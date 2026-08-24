# faceaudit/models/instructions/embedding_aggregation_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.aggregation_config import AggregationConfig
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="EmbeddingAggregationInstruction")


@dataclass
class EmbeddingAggregationInstruction(StrategyInstructionDefinition):
    """Agrega embeddings da mesma identidade em templates sintéticos.

    A mesma matemática do step `IDENTITY_AGGREGATION` do pipeline de
    transformação, exposta como instrução autônoma — para quando se quer
    agregar SEM aplicar nenhuma outra transformação.

    **Por que instrução única, sem par treino/aplicação.** A agregação é
    stateless: não há parâmetro estimado a partir dos dados (diferente de
    Whitening, LDA ou CORAL, que aprendem uma matriz no fit e a reaplicam
    depois). Um `TRAINED_PIPELINE` para ela seria um artefato vazio, e o par
    PIPELINE_TRAINING + PIPELINE_APPLICATION obrigaria duas instruções para
    uma operação que é uma função pura do dataset de entrada.

    **A saída não é um dataset de fotos.** Um template é a média de N faces:
    não corresponde a imagem nenhuma. Por isso os campos que descrevem a FOTO
    (`image_path`, `pose`, `quality_scores`, `face_image`) saem vazios, e o
    `face_hash` deixa de ser "hash do arquivo" para ser um identificador
    determinístico derivado das faces contribuintes. Consequência prática: as
    regras de cobertura passam a bloquear, sobre este dataset, operações que
    dependem de pose ou de qualidade de imagem — o que é o comportamento
    correto, não uma limitação.

    **Cardinalidade muda.** Diferente de toda transformação por amostra
    (R^d→R^d, N constante), a agregação é N→M: `CENTROID` colapsa cada
    identidade em 1 template; `COMBINATORIAL` gera C(N,k) templates por
    identidade, limitados pelo teto configurado.

    **Nota forense.** Um template fundido de N imagens é objeto
    estatisticamente distinto de um embedding de imagem única — a distribuição
    de scores muda. Se a referência for agregada e a questionada não, o
    calibrador precisa ter sido treinado nessa mesma assimetria, sob pena de o
    LR não valer. A proveniência da agregação fica registrada no
    `loaded_metadata` do dataset de saída.

    Saída: novo recurso `DATASET` (a entrada é preservada intacta).
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "<html>Dataset cujos embeddings serão agregados por identidade."
            "<br><br><b>Exige identidade em todas as faces</b> — agregar sem "
            "identidade não tem significado e a execução aborta.</html>"
        )},
    )

    output_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída com os templates agregados."},
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Nome opcional do dataset agregado. Em branco, deriva do nome de "
            "entrada com o sufixo do método."
        )},
    )

    aggregation_config: AggregationConfig = field(
        default_factory=AggregationConfig,
        metadata={"help": (
            "<html>Regras de fusão: método topológico (CENTROID/COMBINATORIAL), "
            "tamanho do subgrupo e teto por identidade, e ponderação opcional "
            "por score de qualidade.</html>"
        )},
    )

    def __post_init__(self):
        self.instruction_type = StrategyInstructionType.EMBEDDING_AGGREGATION
        if not self.name:
            self.name = "Agregação de Embeddings"
        # JSON legado / construção mínima pela UI podem deixar o config nulo.
        # `AggregationConfig.from_dict` devolve None para dict vazio, então o
        # auto-preenchimento aqui é o que impede o painel de renderizar
        # "(sem configuração)".
        if self.aggregation_config is None:
            self.aggregation_config = AggregationConfig()
        self._sync_declared_io()

    # ── Sincronização de IO declarado ────────────────────────────────

    def _sync_declared_io(self) -> None:
        inputs: List[str] = []
        outputs: List[str] = []

        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)
        if self.output_dataset_resource_id:
            outputs.append(self.output_dataset_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """Defaults persistentes (ajustáveis/removíveis pelo usuário na UI).

        - MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES: identidade com uma amostra só
          produz um "template" idêntico ao embedding original — a agregação
          vira no-op. `min_required=1` (e não o default 2, pensado para o fit
          do PLDA): agregar UMA identidade é caso legítimo e frequente em
          casework, onde se funde as fotos de referência de um único indivíduo.
          O que a regra precisa barrar é o dataset em que NENHUMA identidade
          tem o que fundir. `safety_threshold=1` pelo mesmo motivo — avisar a
          partir de 10 identidades seria ruído no fluxo de caso.
        - QUALITY_METRIC_AVAILABLE (só com ponderação ligada): ponderar por uma
          métrica ausente do dataset faz todos os pesos caírem no fallback 1.0,
          e a "média ponderada" vira média simples em silêncio.
        - MAX_SAMPLES_WARNING (só em COMBINATORIAL): o custo é combinatório em
          C(N,k) por identidade antes do teto.
        """
        if self.validation_rules_user_managed:
            return

        self.validation_rule_configs = []

        if not self.dataset_resource_id:
            return

        cfgs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES,
                params={
                    "dataset_resource_id": self.dataset_resource_id,
                    "min_required": 1,
                    "safety_threshold": 1,
                },
            ),
        ]

        cfg = self.aggregation_config or AggregationConfig()

        if cfg.use_quality_weighting and cfg.quality_weight_type is not None:
            # `.name` e NUNCA `.value`: QualityScoreType é Smart Enum cujo
            # value é a tupla (display_name, range, lambda) — o lambda vaza
            # pro JSON e quebra o save da definition.
            metric = getattr(cfg.quality_weight_type, "name", str(cfg.quality_weight_type))
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.QUALITY_METRIC_AVAILABLE,
                params={
                    "dataset_resource_id": self.dataset_resource_id,
                    "quality_metric": metric,
                },
            ))

        if cfg.method == AggregationMethod.COMBINATORIAL:
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.MAX_SAMPLES_WARNING,
                params={
                    "dataset_resource_id": self.dataset_resource_id,
                    "max_samples": 50_000,
                },
            ))

        self.validation_rule_configs = cfgs

    # ── Properties para UI/executor ──────────────────────────────────

    @property
    def uses_aggregation(self) -> bool:
        """Gate do editor inline do config aninhado — sempre verdadeiro aqui
        (a instrução existe para isto), mas o painel exige o hook."""
        return True

    @property
    def uses_combinatorial(self) -> bool:
        cfg = self.aggregation_config
        return bool(cfg and cfg.method == AggregationMethod.COMBINATORIAL)

    @property
    def uses_quality_weighting(self) -> bool:
        cfg = self.aggregation_config
        return bool(cfg and cfg.use_quality_weighting)

    # ── Hooks de UI ──────────────────────────────────────────────────

    def get_resource_field_specs(self) -> list:
        return [
            ResourceFieldSpec(
                field_name="dataset_resource_id",
                label="Dataset de entrada",
                accepted_types=(StrategyResourceType.DATASET,),
                is_output=False,
            ),
            ResourceFieldSpec(
                field_name="output_dataset_resource_id",
                label="Dataset agregado (saída)",
                accepted_types=(StrategyResourceType.DATASET,),
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> list:
        return [
            ValueFieldSpec(
                field_name="output_dataset_name",
                label="Nome do dataset agregado",
                value_type="str",
                placeholder="(deriva do dataset de entrada)",
            ),
        ]

    def should_rebuild_on_nested_change(self, attr_name: str, new_dto) -> bool:
        """Hook do InstructionConfigPanel: editar o config aninhado muda quais
        regras de validação se aplicam.

        `_ensure_default_validation_rules` ramifica em DOIS campos que moram
        dentro do `aggregation_config`: `method` (COMBINATORIAL adiciona
        MAX_SAMPLES_WARNING) e `use_quality_weighting` (adiciona
        QUALITY_METRIC_AVAILABLE). Sem este hook, o painel redesenha o editor
        inline mas não a seção âmbar de regras — o usuário trocaria o método e
        continuaria vendo a lista antiga, que só se corrigiria ao trocar de
        instrução e voltar.
        """
        if attr_name != "aggregation_config":
            return False
        self._sync_declared_io()   # recalcula as regras antes do redesenho
        return True

    def get_configuration_help(self) -> Optional[str]:
        return (
            "<html><b>O resultado é um dataset de templates, não de fotos.</b><br>"
            "Cada template é a média L2-normalizada de N faces da mesma "
            "identidade, e por isso não carrega caminho de imagem, pose nem "
            "score de qualidade — esses campos descrevem uma foto, e um "
            "template não é foto nenhuma.<br><br>"
            "<b>Atenção forense:</b> um template de N imagens tem distribuição "
            "de score diferente de um embedding de imagem única. Se a "
            "referência for agregada e a questionada não, o calibrador precisa "
            "ter sido treinado nessa mesma assimetria.</html>"
        )

    # ── Serialização ─────────────────────────────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        processed_data = dict(data or {})
        processed_data["instruction_type"] = StrategyInstructionType.EMBEDDING_AGGREGATION

        raw_cfg = processed_data.get("aggregation_config")
        if isinstance(raw_cfg, AggregationConfig):
            pass  # já hidratado
        elif isinstance(raw_cfg, dict):
            processed_data["aggregation_config"] = (
                AggregationConfig.from_dict(raw_cfg) or AggregationConfig()
            )
        else:
            processed_data["aggregation_config"] = AggregationConfig()

        # Descarta chaves extintas de JSON legado — sem isto, um campo
        # renomeado no futuro vira TypeError na abertura do projeto.
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in processed_data.items() if k in valid_keys}
        return cls(**filtered)
