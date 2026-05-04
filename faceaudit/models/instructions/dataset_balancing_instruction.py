# faceaudit/models/instructions/dataset_balancing_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.dataset_balancing_mode import DatasetBalancingMode
from faceaudit.enums.dataset_balancing_selection import DatasetBalancingSelection
from faceaudit.enums.quality_score_type import QualityScoreType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)


T = TypeVar("T", bound="DatasetBalancingInstruction")


@dataclass
class DatasetBalancingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para balancear o número de amostras por identidade.

    Duas dimensões ortogonais:

    1. `balancing_mode` — define o alvo de samples por identidade:
       - CAP_ONLY: cap em `target_n`; IDs com menos amostras são mantidas.
       - FLOOR_AND_CAP: cap + floor; IDs com < `target_n` são descartadas.
       - AUTO_MIN_BALANCE: N = min observado; todas as IDs ficam com N.

    2. `selection_criterion` — define como descartar excesso (quando há):
       - RANDOM: sorteio uniforme reprodutível via `random_seed`.
       - BY_QUALITY: top-N por `quality_metric` (maior = melhor).

    Saída: novo `INTERMEDIATE_DATASET` (input preservado).
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset que será balanceado."}
    )

    output_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset balanceado."}
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do dataset balanceado."}
    )

    balancing_mode: DatasetBalancingMode = field(
        default=DatasetBalancingMode.CAP_ONLY,
        metadata={"help": (
            "<html><b>Modo de balanceamento</b> — define quantas amostras "
            "cada identidade vai ter no dataset de saída.<br/><br/>"
            "<b>Exemplo:</b> dataset com IDs A=25, B=12, C=8, D=3 fotos e "
            "<code>target_n=10</code>:<br/><br/>"
            "<b>CAP_ONLY</b> — só corta o que excede target_n; quem tem "
            "menos fica intocado.<br/>"
            "&nbsp;&nbsp;A:25→10, B:12→10, C:8→8, D:3→3 — total 31, "
            "ainda desbalanceado.<br/>"
            "&nbsp;&nbsp;<i>Use quando:</i> quer evitar dominância de IDs "
            "populosos sem perder identidades.<br/><br/>"
            "<b>FLOOR_AND_CAP</b> — corta excesso E descarta IDs inteiras "
            "com menos de target_n.<br/>"
            "&nbsp;&nbsp;A:25→10, B:12→10, C:8→✘, D:3→✘ — total 20, "
            "perfeitamente balanceado.<br/>"
            "&nbsp;&nbsp;<i>Use quando:</i> precisa de balanceamento "
            "estrito (PLDA, validação) e aceita perder IDs.<br/><br/>"
            "<b>AUTO_MIN_BALANCE</b> — ignora target_n; usa o mínimo "
            "observado em runtime.<br/>"
            "&nbsp;&nbsp;min=3 (D) → todas viram 3: A:25→3, B:12→3, C:8→3, "
            "D:3→3 — total 12.<br/>"
            "&nbsp;&nbsp;<i>Use quando:</i> precisa balanceamento sem "
            "descartar IDs e o mínimo já está aceitável. Em datasets "
            "heterogêneos costuma ser ruim — uma ID com poucas amostras "
            "força todo mundo a perder dados.</html>"
        )}
    )

    target_n: int = field(
        default=10,
        metadata={"help": (
            "<html><b>Número alvo de amostras por identidade.</b><br/><br/>"
            "Em <b>CAP_ONLY</b>: limite máximo — IDs com menos amostras "
            "ficam como estão.<br/>"
            "Em <b>FLOOR_AND_CAP</b>: piso E cap — IDs com menos que "
            "isso são descartadas inteiras.<br/>"
            "Em <b>AUTO_MIN_BALANCE</b>: ignorado (o N é computado como "
            "min(samples_per_identity) em runtime).</html>"
        )}
    )

    selection_criterion: DatasetBalancingSelection = field(
        default=DatasetBalancingSelection.RANDOM,
        metadata={"help": (
            "<html><b>Critério de seleção</b> — quando uma ID precisa "
            "ser cortada (ex: 25 fotos → 10), como decidir quais 10 "
            "ficam?<br/><br/>"
            "<b>RANDOM</b> — sorteio uniforme controlado por "
            "<code>random_seed</code>. Reprodutível: mesma seed → mesma "
            "escolha. Útil quando você não tem métrica de qualidade "
            "confiável ou quer evitar viés.<br/><br/>"
            "<b>BY_QUALITY</b> — mantém as N <i>melhores</i> de cada "
            "ID por uma métrica (FIQA/OFIQ/SERFIQ/EMBEDDING_NORM). "
            "Direção: maior = melhor sempre. Empates são quebrados "
            "deterministicamente via SHA1 de <code>face_hash</code>/"
            "<code>file_name</code> — mesma escolha entre máquinas.</html>"
        )}
    )

    quality_metric: QualityScoreType = field(
        default=QualityScoreType.OFIQ,
        metadata={"help": (
            "<html><b>Métrica de qualidade</b> usada quando "
            "<code>selection_criterion=BY_QUALITY</code>. "
            "Direção: <b>maior = melhor</b> para todas.<br/><br/>"
            "<b>FIQA</b> — score do ForensicFace/sepaelv2 (0.0–1.0).<br/>"
            "<b>OFIQ</b> — score unificado ISO/OFIQ (0.0–100.0).<br/>"
            "<b>SERFIQ</b> — Stochastic Embedding Robustness FIQ.<br/>"
            "<b>EMBEDDING_NORM</b> — norma L2 do embedding "
            "(<code>np.linalg.norm</code>), calculada on-the-fly. "
            "Sempre disponível, não precisa estar no parquet.<br/><br/>"
            "<i>O dropdown filtra automaticamente as métricas com "
            "cobertura ≥ 50% no dataset selecionado.</i></html>"
        )}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": (
            "<html><b>Semente aleatória.</b><br/><br/>"
            "Em <b>RANDOM</b>: controla totalmente o sorteio — "
            "mesma seed → mesmo conjunto escolhido.<br/>"
            "Em <b>BY_QUALITY</b>: usado apenas como tie-breaker "
            "determinístico em empates de score.</html>"
        )}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.DATASET_BALANCING

        if not self.name:
            self.name = "Dataset Balancing"

        self._sync_declared_io()

    # ── Field specs para a UI ────────────────────────────────────────

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        specs: List[ChoiceFieldSpec] = [
            ChoiceFieldSpec(
                field_name="balancing_mode",
                label="Modo de Balanceamento",
                choices=("CAP_ONLY", "FLOOR_AND_CAP", "AUTO_MIN_BALANCE"),
                enum_type=DatasetBalancingMode,
                descriptions={
                    "CAP_ONLY":
                        "Corta o que excede target_n; IDs com menos amostras "
                        "ficam intocadas. Ex: A=25→10, B=8→8 (target_n=10). "
                        "Preserva diversidade, mas resultado segue desbalanceado.",
                    "FLOOR_AND_CAP":
                        "Cap em target_n + DESCARTA IDs inteiras com menos "
                        "amostras que isso. Ex: A=25→10, B=8→removida "
                        "(target_n=10). Balanceamento perfeito ao custo de "
                        "perder identidades.",
                    "AUTO_MIN_BALANCE":
                        "Ignora target_n e usa o mínimo observado em runtime. "
                        "Ex: min=3 → todas as IDs viram 3. Balanceado sem "
                        "descartar IDs, mas pode perder muitos dados se uma "
                        "ID tem poucas amostras.",
                },
            ),
            ChoiceFieldSpec(
                field_name="selection_criterion",
                label="Critério de Seleção",
                choices=("RANDOM", "BY_QUALITY"),
                enum_type=DatasetBalancingSelection,
                descriptions={
                    "RANDOM":
                        "Sorteio uniforme controlado por random_seed — "
                        "reprodutível, sem viés. Use quando não há métrica "
                        "de qualidade confiável no dataset.",
                    "BY_QUALITY":
                        "Mantém as N melhores fotos de cada ID por uma "
                        "métrica (FIQA/OFIQ/SERFIQ/EMBEDDING_NORM). Maior = "
                        "melhor. Empates resolvidos por SHA1 de face_hash "
                        "(determinístico entre máquinas).",
                },
            ),
        ]

        # Métrica de qualidade só aparece quando o critério é BY_QUALITY.
        # Filtragem dinâmica por disponibilidade no dataset acontece via
        # `dynamic_choices`: o painel chama o callback a cada rebuild do form
        # passando `available_resources`; retornamos só as métricas que de
        # fato existem no dataset selecionado (≥ 50% de cobertura).
        # EMBEDDING_NORM sempre está disponível (calculada on-the-fly).
        if self.uses_quality_selection:
            specs.append(
                ChoiceFieldSpec(
                    field_name="quality_metric",
                    label="Métrica de Qualidade",
                    choices=tuple(qt.name for qt in QualityScoreType),
                    enum_type=QualityScoreType,
                    dynamic_choices=self._available_quality_metric_choices,
                    descriptions={
                        "FIQA":
                            "Score do ForensicFace (0.0–1.0). Lê de "
                            "`face.ff_info.fiqa_score`.",
                        "OFIQ":
                            "Score unificado ISO/OFIQ (0.0–100.0). Lê de "
                            "`face.ofiq_info.unified_quality_score_scalar`.",
                        "SERFIQ":
                            "Score SER-FIQ estocástico (0.0–100.0). Lê de "
                            "`face.serfiq_info.score`.",
                        "EMBEDDING_NORM":
                            "Norma L2 do embedding sepaelv2/AdaFace. "
                            "Maior norma = mais confiança no extrator. "
                            "Calculada on-the-fly — não requer coluna no parquet.",
                    },
                )
            )

        return specs

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET, StrategyResourceType.INTERMEDIATE_DATASET)
        OUT = (StrategyResourceType.INTERMEDIATE_DATASET,)

        return [
            # triggers_rebuild=True: a métrica de qualidade é filtrada
            # dinamicamente pelo dataset escolhido, então mudar de dataset
            # precisa re-renderizar o dropdown de quality_metric.
            ResourceFieldSpec(
                "dataset_resource_id", "Dataset de entrada", DS,
                triggers_rebuild=True,
            ),
            ResourceFieldSpec(
                "output_dataset_resource_id",
                "Dataset balanceado",
                OUT,
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> List[ValueFieldSpec]:
        specs: List[ValueFieldSpec] = []

        # target_n não é editável em AUTO_MIN_BALANCE — N é computado em runtime.
        if not self.uses_auto_min:
            specs.append(ValueFieldSpec(
                field_name="target_n",
                label="Amostras por Identidade (N)",
                value_type="int",
                min_value=1,
                max_value=10**5,
            ))

        specs.append(ValueFieldSpec(
            field_name="random_seed",
            label="Semente Aleatória",
            value_type="int",
            min_value=0,
            max_value=10**9,
        ))

        return specs

    def get_configuration_help(self) -> Optional[str]:
        if self.uses_floor_and_cap:
            n = self.target_n
            return (
                f"⚠️ Identidades com menos de {n} amostra(s) serão DESCARTADAS "
                "por completo (não aparecem no dataset balanceado)."
            )
        if self.uses_auto_min:
            return (
                "ℹ️ N será computado em runtime como o MÍNIMO de amostras "
                "observado entre as identidades. Se houver alguma identidade "
                "com pouquíssimas amostras, todas as outras serão reduzidas "
                "ao mesmo nível — atenção a outliers."
            )
        return None

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

    # ── Properties para UI/executor ──────────────────────────────────

    @property
    def uses_cap_only(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.CAP_ONLY

    @property
    def uses_floor_and_cap(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.FLOOR_AND_CAP

    @property
    def uses_auto_min(self) -> bool:
        return self.balancing_mode == DatasetBalancingMode.AUTO_MIN_BALANCE

    @property
    def uses_random_selection(self) -> bool:
        return self.selection_criterion == DatasetBalancingSelection.RANDOM

    # ── Hooks dinâmicos para a UI ────────────────────────────────────

    def _available_quality_metric_choices(
        self, available_resources
    ):
        """
        Callback de `dynamic_choices` do ChoiceFieldSpec `quality_metric`.

        Inspeciona o dataset selecionado (`self.dataset_resource_id`) e
        retorna apenas as métricas com cobertura suficiente (≥ 50%).
        Quando `dataset_resource_id` está vazio, ou o recurso não está em
        `available_resources`, ou a inspeção falha, retorna None — o
        painel mantém as 4 opções default.

        EMBEDDING_NORM sempre aparece (calculada on-the-fly, não exige
        coluna no parquet nem campo no objeto).

        Args:
            available_resources: lista de StrategyResource passada pelo
                painel (set_available_resources do panel V2).

        Returns:
            Optional[Tuple[str, ...]] com nomes de QualityScoreType
            disponíveis, ou None para preservar `choices` default.
        """
        if not self.dataset_resource_id or not available_resources:
            return None

        target = None
        for r in available_resources:
            if getattr(r, "resource_id", None) == self.dataset_resource_id:
                target = r
                break

        if target is None:
            return None

        try:
            from faceaudit.services.quality_metric_availability_service import (
                QualityMetricAvailabilityService,
            )
            project_root = getattr(self, "_project_root_hint", None)
            available = QualityMetricAvailabilityService.available_metrics(
                resource=target,
                project_root=project_root,
            )
        except Exception:
            return None

        if not available:
            return None

        # Mantém a ordem canônica do enum (FIQA, OFIQ, SERFIQ, EMBEDDING_NORM).
        return tuple(qt.name for qt in QualityScoreType if qt in available)

    @property
    def uses_quality_selection(self) -> bool:
        return self.selection_criterion == DatasetBalancingSelection.BY_QUALITY

    # ── Reconstrução a partir de dict ────────────────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        if not data:
            raise ValueError("data não pode ser vazio")

        d = dict(data)

        # Hidratação de enums (com fallbacks seguros via from_str).
        d["balancing_mode"] = DatasetBalancingMode.from_str(
            d.get("balancing_mode", DatasetBalancingMode.CAP_ONLY.value)
        )
        d["selection_criterion"] = DatasetBalancingSelection.from_str(
            d.get("selection_criterion", DatasetBalancingSelection.RANDOM.value)
        )
        d["quality_metric"] = QualityScoreType.from_str(
            d.get("quality_metric", QualityScoreType.OFIQ.name)
        )

        # instruction_type sempre fixo — força para o tipo correto.
        d["instruction_type"] = StrategyInstructionType.DATASET_BALANCING

        # Filtra apenas chaves válidas para evitar TypeError.
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in d.items() if k in valid_keys}

        return cls(**filtered)
