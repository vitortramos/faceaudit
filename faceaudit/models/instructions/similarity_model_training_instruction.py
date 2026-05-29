# faceaudit/models/instructions/similarity_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.similarity_model_type import SimilarityModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="SimilarityModelTrainingInstruction")


@dataclass
class SimilarityModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de similaridade.

    Esta instrução descreve:
    - qual família de modelo de similaridade será usada
    - qual recurso de entrada (dataset) será usado no treinamento/preparação
    - qual config de hyperparâmetros (embutido na instrução)
    - qual recurso de saída representará o modelo treinado/preparado

    Observação:
    O termo "training" aqui é usado em sentido amplo. Cosine não exige treino —
    é apenas instanciação de uma função de similaridade. PLDA exige treino real
    com hyperparâmetros descritos em `similarity_model_config`.
    """

    similarity_model_type: SimilarityModelType = field(
        default=SimilarityModelType.COSINE,
        metadata={"help": "Tipo de modelo de similaridade a ser usado."}
    )

    input_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset de entrada usado pelo modelo de similaridade."}
    )

    secondary_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "<html><b>Dataset secundário (opcional, condição cruzada).</b><br/><br/>"
            "Quando preenchido, ativa o estimador PLDA <b>cross-condition</b> "
            "(Brümmer 2010 com peso igual por condição). Útil para comparações "
            "forenses entre condições assimétricas — ex.: foto-padrão frontal "
            "(primário) vs frame de CFTV lateral (secundário).<br/><br/>"
            "<b>Apenas para variantes TWO_COVARIANCE e SHRINKAGE_TWO_COVARIANCE.</b> "
            "SIMPLE + secondary → erro.<br/><br/>"
            "<i>Premissa crítica:</i> ambos os datasets devem estar no mesmo "
            "espaço transformado (i.e. produzidos pela mesma cadeia de "
            "pipeline — tipicamente um DUAL com CORAL/Whitening). A "
            "validação atual confirma apenas coluna e dimensão de embedding, "
            "não a equivalência completa do espaço vetorial — verifique "
            "manualmente que os datasets vêm do mesmo pipeline.</html>"
        )}
    )

    similarity_model_config: Optional[PLDASimilarityModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do modelo (PLDA). None para Cosine."}
    )

    trained_similarity_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o modelo de similaridade treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SIMILARITY_MODEL_TRAINING

        if not self.name:
            self.name = "Similarity Model Training"

        # Auto-instancia o config quando o tipo o exige e ele ainda não foi
        # preenchido. Para Cosine, mantém None (não tem hyperparâmetros).
        if self.uses_plda and self.similarity_model_config is None:
            self.similarity_model_config = PLDASimilarityModelConfig()

        # _sync_declared_io já chama _ensure_default_validation_rules
        # internamente, então a inicialização cobre tanto o caso "tudo
        # informado na construção" quanto "campo atualizado depois".
        self._sync_declared_io()

    def _ensure_default_validation_rules(self) -> None:
        """
        Popula `validation_rule_configs` com as regras padrão recomendadas
        para o estado atual da instrução, **apenas se o usuário não tiver
        assumido o controle** das regras (`validation_rules_user_managed=True`
        preserva escolha explícita).

        Regras default por cenário:
        - PLDA com `input_dataset_resource_id` preenchido (modo single-dataset):
            - FISHER (n_identidades > embedding_dim) no primário
            - MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES no primário
        - PLDA com `input_dataset_resource_id` E `secondary_dataset_resource_id`
          preenchidos (modo cross-condition):
            - FISHER no primário (mantém compat)
            - MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES no primário
            - MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES no secundário
              (sem FISHER — secundário tipicamente esparso; W_B é resgatada
              por shrinkage, daí FISHER bloquearia o cenário-alvo)
            - DATASETS_EMBEDDING_COMPAT([primário, secundário])
            - DATASET_OVERLAP_REQUIRED(primário, secundário) — warning
              (estimador não exige overlap, mas overlap baixo torna B/μ ruidosa)
        - Cosine: nenhuma regra padrão (modelo stateless).
        """
        if self.validation_rules_user_managed:
            return

        # Reset & repopulate.
        self.validation_rule_configs = []

        if not self.uses_plda:
            return

        if not self.input_dataset_resource_id:
            return

        primary = self.input_dataset_resource_id
        secondary = self.secondary_dataset_resource_id

        rules = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.FISHER,
                params={"dataset_resource_id": primary},
            ),
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES,
                params={"dataset_resource_id": primary},
            ),
        ]

        if secondary:
            # Secundário: só MIN (sem FISHER — n_ids > D bloquearia condição
            # tipicamente esparsa que é resgatada por shrinkage).
            rules.append(
                ValidationRuleConfig(
                    rule_type=ValidationRuleType.MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES,
                    params={"dataset_resource_id": secondary},
                )
            )
            rules.append(
                ValidationRuleConfig(
                    rule_type=ValidationRuleType.DATASETS_EMBEDDING_COMPAT,
                    params={"dataset_resource_ids": [primary, secondary]},
                )
            )
            # Overlap mínimo brando: 2 IDs comuns é tecnicamente suficiente
            # pra estimar B (peso igual por classe), mas warn_below_ratio
            # generoso sinaliza overlap baixo sem bloquear.
            rules.append(
                ValidationRuleConfig(
                    rule_type=ValidationRuleType.DATASET_OVERLAP_REQUIRED,
                    params={
                        "dataset_a_resource_id": primary,
                        "dataset_b_resource_id": secondary,
                        "min_overlap": 2,
                        "warn_below_ratio": 0.3,
                    },
                )
            )

        self.validation_rule_configs = rules

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        triples = SimilarityModelType.choices_with_description()
        return [
            ChoiceFieldSpec(
                field_name="similarity_model_type",
                label="Tipo do Modelo",
                choices=tuple(value for value, _label, _desc in triples),
                enum_type=SimilarityModelType,
                descriptions={value: desc for value, _label, desc in triples},
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TS = (StrategyResourceType.TRAINED_SIMILARITY_MODEL,)
        specs: list[ResourceFieldSpec] = []
        # Dataset só aparece para PLDA — COSINE é stateless e não precisa de
        # treino.
        if self.uses_plda:
            specs.append(
                ResourceFieldSpec(
                    "input_dataset_resource_id",
                    "Dataset",
                    DS,
                    triggers_rebuild=True,
                )
            )
            # Dataset secundário só faz sentido em variantes two-cov.
            # Marcado triggers_rebuild para repopular regras de validação
            # (cross-condition adiciona MIN no secundário + COMPAT + OVERLAP).
            if self._uses_any_two_covariance:
                specs.append(
                    ResourceFieldSpec(
                        "secondary_dataset_resource_id",
                        "Dataset 2 (condição cruzada, opcional)",
                        DS,
                        triggers_rebuild=True,
                    )
                )
        specs.append(ResourceFieldSpec(
            "trained_similarity_model_resource_id",
            "Modelo Treinado",
            TS,
            is_output=True,
        ))
        return specs

    @property
    def _uses_any_two_covariance(self) -> bool:
        """Acesso seguro à propriedade do config — None quando Cosine."""
        if not self.uses_plda or self.similarity_model_config is None:
            return False
        return bool(
            getattr(
                self.similarity_model_config,
                "uses_any_two_covariance_variant",
                False,
            )
        )

    def prune_unused_fields(self) -> None:
        """
        Zera `input_dataset_resource_id`, `secondary_dataset_resource_id`
        e `similarity_model_config` quando o tipo escolhido é COSINE
        (stateless, sem treino, sem hyperparams). Também zera
        `secondary_dataset_resource_id` quando a variante PLDA é SIMPLE
        (cross-condition exige two-cov). Chamado só no Apply.
        """
        if self.uses_cosine:
            self.input_dataset_resource_id = None
            self.secondary_dataset_resource_id = None
            self.similarity_model_config = None
            return

        # PLDA: secondary só pra two-cov.
        if not self._uses_any_two_covariance:
            self.secondary_dataset_resource_id = None

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos específicos
        desta instrução. A config do modelo é embutida no DTO e não conta como input.

        Cosine não exige dataset de entrada (stateless), então só PLDA inclui
        input_dataset_resource_id no gate de dependência. PLDA two-cov com
        `secondary_dataset_resource_id` definido também o inclui (modo
        cross-condition).
        """
        inputs = []
        outputs = []

        if self.uses_plda and self.input_dataset_resource_id:
            inputs.append(self.input_dataset_resource_id)

        if (
            self.uses_plda
            and self._uses_any_two_covariance
            and self.secondary_dataset_resource_id
        ):
            inputs.append(self.secondary_dataset_resource_id)

        if self.trained_similarity_model_resource_id:
            outputs.append(self.trained_similarity_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # _sync_declared_io é chamado pelo config panel após cada edição
        # de campo — bom ponto pra repopular defaults quando o usuário
        # acabou de definir o dataset (que não existia em __post_init__).
        self._ensure_default_validation_rules()

    @property
    def uses_cosine(self) -> bool:
        return self.similarity_model_type == SimilarityModelType.COSINE

    @property
    def uses_plda(self) -> bool:
        return self.similarity_model_type == SimilarityModelType.PLDA

    def should_rebuild_on_nested_change(self, attr_name: str, new_dto) -> bool:
        """
        Hook do InstructionConfigPanel: sinaliza quando uma mudança no DTO
        aninhado (similarity_model_config) afeta a visibilidade dos resource
        specs do panel — caso canônico: mudar `plda_variant` para SIMPLE
        esconde o dropdown de `secondary_dataset_resource_id`, mudar para
        TWO_COVARIANCE o mostra.
        """
        return attr_name == "similarity_model_config"

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - similarity_model_type
        - similarity_model_config
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if (
            "similarity_model_type" in processed_data
            and processed_data["similarity_model_type"]
        ):
            processed_data["similarity_model_type"] = SimilarityModelType.from_str(
                processed_data["similarity_model_type"]
            )

        if "similarity_model_config" in processed_data and isinstance(
            processed_data["similarity_model_config"], dict
        ):
            processed_data["similarity_model_config"] = PLDASimilarityModelConfig.from_dict(
                processed_data["similarity_model_config"]
            )

        # Retrocompat: campo removido
        processed_data.pop("similarity_model_config_resource_id", None)

        return cls(**processed_data)
