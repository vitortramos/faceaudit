# faceaudit/models/instructions/typicality_model_training_instruction.py

"""Instrução unificada de treinamento de modelo de tipicidade (PR 2).

Substitui `KNN_TYPICALITY_MODEL_TRAINING` da V2.14 — o tipo legado é
removido do enum e do dispatcher. Estratégias salvas com o tipo antigo
são migradas pelo `LegacyStrategyMigrationService` antes do dispatcher
ver o JSON.

Padrão segue `SimilarityModelTrainingInstruction`:
    - enum discriminador (`typicality_model_type`)
    - config DTO polimórfico (Union de DTOs concretos)
    - `__post_init__` auto-instancia o config quando o tipo o exige
    - `from_dict` dispatcher hidrata config certo baseado em `typicality_model_type`
    - `prune_unused_fields` zera campos irrelevantes ao tipo
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar, Union

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)
from faceaudit.models.typicality.knn_typicality_model_config import (
    KNNTypicalityModelConfig,
)
from faceaudit.models.typicality.movmf_typicality_model_config import (
    MoVMFTypicalityModelConfig,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="TypicalityModelTrainingInstruction")

# Union do config polimórfico — ambos os DTOs.
AnyTypicalityModelConfig = Union[
    KNNTypicalityModelConfig, MoVMFTypicalityModelConfig
]


@dataclass
class TypicalityModelTrainingInstruction(StrategyInstructionDefinition):
    """Treina um modelo de tipicidade (k-NN ou MoVMF EXPERIMENTAL).

    Recebe um dataset de referência representativo da população relevante
    ao caso forense (estratificação prévia é responsabilidade do
    usuário). O backend concreto é escolhido pelo
    `typicality_model_type`; os hiperparâmetros vivem no
    `typicality_model_config` polimórfico, embutido no DTO.
    """

    typicality_model_type: TypicalityModelType = field(
        default=TypicalityModelType.KNN,
        metadata={"help": (
            "<html><b>Tipo do backend de tipicidade.</b><br/>"
            "<b>KNN</b> (estável): não-paramétrico local. Defensável "
            "forense; comportamento histórico do FaceAudit.<br/>"
            "<b>MOVMF</b> (EXPERIMENTAL): paramétrico esférico. "
            "Aguarde validação científica documentada em "
            "<code>docs/typicality_models.md</code> antes de usar em "
            "casework.</html>"
        )}
    )

    input_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Dataset de referência usado para construir o modelo. "
            "Deve ser representativo da subpopulação relevante."
        )}
    )

    typicality_model_config: Optional[AnyTypicalityModelConfig] = field(
        default=None,
        metadata={"help": (
            "Hiperparâmetros embutidos do backend escolhido. Auto-"
            "instanciado pelo __post_init__ quando ausente."
        )}
    )

    trained_typicality_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": (
            "Recurso de saída que representará o modelo de tipicidade "
            "treinado."
        )}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.TYPICALITY_MODEL_TRAINING

        if not self.name:
            self.name = "Typicality Model Training"

        # Auto-instancia o config de acordo com o tipo, ou re-instancia
        # quando o tipo mudou e o config existente não é mais compatível.
        self._reconcile_config_type()

        self._sync_declared_io()

    # ─────────────────────────── helpers internos ──────────────────────

    def _reconcile_config_type(self) -> None:
        """Garante que `typicality_model_config` é compatível com
        `typicality_model_type`. Se incompatível ou ausente, cria um novo
        DTO default — o usuário re-edita os hyperparams.
        """
        expected_cls = self._expected_config_class()
        if expected_cls is None:
            self.typicality_model_config = None
            return
        if (
            self.typicality_model_config is None
            or not isinstance(self.typicality_model_config, expected_cls)
        ):
            self.typicality_model_config = expected_cls()

    def _expected_config_class(self) -> Optional[type]:
        if self.uses_knn:
            return KNNTypicalityModelConfig
        if self.uses_movmf:
            return MoVMFTypicalityModelConfig
        return None

    @property
    def uses_knn(self) -> bool:
        return self.typicality_model_type == TypicalityModelType.KNN

    @property
    def uses_movmf(self) -> bool:
        return self.typicality_model_type == TypicalityModelType.MOVMF

    def should_rebuild_on_nested_change(self, attr_name: str, new_dto) -> bool:
        """Mudança no config aninhado pode afetar visibilidade de campos
        do panel (ex.: `reference_score_calibration_mode` →
        visibilidade de `cv_n_folds`).
        """
        return attr_name == "typicality_model_config"

    # ─────────────────────────── form specs ───────────────────────────

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TT = (StrategyResourceType.TRAINED_TYPICALITY_MODEL,)
        return [
            ResourceFieldSpec(
                "input_dataset_resource_id", "Dataset de Referência", DS,
            ),
            ResourceFieldSpec(
                "trained_typicality_model_resource_id",
                "Modelo de Tipicidade Treinado",
                TT,
                is_output=True,
            ),
        ]

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        triples = TypicalityModelType.choices_with_description()
        return [
            ChoiceFieldSpec(
                field_name="typicality_model_type",
                label="Tipo do Backend",
                choices=tuple(value for value, _label, _desc in triples),
                enum_type=TypicalityModelType,
                descriptions={value: desc for value, _label, desc in triples},
            ),
        ]

    # ─────────────────────────── IO sync + defaults ───────────────────

    def prune_unused_fields(self) -> None:
        """Zera o config quando o tipo não existe (defensivo)."""
        if self._expected_config_class() is None:
            self.typicality_model_config = None

    def _sync_declared_io(self) -> None:
        inputs: list[str] = []
        outputs: list[str] = []

        if self.input_dataset_resource_id:
            inputs.append(self.input_dataset_resource_id)
        if self.trained_typicality_model_resource_id:
            outputs.append(self.trained_typicality_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        # Reconcilia o config nested com o backend escolhido. Caminho do
        # `_on_choice_changed` do panel: setattr(type) → _sync_declared_io →
        # sem isso, o `typicality_model_config` continua com o DTO antigo
        # (KNN ↔ MoVMF) e a renderização inline mostra os campos errados.
        self._reconcile_config_type()

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """Defaults persistentes — mesmas regras do executor legado.

        - MIN_SAMPLES: requisito do fit; warning abaixo de 1000.
        - MAX_SAMPLES_WARNING: aviso de custo O(N²) acima de 50k.
        """
        if self.validation_rules_user_managed:
            return

        self.validation_rule_configs = []
        if not self.input_dataset_resource_id:
            return

        # Para k-NN, min_required = k + 1 (LOO). Para MoVMF, min_required
        # depende de n_components — defensivamente usamos n_components.
        min_required = 2
        if isinstance(self.typicality_model_config, KNNTypicalityModelConfig):
            min_required = max(self.typicality_model_config.k + 1, 2)
        elif isinstance(self.typicality_model_config, MoVMFTypicalityModelConfig):
            min_required = max(self.typicality_model_config.n_components + 1, 2)

        self.validation_rule_configs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_SAMPLES,
                params={
                    "dataset_resource_id": self.input_dataset_resource_id,
                    "min_required": min_required,
                    "safety_threshold": 1000,
                },
            ),
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MAX_SAMPLES_WARNING,
                params={
                    "dataset_resource_id": self.input_dataset_resource_id,
                    "max_samples": 50_000,
                },
            ),
        ]

    # ─────────────────────────── deserialização ───────────────────────

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if (
            "typicality_model_type" in processed_data
            and processed_data["typicality_model_type"]
        ):
            processed_data["typicality_model_type"] = TypicalityModelType.from_str(
                processed_data["typicality_model_type"]
            )

        # Hidrata config polimórfico via o tipo.
        raw_cfg = processed_data.get("typicality_model_config")
        if isinstance(raw_cfg, dict):
            chosen_type = processed_data.get("typicality_model_type", TypicalityModelType.KNN)
            if not isinstance(chosen_type, TypicalityModelType):
                chosen_type = TypicalityModelType.from_str(chosen_type)
            if chosen_type == TypicalityModelType.KNN:
                processed_data["typicality_model_config"] = KNNTypicalityModelConfig.from_dict(raw_cfg)
            elif chosen_type == TypicalityModelType.MOVMF:
                processed_data["typicality_model_config"] = MoVMFTypicalityModelConfig.from_dict(raw_cfg)

        # Retrocompat: campos extintos da V2.14 (já tratados pelo migration
        # service mas tolerantes a JSONs híbridos).
        processed_data.pop("k", None)
        processed_data.pop("ood_quantile", None)
        processed_data.pop("reference_mode", None)
        processed_data.pop("embedding_column", None)
        processed_data.pop("population_label", None)

        return cls(**processed_data)
