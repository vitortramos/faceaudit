# faceaudit/models/instructions/hubness_ablation_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.hubness_ablation_mode import HubnessAblationMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_rule_type import ValidationRuleType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
    ValueFieldSpec,
)
from faceaudit.models.validation.validation_rule_config import ValidationRuleConfig


T = TypeVar("T", bound="HubnessAblationInstruction")


@dataclass
class HubnessAblationInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que executa um estudo de ablação topológica:
    recorta progressivamente o dataset (1, 2, 3 fotos por ID + total)
    seguindo o critério escolhido (qualidade FIQA ou frontalidade) e
    mede o hubness em cada cenário.

    Custoso: refaz O(N²) em N subsets. Use HUBNESS_ANALYSIS para o
    diagnóstico geral barato; use ABLATION quando precisar do estudo
    formal de evolução do Nk com a redução do dataset.
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso DATASET ou INTERMEDIATE_DATASET (EmbeddingDataset) a analisar."}
    )

    output_ablation_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do tipo HUBNESS_ABLATION_RESULT (pickle)."}
    )

    ablation_mode: HubnessAblationMode = field(
        default=HubnessAblationMode.BY_QUANTITY,
        metadata={"help": "Critério de ordenação por identidade ao recortar os cenários."}
    )

    k_neighbors: int = field(
        default=20,
        metadata={"help": "K do KNN exaustivo aplicado em cada cenário."}
    )

    batch_size: int = field(
        default=1000,
        metadata={"help": "Tamanho do lote em cada cenário."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.HUBNESS_ABLATION

        if not self.name:
            self.name = "Hubness Ablation"

        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec(
                field_name="ablation_mode",
                label="Modo de Ablação",
                choices=("BY_QUANTITY", "BY_POSE"),
                enum_type=HubnessAblationMode,
                descriptions={
                    "BY_QUANTITY":
                        "Prioriza as N MELHORES fotos por identidade ordenadas por "
                        "qualidade (FIQA). Cenários: 1, 2, 3 fotos por ID + total. "
                        "Responde: o hubness piora à medida que adicionamos fotos?",
                    "BY_POSE":
                        "Prioriza as N MAIS FRONTAIS fotos por identidade "
                        "(menor abs(yaw)+abs(pitch)). Cenários: 1, 2, 3 fotos por "
                        "ID + total. Responde: filtrar pra só frontais reduz o "
                        "hubness?",
                },
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (StrategyResourceType.DATASET,)
        OUT = (StrategyResourceType.HUBNESS_ABLATION_RESULT,)
        return [
            ResourceFieldSpec("dataset_resource_id", "Dataset", IN),
            ResourceFieldSpec(
                "output_ablation_resource_id",
                "Resultado de Ablação",
                OUT,
                is_output=True,
            ),
        ]

    def get_value_field_specs(self) -> list[ValueFieldSpec]:
        return [
            ValueFieldSpec(
                field_name="k_neighbors",
                label="Vizinhos (K)",
                value_type="int",
                min_value=1, max_value=10**4,
            ),
            ValueFieldSpec(
                field_name="batch_size",
                label="Batch size",
                value_type="int",
                min_value=1, max_value=10**6,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "⚠ <b>Operação MUITO cara:</b> refaz O(N²) em cada cenário (4 "
            "execuções). Para datasets grandes, considere primeiro rodar "
            "HUBNESS_ANALYSIS (mais barato) e usar essa instrução só se "
            "precisar do estudo formal de evolução."
        )

    def _sync_declared_io(self) -> None:
        inputs: list = []
        outputs: list = []
        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)
        if self.output_ablation_resource_id:
            outputs.append(self.output_ablation_resource_id)
        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

        self._ensure_default_validation_rules()

    def _ensure_default_validation_rules(self) -> None:
        """
        Defaults:
        - MIN_SAMPLES no source.
        - MAX_SAMPLES_WARNING — ablação é O(K·N²); threshold conservador.
        - HAS_POSE_METADATA quando BY_POSE — yaw/pitch obrigatórios.
        """
        if self.validation_rules_user_managed:
            return

        # Reset & repopulate: params apontam pros input fields E pro
        # ablation_mode. `user_managed=True` (acima) preserva escolha
        # explícita do usuário.
        self.validation_rule_configs = []

        if not self.dataset_resource_id:
            return

        cfgs = [
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MIN_SAMPLES,
                params={"dataset_resource_id": self.dataset_resource_id},
            ),
            ValidationRuleConfig(
                rule_type=ValidationRuleType.MAX_SAMPLES_WARNING,
                params={
                    "dataset_resource_id": self.dataset_resource_id,
                    "max_samples": 50_000,
                },
            ),
        ]
        if self.ablation_mode == HubnessAblationMode.BY_POSE:
            cfgs.append(ValidationRuleConfig(
                rule_type=ValidationRuleType.HAS_POSE_METADATA,
                params={"dataset_resource_id": self.dataset_resource_id},
            ))
        self.validation_rule_configs = cfgs

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return None
        d = dict(data)
        if "instruction_type" in d and d["instruction_type"]:
            d["instruction_type"] = StrategyInstructionType.from_str(d["instruction_type"])
        if "ablation_mode" in d and d["ablation_mode"]:
            d["ablation_mode"] = HubnessAblationMode.from_str(d["ablation_mode"])
        return cls(**d)
