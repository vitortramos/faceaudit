# faceaudit/models/instructions/pair_mining_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.comparison_mining_config import ComparisonMiningConfig
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="PairMiningInstruction")


@dataclass
class PairMiningInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para curadoria estatística de pares de treino.

    Recebe um EmbeddingComparisonDataset já computado (pares com scores) e
    aplica as regras de mineração definidas em ComparisonMiningConfig, produzindo
    dois datasets curados: um de impostores e um de genuínos.

    Os datasets de saída são usados tipicamente como entrada para
    CalibrationModelTrainingInstruction.
    """

    comparison_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada: EmbeddingComparisonDataset com pares pré-calculados."}
    )

    mining_config: Optional[ComparisonMiningConfig] = field(
        default=None,
        metadata={"help": "Configuração de mineração (regras genuínos/impostores, estratégias)."}
    )

    output_impostors_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída: impostores minerados (EmbeddingComparisonDataset)."}
    )

    output_genuines_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída: genuínos minerados (EmbeddingComparisonDataset)."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.PAIR_MINING

        if not self.name:
            self.name = "Pair Mining"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        ART = (StrategyResourceType.ARTIFACT,)
        OUT = (StrategyResourceType.INTERMEDIATE_DATASET,)
        return [
            ResourceFieldSpec("comparison_dataset_resource_id", "Dataset de Comparação", ART),
            ResourceFieldSpec("output_impostors_resource_id", "Impostores Minerados", OUT, is_output=True),
            ResourceFieldSpec("output_genuines_resource_id", "Genuínos Minerados", OUT, is_output=True),
        ]

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.comparison_dataset_resource_id:
            inputs.append(self.comparison_dataset_resource_id)

        if self.output_impostors_resource_id:
            outputs.append(self.output_impostors_resource_id)

        if self.output_genuines_resource_id:
            outputs.append(self.output_genuines_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, hidratando instruction_type e mining_config.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "mining_config" in processed_data and isinstance(
            processed_data["mining_config"], dict
        ):
            processed_data["mining_config"] = ComparisonMiningConfig.from_dict(
                processed_data["mining_config"]
            )

        return cls(**processed_data)
