# faceaudit/models/instructions/dataset_split_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.dataset_split_mode import DatasetSplitMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="DatasetSplitInstruction")


@dataclass
class DatasetSplitInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para particionamento de datasets.

    Esta instrução descreve:
    - qual dataset de entrada será particionado
    - qual modo de split será utilizado
    - quais parâmetros controlam o particionamento
    - quais recursos de saída serão produzidos

    Modos atualmente suportados:
    - BY_IDENTITY_RATIO
    - BY_IDENTITY_LIST
    - BY_INDEX_RATIO
    - BY_INDEX_LIST
    """

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset que será particionado."}
    )

    split_mode: DatasetSplitMode = field(
        default=DatasetSplitMode.BY_IDENTITY_RATIO,
        metadata={"help": "Modo de split do dataset."}
    )

    split_ratio: float = field(
        default=0.8,
        metadata={"help": "Proporção do split quando o modo usar razão."}
    )

    random_seed: int = field(
        default=42,
        metadata={"help": "Semente aleatória para splits reprodutíveis."}
    )

    identity_list: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista explícita de identidades usada em modos manuais."}
    )

    index_list: List[int] = field(
        default_factory=list,
        metadata={"help": "Lista explícita de índices usada em modos manuais."}
    )

    output_dataset_a_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do primeiro dataset gerado."}
    )

    output_dataset_b_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do segundo dataset gerado."}
    )

    output_dataset_a_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do primeiro dataset gerado."}
    )

    output_dataset_b_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do segundo dataset gerado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.DATASET_SPLIT

        if not self.name:
            self.name = "Dataset Split"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET, StrategyResourceType.INTERMEDIATE_DATASET)
        OUT = (StrategyResourceType.INTERMEDIATE_DATASET,)
        return [
            ResourceFieldSpec("dataset_resource_id", "Dataset", DS),
            ResourceFieldSpec("output_dataset_a_resource_id", "Dataset A", OUT, is_output=True),
            ResourceFieldSpec("output_dataset_b_resource_id", "Dataset B", OUT, is_output=True),
        ]

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos
        específicos da instrução.
        """
        inputs = []
        outputs = []

        if self.dataset_resource_id:
            inputs.append(self.dataset_resource_id)

        if self.output_dataset_a_resource_id:
            outputs.append(self.output_dataset_a_resource_id)

        if self.output_dataset_b_resource_id:
            outputs.append(self.output_dataset_b_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @property
    def uses_identity_ratio(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_IDENTITY_RATIO

    @property
    def uses_identity_list(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_IDENTITY_LIST

    @property
    def uses_index_ratio(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_INDEX_RATIO

    @property
    def uses_index_list(self) -> bool:
        return self.split_mode == DatasetSplitMode.BY_INDEX_LIST

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - split_mode
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "split_mode" in processed_data and processed_data["split_mode"]:
            processed_data["split_mode"] = DatasetSplitMode.from_str(
                processed_data["split_mode"]
            )

        return cls(**processed_data)