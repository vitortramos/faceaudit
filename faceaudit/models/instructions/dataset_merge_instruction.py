# faceaudit/models/instructions/dataset_merge_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.dataset_merge_mode import DatasetMergeMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="DatasetMergeInstruction")


@dataclass
class DatasetMergeInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para combinação/atualização de datasets.

    Esta instrução descreve:
    - qual é o dataset base
    - qual é o dataset secundário
    - qual modo de merge/atualização será utilizado
    - quais parâmetros controlam a operação
    - qual recurso de saída será produzido

    Modos atualmente suportados:
    - CONCAT
    - CONCAT_PRESERVE_ORDER
    - REPLACE_BY_INDEX_LIST
    - UPDATE_BY_INDEX_LIST
    """

    base_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do dataset base da operação."}
    )

    secondary_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do dataset secundário usado na combinação/atualização."}
    )

    merge_mode: DatasetMergeMode = field(
        default=DatasetMergeMode.CONCAT,
        metadata={"help": "Modo de merge/atualização do dataset."}
    )

    index_list: List[int] = field(
        default_factory=list,
        metadata={"help": "Lista explícita de índices usada nos modos orientados por índice."}
    )

    output_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset resultante."}
    )

    output_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome opcional do dataset resultante."}
    )

    preserve_base_description: bool = field(
        default=True,
        metadata={"help": "Indica se a descrição do dataset base deve ser preservada por padrão."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.DATASET_MERGE

        if not self.name:
            self.name = "Dataset Merge"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        OUT = (StrategyResourceType.DATASET,)
        return [
            ResourceFieldSpec("base_dataset_resource_id", "Dataset Base", DS),
            ResourceFieldSpec("secondary_dataset_resource_id", "Dataset Secundário", DS),
            ResourceFieldSpec("output_dataset_resource_id", "Dataset Resultante", OUT, is_output=True),
        ]

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos
        específicos da instrução.
        """
        inputs = []
        outputs = []

        if self.base_dataset_resource_id:
            inputs.append(self.base_dataset_resource_id)

        if self.secondary_dataset_resource_id:
            inputs.append(self.secondary_dataset_resource_id)

        if self.output_dataset_resource_id:
            outputs.append(self.output_dataset_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    def build_validation_rules(self) -> list:
        """
        Regra dinâmica (não persistida): DATASETS_EMBEDDING_COMPAT entre
        base e secondary — necessário pra CONCAT/REPLACE/UPDATE não
        produzir dataset com schema misturado.
        """
        from faceaudit.validation_rules.datasets_embedding_compat_rule import (
            DatasetsEmbeddingCompatRule,
        )

        if not (self.base_dataset_resource_id and self.secondary_dataset_resource_id):
            return []

        return [
            DatasetsEmbeddingCompatRule(
                rule_id=f"{self.instruction_id}_emb_compat",
                dataset_resource_ids=[
                    self.base_dataset_resource_id,
                    self.secondary_dataset_resource_id,
                ],
            ),
        ]

    @property
    def uses_concat(self) -> bool:
        return self.merge_mode == DatasetMergeMode.CONCAT

    @property
    def uses_concat_preserve_order(self) -> bool:
        return self.merge_mode == DatasetMergeMode.CONCAT_PRESERVE_ORDER

    @property
    def uses_replace_by_index_list(self) -> bool:
        return self.merge_mode == DatasetMergeMode.REPLACE_BY_INDEX_LIST

    @property
    def uses_update_by_index_list(self) -> bool:
        return self.merge_mode == DatasetMergeMode.UPDATE_BY_INDEX_LIST

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - merge_mode
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "merge_mode" in processed_data and processed_data["merge_mode"]:
            processed_data["merge_mode"] = DatasetMergeMode.from_str(
                processed_data["merge_mode"]
            )

        return cls(**processed_data)