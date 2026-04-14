# faceaudit/models/instructions/strategy_instruction_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="StrategyInstructionDefinition")


@dataclass
class StrategyInstructionDefinition(JSONSerializableMixin):
    """
    Representa a definição declarativa de uma instrução da estratégia.
    """

    instruction_id: str = field(
        metadata={"help": "Identificador único da instrução dentro da estratégia."}
    )
    name: str = field(
        metadata={"help": "Nome amigável da instrução."}
    )
    instruction_type: StrategyInstructionType = field(
        metadata={"help": "Tipo da instrução."}
    )

    input_resource_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de recursos de entrada consumidos pela instrução."}
    )
    output_resource_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de identificadores de recursos produzidos pela instrução."}
    )

    parameters: Dict[str, Any] = field(
        default_factory=dict,
        metadata={"help": "Parâmetros específicos da instrução."}
    )

    validation_rule_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de identificadores das regras de validação associadas."}
    )

    description: Optional[str] = field(
        default=None,
        metadata={"help": "Descrição opcional da finalidade da instrução."}
    )

    @property
    def has_inputs(self) -> bool:
        return bool(self.input_resource_ids)

    @property
    def has_outputs(self) -> bool:
        return bool(self.output_resource_ids)

    @property
    def has_validation_rules(self) -> bool:
        return bool(self.validation_rule_ids)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói a instrução a partir de um dicionário, realizando dispatch
        para o DTO especializado correto com base em instruction_type.
        """
        if not data:
            return None

        processed_data = data.copy()

        raw_type = processed_data.get("instruction_type")
        if raw_type:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(raw_type)

        instruction_type = processed_data.get("instruction_type")

        if instruction_type == StrategyInstructionType.PIPELINE_TRAINING:
            from faceaudit.models.instructions.pipeline_training_instruction import (
                PipelineTrainingInstruction,
            )
            return PipelineTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.PIPELINE_APPLICATION:
            from faceaudit.models.instructions.pipeline_application_instruction import (
                PipelineApplicationInstruction,
            )
            return PipelineApplicationInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.DATASET_SPLIT:
            from faceaudit.models.instructions.dataset_split_instruction import (
                DatasetSplitInstruction,
            )
            return DatasetSplitInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.DATASET_MERGE:
            from faceaudit.models.instructions.dataset_merge_instruction import (
                DatasetMergeInstruction,
            )
            return DatasetMergeInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SIMILARITY_MODEL_TRAINING:
            from faceaudit.models.instructions.similarity_model_training_instruction import (
                SimilarityModelTrainingInstruction,
            )
            return SimilarityModelTrainingInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.SIMILARITY_SCORING:
            from faceaudit.models.instructions.similarity_scoring_instruction import (
                SimilarityScoringInstruction,
            )
            return SimilarityScoringInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.PAIR_MINING:
            from faceaudit.models.instructions.pair_mining_instruction import (
                PairMiningInstruction,
            )
            return PairMiningInstruction.from_dict(processed_data)

        if instruction_type == StrategyInstructionType.CALIBRATION_MODEL_TRAINING:
            from faceaudit.models.instructions.calibration_model_training_instruction import (
                CalibrationModelTrainingInstruction,
            )
            return CalibrationModelTrainingInstruction.from_dict(processed_data)

        return cls(**processed_data)