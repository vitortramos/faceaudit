# faceaudit/models/instructions/strategy_instruction_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="StrategyInstructionDefinition")


@dataclass(frozen=True)
class ResourceFieldSpec:
    """Especificação de um campo de recurso para renderização no config panel."""
    field_name: str
    label: str
    accepted_types: Tuple[StrategyResourceType, ...]
    is_output: bool = False


@dataclass(frozen=True)
class ChoiceFieldSpec:
    """Especificação de um campo de escolha (enum) para renderização no config panel."""
    field_name: str
    label: str
    choices: Tuple[str, ...]   # valores do enum (strings)
    enum_type: Optional[type] = None  # classe do enum para conversão segura via from_str


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

    def get_choice_field_specs(self) -> List[ChoiceFieldSpec]:
        """Retorna campos de escolha (enums) para renderização no config panel. Override nos DTOs concretos."""
        return []

    def get_resource_field_specs(self) -> List[ResourceFieldSpec]:
        """Retorna specs para renderização no config panel. Override nos DTOs concretos."""
        return []

    def build_validation_rules(self) -> list:
        """
        Retorna regras de validação dinâmicas baseadas no estado atual da instrução.
        Override nos DTOs concretos para regras semânticas específicas.
        """
        return []

    def get_output_resource_placeholders(self) -> list:
        """
        Retorna StrategyResource placeholders para cada campo de output preenchido.
        payload=None, artifact_ref=None — preenchidos ao executar a instrução.
        """
        from faceaudit.models.strategy.strategy_resource import StrategyResource

        placeholders = []
        for spec in self.get_resource_field_specs():
            if not spec.is_output:
                continue
            resource_id = getattr(self, spec.field_name, None)
            if not resource_id:
                continue
            resource_type = (
                spec.accepted_types[0]
                if spec.accepted_types
                else StrategyResourceType.UNKNOWN
            )
            placeholders.append(
                StrategyResource(
                    resource_id=resource_id,
                    name=resource_id,
                    resource_type=resource_type,
                )
            )
        return placeholders

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