# faceaudit/models/instructions/pipeline_application_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy


T = TypeVar("T", bound="PipelineApplicationInstruction")


@dataclass
class PipelineApplicationInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para aplicação de um EmbeddingTransformationPipeline treinado.

    Esta instrução descreve:
    - qual pipeline treinado será usado
    - em qual modo ele será aplicado (LINEAR ou DUAL)
    - quais datasets entram na aplicação
    - qual política de aplicação será usada
    - quais recursos de saída serão produzidos
    """

    pipeline_mode: PipelineMode = field(
        default=PipelineMode.LINEAR,
        metadata={"help": "Modo estrutural de aplicação do pipeline."}
    )

    trained_pipeline_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso que referencia o pipeline treinado a ser aplicado."}
    )

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset usado na aplicação linear."}
    )

    source_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset do ramo Source na aplicação dual."}
    )

    target_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset do ramo Target na aplicação dual."}
    )

    application_policy: Optional[PipelineApplicationPolicy] = field(
        default=None,
        metadata={"help": "Política de aplicação do pipeline."}
    )

    transformed_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset transformado na aplicação linear."}
    )

    transformed_source_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset transformado do ramo Source na aplicação dual."}
    )

    transformed_target_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do dataset transformado do ramo Target na aplicação dual."}
    )

    def __post_init__(self) -> None:
        """
        Garante o tipo da instrução e sincroniza entradas/saídas declarativas mínimas.
        """
        self.instruction_type = StrategyInstructionType.PIPELINE_APPLICATION

        if not self.name:
            self.name = "Pipeline Application"

        self._sync_declared_io()

    @property
    def is_linear(self) -> bool:
        return self.pipeline_mode == PipelineMode.LINEAR

    @property
    def is_dual(self) -> bool:
        return self.pipeline_mode == PipelineMode.DUAL

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos específicos
        desta instrução.
        """
        inputs = []
        outputs = []

        if self.trained_pipeline_resource_id:
            inputs.append(self.trained_pipeline_resource_id)

        if self.is_linear:
            if self.dataset_resource_id:
                inputs.append(self.dataset_resource_id)
            if self.transformed_dataset_resource_id:
                outputs.append(self.transformed_dataset_resource_id)
        else:
            if self.source_dataset_resource_id:
                inputs.append(self.source_dataset_resource_id)
            if self.target_dataset_resource_id:
                inputs.append(self.target_dataset_resource_id)

            if self.transformed_source_dataset_resource_id:
                outputs.append(self.transformed_source_dataset_resource_id)
            if self.transformed_target_dataset_resource_id:
                outputs.append(self.transformed_target_dataset_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - pipeline_mode
        - application_policy
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        if "pipeline_mode" in processed_data and processed_data["pipeline_mode"]:
            processed_data["pipeline_mode"] = PipelineMode.from_str(
                processed_data["pipeline_mode"]
            )

        if "application_policy" in processed_data and isinstance(processed_data["application_policy"], dict):
            processed_data["application_policy"] = PipelineApplicationPolicy.from_dict(
                processed_data["application_policy"]
            )

        return cls(**processed_data)