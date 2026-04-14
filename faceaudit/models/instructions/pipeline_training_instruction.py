# faceaudit/models/instructions/pipeline_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy


T = TypeVar("T", bound="PipelineTrainingInstruction")


@dataclass
class PipelineTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento de um EmbeddingTransformationPipeline.

    Esta instrução descreve:
    - o modo do pipeline (LINEAR ou DUAL)
    - quais recursos de configuração devem ser usados
    - quais datasets entram no treino
    - qual política de treino será aplicada
    - qual recurso de saída representará o pipeline treinado
    """

    pipeline_mode: PipelineMode = field(
        default=PipelineMode.LINEAR,
        metadata={"help": "Modo estrutural do pipeline a ser treinado."}
    )

    pipeline_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de configuração do pipeline linear."}
    )

    source_pipeline_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de configuração do ramo Source no modo DUAL."}
    )

    target_pipeline_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de configuração do ramo Target no modo DUAL."}
    )

    dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset usado no treino linear."}
    )

    source_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset do ramo Source no treino dual."}
    )

    target_dataset_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de dataset do ramo Target no treino dual."}
    )

    training_policy: Optional[PipelineTrainingPolicy] = field(
        default=None,
        metadata={"help": "Política neutra de treino do pipeline."}
    )

    trained_pipeline_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Identificador do recurso de saída que representará o pipeline treinado."}
    )

    def __post_init__(self) -> None:
        """
        Garante o tipo da instrução e sincroniza entradas/saídas declarativas mínimas.
        """
        self.instruction_type = StrategyInstructionType.PIPELINE_TRAINING

        if not self.name:
            self.name = "Pipeline Training"

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

        Mantém a instrução consistente tanto como DTO especializado quanto como
        StrategyInstructionDefinition genérica.
        """
        inputs = []
        outputs = []

        if self.is_linear:
            if self.pipeline_config_resource_id:
                inputs.append(self.pipeline_config_resource_id)
            if self.dataset_resource_id:
                inputs.append(self.dataset_resource_id)

        else:
            if self.source_pipeline_config_resource_id:
                inputs.append(self.source_pipeline_config_resource_id)
            if self.target_pipeline_config_resource_id:
                inputs.append(self.target_pipeline_config_resource_id)
            if self.source_dataset_resource_id:
                inputs.append(self.source_dataset_resource_id)
            if self.target_dataset_resource_id:
                inputs.append(self.target_dataset_resource_id)

        if self.trained_pipeline_resource_id:
            outputs.append(self.trained_pipeline_resource_id)

        # Remove duplicados preservando ordem
        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        - pipeline_mode
        - training_policy
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

        if "training_policy" in processed_data and isinstance(processed_data["training_policy"], dict):
            processed_data["training_policy"] = PipelineTrainingPolicy.from_dict(
                processed_data["training_policy"]
            )

        return cls(**processed_data)