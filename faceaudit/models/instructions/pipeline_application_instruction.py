# faceaudit/models/instructions/pipeline_application_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
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

    trajectory_data_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída opcional contendo TransformationTrajectoryData "
                          "para visualização 3D. Auto-preenchido quando "
                          "application_policy.save_trajectory_data=True."}
    )

    def __post_init__(self) -> None:
        """
        Garante o tipo da instrução e sincroniza entradas/saídas declarativas mínimas.
        """
        self.instruction_type = StrategyInstructionType.PIPELINE_APPLICATION

        if not self.name:
            self.name = "Pipeline Application"

        # Auto-preenche o resource_id da trajetória quando a política está
        # configurada para salvar e o usuário ainda não nomeou o recurso.
        if (
            self._should_save_trajectory()
            and not self.trajectory_data_resource_id
        ):
            self.trajectory_data_resource_id = f"{self.instruction_id}_trajectory"

        self._sync_declared_io()

    def _should_save_trajectory(self) -> bool:
        """
        Retorna True se a política de aplicação está configurada para salvar
        os dados de trajetória 3D.
        """
        return bool(
            self.application_policy is not None
            and getattr(self.application_policy, "save_trajectory_data", False)
        )

    @property
    def is_linear(self) -> bool:
        return self.pipeline_mode == PipelineMode.LINEAR

    @property
    def is_dual(self) -> bool:
        return self.pipeline_mode == PipelineMode.DUAL

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        return [
            ChoiceFieldSpec("pipeline_mode", "Modo do Pipeline", ("LINEAR", "DUAL"), PipelineMode),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        DS = (StrategyResourceType.DATASET,)
        TP = (StrategyResourceType.TRAINED_PIPELINE,)
        ID = (StrategyResourceType.DATASET,)
        TRAJ = (StrategyResourceType.TRAJECTORY_DATA,)

        if self.is_linear:
            specs = [
                ResourceFieldSpec("trained_pipeline_resource_id", "Pipeline Treinado", TP),
                ResourceFieldSpec("dataset_resource_id", "Dataset", DS),
                ResourceFieldSpec("transformed_dataset_resource_id", "Dataset Transformado", ID, is_output=True),
            ]
        else:
            specs = [
                ResourceFieldSpec("trained_pipeline_resource_id", "Pipeline Treinado", TP),
                ResourceFieldSpec("source_dataset_resource_id", "Dataset (Source)", DS),
                ResourceFieldSpec("target_dataset_resource_id", "Dataset (Target)", DS),
                ResourceFieldSpec("transformed_source_dataset_resource_id", "Transformado (Source)", ID, is_output=True),
                ResourceFieldSpec("transformed_target_dataset_resource_id", "Transformado (Target)", ID, is_output=True),
            ]

        # Output condicional: só aparece quando a política está configurada
        # para salvar os dados de trajetória 3D.
        if self._should_save_trajectory():
            specs.append(
                ResourceFieldSpec("trajectory_data_resource_id", "Trajetória 3D", TRAJ, is_output=True)
            )

        return specs

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

        # Output de trajetória 3D — só entra na lista se policy.save_trajectory_data
        # estiver ativo e o id já tiver sido atribuído (auto-fill em __post_init__).
        if self._should_save_trajectory() and self.trajectory_data_resource_id:
            outputs.append(self.trajectory_data_resource_id)

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