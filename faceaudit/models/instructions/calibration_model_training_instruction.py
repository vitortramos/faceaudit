# faceaudit/models/instructions/calibration_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.calibration_model_type import CalibrationModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="CalibrationModelTrainingInstruction")


@dataclass
class CalibrationModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de calibração.
    """

    calibration_model_type: CalibrationModelType = field(
        default=CalibrationModelType.LOGISTIC_REGRESSION,
        metadata={"help": "Tipo de modelo de calibração a ser usado."}
    )

    input_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada usado no treino/preparação do calibrador."}
    )

    calibration_model_config_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de configuração do modelo de calibração."}
    )

    trained_calibration_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o calibrador treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.CALIBRATION_MODEL_TRAINING

        if not self.name:
            self.name = "Calibration Model Training"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        ART = (StrategyResourceType.ARTIFACT,)
        CC = (StrategyResourceType.CALIBRATION_MODEL_CONFIG,)
        TC = (StrategyResourceType.TRAINED_CALIBRATION_MODEL,)
        return [
            ResourceFieldSpec("input_resource_id", "Dados de Entrada (scores)", ART),
            ResourceFieldSpec("calibration_model_config_resource_id", "Config do Calibrador", CC),
            ResourceFieldSpec("trained_calibration_model_resource_id", "Calibrador Treinado", TC, is_output=True),
        ]

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.input_resource_id:
            inputs.append(self.input_resource_id)

        if self.calibration_model_config_resource_id:
            inputs.append(self.calibration_model_config_resource_id)

        if self.trained_calibration_model_resource_id:
            outputs.append(self.trained_calibration_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @property
    def uses_logistic_regression(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.LOGISTIC_REGRESSION

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
            "calibration_model_type" in processed_data
            and processed_data["calibration_model_type"]
        ):
            processed_data["calibration_model_type"] = CalibrationModelType.from_str(
                processed_data["calibration_model_type"]
            )

        return cls(**processed_data)