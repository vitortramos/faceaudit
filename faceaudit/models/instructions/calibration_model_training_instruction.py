# faceaudit/models/instructions/calibration_model_training_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.calibration_model_type import CalibrationModelType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.calibration.kde_calibration_model_config import (
    KDECalibrationModelConfig,
)
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    ChoiceFieldSpec,
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="CalibrationModelTrainingInstruction")


@dataclass
class CalibrationModelTrainingInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa para treinamento/preparação de um modelo de calibração.

    A config do calibrador é embutida no DTO da instrução, em campos específicos
    para cada tipo (logreg_config / kde_config). Apenas o config do tipo
    selecionado é usado pelo executor.
    """

    calibration_model_type: CalibrationModelType = field(
        default=CalibrationModelType.LOGISTIC_REGRESSION,
        metadata={"help": "Tipo de modelo de calibração a ser usado."}
    )

    input_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada usado no treino/preparação do calibrador."}
    )

    logreg_config: Optional[LogisticRegressionCalibrationModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do calibrador por Regressão Logística (quando esse tipo é selecionado)."}
    )

    kde_config: Optional[KDECalibrationModelConfig] = field(
        default=None,
        metadata={"help": "Hyperparâmetros do calibrador KDE (quando esse tipo é selecionado)."}
    )

    trained_calibration_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída que representará o calibrador treinado/preparado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.CALIBRATION_MODEL_TRAINING

        if not self.name:
            self.name = "Calibration Model Training"

        # Auto-instancia o config relevante. O config "do outro tipo" pode
        # permanecer no DTO se já tiver sido preenchido — útil para o usuário
        # alternar entre tipos sem perder valores ajustados anteriormente.
        if self.uses_logistic_regression and self.logreg_config is None:
            self.logreg_config = LogisticRegressionCalibrationModelConfig()
        if self.uses_kde and self.kde_config is None:
            self.kde_config = KDECalibrationModelConfig()

        self._sync_declared_io()

    def get_choice_field_specs(self) -> list[ChoiceFieldSpec]:
        triples = CalibrationModelType.choices_with_description()
        return [
            ChoiceFieldSpec(
                field_name="calibration_model_type",
                label="Tipo do Modelo",
                choices=tuple(value for value, _label, _desc in triples),
                enum_type=CalibrationModelType,
                descriptions={value: desc for value, _label, desc in triples},
            ),
        ]

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (
            StrategyResourceType.SIMILARITY_SCORES,
            StrategyResourceType.ARTIFACT,
        )
        TC = (StrategyResourceType.TRAINED_CALIBRATION_MODEL,)
        return [
            ResourceFieldSpec("input_resource_id", "Scores de Entrada", IN),
            ResourceFieldSpec(
                "trained_calibration_model_resource_id",
                "Calibrador Treinado",
                TC,
                is_output=True,
            ),
        ]

    def _sync_declared_io(self) -> None:
        inputs = []
        outputs = []

        if self.input_resource_id:
            inputs.append(self.input_resource_id)

        if self.trained_calibration_model_resource_id:
            outputs.append(self.trained_calibration_model_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @property
    def uses_logistic_regression(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.LOGISTIC_REGRESSION

    @property
    def uses_kde(self) -> bool:
        return self.calibration_model_type == CalibrationModelType.KDE

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

        if "logreg_config" in processed_data and isinstance(
            processed_data["logreg_config"], dict
        ):
            processed_data["logreg_config"] = (
                LogisticRegressionCalibrationModelConfig.from_dict(
                    processed_data["logreg_config"]
                )
            )

        if "kde_config" in processed_data and isinstance(
            processed_data["kde_config"], dict
        ):
            processed_data["kde_config"] = KDECalibrationModelConfig.from_dict(
                processed_data["kde_config"]
            )

        # Retrocompat: campo removido
        processed_data.pop("calibration_model_config_resource_id", None)

        return cls(**processed_data)
