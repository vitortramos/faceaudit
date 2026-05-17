# faceaudit/models/instructions/score_calibration_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="ScoreCalibrationInstruction")


@dataclass
class ScoreCalibrationInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que aplica um modelo de calibração TREINADO a um
    SimilarityScoreDataset com `raw_score`, produzindo um novo dataset com
    `log_lr` por par.

    O output mantém todas as informações do input (raw_score, same_person,
    probe/reference) e ganha `sample.metadata["log_lr"] = float(log_lr)`.
    A persistência em parquet inclui uma coluna `log_lr` nativa (a classe
    SimilarityScoreDataset estende seu schema automaticamente quando há
    log_lr nos samples).

    Sem hyperparâmetros próprios — toda a configuração já está embutida no
    calibrador treinado.
    """

    trained_calibration_model_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso do tipo TRAINED_CALIBRATION_MODEL a ser aplicado."}
    )

    input_similarity_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada do tipo SIMILARITY_SCORES (raw_score por par)."}
    )

    output_calibrated_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do tipo CALIBRATED_SCORES (log_lr por par)."}
    )

    output_calibrated_scores_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome amigável opcional do dataset de scores calibrados gerado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.SCORE_CALIBRATION

        if not self.name:
            self.name = "Score Calibration"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        TC = (StrategyResourceType.TRAINED_CALIBRATION_MODEL,)
        IN = (StrategyResourceType.SIMILARITY_SCORES,)
        OUT = (StrategyResourceType.CALIBRATED_SCORES,)
        return [
            ResourceFieldSpec(
                "trained_calibration_model_resource_id",
                "Modelo de Calibração",
                TC,
            ),
            ResourceFieldSpec(
                "input_similarity_scores_resource_id",
                "Scores de Entrada",
                IN,
            ),
            ResourceFieldSpec(
                "output_calibrated_scores_resource_id",
                "Scores Calibrados",
                OUT,
                is_output=True,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "Aplica o calibrador treinado aos scores brutos de entrada. "
            "Cada par recebe um log-LR persistido em uma coluna `log_lr` "
            "no parquet de saída. Sem hyperparâmetros próprios — toda a "
            "calibração já está embutida no modelo treinado."
        )

    def build_validation_rules(self) -> list:
        """
        Regras dinâmicas (não persistidas):
        - MODEL_FITTED no calibrador
        - SCORE_CALIBRATION_STATE(RAW) no input — evita dupla calibração
        - SCORE_VECTOR_ORDER_CONSISTENCY — quando o calibrador foi
          treinado em score vector, verifica que a ordem dos sistemas
          do input bate com a do treino.
        """
        from faceaudit.validation_rules.model_fitted_rule import ModelFittedRule
        from faceaudit.validation_rules.score_calibration_state_rule import (
            ScoreCalibrationStateRule,
        )
        from faceaudit.validation_rules.score_vector_order_consistency_rule import (
            ScoreVectorOrderConsistencyRule,
        )

        rules: list = []
        if self.trained_calibration_model_resource_id:
            rules.append(ModelFittedRule(
                rule_id=f"{self.instruction_id}_calibrator_fitted",
                model_resource_id=self.trained_calibration_model_resource_id,
            ))
        if self.input_similarity_scores_resource_id:
            rules.append(ScoreCalibrationStateRule(
                rule_id=f"{self.instruction_id}_scores_raw",
                scores_resource_id=self.input_similarity_scores_resource_id,
                expected="RAW",
            ))
        if (
            self.trained_calibration_model_resource_id
            and self.input_similarity_scores_resource_id
        ):
            rules.append(ScoreVectorOrderConsistencyRule(
                rule_id=f"{self.instruction_id}_score_vector_order_consistency",
                scores_resource_id=self.input_similarity_scores_resource_id,
                calibrator_resource_id=self.trained_calibration_model_resource_id,
            ))
        return rules

    def _sync_declared_io(self) -> None:
        """
        Sincroniza input_resource_ids e output_resource_ids com os campos
        específicos. Ordem: calibrator primeiro, scores depois.
        """
        inputs = []
        outputs = []

        if self.trained_calibration_model_resource_id:
            inputs.append(self.trained_calibration_model_resource_id)

        if self.input_similarity_scores_resource_id:
            inputs.append(self.input_similarity_scores_resource_id)

        if self.output_calibrated_scores_resource_id:
            outputs.append(self.output_calibrated_scores_resource_id)

        self.input_resource_ids = list(dict.fromkeys(inputs))
        self.output_resource_ids = list(dict.fromkeys(outputs))

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO especializado, hidratando:
        - instruction_type
        """
        if not data:
            return None

        processed_data = data.copy()

        if "instruction_type" in processed_data and processed_data["instruction_type"]:
            processed_data["instruction_type"] = StrategyInstructionType.from_str(
                processed_data["instruction_type"]
            )

        return cls(**processed_data)
