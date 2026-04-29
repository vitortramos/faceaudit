# faceaudit/models/instructions/strategy_evaluation_instruction.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.instructions.strategy_instruction_definition import (
    ResourceFieldSpec,
    StrategyInstructionDefinition,
)


T = TypeVar("T", bound="StrategyEvaluationInstruction")


@dataclass
class StrategyEvaluationInstruction(StrategyInstructionDefinition):
    """
    Instrução declarativa que avalia a qualidade forense da estratégia
    a partir de um SimilarityScoreDataset CALIBRADO (CALIBRATED_SCORES,
    com `metadata["log_lr"]` populado em cada sample) e produz um
    StrategyValidationResult contendo:

    - métricas escalares (EER + threshold, Cllr, Cllr_min via PAVA, AUC);
    - dados de plot prontos (DET FPR/FNR, log-LRs gen/imp para Tippett,
      eixo do ECE com curvas real e mínima).

    Sem hyperparâmetros próprios — o cálculo é determinístico via
    `ValidationMathService.calculate_metrics`.
    """

    input_calibrated_scores_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de entrada do tipo CALIBRATED_SCORES (log_lr por par)."}
    )

    output_validation_result_resource_id: Optional[str] = field(
        default=None,
        metadata={"help": "Recurso de saída do tipo VALIDATION_RESULT (StrategyValidationResult em pickle)."}
    )

    output_validation_result_name: Optional[str] = field(
        default=None,
        metadata={"help": "Nome amigável opcional do resultado de validação gerado."}
    )

    def __post_init__(self) -> None:
        self.instruction_type = StrategyInstructionType.STRATEGY_EVALUATION

        if not self.name:
            self.name = "Strategy Evaluation"

        self._sync_declared_io()

    def get_resource_field_specs(self) -> list[ResourceFieldSpec]:
        IN = (StrategyResourceType.CALIBRATED_SCORES,)
        OUT = (StrategyResourceType.VALIDATION_RESULT,)
        return [
            ResourceFieldSpec(
                "input_calibrated_scores_resource_id",
                "Scores Calibrados",
                IN,
            ),
            ResourceFieldSpec(
                "output_validation_result_resource_id",
                "Resultado de Validação",
                OUT,
                is_output=True,
            ),
        ]

    def get_configuration_help(self) -> Optional[str]:
        return (
            "Calcula EER, Cllr, Cllr_min (via PAVA), AUC e dados de plot "
            "(DET, Tippett, ECE) a partir dos log-LRs calibrados. Sem "
            "hyperparâmetros — métricas são determinísticas."
        )

    def _sync_declared_io(self) -> None:
        """Sincroniza input_resource_ids e output_resource_ids."""
        inputs = []
        outputs = []

        if self.input_calibrated_scores_resource_id:
            inputs.append(self.input_calibrated_scores_resource_id)

        if self.output_validation_result_resource_id:
            outputs.append(self.output_validation_result_resource_id)

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
