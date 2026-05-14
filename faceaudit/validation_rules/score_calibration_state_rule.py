# faceaudit/validation_rules/score_calibration_state_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.similarity.similarity_score_dataset import (
    SimilarityScoreDataset,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class ScoreCalibrationStateRule(ValidationRuleBase):
    """
    Verifica se o score dataset está no estado de calibração esperado.

    Detecta calibração via presença de `log_lr` no metadata dos items —
    `SimilarityScoreDataset` calibrado tem `metadata["log_lr"]` populado
    em cada sample (gravado pelo ScoreCalibrationInstructionExecutor).

    Aplicável a:
    - CALIBRATION_MODEL_TRAINING (espera RAW; evita treinar sobre log-LR
      já calibrado, distorcendo o modelo)
    - SCORE_CALIBRATION (espera RAW; evita dupla calibração)
    - STRATEGY_EVALUATION (espera CALIBRATED; métricas só fazem sentido
      em log-LR)

    Parâmetros:
        scores_resource_id: id do recurso de score dataset
        expected: "RAW" ou "CALIBRATED" — estado esperado
    """

    _EXPECTED_VALUES = ("RAW", "CALIBRATED")

    def __init__(
        self,
        rule_id: str,
        scores_resource_id: str,
        expected: str = "RAW",
        name: str = "Estado de calibração do score dataset",
        description: Optional[str] = None,
    ) -> None:
        expected_norm = str(expected).strip().upper()
        if expected_norm not in self._EXPECTED_VALUES:
            raise ValueError(
                f"ScoreCalibrationStateRule: expected deve ser um de "
                f"{self._EXPECTED_VALUES}, recebido '{expected}'."
            )

        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{scores_resource_id}' está {expected_norm}."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._scores_id = scores_resource_id
        self._expected = expected_norm

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._scores_id):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._scores_id}' não encontrado no contexto."
            )
            return result

        resource = context.ensure_payload(self._scores_id)
        payload = resource.payload if resource is not None else None
        if payload is None:
            result.add_warning(
                f"[{self.rule_id}] Payload de '{self._scores_id}' indisponível. "
                "Pulando checagem de estado."
            )
            return result

        if not isinstance(payload, SimilarityScoreDataset):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._scores_id}' não é um "
                f"SimilarityScoreDataset (tipo: {type(payload).__name__})."
            )
            return result

        # Amostragem rápida: em datasets grandes (5M+) iterar todos os
        # items é caro. Suficiente checar os primeiros 100 — se nenhum
        # tem log_lr, é RAW; se algum tem, é CALIBRATED.
        SAMPLE = 100
        has_log_lr = False
        for s in payload.items[:SAMPLE]:
            md = s.metadata if isinstance(s.metadata, dict) else None
            if md is not None and md.get("log_lr") is not None:
                has_log_lr = True
                break

        actual_state = "CALIBRATED" if has_log_lr else "RAW"

        if actual_state != self._expected:
            if self._expected == "RAW":
                result.add_error(
                    f"[{self.rule_id}] '{self._scores_id}' já está CALIBRATED "
                    "(log_lr presente). Re-calibrar produziria scores "
                    "inválidos — use um score dataset RAW como entrada."
                )
            else:
                result.add_error(
                    f"[{self.rule_id}] '{self._scores_id}' está RAW (sem log_lr). "
                    "Esta instrução exige scores CALIBRATED — adicione um "
                    "SCORE_CALIBRATION antes."
                )
            return result

        result.mark_ok()
        return result
