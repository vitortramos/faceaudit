# faceaudit/validation_rules/min_calibration_pairs_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class MinCalibrationPairsRule(ValidationRuleBase):
    """
    Verifica que o dataset de scores possui pares genuínos e impostores suficientes
    para treinar o modelo de calibração.

    Sem pares suficientes a Regressão Logística ou o KDE não conseguem ajustar
    uma curva estável, produzindo LRs sem significado forense.

    Parâmetros configuráveis:
        scores_resource_id: recurso SimilarityScoreDataset com os scores
        min_genuines: mínimo de pares genuínos (default: 100)
        min_impostors: mínimo de pares impostores (default: 500)
        warn_ratio_threshold: emite WARNING se genuínos/(genuínos+impostores) < limiar
                              (default: 0.02 = 2%)
    """

    def __init__(
        self,
        rule_id: str,
        scores_resource_id: str,
        min_genuines: int = 100,
        min_impostors: int = 500,
        warn_ratio_threshold: float = 0.02,
        name: str = "Minimum Calibration Pairs",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{scores_resource_id}' possui pares genuínos e impostores "
                "suficientes para treinar o modelo de calibração."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._scores_id = scores_resource_id
        self._min_genuines = min_genuines
        self._min_impostors = min_impostors
        self._warn_ratio_threshold = warn_ratio_threshold

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

        # Hidratação preguiçosa: scores podem ter milhões de pares; só carrega
        # quando a regra precisa contar genuínos/impostores.
        resource = context.ensure_payload(self._scores_id)
        payload = resource.payload if resource is not None else None

        if not isinstance(payload, SimilarityScoreDataset):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._scores_id}' deve ser "
                f"SimilarityScoreDataset, mas é {type(payload).__name__}."
            )
            return result

        n_genuines = payload.total_genuines
        n_impostors = payload.total_impostors

        if n_genuines < self._min_genuines:
            result.add_error(
                f"[{self.rule_id}] Pares genuínos insuficientes: {n_genuines} disponíveis, "
                f"mínimo exigido: {self._min_genuines}. "
                "A curva de genuínos do calibrador será estatisticamente instável."
            )

        if n_impostors < self._min_impostors:
            result.add_error(
                f"[{self.rule_id}] Pares impostores insuficientes: {n_impostors} disponíveis, "
                f"mínimo exigido: {self._min_impostors}. "
                "A curva de impostores do calibrador será estatisticamente instável."
            )

        if not result.is_valid:
            return result

        # Warning: proporção de genuínos muito baixa
        total = n_genuines + n_impostors
        if total > 0:
            ratio = n_genuines / total
            if ratio < self._warn_ratio_threshold:
                result.add_warning(
                    f"[{self.rule_id}] Proporção de genuínos muito baixa: "
                    f"{n_genuines}/{total} ({ratio:.1%}). "
                    "Dataset fortemente desbalanceado — considere aplicar PAIR_MINING "
                    "antes do treino de calibração."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
