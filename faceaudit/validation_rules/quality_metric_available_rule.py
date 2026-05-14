# faceaudit/validation_rules/quality_metric_available_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class QualityMetricAvailableRule(ValidationRuleBase):
    """
    Verifica que a métrica de qualidade escolhida tem cobertura suficiente
    no dataset.

    Aplicável a DATASET_BALANCING modo BY_QUALITY — sem cobertura mínima
    a seleção fica pesada para um subset enviesado dos samples.

    A cobertura é lida do `DatasetStats.column_coverage` (populado pelo
    `DatasetStatsService` a partir do parquet metadata). Para a métrica
    EMBEDDING_NORM (computada on-the-fly), a regra é trivialmente OK
    (a métrica está sempre disponível se há embeddings).

    Parâmetros:
        dataset_resource_id: id do recurso DATASET
        quality_metric: nome da métrica (ex: "fiqa_score", "ofiq_score",
                        "serfiq_score", "embedding_norm")
        min_coverage: cobertura mínima (default 0.5 = 50%)
    """

    _ALWAYS_AVAILABLE = {"embedding_norm", "EMBEDDING_NORM"}

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        quality_metric: str,
        min_coverage: float = 0.5,
        name: str = "Métrica de qualidade disponível",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{quality_metric}' tem cobertura ≥ "
                f"{min_coverage:.0%} em '{dataset_resource_id}'."
            ),
            default_severity=ValidationSeverity.WARNING,
        )
        self._dataset_id = dataset_resource_id
        self._metric = str(quality_metric)
        self._min_coverage = float(min_coverage)

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        # EMBEDDING_NORM é computada on-the-fly — sempre disponível.
        if self._metric in self._ALWAYS_AVAILABLE:
            result.mark_ok()
            return result

        stats = self._get_or_compute_stats(self._dataset_id, context)
        if stats is None:
            result.add_error(
                f"[{self.rule_id}] Dataset '{self._dataset_id}' não encontrado no contexto."
            )
            return result

        coverage_map = getattr(stats, "column_coverage", None) or {}
        col_stats = coverage_map.get(self._metric)
        if col_stats is None:
            result.add_warning(
                f"[{self.rule_id}] Coluna '{self._metric}' ausente em "
                f"'{self._dataset_id}'. Disponíveis: "
                f"{sorted(coverage_map.keys())}."
            )
            return result

        coverage = float(getattr(col_stats, "coverage", 0.0) or 0.0)
        if coverage < self._min_coverage:
            result.add_warning(
                f"[{self.rule_id}] Cobertura de '{self._metric}' em "
                f"'{self._dataset_id}' é {coverage:.1%}, abaixo do mínimo "
                f"{self._min_coverage:.0%}. Balanceamento será enviesado."
            )
            return result

        result.mark_ok()
        return result
