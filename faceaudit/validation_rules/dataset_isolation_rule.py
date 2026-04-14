# faceaudit/validation_rules/dataset_isolation_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class DatasetIsolationRule(ValidationRuleBase):
    """
    Verifica que dois datasets são completamente disjuntos em identidades.

    Uso típico: garantir que o dataset de calibração não compartilhe identidades
    com o dataset de treino do modelo de similaridade (data leakage).

    Severidade padrão: CRITICAL — leakage invalida completamente os resultados.
    """

    def __init__(
        self,
        rule_id: str,
        dataset_a_resource_id: str,
        dataset_b_resource_id: str,
        name: str = "Dataset Isolation (No Leakage)",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_a_resource_id}' e '{dataset_b_resource_id}' "
                "não compartilham identidades (sem data leakage)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._id_a = dataset_a_resource_id
        self._id_b = dataset_b_resource_id

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        stats_a = self._get_or_compute_stats(self._id_a, context)
        stats_b = self._get_or_compute_stats(self._id_b, context)

        if stats_a is None:
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._id_a}' não encontrado no contexto."
            )
        if stats_b is None:
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._id_b}' não encontrado no contexto."
            )
        if not result.is_valid:
            return result

        overlap = stats_a.identity_set & stats_b.identity_set
        overlap_count = len(overlap)

        if overlap_count > 0:
            result.add_error(
                f"[{self.rule_id}] LEAKAGE: {overlap_count} identidade(s) aparecem em ambos "
                f"'{self._id_a}' e '{self._id_b}'. Os datasets devem ser completamente disjuntos."
            )
        else:
            result.mark_ok()

        return result
