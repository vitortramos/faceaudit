# faceaudit/validation_rules/dataset_overlap_required_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class DatasetOverlapRequiredRule(ValidationRuleBase):
    """
    Verifica que dois datasets compartilham um mínimo de identidades em comum.

    Uso típico: garantir que o dataset Probe e o dataset Reference têm identidades
    em comum para que pares genuínos possam ser formados (KDE, scoring cruzado).

    Parâmetros configuráveis:
        min_overlap: número mínimo absoluto de identidades em comum (default: 1)
        warn_below_ratio: emite WARNING se overlap < essa fração do menor dataset
                          (default: 0.1 = 10%)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_a_resource_id: str,
        dataset_b_resource_id: str,
        min_overlap: int = 1,
        warn_below_ratio: float = 0.1,
        name: str = "Dataset Overlap Required",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_a_resource_id}' e '{dataset_b_resource_id}' "
                f"compartilham no mínimo {min_overlap} identidade(s) em comum."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._id_a = dataset_a_resource_id
        self._id_b = dataset_b_resource_id
        self._min_overlap = min_overlap
        self._warn_below_ratio = warn_below_ratio

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

        if overlap_count < self._min_overlap:
            result.add_error(
                f"[{self.rule_id}] Sobreposição insuficiente: apenas {overlap_count} identidade(s) "
                f"em comum entre '{self._id_a}' ({stats_a.n_ids} IDs) e "
                f"'{self._id_b}' ({stats_b.n_ids} IDs). "
                f"Mínimo exigido: {self._min_overlap}. "
                "Sem identidades em comum não é possível formar pares genuínos."
            )
            return result

        # Warning se overlap é baixo em relação ao menor dataset
        smaller_n_ids = min(stats_a.n_ids, stats_b.n_ids)
        if smaller_n_ids > 0:
            ratio = overlap_count / smaller_n_ids
            if ratio < self._warn_below_ratio:
                result.add_warning(
                    f"[{self.rule_id}] Sobreposição baixa: {overlap_count} identidade(s) em comum "
                    f"representam apenas {ratio:.1%} do menor dataset ({smaller_n_ids} IDs). "
                    "A curva de genuínos pode ser instável."
                )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
