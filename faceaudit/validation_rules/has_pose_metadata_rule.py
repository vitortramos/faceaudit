# faceaudit/validation_rules/has_pose_metadata_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class HasPoseMetadataRule(ValidationRuleBase):
    """
    Verifica que o dataset tem colunas de pose (yaw e pitch).

    Aplicável a HUBNESS_ABLATION modo BY_POSE — sem yaw/pitch a estratégia
    de stratification não consegue agrupar samples por faixa de pose.

    Lê do `DatasetStats.column_coverage` — não exige hidratação do dataset
    inteiro.

    Parâmetros:
        dataset_resource_id: id do recurso DATASET
    """

    _REQUIRED_COLUMNS = ("yaw", "pitch")

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        name: str = "Metadata de pose disponível",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_resource_id}' tem colunas yaw e pitch."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        stats = self._get_or_compute_stats(self._dataset_id, context)
        if stats is None:
            result.add_error(
                f"[{self.rule_id}] Dataset '{self._dataset_id}' não encontrado no contexto."
            )
            return result

        coverage_map = getattr(stats, "column_coverage", None) or {}
        missing = [c for c in self._REQUIRED_COLUMNS if c not in coverage_map]
        if missing:
            result.add_error(
                f"[{self.rule_id}] '{self._dataset_id}' não tem coluna(s) de "
                f"pose: {', '.join(missing)}. Modo BY_POSE exige yaw e pitch."
            )
            return result

        # Cobertura também precisa ser não-nula — coluna no schema mas com
        # 0 valores é tão útil quanto coluna ausente.
        for col in self._REQUIRED_COLUMNS:
            cs = coverage_map.get(col)
            cov = float(getattr(cs, "coverage", 0.0) or 0.0)
            if cov <= 0.0:
                result.add_error(
                    f"[{self.rule_id}] Coluna '{col}' em '{self._dataset_id}' "
                    "tem cobertura 0 (todos os valores nulos)."
                )
                return result

        result.mark_ok()
        return result
