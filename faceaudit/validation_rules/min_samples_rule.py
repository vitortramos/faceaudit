# faceaudit/validation_rules/min_samples_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class MinSamplesRule(ValidationRuleBase):
    """
    Verifica que o dataset tem pelo menos `min_required` amostras totais.

    Aplicável a:
    - HUBNESS_ANALYSIS / HUBNESS_ABLATION (≥ 2 pra fazer sentido topológico)
    - DATASET_SPLIT modos BY_INDEX_*
    - PIPELINE_TRAINING com Whitening/Centering (covariância estável exige
      idealmente N ≥ embedding_dim)

    Parâmetros:
        dataset_resource_id: id do recurso DATASET
        min_required: contagem mínima absoluta (default 2)
        safety_threshold: emite WARNING se contagem ≥ min_required mas
                          < safety_threshold (default 100)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        min_required: int = 2,
        safety_threshold: int = 100,
        name: str = "Mínimo de amostras",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_resource_id}' tem pelo menos "
                f"{min_required} amostra(s)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._min_required = int(min_required)
        self._safety_threshold = int(safety_threshold)

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        stats = self._get_or_compute_stats(self._dataset_id, context)
        if stats is None:
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._dataset_id}' não encontrado no contexto."
            )
            return result

        n_imgs = int(getattr(stats, "n_imgs", 0) or 0)

        if n_imgs < self._min_required:
            result.add_error(
                f"[{self.rule_id}] Amostras insuficientes em "
                f"'{self._dataset_id}': {n_imgs} (mínimo {self._min_required})."
            )
            return result

        if n_imgs < self._safety_threshold:
            result.add_warning(
                f"[{self.rule_id}] '{self._dataset_id}' tem apenas {n_imgs} "
                f"amostra(s) (recomendado: ≥ {self._safety_threshold})."
            )
        else:
            result.mark_ok()

        return result
