# faceaudit/validation_rules/max_samples_warning_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class MaxSamplesWarningRule(ValidationRuleBase):
    """
    Emite WARNING quando o dataset excede um threshold de tamanho — útil
    pra avisar o usuário sobre custo computacional alto.

    Aplicável a HUBNESS_ANALYSIS / HUBNESS_ABLATION (O(N²) e O(K·N²)
    respectivamente). NUNCA emite CRITICAL — é só aviso, não bloqueia
    execução.

    Parâmetros:
        dataset_resource_id: id do recurso DATASET
        max_samples: threshold acima do qual avisa (default 100_000)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        max_samples: int = 100_000,
        name: str = "Aviso de tamanho excessivo",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Avisa se '{dataset_resource_id}' excede {max_samples:,} amostras."
            ),
            default_severity=ValidationSeverity.WARNING,
        )
        self._dataset_id = dataset_resource_id
        self._max_samples = int(max_samples)

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        stats = self._get_or_compute_stats(self._dataset_id, context)
        if stats is None:
            # Sem stats não dá pra avisar — não escala a CRITICAL por isso.
            result.mark_ok()
            return result

        n_imgs = int(getattr(stats, "n_imgs", 0) or 0)

        if n_imgs > self._max_samples:
            result.add_warning(
                f"[{self.rule_id}] '{self._dataset_id}' tem {n_imgs:,} amostras "
                f"(> {self._max_samples:,}). Custo computacional pode ser "
                "elevado — considere amostrar antes."
            )
            return result

        result.mark_ok()
        return result
