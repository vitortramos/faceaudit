# faceaudit/validation_rules/holdout_leakage_rule.py

from __future__ import annotations

from typing import Any, List, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class HoldoutLeakageRule(ValidationRuleBase):
    """
    Verifica que o(s) dataset(s) de hold-out (validação/teste) não compartilham
    identidades com nenhum dos datasets de treino.

    Garante que a avaliação final da estratégia seja feita em dados que o modelo
    nunca viu — condição essencial para estimativas forenses válidas.

    Parâmetros configuráveis:
        holdout_resource_ids: lista de resource_ids dos datasets de hold-out
        train_resource_ids: lista de resource_ids dos datasets de treino
    """

    def __init__(
        self,
        rule_id: str,
        holdout_resource_ids: List[str],
        train_resource_ids: List[str],
        name: str = "Holdout Leakage Check",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                "Verifica que os datasets de hold-out não compartilham identidades "
                "com nenhum dataset de treino."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._holdout_ids = holdout_resource_ids
        self._train_ids = train_resource_ids

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        # Carrega stats de todos os datasets envolvidos
        holdout_stats = {}
        for rid in self._holdout_ids:
            stats = self._get_or_compute_stats(rid, context)
            if stats is None:
                result.add_error(
                    f"[{self.rule_id}] Dataset de hold-out '{rid}' não encontrado no contexto."
                )
            else:
                holdout_stats[rid] = stats

        train_stats = {}
        for rid in self._train_ids:
            stats = self._get_or_compute_stats(rid, context)
            if stats is None:
                result.add_error(
                    f"[{self.rule_id}] Dataset de treino '{rid}' não encontrado no contexto."
                )
            else:
                train_stats[rid] = stats

        if not result.is_valid:
            return result

        # Verifica cruzamentos hold-out × treino
        total_leakage = 0
        for h_id, h_stats in holdout_stats.items():
            for t_id, t_stats in train_stats.items():
                overlap = h_stats.identity_set & t_stats.identity_set
                if overlap:
                    total_leakage += len(overlap)
                    result.add_error(
                        f"[{self.rule_id}] LEAKAGE: {len(overlap)} identidade(s) do "
                        f"hold-out '{h_id}' aparecem no dataset de treino '{t_id}'. "
                        "A avaliação será otimisticamente enviesada."
                    )

        if total_leakage == 0:
            result.mark_ok()

        return result
