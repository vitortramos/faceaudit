# faceaudit/validation_rules/casework_state_rule.py

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


class CaseworkStateRule(ValidationRuleBase):
    """
    Verifica que o score dataset tem `metadata.is_casework` no estado esperado.

    Aplicável a:
    - STRATEGY_EVALUATION (espera False — métricas exigem ground truth)
    - CASEWORK_VISUALIZATION (espera False na referência, True no caso)

    Parâmetros:
        scores_resource_id: id do recurso de score dataset
        expected: True (deve ser casework) ou False (deve NÃO ser casework)
    """

    def __init__(
        self,
        rule_id: str,
        scores_resource_id: str,
        expected: bool,
        name: str = "Estado de casework do score dataset",
        description: Optional[str] = None,
    ) -> None:
        expected_bool = bool(expected)
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{scores_resource_id}' tem "
                f"is_casework={expected_bool}."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._scores_id = scores_resource_id
        self._expected = expected_bool

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
                "Pulando checagem de casework."
            )
            return result

        if not isinstance(payload, SimilarityScoreDataset):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._scores_id}' não é um "
                f"SimilarityScoreDataset (tipo: {type(payload).__name__})."
            )
            return result

        actual = bool((payload.metadata or {}).get("is_casework", False))
        if actual != self._expected:
            if self._expected:
                result.add_error(
                    f"[{self.rule_id}] '{self._scores_id}' tem is_casework=False, "
                    "esperado True (este recurso deve representar o caso "
                    "forense em análise)."
                )
            else:
                result.add_error(
                    f"[{self.rule_id}] '{self._scores_id}' tem is_casework=True, "
                    "esperado False (este recurso deve ser de validação com "
                    "ground truth, não caso forense)."
                )
            return result

        result.mark_ok()
        return result
