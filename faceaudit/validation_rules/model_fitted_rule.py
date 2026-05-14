# faceaudit/validation_rules/model_fitted_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class ModelFittedRule(ValidationRuleBase):
    """
    Verifica que o recurso indicado é um modelo treinado (`is_fitted=True`).

    Aplicável a TRAINED_PIPELINE, TRAINED_SIMILARITY_MODEL e
    TRAINED_CALIBRATION_MODEL — todos expõem o atributo `is_fitted`.
    Sem isso, o executor que consome o modelo crasha em runtime com
    mensagem genérica.

    Parâmetros:
        model_resource_id: id do recurso de modelo treinado
    """

    def __init__(
        self,
        rule_id: str,
        model_resource_id: str,
        name: str = "Modelo treinado",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{model_resource_id}' é um modelo treinado "
                "(is_fitted=True)."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._model_id = model_resource_id

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._model_id):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._model_id}' não encontrado no contexto."
            )
            return result

        # Hidratação preguiçosa — modelo pode estar só com artifact_ref.
        resource = context.ensure_payload(self._model_id)
        payload = resource.payload if resource is not None else None

        if payload is None:
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._model_id}' não tem payload "
                "(falha ao carregar do disco ou ainda não foi produzido)."
            )
            return result

        is_fitted = getattr(payload, "is_fitted", None)
        if is_fitted is None:
            # Modelo sem atributo `is_fitted` — assume OK silenciosamente
            # pra não quebrar tipos atípicos. Preferível a falsa CRITICAL.
            result.mark_ok()
            return result

        if not bool(is_fitted):
            result.add_error(
                f"[{self.rule_id}] Modelo '{self._model_id}' não está treinado "
                "(is_fitted=False). Re-execute a instrução que o produz."
            )
            return result

        result.mark_ok()
        return result
