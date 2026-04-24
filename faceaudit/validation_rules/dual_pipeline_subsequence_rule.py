# faceaudit/validation_rules/dual_pipeline_subsequence_rule.py

from __future__ import annotations

from typing import Any

from faceaudit.enums.transformation_type import TransformationType
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class DualPipelineSubsequenceRule(ValidationRuleBase):
    """
    Verifica que os tipos do pipeline target são subsequência ordenada do source,
    sem CORAL.

    Regras:
    - O target NÃO pode conter CORAL_TRANSFORMATION (exclusivo do source)
    - Os tipos do target devem ser uma subsequência ordenada dos tipos do source
      (excluindo CORAL do source)
    """

    def __init__(
        self,
        rule_id: str,
        source_config_resource_id: str,
        target_config_resource_id: str,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name="Subsequência DUAL Pipeline",
            description=(
                f"Verifica que o pipeline target ('{target_config_resource_id}') "
                f"é subsequência do source ('{source_config_resource_id}') sem CORAL."
            ),
        )
        self._source_id = source_config_resource_id
        self._target_id = target_config_resource_id

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._source_id) or not context.has_resource(self._target_id):
            # Existência já verificada por _validate_instruction_specifics
            result.mark_ok()
            return result

        source_payload = context.get_resource(self._source_id).payload
        target_payload = context.get_resource(self._target_id).payload

        if (
            not isinstance(source_payload, TransformationPipelineConfig)
            or not isinstance(target_payload, TransformationPipelineConfig)
        ):
            # Tipo já verificado por _validate_instruction_specifics
            result.mark_ok()
            return result

        source_types = [t.type for t in source_payload.transformators]
        target_types = [t.type for t in target_payload.transformators]

        # Target não pode conter CORAL
        if any(t == TransformationType.CORAL_TRANSFORMATION for t in target_types):
            result.add_error(
                f"[{self.rule_id}] O pipeline Target não pode conter CORAL_TRANSFORMATION. "
                f"CORAL é exclusivo do ramo Source (adaptação de domínio)."
            )

        # Target deve ser subsequência ordenada de source (excluindo CORAL do source)
        source_non_coral = [
            t for t in source_types if t != TransformationType.CORAL_TRANSFORMATION
        ]
        ptr = 0
        for tgt_type in target_types:
            if tgt_type == TransformationType.CORAL_TRANSFORMATION:
                continue  # já reportado acima
            found = False
            while ptr < len(source_non_coral):
                if source_non_coral[ptr] == tgt_type:
                    ptr += 1
                    found = True
                    break
                ptr += 1
            if not found:
                source_names = [self._type_display(t) for t in source_non_coral]
                target_names = [self._type_display(t) for t in target_types]
                result.add_error(
                    f"[{self.rule_id}] O pipeline Target não é subsequência válida "
                    f"do Source (sem CORAL). "
                    f"Source (sem CORAL): {source_names}, Target: {target_names}"
                )
                break

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    @staticmethod
    def _type_display(t: TransformationType) -> str:
        """Retorna o nome legível do tipo de transformação."""
        try:
            return t.value[1]  # display name da tupla do enum
        except (IndexError, TypeError):
            return str(t)
