# faceaudit/validation_rules/dual_pipeline_dimension_compatibility_rule.py

from __future__ import annotations

from typing import Any

from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class DualPipelineDimensionCompatibilityRule(ValidationRuleBase):
    """
    Verifica que source e target pipeline configs produzem a mesma dimensão de saída.

    Uso típico: modo DUAL do PIPELINE_TRAINING — source e target devem produzir
    embeddings com a mesma dimensionalidade para que o scoring cruzado funcione.
    """

    def __init__(
        self,
        rule_id: str,
        source_config_resource_id: str,
        target_config_resource_id: str,
        embedding_dimensions: int | None = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name="Compatibilidade de Dimensões DUAL",
            description=(
                f"Verifica que '{source_config_resource_id}' e '{target_config_resource_id}' "
                f"produzem a mesma dimensão de saída."
            ),
        )
        self._source_id = source_config_resource_id
        self._target_id = target_config_resource_id
        self._embedding_dimensions = embedding_dimensions

    def validate(
        self,
        target: Any,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        result = ValidationResult.neutral()

        if not context.has_resource(self._source_id):
            result.add_error(
                f"[{self.rule_id}] Config source '{self._source_id}' não encontrada no contexto."
            )
            return result
        if not context.has_resource(self._target_id):
            result.add_error(
                f"[{self.rule_id}] Config target '{self._target_id}' não encontrada no contexto."
            )
            return result

        # Hidratação preguiçosa: pipeline configs são JSONs pequenos —
        # custo é baixo e ocorre só quando a regra é avaliada.
        source_resource = context.ensure_payload(self._source_id)
        target_resource = context.ensure_payload(self._target_id)
        source_payload = source_resource.payload if source_resource is not None else None
        target_payload = target_resource.payload if target_resource is not None else None

        if not isinstance(source_payload, TransformationPipelineConfig):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._source_id}' não contém "
                "um TransformationPipelineConfig válido."
            )
            return result
        if not isinstance(target_payload, TransformationPipelineConfig):
            result.add_error(
                f"[{self.rule_id}] Recurso '{self._target_id}' não contém "
                "um TransformationPipelineConfig válido."
            )
            return result

        source_dim = source_payload.output_dimension
        target_dim = target_payload.output_dimension

        # Quando um pipeline não tem transformação dimensional, a saída é
        # a dimensão original do embedding selecionado na estratégia.
        emb_dim = self._embedding_dimensions
        if emb_dim is None:
            emb_dim = context.get_metadata("embedding_dimensions")

        effective_source = source_dim if source_dim is not None else emb_dim
        effective_target = target_dim if target_dim is not None else emb_dim

        if effective_source is not None and effective_target is not None:
            if effective_source != effective_target:
                result.add_error(
                    f"[{self.rule_id}] Dimensão de saída incompatível: "
                    f"source={effective_source}, target={effective_target}. "
                    f"Ambos os pipelines devem produzir embeddings com a mesma dimensionalidade."
                )
        elif effective_source is None and effective_target is None:
            result.add_warning(
                f"[{self.rule_id}] Nenhum pipeline define dimensão de saída explícita "
                f"e o tipo de embedding não está configurado na estratégia. "
                f"A compatibilidade será verificada em runtime."
            )

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result
