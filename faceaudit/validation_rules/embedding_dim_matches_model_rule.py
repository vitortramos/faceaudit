# faceaudit/validation_rules/embedding_dim_matches_model_rule.py

from __future__ import annotations

from typing import Any, Optional

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.validation_rules.validation_rule_base import ValidationRuleBase


class EmbeddingDimMatchesModelRule(ValidationRuleBase):
    """
    Verifica que `dataset.embedding_dim == model.input_dimension`.

    Aplicável a:
    - PIPELINE_APPLICATION (dataset × pipeline.input_dimension)
    - SIMILARITY_SCORING (probe/reference × similarity_model.input_dimension)

    Sem essa checagem, o consumo do modelo dispara erros de shape no
    primeiro forward.

    Parâmetros:
        dataset_resource_id: id do recurso DATASET
        model_resource_id: id do modelo treinado (com `input_dimension`)
    """

    def __init__(
        self,
        rule_id: str,
        dataset_resource_id: str,
        model_resource_id: str,
        name: str = "Compatibilidade dimensional dataset×modelo",
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            rule_id=rule_id,
            name=name,
            description=description or (
                f"Verifica que '{dataset_resource_id}' tem embeddings "
                f"compatíveis com '{model_resource_id}'."
            ),
            default_severity=ValidationSeverity.CRITICAL,
        )
        self._dataset_id = dataset_resource_id
        self._model_id = model_resource_id

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

        if not context.has_resource(self._model_id):
            result.add_error(
                f"[{self.rule_id}] Modelo '{self._model_id}' não encontrado no contexto."
            )
            return result

        dataset_dim = int(getattr(stats, "embedding_dim", 0) or 0)
        if dataset_dim <= 0:
            result.add_warning(
                f"[{self.rule_id}] Não foi possível inferir embedding_dim de "
                f"'{self._dataset_id}'. Pulando checagem dimensional."
            )
            return result

        resource = context.ensure_payload(self._model_id)
        payload = resource.payload if resource is not None else None
        if payload is None:
            result.add_warning(
                f"[{self.rule_id}] Payload de '{self._model_id}' indisponível. "
                "Pulando checagem dimensional."
            )
            return result

        model_dim = getattr(payload, "input_dimension", None)
        if model_dim is None or model_dim <= 0:
            # Modelo sem dimensão declarada (ex: Cosine stateless) — não dá
            # pra checar; assume OK silenciosamente.
            result.mark_ok()
            return result

        if int(model_dim) != dataset_dim:
            result.add_error(
                f"[{self.rule_id}] Incompatibilidade dimensional: "
                f"'{self._dataset_id}' tem embedding_dim={dataset_dim} mas "
                f"'{self._model_id}' espera input_dimension={int(model_dim)}."
            )
            return result

        result.mark_ok()
        return result
