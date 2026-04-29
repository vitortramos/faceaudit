# faceaudit/validation_rules/validation_rule_base.py

from __future__ import annotations

from typing import Optional

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.dataset_stats import DatasetStats
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.services.dataset_stats_service import DatasetStatsService


class ValidationRuleBase:
    """
    Classe base para implementações concretas de ValidationRuleProtocol.

    Fornece:
    - Campos obrigatórios do protocolo (rule_id, name, description, default_severity)
    - Helper _get_or_compute_stats(): obtém DatasetStats do contexto ou computa
      e armazena em cache para reutilização por outras regras e executores
      da mesma execução (Opção B — cache no contexto).

    Convenção de nomes:
        O stats de um recurso com id "my_dataset" é armazenado no contexto
        com id "my_dataset_stats". As regras seguem essa convenção automaticamente.
    """

    def __init__(
        self,
        rule_id: str,
        name: str,
        description: Optional[str] = None,
        default_severity: ValidationSeverity = ValidationSeverity.CRITICAL,
    ) -> None:
        self._rule_id = rule_id
        self._name = name
        self._description = description
        self._default_severity = default_severity
        self._stats_service = DatasetStatsService()

    # =========================================================================
    # ValidationRuleProtocol — campos
    # =========================================================================

    @property
    def rule_id(self) -> str:
        return self._rule_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> Optional[str]:
        return self._description

    @property
    def default_severity(self) -> ValidationSeverity:
        return self._default_severity

    # =========================================================================
    # HELPER — stats com cache no contexto
    # =========================================================================

    def _get_or_compute_stats(
        self,
        resource_id: str,
        context: StrategyExecutionContextProtocol,
    ) -> Optional[DatasetStats]:
        """
        Retorna o DatasetStats para o recurso indicado.

        Fluxo:
        1. Verifica se "{resource_id}_stats" já está no contexto (cache hit).
        2. Se não, resolve o recurso, determina o tipo do payload e computa.
        3. Armazena o resultado no contexto para reutilização.
        4. Retorna None se o recurso não existir ou o tipo não for suportado.
        """
        stats_id = f"{resource_id}_stats"

        if context.has_resource(stats_id):
            return context.get_resource(stats_id).payload

        if not context.has_resource(resource_id):
            return None

        # Hidratação preguiçosa: se o resource veio do disco, carrega o
        # payload sob demanda. Essencial para validação não falhar quando
        # prepare_context não hidrata mais eager.
        resource = context.ensure_payload(resource_id)
        payload = resource.payload if resource is not None else None
        stats: Optional[DatasetStats] = None

        if isinstance(payload, EmbeddingDataset):
            stats = self._stats_service.compute_from_embedding_dataset(payload)
        elif isinstance(payload, FaceDataset):
            stats = self._stats_service.compute_from_dataset(payload)

        if stats is not None:
            context.set_resource(StrategyResource(
                resource_id=stats_id,
                name=stats_id,
                resource_type=StrategyResourceType.DATASET_STATS,
                payload=stats,
                artifact_ref=None,
            ))

        return stats
