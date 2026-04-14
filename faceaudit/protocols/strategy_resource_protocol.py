# faceaudit/protocols/strategy_resource_protocol.py

from __future__ import annotations
from typing import Any, Optional, Protocol, runtime_checkable

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference


@runtime_checkable
class StrategyResourceProtocol(Protocol):
    """
    Protocolo para recursos lógicos da estratégia.

    Um recurso representa qualquer objeto reutilizável dentro da estratégia,
    como datasets, configurações, modelos treinados ou artefatos intermediários.
    """

    @property
    def resource_id(self) -> str:
        """
        Identificador único do recurso dentro da estratégia.
        """
        ...

    @property
    def name(self) -> str:
        """
        Nome amigável do recurso.
        """
        ...

    @property
    def resource_type(self) -> StrategyResourceType:
        """
        Tipo lógico do recurso.
        """
        ...

    @property
    def payload(self) -> Any:
        """
        Objeto em memória associado ao recurso.
        """
        ...

    @property
    def artifact_ref(self) -> Optional[StrategyArtifactReference]:
        """
        Referência opcional ao artefato físico persistido.
        """
        ...

    @property
    def has_payload(self) -> bool:
        """
        Retorna True se o recurso possui payload em memória.
        """
        ...

    @property
    def has_artifact(self) -> bool:
        """
        Retorna True se o recurso possui referência de artefato persistido.
        """
        ...