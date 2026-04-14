# faceaudit/protocols/lr_strategy_trained_model_protocol.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.protocols.strategy_resource_protocol import StrategyResourceProtocol


@runtime_checkable
class LRStrategyTrainedModelProtocol(Protocol):
    """
    Protocolo para o modelo treinado final da estratégia.

    Um modelo treinado de estratégia representa o resultado materializado da
    preparação do workflow declarativo, contendo:
    - recursos finais e intermediários relevantes
    - artefatos persistidos
    - metadados da execução
    """

    @property
    def strategy_id(self) -> str:
        """
        Identificador da estratégia que originou o modelo treinado.
        """
        ...

    @property
    def trained_resources(self) -> List[StrategyResourceProtocol]:
        """
        Lista de recursos treinados/finais associados ao modelo da estratégia.
        """
        ...

    @property
    def artifacts(self) -> List[StrategyArtifactReference]:
        """
        Lista de artefatos físicos/lógicos associados ao modelo treinado.
        """
        ...

    @property
    def execution_metadata(self) -> Dict[str, Any]:
        """
        Metadados consolidados da execução/preparação da estratégia.
        """
        ...

    def get_resource_by_id(self, resource_id: str) -> Optional[StrategyResourceProtocol]:
        """
        Retorna o recurso treinado correspondente ao identificador informado.
        """
        ...

    def get_artifact_by_path(self, path: str) -> Optional[StrategyArtifactReference]:
        """
        Retorna o artefato correspondente ao caminho informado.
        """
        ...

    def add_trained_resource(self, resource: StrategyResourceProtocol) -> None:
        """
        Registra um recurso treinado no modelo da estratégia.
        """
        ...

    def add_artifact(self, artifact: StrategyArtifactReference) -> None:
        """
        Registra um artefato associado ao modelo da estratégia.
        """
        ...

    def set_execution_metadata(self, key: str, value: Any) -> None:
        """
        Registra ou atualiza um metadado de execução.
        """
        ...

    def get_execution_metadata(self, key: str, default: Any = None) -> Any:
        """
        Recupera um metadado de execução.
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o modelo treinado da estratégia em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o modelo treinado da estratégia a partir do disco.
        """
        ...