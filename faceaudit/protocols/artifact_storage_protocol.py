# faceaudit/protocols/artifact_storage_protocol.py

from __future__ import annotations
from pathlib import Path
from typing import Any, List, Optional, Protocol, Union, runtime_checkable

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference


@runtime_checkable
class ArtifactStorageProtocol(Protocol):
    """
    Protocolo para persistência física de artefatos da nova arquitetura.

    A implementação concreta pode salvar artefatos em disco, copiá-los para o
    projeto, resolvê-los posteriormente e listar os artefatos disponíveis.
    """

    def save_artifact(
        self,
        artifact_type: StrategyArtifactType,
        data: Any,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Persiste um artefato e retorna sua referência lógica.

        Args:
            artifact_type:
                Tipo semântico do artefato.
            data:
                Objeto a ser persistido.
            **kwargs:
                Parâmetros auxiliares para nomenclatura, localização ou metadata.

        Returns:
            StrategyArtifactReference:
                Referência ao artefato persistido.
        """
        ...

    def copy_artifact_to_project(
        self,
        source_path: Union[str, Path],
        artifact_type: StrategyArtifactType,
        **kwargs,
    ) -> StrategyArtifactReference:
        """
        Copia um artefato existente para a estrutura do projeto e retorna sua referência.

        Args:
            source_path:
                Caminho de origem do artefato.
            artifact_type:
                Tipo semântico do artefato.
            **kwargs:
                Parâmetros auxiliares para nomenclatura e localização.

        Returns:
            StrategyArtifactReference:
                Referência ao artefato copiado.
        """
        ...

    def resolve_artifact(
        self,
        artifact_ref: StrategyArtifactReference,
    ) -> Optional[Path]:
        """
        Resolve a referência lógica de um artefato para um caminho físico.

        Args:
            artifact_ref:
                Referência do artefato a ser resolvido.

        Returns:
            Optional[Path]:
                Caminho físico do artefato, se encontrado.
        """
        ...

    def list_artifacts(
        self,
        artifact_type: Optional[StrategyArtifactType] = None,
    ) -> List[StrategyArtifactReference]:
        """
        Lista artefatos disponíveis, opcionalmente filtrando por tipo.

        Args:
            artifact_type:
                Tipo semântico opcional para filtro.

        Returns:
            List[StrategyArtifactReference]:
                Lista de referências de artefatos disponíveis.
        """
        ...