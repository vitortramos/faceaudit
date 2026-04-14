# faceaudit/services/strategy_resource_loader_service.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol


class StrategyResourceLoaderService:
    """
    Carrega (hidrata) o payload de um StrategyResource a partir de seu artifact_ref.

    Usado pelo StrategyExecutor para carregar recursos declarados na estratégia
    antes da execução — caso a definição tenha sido carregada de JSON
    (payload=None, artifact_ref preenchido).

    A hidratação é transparente:
    - Se payload já existe em memória → retorna sem modificação.
    - Se artifact_ref é None → retorna sem modificação (executor detectará na validação).
    - Se o arquivo não for encontrado → retorna sem modificação (executor detectará na validação).
    - Se o tipo não tiver loader registrado → retorna sem modificação.

    Tipos suportados:
        DATASET          → EmbeddingDataset.from_parquet() ou FaceDataset.from_parquet()
        PIPELINE_CONFIG  → TransformationPipelineConfig.from_dict()
    """

    def __init__(self, artifact_storage: ArtifactStorageProtocol) -> None:
        self._storage = artifact_storage

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def hydrate(self, resource: StrategyResource) -> StrategyResource:
        """
        Carrega o payload do recurso a partir do artifact_ref, se necessário.

        A operação é in-place: modifica resource.payload diretamente.
        Retorna sempre o mesmo objeto para permitir encadeamento.

        Args:
            resource: Recurso a ser hidratado.

        Returns:
            O mesmo objeto StrategyResource, com payload preenchido se possível.
        """
        if resource.payload is not None:
            return resource

        if resource.artifact_ref is None:
            return resource

        abs_path = self._storage.resolve_artifact(resource.artifact_ref)
        if abs_path is None:
            return resource  # arquivo ausente — executor falhará na validação

        payload = self._load_by_type(resource.resource_type, abs_path)
        if payload is not None:
            resource.payload = payload

        return resource

    def hydrate_all(self, resources: list[StrategyResource]) -> None:
        """
        Hidrata uma lista de recursos in-place.

        Args:
            resources: Lista de StrategyResource a ser hidratada.
        """
        for resource in resources:
            self.hydrate(resource)

    # =========================================================================
    # DESPACHO POR TIPO
    # =========================================================================

    def _load_by_type(
        self,
        resource_type: StrategyResourceType,
        path: Path,
    ) -> Optional[Any]:
        """
        Despacha o carregamento com base no tipo do recurso.

        Retorna None se o tipo não tiver loader registrado — o recurso
        permanecerá sem payload e a validação do executor reportará o erro.
        """
        if resource_type == StrategyResourceType.DATASET:
            return self._load_dataset(path)

        if resource_type == StrategyResourceType.PIPELINE_CONFIG:
            return self._load_pipeline_config(path)

        # Tipos intermediários/treinados não precisam de loader:
        # são escritos diretamente no contexto pelos executores.
        return None

    # =========================================================================
    # LOADERS CONCRETOS
    # =========================================================================

    def _load_dataset(self, path: Path) -> Optional[Any]:
        """
        Carrega um dataset a partir de um arquivo .parquet.

        Tenta EmbeddingDataset primeiro (caso mais comum).
        Se falhar, tenta FaceDataset.
        """
        try:
            return EmbeddingDataset.from_parquet(path)
        except Exception:
            pass

        try:
            return FaceDataset.from_parquet(path)
        except Exception:
            pass

        return None

    def _load_pipeline_config(self, path: Path) -> Optional[TransformationPipelineConfig]:
        """
        Carrega uma TransformationPipelineConfig a partir de um arquivo .json.
        """
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TransformationPipelineConfig.from_dict(data)
        except Exception:
            return None
