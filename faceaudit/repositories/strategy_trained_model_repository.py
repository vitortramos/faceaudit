# faceaudit/repositories/strategy_trained_model_repository.py

from __future__ import annotations

import pickle
from typing import List, Optional

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.strategy_trained_model_repository_protocol import (
    StrategyTrainedModelRepositoryProtocol,
)


class StrategyTrainedModelRepository(StrategyTrainedModelRepositoryProtocol):
    """
    Repositório concreto para persistência do modelo treinado da estratégia.

    Usa exclusivamente a interface ArtifactStorageProtocol — sem acesso direto
    ao FileSystemService ou a paths absolutos hard-coded.
    """

    def __init__(self, artifact_storage: ArtifactStorageProtocol):
        self._storage = artifact_storage

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def save_model(self, trained_model: LRStrategyTrainedModel) -> None:
        """
        Persiste o modelo treinado da estratégia.
        """
        if trained_model is None:
            raise ValueError("O modelo treinado não pode ser nulo.")

        self._storage.save_artifact(
            artifact_type=StrategyArtifactType.STRATEGY_MODEL,
            data=trained_model,
            strategy_id=trained_model.strategy_id,
        )

    def load_model(self, strategy_id: str) -> Optional[LRStrategyTrainedModel]:
        """
        Carrega o modelo treinado correspondente ao strategy_id.

        O caminho esperado é derivado de StrategyFileType — não há strings hard-coded.
        Retorna None se o artefato não existir no projeto.
        """
        if not strategy_id:
            return None

        relative_path = StrategyFileType.TRAINED_STRATEGY_MODEL.build_path(
            strategy_id=strategy_id
        )
        ref = StrategyArtifactReference(
            artifact_type=StrategyArtifactType.STRATEGY_MODEL,
            path=str(relative_path),
            hash=None,
        )

        abs_path = self._storage.resolve_artifact(ref)
        if abs_path is None:
            return None

        with open(abs_path, "rb") as f:
            model = pickle.load(f)

        if not isinstance(model, LRStrategyTrainedModel):
            raise TypeError(
                "O arquivo carregado não contém um LRStrategyTrainedModel válido."
            )

        return model

    def list_models(self) -> List[LRStrategyTrainedModel]:
        """
        Lista todos os modelos treinados disponíveis no projeto.
        """
        refs = self._storage.list_artifacts(StrategyArtifactType.STRATEGY_MODEL)

        found: List[LRStrategyTrainedModel] = []

        for ref in refs:
            abs_path = self._storage.resolve_artifact(ref)
            if abs_path is None:
                continue
            try:
                with open(abs_path, "rb") as f:
                    model = pickle.load(f)

                if isinstance(model, LRStrategyTrainedModel):
                    found.append(model)
            except Exception:
                continue

        return found
