# faceaudit/repositories/strategy_definition_repository.py

from __future__ import annotations

import json
from typing import List, Optional

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_file_type import StrategyFileType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.strategy_definition_repository_protocol import (
    StrategyDefinitionRepositoryProtocol,
)


class StrategyDefinitionRepository(StrategyDefinitionRepositoryProtocol):
    """
    Repositório concreto para persistência da definição declarativa da estratégia.

    Salva cada definição como arquivo JSON autocontido dentro do projeto.
    Usa exclusivamente a interface ArtifactStorageProtocol — sem acesso direto
    ao FileSystemService ou a paths absolutos hard-coded.
    """

    def __init__(self, artifact_storage: ArtifactStorageProtocol):
        self._storage = artifact_storage

    def save_definition(self, strategy_definition: LRStrategyDefinition) -> None:
        """
        Persiste a definição da estratégia como artefato JSON.
        """
        if strategy_definition is None:
            raise ValueError("A definição da estratégia não pode ser nula.")

        self._storage.save_artifact(
            artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
            data=strategy_definition.to_dict(),
            strategy_id=strategy_definition.strategy_id,
        )

    def load_definition(self, strategy_id: str) -> Optional[LRStrategyDefinition]:
        """
        Carrega a definição da estratégia a partir do arquivo JSON correspondente.

        O caminho esperado é derivado de StrategyFileType — não há strings hard-coded.
        Retorna None se o artefato não existir no projeto.
        """
        if not strategy_id:
            return None

        relative_path = StrategyFileType.STRATEGY_DEFINITION.build_path(
            strategy_id=strategy_id
        )
        ref = StrategyArtifactReference(
            artifact_type=StrategyArtifactType.STRATEGY_DEFINITION,
            path=str(relative_path),
            hash=None,
        )

        abs_path = self._storage.resolve_artifact(ref)
        if abs_path is None:
            return None

        data = json.loads(abs_path.read_text(encoding="utf-8"))
        return LRStrategyDefinition.from_dict(data)

    def list_definitions(self) -> List[LRStrategyDefinition]:
        """
        Lista todas as definições de estratégia presentes no projeto.
        """
        refs = self._storage.list_artifacts(StrategyArtifactType.STRATEGY_DEFINITION)

        found: List[LRStrategyDefinition] = []

        for ref in refs:
            abs_path = self._storage.resolve_artifact(ref)
            if abs_path is None:
                continue
            try:
                data = json.loads(abs_path.read_text(encoding="utf-8"))
                definition = LRStrategyDefinition.from_dict(data)
                if definition is not None:
                    found.append(definition)
            except Exception:
                continue

        return found
