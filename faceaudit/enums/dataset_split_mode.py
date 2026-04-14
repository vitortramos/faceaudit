# faceaudit/enums/dataset_split_mode.py

from __future__ import annotations
from enum import Enum


class DatasetSplitMode(Enum):
    """
    Define os modos suportados de particionamento de datasets.

    Os modos foram desenhados para cobrir tanto datasets ricos (`FaceDataset`)
    quanto datasets vetoriais (`EmbeddingDataset`), mantendo a lógica de split
    declarativa e reutilizável dentro da strategy engine.
    """

    BY_IDENTITY_RATIO = "by_identity_ratio"
    BY_IDENTITY_LIST = "by_identity_list"
    BY_INDEX_RATIO = "by_index_ratio"
    BY_INDEX_LIST = "by_index_list"

    def __str__(self) -> str:
        """Retorna o identificador textual do modo."""
        return self.value

    def to_str(self) -> str:
        """Retorna o valor persistível para serialização."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> DatasetSplitMode:
        """
        Reconstrói o enum a partir de string com fallback seguro.

        Fallback padrão:
        - BY_IDENTITY_RATIO
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.BY_IDENTITY_RATIO