# faceaudit/enums/dataset_merge_mode.py

from __future__ import annotations
from enum import Enum


class DatasetMergeMode(Enum):
    """
    Define os modos suportados de combinação e atualização de datasets.

    Os modos foram desenhados para cobrir tanto operações clássicas de merge
    (concatenação) quanto operações cirúrgicas de substituição/atualização
    orientadas por índice.
    """

    CONCAT = "concat"
    CONCAT_PRESERVE_ORDER = "concat_preserve_order"
    REPLACE_BY_INDEX_LIST = "replace_by_index_list"
    UPDATE_BY_INDEX_LIST = "update_by_index_list"

    @property
    def label_pt(self) -> str:
        return {
            "concat":                "Concatenar",
            "concat_preserve_order": "Concatenar (preservar ordem)",
            "replace_by_index_list": "Substituir por lista de índices",
            "update_by_index_list":  "Atualizar por lista de índices",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        """Retorna o identificador textual do modo."""
        return self.value

    def to_str(self) -> str:
        """Retorna o valor persistível para serialização."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> DatasetMergeMode:
        """
        Reconstrói o enum a partir de string com fallback seguro.

        Fallback padrão:
        - CONCAT
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.CONCAT