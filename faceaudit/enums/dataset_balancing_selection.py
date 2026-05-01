# faceaudit/enums/dataset_balancing_selection.py

from __future__ import annotations
from enum import Enum


class DatasetBalancingSelection(Enum):
    """
    Critério de seleção de quais amostras descartar quando uma identidade
    excede o cap de samples (`target_n`).

    - RANDOM: sorteio uniforme com `random_seed` para reprodutibilidade.
    - BY_QUALITY: ordena por uma `QualityScoreType` (maior = melhor) e
      mantém o top-N. Empates são quebrados por hash do face_hash/file_name
      para determinismo.
    """

    RANDOM = "random"
    BY_QUALITY = "by_quality"

    def __str__(self) -> str:
        return self.value

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> DatasetBalancingSelection:
        """Reconstrói o enum a partir de string com fallback para RANDOM."""
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.RANDOM
