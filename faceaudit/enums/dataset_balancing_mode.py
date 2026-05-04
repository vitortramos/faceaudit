# faceaudit/enums/dataset_balancing_mode.py

from __future__ import annotations
from enum import Enum


class DatasetBalancingMode(Enum):
    """
    Define os modos suportados de balanceamento de amostras por identidade.

    - CAP_ONLY: limita o número máximo a `target_n` por identidade.
      Identidades com menos amostras permanecem inalteradas.
    - FLOOR_AND_CAP: além do cap, descarta identidades inteiras que
      possuírem menos de `target_n` amostras.
    - AUTO_MIN_BALANCE: usa como N o mínimo de amostras observado entre
      as identidades — todas terminam com exatamente N amostras (perda
      máxima de dados, mas garante balanceamento perfeito).
    """

    CAP_ONLY = "cap_only"
    FLOOR_AND_CAP = "floor_and_cap"
    AUTO_MIN_BALANCE = "auto_min_balance"

    def __str__(self) -> str:
        return self.value

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> DatasetBalancingMode:
        """Reconstrói o enum a partir de string com fallback para CAP_ONLY."""
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.CAP_ONLY
