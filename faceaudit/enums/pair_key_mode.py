# faceaudit/enums/pair_key_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class PairKeyMode(Enum):
    """
    Modo de tratamento das chaves de par num dataset de scores.

    Conceito diferente de `PairRoleOrientation`: este enum não trata de
    semântica de papéis, mas de COMO a chave do par é interpretada para
    fins de alinhamento e deduplicação.

    Valores:

    - `ORDERED`: pares (a, b) e (b, a) são observações DISTINTAS. Usado
      em cross-comparison cross-pose onde (frontal_X, profile_Y) e
      (profile_X, frontal_Y) são experimentos diferentes.
    - `UNORDERED`: (a, b) e (b, a) são a MESMA observação. O alinhamento
      pode reordenar livremente; deduplicação rejeita ambas versões.
    """

    ORDERED = (
        "ORDERED",
        "Ordenado",
        "Pares (a, b) e (b, a) são observações distintas.",
    )

    UNORDERED = (
        "UNORDERED",
        "Não ordenado",
        "Pares (a, b) e (b, a) são a mesma observação.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "PairKeyMode":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.ORDERED
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"PairKeyMode desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
