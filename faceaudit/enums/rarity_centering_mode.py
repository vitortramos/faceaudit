# faceaudit/enums/rarity_centering_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class RarityCenteringMode(Enum):
    """
    Política de centragem da raridade aplicada pelas arquiteturas que
    incluem termos de interação `*_X_RARITY_*_CENTERED_MEAN`.

    NONE: sem centragem — interações usam r_F e r_P diretamente.
    MEAN: r̃ = r − r̄, onde r̄ é a média de r no conjunto D_cal.

    Em arquiteturas que mesclam termos com e sem centragem, este enum
    descreve o modo padrão. Termos individuais carregam a variante exata
    no próprio identificador (`*_RAW` vs `*_CENTERED_MEAN`).
    """

    NONE = (
        "NONE",
        "Sem centragem",
        "Interações usam r_F e r_P brutos.",
    )

    MEAN = (
        "MEAN",
        "Centragem pela média (r̃ = r − r̄)",
        "Subtrai a média r̄ estimada em D_cal de cada raridade antes de "
        "multiplicar. Inspirada parcialmente no princípio hierárquico "
        "de Tukey.",
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
    def from_str(cls, value: str) -> "RarityCenteringMode":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.NONE
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"RarityCenteringMode desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
