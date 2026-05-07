"""Tipos de oclusão (PR I1).

I1 implementa só RECTANGLE_BLACK. RECTANGLE_NOISE, CIRCLE_BLACK,
MASK_FACE entram em I2/I7.
"""

from __future__ import annotations

from enum import Enum


class OcclusionKind(str, Enum):
    RECTANGLE_BLACK = "rectangle_black"
    RECTANGLE_NOISE = "rectangle_noise"   # I2
    CIRCLE_BLACK = "circle_black"          # I2
    MASK_FACE = "mask_face"                # I7 — desenho livre via canvas

    @property
    def label_pt(self) -> str:
        return {
            "rectangle_black": "Retângulo preto",
            "rectangle_noise": "Retângulo com ruído",
            "circle_black": "Círculo preto",
            "mask_face": "Máscara desenhada",
        }.get(self.value, self.value)
