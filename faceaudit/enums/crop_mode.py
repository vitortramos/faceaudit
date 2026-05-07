"""Modos de recorte do CropProcessor (PR I1).

I1 implementa CENTERED e FACE_DETECTED. USER_RECTANGLE (usuário
desenha o retângulo no canvas) entra em I6 quando o canvas
interativo for criado.
"""

from __future__ import annotations

from enum import Enum


class CropMode(str, Enum):
    CENTERED = "centered"
    FACE_DETECTED = "face_detected"
    USER_RECTANGLE = "user_rectangle"  # I6

    @property
    def label_pt(self) -> str:
        return {
            "centered": "Centralizado",
            "face_detected": "Face detectada",
            "user_rectangle": "Retângulo manual",
        }.get(self.value, self.value)
