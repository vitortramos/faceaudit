"""Tipos de desfoque suportados (PR I0).

I0 implementa só GAUSSIAN. MOTION e BOX entram em I1.

Values estáveis em disco — JSON do BlurConfig usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class BlurKind(str, Enum):
    GAUSSIAN = "gaussian"
    MOTION = "motion"  # I1
    BOX = "box"        # I1

    @property
    def label_pt(self) -> str:
        return {
            "gaussian": "Gaussiano",
            "motion": "Movimento",
            "box": "Box (média)",
        }.get(self.value, self.value)
