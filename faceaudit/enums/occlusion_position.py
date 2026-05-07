"""Posicionamento da oclusão (PR I1).

I1 implementa CENTER e RANDOM. EYES e MOUTH (oclusão semântica via
landmarks do FFInfo) entram em I7.

RANDOM exige RNG — o pipeline propaga via `apply(image, *, rng)`
deterministicamente derivado de `(seed, image_path)`.
"""

from __future__ import annotations

from enum import Enum


class OcclusionPosition(str, Enum):
    CENTER = "center"
    RANDOM = "random"
    EYES = "eyes"      # I7 — semântica via landmarks
    MOUTH = "mouth"    # I7 — semântica via landmarks

    @property
    def label_pt(self) -> str:
        return {
            "center": "Centralizada",
            "random": "Aleatória",
            "eyes": "Sobre os olhos",
            "mouth": "Sobre a boca",
        }.get(self.value, self.value)
