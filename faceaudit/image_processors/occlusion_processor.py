"""OcclusionProcessor — oclusão geométrica (PR I1).

I1 implementa OcclusionKind.RECTANGLE_BLACK + position={CENTER, RANDOM}.

Largura e altura do retângulo derivam de
`sqrt(coverage_pct) * (W, H)`, mantendo proporção da imagem.

`position=RANDOM` requer RNG (passada via `apply(image, *, rng)`).
Se RNG for None, cai em CENTER (defensivo).
"""

from __future__ import annotations

import math
from random import Random
from typing import Optional

import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.occlusion_config import OcclusionConfig


class OcclusionProcessor:
    """Aplica oclusão geométrica conforme `OcclusionConfig`."""

    def __init__(
        self,
        config: OcclusionConfig,
        *,
        ff_service=None,  # uniforme com outros processors; ignorado em I1
    ):
        self._config = config

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.OCCLUSION

    @property
    def config(self) -> OcclusionConfig:
        return self._config

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
    ) -> np.ndarray:
        if self._config.kind == OcclusionKind.RECTANGLE_BLACK:
            return self._apply_rectangle_black(image, rng)
        raise NotImplementedError(
            f"OcclusionKind.{self._config.kind.name} ainda não implementado "
            "(virá em I2/I7: RECTANGLE_NOISE, CIRCLE_BLACK, MASK_FACE)."
        )

    def _apply_rectangle_black(
        self,
        image: np.ndarray,
        rng: Optional[Random],
    ) -> np.ndarray:
        h, w = image.shape[:2]

        # Lado do retângulo derivado da fração de área
        ratio = math.sqrt(float(self._config.coverage_pct))
        rect_w = max(1, int(w * ratio))
        rect_h = max(1, int(h * ratio))

        # Posição
        position = self._config.position
        if position == OcclusionPosition.RANDOM:
            if rng is None:
                # Defensivo: sem RNG, cai pra CENTER pra ficar
                # determinístico (não bagunça smoke tests)
                position = OcclusionPosition.CENTER
            else:
                x = rng.randint(0, max(0, w - rect_w))
                y = rng.randint(0, max(0, h - rect_h))

        if position == OcclusionPosition.CENTER:
            x = (w - rect_w) // 2
            y = (h - rect_h) // 2
        elif position in (OcclusionPosition.EYES, OcclusionPosition.MOUTH):
            raise NotImplementedError(
                f"OcclusionPosition.{position.name} virá em I7 "
                "(via landmarks do FFInfo)."
            )

        out = image.copy()
        out[y:y + rect_h, x:x + rect_w] = 0  # preto BGR
        return out
