"""BlurProcessor — desfoque (PR I0, só Gaussian).

I0 implementa apenas `BlurKind.GAUSSIAN` via `cv2.GaussianBlur`.
MOTION e BOX entram em I1.

Determinístico — `rng` é ignorado.
"""

from __future__ import annotations

from random import Random
from typing import Optional

import cv2
import numpy as np

from faceaudit.enums.blur_kind import BlurKind
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.blur_config import BlurConfig


class BlurProcessor:
    """Aplica desfoque conforme `BlurConfig`."""

    def __init__(self, config: BlurConfig):
        self._config = config

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.BLUR

    @property
    def config(self) -> BlurConfig:
        return self._config

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
    ) -> np.ndarray:
        if self._config.kind == BlurKind.GAUSSIAN:
            return self._apply_gaussian(image)
        raise NotImplementedError(
            f"BlurKind.{self._config.kind.name} ainda não implementado "
            "(virá em I1: MOTION, BOX)."
        )

    def _apply_gaussian(self, image: np.ndarray) -> np.ndarray:
        # cv2.GaussianBlur exige kernel ímpar — corrige sem falhar
        # quando o usuário passa par via UI/JSON.
        k = int(self._config.kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(
            image, (k, k), float(self._config.sigma),
        )
