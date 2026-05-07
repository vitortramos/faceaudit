"""WebpCompressionProcessor — recompressão WebP (PR I1).

Análogo ao JpegCompressionProcessor: encode + decode pra simular o
impacto de transit por canal lossy. WebP geralmente preserva
melhor que JPEG na mesma quality, mas ainda introduz artefatos
abaixo de 95.

Determinístico — `rng` é ignorado.
"""

from __future__ import annotations

from random import Random
from typing import Optional

import cv2
import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.webp_compression_config import (
    WebpCompressionConfig,
)


class WebpCompressionProcessor:
    """Aplica recompressão WebP no quality declarado no config."""

    def __init__(
        self,
        config: WebpCompressionConfig,
        *,
        ff_service=None,  # PR I1 — uniforme com outros processors; ignorado aqui
    ):
        self._config = config

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.WEBP

    @property
    def config(self) -> WebpCompressionConfig:
        return self._config

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
    ) -> np.ndarray:
        ok, buf = cv2.imencode(
            ".webp", image,
            [cv2.IMWRITE_WEBP_QUALITY, int(self._config.quality)],
        )
        if not ok:
            raise RuntimeError(
                "cv2.imencode falhou pra WebP (input inválido?)."
            )
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(
                "cv2.imdecode falhou após encode WebP (buffer inválido?)."
            )
        return decoded
