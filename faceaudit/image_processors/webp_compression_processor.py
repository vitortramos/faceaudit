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
from faceaudit.image_processors.image_step_base import ImageStepProcessorBase
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)
from faceaudit.models.image_processing.webp_compression_config import (
    WebpCompressionConfig,
)


class WebpCompressionProcessor(ImageStepProcessorBase):
    """Aplica recompressão WebP no quality declarado no config."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.WEBP

    def _process(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
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
