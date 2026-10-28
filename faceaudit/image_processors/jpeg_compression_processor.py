"""JpegCompressionProcessor — recompressão JPEG (PR I0).

Encode com `cv2.imencode(".jpg", img, [IMWRITE_JPEG_QUALITY, q])`
seguido de decode imediato. Simula o impacto de upload/transit por
canal lossy (mensageiro, e-mail, redes sociais) sem persistir
arquivo intermediário.

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
from faceaudit.models.image_processing.jpeg_compression_config import (
    JpegCompressionConfig,
)


class JpegCompressionProcessor(ImageStepProcessorBase):
    """Aplica recompressão JPEG no quality declarado no config."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.JPEG

    def _process(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
    ) -> np.ndarray:
        ok, buf = cv2.imencode(
            ".jpg", image,
            [cv2.IMWRITE_JPEG_QUALITY, int(self._config.quality)],
        )
        if not ok:
            raise RuntimeError(
                "cv2.imencode falhou pra JPEG (input inválido?)."
            )
        decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError(
                "cv2.imdecode falhou após encode JPEG (buffer inválido?)."
            )
        return decoded
