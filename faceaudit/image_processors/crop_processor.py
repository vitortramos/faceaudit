"""CropProcessor — recorte (PR I1).

Modos suportados:
  - CENTERED — corta retângulo centralizado de
    `(target_width, target_height)`. Erro se imagem é menor que o
    destino (sem padding automático).
  - FACE_DETECTED — corta a caixa da face expandida pelo fator de
    margem. A caixa vem da geometria que acompanha a imagem pelo
    pipeline (a do dataset, ou a detecção feita na imagem de entrada) —
    o passo não redetecta na imagem que recebe, que pode já estar
    degradada pelos passos anteriores.

Nos dois modos a geometria da face é transladada para a origem do
recorte e segue para os passos seguintes.

USER_RECTANGLE entra em I6 com canvas interativo.

Determinístico — `rng` é ignorado.
"""

from __future__ import annotations

from random import Random
from typing import Optional, Tuple

import numpy as np

from faceaudit.enums.crop_mode import CropMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.image_processors.image_step_base import ImageStepProcessorBase
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)


class CropProcessor(ImageStepProcessorBase):
    """Aplica recorte conforme `CropConfig.mode`."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.CROP

    def apply_step(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> Tuple[np.ndarray, Optional[ImageFaceGeometry]]:
        if self._config.mode == CropMode.CENTERED:
            x1, y1, x2, y2 = self._centered_box(image)
        elif self._config.mode == CropMode.FACE_DETECTED:
            x1, y1, x2, y2 = self._face_box(
                image, self._require_face(face, "O recorte pela face"),
            )
        else:
            raise NotImplementedError(
                f"CropMode.{self._config.mode.name} ainda não implementado "
                "(virá em I6: USER_RECTANGLE)."
            )
        out = image[y1:y2, x1:x2].copy()
        return out, (None if face is None else face.translated(-x1, -y1))

    # ── CENTERED ───────────────────────────────────────────────────────────

    def _centered_box(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = image.shape[:2]
        tw = int(self._config.target_width)
        th = int(self._config.target_height)
        if tw > w or th > h:
            raise ValueError(
                f"CropMode.CENTERED: imagem ({w}×{h}) é menor que o alvo "
                f"({tw}×{th}). Adicione um redimensionamento antes ou "
                f"reduza o alvo."
            )
        x = (w - tw) // 2
        y = (h - th) // 2
        return x, y, x + tw, y + th

    # ── FACE_DETECTED ─────────────────────────────────────────────────────

    def _face_box(
        self, image: np.ndarray, face: ImageFaceGeometry,
    ) -> Tuple[int, int, int, int]:
        caixa = face.face_box()
        if caixa is None:
            raise RuntimeError("O recorte pela face não tem caixa nem pontos da face.")

        # Expande a caixa pelo fator configurado. Usa a MESMA primitiva da
        # extração de faces de vídeo — o mesmo número tem de produzir o
        # mesmo enquadramento nos dois fluxos.
        from forensicface.geometry import extend_bbox

        x1, y1, x2, y2 = extend_bbox(
            np.asarray(caixa, dtype=np.float64),
            image.shape,
            margin_factor=float(self._config.margin_factor),
        )

        # `extend_bbox` já corta ao quadro; resta garantir área não-nula
        # (caixa degenerada na borda viraria fatia vazia em silêncio).
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))
        return x1, y1, x2, y2
