"""CropProcessor — recorte (PR I1).

Modos suportados:
  - CENTERED — corta retângulo centralizado de
    `(target_width, target_height)`. Erro se imagem é menor que o
    destino (sem padding automático).
  - FACE_DETECTED — usa `ff_service.app.backend.detect_faces` →
    pega a maior face → corta bbox + `face_padding` opcional. Sem
    rotação.

USER_RECTANGLE entra em I6 com canvas interativo.

Determinístico — `rng` é ignorado.
"""

from __future__ import annotations

from random import Random
from typing import Optional

import numpy as np

from faceaudit.enums.crop_mode import CropMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.crop_config import CropConfig


class CropProcessor:
    """Aplica recorte conforme `CropConfig.mode`."""

    def __init__(
        self,
        config: CropConfig,
        *,
        ff_service=None,
    ):
        # `ff_service` é obrigatório quando mode == FACE_DETECTED.
        # Pra outros modos, ignorado. Validação feita lazy no apply.
        self._config = config
        self._ff_service = ff_service

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.CROP

    @property
    def config(self) -> CropConfig:
        return self._config

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
    ) -> np.ndarray:
        if self._config.mode == CropMode.CENTERED:
            return self._apply_centered(image)
        if self._config.mode == CropMode.FACE_DETECTED:
            return self._apply_face_detected(image)
        raise NotImplementedError(
            f"CropMode.{self._config.mode.name} ainda não implementado "
            "(virá em I6: USER_RECTANGLE)."
        )

    # ── CENTERED ───────────────────────────────────────────────────────────

    def _apply_centered(self, image: np.ndarray) -> np.ndarray:
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
        return image[y:y + th, x:x + tw].copy()

    # ── FACE_DETECTED ─────────────────────────────────────────────────────

    def _apply_face_detected(self, image: np.ndarray) -> np.ndarray:
        if self._ff_service is None:
            raise RuntimeError(
                "CropMode.FACE_DETECTED requer FFService injetado. "
                "Construa o pipeline passando ff_service=... "
                "(`ImageProcessingPipeline(config, ff_service=...)`)."
            )
        # Acessa o backend diretamente — mesmo padrão do
        # FFService.align_face_only (PR L3).
        engine = self._ff_service.app
        faces = engine.backend.detect_faces(image)
        if not faces:
            raise RuntimeError(
                "CropMode.FACE_DETECTED: nenhuma face detectada na imagem."
            )
        # forensicface >= 0.7.1 removeu ForensicFace._get_best_face e expôs
        # a seleção como função de módulo em geometry.select_best_face
        # (recebe img_shape, não o array). Import local: este caminho só
        # roda com FFService presente (logo, forensicface instalada).
        from forensicface.geometry import select_best_face
        face = select_best_face(image.shape, faces, criterion="size")
        x1, y1, x2, y2 = self._extract_bbox(face)

        # Expande a bbox pelo fator configurado. Usa a MESMA primitiva da
        # extração de faces de vídeo — o mesmo número tem de produzir o
        # mesmo enquadramento nos dois fluxos.
        import numpy as _np
        from forensicface.geometry import extend_bbox

        x1, y1, x2, y2 = extend_bbox(
            _np.asarray([x1, y1, x2, y2]),
            image.shape,
            margin_factor=float(self._config.margin_factor),
        )

        # `extend_bbox` já corta ao quadro; resta garantir área não-nula
        # (bbox degenerada na borda viraria fatia vazia em silêncio).
        h, w = image.shape[:2]
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y2 = max(y1 + 1, min(h, int(y2)))

        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def _extract_bbox(face) -> tuple:
        """Extrai (x1, y1, x2, y2) do objeto face do backend.

        InsightFace expõe `face.bbox` como ndarray [x1, y1, x2, y2].
        Conversão pra int defensiva.
        """
        bbox = face.bbox
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])
        return x1, y1, x2, y2
