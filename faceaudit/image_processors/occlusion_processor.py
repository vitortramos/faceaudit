"""OcclusionProcessor — oclusão geométrica (PR I1 + I7) ou desenhada
(PR I6).

Implementações suportadas:
  - RECTANGLE_BLACK + CENTER  → retângulo central, lado=√(coverage_pct)·(W,H)
  - RECTANGLE_BLACK + RANDOM  → retângulo aleatório (mesma dim, posição via RNG)
  - CIRCLE_BLACK    + CENTER  → círculo central, raio=radius_pct·min(H,W)
  - CIRCLE_BLACK    + RANDOM  → círculo aleatório (mantém dentro do frame)
  - CIRCLE_BLACK    + EYES    → círculo entre olhos (via FFService landmarks)
  - CIRCLE_BLACK    + MOUTH   → círculo entre cantos da boca
  - MASK_FACE       + (N/A)   → polígonos/retângulos/elipses desenhados
                                no canvas (coords em [0..1])

Combinações inválidas (RECTANGLE_BLACK + EYES/MOUTH) levantam erro
claro — esse caso não tem dimensionamento natural.

EYES/MOUTH exigem `ff_service` injetado no constructor (uniforme via
factory `ImageProcessorFactory.build`). Sem face detectada na
imagem, cai em CENTER (fallback defensivo).

MASK_FACE não usa `position`. Coordenadas armazenadas em [0..1] são
escaladas pra dimensões da imagem em runtime — mesma máscara aplica
em imagens de tamanhos diferentes.
"""

from __future__ import annotations

import math
from random import Random
from typing import Optional, Tuple

import cv2
import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_mask_shape import OcclusionMaskShape
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.occlusion_config import OcclusionConfig


_BLACK_BGR = (0, 0, 0)


class OcclusionProcessor:
    """Aplica oclusão geométrica conforme `OcclusionConfig`."""

    def __init__(
        self,
        config: OcclusionConfig,
        *,
        ff_service=None,  # PR I7 — necessário pra EYES/MOUTH
    ):
        self._config = config
        self._ff_service = ff_service

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
        kind = self._config.kind
        position = self._config.position

        # Validação de combinação inválida
        if kind == OcclusionKind.RECTANGLE_BLACK and position in (
            OcclusionPosition.EYES, OcclusionPosition.MOUTH,
        ):
            raise ValueError(
                f"Combinação inválida: {kind.name} + {position.name}. "
                f"Posições EYES/MOUTH só são suportadas com "
                f"OcclusionKind.CIRCLE_BLACK (sem dimensionamento "
                f"natural pra retângulo sobre região semântica)."
            )

        if kind == OcclusionKind.RECTANGLE_BLACK:
            return self._apply_rectangle_black(image, rng)
        if kind == OcclusionKind.CIRCLE_BLACK:
            return self._apply_circle_black(image, rng)
        if kind == OcclusionKind.MASK_FACE:
            return self._apply_mask_face(image)
        raise NotImplementedError(
            f"OcclusionKind.{kind.name} ainda não implementado."
        )

    # ── RECTANGLE_BLACK (PR I1) ──────────────────────────────────────────────

    def _apply_rectangle_black(
        self, image: np.ndarray, rng: Optional[Random],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        ratio = math.sqrt(float(self._config.coverage_pct))
        rect_w = max(1, int(w * ratio))
        rect_h = max(1, int(h * ratio))

        position = self._config.position
        if position == OcclusionPosition.RANDOM:
            if rng is None:
                position = OcclusionPosition.CENTER  # defensivo
            else:
                x = rng.randint(0, max(0, w - rect_w))
                y = rng.randint(0, max(0, h - rect_h))

        if position == OcclusionPosition.CENTER:
            x = (w - rect_w) // 2
            y = (h - rect_h) // 2

        out = image.copy()
        out[y:y + rect_h, x:x + rect_w] = _BLACK_BGR
        return out

    # ── CIRCLE_BLACK (PR I7) ─────────────────────────────────────────────────

    def _apply_circle_black(
        self, image: np.ndarray, rng: Optional[Random],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        radius = max(1, int(min(h, w) * self._config.radius_pct))

        center = self._compute_center(image, rng, radius)
        if center is None:
            # Fallback defensivo — face não detectada em EYES/MOUTH
            center = (w // 2, h // 2)

        # cv2.circle muta in-place — copiamos antes
        out = image.copy()
        cv2.circle(out, center, radius, _BLACK_BGR, thickness=-1)
        return out

    # ── Cálculo de centro ────────────────────────────────────────────────────

    def _compute_center(
        self,
        image: np.ndarray,
        rng: Optional[Random],
        radius: int,
    ) -> Optional[Tuple[int, int]]:
        """Devolve (cx, cy) conforme `position`. None pra EYES/MOUTH
        quando detecção falha (caller faz fallback)."""
        h, w = image.shape[:2]
        position = self._config.position

        if position == OcclusionPosition.CENTER:
            return (w // 2, h // 2)

        if position == OcclusionPosition.RANDOM:
            if rng is None:
                return (w // 2, h // 2)  # defensivo
            # Mantém círculo dentro da imagem
            min_x = radius
            min_y = radius
            max_x = max(min_x, w - radius)
            max_y = max(min_y, h - radius)
            return (rng.randint(min_x, max_x), rng.randint(min_y, max_y))

        if position in (OcclusionPosition.EYES, OcclusionPosition.MOUTH):
            return self._compute_semantic_center(image, position)

        return None

    def _compute_semantic_center(
        self, image: np.ndarray, position: OcclusionPosition,
    ) -> Optional[Tuple[int, int]]:
        """Centro semântico baseado em landmarks do FFService.

        Keypoints do RetinaFace (insightface):
          [0] = olho esquerdo
          [1] = olho direito
          [2] = nariz
          [3] = canto esquerdo da boca
          [4] = canto direito da boca

        EYES: midpoint(kps[0], kps[1])
        MOUTH: midpoint(kps[3], kps[4])
        """
        if self._ff_service is None:
            raise RuntimeError(
                f"OcclusionPosition.{position.name} requer FFService "
                "injetado. Construa o pipeline passando ff_service=... "
                "(ImageProcessingPipeline(config, ff_service=...))."
            )
        engine = self._ff_service.app
        try:
            faces = engine.backend.detect_faces(image)
        except Exception as e:
            print(f"Error detect_faces in OcclusionProcessor: {e}")
            return None
        if not faces:
            return None

        face = engine._get_best_face(image, faces, criterion="size")
        kps = getattr(face, "kps", None)
        if kps is None or len(kps) < 5:
            return None

        if position == OcclusionPosition.EYES:
            cx = int((float(kps[0][0]) + float(kps[1][0])) / 2)
            cy = int((float(kps[0][1]) + float(kps[1][1])) / 2)
        else:  # MOUTH
            cx = int((float(kps[3][0]) + float(kps[4][0])) / 2)
            cy = int((float(kps[3][1]) + float(kps[4][1])) / 2)

        return (cx, cy)

    # ── MASK_FACE (PR I6) ────────────────────────────────────────────────────

    def _apply_mask_face(self, image: np.ndarray) -> np.ndarray:
        """Pinta em preto cada shape da `OcclusionMask` configurada.

        Coordenadas dos shapes estão em [0..1] — escaladas pra
        dimensões da imagem corrente.

        Mask vazia (sem shapes) → no-op (devolve cópia inalterada).
        Esse caso ocorre quando o usuário escolhe MASK_FACE mas ainda
        não desenhou nada no canvas.
        """
        mask = self._config.mask
        if mask is None or mask.is_empty:
            return image.copy()

        h, w = image.shape[:2]
        out = image.copy()

        for shape in mask.shapes:
            if shape.shape_type == OcclusionMaskShape.POLYGON:
                if len(shape.points) < 3:
                    continue  # polígono inválido, pula
                pts = np.array(
                    [[int(p[0] * w), int(p[1] * h)] for p in shape.points],
                    dtype=np.int32,
                )
                cv2.fillPoly(out, [pts], _BLACK_BGR)
            elif shape.shape_type == OcclusionMaskShape.RECTANGLE:
                if len(shape.points) < 2:
                    continue
                (x1, y1), (x2, y2) = shape.points[0], shape.points[1]
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                cv2.rectangle(
                    out, (px1, py1), (px2, py2),
                    _BLACK_BGR, thickness=-1,
                )
            elif shape.shape_type == OcclusionMaskShape.ELLIPSE:
                if len(shape.points) < 2:
                    continue
                (x1, y1), (x2, y2) = shape.points[0], shape.points[1]
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2
                rx = max(1, abs(px2 - px1) // 2)
                ry = max(1, abs(py2 - py1) // 2)
                cv2.ellipse(
                    out, (cx, cy), (rx, ry),
                    angle=0, startAngle=0, endAngle=360,
                    color=_BLACK_BGR, thickness=-1,
                )

        return out
