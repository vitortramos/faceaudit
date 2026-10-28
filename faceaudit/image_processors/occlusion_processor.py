"""OcclusionProcessor — oclusão geométrica (PR I1 + I7) ou desenhada
(PR I6).

Implementações suportadas:
  - RECTANGLE_BLACK + CENTER  → retângulo central
                                (W·width_pct × H·height_pct)
  - RECTANGLE_BLACK + RANDOM  → retângulo aleatório (mesma dim, posição via RNG)
  - RECTANGLE_BLACK + EYES    → DOIS retângulos, um sobre cada olho
                                (via FFService landmarks)
  - RECTANGLE_BLACK + MOUTH   → retângulo sobre a boca
  - CIRCLE_BLACK    + CENTER  → círculo central, raio=radius_pct·min(H,W)
  - CIRCLE_BLACK    + RANDOM  → círculo aleatório (mantém dentro do frame)
  - CIRCLE_BLACK    + EYES    → DOIS círculos, um sobre cada olho
                                (via FFService landmarks)
  - CIRCLE_BLACK    + MOUTH   → círculo entre cantos da boca
  - MASK_FACE       + (N/A)   → polígonos/retângulos/elipses desenhados
                                no canvas (coords em [0..1])

EYES/MOUTH usam os pontos da face que acompanham a imagem pelo
pipeline (os do dataset, ou a detecção feita na imagem de entrada).
Sem pontos, o passo FALHA — antes caía em CENTER, o que produzia uma
oclusão no lugar errado sem nada no resultado denunciar.

MASK_FACE não usa `position`. Coordenadas armazenadas em [0..1] são
escaladas pra dimensões da imagem em runtime — mesma máscara aplica
em imagens de tamanhos diferentes.
"""

from __future__ import annotations

from random import Random
from typing import List, Optional, Tuple

import cv2
import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_mask_shape import OcclusionMaskShape
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.image_processors.image_step_base import (
    FaceGeometryRequiredError,
    ImageStepProcessorBase,
)
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)


_BLACK_BGR = (0, 0, 0)


class OcclusionProcessor(ImageStepProcessorBase):
    """Aplica oclusão geométrica conforme `OcclusionConfig`."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        return ImageProcessingStepType.OCCLUSION

    def _process(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
    ) -> np.ndarray:
        kind = self._config.kind

        if kind == OcclusionKind.RECTANGLE_BLACK:
            return self._apply_rectangle_black(image, rng, face)
        if kind == OcclusionKind.CIRCLE_BLACK:
            return self._apply_circle_black(image, rng, face)
        if kind == OcclusionKind.MASK_FACE:
            return self._apply_mask_face(image)
        raise NotImplementedError(
            f"OcclusionKind.{kind.name} ainda não implementado."
        )

    # ── RECTANGLE_BLACK ──────────────────────────────────────────────────────

    def _apply_rectangle_black(
        self, image: np.ndarray, rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        rect_w = max(1, int(w * self._config.width_pct))
        rect_h = max(1, int(h * self._config.height_pct))

        centers = self._compute_centers(
            image, rng, face,
            half_extent_x=rect_w // 2,
            half_extent_y=rect_h // 2,
        )
        if not centers:
            centers = [(w // 2, h // 2)]

        # Cada centro vira um retângulo centrado nele. EYES devolve 2
        # centros (um sobre cada olho); demais devolvem 1.
        out = image.copy()
        for (cx, cy) in centers:
            x1 = max(0, cx - rect_w // 2)
            y1 = max(0, cy - rect_h // 2)
            x2 = min(w, x1 + rect_w)
            y2 = min(h, y1 + rect_h)
            out[y1:y2, x1:x2] = _BLACK_BGR
        return out

    # ── CIRCLE_BLACK (PR I7) ─────────────────────────────────────────────────

    def _apply_circle_black(
        self, image: np.ndarray, rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        radius = max(1, int(min(h, w) * self._config.radius_pct))

        centers = self._compute_centers(
            image, rng, face, half_extent_x=radius, half_extent_y=radius,
        )
        if not centers:
            centers = [(w // 2, h // 2)]

        # cv2.circle muta in-place — copiamos antes e iteramos. EYES
        # devolve 2 centros (um por olho); demais devolvem 1.
        # `lineType=LINE_AA` suaviza a borda — sem isso, círculos pequenos
        # ficam visivelmente serrilhados e dão impressão de não serem
        # circulares.
        out = image.copy()
        for center in centers:
            cv2.circle(
                out, center, radius, _BLACK_BGR,
                thickness=-1, lineType=cv2.LINE_AA,
            )
        return out

    # ── Cálculo de centros ───────────────────────────────────────────────────

    def _compute_centers(
        self,
        image: np.ndarray,
        rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
        *,
        half_extent_x: int,
        half_extent_y: int,
    ) -> List[Tuple[int, int]]:
        """Devolve a lista de centros `(cx, cy)` onde a forma deve ser
        desenhada — geralmente 1, exceto EYES que devolve 2 (um por olho).

        `half_extent_x`/`half_extent_y` definem o "raio" em cada eixo —
        círculo usa o raio mesmo nos dois; retângulo usa metade da
        largura e metade da altura. Usado pra manter o desenho dentro do
        frame quando `position=RANDOM`.

        EYES/MOUTH sem pontos da face levantam
        `FaceGeometryRequiredError`.
        """
        h, w = image.shape[:2]
        position = self._config.position

        if position == OcclusionPosition.CENTER:
            return [(w // 2, h // 2)]

        if position == OcclusionPosition.RANDOM:
            if rng is None:
                return [(w // 2, h // 2)]  # defensivo
            # Mantém a forma dentro da imagem
            min_x = half_extent_x
            min_y = half_extent_y
            max_x = max(min_x, w - half_extent_x)
            max_y = max(min_y, h - half_extent_y)
            return [(rng.randint(min_x, max_x), rng.randint(min_y, max_y))]

        if position in (OcclusionPosition.EYES, OcclusionPosition.MOUTH):
            return self._compute_semantic_centers(face, position)

        return []

    @staticmethod
    def _compute_semantic_centers(
        face: Optional[ImageFaceGeometry], position: OcclusionPosition,
    ) -> List[Tuple[int, int]]:
        """Centros semânticos a partir dos pontos da face.

        EYES: **dois** centros, um sobre cada olho — o usuário espera ver
        os dois olhos cobertos, não um único círculo entre eles.
        MOUTH: **um** centro, no meio dos cantos da boca.
        """
        if face is None or face.keypoints is None:
            raise FaceGeometryRequiredError(
                f"A oclusão '{position.label_pt}' precisa dos pontos da "
                "face, e eles não estão disponíveis para esta imagem."
            )
        if position == OcclusionPosition.EYES:
            return [(int(round(x)), int(round(y))) for x, y in face.eyes]
        cx, cy = face.mouth_center
        return [(int(round(cx)), int(round(cy)))]

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
