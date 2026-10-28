"""Geometria da face que acompanha a imagem pelos passos do pipeline.

Os passos de imagem precisam saber onde está a face: redimensionar para
uma distância interocular alvo, ocluir os olhos, recortar a face. Antes
cada passo que precisava disso redetectava a face na imagem que recebia —
já degradada pelos passos anteriores, onde a detecção pode falhar ou
achar a face em outro lugar sem erro aparente.

Agora a geometria entra UMA vez no pipeline (a do dataset, que é a que
gerou o embedding; ou uma detecção na imagem de entrada) e cada passo que
muda o enquadramento a converte para a imagem que devolve. Passo que só
mexe nos pixels (tom, desfoque, compressão) a repassa intacta.

Coordenadas em pixels da imagem corrente. Pontos na ordem do detector:
olho esquerdo, olho direito, nariz, canto esquerdo da boca, canto direito.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ImageFaceGeometry:
    """Caixa `[x1, y1, x2, y2]` e/ou 5 pontos `(5, 2)` de uma face."""

    bbox: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.bbox is not None:
            b = np.asarray(self.bbox, dtype=np.float64).reshape(-1)
            object.__setattr__(self, "bbox", b[:4] if b.size >= 4 else None)
        if self.keypoints is not None:
            k = np.asarray(self.keypoints, dtype=np.float64).reshape(-1)
            object.__setattr__(
                self, "keypoints", k[:10].reshape(5, 2) if k.size >= 10 else None,
            )

    # ── Construção ─────────────────────────────────────────────────────────

    @classmethod
    def from_values(
        cls, bbox: Optional[Sequence[float]], keypoints: Optional[Sequence[float]],
    ) -> Optional["ImageFaceGeometry"]:
        """None quando não há caixa nem pontos utilizáveis."""
        g = cls(
            bbox=None if bbox is None else np.asarray(bbox, dtype=np.float64),
            keypoints=None if keypoints is None else np.asarray(keypoints, dtype=np.float64),
        )
        if g.bbox is None and g.keypoints is None:
            return None
        if g.bbox is not None and not np.all(np.isfinite(g.bbox)):
            g = cls(bbox=None, keypoints=g.keypoints)
        if g.keypoints is not None and not np.all(np.isfinite(g.keypoints)):
            g = cls(bbox=g.bbox, keypoints=None)
        if g.bbox is None and g.keypoints is None:
            return None
        return g

    @classmethod
    def from_face_info(cls, face) -> Optional["ImageFaceGeometry"]:
        """Geometria gravada no dataset (sistema de `image_path`)."""
        ff = getattr(face, "ff_info", None)
        if ff is None:
            return None
        return cls.from_values(getattr(ff, "bbox", None), getattr(ff, "keypoints", None))

    @classmethod
    def from_detection(cls, det_face) -> Optional["ImageFaceGeometry"]:
        """Face devolvida pelo backend do detector (`.bbox`, `.kps`)."""
        return cls.from_values(
            getattr(det_face, "bbox", None), getattr(det_face, "kps", None),
        )

    # ── Medidas ────────────────────────────────────────────────────────────

    @property
    def ipd(self) -> Optional[float]:
        """Distância entre os centros dos olhos, em pixels."""
        if self.keypoints is None:
            return None
        return float(np.linalg.norm(self.keypoints[1] - self.keypoints[0]))

    @property
    def eyes(self) -> Optional[np.ndarray]:
        return None if self.keypoints is None else self.keypoints[:2].copy()

    @property
    def mouth_center(self) -> Optional[np.ndarray]:
        if self.keypoints is None:
            return None
        return (self.keypoints[3] + self.keypoints[4]) / 2.0

    def face_box(self) -> Optional[np.ndarray]:
        """Caixa da face; sem caixa, uma estimada pelos pontos."""
        if self.bbox is not None:
            return self.bbox.copy()
        if self.keypoints is None or self.ipd is None or self.ipd <= 0:
            return None
        # Proporções médias do detector: a caixa tem ~2,4 IPD de largura,
        # e os olhos ficam a ~40% da altura.
        cx = float(self.keypoints[:, 0].mean())
        olhos_y = float(self.keypoints[:2, 1].mean())
        w = 2.4 * self.ipd
        h = 3.0 * self.ipd
        return np.array([cx - w / 2, olhos_y - 0.4 * h, cx + w / 2, olhos_y + 0.6 * h])

    # ── Conversão ──────────────────────────────────────────────────────────

    def scaled(self, sx: float, sy: float) -> "ImageFaceGeometry":
        """Geometria depois de a imagem ser redimensionada por (sx, sy).

        Convenção de pixel por área (a do `cv2.resize`): o centro do
        pixel `i` vai para `(i + 0,5)·s − 0,5`.
        """
        def _s(v: np.ndarray, fx: float, fy: float) -> np.ndarray:
            out = v.copy()
            out[..., 0] = (v[..., 0] + 0.5) * fx - 0.5
            out[..., 1] = (v[..., 1] + 0.5) * fy - 0.5
            return out

        bbox = None
        if self.bbox is not None:
            bbox = _s(self.bbox.reshape(2, 2), sx, sy).reshape(-1)
        kps = None if self.keypoints is None else _s(self.keypoints, sx, sy)
        return ImageFaceGeometry(bbox=bbox, keypoints=kps)

    def translated(self, dx: float, dy: float) -> "ImageFaceGeometry":
        """Geometria depois de um recorte com origem em (−dx, −dy)."""
        bbox = None if self.bbox is None else self.bbox + np.array([dx, dy, dx, dy])
        kps = None if self.keypoints is None else self.keypoints + np.array([dx, dy])
        return ImageFaceGeometry(bbox=bbox, keypoints=kps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": None if self.bbox is None else [round(float(v), 3) for v in self.bbox],
            "keypoints": (
                None if self.keypoints is None
                else [round(float(v), 3) for v in self.keypoints.reshape(-1)]
            ),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["ImageFaceGeometry"]:
        if not data:
            return None
        return cls.from_values(data.get("bbox"), data.get("keypoints"))
