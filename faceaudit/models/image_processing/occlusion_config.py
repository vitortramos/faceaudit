"""Config de oclusão (PR I1 + I7).

I1 implementou OcclusionKind.RECTANGLE_BLACK + position={CENTER, RANDOM}.

I7 adiciona OcclusionKind.CIRCLE_BLACK + position={EYES, MOUTH} via
landmarks do FFService. CIRCLE_BLACK aceita qualquer position;
EYES/MOUTH só funcionam com CIRCLE_BLACK no MVP (RECTANGLE_BLACK
sobre região semântica não é trivial de dimensionar).

Campos relevantes por modo:
  - RECTANGLE_BLACK (CENTER ou RANDOM): usa `coverage_pct` (fração
    da ÁREA da imagem coberta). Lado do retângulo derivado de
    `sqrt(coverage_pct) * (W, H)`.
  - CIRCLE_BLACK (qualquer position): usa `radius_pct` (fração do
    MENOR LADO da imagem). Raio = `radius_pct * min(H, W)`.

`position=RANDOM` requer RNG (passada via `apply(image, *, rng)`).
Sem RNG, cai em CENTER (defensivo).

`position=EYES` e `MOUTH` requerem FFService injetado no processor
(via factory). Sem ele, o apply levanta RuntimeError. Se a detecção
falhar (sem face), cai em CENTER.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.occlusion_mask import OcclusionMask


@dataclass
class OcclusionConfig(ImageProcessingStepConfig):
    """Oclusão geométrica (I1 + I7) ou desenhada (I6)."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.OCCLUSION
    kind: OcclusionKind = OcclusionKind.MASK_FACE
    position: OcclusionPosition = OcclusionPosition.CENTER
    coverage_pct: float = 0.25  # PR I1 — RECTANGLE_BLACK em CENTER/RANDOM
    radius_pct: float = 0.10    # PR I7 — CIRCLE_BLACK em qualquer position
    # PR I6 — máscara desenhada pelo usuário no canvas. Só relevante
    # quando kind=MASK_FACE. Position é ignorada nesse modo.
    mask: Optional[OcclusionMask] = None

    def __post_init__(self):
        # Clamp coverage
        if self.coverage_pct <= 0:
            self.coverage_pct = 0.01
        elif self.coverage_pct > 1.0:
            self.coverage_pct = 1.0
        # Clamp radius (≤ 0.5 mantém círculo dentro da imagem mesmo
        # em posição CENTER de imagem quadrada)
        if self.radius_pct <= 0:
            self.radius_pct = 0.01
        elif self.radius_pct > 0.5:
            self.radius_pct = 0.5

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            **super().to_dict(),
            "kind": self.kind.value,
            "position": self.position.value,
            "coverage_pct": float(self.coverage_pct),
            "radius_pct": float(self.radius_pct),
        }
        # PR I6 — só serializa mask quando há shapes (mantém JSON enxuto)
        if self.mask is not None and not self.mask.is_empty:
            out["mask"] = self.mask.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionConfig":
        mask = None
        if "mask" in data and data["mask"]:
            mask = OcclusionMask.from_dict(data["mask"])
        return cls(
            kind=OcclusionKind(
                data.get("kind", OcclusionKind.RECTANGLE_BLACK.value)
            ),
            position=OcclusionPosition(
                data.get("position", OcclusionPosition.CENTER.value)
            ),
            coverage_pct=float(data.get("coverage_pct", 0.25)),
            radius_pct=float(data.get("radius_pct", 0.10)),
            mask=mask,
        )
