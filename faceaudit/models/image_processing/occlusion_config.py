"""Config de oclusão (PR I1).

I1 implementa OcclusionKind.RECTANGLE_BLACK + position={CENTER, RANDOM}.

`coverage_pct` ∈ (0, 1] — fração da ÁREA da imagem coberta. Largura
e altura do retângulo derivam de `sqrt(coverage_pct) * (W, H)`,
mantendo proporção da imagem original.

`position=RANDOM` requer RNG (passada via `apply(image, *, rng)`).
Se RNG for None, a posição cai em CENTER (defensivo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)


@dataclass
class OcclusionConfig(ImageProcessingStepConfig):
    """Oclusão geométrica simples (I1)."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.OCCLUSION
    kind: OcclusionKind = OcclusionKind.RECTANGLE_BLACK
    position: OcclusionPosition = OcclusionPosition.CENTER
    coverage_pct: float = 0.25  # 25% da área da imagem

    def __post_init__(self):
        # Clamp
        if self.coverage_pct <= 0:
            self.coverage_pct = 0.01
        elif self.coverage_pct > 1.0:
            self.coverage_pct = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "kind": self.kind.value,
            "position": self.position.value,
            "coverage_pct": float(self.coverage_pct),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OcclusionConfig":
        return cls(
            kind=OcclusionKind(
                data.get("kind", OcclusionKind.RECTANGLE_BLACK.value)
            ),
            position=OcclusionPosition(
                data.get("position", OcclusionPosition.CENTER.value)
            ),
            coverage_pct=float(data.get("coverage_pct", 0.25)),
        )
