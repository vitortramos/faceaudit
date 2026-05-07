"""Config de desfoque (PR I0).

I0 implementa só BlurKind.GAUSSIAN. Os campos `kernel_size` e
`sigma` controlam o filtro `cv2.GaussianBlur`. MOTION e BOX (em I1)
vão usar campos adicionais (`angle` pra motion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from faceaudit.enums.blur_kind import BlurKind
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)


@dataclass
class BlurConfig(ImageProcessingStepConfig):
    """Desfoque gaussiano (I0). Kernel ímpar exigido pelo cv2.

    Defaults: kernel 5×5, sigma 1.0 — desfoque leve.
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.BLUR
    kind: BlurKind = BlurKind.GAUSSIAN
    kernel_size: int = 5
    sigma: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "kind": self.kind.value,
            "kernel_size": int(self.kernel_size),
            "sigma": float(self.sigma),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlurConfig":
        return cls(
            kind=BlurKind(data.get("kind", BlurKind.GAUSSIAN.value)),
            kernel_size=int(data.get("kernel_size", 5)),
            sigma=float(data.get("sigma", 1.0)),
        )
