"""Config de compressão WebP (PR I1).

Análogo ao JPEG, mas com algoritmo WebP (geralmente melhor
compressão por bit que JPEG, mas ainda lossy quando quality < 100).

`quality` ∈ [1, 100] — clamp idêntico ao JPEG. Quality=100 dá
quase-lossless (cv2.IMWRITE_WEBP_QUALITY aceita até 101 pra
lossless real, mas 100 já é "near lossless" e simplifica a API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
)


@dataclass
class WebpCompressionConfig(ImageProcessingStepConfig):
    """Recompressão WebP no quality dado.

    `quality` ∈ [1, 100]. Defaults: 80 (compressão eficiente).
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.WEBP
    quality: int = 80

    def __post_init__(self):
        if self.quality < 1:
            self.quality = 1
        elif self.quality > 100:
            self.quality = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "quality": int(self.quality),
        }

    def param_specs(self) -> List[StepParamSpec]:
        return [
            StepParamSpec(
                "quality", "Qualidade", StepParamKind.INT,
                minimum=1, maximum=100, step=1, suffix=" / 100",
                help="WebP geralmente preserva melhor que JPEG na mesma qualidade. 80 é eficiente.",
            ),
        ]

    @classmethod
    def editor_default(cls) -> "WebpCompressionConfig":
        return cls(quality=50)

    def summary_pt(self) -> str:
        return f"qualidade {int(self.quality)}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebpCompressionConfig":
        return cls(quality=int(data.get("quality", 80)))
