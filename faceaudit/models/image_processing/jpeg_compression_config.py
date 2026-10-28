"""Config de compressão JPEG (PR I0).

Aplica encode JPEG + decode pra simular o efeito de upload/transit
via canais lossy (mensageiros, e-mail, redes sociais). Defaults:
quality 85 — qualidade típica de mensageiros como WhatsApp.
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
class JpegCompressionConfig(ImageProcessingStepConfig):
    """Recompressão JPEG no quality dado.

    `quality` ∈ [1, 100] — 1 é destruição agressiva, 95 quase
    transparente, 85 simula mensageiros típicos.
    """

    step_type: ImageProcessingStepType = ImageProcessingStepType.JPEG
    quality: int = 85

    def __post_init__(self):
        # Clamp pra range válido sem levantar — facilita roundtrip
        # de configs vindos de UI ou JSON corrompido.
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
                help="1 = destruição agressiva, 85 = mensageiros típicos, 100 = quase sem perda.",
            ),
        ]

    @classmethod
    def editor_default(cls) -> "JpegCompressionConfig":
        return cls(quality=50)

    def summary_pt(self) -> str:
        return f"qualidade {int(self.quality)}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JpegCompressionConfig":
        return cls(quality=int(data.get("quality", 85)))
