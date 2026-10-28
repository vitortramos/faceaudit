"""Config de desfoque (PR I0).

I0 implementa só BlurKind.GAUSSIAN. Os campos `kernel_size` e
`sigma` controlam o filtro `cv2.GaussianBlur`. MOTION e BOX (em I1)
vão usar campos adicionais (`angle` pra motion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from faceaudit.enums.blur_kind import BlurKind
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
    enum_choices,
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

    def param_specs(self) -> List[StepParamSpec]:
        return [
            StepParamSpec(
                "kind", "Tipo", StepParamKind.CHOICE,
                choices=enum_choices([BlurKind.GAUSSIAN]),
            ),
            StepParamSpec(
                "kernel_size", "Kernel", StepParamKind.INT,
                minimum=1, maximum=99, step=2, suffix=" px",
                help="Tamanho do kernel gaussiano. Ímpar (corrigido "
                     "automaticamente se par).",
            ),
            StepParamSpec(
                "sigma", "Sigma", StepParamKind.FLOAT,
                minimum=0.1, maximum=50.0, step=0.5, decimals=2,
                tunable=True, search_min=0.1, search_max=5.0,
                help="Desvio-padrão do kernel gaussiano. Quanto maior, "
                     "mais borrado.",
            ),
        ]

    @classmethod
    def editor_default(cls) -> "BlurConfig":
        return cls(kernel_size=7, sigma=2.0)

    def summary_pt(self) -> str:
        return f"σ {self.sigma:g}"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlurConfig":
        return cls(
            kind=BlurKind(data.get("kind", BlurKind.GAUSSIAN.value)),
            kernel_size=int(data.get("kernel_size", 5)),
            sigma=float(data.get("sigma", 1.0)),
        )
