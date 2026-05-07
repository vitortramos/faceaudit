"""Config de recorte (PR I1).

Modos suportados em I1:
  - CENTERED — recorta um retângulo centralizado de
    `(target_width, target_height)`. Erro se a imagem for menor que
    o destino (sem padding automático — explicitar é mais seguro
    forensicamente).
  - FACE_DETECTED — usa `FFService.align_face_only`-style detect
    (apenas detect, sem alinhamento) → bbox da maior face → corta
    bbox + padding opcional. Sem rotação — preserva pose original.

USER_RECTANGLE entra em I6 quando o canvas interativo for criado
(armazena (x, y, w, h) explícitos no config).

`face_padding` é fração da bbox detectada (0.0 = bbox exata,
0.2 = 20% de margem em cada lado).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from faceaudit.enums.crop_mode import CropMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)


@dataclass
class CropConfig(ImageProcessingStepConfig):
    """Recorte conforme o modo escolhido."""

    step_type: ImageProcessingStepType = ImageProcessingStepType.CROP
    mode: CropMode = CropMode.CENTERED
    # CENTERED — alvo
    target_width: int = 224
    target_height: int = 224
    # FACE_DETECTED — margem ao redor da bbox (fração)
    face_padding: float = 0.0

    def __post_init__(self):
        if self.target_width < 1:
            self.target_width = 1
        if self.target_height < 1:
            self.target_height = 1
        if self.face_padding < 0:
            self.face_padding = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "mode": self.mode.value,
            "target_width": int(self.target_width),
            "target_height": int(self.target_height),
            "face_padding": float(self.face_padding),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CropConfig":
        return cls(
            mode=CropMode(data.get("mode", CropMode.CENTERED.value)),
            target_width=int(data.get("target_width", 224)),
            target_height=int(data.get("target_height", 224)),
            face_padding=float(data.get("face_padding", 0.0)),
        )
