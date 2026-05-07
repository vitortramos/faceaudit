"""Base abstrata pros configs de step (PR I0).

Cada step concreto (BlurConfig, JpegConfig, ...) herda dessa base
e adiciona seus próprios campos. O `step_type` discrimina o tipo
durante deserialização polimórfica via
`ImageProcessorFactory.config_from_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType


@dataclass
class ImageProcessingStepConfig:
    """Base — não usar diretamente. Concrete configs herdam e
    sobrescrevem `step_type` com o valor correto."""

    step_type: ImageProcessingStepType

    def to_dict(self) -> Dict[str, Any]:
        """Serializa pra dict JSON-safe.

        Concrete configs devem chamar `super().to_dict()` e adicionar
        seus campos específicos.
        """
        return {"step_type": self.step_type.value}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageProcessingStepConfig":
        """Não chamar direto na base — usar
        `ImageProcessorFactory.config_from_dict(data)` que despacha
        polimorficamente baseado em `data["step_type"]`."""
        raise NotImplementedError(
            "Use ImageProcessorFactory.config_from_dict(data) — não "
            "chame ImageProcessingStepConfig.from_dict diretamente."
        )
