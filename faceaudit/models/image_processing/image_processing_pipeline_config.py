"""Config do pipeline completo de processamento de imagens (PR I0).

Lista ordenada de steps + parâmetros globais (seed, input_source,
metadata). Persistido como JSON em `<global_library>/image_pipelines/<id>.json`
(PR I4).

Decisões consolidadas:
  - input_source: ORIGINAL ou ALIGNED_FACE (decisão #1 — opção B)
  - seed global no DTO mesmo que I0 não tenha steps aleatórios
    (decisão #5 — preparação pro futuro)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from faceaudit.enums.image_input_source import ImageInputSource
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)


@dataclass
class ImageProcessingPipelineConfig:
    """Pipeline declarativo: nome + lista de steps + globals.

    Attributes:
        name: identificador legível (também usado pra `<id>.json` na
            biblioteca).
        description: descrição opcional.
        input_source: ORIGINAL (foto inteira) ou ALIGNED_FACE (face
            já recortada via FFService.align_face_only).
        seed: semente global pra steps aleatórios (default 42).
        steps: lista ordenada de step configs.
    """

    name: str = "untitled"
    description: Optional[str] = None
    input_source: ImageInputSource = ImageInputSource.ORIGINAL
    seed: int = 42
    steps: List[ImageProcessingStepConfig] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description or "",
            "input_source": self.input_source.value,
            "seed": int(self.seed),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any],
    ) -> "ImageProcessingPipelineConfig":
        # Import local pra evitar ciclo (factory importa configs).
        from faceaudit.factories.image_processor_factory import (
            ImageProcessorFactory,
        )
        steps = [
            ImageProcessorFactory.config_from_dict(s)
            for s in data.get("steps", [])
        ]
        return cls(
            name=str(data.get("name", "untitled")),
            description=data.get("description") or None,
            input_source=ImageInputSource(
                data.get("input_source", ImageInputSource.ORIGINAL.value)
            ),
            seed=int(data.get("seed", 42)),
            steps=steps,
        )

    def to_json_string(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json_string(cls, raw: str) -> "ImageProcessingPipelineConfig":
        return cls.from_dict(json.loads(raw))
