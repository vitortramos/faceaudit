"""ImageProcessingPipeline — orquestra a sequência de steps (PR I0).

Recebe um `ImageProcessingPipelineConfig`, instancia os processors via
factory, e aplica os steps em sequência levando junto a geometria da
face (`ImageFaceGeometry`).

Geometria: quem chama passa a do dataset quando a tem (é a da detecção
que gerou o embedding). Sem ela, e só se algum passo precisa, o pipeline
detecta UMA vez, na imagem de entrada — nunca na imagem já degradada por
passos anteriores, onde a detecção pode falhar ou achar a face em outro
lugar sem erro aparente.

Determinismo: `config.seed` + `seed_jitter` (por imagem, ex.: o
image_path) → semente derivada por SHA-256. O `hash()` do Python é
sorteado a cada execução (PYTHONHASHSEED), e com ele a mesma oclusão
"aleatória" caía em lugar diferente cada vez que o programa abria.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from random import Random
from typing import List, Optional

import numpy as np

from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.image_processors.image_step_base import FaceGeometryRequiredError
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.protocols.image_processing_step_protocol import (
    ImageProcessingStepProtocol,
)


def stable_rng(seed: int, jitter: Optional[str]) -> Random:
    """RNG determinística entre execuções para (seed, jitter)."""
    chave = f"{int(seed)}\x00{jitter or ''}".encode("utf-8")
    return Random(int.from_bytes(hashlib.sha256(chave).digest()[:8], "big"))


def detect_face_geometry(ff_service, image: np.ndarray) -> Optional[ImageFaceGeometry]:
    """Detecta a maior face da imagem; None se não houver."""
    if ff_service is None:
        return None
    faces = ff_service.app.backend.detect_faces(image)
    if not faces:
        return None
    # forensicface >= 0.7.1 expõe a seleção como função de módulo.
    from forensicface.geometry import select_best_face

    melhor = select_best_face(image.shape, faces, criterion="size")
    return ImageFaceGeometry.from_detection(melhor)


@dataclass(frozen=True)
class PipelineOutput:
    """Imagem processada + geometria da face nela (None se perdida)."""

    image: np.ndarray
    face: Optional[ImageFaceGeometry]


class ImageProcessingPipeline:
    """Sequência ordenada de steps aplicada a uma imagem (BGR uint8)."""

    def __init__(
        self,
        config: ImageProcessingPipelineConfig,
        *,
        ff_service=None,
    ):
        self._config = config
        self._ff_service = ff_service
        self._steps: List[ImageProcessingStepProtocol] = [
            ImageProcessorFactory.build(step_config, ff_service=ff_service)
            for step_config in config.steps
        ]

    @property
    def config(self) -> ImageProcessingPipelineConfig:
        return self._config

    @property
    def steps(self) -> List[ImageProcessingStepProtocol]:
        """Cópia da lista — modificar não afeta o pipeline."""
        return list(self._steps)

    @property
    def requires_face(self) -> bool:
        return any(s.requires_face for s in self._steps)

    def process(
        self,
        image: np.ndarray,
        *,
        seed_jitter: Optional[str] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> PipelineOutput:
        """Aplica os steps em sequência. Não muta o input.

        Args:
            image: ndarray BGR uint8 — original ou face alinhada,
                conforme `config.input_source`.
            seed_jitter: string para derivar a RNG por imagem (típico:
                image_path).
            face: geometria da face no sistema de `image`. Sem ela, o
                pipeline detecta na entrada quando algum passo precisa.

        Raises:
            FaceGeometryRequiredError: um passo precisa da face e ela
                não veio nem foi encontrada.
        """
        if image is None:
            raise ValueError("image não pode ser None")

        if face is None and self.requires_face:
            face = detect_face_geometry(self._ff_service, image)
            if face is None:
                raise FaceGeometryRequiredError(
                    "O pipeline tem passo que precisa da posição da face, a "
                    "imagem não traz a geometria e "
                    + ("nenhuma face foi detectada nela."
                       if self._ff_service is not None
                       else "o detector não está disponível.")
                )

        rng = stable_rng(self._config.seed, seed_jitter)
        out, geom = image, face
        for step in self._steps:
            out, geom = step.apply_step(out, rng=rng, face=geom)
        return PipelineOutput(image=out, face=geom)

    def process_image(
        self,
        image: np.ndarray,
        *,
        seed_jitter: Optional[str] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> np.ndarray:
        """Atalho de `process` que devolve só a imagem."""
        return self.process(image, seed_jitter=seed_jitter, face=face).image
