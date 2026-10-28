"""Base comum dos processadores de passo de imagem.

Concentra o que todo passo faz igual:
  - guardar o config;
  - `apply_step(image, rng=, face=)` → `(imagem, geometria)`: o pipeline
    chama este método, que devolve a imagem processada e a geometria da
    face convertida para ela.

Passo que só mexe nos pixels implementa `apply` e herda a regra padrão
de geometria: tamanho igual → a geometria continua valendo; tamanho
diferente → a geometria é descartada (passo que muda o enquadramento e
não sabe convertê-la não pode deixá-la seguir errada). Passo que muda o
enquadramento (recorte, redimensionamento) sobrescreve `apply_step` e
converte.
"""

from __future__ import annotations

from random import Random
from typing import Optional, Tuple

import numpy as np

from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)


class FaceGeometryRequiredError(RuntimeError):
    """O passo precisa saber onde está a face e a geometria não veio."""


class ImageStepProcessorBase:
    """Base dos processadores. Subclasses definem `step_type` e `apply`."""

    def __init__(self, config, *, ff_service=None):
        self._config = config
        # Mantido por uniformidade com a factory; os passos não detectam
        # mais por conta própria — a geometria vem pelo pipeline.
        self._ff_service = ff_service

    @property
    def config(self):
        return self._config

    @property
    def requires_face(self) -> bool:
        return bool(getattr(self._config, "requires_face", False))

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> np.ndarray:
        return self.apply_step(image, rng=rng, face=face)[0]

    def apply_step(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> Tuple[np.ndarray, Optional[ImageFaceGeometry]]:
        if self.requires_face and face is None:
            raise FaceGeometryRequiredError(
                f"O passo '{self.step_type.label_pt}' precisa da posição da "
                "face, e ela não está disponível para esta imagem."
            )
        out = self._process(image, rng=rng, face=face)
        mesma_forma = out.shape[:2] == image.shape[:2]
        return out, (face if mesma_forma else None)

    def _process(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random],
        face: Optional[ImageFaceGeometry],
    ) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _require_face(face: Optional[ImageFaceGeometry], o_que: str) -> ImageFaceGeometry:
        if face is None:
            raise FaceGeometryRequiredError(
                f"{o_que} precisa da posição da face, e ela não está "
                "disponível para esta imagem."
            )
        return face
