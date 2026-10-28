"""Protocolo dos steps do pipeline de processamento de imagens (PR I0).

Contraste com `EmbeddingTransformationProtocol`: aqui é magro porque
processadores de imagem são determinísticos — não aprendem
parâmetros (sem fit), não têm state pra DUAL mode (não há DUAL),
não persistem em pickle (config JSON basta).

Cada implementação concreta vive em `faceaudit/image_processors/`,
estende `ImageStepProcessorBase` e recebe seu config DTO via
constructor. A factory (`ImageProcessorFactory`) instancia a partir do
config.

A geometria da face (`ImageFaceGeometry`) acompanha a imagem: cada passo
a recebe e devolve convertida para a imagem que produziu.
"""

from __future__ import annotations

from random import Random
from typing import Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.models.image_processing.image_face_geometry import (
    ImageFaceGeometry,
)


@runtime_checkable
class ImageProcessingStepProtocol(Protocol):
    """Contrato comum a todos os steps do pipeline de imagens."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        """Tipo do step. Deve bater com `config.step_type`."""
        ...

    @property
    def config(self):  # type: ignore[no-untyped-def]
        """Config DTO usado na construção (BlurConfig, JpegConfig, ...)."""
        ...

    @property
    def requires_face(self) -> bool:
        """True quando o passo precisa da geometria da face."""
        ...

    def apply_step(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> Tuple[np.ndarray, Optional[ImageFaceGeometry]]:
        """Aplica o step e devolve `(imagem, geometria convertida)`.

        Args:
            image: ndarray BGR uint8 (H, W, 3). Não é mutado.
            rng: gerador para steps com aleatoriedade.
            face: geometria da face no sistema de `image`, ou None.

        Returns:
            Imagem nova (BGR uint8; a forma pode mudar) e a geometria no
            sistema dela — None quando não havia ou não pôde ser convertida.
        """
        ...

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
        face: Optional[ImageFaceGeometry] = None,
    ) -> np.ndarray:
        """Atalho de `apply_step` que devolve só a imagem."""
        ...
