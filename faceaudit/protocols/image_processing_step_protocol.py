"""Protocolo dos steps do pipeline de processamento de imagens (PR I0).

Contraste com `EmbeddingTransformationProtocol`: aqui é magro porque
processadores de imagem são determinísticos — não aprendem
parâmetros (sem fit), não têm state pra DUAL mode (não há DUAL),
não persistem em pickle (config JSON basta).

Cada implementação concreta vive em `faceaudit/image_processors/`
e recebe seu config DTO via constructor. A factory
(`ImageProcessorFactory`) instancia a partir do config.
"""

from __future__ import annotations

from random import Random
from typing import Optional, Protocol, runtime_checkable

import numpy as np

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType


@runtime_checkable
class ImageProcessingStepProtocol(Protocol):
    """Contrato comum a todos os steps do pipeline de imagens."""

    @property
    def step_type(self) -> ImageProcessingStepType:
        """Tipo do step. Deve bater com `config.step_type`."""
        ...

    @property
    def config(self):  # type: ignore[no-untyped-def]
        """Config DTO usado na construção (BlurConfig, JpegConfig, ...).
        Mantido como `Any` no protocolo pra evitar acoplar todos os
        concrete configs aqui — checagem in-place via `step_type`."""
        ...

    def apply(
        self,
        image: np.ndarray,
        *,
        rng: Optional[Random] = None,
    ) -> np.ndarray:
        """Aplica o step na imagem.

        Args:
            image: ndarray BGR uint8 (formato cv2). Forma (H, W, 3) pra
                imagens originais ou (112, 112, 3) pra faces alinhadas.
            rng: gerador opcional pra steps com aleatoriedade (random
                crop, oclusão random). I0 (BLUR + JPEG) ignora.

        Returns:
            ndarray BGR uint8 — sempre uma imagem nova (não muta input).
            Forma pode mudar (ex: CROP).
        """
        ...
