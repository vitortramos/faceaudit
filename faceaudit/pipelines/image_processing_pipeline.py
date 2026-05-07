"""ImageProcessingPipeline — orquestra a sequência de steps (PR I0).

Recebe um `ImageProcessingPipelineConfig`, instancia os processors
via factory, e expõe `process_image(image, *, seed_jitter)` que
aplica os steps em sequência.

Determinismo: `config.seed` (global) + `seed_jitter` opcional (per-
imagem, ex: image_path) → `Random(hash((seed, jitter)))` propagado
pra cada step. Steps determinísticos do I0 ignoram, mas I1+ vão
usar (random crop, oclusão random).
"""

from __future__ import annotations

from random import Random
from typing import List, Optional

import numpy as np

from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.protocols.image_processing_step_protocol import (
    ImageProcessingStepProtocol,
)


class ImageProcessingPipeline:
    """Sequência ordenada de steps aplicada a uma imagem (BGR uint8)."""

    def __init__(self, config: ImageProcessingPipelineConfig):
        self._config = config
        self._steps: List[ImageProcessingStepProtocol] = [
            ImageProcessorFactory.build(step_config)
            for step_config in config.steps
        ]

    @property
    def config(self) -> ImageProcessingPipelineConfig:
        return self._config

    @property
    def steps(self) -> List[ImageProcessingStepProtocol]:
        """Cópia da lista — modificar não afeta o pipeline."""
        return list(self._steps)

    def process_image(
        self,
        image: np.ndarray,
        *,
        seed_jitter: Optional[str] = None,
    ) -> np.ndarray:
        """Aplica os steps em sequência. Não muta o input.

        Args:
            image: ndarray BGR uint8 — original (foto inteira) ou
                aligned face (112×112), conforme `config.input_source`.
            seed_jitter: string opcional pra derivar a RNG per-imagem
                deterministicamente (típico: image_path). Combinada
                com `config.seed` via hash. Sem isso, todas as
                imagens processadas no mesmo pipeline usam a mesma
                RNG — determinístico mas idêntico entre imagens.

        Returns:
            ndarray BGR uint8 — imagem processada. Forma pode mudar
            conforme os steps (ex: CROP altera dimensões).
        """
        if image is None:
            raise ValueError("image não pode ser None")

        rng = Random(hash((self._config.seed, seed_jitter or "")))

        out = image
        for step in self._steps:
            out = step.apply(out, rng=rng)
        return out
