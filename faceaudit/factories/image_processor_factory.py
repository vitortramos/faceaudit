"""ImageProcessorFactory — registry extensível pros steps de imagem
(PR I0).

Centraliza dois mapeamentos:
  - `step_type → processor_class` (constrói processor a partir do
    config)
  - `step_type → config_class` (deserializa config polimorficamente
    a partir de dict)

Pra adicionar um novo step (ex: WEBP em I1), basta:
  1. Criar o config DTO em `models/image_processing/`.
  2. Criar o processor em `image_processors/`.
  3. Registrar ambos nos dois dicts abaixo.

Sem precisar tocar em mais nenhum arquivo.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.image_processors.blur_processor import BlurProcessor
from faceaudit.image_processors.jpeg_compression_processor import (
    JpegCompressionProcessor,
)
from faceaudit.models.image_processing.blur_config import BlurConfig
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.jpeg_compression_config import (
    JpegCompressionConfig,
)
from faceaudit.protocols.image_processing_step_protocol import (
    ImageProcessingStepProtocol,
)


class ImageProcessorFactory:
    """Constrói processors a partir de configs e vice-versa."""

    # step_type → processor class
    _PROCESSOR_REGISTRY: Dict[ImageProcessingStepType, Type] = {
        ImageProcessingStepType.BLUR: BlurProcessor,
        ImageProcessingStepType.JPEG: JpegCompressionProcessor,
        # I1+: WEBP, CROP, OCCLUSION
    }

    # step_type → config class (pra deserialização polimórfica)
    _CONFIG_REGISTRY: Dict[ImageProcessingStepType, Type] = {
        ImageProcessingStepType.BLUR: BlurConfig,
        ImageProcessingStepType.JPEG: JpegCompressionConfig,
        # I1+: WEBP, CROP, OCCLUSION
    }

    @classmethod
    def build(
        cls, config: ImageProcessingStepConfig,
    ) -> ImageProcessingStepProtocol:
        """Constrói um processor a partir do config.

        Raises:
            ValueError: se o step_type não estiver registrado.
        """
        processor_cls = cls._PROCESSOR_REGISTRY.get(config.step_type)
        if processor_cls is None:
            raise ValueError(
                f"Sem processor registrado pra step_type={config.step_type.value}. "
                f"Registrados: "
                f"{sorted(t.value for t in cls._PROCESSOR_REGISTRY)}."
            )
        return processor_cls(config)

    @classmethod
    def config_from_dict(
        cls, data: Dict[str, Any],
    ) -> ImageProcessingStepConfig:
        """Deserializa um config polimorficamente a partir de dict.

        Lê `data["step_type"]` e despacha pra classe correta de
        config. Usado por `ImageProcessingPipelineConfig.from_dict`
        ao reconstruir a lista de steps.

        Raises:
            ValueError: se o step_type não estiver registrado.
            KeyError: se `data` não tiver "step_type".
        """
        try:
            step_type = ImageProcessingStepType(data["step_type"])
        except KeyError as e:
            raise KeyError(
                "Config dict não tem chave 'step_type'."
            ) from e
        config_cls = cls._CONFIG_REGISTRY.get(step_type)
        if config_cls is None:
            raise ValueError(
                f"Sem config class registrada pra step_type={step_type.value}. "
                f"Registrados: "
                f"{sorted(t.value for t in cls._CONFIG_REGISTRY)}."
            )
        return config_cls.from_dict(data)
