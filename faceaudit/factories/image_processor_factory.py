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

from typing import Any, Dict, List, Type

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.image_processors.blur_processor import BlurProcessor
from faceaudit.image_processors.crop_processor import CropProcessor
from faceaudit.image_processors.jpeg_compression_processor import (
    JpegCompressionProcessor,
)
from faceaudit.image_processors.occlusion_processor import OcclusionProcessor
from faceaudit.image_processors.webp_compression_processor import (
    WebpCompressionProcessor,
)
from faceaudit.image_processors.copy_degradation_processors import (
    GrayscaleProcessor,
    HalftoneProcessor,
    ResizeProcessor,
    ToneCurveProcessor,
    TonerGainProcessor,
    TonerNoiseProcessor,
)
from faceaudit.models.image_processing.blur_config import BlurConfig
from faceaudit.models.image_processing.copy_degradation_configs import (
    GrayscaleConfig,
    HalftoneConfig,
    ResizeConfig,
    ToneCurveConfig,
    TonerGainConfig,
    TonerNoiseConfig,
)
from faceaudit.models.image_processing.crop_config import CropConfig
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.jpeg_compression_config import (
    JpegCompressionConfig,
)
from faceaudit.models.image_processing.occlusion_config import OcclusionConfig
from faceaudit.models.image_processing.webp_compression_config import (
    WebpCompressionConfig,
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
        ImageProcessingStepType.WEBP: WebpCompressionProcessor,  # PR I1
        ImageProcessingStepType.CROP: CropProcessor,             # PR I1
        ImageProcessingStepType.OCCLUSION: OcclusionProcessor,   # PR I1
        ImageProcessingStepType.RESIZE: ResizeProcessor,
        ImageProcessingStepType.GRAYSCALE: GrayscaleProcessor,
        ImageProcessingStepType.TONE_CURVE: ToneCurveProcessor,
        ImageProcessingStepType.TONER_GAIN: TonerGainProcessor,
        ImageProcessingStepType.HALFTONE: HalftoneProcessor,
        ImageProcessingStepType.TONER_NOISE: TonerNoiseProcessor,
    }

    # step_type → config class (pra deserialização polimórfica)
    _CONFIG_REGISTRY: Dict[ImageProcessingStepType, Type] = {
        ImageProcessingStepType.BLUR: BlurConfig,
        ImageProcessingStepType.JPEG: JpegCompressionConfig,
        ImageProcessingStepType.WEBP: WebpCompressionConfig,     # PR I1
        ImageProcessingStepType.CROP: CropConfig,                # PR I1
        ImageProcessingStepType.OCCLUSION: OcclusionConfig,      # PR I1
        ImageProcessingStepType.RESIZE: ResizeConfig,
        ImageProcessingStepType.GRAYSCALE: GrayscaleConfig,
        ImageProcessingStepType.TONE_CURVE: ToneCurveConfig,
        ImageProcessingStepType.TONER_GAIN: TonerGainConfig,
        ImageProcessingStepType.HALFTONE: HalftoneConfig,
        ImageProcessingStepType.TONER_NOISE: TonerNoiseConfig,
    }

    @classmethod
    def build(
        cls,
        config: ImageProcessingStepConfig,
        *,
        ff_service=None,
    ) -> ImageProcessingStepProtocol:
        """Constrói um processor a partir do config.

        Args:
            config: DTO específico do step (BlurConfig, CropConfig...).
            ff_service: opcional. Necessário pra `CropMode.FACE_DETECTED`
                e (futuramente) `OcclusionPosition.EYES/MOUTH`. Os
                outros processors aceitam o kwarg uniformemente mas
                ignoram.

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
        return processor_cls(config, ff_service=ff_service)

    @classmethod
    def available_step_types(cls) -> List[ImageProcessingStepType]:
        """Tipos com processador registrado, agrupados pela categoria do
        menu (e na ordem do enum dentro dela)."""
        tipos = [t for t in ImageProcessingStepType if t in cls._PROCESSOR_REGISTRY]
        return sorted(tipos, key=lambda t: t.category_order)

    @classmethod
    def new_config(cls, step_type: ImageProcessingStepType) -> ImageProcessingStepConfig:
        """Config de um passo recém-adicionado no editor."""
        config_cls = cls._CONFIG_REGISTRY.get(step_type)
        if config_cls is None:
            raise ValueError(f"Sem config registrado pra step_type={step_type.value}.")
        return config_cls.editor_default()

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
