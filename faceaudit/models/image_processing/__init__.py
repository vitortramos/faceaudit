"""Configs do pipeline de processamento de imagens (PR I0).

DTOs serializáveis em JSON que descrevem cada step (BlurConfig,
JpegConfig, ...) e o pipeline inteiro
(ImageProcessingPipelineConfig). Pipelines são salvos em
`<global_library>/image_pipelines/<id>.json` (PR I4).
"""

from .blur_config import BlurConfig
from .image_processing_pipeline_config import ImageProcessingPipelineConfig
from .image_processing_step_config import ImageProcessingStepConfig
from .jpeg_compression_config import JpegCompressionConfig

__all__ = [
    "BlurConfig",
    "ImageProcessingPipelineConfig",
    "ImageProcessingStepConfig",
    "JpegCompressionConfig",
]
