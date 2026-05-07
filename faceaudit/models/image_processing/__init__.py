"""Configs do pipeline de processamento de imagens (PR I0).

DTOs serializáveis em JSON que descrevem cada step (BlurConfig,
JpegConfig, ...) e o pipeline inteiro
(ImageProcessingPipelineConfig). Pipelines são salvos em
`<global_library>/image_pipelines/<id>.json` (PR I4).
"""

from .batch_processing_result import BatchProcessingResult
from .blur_config import BlurConfig
from .crop_config import CropConfig
from .image_processing_pipeline_config import ImageProcessingPipelineConfig
from .image_processing_step_config import ImageProcessingStepConfig
from .jpeg_compression_config import JpegCompressionConfig
from .occlusion_config import OcclusionConfig
from .quality_scores_result import QualityScoresResult
from .webp_compression_config import WebpCompressionConfig

__all__ = [
    "BatchProcessingResult",
    "BlurConfig",
    "CropConfig",
    "ImageProcessingPipelineConfig",
    "ImageProcessingStepConfig",
    "JpegCompressionConfig",
    "OcclusionConfig",
    "QualityScoresResult",
    "WebpCompressionConfig",
]
