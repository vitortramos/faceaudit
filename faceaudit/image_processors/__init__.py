"""Processadores de imagem (PR I0).

Implementações concretas do `ImageProcessingStepProtocol`. Cada
processador recebe seu config DTO no constructor (injection) e
implementa `apply(image, *, rng)`.

I0 expõe apenas BLUR (Gaussian) e JPEG. I1+ adiciona WebP, CROP,
OCCLUSION.
"""

from .blur_processor import BlurProcessor
from .crop_processor import CropProcessor
from .jpeg_compression_processor import JpegCompressionProcessor
from .occlusion_processor import OcclusionProcessor
from .webp_compression_processor import WebpCompressionProcessor

__all__ = [
    "BlurProcessor",
    "CropProcessor",
    "JpegCompressionProcessor",
    "OcclusionProcessor",
    "WebpCompressionProcessor",
]
