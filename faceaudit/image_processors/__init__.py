"""Processadores de imagem (PR I0).

Implementações concretas do `ImageProcessingStepProtocol`. Cada
processador recebe seu config DTO no constructor (injection) e
implementa `apply(image, *, rng)`.

I0 expõe apenas BLUR (Gaussian) e JPEG. I1+ adiciona WebP, CROP,
OCCLUSION.
"""

from .blur_processor import BlurProcessor
from .jpeg_compression_processor import JpegCompressionProcessor

__all__ = ["BlurProcessor", "JpegCompressionProcessor"]
