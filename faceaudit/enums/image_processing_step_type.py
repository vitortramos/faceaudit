"""Tipos de step do pipeline de processamento de imagens (PR I0).

5 tipos cobertos pelo plano:
  - BLUR — convoluções (gaussian/motion/box)
  - JPEG — compressão lossy (re-encode + decode)
  - WEBP — compressão lossy/lossless (re-encode + decode)
  - CROP — recorte (modos: usuário desenha / face detectada / centralizado)
  - OCCLUSION — oclusão (manual via canvas / semântica via landmarks)

I0 implementa apenas BLUR e JPEG. Demais entram em I1+.

Passos de impressão, cópia e digitalização (xerox de documento):
  - RESIZE, GRAYSCALE, TONE_CURVE, TONER_GAIN, HALFTONE, TONER_NOISE

Values estáveis em disco — JSON da pipeline usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class ImageProcessingStepType(str, Enum):
    BLUR = "blur"
    JPEG = "jpeg"
    WEBP = "webp"
    CROP = "crop"
    OCCLUSION = "occlusion"
    RESIZE = "resize"
    GRAYSCALE = "grayscale"
    TONE_CURVE = "tone_curve"
    TONER_GAIN = "toner_gain"
    HALFTONE = "halftone"
    TONER_NOISE = "toner_noise"

    @property
    def label_pt(self) -> str:
        """Rótulo em português pra UI."""
        return {
            "blur": "Desfoque",
            "jpeg": "Compressão JPEG",
            "webp": "Compressão WebP",
            "crop": "Recorte",
            "occlusion": "Oclusão",
            "resize": "Redimensionar",
            "grayscale": "Tons de cinza",
            "tone_curve": "Curva de tons",
            "toner_gain": "Ganho de toner",
            "halftone": "Meio-tom",
            "toner_noise": "Ruído de toner",
        }.get(self.value, self.value)

    @property
    def category_pt(self) -> str:
        """Grupo do passo no menu Adicionar do editor."""
        return _CATEGORIAS[self._categoria()]

    @property
    def category_order(self) -> int:
        """Posição do grupo no menu Adicionar."""
        return self._categoria()

    def _categoria(self) -> int:
        return {
            "crop": 0, "resize": 0,
            "grayscale": 1, "tone_curve": 1,
            "toner_gain": 2, "halftone": 2, "toner_noise": 2,
            "blur": 3,
            "jpeg": 4, "webp": 4,
            "occlusion": 5,
        }.get(self.value, 6)


_CATEGORIAS = (
    "Enquadramento",
    "Tom",
    "Impressão e cópia",
    "Desfoque",
    "Compressão",
    "Oclusão",
    "Outros",
)
