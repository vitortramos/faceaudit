"""Tipos de step do pipeline de processamento de imagens (PR I0).

5 tipos cobertos pelo plano:
  - BLUR — convoluções (gaussian/motion/box)
  - JPEG — compressão lossy (re-encode + decode)
  - WEBP — compressão lossy/lossless (re-encode + decode)
  - CROP — recorte (modos: usuário desenha / face detectada / centralizado)
  - OCCLUSION — oclusão (manual via canvas / semântica via landmarks)

I0 implementa apenas BLUR e JPEG. Demais entram em I1+.

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

    @property
    def label_pt(self) -> str:
        """Rótulo em português pra UI."""
        return {
            "blur": "Desfoque",
            "jpeg": "Compressão JPEG",
            "webp": "Compressão WebP",
            "crop": "Recorte",
            "occlusion": "Oclusão",
        }.get(self.value, self.value)

    @property
    def category_pt(self) -> str:
        """Grupo do passo no menu Adicionar do editor."""
        return {
            "crop": "Enquadramento",
            "blur": "Degradação",
            "jpeg": "Compressão",
            "webp": "Compressão",
            "occlusion": "Oclusão",
        }.get(self.value, "Outros")
