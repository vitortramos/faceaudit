"""Fonte de imagem que alimenta o pipeline (PR I0).

Decisão #1 do usuário (opção B): pipeline aceita ambas as fontes,
escolhidas via `ImageProcessingPipelineConfig.input_source`:

  - ORIGINAL — foto inteira (face com background, pose variada).
    Útil pra simular degradação no upload/captura (compressão JPEG
    de mensageiro, blur de movimento na câmera).
  - ALIGNED_FACE — face já detectada e alinhada via ForensicFace
    (`align_face_only`, 112×112 BGR). Útil pra degradar exatamente
    o que o modelo vê — sem ruído de background ou variação de pose.

Pipelines podem ter steps idênticos mas semântica forense distinta
dependendo da fonte escolhida.
"""

from __future__ import annotations

from enum import Enum


class ImageInputSource(str, Enum):
    ORIGINAL = "original"
    ALIGNED_FACE = "aligned_face"

    @property
    def label_pt(self) -> str:
        return {
            "original": "Imagem original",
            "aligned_face": "Face alinhada (112×112)",
        }.get(self.value, self.value)
