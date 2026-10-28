"""Técnica de meio-tom do passo Meio-tom.

Values estáveis em disco — JSON do HalftoneConfig usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class HalftoneMode(str, Enum):
    # Difusão de erro (Floyd–Steinberg): grão irregular, típico de
    # copiadora e impressora a laser digitais.
    ERROR_DIFFUSION = "error_diffusion"
    # Pontilhado ordenado (matriz de Bayer 8×8): padrão regular em xadrez.
    ORDERED = "ordered"
    # Retícula de pontos (AM): pontos que crescem com o escuro, numa grade
    # inclinada — impressão gráfica e algumas copiadoras.
    AM_SCREEN = "am_screen"

    @property
    def label_pt(self) -> str:
        return {
            "error_diffusion": "Difusão de erro (Floyd–Steinberg)",
            "ordered": "Pontilhado ordenado (Bayer)",
            "am_screen": "Retícula de pontos",
        }.get(self.value, self.value)
