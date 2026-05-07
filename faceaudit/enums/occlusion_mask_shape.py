"""Tipos de figura geométrica suportados pelo canvas de oclusão livre
(PR I6).

Cada `OcclusionMask` carrega uma lista de shapes; cada shape tem
um `OcclusionMaskShape` + uma lista de pontos em coordenadas
percentuais [0..1] (independente do tamanho da imagem):
  - POLYGON: lista variável de vértices (≥3)
  - RECTANGLE: 2 pontos (cantos do bbox)
  - ELLIPSE: 2 pontos (cantos do bbox que circunscreve a elipse)

Values estáveis em disco — JSON da OcclusionMask usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class OcclusionMaskShape(str, Enum):
    POLYGON = "polygon"
    RECTANGLE = "rectangle"
    ELLIPSE = "ellipse"

    @property
    def label_pt(self) -> str:
        return {
            "polygon": "Polígono",
            "rectangle": "Retângulo",
            "ellipse": "Elipse",
        }.get(self.value, self.value)
