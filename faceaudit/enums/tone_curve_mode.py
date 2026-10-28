"""Como a Curva de tons define o ponto de preto e o de branco.

Values estáveis em disco — JSON do ToneCurveConfig usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class ToneCurveMode(str, Enum):
    # Níveis fixos (0–255), iguais para todas as imagens.
    FIXED_LEVELS = "fixed_levels"
    # Percentis da região da face, calculados POR IMAGEM. Absorve a
    # diferença de exposição entre as fotos da base: com níveis fixos, uma
    # janela estreita (ex.: 90–160) transformava fotos um pouco mais
    # escuras em cópias quase pretas e um pouco mais claras em cópias
    # quase brancas.
    FACE_PERCENTILES = "face_percentiles"

    @property
    def label_pt(self) -> str:
        return {
            "fixed_levels": "Níveis fixos",
            "face_percentiles": "Percentis da face (por imagem)",
        }.get(self.value, self.value)
