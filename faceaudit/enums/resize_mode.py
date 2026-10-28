"""Como o passo Redimensionar escolhe a escala.

Values estáveis em disco — JSON do ResizeConfig usa o `.value`.
"""

from __future__ import annotations

from enum import Enum


class ResizeMode(str, Enum):
    # Escala que leva a distância entre os olhos ao valor alvo. É o modo
    # que iguala o tamanho da FACE entre imagens de resoluções diferentes.
    TARGET_IPD = "target_ipd"
    FACTOR = "factor"

    @property
    def label_pt(self) -> str:
        return {
            "target_ipd": "Distância entre os olhos (IPD)",
            "factor": "Fator fixo",
        }.get(self.value, self.value)


class ResizeInterpolation(str, Enum):
    # Média de área ao reduzir (não cria serrilhado) e bicúbica ao ampliar.
    AUTO = "auto"
    AREA = "area"
    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"

    @property
    def label_pt(self) -> str:
        return {
            "auto": "Automática (área ao reduzir, bicúbica ao ampliar)",
            "area": "Média de área",
            "linear": "Bilinear",
            "cubic": "Bicúbica",
            "nearest": "Vizinho mais próximo",
        }.get(self.value, self.value)
