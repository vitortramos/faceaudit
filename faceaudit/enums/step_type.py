from __future__ import annotations
from enum import Enum
from typing import Optional

class StepType(Enum):
    """
    Define os tipos fundamentais de etapas de processamento biométrico.
    Utilizado para identificar a natureza matemática de cada camada no pipeline.
    """
    RAW         = "RAW"
    WHITENING   = "WHITENING"
    CORAL       = "CORAL"
    LDA         = "LDA"
    L2_NORM     = "L2_NORM"
    CENTERING   = "CENTERING"
    AGGREGATION = "AGGREGATION"
    RADIAL_GAUSSIANIZATION = "RADIAL_GAUSSIANIZATION"
    UNKNOWN     = "UNKNOWN"

    @property
    def label_pt(self) -> str:
        return {
            "RAW":         "Bruto",
            "WHITENING":   "Branqueamento",
            "CORAL":       "CORAL",
            "LDA":         "LDA",
            "L2_NORM":     "Normalização L2",
            "CENTERING":   "Centralização",
            "AGGREGATION": "Agregação",
            "RADIAL_GAUSSIANIZATION": "Gaussianização Radial",
            "UNKNOWN":     "Desconhecido",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        """Retorna o nome técnico da etapa."""
        return self.value

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o valor identificador para persistência (ex: 'CORAL')."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> StepType:
        """
        Reconstrói o Enum a partir da string. 
        Retorna UNKNOWN como fallback em caso de valor não mapeado.
        """
        try:
            return cls(value.upper() if value else "")
        except ValueError:
            return cls.UNKNOWN