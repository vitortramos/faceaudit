from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ScoreMetricInfo:
    """Estrutura de metadados para as métricas de comparação."""
    label: str
    description: str

class ScoreMetric(Enum):
    """
    Métrica de distância ou similaridade para comparar vetores biométricos.
    """
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"

    @property
    def info(self) -> ScoreMetricInfo:
        """Retorna metadados ricos para UI e Laudos Técnicos."""
        if self is ScoreMetric.COSINE:
            return ScoreMetricInfo(
                "Similaridade de Cosseno", 
                "Mede o ângulo entre vetores. Padrão para reconhecimento facial (ArcFace/AdaFace). Intervalo [-1, 1]."
            )
        if self is ScoreMetric.EUCLIDEAN:
            return ScoreMetricInfo(
                "Distância Euclidiana (L2)", 
                "Mede a distância geométrica linear. Comum em modelos mais antigos ou triplet loss. Intervalo [0, inf)."
            )
        return ScoreMetricInfo(self.name, "")

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o valor identificador (ex: 'cosine') para o JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> ScoreMetric:
        """
        Reconstrói o Enum a partir da string.
        Retorna COSINE como fallback padrão.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.COSINE