# faceaudit/enums/aggregation_method.py
from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class AggregationMethodInfo:
    label: str
    description: str

class AggregationMethod(Enum):
    """
    Define a estratégia topológica para combinar embeddings (Agrupamento).
    
    Nota: A decisão matemática de usar 'Média Simples' ou 'Média Ponderada' 
    é controlada separadamente pelo parâmetro 'use_quality_weighting'.
    """

    CENTROID = "centroid"
    """
    Agrupamento Global (N -> 1).
    Funde TODAS as amostras da identidade em 1 único template.
    """

    COMBINATORIAL = "combinatorial"
    """
    Agrupamento por Subconjuntos (N -> M).
    Gera múltiplos templates combinando grupos menores (ex: 3 a 3).
    """

    @property
    def info(self) -> AggregationMethodInfo:
        """Retorna metadados ricos para interface e relatórios."""
        if self is AggregationMethod.CENTROID:
            return AggregationMethodInfo(
                label="Centróide Global (Single Template)",
                description="Consolida todas as amostras em um único vetor (Média Simples ou Ponderada)."
            )
        
        if self is AggregationMethod.COMBINATORIAL:
            return AggregationMethodInfo(
                label="Combinatória (Data Augmentation)",
                description="Gera múltiplos templates sintéticos via combinação de subgrupos."
            )
            
        return AggregationMethodInfo(self.name, "")

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Converte o membro para string (usado pelo JSONSerializableMixin)."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> AggregationMethod:
        """
        Reconstrói o Enum a partir de uma string. 
        Retorna CENTROID como fallback em caso de valor inválido.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.CENTROID