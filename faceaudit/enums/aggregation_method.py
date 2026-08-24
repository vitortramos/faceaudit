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
    def label_pt(self) -> str:
        return {
            "centroid":      "Centróide Global (template único)",
            "combinatorial": "Combinatória (múltiplos templates sintéticos)",
        }.get(self.value, self.value)

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

    # `label`/`description` são os nomes que o editor inline de DTO aninhado
    # procura via getattr (`NestedDTOInlineWidget`) para montar o texto do
    # combo e a linha de ajuda contextual. Sem eles, o combo mostraria
    # "CENTROID"/"COMBINATORIAL" crus e sem descrição. Delegam ao `info`,
    # que já era a fonte de verdade — nada duplicado.
    @property
    def label(self) -> str:
        return self.info.label

    @property
    def description(self) -> str:
        return self.info.description

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
        # Idempotência: receber um membro já hidratado é comum em round-trips
        # (`AggregationConfig.from_dict` sobre um dict já processado). Sem este
        # guard, `value.lower()` levantaria AttributeError não capturado — o
        # `except ValueError` abaixo não pega.
        if isinstance(value, cls):
            return value
        try:
            return cls(value.lower() if value else "")
        except (ValueError, AttributeError):
            return cls.CENTROID