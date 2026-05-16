from __future__ import annotations
from enum import Enum
from typing import Optional

class LRModelType(Enum):
    """
    Define os tipos de modelos de Likelihood Ratio (LR) disponíveis no sistema.
    Cada membro corresponde ao nome identificador da estratégia técnica.
    """
    PLDA_STRATEGY = "PLDAStrategy"
    KDE_STRATEGY = "KDEStrategy"
    # Adicione outros modelos de LR aqui conforme forem criados (ex: WEIBULL_STRATEGY)

    @property
    def label_pt(self) -> str:
        return {
            "PLDAStrategy": "Estratégia PLDA",
            "KDEStrategy":  "Estratégia KDE",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        """Retorna o valor identificador da estratégia (ex: 'PLDAStrategy')."""
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o nome da constante para persistência (ex: 'PLDA_STRATEGY')."""
        return self.name

    @classmethod
    def from_str(cls, value: str) -> LRModelType:
        """
        Reconstrói o Enum a partir do nome da constante.
        Fallback para KDE_STRATEGY por ser o modelo estatístico base.
        """
        if not value:
            return cls.KDE_STRATEGY
            
        try:
            # Tenta buscar pelo nome da constante (ex: 'PLDA_STRATEGY')
            return cls[value.upper()]
        except (KeyError, AttributeError):
            # Fallback secundário: tenta buscar pelo valor da string (ex: 'PLDAStrategy')
            # Útil para retrocompatibilidade com JSONs antigos que salvaram o .value
            for member in cls:
                if member.value == value:
                    return member
            return cls.KDE_STRATEGY