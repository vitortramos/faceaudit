from __future__ import annotations
from enum import Enum
from typing import Optional

class LRMethodType(Enum):
    """
    Define os tipos de abordagens de Likelihood Ratio (LR) disponíveis no sistema.
    
    Attributes:
        SCORE_BASED: Modelos que operam sobre a distribuição de scores de similaridade (ex: KDE).
        FEATURE_BASED: Modelos que operam diretamente no subespaço latente (ex: PLDA).
    """
    SCORE_BASED = "Baseada no escore de similaridade"
    FEATURE_BASED = "Baseada no vetor de características"

    @property
    def label_pt(self) -> str:
        """Alias para `value` (já em PT-BR). Mantido por consistência de API."""
        return self.value

    def __str__(self) -> str:
        """Retorna a descrição amigável do método."""
        return self.value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}.{self.name}: '{self.value}'>"

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o nome da constante para persistência (ex: 'SCORE_BASED')."""
        return self.name

    @classmethod
    def from_str(cls, value: str) -> LRMethodType:
        """
        Reconstrói o Enum a partir do nome da constante.
        Fallback para SCORE_BASED em caso de erro.
        """
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.SCORE_BASED