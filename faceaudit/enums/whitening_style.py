from __future__ import annotations
from enum import Enum
from typing import Optional

class WhiteningStyle(Enum):
    """
    Define o estilo de aplicação do Whitening (Branqueamento).
    
    Attributes:
        PCA: Rotaciona para os eixos principais. Padrão para PLDA e redução de dimensionalidade.
        ZCA: Mantém a orientação original. Frequentemente usado em processamento de imagem/CNNs.
    """
    PCA = "pca"
    ZCA = "zca"

    @property
    def label_pt(self) -> str:
        return {
            "pca": "PCA (rotação aos eixos principais)",
            "zca": "ZCA (mantém orientação original)",
        }.get(self.value, self.value)

    def __str__(self) -> str:
        """Retorna o valor técnico em minúsculas (ex: 'pca')."""
        return self.value

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna a string técnica para o JSON (ex: 'pca')."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> WhiteningStyle:
        """
        Reconstrói o Enum a partir da string com fallback seguro.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.PCA