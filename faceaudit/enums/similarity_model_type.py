# faceaudit/enums/similarity_model_type.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class SimilarityModelType(Enum):
    """
    Define os tipos de modelo de similaridade suportados pela nova arquitetura.

    Este enum identifica explicitamente qual família de modelo de similaridade
    a instrução pretende usar, evitando inferência implícita por tipo de DTO
    ou presença/ausência de configuração.
    """

    COSINE = (
        "COSINE",
        "Cosine Similarity",
        "Usa similaridade cosseno como função de scoring. "
        "Normalmente não exige treino real de modelo.",
    )

    PLDA = (
        "PLDA",
        "Probabilistic Linear Discriminant Analysis",
        "Usa um modelo PLDA treinado sobre embeddings e identidades para produzir scores de similaridade.",
    )

    def __init__(self, value: str, label: str, description: str):
        self._value_ = value
        self.label = label
        self.description = description

    def __str__(self) -> str:
        """Retorna o label amigável para a interface."""
        return self.label

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    def to_str(self) -> str:
        """Retorna a chave string persistível."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> SimilarityModelType:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        if not value:
            return cls.COSINE

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for item in cls:
                if item.value == val_norm:
                    return item
            return cls.COSINE

    # =========================================================================
    # HELPERS PARA UI
    # =========================================================================

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Útil para combos simples: [(value, label), ...]."""
        return [(item.value, item.label) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        """Útil para UI com tooltips: [(value, label, description), ...]."""
        return [(item.value, item.label, item.description) for item in cls]