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

    Estrutura de cada membro:
    - `value`: chave canônica (persistida em JSON).
    - `label`: rótulo em inglês técnico (docs/logs).
    - `label_pt`: rótulo em português para apresentação na UI.
    - `description`: descrição longa em PT-BR (tooltip/help).
    """

    COSINE = (
        "COSINE",
        "Cosine Similarity",
        "Similaridade Cosseno",
        "Usa similaridade cosseno como função de scoring. "
        "Normalmente não exige treino real de modelo.",
    )

    PLDA = (
        "PLDA",
        "Probabilistic Linear Discriminant Analysis",
        "PLDA (Análise Discriminante Linear Probabilística)",
        "Usa um modelo PLDA treinado sobre embeddings e identidades para produzir scores de similaridade.",
    )

    def __new__(cls, value: str, label: str, label_pt: str, description: str):
        # Pattern canônico de Enum com valor composto — ver
        # typicality_reference_mode.py pra explicação completa.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        """Retorna o label PT-BR para a interface."""
        return self.label_pt

    def to_str(self) -> str:
        """Retorna a chave string persistível."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> SimilarityModelType:
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

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Útil para combos simples: [(value, label_pt), ...]."""
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        """Útil para UI com tooltips: [(value, label_pt, description), ...]."""
        return [(item.value, item.label_pt, item.description) for item in cls]
