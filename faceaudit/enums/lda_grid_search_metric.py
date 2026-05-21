from __future__ import annotations
from enum import Enum


class LdaGridSearchMetric(Enum):
    """
    Métrica de seleção usada pelo grid search do LDA.

    Define qual estatística é otimizada em um holdout interno para escolher
    o melhor valor de regularização (ε) — ou o melhor projeto, dependendo
    da implementação.
    """

    FISHER_RATIO = (
        "fisher_ratio",
        "Fisher Ratio",
        "Razão de Fisher",
        "Maximiza trace(Sb_proj)/trace(Sw_proj) no espaço projetado pelo "
        "holdout (separabilidade direta após a projeção LDA). Default — "
        "barato e bem alinhado ao objetivo do LDA.",
    )

    COSINE_AUC = (
        "cosine_auc",
        "Cosine AUC",
        "AUC Coseno",
        "Maximiza AUC de cossenos em pares genuíno/impostor do holdout "
        "PROJETADO. Mais caro (limitado a 200 identidades para evitar "
        "O(N²)), porém alinhado ao critério forense final de comparação.",
    )

    def __new__(cls, value: str, label: str, label_pt: str, description: str) -> "LdaGridSearchMetric":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "LdaGridSearchMetric":
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.FISHER_RATIO
