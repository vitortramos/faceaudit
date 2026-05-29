# faceaudit/enums/cross_condition_within_mode.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class CrossConditionWithinMode(Enum):
    """
    Define como os resíduos intra-classe são centrados no estimador PLDA
    cross-condition (`fit_cross_condition_two_covariance`).

    Para um par genuíno cross: `x_A − x_B = (μ_c^A − μ_c^B) + r_A − r_B`,
    onde `(μ_c^A − μ_c^B)` é o deslocamento condição×identidade — varia por
    pessoa. A escolha de centragem decide se esse termo entra no W.

    Modos:
    - PER_CONDITION: `r_A = x_A − μ_c^A`, `r_B = x_B − μ_c^B`. Remove o
      deslocamento per-identidade do W. Mais "limpo" se você confia que
      CORAL/Whitening eliminou o shift entre condições. Arrisca subestimar
      a variabilidade cross-condição se houver interação residual.
    - BALANCED_IDENTITY (padrão): `μ_c = (μ_c^A + μ_c^B)/2`; ambos resíduos
      centrados em μ_c balanceada. O W absorve o deslocamento per-identidade.
      Mais conservador forensicamente — evita LR superconfiante. Recomendado
      como default por defensibilidade pericial.

    Só tem efeito quando há `secondary_dataset_resource_id` definido no
    treino. Em modo single-dataset o campo é ignorado.
    """

    PER_CONDITION = (
        "PER_CONDITION",
        "Por condição",
        "Resíduos centrados na média da própria condição (r = x − μ_c^cond). "
        "Remove o deslocamento per-identidade do W. Mais limpo se CORAL já "
        "removeu o shift, mas arrisca subestimar variabilidade cross-condição.",
    )

    BALANCED_IDENTITY = (
        "BALANCED_IDENTITY",
        "Identidade balanceada (recomendado)",
        "Resíduos centrados na média balanceada da identidade "
        "(μ_c = (μ_c^A + μ_c^B)/2). O W absorve o deslocamento per-identidade — "
        "mais conservador forensicamente, evita LR superconfiante.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "CrossConditionWithinMode":
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "CrossConditionWithinMode":
        if not value:
            return cls.BALANCED_IDENTITY

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for m in cls:
                if m.value == val_norm:
                    return m
            return cls.BALANCED_IDENTITY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(m.value, m.label) for m in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(m.value, m.label, m.description) for m in cls]
