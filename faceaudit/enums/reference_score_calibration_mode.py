# faceaudit/enums/reference_score_calibration_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class ReferenceScoreCalibrationMode(Enum):
    """
    Como o modelo de tipicidade constrói a CDF empírica e o threshold OOD
    para suportar `TYPICALITY_PERCENTILE` e `is_in_distribution`.

    A distinção é metodologicamente importante: percentis in-sample são
    otimistas (uma face que ajudou a definir a CDF aparece com rank
    artificialmente "central"), enquanto cross-fit por identidade
    constrói a CDF apenas com scores out-of-fold, oferecendo medida
    defensável de tipicidade populacional.

    - **CROSS_FIT_IDENTITY** (default forense, PR 4):
      1. Particiona identidades em K folds disjuntos.
      2. Para cada fold: ajusta o modelo nos folds train; pontua
         identidades do fold test (scores OOF).
      3. Concatena scores OOF → CDF empírica + threshold OOD.
      4. Refit final em toda a população: o modelo persistido é este.
      Modelo final = refit em D completo; CDF/threshold = OOF.

    - **IN_SAMPLE_LEGACY**: comportamento histórico do k-NN — LOO de
      imagem (exclui só a própria face, não a identidade inteira). CDF
      construída sobre o mesmo conjunto usado no fit. Otimista, preserva
      reprodutibilidade do V2.14.

    PR 2 introduz; implementação MoVMF do `CROSS_FIT_IDENTITY` entra no
    PR 4 via `ReferenceScoreCalibrationService`. Em PR 2 o k-NN nativo
    usa `IN_SAMPLE_LEGACY` como default migratório; novas estratégias
    devem escolher `CROSS_FIT_IDENTITY`.
    """

    CROSS_FIT_IDENTITY = (
        "CROSS_FIT_IDENTITY",
        "Cross-fit por identidade (recomendado)",
        "K-fold por identidade; CDF construída com scores OOF; modelo "
        "persistido é refit em toda a população. Defensável forense.",
    )
    IN_SAMPLE_LEGACY = (
        "IN_SAMPLE_LEGACY",
        "In-sample (legado)",
        "CDF construída sobre o mesmo conjunto do fit. Comportamento "
        "histórico do k-NN; usado pela migração preservativa.",
    )

    def __new__(cls, value: str, label_pt: str, description: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "ReferenceScoreCalibrationMode":
        if not value:
            return cls.CROSS_FIT_IDENTITY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.CROSS_FIT_IDENTITY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
