# faceaudit/enums/covariance_shrinkage_mode.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class CovarianceShrinkageMode(Enum):
    """
    Define como o coeficiente de shrinkage α é determinado em variantes de
    PLDA com shrinkage de covariância (caso canônico: `PLDAVariant.SHRINKAGE_TWO_COVARIANCE`).

    O shrinkage substitui a estimativa empírica de uma covariância Σ por
    uma mistura convexa com um alvo bem-condicionado F:

        Σ_shrunk = (1 − α) · Σ_emp + α · F

    No FaceAudit, o alvo é sempre `F = (tr(Σ)/D) · I` (scaled identity) e o
    shrinkage é aplicado apenas à matriz intra-classe W. B permanece empírica.

    Modos:
    - NONE: nenhum shrinkage; usa Σ_emp diretamente. Serve principalmente
      para ablação e testes (SHRINKAGE_TWO_COVARIANCE + NONE deve reproduzir
      TWO_COVARIANCE puro com `regularization=0`).
    - LEDOIT_WOLF (padrão): α ótimo computado em forma fechada via Ledoit &
      Wolf (2004). Sem hiperparâmetro — defensável em corte sem justificar
      valor. Recomendado como default.
    - MANUAL: α fixo fornecido pelo usuário em [0, 1]. Útil para experimentos
      controlados e comparativos.
    """

    NONE = (
        "NONE",
        "Nenhum",
        "Sem shrinkage — usa a covariância empírica diretamente. "
        "Útil para ablação e comparativo.",
    )

    LEDOIT_WOLF = (
        "LEDOIT_WOLF",
        "Ledoit-Wolf (analítico)",
        "α ótimo computado em forma fechada (Ledoit & Wolf 2004) contra o "
        "alvo tr(Σ)/D · I. Sem hiperparâmetro — defensável em corte.",
    )

    MANUAL = (
        "MANUAL",
        "Manual",
        "α fixo fornecido pelo usuário em [0, 1]. Use para experimentos "
        "controlados ou quando o valor de Ledoit-Wolf precisar ser ajustado.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "CovarianceShrinkageMode":
        # Definir _value_ em __new__ é crítico para que o enum seja serializável
        # via pickle e reconstruído por value=str (mesmo padrão de PLDAVariant).
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
    def from_str(cls, value: str) -> "CovarianceShrinkageMode":
        if not value:
            return cls.LEDOIT_WOLF

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for m in cls:
                if m.value == val_norm:
                    return m
            return cls.LEDOIT_WOLF

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(m.value, m.label) for m in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(m.value, m.label, m.description) for m in cls]
