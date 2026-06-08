# faceaudit/enums/typicality_percentile_basis.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class TypicalityPercentileBasis(Enum):
    """
    Base subjacente do score `TYPICALITY_PERCENTILE`.

    Percentil sem base persistida é ambíguo: a mesma rotulação
    `TYPICALITY_PERCENTILE` pode ser calculada sobre `MEAN_KNN_SIMILARITY`
    (T_k bruto, comportamento histórico do k-NN) ou sobre
    `LOG_DENSITY_ESTIMATE` (log-likelihood/log-densidade L-Q). Duas
    estratégias com mesma rotulação podem aparentar usar o mesmo score
    type consumindo features semanticamente diferentes.

    Esta enum persistida no envelope do modelo e no `TypicalityScoreBatch`
    elimina a ambiguidade. Defaults:

    - **k-NN**: `MEAN_KNN_SIMILARITY` (comportamento histórico — o k-NN
      atual ranqueia T_k da referência, não a log-densidade).
    - **MoVMF**: `LOG_DENSITY_ESTIMATE` (não tem T_k análogo paramétrico).

    Os defaults são selecionados pelos configs de cada backend
    (`KNNTypicalityModelConfig.percentile_basis_score_type`,
    `MoVMFTypicalityModelConfig.percentile_basis_score_type`).
    """

    MEAN_KNN_SIMILARITY = (
        "MEAN_KNN_SIMILARITY",
        "Sobre T_k (similaridade média k-NN)",
        "Percentil empírico de T_k(x) na distribuição leave-one-out da "
        "referência. Comportamento histórico do k-NN.",
    )
    LOG_DENSITY_ESTIMATE = (
        "LOG_DENSITY_ESTIMATE",
        "Sobre log-densidade estimada",
        "Percentil empírico de log p̂(x) — Loftsgaarden-Quesenberry no "
        "k-NN, log-likelihood da mistura no MoVMF.",
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
    def from_str(cls, value: str) -> "TypicalityPercentileBasis":
        if not value:
            return cls.MEAN_KNN_SIMILARITY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.MEAN_KNN_SIMILARITY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
