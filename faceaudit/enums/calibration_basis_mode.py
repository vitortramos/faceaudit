# faceaudit/enums/calibration_basis_mode.py

"""Modo de expansão de basis functions internas do calibrador.

Encapsulado em enum (e não em `bool enabled`) pra permitir extensão
futura sem refatoração: SPLINE_MONOTONIC, GAUSSIAN_KERNELS, etc.
poderão ser adicionados como novos membros sem mudar a semântica do
pickle.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class CalibrationBasisMode(Enum):
    """Modo de expansão interna de basis functions aplicada ao score
    antes de alimentar o calibrador final.

    Aplicável apenas a calibradores que recebem score 1D na V1
    (cosseno, PLDA escalar). Calibradores multivariados (`SCORE_VECTORIZATION`
    de várias colunas) NÃO suportam expansão interna na V1 — usar
    `SCORE_INTERACTION_FEATURE` externa.

    Modos:

    - `NONE` (default): nenhuma expansão. O calibrador vê o score bruto
      diretamente. Backward-compat com calibradores treinados antes
      da introdução de `PiecewiseLinearCalibrationConfig`.

    - `PIECEWISE_LINEAR`: aplica expansão `[z, max(z − offset_j, 0)]`
      com `z = score − hinge_anchor_s0` e `offset_j ∈ knot_offsets`.
      Uma única hinge corresponde a `knot_offsets = [0.0]`; múltiplas
      dobras a `knot_offsets = [-δ, 0, +δ]` etc. `hinge_anchor_s0` é
      aprendido fold-aware no treino (sem leakage no GridSearchCV).
    """

    NONE = (
        "NONE",
        "Nenhuma expansão",
        "Calibrador opera diretamente sobre o score bruto. "
        "Backward-compat com calibradores treinados antes da V1 do "
        "piecewise-linear.",
    )

    PIECEWISE_LINEAR = (
        "PIECEWISE_LINEAR",
        "Piecewise-linear (hinge expansion)",
        "Aplica expansão piecewise-linear em torno de hinge_anchor_s0 "
        "aprendido. Habilita ao calibrador capturar mudança de "
        "inclinação no ponto neutro. Restrito a score 1D na V1.",
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
    def from_str(cls, value: str) -> "CalibrationBasisMode":
        if not value:
            return cls.NONE
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.NONE

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
