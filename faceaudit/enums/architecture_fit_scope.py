# faceaudit/enums/architecture_fit_scope.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class ArchitectureFitScope(Enum):
    """
    Escopo do ajuste original de s₀ e r₀ persistido no `LRArchitectureState`.

    Distinto de `ArchitectureParameterMode`: este enum descreve **como o
    state foi originalmente construído** (e fica congelado lá); o
    `ArchitectureParameterMode` descreve a **operação atual** da instrução.

    Quando um state é herdado via INHERIT, seu `fit_scope` continua refletindo
    o ajuste original — INHERIT não é "um modo de ajuste", é uma forma de
    aplicação que reaplica o ajuste já feito.

    Valores:

    - `NOT_APPLICABLE`: arquitetura não tem parâmetros ajustáveis
      (COSINE_ONLY). State tem `s0=None`, `r0_*=None`.
    - `FULL_CALIBRATION_LEGACY`: ajustado sobre todo D_cal antes da CV
      interna do calibrador (modo legado da v1, com leakage residual).
    - `PER_FOLD_REFIT_FULL_CALIBRATION`: ajustado por fold via Pipeline
      sklearn (v2); após escolha de C, refit final em todo D_cal.
    - `MANUAL`: valores fornecidos pelo usuário, não derivados de D_cal.
    """

    NOT_APPLICABLE = (
        "NOT_APPLICABLE",
        "Não aplicável",
        "Arquitetura sem parâmetros ajustáveis (e.g., COSINE_ONLY).",
    )

    FULL_CALIBRATION_LEGACY = (
        "FULL_CALIBRATION_LEGACY",
        "Ajuste em todo D_cal (legado)",
        "s₀ e r₀ estimados sobre todo D_cal antes da CV interna. Sujeito "
        "a leakage residual na seleção interna de C. Modo padrão da v1.",
    )

    PER_FOLD_REFIT_FULL_CALIBRATION = (
        "PER_FOLD_REFIT_FULL_CALIBRATION",
        "Ajuste por fold + refit em D_cal (v2)",
        "s₀ e r₀ estimados independentemente em cada train fold da CV "
        "interna. Após escolha de C, refit final em todo D_cal. Elimina "
        "leakage residual.",
    )

    MANUAL = (
        "MANUAL",
        "Valores manuais",
        "Parâmetros fornecidos pelo usuário, sem derivação de D_cal.",
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
    def from_str(cls, value: str) -> "ArchitectureFitScope":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.NOT_APPLICABLE
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"ArchitectureFitScope desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @property
    def has_leakage_risk(self) -> bool:
        """True para o modo legado que ajusta antes da CV interna."""
        return self == ArchitectureFitScope.FULL_CALIBRATION_LEGACY
