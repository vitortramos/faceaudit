# faceaudit/enums/s0_resolution_method.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class S0ResolutionMethod(Enum):
    """
    Método específico de resolução de s₀ persistido no `LRArchitectureState`.

    Documenta auditavelmente como s₀ foi obtido. Distinto de
    `ArchitectureParameterMode` (operação atual do DTO) e de
    `ArchitectureFitScope` (escopo do ajuste — full vs per-fold).

    Valores:

    - `NOT_APPLICABLE`: arquitetura não usa s₀ (COSINE_ONLY).
    - `AUTO_LOGREG`: ajustado via LogReg univariado base ~ same_person,
      `s₀ = −intercept/coef`. Exige `same_person ∈ {0, 1}`, duas classes
      presentes, convergência, e `|coef| > tolerance`.
    - `MANUAL`: valor fornecido pelo usuário (`manual_s0`).
    """

    NOT_APPLICABLE = (
        "NOT_APPLICABLE",
        "Não aplicável",
        "Arquitetura não usa s₀.",
    )

    AUTO_LOGREG = (
        "AUTO_LOGREG",
        "LogReg univariado (auxiliar)",
        "Ajusta LogReg univariado base ~ same_person e calcula "
        "s₀ = −intercept/coef. Exige ambas as classes presentes, "
        "convergência e |coef| acima de tolerância numérica.",
    )

    MANUAL = (
        "MANUAL",
        "Valor manual",
        "Valor fornecido pelo usuário.",
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
    def from_str(cls, value: str) -> "S0ResolutionMethod":
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
            raise ValueError(f"S0ResolutionMethod desconhecido: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
