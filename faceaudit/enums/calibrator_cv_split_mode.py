# faceaudit/enums/calibrator_cv_split_mode.py

"""Modo de partição dos folds em CV diagnóstico do calibrador.

Separado em módulo `enums` (não em `services`) pra evitar import circular:
configs de calibração precisam dessa enum e são importadas durante
`faceaudit.models.__init__`, antes do `faceaudit.services.__init__` que
carrega serviços pesados (OFIQ, ForensicFace, etc.).
"""

from __future__ import annotations

from enum import Enum


class CalibratorCVSplitMode(Enum):
    """Estratégia de partição de folds em CV diagnóstico do calibrador."""

    STRATIFIED_PAIR = (
        "STRATIFIED_PAIR",
        "Estratificado por par (otimista)",
        "sklearn.StratifiedKFold — preserva proporção gen/imp. Pares "
        "da mesma identidade podem aparecer em ambos os folds, gerando "
        "estimativa otimista do Cllr CV.",
    )
    STRATIFIED_GROUP = (
        "STRATIFIED_GROUP",
        "Estratificado por grupo (identidade do probe)",
        "sklearn.StratifiedGroupKFold por probe_identity — nenhuma "
        "identidade de probe usada no treino do fold aparece no held-out. "
        "Forense-defensável. Mais pessimista, mais honesto.",
    )

    def __new__(cls, value: str, label: str, description: str) -> "CalibratorCVSplitMode":
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
    def from_str(cls, value) -> "CalibratorCVSplitMode":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.STRATIFIED_PAIR
        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[val_norm]
        except KeyError:
            for m in cls:
                if m.value == val_norm:
                    return m
            return cls.STRATIFIED_PAIR
