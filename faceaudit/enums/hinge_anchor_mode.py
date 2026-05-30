# faceaudit/enums/hinge_anchor_mode.py

"""Modo de resolução da âncora `s0` das basis functions piecewise-linear
do calibrador.

Separado em módulo `enums` (não em `services`) pra evitar import circular:
configs de calibração precisam dessa enum e são importadas durante
`faceaudit.models.__init__`, antes do `faceaudit.services.__init__`.

`hinge_anchor_s0` é o ponto-âncora em torno do qual a basis function
piecewise-linear é posicionada. NÃO é necessariamente o ponto onde o
log-LR final cruza zero — esse cruzamento é função do calibrador
treinado e é registrado separadamente como `effective_log_lr_zero_crossing`
em diagnóstico.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class HingeAnchorMode(Enum):
    """Como o `hinge_anchor_s0` é resolvido em `CALIBRATION_MODEL_TRAINING`
    quando `piecewise_linear_config.mode = PIECEWISE_LINEAR`.

    Pickle do calibrador treinado carrega o valor numérico de
    `hinge_anchor_s0` aprendido ou informado, independente do modo.

    Modos:

    - `AUTO_LOGREG` (default): estima `s0` via LogReg auxiliar
      univariado `score → label` no domínio de calibração e calcula
      `s0 = −intercept/coef`. Fold-aware via `PiecewiseLinearScoreTransformer`
      — em cada fold do GridSearchCV, `s0` é recomputado APENAS sobre
      os índices de treino do fold. Sem leakage. Exige `same_person ∈
      {0, 1}` em todos os samples e duas classes.

    - `MANUAL`: usa `hinge_anchor_manual_value` diretamente. Útil pra
      reusar `s0` de literatura, de calibrador externo já auditado, ou
      de regra de negócio explícita. Não envolve fit auxiliar — o
      transformer apenas usa o valor passado.
    """

    AUTO_LOGREG = (
        "AUTO_LOGREG",
        "Auto (LogReg auxiliar)",
        "Estima s0 via LogReg auxiliar univariado fold a fold "
        "(sklearn-compatible com GridSearchCV — sem leakage). "
        "Default recomendado para a maioria dos casos.",
    )

    MANUAL = (
        "MANUAL",
        "Manual (valor fixo)",
        "Usa o valor de `hinge_anchor_manual_value` informado pelo "
        "usuário. Útil pra reusar s0 de literatura ou de calibrador "
        "externo já analisado.",
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
    def from_str(cls, value: str) -> "HingeAnchorMode":
        if not value:
            return cls.AUTO_LOGREG
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.AUTO_LOGREG

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
