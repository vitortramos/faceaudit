# faceaudit/enums/knot_placement.py

"""Modo de posicionamento dos knots na expansão piecewise-linear.

`MANUAL`: usuário escreve `knot_offsets` no DTO (em RAW ou LOG_LR).
`AUTO_FROM_MEDIANS`: o transformer popula `knot_offsets` no fit usando
a heurística "espelho em zero das medianas log-LR de cada classe"
(detalhada em `project_piecewise_knot_heuristic`).

Em `AUTO_FROM_MEDIANS`:

    1. LogReg auxiliar univariado é treinado sobre o subset de treino
       do fold (já feito hoje para `hinge_anchor_s0`).
    2. log-LR predito pelo auxiliar é `aux_coef · (score − s0)`.
    3. Medianas log-LR por classe:
       - `log_lr_Hp_P50 = aux_coef · (median(score|Hp_train) − s0)`
       - `log_lr_Hd_P50 = aux_coef · (median(score|Hd_train) − s0)`
    4. Knots fixos em `sorted([-log_lr_Hp_P50, -log_lr_Hd_P50])` —
       2 knots espelhando as medianas. Heurística robusta nos casos
       PLDA-like; em cosseno-like com forte assimetria entre |Hp_P50|
       e |Hd_P50|, 1 knot em `[0.0]` manual pode ser marginalmente
       melhor — esses casos ficam pro modo MANUAL.
    5. Knots são sempre em escala LOG_LR (consequência natural da
       fórmula).

Fold-aware sem leakage: cada fold do GridSearchCV recomputa medianas só
sobre seus dados de treino — `s0`, `aux_coef`, knots e LogReg final são
todos estimados no mesmo subset.

Restrição: requer `HingeAnchorMode.AUTO_LOGREG` (precisa do `aux_coef`
do LogReg auxiliar pra mapear log-LR ↔ raw).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class KnotPlacement(Enum):
    """Modo de posicionamento dos knots."""

    MANUAL = (
        "MANUAL",
        "Manual",
        "Usuário escreve knot_offsets no DTO (em RAW ou LOG_LR). "
        "Default para backward-compat com configs antigos.",
    )

    AUTO_FROM_MEDIANS = (
        "AUTO_FROM_MEDIANS",
        "Auto (medianas log-LR)",
        "Transformer popula knot_offsets no fit com 2 knots em "
        "[-log_lr_Hp_P50, -log_lr_Hd_P50] (espelho em zero das medianas "
        "log-LR de cada classe, via LogReg auxiliar). Fold-aware. "
        "Exige hinge_anchor_mode=AUTO_LOGREG.",
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
    def from_str(cls, value: str) -> "KnotPlacement":
        if not value:
            return cls.MANUAL
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.MANUAL

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
