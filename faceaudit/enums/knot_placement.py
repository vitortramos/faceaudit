# faceaudit/enums/knot_placement.py

"""Modo de posicionamento dos knots na expansão piecewise-linear.

`MANUAL`: usuário escreve `knot_offsets` no DTO (em RAW ou AUX_LOG_LR).
`AUTO_FROM_MIRRORED_MEDIANS`: o transformer popula `knot_offsets` no
fit usando a heurística experimental "espelho em zero das medianas
log-LR de cada classe" (detalhada em `docs/piecewise_linear_calibration.md`,
§6.2).

Em `AUTO_FROM_MIRRORED_MEDIANS`:

    1. LogReg auxiliar univariado é treinado sobre o subset de treino
       do fold (já feito hoje para `hinge_anchor_s0`).
    2. log-LR predito pelo auxiliar é `aux_coef · (score − s0)`.
    3. Medianas log-LR por classe:
       - `log_lr_Hp_P50 = aux_coef · (median(score|Hp_train) − s0)`
       - `log_lr_Hd_P50 = aux_coef · (median(score|Hd_train) − s0)`
    4. Knots fixos em `sorted([-log_lr_Hp_P50, -log_lr_Hd_P50])` —
       2 knots **refletidos** em torno de log-LR=0. Heurística
       experimental. Em distribuições aproximadamente simétricas
       (i.e. `log_lr_Hd_P50 ≈ -log_lr_Hp_P50`), o espelhamento é
       cosmético; em distribuições assimétricas é uma decisão de
       regularização. Validar empiricamente em cada domínio.
    5. Knots são sempre em escala AUX_LOG_LR (consequência natural da
       fórmula).

Fold-aware sem leakage: cada fold do GridSearchCV recomputa medianas só
sobre seus dados de treino — `s0`, `aux_coef`, knots e LogReg final são
todos estimados no mesmo subset.

Restrição: requer `HingeAnchorMode.AUTO_LOGREG` (precisa do `aux_coef`
do LogReg auxiliar pra mapear log-LR ↔ raw).

# Nomenclatura

Versões anteriores expunham este modo como `AUTO_FROM_MEDIANS`. O
nome `AUTO_FROM_MIRRORED_MEDIANS` é mais preciso porque a heurística
usa as medianas **refletidas** em torno de zero, não as medianas
diretas. O nome antigo é aceito como alias retroativo pela
`from_str()` (JSONs persistidos com `"AUTO_FROM_MEDIANS"` continuam
funcionando) e o valor serializado pelo `to_str()` continua sendo
`"AUTO_FROM_MEDIANS"` para preservar compatibilidade dos pickles
existentes. Apenas a constante Python e o `label_pt` foram
atualizados para refletir o nome correto.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class KnotPlacement(Enum):
    """Modo de posicionamento dos knots."""

    MANUAL = (
        "MANUAL",
        "Manual",
        "Usuário escreve knot_offsets no DTO (em RAW ou AUX_LOG_LR). "
        "Default operacional recomendado.",
    )

    AUTO_FROM_MIRRORED_MEDIANS = (
        "AUTO_FROM_MEDIANS",  # valor serializado preservado para compat
        "Auto (medianas log-LR refletidas) — experimental",
        "Transformer popula knot_offsets no fit com 2 knots em "
        "sorted([-log_lr_Hp_P50, -log_lr_Hd_P50]) — espelho das medianas "
        "log-LR auxiliares de cada classe em torno de zero. "
        "Heurística experimental, não estabelecida na literatura. "
        "Fold-aware. Exige hinge_anchor_mode=AUTO_LOGREG. Validar em "
        "cada domínio antes de uso operacional.",
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
        """Aceita o nome canônico e aliases retroativos.

        Aliases aceitos:
        - `AUTO_FROM_MEDIANS` → `AUTO_FROM_MIRRORED_MEDIANS` (nome antigo).
        """
        if not value:
            return cls.MANUAL
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        # Aliases retroativos antes do lookup canônico
        _ALIASES = {
            "AUTO_FROM_MEDIANS": cls.AUTO_FROM_MIRRORED_MEDIANS,
        }
        if norm in _ALIASES:
            return _ALIASES[norm]
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
