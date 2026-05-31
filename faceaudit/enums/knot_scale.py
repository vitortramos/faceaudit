# faceaudit/enums/knot_scale.py

"""Escala em que o usuário expressa os `knot_offsets` da expansão
piecewise-linear do calibrador.

Separado em módulo `enums` pra evitar import circular com os configs
de calibração que dependem dessa escala.

Motivação: `knot_offsets` em escala bruta exige conhecer o domínio
numérico de cada modelo de similaridade (cosseno em [−1, 1], PLDA pode
variar em [−200, +200] dependendo da população). Expressar em escala
log-LR é universal: "dobra a 2 unidades log-LR acima da neutralidade"
significa a mesma coisa para qualquer calibrador.

A conversão `log_lr → raw` é feita no transformer no momento do fit:

    offset_raw_j = offset_log_lr_j / aux_coef

onde `aux_coef` é a inclinação do LogReg auxiliar univariado treinado
sobre o score (i.e., a escala em que `score` é mapeado para log-LR
linearmente em torno de `s0`). Cada fold do GridSearchCV recomputa
`aux_coef` sobre apenas seus dados de treino — mantém a propriedade
fold-aware sem leakage.

Restrição: `LOG_LR` requer `HingeAnchorMode.AUTO_LOGREG`. Em `MANUAL`,
não há LogReg auxiliar para fornecer a escala — use `RAW` com offsets
diretos no domínio do score.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class KnotScale(Enum):
    """Unidade dos `knot_offsets` informados pelo usuário.

    Modos:

    - `RAW`: offsets em unidades de **score bruto** (compatível com
      qualquer modo de âncora). Default em projetos legados sem o
      campo `knot_scale` no JSON. Exige conhecer o domínio numérico
      do score (-1..+1 para cosseno, etc.).

    - `LOG_LR`: offsets em unidades de **log-LR auxiliar** (i.e., a
      escala em que o score é projetado pelo LogReg auxiliar univariado).
      Universal entre modelos — "+2 log-LR acima da neutralidade"
      significa "10× mais provável Hp". Exige `HingeAnchorMode.AUTO_LOGREG`.
    """

    RAW = (
        "RAW",
        "Bruto (unidades do score)",
        "Offsets diretamente em unidades do score (cosseno em ±1, PLDA "
        "em ±100, etc.). Backward-compat com calibradores legados.",
    )

    LOG_LR = (
        "LOG_LR",
        "Log-LR (universal entre modelos)",
        "Offsets em unidades de log-LR auxiliar. '+2 log-LR' = '10× mais "
        "provável Hp'. Conversão raw_offset = log_lr_offset / aux_coef "
        "feita fold-aware no fit. Exige âncora AUTO_LOGREG.",
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
    def from_str(cls, value: str) -> "KnotScale":
        if not value:
            return cls.RAW
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.RAW

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]

    @property
    def unit_label(self) -> str:
        """Label curto pra logs/mensagens — ex: 'log-LR' ou 'score bruto'."""
        if self is KnotScale.LOG_LR:
            return "log-LR"
        return "score bruto"
