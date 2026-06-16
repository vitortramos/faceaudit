# faceaudit/enums/monotonicity_policy.py

"""Política de validação de monotonicidade da curva de calibração
piecewise-linear.

A curva final do calibrador deve manter `score_maior → log_LR_maior`
em todos os trechos. A expansão piecewise-linear permite mudanças de
inclinação, mas inclinações negativas (curvas decrescentes em algum
trecho) tornam o calibrador inválido forensicamente.

Esta enum define o que fazer quando a curva treinada não é monotônica
na escala bruta.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class MonotonicityPolicy(Enum):
    """Política a ser aplicada quando a curva piecewise-linear treinada
    apresenta algum trecho com inclinação ≤ `min_allowed_slope` na
    escala bruta.

    Modos:

    - `ERROR` (default): rejeita o calibrador. A instrução
      `CALIBRATION_MODEL_TRAINING` falha com erro explícito listando
      as inclinações por trecho. Defensável forensicamente — um
      calibrador não monotônico produz LRs inconsistentes (score maior
      pode dar evidência menor).

    - `WARNING`: emite WARNING forte mas permite o calibrador. Útil
      em modo experimental para investigar por que a curva ficou
      não monotônica (multicolinearidade, regularização insuficiente,
      knots mal posicionados). NÃO deve ser usado em pipeline
      operacional pericial.
    """

    ERROR = (
        "ERROR",
        "Erro (rejeita calibrador não monotônico)",
        "Default operacional. Calibrador com trecho não monotônico é "
        "rejeitado com erro listando inclinações por trecho.",
    )

    WARNING = (
        "WARNING",
        "Warning (permite, mas alerta)",
        "Modo experimental. Permite calibrador não monotônico com "
        "warning forte. NÃO usar em pipeline pericial — apenas para "
        "diagnóstico.",
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
    def from_str(cls, value: str) -> "MonotonicityPolicy":
        if not value:
            return cls.ERROR
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.ERROR

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
