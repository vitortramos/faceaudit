# faceaudit/enums/unit_norm_policy.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class UnitNormPolicy(Enum):
    """
    Política do modelo de tipicidade ao receber embeddings que NÃO estão
    em norma L2 unitária.

    Modelos paramétricos esféricos (MoVMF) assumem `‖x‖₂ = 1` por
    definição. Modelos não-paramétricos esféricos (k-NN com kernel
    cosseno) também. Embeddings com norma fora de tolerância levantam a
    seguinte questão pericial: silenciar normalizando, ou rejeitar?

    - **ERROR** (default forense): valida `|‖x‖ − 1| < tolerance` e
      levanta `UnitNormViolationError` em caso de violação. Rastreabilidade
      máxima — o usuário documenta o pré-processamento que produz
      embeddings unitários e qualquer divergência é detectada.

    - **NORMALIZE_WITH_WARNING**: normaliza para norma 1 e emite warning
      visível com a magnitude máxima da divergência observada. Útil
      durante exploração científica.

    - **NORMALIZE_SILENTLY**: normaliza sem aviso. Comportamento histórico
      do k-NN (compat). Usado pela migração de estratégias V2.14.
    """

    ERROR = (
        "ERROR",
        "Erro (default forense)",
        "Levanta UnitNormViolationError se algum embedding violar a "
        "tolerância. Rastreabilidade máxima.",
    )
    NORMALIZE_WITH_WARNING = (
        "NORMALIZE_WITH_WARNING",
        "Normalizar com warning",
        "Normaliza e emite warning visível indicando a divergência "
        "máxima observada.",
    )
    NORMALIZE_SILENTLY = (
        "NORMALIZE_SILENTLY",
        "Normalizar silenciosamente (legado)",
        "Normaliza sem aviso. Comportamento histórico do k-NN; usado "
        "pela migração preservativa.",
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
    def from_str(cls, value: str) -> "UnitNormPolicy":
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
