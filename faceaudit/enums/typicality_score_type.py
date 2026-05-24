# faceaudit/enums/typicality_score_type.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class TypicalityScoreType(Enum):
    """
    Qual das três métricas do KNNTypicalityModel é usada como score
    individual por face antes de combinar via TypicalityCombiner.

    - MEAN_KNN_SIMILARITY: T_k bruto, em [-1, 1]. Intuitivo, mas a escala
      depende da concentração da população — não é diretamente comparável
      entre populações de referência distintas.

    - TYPICALITY_PERCENTILE: rank empírico, normalizado em [0, 1].
      Comparável entre populações (sempre 0..1). Recomendado para
      vetorização com cosseno, porque tem escala consistente e é
      defensável em corte como estatística não-paramétrica.

    - LOG_DENSITY_KNN: estimador de Loftsgaarden-Quesenberry esférico.
      Escala real, pode ser muito negativo em D=512 (proporcional a
      -(d-1)·log θ_k). Indicado para LR generativo direto. Para
      vetorização junto com cosseno, considere padronizar antes do
      LogReg via `StandardScaler` (não implementado aqui).
    """

    MEAN_KNN_SIMILARITY = ("MEAN_KNN_SIMILARITY", "Similaridade média kNN")
    TYPICALITY_PERCENTILE = ("TYPICALITY_PERCENTILE", "Percentil de tipicidade")
    LOG_DENSITY_KNN = ("LOG_DENSITY_KNN", "Log-densidade kNN")

    def __new__(cls, value: str, label_pt: str):
        # Pattern canônico de Enum com valor composto — ver
        # typicality_reference_mode.py pra explicação completa.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label_pt = label_pt
        return obj

    def __str__(self) -> str:
        return self.label_pt

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "TypicalityScoreType":
        if not value:
            return cls.MEAN_KNN_SIMILARITY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.MEAN_KNN_SIMILARITY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
