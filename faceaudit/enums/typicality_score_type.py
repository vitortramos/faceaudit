# faceaudit/enums/typicality_score_type.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class TypicalityScoreType(Enum):
    """
    Qual das métricas de tipicidade é usada como score individual por
    face antes de combinar via TypicalityCombiner.

    Compartilhada entre k-NN e MoVMF (PR 3 unificou nomes):

    - **MEAN_KNN_SIMILARITY**: T_k bruto, em [-1, 1]. Específico ao k-NN
      (k vizinhos mais próximos esférico). Sem análogo paramétrico
      direto no MoVMF.

    - **TYPICALITY_PERCENTILE**: rank empírico, escalado 0..100 pelo
      modelo (pair scoring divide por 100 para [0,1]). Universal entre
      backends. A base do percentil é declarada pelo modelo via
      `TypicalityPercentileBasis` (k-NN default: T_k; MoVMF default:
      log-densidade) e propagada no `TypicalityScoreBatch.percentile_basis_score_type`.

    - **LOG_DENSITY_ESTIMATE** (renomeado de `LOG_DENSITY_KNN` no PR 3):
      estimativa de log-densidade. No k-NN é Loftsgaarden-Quesenberry
      esférico (proporcional a -(d-1)·log θ_k); no MoVMF é
      log-likelihood da mistura. **As escalas absolutas diferem entre
      backends** — o nome `ESTIMATE` reconhece isso.

    Migração de JSONs antigos (V2.14) com `score_type=LOG_DENSITY_KNN`
    via `LegacyStrategyMigrationService` (não pelo `from_str` direto).
    """

    MEAN_KNN_SIMILARITY = ("MEAN_KNN_SIMILARITY", "Similaridade média kNN")
    TYPICALITY_PERCENTILE = ("TYPICALITY_PERCENTILE", "Percentil de tipicidade")
    LOG_DENSITY_ESTIMATE = (
        "LOG_DENSITY_ESTIMATE", "Log-densidade estimada (k-NN ou MoVMF)"
    )

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
        # None/empty: default por back-compat de JSONs antigos.
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
            # P1-10 (revisão externa 2026-06-09): strings desconhecidas
            # levantam em vez de cair em default silencioso.
            raise ValueError(
                f"TypicalityScoreType.from_str: valor desconhecido "
                f"{value!r}. Aceitos: {[m.value for m in cls]}."
            )

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
