# faceaudit/enums/quality_gate_policy.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class QualityGatePolicy(Enum):
    """
    Política de aplicação de um gate de QUALIDADE/CONFIABILIDADE
    (det_score, fiqa_score) na extração de faces de vídeo.

    Estes gates são **separados** do gate de IDENTIDADE
    (`similarity_to_anchor`), que é sempre HARD_REJECT abaixo do limiar
    e é o árbitro de "é o alvo?". det_score e fiqa_score apenas
    *qualificam a confiabilidade da decisão* — não devem virar filtro
    opaco que apaga evidência potencialmente relevante (em vídeo, os
    quadros ruins podem ser justamente a condição da evidência).

    Modos:

    - `DISABLED`: o score não é nem usado nem gravado como gate.
    - `TAG_ONLY` (default da v1): grava o score em toda face aceita
      (sidecar + parquet), mas **não bloqueia** nada.
    - `ROUTE_TO_REVIEW`: a face passa no gate de identidade, mas é
      enviada **obrigatoriamente** à revisão humana (acrescenta
      `low_det`/`low_fiqa` em `review_reasons`).
    - `HARD_REJECT`: a face **não** entra como candidato final.
      Para `det_score` pode pré-filtrar ANTES do embedding (economia);
      `fiqa_score` só existe pós-embedding.

    Os limiares (`min_det_score`/`min_fiqa_score`) começam em `None` e
    são calibrados pela distribuição empírica observada na prévia — o
    `fiqa_score` do CR-FIQA não deve ser presumido normalizado.
    """

    DISABLED = (
        "DISABLED",
        "Desabilitado",
        "O score não é usado nem gravado como gate.",
    )
    TAG_ONLY = (
        "TAG_ONLY",
        "Só registrar (tag)",
        "Grava o score na face aceita (sidecar + parquet) sem bloquear. "
        "Default da v1.",
    )
    ROUTE_TO_REVIEW = (
        "ROUTE_TO_REVIEW",
        "Rotear para revisão",
        "A face passa no gate de identidade mas vai obrigatoriamente "
        "para a revisão humana (review_reasons += low_*).",
    )
    HARD_REJECT = (
        "HARD_REJECT",
        "Rejeitar (descarte duro)",
        "A face não entra como candidato final. Para det_score pode "
        "pré-filtrar antes do embedding.",
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

    @property
    def blocks_or_routes(self) -> bool:
        """True quando a política pode alterar o destino da face
        (ROUTE_TO_REVIEW ou HARD_REJECT)."""
        return self in (QualityGatePolicy.ROUTE_TO_REVIEW, QualityGatePolicy.HARD_REJECT)

    @classmethod
    def from_str(cls, value) -> "QualityGatePolicy":
        if not value:
            return cls.TAG_ONLY
        if isinstance(value, cls):
            return value
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.TAG_ONLY

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
