# faceaudit/enums/population_weighting_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class PopulationWeightingMode(Enum):
    """
    Como o modelo de tipicidade trata a relação entre faces e identidades
    ao construir a referência populacional.

    Substitui o `TypicalityReferenceMode` (k-NN) legado, harmonizado para
    ambos os backends (k-NN + MoVMF). PR 2 introduz; PR 4 endurece a
    implementação no MoVMF e adiciona a semântica matemática do
    `INVERSE_IDENTITY_FREQUENCY` no k-NN documentada em
    `docs/typicality_models.md`.

    Mapeamento legado k-NN (migração preservativa):
        TypicalityReferenceMode.PER_FACE      → FACE_LEVEL_RAW
        TypicalityReferenceMode.PER_IDENTITY  → IDENTITY_CENTROIDS

    - **FACE_LEVEL_RAW**: cada face é uma observação independente. Pesa
      identidades proporcionalmente à frequência de coleta (uma pessoa
      com 30 fotos pesa 30× mais que uma pessoa com 1 foto). Útil para
      análise pura de fotos atípicas, vulnerável a desbalanceamento.

    - **INVERSE_IDENTITY_FREQUENCY**: cada face entra com peso
      `1/count(identity)`. Iguala o peso das identidades sem perder
      diversidade intra-identidade. Implementação no k-NN documentada em
      `docs/typicality_models.md` (afeta CDF de referência + peso de
      vizinhos na agregação T_k). PR 4 implementa.

    - **IDENTITY_CENTROIDS** (default forense): agrega por identidade,
      cada centroide L2-normalizado entra como ponto único. A referência
      tem uma observação por pessoa, defensável em corte ("esta pessoa
      está no 73º percentil da população de identidades"). Robusto a
      desbalanceamento.
    """

    FACE_LEVEL_RAW = (
        "FACE_LEVEL_RAW",
        "Por face (sem ponderar)",
        "Cada foto é uma observação. Identidades com mais fotos pesam "
        "mais. Vulnerável a desbalanceamento de coleta.",
    )
    INVERSE_IDENTITY_FREQUENCY = (
        "INVERSE_IDENTITY_FREQUENCY",
        "Por face com peso 1/n_faces(identidade)",
        "Cada foto entra com peso inversamente proporcional ao número de "
        "fotos da identidade. Iguala identidades sem perder diversidade "
        "intra-identidade.",
    )
    IDENTITY_CENTROIDS = (
        "IDENTITY_CENTROIDS",
        "Por identidade (centroide)",
        "Agrega faces de cada identidade em centroide L2-normalizado. "
        "Uma observação por pessoa. Defensável em corte forense; robusto "
        "a desbalanceamento de coleta.",
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
    def from_str(cls, value: str) -> "PopulationWeightingMode":
        # None/empty: default por back-compat de JSONs antigos.
        if not value:
            return cls.IDENTITY_CENTROIDS
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
            # levantam — "IDENTITY_CENTROD" não pode virar
            # "IDENTITY_CENTROIDS" silenciosamente.
            raise ValueError(
                f"PopulationWeightingMode.from_str: valor desconhecido "
                f"{value!r}. Aceitos: {[m.value for m in cls]}."
            )

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
