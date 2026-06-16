# faceaudit/enums/typicality_model_type.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class TypicalityModelType(Enum):
    """
    Discriminador da família de modelos de tipicidade do FaceAudit.

    Análogo a `SimilarityModelType` (COSINE/PLDA): nomeia o backend
    concreto persistido em `TRAINED_TYPICALITY_MODEL`.

    - `KNN`: `KNNTypicalityModel` — não-paramétrico local (k-NN
      esférico, Loftsgaarden-Quesenberry).
    - `MOVMF`: `MoVMFTypicalityModel` — mistura paramétrica de
      von Mises-Fisher na hiperesfera. Status **EXPERIMENTAL** até
      validação científica concluída (ver `docs/typicality_models.md`).
    """

    KNN = ("KNN", "k-NN não-paramétrico",
           "Estimador local não-paramétrico baseado em vizinhos esféricos. "
           "Hubness vira sinal (densidade local).")
    MOVMF = ("MOVMF", "Mistura von Mises-Fisher (EXPERIMENTAL)",
             "Mistura paramétrica de vMF na hiperesfera. Fornece densidade "
             "probabilística explícita; status experimental até validação.")

    def __new__(cls, value: str, label_pt: str, description: str):
        # Pattern canônico — ver typicality_reference_mode.py para a
        # explicação completa sobre `_value_` em `__new__`.
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
    def from_str(cls, value: str) -> "TypicalityModelType":
        # None/empty: default por back-compat de JSONs antigos sem o campo.
        if not value:
            return cls.KNN
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
            # levantam em vez de cair em default silencioso. Antes,
            # "IDENTITY_CENTROD" virava "KNN" sem aviso — bug perigoso
            # em audit forense.
            raise ValueError(
                f"TypicalityModelType.from_str: valor desconhecido "
                f"{value!r}. Aceitos: {[m.value for m in cls]}."
            )

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label_pt, item.description) for item in cls]
