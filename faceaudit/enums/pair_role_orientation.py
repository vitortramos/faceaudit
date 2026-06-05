# faceaudit/enums/pair_role_orientation.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class PairRoleOrientation(Enum):
    """
    Orientação semântica dos papéis probe/reference nos pares de um
    dataset de scores.

    Conceito diferente de `PairKeyMode`: aqui descrevemos QUAL face
    representa qual papel (probe vs reference) na semântica do experimento.
    Em comparações cross-pose forenses, probe é tipicamente a face questionada
    (e.g., frontal CFTV) e reference é o material de comparação curado
    (e.g., perfil mug shot). A semântica importa para a interpretação de
    coeficientes (β_F vs β_P) em arquiteturas que distinguem os dois lados.

    Valores:

    - `PROBE_REFERENCE`: orientação canônica — primeiro lado é probe,
      segundo é reference.
    - `REFERENCE_PROBE`: orientação invertida — primeiro lado é reference,
      segundo é probe. Útil para detectar e bloquear inversão silenciosa.
    - `NOT_APPLICABLE`: arquitetura simétrica (COSINE_ONLY, HINGE_ONLY,
      TWO_SIDED_HINGE_ONLY) não distingue lados — a orientação não muda
      a matriz de features.
    """

    PROBE_REFERENCE = (
        "PROBE_REFERENCE",
        "Probe → Reference",
        "Orientação canônica: primeira face é probe, segunda é reference.",
    )

    REFERENCE_PROBE = (
        "REFERENCE_PROBE",
        "Reference → Probe",
        "Orientação invertida: primeira face é reference, segunda é probe. "
        "Útil para detectar inversão silenciosa entre datasets de origem.",
    )

    NOT_APPLICABLE = (
        "NOT_APPLICABLE",
        "Não aplicável",
        "Arquitetura simétrica — orientação não muda a matriz produzida.",
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
    def from_str(cls, value: str) -> "PairRoleOrientation":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.NOT_APPLICABLE
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            raise ValueError(f"PairRoleOrientation desconhecida: {value!r}")

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
