# faceaudit/enums/calibration_compatibility_mode.py

from __future__ import annotations

from enum import Enum
from typing import List, Tuple


class CalibrationCompatibilityMode(Enum):
    """
    Política de validação do vínculo dataset vetorizado ↔ calibrador
    aplicada por `CalibrationModelMetadataService.validate_compatibility`.

    Modos:

    - `STRICT` (default): aplica a tabela assimétrica completa. Mismatch
      de `vectorization_contract_fingerprint` entre dataset e calibrador
      sempre falha. Dataset sem fingerprint + calibrador com fingerprint
      também falha (impossível garantir que as features atuais
      correspondem ao calibrador). Apenas o caso "ambos legados" passa
      com warning.

    - `LEGACY_PERMISSIVE`: opt-in explícito do usuário para rodar
      híbridos durante migração — calibrador novo aceita dataset legado
      sem fingerprint COM warning forte. Não relaxa mismatch entre dois
      fingerprints presentes (esse continua erro).
    """

    STRICT = (
        "STRICT",
        "Estrito (default)",
        "Vínculo dataset↔calibrador validado integralmente. Dataset sem "
        "fingerprint + calibrador com fingerprint → erro.",
    )

    LEGACY_PERMISSIVE = (
        "LEGACY_PERMISSIVE",
        "Permissivo legado (opt-in)",
        "Calibrador novo aceita dataset legado sem fingerprint com "
        "warning forte. Não tolera mismatch entre fingerprints presentes.",
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
    def from_str(cls, value: str) -> "CalibrationCompatibilityMode":
        if isinstance(value, cls):
            return value
        if not value:
            return cls.STRICT
        norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        try:
            return cls[norm]
        except KeyError:
            for item in cls:
                if item.value == norm:
                    return item
            return cls.STRICT

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label_pt) for item in cls]
