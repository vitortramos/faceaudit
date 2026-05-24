# faceaudit/enums/calibration_model_type.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class CalibrationModelType(Enum):
    """
    Define os tipos de modelo de calibração suportados pela nova arquitetura.

    Este enum identifica explicitamente qual família de calibrador a instrução
    pretende usar, evitando inferência implícita por tipo de DTO.

    Estrutura de cada membro:
    - `value`: chave canônica (persistida em JSON).
    - `label`: rótulo em inglês técnico (docs/logs).
    - `label_pt`: rótulo em português para apresentação na UI.
    - `description`: descrição longa em PT-BR (tooltip/help).
    """

    LOGISTIC_REGRESSION = (
        "LOGISTIC_REGRESSION",
        "Logistic Regression",
        "Regressão Logística",
        "Usa regressão logística para calibrar scores brutos em probabilidades/LLR calibrado.",
    )

    KDE = (
        "KDE",
        "Kernel Density Estimation",
        "Estimação por Densidade Kernel (KDE)",
        "Usa estimativa de densidade para modelar separadamente scores genuínos e impostores, "
        "convertendo score bruto em LR/log-LR calibrado.",
    )

    BI_GAUSSIANIZED = (
        "BI_GAUSSIANIZED",
        "Bi-Gaussianized Calibration",
        "Calibração Bi-Gaussianizada",
        "Mapeia scores para um sistema bi-gaussiano perfeitamente calibrado com σ² alvo "
        "calculado a partir do Cllr ou EER do treino. Robusto a violação da suposição de "
        "duas gaussianas equi-variantes (Morrison 2024).",
    )

    def __new__(cls, value: str, label: str, label_pt: str, description: str):
        # Definir `_value_` em `__new__` (não `__init__`) é o pattern
        # canônico do CPython pra enums com valor composto. Senão,
        # `_value2member_map_` fica com a tupla original como chave e
        # o roundtrip de pickle quebra (ValueError no pickle.loads).
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label_pt
        obj.description = description
        return obj

    def __str__(self) -> str:
        """Retorna o label PT-BR para a interface."""
        return self.label_pt

    def to_str(self) -> str:
        """Retorna a chave string persistível."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> CalibrationModelType:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        if not value:
            return cls.LOGISTIC_REGRESSION

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for item in cls:
                if item.value == val_norm:
                    return item
            return cls.LOGISTIC_REGRESSION

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Útil para combos simples: [(value, label_pt), ...]."""
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        """Útil para UI com tooltips: [(value, label_pt, description), ...]."""
        return [(item.value, item.label_pt, item.description) for item in cls]
