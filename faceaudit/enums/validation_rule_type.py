# faceaudit/enums/validation_rule_type.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class ValidationRuleType(Enum):
    """
    Tipos de regra de validação plugáveis a uma instrução ou estratégia.

    Cada valor mapeia para uma classe concreta em
    `faceaudit/validation_rules/` via `ValidationRuleFactory`. Os parâmetros
    de construção da regra ficam no `ValidationRuleConfig.params` que
    acompanha o tipo.
    """

    FISHER = (
        "FISHER",
        "Fisher (viabilidade PLDA)",
        "Verifica n_identidades > embedding_dim no dataset de treino — "
        "condição geométrica necessária para o PLDA não singularizar.",
    )

    MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES = (
        "MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES",
        "Mínimo de identidades com 2+ amostras",
        "Verifica que o dataset tem pelo menos N identidades com 2 ou mais "
        "amostras — PLDA descarta singletons e exige ≥ 2 classes válidas.",
    )

    MIN_CALIBRATION_PAIRS = (
        "MIN_CALIBRATION_PAIRS",
        "Mínimo de pares de calibração",
        "Verifica que o score dataset de calibração tem pelo menos N pares "
        "genuínos e N pares impostores.",
    )

    DATASET_ISOLATION = (
        "DATASET_ISOLATION",
        "Isolamento entre datasets",
        "Verifica que dois datasets são completamente disjuntos em "
        "identidades (anti-leakage).",
    )

    DATASET_OVERLAP_REQUIRED = (
        "DATASET_OVERLAP_REQUIRED",
        "Overlap obrigatório entre datasets",
        "Verifica que dois datasets compartilham pelo menos N identidades "
        "(necessário em supervised adaptation, cross-pose).",
    )

    HOLDOUT_LEAKAGE = (
        "HOLDOUT_LEAKAGE",
        "Vazamento de holdout",
        "Verifica que o dataset de holdout não compartilha identidades com "
        "treino ou calibração.",
    )

    DUAL_PIPELINE_DIM_COMPAT = (
        "DUAL_PIPELINE_DIM_COMPAT",
        "Compatibilidade dimensional (DUAL)",
        "Em pipeline DUAL, verifica que source e target produzem embeddings "
        "com a mesma dimensão final.",
    )

    DUAL_PIPELINE_SUBSEQUENCE = (
        "DUAL_PIPELINE_SUBSEQUENCE",
        "Subsequência de steps (DUAL)",
        "Em pipeline DUAL, verifica que os steps do target são prefixo dos "
        "steps do source.",
    )

    def __init__(self, value: str, label: str, description: str):
        self._value_ = value
        self.label = label
        self.description = description

    def __str__(self) -> str:
        return self.label

    def to_str(self) -> str:
        return self.value

    @classmethod
    def from_str(cls, value: str) -> "ValidationRuleType":
        if not value:
            raise ValueError("ValidationRuleType: valor vazio não suportado.")

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for item in cls:
                if item.value == val_norm:
                    return item
            raise ValueError(
                f"ValidationRuleType desconhecido: '{value}'."
            )

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label, item.description) for item in cls]
