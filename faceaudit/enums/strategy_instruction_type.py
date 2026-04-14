# faceaudit/enums/strategy_instruction_type.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class StrategyInstructionType(Enum):
    """
    Define os tipos de instrução suportados pela strategy engine.
    """

    PIPELINE_TRAINING = (
        "PIPELINE_TRAINING",
        "Pipeline Training",
        "Treina um pipeline de transformação de embeddings.",
    )

    PIPELINE_APPLICATION = (
        "PIPELINE_APPLICATION",
        "Pipeline Application",
        "Aplica um pipeline treinado a um ou mais datasets.",
    )

    DATASET_SPLIT = (
        "DATASET_SPLIT",
        "Dataset Split",
        "Divide um dataset em dois subconjuntos segundo uma regra declarada.",
    )

    DATASET_MERGE = (
        "DATASET_MERGE",
        "Dataset Merge",
        "Combina ou atualiza datasets segundo uma regra declarada.",
    )

    SIMILARITY_MODEL_TRAINING = (
        "SIMILARITY_MODEL_TRAINING",
        "Similarity Model Training",
        "Treina ou prepara um modelo de similaridade.",
    )

    SIMILARITY_SCORING = (
        "SIMILARITY_SCORING",
        "Similarity Scoring",
        "Gera scores de similaridade a partir de um modelo treinado e um ou mais datasets.",
    )

    PAIR_MINING = (
        "PAIR_MINING",
        "Pair Mining",
        "Curadoria estatística de pares para treino de calibração.",
    )

    CALIBRATION_MODEL_TRAINING = (
        "CALIBRATION_MODEL_TRAINING",
        "Calibration Model Training",
        "Treina ou prepara um modelo de calibração a partir de scores e labels.",
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
    def from_str(cls, value: str) -> "StrategyInstructionType":
        if not value:
            return cls.PIPELINE_TRAINING

        if isinstance(value, cls):
            return value

        val_norm = str(value).strip().upper().replace("-", "_").replace(" ", "_")

        try:
            return cls[val_norm]
        except KeyError:
            for item in cls:
                if item.value == val_norm:
                    return item
            return cls.PIPELINE_TRAINING

    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        return [(item.value, item.label) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        return [(item.value, item.label, item.description) for item in cls]