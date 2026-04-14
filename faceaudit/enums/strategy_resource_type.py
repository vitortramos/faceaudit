# faceaudit/enums/strategy_resource_type.py

from __future__ import annotations
from enum import Enum


class StrategyResourceType(Enum):
    """
    Define os tipos lógicos de recursos que podem existir dentro de uma estratégia.

    Esses recursos representam insumos, configurações, artefatos intermediários
    e modelos treinados utilizados ao longo da preparação e execução de uma
    estratégia de cálculo de LR.
    """

    # --- DADOS ---
    DATASET = "DATASET"
    INTERMEDIATE_DATASET = "INTERMEDIATE_DATASET"

    # --- CONFIGURAÇÕES ---
    PIPELINE_CONFIG = "PIPELINE_CONFIG"
    SPLIT_CONFIG = "SPLIT_CONFIG"
    MERGE_CONFIG = "MERGE_CONFIG"
    SIMILARITY_MODEL_CONFIG = "SIMILARITY_MODEL_CONFIG"
    CALIBRATION_MODEL_CONFIG = "CALIBRATION_MODEL_CONFIG"

    # --- MODELOS TREINADOS ---
    TRAINED_PIPELINE = "TRAINED_PIPELINE"
    TRAINED_SIMILARITY_MODEL = "TRAINED_SIMILARITY_MODEL"
    TRAINED_CALIBRATION_MODEL = "TRAINED_CALIBRATION_MODEL"

    # --- OUTROS / APOIO ---
    DATASET_STATS = "DATASET_STATS"
    ARTIFACT = "ARTIFACT"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        """Retorna o nome técnico do tipo de recurso."""
        return self.value

    def to_str(self) -> str:
        """Retorna o valor persistível para serialização."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> StrategyResourceType:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        try:
            return cls(value.upper() if value else "")
        except ValueError:
            return cls.UNKNOWN