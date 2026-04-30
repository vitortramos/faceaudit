# faceaudit/enums/strategy_resource_type.py

from __future__ import annotations
from enum import Enum
from typing import Optional


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
    MINING_CONFIG = "MINING_CONFIG"
    MERGE_CONFIG = "MERGE_CONFIG"
    SIMILARITY_MODEL_CONFIG = "SIMILARITY_MODEL_CONFIG"
    CALIBRATION_MODEL_CONFIG = "CALIBRATION_MODEL_CONFIG"

    # --- MODELOS TREINADOS ---
    TRAINED_PIPELINE = "TRAINED_PIPELINE"
    TRAINED_SIMILARITY_MODEL = "TRAINED_SIMILARITY_MODEL"
    TRAINED_CALIBRATION_MODEL = "TRAINED_CALIBRATION_MODEL"

    # --- SCORES DE SIMILARIDADE ---
    SIMILARITY_SCORES = "SIMILARITY_SCORES"
    CALIBRATED_SCORES = "CALIBRATED_SCORES"

    # --- VISUALIZAÇÃO ---
    TRAJECTORY_DATA = "TRAJECTORY_DATA"

    # --- VALIDAÇÃO / MÉTRICAS ---
    VALIDATION_RESULT = "VALIDATION_RESULT"

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

    def to_artifact_type(self) -> Optional["StrategyArtifactType"]:
        """Retorna o StrategyArtifactType correspondente para persistência."""
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType

        _map = {
            StrategyResourceType.TRAINED_PIPELINE:          StrategyArtifactType.TRAINED_PIPELINE,
            StrategyResourceType.TRAINED_SIMILARITY_MODEL:  StrategyArtifactType.TRAINED_SIMILARITY_MODEL,
            StrategyResourceType.TRAINED_CALIBRATION_MODEL: StrategyArtifactType.TRAINED_CALIBRATION_MODEL,
            StrategyResourceType.INTERMEDIATE_DATASET:      StrategyArtifactType.INTERMEDIATE_DATASET,
            StrategyResourceType.TRAJECTORY_DATA:           StrategyArtifactType.TRAJECTORY_DATA,
            StrategyResourceType.SIMILARITY_SCORES:         StrategyArtifactType.SIMILARITY_SCORES,
            StrategyResourceType.CALIBRATED_SCORES:         StrategyArtifactType.CALIBRATED_SCORES,
            StrategyResourceType.VALIDATION_RESULT:         StrategyArtifactType.VALIDATION_RESULT,
            StrategyResourceType.MINING_CONFIG:             StrategyArtifactType.MINING_CONFIG,
            StrategyResourceType.PIPELINE_CONFIG:           StrategyArtifactType.PIPELINE_CONFIG,
            StrategyResourceType.SPLIT_CONFIG:              StrategyArtifactType.SPLIT_CONFIG,
        }
        return _map.get(self)

    @classmethod
    def from_str(cls, value: str) -> StrategyResourceType:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        try:
            return cls(value.upper() if value else "")
        except ValueError:
            return cls.UNKNOWN