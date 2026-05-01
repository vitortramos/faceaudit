# faceaudit/enums/strategy_artifact_type.py

from __future__ import annotations
from enum import Enum


class StrategyArtifactType(Enum):
    """
    Define os tipos de artefatos produzidos ou consumidos pela nova arquitetura
    de estratégias de cálculo de LR.

    Este enum representa a semântica do artefato no domínio da estratégia,
    independentemente do formato físico final em disco.
    """

    # --- DEFINIÇÕES / MODELOS ---
    STRATEGY_DEFINITION = "strategy_definition"
    STRATEGY_MODEL = "strategy_model"
    TRAINED_PIPELINE = "trained_pipeline"
    TRAINED_SIMILARITY_MODEL = "trained_similarity_model"
    TRAINED_CALIBRATION_MODEL = "trained_calibration_model"

    # --- DADOS DERIVADOS ---
    TRANSFORMED_DATASET = "transformed_dataset"
    INTERMEDIATE_DATASET = "intermediate_dataset"
    SPLIT_RESULT = "split_result"
    MERGE_RESULT = "merge_result"

    # --- METADADOS / EXECUÇÃO ---
    EXECUTION_LOG = "execution_log"
    CHECKPOINT = "checkpoint"
    METRICS = "metrics"
    TRAJECTORY_DATA = "trajectory_data"
    SIMILARITY_SCORES = "similarity_scores"
    CALIBRATED_SCORES = "calibrated_scores"
    VALIDATION_RESULT = "validation_result"
    CASEWORK_VISUALIZATION = "casework_visualization"

    # --- VISUALIZAÇÕES ---
    GRAPHIC = "graphic"

    # --- BIBLIOTECA GLOBAL (recursos reutilizáveis entre projetos) ---
    GLOBAL_DATASET = "global_dataset"        # Dataset parquet na biblioteca global
    SPLIT_CONFIG = "split_config"            # Configuração de split (SplitConfig JSON)
    MINING_CONFIG = "mining_config"          # Configuração de mineração (ComparisonMiningConfig JSON)
    PIPELINE_CONFIG = "pipeline_config"      # Configuração de pipeline de transformação (JSON)
    STRATEGY_TEMPLATE = "strategy_template"  # Template de estratégia (LRStrategyDefinition JSON sem recursos vinculados)

    # --- FALLBACK ---
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Retorna o identificador textual do tipo de artefato."""
        return self.value

    def to_str(self) -> str:
        """Retorna o valor persistível para serialização."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> StrategyArtifactType:
        """
        Reconstrói o enum a partir de string com fallback seguro.
        """
        try:
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.UNKNOWN