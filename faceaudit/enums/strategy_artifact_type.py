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
    # Datasets produzidos por execução de instrução (PIPELINE_APPLICATION,
    # DATASET_SPLIT, BALANCING, MERGE, PAIRED_SPLIT). Path no disco:
    # `<project>/datasets/{strategy_id}_{resource_id}.parquet`.
    # Distinto do GLOBAL_DATASET (importado pelo usuário, path
    # `{dataset_id}.parquet`) — o tipo lógico é o mesmo
    # (`StrategyResourceType.DATASET`), mas a provenância (capturada na
    # `ResourceCategory`) e o path físico diferem.
    TRANSFORMED_DATASET = "transformed_dataset"
    DERIVED_DATASET = "derived_dataset"
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
    HUBNESS_REPORT = "hubness_report"
    HUBNESS_ABLATION_RESULT = "hubness_ablation_result"

    # --- VISUALIZAÇÕES ---
    GRAPHIC = "graphic"

    # --- BIBLIOTECA GLOBAL (recursos reutilizáveis entre projetos) ---
    GLOBAL_DATASET = "global_dataset"        # Dataset parquet na biblioteca global
    SPLIT_CONFIG = "split_config"            # Configuração de split (SplitConfig JSON)
    MINING_CONFIG = "mining_config"          # Configuração de mineração (ComparisonMiningConfig JSON)
    PIPELINE_CONFIG = "pipeline_config"      # Configuração de pipeline de transformação (JSON)
    IMAGE_PIPELINE_CONFIG = "image_pipeline_config"  # PR I4 — configuração de pipeline de processamento de imagens (JSON)
    FILTER_CONFIG = "filter_config"          # FilterConfig polimórfico (lista de ColumnFilter, JSON)
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

        Legacy shim: `"intermediate_dataset"` (valor extinto) mapeia pra
        `DERIVED_DATASET` — JSONs de `StrategyArtifactReference` antigos
        carregam sem erro.
        """
        if not value:
            return cls.UNKNOWN
        normalized = value.lower().strip()
        if normalized == "intermediate_dataset":
            return cls.DERIVED_DATASET
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN