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
    # DATASET cobre tanto datasets importados pelo usuário ("primários")
    # quanto datasets produzidos por execução de instrução ("derivados").
    # A distinção fica no campo `category` (ResourceCategory) do
    # StrategyResource. Antes existia INTERMEDIATE_DATASET separado pra
    # marcar a versão derivada — mas isso cindia artificialmente um único
    # conceito (FaceDataset/EmbeddingDataset) por provenância.
    DATASET = "DATASET"

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

    # --- VISUALIZAÇÃO FORENSE ---
    CASEWORK_VISUALIZATION = "CASEWORK_VISUALIZATION"

    # --- DIAGNÓSTICO DE HUBNESS ---
    HUBNESS_REPORT = "HUBNESS_REPORT"
    HUBNESS_ABLATION_RESULT = "HUBNESS_ABLATION_RESULT"

    # --- OUTROS / APOIO ---
    DATASET_STATS = "DATASET_STATS"
    ARTIFACT = "ARTIFACT"
    UNKNOWN = "UNKNOWN"

    @property
    def label_pt(self) -> str:
        return {
            "DATASET":                  "Dataset",
            "PIPELINE_CONFIG":          "Configuração de Pipeline",
            "SPLIT_CONFIG":             "Configuração de Divisão",
            "MINING_CONFIG":            "Configuração de Mineração",
            "MERGE_CONFIG":             "Configuração de Mesclagem",
            "SIMILARITY_MODEL_CONFIG":  "Configuração de Modelo de Similaridade",
            "CALIBRATION_MODEL_CONFIG": "Configuração de Calibrador",
            "TRAINED_PIPELINE":         "Pipeline Treinado",
            "TRAINED_SIMILARITY_MODEL": "Modelo de Similaridade Treinado",
            "TRAINED_CALIBRATION_MODEL":"Calibrador Treinado",
            "SIMILARITY_SCORES":        "Scores de Similaridade",
            "CALIBRATED_SCORES":        "Scores Calibrados (log-LR)",
            "TRAJECTORY_DATA":          "Trajetória de Pipeline",
            "VALIDATION_RESULT":        "Resultado de Validação",
            "CASEWORK_VISUALIZATION":   "Visualização Forense",
            "HUBNESS_REPORT":           "Relatório de Hubness",
            "HUBNESS_ABLATION_RESULT":  "Resultado de Ablação Hubness",
            "DATASET_STATS":            "Estatísticas do Dataset",
            "ARTIFACT":                 "Artefato",
            "UNKNOWN":                  "Desconhecido",
        }.get(self.value, self.value)

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
            # DATASET (derivado) usa o tipo físico DERIVED_DATASET — path
            # `{strategy_id}_{resource_id}.parquet` em `datasets/`.
            # Primários nunca chegam aqui: `StrategyResource.to_artifact_type`
            # (ciente da `category`) retorna None pra PRIMARY, fazendo
            # `persist_produced_resource` pular.
            StrategyResourceType.DATASET:                   StrategyArtifactType.DERIVED_DATASET,
            StrategyResourceType.TRAJECTORY_DATA:           StrategyArtifactType.TRAJECTORY_DATA,
            StrategyResourceType.SIMILARITY_SCORES:         StrategyArtifactType.SIMILARITY_SCORES,
            StrategyResourceType.CALIBRATED_SCORES:         StrategyArtifactType.CALIBRATED_SCORES,
            StrategyResourceType.VALIDATION_RESULT:         StrategyArtifactType.VALIDATION_RESULT,
            StrategyResourceType.CASEWORK_VISUALIZATION:    StrategyArtifactType.CASEWORK_VISUALIZATION,
            StrategyResourceType.MINING_CONFIG:             StrategyArtifactType.MINING_CONFIG,
            StrategyResourceType.PIPELINE_CONFIG:           StrategyArtifactType.PIPELINE_CONFIG,
            StrategyResourceType.SPLIT_CONFIG:              StrategyArtifactType.SPLIT_CONFIG,
            StrategyResourceType.HUBNESS_REPORT:            StrategyArtifactType.HUBNESS_REPORT,
            StrategyResourceType.HUBNESS_ABLATION_RESULT:   StrategyArtifactType.HUBNESS_ABLATION_RESULT,
        }
        return _map.get(self)

    @classmethod
    def from_str(cls, value: str) -> StrategyResourceType:
        """
        Reconstrói o enum a partir de string com fallback seguro.

        Legacy shim: `"INTERMEDIATE_DATASET"` (tipo extinto) mapeia pra
        `DATASET` — a provenância (derivado) é capturada pela `category`
        no `StrategyResource.from_dict`. JSONs antigos abrem normalmente.
        """
        if not value:
            return cls.UNKNOWN
        normalized = value.upper().strip()
        if normalized == "INTERMEDIATE_DATASET":
            return cls.DATASET
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN