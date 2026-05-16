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

    MIN_IDENTITIES = (
        "MIN_IDENTITIES",
        "Mínimo de identidades",
        "Verifica que o dataset tem pelo menos N identidades distintas. "
        "Aplicável a SPLIT, BALANCING e treinos com LDA.",
    )

    MIN_SAMPLES = (
        "MIN_SAMPLES",
        "Mínimo de amostras",
        "Verifica que o dataset tem pelo menos N amostras totais. "
        "Aplicável a HUBNESS, SPLIT por índice e treinos com Whitening.",
    )

    MODEL_FITTED = (
        "MODEL_FITTED",
        "Modelo treinado",
        "Verifica que o recurso (pipeline, modelo de similaridade ou "
        "calibrador) foi efetivamente treinado (`is_fitted=True`). "
        "Sem isso o executor crasha em runtime.",
    )

    EMBEDDING_DIM_MATCHES_MODEL = (
        "EMBEDDING_DIM_MATCHES_MODEL",
        "Compatibilidade dimensional dataset×modelo",
        "Verifica que `dataset.embedding_dim == model.input_dimension`. "
        "Aplicável a PIPELINE_APPLICATION e SIMILARITY_SCORING.",
    )

    SCORE_CALIBRATION_STATE = (
        "SCORE_CALIBRATION_STATE",
        "Estado de calibração do score dataset",
        "Verifica se o score dataset está RAW ou CALIBRATED. Evita dupla "
        "calibração e garante que avaliação só rode em scores calibrados.",
    )

    CASEWORK_STATE = (
        "CASEWORK_STATE",
        "Estado de casework do score dataset",
        "Verifica se o score dataset tem `metadata.is_casework` no estado "
        "esperado (True para caso, False para validação).",
    )

    SCORE_TYPE_MATCH = (
        "SCORE_TYPE_MATCH",
        "Mesmo tipo de score (referência × caso)",
        "Em CASEWORK_VISUALIZATION, verifica que reference e case têm "
        "o mesmo tipo de score (RAW ou CALIBRATED).",
    )

    DATASETS_EMBEDDING_COMPAT = (
        "DATASETS_EMBEDDING_COMPAT",
        "Compatibilidade de embedding entre datasets",
        "Verifica que múltiplos datasets compartilham a mesma coluna de "
        "embedding e a mesma dimensão (pré-condição pra DATASET_MERGE).",
    )

    QUALITY_METRIC_AVAILABLE = (
        "QUALITY_METRIC_AVAILABLE",
        "Métrica de qualidade disponível",
        "Verifica que a métrica de qualidade escolhida (FIQA/OFIQ/SERFIQ/"
        "EMBEDDING_NORM) tem cobertura suficiente no dataset (default 50%).",
    )

    HAS_POSE_METADATA = (
        "HAS_POSE_METADATA",
        "Metadata de pose disponível",
        "Verifica que o dataset tem colunas yaw/pitch necessárias pro "
        "modo BY_POSE de HUBNESS_ABLATION.",
    )

    MAX_SAMPLES_WARNING = (
        "MAX_SAMPLES_WARNING",
        "Aviso de tamanho excessivo",
        "Emite WARNING quando o dataset excede um threshold (custo O(N²) "
        "ou similar). Usado em HUBNESS_ANALYSIS e HUBNESS_ABLATION.",
    )

    SCORE_VECTOR_PAIR_COMPATIBILITY = (
        "SCORE_VECTOR_PAIR_COMPATIBILITY",
        "Compatibilidade de pares (SCORE_VECTORIZATION)",
        "Verifica que dois ou mais SimilarityScoreDataset compartilham o "
        "mesmo conjunto de pares (probe.face_hash + reference.face_hash) — "
        "pré-condição pra combinar via SCORE_VECTORIZATION.",
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
