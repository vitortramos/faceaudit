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

    PAIRED_DATASET_SPLIT = (
        "PAIRED_DATASET_SPLIT",
        "Paired Dataset Split",
        "Divide DOIS datasets de forma sincronizada por identidade — o mesmo "
        "sorteio de IDs é aplicado em ambos. Ideal para preparar pares "
        "cross-pose (frontal × lateral) com identidades pareadas em "
        "treino e calibração.",
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

    SCORE_CALIBRATION = (
        "SCORE_CALIBRATION",
        "Score Calibration",
        "Aplica um modelo de calibração treinado a um SimilarityScoreDataset, "
        "produzindo log-LR por par.",
    )

    SCORE_VECTORIZATION = (
        "SCORE_VECTORIZATION",
        "Score Vectorization",
        "Combina dois ou mais SimilarityScoreDataset (mesmos pares) num "
        "único dataset onde cada amostra carrega um vetor de scores. "
        "Permite treinar calibrador multivariado (LogReg 2D+) que aprende "
        "pesos ótimos por sistema. Score-level combination forense — "
        "Brümmer & du Preez 2006.",
    )

    STRATEGY_EVALUATION = (
        "STRATEGY_EVALUATION",
        "Strategy Evaluation",
        "Calcula métricas forenses (EER, Cllr, Cllr_min via PAVA, AUC) e dados "
        "de plot (DET, Tippett, ECE) a partir de um CALIBRATED_SCORES.",
    )

    CASEWORK_VISUALIZATION = (
        "CASEWORK_VISUALIZATION",
        "Casework Visualization",
        "Plota distribuições de referência (gen/imp) sobrepostas com escores "
        "do caso forense — KDE quando há ≥30 pares de caso, linhas verticais "
        "tracejadas com valor numérico anotado quando há menos.",
    )

    DATASET_BALANCING = (
        "DATASET_BALANCING",
        "Dataset Balancing",
        "Balanceia o número de amostras por identidade (cap, floor+cap ou "
        "auto-min) selecionando descartes por sorteio aleatório ou por "
        "métrica de qualidade (FIQA, OFIQ, SERFIQ, norma do embedding).",
    )

    HUBNESS_ANALYSIS = (
        "HUBNESS_ANALYSIS",
        "Hubness Analysis",
        "Diagnóstico de viés topológico (hubness) em alta dimensionalidade — "
        "quantifica quais faces atraem vizinhos errados via Nk score, "
        "similaridade impostora e correlações com pose/norma. Produz um "
        "HubnessReport compósito com 5 estudos (analysis base, "
        "stratification, paired study, quality diagnostics, "
        "optimization study).",
    )

    HUBNESS_ABLATION = (
        "HUBNESS_ABLATION",
        "Hubness Ablation",
        "Estudo de ablação topológica — recorta progressivamente o dataset "
        "(por quantidade de fotos por ID ou priorizando frontalidade) e "
        "mede como o hubness evolui em cada cenário. Custoso: refaz "
        "O(N²) em N subsets.",
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