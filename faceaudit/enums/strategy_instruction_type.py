# faceaudit/enums/strategy_instruction_type.py

from __future__ import annotations
from enum import Enum
from typing import List, Tuple


class StrategyInstructionType(Enum):
    """
    Define os tipos de instrução suportados pela strategy engine.

    Cada membro tem:
    - `value`: nome canônico (persistido em JSON, usado em código).
    - `label`: rótulo amigável em inglês técnico (usado em docs/logs).
    - `label_pt`: rótulo em português para apresentação na UI ao perito.
    - `description`: descrição longa em PT-BR (usada em tooltips/help).
    """

    PIPELINE_TRAINING = (
        "PIPELINE_TRAINING",
        "Pipeline Training",
        "Treinar Pipeline",
        "Treina um pipeline de transformação de embeddings.",
    )

    PIPELINE_APPLICATION = (
        "PIPELINE_APPLICATION",
        "Pipeline Application",
        "Aplicar Pipeline",
        "Aplica um pipeline treinado a um ou mais datasets.",
    )

    DATASET_SPLIT = (
        "DATASET_SPLIT",
        "Dataset Split",
        "Dividir Dataset",
        "Divide um dataset em dois subconjuntos segundo uma regra declarada.",
    )

    PAIRED_DATASET_SPLIT = (
        "PAIRED_DATASET_SPLIT",
        "Paired Dataset Split",
        "Dividir Datasets Pareados",
        "Divide DOIS datasets de forma sincronizada por identidade — o mesmo "
        "sorteio de IDs é aplicado em ambos. Ideal para preparar pares "
        "cross-pose (frontal × lateral) com identidades pareadas em "
        "treino e calibração.",
    )

    DATASET_MERGE = (
        "DATASET_MERGE",
        "Dataset Merge",
        "Mesclar Datasets",
        "Combina ou atualiza datasets segundo uma regra declarada.",
    )

    SIMILARITY_MODEL_TRAINING = (
        "SIMILARITY_MODEL_TRAINING",
        "Similarity Model Training",
        "Treinar Modelo de Similaridade",
        "Treina ou prepara um modelo de similaridade.",
    )

    SIMILARITY_SCORING = (
        "SIMILARITY_SCORING",
        "Similarity Scoring",
        "Calcular Scores de Similaridade",
        "Gera scores de similaridade a partir de um modelo treinado e um ou mais datasets.",
    )

    PAIR_MINING = (
        "PAIR_MINING",
        "Pair Mining",
        "Mineração de Pares",
        "Curadoria estatística de pares para treino de calibração.",
    )

    CALIBRATION_MODEL_TRAINING = (
        "CALIBRATION_MODEL_TRAINING",
        "Calibration Model Training",
        "Treinar Modelo de Calibração",
        "Treina ou prepara um modelo de calibração a partir de scores e labels.",
    )

    SCORE_CALIBRATION = (
        "SCORE_CALIBRATION",
        "Score Calibration",
        "Calibrar Scores",
        "Aplica um modelo de calibração treinado a um SimilarityScoreDataset, "
        "produzindo log-LR por par.",
    )

    SCORE_VECTORIZATION = (
        "SCORE_VECTORIZATION",
        "Score Vectorization",
        "Vetorização de Scores",
        "Combina dois ou mais SimilarityScoreDataset (mesmos pares) num "
        "único dataset onde cada amostra carrega um vetor de scores. "
        "Permite treinar calibrador multivariado (LogReg 2D+) que aprende "
        "pesos ótimos por sistema. Score-level combination forense — "
        "Brümmer & du Preez 2006.",
    )

    STRATEGY_EVALUATION = (
        "STRATEGY_EVALUATION",
        "Strategy Evaluation",
        "Avaliar Estratégia",
        "Calcula métricas forenses (EER, Cllr, Cllr_min via PAVA, AUC) e dados "
        "de plot (DET, Tippett, ECE) a partir de um CALIBRATED_SCORES.",
    )

    CASEWORK_VISUALIZATION = (
        "CASEWORK_VISUALIZATION",
        "Casework Visualization",
        "Visualizar Caso Forense",
        "Plota distribuições de referência (gen/imp) sobrepostas com escores "
        "do caso forense — KDE quando há ≥30 pares de caso, linhas verticais "
        "tracejadas com valor numérico anotado quando há menos.",
    )

    DATASET_BALANCING = (
        "DATASET_BALANCING",
        "Dataset Balancing",
        "Balancear Dataset",
        "Balanceia o número de amostras por identidade (cap, floor+cap ou "
        "auto-min) selecionando descartes por sorteio aleatório ou por "
        "métrica de qualidade (FIQA, OFIQ, SERFIQ, norma do embedding).",
    )

    HUBNESS_ANALYSIS = (
        "HUBNESS_ANALYSIS",
        "Hubness Analysis",
        "Análise de Hubness",
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
        "Ablação de Hubness",
        "Estudo de ablação topológica — recorta progressivamente o dataset "
        "(por quantidade de fotos por ID ou priorizando frontalidade) e "
        "mede como o hubness evolui em cada cenário. Custoso: refaz "
        "O(N²) em N subsets.",
    )

    KNN_TYPICALITY_MODEL_TRAINING = (
        "KNN_TYPICALITY_MODEL_TRAINING",
        "kNN Typicality Model Training",
        "Treinar Modelo de Tipicidade (kNN)",
        "Treina um modelo de tipicidade não-paramétrico baseado em k-NN esférico "
        "a partir de um dataset de referência da subpopulação relevante (já "
        "estratificada). Persiste a referência inteira como modelo e pré-computa "
        "a CDF empírica leave-one-out de T_k pra inferência de percentile.",
    )

    TYPICALITY_PAIR_SCORING = (
        "TYPICALITY_PAIR_SCORING",
        "Typicality Pair Scoring",
        "Calcular Score de Tipicidade por Par",
        "Aplica modelo(s) de tipicidade treinado(s) a cada face de cada par "
        "(probe, reference) de um dataset, e produz um SimilarityScoreDataset "
        "onde o score de cada par é a combinação simétrica (MEAN/MIN/MAX/PRODUCT/"
        "ABS_DIFF) das tipicidades individuais. Suporta dois modelos distintos "
        "(probe vs reference) para casos cross-população (ex.: cross-pose: "
        "lateral × frontal). Pareado com SIMILARITY_SCORES pelo "
        "SCORE_VECTORIZATION pra treinar calibrador 2D+ que conheça raridade "
        "local.",
    )

    def __init__(
        self,
        value: str,
        label: str,
        label_pt: str,
        description: str,
    ):
        self._value_ = value
        self.label = label
        self.label_pt = label_pt
        self.description = description

    def __str__(self) -> str:
        """Default representation usa label PT (apresentação ao perito)."""
        return self.label_pt

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
        """Lista `(value, label_pt)` — usada em combos da UI."""
        return [(item.value, item.label_pt) for item in cls]

    @classmethod
    def choices_with_description(cls) -> List[Tuple[str, str, str]]:
        """Lista `(value, label_pt, description)` — combos + tooltip."""
        return [(item.value, item.label_pt, item.description) for item in cls]
