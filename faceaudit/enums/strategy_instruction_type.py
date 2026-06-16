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

    TYPICALITY_MODEL_TRAINING = (
        "TYPICALITY_MODEL_TRAINING",
        "Typicality Model Training",
        "Treinar Modelo de Tipicidade",
        "Treina um modelo de tipicidade (k-NN não-paramétrico ou MoVMF "
        "paramétrico EXPERIMENTAL) a partir de um dataset de referência "
        "da subpopulação relevante. Backend escolhido via "
        "typicality_model_type; hiperparâmetros embutidos no "
        "typicality_model_config polimórfico.",
    )

    TYPICALITY_PAIR_SCORING = (
        "TYPICALITY_PAIR_SCORING",
        "Typicality Pair Scoring",
        "Calcular Score de Tipicidade por Par",
        "Aplica modelo(s) de tipicidade treinado(s) a cada face de cada par "
        "(probe, reference) de um dataset, e produz um SimilarityScoreDataset "
        "onde o score de cada par é a combinação simétrica (MEAN/MIN/MAX/PRODUCT/"
        "ABS_DIFF) das tipicidades individuais OU a tipicidade de um papel só "
        "(REFERENCE_ONLY/PROBE_ONLY — alimentam SCORE_INTERACTION_FEATURE). "
        "Suporta dois modelos distintos (probe vs reference) para casos "
        "cross-população (ex.: cross-pose: lateral × frontal). Pareado com "
        "SIMILARITY_SCORES pelo SCORE_VECTORIZATION pra treinar calibrador "
        "2D+ que conheça raridade local.",
    )

    SCORE_INTERACTION_FEATURE = (
        "SCORE_INTERACTION_FEATURE",
        "Score Interaction Feature",
        "Feature de Interação Cosseno × Raridade",
        "Constrói feature forense de interação `(base − s0)·rarity` "
        "(modos ONE_SIDED_POSITIVE/LINEAR/HINGE_BASE) a partir de "
        "DOIS SimilarityScoreDataset alinháveis por par: base (cosseno/PLDA) "
        "e modifier (tipicidade do TYPICALITY_PAIR_SCORING). `s0` (ponto "
        "de evidência neutra, logLR=0) é resolvido em três modos: "
        "AUTO_LOGREG (calibrador auxiliar univariado), MANUAL (valor "
        "fixo), INHERIT (herda metadata de execução anterior — "
        "obrigatório em validação e casework). Inversão typicality→rarity "
        "automática por score_type (PERCENTILE→1−v/100; MEAN_KNN/LOG_DENSITY→−v). "
        "Output `raw_score` 1D consumível por SCORE_VECTORIZATION para "
        "fusão LogReg multivariada. Bloqueia AUTO_LOGREG em casework; "
        "valida `same_person` e família do modifier.",
    )

    LR_ARCHITECTURE_VECTORIZATION = (
        "LR_ARCHITECTURE_VECTORIZATION",
        "LR Architecture Vectorization",
        "Vetorização por Arquitetura LR",
        "Instrução declarativa de alto nível que substitui o "
        "encadeamento manual SCORE_INTERACTION_FEATURE×N + "
        "SCORE_VECTORIZATION para uma das 9 arquiteturas de calibração "
        "LR (COSINE_ONLY a BILATERAL_FULL_CENTERED D=7). O usuário "
        "escolhe a arquitetura via enum e a instrução cuida da "
        "construção interna (hinges, raridades centradas, vetorização) "
        "e do binding do fingerprint para o calibrador. As instruções "
        "SCORE_INTERACTION_FEATURE e SCORE_VECTORIZATION continuam "
        "disponíveis para composições exóticas. Fase derivada do "
        "InstructionExecutionFrame; bloqueia ajuste em VALIDATION/"
        "CASEWORK exigindo parameter_mode=INHERIT.",
    )

    def __new__(
        cls,
        value: str,
        label: str,
        label_pt: str,
        description: str,
    ):
        # Pattern canônico de Enum com valor composto — ver
        # typicality_reference_mode.py pra explicação completa.
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.label_pt = label_pt
        obj.description = description
        return obj

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
