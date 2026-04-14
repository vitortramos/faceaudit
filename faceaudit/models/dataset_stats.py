# faceaudit/models/dataset_stats.py
from dataclasses import dataclass, field
from typing import Set, Dict

@dataclass
class DatasetStats:
    """
    Representa metadados agnósticos de um dataset facial.
    Enriquecido com histogramas demográficos.
    """
    # Totais
    n_imgs: int = 0
    n_ids: int = 0

    # Dimensão dos embeddings (0 se não aplicável ou dataset vazio)
    embedding_dim: int = 0

    # Identificação
    identity_set: Set[str] = field(default_factory=set)
    hash_set: Set[str] = field(default_factory=set)

    embedding_columns: Set[str] = field(default_factory=set)

    # Estatísticas Demográficas (Histogramas)
    # Ex: {'Male': 150, 'Female': 120}
    gender_counts: Dict[str, int] = field(default_factory=dict)
    ethnicity_counts: Dict[str, int] = field(default_factory=dict)
    origin_counts: Dict[str, int] = field(default_factory=dict)