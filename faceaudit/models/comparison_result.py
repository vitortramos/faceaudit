from dataclasses import dataclass, field
from typing import List
from .category_diff import CategoryDiff

@dataclass
class ComparisonResult:
    """
    Objeto consolidado (DTO) contendo o relatório de comparação entre dois datasets.
    Gerado pelo DatasetComparisonService para alimentar o CompareDialog.
    """
    # Metadados de Identificação
    dataset_a_name: str
    dataset_b_name: str

    # --- Estatísticas de Volume (Imagens) ---
    total_imgs_a: int
    total_imgs_b: int

    # --- Estatísticas de Identidade (Sets) ---
    total_ids_a: int
    total_ids_b: int
    
    # Quantas identidades existem em AMBOS (Interseção)
    common_ids_count: int
    
    # Quantas identidades são exclusivas de cada um (Diferença de Sets)
    unique_ids_a_count: int
    unique_ids_b_count: int

    # --- Detalhes Demográficos (Listas para as Tabelas) ---
    # Cada lista contém objetos CategoryDiff prontos para exibição
    gender_diffs: List[CategoryDiff] = field(default_factory=list)
    ethnicity_diffs: List[CategoryDiff] = field(default_factory=list)
    origin_diffs: List[CategoryDiff] = field(default_factory=list)
    
    # --- Comparação Técnica (Colunas de Vetores) ---
    # Ex: A tem ['embedding_v2'], B tem ['embedding_v4']
    common_columns: List[str] = field(default_factory=list)
    missing_in_b: List[str] = field(default_factory=list) # O que tem no A mas falta no B