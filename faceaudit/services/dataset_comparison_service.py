from typing import List, Dict, Set
from faceaudit.models.dataset_stats import DatasetStats
from faceaudit.models.comparison_result import ComparisonResult
from faceaudit.models.category_diff import CategoryDiff

class DatasetComparisonService:
    """
    Serviço de lógica pura responsável por comparar dois objetos DatasetStats.
    
    Calcula interseções de conjuntos (IDs), diferenças demográficas e
    discrepâncias de colunas técnicas. Não realiza I/O.
    """

    def compare_stats(self, 
                      stats_a: DatasetStats, 
                      stats_b: DatasetStats, 
                      name_a: str, 
                      name_b: str) -> ComparisonResult:
        """
        Realiza a comparação completa entre dois snapshots estatísticos.

        Args:
            stats_a (DatasetStats): Estatísticas do primeiro dataset (Referência A).
            stats_b (DatasetStats): Estatísticas do segundo dataset (Referência B).
            name_a (str): Nome de exibição do dataset A.
            name_b (str): Nome de exibição do dataset B.

        Returns:
            ComparisonResult: O DTO preenchido com todo o relatório de diferenças.
        """
        
        # 1. Comparação de Identidades (Set Operations)
        # Calcula interseção e exclusivos usando álgebra de conjuntos
        set_a = stats_a.identity_set
        set_b = stats_b.identity_set
        
        common_ids = len(set_a.intersection(set_b))
        unique_a = len(set_a - set_b)
        unique_b = len(set_b - set_a)

        # 2. Comparação Demográfica (Gerar Listas de CategoryDiff)
        gender_diffs = self._generate_diff_list(stats_a.gender_counts, stats_b.gender_counts)
        ethnicity_diffs = self._generate_diff_list(stats_a.ethnicity_counts, stats_b.ethnicity_counts)
        origin_diffs = self._generate_diff_list(stats_a.origin_counts, stats_b.origin_counts)

        # 3. Comparação Técnica (Colunas de Vetores)
        cols_a = stats_a.embedding_columns
        cols_b = stats_b.embedding_columns
        
        common_cols = sorted(list(cols_a.intersection(cols_b)))
        missing_in_b = sorted(list(cols_a - cols_b))

        # 4. Construção do Resultado
        return ComparisonResult(
            dataset_a_name=name_a,
            dataset_b_name=name_b,
            
            # Totais Brutos
            total_imgs_a=stats_a.n_imgs,
            total_imgs_b=stats_b.n_imgs,
            total_ids_a=stats_a.n_ids,
            total_ids_b=stats_b.n_ids,
            
            # Análise de Conjuntos
            common_ids_count=common_ids,
            unique_ids_a_count=unique_a,
            unique_ids_b_count=unique_b,
            
            # Detalhes (Listas)
            gender_diffs=gender_diffs,
            ethnicity_diffs=ethnicity_diffs,
            origin_diffs=origin_diffs,
            
            # Técnico
            common_columns=common_cols,
            missing_in_b=missing_in_b
        )

    def _generate_diff_list(self, dict_a: Dict[str, int], dict_b: Dict[str, int]) -> List[CategoryDiff]:
        """
        Helper que compara dois dicionários de contagem e gera a lista de diferenças.
        
        Exemplo:
            A={'Male': 10}, B={'Male': 12, 'Female': 5}
            Retorna: [
                CategoryDiff('Male', 10, 12),  -> Diff +2
                CategoryDiff('Female', 0, 5)   -> Diff +5
            ]
        """
        # Une todas as chaves (categorias) presentes em A ou B
        all_categories = set(dict_a.keys()).union(set(dict_b.keys()))
        
        diff_list = []
        for cat in sorted(list(all_categories)):
            count_a = dict_a.get(cat, 0)
            count_b = dict_b.get(cat, 0)
            
            diff_list.append(CategoryDiff(
                category=cat,
                count_a=count_a,
                count_b=count_b
            ))
            
        return diff_list