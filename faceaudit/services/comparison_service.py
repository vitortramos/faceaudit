# faceaudit/services/comparison_service.py

import numpy as np
from typing import Optional, Tuple, Callable, Dict, List
import pandas as pd

# Imports da Arquitetura
from faceaudit.models import EmbeddingDataset, EmbeddingComparisonDataset, EmbeddingComparison


class ComparisonService:
    """
    Serviço core responsável por realizar o confronto (matching) 1:N ou N:M entre datasets.
    
    Este serviço é a espinha dorsal de qualquer análise biométrica no sistema. Ele calcula a 
    SIMILARIDADE DE COSSENO de forma altamente vetorizada usando NumPy, bypassando loops 
    em Python e garantindo máxima performance de CPU e RAM. 
    
    Ele não depende do pipeline do modelo, operando exclusivamente em cima das matrizes extraídas.
    """

    def __init__(self, eps: float = 1e-12, clip: bool = True, warn_if_not_unit: bool = True):
        """
        Inicializa o serviço de comparação com travas de segurança numérica.

        Args:
            eps (float): Épsilon usado para evitar divisão por zero durante a normalização L2.
            clip (bool): Se True, faz clamp final do resultado matricial para [-1.0, 1.0], 
                         compensando imprecisões de ponto flutuante do NumPy.
            warn_if_not_unit (bool): Se True, emite um aviso silencioso no console se detectar 
                                     vetores de entrada com normas muito distantes de 1.0.
        """
        self.eps = float(eps)
        self.clip = bool(clip)
        self.warn_if_not_unit = bool(warn_if_not_unit)

    # =========================================================================
    # PROCESSAMENTO MASSIVO (BATCHING PARA BIG DATA)
    # =========================================================================

    def compare_batch_against_all(self, batch_matrix: np.ndarray, reference_matrix: np.ndarray, is_normalized: bool = False) -> np.ndarray:
        """
        Calcula a similaridade de cosseno de um LOTE (batch) contra a base de referência inteira.
        Usado para evitar MemoryError (OOM) em datasets com N > 100.000.

        Args:
            batch_matrix: Matriz (B, D) das amostras a serem consultadas.
            reference_matrix: Matriz (N, D) do banco de dados total.
            is_normalized: Se True, pula a etapa de normalização L2 (ganho brutal de performance 
                           se as matrizes já chegam unitárias).

        Returns:
            np.ndarray: Matriz (B, N) de similaridades de cosseno.
        """
        if is_normalized:
            norm_batch = batch_matrix
            norm_ref = reference_matrix
        else:
            norm_batch = self._l2_normalize_rows(batch_matrix)
            norm_ref = self._l2_normalize_rows(reference_matrix)
        
        sim_matrix = np.dot(norm_batch, norm_ref.T)
        
        if self.clip:
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            
        return sim_matrix

    def get_top_k_neighbors_batch(self, batch_matrix: np.ndarray, reference_matrix: np.ndarray, k: int, is_normalized: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compara o lote contra a base toda e retorna APENAS os Top-K vizinhos mais similares.
        Altamente otimizado para o algoritmo de Hubness, reduzindo o tráfego de memória.

        Args:
            batch_matrix: Matriz (B, D) das amostras de consulta.
            reference_matrix: Matriz (N, D) do banco total.
            k (int): Número de vizinhos a reter.
            is_normalized: Se True, avisa que as matrizes já estão normalizadas.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (Índices dos vizinhos (B, K), Scores de Similaridade (B, K))
        """
        # Calcula similaridade
        sim_matrix = self.compare_batch_against_all(batch_matrix, reference_matrix, is_normalized=is_normalized)
        
        B, N = sim_matrix.shape
        k_eff = min(k, N)
        
        # argpartition é O(N) em vez de O(N log N) do sort completo
        # Pega os 'k_eff' maiores índices ao longo da dimensão N (eixo 1)
        top_k_idx = np.argpartition(sim_matrix, -k_eff, axis=1)[:, -k_eff:]
        
        # Precisamos de um array auxiliar para indexar de forma avançada no NumPy
        row_indices = np.arange(B)[:, None]
        top_k_scores = sim_matrix[row_indices, top_k_idx]
        
        return top_k_idx, top_k_scores

    # =========================================================================
    # COMPARAÇÕES COMPLETAS (LEGACY)
    # =========================================================================

    def process_comparison(
        self,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
        comparison_name: str = "Comparison",
        logger_callback: Callable[[str], None] = print
    ) -> EmbeddingComparisonDataset:
        """
        Realiza a comparação exaustiva e gera objetos instanciados de todos os cruzamentos.

        Comportamento Inteligente de Matriz:
        - Datasets Distintos (A vs B): Realiza a matriz retangular total (M x N).
        - Auto-comparação (A vs A): Otimiza a construção iterando apenas sobre a
          parte triangular superior da matriz de distâncias (i < j). Isso elimina
          a diagonal principal (foto contra si mesma) e o espelhamento inferior, 
          economizando 50% de RAM e tempo de construção no Python.

        ⚠️ Aviso: Este método instancia objetos 'EmbeddingComparison' individuais na memória RAM.
        Usar para bases N > 10.000 (onde os pares chegam a 50 milhões) causará Out Of Memory (OOM).
        Para mineração massiva, use o método 'mine_hard_samples'.

        Args:
            probe_dataset (EmbeddingDataset): Dataset contendo as imagens Questionadas (Linhas).
            reference_dataset (EmbeddingDataset): Dataset contendo as imagens Padrão (Colunas).
            comparison_name (str): Nome do Dataset gerado.
            logger_callback (Callable): Função para injeção de logs na UI.

        Returns:
            EmbeddingComparisonDataset: Dataset resultante com a lista de pares pontuados.
            
        Raises:
            ValueError: Se os datasets possuírem dimensões latentes incompatíveis.
        """
        logger_callback(
            f"[{self.__class__.__name__}] Calculando similaridades (COSSENO): "
            f"{len(probe_dataset)} (Probe) x {len(reference_dataset)} (Ref) pares originais..."
        )

        is_auto_comparison = (probe_dataset is reference_dataset)
        
        if not is_auto_comparison:
            self._verify_duplicates(probe_dataset, reference_dataset, logger_callback)
        else:
            logger_callback(f"[{self.__class__.__name__}] ℹ️ Modo Auto-Comparação detectado. Otimizando processamento.")

        matrix_p = probe_dataset.matrix
        matrix_r = reference_dataset.matrix

        if matrix_p.size == 0 or matrix_r.size == 0:
            raise ValueError("Um dos datasets está vazio ou sem embeddings válidos.")

        if matrix_p.ndim != 2 or matrix_r.ndim != 2:
            raise ValueError("Embeddings devem ser matrizes 2D (N, D).")

        if matrix_p.shape[1] != matrix_r.shape[1]:
            raise ValueError(
                f"Dimensionalidades diferentes: Probe D={matrix_p.shape[1]} vs Ref D={matrix_r.shape[1]}"
            )

        if self.warn_if_not_unit:
            self._warn_if_vectors_not_unit(matrix_p, "Probe", logger_callback)
            self._warn_if_vectors_not_unit(matrix_r, "Ref", logger_callback)

        similarity_matrix = self.compare_batch_against_all(matrix_p, matrix_r)

        comparisons = []
        items_p = probe_dataset.items
        items_r = reference_dataset.items

        total_ops = len(items_p) * len(items_r)
        if is_auto_comparison:
            total_ops = (len(items_p) * (len(items_p) - 1)) // 2

        if total_ops > 50_000_000:
            logger_callback(f"[{self.__class__.__name__}] ⚠️ Atenção: Gerando {total_ops} objetos. Alto consumo de RAM esperado.")

        for i, item_p in enumerate(items_p):
            id_p = getattr(item_p, "identity", None)
            start_j = i + 1 if is_auto_comparison else 0
            
            for j in range(start_j, len(items_r)):
                item_r = items_r[j]
                sim_score = float(similarity_matrix[i, j])

                id_r = getattr(item_r, "identity", None)
                is_same = (id_p is not None and id_r is not None and id_p == id_r)

                comparisons.append(
                    EmbeddingComparison(
                        embedding_1=item_p,
                        embedding_2=item_r,
                        cosine_similarity=sim_score,
                        same_person=is_same,
                    )
                )

        logger_callback(f"[{self.__class__.__name__}] ✅ Sucesso! {len(comparisons)} pares extraídos para a RAM.")

        return EmbeddingComparisonDataset(
            name=comparison_name,
            description=f"Comparação {probe_dataset.name} vs {reference_dataset.name}",
            items=comparisons,
        )

    def compare_all(self, dataset: EmbeddingDataset) -> np.ndarray:
        """
        Realiza a auto-comparação N x N e retorna a matriz bruta de similaridade.
        
        Diferente do 'process_comparison', este método não instancia objetos DTO,
        operando estritamente com tensores NumPy para máxima eficiência de memória.
        Utilizado por serviços de análise estatística e topológica (Hubness).

        Args:
            dataset (EmbeddingDataset): O dataset para auto-confronto.

        Returns:
            np.ndarray: Matriz (N, N) de similaridades de cosseno.
        """
        matrix = dataset.matrix
        if matrix.size == 0:
            return np.array([])

        return self.compare_batch_against_all(matrix, matrix)

    def generate_pairs_only(
        self,
        probe_dataset: EmbeddingDataset,
        reference_dataset: EmbeddingDataset,
        comparison_name: str = "Comparison_Pairs_Only",
        logger_callback: Callable[[str], None] = print
    ) -> EmbeddingComparisonDataset:
        """
        Gera a estrutura de pareamento (Identifica Genuínos e Impostores) SEM calcular cosseno.
        
        Otimizado para preparar a entrada de modelos gerativos e probabilísticos (como PLDA). 
        Nestes casos, o próprio modelo se encarrega de aplicar sua métrica de distância 
        (Log-Likelihood Ratio), tornando o cálculo prévio de cosseno desperdício computacional.

        Args:
            probe_dataset: Dataset de "Questionadas".
            reference_dataset: Dataset de "Padrões".
            comparison_name: Nome do dataset resultante.
            logger_callback: Função para injeção de logs na UI.

        Returns:
            EmbeddingComparisonDataset: Dataset contendo as combinações, com cosine_similarity = 0.0.
        """
        logger_callback(
            f"[{self.__class__.__name__}] Gerando estrutura de pares (Ground Truth): "
            f"{len(probe_dataset)} x {len(reference_dataset)}..."
        )

        is_auto_comparison = (probe_dataset is reference_dataset)
        
        if not is_auto_comparison:
            self._verify_duplicates(probe_dataset, reference_dataset, logger_callback)
        else:
            logger_callback(f"[{self.__class__.__name__}] ℹ️ Modo Auto-Comparação detectado. Otimizando para matriz triangular.")

        comparisons = []
        items_p = probe_dataset.items
        items_r = reference_dataset.items

        total_ops = len(items_p) * len(items_r)
        if is_auto_comparison:
            total_ops = (len(items_p) * (len(items_p) - 1)) // 2

        if total_ops > 50_000_000:
            logger_callback(f"[{self.__class__.__name__}] ⚠️ Gerando {total_ops} objetos em branco (sem score).")

        for i, item_p in enumerate(items_p):
            id_p = getattr(item_p, "identity", None)
            start_j = i + 1 if is_auto_comparison else 0
            
            for j in range(start_j, len(items_r)):
                item_r = items_r[j]
                
                id_r = getattr(item_r, "identity", None)
                is_same = (id_p is not None and id_r is not None and id_p == id_r)

                comparisons.append(
                    EmbeddingComparison(
                        embedding_1=item_p,
                        embedding_2=item_r,
                        cosine_similarity=0.0, # Ignorado deliberadamente
                        same_person=is_same,
                    )
                )

        logger_callback(f"[{self.__class__.__name__}] ✅ Estrutura de {len(comparisons)} pares gerada.")

        return EmbeddingComparisonDataset(
            name=comparison_name,
            description=f"Pares Brutos {probe_dataset.name} vs {reference_dataset.name}",
            items=comparisons,
        )

    def mine_hard_samples(
        self,
        dataset: EmbeddingDataset,
        top_n: int = 100,
        impostor_threshold: float = 0.3,
        genuine_threshold: float = 0.5,
        logger_callback: Callable[[str], None] = print,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Mineração profunda vetorizada (Hard Negative & Hard Positive Mining).

        Calcula a similaridade da matriz completa N x N (Triângulo superior O(N²/2)) e
        extrai diretamente dos tensores NumPy os pares limítrofes, evitando a criação
        massiva de objetos na memória RAM.

        Ideal para diagnósticos rápidos de Overfitting em datasets enormes e geração de relatórios.

        Args:
            dataset (EmbeddingDataset): O dataset base que será cruzado contra si mesmo.
            top_n (int): A quantidade de "piores exemplos" a ser extraída para cada categoria.
            impostor_threshold (float): Cosseno mínimo para um par de identidades
                diferentes ser considerado *hard impostor* (FP potencial). Pares
                abaixo disso são claramente impostores e NÃO geram risco de FP —
                não interessam à auditoria. Default 0.3 (zona de decisão típica
                de modelos como AdaFace SEPAEL).
            genuine_threshold (float): Cosseno máximo para um par da mesma
                identidade ser considerado *hard genuine* (FN potencial). Pares
                acima disso já são bem reconhecidos. Default 0.5.
            logger_callback (Callable): Função para injeção de logs na UI.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                Uma tupla com dois DataFrames Pandas:
                1. O Top N dos piores pares Genuínos (mesma pessoa, cosseno
                   ≤ `genuine_threshold`).
                2. O Top N dos piores pares Impostores (pessoas diferentes,
                   cosseno ≥ `impostor_threshold`).
                Quando nenhum par passa o threshold, o DataFrame correspondente
                retorna vazio.
        """
        logger_callback(f"[{self.__class__.__name__}] Iniciando mineração vetorizada O(N²) para {len(dataset)} faces...")
        
        items = dataset.items
        matrix = dataset.matrix
        n = len(items)
        
        if n < 2: 
            return pd.DataFrame(), pd.DataFrame()

        sim_matrix = self.compare_batch_against_all(matrix, matrix)

        ids = np.array([getattr(item, "identity", None) for item in items])
        
        i_idx, j_idx = np.triu_indices(n, k=1)
        sims = sim_matrix[i_idx, j_idx]
        
        valid_ids_mask = (ids[i_idx] != None) & (ids[j_idx] != None)
        is_genuine = (ids[i_idx] == ids[j_idx]) & valid_ids_mask

        gen_mask = is_genuine
        imp_mask = ~is_genuine & valid_ids_mask

        gen_sims = sims[gen_mask]
        imp_sims = sims[imp_mask]

        # Piores Genuínos (Busca do Menor para o Maior).
        # Filtragem por threshold ANTES de tirar o top — pares com cosseno
        # acima de `genuine_threshold` já estão bem reconhecidos e não são
        # FN potenciais; não devem aparecer na auditoria.
        worst_gen_rows = []
        if len(gen_sims) > 0:
            gen_risk_mask = gen_sims <= genuine_threshold
            risky_gen_sims = gen_sims[gen_risk_mask]
            if len(risky_gen_sims) > 0:
                # Mapeia índices "risky" de volta pros índices originais do gen pool.
                gen_risk_local_idx = np.flatnonzero(gen_risk_mask)
                k_gen = min(top_n, len(risky_gen_sims))
                worst_gen_local = np.argpartition(risky_gen_sims, k_gen - 1)[:k_gen]
                worst_gen_local = worst_gen_local[np.argsort(risky_gen_sims[worst_gen_local])]
                worst_gen_idx = gen_risk_local_idx[worst_gen_local]

                abs_gen_i = i_idx[gen_mask][worst_gen_idx]
                abs_gen_j = j_idx[gen_mask][worst_gen_idx]
                final_gen_sims = gen_sims[worst_gen_idx]

                for rank_idx in range(k_gen):
                    item_1, item_2 = items[abs_gen_i[rank_idx]], items[abs_gen_j[rank_idx]]
                    worst_gen_rows.append({
                        "identity_1": getattr(item_1, "identity", "N/A"),
                        "file_name_1": item_1.file_name or "N/A",
                        "file_name_2": item_2.file_name or "N/A",
                        "cosine_similarity": final_gen_sims[rank_idx],
                        "image_path_1": item_1.image_path,
                        "image_path_2": item_2.image_path
                    })

        # Piores Impostores (Busca do Maior para o Menor).
        # Filtra ANTES de tirar o top — pares com cosseno abaixo de
        # `impostor_threshold` são claramente impostores e NÃO oferecem
        # risco de FP, não interessam à auditoria forense (a "estética" de
        # serem o topo dentro do dataset é irrelevante quando o cosseno
        # deles está bem abaixo da zona de decisão).
        worst_imp_rows = []
        if len(imp_sims) > 0:
            imp_risk_mask = imp_sims >= impostor_threshold
            risky_imp_sims = imp_sims[imp_risk_mask]
            if len(risky_imp_sims) > 0:
                imp_risk_local_idx = np.flatnonzero(imp_risk_mask)
                k_imp = min(top_n, len(risky_imp_sims))
                worst_imp_local = np.argpartition(-risky_imp_sims, k_imp - 1)[:k_imp]
                worst_imp_local = worst_imp_local[np.argsort(-risky_imp_sims[worst_imp_local])]
                worst_imp_idx = imp_risk_local_idx[worst_imp_local]

                abs_imp_i = i_idx[imp_mask][worst_imp_idx]
                abs_imp_j = j_idx[imp_mask][worst_imp_idx]
                final_imp_sims = imp_sims[worst_imp_idx]

                for rank_idx in range(k_imp):
                    item_1, item_2 = items[abs_imp_i[rank_idx]], items[abs_imp_j[rank_idx]]
                    worst_imp_rows.append({
                        "identity_1": getattr(item_1, "identity", "N/A"),
                        "identity_2": getattr(item_2, "identity", "N/A"),
                        "file_name_1": item_1.file_name or "N/A",
                        "file_name_2": item_2.file_name or "N/A",
                        "cosine_similarity": final_imp_sims[rank_idx],
                        "image_path_1": item_1.image_path,
                        "image_path_2": item_2.image_path
                    })

        df_gen = pd.DataFrame(worst_gen_rows) if worst_gen_rows else pd.DataFrame()
        df_imp = pd.DataFrame(worst_imp_rows) if worst_imp_rows else pd.DataFrame()

        logger_callback(f"[{self.__class__.__name__}] ✅ Mineração vetorizada concluída instantaneamente.")
        return df_gen, df_imp

    def get_identity_confusion_matrix(
        self, 
        dataset: EmbeddingDataset,
        logger_callback: Callable[[str], None] = print
    ) -> Dict[str, List[str]]:
        """
        Calcula a matriz de similaridade entre IDENTIDADES para análise de 'Sósias Globais'.
        
        Em vez de cruzar todas as imagens (N x N), este método tira o centróide (vetor médio L2) 
        de todas as fotos de uma mesma pessoa e compara a pessoa diretamente com as outras 
        (Número_de_Identidades x Número_de_Identidades). 
        
        Isso é usado pelo SplitService para criar "Clusters de Dificuldade" (Clusterização de Sósias),
        garantindo que se uma pessoa for separada para o grupo de calibração, seus sósias
        naturais a acompanhem, viabilizando o Hard Mining futuro.

        Args:
            dataset (EmbeddingDataset): O dataset com múltiplas fotos por indivíduo.
            logger_callback (Callable): Função para injeção de logs na UI.

        Returns:
            Dict[str, List[str]]: Dicionário mapeando cada Identidade Pivô para uma lista 
                                  das demais identidades, ordenadas da mais confusa (pior sósia)
                                  para a menos confusa.
        """
        logger_callback(f"[{self.__class__.__name__}] Calculando Matriz de Confusão Média por Identidade...")
        
        items = dataset.items
        matrix = dataset.matrix
        
        if len(items) == 0:
            return {}

        ids = np.array([getattr(item, "identity", None) for item in items])
        unique_ids = np.unique(ids[ids != None])
        n_ids = len(unique_ids)
        
        if n_ids < 2:
            return {str(uid): [] for uid in unique_ids}

        # Constrói centróide médio (Mean Vector) e L2-Normaliza
        centroids = np.zeros((n_ids, matrix.shape[1]), dtype=np.float64)
        for i, uid in enumerate(unique_ids):
            mask = (ids == uid)
            person_embs = matrix[mask]
            centroids[i] = np.mean(person_embs, axis=0)
            
        centroids = self._l2_normalize_rows(centroids)

        # Cross-comparison massiva das personalidades
        sim_matrix = self.compare_batch_against_all(centroids, centroids, is_normalized=True)
            
        np.fill_diagonal(sim_matrix, -np.inf)
        
        confusion_dict = {}
        for i, pivot_id in enumerate(unique_ids):
            # Argsort invertido (do maior Cosseno para o Menor)
            sorted_indices = np.argsort(-sim_matrix[i])
            sorted_similar_ids = [str(unique_ids[idx]) for idx in sorted_indices]
            confusion_dict[str(pivot_id)] = sorted_similar_ids
            
        logger_callback(f"[{self.__class__.__name__}] ✅ Matriz de Confusão concluída para {n_ids} identidades.")
        return confusion_dict

    # =========================================================================
    # Helpers Internos
    # =========================================================================

    def _l2_normalize_rows(self, X: np.ndarray) -> np.ndarray:
        """Normaliza cada vetor (linha) para norma matemática 1.0 (Unit Vector)."""
        # Agora usando float32 para dobrar a velocidade do BLAS e economizar RAM
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, self.eps, dtype=np.float32)
        return X / norms

    def _warn_if_vectors_not_unit(
        self, 
        X: np.ndarray, 
        label: str, 
        logger_callback: Callable[[str], None]
    ) -> None:
        """
        Sistema de diagnóstico preventivo. Alerta se o pipeline anterior esqueceu 
        de fazer a normalização L2 nos tensores (Embora o Cosseno funcione igual, 
        isso indica anomalia na extração do modelo neural).
        """
        X = np.asarray(X, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1)
        if norms.size == 0:
            return

        mean_norm = float(np.mean(norms))
        min_norm = float(np.min(norms))
        max_norm = float(np.max(norms))

        if mean_norm < 0.90 or mean_norm > 1.10 or min_norm < 0.5 or max_norm > 2.0:
            logger_callback(
                f"[{self.__class__.__name__}] ⚠️ AVISO: {label} Embeddings não parecem L2-Unitários "
                f"(media={mean_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}). "
                f"Normalizando internamente por segurança."
            )

    def _verify_duplicates(
        self, 
        ds1: EmbeddingDataset, 
        ds2: EmbeddingDataset,
        logger_callback: Callable[[str], None]
    ) -> None:
        """
        Verifica vazamento/contaminação de dados (Overlapping Data Leakage) entre a 
        base Probe e a base Reference.
        """
        hashes_ds1 = {item.face_hash for item in ds1.items if item.face_hash is not None}
        hashes_ds2 = {item.face_hash for item in ds2.items if item.face_hash is not None}

        intersection = hashes_ds1.intersection(hashes_ds2)

        if intersection:
            logger_callback(
                f"[{self.__class__.__name__}] ⚠️ ALERTA DE VAZAMENTO: {len(intersection)} imagens idênticas (mesmo hash SHA-256) "
                f"encontradas trafegando nos dois datasets distintos."
            )