# interface/tabs/strategy_manager/new_mining_config
import itertools
import math
import random
import numpy as np
import pickle
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Union, Any
from scipy import linalg
from scipy.linalg import sqrtm, inv, orthogonal_procrustes
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf, OAS
from sklearn.decomposition import PCA
from collections import defaultdict

# Imports gráficos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Imports dos Modelos e Enums do Domínio
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.models import EmbeddingDataset, EmbeddingInfo

# --- NOVOS IMPORTS ---
from faceaudit.models.transformation_step_3d import TransformationStep3D
from faceaudit.models.transformation_trajectory_data import TransformationTrajectoryData
from faceaudit.enums.whitening_style import WhiteningStyle
from faceaudit.enums.step_type import StepType 
from faceaudit.enums.coral_estimator_type import CoralEstimatorType
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.wccn_estimator_type import WccnEstimatorType
from faceaudit.enums.wccn_grid_search_metric import WccnGridSearchMetric
from faceaudit.enums.lda_estimator_type import LdaEstimatorType
from faceaudit.enums.lda_grid_search_metric import LdaGridSearchMetric

# --- INJEÇÃO DO DTO DE AGREGAÇÃO ---
from faceaudit.models.aggregation_config import AggregationConfig
from faceaudit.models.wccn_config import WCCNConfig
from faceaudit.models.lda_config import LDAConfig


# Prefixo do `face_hash` de template agregado. Marca a linha como SINTÉTICA —
# nenhuma foto real produz um hash com este prefixo, o que torna templates
# distinguíveis de imagens por inspeção direta (e por grep).
_SYNTH_PREFIX = "SYNTH_"

# Versão do esquema de derivação do hash sintético. Fica gravada no
# `loaded_metadata` do dataset agregado: se a fórmula mudar, datasets antigos
# continuam auditáveis e a incompatibilidade de pareamento fica explícita em
# vez de aparecer como "N pares não encontrados" lá na frente.
_SYNTH_HASH_SCHEME = "agg-template-sha256-v2"


class EmbeddingMathService:
    """
    Serviço utilitário de 'Baixo Nível' para Álgebra Linear em Embeddings Faciais.
    Responsável por todas as transformações geométricas, reduções de dimensionalidade,
    agregações biométricas e cálculos de projeção visual 3D do sistema.
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Inicializa o serviço matemático.
        
        Args:
            seed (Optional[int]): Semente aleatória para travar solvers estocásticos 
                                  (como PCA e Subamostragem), garantindo reprodutibilidade 
                                  exata nas execuções e auditorias.
        """
        self.seed = seed
        # Cria um gerador local do Numpy isolado para não poluir o ambiente global
        self.rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    # =========================================================================
    # 1. UTILITÁRIOS MATEMÁTICOS E VISUALIZAÇÃO
    # =========================================================================

    def get_cov_ellipsoid(self, data: np.ndarray, n_std: float = 2.0, force_eigvecs: Optional[np.ndarray] = None):
        """
        Calcula as coordenadas paramétricas de um elipsoide 3D que representa 
        a matriz de covariância de um conjunto de dados.
        
        Args:
            data (np.ndarray): Matriz 3D de dados (Nx3).
            n_std (float): Número de desvios padrões para dimensionar o elipsoide (escala).
            force_eigvecs (Optional[np.ndarray]): Matriz de autovetores para travar a rotação
                                                  visual (Evita o Paradoxo da Esfera).
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Malhas (X, Y, Z) para o plot_wireframe.
        """
        if data is None or data.shape[1] != 3: return None
        if data.shape[0] < 4: return None

        try:
            cov = np.cov(data, rowvar=False)
            center = np.mean(data, axis=0)
            
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # --- TRAVA DE ROTAÇÃO INJETADA ---
            if force_eigvecs is not None:
                # Usa a rotação pré-calculada (estável), mas mantém a escala (autovalores) real
                eigvecs = force_eigvecs
            # ---------------------------------
            
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = np.cos(u)[:, None] * np.sin(v)[None, :]
            y = np.sin(u)[:, None] * np.sin(v)[None, :]
            z = np.outer(np.ones_like(u), np.cos(v))
            
            radii = np.sqrt(np.maximum(eigvals, 0)) * n_std
            
            xyz = np.stack([x.flatten(), y.flatten(), z.flatten()])
            T = np.dot(eigvecs, np.diag(radii))
            xyz_trans = np.dot(T, xyz)
            
            return (xyz_trans[0].reshape(x.shape) + center[0],
                    xyz_trans[1].reshape(y.shape) + center[1],
                    xyz_trans[2].reshape(z.shape) + center[2])
                    
        except Exception as e:
            print(f"Erro ao calcular elipsoide: {e}")
            return None

    def visualize_embedding_transformation(self,
                                           source_dataset: EmbeddingDataset,
                                           target_dataset: Optional[EmbeddingDataset] = None,
                                           mode: str = 'whitening',
                                           n_samples: int = 1000,
                                           coral_matrix: Optional[np.ndarray] = None,
                                           whitening_mean: Optional[np.ndarray] = None,
                                           whitening_matrix: Optional[np.ndarray] = None):
        """
        Gera uma visualização 3D estática (Matplotlib) comparando a distribuição 
        dos embeddings antes e depois de uma transformação geométrica.
        
        Args:
            source_dataset: O dataset de origem.
            target_dataset: O dataset de destino (Requerido para CORAL).
            mode: O tipo de transformação ('whitening' ou 'coral').
            n_samples: Número máximo de pontos para renderizar (evita travamentos na UI).
            coral_matrix: Matriz de alinhamento pré-calculada.
            whitening_mean: Vetor média do Whitening.
            whitening_matrix: Matriz de projeção Whitening.
        """
        print(f"--- Gerando Visualização 3D (Modo: {mode.upper()}) ---")
        fig = plt.figure(figsize=(16, 8))
        
        def sample_matrix(ds: EmbeddingDataset, n: int) -> np.ndarray:
            mat = ds.matrix
            if len(mat) > n:
                # Uso do RNG travado na semente global para garantir a mesma nuvem visual
                indices = self.rng.choice(len(mat), n, replace=False)
                return mat[indices]
            return mat

        if mode == 'whitening':
            matrix = sample_matrix(source_dataset, n_samples)
            if whitening_mean is None or whitening_matrix is None:
                print("    ⚠️ Parâmetros de Whitening não fornecidos. Simulando localmente...")
                mean_local = np.mean(matrix, axis=0)
                centered_local = matrix - mean_local
                U, S, Vt = np.linalg.svd(np.cov(centered_local, rowvar=False))
                X_before = np.dot(centered_local, U[:, :3])
                cov_3d = np.cov(X_before, rowvar=False)
                l3d, v3d = np.linalg.eigh(cov_3d)
                inv_sqrt = np.diag(1. / np.sqrt(l3d + 1e-10))
                whiten_mat_3d = np.dot(np.dot(v3d, inv_sqrt), v3d.T)
                X_after = np.dot(X_before, whiten_mat_3d)
            else:
                from sklearn.decomposition import PCA
                # Trava de semente no solver estocástico do PCA
                pca = PCA(n_components=3, random_state=self.seed).fit(matrix)
                X_before = pca.transform(matrix)
                centered_full = matrix - whitening_mean
                white_full = np.dot(centered_full, whitening_matrix)
                X_after = pca.transform(white_full)
            title_before, title_after = "ANTES: Distribuição Original", "DEPOIS: Whitening (Esférico)"
            data_pairs = [(X_before, None), (X_after, None)]

        elif mode == 'coral':
            if target_dataset is None: raise ValueError("target_dataset obrigatório para CORAL.")
            X_src = sample_matrix(source_dataset, n_samples)
            X_tgt = sample_matrix(target_dataset, n_samples)
            if coral_matrix is None:
                print("    ⚠️ Matriz CORAL não fornecida. Simulando localmente...")
                cov_src = np.cov(X_src, rowvar=False) + np.eye(X_src.shape[1]) * 1e-6
                cov_tgt = np.cov(X_tgt, rowvar=False) + np.eye(X_tgt.shape[1]) * 1e-6
                try:
                    A_coral_local = np.dot(inv(sqrtm(cov_src)), sqrtm(cov_tgt))
                except:
                    A_coral_local = np.eye(X_src.shape[1])
                X_src_adapted = np.dot(X_src, A_coral_local).real
            else:
                X_src_adapted = np.dot(X_src, coral_matrix).real
            mean_tgt = np.mean(X_tgt, axis=0)
            target_centered = X_tgt - mean_tgt
            U_tgt, _, _ = np.linalg.svd(np.cov(target_centered, rowvar=False))
            proj_matrix = U_tgt[:, :3]
            P_tgt = np.dot(X_tgt - mean_tgt, proj_matrix)
            P_src = np.dot(X_src - mean_tgt, proj_matrix)
            P_src_adapt = np.dot(X_src_adapted - mean_tgt, proj_matrix)
            title_before, title_after = "ANTES: Desalinhado (Source/Target)", "DEPOIS: CORAL (Alinhado)"
            data_pairs = [(P_src, P_tgt), (P_src_adapt, P_tgt)]
        else:
            raise ValueError("Mode inválido.")

        titles = [title_before, title_after]
        for i, (data_main, data_ref) in enumerate(data_pairs):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            if data_ref is not None:
                ax.scatter(data_ref[:,0], data_ref[:,1], data_ref[:,2], c='blue', s=2, alpha=0.1, label='Target')
                try:
                    w = self.get_cov_ellipsoid(data_ref)
                    if w is not None: ax.plot_wireframe(*w, color='blue', alpha=0.2, linewidth=0.5)
                except: pass
            color = 'red' if i == 0 and mode == 'coral' else 'green'
            if mode == 'whitening' and i == 0: color = 'navy'
            label_main = 'Source'
            ax.scatter(data_main[:,0], data_main[:,1], data_main[:,2], c=color, s=2, alpha=0.2, label=label_main)
            try:
                w = self.get_cov_ellipsoid(data_main)
                if w is not None: ax.plot_wireframe(*w, color=color, alpha=0.3, linewidth=0.8)
            except: pass
            ax.set_title(titles[i])
            ax.legend()
        plt.tight_layout()
        plt.show()

    
    # =========================================================================
    # 2. TRANSFORMAÇÃO CORAL (CORRELATION ALIGNMENT)
    # =========================================================================

    def calculate_coral_parameters(self,
                                   source_dataset: EmbeddingDataset,
                                   target_dataset: EmbeddingDataset,
                                   regularization: float = 1e-6,
                                   use_grid_search: bool = True,
                                   estimator: CoralEstimatorType = CoralEstimatorType.STANDARD) -> Tuple[np.ndarray, float]:
        """
        Calcula a matriz de transformação CORAL para alinhar o domínio Source ao Target.
        
        Mapeia a covariância (2º momento estatístico) do dataset de origem para se
        assemelhar à covariância do dataset de destino, minimizando o Domain Shift.
        
        Args:
            source_dataset: Dataset de Origem (A ser transformado).
            target_dataset: Dataset Alvo (O domínio esperado).
            regularization: Fator de Shrinkage padrão para evitar matrizes singulares.
            use_grid_search: Se True, busca o melhor fator de regularização usando ROC AUC.
            estimator: Enum (STANDARD, LEDOIT_WOLF, OAS) para matrizes robustas.
            
        Returns:
            Tuple[np.ndarray, float]: Matriz de projeção e a regularização final aplicada.
        """
        print(f"--- Calculando Parâmetros CORAL (Modo: {estimator.label}) ---")
        
        Xs = source_dataset.matrix
        Xt = target_dataset.matrix
        
        if Xs.size == 0 or Xt.size == 0:
            raise ValueError("Datasets vazios. Não é possível calcular CORAL.")

        Cs = None
        Ct = None
        final_reg_used = 0.0

        # --- CAMINHO 1: CORAL CLÁSSICO (Covariância Simples + Grid Search) ---
        if estimator == CoralEstimatorType.STANDARD:
            reg_to_use = regularization
            
            if use_grid_search:
                has_ids_src = len(source_dataset.identities) > 0
                has_ids_tgt = len(target_dataset.identities) > 0
                
                if not has_ids_src or not has_ids_tgt:
                    print("AVISO: Identidades não encontradas. Grid Search desativado.")
                else:
                    print("Iniciando Grid Search automático para CORAL Standard...")
                    try:
                        best_reg, best_auc = self._perform_grid_search(source_dataset, target_dataset)
                        print(f"-> Melhor Regularização: {best_reg:.1e} (AUC: {best_auc:.4f})")
                        reg_to_use = best_reg
                    except Exception as e:
                        print(f"AVISO: Falha no Grid Search ({e}). Usando padrão: {regularization}")

            d = Xs.shape[1]
            # Covariância Padrão + Regularização Diagonal Fixa
            Cs = np.cov(Xs, rowvar=False) + np.eye(d) * reg_to_use
            Ct = np.cov(Xt, rowvar=False) + np.eye(d) * reg_to_use
            final_reg_used = reg_to_use

        # --- CAMINHO 2: CORAL ROBUSTO (Ledoit-Wolf ou OAS) ---
        elif estimator in [CoralEstimatorType.LEDOIT_WOLF, CoralEstimatorType.OAS]:
            print(f"Usando estimador robusto {estimator.label} (Grid Search manual ignorado).")
            
            # Seleciona a Classe do Sklearn
            CovModel = LedoitWolf if estimator == CoralEstimatorType.LEDOIT_WOLF else OAS
            
            # O fit calcula a covariância com o shrinkage ideal automaticamente
            lw_src = CovModel(store_precision=False, assume_centered=False).fit(Xs)
            lw_tgt = CovModel(store_precision=False, assume_centered=False).fit(Xt)
            
            Cs = lw_src.covariance_
            Ct = lw_tgt.covariance_
            
            # Apenas para log/retorno (pega o shrinkage médio usado)
            final_reg_used = (lw_src.shrinkage_ + lw_tgt.shrinkage_) / 2
            print(f"-> Shrinkage Automático Médio aplicado: {final_reg_used:.4f}")

        else:
            raise ValueError(f"Estimador desconhecido: {estimator}")

        # --- CÁLCULO FINAL DA MATRIZ DE ROTAÇÃO ---
        try:
            Cs_sqrt_inv = inv(sqrtm(Cs))
            Ct_sqrt = sqrtm(Ct)
            coral_matrix = np.dot(Cs_sqrt_inv, Ct_sqrt)
        except np.linalg.LinAlgError:
            print("ERRO CRÍTICO: Matriz singular. Adicionando jitter de emergência.")
            Cs += np.eye(Cs.shape[0]) * 1e-5
            Ct += np.eye(Ct.shape[0]) * 1e-5
            Cs_sqrt_inv = inv(sqrtm(Cs))
            Ct_sqrt = sqrtm(Ct)
            coral_matrix = np.dot(Cs_sqrt_inv, Ct_sqrt)

        print("Parâmetros CORAL calculados com sucesso.")
        return coral_matrix, final_reg_used

    def _perform_grid_search(self, src_ds: EmbeddingDataset, tgt_ds: EmbeddingDataset) -> Tuple[float, float]:
        """
        Executa busca por força bruta (Grid Search) para achar a regularização ideal
        que maximize o AUC da curva ROC na comparação Source vs Target.
        """
        def build_id_map(ds: EmbeddingDataset):
            mapping = {}
            for item in ds:
                if item.identity and item.identity not in mapping:
                    mapping[item.identity] = item.embedding
            return mapping
            
        src_dict = build_id_map(src_ds)
        tgt_dict = build_id_map(tgt_ds)
        
        # Garante ordenação alfabética cruzada para reprodutibilidade no Grid Search!
        ids_comuns = sorted(list(set(src_dict.keys()) & set(tgt_dict.keys())))
        
        if len(ids_comuns) < 5:
            raise ValueError(f"IDs comuns insuficientes ({len(ids_comuns)}) para Grid Search.")
        if len(ids_comuns) > 200: 
            # Trava a semente no shuffle se precisar cortar
            if self.seed is not None:
                random.Random(self.seed).shuffle(ids_comuns)
            else:
                random.shuffle(ids_comuns)
            ids_comuns = ids_comuns[:200]
            
        vecs_src_gen, vecs_tgt_gen = [], []
        vecs_src_imp, vecs_tgt_imp = [], []
        for i, id_atual in enumerate(ids_comuns):
            emb_s = src_dict[id_atual]
            emb_t = tgt_dict[id_atual]
            vecs_src_gen.append(emb_s)
            vecs_tgt_gen.append(emb_t)
            id_imp = ids_comuns[(i + 1) % len(ids_comuns)]
            emb_t_imp = tgt_dict[id_imp]
            vecs_src_imp.append(emb_s)
            vecs_tgt_imp.append(emb_t_imp)
            
        S_gen = np.stack(vecs_src_gen); T_gen = np.stack(vecs_tgt_gen)
        S_imp = np.stack(vecs_src_imp); T_imp = np.stack(vecs_tgt_imp)
        candidates = [1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0]
        best_auc = -1.0
        best_reg = candidates[0] 
        success_flag = False 

        Ct_fixed = np.cov(np.concatenate([T_gen, T_imp]), rowvar=False)
        Cs_fixed = np.cov(np.concatenate([S_gen, S_imp]), rowvar=False)
        d = S_gen.shape[1]
        
        for reg in candidates:
            try:
                Cs_reg = Cs_fixed + np.eye(d) * reg
                Ct_reg = Ct_fixed + np.eye(d) * reg
                A = np.dot(inv(sqrtm(Cs_reg)), sqrtm(Ct_reg))
                if np.iscomplexobj(A): A = A.real
                S_gen_trans = np.dot(S_gen, A)
                S_imp_trans = np.dot(S_imp, A)
                
                def cos_sim(a, b):
                    na = np.linalg.norm(a, axis=1, keepdims=True) + 1e-10
                    nb = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
                    return np.sum((a / na) * (b / nb), axis=1)
                
                sims_gen = cos_sim(S_gen_trans, T_gen)
                sims_imp = cos_sim(S_imp_trans, T_imp)
                y_true = np.concatenate([np.ones(len(sims_gen)), np.zeros(len(sims_imp))])
                y_scores = np.concatenate([sims_gen, sims_imp])
                auc_val = roc_auc_score(y_true, y_scores)
                
                success_flag = True 
                if auc_val > best_auc:
                    best_auc = auc_val
                    best_reg = reg
            except Exception: 
                continue
                
        if not success_flag:
            raise ValueError("Todos os candidatos de regularização falharam na inversão de matriz.")
            
        return best_reg, best_auc

    def apply_coral_transformation(self, dataset: EmbeddingDataset, coral_matrix: np.ndarray) -> EmbeddingDataset:
        """
        Aplica a matriz CORAL calculada em um dataset gerando um novo.
        """
        if coral_matrix is None:
            raise ValueError("coral_matrix não pode ser None.")
        
        print(f"Aplicando CORAL em {dataset.name}...")
        
        new_items = []
        for item in dataset.items:
            new_emb = np.dot(item.embedding, coral_matrix)
            if np.iscomplexobj(new_emb): 
                new_emb = new_emb.real
            
            new_items.append(EmbeddingInfo(
                embedding=new_emb.astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores
            ))
            
        return EmbeddingDataset(
            name=f"{dataset.name}_CORAL",
            description=f"Adaptado via CORAL de {dataset.name}",
            items=new_items
        )

    # =========================================================================
    # 3. TRANSFORMAÇÃO WHITENING (PCA & ZCA)
    # =========================================================================

    def calculate_whitening_parameters(self, 
                                     dataset: EmbeddingDataset, 
                                     n_components: Optional[int] = None,
                                     style: WhiteningStyle = WhiteningStyle.PCA) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula os parâmetros (Média e Matriz de Projeção) para transformar 
        a matriz de covariância em uma esfera unitária (Descorrelação).
        
        Args:
            dataset: Base de dados de treinamento.
            n_components: Se fornecido, reduz a dimensionalidade retendo a maior variância.
            style: Enum (PCA_WHITENING ou ZCA_WHITENING).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Vetor médio e a matriz de transformação.
        """
        print(f"--- Calculando Whitening (Style: {style.value.upper()}, Components: {n_components if n_components else 'All'}) ---")
        X = dataset.matrix
        if X.size == 0: raise ValueError("Dataset vazio.")
        whitening_mean = np.mean(X, axis=0)
        X_centered = X - whitening_mean
        cov = np.cov(X_centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        S = eigvals[::-1]
        U = eigvecs[:, ::-1]
        S = np.maximum(S, 1e-10)
        
        if n_components is not None and n_components < len(S):
            S = S[:n_components]
            U = U[:, :n_components]
            
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S))
        
        if style == WhiteningStyle.PCA:
            whitening_matrix = np.dot(U, S_inv_sqrt)
        elif style == WhiteningStyle.ZCA:
            if n_components is not None and n_components < X.shape[1]:
                print("⚠️ Aviso: ZCA com redução de dimensionalidade perde informação espacial.")
            W_pca = np.dot(U, S_inv_sqrt)
            whitening_matrix = np.dot(W_pca, U.T)
        else:
            raise ValueError(f"Estilo desconhecido: {style}")
            
        print(f"Parâmetros Whitening ({style.value}) calculados.")
        return whitening_mean, whitening_matrix

    def apply_whitening_transformation(
        self,
        dataset: EmbeddingDataset,
        whitening_mean: np.ndarray,
        whitening_matrix: np.ndarray
    ) -> EmbeddingDataset:
        """
        Aplica a projeção de Whitening e gera um novo dataset.
        """
        if whitening_mean is None or whitening_matrix is None:
            raise ValueError("Parâmetros de Whitening obrigatórios.")

        print(f"Aplicando Whitening em {dataset.name}...")

        new_items = []
        for item in dataset.items:
            vec_centered = item.embedding - whitening_mean
            vec_white = np.dot(vec_centered, whitening_matrix)

            new_items.append(EmbeddingInfo(
                embedding=vec_white.astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores
            ))

        return EmbeddingDataset(
            name=f"{dataset.name}_White",
            description=f"Whitening de {dataset.name}",
            items=new_items
        )


    # =========================================================================
    # 4. TRANSFORMAÇÃO LDA (LINEAR DISCRIMINANT ANALYSIS)
    # =========================================================================

    def calculate_lda_parameters(
        self,
        dataset: EmbeddingDataset,
        config: "LDAConfig",
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Calcula os eixos discriminantes de Fisher (LDA).

        Sequência:
          1. Agrupa amostras por identidade; valida ≥ 2 classes com ≥ 2 amostras.
          2. Se config.use_grid_search=True, busca o melhor ε via holdout
             por identidade usando a métrica configurada.
          3. Refita (mean, matrix) no dataset completo com ε escolhido,
             estimator e shrinkage da config.
          4. Retorna (mean_vector, lda_matrix, metrics_dict).

        Args:
            dataset: Dataset rotulado (cada EmbeddingInfo precisa de `identity`).
            config: LDAConfig com n_components/regularization/shrinkage/
                estimator/grid search.

        Returns:
            Tuple (mean_vector, lda_matrix, metrics_dict).
            - mean_vector: ``(D,)`` — centro global usado em apply.
            - lda_matrix: ``(D, n_components)`` — eixos discriminantes.
            - metrics_dict: estatísticas de auditoria (estimator usado, ε
              final, condicionamento, etc.).
        """
        if dataset.matrix.size == 0:
            raise ValueError("[LDA] Dataset vazio.")

        X = dataset.matrix
        identities = [item.identity for item in dataset.items]
        if any(ident is None or ident == "" for ident in identities):
            raise ValueError(
                "[LDA] Todos os EmbeddingInfo precisam ter 'identity' populado."
            )

        ids_array = np.asarray(identities)
        classes, counts = np.unique(ids_array, return_counts=True)
        n_singletons = int(np.sum(counts == 1))
        valid_classes = classes[counts >= 2]
        n_classes_used = int(len(valid_classes))

        if n_classes_used < 2:
            raise ValueError(
                f"[LDA] São necessárias ≥ 2 identidades com ≥ 2 amostras. "
                f"Encontrado: {n_classes_used}."
            )

        print(
            f"--- Calculando Parâmetros LDA (Estimador: {config.estimator.label}, "
            f"GridSearch: {config.use_grid_search}) ---"
        )
        print(
            f"   -> {n_classes_used} identidades válidas, "
            f"{n_singletons} singletons descartados."
        )

        grid_search_scores: Dict[float, float] = {}
        best_score: Optional[float] = None
        final_eps = float(config.regularization)

        if config.use_grid_search:
            best_eps, best_score, grid_search_scores = self._perform_lda_grid_search(
                dataset=dataset,
                config=config,
                valid_classes=valid_classes,
            )
            final_eps = best_eps
            print(
                f"   -> Grid search escolheu ε={final_eps:.1e} "
                f"({config.grid_search_metric.label}={best_score:.4f})"
            )

        mean_vector, lda_matrix, metrics_fit = self._lda_fit_single_eps(
            X=X,
            identities=ids_array,
            config=config,
            eps=final_eps,
            valid_classes=valid_classes,
        )

        metrics: Dict[str, Any] = {
            "step_type": "LDA",
            "n_classes_used": n_classes_used,
            "n_singletons_skipped": n_singletons,
            "input_dimension": int(X.shape[1]),
            "output_dimension": int(lda_matrix.shape[1]),
            "estimator_used": config.estimator.value,
            "final_regularization": float(final_eps),
            "final_shrinkage": float(config.shrinkage),
            "grid_search_used": bool(config.use_grid_search),
            "grid_search_metric": (
                config.grid_search_metric.value if config.use_grid_search else None
            ),
            "grid_search_scores": (
                {f"{eps:.3e}": float(score) for eps, score in grid_search_scores.items()}
                if config.use_grid_search else {}
            ),
            "grid_search_best_eps": (
                float(final_eps) if config.use_grid_search else None
            ),
            "grid_search_best_score": (
                float(best_score) if best_score is not None else None
            ),
        }
        metrics.update(metrics_fit)
        return mean_vector, lda_matrix, metrics

    def _lda_fit_single_eps(
        self,
        X: np.ndarray,
        identities: np.ndarray,
        config: "LDAConfig",
        eps: float,
        valid_classes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit do LDA para um ε específico. Helper compartilhado entre o fit
        normal e o grid search.

        Retorna (mean_vector, lda_matrix, fit_metrics).
        """
        from faceaudit.enums.lda_estimator_type import LdaEstimatorType
        if valid_classes is None:
            classes, counts = np.unique(identities, return_counts=True)
            valid_classes = classes[counts >= 2]

        d = int(X.shape[1])
        global_mean = X.mean(axis=0)

        # --- Sw via estimator escolhido ---
        if config.estimator == LdaEstimatorType.STANDARD:
            # Soma de outer products (Fisher LDA clássico).
            Sw = np.zeros((d, d), dtype=np.float64)
            for cls in valid_classes:
                Xc = X[identities == cls]
                mean_c = Xc.mean(axis=0)
                centered = Xc - mean_c
                Sw += centered.T @ centered
        elif config.estimator in (LdaEstimatorType.LEDOIT_WOLF, LdaEstimatorType.OAS):
            # Concatena vetores centrados por classe e aplica estimador
            # robusto sobre o conjunto centralizado intraclasse. Escala
            # pelo número de amostras pra ficar compatível com a soma de
            # outer products (mantém o critério Fisher sensível a tamanho).
            centered_chunks = []
            for cls in valid_classes:
                Xc = X[identities == cls]
                centered_chunks.append(Xc - Xc.mean(axis=0))
            X_centered_all = np.concatenate(centered_chunks, axis=0)
            n_centered = X_centered_all.shape[0]
            CovModel = LedoitWolf if config.estimator == LdaEstimatorType.LEDOIT_WOLF else OAS
            cov_model = CovModel(store_precision=False, assume_centered=True).fit(X_centered_all)
            Sw = cov_model.covariance_ * n_centered
        else:
            raise ValueError(f"Estimador LDA desconhecido: {config.estimator}")

        within_trace_before = float(np.trace(Sw))

        # --- Shrinkage explícito (combina com qualquer estimator) ---
        if config.shrinkage > 0.0:
            Sw = (1.0 - config.shrinkage) * Sw + config.shrinkage * np.eye(d)

        # --- Sb (between-class scatter) — sempre standard ---
        Sb = np.zeros((d, d), dtype=np.float64)
        for cls in valid_classes:
            Xc = X[identities == cls]
            n_c = Xc.shape[0]
            mean_diff = (Xc.mean(axis=0) - global_mean).reshape(-1, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)

        # --- Ridge final no Sw + eig generalizado ---
        eye_d = np.eye(d)
        eps_current = eps
        for attempt in range(3):
            try:
                Sw_reg = Sw + eps_current * eye_d
                eigvals, eigvecs = linalg.eigh(Sb, Sw_reg)
                break
            except (np.linalg.LinAlgError, ValueError):
                eps_current *= 100.0
                if attempt == 2:
                    raise RuntimeError(
                        f"[LDA] Falha persistente no eigh generalizado mesmo após "
                        f"escalar ε para {eps_current:.1e}."
                    )

        # Ordena por autovalor descendente (maior separabilidade primeiro)
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, sorted_indices]

        max_valid = len(valid_classes) - 1
        n_components = config.n_components
        if n_components is None or n_components > max_valid:
            n_components = max_valid

        lda_matrix = eigvecs[:, :n_components]

        # Métricas de auditoria
        try:
            cond = float(np.linalg.cond(Sw_reg))
        except Exception:
            cond = float("nan")
        try:
            # Fisher ratio antes/depois (sob a projeção)
            tr_sw = float(np.trace(Sw))
            tr_sb = float(np.trace(Sb))
            fisher_before = tr_sb / max(tr_sw, 1e-12)
            Sb_proj = lda_matrix.T @ Sb @ lda_matrix
            Sw_proj = lda_matrix.T @ Sw @ lda_matrix
            fisher_after = float(np.trace(Sb_proj)) / max(float(np.trace(Sw_proj)), 1e-12)
        except Exception:
            fisher_before = float("nan")
            fisher_after = float("nan")

        return global_mean, lda_matrix, {
            "within_class_trace_before": within_trace_before,
            "within_class_trace_after": float(np.trace(Sw)) if config.shrinkage == 0 else float("nan"),
            "fisher_ratio_before": float(fisher_before),
            "fisher_ratio_after": float(fisher_after),
            "condition_number": cond,
        }

    def _perform_lda_grid_search(
        self,
        dataset: EmbeddingDataset,
        config: "LDAConfig",
        valid_classes: np.ndarray,
    ) -> Tuple[float, float, Dict[float, float]]:
        """
        Busca o melhor ε em config.reg_grid usando holdout por identidade.
        Mesma estrutura do `_perform_wccn_grid_search` (split estratificado
        por ID, refit per ε, score holdout, escolhe maior score).
        """
        rng = random.Random(self.seed) if self.seed is not None else random.Random()

        all_ids = sorted([str(c) for c in valid_classes])
        rng.shuffle(all_ids)
        n_holdout = max(1, int(round(len(all_ids) * config.grid_search_holdout_ratio)))
        holdout_ids = set(all_ids[:n_holdout])
        train_ids = set(all_ids[n_holdout:])

        if len(train_ids) < 2:
            raise ValueError(
                f"[LDA] Grid search precisa de ≥ 2 identidades de treino após holdout. "
                f"Total={len(all_ids)}, holdout={n_holdout}, treino={len(train_ids)}."
            )

        X = dataset.matrix
        identities = np.asarray([item.identity for item in dataset.items])
        train_mask = np.isin(identities, list(train_ids))
        holdout_mask = np.isin(identities, list(holdout_ids))
        X_train = X[train_mask]
        ids_train = identities[train_mask]
        X_holdout = X[holdout_mask]
        ids_holdout = identities[holdout_mask]

        ho_classes, ho_counts = np.unique(ids_holdout, return_counts=True)
        ho_valid = ho_classes[ho_counts >= 2]
        if len(ho_valid) < 5:
            raise ValueError(
                f"[LDA] Holdout insuficiente para grid search: precisa de ≥ 5 "
                f"identidades com ≥ 2 amostras, encontrado {len(ho_valid)}."
            )

        train_classes, train_counts = np.unique(ids_train, return_counts=True)
        train_valid = train_classes[train_counts >= 2]
        if len(train_valid) < 2:
            raise ValueError(
                f"[LDA] Treino do grid search precisa de ≥ 2 identidades com "
                f"≥ 2 amostras, encontrado {len(train_valid)}."
            )

        scores: Dict[float, float] = {}
        for eps in config.reg_grid:
            try:
                mean_eps, matrix_eps, _ = self._lda_fit_single_eps(
                    X=X_train,
                    identities=ids_train,
                    config=config,
                    eps=float(eps),
                    valid_classes=train_valid,
                )
                # Projeta holdout pelo modelo treinado
                X_holdout_proj = (X_holdout - mean_eps) @ matrix_eps
                score = self._score_lda_holdout(
                    X_holdout_proj=X_holdout_proj,
                    ids_holdout=ids_holdout,
                    valid_classes=ho_valid,
                    metric=config.grid_search_metric,
                    rng=rng,
                )
                scores[float(eps)] = float(score)
            except Exception as e:
                print(f"   [grid] ε={eps:.1e} falhou: {e}")
                scores[float(eps)] = float("-inf")

        finite_scores = {k: v for k, v in scores.items() if v != float("-inf") and not np.isnan(v)}
        if not finite_scores:
            raise RuntimeError(
                "[LDA] Grid search falhou para todos os valores de ε avaliados."
            )
        max_score = max(finite_scores.values())
        best_eps = min(eps for eps, s in finite_scores.items() if s == max_score)
        return float(best_eps), float(max_score), scores

    def _score_lda_holdout(
        self,
        X_holdout_proj: np.ndarray,
        ids_holdout: np.ndarray,
        valid_classes: np.ndarray,
        metric: "LdaGridSearchMetric",
        rng: random.Random,
    ) -> float:
        """
        Score do holdout JÁ PROJETADO segundo a métrica escolhida.

        FISHER_RATIO: trace(Sb_proj)/trace(Sw_proj) — barato, sem pares.
        COSINE_AUC: AUC de cossenos em pares genuíno/impostor (≤ 200 ids).
        """
        from faceaudit.enums.lda_grid_search_metric import LdaGridSearchMetric

        if metric == LdaGridSearchMetric.FISHER_RATIO:
            d = X_holdout_proj.shape[1]
            global_mean = X_holdout_proj.mean(axis=0)
            Sw = np.zeros((d, d), dtype=np.float64)
            Sb = np.zeros((d, d), dtype=np.float64)
            for cls in valid_classes:
                Xc = X_holdout_proj[ids_holdout == cls]
                mean_c = Xc.mean(axis=0)
                centered = Xc - mean_c
                Sw += centered.T @ centered
                diff = (mean_c - global_mean).reshape(-1, 1)
                Sb += Xc.shape[0] * (diff @ diff.T)
            tr_sw = float(np.trace(Sw))
            tr_sb = float(np.trace(Sb))
            return tr_sb / (tr_sw + 1e-12)

        if metric == LdaGridSearchMetric.COSINE_AUC:
            ids = list(valid_classes)
            if len(ids) > 200:
                rng_local = random.Random(self.seed) if self.seed is not None else rng
                rng_local.shuffle(ids)
                ids = sorted(ids[:200])

            id_to_idx: Dict[str, list] = {
                cls: list(np.where(ids_holdout == cls)[0]) for cls in ids
            }
            gen_a, gen_b, imp_a, imp_b = [], [], [], []
            for i, cls in enumerate(ids):
                idxs = id_to_idx[cls]
                if len(idxs) < 2:
                    continue
                gen_a.append(idxs[0])
                gen_b.append(idxs[1])
                other_cls = ids[(i + 1) % len(ids)]
                other_idxs = id_to_idx[other_cls]
                if not other_idxs:
                    continue
                imp_a.append(idxs[0])
                imp_b.append(other_idxs[0])

            if not gen_a or not imp_a:
                return float("-inf")

            def cos_sim_pairs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                na = np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
                nb = np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
                return np.sum((A / na) * (B / nb), axis=1)

            sims_gen = cos_sim_pairs(X_holdout_proj[gen_a], X_holdout_proj[gen_b])
            sims_imp = cos_sim_pairs(X_holdout_proj[imp_a], X_holdout_proj[imp_b])
            y_true = np.concatenate([np.ones(len(sims_gen)), np.zeros(len(sims_imp))])
            y_scores = np.concatenate([sims_gen, sims_imp])
            try:
                return float(roc_auc_score(y_true, y_scores))
            except ValueError:
                return float("-inf")

        raise ValueError(f"Métrica de grid search LDA desconhecida: {metric}")

    def apply_lda_transformation(self, dataset: EmbeddingDataset, mean_vector: np.ndarray, lda_matrix: np.ndarray) -> EmbeddingDataset:
        """Aplica a matriz de projeção LDA a um dataset."""
        if mean_vector is None or lda_matrix is None:
            raise ValueError("Parâmetros LDA não fornecidos.")
            
        print(f"Aplicando LDA em {dataset.name}...")
        X_centered = dataset.matrix - mean_vector
        X_lda = np.dot(X_centered, lda_matrix)
        
        new_items = []
        for i, item in enumerate(dataset.items):
            new_items.append(EmbeddingInfo(
                embedding=X_lda[i].astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores
            ))
            
        return EmbeddingDataset(
            name=f"{dataset.name}_LDA",
            description=f"Projeção LDA ({lda_matrix.shape[1]}D)",
            items=new_items
        )

    # =========================================================================
    # 4b. TRANSFORMAÇÃO WCCN (Within-Class Covariance Normalization)
    # =========================================================================

    def calculate_wccn_parameters(
        self,
        dataset: EmbeddingDataset,
        config: WCCNConfig,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Calcula a matriz WCCN (B) para normalizar a covariância intraclasse.

        Sequência:
          1. Agrupa amostras por identidade; descarta singletons.
          2. Se config.use_grid_search=True, busca o melhor ε via holdout.
          3. Refita W no dataset completo com ε escolhido.
          4. Retorna (B, metrics_dict).

        Args:
            dataset: Dataset rotulado (cada EmbeddingInfo precisa de `identity`).
            config: WCCNConfig com regularization/shrinkage/estimator/grid search.

        Returns:
            Tuple (B_matrix, metrics_dict). B é D×D; aplicação: x' = x @ B.
        """
        if dataset.matrix.size == 0:
            raise ValueError("[WCCN] Dataset vazio.")

        X = dataset.matrix
        identities = [item.identity for item in dataset.items]
        if any(ident is None or ident == "" for ident in identities):
            raise ValueError(
                "[WCCN] Todos os EmbeddingInfo precisam ter 'identity' populado. "
                "WCCN é supervisionado."
            )

        ids_array = np.asarray(identities)
        classes, counts = np.unique(ids_array, return_counts=True)
        n_singletons = int(np.sum(counts == 1))
        valid_classes = classes[counts >= 2]
        n_classes_used = int(len(valid_classes))

        if n_classes_used < 2:
            raise ValueError(
                f"[WCCN] São necessárias ≥ 2 identidades com ≥ 2 amostras. "
                f"Encontrado: {n_classes_used}."
            )

        print(
            f"--- Calculando Parâmetros WCCN (Estimador: {config.estimator.label}, "
            f"GridSearch: {config.use_grid_search}) ---"
        )
        print(
            f"   -> {n_classes_used} identidades válidas, "
            f"{n_singletons} singletons descartados."
        )

        grid_search_scores: Dict[float, float] = {}
        best_score: Optional[float] = None
        final_eps = float(config.regularization)

        if config.use_grid_search:
            best_eps, best_score, grid_search_scores = self._perform_wccn_grid_search(
                dataset=dataset,
                config=config,
                valid_classes=valid_classes,
            )
            final_eps = best_eps
            print(
                f"   -> Grid search escolheu ε={final_eps:.1e} "
                f"({config.grid_search_metric.label}={best_score:.4f})"
            )

        B, metrics_fit = self._wccn_fit_single_eps(
            X=X,
            identities=ids_array,
            config=config,
            eps=final_eps,
            valid_classes=valid_classes,
        )

        metrics: Dict[str, Any] = {
            "step_type": StepType.WCCN.value,
            "n_classes_used": n_classes_used,
            "n_singletons_skipped": n_singletons,
            "input_dimension": int(X.shape[1]),
            "output_dimension": int(B.shape[0]),
            "estimator_used": config.estimator.value,
            "final_regularization": float(final_eps),
            "final_shrinkage": float(config.shrinkage),
            "grid_search_used": bool(config.use_grid_search),
            "grid_search_metric": (
                config.grid_search_metric.value if config.use_grid_search else None
            ),
            "grid_search_scores": (
                {f"{eps:.3e}": float(score) for eps, score in grid_search_scores.items()}
                if config.use_grid_search else {}
            ),
            "grid_search_best_eps": (
                float(final_eps) if config.use_grid_search else None
            ),
            "grid_search_best_score": (
                float(best_score) if best_score is not None else None
            ),
        }
        metrics.update(metrics_fit)
        return B, metrics

    def _wccn_fit_single_eps(
        self,
        X: np.ndarray,
        identities: np.ndarray,
        config: WCCNConfig,
        eps: float,
        valid_classes: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit do WCCN para um ε específico. Helper compartilhado entre o fit
        normal e o grid search.

        Retorna (B, fit_metrics) onde B = chol(inv(W_reg)).
        """
        if valid_classes is None:
            classes, counts = np.unique(identities, return_counts=True)
            valid_classes = classes[counts >= 2]

        d = int(X.shape[1])

        # Computa W segundo o estimador escolhido
        if config.estimator == WccnEstimatorType.STANDARD:
            # Média das covariâncias por classe (Hatch et al. 2006)
            W = np.zeros((d, d), dtype=np.float64)
            K = 0
            for cls in valid_classes:
                X_c = X[identities == cls]
                mean_c = X_c.mean(axis=0)
                centered = X_c - mean_c
                cov_c = (centered.T @ centered) / centered.shape[0]
                W += cov_c
                K += 1
            W /= max(K, 1)

        elif config.estimator in (WccnEstimatorType.LEDOIT_WOLF, WccnEstimatorType.OAS):
            # Concatena todos os vetores centrados por classe e aplica
            # o estimador robusto sobre o conjunto centralizado intraclasse.
            centered_chunks = []
            for cls in valid_classes:
                X_c = X[identities == cls]
                centered_chunks.append(X_c - X_c.mean(axis=0))
            X_centered_all = np.concatenate(centered_chunks, axis=0)
            CovModel = LedoitWolf if config.estimator == WccnEstimatorType.LEDOIT_WOLF else OAS
            cov_model = CovModel(store_precision=False, assume_centered=True).fit(X_centered_all)
            W = cov_model.covariance_
        else:
            raise ValueError(f"Estimador WCCN desconhecido: {config.estimator}")

        within_trace_before = float(np.trace(W))

        # Shrinkage explícito (combina com ridge)
        if config.shrinkage > 0.0:
            W = (1.0 - config.shrinkage) * W + config.shrinkage * np.eye(d)

        # Ridge final
        eye_d = np.eye(d)
        W_reg = W + eps * eye_d

        # Inversa via Cholesky de W_reg + chol(inv(W_reg)).
        # Fallback aumenta ε se a primeira tentativa falhar.
        eps_current = eps
        for attempt in range(3):
            try:
                W_inv = np.linalg.inv(W_reg)
                # B = chol(W_inv) tal que W_inv = B @ B.T
                B = np.linalg.cholesky(W_inv)
                break
            except np.linalg.LinAlgError:
                eps_current *= 100.0
                W_reg = W + eps_current * eye_d
                if attempt == 2:
                    raise RuntimeError(
                        f"[WCCN] Falha persistente na inversão de W mesmo "
                        f"após escalar ε para {eps_current:.1e}."
                    )

        # Métrica de condicionamento
        try:
            cond = float(np.linalg.cond(W_reg))
        except Exception:
            cond = float("nan")

        # Fisher ratio antes/depois: usa Sb global como referência
        # (não é o critério de seleção do grid search; serve para auditoria).
        try:
            global_mean = X.mean(axis=0)
            Sb = np.zeros((d, d), dtype=np.float64)
            for cls in valid_classes:
                X_c = X[identities == cls]
                diff = (X_c.mean(axis=0) - global_mean).reshape(-1, 1)
                Sb += X_c.shape[0] * (diff @ diff.T)
            Sb_trace = float(np.trace(Sb))
            fisher_before = Sb_trace / max(within_trace_before, 1e-12)

            # Após WCCN, W' = B.T @ W @ B = I (idealmente)
            W_after = B.T @ W @ B
            Sb_after = B.T @ Sb @ B
            fisher_after = float(np.trace(Sb_after)) / max(float(np.trace(W_after)), 1e-12)
        except Exception:
            fisher_before = float("nan")
            fisher_after = float("nan")

        return B, {
            "within_class_trace_before": within_trace_before,
            "within_class_trace_after": float(np.trace(B.T @ W @ B)),
            "fisher_ratio_before": float(fisher_before),
            "fisher_ratio_after": float(fisher_after),
            "condition_number": cond,
        }

    def _perform_wccn_grid_search(
        self,
        dataset: EmbeddingDataset,
        config: WCCNConfig,
        valid_classes: np.ndarray,
    ) -> Tuple[float, float, Dict[float, float]]:
        """
        Busca o melhor ε em config.reg_grid usando holdout por identidade.

        Métrica: FISHER_RATIO (default) ou COSINE_AUC.
        Retorna (best_eps, best_score, scores_dict).
        """
        rng = random.Random(self.seed) if self.seed is not None else random.Random()

        # Split por identidade
        all_ids = sorted([str(c) for c in valid_classes])
        rng.shuffle(all_ids)
        n_holdout = max(1, int(round(len(all_ids) * config.grid_search_holdout_ratio)))
        holdout_ids = set(all_ids[:n_holdout])
        train_ids = set(all_ids[n_holdout:])

        if len(train_ids) < 2:
            raise ValueError(
                f"[WCCN] Grid search precisa de ≥ 2 identidades de treino após holdout. "
                f"Total={len(all_ids)}, holdout={n_holdout}, treino={len(train_ids)}."
            )

        X = dataset.matrix
        identities = np.asarray([item.identity for item in dataset.items])

        train_mask = np.isin(identities, list(train_ids))
        holdout_mask = np.isin(identities, list(holdout_ids))
        X_train = X[train_mask]
        ids_train = identities[train_mask]
        X_holdout = X[holdout_mask]
        ids_holdout = identities[holdout_mask]

        # Pré-requisito do holdout: ≥ 5 identidades com ≥ 2 amostras
        ho_classes, ho_counts = np.unique(ids_holdout, return_counts=True)
        ho_valid = ho_classes[ho_counts >= 2]
        if len(ho_valid) < 5:
            raise ValueError(
                f"[WCCN] Holdout insuficiente para grid search: precisa de ≥ 5 "
                f"identidades com ≥ 2 amostras, encontrado {len(ho_valid)}. "
                f"Reduza holdout_ratio ou adicione mais identidades ao dataset."
            )

        # Pré-computa classes válidas no treino
        train_classes, train_counts = np.unique(ids_train, return_counts=True)
        train_valid = train_classes[train_counts >= 2]
        if len(train_valid) < 2:
            raise ValueError(
                f"[WCCN] Treino do grid search precisa de ≥ 2 identidades com "
                f"≥ 2 amostras, encontrado {len(train_valid)}."
            )

        scores: Dict[float, float] = {}
        for eps in config.reg_grid:
            try:
                B_eps, _ = self._wccn_fit_single_eps(
                    X=X_train,
                    identities=ids_train,
                    config=config,
                    eps=float(eps),
                    valid_classes=train_valid,
                )
                X_holdout_trans = X_holdout @ B_eps
                score = self._score_wccn_holdout(
                    X_holdout=X_holdout_trans,
                    ids_holdout=ids_holdout,
                    valid_classes=ho_valid,
                    metric=config.grid_search_metric,
                    rng=rng,
                )
                scores[float(eps)] = float(score)
            except Exception as e:
                print(f"   [grid] ε={eps:.1e} falhou: {e}")
                scores[float(eps)] = float("-inf")

        # Seleção: maior score; em empate, menor ε (mais conservador)
        finite_scores = {k: v for k, v in scores.items() if v != float("-inf") and not np.isnan(v)}
        if not finite_scores:
            raise RuntimeError(
                "[WCCN] Grid search falhou para todos os valores de ε avaliados. "
                "Verifique a integridade do dataset ou amplie reg_grid."
            )
        max_score = max(finite_scores.values())
        best_eps = min(eps for eps, s in finite_scores.items() if s == max_score)
        return float(best_eps), float(max_score), scores

    def _score_wccn_holdout(
        self,
        X_holdout: np.ndarray,
        ids_holdout: np.ndarray,
        valid_classes: np.ndarray,
        metric: WccnGridSearchMetric,
        rng: random.Random,
    ) -> float:
        """
        Computa o score do holdout transformado segundo a métrica escolhida.
        """
        if metric == WccnGridSearchMetric.FISHER_RATIO:
            d = X_holdout.shape[1]
            global_mean = X_holdout.mean(axis=0)
            Sw = np.zeros((d, d), dtype=np.float64)
            Sb = np.zeros((d, d), dtype=np.float64)
            for cls in valid_classes:
                Xc = X_holdout[ids_holdout == cls]
                mean_c = Xc.mean(axis=0)
                centered = Xc - mean_c
                Sw += centered.T @ centered
                diff = (mean_c - global_mean).reshape(-1, 1)
                Sb += Xc.shape[0] * (diff @ diff.T)
            tr_sw = float(np.trace(Sw))
            tr_sb = float(np.trace(Sb))
            return tr_sb / (tr_sw + 1e-12)

        if metric == WccnGridSearchMetric.COSINE_AUC:
            # Limita a 200 identidades para evitar O(N²) explosivo
            ids = list(valid_classes)
            if len(ids) > 200:
                rng_local = random.Random(self.seed) if self.seed is not None else rng
                rng_local.shuffle(ids)
                ids = sorted(ids[:200])

            # Constrói pares: 1 genuíno + 1 impostor por identidade
            id_to_idx: Dict[str, list] = {cls: list(np.where(ids_holdout == cls)[0]) for cls in ids}
            gen_a, gen_b, imp_a, imp_b = [], [], [], []
            for i, cls in enumerate(ids):
                idxs = id_to_idx[cls]
                if len(idxs) < 2:
                    continue
                # Genuíno: dois índices distintos
                a, b = idxs[0], idxs[1]
                gen_a.append(a)
                gen_b.append(b)
                # Impostor: amostra desse cls vs amostra da próxima id no ciclo
                other_cls = ids[(i + 1) % len(ids)]
                other_idxs = id_to_idx[other_cls]
                if not other_idxs:
                    continue
                imp_a.append(a)
                imp_b.append(other_idxs[0])

            if not gen_a or not imp_a:
                return float("-inf")

            def cos_sim_pairs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                na = np.linalg.norm(A, axis=1, keepdims=True) + 1e-10
                nb = np.linalg.norm(B, axis=1, keepdims=True) + 1e-10
                return np.sum((A / na) * (B / nb), axis=1)

            sims_gen = cos_sim_pairs(X_holdout[gen_a], X_holdout[gen_b])
            sims_imp = cos_sim_pairs(X_holdout[imp_a], X_holdout[imp_b])
            y_true = np.concatenate([np.ones(len(sims_gen)), np.zeros(len(sims_imp))])
            y_scores = np.concatenate([sims_gen, sims_imp])
            try:
                return float(roc_auc_score(y_true, y_scores))
            except ValueError:
                return float("-inf")

        raise ValueError(f"Métrica de grid search desconhecida: {metric}")

    def apply_wccn_transformation(
        self,
        dataset: EmbeddingDataset,
        wccn_matrix: np.ndarray,
    ) -> EmbeddingDataset:
        """
        Aplica a matriz WCCN B em um dataset, gerando um novo dataset
        com x' = x @ B em cada embedding.
        """
        if wccn_matrix is None:
            raise ValueError("wccn_matrix não pode ser None.")

        print(f"Aplicando WCCN em {dataset.name}...")

        new_items = []
        for item in dataset.items:
            new_emb = (item.embedding @ wccn_matrix).astype(np.float32)
            new_items.append(EmbeddingInfo(
                embedding=new_emb,
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                pose=dict(item.pose) if item.pose else None,
                quality_scores=dict(item.quality_scores) if item.quality_scores else None,
            ))

        return EmbeddingDataset(
            name=f"{dataset.name}_WCCN",
            description=f"Adaptado via WCCN de {dataset.name}",
            items=new_items,
        )

    # =========================================================================
    # 5. UTILITÁRIOS: CENTERING, L2, AGGREGATION
    # =========================================================================

    def calculate_centering_parameters(self, dataset: EmbeddingDataset) -> np.ndarray:
        """Calcula a média global para subtrair e mover o espaço latente para o eixo (0,0)."""
        print(f"--- Calculando Parâmetros Centering ({dataset.name}) ---")
        if dataset.matrix.size == 0: raise ValueError("Dataset vazio.")
        return np.mean(dataset.matrix, axis=0)

    def apply_centering_transformation(self, dataset: EmbeddingDataset, mean_vector: np.ndarray) -> EmbeddingDataset:
        """Centraliza o dataset dado um vetor média base."""
        if mean_vector is None: raise ValueError("mean_vector é obrigatório.")
        print(f"Aplicando Centering em {dataset.name}...")
        new_matrix = dataset.matrix - mean_vector
        new_items = []
        for i, item in enumerate(dataset.items):
            new_items.append(EmbeddingInfo(
                embedding=new_matrix[i].astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores
            ))
        return EmbeddingDataset(
            name=f"{dataset.name}_Centered",
            description="Centralizado pela média",
            items=new_items
        )

    def apply_l2_normalization(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """Garante que a norma de todos os vetores seja igual a 1 (L2-Norm)."""
        X = dataset.matrix.astype(np.float64)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X / norms
        new_items = []
        for i, item in enumerate(dataset.items):
            new_items.append(EmbeddingInfo(
                embedding=X_norm[i].astype(np.float32),
                face_hash=item.face_hash,
                image_path=item.image_path,
                identity=item.identity,
                file_name=item.file_name,
                source=item.source,
                origin=item.origin,
                face_image=item.face_image,
                yaw=item.yaw,
                pitch=item.pitch,
                quality_scores=item.quality_scores
            ))
        return EmbeddingDataset(
            name=f"{dataset.name}_L2Norm",
            description="Normalização Unitária (L2)",
            items=new_items
        )

    # `apply_identity_aggregation` foi REMOVIDA. Era agregação legada com zero
    # chamadores em toda a árvore, e não era sequer executável: construía
    # `EmbeddingInfo` sem `image_path`, que é parâmetro obrigatório — qualquer
    # chamada estourava `TypeError`. Pior que morta, documentava a convenção
    # ERRADA ao lado da certa: herdava o `face_hash` de uma das fotos reais,
    # fazendo um template sintético se passar por imagem na detecção de
    # vazamento probe↔referência (`comparison_service.py`), que compara
    # conjuntos de hash. Herdar o hash faria essa checagem pegar no máximo UMA
    # das k fotos fundidas e perder as demais — garantia falsa. Use
    # `apply_smart_aggregation` com `AggregationMethod.CENTROID`, que é o
    # mesmo resultado matemático com identidade de linha correta.

    # =========================================================================
    # 6. UTILITÁRIOS: VISUALIZAÇÃO E PROJEÇÃO (MÁQUINA DE ESTADOS ORIENTADA A DOMÍNIO)
    # =========================================================================

    def compute_trajectory_projection(self, raw_frames: List[Any]) -> TransformationTrajectoryData:
        """
        Calcula a projeção 3D da trajetória de transformações biométricas.
        
        Utiliza uma abordagem de Joint PCA (PCA Conjunto) para garantir que 
        Source e Target sejam projetados sob a mesma "lente" matemática em 
        cada passo, preservando as distâncias e escalas relativas. Em seguida, 
        aplica a estabilização de Procrustes para suavizar a transição visual 
        da câmera entre os frames.
        
        Implementa a Máquina de Estados de Orientação Orientada a Domínio:
        As regras de travar/destravar os eixos visuais (wireframes) não são mais 
        "hardcoded". O serviço interroga o Enum `TransformationType` via propriedade 
        `requires_axis_recalculation`. Transformações que alteram severamente o subespaço 
        (como CORAL ou LDA) destravam e recalculam os eixos. Transformações isotrópicas 
        (como Whitening ou L2) mantêm os eixos travados, prevenindo o "Paradoxo da Esfera".

        Args:
            raw_frames (List[Any]): Lista de frames (SimpleNamespace) sincronizados 
                                    gerados pelo TransformationTrajectoryController.

        Returns:
            TransformationTrajectoryData: Objeto empacotando todos os passos 3D prontos 
                                          para renderização pela UI.
        """
        print("[MathService] Calculando projeção 3D (Máquina de Estados Orientada a Domínio)...")

        if not raw_frames:
            raise ValueError("Lista de frames vazia.")

        from faceaudit.enums.transformation_type import TransformationType
        from faceaudit.enums.step_type import StepType

        steps_3d = []
        max_fit_samples = 10000 

        # --- ESTADOS DE CONTINUIDADE (CÂMERA) ---
        active_pca = None
        active_dim = -1 
        prev_src_3d_points = None
        
        # --- MÁQUINA DE ESTADOS DE ORIENTAÇÃO (MEMÓRIAS INDEPENDENTES) ---
        target_locked_orientation = None
        source_locked_orientation = None

        for frame in raw_frames:
            # --- 1. PREPARAÇÃO E INJEÇÃO DE METADADOS ---
            src_mat = frame.source_matrix
            tgt_mat = frame.target_matrix
            metrics = getattr(frame, 'metrics', {})
            
            # Garantir que métricas vitais existam mesmo se o passo original não as fornecer
            if not metrics:
                metrics = {}
            if 'n_samples_source' not in metrics: 
                metrics['n_samples_source'] = len(src_mat)
            if 'n_samples_target' not in metrics and tgt_mat is not None: 
                metrics['n_samples_target'] = len(tgt_mat)
            if 'dimensions' not in metrics: 
                metrics['dimensions'] = f"{src_mat.shape[1]}D"
                
            type_str = metrics.get("step_type", StepType.RAW.value)
            
            # --- PARSING INTELIGENTE E CONSULTA AO DOMÍNIO ---
            try:
                # Tenta ler diretamente como StepType (ex: "RAW" no Step 0)
                step_type_enum = StepType(type_str)
                transf_enum = None
            except ValueError:
                # Caso seja uma transformação formatada (ex: "CORAL_TRANSFORMATION")
                transf_enum = TransformationType.from_str(type_str)
                step_type_enum = transf_enum.step_type if transf_enum else StepType.UNKNOWN

            # Consulta a regra de negócio do Domínio (False por padrão para passos Raw/Desconhecidos)
            requires_recalc = transf_enum.requires_axis_recalculation if transf_enum else False

            # Garantia do Controller de que as dimensões são simétricas
            src_dim = src_mat.shape[1]
            tgt_dim = tgt_mat.shape[1] if tgt_mat is not None else src_dim

            # --- 2. CÁLCULO DA CÂMERA (JOINT PCA UNIFICADO) ---
            can_reuse_pca = (active_pca is not None) and (src_dim == active_dim)

            if not can_reuse_pca:
                # Treina um novo Joint PCA empilhando ambos os datasets
                if tgt_mat is not None:
                    total = len(src_mat) + len(tgt_mat)
                    if total > max_fit_samples:
                        # Uso do RNG travado na Semente Global!
                        idx_s = self.rng.choice(len(src_mat), min(len(src_mat), max_fit_samples // 2), replace=False)
                        idx_t = self.rng.choice(len(tgt_mat), min(len(tgt_mat), max_fit_samples // 2), replace=False)
                        data_fit = np.vstack([src_mat[idx_s], tgt_mat[idx_t]])
                    else:
                        data_fit = np.vstack([src_mat, tgt_mat])
                else:
                    data_fit = src_mat

                # Trava a semente do PCA para a Câmera ser idêntica em repetições da auditoria
                active_pca = PCA(n_components=3, random_state=self.seed)
                active_pca.fit(data_fit)
                active_dim = src_dim 
            
            variance = float(np.sum(active_pca.explained_variance_ratio_))
            src_3d = active_pca.transform(src_mat)
            tgt_3d = active_pca.transform(tgt_mat) if tgt_mat is not None else None

            # --- 3. ESTABILIZAÇÃO DE CÂMERA (PROCRUSTES) ---
            if prev_src_3d_points is not None and src_3d.shape == prev_src_3d_points.shape:
                curr_centered = src_3d - np.mean(src_3d, axis=0)
                prev_centered = prev_src_3d_points - np.mean(prev_src_3d_points, axis=0)

                norm_curr = np.linalg.norm(curr_centered) + 1e-10
                norm_prev = np.linalg.norm(prev_centered) + 1e-10

                R, _ = orthogonal_procrustes(curr_centered / norm_curr, prev_centered / norm_prev)

                src_3d = np.dot(src_3d, R)
                if tgt_3d is not None:
                    tgt_3d = np.dot(tgt_3d, R)

            prev_src_3d_points = src_3d.copy()

            # --- 4. EXECUÇÃO DA MÁQUINA DE ESTADOS DE ORIENTAÇÃO (AGORA AGNÓSTICA) ---
            
            # Regra do TARGET: 
            # Calcula no Step 0 (quando target_locked_orientation é None). 
            # Só recalcula se a transformação atual exigir explicitamente (ex: ambos passaram por LDA).
            if tgt_3d is not None:
                if target_locked_orientation is None or requires_recalc:
                    try:
                        _, eigvecs_t = np.linalg.eigh(np.cov(tgt_3d, rowvar=False))
                        target_locked_orientation = eigvecs_t
                    except:
                        if target_locked_orientation is None:
                            target_locked_orientation = np.eye(3)
            
            # Regra do SOURCE: 
            # Calcula no Step 0, e recalcula de forma dinâmica se o Domínio autorizar.
            if source_locked_orientation is None or requires_recalc:
                try:
                    _, eigvecs_s = np.linalg.eigh(np.cov(src_3d, rowvar=False))
                    source_locked_orientation = eigvecs_s
                except:
                    if source_locked_orientation is None:
                        source_locked_orientation = np.eye(3)

            # --- 5. EMPACOTAMENTO FORTEMENTE TIPADO ---
            step_obj = TransformationStep3D(
                step_name=frame.step_name,
                source_points=src_3d,
                target_points=tgt_3d,
                variance=variance,
                original_dimension=src_dim,
                step_type=step_type_enum,
                metrics=metrics,
                source_orientation_matrix=source_locked_orientation,  
                target_orientation_matrix=target_locked_orientation   
            )
            steps_3d.append(step_obj)

        return TransformationTrajectoryData(steps=steps_3d)


    # =========================================================================
    # 7. AGREGAÇÃO AVANÇADA (SMART ENROLLMENT)
    # =========================================================================

    def apply_smart_aggregation(
        self,
        dataset: EmbeddingDataset,
        config: AggregationConfig, # <--- RECEBE O DTO AQUI
        logger_callback: Optional[Callable[[str], None]] = None 
    ) -> EmbeddingDataset:
        """
        Gera templates sintéticos agregando embeddings da mesma identidade, 
        aplicando técnicas avançadas como Combinatória Restrita (K-Combos) 
        e média ponderada baseada nos escores de Qualidade de Imagem (FIQ).
        
        Args:
            dataset: Dataset de entrada.
            config: DTO contendo todas as regras de agregação.
            logger_callback: Função opcional para receber logs (ex: UI).
            
        Returns:
            EmbeddingDataset: Novo dataset contendo apenas os templates virtuais.
        """
        log = logger_callback if logger_callback else print

        # Extrai variáveis do DTO para facilitar a leitura no resto do método
        method = config.method
        group_size = config.combination_group_size
        max_output_per_id = config.max_synthetic_templates_per_id
        quality_weight_key = config.quality_key_str

        log(f"--- Iniciando Smart Aggregation ({method.value.upper()}) ---")
        if quality_weight_key:
            log(f"    -> Usando ponderação por qualidade: '{quality_weight_key}'")

        # 1. Gate fail-closed: identidade é o que DEFINE o agrupamento.
        # Antes, faces sem identidade caíam numa chave literal "unknown" e eram
        # fundidas num único template — pessoas diferentes viravam uma só, em
        # silêncio, e "unknown" passava a contar como identidade real em
        # DatasetStats.n_ids, MIN_IDENTITIES, FisherRule e na matriz
        # intraclasse do PLDA. Agregar o que não tem identidade não é uma
        # operação com significado; abortar é a única saída honesta.
        sem_id = [
            i for i, item in enumerate(dataset.items)
            if item.identity is None or not str(item.identity).strip()
        ]
        if sem_id:
            amostra = ", ".join(str(i) for i in sem_id[:10])
            reticencias = "…" if len(sem_id) > 10 else ""
            raise ValueError(
                f"Agregação exige identidade em todas as faces: "
                f"{len(sem_id)} de {len(dataset.items)} estão sem identidade "
                f"(índices {amostra}{reticencias}). Faces sem identidade não podem "
                f"ser agregadas — corrija o dataset ou filtre-as antes."
            )

        # 2. Agrupamento (Identity -> List[Item])
        groups = defaultdict(list)
        for item in dataset.items:
            groups[str(item.identity)].append(item)

        new_items = []
        
        # 2. Processamento por Identidade
        for identity, items in groups.items():
            n_items = len(items)
            
            # Extrai matriz de embeddings do grupo
            embeddings = np.vstack([item.embedding for item in items])
            
            # Prepara os pesos (se solicitado)
            weights = None
            if quality_weight_key:
                weights = self._extract_weights(items, quality_weight_key)
                # Garante que seja um array numpy para fatiamento seguro nos subgrupos
                if not isinstance(weights, np.ndarray):
                    weights = np.array(weights)

            # 3. Define as Tarefas (Quais índices serão agregados)
            tasks_indices = []

            if method == AggregationMethod.COMBINATORIAL and n_items >= group_size:
                tasks_indices = self._sample_combinations(
                    n_items, group_size, max_output_per_id, self.seed,
                    sampling=getattr(config, "combination_sampling", None),
                )
            else:
                # Centróide Padrão: Uma única tarefa contendo todos os índices do grupo
                # (Se for Combinatorial mas não tiver itens suficientes, faz fallback pra cá)
                tasks_indices = [list(range(n_items))]

            # 4. Executa a Álgebra para cada subgrupo
            for idx_group in tasks_indices:
                indices = list(idx_group)
                sub_embeddings = embeddings[indices]
                
                # Fatiamento seguro dos pesos (evita erro de indexar 'None')
                sub_weights = weights[indices] if weights is not None else None

                # Média + Normalização
                mean_vector = self._compute_mean_and_normalize(sub_embeddings, sub_weights)
                
                sub_items = [items[i] for i in indices]

                # Identidade da linha: derivada do CONJUNTO de faces que
                # entraram, determinística entre execuções e distinta de
                # qualquer hash de foto real (ver _synthetic_template_hash).
                template_hash = self._synthetic_template_hash(
                    identity, sub_items, method, len(indices),
                )
                # O nome antigo derivava só de items[indices[0]].file_name:
                # em COMBINATORIAL, todo combo que compartilhava o primeiro
                # índice gerava nome IDÊNTICO — e, por tabela, hash idêntico.
                # O sufixo de hash desambigua e mantém o rastro.
                new_fname = f"AGG_{identity}_k{len(indices)}_{template_hash[len(_SYNTH_PREFIX):][:12]}"

                new_items.append(EmbeddingInfo(
                    embedding=mean_vector.astype(np.float32),
                    face_hash=template_hash,
                    identity=identity,
                    file_name=new_fname,
                    # ── Metadado de FOTO não sobrevive à agregação ──
                    # Um template é a média de N faces: não é imagem nenhuma,
                    # não tem pose, e "qualidade de imagem" (FIQA/OFIQ) não é
                    # definida para ele. Herdar do primeiro item, como se
                    # fazia, produzia um template que se declarava uma foto
                    # específica — com pose e qualidade que não descrevem o
                    # que o vetor representa. Vazio é a resposta correta, e é
                    # o que faz QualityMetricAvailableRule/HasPoseMetadataRule
                    # bloquearem operações que dependem desses campos.
                    image_path="",
                    pose=None,
                    quality_scores=None,
                    face_image=None,
                    # Proveniência da coleção (não da foto): só sobrevive se
                    # for unânime entre as faces fundidas.
                    source=self._unanimous_attr(sub_items, "source"),
                    origin=self._unanimous_attr(sub_items, "origin"),
                ))

        # `face_hash` é a chave de par de toda a infra a jusante (PAIR_MINING,
        # SCORE_INTERACTION_FEATURE, LR_ARCHITECTURE_VECTORIZATION,
        # ScoreVectorPairCompatibilityRule) e é coluna obrigatória nos parquets
        # de score. Duas linhas com o mesmo hash e embeddings diferentes é
        # corrupção silenciosa: o PAIR_MINING as trata como auto-comparação e
        # descarta os genuínos daquela identidade. Falha aqui, alto e claro.
        from collections import Counter
        colisoes = {
            h: n for h, n in Counter(i.face_hash for i in new_items).items() if n > 1
        }
        if colisoes:
            amostra = ", ".join(f"{h[:24]}…×{n}" for h, n in list(colisoes.items())[:5])
            raise ValueError(
                f"Agregação produziu {sum(colisoes.values())} templates com "
                f"{len(colisoes)} hash(es) repetido(s) ({amostra}). Causa provável: "
                f"faces byte-idênticas (o mesmo arquivo listado mais de uma vez) "
                f"dentro da mesma identidade. Remova as duplicatas antes de agregar."
            )

        log(f"--- Agregação Finalizada. {len(dataset)} itens -> {len(new_items)} templates sintéticos. ---")

        # A ponderação só é EFETIVA se a métrica existia em alguma face. Quando
        # falta em todas, `_extract_weights` devolve pesos unitários e a média
        # degenera em simples — e afirmar "ponderada" no log e na proveniência
        # de um dataset que vai a laudo seria falso.
        n_com_peso, n_total_peso = getattr(self, "_last_weight_presence", (0, 0))
        ponderacao_efetiva = bool(quality_weight_key) and n_com_peso > 0
        if quality_weight_key and not ponderacao_efetiva:
            log(
                f"⚠ Ponderação por '{quality_weight_key}' pedida, mas a métrica "
                f"está AUSENTE em todas as faces — a fusão foi por média simples."
            )

        desc_extra = f" | Weighted({quality_weight_key})" if ponderacao_efetiva else ""

        # Propaga o metadado de dataset (o split e o merge fazem o mesmo em
        # toda saída) e registra a proveniência da agregação. Sem isso, a
        # jusante não há como distinguir "campos de foto vazios por design"
        # de "vazios por falha de ingestão" — e a distinção importa porque
        # as regras de cobertura passam a bloquear operações sobre este
        # dataset.
        meta = dict(getattr(dataset, "loaded_metadata", None) or {})
        meta.update({
            "aggregation_applied": "true",
            "aggregation_method": method.value,
            "aggregation_source_dataset": str(dataset.name),
            "aggregation_source_items": str(len(dataset.items)),
            "aggregation_output_templates": str(len(new_items)),
            "aggregation_identities": str(len(groups)),
            # Só declara ponderação quando ela de fato ocorreu — ver acima.
            "aggregation_quality_weighting": quality_weight_key if ponderacao_efetiva else "",
            "aggregation_quality_weighting_requested": quality_weight_key or "",
            "aggregation_quality_weighting_effective": "true" if ponderacao_efetiva else "false",
            "aggregation_cleared_photo_fields": "image_path,pose,quality_scores,face_image",
            "aggregation_hash_scheme": _SYNTH_HASH_SCHEME,
        })
        if method == AggregationMethod.COMBINATORIAL:
            meta["aggregation_group_size"] = str(group_size)
            meta["aggregation_max_per_identity"] = str(max_output_per_id)
            _amostragem = getattr(config, "combination_sampling", None)
            meta["aggregation_combination_sampling"] = (
                _amostragem.value if _amostragem is not None else ""
            )

        return EmbeddingDataset(
            name=f"{dataset.name}_SmartAgg_{method.value}",
            items=new_items,
            description=f"Smart Aggregation ({method.value}){desc_extra}",
            loaded_metadata=meta,
        )

    # --- HELPERS PRIVADOS ---

    @staticmethod
    def _sample_combinations(
        n_items: int,
        group_size: int,
        max_output: int,
        seed: Optional[int],
        sampling: Optional["CombinationSamplingMode"] = None,
    ) -> List[Tuple[int, ...]]:
        """Até `max_output` combinações de `group_size` índices, SEM materializar C(n,k).

        A versão anterior fazia `list(itertools.combinations(...))` e só depois
        aplicava o teto — pagando o custo inteiro antes de descartar quase tudo.
        Com 60 faces numa identidade e k=5 isso são 5.461.512 tuplas (~480 MB)
        para ficar com 100; com 100 faces, C(100,5)=75 milhões e o processo
        morre por falta de memória. O teto existe justamente para evitar essa
        explosão, então precisa ser aplicado ANTES da geração.

        Quando o total CABE no teto, enumera direto — a cobertura por face é
        perfeitamente uniforme por construção e o modo de amostragem é
        irrelevante. Quando EXCEDE, o critério vem de `sampling`:

        - BALANCED_COVERAGE (default): escolhe a cada passo as k faces menos
          usadas até então, deixando a cobertura por face com amplitude 1.
        - UNIFORM: sorteia posições distintas no espaço combinatório e as
          converte por unranking — amostra não-enviesada, custo O(teto · k).

        Em ambos os casos a seleção é semeada e portanto reprodutível, que é o
        que a agregação sintética exige para ser auditável.
        """
        from faceaudit.enums.combination_sampling_mode import CombinationSamplingMode

        total = math.comb(n_items, group_size)
        rng = random.Random(seed if seed is not None else 0)

        if total <= max_output:
            combos = list(itertools.combinations(range(n_items), group_size))
            rng.shuffle(combos)
            return combos

        if sampling == CombinationSamplingMode.UNIFORM:
            ranks = rng.sample(range(total), max_output)
            ranks.sort()  # ordem estável do resultado, independente do sorteio
            return [
                EmbeddingMathService._unrank_combination(n_items, group_size, r)
                for r in ranks
            ]

        return EmbeddingMathService._balanced_combinations(
            n_items, group_size, max_output, rng,
        )

    @staticmethod
    def _balanced_combinations(
        n_items: int,
        group_size: int,
        max_output: int,
        rng: random.Random,
        max_tentativas: int = 8,
    ) -> List[Tuple[int, ...]]:
        """Combinações escolhidas para equilibrar a participação de cada face.

        O sorteio uniforme é uniforme sobre COMBINAÇÕES, não sobre FACES: com o
        teto cortando, algumas fotos entram em muitos templates e outras em
        nenhum — o que, num dataset de enrollment, significa que parte do
        material de referência simplesmente não foi representada.

        A escolha aqui é gulosa: a cada passo pega as `group_size` faces menos
        usadas até então, com desempate aleatório semeado entre faces de mesma
        cobertura. Isso mantém a cobertura por face dentro de uma unidade
        (⌊teto·k/N⌋ ou ⌈teto·k/N⌉) e garante que nenhuma face fique de fora
        enquanto houver teto disponível.

        O desempate é re-sorteado a cada tentativa porque a mesma escolha
        gulosa pode reincidir numa combinação já usada; esgotadas as
        tentativas, cai num sorteio livre de combinação inédita. Na prática o
        fallback não dispara (medido em n=6..60, k=2..5).
        """
        cobertura = [0] * n_items
        escolhidas: List[Tuple[int, ...]] = []
        vistas: set = set()

        while len(escolhidas) < max_output:
            combo: Optional[Tuple[int, ...]] = None

            for _ in range(max_tentativas):
                ordem = sorted(
                    range(n_items), key=lambda i: (cobertura[i], rng.random()),
                )
                candidata = tuple(sorted(ordem[:group_size]))
                if candidata not in vistas:
                    combo = candidata
                    break

            if combo is None:
                while True:
                    candidata = tuple(sorted(rng.sample(range(n_items), group_size)))
                    if candidata not in vistas:
                        combo = candidata
                        break

            vistas.add(combo)
            escolhidas.append(combo)
            for idx in combo:
                cobertura[idx] += 1

        return escolhidas

    @staticmethod
    def _unrank_combination(n: int, k: int, rank: int) -> Tuple[int, ...]:
        """A combinação de índices na posição `rank` da ordem lexicográfica.

        Sistema numérico combinatório: permite acessar a r-ésima combinação sem
        gerar as anteriores. É o que torna possível respeitar o teto sem
        enumerar o espaço inteiro.
        """
        resultado: List[int] = []
        atual = 0
        restante = rank
        for pos in range(k):
            faltam = k - pos
            while True:
                bloco = math.comb(n - atual - 1, faltam - 1)
                if restante < bloco:
                    break
                restante -= bloco
                atual += 1
            resultado.append(atual)
            atual += 1
        return tuple(resultado)

    @staticmethod
    def _contributor_key(item: EmbeddingInfo) -> str:
        """Chave da face contribuinte usada na derivação do hash do template.

        Usa `face_hash` quando existe; senão, um digest do CONTEÚDO.

        `face_hash` vazio é estado NORMAL e frequente aqui — `to_embedding_dataset`
        faz `face_hash=face.hash or ""` e `EmbeddingDataset.from_parquet` faz
        `hash_value or ''` para qualquer parquet sem a coluna. Sem este fallback,
        todo k-combo de uma identidade cujas faces têm hash "" produzia payload
        canônico byte-idêntico e, portanto, o MESMO digest: N templates com N
        embeddings distintos colapsavam numa única chave de par — e o PAIR_MINING,
        que exclui auto-comparação por igualdade de hash, descartaria em silêncio
        todos os pares genuínos daquela identidade.

        O fallback é derivado de conteúdo (e não da posição no dataset) para
        preservar a invariância à ordem, que os testes cobram.
        """
        h = str(getattr(item, "face_hash", "") or "").strip()
        if h:
            return f"h:{h}"

        from faceaudit.models.lr._canonical_hash import sha256_hex

        emb = np.ascontiguousarray(np.asarray(item.embedding, dtype="<f8"))
        return "c:" + sha256_hex(b"|".join([
            str(getattr(item, "image_path", "") or "").encode("utf-8"),
            str(getattr(item, "file_name", "") or "").encode("utf-8"),
            emb.tobytes(),
        ]))

    @staticmethod
    def _synthetic_template_hash(
        identity: str,
        items: List[EmbeddingInfo],
        method: AggregationMethod,
        group_size: int,
    ) -> str:
        """Identificador determinístico de um template agregado.

        `face_hash` não é decoração: é a CHAVE DE PAR de toda a infra de
        vetorização LR (`_align_pairs` do SCORE_INTERACTION_FEATURE,
        `_extract_canonical_rarity` do LR_ARCHITECTURE_VECTORIZATION,
        ScoreVectorPairCompatibilityRule, exclusão de auto-comparação no
        PAIR_MINING) e é persistido como coluna obrigatória
        `probe_face_hash`/`reference_face_hash` nos parquets de score.

        Duas exigências, por isso:

        1. **Determinístico.** O `hash()` embutido do Python é salgado por
           processo (PYTHONHASHSEED) desde a 3.3 — a mesma agregação rodada
           duas vezes produzia identificadores diferentes, quebrando o
           pareamento entre execuções. Aqui usa-se SHA-256 sobre payload
           canônico (mesma convenção dos fingerprints do subsistema LR).
        2. **Distinto de qualquer foto real.** Derivar do CONJUNTO ordenado
           de faces contribuintes torna o template rastreável sem que ele
           colida com nenhuma das fotos que o formaram — o que era o defeito
           da agregação legada, que herdava o hash de uma delas e fazia um
           template sintético se passar por imagem real na detecção de
           vazamento probe↔referência.

        A ordenação dos hashes contribuintes torna o resultado invariante à
        ordem em que as faces aparecem no dataset.
        """
        from faceaudit.models.lr._canonical_hash import hash_canonical_payload

        contribuintes = sorted(
            EmbeddingMathService._contributor_key(it) for it in items
        )
        digest = hash_canonical_payload({
            "kind": "aggregated_template",
            "scheme": _SYNTH_HASH_SCHEME,
            "identity": str(identity),
            "method": method.value,
            "group_size": int(group_size),
            "sources": contribuintes,
        })
        return f"{_SYNTH_PREFIX}{digest}"

    @staticmethod
    def _unanimous_attr(items: List[EmbeddingInfo], attr: str) -> Optional[str]:
        """Valor do atributo quando TODAS as faces concordam; None se divergem.

        `source`/`origin` descrevem a coleção de onde a face veio, não a foto
        em si — então sobrevivem à fusão desde que sejam os mesmos. Fundir
        faces de origens diferentes num template torna a origem ambígua, e
        ambíguo vira ausente em vez de virar "a primeira".
        """
        valores = {getattr(it, attr, None) for it in items}
        if len(valores) == 1:
            return valores.pop()
        return None

    def _extract_weights(self, items: List[EmbeddingInfo], key: str) -> np.ndarray:
        """
        Lê os dicionários de qualidade de cada item, retornando um vetor NumPY
        seguro. Aplica um fallback para peso=1.0 se a qualidade for Nula ou inexistente.
        """
        weights = []
        n_presentes = 0
        for item in items:
            val = 1.0
            if item.quality_scores and key in item.quality_scores:
                v = item.quality_scores[key]
                if v is not None:
                    try:
                        candidato = float(v)
                    except (TypeError, ValueError):
                        candidato = None
                    # NaN/inf atravessavam TODAS as guardas: `float('nan')` não
                    # é None, `np.sum(...) == 0` é False quando a soma é NaN, e
                    # `norm > 1e-9` é False para NaN — o vetor saía inteiramente
                    # NaN, sem exceção e sem log. Score não-finito é score
                    # ausente: cai no fallback 1.0.
                    if candidato is not None and np.isfinite(candidato):
                        val = candidato
                        n_presentes += 1
            weights.append(val)

        w_arr = np.array(weights, dtype=np.float32)
        # `_last_weight_presence` deixa o chamador saber se a ponderação foi
        # EFETIVA. Sem isso, a proveniência gravada no dataset afirmava fusão
        # ponderada mesmo quando a métrica faltava em todas as faces e a média
        # degenerou em simples.
        self._last_weight_presence = (n_presentes, len(items))
        if np.sum(w_arr) == 0:
            return np.ones(len(items), dtype=np.float32)
        return w_arr

    def _compute_mean_and_normalize(self, vectors: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
        """
        Calcula a média ponderada ou simples de tensores 1D e reprojeta
        o vetor resultante de volta à hiperesfera (Norma L2 = 1.0).
        """
        # A guarda de soma-zero do `_extract_weights` é do GRUPO inteiro, mas em
        # COMBINATORIAL o vetor é fatiado por SUBGRUPO antes de chegar aqui: um
        # combo cujas k faces têm todas qualidade 0.0 soma zero mesmo com o
        # grupo somando positivo, e `np.average` levantava ZeroDivisionError,
        # abortando a instrução inteira. Peso degenerado = sem informação de
        # qualidade neste subgrupo; média simples é a resposta.
        if weights is not None:
            soma = float(np.sum(weights))
            if not np.isfinite(soma) or soma <= 0.0:
                weights = None

        if weights is not None:
            mean = np.average(vectors, axis=0, weights=weights)
        else:
            mean = np.mean(vectors, axis=0)

        norm = float(np.linalg.norm(mean))
        if np.isfinite(norm) and norm > 1e-9:
            mean = mean / norm
        return mean