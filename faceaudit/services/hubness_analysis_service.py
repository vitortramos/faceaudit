# faceaudit/services/hubness_analysis_service.py
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from typing import Optional, List
from collections import defaultdict
from tqdm.auto import tqdm

from faceaudit.models.hubness_analysis_info import HubnessAnalysisInfo
from faceaudit.models.hubness_optimization_study_info import HubnessOptimizationSample, HubnessOptimizationStudyInfo
from faceaudit.models.hubness_sample_info import HubnessSampleInfo
from faceaudit.models.hubness_stratification_info import HubnessStratificationInfo
from faceaudit.models.hubness_paired_study_info import HubnessPairedStudyInfo, HubnessPairedSample
from faceaudit.models.ablation_study_info import AblationStudyInfo
from faceaudit.models.ablation_scenario_info import AblationScenarioInfo
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.hubness_quality_diagnostics_info import HubnessQualityDiagnosticsInfo, HubnessQualityRange


class HubnessAnalysisService:
    """
    Serviço de cálculo para análise de viés topológico (Hubness).
    Otimizado para Big Data: Vetorização extrema, Zero-Copy Masks e Label Encoding.
    """

    def __init__(self, comparison_service, k_neighbors: int = 20):
        self.comparison_service = comparison_service
        self.k = k_neighbors

    def compute_analysis(
        self,
        dataset: EmbeddingDataset,
        batch_size: int = 1000,
        chunk_size: int = 10000,
        logger_callback: Optional[callable] = None,
    ) -> HubnessAnalysisInfo:
        """
        Executa a análise estatística base sobre o EmbeddingDataset de forma LOTEADA DUPLA.
        Versão Ultimate CPU: Zero-Copy Masks, Pré-computação de Índices e Soma por Exclusão.

        `logger_callback` opcional redireciona mensagens de progresso para um
        sink customizado (ex: log do StrategyExecutionContext). Quando None,
        mantém o comportamento original com `print` + `tqdm`.
        """
        log = logger_callback if logger_callback is not None else print
        items = dataset.items
        n = len(items)
        matrix = dataset.matrix
        identities_array = np.array(dataset.identities)
        
        valid_mask = np.array([idx is not None for idx in identities_array], dtype=bool)
        k_eff = min(self.k, max(1, n - 1))
        
        nk_scores = np.zeros(n, dtype=int)
        mean_impostors = np.zeros(n, dtype=np.float32)
        max_impostors = np.zeros(n, dtype=np.float32)
        mean_delta_pose = np.zeros(n, dtype=np.float32)
        
        poses = np.array([abs(float(item.yaw)) + abs(float(item.pitch)) for item in items])
        raw_norms = np.linalg.norm(matrix, axis=1)

        # 1. Label Encoding
        _, integer_ids = np.unique(identities_array[valid_mask], return_inverse=True)
        id_ints = np.full(n, -1, dtype=np.int32)
        id_ints[valid_mask] = integer_ids

        # Dicionário Global para cruzamento O(1) sem np.where
        from collections import defaultdict
        id_to_indices = defaultdict(list)
        for idx, i_id in enumerate(id_ints):
            if i_id != -1:
                id_to_indices[i_id].append(idx)
        id_ints_map = {k: np.array(v, dtype=np.int32) for k, v in id_to_indices.items()}

        # 2. Normalização Global Única
        norm_matrix = self.comparison_service._l2_normalize_rows(matrix)

        log(f">>> [Processamento Block-Wise] Hubness {n} faces. Batch: {batch_size}, Chunk: {chunk_size}...")

        # Quando o caller injetou um logger custom, evitamos tqdm (que escreve
        # direto em stderr e pode confundir quem está consumindo via UI/log).
        # Em vez disso, emitimos progresso periódico via o callback.
        use_tqdm = logger_callback is None
        outer_range = range(0, n, batch_size)
        outer_iter = (
            tqdm(outer_range, desc="Processando Lotes (Queries)", unit="lote")
            if use_tqdm else outer_range
        )

        total_batches = (n + batch_size - 1) // batch_size

        # =====================================================================
        # LOOP EXTERNO (Lotes de Consulta - Query)
        # =====================================================================
        for batch_idx, i in enumerate(outer_iter):
            end_i = min(i + batch_size, n)
            batch_query = norm_matrix[i:end_i]
            batch_ids = id_ints[i:end_i]
            B = end_i - i
            
            # Acumuladores do Lote (float64 para evitar perda de precisão na soma longa)
            batch_sum_impostors = np.zeros(B, dtype=np.float64)
            batch_max_impostors = np.full(B, -np.inf, dtype=np.float32)
            batch_impostor_counts = np.zeros(B, dtype=np.int32)
            
            running_top_k_scores = np.full((B, k_eff), -np.inf, dtype=np.float32)
            running_top_k_indices = np.full((B, k_eff), -1, dtype=np.int32)

            # Pré-computação: Posições locais de cada identidade dentro do Batch
            unique_batch_ids = np.unique(batch_ids[batch_ids != -1])
            batch_pos_by_id = {uid: np.where(batch_ids == uid)[0] for uid in unique_batch_ids}

            # =================================================================
            # LOOP INTERNO (Blocos de Referência - Chunks)
            # =================================================================
            for j in range(0, n, chunk_size):
                end_j = min(j + chunk_size, n)
                chunk_ref = norm_matrix[j:end_j]
                chunk_ids = id_ints[j:end_j]
                C = end_j - j
                
                # Produto escalar otimizado (B x C)
                sim = np.dot(batch_query, chunk_ref.T)
                
                # -------------------------------------------------------------
                # Truque Matemático: Soma Total - Soma dos Excluídos
                # -------------------------------------------------------------
                chunk_sum_all = np.sum(sim, axis=1, dtype=np.float64)
                chunk_sum_exclude = np.zeros(B, dtype=np.float64)
                chunk_count_exclude = np.zeros(B, dtype=np.int32)
                
                # A. Exclusão de Inválidos do Chunk (-1)
                invalid_chunk_mask = (chunk_ids == -1)
                if np.any(invalid_chunk_mask):
                    invalid_cols = np.where(invalid_chunk_mask)[0]
                    chunk_sum_exclude += np.sum(sim[:, invalid_cols], axis=1)
                    chunk_count_exclude += invalid_cols.size
                    sim[:, invalid_cols] = -np.inf
                    
                # B. Exclusão de Genuínos via Interseção de Dicionários (O(1))
                # Nota: A auto-comparação (diagonal) é implicitamente tratada aqui!
                for uid in unique_batch_ids:
                    global_c_idx = id_ints_map[uid]
                    # Filtra os índices da pessoa que caem na janela deste chunk
                    c_idx = global_c_idx[(global_c_idx >= j) & (global_c_idx < end_j)] - j
                    
                    if c_idx.size > 0:
                        b_idx = batch_pos_by_id[uid]
                        genuine_scores = sim[b_idx[:, None], c_idx]
                        
                        chunk_sum_exclude[b_idx] += np.sum(genuine_scores, axis=1)
                        chunk_count_exclude[b_idx] += c_idx.size
                        sim[b_idx[:, None], c_idx] = -np.inf

                # Atualiza as métricas globais do lote (Sem criar máscaras booleanas BxC)
                batch_sum_impostors += (chunk_sum_all - chunk_sum_exclude)
                batch_impostor_counts += (C - chunk_count_exclude)
                batch_max_impostors = np.maximum(batch_max_impostors, np.max(sim, axis=1, initial=-np.inf))
                
                # -------------------------------------------------------------
                # Top-K Contínuo
                # -------------------------------------------------------------
                curr_k = min(k_eff, C)
                chunk_top_k_idx = np.argpartition(sim, -curr_k, axis=1)[:, -curr_k:]
                chunk_top_k_scores = np.take_along_axis(sim, chunk_top_k_idx, axis=1)
                chunk_top_k_global_idx = chunk_top_k_idx + j
                
                # Alerta: Se K for muito grande (>1000), a concatenação pode pesar.
                merged_scores = np.concatenate([running_top_k_scores, chunk_top_k_scores], axis=1)
                merged_indices = np.concatenate([running_top_k_indices, chunk_top_k_global_idx], axis=1)
                
                best_merged_idx = np.argpartition(merged_scores, -k_eff, axis=1)[:, -k_eff:]
                running_top_k_scores = np.take_along_axis(merged_scores, best_merged_idx, axis=1)
                running_top_k_indices = np.take_along_axis(merged_indices, best_merged_idx, axis=1)

            # =================================================================
            # FIM DO LOOP DE CHUNKS: Consolidação Vetorizada do Batch
            # =================================================================
            
            max_impostors[i:end_i] = batch_max_impostors
            
            valid_mean_mask = batch_impostor_counts > 0
            mean_impostors[i:end_i] = np.divide(
                batch_sum_impostors, 
                batch_impostor_counts, 
                out=np.zeros(B, dtype=np.float32), 
                where=valid_mean_mask
            )
            
            # Hubness (Nk) otimizado usando bincount
            valid_k_mask = (running_top_k_indices != -1)
            valid_k_flat = running_top_k_indices[valid_k_mask]
            nk_scores += np.bincount(valid_k_flat, minlength=n)
            
            # Delta Pose Vetorizado
            safe_indices = np.maximum(running_top_k_indices, 0)
            neighbor_poses = poses[safe_indices]
            pose_diffs = np.abs(poses[i:end_i, None] - neighbor_poses)
            
            sum_diffs = np.sum(pose_diffs, axis=1, where=valid_k_mask, initial=0.0)
            count_valid_diffs = np.sum(valid_k_mask, axis=1)
            
            mean_delta_pose[i:end_i] = np.divide(
                sum_diffs,
                count_valid_diffs,
                out=np.zeros(B, dtype=np.float32),
                where=count_valid_diffs > 0
            )

            # Progresso periódico via callback (a cada 10% ou no fim).
            if not use_tqdm and (
                batch_idx == total_batches - 1
                or batch_idx % max(1, total_batches // 10) == 0
            ):
                pct = int(100 * (batch_idx + 1) / max(1, total_batches))
                log(f"  ↳ Hubness analysis: {pct}% ({batch_idx + 1}/{total_batches} lotes)")

        log("\n>>> Matemática O(N²) concluída. Instanciando DTOs...")

        # Construção da Lista de Resultados
        samples = []
        for i, item in enumerate(items):
            samples.append(HubnessSampleInfo(
                index=i, identity=item.identity, norm=float(raw_norms[i]),
                quality_score=float(item.quality_score or 0.0),
                serfiq=float(item.quality_scores.get('serfiq', 0.0)),
                yaw=float(item.yaw), pitch=float(item.pitch), abs_pose=float(poses[i]),
                nk_score=int(nk_scores[i]), mean_impostor_score=float(mean_impostors[i]),
                max_impostor_score=float(max_impostors[i]), mean_delta_pose_to_neighbors=float(mean_delta_pose[i])
            ))

        return self._finalize_analysis(n, identities_array, valid_mask, samples, k_eff, nk_scores)

    def compute_stratification(self, analysis: HubnessAnalysisInfo) -> List[HubnessStratificationInfo]:
        """Estratifica os resultados em 5 faixas rigorosas de Yaw."""
        df = pd.DataFrame([vars(s) for s in analysis.samples])
        bins = [0, 15, 30, 45, 60, 180]
        labels = ["Frontal (<15°)", "Leve (15-30°)", "Moderada (30-45°)", "Forte (45-60°)", "Extrema (>60°)"]
        
        df['yaw_abs'] = df['yaw'].abs()
        df['range'] = pd.cut(df['yaw_abs'], bins=bins, labels=labels, right=False)
        
        strat_results = []
        for i, label in enumerate(labels):
            subset = df[df['range'] == label]
            if subset.empty: 
                continue
                
            strat_results.append(HubnessStratificationInfo(
                label=label, 
                min_yaw=bins[i], 
                max_yaw=bins[i+1], 
                sample_count=len(subset),
                avg_nk=float(subset['nk_score'].mean()), 
                avg_norm=float(subset['norm'].mean()),
                avg_mean_impostor_score=float(subset['mean_impostor_score'].mean()),
                avg_max_impostor_score=float(subset['max_impostor_score'].mean()),
                nk_scores=subset['nk_score'].tolist(), 
                norms=subset['norm'].tolist(),
                max_impostor_scores=subset['max_impostor_score'].tolist()
            ))
            
        return strat_results

    def compute_paired_study(self, analysis: HubnessAnalysisInfo, threshold_nf: float = 30.0) -> HubnessPairedStudyInfo:
        """Realiza o estudo pareado: Foto Frontal (<15°) vs Melhor Não-Frontal (>30°)."""
        df = pd.DataFrame([vars(s) for s in analysis.samples])
        df['yaw_abs'] = df['yaw'].abs()
        
        # Filtra identidades válidas
        df = df[df['identity'].notnull()]
        
        paired_samples = []
        for identity, group in df.groupby('identity'):
            frontal = group[group['yaw_abs'] < 15.0].sort_values('yaw_abs').head(1)
            non_frontal = group[group['yaw_abs'] >= threshold_nf].sort_values('yaw_abs').head(1)
            
            if not frontal.empty and not non_frontal.empty:
                f, nf = frontal.iloc[0], non_frontal.iloc[0]
                
                paired_samples.append(HubnessPairedSample(
                    identity=str(identity),
                    f_nk=int(f['nk_score']), 
                    f_norm=float(f['norm']), 
                    f_max_impostor=float(f['max_impostor_score']),
                    nf_nk=int(nf['nk_score']), 
                    nf_norm=float(nf['norm']), 
                    nf_max_impostor=float(nf['max_impostor_score']),
                    nf_yaw=float(nf['yaw']), 
                    delta_nk=int(nf['nk_score'] - f['nk_score']),
                    delta_norm=float(nf['norm'] - f['norm']), 
                    delta_max_impostor=float(nf['max_impostor_score'] - f['max_impostor_score'])
                ))
                
        return HubnessPairedStudyInfo(
            comparison_label=f"Frontal (<15°) vs Não-Frontal (>{threshold_nf}°)", 
            paired_samples=paired_samples
        )
    
    def compute_quality_diagnostics(self, analysis: HubnessAnalysisInfo) -> HubnessQualityDiagnosticsInfo:
        """
        Diagnóstico de Covariáveis: Verifica se a qualidade (FIQA) ou a 
        Norma L2 variam sistematicamente com a pose.
        """
        df = pd.DataFrame([vars(s) for s in analysis.samples])
        bins = [0, 15, 30, 45, 60, 180]
        labels = ["Frontal (<15°)", "Leve (15-30°)", "Moderada (30-45°)", "Forte (45-60°)", "Extrema (>60°)"]
        
        df['yaw_abs'] = df['yaw'].abs()
        df['range'] = pd.cut(df['yaw_abs'], bins=bins, labels=labels, right=False)
        
        range_results = []
        for label in labels:
            subset = df[df['range'] == label]
            if subset.empty: 
                continue
                
            range_results.append(HubnessQualityRange(
                label=label, 
                sample_count=len(subset),
                avg_quality=float(subset['quality_score'].mean()), 
                median_quality=float(subset['quality_score'].median()), 
                std_quality=float(subset['quality_score'].std()), 
                quality_values=subset['quality_score'].tolist(),
                avg_norm=float(subset['norm'].mean()), 
                median_norm=float(subset['norm'].median()), 
                std_norm=float(subset['norm'].std()), 
                norm_values=subset['norm'].tolist()
            ))
            
        return HubnessQualityDiagnosticsInfo(ranges=range_results)
    
    def compute_optimization_study(self, analysis: HubnessAnalysisInfo) -> HubnessOptimizationStudyInfo:
        """Analisa se identidades performam melhor em 15-30° (Wild) do que em 0-8° (Lab)."""
        df = pd.DataFrame([vars(s) for s in analysis.samples])
        df['yaw_abs'] = df['yaw'].abs()
        df = df[df['identity'].notnull()]
        
        opt_samples = []
        for identity, group in df.groupby('identity'):
            s_row = group[group['yaw_abs'] < 8.0].sort_values('yaw_abs').head(1)
            l_row = group[(group['yaw_abs'] >= 15.0) & (group['yaw_abs'] <= 30.0)].sort_values('nk_score').head(1)
            p_row = group[group['yaw_abs'] > 45.0].sort_values('yaw_abs').head(1)
            
            if not (s_row.empty or l_row.empty or p_row.empty):
                s, l, p = s_row.iloc[0], l_row.iloc[0], p_row.iloc[0]
                
                opt_samples.append(HubnessOptimizationSample(
                    identity=str(identity), 
                    s_nk=int(s['nk_score']), 
                    s_norm=float(s['norm']),
                    l_nk=int(l['nk_score']), 
                    l_norm=float(l['norm']), 
                    l_yaw=float(l['yaw']),
                    p_nk=int(p['nk_score']), 
                    p_norm=float(p['norm']),
                    delta_nk_optimization=int(l['nk_score'] - s['nk_score']),
                    delta_nk_degradation=int(p['nk_score'] - s['nk_score']),
                    delta_norm_optimization=float(l['norm'] - s['norm'])
                ))
                
        return HubnessOptimizationStudyInfo(
            label="Análise de Sweet Spot: Estrito (Lab) vs Leve (Wild) vs Perfil", 
            samples=opt_samples
        )

    def filter_dataset_by_quality(self, dataset: EmbeddingDataset, threshold: Optional[float] = None, percentile: Optional[float] = None) -> EmbeddingDataset:
        """Cria um novo EmbeddingDataset contendo apenas a elite de qualidade."""
        qualities = np.array([float(item.quality_score or 0.0) for item in dataset.items])
        
        if percentile is not None:
            cutoff = np.percentile(qualities, (1 - percentile) * 100)
            mode_label = f"Top {percentile*100:.0f}% (FIQA >= {cutoff:.3f})"
        elif threshold is not None:
            cutoff = threshold
            mode_label = f"Absoluto (FIQA >= {cutoff:.3f})"
        else:
            raise ValueError("Forneça 'threshold' ou 'percentile'.")

        filtered_items = [item for item in dataset.items if (item.quality_score or 0.0) >= cutoff]
        
        if not filtered_items: 
            raise ValueError(f"Filtro restritivo demais: nenhuma imagem restou com {mode_label}")
            
        print(f">>> [FILTRO DE PUREZA] {mode_label}: {len(filtered_items)} de {len(dataset.items)} preservadas.")
        
        return EmbeddingDataset(
            name=f"{dataset.name}_Elite", 
            items=filtered_items, 
            description=f"Subset filtrado por qualidade ({mode_label})", 
            loaded_metadata=dataset.loaded_metadata.copy()
        )

    def _finalize_analysis(self, n, id_array, valid_mask, samples, k_eff, nk_scores) -> HubnessAnalysisInfo:
        """Calcula correlações globais e popula o DTO final com as estatísticas de distribuição."""
        valid_samples = [s for s in samples if s.identity is not None]
        stats = {k: None for k in ["ri", "pi", "rhoi", "si", "rn", "pn", "rhon", "sn"]}
        
        if len(valid_samples) > 1:
            nk_v = [s.nk_score for s in valid_samples]
            imp_v = [s.mean_impostor_score for s in valid_samples]
            norm_v = [s.norm for s in valid_samples]
            
            stats["ri"], stats["pi"] = pearsonr(nk_v, imp_v)
            stats["rhoi"], stats["si"] = spearmanr(nk_v, imp_v)
            stats["rn"], stats["pn"] = pearsonr(norm_v, nk_v)
            stats["rhon"], stats["sn"] = spearmanr(norm_v, nk_v)

        return HubnessAnalysisInfo(
            k_neighbors=int(k_eff), 
            sample_count=int(n), 
            valid_identity_count=len(set(id_array[valid_mask])), 
            samples=samples,
            max_nk=int(np.max(nk_scores)), 
            mean_nk=float(np.mean(nk_scores)), 
            std_nk=float(np.std(nk_scores)),
            pearson_hubness_vs_mean_impostor=stats["ri"], 
            pearson_hubness_vs_mean_impostor_p_value=stats["pi"],
            spearman_hubness_vs_mean_impostor=stats["rhoi"], 
            spearman_hubness_vs_mean_impostor_p_value=stats["si"],
            pearson_norm_vs_hubness=stats["rn"], 
            pearson_norm_vs_hubness_p_value=stats["pn"],
            spearman_norm_vs_hubness=stats["rhon"], 
            spearman_norm_vs_hubness_p_value=stats["sn"]
        )

    # =========================================================================
    # MOTORES DE ABLAÇÃO
    # =========================================================================

    def compute_ablation_by_quantity(
        self,
        dataset: EmbeddingDataset,
        batch_size: int = 1000,
        logger_callback: Optional[callable] = None,
    ) -> AblationStudyInfo:
        """Ablação topológica baseada na quantidade de melhores fotos (FIQA) por identidade."""
        temp_data = [
            {'index': i, 'identity': item.identity, 'quality': float(item.quality_score or 0.0)}
            for i, item in enumerate(dataset.items)
        ]
        df_order = pd.DataFrame(temp_data).sort_values(by=['identity', 'quality'], ascending=[True, False])

        scenarios = [
            ("1 Foto (Melhor)/ID", 1),
            ("2 Fotos (Melhores)/ID", 2),
            ("3 Fotos (Melhores)/ID", 3),
            ("Dataset Total", None)
        ]

        return self._run_ablation_engine(
            dataset, df_order, scenarios,
            batch_size=batch_size, logger_callback=logger_callback,
        )

    def compute_ablation_by_pose(
        self,
        dataset: EmbeddingDataset,
        batch_size: int = 1000,
        logger_callback: Optional[callable] = None,
    ) -> AblationStudyInfo:
        """Ablação topológica priorizando as fotos mais frontais por identidade."""
        temp_data = [
            {'index': i, 'identity': item.identity, 'pose_err': abs(item.yaw) + abs(item.pitch)}
            for i, item in enumerate(dataset.items)
        ]
        df_order = pd.DataFrame(temp_data).sort_values(by=['identity', 'pose_err'], ascending=[True, True])

        scenarios = [
            ("Top 1 (Mais Frontal)/ID", 1),
            ("Top 2 (Mais Frontais)/ID", 2),
            ("Top 3 (Mais Frontais)/ID", 3),
            ("Dataset Total", None)
        ]

        return self._run_ablation_engine(
            dataset, df_order, scenarios,
            batch_size=batch_size, logger_callback=logger_callback,
        )

    def _run_ablation_engine(
        self,
        dataset: EmbeddingDataset,
        df_ordered: pd.DataFrame,
        scenarios_config: list,
        batch_size: int = 1000,
        logger_callback: Optional[callable] = None,
    ) -> AblationStudyInfo:
        """
        Motor Vetorizado Zero-Copy para Ablação.
        Garante que recortes do dataset também sejam processados em lote sem estourar a RAM.
        """
        log = logger_callback if logger_callback is not None else print
        use_tqdm = logger_callback is None
        identities_array = np.array(dataset.identities)
        valid_mask_global = np.array([idx is not None for idx in identities_array], dtype=bool)
        k_eff = min(self.k, max(1, len(set(identities_array[valid_mask_global])) - 2))
        
        _, integer_ids = np.unique(identities_array[valid_mask_global], return_inverse=True)
        id_ints_global = np.full(len(identities_array), -1, dtype=np.int32)
        id_ints_global[valid_mask_global] = integer_ids
        
        results = []
        for name, limit in scenarios_config:
            log(f"\n>>> [Ablação] Processando cenário: {name}...")
            
            indices = df_ordered.groupby('identity').head(limit)['index'].values if limit else df_ordered['index'].values
            sub_emb = dataset.matrix[indices]
            sub_ints = id_ints_global[indices]
            n_sub = len(indices)
            
            if n_sub == 0: 
                continue
            
            # Dicionário do subconjunto (Zero-Copy Masking)
            id_to_indices_sub = defaultdict(list)
            for idx, i_id in enumerate(sub_ints):
                if i_id != -1:
                    id_to_indices_sub[i_id].append(idx)
            id_ints_map_sub = {k: np.array(v, dtype=np.int32) for k, v in id_to_indices_sub.items()}
            
            curr_k = min(k_eff, max(1, n_sub - 1))
            nk_scores = np.zeros(n_sub, dtype=int)
            normalized_sub_emb = self.comparison_service._l2_normalize_rows(sub_emb)
            
            inner_range = range(0, n_sub, batch_size)
            inner_iter = (
                tqdm(inner_range, desc=f"    Calculando {name}", unit="lote", leave=False)
                if use_tqdm else inner_range
            )
            for i in inner_iter:
                end = min(i + batch_size, n_sub)
                batch_matrix = normalized_sub_emb[i:end]
                batch_ints = sub_ints[i:end]
                B = end - i
                
                batch_scores = self.comparison_service.compare_batch_against_all(
                    batch_matrix, 
                    normalized_sub_emb, 
                    is_normalized=True
                )
                
                # Zero-Copy Anulação Cirúrgica
                for r in range(B):
                    curr_id = batch_ints[r]
                    if curr_id != -1:
                        batch_scores[r, id_ints_map_sub[curr_id]] = -np.inf
                    else:
                        batch_scores[r, i + r] = -np.inf
                    
                batch_top_k = np.argpartition(batch_scores, -curr_k, axis=1)[:, -curr_k:]
                nk_scores += np.bincount(batch_top_k.flatten(), minlength=n_sub)
            
            results.append(AblationScenarioInfo(
                scenario_name=name, 
                image_count_per_identity=limit, 
                max_nk=int(np.max(nk_scores)), 
                nk_scores=nk_scores.tolist()
            ))
        
        return AblationStudyInfo(k_neighbors=k_eff, scenarios=results)