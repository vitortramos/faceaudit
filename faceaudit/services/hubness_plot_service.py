from __future__ import annotations
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List

from faceaudit.models.ablation_study_info import AblationStudyInfo
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.hubness_analysis_info import HubnessAnalysisInfo
from faceaudit.models.hubness_optimization_study_info import HubnessOptimizationStudyInfo
from faceaudit.models.hubness_quality_diagnostics_info import HubnessQualityDiagnosticsInfo


class HubnessPlotService:
    """
    Serviço de visualização de dados para auditoria de viés topológico.
    
    Responsável por converter DTOs de análise em representações gráficas formais
    adequadas para laudos periciais e publicações científicas. 
    Requer 'statsmodels' instalado para cálculos de regressão LOWESS.
    """

    def __init__(self):
        # Configuração estética estritamente acadêmica (e.g., IEEE, Elsevier)
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'figure.dpi': 200,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'savefig.bbox': 'tight'
        })

    def _prepare_df(self, analysis: HubnessAnalysisInfo) -> pd.DataFrame:
        """Converte a lista de HubnessSampleInfo em um DataFrame interno para o Seaborn."""
        return pd.DataFrame([vars(s) for s in analysis.samples])

    # --- MÉTODOS ORIGINAIS MANTIDOS ---

    def plot_hubness_vs_impostor_similarity(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df, x='nk_score', y='mean_impostor_score', s=15, alpha=0.15, color='purple', edgecolor='white', linewidth=0.2)
        sns.regplot(data=df, x='nk_score', y='mean_impostor_score', scatter=False, color='black', lowess=True)
        stats = (f"Pearson $r$: {analysis.pearson_hubness_vs_mean_impostor:.3f}\n"
                 f"Spearman $\\rho_s$: {analysis.spearman_hubness_vs_mean_impostor:.3f}")
        plt.annotate(stats, xy=(0.05, 0.9), xycoords='axes fraction', bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
        plt.title(f"Risco Biométrico: Hubness ($N_{{{analysis.k_neighbors}}}$) vs Similaridade Impostora")
        plt.xlabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.ylabel("Similaridade Média (Pares Impostores)")
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_norm_vs_hubness_distribution(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x='norm', y='nk_score', levels=5, color="steelblue", alpha=0.4)
        sns.scatterplot(data=df, x='norm', y='nk_score', s=10, alpha=0.15, color='steelblue', edgecolor='none')
        sns.regplot(data=df, x='norm', y='nk_score', scatter=False, color='darkorange', lowess=True)
        plt.title("Topologia Hiperesférica: Norma L2 ($||z||$) vs Hubness")
        plt.xlabel("Norma L2 ($||z||$)")
        plt.ylabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_norm_vs_pose_impact(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='abs_pose', y='norm', alpha=0.3, s=15, color='teal', edgecolor='none')
        sns.regplot(data=df, x='abs_pose', y='norm', scatter=False, color='red', lowess=True)
        plt.title("Degradação de Magnitude: Pose Absoluta vs Norma L2")
        plt.xlabel("Pose Acumulada ($|Yaw| + |Pitch|$)")
        plt.ylabel("Norma L2 ($||z||$)")
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_pose_vs_hubness_direct(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(9, 6))
        sns.scatterplot(data=df, x='abs_pose', y='nk_score', s=20, alpha=0.2, color='darkblue', edgecolor='none')
        sns.regplot(data=df, x='abs_pose', y='nk_score', scatter=False, color='red', lowess=True, line_kws={'linewidth': 3})
        plt.title("Assimetria Topológica: Desvio de Pose vs Hubness", fontweight='bold')
        plt.xlabel("Desvio de Pose Acumulado ($|Yaw| + |Pitch|$)")
        plt.ylabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_pose_norm_interaction(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(11, 7))
        scatter = plt.scatter(data=df, x='abs_pose', y='nk_score', c='norm', cmap='viridis_r', s=25, alpha=0.4, edgecolor='none')
        sns.regplot(data=df, x='abs_pose', y='nk_score', scatter=False, color='red', lowess=True, line_kws={'linewidth': 3})
        cbar = plt.colorbar(scatter)
        cbar.set_label('Magnitude do Embedding ($||z||$)')
        plt.axhline(analysis.k_neighbors, color='black', linestyle='--', alpha=0.3, label='Expectativa Ideal')
        plt.title("Interação Topológica: Pose, Magnitude e Hubness", fontweight='bold')
        plt.xlabel("Desvio de Pose Acumulado ($|Yaw| + |Pitch|$)")
        plt.ylabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.legend(frameon=True, facecolor='white', loc='upper right')
        if save_path: plt.savefig(save_path)
        plt.show()

    # --- NOVOS MÉTODOS DE ESTRATIFICAÇÃO E ESTUDO PAREADO ---

    def plot_stratified_hubness(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """Gera um Violin Plot do Nk score por faixas de Yaw para mostrar a densidade do risco."""
        if not analysis.stratification: return
        
        data = []
        for strat in analysis.stratification:
            for score in strat.nk_scores:
                data.append({'Faixa de Yaw': strat.label, 'Nk Score': score})
        
        df_strat = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_strat, x='Faixa de Yaw', y='Nk Score', inner="quart", palette="muted")
        plt.title("Distribuição de Hubness por Estratos de Pose (Yaw)")
        plt.ylabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.xticks(rotation=15)
        
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_paired_slope_impact(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """Gera um Slope Plot (gráfico de inclinação) para o estudo pareado (Frontal vs Não-Frontal)."""
        if not analysis.paired_study: return
        
        samples = analysis.paired_study.paired_samples
        plt.figure(figsize=(7, 8))
        
        for s in samples:
            color = 'red' if s.delta_nk > 0 else 'gray'
            alpha = 0.3 if s.delta_nk > 0 else 0.1
            plt.plot(['Frontal', 'Não-Frontal'], [s.f_nk, s.nf_nk], marker='o', color=color, alpha=alpha, linewidth=1)
        
        plt.title(f"Estudo Pareado: Impacto Individual da Pose no Hubness\n({analysis.paired_study.comparison_label})")
        plt.ylabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.grid(axis='x')
        
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_paired_delta_impostor_risk(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """Histograma do aumento do risco impostor (Delta Max Impostor) no estudo pareado."""
        if not analysis.paired_study: return
        
        deltas = [s.delta_max_impostor for s in analysis.paired_study.paired_samples]
        plt.figure(figsize=(9, 5))
        sns.histplot(deltas, kde=True, color='crimson', bins=30)
        plt.axvline(0, color='black', linestyle='--')
        plt.title("Impacto da Pose no Risco Biométrico: $\Delta$ Similaridade Impostora Máxima")
        plt.xlabel("Aumento do Score Impostor (Não-Frontal - Frontal)")
        plt.ylabel("Frequência (Identidades)")
        
        if save_path: plt.savefig(save_path)
        plt.show()

    # --- MÉTODOS DE ABLAÇÃO E QUALIDADE MANTIDOS ---

    def plot_ablation_study_topology(self, study: AblationStudyInfo, save_path: Optional[str] = None):
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(study.scenarios)))
        is_pose = "Frontal" in study.scenarios[0].scenario_name or "Pose" in study.scenarios[0].scenario_name
        title_suffix = "por Pose (Yaw)" if is_pose else "por Qualidade (Norma/FIQA)"
        for scenario, color in zip(study.scenarios, colors):
            nk_vals = [n for n in scenario.nk_scores if n > 0]
            sns.kdeplot(nk_vals, label=f"{scenario.scenario_name} (Máx $N_k$: {scenario.max_nk})", color=color, linewidth=2.5, fill=True, alpha=0.1)
        plt.axvline(study.k_neighbors, color='black', linestyle='--', alpha=0.6, label=f'Expectativa Ideal ($K={study.k_neighbors}$)')
        plt.title(f"Estudo de Ablação Topológica: Evolução do Hubness {title_suffix}")
        plt.xlabel(f"Grau de Hubness ($N_{{{study.k_neighbors}}}$)")
        plt.ylabel("Densidade Probabilística ($p(N_k)$)")
        plt.legend(frameon=True, facecolor='white')
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_hubness_vs_pose_delta(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='nk_score', y='mean_delta_pose_to_neighbors', s=15, alpha=0.2, color='darkgreen', edgecolor='none')
        sns.regplot(data=df, x='nk_score', y='mean_delta_pose_to_neighbors', scatter=False, color='black', lowess=True)
        plt.title("Tolerância Geométrica: Hubness vs Delta de Pose Médio")
        plt.xlabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.ylabel("$\Delta$ Pose Médio (Face vs Vizinhos Impostores)")
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_norm_vs_stochastic_quality(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        df = self._prepare_df(analysis)
        if df['serfiq'].sum() == 0: return
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x='serfiq', y='norm', alpha=0.3, s=15, color='navy', edgecolor='none')
        sns.regplot(data=df, x='serfiq', y='norm', scatter=False, color='red', lowess=True)
        plt.title("Consistência Métrica: SER-FIQ vs Norma L2")
        plt.xlabel("Escore SER-FIQ")
        plt.ylabel("Norma L2 ($||z||$)")
        if save_path: plt.savefig(save_path)
        plt.show()


    def plot_optimization_trends(self, study: HubnessOptimizationStudyInfo, save_path: Optional[str] = None):
        """Gera o gráfico de tendências triplas (Estrito -> Leve -> Perfil) por identidade."""
        if not study.samples: return
        
        plt.figure(figsize=(9, 7))
        categories = ['Estrito (<8°)', 'Leve (15-30°)', 'Perfil (>45°)']
        
        for s in study.samples:
            y_values = [s.s_nk, s.l_nk, s.p_nk]
            # Se o Nk caiu no 'Leve', usamos azul (otimização), senão cinza
            color = 'royalblue' if s.delta_nk_optimization < 0 else 'lightgray'
            alpha = 0.4 if color == 'royalblue' else 0.1
            plt.plot(categories, y_values, marker='o', color=color, alpha=alpha, linewidth=1)
        
        plt.title("Dinâmica de Pose: Otimização em Faixas Intermediárias", fontweight='bold')
        plt.ylabel("Grau de Hubness ($N_{20}$)")
        plt.grid(axis='y', alpha=0.3)
        
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_norm_ridgeline(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """Ridgeline Plot para comparar a 'Confiança' (Norma L2) entre as faixas de pose."""
        if not analysis.stratification: return
        
        data = []
        for strat in analysis.stratification:
            for n in strat.norms:
                data.append({'Faixa': strat.label, 'Norma L2': n})
        
        df = pd.DataFrame(data)
        plt.figure(figsize=(10, 6))
        
        # Gráfico de densidade sobreposto por faixa
        for label in df['Faixa'].unique():
            subset = df[df['Faixa'] == label]
            sns.kdeplot(subset['Norma L2'], label=label, fill=True, alpha=0.2)
            
        plt.title("Distribuição de Confiança (Norma L2) por Faixa de Pose")
        plt.xlabel("Magnitude do Embedding ($||z||$)")
        plt.ylabel("Densidade de Amostras")
        plt.legend(title="Pose (Yaw)")
        
        if save_path: plt.savefig(save_path)
        plt.show()

    def plot_quality_diagnostics(self, diagnostics: HubnessQualityDiagnosticsInfo, save_path: Optional[str] = None):
        """
        Gera Violin Plots para FIQA e Norma L2 estratificados por pose.
        Essencial para descartar viés de amostragem na tese do 'Sweet Spot'.
        """
        if not diagnostics.ranges: return
        
        # Prepara Dataframe para o Seaborn
        data = []
        for r in diagnostics.ranges:
            for q, n in zip(r.quality_values, r.norm_values):
                data.append({'Pose': r.label, 'FIQA': q, 'Norma L2': n})
        df = pd.DataFrame(data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Distribuição de Qualidade (FIQA)
        sns.violinplot(data=df, x='Pose', y='FIQA', inner="quart", palette="vlag", ax=ax1)
        ax1.set_title("Distribuição de Qualidade (FIQA) por Faixa de Pose")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15)
        
        # Subplot 2: Distribuição de Magnitude (Norma L2)
        sns.violinplot(data=df, x='Pose', y='Norma L2', inner="quart", palette="rocket", ax=ax2)
        ax2.set_title("Distribuição de Magnitude (Norma L2) por Faixa de Pose")
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=15)
        
        plt.tight_layout()
        if save_path: plt.savefig(save_path)
        plt.show()
    
    def plot_hubness_density(self, analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """
        Usa os dados consolidados no DTO HubnessAnalysisInfo para 
        renderizar a assinatura topológica (com Percentil 95).
        """
        nk_vals = [s.nk_score for s in analysis.samples]
        p95_nk = np.percentile(nk_vals, 95)
        
        plt.figure(figsize=(9, 6))
        sns.kdeplot(nk_vals, fill=True, color="teal", alpha=0.3, linewidth=2.5)
        
        plt.axvline(analysis.k_neighbors, color='black', linestyle='--', alpha=0.6, 
                    label=f'Expectativa Ideal ($K={analysis.k_neighbors}$)')
        
        # Agora incluindo o Percentil 95
        stats_text = (f"Máximo $N_k$: {analysis.max_nk}\n"
                      f"Percentil 95: {p95_nk:.0f}\n"
                      f"Média $N_k$: {analysis.mean_nk:.2f}\n"
                      f"Desvio Padrão: {analysis.std_nk:.2f}")
        
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                     ha='right', va='top', fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8))

        plt.title(f"Assinatura Topológica do Dataset ($N_{{{analysis.k_neighbors}}}$)")
        plt.xlabel(f"Grau de Hubness ($N_{{{analysis.k_neighbors}}}$)")
        plt.ylabel("Densidade Relativa $p(N_k)$")
        plt.legend()
        
        if save_path: plt.savefig(save_path)
        plt.show()
    
    def plot_multiple_hubness_densities(
        self, 
        analyses: List[HubnessAnalysisInfo], 
        labels: List[str], 
        save_path: Optional[str] = None
    ):
        """
        Sobrepõe a assinatura topológica de múltiplos datasets no mesmo gráfico.
        Ideal para comparar o Antes/Depois (Ex: Base Bruta vs. Base Filtrada por Qualidade).
        """
        if len(analyses) != len(labels):
            raise ValueError("A quantidade de análises deve ser igual à quantidade de labels.")

        plt.figure(figsize=(10, 6))
        
        # Paleta de cores dinâmica e acessível
        colors = sns.color_palette("husl", len(analyses))
        
        # Assume que todos os testes usaram o mesmo K para a linha ideal
        k_neighbors = analyses[0].k_neighbors 
        
        for i, analysis in enumerate(analyses):
            nk_vals = [s.nk_score for s in analysis.samples]
            p95_nk = np.percentile(nk_vals, 95)
            
            # Construindo uma legenda rica para cada curva
            label_text = (f"{labels[i]} "
                          f"(Média: {analysis.mean_nk:.1f} | P95: {p95_nk:.0f} | Max: {analysis.max_nk})")
            
            sns.kdeplot(nk_vals, fill=True, color=colors[i], alpha=0.2, linewidth=2, label=label_text)
            
        # Linha de expectativa ideal matemática
        plt.axvline(k_neighbors, color='black', linestyle='--', alpha=0.6, 
                    label=f'Expectativa Ideal ($K={k_neighbors}$)')
        
        plt.title(f"Comparativo de Assinaturas Topológicas ($N_{{{k_neighbors}}}$)")
        plt.xlabel(f"Grau de Hubness ($N_{{{k_neighbors}}}$)")
        plt.ylabel("Densidade Relativa $p(N_k)$")
        
        # Ajusta a legenda para não cobrir o gráfico
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=9)
        plt.tight_layout()
        
        if save_path: plt.savefig(save_path)
        plt.show()
    
    def plot_hubness_by_norm_strata(self, balanced_analysis: HubnessAnalysisInfo, save_path: Optional[str] = None):
        """
        Gera o gráfico comparativo do Experimento de Qualidade (Norma L2).
        Utiliza um layout de auditoria com blocos de métricas agrupados e isolados.
        Versão Vazada (Sem preenchimento) com caixas de legenda na cor das curvas.
        """
        from collections import defaultdict
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.lines import Line2D

        # 1. Agrupar os DTOs por identidade
        samples_by_id = defaultdict(list)
        for s in balanced_analysis.samples:
            if s.identity is not None:
                samples_by_id[s.identity].append(s)

        best_nk, best_norms = [], []
        median_nk, median_norms = [], []
        worst_nk, worst_norms = [], []
        
        all_nk = [s.nk_score for s in balanced_analysis.samples]
        all_norms = [s.norm for s in balanced_analysis.samples]

        # 2. Extrai Top, Middle e Bottom de cada pessoa
        for identity, samples in samples_by_id.items():
            if len(samples) >= 3:
                sorted_samples = sorted(samples, key=lambda x: x.norm, reverse=True)
                
                best_nk.append(sorted_samples[0].nk_score)
                best_norms.append(sorted_samples[0].norm)
                
                median_nk.append(sorted_samples[len(sorted_samples) // 2].nk_score)
                median_norms.append(sorted_samples[len(sorted_samples) // 2].norm)
                
                worst_nk.append(sorted_samples[-1].nk_score)
                worst_norms.append(sorted_samples[-1].norm)

        total_ids = len(best_nk)
        total_images = len(all_nk)

        # 3. Renderização do Gráfico
        plt.figure(figsize=(13, 8)) 

        # A) Renderiza as curvas SEM PREENCHIMENTO (fill=False) e opacidade alta
        # Alterei a cor da base total para um cinza escuro/preto para bater com sua ideia
        sns.kdeplot(all_nk, fill=False, color="#404040", alpha=0.7, linewidth=3, linestyle="--")
        sns.kdeplot(best_nk, fill=False, color="forestgreen", alpha=0.9, linewidth=2.5)
        sns.kdeplot(median_nk, fill=False, color="goldenrod", alpha=0.9, linewidth=2.5)
        sns.kdeplot(worst_nk, fill=False, color="crimson", alpha=0.9, linewidth=2.5)

        k = balanced_analysis.k_neighbors
        plt.axvline(k, color='black', linestyle=':', alpha=0.8)

        # Título e Eixos
        plt.title(f"Decomposição do Hubness por Magnitude do Embedding (Norma L2)\n"
                  f"Dataset Balanceado: {total_images} Imagens | {total_ids} Identidades", 
                  pad=15, fontsize=14, fontweight='bold')
        plt.xlabel(f"Grau de Hubness ($N_{{{k}}}$)", fontsize=12)
        plt.ylabel("Densidade Relativa $p(N_k)$", fontsize=12)
        
        # =====================================================================
        # LAYOUT DE AUDITORIA: Blocos de Métricas Customizados (fora do gráfico)
        # =====================================================================
        fig = plt.gcf()
        fig.subplots_adjust(right=0.74) # Margem para as caixas
        
        def format_pericial_box_text(name, nk_list, norm_list):
            mean_nk = np.mean(nk_list)
            std_nk = np.std(nk_list)
            p95_nk = np.percentile(nk_list, 95)
            max_nk = np.max(nk_list)
            
            mean_norm = np.mean(norm_list)
            std_norm = np.std(norm_list)
            min_norm = np.min(norm_list)
            max_norm = np.max(norm_list)
            n_samples = len(nk_list)
            
            return (f"$\\mathbf{{{name}}}$ (n={n_samples})\n"
                    f"Média $N_k$: {mean_nk:.1f} | DP $N_k$: {std_nk:.1f} | P95: {p95_nk:.0f} | Máx: {max_nk}\n"
                    f"Norma: {mean_norm:.1f} ($\\pm${std_norm:.1f}) | Min {min_norm:.1f} | Max {max_norm:.1f}")

        box_y_start = 0.84
        box_y_padding = 0.16
        box_x_start = 0.77

        # Definições: Nome, Nk, Norma, Cor, Estilo da Linha, Espessura da Linha
        strata_definitions = [
            ("Base Balanceada Total", all_nk, all_norms, "#404040", "--", 3.0),
            ("Maior Norma L2 (HQ)", best_nk, best_norms, "forestgreen", "-", 2.5),
            ("Norma L2 Mediana", median_nk, median_norms, "goldenrod", "-", 2.5),
            ("Menor Norma L2 (LQ)", worst_nk, worst_norms, "crimson", "-", 2.5)
        ]

        for i, (name, nk_vals, norm_vals, color, linestyle, lw) in enumerate(strata_definitions):
            box_text = format_pericial_box_text(name, nk_vals, norm_vals)
            current_y = box_y_start - (i * box_y_padding)
            
            # A MÁGICA ESTÁ AQUI: edgecolor=color e linewidth=2.0
            props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=1.0, linewidth=2.0)
            
            fig.text(box_x_start, current_y, box_text, fontsize=9.5, transform=fig.transFigure,
                     bbox=props, verticalalignment='top', horizontalalignment='left')
            
            # Adiciona a linha de amostra na legenda (mantive para reforçar se é tracejado ou liso)
            line = Line2D([box_x_start + 0.015, box_x_start + 0.035], [current_y - 0.012, current_y - 0.012], 
                          transform=fig.transFigure, color=color, linestyle=linestyle, linewidth=lw)
            fig.lines.append(line)

        ideal_y = box_y_start - (4 * box_y_padding)
        fig.text(box_x_start, ideal_y + 0.02, f"$\\dots\\dots$ Ideal ($K={k}$)", 
                 fontsize=10, transform=fig.transFigure, fontweight='bold')

        if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_top_hubs_gallery(
        analysis, 
        dataset, 
        top_n: int = 10, 
        num_cols: int = 5, 
        save_path: Optional[str] = None
    ):
        """
        Renderiza um grid com as imagens reais dos maiores Hubs do dataset.
        Cruza o DTO de análise com as imagens armazenadas no EmbeddingDataset.
        """
        print(f"Buscando os {top_n} maiores hubs na base...")
        
        # 1. Ordena os samples pelo grau de Hubness (Nk) em ordem decrescente
        sorted_samples = sorted(analysis.samples, key=lambda s: s.nk_score, reverse=True)
        top_samples = sorted_samples[:top_n]
        
        # 2. Configuração do Grid
        num_rows = math.ceil(top_n / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.5, num_rows * 4.5))
        
        # Transforma axes em um array 1D para facilitar o loop, mesmo se for só 1 linha
        if num_rows == 1 and num_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # 3. Plotagem das imagens
        for i, ax in enumerate(axes):
            if i < top_n:
                sample = top_samples[i]
                # Recupera a imagem original usando o índice armazenado no sample
                item = dataset.items[sample.index]
                
                # Tenta pegar a imagem do array. Se não existir, avisa.
                img = item.face_image
                
                if img is not None:
                    # Se a imagem for do OpenCV (BGR), costuma precisar inverter para RGB no Matplotlib
                    # plt.imshow(img[..., ::-1]) # Descomente se as cores ficarem azuis/estranhas
                    ax.imshow(img)
                else:
                    # Placeholder caso a imagem não esteja carregada na memória
                    ax.text(0.5, 0.5, 'Imagem\nIndisponível', ha='center', va='center', fontsize=12)
                    ax.set_facecolor('#eeeeee')
                
                ax.axis('off')
                
                # Monta a ficha criminal do Hub
                # Usando a property genérica quality_score que você tem no DTO
                fiqa_val = item.quality_score
                fiqa_str = f"{fiqa_val:.2f}" if fiqa_val is not None else "N/A"
                
                title = (
                    f"Rank #{i+1}\n"
                    f"ID: {sample.identity}\n"
                    f"Hubness ($N_k$): {sample.nk_score}\n"
                    f"Norma L2: {sample.norm:.1f}\n"
                    f"Qualidade: {fiqa_str}"
                )
                
                # Pinta o fundo do título de vermelho se for um Hub extremo (ex: Nk > 50)
                title_color = 'darkred' if sample.nk_score > (analysis.k_neighbors * 2) else 'black'
                ax.set_title(title, fontsize=10, color=title_color, pad=10, fontweight='bold')
                
            else:
                # Oculta os eixos que sobrarem vazios no grid
                ax.axis('off')

        plt.suptitle(f"Galeria de Investigação: Top {top_n} Maiores Hubs do Dataset", 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            
        plt.show()
    
    def plot_norm_vs_nk_lowess(self, analysis: 'HubnessAnalysisInfo', save_path: Optional[str] = None):
        """
        Gera um gráfico de densidade hexagonal cruzando a Magnitude (Norma L2) 
        com o Grau de Hubness (Nk), sobreposto com uma curva de tendência LOWESS.
        Revela o "sweet spot" de qualidade e os extremos anômalos.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        try:
            import statsmodels.api as sm
        except ImportError:
            print("Erro: A biblioteca 'statsmodels' é necessária para a curva LOWESS.")
            print("Instale usando: pip install statsmodels")
            return

        print("Preparando dados para o gráfico Hexbin + LOWESS...")
        
        # 1. Extração bruta
        norms = np.array([s.norm for s in analysis.samples])
        nks = np.array([s.nk_score for s in analysis.samples])
        
        # 2. Configuração do Gráfico
        plt.figure(figsize=(12, 7))
        
        # 3. Hexbin para a Densidade da Nuvem (Excelente para Big Data)
        hb = plt.hexbin(norms, nks, gridsize=60, cmap='viridis', mincnt=1, alpha=0.7)
        cb = plt.colorbar(hb, label='Densidade de Imagens')
        
        # 4. Cálculo do LOWESS (Suavização Local)
        # Proteção de performance: LOWESS em 150k pontos trava a máquina. 10k é instantâneo e fiel.
        sample_size = min(len(norms), 10000)
        if len(norms) > sample_size:
            idx = np.random.choice(len(norms), sample_size, replace=False)
            n_sample = norms[idx]
            nk_sample = nks[idx]
        else:
            n_sample = norms
            nk_sample = nks
            
        print("Calculando curva de tendência não-linear...")
        # frac=0.2 significa que cada ponto da linha considera os 20% vizinhos mais próximos
        lowess_fit = sm.nonparametric.lowess(nk_sample, n_sample, frac=0.2)
        
        # O retorno é uma matriz ordenada [X_suavizado, Y_suavizado]
        lowess_x = lowess_fit[:, 0]
        lowess_y = lowess_fit[:, 1]
        
        # Plota a linha de tendência sobre a nuvem
        plt.plot(lowess_x, lowess_y, color='crimson', linewidth=3, 
                 label='Tendência LOWESS (Comportamento Real)')
        
        # 5. Linha de Referência Ideal
        k = analysis.k_neighbors
        plt.axhline(k, color='black', linestyle=':', linewidth=2, label=f'Expectativa Ideal ($N_k = {k}$)')
        
        # 6. Perfumaria Pericial
        plt.title(f"Mapeamento Não-Linear: Magnitude do Embedding vs Hubness\n"
                  f"Dataset: {len(norms)} Imagens Avaliadas", pad=15, fontsize=14, fontweight='bold')
        plt.xlabel("Norma L2 Bruta da Imagem", fontsize=12)
        plt.ylabel(f"Grau de Hubness ($N_{{{k}}}$)", fontsize=12)
        
        # Recorta o eixo Y para não ser arruinado por um único outlier de Nk=800, 
        # focando onde 99.5% dos dados vivem, mas dando uma margem para ver a curva subir.
        p99_nk = np.percentile(nks, 99.5)
        plt.ylim(-2, p99_nk + 15)
        
        # Cria uma caixinha de legenda bonita
        plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='gray', framealpha=0.9)
        plt.tight_layout()
        
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def plot_top_outliers_analysis(self, analysis: 'HubnessAnalysisInfo', dataset: 'EmbeddingDataset', save_path: Optional[str] = None):
        """
        Analisa os outliers extremos de Hubness (Top 1% de Nk) contra o resto do dataset.
        Compara duas distribuições cruciais: 
        1. Magnitude (Norma L2 - Proxy de Qualidade)
        2. Tipicidade (Similaridade com o Rosto Médio Global)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        print("Calculando Tipicidade Global e isolando o Top 1% dos Hubs...")

        # 1. Isola as pontuações de Nk e define o corte do Top 1% (Percentil 99)
        nks = np.array([s.nk_score for s in analysis.samples])
        threshold_99 = np.percentile(nks, 99)
        
        top_mask = nks >= threshold_99
        rest_mask = ~top_mask
        
        top_count = np.sum(top_mask)
        rest_count = np.sum(rest_mask)

        # 2. Extração de Normas
        norms = np.array([s.norm for s in analysis.samples])

        # 3. Cálculo de Tipicidade (Cosseno com o Centróide Global)
        raw_matrix = dataset.matrix
        row_norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1e-10 # Proteção contra div por 0
        normed_matrix = raw_matrix / row_norms
        
        # O vetor médio do dataset inteiro (O "Rosto Mais Genérico")
        global_mean_vector = np.mean(normed_matrix, axis=0)
        global_mean_vector /= np.linalg.norm(global_mean_vector)
        
        # Tipicidade = Similaridade de Cosseno com o vetor médio global
        typicalities = np.dot(normed_matrix, global_mean_vector)

        # 4. Renderização dos Gráficos Comparativos (1x2 Subplots)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Autópsia Forense: Top 1% Maiores Hubs ($N_k \\geq {threshold_99:.0f}$)\n"
                     f"Analisando {top_count} Outliers vs {rest_count} Imagens Normais", 
                     fontsize=15, fontweight='bold', y=1.05)

        # Helper para legendas
        def get_label(name, data):
            return f"{name}\n(Média: {np.mean(data):.2f} | Mediana: {np.median(data):.2f})"

        # --- Subplot 1: Distribuição da Norma L2 ---
        sns.kdeplot(norms[rest_mask], fill=True, color="gray", alpha=0.3, linewidth=2, 
                    label=get_label("Resto da Base (99%)", norms[rest_mask]), ax=axes[0])
        sns.kdeplot(norms[top_mask], fill=True, color="crimson", alpha=0.5, linewidth=2.5, 
                    label=get_label("Top 1% Hubs", norms[top_mask]), ax=axes[0])
        
        axes[0].set_title("Decomposição de Qualidade (Norma L2)", fontweight='bold', pad=10)
        axes[0].set_xlabel("Norma L2 Bruta")
        axes[0].set_ylabel("Densidade")
        axes[0].legend(loc='upper right')

        # --- Subplot 2: Distribuição de Tipicidade ---
        sns.kdeplot(typicalities[rest_mask], fill=True, color="gray", alpha=0.3, linewidth=2, 
                    label=get_label("Resto da Base (99%)", typicalities[rest_mask]), ax=axes[1])
        sns.kdeplot(typicalities[top_mask], fill=True, color="darkorange", alpha=0.5, linewidth=2.5, 
                    label=get_label("Top 1% Hubs", typicalities[top_mask]), ax=axes[1])
        
        axes[1].set_title("Decomposição de Tipicidade (Similaridade com Centróide)", fontweight='bold', pad=10)
        axes[1].set_xlabel("Score de Tipicidade (Cosseno)")
        axes[1].set_ylabel("Densidade")
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        if save_path: plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()