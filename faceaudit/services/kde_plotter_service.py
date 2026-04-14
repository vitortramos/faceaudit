from pathlib import Path
import matplotlib.figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

from faceaudit.models.kde_visualization_data import KDEVisualizationData

class KDEPlotterService:
    """
    Serviço responsável por gerar e salvar gráficos de distribuição do KDE (Modo Headless).
    Isola a lógica gráfica (Matplotlib) do Manager e opera 100% em background (Thread-safe).
    """
    
    def save_distribution_plot(self, viz_data: KDEVisualizationData, save_path: Path, title_suffix: str = "") -> Path:
        """
        Gera o gráfico de Genuínos vs Impostores e salva no disco.
        """
        # API Orientada a Objetos pura (Evita vazamento de memória e popups de janelas)
        fig = matplotlib.figure.Figure(figsize=(10, 6))
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # 1. Plota os Histogramas (Barras ao fundo)
        if viz_data.hist_gen_counts is not None and viz_data.hist_gen_bins is not None:
            widths_gen = viz_data.hist_gen_bins[1:] - viz_data.hist_gen_bins[:-1]
            centers_gen = (viz_data.hist_gen_bins[:-1] + viz_data.hist_gen_bins[1:]) / 2
            ax.bar(centers_gen, viz_data.hist_gen_counts, width=widths_gen, 
                   color='#2ca02c', alpha=0.3, label='Distribuição (Gen)')

        if viz_data.hist_imp_counts is not None and viz_data.hist_imp_bins is not None:
            widths_imp = viz_data.hist_imp_bins[1:] - viz_data.hist_imp_bins[:-1]
            centers_imp = (viz_data.hist_imp_bins[:-1] + viz_data.hist_imp_bins[1:]) / 2
            ax.bar(centers_imp, viz_data.hist_imp_counts, width=widths_imp, 
                   color='#d62728', alpha=0.3, label='Distribuição (Imp)')

        # 2. Plota as Linhas do KDE (Linhas sólidas na frente)
        if viz_data.x_grid is not None:
            if viz_data.kde_gen_density is not None:
                ax.plot(viz_data.x_grid, viz_data.kde_gen_density, color='#155724', 
                        label='KDE (Gen)', linewidth=2.5)
                
            if viz_data.kde_imp_density is not None:
                ax.plot(viz_data.x_grid, viz_data.kde_imp_density, color='#721c24', 
                        label='KDE (Imp)', linewidth=2.5)

        # 3. Plota o Threshold (Lê do dicionário dinâmico de stats)
        # Procura chaves comuns de limiar (EER ou threshold genérico)
        threshold = viz_data.stats.get("eer_threshold") or viz_data.stats.get("threshold")
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', 
                       label=f'Limiar Sugerido ({threshold:.3f})', linewidth=2)

        # 4. Estilização Forense
        titulo = "Distribuição de Densidade Probabilística (KDE)"
        if title_suffix:
            titulo += f"\n{title_suffix}"
            
        ax.set_title(titulo, fontsize=14, fontweight='bold')
        ax.set_xlabel("Escore de Similaridade (RAW)", fontsize=12)
        ax.set_ylabel("Densidade", fontsize=12)
        
        # Ajusta a legenda para ficar fora da área principal se necessário, 
        # mas upper right costuma ser seguro.
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(True, linestyle=':', alpha=0.7)

        # 5. Persistência
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        # Limpeza agressiva da memória
        fig.clf()
        
        return save_path