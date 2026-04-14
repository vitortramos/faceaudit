from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from faceaudit.models.gaussianity_analysis_info import GaussianityAnalysisInfo

class GaussianityPlotter:
    """
    Serviço de renderização para laudos de Gaussianidade (Q-Q Plots).
    """
    
    def plot_mahalanobis_qq(self, info: GaussianityAnalysisInfo, save_path: str = None):
        """
        Gera o Q-Q Plot comparando a nuvem real contra a Gaussiana perfeita teórica.
        """
        plt.figure(figsize=(10, 10))
        
        # Plota os pontos reais (Eixo X = Teórico Perfeito, Eixo Y = Realidade da sua Rede)
        # Subamostragem apenas visual para o matplotlib não engasgar com 150k pontos
        step = max(1, len(info.mahalanobis_sq_distances) // 10000) 
        
        plt.scatter(info.theoretical_chi2_quantiles[::step], 
                    info.mahalanobis_sq_distances[::step], 
                    alpha=0.4, color='darkblue', edgecolors='none', s=15,
                    label='Imagens do Dataset (Amostra Visual)')
        
        # Plota a Linha de Identidade (A "Perfeição" Gaussiana)
        max_val = max(np.max(info.theoretical_chi2_quantiles), np.max(info.mahalanobis_sq_distances))
        plt.plot([0, max_val], [0, max_val], color='crimson', linewidth=2.5, linestyle='--', 
                 label='Gaussiana Multivariada Ideal (Referência)')
        
        # Anotações Periciais
        plt.title(f"Q-Q Plot de Mahalanobis: Avaliação de Gaussianidade\n"
                  f"Dataset: {info.dataset_name} | {info.dimensions} Dimensões", 
                  pad=15, fontsize=14, fontweight='bold')
                  
        plt.xlabel(f"Quantis Teóricos ($\chi^2_{{{info.dimensions}}}$)", fontsize=12)
        plt.ylabel("Distâncias de Mahalanobis ao Quadrado (Nuvem Real)", fontsize=12)
        
        # Caixa de Métricas
        box_text = (f"$\\mathbf{{Diagnóstico da Nuvem}}$\n"
                    f"Amostras: {info.sample_size}\n"
                    f"Massa na Elipse 95%: {info.pct_inside_95_ellipse:.1f}% (Ideal: 95%)")
                    
        props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
        plt.gca().text(0.05, 0.95, box_text, transform=plt.gca().transAxes, fontsize=11,
                       verticalalignment='top', bbox=props)
        
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Limita os eixos para focar nos dados (evita que um outlier de dist=10000 quebre o gráfico)
        plt.xlim(0, np.percentile(info.theoretical_chi2_quantiles, 99.9) * 1.2)
        plt.ylim(0, np.percentile(info.mahalanobis_sq_distances, 99.9) * 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()
    
    def plot_multiple_mahalanobis_qq(self, infos: List[GaussianityAnalysisInfo], title: str = "Comparativo de Gaussianidade (Q-Q Plot)", save_path: Optional[str] = None):
        """
        Plota múltiplas curvas de Mahalanobis no mesmo gráfico para análise comparativa.
        Útil para cruzar as distorções geométricas entre datasets de diferentes qualidades (ex: HQ vs LQ).
        """
        if not infos:
            print("Aviso: Nenhuma informação de gaussianidade fornecida para plotagem.")
            return

        plt.figure(figsize=(12, 10))
        
        # Gera cores distintas dinamicamente com base na quantidade de laudos
        # Se forem até 4, usamos as cores periciais que adotamos antes; senão, delegamos ao seaborn
        if len(infos) <= 4:
            colors = ["#404040", "forestgreen", "goldenrod", "crimson"]
        else:
            colors = sns.color_palette("tab10", len(infos))
            
        global_max_x = 0
        global_max_y = 0
        all_x = []
        all_y = []

        # Itera sobre todos os DTOs para extrair e plotar
        for idx, info in enumerate(infos):
            step = max(1, len(info.mahalanobis_sq_distances) // 10000) 
            
            x_vals = info.theoretical_chi2_quantiles[::step]
            y_vals = info.mahalanobis_sq_distances[::step]
            
            global_max_x = max(global_max_x, np.max(x_vals))
            global_max_y = max(global_max_y, np.max(y_vals))
            
            all_x.extend(x_vals)
            all_y.extend(y_vals)
            
            # Incorpora o Diagnóstico (Massa na Elipse) direto na legenda para limpeza visual
            label_text = f"{info.dataset_name} (Massa 95%: {info.pct_inside_95_ellipse:.1f}%)"
            
            plt.scatter(x_vals, y_vals, alpha=0.5, color=colors[idx % len(colors)], 
                        edgecolors='none', s=15, label=label_text)

        # Plota a Linha de Identidade (A "Perfeição" Gaussiana) cobrindo todos os dados
        max_val = max(global_max_x, global_max_y)
        plt.plot([0, max_val], [0, max_val], color='black', linewidth=2.5, linestyle='--', 
                 label='Gaussiana Multivariada Ideal (Referência)')

        # Anotações e perfumaria
        dimensions = infos[0].dimensions
        total_samples = sum(info.sample_size for info in infos)
        
        plt.title(f"{title}\n"
                  f"Comparativo de {len(infos)} Datasets | Amostras Totais: {total_samples} | {dimensions} Dimensões", 
                  pad=15, fontsize=14, fontweight='bold')
                  
        plt.xlabel(f"Quantis Teóricos ($\chi^2_{{{dimensions}}}$)", fontsize=12)
        plt.ylabel("Distâncias de Mahalanobis ao Quadrado (Nuvem Real)", fontsize=12)
        
        # Organiza a legenda (se forem muitos itens, joga para fora ou mantém dentro no canto)
        plt.legend(loc='lower right', fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # Limita os eixos globalmente com base no P99.9 de TODAS as massas reunidas
        plt.xlim(0, np.percentile(all_x, 99.9) * 1.2)
        plt.ylim(0, np.percentile(all_y, 99.9) * 1.2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()