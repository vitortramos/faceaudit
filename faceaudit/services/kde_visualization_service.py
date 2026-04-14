from __future__ import annotations

from typing import Tuple
import numpy as np

from faceaudit.models.kde_training_artifacts import KDETrainingArtifacts
from faceaudit.models.kde_visualization_data import KDEVisualizationData
from faceaudit.strategies.kde_strategy import KDEStrategy


class KDEVisualizationService:
    """
    Serviço responsável por preparar dados para visualização:
    - Histograma + KDE no espaço RAW (cosine)
    - Mesma escala de densidade
    - Cálculo de Limiar (Threshold) via Equal Error Rate (EER)
    """

    def build_visualization_data(
        self,
        artifacts: KDETrainingArtifacts,
        strategy: KDEStrategy,
        *,
        n_grid: int = 1000,
        n_bins: int = 50,
        x_min: float = -0.25,
        x_max: float = 1.0
    ) -> KDEVisualizationData:
        """
        Constrói os dados necessários para plot único:
        Histograma + Curvas KDE no mesmo eixo.

        Parâmetros:
        - n_grid: resolução da curva KDE
        - n_bins: número de bins do histograma
        - x_min/x_max: limites do eixo X
        """

        if not strategy.fitted:
            raise RuntimeError("Estratégia KDE não treinada.")

        # ===============================
        # 1. GRID RAW E DENSIDADES
        # ===============================
        x_grid = np.linspace(x_min, x_max, n_grid).reshape(-1, 1)

        # Avaliação no espaço Fisher (modelo foi treinado lá)
        x_fisher = strategy._to_fisher_space(x_grid)

        log_pdf_gen = strategy.kde_gen.score_samples(x_fisher)
        log_pdf_imp = strategy.kde_imp.score_samples(x_fisher)

        kde_gen_density_fisher = np.exp(log_pdf_gen)
        kde_imp_density_fisher = np.exp(log_pdf_imp)

        # --- CORREÇÃO MATEMÁTICA: Jacobiano da Transformação de Fisher ---
        x_grid_safe = np.clip(x_grid, -0.99999, 0.99999) 
        derivative = 1.0 / (1.0 - x_grid_safe**2)

        # Traz a densidade de volta para a escala do espaço RAW
        kde_gen_density = kde_gen_density_fisher * derivative.flatten()
        kde_imp_density = kde_imp_density_fisher * derivative.flatten()

        # ===============================
        # 2. CÁLCULO DO EER / THRESHOLD
        # ===============================
        # Tamanho do passo de integração
        dx = (x_max - x_min) / (n_grid - 1)
        
        # CDF (Cumulative Distribution Function) via integração numérica simples
        cdf_gen = np.cumsum(kde_gen_density) * dx
        cdf_imp = np.cumsum(kde_imp_density) * dx
        
        # Normalização de segurança (garante que a área total é cravada em 1.0)
        cdf_gen /= cdf_gen[-1]
        cdf_imp /= cdf_imp[-1]
        
        # FNMR(t): Fração de genuínos abaixo do limiar (Falsa Rejeição)
        fnmr = cdf_gen
        # FMR(t): Fração de impostores acima do limiar (Falsa Aceitação)
        fmr = 1.0 - cdf_imp
        
        # Encontra o índice onde as duas taxas se cruzam (menor diferença absoluta)
        diff = np.abs(fnmr - fmr)
        eer_idx = np.argmin(diff)
        
        eer_threshold = float(x_grid[eer_idx][0])
        eer_value = float((fnmr[eer_idx] + fmr[eer_idx]) / 2.0)

        # ===============================
        # 3. HISTOGRAMAS (RAW)
        # ===============================
        hist_gen_counts, hist_gen_bins = np.histogram(
            artifacts.genuines_scores_raw,
            bins=n_bins,
            range=(x_min, x_max),
            density=True
        )

        hist_imp_counts, hist_imp_bins = np.histogram(
            artifacts.impostors_scores_raw,
            bins=n_bins,
            range=(x_min, x_max),
            density=True
        )

        # ===============================
        # 4. Estatísticas
        # ===============================
        stats = {
            "n_genuines": artifacts.n_genuines,
            "n_impostors": artifacts.n_impostors,
            "mean_gen": float(np.mean(artifacts.genuines_scores_raw)),
            "std_gen": float(np.std(artifacts.genuines_scores_raw)),
            "mean_imp": float(np.mean(artifacts.impostors_scores_raw)),
            "std_imp": float(np.std(artifacts.impostors_scores_raw)),
            "bw_gen_fisher": strategy.metadata.get("bw_gen_fisher"),
            "bw_imp_fisher": strategy.metadata.get("bw_imp_fisher"),
            "eer_threshold": eer_threshold,   # A mágica acontece aqui!
            "eer_value": eer_value
        }

        return KDEVisualizationData(
            x_grid=x_grid.flatten(),
            kde_gen_density=kde_gen_density,
            kde_imp_density=kde_imp_density,
            hist_gen_counts=hist_gen_counts,
            hist_gen_bins=hist_gen_bins,
            hist_imp_counts=hist_imp_counts,
            hist_imp_bins=hist_imp_bins,
            stats=stats
        )