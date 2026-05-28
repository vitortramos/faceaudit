from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class ValidationPlotData:
    """
    Estruturas de dados pré-computadas contendo as coordenadas prontas para
    injeção no motor gráfico (Matplotlib/PyQtGraph), evitando recálculo na View.
    """
    # --- Para Tippett Plot e Histogramas ---
    genuine_log_lrs: np.ndarray = field(repr=False)
    impostor_log_lrs: np.ndarray = field(repr=False)

    # --- Para Curva DET (Detection Error Tradeoff) ---
    fpr_array: np.ndarray = field(repr=False) # False Positive Rates (eixo X)
    fnr_array: np.ndarray = field(repr=False) # False Negative Rates (eixo Y)

    # --- Para Gráfico de Calibração (ECE - Opcional) ---
    ece_prior_log_odds: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    ece_values: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    ece_min_values: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # --- Para Curva de Calibração (raw_score × log-LR) ---
    # Mesmos índices que genuine_log_lrs / impostor_log_lrs — i-ésimo
    # elemento de raw_scores corresponde ao i-ésimo log-LR. Permite
    # plotar a função de mapeamento do calibrador. Empty quando o
    # pickle foi gerado por versão anterior do executor (pre-curva);
    # nesse caso o draw cai em placeholder gracioso.
    genuine_raw_scores: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    impostor_raw_scores: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # --- Para Curva de Calibração v2 (pós-afim Brümmer + resíduo) ---
    # Grid pré-computado de f(x) (calibrador atual) e da recalibração
    # afim Brümmer-Doddington: slope·logLR + intercept. O slope/intercept
    # operam sobre o log-LR atual, não sobre o raw_score; por isso a
    # "pós-afim" plotada no plano raw×logLR só é reta quando f(x) é
    # linear (caso LogReg). Para KDE/BiGauss, é curva.
    #
    # Empty quando o pickle foi gerado por versão anterior do executor
    # (sem grid pré-computado); o draw faz fallback pra single-panel.
    cal_curve_grid_x: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_llr_y: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_affine_y: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_affine_delta_y: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    # Raw_score onde a curva pós-afim cruza log-LR=0 (LR=1, neutralidade
    # forense). None quando: (a) não houve cruzamento na faixa observada,
    # ou (b) grid não foi computado (compat). UI plota linha vertical
    # tracejada nesse x quando presente.
    cal_curve_zero_raw_score: Optional[float] = field(default=None, repr=False)
    # Rug plot subsampled (cap ~2000 por classe) — apenas raw_scores.
    cal_curve_rug_gen_x: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_rug_imp_x: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    # --- Curva de Calibração: dimensionalidade + visualização 2D ---
    # `cal_curve_dim` indica quantos sistemas alimentam o calibrador:
    #   1 → calibrador 1D, plot 2-panel raw×logLR (atual).
    #   2 → calibrador 2D (typ. SCORE_VECTORIZATION com 2 sistemas);
    #       plot vira heatmap log-LR no plano (x₀, x₁) com scatter +
    #       contornos de decisão.
    #   3+ → suprime visualização raw→logLR (1D collapse é enganoso);
    #       plot mostra placeholder textual apontando pro Diagnóstico
    #       Afim em Escala log-LR (que opera em qualquer D).
    # Default 1 mantém retrocompat com pickles antigos.
    cal_curve_dim: int = field(default=1, repr=False)
    # Labels dos eixos pra D ≥ 2 — vem de `score_vector_source_resources`
    # do dataset CALIBRATED_SCORES (nomes dos modelos de similaridade).
    # Empty pra D=1.
    cal_curve_axis_labels: List[str] = field(default_factory=list, repr=False)

    # --- Curva de Calibração 2D: heatmap log-LR no plano (x₀, x₁) ---
    # Grid regular pré-computado via interpolação 2D
    # (LinearNDInterpolator) dos observados (x₀, x₁, log_lr). Pontos
    # fora do casco convexo viram NaN — matplotlib renderiza como
    # masked area. Empty quando D != 2 ou pickle pré-2D.
    cal_curve_2d_grid_x0: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_2d_grid_x1: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_2d_grid_log_lr: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    # Scatter subsampled (cap ~5000 por classe) pra overlay sobre o
    # heatmap. Mantém densidade visual sem matar render.
    cal_curve_2d_gen_x0: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_2d_gen_x1: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_2d_imp_x0: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    cal_curve_2d_imp_x1: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)