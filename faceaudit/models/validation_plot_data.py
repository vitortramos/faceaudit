from dataclasses import dataclass, field
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