from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class KDEVisualizationData:
    """
    Dados prontos para plotagem de:
    - Histograma (mesma escala)
    - Curvas KDE (avaliadas no espaço RAW)
    """

    # Grid comum em score RAW [-1, 1]
    x_grid: np.ndarray

    # Densidades KDE avaliadas no espaço RAW
    kde_gen_density: np.ndarray
    kde_imp_density: np.ndarray

    # Histogramas normalizados (density=True)
    hist_gen_counts: np.ndarray
    hist_gen_bins: np.ndarray

    hist_imp_counts: np.ndarray
    hist_imp_bins: np.ndarray

    # Estatísticas
    stats: Dict[str, Any]
