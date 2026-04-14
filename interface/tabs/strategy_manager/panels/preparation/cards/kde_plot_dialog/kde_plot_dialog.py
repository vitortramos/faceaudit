# faceaudit/interface/tabs/strategy_manager/panels/preparation/cards/kde_plot_dialog/kde_plot_dialog.py

import numpy as np
from PyQt6 import QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from faceaudit.models.kde_visualization_data import KDEVisualizationData


class KDEPlotDialog(QtWidgets.QDialog):
    """
    Janela para visualizar:
      - Histograma + KDE no espaço do score (cosseno)
      - Mesma escala (density)
      - Médias
      - Bandwidth usado
    """

    def __init__(self, parent=None, title="Análise KDE (Histograma + Curva)"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 600)

        layout = QtWidgets.QVBoxLayout(self)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        layout.addWidget(self.btn_close)

    # ============================================================
    # NOVO MÉTODO
    # ============================================================

    def plot_visualization(self, data: KDEVisualizationData, title_suffix: str = ""):
        """
        Plota histograma + KDE no espaço RAW.
        """

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        x = data.x_grid

        # ======================================================
        # 1) HISTOGRAMAS (já normalizados pelo service)
        # ======================================================

        # Centros dos bins
        gen_bin_centers = (data.hist_gen_bins[:-1] + data.hist_gen_bins[1:]) / 2
        imp_bin_centers = (data.hist_imp_bins[:-1] + data.hist_imp_bins[1:]) / 2

        # Largura dos bins
        gen_bin_width = data.hist_gen_bins[1] - data.hist_gen_bins[0]
        imp_bin_width = data.hist_imp_bins[1] - data.hist_imp_bins[0]

        ax.bar(
            gen_bin_centers,
            data.hist_gen_counts,
            width=gen_bin_width,
            alpha=0.25,
            color="green",
            label="Hist Genuínos"
        )

        ax.bar(
            imp_bin_centers,
            data.hist_imp_counts,
            width=imp_bin_width,
            alpha=0.25,
            color="red",
            label="Hist Impostores"
        )

        # ======================================================
        # 2) CURVAS KDE
        # ======================================================

        ax.plot(
            x,
            data.kde_gen_density,
            color="green",
            linewidth=2,
            label="KDE Genuínos"
        )

        ax.plot(
            x,
            data.kde_imp_density,
            color="red",
            linewidth=2,
            label="KDE Impostores"
        )

        # ======================================================
        # 3) MÉDIAS
        # ======================================================

        ax.axvline(
            data.stats["mean_gen"],
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"Média Gen: {data.stats['mean_gen']:.3f}"
        )

        ax.axvline(
            data.stats["mean_imp"],
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Média Imp: {data.stats['mean_imp']:.3f}"
        )

        # ======================================================
        # 4) Título informativo
        # ======================================================

        bw_gen = data.stats.get("bw_gen_fisher", None)
        bw_imp = data.stats.get("bw_imp_fisher", None)

        bw_text = ""
        if bw_gen is not None and bw_imp is not None:
            bw_text = f"\nBandwidth (Fisher): Gen={bw_gen:.4f} | Imp={bw_imp:.4f}"

        ax.set_title(
            f"Histograma + KDE no espaço do Score {title_suffix}{bw_text}".strip()
        )

        ax.set_xlabel("Score de Similaridade (Cosseno)")
        ax.set_ylabel("Densidade")
        ax.set_xlim([min(x), max(x)])
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()

        self.figure.tight_layout()
        self.canvas.draw()
