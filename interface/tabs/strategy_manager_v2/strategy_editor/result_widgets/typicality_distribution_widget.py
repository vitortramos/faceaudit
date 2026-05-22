# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/typicality_distribution_widget.py

"""
Widget de visualização do KNNTypicalityModel treinado.

Plota a CDF empírica leave-one-out de T_k (similaridade média aos k
vizinhos mais próximos) computada durante o fit, sobreposta com:
  - KDE da distribuição
  - linha vertical no `ood_threshold` (corte de fora-de-distribuição)
  - métricas-resumo no canto

Diagnóstico crítico em uso forense: se a curva é claramente bimodal,
significa que o `population_label` cobre subpopulações distintas e o
modelo deveria ser estratificado mais fino (gênero, etnia, idade).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class TypicalityDistributionWidget(QtWidgets.QWidget):
    """
    Plot da distribuição empírica de T_k (LOO) do modelo treinado.

    Args:
        parent: QWidget pai opcional.
        model: instância de KNNTypicalityModel já treinada (is_fitted=True).
            Pode ser None — widget mostra placeholder até set_model().
        show_toolbar: se True, exibe NavigationToolbar matplotlib no topo.
        show_title: se True, desenha título no eixo (oculto em comparison).
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        model: Any = None,
        show_toolbar: bool = True,
        show_title: bool = True,
    ):
        super().__init__(parent)
        self._model = model
        self._show_title = bool(show_title)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

        if show_toolbar:
            self.toolbar: Optional[NavigationToolbar] = NavigationToolbar(
                self.canvas, self,
            )
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas, 1)
        self._render()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_model(self, model: Any) -> None:
        """Troca o modelo exibido e redesenha."""
        self._model = model
        self._render()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        self.figure.clear()
        self.figure.set_layout_engine("constrained")

        if self._model is None or not getattr(self._model, "is_fitted", lambda: False)():
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "(modelo não treinado)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888",
            )
            ax.axis("off")
            self.canvas.draw_idle()
            return

        try:
            self._draw(self.figure, self._model, show_title=self._show_title)
        except Exception as exc:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, f"Erro ao gerar plot:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                color="#a33", fontsize=10,
            )
            ax.axis("off")
        self.canvas.draw_idle()

    @staticmethod
    def _draw(figure: Figure, model: Any, show_title: bool = True) -> None:
        """Histograma + KDE da distribuição LOO de T_k, com OOD threshold."""
        figure.clear()
        ax = figure.add_subplot(111)

        loo = model.loo_distribution
        if loo is None or loo.size == 0:
            ax.text(
                0.5, 0.5, "(sem distribuição LOO)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888",
            )
            return

        loo = np.asarray(loo, dtype=np.float64)
        n = int(loo.size)

        x_min = float(np.min(loo))
        x_max = float(np.max(loo))
        if x_min == x_max:
            x_min -= 0.01
            x_max += 0.01
        # Margem de 5% pra esquerda/direita pra OOD vline não ficar colada
        # na borda.
        span = x_max - x_min
        x_min -= 0.05 * span
        x_max += 0.05 * span

        n_bins = min(60, max(20, int(np.sqrt(n))))
        bins = np.linspace(max(-1.0, x_min), min(1.0, x_max), n_bins + 1)

        ax.hist(
            loo, bins=bins, density=True, alpha=0.45,
            color="#1565c0", edgecolor="#0d3a6c", linewidth=0.5,
            label=f"T_k LOO  (n={n:,})".replace(",", "."),
        )

        # KDE suavizado (falha silenciosa se scipy ausente — improvável)
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x_min, x_max, 400)
            kde = gaussian_kde(loo)
            ax.plot(xs, kde(xs), color="#0d3a6c", linewidth=1.6, alpha=0.95)
        except Exception:
            pass

        ood = getattr(model, "ood_threshold", None)
        if ood is not None and np.isfinite(ood):
            ax.axvline(
                float(ood), color="#c62828", linestyle="--", linewidth=1.6,
                alpha=0.85,
                label=f"OOD threshold = {ood:.4f}",
            )

        if show_title:
            pop = getattr(model, "population_label", None) or "(sem rótulo)"
            ax.set_title(
                f"Distribuição empírica de T_k (LOO) — {pop}", fontsize=11,
            )
        ax.set_xlabel(r"$T_k$ — similaridade média aos $k$ vizinhos")
        ax.set_ylabel("Densidade")
        ax.grid(True, linestyle="--", alpha=0.4)

        # Box de métricas no canto superior direito (somente se houver espaço).
        k = getattr(model, "k", "?")
        d = getattr(model, "embedding_dim", "?")
        med = float(np.median(loo))
        mu = float(np.mean(loo))
        sd = float(np.std(loo))
        info = (
            f"k = {k}\n"
            f"d = {d}\n"
            f"mediana = {med:.4f}\n"
            f"média = {mu:.4f}\n"
            f"std = {sd:.4f}"
        )
        ax.text(
            0.985, 0.985, info, transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fafafa", ec="#bbb", lw=0.7),
        )

        ax.legend(loc="upper left", fontsize=9)
