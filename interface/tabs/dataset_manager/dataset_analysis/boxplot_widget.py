"""Widget matplotlib pra plotar boxplots estratificados por quartil.

Consumido pela `DatasetAnalysisView` quando o usuário seleciona uma
auditoria do tipo `boxplot_*` na árvore. Renderiza N boxplots lado-a-lado
(um por grupo) com swarm plot leve por cima e linha conectando as
medianas — dá uma leitura imediata se o efeito é monotônico ou flat.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class BoxplotWidget(QtWidgets.QWidget):
    """Boxplots estratificados com toolbar de navegação."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self._figure = Figure(figsize=(6, 5), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas, 1)

    def render_boxplot(
        self,
        groups: Sequence[Sequence[float]],
        labels: Sequence[str],
        x_label: str,
        y_label: str,
    ) -> None:
        """Plota um boxplot por grupo. Liga as medianas com uma linha
        contínua pra realçar tendência (monótona vs flat)."""
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        valid_groups: List[np.ndarray] = []
        valid_labels: List[str] = []
        for g, lbl in zip(groups, labels):
            arr = np.asarray(list(g), dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size > 0:
                valid_groups.append(arr)
                valid_labels.append(lbl)

        if not valid_groups:
            ax.text(
                0.5, 0.5, "Sem dados nos grupos.",
                ha="center", va="center", transform=ax.transAxes,
                color="#666",
            )
            self._canvas.draw_idle()
            return

        positions = list(range(1, len(valid_groups) + 1))

        bp = ax.boxplot(
            valid_groups, positions=positions, widths=0.55,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="#c62828", linewidth=1.6),
            boxprops=dict(facecolor="#bbdefb", edgecolor="#1976d2"),
            whiskerprops=dict(color="#1976d2"),
            capprops=dict(color="#1976d2"),
            flierprops=dict(
                marker="o", markersize=2.5,
                markerfacecolor="#90a4ae", markeredgecolor="none",
                alpha=0.4,
            ),
        )
        # Ignora retorno (`bp`) — só queríamos o estilo.
        _ = bp

        # Linha conectando as medianas — visualmente revela
        # monotonicidade. Se as 4 medianas formam uma reta descendente,
        # correlação negativa é real; se ficam quase paralelas ao eixo X,
        # a correlação observada no scatter é artefato de outliers.
        medians = [float(np.median(g)) for g in valid_groups]
        ax.plot(
            positions, medians,
            color="#c62828", linestyle="-", linewidth=1.4,
            marker="o", markersize=5.5,
            label="Mediana por grupo",
        )

        # Anota cada mediana com seu valor numérico
        for pos, med in zip(positions, medians):
            ax.annotate(
                f"{med:.2f}", xy=(pos, med),
                xytext=(8, 0), textcoords="offset points",
                fontsize=9, color="#c62828",
                va="center",
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(valid_labels, fontsize=9)
        ax.set_xlabel(f"Quartis de {x_label}")
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label}  por quartil de  {x_label}")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Reseta o canvas — chamado ao trocar pra outro tipo de finding."""
        self._figure.clear()
        self._canvas.draw_idle()
