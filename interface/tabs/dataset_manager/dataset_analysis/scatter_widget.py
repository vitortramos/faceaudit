"""Widget matplotlib pra plotar scatter de correlação entre duas
métricas de qualidade.

Consumido pela `DatasetAnalysisView` quando o usuário seleciona uma
auditoria do tipo `correlation_*` na árvore. Renderiza:
  - Scatter (alpha baixo) ou hexbin (densidade) conforme N
  - Linha de regressão linear OLS
  - Coeficiente de Pearson na legenda
  - Linhas guia formando 4 quadrantes (thresholds em x e y) — quadrantes
    são anotados com a contagem de pontos pra leitura rápida
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


# Acima deste N, scatter puro vira mancha — usa hexbin pra ler densidade.
_SCATTER_HEXBIN_THRESHOLD = 5_000


class ScatterWidget(QtWidgets.QWidget):
    """Scatter de correlação com toolbar de navegação."""

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

    def render_correlation(
        self,
        x_values: Sequence[float],
        y_values: Sequence[float],
        x_label: str,
        y_label: str,
        *,
        x_threshold: Optional[float] = None,
        y_threshold: Optional[float] = None,
    ) -> None:
        """Plota scatter + regressão + quadrantes. Limpa axes anterior.

        x_threshold / y_threshold opcionais — se None, usa a mediana
        de cada eixo (quadrantes balanceados).
        """
        self._figure.clear()
        ax = self._figure.add_subplot(111)

        if not x_values or not y_values:
            ax.text(
                0.5, 0.5, "Sem dados pareados para correlacionar.",
                ha="center", va="center", transform=ax.transAxes,
                color="#666",
            )
            self._canvas.draw_idle()
            return

        x = np.asarray(list(x_values), dtype=float)
        y = np.asarray(list(y_values), dtype=float)
        n = x.size

        # Remove pares inválidos (NaN/inf) antes de tudo
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        n = x.size
        if n == 0:
            ax.text(
                0.5, 0.5, "Todos os pares são NaN/inf.",
                ha="center", va="center", transform=ax.transAxes,
                color="#666",
            )
            self._canvas.draw_idle()
            return

        # Pearson r e regressão linear (numpy puro — evita scipy)
        if n > 1 and np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, 1)
        else:
            r = 0.0
            slope, intercept = 0.0, float(np.mean(y)) if n else 0.0

        # Plot principal: hexbin pra N grande, scatter pra N moderado.
        if n >= _SCATTER_HEXBIN_THRESHOLD:
            hb = ax.hexbin(
                x, y, gridsize=50, cmap="Blues",
                mincnt=1, linewidths=0.0,
            )
            cb = self._figure.colorbar(hb, ax=ax)
            cb.set_label("Densidade (faces)", fontsize=9)
        else:
            ax.scatter(
                x, y, s=14, c="#1976d2", alpha=0.45,
                edgecolors="none",
            )

        # Linha de regressão sobre o range observado
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(
            x_line, slope * x_line + intercept,
            color="#c62828", linewidth=1.6,
            label=f"y = {slope:+.3f}·x + {intercept:+.3f}",
        )

        # Linhas de quadrante (thresholds)
        x_thr = float(x_threshold) if x_threshold is not None else float(np.median(x))
        y_thr = float(y_threshold) if y_threshold is not None else float(np.median(y))
        ax.axvline(
            x_thr, color="#666", linestyle=":", linewidth=1.2,
            label=f"limiar X = {x_thr:.3f}",
        )
        ax.axhline(
            y_thr, color="#666", linestyle=":", linewidth=1.2,
            label=f"limiar Y = {y_thr:.3f}",
        )

        # Anota contagem de cada quadrante — ajuda a quantificar a
        # suspeita do usuário (ex: "FIQA alto + norma baixa = N faces").
        q_tr = int(np.sum((x >= x_thr) & (y >= y_thr)))   # alto X / alto Y
        q_br = int(np.sum((x >= x_thr) & (y < y_thr)))    # alto X / baixo Y ← suspeito
        q_tl = int(np.sum((x < x_thr) & (y >= y_thr)))    # baixo X / alto Y
        q_bl = int(np.sum((x < x_thr) & (y < y_thr)))     # baixo X / baixo Y
        self._annotate_quadrants(ax, x_thr, y_thr, q_tr, q_br, q_tl, q_bl)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(
            f"{x_label}  ×  {y_label}    "
            f"(n={n:,};  Pearson r = {r:+.3f})"
        )
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        self._canvas.draw_idle()

    def clear(self) -> None:
        """Reseta o canvas — chamado ao trocar pra outro tipo de finding."""
        self._figure.clear()
        self._canvas.draw_idle()

    # ── Helpers internos ─────────────────────────────────────────────────

    @staticmethod
    def _annotate_quadrants(
        ax, x_thr: float, y_thr: float,
        q_tr: int, q_br: int, q_tl: int, q_bl: int,
    ) -> None:
        """Põe um texto leve no canto de cada quadrante com a contagem
        e um rótulo curto. Posição relativa ao threshold + offset em
        coords de display via `transform=ax.transData` evita problemas
        com ranges arbitrários."""
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        # Margem visual: 4% do range em cada direção
        dx = (x_max - x_min) * 0.02
        dy = (y_max - y_min) * 0.02

        common = dict(
            fontsize=8.5, color="#444",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff",
                      edgecolor="#ccc", alpha=0.85),
        )
        # Top-right (alto X, alto Y) — saudável
        ax.text(
            x_max - dx, y_max - dy,
            f"alto X / alto Y\nn = {q_tr:,}",
            ha="right", va="top", **common,
        )
        # Bottom-right (alto X, baixo Y) — quadrante suspeito
        ax.text(
            x_max - dx, y_min + dy,
            f"alto X / BAIXO Y\nn = {q_br:,}",
            ha="right", va="bottom",
            **{**common, "color": "#c62828"},
        )
        # Top-left
        ax.text(
            x_min + dx, y_max - dy,
            f"baixo X / alto Y\nn = {q_tl:,}",
            ha="left", va="top", **common,
        )
        # Bottom-left
        ax.text(
            x_min + dx, y_min + dy,
            f"baixo X / baixo Y\nn = {q_bl:,}",
            ha="left", va="bottom", **common,
        )
