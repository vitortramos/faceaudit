# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/score_distribution_widget.py

"""
Widget de distribuição de scores (genuínos × impostores).

Aplicável a:
- SIMILARITY_SCORING (raw_score por par)
- SCORE_CALIBRATION (log-LR calibrado, lido de sample.metadata["log_lr"])
- PAIR_MINING (raw_score sobre pares curados)

A lógica de plot foi extraída do método `_draw_score_distribution` do
controller V2 — comportamento e estilo idênticos. O widget pode operar
em dois modos: com ou sem `NavigationToolbar2QT` no topo.
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


class ScoreDistributionWidget(QtWidgets.QWidget):
    """
    Histograma + KDE de scores genuínos vs impostores.

    Args:
        parent: QWidget pai opcional.
        score_dataset: SimilarityScoreDataset a ser plotado. Pode ser None
            (widget mostra placeholder até `set_dataset(...)` ser chamado).
        use_log_lr: se True, lê scores de `sample.metadata["log_lr"]` em
            vez de `sample.raw_score`. Usado quando o dataset é
            `CALIBRATED_SCORES`.
        show_toolbar: se True, exibe a `NavigationToolbar2QT` matplotlib
            no topo (zoom/pan/salvar). Default True para uso em dialog;
            no painel inline geralmente passa False.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        score_dataset: Any = None,
        use_log_lr: bool = False,
        show_toolbar: bool = True,
    ):
        super().__init__(parent)
        self._dataset = score_dataset
        self._use_log_lr = bool(use_log_lr)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        if show_toolbar:
            self.toolbar: Optional[NavigationToolbar] = NavigationToolbar(
                self.canvas, self
            )
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None

        layout.addWidget(self.canvas, 1)

        self._render()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_dataset(
        self,
        score_dataset: Any,
        use_log_lr: Optional[bool] = None,
    ) -> None:
        """Troca o dataset exibido e redesenha o gráfico."""
        self._dataset = score_dataset
        if use_log_lr is not None:
            self._use_log_lr = bool(use_log_lr)
        self._render()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        """Desenha o gráfico — placeholder se sem dataset."""
        self.figure.clear()
        if self._dataset is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "(sem dados)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888",
            )
            ax.axis("off")
            self.canvas.draw_idle()
            return

        try:
            self._draw(self.figure, self._dataset, self._use_log_lr)
        except Exception as exc:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"Erro ao gerar plot:\n{exc}",
                ha="center", va="center", transform=ax.transAxes,
                color="#a33", fontsize=10,
            )
            ax.axis("off")
        self.canvas.draw_idle()

    @staticmethod
    def _draw(figure: Figure, score_dataset: Any, use_log_lr: bool) -> None:
        """
        Desenha histograma + KDE no `figure`. Robusto a casos onde só uma
        das classes está presente.

        Quando `use_log_lr=True`, lê scores de `sample.metadata["log_lr"]`;
        caso contrário, usa `score_dataset.scores`.
        """
        figure.clear()
        ax = figure.add_subplot(111)

        if use_log_lr:
            raw: list = []
            raw_labels: list = []
            for s in score_dataset.items:
                md = getattr(s, "metadata", None)
                v = md.get("log_lr") if isinstance(md, dict) else None
                if v is None:
                    continue
                raw.append(float(v))
                raw_labels.append(int(s.label))
            scores = np.asarray(raw, dtype=np.float64)
            labels = np.asarray(raw_labels, dtype=np.int32)
        else:
            scores = np.asarray(score_dataset.scores, dtype=np.float64)
            labels = np.asarray(score_dataset.labels, dtype=np.int32)

        if scores.size == 0:
            ax.text(
                0.5, 0.5, "(sem scores)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#888",
            )
            return

        gen_mask = labels == 1
        imp_mask = labels == 0
        gen_scores = scores[gen_mask]
        imp_scores = scores[imp_mask]
        n_gen = int(gen_scores.size)
        n_imp = int(imp_scores.size)

        x_min = float(np.min(scores))
        x_max = float(np.max(scores))
        if x_min == x_max:
            x_min -= 0.5
            x_max += 0.5
        bins = np.linspace(x_min, x_max, 60)

        if n_imp > 0:
            ax.hist(
                imp_scores, bins=bins, density=True, alpha=0.45,
                color="#c62828", edgecolor="#7a0000", linewidth=0.5,
                label=f"Impostores  (n={n_imp:,})".replace(",", "."),
            )
        if n_gen > 0:
            ax.hist(
                gen_scores, bins=bins, density=True, alpha=0.45,
                color="#2e7d32", edgecolor="#0d3a10", linewidth=0.5,
                label=f"Genuínos  (n={n_gen:,})".replace(",", "."),
            )

        # KDE leve via gaussian_kde — falha silenciosa se scipy não
        # disponível (histograma ainda funciona).
        try:
            from scipy.stats import gaussian_kde
            xs = np.linspace(x_min, x_max, 400)
            if n_imp > 1:
                kde_i = gaussian_kde(imp_scores)
                ax.plot(xs, kde_i(xs), color="#7a0000", linewidth=1.4, alpha=0.9)
            if n_gen > 1:
                kde_g = gaussian_kde(gen_scores)
                ax.plot(xs, kde_g(xs), color="#0d3a10", linewidth=1.4, alpha=0.9)
        except Exception:
            pass

        title_prefix = (
            "Distribuição de log-LR" if use_log_lr else "Distribuição de Scores"
        )
        title_suffix = ""
        ds_name = getattr(score_dataset, "name", None)
        if ds_name:
            title_suffix = f" — {ds_name}"
        ax.set_title(
            f"{title_prefix}{title_suffix}\n"
            f"Total: {n_gen + n_imp:,} pares".replace(",", "."),
            fontsize=11,
        )
        ax.set_xlabel("log-LR calibrado" if use_log_lr else "Score bruto")
        ax.set_ylabel("Densidade")
        if use_log_lr:
            ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                       alpha=0.6, label="log-LR = 0")
        ax.grid(True, linestyle="--", alpha=0.4)
        if n_gen > 0 or n_imp > 0:
            ax.legend(loc="best", fontsize=9)

        figure.tight_layout()
