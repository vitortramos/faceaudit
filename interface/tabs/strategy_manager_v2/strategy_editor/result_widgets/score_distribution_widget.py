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

from faceaudit.models.similarity.kde_curve import kde_curve


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
        show_title: bool = True,
    ):
        super().__init__(parent)
        self._dataset = score_dataset
        self._use_log_lr = bool(use_log_lr)
        # show_title=False suprime o ax.set_title() — útil quando o widget
        # é embedado num cell que já tem header próprio (ex: comparação),
        # evitando título redundante e ganhando área pro plot.
        self._show_title = bool(show_title)
        # Sidecar pré-computado (gravado por SimilarityScoreDataset.to_parquet).
        # Quando presente, o render usa diretamente sem iterar samples nem
        # rodar gaussian_kde — orders of magnitude mais rápido pra datasets
        # com milhões de pares.
        self._dist_summary: Optional[dict] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)

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
        # Trocar pra um dataset em memória anula o sidecar previamente
        # carregado — preferir o dataset (mais fresco, ground truth).
        self._dist_summary = None
        self._render()

    def set_distribution_summary(self, summary: Optional[dict]) -> None:
        """
        Plota a partir do sidecar pré-computado, sem precisar do dataset
        em memória. Usado pelo painel de resultados quando o parquet está
        no disco mas o sidecar `<stem>_dist.json` está disponível —
        evita o custo de carregar 5M+ samples só para desenhar.

        Quando `summary is None`, limpa o estado e mostra placeholder.
        """
        self._dist_summary = summary
        # Ao usar sidecar, descarta dataset previamente setado para evitar
        # ambiguidade no _render.
        if summary is not None:
            self._dataset = None
        self._render()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _render(self) -> None:
        """Desenha o gráfico — placeholder se sem dataset/sidecar."""
        self.figure.clear()
        # Constrained layout reserva espaço pra ylabel/xlabel/legenda
        # automaticamente — substitui o frágil tight_layout() que cortava
        # o ylabel "Densidade" em cells estreitos.
        self.figure.set_layout_engine("constrained")

        # Caminho rápido: sidecar pré-computado.
        if self._dist_summary is not None:
            try:
                self._draw_from_summary(
                    self.figure, self._dist_summary, show_title=self._show_title,
                )
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
            return

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
            self._draw(
                self.figure, self._dataset, self._use_log_lr,
                show_title=self._show_title,
            )
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
    def _draw(
        figure: Figure, score_dataset: Any, use_log_lr: bool,
        show_title: bool = True,
    ) -> None:
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
                # label aqui é Optional[int]; sentinela -1 = casework (sem GT)
                lbl = s.label
                raw_labels.append(int(lbl) if lbl is not None else -1)
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
        # Sentinela -1 marca pares casework (sem ground truth) — `dataset.labels`
        # mapeia `same_person=None` → -1. SIMILARITY_SCORING em modo casework
        # produz datasets puramente casework; sem essa branch o gráfico ficava
        # vazio mostrando "Total: 0 pares" mesmo com pares no recurso.
        case_mask = labels == -1
        gen_scores = scores[gen_mask]
        imp_scores = scores[imp_mask]
        case_scores = scores[case_mask]
        n_gen = int(gen_scores.size)
        n_imp = int(imp_scores.size)
        n_case = int(case_scores.size)

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
        # Pares casework: poucos pontos é o caso típico (1–N pares forenses),
        # plot como hist+linhas verticais — KDE só faz sentido com n>=30.
        if n_case > 0:
            if n_case >= 30:
                ax.hist(
                    case_scores, bins=bins, density=True, alpha=0.45,
                    color="#1565c0", edgecolor="#0d3a6c", linewidth=0.5,
                    label=f"Caso  (n={n_case:,})".replace(",", "."),
                )
            else:
                # n<30 — desenha cada par como linha vertical, primeiro
                # com label e demais sem (evita poluir a legenda).
                for i, val in enumerate(case_scores):
                    ax.axvline(
                        float(val), color="#1565c0", linestyle="--",
                        linewidth=1.4, alpha=0.85,
                        label=(
                            f"Caso  (n={n_case})" if i == 0 else None
                        ),
                    )

        # KDE por série — `kde_curve` devolve None se scipy faltar ou se a
        # série for degenerada (o pico achataria as demais; o histograma
        # continua mostrando a massa concentrada).
        xs = np.linspace(x_min, x_max, 400)
        for series, min_n, color in (
            (imp_scores, 2, "#7a0000"),
            (gen_scores, 2, "#0d3a10"),
            (case_scores, 31, "#0d3a6c"),
        ):
            ys = kde_curve(series, xs) if series.size >= min_n else None
            if ys is not None:
                ax.plot(xs, ys, color=color, linewidth=1.4, alpha=0.9)

        if show_title:
            # Título enxuto — sem nome do dataset nem linha de total
            # (cabia mal em cells estreitos do painel de resultados).
            # Espelha o helper `draw_distribution_from_summary`.
            ax.set_title(
                "Distribuição de log-LR" if use_log_lr else "Distribuição de Escores",
                fontsize=11,
            )
        ax.set_xlabel("log-LR calibrado" if use_log_lr else "Escore Bruto")
        ax.set_ylabel("Densidade")
        if use_log_lr:
            ax.axvline(0.0, color="#444", linestyle=":", linewidth=1.0,
                       alpha=0.6, label="log-LR = 0")
        ax.grid(True, linestyle="--", alpha=0.4)
        if n_gen > 0 or n_imp > 0 or n_case > 0:
            ax.legend(loc="best", fontsize=9)

        # tight_layout removido — _render seta constrained_layout antes,
        # que faz o mesmo trabalho sem cortar o ylabel em cells estreitos.

    @staticmethod
    def _draw_from_summary(
        figure: Figure, summary: dict, show_title: bool = True,
    ) -> None:
        """Delega pro helper puro-matplotlib em
        `faceaudit.models.similarity.distribution_plot.draw_distribution_from_summary`.

        Mantido como wrapper estático pra compat com chamadores existentes;
        a lógica de desenho foi extraída pra que o snapshot PNG gerado
        durante a persistência use exatamente o mesmo render."""
        from faceaudit.models.similarity.distribution_plot import (
            draw_distribution_from_summary,
        )
        draw_distribution_from_summary(figure, summary, show_title=show_title)
