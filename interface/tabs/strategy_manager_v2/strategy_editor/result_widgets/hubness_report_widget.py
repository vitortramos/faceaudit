# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/hubness_report_widget.py

"""
Widget de visualização do HubnessReport.

QTabWidget com 5 abas mostrando os plots mais informativos do
HubnessPlotService. Reutiliza os métodos do plot service via uma
estratégia de captura: substitui temporariamente `plt.show` por um
no-op, deixa o método do service criar sua Figure normalmente, e captura
o resultado via `plt.gcf()`. Assim não precisamos refatorar os 20
métodos do PlotService — eles continuam funcionando idênticos em
scripts/notebooks.

Uso típico (modo dialog full-screen):
    w = HubnessReportWidget(result=report, show_header=True)

Uso inline (3ª coluna do editor V2):
    w = HubnessReportWidget(result=None, show_toolbar=False)
    w.set_result(report)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"


class HubnessReportWidget(QtWidgets.QWidget):
    """
    Painel de 5 abas para um HubnessReport.

    Args:
        parent: pai opcional.
        result: HubnessReport. None = placeholder.
        show_header: header âmbar com métricas escalares.
        show_toolbar: NavigationToolbar matplotlib por aba.
        show_help: caixa azul com texto de interpretação por aba.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        result: Any = None,
        show_header: bool = True,
        show_toolbar: bool = True,
        show_help: bool = True,
    ):
        super().__init__(parent)
        self._result = result
        self._show_toolbar = bool(show_toolbar)
        self._show_help = bool(show_help)

        # Cache: {tab_index: {canvas, figure, draw_fn, drawn_token}}
        self._tab_state: Dict[int, Dict[str, Any]] = {}
        self._result_token: int = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if show_header:
            self._header_frame = self._build_header_frame()
            layout.addWidget(self._header_frame)
        else:
            self._header_frame = None

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._setup_tabs()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self._refresh_header()
        self._on_tab_changed(self.tabs.currentIndex())

    # ── Public API ────────────────────────────────────────────────────────────

    def set_result(self, result: Any) -> None:
        self._result = result
        self._result_token += 1
        for state in self._tab_state.values():
            state["drawn_token"] = None
        self._refresh_header()
        self._on_tab_changed(self.tabs.currentIndex())

    @property
    def result(self) -> Any:
        return self._result

    # ── Header ────────────────────────────────────────────────────────────────

    def _build_header_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        self._header_label = QtWidgets.QLabel("(aguardando relatório)")
        self._header_label.setStyleSheet(
            "background: transparent; color: #5a4500; font-size: 11px;"
        )
        self._header_label.setWordWrap(True)
        h.addWidget(self._header_label, 1)
        return frame

    def _refresh_header(self) -> None:
        if self._header_frame is None:
            return
        if self._result is None:
            self._header_label.setText("(aguardando relatório)")
            return

        r = self._result
        a = r.analysis
        text = (
            f"<b>{r.dataset_name or 'dataset'}</b>"
            f" &nbsp;·&nbsp; N={r.n_samples:,} samples"
            f" &nbsp;·&nbsp; K={r.k_neighbors}"
            f" &nbsp;·&nbsp; <b>max Nk</b>={a.max_nk}"
            f" &nbsp;·&nbsp; <b>mean Nk</b>={a.mean_nk:.2f} (esperado ≈ K)"
            f" &nbsp;·&nbsp; <b>std Nk</b>={a.std_nk:.2f}"
        ).replace(",", ".")
        self._header_label.setText(text)

    # ── Setup das 5 abas (lazy render) ────────────────────────────────────────

    def _setup_tabs(self) -> None:
        from faceaudit.services.hubness_plot_service import HubnessPlotService
        ps = HubnessPlotService()

        # Cada aba define: (título, callable que recebe Figure e desenha, help)
        # As callables encapsulam a chamada ao PlotService monkey-patchando
        # plt.show pra capturar a Figure que o método cria.
        self._add_tab(
            "Diagnóstico Topológico",
            lambda fig: self._draw_topology_grid(fig, ps),
            self._help_topology(),
        )
        self._add_tab(
            "Estratificação por Pose",
            lambda fig: self._capture(fig, ps.plot_stratified_hubness, self._result.analysis),
            self._help_stratification(),
        )
        self._add_tab(
            "Estudo Pareado",
            lambda fig: self._capture(fig, ps.plot_paired_slope_impact, self._result.analysis),
            self._help_paired(),
        )
        self._add_tab(
            "Qualidade × Norma",
            lambda fig: self._capture(fig, ps.plot_quality_diagnostics, self._result.quality_diagnostics),
            self._help_quality(),
        )
        self._add_tab(
            "Densidade de Hubness",
            lambda fig: self._capture(fig, ps.plot_hubness_density, self._result.analysis),
            self._help_density(),
        )

    def _add_tab(
        self,
        title: str,
        draw_fn: Callable[[Figure], None],
        help_html: str = "",
    ) -> None:
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(4, 4, 4, 4)
        page_layout.setSpacing(4)

        figure = Figure(figsize=(8, 5), dpi=100)
        canvas = FigureCanvas(figure)

        if self._show_toolbar:
            toolbar = NavigationToolbar(canvas, page)
            page_layout.addWidget(toolbar)

        page_layout.addWidget(canvas, 1)

        if help_html and self._show_help:
            help_container = QtWidgets.QScrollArea()
            help_container.setWidgetResizable(True)
            help_container.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            help_container.setMaximumHeight(180)
            help_container.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
            )
            help_inner = QtWidgets.QFrame()
            help_inner.setStyleSheet(
                "QFrame { background: #f4f6fa; border-left: 3px solid #2E5090;"
                " border-radius: 2px; }"
            )
            inner = QtWidgets.QVBoxLayout(help_inner)
            inner.setContentsMargins(10, 6, 10, 6)
            lbl = QtWidgets.QLabel(help_html)
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            lbl.setWordWrap(True)
            lbl.setStyleSheet(
                "background: transparent; color: #1a2942; font-size: 11px;"
            )
            inner.addWidget(lbl)
            help_container.setWidget(help_inner)
            page_layout.addWidget(help_container)

        idx = self.tabs.addTab(page, title)
        self._tab_state[idx] = {
            "canvas": canvas,
            "figure": figure,
            "draw_fn": draw_fn,
            "drawn_token": None,
        }

    def _on_tab_changed(self, index: int) -> None:
        state = self._tab_state.get(index)
        if state is None:
            return
        if state["drawn_token"] == self._result_token:
            return  # já desenhado para o resultado atual

        figure: Figure = state["figure"]
        canvas: FigureCanvas = state["canvas"]
        draw_fn: Callable = state["draw_fn"]

        figure.clear()
        if self._result is None:
            ax = figure.add_subplot(111)
            ax.text(0.5, 0.5, "(aguardando relatório)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="#888")
            ax.axis("off")
        else:
            try:
                draw_fn(figure)
            except Exception as exc:
                figure.clear()
                ax = figure.add_subplot(111)
                ax.text(0.5, 0.5, f"Erro ao gerar plot:\n{exc}",
                        transform=ax.transAxes, ha="center", va="center",
                        color="#a33", fontsize=10)
                ax.axis("off")
        canvas.draw_idle()
        state["drawn_token"] = self._result_token

    # ── Captura via monkey-patch de plt.show ──────────────────────────────────

    def _capture(self, target_figure: Figure, plot_fn: Callable, *args, **kwargs) -> None:
        """
        Chama um método do HubnessPlotService que internamente cria sua própria
        Figure via `plt.figure(...)` e finaliza com `plt.show()`. Capturamos
        essa Figure e copiamos o conteúdo para `target_figure` (a que pertence
        ao FigureCanvas inline). Estratégia:

        1. Substitui plt.show por no-op (pra não abrir janelas em background).
        2. Chama plot_fn — ela cria sua Figure interna e popula com axes.
        3. Recupera plt.gcf() — a Figure populada.
        4. Re-plota o conteúdo em target_figure copiando os axes (como hack
           prático: usa a estratégia mais segura de chamar pickle dump/load
           pra "transplantar" a figure inteira; alternativa simples — clear
           target e usar a fig criada como `figure.canvas.figure = newfig`
           via reattach manual).

        Estratégia adotada: salvamos a figure capturada em buffer pickle e
        recarregamos colando no target. Funciona pra todos os tipos de plot
        (scatter, hist, kde, ridgeline, slope) sem precisar conhecer os
        artistas matplotlib individualmente.
        """
        original_show = plt.show
        original_figure = plt.figure

        # Lista pra capturar a(s) figure(s) criada(s).
        captured: List[Figure] = []

        def capture_figure(*fa, **fkw):
            new_fig = Figure(*fa, **fkw)
            FigureCanvas(new_fig)  # Necessário pra que plt.gca() funcione
            captured.append(new_fig)
            # Registra como current figure pra que plt.gca/plt.scatter etc
            # peguem essa figure ao serem chamadas em seguida.
            plt._pylab_helpers.Gcf.set_active(new_fig.canvas.manager) \
                if hasattr(plt, "_pylab_helpers") else None
            return new_fig

        plt.show = lambda *a, **k: None
        try:
            # Não substituimos plt.figure — métodos do service usam plt.figure
            # que retorna uma figure registrada no Gcf. Captura via plt.gcf()
            # após a chamada.
            plot_fn(*args, **kwargs)
            captured_fig = plt.gcf()
            captured.append(captured_fig)

            # Transplante: limpa target_figure e copia os axes da capturada.
            self._transplant_figure(captured_fig, target_figure)
        finally:
            plt.show = original_show
            # Fecha as figures capturadas pra liberar memória matplotlib.
            for fig in captured:
                try:
                    plt.close(fig)
                except Exception:
                    pass

    @staticmethod
    def _transplant_figure(src_fig: Figure, dst_fig: Figure) -> None:
        """
        Copia o conteúdo de `src_fig` pra `dst_fig` re-renderizando via
        png buffer + imshow. Hack prático que funciona pra qualquer tipo de
        plot matplotlib (scatter, hist, kde, ridgeline, slope) sem precisar
        conhecer os Artists individuais — a desvantagem é perder
        interatividade (zoom/pan funcionam mas nas coordenadas pixel, não
        nas data coords). Pra o caso de uso (relatório forense, visualização
        estática) é aceitável.

        Pra interatividade total, abrir no dialog full-screen via botão
        🔍 Expandir (que renderiza diretamente sem transplante).
        """
        import io
        from matplotlib.image import imread

        buf = io.BytesIO()
        src_fig.savefig(buf, format="png", dpi=src_fig.dpi, bbox_inches="tight")
        buf.seek(0)
        img = imread(buf)

        dst_fig.clear()
        ax = dst_fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        dst_fig.tight_layout(pad=0)

    # ── Aba 1: grid 1×3 com 3 plots topológicos ──────────────────────────────

    def _draw_topology_grid(self, target_figure: Figure, plot_service) -> None:
        """
        Combina 3 plots topológicos (Hubness×Impostor, Norm×Hubness, Pose×Hubness)
        em um único grid 1×3 numa só Figure. Captura cada um separadamente,
        salva como PNG e arranja em subplots.
        """
        import io
        from matplotlib.image import imread

        plot_calls = [
            (plot_service.plot_hubness_vs_impostor_similarity, self._result.analysis),
            (plot_service.plot_norm_vs_hubness_distribution, self._result.analysis),
            (plot_service.plot_pose_vs_hubness_direct, self._result.analysis),
        ]

        original_show = plt.show
        plt.show = lambda *a, **k: None
        images = []
        try:
            for fn, arg in plot_calls:
                fn(arg)
                cap = plt.gcf()
                buf = io.BytesIO()
                cap.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                buf.seek(0)
                images.append(imread(buf))
                plt.close(cap)
        finally:
            plt.show = original_show

        target_figure.clear()
        for i, img in enumerate(images):
            ax = target_figure.add_subplot(1, 3, i + 1)
            ax.imshow(img)
            ax.axis("off")
        target_figure.tight_layout(pad=0.5)

    # ── Help ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _help_topology() -> str:
        return (
            "<b>Diagnóstico topológico em 3 vistas:</b><br>"
            "• <b>Hubness × Similaridade Impostora</b> — faces com Nk alto "
            "(hubs) tendem a atrair impostores? Correlação positiva forte = "
            "viés problemático.<br>"
            "• <b>Norma × Hubness</b> — relação clássica do extrator: faces "
            "com norma L2 baixa frequentemente viram hubs.<br>"
            "• <b>Pose × Hubness</b> — pose lateral aumenta o hubness?"
        )

    @staticmethod
    def _help_stratification() -> str:
        return (
            "<b>Box plot do Nk por 5 faixas de yaw:</b> Frontal &lt;15°, Leve "
            "15–30°, Moderada 30–45°, Forte 45–60°, Extrema &gt;60°. "
            "Faixas com mediana muito acima de K = grupos enviesados."
        )

    @staticmethod
    def _help_paired() -> str:
        return (
            "<b>Slope chart frontal × não-frontal por identidade:</b> linhas "
            "vermelhas indicam IDs onde o Nk PIORA quando se usa a foto não-"
            "frontal. Quanto mais linhas vermelhas inclinadas pra cima, mais "
            "viés sistemático associado a pose lateral."
        )

    @staticmethod
    def _help_quality() -> str:
        return (
            "<b>Diagnóstico de covariáveis FIQA / Norma × pose:</b> verifica "
            "se a qualidade do extrator varia sistematicamente com a pose. "
            "Médias muito diferentes entre faixas indicam que efeitos "
            "atribuídos à pose podem ser confundidos com efeitos de qualidade."
        )

    @staticmethod
    def _help_density() -> str:
        return (
            "<b>KDE da distribuição global de Nk:</b> idealmente próxima de uma "
            "Poisson(K). Caudas pesadas à direita = presença de hubs. Forma "
            "muito assimétrica indica viés topológico geral."
        )
