# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/hubness_ablation_widget.py

"""
Widget de visualização do AblationStudyInfo (output da instrução
HUBNESS_ABLATION).

Single-plot widget que reusa `HubnessPlotService.plot_ablation_study_topology`
via mesma estratégia de monkey-patch + transplante de PNG usada pelo
HubnessReportWidget. Plot mostra KDEs sobrepostas das distribuições de
Nk em cada cenário (1, 2, 3 fotos por ID, total) — quanto mais para a
direita o pico, mais forte o hubness.
"""

from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib.pyplot as plt
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.image import imread


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"


class HubnessAblationWidget(QtWidgets.QWidget):
    """
    Single-plot widget para `AblationStudyInfo`.

    Args:
        parent: pai opcional.
        result: AblationStudyInfo (None = placeholder).
        show_header: header âmbar com K + nº de cenários.
        show_toolbar: NavigationToolbar matplotlib.
        show_help: caixa azul de ajuda contextual.
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
        self._show_help = bool(show_help)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if show_header:
            self._header_frame = self._build_header_frame()
            layout.addWidget(self._header_frame)
        else:
            self._header_frame = None

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

        if show_help:
            self._help_frame = self._build_help_frame()
            layout.addWidget(self._help_frame)
        else:
            self._help_frame = None

        self._refresh_header()
        self._render()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_result(self, result: Any) -> None:
        self._result = result
        self._refresh_header()
        self._render()

    @property
    def result(self) -> Any:
        return self._result

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_header_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        self._header_label = QtWidgets.QLabel("(aguardando ablação)")
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
            self._header_label.setText("(aguardando ablação)")
            return

        r = self._result
        scenarios = getattr(r, "scenarios", []) or []
        max_nks = ", ".join(
            f"{s.scenario_name}: max Nk={s.max_nk}" for s in scenarios
        )
        text = (
            f"<b>K</b>={getattr(r, 'k_neighbors', '?')}"
            f" &nbsp;·&nbsp; <b>{len(scenarios)} cenários</b>"
            f" &nbsp;·&nbsp; {max_nks}"
        )
        self._header_label.setText(text)

    def _build_help_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            "QFrame { background: #f4f6fa; border-left: 3px solid #2E5090;"
            " border-radius: 2px; }"
        )
        frame.setMaximumHeight(110)
        inner = QtWidgets.QVBoxLayout(frame)
        inner.setContentsMargins(10, 6, 10, 6)
        lbl = QtWidgets.QLabel(
            "<b>KDEs sobrepostas:</b> distribuição de Nk em cada cenário "
            "(1, 2, 3 fotos por ID, total). Quanto mais à direita o pico = "
            "mais hubs no cenário. Linha tracejada vertical em K marca a "
            "expectativa ideal (Poisson). Cenários com max Nk muito acima "
            "de K indicam que adicionar fotos por ID amplifica o hubness."
        )
        lbl.setStyleSheet(
            "background: transparent; color: #1A2942; font-size: 10px;"
        )
        lbl.setWordWrap(True)
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        inner.addWidget(lbl)
        return frame

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        self.figure.clear()
        if self._result is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "(aguardando ablação)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#888",
            )
            ax.axis("off")
            self.canvas.draw_idle()
            return

        try:
            self._draw(self.figure, self._result)
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
    def _draw(target_figure: Figure, result: Any) -> None:
        """
        Reusa HubnessPlotService.plot_ablation_study_topology via captura
        de plt.gcf() + transplante por PNG. Estratégia idêntica à do
        HubnessReportWidget — evita refatorar o método do plot service.
        """
        from faceaudit.services.hubness_plot_service import HubnessPlotService
        ps = HubnessPlotService()

        original_show = plt.show
        plt.show = lambda *a, **k: None
        try:
            ps.plot_ablation_study_topology(result)
            captured = plt.gcf()
            buf = io.BytesIO()
            captured.savefig(buf, format="png", dpi=captured.dpi, bbox_inches="tight")
            plt.close(captured)
            buf.seek(0)
            img = imread(buf)

            target_figure.clear()
            ax = target_figure.add_subplot(111)
            ax.imshow(img)
            ax.axis("off")
            target_figure.tight_layout(pad=0)
        finally:
            plt.show = original_show
