# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/casework_visualization_widget.py

"""
Widget de visualização de CaseworkVisualizationResult.

Single-plot: sobrepõe a distribuição dos scores do caso forense
(questionada × padrão) sobre as distribuições de referência (genuínos
verde, impostores vermelho). Quando há ≥30 pares de caso, usa KDE;
abaixo disso, usa linhas verticais tracejadas com valor numérico
anotado.

Lógica extraída de `CaseworkVisualizationViewerDialog._draw_visualization`
sem alteração — o dialog agora envolve este widget com botão Fechar.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from PyQt6 import QtCore, QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


_BG_HEADER = "#FFF8E8"
_BORDER_HEADER = "#E8A020"

_KDE_MIN_CASE_SAMPLES = 30


class CaseworkVisualizationWidget(QtWidgets.QWidget):
    """
    Single-plot widget para `CaseworkVisualizationResult`.

    Args:
        parent: pai opcional.
        result: CaseworkVisualizationResult (None = placeholder).
        show_header: se True, header âmbar com modo/contagens/sumário.
        show_toolbar: se True, NavigationToolbar matplotlib no topo.
        show_help: se True, caixa azul com texto de ajuda contextual.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        result: Any = None,
        show_header: bool = True,
        show_toolbar: bool = True,
        show_help: bool = True,
        show_title: bool = True,
    ):
        super().__init__(parent)
        self._result = result
        self._show_help = bool(show_help)
        # show_title=False suprime ax.set_title() — útil quando o widget é
        # embedado num cell que já tem header próprio (ex: comparação).
        self._show_title = bool(show_title)

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
        self._refresh_help()
        self._render()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_result(self, result: Any) -> None:
        self._result = result
        self._refresh_header()
        self._refresh_help()
        self._render()

    @property
    def result(self) -> Any:
        return self._result

    # ── UI building ───────────────────────────────────────────────────────────

    def _build_header_frame(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {_BG_HEADER}; "
            f"border-left: 3px solid {_BORDER_HEADER}; border-radius: 2px; }}"
        )
        h = QtWidgets.QHBoxLayout(frame)
        h.setContentsMargins(10, 6, 10, 6)
        self._header_label = QtWidgets.QLabel("(aguardando resultado)")
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
            self._header_label.setText("(aguardando resultado)")
            return

        r = self._result
        cs = r.case_summary or {}
        case_part = (
            f"min={cs.get('min', 0):+.2f} | "
            f"mediana={cs.get('median', 0):+.2f} | "
            f"max={cs.get('max', 0):+.2f}"
            if cs else "sem amostras"
        )
        eer_part = (
            f" &nbsp;·&nbsp; <b>EER threshold</b>={r.eer_threshold:.4f}"
            if r.eer_threshold is not None else ""
        )

        text = (
            f"<b>Modo: {r.score_type}</b>"
            f" &nbsp;·&nbsp; Referência: {r.n_genuine:,} gen / {r.n_impostor:,} imp"
            f" &nbsp;·&nbsp; Caso: {r.n_case} pares ({case_part})"
            f"{eer_part}"
        ).replace(",", ".")
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
        self._help_label = QtWidgets.QLabel("")
        self._help_label.setStyleSheet(
            "background: transparent; color: #1A2942; font-size: 10px;"
        )
        self._help_label.setWordWrap(True)
        self._help_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        inner.addWidget(self._help_label)
        return frame

    def _refresh_help(self) -> None:
        if self._help_frame is None:
            return
        if self._result is None:
            self._help_label.setText("")
            return
        if self._result.score_type == "CALIBRATED":
            html = (
                "<b>Como ler o gráfico:</b> as duas curvas (verde gen, vermelho imp) "
                "mostram onde o calibrador posiciona pares conhecidos da população "
                "de referência (validação). A linha tracejada azul indica os escores "
                "do caso forense em log-LR. Se a marca do caso cai bem dentro da "
                "curva verde → forte evidência a favor de Hp (mesma pessoa). Bem "
                "dentro da curva vermelha → forte evidência a favor de Hd (pessoas "
                "diferentes). Na zona de sobreposição → evidência fraca/ambígua. "
                "A linha cinza tracejada em log-LR=0 marca o limiar neutro."
            )
        else:
            html = (
                "<b>Como ler o gráfico:</b> distribuição bruta (raw_score) dos pares "
                "de validação por classe (verde gen, vermelho imp). A linha cinza "
                "tracejada marca o threshold de EER calculado on-the-fly do dataset "
                "de referência — ponto onde FNR=FPR. Os escores do caso (azul) "
                "podem ser comparados visualmente: à direita do threshold sugere "
                "Hp, à esquerda sugere Hd. Para evidência probabilística "
                "(log-LR), use o modo CALIBRATED após passar os scores por "
                "SCORE_CALIBRATION."
            )
        self._help_label.setText(html)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        self.figure.clear()
        # Constrained layout reserva espaço pra ylabel/xlabel/legenda
        # automaticamente — substitui o tight_layout() que não dava conta
        # do ylabel "Densidade" em cells embedados estreitos.
        self.figure.set_layout_engine("constrained")
        if self._result is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5, 0.5, "(aguardando resultado)", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#888",
            )
            ax.axis("off")
            self.canvas.draw_idle()
            return

        try:
            self._draw(self.figure, self._result, show_title=self._show_title)
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
    def _draw(figure: Figure, r: Any, show_title: bool = True) -> None:
        """Delega pro helper puro-matplotlib em
        `faceaudit.models.casework_visualization_plot.draw_casework_visualization`.

        Wrapper mantido por compat com chamadores existentes; a lógica de
        desenho foi extraída pra que o snapshot PNG gerado durante a
        persistência use exatamente o mesmo render."""
        from faceaudit.models.casework_visualization_plot import (
            draw_casework_visualization,
        )
        draw_casework_visualization(figure, r, show_title=show_title)
