# interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/typicality_distribution_widget.py

"""
Widget Qt vivo (matplotlib interativo) pra inspeção de um KNNTypicalityModel
treinado. Usado SÓ pelo TypicalityDistributionViewerDialog (botão 🔍 Expandir).

O painel inline NÃO usa este widget — usa _PngSnapshotWidget com o PNG já
gravado pelo executor durante a persistência do modelo (mesmo padrão do
SimilarityScoring/Casework/ValidationResult). Manter este widget enxuto:
ele só existe pra dar interação (zoom, pan, save-as-png via NavigationToolbar)
no dialog full-screen.

O desenho é delegado pra função pura `draw_typicality_distribution_from_model`
(em `faceaudit/models/typicality_distribution_plot.py`), que também é usada
pelo renderizador headless de PNG. Garantia de consistência visual entre
o snapshot inline e o widget vivo.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from faceaudit.models.typicality_distribution_plot import (
    draw_typicality_distribution_from_model,
)


class TypicalityDistributionWidget(QtWidgets.QWidget):
    """Plot vivo da distribuição empírica de T_k (LOO) do modelo treinado."""

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
        # Mesmo padrão de HubnessReportWidget — NÃO chama setParent(self)
        # separadamente; o addWidget abaixo já faz o reparent via layout.
        # Sem isso o canvas existe brevemente como top-level e o showEvent
        # do backend Qt pode chegar num canvas zombie em PyQt6
        # (TypeError: 'NoneType' object is not callable).
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
        self.figure.set_layout_engine("constrained")
        try:
            draw_typicality_distribution_from_model(
                self.figure, self._model, show_title=self._show_title,
            )
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
