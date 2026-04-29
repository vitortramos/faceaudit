# interface/tabs/strategy_manager_v2/strategy_editor/graph_viewer_dialog.py

from __future__ import annotations

from typing import Callable, Optional

from PyQt6 import QtWidgets

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure


class GraphViewerDialog(QtWidgets.QDialog):
    """
    Dialog genérico para exibir gráficos matplotlib em tamanho cheio.

    Uso típico (instrução produz Figure):
        dlg = GraphViewerDialog("KDE Plot", figure=fig, parent=self)
        dlg.exec()

    Uso com função de desenho lazy:
        dlg = GraphViewerDialog.from_draw_fn(
            "Trajetória", draw_fn=lambda f: trajectory.plot(f), parent=self
        )
        dlg.exec()
    """

    def __init__(
        self,
        title: str,
        figure: Optional[Figure] = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 700)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.figure = figure if figure is not None else Figure(figsize=(10, 7), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)

        self.canvas.draw()

    @classmethod
    def from_draw_fn(
        cls,
        title: str,
        draw_fn: Callable[[Figure], None],
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> "GraphViewerDialog":
        dlg = cls(title, figure=None, parent=parent)
        draw_fn(dlg.figure)
        dlg.canvas.draw()
        return dlg
