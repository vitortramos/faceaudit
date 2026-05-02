# interface/tabs/strategy_manager_v2/strategy_editor/casework_visualization_viewer_dialog.py

"""
Dialog wrapper sobre `CaseworkVisualizationWidget`.

Originalmente continha a lógica de plot do CaseworkVisualizationResult.
A lógica foi extraída para o widget reutilizável
`result_widgets.CaseworkVisualizationWidget`, e este dialog agora é um
wrapper fino — usado pelo botão "📊 Visualizar Caso" em pontos onde
se quer abrir em janela cheia (incluindo o botão 🔍 Expandir do
`InstructionResultsPanel`).
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6 import QtWidgets

from .result_widgets.casework_visualization_widget import CaseworkVisualizationWidget


class CaseworkVisualizationViewerDialog(QtWidgets.QDialog):
    """
    Dialog em tela cheia para visualizar um CaseworkVisualizationResult.

    Embute um `CaseworkVisualizationWidget` com header, toolbar matplotlib
    e texto de ajuda contextual.
    """

    def __init__(
        self,
        result: Any,  # CaseworkVisualizationResult
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        title = (
            f"Visualização Forense de Caso — "
            f"{result.case_dataset_name or 'caso'} sobre "
            f"{result.reference_dataset_name or 'referência'}"
        )
        self.setWindowTitle(title)
        self.resize(950, 720)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Widget reutilizável — modo dialog: header + toolbar + ajuda.
        self.widget = CaseworkVisualizationWidget(
            parent=self,
            result=result,
            show_header=True,
            show_toolbar=True,
            show_help=True,
        )
        layout.addWidget(self.widget, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)
