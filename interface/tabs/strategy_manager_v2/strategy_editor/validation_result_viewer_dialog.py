# interface/tabs/strategy_manager_v2/strategy_editor/validation_result_viewer_dialog.py

"""
Dialog wrapper sobre `ValidationResultWidget`.

Originalmente continha toda a lógica de plot dos 4 gráficos do
StrategyValidationResult. A lógica foi extraída para o widget
reutilizável `result_widgets.ValidationResultWidget`, e este dialog
agora é um wrapper fino — usado pelo botão "📊 Visualizar Resultado"
em pontos onde se quer abrir em janela cheia (incluindo o botão
🔍 Expandir do `InstructionResultsPanel`).
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6 import QtWidgets

from .result_widgets.validation_result_widget import ValidationResultWidget


class ValidationResultViewerDialog(QtWidgets.QDialog):
    """
    Dialog em tela cheia para visualizar um StrategyValidationResult.

    Embute um `ValidationResultWidget` com header, toolbar matplotlib
    e textos de ajuda contextuais por aba.
    """

    def __init__(
        self,
        result: Any,  # StrategyValidationResult
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle(f"Resultado de Validação: {result.strategy_name}")
        self.resize(1100, 780)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Widget reutilizável — modo dialog: header + toolbar + ajuda.
        self.widget = ValidationResultWidget(
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
