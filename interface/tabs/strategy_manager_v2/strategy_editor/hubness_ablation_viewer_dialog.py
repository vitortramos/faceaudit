# interface/tabs/strategy_manager_v2/strategy_editor/hubness_ablation_viewer_dialog.py

"""
Dialog full-screen para visualização do AblationStudyInfo.

Wrapper fino sobre HubnessAblationWidget — single-plot por natureza,
não precisa de galeria como o HubnessReportViewerDialog.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6 import QtWidgets

from .result_widgets.hubness_ablation_widget import HubnessAblationWidget


class HubnessAblationViewerDialog(QtWidgets.QDialog):
    """
    Dialog full-screen para visualização de um AblationStudyInfo.

    Embute um HubnessAblationWidget com header, toolbar e ajuda.
    """

    def __init__(
        self,
        result: Any,  # AblationStudyInfo
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._result = result
        self.setWindowTitle("Estudo de Ablação Topológica — Hubness")
        self.resize(950, 720)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.widget = HubnessAblationWidget(
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
