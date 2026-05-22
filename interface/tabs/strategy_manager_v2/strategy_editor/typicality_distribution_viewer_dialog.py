# interface/tabs/strategy_manager_v2/strategy_editor/typicality_distribution_viewer_dialog.py

"""
Dialog wrapper sobre `TypicalityDistributionWidget`.

Aberto pelo botão "🔍 Expandir" na 3ª coluna da aba Strategy Editor V2
quando a instrução é `KNN_TYPICALITY_MODEL_TRAINING`. Reusa o mesmo
widget de plot inline, agora com NavigationToolbar matplotlib (zoom,
pan, salvar PNG).
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6 import QtWidgets

from .result_widgets.typicality_distribution_widget import TypicalityDistributionWidget


class TypicalityDistributionViewerDialog(QtWidgets.QDialog):
    """Janela em tela cheia para inspeção de um KNNTypicalityModel treinado."""

    def __init__(
        self,
        model: Any,  # KNNTypicalityModel
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._model = model
        pop = getattr(model, "population_label", None) or "(sem rótulo)"
        n = getattr(model, "sample_count", "?")
        d = getattr(model, "embedding_dim", "?")
        k = getattr(model, "k", "?")
        self.setWindowTitle(
            f"Modelo de Tipicidade — {pop}  "
            f"|  N={n}  D={d}  k={k}"
        )
        self.resize(900, 620)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.widget = TypicalityDistributionWidget(
            parent=self,
            model=model,
            show_toolbar=True,
            show_title=True,
        )
        layout.addWidget(self.widget, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_close = QtWidgets.QPushButton("Fechar", parent=self)
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        layout.addLayout(btn_row)
