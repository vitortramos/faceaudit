# interface/tabs/strategy_manager_v2/comparison_tab/comparison_tab_widget.py

from PyQt6 import QtWidgets, QtCore


class ComparisonTabWidget(QtWidgets.QWidget):
    """
    Aba de comparação entre estratégias treinadas.

    Implementação futura. Exibe placeholder informativo por enquanto.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        icon = QtWidgets.QLabel("📊")
        icon.setStyleSheet("font-size: 48px;")
        icon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        lbl = QtWidgets.QLabel(
            "Comparação entre estratégias treinadas\n\n"
            "Selecione duas estratégias executadas para comparar métricas lado a lado.\n"
            "Esta funcionalidade será implementada em iteração futura."
        )
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 13px; line-height: 160%;")

        layout.addStretch()
        layout.addWidget(icon)
        layout.addWidget(lbl)
        layout.addStretch()
