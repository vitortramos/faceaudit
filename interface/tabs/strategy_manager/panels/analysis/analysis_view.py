from PyQt6 import QtWidgets, QtCore

class AnalysisView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        
        lbl = QtWidgets.QLabel("Painel 3: Análise de Caso\n(Aplicação Forense e Laudo)")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("border: 2px dashed #d9534f; color: #d9534f; padding: 20px;")
        
        layout.addWidget(lbl)