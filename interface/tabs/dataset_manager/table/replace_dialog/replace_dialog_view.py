from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, 
    QCheckBox, QDialogButtonBox
)

class ReplaceDialogView(QDialog):
    """
    View do Diálogo de Substituição.
    Responsabilidade: Apenas definir os widgets e o layout.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Substituir na Seleção")
        self.setFixedWidth(300)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Formulário de Entradas
        form = QFormLayout()
        self.input_find = QLineEdit()
        self.input_replace = QLineEdit()
        self.check_exact = QCheckBox("Correspondência Exata")
        
        form.addRow("Localizar:", self.input_find)
        form.addRow("Substituir por:", self.input_replace)
        form.addRow("", self.check_exact)
        
        layout.addLayout(form)

        # Botões Padrão (OK/Cancel)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)