from PyQt6 import QtWidgets, QtCore

class SaveProjectView(QtWidgets.QDialog):
    """
    View do Diálogo de Opções de Salvamento.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Opções de Salvamento")
        self.resize(450, 180)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        lbl = QtWidgets.QLabel("O projeto será atualizado no disco. Selecione opções extras:")
        lbl.setStyleSheet("font-weight: bold; color: #333;")
        layout.addWidget(lbl)
        layout.addSpacing(10)
        
        # Checkbox 1: Deep Copy
        self.chk_images = QtWidgets.QCheckBox("Empacotar Fotografias Originais (Deep Copy)")
        self.chk_images.setToolTip("Copia fisicamente as imagens novas para o projeto. Ignora as que já existem.")
        
        # Checkbox 2: Garbage Collector
        self.chk_clean = QtWidgets.QCheckBox("Limpar Arquivos Órfãos (Garbage Collector)")
        self.chk_clean.setToolTip("Apaga fisicamente do pendrive os datasets, pipelines e imagens que não estão mais referenciados neste projeto.")
        self.chk_clean.setStyleSheet("color: #d9534f;") # Tom avermelhado para indicar ação destrutiva
        self.chk_clean.setChecked(False) # Segurança em primeiro lugar
        
        layout.addWidget(self.chk_images)
        layout.addWidget(self.chk_clean)
        layout.addStretch()
        
        # Botões
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Salvar Projeto")
        layout.addWidget(self.buttons)