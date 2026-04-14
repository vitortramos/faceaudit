from PyQt6 import QtWidgets, QtCore

class NewProjectView(QtWidgets.QDialog):
    """
    View do Diálogo de Criação de Novo Projeto.
    Solicita o nome, diretório base e opções de portabilidade.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Criar Novo Projeto FaceAudit")
        self.resize(500, 300)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(15)

        # Info
        lbl_info = QtWidgets.QLabel(
            "Um projeto FaceAudit é <b>autocontido</b>. Uma nova pasta será criada no local escolhido "
            "para armazenar o ficheiro .fap, datasets, pipelines e modelos treinados."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #555; margin-bottom: 5px;")
        layout.addWidget(lbl_info)

        form = QtWidgets.QFormLayout()
        
        # 1. Nome do Projeto
        self.input_name = QtWidgets.QLineEdit()
        self.input_name.setPlaceholderText("Ex: Laudo_Caso_123")
        form.addRow("Nome do Projeto:", self.input_name)

        # 2. Localização Base
        row_loc = QtWidgets.QHBoxLayout()
        self.input_location = QtWidgets.QLineEdit()
        self.input_location.setReadOnly(True)
        self.input_location.setPlaceholderText("Selecione onde guardar...")
        
        self.btn_browse = QtWidgets.QPushButton("Procurar...")
        row_loc.addWidget(self.input_location)
        row_loc.addWidget(self.btn_browse)
        form.addRow("Guardar em:", row_loc)

        layout.addLayout(form)

        # 3. Preview do Caminho Final
        self.lbl_preview = QtWidgets.QLabel("Caminho Final: -")
        self.lbl_preview.setStyleSheet("color: #0078d7; font-family: Consolas, monospace; font-size: 11px;")
        self.lbl_preview.setWordWrap(True)
        layout.addWidget(self.lbl_preview)

        # 4. Opções de Portabilidade (Deep Copy)
        self.group_port = QtWidgets.QGroupBox("Opções de Portabilidade")
        self.group_port.setStyleSheet("QGroupBox { font-weight: bold; color: #495057; }")
        l_port = QtWidgets.QVBoxLayout(self.group_port)
        
        self.chk_deps = QtWidgets.QCheckBox("Empacotar Datasets e Pipelines (Obrigatório para reprodutibilidade)")
        self.chk_deps.setChecked(True)
        self.chk_deps.setEnabled(False) # Sempre true
        
        self.chk_images = QtWidgets.QCheckBox("Empacotar Fotografias Originais (Deep Copy)")
        self.chk_images.setToolTip("Copia as imagens faciais fisicamente para dentro da pasta do projeto.\nAumenta o tamanho em disco, mas garante portabilidade total se abrir noutro computador.")
        
        l_port.addWidget(self.chk_deps)
        l_port.addWidget(self.chk_images)
        layout.addWidget(self.group_port)

        layout.addStretch()

        # Botões OK / Cancel
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Criar Projeto")
        layout.addWidget(self.buttons)