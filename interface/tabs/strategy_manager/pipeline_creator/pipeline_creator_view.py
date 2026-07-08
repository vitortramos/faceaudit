from PyQt6 import QtWidgets, QtCore, QtGui
from faceaudit.enums.transformation_type import TransformationType

from interface.theme import apply_themed_stylesheet

class PipelineCreatorView(QtWidgets.QDialog):
    """
    View do criador de pipelines.
    Layout: [Ferramentas] -> [Ações] -> [Sequência] -> [Propriedades]
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Criador de Pipeline de Transformação")
        self.resize(950, 550) # Aumentei a largura para caber a 3ª coluna
        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        
        # --- 1. Cabeçalho ---
        header_group = QtWidgets.QGroupBox("Identificação")
        header_layout = QtWidgets.QFormLayout(header_group)
        
        self.input_name = QtWidgets.QLineEdit()
        self.input_name.setPlaceholderText("Ex: Coral_Whitening_L2_v1")
        
        self.input_desc = QtWidgets.QLineEdit()
        self.input_desc.setPlaceholderText("Descrição opcional...")
        
        header_layout.addRow("Nome do Pipeline:", self.input_name)
        header_layout.addRow("Descrição:", self.input_desc)
        main_layout.addWidget(header_group)

        # --- 2. Área Principal (3 Colunas) ---
        body_layout = QtWidgets.QHBoxLayout()
        
        # COLUNA 1: Ferramentas Disponíveis
        col1_layout = QtWidgets.QVBoxLayout()
        lbl_avail = QtWidgets.QLabel("🛠️ Disponíveis")
        apply_themed_stylesheet(
            lbl_avail,
            lambda t: "font-weight: bold; color: " + t("TEXT") + ";",
        )
        col1_layout.addWidget(lbl_avail)
        
        self.list_available = QtWidgets.QListWidget()
        self.list_available.setDragEnabled(True)
        col1_layout.addWidget(self.list_available)
        
        # COLUNA 2: Sequência (Com botões de ação no meio)
        # Container para botões de mover (Meio-Esquerda)
        btns_layout = QtWidgets.QVBoxLayout()
        btns_layout.addStretch()
        self.btn_add = QtWidgets.QPushButton("➡")
        self.btn_remove = QtWidgets.QPushButton("❌")
        apply_themed_stylesheet(
            self.btn_remove,
            lambda t: "color: " + t("DANGER") + "; font-weight: bold;",
        )
        btns_layout.addWidget(self.btn_add)
        btns_layout.addWidget(self.btn_remove)
        btns_layout.addStretch()

        # Lista de Sequência
        col2_layout = QtWidgets.QVBoxLayout()
        lbl_seq = QtWidgets.QLabel("🚀 Sequência de Execução")
        apply_themed_stylesheet(
            lbl_seq,
            lambda t: "font-weight: bold; color: " + t("PRIMARY") + ";",
        )
        col2_layout.addWidget(lbl_seq)
        
        self.list_sequence = QtWidgets.QListWidget()
        self.list_sequence.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        col2_layout.addWidget(self.list_sequence)
        
        # Botões de Reordenação
        order_layout = QtWidgets.QHBoxLayout()
        self.btn_up = QtWidgets.QPushButton("⬆")
        self.btn_down = QtWidgets.QPushButton("⬇")
        order_layout.addWidget(self.btn_up)
        order_layout.addWidget(self.btn_down)
        col2_layout.addLayout(order_layout)

        # COLUNA 3: Propriedades (O Inspector)
        col3_layout = QtWidgets.QVBoxLayout()
        self.group_props = QtWidgets.QGroupBox("⚙️ Parâmetros")
        self.group_props.setStyleSheet("QGroupBox { font-weight: bold; color: #6610f2; }")
        self.layout_props = QtWidgets.QVBoxLayout(self.group_props)
        self.layout_props.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        
        # Placeholder inicial
        self.lbl_no_props = QtWidgets.QLabel("Selecione um item na\nsequência para editar.")
        self.lbl_no_props.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        apply_themed_stylesheet(
            self.lbl_no_props,
            lambda t: "color: " + t("TEXT_MUTED") + ";",
        )
        self.layout_props.addWidget(self.lbl_no_props)
        
        col3_layout.addWidget(self.group_props)

        # Adiciona as colunas ao layout principal
        body_layout.addLayout(col1_layout, stretch=2)
        body_layout.addLayout(btns_layout, stretch=0)
        body_layout.addLayout(col2_layout, stretch=2)
        body_layout.addLayout(col3_layout, stretch=3) # Propriedades ganha mais espaço
        
        main_layout.addLayout(body_layout)

        # --- 3. Rodapé ---
        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.rejected.connect(self.reject)
        main_layout.addWidget(self.buttons)

    def add_available_item(self, label: str, tooltip: str, enum_data: TransformationType):
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, enum_data)
        item.setToolTip(tooltip)
        self.list_available.addItem(item)

    def show_warning(self, message: str):
        QtWidgets.QMessageBox.warning(self, "Atenção", message)

    def set_properties_widget(self, widget: QtWidgets.QWidget):
        """
        Substitui o conteúdo do painel de propriedades por um formulário específico.
        """
        # Limpa o layout atual
        while self.layout_props.count():
            child = self.layout_props.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        if widget:
            self.layout_props.addWidget(widget)
        else:
            self.layout_props.addWidget(self.lbl_no_props)
            self.lbl_no_props.setVisible(True)