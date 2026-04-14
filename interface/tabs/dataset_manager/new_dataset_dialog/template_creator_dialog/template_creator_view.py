from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QListWidget, QComboBox, QDialogButtonBox, 
    QTabWidget, QWidget, QCheckBox, QSpinBox, QFormLayout
)

class TemplateCreatorView(QDialog):
    """
    View do Criador de Templates.
    Responsável apenas pela interface gráfica.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Criador de Template de Estrutura")
        self.resize(550, 600)
        
        self.mapping_options = ["identity", "gender", "ethnicity", "origin", "Ignorar"]
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- CABEÇALHO ---
        top_form = QFormLayout()
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ex: BFW_Oficial")
        
        # Separador
        sep_layout = QHBoxLayout()
        self.chk_use_sep = QCheckBox("Usar Separador para dividir campos")
        self.txt_separator = QLineEdit("_")
        self.txt_separator.setFixedWidth(35)
        self.txt_separator.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.txt_separator.setEnabled(False)
        self.chk_use_sep.toggled.connect(self.txt_separator.setEnabled)
        
        sep_layout.addWidget(self.chk_use_sep)
        sep_layout.addWidget(self.txt_separator)
        sep_layout.addStretch()

        top_form.addRow("Nome do Template:", self.txt_name)
        top_form.addRow("Configuração de Split:", sep_layout)
        layout.addLayout(top_form)

        # --- TABS ---
        self.tabs = QTabWidget()
        
        # ABA 1: PASTAS
        self.tab_folders = QWidget()
        self._init_folders_tab()
        self.tabs.addTab(self.tab_folders, "📁 Estrutura de Pastas")
        
        # ABA 2: ARQUIVOS
        self.tab_files = QWidget()
        self._init_files_tab()
        self.tabs.addTab(self.tab_files, "📄 Nome do Arquivo")
        
        layout.addWidget(self.tabs)

        # Atalho Identidade
        self.chk_parent = QCheckBox("Usar nome da pasta pai imediata como Identidade")
        layout.addWidget(self.chk_parent)

        # Botões
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept) 
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def _init_folders_tab(self):
        layout = QVBoxLayout(self.tab_folders)
        layout.addWidget(QLabel("Mapeamento de Pastas (Defina o Nível):"))
        
        self.list_folders = QListWidget()
        layout.addWidget(self.list_folders)
        
        controls = QHBoxLayout()
        
        self.spin_folder_level = QSpinBox()
        self.spin_folder_level.setRange(0, 10)
        self.spin_folder_level.setToolTip("Nível da pasta (0=Raiz do Dataset)")
        self.spin_folder_level.setPrefix("Nível ")
        
        self.cmb_folder_attr = QComboBox()
        self.cmb_folder_attr.addItems(self.mapping_options)
        
        self.btn_add_f = QPushButton("Adicionar")
        self.btn_rem_f = QPushButton("Remover")
        
        controls.addWidget(self.spin_folder_level)
        controls.addWidget(self.cmb_folder_attr)
        controls.addWidget(self.btn_add_f)
        controls.addWidget(self.btn_rem_f)
        layout.addLayout(controls)

    def _init_files_tab(self):
        layout = QVBoxLayout(self.tab_files)
        layout.addWidget(QLabel("Mapeamento de Arquivo (Split):"))
        
        self.list_files = QListWidget()
        layout.addWidget(self.list_files)
        
        controls = QHBoxLayout()
        
        self.spin_file_idx = QSpinBox()
        self.spin_file_idx.setRange(0, 10)
        self.spin_file_idx.setPrefix("Parte ")
        
        self.cmb_file_attr = QComboBox()
        self.cmb_file_attr.addItems(self.mapping_options)
        
        self.btn_add_file = QPushButton("Adicionar")
        self.btn_rem_file = QPushButton("Remover")
        
        controls.addWidget(self.spin_file_idx)
        controls.addWidget(self.cmb_file_attr)
        controls.addWidget(self.btn_add_file)
        controls.addWidget(self.btn_rem_file)
        layout.addLayout(controls)

    # --- API de Manipulação Visual ---

    def add_folder_item(self, level, attr):
        self.list_folders.addItem(f"[Nível {level}] -> {attr}")

    def remove_folder_item(self):
        curr = self.list_folders.currentRow()
        if curr >= 0:
            self.list_folders.takeItem(curr)

    def add_file_item(self, idx, attr):
        self.list_files.addItem(f"[Parte {idx}] -> {attr}")

    def remove_file_item(self):
        curr = self.list_files.currentRow()
        if curr >= 0:
            self.list_files.takeItem(curr)