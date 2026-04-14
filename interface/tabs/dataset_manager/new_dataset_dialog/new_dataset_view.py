from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QPlainTextEdit, QComboBox, QGroupBox, QStackedWidget,
    QWidget
)

class NewDatasetView(QDialog):
    """
    View do Diálogo de Criação de Novo Dataset.
    Responsável apenas pela disposição dos widgets.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Novo Dataset")
        self.resize(550, 600)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # --- SEÇÃO 1: INPUT (Caminho e Nome) ---
        self.form_input = QFormLayout()
        
        self.txt_path = QLineEdit()
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(40)
        
        row_path = QHBoxLayout()
        row_path.addWidget(self.txt_path)
        row_path.addWidget(self.btn_browse)
        self.form_input.addRow("Pasta Imagens:", row_path)
        
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Nome do Dataset")
        self.form_input.addRow("Nome:", self.txt_name)
        
        self.chk_recursive = QCheckBox("Buscar em subpastas (Recursivo)")
        self.chk_recursive.setChecked(True)
        self.form_input.addRow("", self.chk_recursive)
        
        layout.addLayout(self.form_input)

        # --- SEÇÃO 2: ESTRATÉGIA (Stack para alternar visualização) ---
        self.stack_strategy = QStackedWidget()
        
        # Página 0: Modo Flat (Não Recursivo / Simples)
        self.page_flat = QGroupBox("Metadados (Aplicar a todos)")
        form_flat = QFormLayout(self.page_flat)
        self.txt_identity_flat = QLineEdit()
        self.cmb_gender = QComboBox()
        self.cmb_gender.addItems(["", "Male", "Female"])
        self.txt_ethnicity = QLineEdit()
        
        form_flat.addRow("Identidade:", self.txt_identity_flat)
        form_flat.addRow("Gênero:", self.cmb_gender)
        form_flat.addRow("Etnia:", self.txt_ethnicity)
        self.stack_strategy.addWidget(self.page_flat)
        
        # Página 1: Modo Recursivo (Template)
        self.page_recursive = QGroupBox("Extração Automática")
        form_rec = QFormLayout(self.page_recursive)
        
        self.cmb_template = QComboBox()
        self.btn_new_template = QPushButton("Criar Template")
        
        row_template = QHBoxLayout()
        row_template.addWidget(self.cmb_template, 1)
        row_template.addWidget(self.btn_new_template)
        
        form_rec.addRow("Template:", row_template)
        form_rec.addRow(QLabel("<i>Identidade será extraída das pastas ou arquivos conforme o template.</i>"))
        self.stack_strategy.addWidget(self.page_recursive)
        
        layout.addWidget(self.stack_strategy)

        # --- SEÇÃO 3: DETALHES GERAIS ---
        form_bottom = QFormLayout()
        self.txt_origin = QLineEdit("Local")
        self.txt_desc = QPlainTextEdit()
        self.txt_desc.setMaximumHeight(50)
        
        form_bottom.addRow("Origem:", self.txt_origin)
        form_bottom.addRow("Descrição:", self.txt_desc)
        layout.addLayout(form_bottom)

        # --- SEÇÃO 4: PIPELINE ---
        group_pipe = QGroupBox("Processamento")
        vbox_pipe = QVBoxLayout(group_pipe)
        self.chk_ff = QCheckBox("Forensic Face (Biometria)")
        self.chk_ff.setChecked(True)
        self.chk_ofiq = QCheckBox("OFIQ (Qualidade ISO)")
        self.chk_ofiq.setChecked(True)
        self.chk_serfiq = QCheckBox("SER-FIQ (Robustez)")
        self.chk_serfiq.setChecked(True)
        
        vbox_pipe.addWidget(self.chk_ff)
        vbox_pipe.addWidget(self.chk_ofiq)
        vbox_pipe.addWidget(self.chk_serfiq)
        layout.addWidget(group_pipe)

        # --- BOTÕES ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)