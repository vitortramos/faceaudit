from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QStackedWidget, QWidget, QDoubleSpinBox,
    QListWidget, QPushButton, QDialogButtonBox, QFormLayout, QGroupBox
)

from interface.theme import apply_themed_stylesheet

class SplitDatasetDialogView(QDialog):
    """
    Interface gráfica para a ferramenta de Divisão (Split) de Datasets.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dividir Dataset")
        self.resize(600, 550)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # --- 1. CABEÇALHO (Info do Dataset) ---
        group_info = QGroupBox("Dataset Original")
        info_layout = QVBoxLayout(group_info)
        self.lbl_dataset_info = QLabel("Carregando...")
        apply_themed_stylesheet(
            self.lbl_dataset_info,
            lambda t: "color: " + t("TEXT") + "; font-weight: bold;",
        )
        info_layout.addWidget(self.lbl_dataset_info)
        main_layout.addWidget(group_info)

        # --- 2. CONFIGURAÇÃO DE SAÍDA (Nomes) ---
        group_output = QGroupBox("Configuração de Saída")
        form_output = QFormLayout(group_output)
        
        self.txt_name_a = QLineEdit()
        self.txt_name_b = QLineEdit()
        
        form_output.addRow("Nome do Dataset A (Treino/Base):", self.txt_name_a)
        form_output.addRow("Nome do Dataset B (Teste/Alvo):", self.txt_name_b)
        main_layout.addWidget(group_output)

        # --- 3. MÉTODO DE DIVISÃO ---
        group_method = QGroupBox("Critério de Divisão")
        method_layout = QVBoxLayout(group_method)
        
        form_method = QFormLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Divisão Aleatória por Porcentagem",
            "Seleção Manual de Identidades"
        ])
        form_method.addRow("Método:", self.cmb_mode)
        method_layout.addLayout(form_method)

        # Stacked Widget para o modo selecionado
        self.stack = QStackedWidget()
        
        # Página 0: Porcentagem
        page_ratio = QWidget()
        layout_ratio = QVBoxLayout(page_ratio)
        layout_ratio.setContentsMargins(0, 10, 0, 0)
        
        form_ratio = QFormLayout()
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.05, 0.95)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.80)
        self.spin_ratio.setSuffix(" (Fração para o Dataset A)")
        
        form_ratio.addRow("Proporção (A / Total):", self.spin_ratio)
        layout_ratio.addLayout(form_ratio)
        layout_ratio.addStretch()
        self.stack.addWidget(page_ratio)

        # Página 1: Manual (Listas)
        page_manual = QWidget()
        layout_manual = QHBoxLayout(page_manual)
        layout_manual.setContentsMargins(0, 10, 0, 0)
        
        # Lista A (Restante)
        vbox_a = QVBoxLayout()
        vbox_a.addWidget(QLabel("Identidades no Dataset A:"))
        self.list_a = QListWidget()
        self.list_a.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        vbox_a.addWidget(self.list_a)
        
        # Botões do meio
        vbox_btns = QVBoxLayout()
        vbox_btns.addStretch()
        self.btn_move_right = QPushButton(">>")
        self.btn_move_right.setToolTip("Mover para Dataset B")
        self.btn_move_left = QPushButton("<<")
        self.btn_move_left.setToolTip("Mover para Dataset A")
        vbox_btns.addWidget(self.btn_move_right)
        vbox_btns.addWidget(self.btn_move_left)
        vbox_btns.addStretch()
        
        # Lista B (Selecionados)
        vbox_b = QVBoxLayout()
        vbox_b.addWidget(QLabel("Identidades no Dataset B:"))
        self.list_b = QListWidget()
        self.list_b.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        vbox_b.addWidget(self.list_b)
        
        layout_manual.addLayout(vbox_a)
        layout_manual.addLayout(vbox_btns)
        layout_manual.addLayout(vbox_b)
        self.stack.addWidget(page_manual)

        method_layout.addWidget(self.stack)
        main_layout.addWidget(group_method)

        # --- 4. AÇÃO PÓS-DIVISÃO ---
        group_action = QGroupBox("Ação Pós-Divisão")
        action_layout = QFormLayout(group_action)
        self.cmb_action = QComboBox()
        self.cmb_action.addItems([
            "Manter Dataset Original Aberto",
            "Carregar Dataset A no visualizador",
            "Carregar Dataset B no visualizador"
        ])
        action_layout.addRow("Após concluir:", self.cmb_action)
        main_layout.addWidget(group_action)

        # --- BOTÕES DE AÇÃO ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Dividir e Salvar")
        main_layout.addWidget(self.button_box)