from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, 
    QPushButton, QGroupBox, QLabel
)
from PyQt6.QtCore import Qt

from faceaudit.models.split_config import SplitMode, GroupingStrategy

class NewSplitConfigDialogView(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuração de Split de Dados (PLDA)")
        self.setMinimumWidth(450)
        
        self.layout = QVBoxLayout(self)
        self._build_ui()

    def _build_ui(self):
        # --- Identificação (Nome e Descrição) ---
        info_group = QGroupBox("Identificação")
        info_layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ex: Split_Padrao_20_Porcento")
        
        self.desc_input = QLineEdit()
        self.desc_input.setPlaceholderText("Breve descrição desta configuração...")
        
        info_layout.addRow("Nome da Configuração:", self.name_input)
        info_layout.addRow("Descrição:", self.desc_input)
        info_group.setLayout(info_layout)
        self.layout.addWidget(info_group)

        # --- Estratégia de Agrupamento ---
        grouping_group = QGroupBox("Estratégia de Agrupamento (Sósias)")
        grouping_layout = QFormLayout()
        
        self.grouping_combo = QComboBox()
        for strategy in GroupingStrategy:
            self.grouping_combo.addItem(strategy.value, strategy)
            
        self.cluster_size_spin = QSpinBox()
        self.cluster_size_spin.setRange(2, 20)
        self.cluster_size_spin.setValue(3)
        self.cluster_size_spin.setToolTip("O sorteado + N piores sósias (Apenas para PRESERVE_LOOKALIKES)")
        
        grouping_layout.addRow("Estratégia:", self.grouping_combo)
        grouping_layout.addRow("Tamanho do Cluster:", self.cluster_size_spin)
        grouping_group.setLayout(grouping_layout)
        self.layout.addWidget(grouping_group)

        # --- Metas e Proporções ---
        target_group = QGroupBox("Metas de Corte")
        target_layout = QFormLayout()
        
        self.mode_combo = QComboBox()
        for mode in SplitMode:
            self.mode_combo.addItem(mode.value, mode)
            
        self.ratio_spin = QDoubleSpinBox()
        self.ratio_spin.setRange(0.01, 0.99)
        self.ratio_spin.setSingleStep(0.05)
        self.ratio_spin.setValue(0.20)
        self.ratio_spin.setToolTip("Fração da base para calibração (0.0 a 1.0)")
        
        self.min_pairs_spin = QSpinBox()
        self.min_pairs_spin.setRange(0, 1000000)
        self.min_pairs_spin.setSingleStep(1000)
        self.min_pairs_spin.setValue(3000)
        
        self.min_ids_spin = QSpinBox()
        self.min_ids_spin.setRange(2, 10000)
        self.min_ids_spin.setValue(40)
        
        target_layout.addRow("Modo de Corte:", self.mode_combo)
        target_layout.addRow("Fração de Calibração (Ratio):", self.ratio_spin)
        target_layout.addRow("Mínimo de Pares (Alvo/Piso):", self.min_pairs_spin)
        target_layout.addRow("Mínimo de IDs (Alvo/Piso):", self.min_ids_spin)
        target_group.setLayout(target_layout)
        self.layout.addWidget(target_group)

        # --- Filtros e Reprodutibilidade ---
        misc_group = QGroupBox("Filtros e Reprodutibilidade")
        misc_layout = QFormLayout()
        
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(2, 100)
        self.min_samples_spin.setValue(2)
        
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        
        misc_layout.addRow("Mínimo de Fotos por Pessoa:", self.min_samples_spin)
        misc_layout.addRow("Semente Aleatória (Seed):", self.seed_spin)
        misc_group.setLayout(misc_layout)
        self.layout.addWidget(misc_group)

        # --- Botões de Ação ---
        buttons_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_save = QPushButton("Salvar Configuração")
        self.btn_save.setDefault(True)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(self.btn_cancel)
        buttons_layout.addWidget(self.btn_save)
        self.layout.addLayout(buttons_layout)