from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QTextEdit, QSpinBox, QCheckBox, QComboBox,
    QTabWidget, QWidget, QDialogButtonBox, QLabel, QDoubleSpinBox
)
from PyQt6.QtCore import Qt


class NewMiningConfigDialogView(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nova Estratégia de Mineração (Pair Mining)")
        self.resize(650, 750)  # Aumentei para caber o painel do modo Misto
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)

        # --- 1. METADADOS GERAIS ---
        group_general = QGroupBox("Metadados e Configurações Globais")
        form_general = QFormLayout(group_general)

        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ex: Hard Impostors v1 (Balanceado)")
        
        self.txt_desc = QTextEdit()
        self.txt_desc.setMaximumHeight(60)
        self.txt_desc.setPlaceholderText("Descrição opcional...")

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setValue(42)

        self.chk_exclude_self = QCheckBox("Excluir Auto-Comparações (A x A)")
        self.chk_exclude_self.setChecked(True)

        form_general.addRow("Nome da Estratégia:", self.txt_name)
        form_general.addRow("Descrição:", self.txt_desc)
        form_general.addRow("Seed (Aleatoriedade):", self.sp_seed)
        form_general.addRow("", self.chk_exclude_self)

        main_layout.addWidget(group_general)

        # --- 2. REGRAS DE MINERAÇÃO (TABS) ---
        self.tabs = QTabWidget()
        
        self.tab_impostors, self.imp_widgets = self._create_rule_tab("Impostores (Falsos Positivos)")
        self.tab_genuines, self.gen_widgets = self._create_rule_tab("Genuínos (Falsos Negativos)")

        self.tabs.addTab(self.tab_impostors, "Regra para Impostores")
        self.tabs.addTab(self.tab_genuines, "Regra para Genuínos")
        
        main_layout.addWidget(self.tabs)

        # --- 3. BOTÕES (OK / CANCEL) ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)

    def _create_rule_tab(self, title: str):
        """Helper para criar o layout idêntico para Genuínos e Impostores."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # --- Grupo de Estratégia ---
        form_top = QFormLayout()
        lbl_title = QLabel(f"<b>Configuração de {title}</b>")
        form_top.addRow(lbl_title)

        cb_strategy = QComboBox()
        cb_strategy.addItems([
            "Random (Amostragem Aleatória)", 
            "Hard (Focar nos Piores Erros)", 
            "Easy (Focar nos Acertos)",
            "Typical (Comportamento Mediano)",
            "Mixed (Mistura Customizada %)"
        ])
        form_top.addRow("Estratégia de Dificuldade:", cb_strategy)
        layout.addLayout(form_top)

        # --- Grupo Exclusivo para Modo Misto (Escondido por padrão) ---
        group_mixed = QGroupBox("Proporções do Modo Misto (Deve somar 100%)")
        form_mixed = QFormLayout(group_mixed)
        
        sp_hard_pct = QDoubleSpinBox()
        sp_typical_pct = QDoubleSpinBox()
        sp_random_pct = QDoubleSpinBox()
        sp_easy_pct = QDoubleSpinBox()

        for sp in [sp_hard_pct, sp_typical_pct, sp_random_pct, sp_easy_pct]:
            sp.setRange(0.0, 100.0)
            sp.setSuffix(" %")
            sp.setDecimals(1)
            sp.setSingleStep(5.0)

        # Defaults sugeridos (NIST-like)
        sp_hard_pct.setValue(30.0)
        sp_typical_pct.setValue(20.0)
        sp_random_pct.setValue(50.0)
        sp_easy_pct.setValue(0.0)

        form_mixed.addRow("Hard (Difíceis):", sp_hard_pct)
        form_mixed.addRow("Typical (Medianos):", sp_typical_pct)
        form_mixed.addRow("Random (Aleatórios):", sp_random_pct)
        form_mixed.addRow("Easy (Fáceis):", sp_easy_pct)
        
        group_mixed.setVisible(False) # Começa invisível
        layout.addWidget(group_mixed)

        # --- Grupo de Limites e Escopo ---
        form_bottom = QFormLayout()
        
        cb_focus = QComboBox()
        cb_focus.addItems(["Pairs (Minerar Pares Globais)", "Identities (Eger Indivíduos Primeiro)"])

        sp_max_ids = QSpinBox()
        sp_max_ids.setRange(0, 100_000)
        sp_max_ids.setSingleStep(10)
        sp_max_ids.setSpecialValueText("Ilimitado")
        sp_max_ids.setValue(0)
        
        sp_max_per_id = QSpinBox()
        sp_max_per_id.setRange(0, 100_000)
        sp_max_per_id.setSingleStep(10)
        sp_max_per_id.setSpecialValueText("Ilimitado")
        sp_max_per_id.setValue(0)

        sp_max_global = QSpinBox()
        sp_max_global.setRange(0, 10_000_000)
        sp_max_global.setSingleStep(1000)
        sp_max_global.setSpecialValueText("Ilimitado")
        sp_max_global.setValue(0) 

        cb_global_mode = QComboBox()
        cb_global_mode.addItems([
            "Corte Direto (Global Order)", 
            "Balancear por Identidade (Round-Robin Pivot)", 
            "Balancear por Dupla (Round-Robin Identity Pair)"
        ])
        cb_global_mode.setEnabled(False)

        form_bottom.addRow("Foco da Análise:", cb_focus)
        form_bottom.addRow("Máximo de Identidades Analisadas:", sp_max_ids)
        form_bottom.addRow("Máximo de Pares por Identidade:", sp_max_per_id)
        form_bottom.addRow("Máximo de Pares Global (Total):", sp_max_global)
        form_bottom.addRow("Modo do Corte Global:", cb_global_mode)
        
        layout.addLayout(form_bottom)
        layout.addStretch() # Empurra tudo pra cima

        widgets = {
            "strategy": cb_strategy,
            "focus": cb_focus,
            "max_global": sp_max_global,
            "global_mode": cb_global_mode,
            "max_per_id": sp_max_per_id,
            "max_ids": sp_max_ids,
            "group_mixed": group_mixed,
            "sp_hard_pct": sp_hard_pct,
            "sp_typical_pct": sp_typical_pct,
            "sp_random_pct": sp_random_pct,
            "sp_easy_pct": sp_easy_pct
        }
        return tab, widgets