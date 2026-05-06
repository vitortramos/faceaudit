from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QPlainTextEdit, QComboBox, QGroupBox, QStackedWidget,
    QWidget, QSpinBox, QDoubleSpinBox, QFrame
)

# Modelos suportados pelo FFService — duplicado da EnrichDatasetView
# pra manter a UI auto-contida (lista pequena, raramente muda).
_SUPPORTED_FF_MODELS = ("sepaelv2", "sepaelv4")

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
        vbox_pipe.setSpacing(6)

        # Forensic Face — checkbox + botão Avançado + painel
        self.chk_ff = QCheckBox("Forensic Face (Biometria)")
        self.chk_ff.setChecked(True)
        self._ff_adv_btn = QPushButton("▸ Avançado")
        self._ff_adv_btn.setCheckable(True)
        self._ff_adv_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        ff_row = QHBoxLayout()
        ff_row.addWidget(self.chk_ff)
        ff_row.addStretch()
        ff_row.addWidget(self._ff_adv_btn)
        vbox_pipe.addLayout(ff_row)
        self._ff_adv_panel = self._build_ff_advanced_panel()
        self._ff_adv_panel.setVisible(False)
        vbox_pipe.addWidget(self._ff_adv_panel)
        self._ff_adv_btn.toggled.connect(
            lambda c: (
                self._ff_adv_panel.setVisible(c),
                self._ff_adv_btn.setText("▾ Avançado" if c else "▸ Avançado"),
            )
        )

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("color: #e0e0e0; max-height: 1px;")
        vbox_pipe.addWidget(sep1)

        # OFIQ — só checkbox (sem painel avançado)
        self.chk_ofiq = QCheckBox("OFIQ (Qualidade ISO)")
        self.chk_ofiq.setChecked(True)
        vbox_pipe.addWidget(self.chk_ofiq)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #e0e0e0; max-height: 1px;")
        vbox_pipe.addWidget(sep2)

        # SER-FIQ — checkbox + botão Avançado + painel
        self.chk_serfiq = QCheckBox("SER-FIQ (Robustez)")
        self.chk_serfiq.setChecked(True)
        self._serfiq_adv_btn = QPushButton("▸ Avançado")
        self._serfiq_adv_btn.setCheckable(True)
        self._serfiq_adv_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        sf_row = QHBoxLayout()
        sf_row.addWidget(self.chk_serfiq)
        sf_row.addStretch()
        sf_row.addWidget(self._serfiq_adv_btn)
        vbox_pipe.addLayout(sf_row)
        self._serfiq_adv_panel = self._build_serfiq_advanced_panel()
        self._serfiq_adv_panel.setVisible(False)
        vbox_pipe.addWidget(self._serfiq_adv_panel)
        self._serfiq_adv_btn.toggled.connect(
            lambda c: (
                self._serfiq_adv_panel.setVisible(c),
                self._serfiq_adv_btn.setText("▾ Avançado" if c else "▸ Avançado"),
            )
        )

        layout.addWidget(group_pipe)

        # --- BOTÕES ---
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        layout.addWidget(self.buttons)

    # ──────────────────────────────────────────────────────────────────
    # Painéis avançados (mesmo formato da EnrichDatasetView)
    # ──────────────────────────────────────────────────────────────────

    def _build_ff_advanced_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            "QWidget { background: #f7f7f7; border-left: 3px solid #b0bec5; "
            "padding: 8px; }"
        )
        v = QVBoxLayout(panel)
        v.setContentsMargins(20, 4, 4, 4)
        v.setSpacing(6)

        v.addWidget(QLabel("<b>Modelos de embedding</b>"))
        self._ff_model_checks = {}
        models_row = QHBoxLayout()
        for name in _SUPPORTED_FF_MODELS:
            chk = QCheckBox(name.replace("sepaelv", "SEPAEL V"))
            self._ff_model_checks[name] = chk
            models_row.addWidget(chk)
        self._ff_model_checks["sepaelv2"].setChecked(True)
        models_row.addStretch()
        v.addLayout(models_row)

        det_row = QHBoxLayout()
        det_row.addWidget(QLabel("Det size:"))
        self._ff_det_size_spin = QSpinBox()
        self._ff_det_size_spin.setRange(160, 1280)
        self._ff_det_size_spin.setSingleStep(32)
        self._ff_det_size_spin.setValue(320)
        det_row.addWidget(self._ff_det_size_spin)
        det_row.addStretch()
        v.addLayout(det_row)

        self._ff_extended_chk = QCheckBox(
            "Modo completo: FIQA + gender + age + yaw/pitch/roll"
        )
        self._ff_extended_chk.setChecked(True)
        v.addWidget(self._ff_extended_chk)
        return panel

    def _build_serfiq_advanced_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet(
            "QWidget { background: #f7f7f7; border-left: 3px solid #b0bec5; "
            "padding: 8px; }"
        )
        v = QVBoxLayout(panel)
        v.setContentsMargins(20, 4, 4, 4)
        v.setSpacing(6)

        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("T (iterações):"))
        self._serfiq_t_spin = QSpinBox()
        self._serfiq_t_spin.setRange(1, 100)
        self._serfiq_t_spin.setValue(10)
        t_row.addWidget(self._serfiq_t_spin)
        t_row.addStretch()
        v.addLayout(t_row)

        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Alpha:"))
        self._serfiq_alpha_spin = QDoubleSpinBox()
        self._serfiq_alpha_spin.setRange(0.1, 10.0)
        self._serfiq_alpha_spin.setSingleStep(0.1)
        self._serfiq_alpha_spin.setValue(1.5)
        alpha_row.addWidget(self._serfiq_alpha_spin)
        alpha_row.addStretch()
        v.addLayout(alpha_row)

        return panel

    # ──────────────────────────────────────────────────────────────────
    # Getters dos params operacionais (consumidos pelo controller)
    # ──────────────────────────────────────────────────────────────────

    def get_ff_models(self) -> list:
        models = [n for n, c in self._ff_model_checks.items() if c.isChecked()]
        return models or ["sepaelv2"]

    def get_ff_det_size(self) -> int:
        return int(self._ff_det_size_spin.value())

    def get_ff_extended(self) -> bool:
        return bool(self._ff_extended_chk.isChecked())

    def get_serfiq_t(self) -> int:
        return int(self._serfiq_t_spin.value())

    def get_serfiq_alpha(self) -> float:
        return float(self._serfiq_alpha_spin.value())