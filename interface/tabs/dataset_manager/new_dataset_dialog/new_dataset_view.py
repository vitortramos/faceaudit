from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QPushButton, QFormLayout, QDialogButtonBox,
    QPlainTextEdit, QComboBox, QGroupBox, QStackedWidget,
    QWidget, QSpinBox, QDoubleSpinBox, QFrame
)

# Modelos suportados pelo FFService — lista lida em runtime via
# `SepaelModel.available_on_disk()`, pra refletir o que o usuário
# instalou em ~/.forensicface/models/.
from faceaudit.enums.sepael_model import SepaelModel


def _supported_ff_models() -> tuple:
    available = SepaelModel.available_on_disk()
    if available:
        return tuple(m.ff_name for m in available)
    return (SepaelModel.default().ff_name,)

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

        # Marca o dataset como contendo faces JÁ alinhadas (112×112).
        # Sem isso, o FFService re-detectaria a face dentro do crop
        # alinhado durante o enrich, degradando o embedding. Default
        # False — a maioria dos datasets é de fotos originais.
        self.chk_is_aligned = QCheckBox(
            "Estas imagens são faces já alinhadas (112×112)"
        )
        self.chk_is_aligned.setChecked(False)
        self.chk_is_aligned.setToolTip(
            "Marque quando o diretório contém crops 112×112 pós-"
            "ForensicFace align (ex: BFW alinhado, output do Image "
            "Pipeline com Origem=Face alinhada). O FFService usa "
            "process_face_aligned (pula detect+align), ~50% mais "
            "rápido e mais correto."
        )
        self.form_input.addRow("", self.chk_is_aligned)

        # Alinhar faces durante a criação — modo "originais → alinhar
        # agora" em 1 passo. Mutuamente exclusivo com chk_is_aligned.
        self.chk_align_during = QCheckBox(
            "Alinhar faces durante a criação (salvar crops 112×112)"
        )
        self.chk_align_during.setChecked(False)
        self.chk_align_during.setToolTip(
            "Roda detect+align nos originais durante a ingestão e "
            "salva os crops 112×112 como PNG numa pasta destino. "
            "O dataset (.parquet) vai pra biblioteca global de datasets; "
            "a pasta selecionada recebe apenas os PNGs alinhados. "
            "O dataset fica marcado is_aligned=True — evita rodar "
            "detect+align 2× via 'New Dataset + Image Pipeline'. "
            "Requer compute_ff habilitado."
        )
        self.form_input.addRow("", self.chk_align_during)

        # Mutuamente exclusivos: uma face é OU já-alinhada OU vai-ser-
        # alinhada — não as duas. Sinais bloqueados durante o set
        # programático pra evitar feedback loop.
        self.chk_is_aligned.toggled.connect(self._on_is_aligned_toggled)
        self.chk_align_during.toggled.connect(self._on_align_during_toggled)

        # Target dir pros PNGs alinhados (visível só quando
        # chk_align_during=True). Label e widget são escondidos juntos
        # — sem o label explícito, `addRow(str, widget)` cria a QLabel
        # internamente e ela não some quando `row_align_target.hide()`.
        self.row_align_target = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(self.row_align_target)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.txt_align_target = QtWidgets.QLineEdit()
        self.txt_align_target.setPlaceholderText(
            "Pasta destino dos PNGs alinhados (o parquet vai pra biblioteca global)"
        )
        self.btn_browse_align_target = QPushButton("...")
        self.btn_browse_align_target.setFixedWidth(40)
        row_layout.addWidget(self.txt_align_target, 1)
        row_layout.addWidget(self.btn_browse_align_target)
        self._lbl_align_target = QtWidgets.QLabel("Destino PNGs alinhados:")
        self._lbl_align_target.setVisible(False)
        self.row_align_target.setVisible(False)
        self.form_input.addRow(self._lbl_align_target, self.row_align_target)

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
        for name in _supported_ff_models():
            member = SepaelModel.from_ff_name(name)
            label = member.label_pt if member is not None else name
            chk = QCheckBox(label)
            self._ff_model_checks[name] = chk
            models_row.addWidget(chk)
        default_name = SepaelModel.default().ff_name
        if default_name in self._ff_model_checks:
            self._ff_model_checks[default_name].setChecked(True)
        else:
            first = next(iter(self._ff_model_checks.values()), None)
            if first is not None:
                first.setChecked(True)
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

    def set_stage_availability(
        self,
        ff_available: bool,
        ofiq_available: bool,
    ) -> None:
        """Desabilita checkboxes dos estágios cujo serviço externo não
        está disponível. SER-FIQ depende de FF — segue a disponibilidade
        do FF. Cada checkbox desabilitado também limpa a marcação pra
        que `DatasetIngestionConfig` não receba `compute_ff=True` quando
        o FFService não está pronto."""
        ff_tooltip = (
            "ForensicFace não está disponível: nenhum modelo encontrado em "
            "~/.forensicface/models/ ou o serviço falhou ao carregar. "
            "Instale um modelo (sepaelv2/v3/v4/v5) ou troque nas Configurações."
        )
        ofiq_tooltip = (
            "OFIQ não está disponível: executável ou pasta de dados não "
            "encontrados. Verifique os caminhos nas Configurações."
        )

        if not ff_available:
            self.chk_ff.setChecked(False)
            self.chk_ff.setEnabled(False)
            self.chk_ff.setToolTip(ff_tooltip)
            self.chk_serfiq.setChecked(False)
            self.chk_serfiq.setEnabled(False)
            self.chk_serfiq.setToolTip(
                "SER-FIQ depende do ForensicFace, que não está disponível. "
                + ff_tooltip
            )

        if not ofiq_available:
            self.chk_ofiq.setChecked(False)
            self.chk_ofiq.setEnabled(False)
            self.chk_ofiq.setToolTip(ofiq_tooltip)

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

    # ── Slots: mutual exclusion entre is_aligned e align_during ──────
    def _on_is_aligned_toggled(self, checked: bool) -> None:
        if checked and self.chk_align_during.isChecked():
            self.chk_align_during.blockSignals(True)
            self.chk_align_during.setChecked(False)
            self.chk_align_during.blockSignals(False)
            self._lbl_align_target.setVisible(False)
            self.row_align_target.setVisible(False)

    def _on_align_during_toggled(self, checked: bool) -> None:
        if checked and self.chk_is_aligned.isChecked():
            self.chk_is_aligned.blockSignals(True)
            self.chk_is_aligned.setChecked(False)
            self.chk_is_aligned.blockSignals(False)
        self._lbl_align_target.setVisible(checked)
        self.row_align_target.setVisible(checked)