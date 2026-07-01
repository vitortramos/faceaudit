"""View do diálogo "Novo dataset" — redesenho em 3 grupos (Etapa 1).

Layout:
  1 · Pasta de imagens   — caminho + natureza da entrada (originais / já alinhadas)
  2 · Características do dataset — nome, diretório destino, tipo de armazenamento
       (autocontido / caminho absoluto), tipo de imagem (só no autocontido) +
       metadados gerais (origem/descrição) colapsados
  3 · Conteúdo do dataset — embeddings (modelos), qualidade (OFIQ/SER-FIQ),
       e identidade/colunas: checkbox "Usar template de colunas" (off →
       identidade por subdiretório ou única digitada; on → template extrai
       identidade + demais colunas)

Esta classe é SÓ disposição + máquina de estados condicional (mostrar/ocultar/
cinza). Validação, browse, criador de template e get_config ficam no controller.
A Etapa 2 liga gravar-no-destino/autocontido na ingestão; aqui os campos novos
existem e são lidos por getters, mas não mudam o que a ingestão faz ainda.
"""

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QCheckBox, QRadioButton, QButtonGroup, QPushButton, QDialogButtonBox,
    QPlainTextEdit, QComboBox, QGroupBox, QWidget, QSpinBox, QDoubleSpinBox,
    QFrame,
)

from faceaudit.enums.sepael_model import SepaelModel


def _supported_ff_models() -> tuple:
    available = SepaelModel.available_on_disk()
    if available:
        return tuple(m.ff_name for m in available)
    return (SepaelModel.default().ff_name,)


_ADV_STYLE = (
    "QWidget#advPanel { background: #f7f7f7; border-left: 3px solid #b0bec5; }"
)


class NewDatasetView(QDialog):
    """Disposição + visibilidade condicional do diálogo de novo dataset."""

    # Emitido quando qualquer entrada relevante muda — o controller liga em
    # _revalidate() para validação reativa (OK + rodapé).
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None, default_dataset_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Novo dataset")
        self._default_dataset_dir = default_dataset_dir or ""
        self.resize(600, 640)
        self._init_ui()
        self._wire_conditionals()
        self._update_conditional()

    # ──────────────────────────────────────────────────────────────────
    # Construção
    # ──────────────────────────────────────────────────────────────────
    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        outer.addWidget(self._build_group_source())
        outer.addWidget(self._build_group_dataset())
        outer.addWidget(self._build_group_content())

        # Rodapé: resumo + bloqueio + botões
        outer.addWidget(self._hline())
        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet("color: #555; font-size: 11px;")
        self.lbl_summary.setWordWrap(True)
        outer.addWidget(self.lbl_summary)
        self.lbl_blocker = QLabel("")
        self.lbl_blocker.setStyleSheet("color: #b00020; font-size: 11px;")
        self.lbl_blocker.setWordWrap(True)
        outer.addWidget(self.lbl_blocker)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Criar dataset")
        outer.addWidget(self.buttons)

    # ── Grupo 1: Pasta de imagens ──
    def _build_group_source(self) -> QGroupBox:
        grp = QGroupBox("1 · Pasta de imagens", self)
        v = QVBoxLayout(grp)
        v.setSpacing(7)

        row = QHBoxLayout()
        row.addWidget(QLabel("Caminho:", grp))
        self.txt_path = QLineEdit(grp)
        self.btn_browse = QPushButton("…", grp); self.btn_browse.setFixedWidth(40)
        row.addWidget(self.txt_path, 1); row.addWidget(self.btn_browse)
        v.addLayout(row)

        self.rb_input_original = QRadioButton("Fotos originais (detectar e alinhar)", grp)
        self.rb_input_aligned = QRadioButton("Faces já alinhadas 112×112 (pular detecção)", grp)
        self.rb_input_original.setChecked(True)
        self.grp_input = QButtonGroup(self)
        self.grp_input.addButton(self.rb_input_original, 0)
        self.grp_input.addButton(self.rb_input_aligned, 1)
        v.addWidget(self.rb_input_original)
        v.addWidget(self.rb_input_aligned)
        return grp

    # ── Grupo 2: Características do dataset ──
    def _build_group_dataset(self) -> QGroupBox:
        grp = QGroupBox("2 · Características do dataset", self)
        v = QVBoxLayout(grp)
        v.setSpacing(7)

        g = QGridLayout(); g.setHorizontalSpacing(10); g.setVerticalSpacing(7)
        g.addWidget(QLabel("Nome:", grp), 0, 0)
        self.txt_name = QLineEdit(grp); self.txt_name.setPlaceholderText("Nome do dataset")
        g.addWidget(self.txt_name, 0, 1)

        g.addWidget(QLabel("Destino:", grp), 1, 0)
        dest_w = QWidget(grp); dest_h = QHBoxLayout(dest_w); dest_h.setContentsMargins(0, 0, 0, 0)
        self.txt_dest = QLineEdit(grp); self.txt_dest.setText(self._default_dataset_dir)
        self.txt_dest.setPlaceholderText("Pasta onde o dataset (.parquet) será gravado")
        self.btn_browse_dest = QPushButton("…", grp); self.btn_browse_dest.setFixedWidth(40)
        dest_h.addWidget(self.txt_dest, 1); dest_h.addWidget(self.btn_browse_dest)
        g.addWidget(dest_w, 1, 1)

        g.addWidget(QLabel("Tipo:", grp), 2, 0)
        type_w = QWidget(grp); type_h = QHBoxLayout(type_w); type_h.setContentsMargins(0, 0, 0, 0)
        self.rb_self_contained = QRadioButton("Autocontido (copia imagens)", grp)
        self.rb_absolute = QRadioButton("Caminho absoluto (referencia originais)", grp)
        self.rb_absolute.setChecked(True)
        self.rb_self_contained.setToolTip(
            "Copia as imagens para a pasta do dataset (portátil). Caminho absoluto "
            "referencia as imagens originais no lugar (mais leve, não portátil).")
        self.grp_storage = QButtonGroup(self)
        self.grp_storage.addButton(self.rb_absolute, 0)
        self.grp_storage.addButton(self.rb_self_contained, 1)
        type_h.addWidget(self.rb_self_contained); type_h.addWidget(self.rb_absolute); type_h.addStretch()
        g.addWidget(type_w, 2, 1)

        # Tipo de imagem — só no autocontido.
        self._lbl_imgtype = QLabel("Imagem:", grp)
        g.addWidget(self._lbl_imgtype, 3, 0)
        self.row_imgtype = QWidget(grp)
        img_h = QHBoxLayout(self.row_imgtype); img_h.setContentsMargins(0, 0, 0, 0)
        self.rb_img_original = QRadioButton("Face original", grp)
        self.rb_img_aligned = QRadioButton("Face alinhada", grp)
        self.rb_img_aligned.setChecked(True)
        self.grp_imgtype = QButtonGroup(self)
        self.grp_imgtype.addButton(self.rb_img_original, 0)
        self.grp_imgtype.addButton(self.rb_img_aligned, 1)
        img_h.addWidget(self.rb_img_original); img_h.addWidget(self.rb_img_aligned)
        self._lbl_imgtype_hint = QLabel("(o que copiar)", grp)
        self._lbl_imgtype_hint.setStyleSheet("color: #888; font-size: 11px;")
        img_h.addWidget(self._lbl_imgtype_hint); img_h.addStretch()
        g.addWidget(self.row_imgtype, 3, 1)
        v.addLayout(g)

        # Metadados gerais (origem/descrição) — colapsado.
        self._meta_btn = QPushButton("▸ Metadados gerais (origem, descrição)", grp)
        self._meta_btn.setCheckable(True)
        self._meta_btn.setStyleSheet("text-align: left; border: none; color: #555;")
        v.addWidget(self._meta_btn)
        self._meta_panel = QWidget(grp)
        mg = QGridLayout(self._meta_panel); mg.setContentsMargins(14, 2, 2, 2)
        mg.addWidget(QLabel("Origem:", self._meta_panel), 0, 0)
        self.txt_origin = QLineEdit("Local", self._meta_panel)
        mg.addWidget(self.txt_origin, 0, 1)
        mg.addWidget(QLabel("Descrição:", self._meta_panel), 1, 0)
        self.txt_desc = QPlainTextEdit(self._meta_panel); self.txt_desc.setMaximumHeight(48)
        mg.addWidget(self.txt_desc, 1, 1)
        self._meta_panel.setVisible(False)
        v.addWidget(self._meta_panel)
        self._meta_btn.toggled.connect(lambda c: (
            self._meta_panel.setVisible(c),
            self._meta_btn.setText(("▾" if c else "▸") + " Metadados gerais (origem, descrição)"),
        ))
        return grp

    # ── Grupo 3: Conteúdo do dataset ──
    def _build_group_content(self) -> QGroupBox:
        grp = QGroupBox("3 · Conteúdo do dataset", self)
        v = QVBoxLayout(grp)
        v.setSpacing(7)

        # Embeddings (modelos inline) + Avançado.
        emb_row = QHBoxLayout()
        emb_row.addWidget(QLabel("Embeddings:", grp))
        self._ff_model_checks = {}
        for name in _supported_ff_models():
            member = SepaelModel.from_ff_name(name)
            chk = QCheckBox(member.label_pt if member is not None else name, grp)
            self._ff_model_checks[name] = chk
            emb_row.addWidget(chk)
        default_name = SepaelModel.default().ff_name
        (self._ff_model_checks.get(default_name)
         or next(iter(self._ff_model_checks.values()), None)).setChecked(True)
        emb_row.addStretch()
        self._ff_adv_btn = QPushButton("▸ Avançado", grp)
        self._ff_adv_btn.setCheckable(True)
        self._ff_adv_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        emb_row.addWidget(self._ff_adv_btn)
        v.addLayout(emb_row)
        self._ff_adv_panel = self._build_ff_advanced_panel()
        self._ff_adv_panel.setVisible(False)
        v.addWidget(self._ff_adv_panel)
        self._ff_adv_btn.toggled.connect(lambda c: (
            self._ff_adv_panel.setVisible(c),
            self._ff_adv_btn.setText("▾ Avançado" if c else "▸ Avançado"),
        ))

        # Qualidade.
        qual_row = QHBoxLayout()
        qual_row.addWidget(QLabel("Qualidade:", grp))
        # Desmarcados por padrão — são os estágios lentos (opt-in).
        self.chk_ofiq = QCheckBox("OFIQ (qualidade ISO)", grp); self.chk_ofiq.setChecked(False)
        self.chk_serfiq = QCheckBox("SER-FIQ (robustez)", grp); self.chk_serfiq.setChecked(False)
        qual_row.addWidget(self.chk_ofiq)
        qual_row.addWidget(self._slow_badge())
        qual_row.addSpacing(8)
        qual_row.addWidget(self.chk_serfiq)
        self._serfiq_adv_btn = QPushButton("▸ Avançado", grp)
        self._serfiq_adv_btn.setCheckable(True)
        self._serfiq_adv_btn.setStyleSheet("font-size: 10px; padding: 2px 8px;")
        qual_row.addWidget(self._serfiq_adv_btn)
        qual_row.addWidget(self._slow_badge())
        qual_row.addStretch()
        v.addLayout(qual_row)
        self._serfiq_adv_panel = self._build_serfiq_advanced_panel()
        self._serfiq_adv_panel.setVisible(False)
        v.addWidget(self._serfiq_adv_panel)
        self._serfiq_adv_btn.toggled.connect(lambda c: (
            self._serfiq_adv_panel.setVisible(c),
            self._serfiq_adv_btn.setText("▾ Avançado" if c else "▸ Avançado"),
        ))

        v.addWidget(self._hline())

        # Identidade / colunas.
        self.chk_use_template = QCheckBox("Usar template de colunas", grp)
        self.chk_use_template.setToolTip(
            "O template extrai identidade E outras colunas (gênero, etnia, origem, "
            "colunas custom) da estrutura de pastas/arquivos.")
        v.addWidget(self.chk_use_template)

        # Bloco template (visível quando checkbox marcado).
        self.row_template = QWidget(grp)
        th = QVBoxLayout(self.row_template); th.setContentsMargins(18, 0, 0, 0); th.setSpacing(3)
        tr = QHBoxLayout()
        tr.addWidget(QLabel("Template:", self.row_template))
        self.cmb_template = QComboBox(self.row_template)
        self.btn_new_template = QPushButton("Criar template…", self.row_template)
        tr.addWidget(self.cmb_template, 1); tr.addWidget(self.btn_new_template)
        th.addLayout(tr)
        th.addWidget(QLabel(
            "<i>Extrai identidade e demais colunas das pastas/arquivos.</i>", self.row_template))
        self.row_template.setVisible(False)
        v.addWidget(self.row_template)

        # Bloco identidade (visível quando checkbox desmarcado).
        self.row_identity = QWidget(grp)
        idv = QVBoxLayout(self.row_identity); idv.setContentsMargins(0, 0, 0, 0); idv.setSpacing(5)
        idv.addWidget(QLabel("Identidade (como obter):", self.row_identity))
        self.rb_id_subdir = QRadioButton(
            "Do subdiretório (uma identidade por subpasta)", self.row_identity)
        self.rb_id_single = QRadioButton("Identidade única (digitar)", self.row_identity)
        self.rb_id_subdir.setChecked(True)
        self.grp_identity = QButtonGroup(self)
        self.grp_identity.addButton(self.rb_id_subdir, 0)
        self.grp_identity.addButton(self.rb_id_single, 1)
        idv.addWidget(self.rb_id_subdir)
        idv.addWidget(self.rb_id_single)
        # Campos da identidade única.
        self.row_id_single = QWidget(self.row_identity)
        sg = QGridLayout(self.row_id_single); sg.setContentsMargins(20, 0, 0, 0)
        sg.addWidget(QLabel("Identidade:", self.row_id_single), 0, 0)
        self.txt_identity_flat = QLineEdit(self.row_id_single)
        sg.addWidget(self.txt_identity_flat, 0, 1)
        sg.addWidget(QLabel("Gênero:", self.row_id_single), 1, 0)
        self.cmb_gender = QComboBox(self.row_id_single); self.cmb_gender.addItems(["", "Male", "Female"])
        sg.addWidget(self.cmb_gender, 1, 1)
        sg.addWidget(QLabel("Etnia:", self.row_id_single), 2, 0)
        self.txt_ethnicity = QLineEdit(self.row_id_single)
        sg.addWidget(self.txt_ethnicity, 2, 1)
        self.row_id_single.setVisible(False)
        idv.addWidget(self.row_id_single)
        v.addWidget(self.row_identity)
        return grp

    def _build_ff_advanced_panel(self) -> QWidget:
        panel = QWidget(self); panel.setObjectName("advPanel"); panel.setStyleSheet(_ADV_STYLE)
        v = QVBoxLayout(panel); v.setContentsMargins(20, 4, 4, 4); v.setSpacing(6)
        det_row = QHBoxLayout()
        det_row.addWidget(QLabel("Det size:", panel))
        self._ff_det_size_spin = QSpinBox(panel)
        self._ff_det_size_spin.setRange(160, 1280); self._ff_det_size_spin.setSingleStep(32)
        self._ff_det_size_spin.setValue(320)
        det_row.addWidget(self._ff_det_size_spin); det_row.addStretch()
        v.addLayout(det_row)
        self._ff_extended_chk = QCheckBox("Modo completo: FIQA + gender + age + yaw/pitch/roll", panel)
        self._ff_extended_chk.setChecked(True)
        v.addWidget(self._ff_extended_chk)
        return panel

    def _build_serfiq_advanced_panel(self) -> QWidget:
        panel = QWidget(self); panel.setObjectName("advPanel"); panel.setStyleSheet(_ADV_STYLE)
        v = QVBoxLayout(panel); v.setContentsMargins(20, 4, 4, 4); v.setSpacing(6)
        t_row = QHBoxLayout(); t_row.addWidget(QLabel("T (iterações):", panel))
        self._serfiq_t_spin = QSpinBox(panel); self._serfiq_t_spin.setRange(1, 100); self._serfiq_t_spin.setValue(10)
        t_row.addWidget(self._serfiq_t_spin); t_row.addStretch(); v.addLayout(t_row)
        a_row = QHBoxLayout(); a_row.addWidget(QLabel("Alpha:", panel))
        self._serfiq_alpha_spin = QDoubleSpinBox(panel)
        self._serfiq_alpha_spin.setRange(0.1, 10.0); self._serfiq_alpha_spin.setSingleStep(0.1)
        self._serfiq_alpha_spin.setValue(1.5)
        a_row.addWidget(self._serfiq_alpha_spin); a_row.addStretch(); v.addLayout(a_row)
        return panel

    def _slow_badge(self) -> QLabel:
        lbl = QLabel("⏱ lento", self)
        lbl.setStyleSheet(
            "background: #fff3cd; color: #856404; font-size: 10px; "
            "padding: 1px 6px; border-radius: 4px;")
        return lbl

    def _hline(self) -> QFrame:
        ln = QFrame(self); ln.setFrameShape(QFrame.Shape.HLine)
        ln.setStyleSheet("color: #e0e0e0; max-height: 1px;")
        return ln

    # ──────────────────────────────────────────────────────────────────
    # Máquina de estados condicional
    # ──────────────────────────────────────────────────────────────────
    def _wire_conditionals(self):
        # Visibilidade/cinza + sinal `changed` (validação reativa no controller).
        for sig in (
            self.txt_path.textChanged, self.txt_name.textChanged,
            self.txt_dest.textChanged, self.txt_identity_flat.textChanged,
            self.cmb_template.currentIndexChanged,
        ):
            sig.connect(self._on_any_change)
        for grp in (self.grp_input, self.grp_storage, self.grp_imgtype, self.grp_identity):
            grp.buttonToggled.connect(self._on_any_change)
        for chk in (self.chk_use_template, self.chk_ofiq, self.chk_serfiq,
                    *self._ff_model_checks.values()):
            chk.toggled.connect(self._on_any_change)

    def _on_any_change(self, *args):
        self._update_conditional()
        self.changed.emit()

    def _update_conditional(self):
        autocontido = self.rb_self_contained.isChecked()
        input_aligned = self.rb_input_aligned.isChecked()

        # Tipo de imagem só no autocontido.
        self._lbl_imgtype.setVisible(autocontido)
        self.row_imgtype.setVisible(autocontido)
        # "Face original" não existe quando a entrada já é alinhada.
        self.rb_img_original.setEnabled(autocontido and not input_aligned)
        if input_aligned and self.rb_img_original.isChecked():
            self.rb_img_aligned.setChecked(True)

        # Template marcado → bloco template; desmarcado → identidade.
        use_tpl = self.chk_use_template.isChecked()
        self.row_template.setVisible(use_tpl)
        self.row_identity.setVisible(not use_tpl)
        # Campos da identidade única só quando "digitar".
        self.row_id_single.setVisible(not use_tpl and self.rb_id_single.isChecked())

    # ──────────────────────────────────────────────────────────────────
    # Disponibilidade de serviços (FF/OFIQ) — reusada do controller
    # ──────────────────────────────────────────────────────────────────
    def set_stage_availability(self, ff_available: bool, ofiq_available: bool) -> None:
        ff_tip = ("ForensicFace indisponível: nenhum modelo em ~/.forensicface/models/ "
                  "ou o serviço falhou ao carregar.")
        ofiq_tip = "OFIQ indisponível: executável ou pasta de dados não encontrados."
        if not ff_available:
            for chk in self._ff_model_checks.values():
                chk.setChecked(False); chk.setEnabled(False); chk.setToolTip(ff_tip)
            self.chk_serfiq.setChecked(False); self.chk_serfiq.setEnabled(False)
            self.chk_serfiq.setToolTip("SER-FIQ depende do ForensicFace. " + ff_tip)
            # Sem FF não há como detectar+alinhar originais.
            self.rb_img_aligned.setEnabled(False)
            self.rb_input_original.setToolTip(ff_tip)
        if not ofiq_available:
            self.chk_ofiq.setChecked(False); self.chk_ofiq.setEnabled(False)
            self.chk_ofiq.setToolTip(ofiq_tip)
        self._update_conditional()

    # ──────────────────────────────────────────────────────────────────
    # Getters
    # ──────────────────────────────────────────────────────────────────
    def get_ff_models(self) -> list:
        return [n for n, c in self._ff_model_checks.items() if c.isChecked()]

    def any_model_checked(self) -> bool:
        return any(c.isChecked() for c in self._ff_model_checks.values())

    def get_ff_det_size(self) -> int:
        return int(self._ff_det_size_spin.value())

    def get_ff_extended(self) -> bool:
        return bool(self._ff_extended_chk.isChecked())

    def get_serfiq_t(self) -> int:
        return int(self._serfiq_t_spin.value())

    def get_serfiq_alpha(self) -> float:
        return float(self._serfiq_alpha_spin.value())

    def get_dataset_dir(self) -> str:
        return self.txt_dest.text().strip()

    def get_self_contained(self) -> bool:
        return self.rb_self_contained.isChecked()

    def get_image_type(self) -> str:
        """'aligned' ou 'original' — relevante só no autocontido."""
        return "original" if (self.rb_self_contained.isChecked()
                              and self.rb_img_original.isChecked()) else "aligned"

    def is_input_aligned(self) -> bool:
        return self.rb_input_aligned.isChecked()

    def use_template(self) -> bool:
        return self.chk_use_template.isChecked()

    def is_single_identity(self) -> bool:
        return (not self.chk_use_template.isChecked()) and self.rb_id_single.isChecked()
