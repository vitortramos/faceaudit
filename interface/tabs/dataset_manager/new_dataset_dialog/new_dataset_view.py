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

# Usado nas duas telas que ligam a extração (criação e enriquecimento).
TIP_EXIF = (
    'Lê os metadados embutidos no arquivo (EXIF/XMP/IPTC): data da captura, câmera, exposição, GPS. Exige a foto ORIGINAL — recorte e alinhamento gravam pixel novo, sem metadado de captura. As colunas nascem ocultas; o conteúdo completo aparece na aba Metadados.'
)
from interface.theme import apply_themed_stylesheet


def _supported_ff_models() -> tuple:
    available = SepaelModel.available_on_disk()
    if available:
        return tuple(m.ff_name for m in available)
    return (SepaelModel.default().ff_name,)


class NewDatasetView(QDialog):
    """Disposição + visibilidade condicional do diálogo de novo dataset."""

    # Emitido quando qualquer entrada relevante muda — o controller liga em
    # _revalidate() para validação reativa (OK + rodapé).
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None, default_dataset_dir: str = "",
                 default_use_gpu: bool = False, default_batch_size: int = 32):
        super().__init__(parent)
        self.setWindowTitle("Novo dataset")
        self._default_dataset_dir = default_dataset_dir or ""
        self._default_use_gpu = bool(default_use_gpu)
        self._default_batch_size = int(default_batch_size)
        self.resize(600, 660)
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
        outer.addWidget(self._build_group_ff_settings())
        outer.addWidget(self._build_group_content())

        # Rodapé: resumo + bloqueio + botões
        outer.addWidget(self._hline())
        self.lbl_summary = QLabel("")
        apply_themed_stylesheet(
            self.lbl_summary, lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;"
        )
        self.lbl_summary.setWordWrap(True)
        outer.addWidget(self.lbl_summary)
        self.lbl_blocker = QLabel("")
        apply_themed_stylesheet(
            self.lbl_blocker, lambda t: "color: " + t("DANGER") + "; font-size: 11px;"
        )
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
        self.txt_dest.setToolTip(
            "Pasta onde o <b>dataset final</b> (.parquet) será gravado. Default: a "
            "pasta global de datasets, para reuso entre projetos. Gravar fora dela "
            "significa que o dataset <b>não aparece na biblioteca global</b>.")
        self.btn_browse_dest = QPushButton("…", grp); self.btn_browse_dest.setFixedWidth(40)
        dest_h.addWidget(self.txt_dest, 1); dest_h.addWidget(self.btn_browse_dest)
        g.addWidget(dest_w, 1, 1)

        g.addWidget(QLabel("Tipo:", grp), 2, 0)
        type_w = QWidget(grp); type_h = QHBoxLayout(type_w); type_h.setContentsMargins(0, 0, 0, 0)
        self.rb_self_contained = QRadioButton("Autocontido (copia imagens)", grp)
        self.rb_absolute = QRadioButton("Caminho absoluto (referencia originais)", grp)
        self.rb_absolute.setChecked(True)
        self.rb_self_contained.setToolTip(
            "<b>Autocontido</b>: o dataset ganha pasta própria em "
            "<i>&lt;Destino&gt;/&lt;nome&gt;/</i>, com o arquivo e as imagens "
            "(<i>images/</i>) dentro. A pasta inteira pode ser movida, "
            "compactada ou levada para outra máquina sem quebrar nada. Custa "
            "espaço em disco (duplica as imagens).")
        self.rb_absolute.setToolTip(
            "<b>Caminho absoluto</b>: referencia as imagens onde elas estão, sem "
            "copiar. Mais leve, mas <b>quebra</b> se as imagens mudarem de lugar.")
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
        self.rb_img_original = QRadioButton("Foto original", grp)
        self.rb_img_crop = QRadioButton("Face recortada", grp)
        self.rb_img_aligned = QRadioButton("Face alinhada", grp)
        self.rb_img_aligned.setChecked(True)
        self.rb_img_original.setToolTip(
            "<b>Foto original</b>: copia a foto como veio, com a cena "
            "inteira. Preserva o registro íntegro da captura.")
        self.rb_img_crop.setToolTip(
            "<b>Face recortada</b>: recorte da face com margem, <b>sem "
            "rotação</b> — preserva a geometria e as proporções reais do "
            "rosto. É a forma útil para ilustrar laudo e para comparação "
            "visual. O dataset <b>não</b> é marcado como alinhado: um "
            "enriquecimento posterior continua detectando e alinhando "
            "normalmente a partir do recorte.")
        self.rb_img_aligned.setToolTip(
            "<b>Face alinhada</b>: face 112×112, que é o que o modelo "
            "enxerga. Deforma o rosto (rotação, escala e reamostragem), "
            "então não serve para avaliar semelhança a olho nu.")
        self.grp_imgtype = QButtonGroup(self)
        self.grp_imgtype.addButton(self.rb_img_original, 0)
        self.grp_imgtype.addButton(self.rb_img_aligned, 1)
        self.grp_imgtype.addButton(self.rb_img_crop, 2)
        img_h.addWidget(self.rb_img_original)
        img_h.addWidget(self.rb_img_crop)
        img_h.addWidget(self.rb_img_aligned)

        # Margem — só faz sentido no recorte.
        self.lbl_crop_margin = QLabel("Margem:", grp)
        self.spn_crop_margin = QDoubleSpinBox(grp)
        self.spn_crop_margin.setRange(1.0, 5.0)
        self.spn_crop_margin.setSingleStep(0.5)
        self.spn_crop_margin.setValue(2.0)
        self.spn_crop_margin.setToolTip(
            "Fator de expansão da caixa da face: 1,0 = caixa exata, 2,0 = o "
            "dobro em cada dimensão. Mesma unidade da extração de faces de "
            "vídeo.")
        img_h.addWidget(self.lbl_crop_margin)
        img_h.addWidget(self.spn_crop_margin)
        self.rb_img_crop.toggled.connect(self._atualizar_margem_recorte)
        self._lbl_imgtype_hint = QLabel("(o que copiar)", grp)
        apply_themed_stylesheet(
            self._lbl_imgtype_hint, lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;"
        )
        img_h.addWidget(self._lbl_imgtype_hint); img_h.addStretch()
        g.addWidget(self.row_imgtype, 3, 1)
        self._atualizar_margem_recorte()
        v.addLayout(g)

        # Descrição do dataset (metadado do dataset como um todo, NÃO é coluna —
        # a Origem, que é coluna por-face, foi para "Conteúdo do dataset") — colapsado.
        self._meta_btn = QPushButton("▸ Descrição (opcional)", grp)
        self._meta_btn.setCheckable(True)
        apply_themed_stylesheet(
            self._meta_btn,
            lambda t: "text-align: left; border: none; color: " + t("TEXT_MUTED") + ";",
        )
        v.addWidget(self._meta_btn)
        self._meta_panel = QWidget(grp)
        mg = QGridLayout(self._meta_panel); mg.setContentsMargins(14, 2, 2, 2)
        mg.addWidget(QLabel("Descrição:", self._meta_panel), 0, 0)
        self.txt_desc = QPlainTextEdit(self._meta_panel); self.txt_desc.setMaximumHeight(48)
        mg.addWidget(self.txt_desc, 0, 1)
        self._meta_panel.setVisible(False)
        v.addWidget(self._meta_panel)
        self._meta_btn.toggled.connect(lambda c: (
            self._meta_panel.setVisible(c),
            self._meta_btn.setText(("▾" if c else "▸") + " Descrição (opcional)"),
        ))
        return grp

    # ── Grupo 3: Conteúdo do dataset ──
    def _build_group_content(self) -> QGroupBox:
        grp = QGroupBox("4 · Conteúdo do dataset", self)
        v = QVBoxLayout(grp)
        v.setSpacing(7)

        # Conteúdo numa grade de 2 colunas: rótulo da categoria na coluna 0,
        # checkboxes na coluna 1. A grade alinha os checkboxes de TODAS as linhas
        # na mesma coluna (largura da coluna 0 = a do maior rótulo).
        cg = QGridLayout()
        cg.setHorizontalSpacing(8)
        cg.setVerticalSpacing(7)
        cg.setColumnStretch(1, 1)

        # Embeddings (modelos inline).
        cg.addWidget(QLabel("Embeddings:", grp), 0, 0)
        emb_w = QWidget(grp)
        emb_h = QHBoxLayout(emb_w); emb_h.setContentsMargins(0, 0, 0, 0)
        self._ff_model_checks = {}
        for name in _supported_ff_models():
            member = SepaelModel.from_ff_name(name)
            chk = QCheckBox(member.label_pt if member is not None else name, emb_w)
            self._ff_model_checks[name] = chk
            emb_h.addWidget(chk)
        default_name = SepaelModel.default().ff_name
        (self._ff_model_checks.get(default_name)
         or next(iter(self._ff_model_checks.values()), None)).setChecked(True)
        emb_h.addStretch()
        cg.addWidget(emb_w, 0, 1)

        # Qualidade: FIQA (roda no crop; sobrevive a faces alinhadas) + OFIQ
        # (ISO, exige a foto ORIGINAL → some em entrada alinhada).
        cg.addWidget(QLabel("Qualidade:", grp), 1, 0)
        qual_w = QWidget(grp)
        qual_h = QHBoxLayout(qual_w); qual_h.setContentsMargins(0, 0, 0, 0)
        self.chk_fiqa = QCheckBox("FIQA", qual_w); self.chk_fiqa.setChecked(True)
        self.chk_fiqa.setToolTip(
            "Qualidade da face (CR-FIQA) — computada junto do embedding, "
            "disponível também para faces já alinhadas.")
        qual_h.addWidget(self.chk_fiqa)
        qual_h.addSpacing(12)
        self._ofiq_w = QWidget(qual_w)
        ofiq_h = QHBoxLayout(self._ofiq_w); ofiq_h.setContentsMargins(0, 0, 0, 0)
        self.chk_ofiq = QCheckBox("OFIQ (qualidade ISO)", self._ofiq_w); self.chk_ofiq.setChecked(False)
        self.chk_ofiq.setToolTip(
            "Qualidade ISO/IEC 29794-5 — exige a foto ORIGINAL (faz a própria "
            "detecção). Indisponível para faces já alinhadas.")
        ofiq_h.addWidget(self.chk_ofiq); ofiq_h.addWidget(self._slow_badge())
        qual_h.addWidget(self._ofiq_w)
        qual_h.addStretch()
        cg.addWidget(qual_w, 1, 1)

        # Metadados embutidos — leitura de arquivo, não inferência: não
        # entra no grupo de qualidade nem carrega selo de lentidão.
        self._exif_lbl = QLabel("Metadados:", grp)
        cg.addWidget(self._exif_lbl, 5, 0)
        self._exif_w = QWidget(grp)
        exif_h = QHBoxLayout(self._exif_w); exif_h.setContentsMargins(0, 0, 0, 0)
        self.chk_exif = QCheckBox("Extrair EXIF/XMP da imagem", self._exif_w)
        self.chk_exif.setChecked(False)
        self.chk_exif.setToolTip(TIP_EXIF)
        exif_h.addWidget(self.chk_exif)
        exif_h.addStretch()
        cg.addWidget(self._exif_w, 5, 1)

        # Pose (yaw/pitch/roll) — estimada na imagem ORIGINAL durante o
        # alinhamento; NÃO existe para faces já alinhadas.
        self._pose_lbl = QLabel("Pose:", grp)
        cg.addWidget(self._pose_lbl, 2, 0)
        self._pose_row_w = QWidget(grp)
        pose_h = QHBoxLayout(self._pose_row_w); pose_h.setContentsMargins(0, 0, 0, 0)
        self.chk_yaw = QCheckBox("Yaw", self._pose_row_w); self.chk_yaw.setChecked(True)
        self.chk_pitch = QCheckBox("Pitch", self._pose_row_w); self.chk_pitch.setChecked(True)
        self.chk_roll = QCheckBox("Roll", self._pose_row_w); self.chk_roll.setChecked(True)
        for c in (self.chk_yaw, self.chk_pitch, self.chk_roll):
            pose_h.addWidget(c)
        pose_h.addStretch()
        cg.addWidget(self._pose_row_w, 2, 1)

        # Outros atributos — dependem da detecção na imagem ORIGINAL; somem
        # para faces já alinhadas.
        self._outros_lbl = QLabel("Outros:", grp)
        cg.addWidget(self._outros_lbl, 3, 0)
        self._outros_row_w = QWidget(grp)
        outros_h = QHBoxLayout(self._outros_row_w); outros_h.setContentsMargins(0, 0, 0, 0)
        self.chk_det_conf = QCheckBox("Det Conf", self._outros_row_w); self.chk_det_conf.setChecked(True)
        self.chk_ipd = QCheckBox("IPD", self._outros_row_w); self.chk_ipd.setChecked(True)
        self.chk_gender = QCheckBox("Gênero", self._outros_row_w); self.chk_gender.setChecked(True)
        self.chk_age = QCheckBox("Idade", self._outros_row_w); self.chk_age.setChecked(True)
        for c in (self.chk_det_conf, self.chk_ipd, self.chk_gender, self.chk_age):
            outros_h.addWidget(c)
        outros_h.addStretch()
        cg.addWidget(self._outros_row_w, 3, 1)

        v.addLayout(cg)

        v.addWidget(self._hline())

        # Origem — coluna de PROVENIÊNCIA por face (constante nesta ingestão);
        # é dado do dataset (coluna 'origin'), não metadado geral.
        origin_row = QHBoxLayout()
        origin_row.addWidget(QLabel("Origem:", grp))
        self.txt_origin = QLineEdit("Local", grp)
        self.txt_origin.setToolTip(
            "Proveniência de cada face (vira a coluna 'origin' do dataset) — "
            "ex.: nome do caso, base ou câmera. Constante nesta ingestão.")
        origin_row.addWidget(self.txt_origin, 1)
        v.addLayout(origin_row)

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

    def _build_group_ff_settings(self) -> QGroupBox:
        """Grupo 3 — knobs do engine ForensicFace. GPU e Batch size vêm das
        Configurações do Sistema (default) e são editáveis como override DESTA
        extração (não gravam no global). Det size é por-extração e só afeta
        fotos originais (no-op em faces já alinhadas)."""
        grp = QGroupBox("3 · Configurações ForensicFace", self)
        v = QVBoxLayout(grp); v.setSpacing(6)
        row = QHBoxLayout()
        self.chk_use_gpu = QCheckBox("Usar GPU", grp)
        self.chk_use_gpu.setChecked(self._default_use_gpu)
        self.chk_use_gpu.setToolTip(
            "Default das Configurações do Sistema; alterar aqui vale só para "
            "esta extração (não grava no global).")
        row.addWidget(self.chk_use_gpu)
        row.addSpacing(16)
        row.addWidget(QLabel("Batch size:", grp))
        self.spn_batch_size = QSpinBox(grp)
        self.spn_batch_size.setRange(1, 512); self.spn_batch_size.setValue(self._default_batch_size)
        self.spn_batch_size.setToolTip(
            "Faces por chamada do modelo. Maior = mais paralelismo/GPU, mais "
            "memória. Default do Sistema; override por extração.")
        row.addWidget(self.spn_batch_size)
        row.addSpacing(16)
        self._lbl_det_size = QLabel("Det size:", grp)
        row.addWidget(self._lbl_det_size)
        self._ff_det_size_spin = QSpinBox(grp)
        self._ff_det_size_spin.setRange(160, 1280); self._ff_det_size_spin.setSingleStep(32)
        self._ff_det_size_spin.setValue(320)
        self._ff_det_size_spin.setToolTip(
            "Resolução do detector de faces. Só afeta fotos ORIGINAIS — "
            "ignorado em faces já alinhadas.")
        row.addWidget(self._ff_det_size_spin)
        row.addStretch()
        v.addLayout(row)
        self._det_size_hint = QLabel(
            "Det size não se aplica a faces já alinhadas (detecção é pulada).", grp)
        apply_themed_stylesheet(
            self._det_size_hint, lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 11px;"
        )
        self._det_size_hint.setVisible(False)
        v.addWidget(self._det_size_hint)
        return grp

    def _slow_badge(self) -> QLabel:
        lbl = QLabel("⏱ lento", self)
        apply_themed_stylesheet(
            lbl,
            lambda t: (
                "background: " + t("WARNING_BG") + "; color: " + t("WARNING") + "; font-size: 10px; "
                "padding: 1px 6px; border-radius: 4px;"
            ),
        )
        return lbl

    def _hline(self) -> QFrame:
        ln = QFrame(self); ln.setFrameShape(QFrame.Shape.HLine)
        apply_themed_stylesheet(ln, lambda t: "color: " + t("BORDER") + "; max-height: 1px;")
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
        for chk in (self.chk_use_template, self.chk_fiqa, self.chk_ofiq,
                    self.chk_yaw, self.chk_pitch, self.chk_roll,
                    self.chk_det_conf, self.chk_ipd, self.chk_gender, self.chk_age,
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

        # "Some se alinhada": atributos que só existem na foto ORIGINAL
        # desaparecem quando a entrada já é de faces alinhadas 112×112 (rótulo
        # da categoria + os checkboxes da coluna 1 escondidos juntos).
        self._pose_lbl.setVisible(not input_aligned)
        self._pose_row_w.setVisible(not input_aligned)
        self._outros_lbl.setVisible(not input_aligned)
        self._outros_row_w.setVisible(not input_aligned)
        self._ofiq_w.setVisible(not input_aligned)     # OFIQ exige a original
        # Det size é no-op em alinhada — esconde o campo, mostra a dica.
        self._lbl_det_size.setVisible(not input_aligned)
        self._ff_det_size_spin.setVisible(not input_aligned)
        self._det_size_hint.setVisible(input_aligned)

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
            # Todos os atributos FF dependem do ForensicFace.
            for chk in (self.chk_fiqa, self.chk_yaw, self.chk_pitch, self.chk_roll,
                        self.chk_det_conf, self.chk_ipd, self.chk_gender, self.chk_age):
                chk.setChecked(False); chk.setEnabled(False); chk.setToolTip(ff_tip)
            # Sem FF não há como detectar+alinhar originais.
            self.rb_img_aligned.setEnabled(False)
            self.rb_input_original.setToolTip(ff_tip)
        if not ofiq_available:
            self.chk_exif.setChecked(False); self.chk_exif.setEnabled(False)
            self.chk_exif.setToolTip(
                "Faces já alinhadas não carregam metadado de captura — o "
                "recorte é pixel novo, gravado sem EXIF."
            )
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

    def get_use_gpu(self) -> bool:
        return bool(self.chk_use_gpu.isChecked())

    def get_batch_size(self) -> int:
        return int(self.spn_batch_size.value())

    def get_content_flags(self) -> dict:
        """Toggles granulares de atributo → coluna. Chaves batem com
        `DatasetIngestionConfig.ff_kept_attributes`."""
        return {
            "fiqa": self.chk_fiqa.isChecked(),
            "yaw": self.chk_yaw.isChecked(),
            "pitch": self.chk_pitch.isChecked(),
            "roll": self.chk_roll.isChecked(),
            "gender": self.chk_gender.isChecked(),
            "age": self.chk_age.isChecked(),
            "det_score": self.chk_det_conf.isChecked(),
            "ipd": self.chk_ipd.isChecked(),
        }

    def get_dataset_dir(self) -> str:
        return self.txt_dest.text().strip()

    def get_self_contained(self) -> bool:
        return self.rb_self_contained.isChecked()

    def _atualizar_margem_recorte(self) -> None:
        """A margem só existe no recorte — o alinhamento é canônico."""
        recorte = self.rb_img_crop.isChecked()
        self.lbl_crop_margin.setVisible(recorte)
        self.spn_crop_margin.setVisible(recorte)

    def get_image_type(self) -> str:
        """Forma da imagem guardada — relevante só no autocontido.

        Devolve o valor de `DatasetImageKind`: 'original', 'crop' ou
        'aligned'. Fora do autocontido não há imagem copiada para
        escolher, e o dataset referencia as fotos onde elas estão.
        """
        if not self.rb_self_contained.isChecked():
            return "aligned"
        if self.rb_img_original.isChecked():
            return "original"
        if self.rb_img_crop.isChecked():
            return "crop"
        return "aligned"

    def get_extract_exif(self) -> bool:
        return self.chk_exif.isChecked() and self.chk_exif.isEnabled()

    def get_crop_margin_factor(self) -> float:
        """Fator de expansão da bbox no modo recorte."""
        return float(self.spn_crop_margin.value())

    def is_input_aligned(self) -> bool:
        return self.rb_input_aligned.isChecked()

    def use_template(self) -> bool:
        return self.chk_use_template.isChecked()

    def is_single_identity(self) -> bool:
        return (not self.chk_use_template.isChecked()) and self.rb_id_single.isChecked()
