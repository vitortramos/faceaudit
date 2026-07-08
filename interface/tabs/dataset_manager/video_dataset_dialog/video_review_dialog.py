"""Grade de revisão/cull de um run de extração de vídeo + configuração do
dataset (janela B, duas colunas).

Esquerda: grupo **filtros** (sliders de intervalo det/fiqa/yaw/pitch + botão
"Aplicar filtros" que marca/desmarca os recortes visíveis pelos intervalos) e
grupo **recortes** (combos Alvo + Vídeo que escopam a grade + miniaturas +
marcar/desmarcar + contadores). Direita: grupo **Modo** (Só recortes /
Recortes + dataset) e grupo **Dataset** (visível só em "Recortes + dataset":
Nome, pasta, tipo de dataset, tipo de imagem [face alinhada/recorte], embeddings
com o árbitro travado lido do manifesto, OFIQ/SER-FIQ) + ações. Os dois fluxos
(criar / importar run) passam por aqui — é nesta janela que o dataset é
configurado. "Concluir revisão" grava as decisões humanas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.sepael_model import SepaelModel
from faceaudit.enums.video_extraction_mode import VideoExtractionMode
from faceaudit.services.image_io import imread_unicode_safe
from faceaudit.services.video_review_service import VideoReviewService, ReviewFace
from .range_slider import _RangeSlider
from .video_dataset_setup_dialog import _bgr_to_pixmap, _TARGET_COLORS
from ...strategy_manager_v2.comparison.flow_layout import FlowLayout

_THUMB = 100


class _Thumb(QtWidgets.QFrame):
    def __init__(self, rf: ReviewFace, parent=None, *, color: Optional[str] = None):
        super().__init__(parent)
        self.image_file = rf.image_file
        self.label = rf.label
        self.det_score = rf.det_score
        self.fiqa_score = rf.fiqa_score
        self.yaw = rf.yaw
        self.pitch = rf.pitch
        self.source_video_filename = rf.source_video_filename
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFixedWidth(_THUMB + 18)   # célula enxuta ≈ largura da foto
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 4)
        v.setSpacing(2)

        lbl_img = QtWidgets.QLabel(self)
        lbl_img.setFixedSize(_THUMB, _THUMB)
        lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_img.setStyleSheet("background:#222;")
        bgr = imread_unicode_safe(rf.crop_path) if rf.crop_path else None
        if bgr is not None:
            pix = _bgr_to_pixmap(bgr).scaled(
                _THUMB, _THUMB, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation)
            lbl_img.setPixmap(pix)
        else:
            lbl_img.setText("sem\nimagem")
        v.addWidget(lbl_img)

        # Chip do alvo (cor por alvo) — útil sobretudo em multi-alvo.
        if rf.label:
            c = color or "#888"
            chip = QtWidgets.QLabel(rf.label, self)
            chip.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            chip.setStyleSheet(
                f"color:{c}; border:1px solid {c}; border-radius:5px; "
                "padding:0 3px; font-size:9px;")
            v.addWidget(chip)

        sim = rf.similarity_to_anchor
        sim_txt = f"{sim:.2f}" if sim is not None else "—"
        self.chk = QtWidgets.QCheckBox(f"f{rf.frame_index} · {sim_txt}", self)
        self.chk.setChecked(rf.human_review_status != "rejected")
        self.chk.setStyleSheet("font-size:10px;")
        self.chk.setToolTip(
            "Esta marcação <b>é</b> a revisão humana: marcada = a face entra no "
            "conjunto final; desmarcada → vai para <i>_rejected/</i>. "
            "<i>f</i> = quadro; o número é o cosseno com a âncora (cru, não LR).")
        v.addWidget(self.chk)

        # Triagem numa linha compacta (CR-FIQA não é 0–1; pose em graus).
        det_txt = f"{rf.det_score:.2f}" if rf.det_score is not None else "—"
        fiqa_txt = f"{rf.fiqa_score:.2f}" if rf.fiqa_score is not None else "—"
        info = f"det {det_txt} · fiq {fiqa_txt}"
        if rf.yaw is not None or rf.pitch is not None:
            yaw_txt = f"{rf.yaw:+.0f}°" if rf.yaw is not None else "—"
            pit_txt = f"{rf.pitch:+.0f}°" if rf.pitch is not None else "—"
            info += f"\ny {yaw_txt} · p {pit_txt}"
        lbl_q = QtWidgets.QLabel(info, self)
        lbl_q.setToolTip(
            "det_score (confiança) · fiqa_score (qualidade CR-FIQA, não 0–1) · "
            "yaw/pitch (pose, graus)")
        apply_themed_stylesheet(
            lbl_q, lambda t: "color:" + t("TEXT_MUTED") + "; font-size:9px;")
        v.addWidget(lbl_q)

        if rf.review_reasons:
            hint = QtWidgets.QLabel("⚠ " + ", ".join(rf.review_reasons), self)
            hint.setWordWrap(True)
            apply_themed_stylesheet(
                hint, lambda t: "color:" + t("WARNING") + "; font-size:9px;")
            v.addWidget(hint)

    def is_accepted(self) -> bool:
        return self.chk.isChecked()


# (metric_key, rótulo, formato, fallback (lo, hi), tooltip)
_FILTER_SPECS = [
    ("det", "det", "f2", (0.0, 1.0),
     "Intervalo de <b>det_score</b> (confiabilidade da detecção). 'Aplicar "
     "filtros' marca os recortes dentro do intervalo. Confiabilidade ≠ identidade."),
    ("fiqa", "fiqa", "f2", (0.0, 2.0),
     "Intervalo de <b>fiqa_score</b> (qualidade — CR-FIQA, <b>não</b> 0–1; "
     "tipicamente ~1,0–1,4). Use comparativamente pela distribuição deste run."),
    ("yaw", "yaw", "deg", (-90.0, 90.0),
     "Intervalo de <b>yaw</b> (rotação horizontal, graus). Ex.: −20°..+20° ≈ frontal."),
    ("pitch", "pitch", "deg", (-90.0, 90.0),
     "Intervalo de <b>pitch</b> (rotação vertical, graus)."),
]


class VideoReviewDialog(QtWidgets.QDialog):
    def __init__(self, parent, run_dir, *, operator: str = "perito",
                 dataset_output_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Revisão das faces extraídas e configuração do dataset")
        self._run_dir = run_dir
        self._operator = operator
        self._dataset_output_dir = dataset_output_dir
        self._svc = VideoReviewService()
        self._outcome: Optional[str] = None  # completed | skipped | None(cancelado)
        self._thumbs: List[_Thumb] = []
        self._filters: Dict[str, _RangeSlider] = {}
        self._filter_val_lbls: Dict[str, QtWidgets.QLabel] = {}
        self._label_filter: Optional[str] = None
        self._video_filter: Optional[str] = None
        self._manifest = self._read_manifest()
        self._arbiter = self._manifest.get("recognition_model")
        # Carregar as miniaturas (imread por face) pode demorar — mostra o
        # cursor de ocupado durante a montagem da grade.
        # Botões de minimizar/maximizar/restaurar (QDialog só traz fechar por
        # padrão) — como a janela do aplicativo. Definir os flags ANTES de montar
        # a UI/mostrar evita recriação da janela depois.
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._build_ui()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        # Abre grande mas NÃO maximizada — o usuário maximiza/restaura se quiser.
        self._apply_initial_size()

    def outcome(self) -> Optional[str]:
        return self._outcome

    def _apply_initial_size(self):
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        if avail is not None:
            w = min(1480, int(avail.width() * 0.86))
            h = min(900, int(avail.height() * 0.86))
        else:
            w, h = 1280, 820
        self.resize(max(900, w), max(560, h))
        if avail is not None:
            fg = self.frameGeometry()
            fg.moveCenter(avail.center())
            self.move(fg.topLeft())

    def _read_manifest(self) -> dict:
        meta = Path(self._run_dir) / "_meta"
        try:
            for m in sorted(meta.glob("_run_*.json")):
                try:
                    return json.loads(m.read_text(encoding="utf-8"))
                except Exception:
                    continue
        except Exception:
            pass
        return {}

    # ─────────────────────────────── UI ───────────────────────────────
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        faces = [f for f in self._svc.list_faces(self._run_dir)
                 if f.human_review_status != "rejected"]
        outer.addWidget(QtWidgets.QLabel(
            f"<b>{len(faces)} face(s)</b> extraídas. Desmarque as que NÃO são o alvo "
            "(rejeitados são preservados em <code>_rejected/</code>).", self))

        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 1)
        body.addWidget(self._build_left_column(faces), 2)
        body.addWidget(self._build_right_column(faces), 1)

    def _build_left_column(self, faces: List[ReviewFace]) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(self._build_filters_group(faces))
        v.addWidget(self._build_recortes_group(faces), 1)
        return col

    def _build_filters_group(self, faces: List[ReviewFace]) -> QtWidgets.QWidget:
        grp = QtWidgets.QGroupBox("Filtros", self)
        outer = QtWidgets.QVBoxLayout(grp)

        # Os quatro filtros lado a lado (cada um: rótulo + valor em cima, slider
        # embaixo); o botão "Aplicar filtros" fica abaixo de todos.
        sliders_row = QtWidgets.QHBoxLayout()
        sliders_row.setSpacing(18)
        outer.addLayout(sliders_row)
        values = {
            "det": [f.det_score for f in faces],
            "fiqa": [f.fiqa_score for f in faces],
            "yaw": [f.yaw for f in faces],
            "pitch": [f.pitch for f in faces],
        }
        for key, label, fmt, fallback, tip in _FILTER_SPECS:
            lo, hi = self._metric_bounds(values[key], fallback)
            sld = _RangeSlider(grp, minimum=lo, maximum=hi)
            sld.setToolTip(tip)
            sld.setMinimumWidth(130)
            sld.setMaximumWidth(170)
            val_lbl = QtWidgets.QLabel("", grp)
            apply_themed_stylesheet(
                val_lbl, lambda t: "color:" + t("TEXT_MUTED") + "; font-size:11px;")
            sld.rangeChanged.connect(
                lambda a, b, k=key, f=fmt: self._set_filter_label(k, a, b, f))
            self._filters[key] = sld
            self._filter_val_lbls[key] = val_lbl
            cell = QtWidgets.QVBoxLayout()
            cell.setSpacing(1)
            top = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(label, grp)
            name.setStyleSheet("font-weight:500;")
            top.addWidget(name)
            top.addWidget(val_lbl)
            top.addStretch()
            cell.addLayout(top)
            cell.addWidget(sld)
            sliders_row.addLayout(cell)
            self._set_filter_label(key, lo, hi, fmt)
        sliders_row.addStretch(1)

        apply_row = QtWidgets.QHBoxLayout()
        self._btn_apply_filters = QtWidgets.QPushButton("Aplicar filtros", grp)
        self._btn_apply_filters.setToolTip(
            "Marca os recortes <b>visíveis</b> cujas métricas caem <b>dentro</b> dos "
            "quatro intervalos e desmarca os demais. Métrica ausente não exclui a "
            "face. A decisão final continua sua — confira antes de concluir.")
        self._btn_apply_filters.clicked.connect(self._apply_filters_to_selection)
        apply_row.addWidget(self._btn_apply_filters)
        apply_row.addStretch(1)
        outer.addLayout(apply_row)
        return grp

    def _build_recortes_group(self, faces: List[ReviewFace]) -> QtWidgets.QWidget:
        grp = QtWidgets.QGroupBox("Recortes", self)
        v = QtWidgets.QVBoxLayout(grp)

        # Rótulos distintos (ordem de 1ª aparição) → cor por alvo + filtro.
        labels_seen: List[str] = []
        for f in faces:
            if f.label and f.label not in labels_seen:
                labels_seen.append(f.label)
        self._label_colors = {lbl: _TARGET_COLORS[i % len(_TARGET_COLORS)]
                              for i, lbl in enumerate(labels_seen)}
        videos_seen: List[str] = []
        for f in faces:
            if f.source_video_filename and f.source_video_filename not in videos_seen:
                videos_seen.append(f.source_video_filename)

        filt_row = QtWidgets.QHBoxLayout()
        if len(labels_seen) >= 2:
            filt_row.addWidget(QtWidgets.QLabel("Alvo:", grp))
            self.cmb_filter = QtWidgets.QComboBox(grp)
            self.cmb_filter.addItem("todos", None)
            for lbl in labels_seen:
                self.cmb_filter.addItem(lbl, lbl)
            self.cmb_filter.setToolTip(
                "Escopo da grade por alvo. Marcar/desmarcar, aplicar filtros e a "
                "contagem agem sobre as faces visíveis.")
            self.cmb_filter.currentIndexChanged.connect(self._on_filter_changed)
            filt_row.addWidget(self.cmb_filter)
        if len(videos_seen) >= 2:
            filt_row.addWidget(QtWidgets.QLabel("Vídeo:", grp))
            self.cmb_video_filter = QtWidgets.QComboBox(grp)
            self.cmb_video_filter.addItem("todos os vídeos", None)
            for vid in videos_seen:
                self.cmb_video_filter.addItem(vid, vid)
            self.cmb_video_filter.setToolTip(
                "Escopo da grade por vídeo de origem (busca multivídeo).")
            self.cmb_video_filter.currentIndexChanged.connect(self._on_video_filter_changed)
            filt_row.addWidget(self.cmb_video_filter)
        filt_row.addStretch()
        self.lbl_count = QtWidgets.QLabel("", grp)
        apply_themed_stylesheet(
            self.lbl_count, lambda t: "color:" + t("TEXT_MUTED") + "; font-size:11px;")
        filt_row.addWidget(self.lbl_count)
        v.addLayout(filt_row)

        scroll = QtWidgets.QScrollArea(grp)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid_host = QtWidgets.QWidget(scroll)
        # FlowLayout canônico: empacota as miniaturas e quebra linha sozinho —
        # cabem muito mais por linha que o grid fixo de 6 colunas.
        self._flow = FlowLayout(grid_host, margin=4, h_spacing=6, v_spacing=6)
        for rf in faces:
            t = _Thumb(rf, grid_host, color=self._label_colors.get(rf.label))
            t.chk.toggled.connect(lambda _c: self._update_counts())
            self._thumbs.append(t)
            self._flow.addWidget(t)
        scroll.setWidget(grid_host)
        v.addWidget(scroll, 1)

        sel_row = QtWidgets.QHBoxLayout()
        btn_all = QtWidgets.QPushButton("Marcar todos", grp)
        btn_all.setToolTip("Marca todas as faces visíveis da grade.")
        btn_all.clicked.connect(lambda: self._set_all(True))
        btn_none = QtWidgets.QPushButton("Desmarcar todos", grp)
        btn_none.setToolTip("Desmarca todas as faces visíveis da grade.")
        btn_none.clicked.connect(lambda: self._set_all(False))
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        sel_row.addStretch()
        v.addLayout(sel_row)
        self._update_counts()
        return grp

    def _build_right_column(self, faces: List[ReviewFace]) -> QtWidgets.QWidget:
        col = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(col)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        v.addWidget(self._build_mode_group())
        v.addWidget(self._build_dataset_group())
        v.addStretch()

        btn_row = QtWidgets.QHBoxLayout()
        btn_cancel = QtWidgets.QPushButton("Cancelar", col)
        btn_cancel.clicked.connect(self.reject)
        btn_finish = QtWidgets.QPushButton("Concluir revisão ✔", col)
        btn_finish.setToolTip(
            "Grava suas decisões (marcadas = aceitas; desmarcadas → <i>_rejected/</i>) "
            "e encerra a revisão. No modo Recortes + dataset, libera a geração do dataset.")
        btn_finish.setDefault(True)
        btn_finish.clicked.connect(self._on_finish)
        btn_row.addWidget(btn_cancel)
        btn_row.addStretch()
        btn_row.addWidget(btn_finish)
        v.addLayout(btn_row)
        return col

    def _build_mode_group(self) -> QtWidgets.QWidget:
        grp = QtWidgets.QGroupBox("Modo", self)
        v = QtWidgets.QVBoxLayout(grp)
        v.setSpacing(8)
        self.rb_producer = QtWidgets.QRadioButton("Só recortes", grp)
        self.rb_producer.setToolTip(
            "<b>Só recortes:</b> grava a revisão e para — você usa/importa o run "
            "depois. Não gera dataset agora.")
        self.rb_e2e = QtWidgets.QRadioButton("Recortes + dataset", grp)
        self.rb_e2e.setToolTip(
            "<b>Recortes + dataset:</b> após esta revisão, monta o dataset e o "
            "enriquece (embeddings + opcional OFIQ/SER-FIQ) e o registra no Gerenciador.")
        self.rb_producer.setChecked(True)
        self.rb_producer.toggled.connect(self._on_mode_changed)
        v.addWidget(self.rb_producer)
        v.addWidget(self.rb_e2e)
        return grp

    def _build_dataset_group(self) -> QtWidgets.QWidget:
        grp = QtWidgets.QGroupBox("Dataset", self)
        self._dataset_group = grp
        fds = QtWidgets.QFormLayout(grp)
        fds.setContentsMargins(10, 12, 10, 12)
        fds.setVerticalSpacing(12)
        fds.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.txt_name = QtWidgets.QLineEdit(grp)
        self.txt_name.setText(Path(self._run_dir).name)
        self.txt_name.setToolTip("Nome do <b>dataset</b> gerado. Default = nome do run.")

        self.txt_dataset_output = QtWidgets.QLineEdit(grp)
        self.txt_dataset_output.setText(self._dataset_output_dir or "")
        self.txt_dataset_output.setPlaceholderText("Pasta onde o dataset final será gravado")
        self.txt_dataset_output.setToolTip(
            "Pasta onde o <b>dataset final</b> (parquet) será gravado. Default: a "
            "pasta global de datasets, para reuso entre projetos.")
        btn_ds_out = QtWidgets.QPushButton("Procurar…", grp)
        btn_ds_out.clicked.connect(self._on_browse_dataset_output)
        ds_out_w = QtWidgets.QWidget(grp)
        ds_out_h = QtWidgets.QHBoxLayout(ds_out_w)
        ds_out_h.setContentsMargins(0, 0, 0, 0)
        ds_out_h.addWidget(self.txt_dataset_output, 1)
        ds_out_h.addWidget(btn_ds_out)

        # Tipo de dataset: autocontido (copia recortes) vs caminho absoluto.
        self.rb_self_contained = QtWidgets.QRadioButton("Autocontido", grp)
        self.rb_self_contained.setToolTip(
            "<b>Autocontido</b> (recomendado): copia os recortes para a pasta do "
            "dataset, com caminhos relativos. Portátil entre máquinas/drives.")
        self.rb_absolute = QtWidgets.QRadioButton("Caminho absoluto", grp)
        self.rb_absolute.setToolTip(
            "<b>Caminho absoluto</b>: referencia os recortes do run sem copiar. "
            "Mais leve, mas <b>quebra</b> se o run mudar de lugar.")
        self.rb_self_contained.setChecked(True)
        dest_w = QtWidgets.QWidget(grp)
        dest_h = QtWidgets.QHBoxLayout(dest_w)
        dest_h.setContentsMargins(0, 0, 0, 0)
        dest_h.addWidget(self.rb_self_contained)
        dest_h.addWidget(self.rb_absolute)
        dest_h.addStretch()

        # Tipo de imagem: face alinhada (112×112) vs recorte com margem.
        self.rb_img_aligned = QtWidgets.QRadioButton("Face alinhada", grp)
        self.rb_img_aligned.setToolTip(
            "<b>Face alinhada</b> (recomendado): usa a face 112×112 já alinhada — o "
            "embedding é extraído direto, <b>sem re-detectar</b>. Essencial para faces "
            "pequenas de vigilância, onde re-detectar o recorte falharia.")
        self.rb_img_crop = QtWidgets.QRadioButton("Face recortada", grp)
        self.rb_img_crop.setToolTip(
            "<b>Face recortada</b>: usa o recorte com margem (imagem de evidência). O "
            "enrich <b>re-detecta</b> a face no recorte — pode falhar em faces muito "
            "pequenas. Útil quando você quer o recorte com contexto como imagem do dataset.")
        self.rb_img_aligned.setChecked(True)
        img_w = QtWidgets.QWidget(grp)
        img_h = QtWidgets.QHBoxLayout(img_w)
        img_h.setContentsMargins(0, 0, 0, 0)
        img_h.addWidget(self.rb_img_aligned)
        img_h.addWidget(self.rb_img_crop)
        img_h.addStretch()

        emb_w = self._build_embedding_checks(grp)

        self.chk_ofiq = QtWidgets.QCheckBox("OFIQ", grp)
        self.chk_ofiq.setToolTip(
            "Calcula o <b>OFIQ</b> (qualidade de imagem facial) no enriquecimento. "
            "Não altera os recortes nem a seleção; custo extra.")
        self.chk_serfiq = QtWidgets.QCheckBox("SER-FIQ", grp)
        self.chk_serfiq.setToolTip(
            "Calcula o <b>SER-FIQ</b> (qualidade de imagem facial) no enriquecimento. "
            "Não altera os recortes nem a seleção; custo extra.")
        qual_w = QtWidgets.QWidget(grp)
        qual_h = QtWidgets.QHBoxLayout(qual_w)
        qual_h.setContentsMargins(0, 0, 0, 0)
        qual_h.addWidget(self.chk_ofiq)
        qual_h.addWidget(self.chk_serfiq)
        qual_h.addStretch()

        fds.addRow("Nome do dataset:", self.txt_name)
        fds.addRow("Pasta do dataset:", ds_out_w)
        fds.addRow("Tipo de dataset:", dest_w)
        fds.addRow("Tipo de imagem:", img_w)
        fds.addRow("Embeddings:", emb_w)
        fds.addRow("Qualidade:", qual_w)
        self._on_mode_changed()   # esconde o grupo inteiro em "Só recortes"
        return grp

    def _build_embedding_checks(self, parent: QtWidgets.QWidget) -> QtWidgets.QWidget:
        self._emb_checks: Dict[str, QtWidgets.QCheckBox] = {}  # ff_name -> QCheckBox
        avail = SepaelModel.available_on_disk() or list(SepaelModel)
        # Garante o árbitro presente mesmo se não instalado nesta máquina — o
        # dataset PRECISA carregar o embedding que selecionou as faces.
        if self._arbiter and not any(m.ff_name == self._arbiter for m in avail):
            extra = next((m for m in SepaelModel if m.ff_name == self._arbiter), None)
            if extra is not None:
                avail = [extra] + list(avail)
        box = QtWidgets.QWidget(parent)
        h = QtWidgets.QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        for m in avail:
            cb = QtWidgets.QCheckBox(m.label_pt, box)
            cb.setToolTip(
                "Embedding que o dataset vai carregar. O modelo-árbitro (lido do "
                "manifesto do run) fica sempre marcado e travado — o dataset precisa "
                "carregar o embedding que selecionou as faces. Os demais são extras.")
            self._emb_checks[m.ff_name] = cb
            h.addWidget(cb)
        h.addStretch()
        self._sync_embedding_checkboxes()
        return box

    def _sync_embedding_checkboxes(self):
        """O árbitro (manifesto) fica sempre marcado e travado."""
        for ff_name, cb in self._emb_checks.items():
            if ff_name == self._arbiter:
                cb.setChecked(True)
                cb.setEnabled(False)
            else:
                cb.setEnabled(True)

    def _on_mode_changed(self, *_):
        self._dataset_group.setVisible(self.rb_e2e.isChecked())

    def _on_browse_dataset_output(self):
        start = self.txt_dataset_output.text().strip() or self._dataset_output_dir or ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pasta do dataset final", start)
        if d:
            self.txt_dataset_output.setText(d)

    # ──────────────────────────── filtros ────────────────────────────
    @staticmethod
    def _metric_bounds(vals, fallback):
        vv = [v for v in vals if v is not None]
        if not vv:
            return fallback
        lo, hi = min(vv), max(vv)
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)

    def _set_filter_label(self, key: str, lo: float, hi: float, fmt: str):
        if fmt == "deg":
            txt = f"{lo:+.0f}° – {hi:+.0f}°"
        else:
            txt = f"{lo:.2f} – {hi:.2f}"
        self._filter_val_lbls[key].setText(txt)

    def _apply_filters_to_selection(self):
        """Marca os recortes visíveis dentro dos 4 intervalos, desmarca o resto.
        Métrica ausente (None) passa no critério — não exclui a face."""
        for t in self._visible_thumbs():
            ok = (self._filters["det"].contains(t.det_score)
                  and self._filters["fiqa"].contains(t.fiqa_score)
                  and self._filters["yaw"].contains(t.yaw)
                  and self._filters["pitch"].contains(t.pitch))
            t.chk.setChecked(ok)
        self._update_counts()

    # ──────────────────────────── recortes ───────────────────────────
    def _visible_thumbs(self) -> List["_Thumb"]:
        out = self._thumbs
        if self._label_filter is not None:
            out = [t for t in out if t.label == self._label_filter]
        if self._video_filter is not None:
            out = [t for t in out if t.source_video_filename == self._video_filter]
        return out

    def _refresh_visibility(self):
        visible = {id(t) for t in self._visible_thumbs()}
        for t in self._thumbs:
            t.setVisible(id(t) in visible)
        # Recompacta o FlowLayout (itens ocultos colapsam para tamanho zero).
        flow = getattr(self, "_flow", None)
        if flow is not None:
            flow.invalidate()
        self._update_counts()

    def _on_filter_changed(self, _idx: int):
        self._label_filter = self.cmb_filter.currentData()
        self._refresh_visibility()

    def _on_video_filter_changed(self, _idx: int):
        self._video_filter = self.cmb_video_filter.currentData()
        self._refresh_visibility()

    def _set_all(self, checked: bool):
        for t in self._visible_thumbs():
            t.chk.setChecked(checked)
        self._update_counts()

    def _update_counts(self):
        vis = self._visible_thumbs()
        n_marked = sum(1 for t in vis if t.chk.isChecked())
        self.lbl_count.setText(f"{n_marked} marcada(s) · {len(vis)} visível(eis)")

    # ──────────────────────────── decisões ───────────────────────────
    def _on_finish(self):
        for t in self._thumbs:
            status = "accepted" if t.is_accepted() else "rejected"
            try:
                self._svc.set_face_decision(self._run_dir, t.image_file,
                                            status=status, operator=self._operator)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Revisão", f"Falha em {t.image_file}: {e}")
        self._svc.commit_review(self._run_dir, operator=self._operator, skipped=False)
        self._outcome = "completed"
        self.accept()

    # ──────────────────── getters do dataset (lidos pelo controller) ────────────────────
    def get_mode(self) -> VideoExtractionMode:
        return VideoExtractionMode.END_TO_END if self.rb_e2e.isChecked() else VideoExtractionMode.PRODUCER

    def get_dataset_name(self) -> str:
        return self.txt_name.text().strip() or Path(self._run_dir).name

    def get_dataset_embeddings(self) -> List[str]:
        """ff_names marcados — o árbitro está sempre incluído."""
        out = [ff for ff, cb in getattr(self, "_emb_checks", {}).items() if cb.isChecked()]
        if self._arbiter and self._arbiter not in out:
            out.append(self._arbiter)
        return out

    def get_dataset_output_dir(self) -> str:
        return self.txt_dataset_output.text().strip() or (self._dataset_output_dir or "")

    def get_dataset_self_contained(self) -> bool:
        return self.rb_self_contained.isChecked()

    def get_image_type(self) -> str:
        """'aligned' (face 112×112 alinhada) ou 'crop' (recorte com margem)."""
        return "crop" if self.rb_img_crop.isChecked() else "aligned"

    def compute_ofiq(self) -> bool:
        return self.chk_ofiq.isChecked()

    def compute_serfiq(self) -> bool:
        return self.chk_serfiq.isChecked()
