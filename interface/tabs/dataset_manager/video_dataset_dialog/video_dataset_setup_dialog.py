"""Janela de configuração da extração de faces de vídeo (Fase 5).

Fluxo: escolher vídeo + det_size → Carregar → navegar quadros (slider) →
clicar na face do alvo (âncora) → ajustar parâmetros → (opcional) prévia
de limiar → Extrair. Constrói o FFService no det_size escolhido e expõe,
após aceitar, a config + o alvo + o serviço para o orquestrador.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.services.ff_service import FFService
from faceaudit.services.video_frame_face_extraction_service import (
    VideoFrameFaceExtractionService,
)
from faceaudit.models.video.video_extraction_config import VideoExtractionConfig
from faceaudit.enums.video_extraction_mode import VideoExtractionMode

_DET_SIZES = [("320 (rápido)", 320), ("640 (recomendado)", 640),
              ("960 (vigilância / faces pequenas)", 960)]


def _bgr_to_pixmap(bgr: np.ndarray) -> QtGui.QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QtGui.QImage(rgb.data, w, h, ch * w,
                        QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(qimg)


class _FrameView(QtWidgets.QGraphicsView):
    """View do quadro com overlay de caixas; emite o clique em coords da imagem."""
    clicked = QtCore.pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setMinimumHeight(360)
        self._pix_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._rects: List[QtWidgets.QGraphicsRectItem] = []

    def set_frame(self, pixmap: QtGui.QPixmap):
        self._scene.clear()
        self._rects = []
        self._pix_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.fitInView(self._pix_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_boxes(self, boxes):
        for it in self._rects:
            self._scene.removeItem(it)
        self._rects = []
        for (x1, y1, x2, y2, color) in boxes:
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidth(2)
            pen.setCosmetic(True)
            it = self._scene.addRect(x1, y1, x2 - x1, y2 - y1, pen)
            self._rects.append(it)

    def mousePressEvent(self, ev):
        if self._pix_item is not None:
            sp = self.mapToScene(ev.position().toPoint())
            self.clicked.emit(sp.x(), sp.y())
        super().mousePressEvent(ev)

    def resizeEvent(self, ev):
        if self._pix_item is not None:
            self.fitInView(self._pix_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(ev)


class VideoDatasetSetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, use_gpu: bool = False, model: Optional[str] = None,
                 datasets_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("🎬 Criar Dataset a partir de Vídeo")
        self.resize(960, 760)
        self._use_gpu = use_gpu
        self._model = model
        self._datasets_dir = datasets_dir

        self._service: Optional[VideoFrameFaceExtractionService] = None
        self._info = None
        self._video_path: Optional[str] = None
        self._current_frame: Optional[np.ndarray] = None
        self._current_dets: list = []
        self._target = None
        self._anchor_det_index: Optional[int] = None

        self._build_ui()

    # ───────────────────────────── UI ─────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # Linha 1: vídeo + det_size + carregar
        row = QtWidgets.QHBoxLayout()
        self.txt_video = QtWidgets.QLineEdit(self)
        self.txt_video.setPlaceholderText("Caminho do arquivo de vídeo…")
        btn_browse = QtWidgets.QPushButton("Procurar…", self)
        btn_browse.clicked.connect(self._on_browse)
        self.cmb_det = QtWidgets.QComboBox(self)
        for label, _v in _DET_SIZES:
            self.cmb_det.addItem(label)
        self.cmb_det.setCurrentIndex(1)  # 640
        self.btn_load = QtWidgets.QPushButton("Carregar vídeo", self)
        self.btn_load.clicked.connect(self._on_load)
        row.addWidget(QtWidgets.QLabel("Vídeo:", self))
        row.addWidget(self.txt_video, 1)
        row.addWidget(btn_browse)
        row.addWidget(QtWidgets.QLabel("Detector:", self))
        row.addWidget(self.cmb_det)
        row.addWidget(self.btn_load)
        root.addLayout(row)

        # View do quadro + slider
        self.view = _FrameView(self)
        self.view.clicked.connect(self._on_view_clicked)
        root.addWidget(self.view, 1)

        sld_row = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._detect_current)
        self.btn_detect = QtWidgets.QPushButton("Detectar neste quadro", self)
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_current)
        self.lbl_frame = QtWidgets.QLabel("—", self)
        sld_row.addWidget(self.slider, 1)
        sld_row.addWidget(self.lbl_frame)
        sld_row.addWidget(self.btn_detect)
        root.addLayout(sld_row)

        self.lbl_anchor = QtWidgets.QLabel(
            "Âncora: <i>nenhuma</i> — clique na face do alvo num quadro com detecção.", self)
        self.lbl_anchor.setTextFormat(QtCore.Qt.TextFormat.RichText)
        root.addWidget(self.lbl_anchor)

        # Parâmetros
        form = QtWidgets.QFormLayout()
        self.spn_fps = QtWidgets.QDoubleSpinBox(self)
        self.spn_fps.setRange(0.1, 30.0)
        self.spn_fps.setValue(2.0)
        self.spn_thr = QtWidgets.QDoubleSpinBox(self)
        self.spn_thr.setRange(-1.0, 1.0)
        self.spn_thr.setSingleStep(0.05)
        self.spn_thr.setValue(0.30)
        self.spn_margin = QtWidgets.QDoubleSpinBox(self)
        self.spn_margin.setRange(1.0, 5.0)
        self.spn_margin.setSingleStep(0.5)
        self.spn_margin.setValue(2.0)
        self.txt_label = QtWidgets.QLineEdit("alvo", self)
        self.txt_name = QtWidgets.QLineEdit(self)
        self.txt_name.setPlaceholderText("Nome do dataset (default = nome do vídeo)")
        form.addRow("FPS-alvo:", self.spn_fps)
        form.addRow("Limiar de identidade (cosseno):", self.spn_thr)
        form.addRow("Margem do recorte:", self.spn_margin)
        form.addRow("Rótulo da identidade:", self.txt_label)
        form.addRow("Nome do dataset:", self.txt_name)
        root.addLayout(form)

        # Modo + processamento
        mode_row = QtWidgets.QHBoxLayout()
        self.rb_producer = QtWidgets.QRadioButton("Produtor (só recortes + metadados)", self)
        self.rb_e2e = QtWidgets.QRadioButton("Ponta a ponta (gera o dataset)", self)
        self.rb_producer.setChecked(True)
        self.chk_ofiq = QtWidgets.QCheckBox("OFIQ", self)
        self.chk_serfiq = QtWidgets.QCheckBox("SER-FIQ", self)
        mode_row.addWidget(self.rb_producer)
        mode_row.addWidget(self.rb_e2e)
        mode_row.addStretch()
        mode_row.addWidget(QtWidgets.QLabel("Qualidade (ponta a ponta):", self))
        mode_row.addWidget(self.chk_ofiq)
        mode_row.addWidget(self.chk_serfiq)
        root.addLayout(mode_row)

        # Prévia + botões
        self.lbl_preview = QtWidgets.QLabel("", self)
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color:#444; background:#f5f5f5; padding:6px; border-radius:4px;")
        root.addWidget(self.lbl_preview)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_preview = QtWidgets.QPushButton("Pré-visualizar limiar", self)
        self.btn_preview.setEnabled(False)
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_extract = QtWidgets.QPushButton("Extrair ▶", self)
        self.btn_extract.setEnabled(False)
        self.btn_extract.clicked.connect(self._on_accept)
        btn_cancel = QtWidgets.QPushButton("Cancelar", self)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_extract)
        root.addLayout(btn_row)

    # ───────────────────────────── slots ─────────────────────────────
    def _on_browse(self):
        start = self._datasets_dir or ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolher vídeo", start,
            "Vídeos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;Todos (*.*)")
        if path:
            self.txt_video.setText(path)

    def _det_size(self) -> int:
        return _DET_SIZES[self.cmb_det.currentIndex()][1]

    def _on_load(self):
        path = self.txt_video.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Vídeo", "Escolha um arquivo de vídeo válido.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            ff = FFService(models=[self._model] if self._model else None,
                           use_gpu=self._use_gpu, det_size=self._det_size())
            self._service = VideoFrameFaceExtractionService(ff)
            self._info = self._service.probe_video(path)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao carregar:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        self._video_path = path
        if not self.txt_name.text().strip():
            self.txt_name.setText(Path(path).stem)
        total = max(1, self._info.frame_count_reported)
        self.slider.setRange(0, total - 1)
        self.slider.setEnabled(True)
        self.btn_detect.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.slider.setValue(0)
        self._show_frame(0)
        self._detect_current()

    def _show_frame(self, idx: int):
        if not self._service or self._video_path is None:
            return
        frame = self._service.read_frame(self._video_path, idx)
        if frame is None:
            return
        self._current_frame = frame
        self.view.set_frame(_bgr_to_pixmap(frame))
        ts = idx / self._info.fps_reported if self._info.fps_reported else 0
        self.lbl_frame.setText(f"Quadro {idx} / {self._info.frame_count_reported}  ({ts:.1f}s)")

    def _on_slider_changed(self, idx: int):
        self._show_frame(idx)

    def _detect_current(self):
        if self._current_frame is None or not self._service:
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._current_dets = self._service.detect_in_frame(self._current_frame)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self._redraw_boxes()
        if not self._current_dets:
            self.lbl_frame.setText(self.lbl_frame.text() + "  — sem face (tente outro quadro ou det. 960)")

    def _redraw_boxes(self):
        boxes = []
        for i, d in enumerate(self._current_dets):
            b = np.asarray(d["bbox"]).astype(float).flatten()
            color = "#00c853" if i == self._anchor_det_index else "#2196f3"
            boxes.append((b[0], b[1], b[2], b[3], color))
        self.view.set_boxes(boxes)

    def _on_view_clicked(self, x: float, y: float):
        if not self._current_dets or not self._service:
            return
        built = self._service.build_target_from_click(
            self._current_frame, x, y, label=self.txt_label.text().strip() or "alvo",
            similarity_threshold=float(self.spn_thr.value()), model=self._model)
        if built is None:
            return
        target, det_idx, _dets = built
        self._target = target
        self._anchor_det_index = det_idx
        self._redraw_boxes()
        fp = target.anchor_embedding_fingerprint() or ""
        self.lbl_anchor.setText(
            f"Âncora: <b>quadro {self.slider.value()}</b>, face #{det_idx} — "
            f"fingerprint <code>{fp[:24]}…</code>")
        self.btn_extract.setEnabled(True)

    def _on_preview(self):
        if not self._service or self._target is None:
            QtWidgets.QMessageBox.information(self, "Prévia", "Carregue o vídeo e defina a âncora primeiro.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            total = max(1, self._info.frame_count_reported)
            idxs = [int(v) for v in np.linspace(0, total - 1, 14)]
            sims, dets, fiqas = [], [], []
            n_noface = n_accept = n_below = 0
            thr = float(self.spn_thr.value())
            for idx in idxs:
                fr = self._service.read_frame(self._video_path, idx)
                if fr is None:
                    continue
                scored = self._service.score_frame_against_target(fr, self._target, model=self._model)
                if not scored:
                    n_noface += 1
                    continue
                best = max(scored, key=lambda s: s["similarity"])
                sims.append(best["similarity"])
                dets.append(best["det_score"])
                if best["fiqa"] is not None:
                    fiqas.append(best["fiqa"])
                if best["similarity"] >= thr:
                    n_accept += 1
                else:
                    n_below += 1
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        def _q(arr):
            if not arr:
                return "—"
            a = np.asarray(arr, dtype=float)
            return f"min {a.min():.2f} | p50 {np.median(a):.2f} | max {a.max():.2f}"

        self.lbl_preview.setText(
            f"<b>Prévia ({len(idxs)} quadros amostrados, limiar {thr:.2f}):</b><br>"
            f"aceitos (≥ limiar): <b>{n_accept}</b> | abaixo do limiar: {n_below} | sem face: {n_noface}<br>"
            f"similaridade c/ âncora: {_q(sims)}<br>"
            f"det_score: {_q(dets)} &nbsp;|&nbsp; fiqa_score: {_q(fiqas)} "
            f"<i>(CR-FIQA não é 0–1; calibre pela distribuição)</i>")

    def _on_accept(self):
        if self._target is None or not self._video_path:
            QtWidgets.QMessageBox.warning(self, "Extração", "Defina a âncora (clique na face) antes.")
            return
        self.accept()

    # ──────────────────────────── resultados ────────────────────────────
    def get_service(self) -> VideoFrameFaceExtractionService:
        return self._service

    def get_target(self):
        return self._target

    def get_mode(self) -> VideoExtractionMode:
        return VideoExtractionMode.END_TO_END if self.rb_e2e.isChecked() else VideoExtractionMode.PRODUCER

    def get_dataset_name(self) -> str:
        return self.txt_name.text().strip() or Path(self._video_path).stem

    def get_config(self, output_root: str) -> VideoExtractionConfig:
        return VideoExtractionConfig(
            video_path=self._video_path, output_root=output_root,
            target_fps=float(self.spn_fps.value()),
            similarity_threshold=float(self.spn_thr.value()),
            margin=float(self.spn_margin.value()),
            det_size=self._det_size(), recognition_model=self._model,
            mode=self.get_mode(),
            compute_ff=True, compute_ofiq=self.chk_ofiq.isChecked(),
            compute_serfiq=self.chk_serfiq.isChecked(),
        )
