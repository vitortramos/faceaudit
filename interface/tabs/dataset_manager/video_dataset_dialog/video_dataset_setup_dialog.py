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


class _RangeScrubber(QtWidgets.QWidget):
    """Timeline única com 3 alças arrastáveis: início (verde) e fim
    (vermelho) delimitam a janela de extração; o playhead (laranja) é o
    quadro atual exibido. Mantém início ≤ fim; o playhead é livre."""
    currentChanged = QtCore.pyqtSignal(int)
    currentReleased = QtCore.pyqtSignal(int)
    windowChanged = QtCore.pyqtSignal()

    _MARGIN = 12

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = self._max = self._start = self._end = self._current = 0
        self._drag = None
        self.setMinimumHeight(46)
        self.setEnabled(False)
        self.setMouseTracking(True)

    def set_range(self, lo, hi):
        self._min, self._max = int(lo), int(hi)
        self._start, self._end, self._current = self._min, self._max, self._min
        self.update()

    def start(self) -> int:
        return self._start

    def end(self) -> int:
        return self._end

    def current(self) -> int:
        return self._current

    def _clamp(self, v) -> int:
        return max(self._min, min(self._max, int(round(v))))

    def _x(self, val) -> float:
        m = self._MARGIN
        w = max(1, self.width() - 2 * m)
        if self._max <= self._min:
            return float(m)
        return m + (val - self._min) / (self._max - self._min) * w

    def _val(self, x) -> int:
        m = self._MARGIN
        w = max(1, self.width() - 2 * m)
        return self._clamp(self._min + (x - m) / w * (self._max - self._min))

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        y = self.height() // 2
        m = self._MARGIN
        p.setPen(QtGui.QPen(QtGui.QColor("#cfcfcf"), 4))
        p.drawLine(m, y, self.width() - m, y)
        if not self.isEnabled():
            return
        xs, xe, xc = self._x(self._start), self._x(self._end), self._x(self._current)
        p.setPen(QtGui.QPen(QtGui.QColor("#1976d2"), 5))
        p.drawLine(int(xs), y, int(xe), y)
        p.setPen(QtGui.QPen(QtGui.QColor("#ff6d00"), 3))
        p.drawLine(int(xc), 5, int(xc), self.height() - 5)
        self._handle(p, xs, y, QtGui.QColor("#00c853"))
        self._handle(p, xe, y, QtGui.QColor("#d32f2f"))

    @staticmethod
    def _handle(p, x, y, color):
        p.setBrush(color)
        p.setPen(QtGui.QPen(color.darker(130), 1))
        p.drawEllipse(QtCore.QPointF(x, y), 7.0, 7.0)

    def mousePressEvent(self, ev):
        if not self.isEnabled():
            return
        x = ev.position().x()
        # Prioridade no playhead em caso de empate (alça mais usada).
        order = [("current", self._x(self._current)),
                 ("start", self._x(self._start)),
                 ("end", self._x(self._end))]
        self._drag = min(order, key=lambda kv: abs(kv[1] - x))[0]
        self._apply(x)

    def mouseMoveEvent(self, ev):
        if self._drag:
            self._apply(ev.position().x())

    def mouseReleaseEvent(self, ev):
        if self._drag == "current":
            self.currentReleased.emit(self._current)
        self._drag = None

    def _apply(self, x):
        v = self._val(x)
        if self._drag == "start":
            self._start = min(v, self._end)
            self.windowChanged.emit()
        elif self._drag == "end":
            self._end = max(v, self._start)
            self.windowChanged.emit()
        elif self._drag == "current":
            self._current = v
            self.currentChanged.emit(self._current)
        self.update()


class VideoDatasetSetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, use_gpu: bool = False, model: Optional[str] = None,
                 datasets_dir: Optional[str] = None, output_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("🎬 Criar Dataset a partir de Vídeo")
        self.resize(960, 760)
        self._use_gpu = use_gpu
        self._model = model
        self._datasets_dir = datasets_dir
        self._output_dir = output_dir

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

        # ── Grupo 1: Vídeo + Detecção (aplicados ao Carregar) ──
        g_src = QtWidgets.QGroupBox("1. Vídeo e detecção", self)
        gs = QtWidgets.QVBoxLayout(g_src)
        vid_row = QtWidgets.QHBoxLayout()
        self.txt_video = QtWidgets.QLineEdit(self)
        self.txt_video.setPlaceholderText("Caminho do arquivo de vídeo…")
        btn_browse = QtWidgets.QPushButton("Procurar…", self)
        btn_browse.clicked.connect(self._on_browse)
        vid_row.addWidget(QtWidgets.QLabel("Vídeo:", self))
        vid_row.addWidget(self.txt_video, 1)
        vid_row.addWidget(btn_browse)
        gs.addLayout(vid_row)

        det_row = QtWidgets.QHBoxLayout()
        self.spn_det_size = QtWidgets.QSpinBox(self)
        self.spn_det_size.setRange(256, 1280)
        self.spn_det_size.setSingleStep(32)
        self.spn_det_size.setValue(640)
        self.spn_det_size.setSuffix(" px")
        self.spn_det_size.setToolTip(
            "<b>Resolução do detector</b> (px, múltiplos de 32). Tamanho da "
            "imagem que entra no detector de faces.<br>"
            "↑ Maior detecta faces menores/distantes (vigilância), mais lento.<br>"
            "↓ Menor é mais rápido, mas perde faces pequenas.<br>"
            "320 ≈ rápido · 640 ≈ recomendado · 960 ≈ vigilância/faces pequenas.")
        self.spn_det_thresh = QtWidgets.QDoubleSpinBox(self)
        self.spn_det_thresh.setRange(0.05, 0.9)
        self.spn_det_thresh.setSingleStep(0.05)
        self.spn_det_thresh.setValue(0.5)
        self.spn_det_thresh.setToolTip(
            "<b>Confiança mínima da detecção</b>. ↓ aceita detecções de baixa "
            "confiança (faces pequenas/distantes/borradas), ao custo de mais "
            "falsos positivos. Padrão 0.50.")
        self.btn_load = QtWidgets.QPushButton("Carregar vídeo", self)
        self.btn_load.clicked.connect(self._on_load)
        det_row.addWidget(QtWidgets.QLabel("Resolução:", self))
        det_row.addWidget(self.spn_det_size)
        det_row.addSpacing(14)
        det_row.addWidget(QtWidgets.QLabel("Confiança mínima:", self))
        det_row.addWidget(self.spn_det_thresh)
        det_row.addStretch()
        det_row.addWidget(self.btn_load)
        gs.addLayout(det_row)
        root.addWidget(g_src)

        # View do quadro + slider
        self.view = _FrameView(self)
        self.view.clicked.connect(self._on_view_clicked)
        root.addWidget(self.view, 1)

        # Timeline única: 3 alças arrastáveis (início · quadro atual · fim).
        self.scrubber = _RangeScrubber(self)
        self.scrubber.currentChanged.connect(self._show_frame)
        self.scrubber.currentReleased.connect(lambda _v: self._detect_current())
        self.scrubber.windowChanged.connect(self._on_window_changed)
        root.addWidget(self.scrubber)

        info_row = QtWidgets.QHBoxLayout()
        legend = QtWidgets.QLabel(
            "<span style='color:#00c853'>●</span> início&nbsp;&nbsp;"
            "<span style='color:#ff6d00'>▮</span> quadro atual&nbsp;&nbsp;"
            "<span style='color:#d32f2f'>●</span> fim", self)
        legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_frame = QtWidgets.QLabel("—", self)
        self.lbl_window = QtWidgets.QLabel("", self)
        self.btn_detect = QtWidgets.QPushButton("Detectar neste quadro", self)
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_current)
        info_row.addWidget(legend)
        info_row.addStretch()
        info_row.addWidget(self.lbl_frame)
        info_row.addWidget(QtWidgets.QLabel("  ·  ", self))
        info_row.addWidget(self.lbl_window)
        info_row.addStretch()
        info_row.addWidget(self.btn_detect)
        root.addLayout(info_row)

        self.lbl_anchor = QtWidgets.QLabel(
            "Âncora: <i>nenhuma</i> — clique na face do alvo num quadro com detecção.", self)
        self.lbl_anchor.setTextFormat(QtCore.Qt.TextFormat.RichText)
        root.addWidget(self.lbl_anchor)

        # ── Grupo 3: Extração ──
        g_ext = QtWidgets.QGroupBox("3. Extração", self)
        fext = QtWidgets.QFormLayout(g_ext)
        self.spn_fps = QtWidgets.QDoubleSpinBox(self)
        self.spn_fps.setRange(0.1, 30.0)
        self.spn_fps.setDecimals(1)
        self.spn_fps.setSingleStep(0.5)
        self.spn_fps.setValue(2.0)
        self.spn_fps.setToolTip(
            "Quantos quadros por segundo de vídeo serão amostrados para "
            "extração. ↑ mais quadros = mais faces, mais lento.")
        self.spn_fps.valueChanged.connect(self._update_fps_hint)
        self.lbl_fps_hint = QtWidgets.QLabel("", self)
        self.lbl_fps_hint.setStyleSheet("color:#666;")
        fps_w = QtWidgets.QWidget(self)
        fps_h = QtWidgets.QHBoxLayout(fps_w)
        fps_h.setContentsMargins(0, 0, 0, 0)
        fps_h.addWidget(self.spn_fps)
        fps_h.addWidget(self.lbl_fps_hint, 1)
        self.spn_thr = QtWidgets.QDoubleSpinBox(self)
        self.spn_thr.setRange(-1.0, 1.0)
        self.spn_thr.setSingleStep(0.05)
        self.spn_thr.setValue(0.30)
        self.spn_thr.setToolTip(
            "Similaridade de cosseno mínima com a âncora para uma face ser "
            "aceita como o alvo. Calibre com 'Pré-visualizar limiar'.")
        self.spn_margin = QtWidgets.QDoubleSpinBox(self)
        self.spn_margin.setRange(1.0, 5.0)
        self.spn_margin.setSingleStep(0.5)
        self.spn_margin.setValue(2.0)
        self.spn_margin.setToolTip("Contexto ao redor da face no recorte de evidência.")
        fext.addRow("Amostragem (quadros/s):", fps_w)
        fext.addRow("Limiar de identidade (cosseno):", self.spn_thr)
        fext.addRow("Margem do recorte:", self.spn_margin)
        root.addWidget(g_ext)

        # ── Grupo 4: Dataset ──
        g_ds = QtWidgets.QGroupBox("4. Dataset", self)
        fds = QtWidgets.QFormLayout(g_ds)
        self.txt_label = QtWidgets.QLineEdit("alvo", self)
        self.txt_label.setToolTip("Nome da identidade (vira a pasta da face e a coluna 'identity').")
        self.txt_name = QtWidgets.QLineEdit(self)
        self.txt_name.setPlaceholderText("Nome do dataset (default = nome do vídeo)")
        self.txt_output = QtWidgets.QLineEdit(self)
        self.txt_output.setText(self._output_dir or "")
        self.txt_output.setPlaceholderText("Pasta onde o run será criado")
        btn_out = QtWidgets.QPushButton("Procurar…", self)
        btn_out.clicked.connect(self._on_browse_output)
        out_w = QtWidgets.QWidget(self)
        out_h = QtWidgets.QHBoxLayout(out_w)
        out_h.setContentsMargins(0, 0, 0, 0)
        out_h.addWidget(self.txt_output, 1)
        out_h.addWidget(btn_out)
        self.rb_producer = QtWidgets.QRadioButton("Produtor (só recortes + metadados)", self)
        self.rb_e2e = QtWidgets.QRadioButton("Ponta a ponta (gera o dataset)", self)
        self.rb_producer.setChecked(True)
        self.chk_ofiq = QtWidgets.QCheckBox("OFIQ", self)
        self.chk_serfiq = QtWidgets.QCheckBox("SER-FIQ", self)
        mode_w = QtWidgets.QWidget(self)
        mode_h = QtWidgets.QHBoxLayout(mode_w)
        mode_h.setContentsMargins(0, 0, 0, 0)
        mode_h.addWidget(self.rb_producer)
        mode_h.addWidget(self.rb_e2e)
        mode_h.addStretch()
        mode_h.addWidget(QtWidgets.QLabel("Qualidade:", self))
        mode_h.addWidget(self.chk_ofiq)
        mode_h.addWidget(self.chk_serfiq)
        fds.addRow("Rótulo da identidade:", self.txt_label)
        fds.addRow("Nome do dataset:", self.txt_name)
        fds.addRow("Pasta de saída:", out_w)
        fds.addRow("Modo:", mode_w)
        root.addWidget(g_ds)

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

    def _on_browse_output(self):
        start = self.txt_output.text().strip() or self._output_dir or ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pasta de saída do run de vídeo", start)
        if d:
            self.txt_output.setText(d)

    def _det_size(self) -> int:
        return int(self.spn_det_size.value())

    def _on_load(self):
        path = self.txt_video.text().strip()
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Vídeo", "Escolha um arquivo de vídeo válido.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            ff = FFService(models=[self._model] if self._model else None,
                           use_gpu=self._use_gpu, det_size=self._det_size(),
                           det_thresh=float(self.spn_det_thresh.value()))
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
        self.scrubber.setEnabled(True)
        self.scrubber.set_range(0, total - 1)
        self.btn_detect.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self._on_window_changed()
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

    def _on_window_changed(self):
        if self._info is None:
            return
        s, e = self.scrubber.start(), self.scrubber.end()
        fps = self._info.fps_reported or 1.0
        n = max(1, e - s + 1)
        self.lbl_window.setText(f"Janela: quadro {s}–{e} · {n} quadro(s) · {n / fps:.1f}s")
        self._update_fps_hint()

    def _update_fps_hint(self):
        """Mostra ao vivo o efeito da amostragem: 1 a cada N quadros, ~M no total."""
        if self._info is None:
            self.lbl_fps_hint.setText("")
            return
        vfps = self._info.fps_reported or 30.0
        target = max(0.1, self.spn_fps.value())
        every_n = max(1, round(vfps / target))
        s, e = self.scrubber.start(), self.scrubber.end()
        window = max(1, e - s + 1)
        est = max(1, window // every_n)
        self.lbl_fps_hint.setText(f"≈ 1 a cada {every_n} quadros · ~{est} no total")

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
            similarity_threshold=float(self.spn_thr.value()), model=self._model,
            frame_index=self.scrubber.current())
        if built is None:
            return
        target, det_idx, _dets = built
        self._target = target
        self._anchor_det_index = det_idx
        self._redraw_boxes()
        fp = target.anchor_embedding_fingerprint() or ""
        self.lbl_anchor.setText(
            f"Âncora: <b>quadro {self.scrubber.current()}</b>, face #{det_idx} — "
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
            return (f"min {a.min():.2f} | p5 {np.percentile(a, 5):.2f} | "
                    f"p50 {np.median(a):.2f} | p95 {np.percentile(a, 95):.2f} | max {a.max():.2f}")

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
        fps = (self._info.fps_reported if self._info and self._info.fps_reported else 30.0)
        out = self.txt_output.text().strip() or output_root
        return VideoExtractionConfig(
            video_path=self._video_path, output_root=out,
            target_fps=float(self.spn_fps.value()),
            start_s=self.scrubber.start() / fps,
            end_s=self.scrubber.end() / fps,
            similarity_threshold=float(self.spn_thr.value()),
            margin=float(self.spn_margin.value()),
            det_size=self._det_size(), det_thresh=float(self.spn_det_thresh.value()),
            recognition_model=self._model, mode=self.get_mode(),
            compute_ff=True, compute_ofiq=self.chk_ofiq.isChecked(),
            compute_serfiq=self.chk_serfiq.isChecked(),
        )
