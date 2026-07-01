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
from faceaudit.services.image_io import imread_unicode_safe
from faceaudit.services.video_frame_face_extraction_service import (
    VideoFrameFaceExtractionService,
)
from faceaudit.models.video.video_extraction_config import VideoExtractionConfig
from faceaudit.enums.sepael_model import SepaelModel

# Cores estáveis por alvo (índice → cor da caixa no quadro e da bolinha na
# lista). Tons médios, legíveis sobre quadros de vídeo.
_TARGET_COLORS = ["#1D9E75", "#BA7517", "#378ADD", "#D4537E",
                  "#639922", "#7F77DD", "#D85A30", "#E24B4A"]
_DEFAULT_THRESHOLD = 0.30


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
        # Mínimo enxuto + stretch (root.addWidget(view, 1)): cede em telas
        # baixas e cresce quando há espaço. A face continua clicável (a view
        # escala o quadro com KeepAspectRatio).
        self.setMinimumHeight(240)
        self._pix_item: Optional[QtWidgets.QGraphicsPixmapItem] = None
        self._rects: List[QtWidgets.QGraphicsRectItem] = []

    def set_frame(self, pixmap: QtGui.QPixmap):
        self._scene.clear()
        self._rects = []
        self._pix_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        self.fitInView(self._pix_item, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

    def set_boxes(self, boxes):
        """boxes: iterável de (x1, y1, x2, y2, color[, label]). O 6º elemento
        opcional desenha uma tag de texto na cor da caixa (rótulo do alvo)."""
        for it in self._rects:
            self._scene.removeItem(it)
        self._rects = []
        for box in boxes:
            x1, y1, x2, y2, color = box[:5]
            label = box[5] if len(box) > 5 else None
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidth(2)
            pen.setCosmetic(True)
            it = self._scene.addRect(x1, y1, x2 - x1, y2 - y1, pen)
            self._rects.append(it)
            if label:
                txt = self._scene.addText(label)
                txt.setDefaultTextColor(QtGui.QColor(color))
                f = txt.font()
                f.setPointSize(10)
                f.setBold(True)
                txt.setFont(f)
                # Tamanho de tela constante (não escala com o zoom da view).
                txt.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
                txt.setPos(x1, y1)
                self._rects.append(txt)

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

    def set_window(self, start, end):
        """Define as alças de início/fim (quadros), mantendo início ≤ fim."""
        s, e = self._clamp(start), self._clamp(end)
        self._start, self._end = (s, e) if s <= e else (e, s)
        self._current = self._clamp(self._current)
        self.update()
        self.windowChanged.emit()

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


class _TargetRowWidget(QtWidgets.QFrame):
    """Uma linha do painel de Alvos: bolinha de cor + rótulo editável +
    limiar de identidade próprio + status da âncora + remover. Guarda o
    `VideoTarget` construído (None enquanto sem âncora) e o índice da
    detecção ancorada (válido apenas no quadro-âncora)."""
    activated = QtCore.pyqtSignal(object)   # self
    removed = QtCore.pyqtSignal(object)     # self
    label_changed = QtCore.pyqtSignal(object)
    threshold_changed = QtCore.pyqtSignal(object)
    anchors_edit_requested = QtCore.pyqtSignal(object)   # self

    def __init__(self, parent, *, label: str, color: str, threshold: float):
        super().__init__(parent)
        self.color = color
        self.video_target = None
        self.anchor_det_index: Optional[int] = None
        self._active = False
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setObjectName("targetRow")

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(8, 5, 8, 5)
        h.setSpacing(8)
        self._swatch = QtWidgets.QLabel(self)
        self._swatch.setFixedSize(14, 14)
        self._swatch.setStyleSheet(f"background:{color}; border-radius:3px;")
        self.txt_label = QtWidgets.QLineEdit(label, self)
        self.txt_label.setToolTip(
            "Rótulo da identidade deste alvo (vira a pasta da face e a coluna "
            "'identity' no dataset).")
        self.txt_label.textChanged.connect(lambda _t: self.label_changed.emit(self))
        self.spn_thr = QtWidgets.QDoubleSpinBox(self)
        self.spn_thr.setRange(-1.0, 1.0)
        self.spn_thr.setSingleStep(0.05)
        self.spn_thr.setValue(threshold)
        self.spn_thr.setToolTip(
            "<b>Limiar de identidade</b> (cosseno) deste alvo — abaixo dele a face "
            "é descartada. Cada alvo tem o seu; calibre com <b>Testar limiar</b>.")
        self.spn_thr.valueChanged.connect(lambda _v: self.threshold_changed.emit(self))
        self.lbl_status = QtWidgets.QLabel("sem âncora", self)
        self.lbl_status.setStyleSheet("color:#b26a00; font-size:11px;")
        self.btn_anchors = QtWidgets.QPushButton("Âncoras", self)
        self.btn_anchors.setToolTip("Ver, combinar e remover âncoras deste alvo.")
        self.btn_anchors.setEnabled(False)
        self.btn_anchors.clicked.connect(lambda: self.anchors_edit_requested.emit(self))
        self.btn_remove = QtWidgets.QToolButton(self)
        self.btn_remove.setText("🗑")
        self.btn_remove.setAutoRaise(True)
        self.btn_remove.setToolTip("Remover este alvo.")
        self.btn_remove.clicked.connect(lambda: self.removed.emit(self))

        h.addWidget(self._swatch)
        h.addWidget(self.txt_label, 2)
        h.addWidget(QtWidgets.QLabel("limiar", self))
        h.addWidget(self.spn_thr)
        h.addWidget(self.lbl_status, 2)
        h.addWidget(self.btn_anchors)
        h.addWidget(self.btn_remove)
        self.set_active(False)

    def label_text(self) -> str:
        return self.txt_label.text().strip()

    def threshold(self) -> float:
        return float(self.spn_thr.value())

    def has_anchor(self) -> bool:
        return self.video_target is not None

    def anchor_frame_index(self) -> Optional[int]:
        return getattr(self.video_target, "anchor_frame_index", None)

    def set_anchor(self, video_target, det_index: int):
        self.video_target = video_target
        self.anchor_det_index = det_index
        self.refresh_anchor_status()

    def refresh_anchor_status(self):
        """Status com a contagem de âncoras e a mistura de fontes (vídeo/foto)."""
        t = self.video_target
        if t is None:
            self.clear_anchor()
            return
        n = t.anchor_embedding_count
        n_vid, n_img = t.anchor_source_counts()
        parts = []
        if n_vid:
            parts.append(f"{n_vid} vídeo")
        if n_img:
            parts.append(f"{n_img} foto")
        detail = (" · " + " · ".join(parts)) if parts else ""
        plural = "âncora" if n == 1 else "âncoras"
        self.lbl_status.setText(f"{n} {plural}{detail}")
        self.lbl_status.setStyleSheet("color:#2e7d32; font-size:11px;")
        self.btn_anchors.setEnabled(n > 0)

    def clear_anchor(self):
        self.video_target = None
        self.anchor_det_index = None
        self.lbl_status.setText("sem âncora")
        self.lbl_status.setStyleSheet("color:#b26a00; font-size:11px;")
        self.btn_anchors.setEnabled(False)

    def set_active(self, active: bool):
        self._active = active
        self.setStyleSheet(
            "QFrame#targetRow { border:2px solid #1976d2; border-radius:6px; "
            "background:#e8f1fb; }"
            if active else
            "QFrame#targetRow { border:0.5px solid #cfcfcf; border-radius:6px; }")

    def mousePressEvent(self, ev):
        self.activated.emit(self)
        super().mousePressEvent(ev)


class _AnchorsDialog(QtWidgets.QDialog):
    """Lista as âncoras de um alvo (a fonte de cada uma) e permite removê-las.
    A referência do alvo é a média de todas as âncoras."""

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.target = target
        self.setWindowTitle(f"Âncoras · {getattr(target, 'label', '')}")
        self.resize(500, 360)
        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(QtWidgets.QLabel(
            "Âncoras deste alvo — remova as que não quiser. O modo de combinação "
            "define como elas pontuam cada face.", self))

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Combinação:", self))
        self.cmb_mode = QtWidgets.QComboBox(self)
        self.cmb_mode.addItem("Centroide (média das âncoras)", "CENTROID")
        self.cmb_mode.addItem("Todas as âncoras (casa com qualquer uma)", "ANY")
        cur = getattr(target, "anchor_match_mode", "CENTROID")
        i = self.cmb_mode.findData(cur)
        self.cmb_mode.setCurrentIndex(i if i >= 0 else 0)
        self.cmb_mode.setToolTip(
            "<b>Centroide</b>: compara cada face com a <b>média</b> das âncoras — "
            "robusto para várias aparências da MESMA pessoa; mais conservador.<br>"
            "<b>Todas as âncoras</b>: compara com <b>cada</b> âncora e casa se "
            "bater com <b>qualquer</b> uma (máximo) — mais permissivo, mais recall "
            "e mais risco de falso positivo.")
        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.cmb_mode, 1)
        root.addLayout(mode_row)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        self._host = QtWidgets.QWidget(scroll)
        self._list = QtWidgets.QVBoxLayout(self._host)
        self._list.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._host)
        root.addWidget(scroll, 1)
        btn_close = QtWidgets.QPushButton("Fechar", self)
        btn_close.clicked.connect(self.accept)
        foot = QtWidgets.QHBoxLayout()
        foot.addStretch()
        foot.addWidget(btn_close)
        root.addLayout(foot)
        self._rebuild()

    def _rebuild(self):
        while self._list.count():
            it = self._list.takeAt(0)
            w = it.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        n = self.target.anchor_embedding_count
        if n == 0:
            self._list.addWidget(QtWidgets.QLabel("Sem âncoras.", self._host))
            return
        for i in range(n):
            src = (self.target.anchor_sources[i]
                   if i < len(self.target.anchor_sources) else {})
            line = QtWidgets.QFrame(self._host)
            line.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
            h = QtWidgets.QHBoxLayout(line)
            h.setContentsMargins(8, 4, 8, 4)
            thumb = QtWidgets.QLabel(line)
            thumb.setFixedSize(56, 56)
            thumb.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            thumb.setStyleSheet("background:#222; border-radius:4px;")
            bgr = src.get("thumbnail_bgr")
            if bgr is not None:
                thumb.setPixmap(_bgr_to_pixmap(bgr).scaled(
                    56, 56, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation))
            else:
                thumb.setText("—")
            h.addWidget(thumb)
            h.addWidget(QtWidgets.QLabel(self._describe(i + 1, src), line), 1)
            btn = QtWidgets.QPushButton("Remover", line)
            btn.clicked.connect(lambda _c, idx=i: self._remove(idx))
            h.addWidget(btn)
            self._list.addWidget(line)

    @staticmethod
    def _describe(n: int, src: dict) -> str:
        if src.get("type") == "image":
            return f"{n}. 📷 foto · {src.get('filename') or src.get('path') or '—'}"
        fi = src.get("frame_index")
        return f"{n}. 🎞 vídeo · quadro {fi}" if fi is not None else f"{n}. 🎞 vídeo"

    def _on_mode_changed(self, _idx: int):
        self.target.anchor_match_mode = self.cmb_mode.currentData() or "CENTROID"

    def _remove(self, index: int):
        self.target.remove_anchor(index)
        self._rebuild()


def _secs_to_qtime(s):
    s = max(0, int(round(s or 0)))
    return QtCore.QTime(s // 3600, (s % 3600) // 60, s % 60)


def _qtime_to_secs(t) -> float:
    return float(t.hour() * 3600 + t.minute() * 60 + t.second())


def _fmt_window(start_s, end_s) -> str:
    def f(x):
        x = max(0, int(round(x)))
        h, m, sec = x // 3600, (x % 3600) // 60, x % 60
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"
    return f"[{f(start_s)}–{'fim' if end_s is None else f(end_s)}]"


class _VideoRowWidget(QtWidgets.QWidget):
    """Linha de um vídeo na lista: nome + trecho (início/fim em HH:mm:ss)
    editável SEM carregar o vídeo. Default = vídeo inteiro (0 → duração); o fim
    é travado na duração. Sincroniza com o scrubber quando o vídeo é carregado."""
    window_changed = QtCore.pyqtSignal(str, float, float)   # path, start_s, end_s

    def __init__(self, parent, *, path: str, duration_s: float):
        super().__init__(parent)
        self._path = path
        self._duration = max(0.0, float(duration_s or 0.0))

        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(6, 2, 6, 2)
        h.setSpacing(6)

        self._lbl = QtWidgets.QLabel(Path(path).name, self)
        self._lbl.setMinimumWidth(80)
        h.addWidget(self._lbl, 1)

        h.addWidget(QtWidgets.QLabel("Início", self))
        self.te_start = QtWidgets.QTimeEdit(self)
        self.te_start.setDisplayFormat("HH:mm:ss")
        self.te_start.setFixedWidth(84)
        h.addWidget(self.te_start)
        h.addWidget(QtWidgets.QLabel("Fim", self))
        self.te_end = QtWidgets.QTimeEdit(self)
        self.te_end.setDisplayFormat("HH:mm:ss")
        self.te_end.setFixedWidth(84)
        h.addWidget(self.te_end)

        # Máximos travam na duração; default = vídeo inteiro (0 → duração).
        dur_q = _secs_to_qtime(self._duration) if self._duration > 0 else QtCore.QTime(23, 59, 59)
        self.te_start.setMaximumTime(dur_q)
        self.te_end.setMaximumTime(dur_q)
        self.te_start.setTime(QtCore.QTime(0, 0, 0))
        self.te_end.setTime(dur_q)
        self._lbl.setToolTip(f"{path}\nDuração: {dur_q.toString('HH:mm:ss')}")

        self.te_start.timeChanged.connect(self._on_edit)
        self.te_end.timeChanged.connect(self._on_edit)

    @property
    def path(self) -> str:
        return self._path

    def _on_edit(self):
        # Trava cruzada início ≤ fim: empurra o campo que NÃO foi editado.
        if self.te_start.time() > self.te_end.time():
            if self.sender() is self.te_start:
                self.te_end.blockSignals(True); self.te_end.setTime(self.te_start.time()); self.te_end.blockSignals(False)
            else:
                self.te_start.blockSignals(True); self.te_start.setTime(self.te_end.time()); self.te_start.blockSignals(False)
        s, e = self.get_window_secs()
        self.window_changed.emit(self._path, s, e)

    def get_window_secs(self):
        return _qtime_to_secs(self.te_start.time()), _qtime_to_secs(self.te_end.time())

    def set_window_secs(self, start_s, end_s):
        """Atualiza os campos sem emitir (sincronização vinda do scrubber)."""
        self.te_start.blockSignals(True); self.te_end.blockSignals(True)
        try:
            self.te_start.setTime(_secs_to_qtime(start_s))
            self.te_end.setTime(_secs_to_qtime(end_s))
        finally:
            self.te_start.blockSignals(False); self.te_end.blockSignals(False)


class VideoDatasetSetupDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, use_gpu: bool = False, model: Optional[str] = None,
                 datasets_dir: Optional[str] = None, output_dir: Optional[str] = None,
                 dataset_output_dir: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("🎬 Criar Dataset a partir de Vídeo")
        # Mínimo enxuto para caber em notebooks pequenos; a área de rolagem
        # cobre o resto. O tamanho de abertura é calculado em _apply_initial_size.
        self.setMinimumSize(700, 460)
        # Botões de minimizar/maximizar/restaurar (como a janela do app).
        self.setWindowFlags(self.windowFlags()
                            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
                            | QtCore.Qt.WindowType.WindowMaximizeButtonHint)
        self._use_gpu = use_gpu
        self._model = model
        self._datasets_dir = datasets_dir
        self._output_dir = output_dir
        self._dataset_output_dir = dataset_output_dir

        self._service: Optional[VideoFrameFaceExtractionService] = None
        self._info = None
        self._video_path: Optional[str] = None
        self._video_infos: dict = {}      # path → VideoInfo (cache para a prévia)
        self._video_windows: dict = {}    # path → (start_s, end_s) — só quando PARCIAL
        self._video_durations: dict = {}  # path → duração em segundos (probe no add)
        self._syncing_window = False      # evita loop scrubber ↔ campos da linha
        self._current_frame: Optional[np.ndarray] = None
        self._current_dets: list = []
        # Modo "imagem de referência": quando True, o quadro exibido é uma foto
        # estática (não um quadro do vídeo) e a próxima âncora é do tipo "image".
        self._anchor_source_is_image: bool = False
        self._anchor_image_path: Optional[str] = None
        self._target_rows: List[_TargetRowWidget] = []
        self._active_row: Optional[_TargetRowWidget] = None

        # Estrangulamento do scrubber: arrastar a alça dispara muitos
        # currentChanged; ler o quadro (seek) a cada um trava o arraste em
        # vídeos longos. O rótulo atualiza ao vivo (barato); a leitura é adiada.
        self._scrub_timer = QtCore.QTimer(self)
        self._scrub_timer.setSingleShot(True)
        self._scrub_timer.setInterval(90)
        self._scrub_timer.timeout.connect(self._on_scrub_timeout)
        self._pending_scrub_idx: Optional[int] = None
        # Assinatura (modelo, det_size, det_thresh) do serviço já construído —
        # evita recriar a FFService (recarregar modelos) ao só trocar de vídeo.
        self._service_sig: Optional[tuple] = None

        self._build_ui()
        self._add_target(activate=True)  # semeia o 1º alvo (caso single = "1 alvo")
        self._apply_initial_size()         # abre dentro da tela (telas baixas)

    # ───────────────────────────── UI ─────────────────────────────
    def _build_ui(self):
        # `outer` é o layout da janela: [conteúdo rolável | barra de ações fixa].
        # Todo o formulário vai para `content` (dentro da QScrollArea), então a
        # janela pode encolher abaixo do mínimo do conteúdo — essencial em telas
        # baixas, onde antes os botões caíam para fora.
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        content = QtWidgets.QWidget(self)
        self._scroll_content = content
        root = QtWidgets.QVBoxLayout(content)
        # Duas colunas via splitter NÃO-colapsável: prévia + scrubber + detecção
        # à esquerda; vídeo/alvos/extração à direita. A coluna direita nunca
        # some (childrenCollapsible=False + largura mínima); a prévia não a engole.
        self._columns = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, content)
        self._columns.setChildrenCollapsible(False)
        root.addWidget(self._columns, 1)
        left_w = QtWidgets.QWidget(self._columns)
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        right_w = QtWidgets.QWidget(self._columns)
        right = QtWidgets.QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        left_w.setMinimumWidth(340)
        right_w.setMinimumWidth(300)
        self._columns.addWidget(left_w)
        self._columns.addWidget(right_w)
        self._columns.setStretchFactor(0, 3)
        self._columns.setStretchFactor(1, 2)

        # ── Grupo Vídeos (direita): lista multivídeo + carregar ──
        g_src = QtWidgets.QGroupBox("Vídeos", self)
        gs = QtWidgets.QVBoxLayout(g_src)
        add_row = QtWidgets.QHBoxLayout()
        btn_add_videos = QtWidgets.QPushButton("➕ Adicionar vídeos…", self)
        btn_add_videos.setToolTip(
            "Escolhe um ou mais arquivos de vídeo. O alvo é procurado em <b>TODOS</b> "
            "eles, gerando um único dataset (cada face marcada com seu vídeo de origem).")
        btn_add_videos.clicked.connect(self._on_add_videos)
        add_row.addWidget(btn_add_videos)
        add_row.addStretch()
        gs.addLayout(add_row)
        self.lst_videos = QtWidgets.QListWidget(self)
        self.lst_videos.setMaximumHeight(132)
        self.lst_videos.setToolTip(
            "Vídeos do caso. Defina o trecho (início/fim) direto na linha — sem "
            "carregar. Duplo-clique (ou Carregar) abre um na prévia para navegar e "
            "ancorar; a extração percorre todos.")
        self.lst_videos.itemDoubleClicked.connect(lambda _it: self._on_load_selected())
        gs.addWidget(self.lst_videos)
        vbtn = QtWidgets.QHBoxLayout()
        self.btn_load = QtWidgets.QPushButton("Carregar selecionado", self)
        self.btn_load.setToolTip(
            "Abre o vídeo selecionado na prévia (inicializa o detector com a "
            "Resolução/Confiança atuais) para navegar e ancorar.")
        self.btn_load.clicked.connect(self._on_load_selected)
        btn_rm_video = QtWidgets.QPushButton("Remover", self)
        btn_rm_video.setToolTip("Remove o vídeo selecionado da lista.")
        btn_rm_video.clicked.connect(self._on_remove_video)
        vbtn.addWidget(self.btn_load)
        vbtn.addWidget(btn_rm_video)
        vbtn.addStretch()
        gs.addLayout(vbtn)
        case_row = QtWidgets.QHBoxLayout()
        case_row.addWidget(QtWidgets.QLabel("Nome do caso:", self))
        self.txt_case = QtWidgets.QLineEdit(self)
        self.txt_case.setPlaceholderText("opcional — nomeia a run com vários vídeos")
        self.txt_case.setToolTip(
            "Nome da run quando há mais de um vídeo (default = nome do 1º vídeo).")
        case_row.addWidget(self.txt_case, 1)
        gs.addLayout(case_row)
        right.addWidget(g_src)

        # Widgets de detecção — montados na barra "Detecção" (esquerda, sob a prévia).
        self.cmb_model = QtWidgets.QComboBox(self)
        self._populate_model_combo()
        self.cmb_model.setToolTip(
            "<b>Modelo de embedding</b> usado na re-identificação — é o cosseno "
            "dele que decide se a face é o alvo. O limiar de cada alvo e o "
            "<b>Testar limiar</b> são calibrados para este modelo; trocá-lo exige "
            "recarregar o vídeo. Só aparecem os modelos instalados na máquina.")
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
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

        # View do quadro + slider
        self.view = _FrameView(self)
        self.view.setToolTip(
            "Quadro atual do vídeo. Depois de <b>Detectar</b>, clique sobre a face "
            "do alvo para defini-la como <b>âncora</b> — é ela que a ferramenta vai "
            "procurar no vídeo inteiro.")
        self.view.clicked.connect(self._on_view_clicked)
        left.addWidget(self.view, 1)

        # Timeline única: 3 alças arrastáveis (início · quadro atual · fim).
        self.scrubber = _RangeScrubber(self)
        self.scrubber.setToolTip(
            "Linha do tempo com três alças: <b>início</b>, <b>quadro atual</b> e "
            "<b>fim</b>. As alças de início/fim limitam o trecho do vídeo percorrido "
            "na extração; a do meio só navega para visualizar e ancorar.")
        self.scrubber.currentChanged.connect(self._on_scrub_changed)
        self.scrubber.currentReleased.connect(self._on_scrub_released)
        self.scrubber.windowChanged.connect(self._on_window_changed)
        left.addWidget(self.scrubber)

        info_row = QtWidgets.QHBoxLayout()
        legend = QtWidgets.QLabel(
            "<span style='color:#00c853'>●</span> início&nbsp;&nbsp;"
            "<span style='color:#ff6d00'>▮</span> quadro atual&nbsp;&nbsp;"
            "<span style='color:#d32f2f'>●</span> fim", self)
        legend.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.lbl_frame = QtWidgets.QLabel("—", self)
        self.lbl_window = QtWidgets.QLabel("", self)
        self.btn_detect = QtWidgets.QPushButton("Detectar neste quadro", self)
        self.btn_detect.setToolTip(
            "Roda o detector de faces no quadro atual e desenha as caixas (sem "
            "extrair embedding). Em seguida, clique numa caixa para escolher a "
            "âncora. Usa a <b>Resolução</b> e a <b>Confiança mínima</b> definidas "
            "no carregamento.")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self._detect_current)
        info_row.addWidget(legend)
        info_row.addStretch()
        info_row.addWidget(self.lbl_frame)
        info_row.addWidget(QtWidgets.QLabel("  ·  ", self))
        info_row.addWidget(self.lbl_window)
        info_row.addStretch()
        info_row.addWidget(self.btn_detect)
        left.addLayout(info_row)

        # ── Barra de detecção (esquerda, sob a prévia) ──
        g_det = QtWidgets.QGroupBox("Detecção", self)
        det_row = QtWidgets.QHBoxLayout(g_det)
        det_row.addWidget(QtWidgets.QLabel("Embedding:", self))
        det_row.addWidget(self.cmb_model, 1)
        det_row.addSpacing(10)
        det_row.addWidget(QtWidgets.QLabel("Resolução:", self))
        det_row.addWidget(self.spn_det_size)
        det_row.addSpacing(10)
        det_row.addWidget(QtWidgets.QLabel("Confiança:", self))
        det_row.addWidget(self.spn_det_thresh)
        left.addWidget(g_det)

        # ── Grupo Alvos (direita) ──
        g_tgt = QtWidgets.QGroupBox("Alvos", self)
        gt = QtWidgets.QVBoxLayout(g_tgt)
        hdr = QtWidgets.QHBoxLayout()
        self.lbl_targets_count = QtWidgets.QLabel("", self)
        self.lbl_targets_count.setStyleSheet("color:#666;")
        btn_add = QtWidgets.QPushButton("➕ Adicionar alvo", self)
        btn_add.setToolTip(
            "Adiciona um novo alvo. Depois, clique na face dele no quadro para "
            "ancorá-lo. Cada alvo vira uma identidade no dataset.")
        btn_add.clicked.connect(lambda: self._add_target(activate=True))
        self.btn_ref_image = QtWidgets.QPushButton("🖼 Imagem de referência", self)
        self.btn_ref_image.setToolTip(
            "Acrescenta uma âncora ao alvo <b>ativo</b> a partir de uma <b>foto</b> "
            "(busca por imagem): escolha o arquivo, a foto aparece na prévia e você "
            "clica na face. Serve para 'tenho a foto da pessoa, ache nos vídeos' e "
            "para reforçar a âncora com várias aparências do alvo.")
        self.btn_ref_image.clicked.connect(self._on_add_reference_image)
        hdr.addWidget(self.lbl_targets_count)
        hdr.addStretch()
        hdr.addWidget(self.btn_ref_image)
        hdr.addWidget(btn_add)
        gt.addLayout(hdr)
        self._targets_container = QtWidgets.QWidget(self)
        self._targets_layout = QtWidgets.QVBoxLayout(self._targets_container)
        self._targets_layout.setContentsMargins(0, 0, 0, 0)
        self._targets_layout.setSpacing(4)
        gt.addWidget(self._targets_container)
        hint = QtWidgets.QLabel(
            "Clique numa face do quadro para ancorar o alvo <b>ativo</b> (destacado). "
            "O alvo ativo recebe a próxima âncora; clique numa linha para trocá-lo.", self)
        hint.setTextFormat(QtCore.Qt.TextFormat.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        gt.addWidget(hint)
        right.addWidget(g_tgt)

        # ── Grupo Extração (direita) ──
        g_ext = QtWidgets.QGroupBox("Extração", self)
        fext = QtWidgets.QFormLayout(g_ext)
        self.spn_fps = QtWidgets.QDoubleSpinBox(self)
        self.spn_fps.setRange(0.1, 30.0)
        self.spn_fps.setDecimals(1)
        self.spn_fps.setSingleStep(0.5)
        self.spn_fps.setValue(2.0)
        self.spn_fps.setToolTip(
            "Taxa de amostragem: quantos quadros por segundo de vídeo são "
            "examinados (a ferramenta processa 1 a cada N quadros). ↑ amostra "
            "mais — mais faces candidatas e mais lento. Veja o efeito na contagem "
            "ao lado. O passo é derivado do FPS reportado pelo arquivo.")
        self.spn_fps.valueChanged.connect(self._update_fps_hint)
        self.lbl_fps_hint = QtWidgets.QLabel("", self)
        self.lbl_fps_hint.setStyleSheet("color:#666;")
        fps_w = QtWidgets.QWidget(self)
        fps_h = QtWidgets.QHBoxLayout(fps_w)
        fps_h.setContentsMargins(0, 0, 0, 0)
        fps_h.addWidget(self.spn_fps)
        fps_h.addWidget(self.lbl_fps_hint, 1)
        self.spn_margin = QtWidgets.QDoubleSpinBox(self)
        self.spn_margin.setRange(1.0, 5.0)
        self.spn_margin.setSingleStep(0.5)
        self.spn_margin.setValue(2.0)
        self.spn_margin.setToolTip(
            "<b>Fator</b> de expansão da caixa da face no recorte de evidência "
            "(não é em pixels). 2.0 = recorte com o dobro da largura/altura da "
            "caixa; 1.0 = caixa original; abaixo de 1.0 corta a face. Afeta só a "
            "imagem salva para laudo/custódia — não o embedding (face alinhada 112×112).")
        self.txt_output = QtWidgets.QLineEdit(self)
        self.txt_output.setText(self._output_dir or "")
        self.txt_output.setPlaceholderText("Pasta onde o run será criado")
        self.txt_output.setToolTip(
            "Pasta <b>onde a subpasta do run será criada</b> (uma por execução: "
            "nome do vídeo + etiqueta). Dentro dela ficam os recortes por rótulo, "
            "as faces alinhadas (<i>_aligned/</i>) e a proveniência (<i>_meta/</i>).")
        btn_run_out = QtWidgets.QPushButton("Procurar…", self)
        btn_run_out.setToolTip("Escolhe a pasta de saída do run.")
        btn_run_out.clicked.connect(self._on_browse_output)
        run_out_w = QtWidgets.QWidget(self)
        run_out_h = QtWidgets.QHBoxLayout(run_out_w)
        run_out_h.setContentsMargins(0, 0, 0, 0)
        run_out_h.addWidget(self.txt_output, 1)
        run_out_h.addWidget(btn_run_out)
        fext.addRow("Amostragem (quadros/s):", fps_w)
        fext.addRow("Margem do recorte:", self.spn_margin)
        fext.addRow("Pasta do run:", run_out_w)
        right.addWidget(g_ext)
        right.addStretch()

        # Modo + Dataset migraram para a janela de revisão (configurados lá, pois
        # "Importar Run de Vídeo" abre direto naquela janela). Aqui só se extrai.

        # Prévia + botões
        self.lbl_preview = QtWidgets.QLabel("", self)
        self.lbl_preview.setWordWrap(True)
        self.lbl_preview.setStyleSheet("color:#444; background:#f5f5f5; padding:6px; border-radius:4px;")
        root.addWidget(self.lbl_preview)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_preview = QtWidgets.QPushButton("Testar limiar", self)
        self.btn_preview.setToolTip(
            "Varre uma amostra de quadros e mostra a distribuição (mín/p5/mediana/"
            "p95/máx) de similaridade, det_score e fiqa, além de contadores por "
            "limiar (aceitos / abaixo / sem face). Serve para <b>calibrar o limiar "
            "de identidade</b> manualmente antes de extrair.")
        self.btn_preview.setEnabled(False)
        self.btn_preview.clicked.connect(self._on_preview)
        self.btn_extract = QtWidgets.QPushButton("Extrair ▶", self)
        self.btn_extract.setToolTip(
            "Inicia o run: percorre a janela escolhida e salva as faces cuja "
            "similaridade com a âncora atinge o <b>limiar de identidade</b> "
            "(cosseno cru), com recortes, faces alinhadas e proveniência. A "
            "revisão e a geração do dataset acontecem depois, em etapas próprias.")
        self.btn_extract.setEnabled(False)
        self.btn_extract.clicked.connect(self._on_accept)
        btn_cancel = QtWidgets.QPushButton("Cancelar", self)
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(self.btn_preview)
        btn_row.addStretch()
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_extract)

        # Conteúdo rolável + barra de ações fixa embaixo (sempre alcançável).
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        bar = QtWidgets.QWidget(self)
        self._action_bar = bar
        bar.setObjectName("actionBar")
        bar.setStyleSheet("#actionBar { border-top: 1px solid #d0d0d0; }")
        btn_row.setContentsMargins(9, 6, 9, 9)
        bar.setLayout(btn_row)
        outer.addWidget(bar)

    # ───────────────────────────── slots ─────────────────────────────
    def _on_add_videos(self):
        start = self._datasets_dir or ""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Escolher vídeo(s)", start,
            "Vídeos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;Todos (*.*)")
        if not paths:
            return
        existing = set(self._video_paths())
        for p in paths:
            if p in existing:
                continue
            dur = self._probe_duration_secs(p)     # metadados — sem carregar modelo
            self._video_durations[p] = dur
            it = QtWidgets.QListWidgetItem(self.lst_videos)
            it.setData(QtCore.Qt.ItemDataRole.UserRole, p)
            row = _VideoRowWidget(self.lst_videos, path=p, duration_s=dur)
            row.window_changed.connect(self._on_row_window_changed)
            it.setSizeHint(row.sizeHint())
            self.lst_videos.setItemWidget(it, row)
            existing.add(p)
        if self.lst_videos.currentRow() < 0 and self.lst_videos.count() > 0:
            self.lst_videos.setCurrentRow(self.lst_videos.count() - 1)

    def _probe_duration_secs(self, path) -> float:
        """Duração em segundos via metadados do cv2 — sem carregar o vídeo nem o
        detector (rápido, não decodifica quadros)."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                cap.release()
                return 0.0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
            cap.release()
            return (n / fps) if fps > 0 else 0.0
        except Exception:
            return 0.0

    def _row_for_path(self, path):
        for i in range(self.lst_videos.count()):
            it = self.lst_videos.item(i)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == path:
                return self.lst_videos.itemWidget(it)
        return None

    def _on_row_window_changed(self, path: str, start_s: float, end_s: float):
        """Campos de trecho de uma linha mudaram: persiste a janela (parcial-only)
        e, se aquele vídeo estiver carregado, reflete no scrubber."""
        dur = self._video_durations.get(path, 0.0)
        if start_s <= 0.01 and (dur <= 0 or end_s >= dur - 0.5):
            self._video_windows.pop(path, None)        # vídeo inteiro
        else:
            self._video_windows[path] = (start_s, end_s)
        if path == self._video_path and self._info is not None:
            self._syncing_window = True
            try:
                self._apply_window_to_scrubber(path)
            finally:
                self._syncing_window = False

    def _video_paths(self) -> List[str]:
        return [self.lst_videos.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.lst_videos.count())]

    def _on_load_selected(self):
        it = self.lst_videos.currentItem()
        if it is None and self.lst_videos.count() == 1:
            it = self.lst_videos.item(0)
        if it is None:
            QtWidgets.QMessageBox.information(self, "Vídeo", "Selecione um vídeo na lista.")
            return
        self._load_video(it.data(QtCore.Qt.ItemDataRole.UserRole))

    def _on_remove_video(self):
        it = self.lst_videos.currentItem()
        if it is None:
            return
        path = it.data(QtCore.Qt.ItemDataRole.UserRole)
        self.lst_videos.takeItem(self.lst_videos.row(it))   # remove item + row widget
        self._video_infos.pop(path, None)
        self._video_windows.pop(path, None)
        self._video_durations.pop(path, None)
        if path == self._video_path:
            self._video_path = None
            self._info = None
            self._current_frame = None
            self._current_dets = []
            self.scrubber.setEnabled(False)
            self.btn_detect.setEnabled(False)
            self.btn_preview.setEnabled(False)
            self.view.set_boxes([])
            self.lbl_frame.setText("—")

    def _apply_window_to_scrubber(self, path: str):
        """Reflete a janela do vídeo nas alças do scrubber (ao carregar ou ao
        editar os campos da linha). Janela ausente = vídeo inteiro."""
        if self._info is None:
            return
        last = max(0, self._info.frame_count_reported - 1)
        w = self._video_windows.get(path)
        if not w:
            self.scrubber.set_window(0, last)              # vídeo inteiro
            return
        fps = self._info.fps_reported or 30.0
        s_frame = max(0, min(last, int(round(w[0] * fps))))
        e_frame = last if w[1] is None else max(s_frame, min(last, int(round(w[1] * fps))))
        self.scrubber.set_window(s_frame, e_frame)

    def _on_browse_output(self):
        start = self.txt_output.text().strip() or self._output_dir or ""
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pasta de saída do run de vídeo", start)
        if d:
            self.txt_output.setText(d)

    # ──────────────────────────── modelo / embeddings ────────────────────────────
    def _populate_model_combo(self):
        avail = SepaelModel.available_on_disk() or list(SepaelModel)
        for m in avail:
            self.cmb_model.addItem(m.label_pt, m.ff_name)
        target_ff = self._model or SepaelModel.SEPAEL_V2.ff_name
        idx = self.cmb_model.findData(target_ff)
        self.cmb_model.setCurrentIndex(idx if idx >= 0 else 0)

    def _selected_model(self) -> Optional[str]:
        return self.cmb_model.currentData() or self._model

    def _on_model_changed(self, *_):
        # O árbitro mudou: as âncoras (embeddings por modelo) ficam inválidas.
        if self._service is None:
            return  # ainda não carregou — nada a invalidar
        # Modelo trocado após carregar: as âncoras (embeddings por modelo) ficam
        # inválidas e o FFService está obsoleto → exige recarregar o vídeo.
        for r in self._target_rows:
            r.clear_anchor()
        # Libera o capture de prévia antes de soltar o serviço — senão o handle
        # do vídeo fica aberto até o GC (lock no Windows).
        try:
            self._service.close_preview_cap()
        except Exception:
            pass
        self._service = None
        self._service_sig = None
        self._info = None
        self._current_dets = []
        self.scrubber.setEnabled(False)
        self.btn_detect.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_extract.setEnabled(False)
        self.view.set_boxes([])
        self._refresh_targets_count()
        self.lbl_frame.setText("Modelo alterado — clique em “Carregar vídeo” novamente.")

    def _apply_initial_size(self):
        """Abre num tamanho confortável para as DUAS colunas: largura suficiente
        para a coluna de controles aparecer inteira (não só ao aumentar a janela)
        e altura folgada para a prévia, sempre limitado à área útil da tela. Em
        telas estreitas, a área de rolagem cobre o excedente e a coluna direita
        não colapsa (splitter não-colapsável)."""
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        self._scroll_content.layout().activate()
        hint = self._scroll_content.sizeHint()
        desired_w = hint.width() + 24
        desired_h = hint.height() + self._action_bar.sizeHint().height() + 24
        target_w = max(1120, desired_w)
        target_h = max(700, desired_h)
        if avail is not None:
            target_w = min(target_w, avail.width() - 60)
            target_h = min(target_h, avail.height() - 80)
        w = max(820, int(target_w))
        h = max(480, int(target_h))
        self.resize(w, h)
        # Divisão inicial ~57/43 entre prévia e controles (arrastável depois).
        self._columns.setSizes([int(w * 0.57), int(w * 0.43)])

    # ──────────────────────────── alvos ────────────────────────────
    def _add_target(self, *, activate: bool = True, label: Optional[str] = None):
        ci = getattr(self, "_color_counter", 0)
        self._color_counter = ci + 1
        if label is None:
            label = "alvo" if ci == 0 else f"alvo_{ci + 1}"
        thr = (self._active_row.threshold() if self._active_row is not None
               else _DEFAULT_THRESHOLD)
        color = _TARGET_COLORS[ci % len(_TARGET_COLORS)]
        row = _TargetRowWidget(self._targets_container, label=label, color=color, threshold=thr)
        row.activated.connect(self._set_active_row)
        row.removed.connect(self._remove_target)
        row.label_changed.connect(self._on_target_label_changed)
        row.threshold_changed.connect(self._on_target_threshold_changed)
        row.anchors_edit_requested.connect(self._on_edit_anchors)
        self._targets_layout.addWidget(row)
        self._target_rows.append(row)
        if activate or self._active_row is None:
            self._set_active_row(row)
        self._refresh_targets_count()

    def _remove_target(self, row: "_TargetRowWidget"):
        if row not in self._target_rows:
            return
        if len(self._target_rows) == 1:
            # Mantém ao menos um alvo: limpa a âncora em vez de remover a linha.
            row.clear_anchor()
            self._redraw_boxes()
            self._refresh_targets_count()
            self.btn_extract.setEnabled(self._any_anchor())
            return
        self._target_rows.remove(row)
        self._targets_layout.removeWidget(row)
        row.setParent(None)
        row.deleteLater()
        if self._active_row is row:
            self._active_row = None
            self._set_active_row(self._target_rows[-1])
        self._redraw_boxes()
        self._refresh_targets_count()
        self.btn_extract.setEnabled(self._any_anchor())

    def _set_active_row(self, row: "_TargetRowWidget"):
        self._active_row = row
        for r in self._target_rows:
            r.set_active(r is row)

    def _refresh_targets_count(self):
        n = len(self._target_rows)
        anchored = sum(1 for r in self._target_rows if r.has_anchor())
        self.lbl_targets_count.setText(f"{n} alvo(s) · {anchored} com âncora")

    def _any_anchor(self) -> bool:
        return any(r.has_anchor() for r in self._target_rows)

    def _representative_threshold(self) -> float:
        for r in self._target_rows:
            if r.has_anchor():
                return r.threshold()
        return self._active_row.threshold() if self._active_row else _DEFAULT_THRESHOLD

    def _on_target_label_changed(self, row: "_TargetRowWidget"):
        if row.has_anchor():
            row.video_target.label = row.label_text() or "alvo"
        self._redraw_boxes()

    def _on_target_threshold_changed(self, row: "_TargetRowWidget"):
        if row.has_anchor():
            row.video_target.similarity_threshold = row.threshold()

    def _on_edit_anchors(self, row: "_TargetRowWidget"):
        if row.video_target is None:
            return
        _AnchorsDialog(row.video_target, parent=self).exec()
        # Remover todas as âncoras esvazia o alvo (volta a "sem âncora").
        if row.video_target.anchor_embedding_count == 0:
            row.clear_anchor()
        else:
            row.refresh_anchor_status()
        self._redraw_boxes()
        self._refresh_targets_count()
        self.btn_extract.setEnabled(self._any_anchor())

    def _det_size(self) -> int:
        return int(self.spn_det_size.value())

    def _detector_signature(self) -> tuple:
        return (self._selected_model(), self._det_size(),
                float(self.spn_det_thresh.value()))

    def _ensure_service(self, *, rebuild: bool = False):
        """Constrói o FFService/serviço de extração SOB DEMANDA — não exige
        carregar o vídeo (basta o modelo/detector atuais). Usado tanto pelo
        Carregar (rebuild=True, aplica novos parâmetros) quanto pela âncora por
        imagem e pela extração (build se ainda não existe)."""
        if self._service is not None and not rebuild:
            return
        # Libera o capture de prévia do serviço antigo antes de descartá-lo.
        if self._service is not None:
            try:
                self._service.close_preview_cap()
            except Exception:
                pass
        model_ff = self._selected_model()
        ff = FFService(models=[model_ff] if model_ff else None,
                       use_gpu=self._use_gpu, det_size=self._det_size(),
                       det_thresh=float(self.spn_det_thresh.value()))
        self._service = VideoFrameFaceExtractionService(ff)
        self._service_sig = self._detector_signature()

    def _ensure_service_current(self):
        """Garante o serviço aplicando os parâmetros ATUAIS do detector:
        reconstrói só quando a assinatura (modelo/det_size/det_thresh) mudou —
        senão reusa (não recarrega modelos). Usado por Carregar, Testar limiar e
        Imagem de referência, para que mudar Resolução/Confiança tenha efeito sem
        recarregar o vídeo."""
        need = (self._service is None) or (self._service_sig != self._detector_signature())
        self._ensure_service(rebuild=need)

    def _load_video(self, path: str):
        if not path or not Path(path).exists():
            QtWidgets.QMessageBox.warning(self, "Vídeo", "Arquivo de vídeo inválido.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # Só recria a FFService (recarrega modelos) se os parâmetros do
            # detector mudaram; trocar de vídeo reusa o serviço existente.
            self._ensure_service_current()
            # compute_hash=False: o hash do arquivo inteiro é caro e só é
            # necessário na extração (vai para o manifesto), não para a prévia.
            self._info = self._service.probe_video(path, compute_hash=False)
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao carregar:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        self._video_path = path
        self._video_infos[path] = self._info
        self._anchor_source_is_image = False
        total = max(1, self._info.frame_count_reported)
        self.scrubber.setEnabled(True)
        self.scrubber.set_range(0, total - 1)
        self._apply_window_to_scrubber(path)   # restaura a janela salva do vídeo
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
        self._anchor_source_is_image = False   # voltou a exibir um quadro do vídeo
        self.view.set_frame(_bgr_to_pixmap(frame))
        ts = idx / self._info.fps_reported if self._info.fps_reported else 0
        self.lbl_frame.setText(f"Quadro {idx} / {self._info.frame_count_reported}  ({ts:.1f}s)")

    # ─────────────── scrubber: rótulo ao vivo + leitura estrangulada ───────────────
    def _on_scrub_changed(self, idx: int):
        """Arraste da alça do meio. Atualiza o rótulo do quadro imediatamente
        (barato) e adia a leitura do quadro (seek/decode caro) para o timer —
        mantém o arraste fluido mesmo em vídeos longos."""
        if self._info is not None:
            fps = self._info.fps_reported or 0.0
            ts = (idx / fps) if fps else 0.0
            self.lbl_frame.setText(
                f"Quadro {idx} / {self._info.frame_count_reported}  ({ts:.1f}s)")
        self._pending_scrub_idx = int(idx)
        if not self._scrub_timer.isActive():
            self._scrub_timer.start()

    def _on_scrub_timeout(self):
        if self._pending_scrub_idx is not None:
            self._show_frame(self._pending_scrub_idx)

    def _on_scrub_released(self, value: int):
        """Soltou a alça: garante o quadro exato + roda a detecção (cara)."""
        self._scrub_timer.stop()
        self._pending_scrub_idx = None
        self._show_frame(int(value))
        self._detect_current()

    def done(self, r: int):
        # Para o timer de scrub ANTES de liberar o capture — senão um
        # _on_scrub_timeout ainda na fila reabriria o capture logo após o release.
        self._scrub_timer.stop()
        self._pending_scrub_idx = None
        # Libera o capture de prévia ao fechar (aceitar OU cancelar). A extração
        # abre seus próprios captures; o de prévia não deve segurar o arquivo.
        try:
            if self._service is not None:
                self._service.close_preview_cap()
        except Exception:
            pass
        super().done(r)

    def _on_window_changed(self):
        if self._info is None:
            return
        s, e = self.scrubber.start(), self.scrubber.end()
        fps = self._info.fps_reported or 1.0
        n = max(1, e - s + 1)
        self.lbl_window.setText(f"Janela: quadro {s}–{e} · {n} quadro(s) · {n / fps:.1f}s")
        self._update_fps_hint()
        # Se a mudança veio dos CAMPOS da linha, não re-sobrescreve (evita loop
        # + não sobrepõe o que o usuário acabou de digitar).
        if self._syncing_window:
            return
        # Scrubber → persiste a janela POR vídeo (faixa cheia = sem entrada) e
        # reflete nos campos da linha daquele vídeo.
        if self._video_path:
            last = max(0, self._info.frame_count_reported - 1)
            start_s, end_s = s / fps, e / fps
            if s <= 0 and e >= last:
                self._video_windows.pop(self._video_path, None)
            else:
                self._video_windows[self._video_path] = (start_s, end_s)
            row = self._row_for_path(self._video_path)
            if row is not None:
                row.set_window_secs(start_s, end_s)

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
        # Caixas em cinza-azul por padrão; coloridas (cor do alvo) apenas onde
        # há âncora cravada NESTE quadro — o det_index só vale no quadro-âncora.
        cur = self.scrubber.current() if self.scrubber.isEnabled() else None
        anchored = {}
        for row in self._target_rows:
            if (row.has_anchor() and row.anchor_det_index is not None
                    and row.anchor_frame_index() == cur):
                anchored[row.anchor_det_index] = (row.color, row.label_text() or "alvo")
        boxes = []
        for i, d in enumerate(self._current_dets):
            b = np.asarray(d["bbox"]).astype(float).flatten()
            if i in anchored:
                color, lbl = anchored[i]
                boxes.append((b[0], b[1], b[2], b[3], color, lbl))
            else:
                boxes.append((b[0], b[1], b[2], b[3], "#2196f3"))
        self.view.set_boxes(boxes)

    def _on_view_clicked(self, x: float, y: float):
        if not self._current_dets or not self._service:
            return
        row = self._active_row
        if row is None:
            QtWidgets.QMessageBox.information(
                self, "Alvos", "Adicione ou selecione um alvo antes de clicar numa face.")
            return
        model = self._selected_model()
        is_image = self._anchor_source_is_image
        src = self._current_anchor_source()
        if row.has_anchor():
            # Acrescenta âncora ao alvo ativo (multifonte: outro quadro/vídeo/foto).
            before = row.video_target.anchor_embedding_count
            res = self._service.add_anchor_from_click(
                row.video_target, self._current_frame, x, y, model=model, source=src)
            if res is None:
                return
            if row.video_target.anchor_embedding_count == before:
                self.lbl_frame.setText("Essa face já está ancorada — âncora não duplicada.")
            row.refresh_anchor_status()
        else:
            built = self._service.build_target_from_click(
                self._current_frame, x, y, label=row.label_text() or "alvo",
                similarity_threshold=row.threshold(), model=model,
                frame_index=(None if is_image else self.scrubber.current()))
            if built is None:
                return
            target, det_idx, _dets = built
            # Registra a fonte real (vídeo ou foto) preservando bbox/det_index.
            if target.anchor_sources:
                merged = dict(target.anchor_sources[-1])
                merged.update(src)
                target.anchor_sources[-1] = merged
            row.set_anchor(target, det_idx)
        self._redraw_boxes()
        self._refresh_targets_count()
        self.btn_extract.setEnabled(self._any_anchor())

    def _current_anchor_source(self) -> dict:
        """Proveniência da fonte do quadro atualmente exibido (vídeo ou foto)."""
        if self._anchor_source_is_image:
            p = self._anchor_image_path
            return {"type": "image", "path": p,
                    "filename": Path(p).name if p else None, "frame_index": None}
        return {"type": "video", "path": self._video_path,
                "filename": Path(self._video_path).name if self._video_path else None,
                "frame_index": self.scrubber.current() if self.scrubber.isEnabled() else None}

    def _on_add_reference_image(self):
        row = self._active_row
        if row is None:
            QtWidgets.QMessageBox.information(
                self, "Imagem de referência", "Selecione um alvo antes.")
            return
        start = self._datasets_dir or ""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Escolher imagem de referência", start,
            "Imagens (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;Todos (*.*)")
        if not path:
            return
        img = imread_unicode_safe(path)
        if img is None:
            QtWidgets.QMessageBox.warning(self, "Imagem", "Não foi possível ler a imagem.")
            return
        # NÃO exige carregar o vídeo — o detector/embedding é construído sob
        # demanda aqui (pode levar alguns segundos na 1ª vez).
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._ensure_service_current()
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(self, "Erro", f"Falha ao iniciar o detector:\n{e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        # Exibe a foto na MESMA prévia e entra no modo "imagem": a próxima âncora
        # é do tipo foto (frame_index None). Mover o scrubber volta ao vídeo.
        self._anchor_source_is_image = True
        self._anchor_image_path = path
        self._current_frame = img
        self.view.set_frame(_bgr_to_pixmap(img))
        self.lbl_frame.setText(
            f"Imagem de referência: {Path(path).name} — clique na face do alvo ativo")
        self._detect_current()

    def _on_preview(self):
        row = self._active_row
        if row is None or not row.has_anchor():
            QtWidgets.QMessageBox.information(
                self, "Prévia", "Ancore o alvo ativo (clique na face dele) primeiro.")
            return
        paths = self._video_paths() or ([self._video_path] if self._video_path else [])
        if not paths:
            QtWidgets.QMessageBox.information(
                self, "Prévia", "Adicione ao menos um vídeo primeiro.")
            return
        target = row.video_target
        thr = row.threshold()
        n_per = max(2, (14 + len(paths) - 1) // len(paths))   # ~14 quadros no total
        sims, dets, fiqas = [], [], []
        n_noface = n_accept = n_below = n_sampled = 0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            self._ensure_service_current()
            for p in paths:
                info = self._video_infos.get(p)
                if info is None:
                    try:
                        info = self._service.probe_video(p, compute_hash=False)
                        self._video_infos[p] = info
                    except Exception:
                        continue
                total = max(1, info.frame_count_reported)
                for idx in (int(v) for v in np.linspace(0, total - 1, n_per)):
                    fr = self._service.read_frame(p, idx)
                    if fr is None:
                        continue
                    n_sampled += 1
                    scored = self._service.score_frame_against_target(
                        fr, target, model=self._selected_model())
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

        scope = f"{len(paths)} vídeo(s)" if len(paths) > 1 else "1 vídeo"
        self.lbl_preview.setText(
            f"<b>Prévia do alvo '{row.label_text() or 'alvo'}' "
            f"({n_sampled} quadros de {scope}, limiar {thr:.2f}):</b><br>"
            f"aceitos (≥ limiar): <b>{n_accept}</b> | abaixo do limiar: {n_below} | sem face: {n_noface}<br>"
            f"similaridade c/ âncora: {_q(sims)}<br>"
            f"det_score: {_q(dets)} &nbsp;|&nbsp; fiqa_score: {_q(fiqas)} "
            f"<i>(CR-FIQA não é 0–1; calibre pela distribuição)</i>")

    def _on_accept(self):
        paths = self._video_paths()
        if not paths:
            QtWidgets.QMessageBox.warning(self, "Extração", "Adicione ao menos um vídeo.")
            return
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            QtWidgets.QMessageBox.warning(
                self, "Extração", f"Vídeo não encontrado:\n{missing[0]}")
            return
        if not self._any_anchor():
            QtWidgets.QMessageBox.warning(
                self, "Extração", "Ancore ao menos um alvo (clique na face) antes de extrair.")
            return
        # Bloqueia se algum alvo ficou sem âncora — destaca o pendente.
        pending = [r for r in self._target_rows if not r.has_anchor()]
        if pending:
            self._set_active_row(pending[0])
            QtWidgets.QMessageBox.warning(
                self, "Alvo sem âncora",
                f"O alvo '{pending[0].label_text() or 'alvo'}' está sem âncora. "
                "Clique na face dele num quadro ou remova-o antes de extrair.")
            return
        # Rótulos não-vazios e únicos (cada alvo vira uma identidade distinta).
        labels = [r.label_text() for r in self._target_rows]
        if any(not lbl for lbl in labels):
            QtWidgets.QMessageBox.warning(self, "Rótulo vazio", "Todo alvo precisa de um rótulo.")
            return
        if len(set(labels)) != len(labels):
            QtWidgets.QMessageBox.warning(
                self, "Rótulos repetidos",
                "Cada alvo precisa de um rótulo único — eles viram identidades distintas no dataset.")
            return
        self.accept()

    # ──────────────────────────── resultados ────────────────────────────
    def get_service(self) -> VideoFrameFaceExtractionService:
        return self._service

    def get_targets(self) -> List:
        """Lista de VideoTarget ancorados (≥1 garantido após _on_accept)."""
        return [r.video_target for r in self._target_rows if r.has_anchor()]

    def get_target(self):
        """Back-compat: primeiro alvo ancorado, ou None."""
        targets = self.get_targets()
        return targets[0] if targets else None

    def get_selected_model(self) -> Optional[str]:
        return self._selected_model()

    def _effective_video_path(self) -> Optional[str]:
        """1º vídeo efetivo (carregado ou o 1º da lista) — para compat de campo
        único; a extração usa `video_paths` (lista inteira)."""
        if self._video_path:
            return self._video_path
        paths = self._video_paths()
        return paths[0] if paths else None

    def get_config(self, output_root: str) -> VideoExtractionConfig:
        # Modo e configuração do dataset (embeddings/OFIQ/SER-FIQ/destino) são
        # decididos na janela de revisão; aqui só vão os parâmetros de extração.
        out = self.txt_output.text().strip() or output_root
        paths = self._video_paths()
        if not paths and self._effective_video_path():
            paths = [self._effective_video_path()]
        # Janela de tempo POR vídeo (paralela a paths): cada vídeo guarda a sua
        # (definida carregando+scrubando ou pelo diálogo "Definir trecho…").
        # Sem entrada = vídeo inteiro.
        video_windows = []
        for p in paths:
            w = self._video_windows.get(p)
            video_windows.append([w[0], w[1]] if w else None)
        return VideoExtractionConfig(
            video_path=(paths[0] if paths else ""), video_paths=paths,
            video_windows=video_windows,
            case_name=(self.txt_case.text().strip() or None),
            output_root=out,
            target_fps=float(self.spn_fps.value()),
            start_s=0.0, end_s=None,    # janela é por vídeo (video_windows)
            similarity_threshold=self._representative_threshold(),
            margin=float(self.spn_margin.value()),
            det_size=self._det_size(), det_thresh=float(self.spn_det_thresh.value()),
            recognition_model=self._selected_model(),
            compute_ff=True,
        )
