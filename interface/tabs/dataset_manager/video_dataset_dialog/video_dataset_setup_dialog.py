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
            "é descartada. Cada alvo tem o seu; calibre com <b>Pré-visualizar limiar</b>.")
        self.spn_thr.valueChanged.connect(lambda _v: self.threshold_changed.emit(self))
        self.lbl_status = QtWidgets.QLabel("sem âncora", self)
        self.lbl_status.setStyleSheet("color:#b26a00; font-size:11px;")
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
        self.lbl_status.setText(
            f"âncora: quadro {video_target.anchor_frame_index}, face #{det_index}")
        self.lbl_status.setStyleSheet("color:#2e7d32; font-size:11px;")

    def clear_anchor(self):
        self.video_target = None
        self.anchor_det_index = None
        self.lbl_status.setText("sem âncora")
        self.lbl_status.setStyleSheet("color:#b26a00; font-size:11px;")

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
        self._target_rows: List[_TargetRowWidget] = []
        self._active_row: Optional[_TargetRowWidget] = None

        self._build_ui()
        self._add_target(activate=True)  # semeia o 1º alvo (caso single = "1 alvo")

    # ───────────────────────────── UI ─────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # ── Grupo 1: Vídeo + Detecção (aplicados ao Carregar) ──
        g_src = QtWidgets.QGroupBox("1. Vídeo e detecção", self)
        gs = QtWidgets.QVBoxLayout(g_src)
        vid_row = QtWidgets.QHBoxLayout()
        self.txt_video = QtWidgets.QLineEdit(self)
        self.txt_video.setPlaceholderText("Caminho do arquivo de vídeo…")
        self.txt_video.setToolTip(
            "Caminho do arquivo de vídeo a analisar. Use <b>Procurar…</b> para "
            "selecionar. A <b>Resolução</b> e a <b>Confiança mínima</b> do detector "
            "só são aplicadas ao clicar em <b>Carregar vídeo</b>.")
        btn_browse = QtWidgets.QPushButton("Procurar…", self)
        btn_browse.setToolTip("Abre o seletor de arquivos para escolher o vídeo.")
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
        self.btn_load.setToolTip(
            "Abre o vídeo e inicializa o detector com a <b>Resolução</b> e a "
            "<b>Confiança mínima</b> atuais. Alterar esses dois campos depois "
            "<b>só tem efeito ao recarregar</b>.")
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
        self.view.setToolTip(
            "Quadro atual do vídeo. Depois de <b>Detectar</b>, clique sobre a face "
            "do alvo para defini-la como <b>âncora</b> — é ela que a ferramenta vai "
            "procurar no vídeo inteiro.")
        self.view.clicked.connect(self._on_view_clicked)
        root.addWidget(self.view, 1)

        # Timeline única: 3 alças arrastáveis (início · quadro atual · fim).
        self.scrubber = _RangeScrubber(self)
        self.scrubber.setToolTip(
            "Linha do tempo com três alças: <b>início</b>, <b>quadro atual</b> e "
            "<b>fim</b>. As alças de início/fim limitam o trecho do vídeo percorrido "
            "na extração; a do meio só navega para visualizar e ancorar.")
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
        root.addLayout(info_row)

        # ── Grupo 2: Alvos (multi-alvo) ──
        g_tgt = QtWidgets.QGroupBox("2. Alvos", self)
        gt = QtWidgets.QVBoxLayout(g_tgt)
        hdr = QtWidgets.QHBoxLayout()
        self.lbl_targets_count = QtWidgets.QLabel("", self)
        self.lbl_targets_count.setStyleSheet("color:#666;")
        btn_add = QtWidgets.QPushButton("➕ Adicionar alvo", self)
        btn_add.setToolTip(
            "Adiciona um novo alvo. Depois, clique na face dele no quadro para "
            "ancorá-lo. Cada alvo vira uma identidade no dataset.")
        btn_add.clicked.connect(lambda: self._add_target(activate=True))
        hdr.addWidget(self.lbl_targets_count)
        hdr.addStretch()
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
        root.addWidget(g_tgt)

        # ── Grupo 3: Extração ──
        g_ext = QtWidgets.QGroupBox("3. Extração", self)
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
        fext.addRow("Amostragem (quadros/s):", fps_w)
        fext.addRow("Margem do recorte:", self.spn_margin)
        root.addWidget(g_ext)

        # ── Grupo 4: Dataset ──
        g_ds = QtWidgets.QGroupBox("4. Dataset", self)
        fds = QtWidgets.QFormLayout(g_ds)
        self.txt_name = QtWidgets.QLineEdit(self)
        self.txt_name.setPlaceholderText("Nome do dataset (default = nome do vídeo)")
        self.txt_name.setToolTip(
            "Nome do <b>dataset</b> gerado no modo <b>Ponta a ponta</b> (ignorado "
            "no modo Produtor). Se vazio, usa o nome do vídeo. Não nomeia a pasta "
            "do run — só o dataset final.")
        self.txt_output = QtWidgets.QLineEdit(self)
        self.txt_output.setText(self._output_dir or "")
        self.txt_output.setPlaceholderText("Pasta onde o run será criado")
        self.txt_output.setToolTip(
            "Pasta <b>onde a subpasta do run será criada</b> (uma por execução: "
            "nome do vídeo + etiqueta). Dentro dela ficam os recortes por rótulo, "
            "as faces alinhadas (<i>_aligned/</i>) e a proveniência (<i>_meta/</i>).")
        btn_out = QtWidgets.QPushButton("Procurar…", self)
        btn_out.setToolTip("Escolhe a pasta de saída do run.")
        btn_out.clicked.connect(self._on_browse_output)
        out_w = QtWidgets.QWidget(self)
        out_h = QtWidgets.QHBoxLayout(out_w)
        out_h.setContentsMargins(0, 0, 0, 0)
        out_h.addWidget(self.txt_output, 1)
        out_h.addWidget(btn_out)
        self.rb_producer = QtWidgets.QRadioButton("Produtor (só recortes + metadados)", self)
        self.rb_producer.setToolTip(
            "<b>Produtor:</b> extrai os recortes + a proveniência (<i>_meta/</i>) e "
            "para aí — você importa/usa o run depois. Os recortes são idênticos aos "
            "do modo Ponta a ponta; muda só o que acontece em seguida.")
        self.rb_e2e = QtWidgets.QRadioButton("Ponta a ponta (gera o dataset)", self)
        self.rb_e2e.setToolTip(
            "<b>Ponta a ponta:</b> após a extração e a <b>revisão humana</b>, monta "
            "o dataset e o enriquece (ForensicFace e, se marcados, OFIQ/SER-FIQ), "
            "registrando-o no Gerenciador. O dataset só é finalizado após a revisão.")
        self.rb_producer.setChecked(True)
        self.chk_ofiq = QtWidgets.QCheckBox("OFIQ", self)
        self.chk_ofiq.setToolTip(
            "Calcula o <b>OFIQ</b> (qualidade de imagem facial, ISO/IEC 29794-5) no "
            "enriquecimento — só no modo Ponta a ponta. Não altera os recortes nem "
            "a seleção; custo de processamento extra.")
        self.chk_serfiq = QtWidgets.QCheckBox("SER-FIQ", self)
        self.chk_serfiq.setToolTip(
            "Calcula o <b>SER-FIQ</b> (qualidade de imagem facial) no enriquecimento "
            "— só no modo Ponta a ponta. Não altera os recortes nem a seleção; "
            "custo de processamento extra.")
        mode_w = QtWidgets.QWidget(self)
        mode_h = QtWidgets.QHBoxLayout(mode_w)
        mode_h.setContentsMargins(0, 0, 0, 0)
        mode_h.addWidget(self.rb_producer)
        mode_h.addWidget(self.rb_e2e)
        mode_h.addStretch()
        mode_h.addWidget(QtWidgets.QLabel("Qualidade:", self))
        mode_h.addWidget(self.chk_ofiq)
        mode_h.addWidget(self.chk_serfiq)
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
        built = self._service.build_target_from_click(
            self._current_frame, x, y, label=row.label_text() or "alvo",
            similarity_threshold=row.threshold(), model=self._model,
            frame_index=self.scrubber.current())
        if built is None:
            return
        target, det_idx, _dets = built
        row.set_anchor(target, det_idx)
        self._redraw_boxes()
        self._refresh_targets_count()
        self.btn_extract.setEnabled(self._any_anchor())

    def _on_preview(self):
        row = self._active_row
        if not self._service or row is None or not row.has_anchor():
            QtWidgets.QMessageBox.information(
                self, "Prévia",
                "Carregue o vídeo e ancore o alvo ativo (clique na face dele) primeiro.")
            return
        target = row.video_target
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            total = max(1, self._info.frame_count_reported)
            idxs = [int(v) for v in np.linspace(0, total - 1, 14)]
            sims, dets, fiqas = [], [], []
            n_noface = n_accept = n_below = 0
            thr = row.threshold()
            for idx in idxs:
                fr = self._service.read_frame(self._video_path, idx)
                if fr is None:
                    continue
                scored = self._service.score_frame_against_target(fr, target, model=self._model)
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
            f"<b>Prévia do alvo '{row.label_text() or 'alvo'}' "
            f"({len(idxs)} quadros amostrados, limiar {thr:.2f}):</b><br>"
            f"aceitos (≥ limiar): <b>{n_accept}</b> | abaixo do limiar: {n_below} | sem face: {n_noface}<br>"
            f"similaridade c/ âncora: {_q(sims)}<br>"
            f"det_score: {_q(dets)} &nbsp;|&nbsp; fiqa_score: {_q(fiqas)} "
            f"<i>(CR-FIQA não é 0–1; calibre pela distribuição)</i>")

    def _on_accept(self):
        if not self._video_path:
            QtWidgets.QMessageBox.warning(self, "Extração", "Carregue um vídeo antes.")
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
            similarity_threshold=self._representative_threshold(),
            margin=float(self.spn_margin.value()),
            det_size=self._det_size(), det_thresh=float(self.spn_det_thresh.value()),
            recognition_model=self._model, mode=self.get_mode(),
            compute_ff=True, compute_ofiq=self.chk_ofiq.isChecked(),
            compute_serfiq=self.chk_serfiq.isChecked(),
        )
