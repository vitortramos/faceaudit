"""StepConfigPanel — form dinâmico de configuração por step type
(PR I2).

Renderiza widgets diferentes conforme o `step_type` do config
selecionado. Emite `config_changed(config)` em todas as edições para
que o controller principal possa atualizar o pipeline em RAM e
disparar re-render do preview.

Suporta os 5 step types do MVP:
  - BLUR (kind, kernel_size, sigma)
  - JPEG (quality)
  - WEBP (quality)
  - CROP (mode, target_w/h, face_padding)
  - OCCLUSION (kind, position, width_pct/height_pct, radius_pct, mask)
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.blur_kind import BlurKind
from faceaudit.enums.crop_mode import CropMode
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.enums.occlusion_kind import OcclusionKind
from faceaudit.enums.occlusion_position import OcclusionPosition
from faceaudit.models.image_processing.blur_config import BlurConfig
from faceaudit.models.image_processing.crop_config import CropConfig
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.jpeg_compression_config import (
    JpegCompressionConfig,
)
from faceaudit.models.image_processing.occlusion_config import OcclusionConfig
from faceaudit.models.image_processing.webp_compression_config import (
    WebpCompressionConfig,
)


class StepConfigPanel(QtWidgets.QWidget):
    """Form dinâmico que renderiza os campos do step selecionado.

    Emite `config_changed` quando qualquer campo muda. O controller
    pai recebe o config atualizado e re-renderiza o preview.
    """

    config_changed = QtCore.pyqtSignal(object)  # ImageProcessingStepConfig
    # PR I6 — sinal emitido quando o usuário clica "Desenhar máscara".
    # Carrega o config (que tem `mask` attr) — controller intercepta,
    # abre canvas com a imagem corrente, recebe nova mask, atualiza
    # o config in-place e emite config_changed pra refresh.
    mask_canvas_requested = QtCore.pyqtSignal(object)  # OcclusionConfig

    # Tipos disponíveis no MVP — os demais (MOTION/BOX, USER_RECTANGLE,
    # RECTANGLE_NOISE) ficam ocultos até serem implementados.
    _BLUR_KINDS_AVAILABLE = [BlurKind.GAUSSIAN]
    _CROP_MODES_AVAILABLE = [CropMode.CENTERED, CropMode.FACE_DETECTED]
    # PR I7 — CIRCLE_BLACK + EYES/MOUTH adicionados.
    # PR I6 — MASK_FACE adicionado. Default agora — usuário desenha
    # diretamente em vez de configurar geometria abstrata.
    _OCCLUSION_KINDS_AVAILABLE = [
        OcclusionKind.MASK_FACE,
        OcclusionKind.RECTANGLE_BLACK,
        OcclusionKind.CIRCLE_BLACK,
    ]
    # Positions liberadas por kind (combos dinâmicos).
    # MASK_FACE não usa position (apenas a máscara desenhada).
    _OCCLUSION_POSITIONS_BY_KIND = {
        OcclusionKind.RECTANGLE_BLACK: [
            OcclusionPosition.CENTER,
            OcclusionPosition.RANDOM,
            OcclusionPosition.EYES,
            OcclusionPosition.MOUTH,
        ],
        OcclusionKind.CIRCLE_BLACK: [
            OcclusionPosition.CENTER,
            OcclusionPosition.RANDOM,
            OcclusionPosition.EYES,
            OcclusionPosition.MOUTH,
        ],
        OcclusionKind.MASK_FACE: [],  # vazio — combo escondido
    }

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._config: Optional[ImageProcessingStepConfig] = None
        self._suppress_signals = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._title = QtWidgets.QLabel("<i>Selecione um step à esquerda</i>")
        apply_themed_stylesheet(
            self._title,
            lambda t: (
                "color: " + t("TEXT_MUTED") + "; padding: 4px; font-size: 11px;"
            ),
        )
        layout.addWidget(self._title)

        # Form host — limpo e reconstruído sempre que o step muda
        self._form_host = QtWidgets.QWidget()
        self._form_layout = QtWidgets.QFormLayout(self._form_host)
        layout.addWidget(self._form_host)
        layout.addStretch()

    # ── API ────────────────────────────────────────────────────────────────

    def show_config(self, config: Optional[ImageProcessingStepConfig]):
        """Renderiza o form pra `config`. None limpa o panel."""
        self._config = config
        self._suppress_signals = True
        self._clear_form()
        if config is None:
            self._title.setText("<i>Selecione um step à esquerda</i>")
            self._suppress_signals = False
            return

        self._title.setText(
            f"<b>{config.step_type.label_pt}</b>"
        )

        if config.step_type == ImageProcessingStepType.BLUR:
            self._build_blur_form(config)
        elif config.step_type == ImageProcessingStepType.JPEG:
            self._build_jpeg_form(config)
        elif config.step_type == ImageProcessingStepType.WEBP:
            self._build_webp_form(config)
        elif config.step_type == ImageProcessingStepType.CROP:
            self._build_crop_form(config)
        elif config.step_type == ImageProcessingStepType.OCCLUSION:
            self._build_occlusion_form(config)

        self._suppress_signals = False

    def _clear_form(self):
        # `removeRow` é a API correta pra limpar QFormLayout — `takeAt`
        # não respeita rows e deixa widgets pendurados.
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)

    def _emit_changed(self):
        if self._suppress_signals or self._config is None:
            return
        self.config_changed.emit(self._config)

    # ── Forms por tipo ─────────────────────────────────────────────────────

    def _build_blur_form(self, config: BlurConfig):
        cmb_kind = QtWidgets.QComboBox()
        for k in self._BLUR_KINDS_AVAILABLE:
            cmb_kind.addItem(k.label_pt, userData=k)
        cmb_kind.setCurrentIndex(self._BLUR_KINDS_AVAILABLE.index(config.kind))

        spn_kernel = QtWidgets.QSpinBox()
        spn_kernel.setRange(1, 99)
        spn_kernel.setSingleStep(2)  # ímpar
        spn_kernel.setValue(int(config.kernel_size))
        spn_kernel.setSuffix(" px")
        spn_kernel.setToolTip(
            "Tamanho do kernel gaussiano. Ímpar (corrigido "
            "automaticamente se par)."
        )

        spn_sigma = QtWidgets.QDoubleSpinBox()
        spn_sigma.setRange(0.1, 50.0)
        spn_sigma.setSingleStep(0.5)
        spn_sigma.setValue(float(config.sigma))
        spn_sigma.setDecimals(2)
        spn_sigma.setToolTip(
            "Desvio-padrão do kernel gaussiano. Quanto maior, mais "
            "borrado."
        )

        def _on_kind():
            config.kind = cmb_kind.currentData()
            self._emit_changed()

        def _on_kernel():
            config.kernel_size = int(spn_kernel.value())
            self._emit_changed()

        def _on_sigma():
            config.sigma = float(spn_sigma.value())
            self._emit_changed()

        cmb_kind.currentIndexChanged.connect(_on_kind)
        spn_kernel.valueChanged.connect(_on_kernel)
        spn_sigma.valueChanged.connect(_on_sigma)

        self._form_layout.addRow("Tipo:", cmb_kind)
        self._form_layout.addRow("Kernel:", spn_kernel)
        self._form_layout.addRow("Sigma:", spn_sigma)

    def _build_jpeg_form(self, config: JpegCompressionConfig):
        spn_quality = QtWidgets.QSpinBox()
        spn_quality.setRange(1, 100)
        spn_quality.setValue(int(config.quality))
        spn_quality.setSuffix(" / 100")
        spn_quality.setToolTip(
            "1 = destruição agressiva, 85 = mensageiros típicos, "
            "100 = quase sem perda."
        )

        def _on_quality():
            config.quality = int(spn_quality.value())
            self._emit_changed()

        spn_quality.valueChanged.connect(_on_quality)
        self._form_layout.addRow("Qualidade:", spn_quality)

    def _build_webp_form(self, config: WebpCompressionConfig):
        spn_quality = QtWidgets.QSpinBox()
        spn_quality.setRange(1, 100)
        spn_quality.setValue(int(config.quality))
        spn_quality.setSuffix(" / 100")
        spn_quality.setToolTip(
            "WebP geralmente preserva melhor que JPEG na mesma "
            "quality. 80 é eficiente."
        )

        def _on_quality():
            config.quality = int(spn_quality.value())
            self._emit_changed()

        spn_quality.valueChanged.connect(_on_quality)
        self._form_layout.addRow("Qualidade:", spn_quality)

    def _build_crop_form(self, config: CropConfig):
        cmb_mode = QtWidgets.QComboBox()
        for m in self._CROP_MODES_AVAILABLE:
            cmb_mode.addItem(m.label_pt, userData=m)
        cmb_mode.setCurrentIndex(self._CROP_MODES_AVAILABLE.index(config.mode))

        spn_w = QtWidgets.QSpinBox()
        spn_w.setRange(8, 4096)
        spn_w.setValue(int(config.target_width))
        spn_w.setSuffix(" px")

        spn_h = QtWidgets.QSpinBox()
        spn_h.setRange(8, 4096)
        spn_h.setValue(int(config.target_height))
        spn_h.setSuffix(" px")

        spn_padding = QtWidgets.QDoubleSpinBox()
        spn_padding.setRange(0.0, 2.0)
        spn_padding.setSingleStep(0.05)
        spn_padding.setValue(float(config.face_padding))
        spn_padding.setDecimals(2)
        spn_padding.setToolTip(
            "Margem ao redor da bbox detectada (fração — 0.2 = 20%)."
        )

        # Linhas condicionais — guarda referências pra mostrar/esconder
        row_w_label = QtWidgets.QLabel("Largura:")
        row_h_label = QtWidgets.QLabel("Altura:")
        row_pad_label = QtWidgets.QLabel("Padding face:")

        def _refresh_visibility():
            is_centered = cmb_mode.currentData() == CropMode.CENTERED
            row_w_label.setVisible(is_centered)
            spn_w.setVisible(is_centered)
            row_h_label.setVisible(is_centered)
            spn_h.setVisible(is_centered)
            row_pad_label.setVisible(not is_centered)
            spn_padding.setVisible(not is_centered)

        def _on_mode():
            config.mode = cmb_mode.currentData()
            _refresh_visibility()
            self._emit_changed()

        def _on_w():
            config.target_width = int(spn_w.value())
            self._emit_changed()

        def _on_h():
            config.target_height = int(spn_h.value())
            self._emit_changed()

        def _on_pad():
            config.face_padding = float(spn_padding.value())
            self._emit_changed()

        cmb_mode.currentIndexChanged.connect(_on_mode)
        spn_w.valueChanged.connect(_on_w)
        spn_h.valueChanged.connect(_on_h)
        spn_padding.valueChanged.connect(_on_pad)

        self._form_layout.addRow("Modo:", cmb_mode)
        self._form_layout.addRow(row_w_label, spn_w)
        self._form_layout.addRow(row_h_label, spn_h)
        self._form_layout.addRow(row_pad_label, spn_padding)
        _refresh_visibility()

    def _build_occlusion_form(self, config: OcclusionConfig):
        """Combos dinâmicos: kind define quais positions estão
        disponíveis. Spinners de largura/altura (retângulo) vs raio
        (círculo) alternam conforme kind selecionado.
        """
        cmb_kind = QtWidgets.QComboBox()
        for k in self._OCCLUSION_KINDS_AVAILABLE:
            cmb_kind.addItem(k.label_pt, userData=k)
        if config.kind in self._OCCLUSION_KINDS_AVAILABLE:
            cmb_kind.setCurrentIndex(
                self._OCCLUSION_KINDS_AVAILABLE.index(config.kind)
            )

        cmb_pos = QtWidgets.QComboBox()

        # Largura (X) e altura (Y) independentes — substituem o antigo
        # `coverage_pct` único, que forçava o retângulo a ter o mesmo
        # aspecto da imagem. Permite ex: barra horizontal sobre os olhos
        # (width alto + height baixo) ou faixa vertical.
        spn_width = QtWidgets.QDoubleSpinBox()
        spn_width.setRange(0.01, 1.0)
        spn_width.setSingleStep(0.05)
        spn_width.setValue(float(config.width_pct))
        spn_width.setDecimals(2)
        spn_width.setToolTip(
            "Largura do retângulo como fração da LARGURA da imagem "
            "(0.5 = metade da largura)."
        )

        spn_height = QtWidgets.QDoubleSpinBox()
        spn_height.setRange(0.01, 1.0)
        spn_height.setSingleStep(0.05)
        spn_height.setValue(float(config.height_pct))
        spn_height.setDecimals(2)
        spn_height.setToolTip(
            "Altura do retângulo como fração da ALTURA da imagem "
            "(0.5 = metade da altura)."
        )

        spn_radius = QtWidgets.QDoubleSpinBox()
        spn_radius.setRange(0.01, 0.50)
        spn_radius.setSingleStep(0.02)
        spn_radius.setValue(float(config.radius_pct))
        spn_radius.setDecimals(3)
        spn_radius.setToolTip(
            "Raio do círculo como fração do MENOR LADO da imagem "
            "(0.10 = 10% do min(H, W)). Limite 0.50 mantém círculo "
            "dentro do frame em CENTER."
        )

        # Linhas condicionais — guardamos QLabel pra esconder/mostrar
        width_label = QtWidgets.QLabel("Largura:")
        height_label = QtWidgets.QLabel("Altura:")
        radius_label = QtWidgets.QLabel("Raio:")

        # PR I6 — botão pra abrir canvas (só visível em MASK_FACE)
        position_label = QtWidgets.QLabel("Posição:")
        btn_draw_mask = QtWidgets.QPushButton("🎨 Desenhar máscara…")
        btn_draw_mask.setToolTip(
            "Abre canvas pra desenhar polígonos/retângulos/elipses sobre "
            "a imagem do preview. Coordenadas salvas em [0..1] — mesma "
            "máscara aplica a imagens de tamanhos diferentes."
        )
        btn_draw_mask.clicked.connect(
            lambda: self.mask_canvas_requested.emit(config)
        )
        mask_label = QtWidgets.QLabel("Máscara:")

        def _refresh_mask_status():
            """Atualiza texto do botão refletindo estado da mask."""
            from faceaudit.models.image_processing.occlusion_mask import OcclusionMask
            if config.mask is None or config.mask.is_empty:
                btn_draw_mask.setText("🎨 Desenhar máscara…")
            else:
                n = len(config.mask.shapes)
                shape_word = "shape" if n == 1 else "shapes"
                btn_draw_mask.setText(
                    f"🎨 Editar ({n} {shape_word})…"
                )

        def _refresh_positions():
            """Repopula combo de positions conforme kind selecionado."""
            current_pos = cmb_pos.currentData() if cmb_pos.count() else None
            kind = cmb_kind.currentData()
            allowed = self._OCCLUSION_POSITIONS_BY_KIND.get(kind, [])
            cmb_pos.blockSignals(True)
            cmb_pos.clear()
            for p in allowed:
                cmb_pos.addItem(p.label_pt, userData=p)
            # Restaura seleção quando possível, senão CENTER (sempre presente)
            if current_pos in allowed:
                cmb_pos.setCurrentIndex(allowed.index(current_pos))
            elif config.position in allowed:
                cmb_pos.setCurrentIndex(allowed.index(config.position))
            else:
                cmb_pos.setCurrentIndex(0)
            cmb_pos.blockSignals(False)
            # Sincroniza config (mudança de kind pode forçar position)
            new_pos = cmb_pos.currentData()
            if new_pos is not None and new_pos != config.position:
                config.position = new_pos

        def _refresh_visibility():
            """Mostra largura/altura / raio / mask button conforme kind."""
            kind = cmb_kind.currentData()
            is_mask = kind == OcclusionKind.MASK_FACE
            is_rect = kind == OcclusionKind.RECTANGLE_BLACK
            is_circle = kind == OcclusionKind.CIRCLE_BLACK

            # MASK_FACE: esconde position/dimensões, mostra botão de canvas
            position_label.setVisible(not is_mask)
            cmb_pos.setVisible(not is_mask)
            mask_label.setVisible(is_mask)
            btn_draw_mask.setVisible(is_mask)

            width_label.setVisible(is_rect)
            spn_width.setVisible(is_rect)
            height_label.setVisible(is_rect)
            spn_height.setVisible(is_rect)
            radius_label.setVisible(is_circle)
            spn_radius.setVisible(is_circle)

            if is_mask:
                _refresh_mask_status()

        def _on_kind():
            config.kind = cmb_kind.currentData()
            _refresh_positions()
            _refresh_visibility()
            self._emit_changed()

        def _on_pos():
            new = cmb_pos.currentData()
            if new is not None:
                config.position = new
            self._emit_changed()

        def _on_width():
            config.width_pct = float(spn_width.value())
            self._emit_changed()

        def _on_height():
            config.height_pct = float(spn_height.value())
            self._emit_changed()

        def _on_radius():
            config.radius_pct = float(spn_radius.value())
            self._emit_changed()

        cmb_kind.currentIndexChanged.connect(_on_kind)
        cmb_pos.currentIndexChanged.connect(_on_pos)
        spn_width.valueChanged.connect(_on_width)
        spn_height.valueChanged.connect(_on_height)
        spn_radius.valueChanged.connect(_on_radius)

        self._form_layout.addRow("Tipo:", cmb_kind)
        self._form_layout.addRow(position_label, cmb_pos)
        self._form_layout.addRow(width_label, spn_width)
        self._form_layout.addRow(height_label, spn_height)
        self._form_layout.addRow(radius_label, spn_radius)
        self._form_layout.addRow(mask_label, btn_draw_mask)

        # Helper exposto pro controller atualizar texto do botão
        # após callback do canvas.
        self._refresh_mask_button = _refresh_mask_status

        # Estado inicial coerente com o kind do config
        _refresh_positions()
        _refresh_visibility()
