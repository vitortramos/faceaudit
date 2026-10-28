"""ImagePipelineDialogView — QDialog modal que hospeda o editor
(PR I2).

Layout:
  - Topo: campo Nome + combo Input Source + spin Seed +
    botão Salvar (desabilitado em I2 — virá em I4)
  - QSplitter horizontal:
    - Esquerda (35%): StepsListWidget (em cima) + StepConfigPanel
      (embaixo)
    - Direita (65%): ImageCompareWidget
  - Fechar (sem prompt — sem auto-save em I2)
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.image_input_source import ImageInputSource

from .degradation_measures_panel import DegradationMeasuresPanel
from .image_compare_widget import ImageCompareWidget
from .quality_scores_panel import QualityScoresPanel
from .step_config_panel import StepConfigPanel
from .steps_list_widget import StepsListWidget


class ImagePipelineDialogView(QtWidgets.QDialog):
    """Dialog top-level. Modal, 1300×820."""

    save_clicked = QtCore.pyqtSignal()
    load_clicked = QtCore.pyqtSignal()  # PR I4
    name_changed = QtCore.pyqtSignal(str)
    input_source_changed = QtCore.pyqtSignal(object)  # ImageInputSource
    seed_changed = QtCore.pyqtSignal(int)
    questioned_load_clicked = QtCore.pyqtSignal()
    questioned_clear_clicked = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Editor de Pipeline de Imagens")
        self.setModal(True)
        self.resize(1500, 920)
        # PR I4 — controller injeta callback aqui pra interceptar
        # close quando há mudanças não salvas.
        self.confirm_close_callback: Optional[Callable[[], bool]] = None
        self._build_ui()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """PR I4 — pergunta confirmação se o controller marcou dirty.

        O controller injeta `confirm_close_callback` que retorna True
        pra fechar ou False pra cancelar. Sem callback, fecha direto.
        """
        if self.confirm_close_callback is not None:
            if not self.confirm_close_callback():
                event.ignore()
                return
        event.accept()
        super().closeEvent(event)

    def _build_ui(self):
        main = QtWidgets.QVBoxLayout(self)
        main.setContentsMargins(8, 8, 8, 8)
        main.setSpacing(6)

        # ── Topo: meta do pipeline ──
        top_row = QtWidgets.QHBoxLayout()

        top_row.addWidget(QtWidgets.QLabel("<b>Nome:</b>"))
        self.input_name = QtWidgets.QLineEdit()
        self.input_name.setPlaceholderText("untitled")
        self.input_name.editingFinished.connect(
            lambda: self.name_changed.emit(self.input_name.text().strip())
        )
        top_row.addWidget(self.input_name, 1)

        top_row.addWidget(QtWidgets.QLabel("Origem:"))
        self.cmb_input_source = QtWidgets.QComboBox()
        for src in ImageInputSource:
            self.cmb_input_source.addItem(src.label_pt, userData=src)
        self.cmb_input_source.setToolTip(
            "<b>Imagem original</b>: pipeline opera na foto inteira "
            "(face com background, pose variada). Útil pra simular "
            "degradação no upload/captura (compressão JPEG, blur, etc.).<br><br>"
            "<b>Face alinhada (112×112)</b>: o ForensicFace detecta e "
            "alinha a face ANTES de qualquer step rodar. Useful pra "
            "degradar exatamente o que o modelo vê. <b>Truque</b>: "
            "selecionar esta opção + zero steps = pipeline produz "
            "dataset com APENAS as faces alinhadas como PNG "
            "(substitui o antigo 'Copiar Faces Alinhadas' do "
            "Localizar Dataset)."
        )
        self.cmb_input_source.currentIndexChanged.connect(
            lambda _i: self.input_source_changed.emit(
                self.cmb_input_source.currentData()
            )
        )
        top_row.addWidget(self.cmb_input_source)

        # Help text dinâmico: muda conforme a origem selecionada.
        # Cinza pequeno, abaixo do dropdown. Critical pra que o usuário
        # entenda o efeito sem ter que descobrir empiricamente.
        self.lbl_input_source_help = QtWidgets.QLabel()
        apply_themed_stylesheet(
            self.lbl_input_source_help,
            lambda t: (
                "color: " + t("TEXT_MUTED")
                + "; font-size: 10px; padding: 0 4px;"
            ),
        )
        self.lbl_input_source_help.setWordWrap(True)

        top_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.spn_seed = QtWidgets.QSpinBox()
        self.spn_seed.setRange(0, 999_999_999)
        self.spn_seed.setValue(42)
        self.spn_seed.valueChanged.connect(self.seed_changed.emit)
        top_row.addWidget(self.spn_seed)

        top_row.addSpacing(20)

        # PR I4 — Carregar pipeline da biblioteca global
        self.btn_load = QtWidgets.QPushButton("📂 Carregar")
        self.btn_load.setToolTip(
            "Carregar pipeline salvo da biblioteca global (substitui "
            "o pipeline atual em memória — mudanças não salvas são "
            "descartadas após confirmação)."
        )
        self.btn_load.clicked.connect(self.load_clicked.emit)
        top_row.addWidget(self.btn_load)

        # PR I4 — Salvar pipeline na biblioteca global
        self.btn_save = QtWidgets.QPushButton("💾 Salvar")
        self.btn_save.setToolTip(
            "Salvar pipeline atual na biblioteca global como JSON."
        )
        self.btn_save.clicked.connect(self.save_clicked.emit)
        top_row.addWidget(self.btn_save)

        top_row.addSpacing(20)
        self.btn_questioned = QtWidgets.QPushButton("📄 Questionada…", parent=self)
        self.btn_questioned.setToolTip(
            "Carrega a imagem questionada que o pipeline deve reproduzir. "
            "Ela aparece ao lado da processada, com as medidas de "
            "degradação das duas, e fica registrada no pipeline (nome e "
            "SHA-256) e na cadeia de custódia do dataset adaptado."
        )
        self.btn_questioned.clicked.connect(self.questioned_load_clicked.emit)
        top_row.addWidget(self.btn_questioned)
        self.btn_questioned_clear = QtWidgets.QToolButton(parent=self)
        self.btn_questioned_clear.setText("✕")
        self.btn_questioned_clear.setToolTip("Retirar a questionada do pipeline")
        self.btn_questioned_clear.setEnabled(False)
        self.btn_questioned_clear.clicked.connect(self.questioned_clear_clicked.emit)
        top_row.addWidget(self.btn_questioned_clear)

        main.addLayout(top_row)

        self.lbl_questioned_info = QtWidgets.QLabel("", parent=self)
        self.lbl_questioned_info.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        apply_themed_stylesheet(
            self.lbl_questioned_info,
            lambda t: "color: " + t("TEXT_MUTED") + "; font-size: 10px; padding: 0 4px;",
        )
        self.lbl_questioned_info.setVisible(False)
        main.addWidget(self.lbl_questioned_info)
        main.addWidget(self.lbl_input_source_help)
        # Texto inicial — atualizado dinamicamente em set_pipeline_meta
        # e quando o usuário troca o dropdown.
        self.cmb_input_source.currentIndexChanged.connect(
            lambda _i: self._update_input_source_help()
        )
        self._update_input_source_help()

        # ── Splitter: esquerda (lista + config) | direita (preview) ──
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Esquerda: vertical split entre lista e config
        left_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.steps_list = StepsListWidget()
        left_split.addWidget(self.steps_list)

        # Frame ao redor do StepConfigPanel pra dar contraste visual
        config_frame = QtWidgets.QFrame()
        config_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        apply_themed_stylesheet(
            config_frame,
            lambda t: (
                "QFrame { background: " + t("SURFACE_RAISED") + "; "
                "border: 1px solid " + t("BORDER") + "; "
                "border-radius: 4px; }"
            ),
        )
        config_layout = QtWidgets.QVBoxLayout(config_frame)
        config_layout.setContentsMargins(0, 0, 0, 0)
        self.step_config = StepConfigPanel()
        config_layout.addWidget(self.step_config)
        left_split.addWidget(config_frame)

        left_split.setSizes([320, 480])
        self.splitter.addWidget(left_split)

        # Direita: container vertical com preview + painel de scores
        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.image_compare = ImageCompareWidget()
        right_layout.addWidget(self.image_compare, 1)

        # Embaixo do preview: scores de qualidade e medidas de degradação
        paineis = QtWidgets.QHBoxLayout()
        self.quality_scores = QualityScoresPanel(parent=right_container)
        paineis.addWidget(self.quality_scores, 1)
        self.degradation_measures = DegradationMeasuresPanel(parent=right_container)
        paineis.addWidget(self.degradation_measures, 1)
        for painel in (self.quality_scores, self.degradation_measures):
            paineis.setAlignment(painel, QtCore.Qt.AlignmentFlag.AlignTop)
        right_layout.addLayout(paineis)

        self.splitter.addWidget(right_container)

        self.splitter.setSizes([460, 840])
        main.addWidget(self.splitter, 1)

        # ── Botão Fechar embaixo ──
        bottom_row = QtWidgets.QHBoxLayout()
        bottom_row.addStretch()
        self.btn_close = QtWidgets.QPushButton("Fechar")
        self.btn_close.clicked.connect(self.accept)
        bottom_row.addWidget(self.btn_close)
        main.addLayout(bottom_row)

    def _update_input_source_help(self) -> None:
        """Texto cinza embaixo do dropdown — descreve o efeito da
        opção corrente. Permite o usuário entender que zero steps
        + Face alinhada produz dataset de faces alinhadas."""
        src = self.cmb_input_source.currentData()
        if src == ImageInputSource.ALIGNED_FACE:
            self.lbl_input_source_help.setText(
                "ℹ Detect + align rodam <b>antes</b> dos steps. "
                "Pipeline com zero steps salva apenas a face alinhada "
                "como PNG (substitui 'Copiar Faces Alinhadas' do "
                "Localizar Dataset)."
            )
        else:
            self.lbl_input_source_help.setText(
                "ℹ Steps operam na foto inteira — útil pra simular "
                "degradação no upload/captura (compressão, blur, "
                "oclusão, etc.)."
            )

    # ── Setters de estado ──────────────────────────────────────────────────

    def set_pipeline_meta(
        self,
        name: str,
        input_source: ImageInputSource,
        seed: int,
    ):
        """Popula os widgets do topo com o estado do pipeline."""
        self.input_name.blockSignals(True)
        self.input_name.setText(name)
        self.input_name.blockSignals(False)

        self.cmb_input_source.blockSignals(True)
        for i in range(self.cmb_input_source.count()):
            if self.cmb_input_source.itemData(i) == input_source:
                self.cmb_input_source.setCurrentIndex(i)
                break
        self.cmb_input_source.blockSignals(False)
        self._update_input_source_help()

        self.spn_seed.blockSignals(True)
        self.spn_seed.setValue(int(seed))
        self.spn_seed.blockSignals(False)

    def set_questioned_info(self, texto: Optional[str]):
        """Linha com nome e SHA-256 da questionada; None esconde."""
        self.lbl_questioned_info.setVisible(bool(texto))
        self.lbl_questioned_info.setText(texto or "")
        self.btn_questioned_clear.setEnabled(bool(texto))

    def set_save_load_enabled(self, enabled: bool, reason: str = ""):
        """PR I4 — desabilita Salvar/Carregar quando library_root
        não está disponível. `reason` vai pro tooltip."""
        self.btn_save.setEnabled(enabled)
        self.btn_load.setEnabled(enabled)
        if not enabled and reason:
            tip = f"Indisponível: {reason}"
            self.btn_save.setToolTip(tip)
            self.btn_load.setToolTip(tip)

    def set_dirty_indicator(self, dirty: bool):
        """PR I4 — adiciona/remove asterisco no título da janela
        quando há mudanças não salvas."""
        title = "Editor de Pipeline de Imagens"
        self.setWindowTitle(f"{title} *" if dirty else title)
