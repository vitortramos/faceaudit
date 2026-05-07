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

from typing import Optional

from PyQt6 import QtCore, QtWidgets

from faceaudit.enums.image_input_source import ImageInputSource

from .image_compare_widget import ImageCompareWidget
from .quality_scores_panel import QualityScoresPanel
from .step_config_panel import StepConfigPanel
from .steps_list_widget import StepsListWidget


class ImagePipelineDialogView(QtWidgets.QDialog):
    """Dialog top-level. Modal, 1300×820."""

    save_clicked = QtCore.pyqtSignal()
    name_changed = QtCore.pyqtSignal(str)
    input_source_changed = QtCore.pyqtSignal(object)  # ImageInputSource
    seed_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Editor de Pipeline de Imagens")
        self.setModal(True)
        self.resize(1300, 820)
        self._build_ui()

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
        self.cmb_input_source.currentIndexChanged.connect(
            lambda _i: self.input_source_changed.emit(
                self.cmb_input_source.currentData()
            )
        )
        top_row.addWidget(self.cmb_input_source)

        top_row.addWidget(QtWidgets.QLabel("Seed:"))
        self.spn_seed = QtWidgets.QSpinBox()
        self.spn_seed.setRange(0, 999_999_999)
        self.spn_seed.setValue(42)
        self.spn_seed.valueChanged.connect(self.seed_changed.emit)
        top_row.addWidget(self.spn_seed)

        top_row.addSpacing(20)
        self.btn_save = QtWidgets.QPushButton("💾 Salvar")
        self.btn_save.setToolTip(
            "Persistência na biblioteca global virá no PR I4."
        )
        self.btn_save.setEnabled(False)  # I4
        self.btn_save.clicked.connect(self.save_clicked.emit)
        top_row.addWidget(self.btn_save)

        main.addLayout(top_row)

        # ── Splitter: esquerda (lista + config) | direita (preview) ──
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Esquerda: vertical split entre lista e config
        left_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        self.steps_list = StepsListWidget()
        left_split.addWidget(self.steps_list)

        # Frame ao redor do StepConfigPanel pra dar contraste visual
        config_frame = QtWidgets.QFrame()
        config_frame.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        config_frame.setStyleSheet(
            "QFrame { background: #fafafa; border: 1px solid #e0e0e0; "
            "border-radius: 4px; }"
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

        # PR I3 — painel de scores embaixo do preview
        self.quality_scores = QualityScoresPanel()
        right_layout.addWidget(self.quality_scores)

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

        self.spn_seed.blockSignals(True)
        self.spn_seed.setValue(int(seed))
        self.spn_seed.blockSignals(False)
