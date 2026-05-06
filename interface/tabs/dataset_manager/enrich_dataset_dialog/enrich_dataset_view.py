from typing import List

from PyQt6 import QtCore
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class EnrichDatasetView(QDialog):
    """
    Janela de seleção de opções de Enriquecimento.

    Para cada serviço (FF / OFIQ / SER-FIQ):
    - Checkbox liga/desliga.
    - Status visual (Ausente / Parcial / Completo).
    - Botão "▾ Avançado" expande/colapsa painel de params operacionais.

    OFIQ não tem painel avançado — todos os params estão no config_path
    do executável externo (passado via Configurações do sistema).
    """

    # Modelos suportados pelo FFService.
    _SUPPORTED_FF_MODELS = ("sepaelv2", "sepaelv4")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enriquecer Dataset")
        self.resize(560, 480)

        self.checkboxes: dict = {}
        self.status_labels: dict = {}
        self._adv_widgets: dict = {}
        self._adv_buttons: dict = {}

        # Widgets de params (referenciados pelo controller via getters).
        self._ff_model_checks: dict = {}
        self._ff_det_size_spin: QSpinBox | None = None
        self._ff_extended_chk: QCheckBox | None = None
        self._serfiq_t_spin: QSpinBox | None = None
        self._serfiq_alpha_spin: QDoubleSpinBox | None = None

        self._init_ui()

    # ──────────────────────────────────────────────────────────────────
    # UI
    # ──────────────────────────────────────────────────────────────────

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        lbl_info = QLabel(
            "Selecione quais processos deseja executar nas imagens deste dataset.\n"
            "O sistema analisou os dados atuais para sugerir o que falta."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #666; margin-bottom: 10px;")
        main_layout.addWidget(lbl_info)

        group = QGroupBox("Processadores Disponíveis")
        group_layout = QVBoxLayout(group)
        group_layout.setSpacing(8)

        # Forensic Face — header + painel avançado
        self._add_service_header(
            group_layout,
            key="ff",
            title="Forensic Face",
            subtitle="Embedding (sepaelv2/v4), detecção, pose, qualidade, demografia",
            advanced_panel=self._build_ff_advanced_panel(),
        )

        # OFIQ — só header (sem params na UI; tudo está no config externo)
        self._add_service_header(
            group_layout,
            key="ofiq",
            title="OFIQ (Qualidade ISO)",
            subtitle="Métricas vetoriais de qualidade (ISO/IEC 29794-5)",
            advanced_panel=None,
        )

        # SER-FIQ — header + painel avançado
        self._add_service_header(
            group_layout,
            key="serfiq",
            title="SER-FIQ (Robustez)",
            subtitle="Score estocástico de estabilidade do embedding",
            advanced_panel=self._build_serfiq_advanced_panel(),
        )

        group_layout.addStretch()
        main_layout.addWidget(group)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Iniciar Processamento"
        )
        main_layout.addWidget(self.button_box)

    def _add_service_header(
        self,
        layout: QVBoxLayout,
        *,
        key: str,
        title: str,
        subtitle: str,
        advanced_panel: QWidget | None,
    ) -> None:
        """Linha com: checkbox + título/subtítulo + status + botão Avançado.
        Se advanced_panel for None, o botão Avançado não é criado."""
        # Linha principal
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        chk = QCheckBox()
        self.checkboxes[key] = chk

        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)
        lbl_title = QLabel(f"<b>{title}</b>")
        lbl_sub = QLabel(subtitle)
        lbl_sub.setStyleSheet("font-size: 10px; color: #888;")
        text_layout.addWidget(lbl_title)
        text_layout.addWidget(lbl_sub)

        lbl_status = QLabel("Carregando...")
        lbl_status.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        lbl_status.setStyleSheet("font-weight: bold; color: #999;")
        self.status_labels[key] = lbl_status

        row.addWidget(chk)
        row.addWidget(text_container, stretch=1)
        row.addWidget(lbl_status)

        if advanced_panel is not None:
            btn_adv = QPushButton("▸ Avançado")
            btn_adv.setCheckable(True)
            btn_adv.setStyleSheet(
                "QPushButton { font-size: 10px; padding: 2px 8px; }"
            )
            btn_adv.toggled.connect(
                lambda checked, k=key: self._toggle_advanced(k, checked)
            )
            self._adv_buttons[key] = btn_adv
            row.addWidget(btn_adv)

        layout.addWidget(row_widget)

        # Painel avançado (oculto inicialmente)
        if advanced_panel is not None:
            advanced_panel.setVisible(False)
            # Recuo visual + borda esquerda pra deixar claro que pertence
            # ao serviço.
            advanced_panel.setStyleSheet(
                "QWidget#AdvancedPanel { "
                "background: #f7f7f7; "
                "border-left: 3px solid #b0bec5; "
                "padding: 8px; "
                "}"
            )
            advanced_panel.setObjectName("AdvancedPanel")
            self._adv_widgets[key] = advanced_panel
            layout.addWidget(advanced_panel)

        # Linha separadora fina entre serviços.
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0; max-height: 1px;")
        layout.addWidget(sep)

    def _toggle_advanced(self, key: str, expanded: bool) -> None:
        widget = self._adv_widgets.get(key)
        btn = self._adv_buttons.get(key)
        if widget is not None:
            widget.setVisible(expanded)
        if btn is not None:
            btn.setText("▾ Avançado" if expanded else "▸ Avançado")

    # ──────────────────────────────────────────────────────────────────
    # Painéis avançados
    # ──────────────────────────────────────────────────────────────────

    def _build_ff_advanced_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 4, 4, 4)
        layout.setSpacing(6)

        # Modelos (multi-select)
        layout.addWidget(QLabel("<b>Modelos de embedding</b>"))
        models_row = QHBoxLayout()
        for model_name in self._SUPPORTED_FF_MODELS:
            chk = QCheckBox(model_name.replace("sepaelv", "SEPAEL V"))
            self._ff_model_checks[model_name] = chk
            models_row.addWidget(chk)
        # Default: sepaelv2 marcado
        self._ff_model_checks["sepaelv2"].setChecked(True)
        models_row.addStretch()
        layout.addLayout(models_row)
        models_help = QLabel(
            "Cada modelo extra dobra o uso de memória GPU. O <i>primeiro</i> "
            "modelo é o default — embeddings de outros modelos selecionados "
            "ficam no slot correspondente do FFInfo."
        )
        models_help.setWordWrap(True)
        models_help.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(models_help)

        # Det size
        det_row = QHBoxLayout()
        det_row.addWidget(QLabel("Det size:"))
        self._ff_det_size_spin = QSpinBox()
        self._ff_det_size_spin.setRange(160, 1280)
        self._ff_det_size_spin.setSingleStep(32)
        self._ff_det_size_spin.setValue(320)
        det_row.addWidget(self._ff_det_size_spin)
        det_help = QLabel(
            "<i>(maior = melhor pra fotos de baixa res, ~quadrático em GPU)</i>"
        )
        det_help.setStyleSheet("font-size: 10px; color: #888;")
        det_row.addWidget(det_help)
        det_row.addStretch()
        layout.addLayout(det_row)

        # Extended
        self._ff_extended_chk = QCheckBox(
            "Modo completo: FIQA + gender + age + yaw/pitch/roll"
        )
        self._ff_extended_chk.setChecked(True)
        layout.addWidget(self._ff_extended_chk)
        ext_help = QLabel(
            "<i>Desmarque para gerar dataset enxuto (apenas embedding + "
            "bbox + keypoints). Mais rápido.</i>"
        )
        ext_help.setWordWrap(True)
        ext_help.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(ext_help)

        return panel

    def _build_serfiq_advanced_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 4, 4, 4)
        layout.setSpacing(6)

        # T
        t_row = QHBoxLayout()
        t_row.addWidget(QLabel("T (iterações):"))
        self._serfiq_t_spin = QSpinBox()
        self._serfiq_t_spin.setRange(1, 100)
        self._serfiq_t_spin.setValue(10)
        t_row.addWidget(self._serfiq_t_spin)
        t_help = QLabel(
            "<i>(cada imagem é processada T vezes — mais = score mais "
            "estável, mais lento)</i>"
        )
        t_help.setStyleSheet("font-size: 10px; color: #888;")
        t_row.addWidget(t_help)
        t_row.addStretch()
        layout.addLayout(t_row)

        # Alpha
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Alpha:"))
        self._serfiq_alpha_spin = QDoubleSpinBox()
        self._serfiq_alpha_spin.setRange(0.1, 10.0)
        self._serfiq_alpha_spin.setSingleStep(0.1)
        self._serfiq_alpha_spin.setValue(1.5)
        alpha_row.addWidget(self._serfiq_alpha_spin)
        alpha_help = QLabel(
            "<i>(fator de escala do score; menor = compressão maior)</i>"
        )
        alpha_help.setStyleSheet("font-size: 10px; color: #888;")
        alpha_row.addWidget(alpha_help)
        alpha_row.addStretch()
        layout.addLayout(alpha_row)

        return panel

    # ──────────────────────────────────────────────────────────────────
    # API consumida pelo controller
    # ──────────────────────────────────────────────────────────────────

    def set_status(self, key: str, text: str, color: str):
        if key in self.status_labels:
            self.status_labels[key].setText(text)
            self.status_labels[key].setStyleSheet(
                f"font-weight: bold; color: {color};"
            )

    def set_checked(self, key: str, checked: bool):
        if key in self.checkboxes:
            self.checkboxes[key].setChecked(checked)

    def is_checked(self, key: str) -> bool:
        if key in self.checkboxes:
            return self.checkboxes[key].isChecked()
        return False

    # Getters dos params operacionais (consumidos pelo controller).

    def get_ff_models(self) -> List[str]:
        models = [
            name for name, chk in self._ff_model_checks.items() if chk.isChecked()
        ]
        return models or ["sepaelv2"]  # nunca devolve lista vazia

    def get_ff_det_size(self) -> int:
        return int(self._ff_det_size_spin.value()) if self._ff_det_size_spin else 320

    def get_ff_extended(self) -> bool:
        return bool(self._ff_extended_chk.isChecked()) if self._ff_extended_chk else True

    def get_serfiq_t(self) -> int:
        return int(self._serfiq_t_spin.value()) if self._serfiq_t_spin else 10

    def get_serfiq_alpha(self) -> float:
        return float(self._serfiq_alpha_spin.value()) if self._serfiq_alpha_spin else 1.5
