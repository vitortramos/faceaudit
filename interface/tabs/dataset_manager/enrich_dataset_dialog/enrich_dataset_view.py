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

    # Lista dinâmica: lê os modelos detectados em
    # `~/.forensicface/models/` a cada open do dialog. Permite que o
    # usuário adicione modelos novos sem alterar código.
    @property
    def _supported_ff_models(self) -> tuple:
        from faceaudit.enums.sepael_model import SepaelModel
        available = SepaelModel.available_on_disk()
        if available:
            return tuple(m.ff_name for m in available)
        # Fallback histórico se nenhum modelo está em disco — mantém UI
        # consistente em primeira execução; o usuário vai falhar na execução
        # se de fato não tiver nada instalado.
        return (SepaelModel.default().ff_name,)

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
        self._ff_is_aligned_chk: QCheckBox | None = None
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

        # PR 18: idempotência. Default desmarcado — pula faces que já têm
        # o enrichment pedido. Marcado = força reprocessamento de tudo
        # (sobrescreve *_info existentes).
        self.chk_force_reprocess = QCheckBox(
            "Reprocessar mesmo as faces já enriquecidas"
        )
        self.chk_force_reprocess.setChecked(False)
        self.chk_force_reprocess.setToolTip(
            "Por padrão, cada estágio (FF/OFIQ/SER-FIQ) processa apenas as "
            "faces que ainda não têm aquele dado preenchido — útil para "
            "completar um dataset parcialmente enriquecido sem desperdiçar "
            "GPU. Marque esta opção para reprocessar todas as faces."
        )
        main_layout.addWidget(self.chk_force_reprocess)

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

        # Modelos (multi-select) — lista dinâmica vinda de
        # `SepaelModel.available_on_disk()`. Label exibido é o
        # `label_pt` do enum (ex: "Sepael V2").
        from faceaudit.enums.sepael_model import SepaelModel
        layout.addWidget(QLabel("<b>Modelos de embedding</b>"))
        models_row = QHBoxLayout()
        for model_name in self._supported_ff_models:
            member = SepaelModel.from_ff_name(model_name)
            label = member.label_pt if member is not None else model_name
            chk = QCheckBox(label)
            self._ff_model_checks[model_name] = chk
            models_row.addWidget(chk)
        # Default selecionado: SepaelModel.default() (sepaelv2 hoje).
        default_name = SepaelModel.default().ff_name
        if default_name in self._ff_model_checks:
            self._ff_model_checks[default_name].setChecked(True)
        else:
            # default não está disponível em disco — marca o primeiro
            # da lista pra evitar UI sem nada selecionado.
            first = next(iter(self._ff_model_checks.values()), None)
            if first is not None:
                first.setChecked(True)
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

        # is_aligned — override. Pré-preenchido pelo controller com o
        # valor atual do dataset; o usuário só toca pra corrigir datasets
        # cuja metadata está errada (ex: parquets legados sem o flag).
        # Sem esse override, FF tenta detect+align num crop 112×112 já
        # alinhado → falha na detecção ou gera embedding lixo
        # silenciosamente.
        self._ff_is_aligned_chk = QCheckBox(
            "Imagens são faces já alinhadas 112×112"
        )
        self._ff_is_aligned_chk.setChecked(False)
        self._ff_is_aligned_chk.setToolTip(
            "Marque se as imagens já são crops alinhados 112×112 (não fotos "
            "originais). FF usa process_face_aligned (pula detect+align, "
            "~50% mais rápido). Pré-preenchido com o valor atual do "
            "dataset — só altere pra corrigir metadata errada."
        )
        layout.addWidget(self._ff_is_aligned_chk)
        aligned_help = QLabel(
            "<i>Aplica-se ao dataset todo. Após enriquecer, o flag fica "
            "persistido na metadata do parquet.</i>"
        )
        aligned_help.setWordWrap(True)
        aligned_help.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(aligned_help)

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

    def get_ff_is_aligned(self) -> bool:
        return bool(self._ff_is_aligned_chk.isChecked()) if self._ff_is_aligned_chk else False

    def set_ff_is_aligned(self, value: bool) -> None:
        if self._ff_is_aligned_chk is not None:
            self._ff_is_aligned_chk.setChecked(bool(value))

    def set_stage_availability(
        self,
        ff_available: bool,
        ofiq_available: bool,
    ) -> None:
        """Desabilita visualmente os checkboxes dos estágios cujo serviço
        externo não pôde ser inicializado. SER-FIQ depende de FF, então
        segue a disponibilidade do FF. Cada checkbox desabilitado ganha
        tooltip explicando o motivo + sai forçadamente desmarcado pra
        que `get_config()` não devolva o estágio como solicitado."""
        ff_tooltip = (
            "ForensicFace não está disponível: nenhum modelo encontrado em "
            "~/.forensicface/models/ ou o serviço falhou ao carregar. "
            "Instale um modelo (sepaelv2/v3/v4/v5) ou troque nas Configurações."
        )
        ofiq_tooltip = (
            "OFIQ não está disponível: executável ou pasta de dados não "
            "encontrados. Verifique os caminhos nas Configurações."
        )

        chk_ff = self.checkboxes.get("ff")
        if chk_ff is not None and not ff_available:
            chk_ff.setChecked(False)
            chk_ff.setEnabled(False)
            chk_ff.setToolTip(ff_tooltip)

        chk_serfiq = self.checkboxes.get("serfiq")
        if chk_serfiq is not None and not ff_available:
            chk_serfiq.setChecked(False)
            chk_serfiq.setEnabled(False)
            chk_serfiq.setToolTip(
                "SER-FIQ depende do ForensicFace, que não está disponível. "
                + ff_tooltip
            )

        chk_ofiq = self.checkboxes.get("ofiq")
        if chk_ofiq is not None and not ofiq_available:
            chk_ofiq.setChecked(False)
            chk_ofiq.setEnabled(False)
            chk_ofiq.setToolTip(ofiq_tooltip)

    def get_serfiq_t(self) -> int:
        return int(self._serfiq_t_spin.value()) if self._serfiq_t_spin else 10

    def get_serfiq_alpha(self) -> float:
        return float(self._serfiq_alpha_spin.value()) if self._serfiq_alpha_spin else 1.5
