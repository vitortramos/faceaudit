"""StepConfigPanel — formulário do passo selecionado no editor.

Montado a partir da declaração de parâmetros do config
(`config.param_specs()`, ver `faceaudit/models/image_processing/
step_param_spec.py`). Não há código por tipo de passo aqui: passo novo
declara seus campos no config e o formulário aparece.

Cada edição grava no config (mutação in-place — o mesmo objeto que está
na lista do pipeline), reaplica as correções de faixa do config, ressincroniza
os campos (um campo pode mudar as opções ou a visibilidade de outro) e
emite `config_changed`.

Campos do tipo ACTION (ex.: desenhar a máscara de oclusão) emitem
`action_requested(nome, config)`; o controller executa e chama
`refresh()`.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from PyQt6 import QtCore, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)
from faceaudit.models.image_processing.step_param_spec import (
    StepParamKind,
    StepParamSpec,
)


class StepConfigPanel(QtWidgets.QWidget):
    """Formulário do passo selecionado."""

    config_changed = QtCore.pyqtSignal(object)  # ImageProcessingStepConfig
    action_requested = QtCore.pyqtSignal(str, object)  # nome, config

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._config: Optional[ImageProcessingStepConfig] = None
        self._rows: List[Tuple[StepParamSpec, QtWidgets.QLabel, QtWidgets.QWidget]] = []
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._title = QtWidgets.QLabel("<i>Selecione um passo à esquerda</i>", parent=self)
        self._title.setWordWrap(True)
        apply_themed_stylesheet(
            self._title,
            lambda t: "color: " + t("TEXT_MUTED") + "; padding: 4px; font-size: 11px;",
        )
        layout.addWidget(self._title)

        self._form_host = QtWidgets.QWidget(parent=self)
        self._form_layout = QtWidgets.QFormLayout(self._form_host)
        layout.addWidget(self._form_host)
        layout.addStretch()

    # ── API ────────────────────────────────────────────────────────────────

    def show_config(self, config: Optional[ImageProcessingStepConfig]):
        """Renderiza o formulário de `config`. None limpa o painel."""
        self._config = config
        self._clear_form()
        if config is None:
            self._title.setText("<i>Selecione um passo à esquerda</i>")
            return

        self._title.setText(f"<b>{config.step_type.label_pt}</b>")
        specs = config.param_specs()
        if not specs:
            vazio = QtWidgets.QLabel("<i>Este passo não tem parâmetros.</i>", parent=self._form_host)
            self._form_layout.addRow(vazio)
            return

        self._syncing = True
        for spec in specs:
            rotulo = QtWidgets.QLabel(
                "" if spec.kind == StepParamKind.ACTION and not spec.label
                else f"{spec.label}:",
                parent=self._form_host,
            )
            widget = self._build_widget(spec, config)
            if spec.help:
                rotulo.setToolTip(spec.help)
                widget.setToolTip(spec.help)
            self._form_layout.addRow(rotulo, widget)
            self._rows.append((spec, rotulo, widget))
        self._syncing = False
        self.refresh()

    def refresh(self):
        """Ressincroniza valores, opções, visibilidade e rótulos de botão
        com o config (após edição de outro campo ou ação externa)."""
        if self._config is None:
            return
        self._syncing = True
        try:
            for spec, rotulo, widget in self._rows:
                self._sync_widget(spec, widget, self._config)
                visivel = spec.is_visible(self._config)
                rotulo.setVisible(visivel)
                widget.setVisible(visivel)
        finally:
            self._syncing = False

    # ── Construção ─────────────────────────────────────────────────────────

    def _clear_form(self):
        self._rows = []
        # `removeRow` é a API correta pra limpar QFormLayout — `takeAt`
        # não respeita rows e deixa widgets pendurados.
        while self._form_layout.rowCount() > 0:
            self._form_layout.removeRow(0)

    def _build_widget(self, spec: StepParamSpec, config: Any) -> QtWidgets.QWidget:
        host = self._form_host
        if spec.kind == StepParamKind.INT:
            w = QtWidgets.QSpinBox(parent=host)
            w.setRange(
                int(spec.minimum if spec.minimum is not None else -2**31 + 1),
                int(spec.maximum if spec.maximum is not None else 2**31 - 1),
            )
            w.setSingleStep(int(spec.step or 1))
            w.setSuffix(spec.suffix)
            w.valueChanged.connect(lambda v, s=spec: self._on_value(s, v))
            return w
        if spec.kind == StepParamKind.FLOAT:
            w = QtWidgets.QDoubleSpinBox(parent=host)
            w.setDecimals(int(spec.decimals))
            w.setRange(
                float(spec.minimum if spec.minimum is not None else -1e9),
                float(spec.maximum if spec.maximum is not None else 1e9),
            )
            w.setSingleStep(float(spec.step or 0.1))
            w.setSuffix(spec.suffix)
            w.valueChanged.connect(lambda v, s=spec: self._on_value(s, v))
            return w
        if spec.kind == StepParamKind.CHOICE:
            w = QtWidgets.QComboBox(parent=host)
            w.currentIndexChanged.connect(
                lambda _i, s=spec, cb=w: self._on_value(s, cb.currentData())
            )
            return w
        if spec.kind == StepParamKind.BOOL:
            w = QtWidgets.QCheckBox(parent=host)
            w.toggled.connect(lambda v, s=spec: self._on_value(s, bool(v)))
            return w
        if spec.kind == StepParamKind.ACTION:
            w = QtWidgets.QPushButton(parent=host)
            w.clicked.connect(
                lambda _c=False, s=spec: self.action_requested.emit(s.name, self._config)
            )
            return w
        raise ValueError(f"Tipo de campo desconhecido: {spec.kind}")

    def _sync_widget(self, spec: StepParamSpec, widget: QtWidgets.QWidget, config: Any):
        if spec.kind == StepParamKind.ACTION:
            if spec.action_label is not None:
                widget.setText(spec.action_label(config))
            else:
                widget.setText(spec.label)
            return
        valor = getattr(config, spec.name, None)
        widget.blockSignals(True)
        try:
            if spec.kind == StepParamKind.INT:
                widget.setValue(int(valor))
            elif spec.kind == StepParamKind.FLOAT:
                widget.setValue(float(valor))
            elif spec.kind == StepParamKind.BOOL:
                widget.setChecked(bool(valor))
            elif spec.kind == StepParamKind.CHOICE:
                opcoes = list(spec.choices_for(config))
                widget.clear()
                for v, rotulo in opcoes:
                    widget.addItem(rotulo, userData=v)
                valores = [v for v, _ in opcoes]
                if valor in valores:
                    widget.setCurrentIndex(valores.index(valor))
                elif opcoes:
                    # As opções mudaram e o valor atual saiu delas: o
                    # config passa a refletir o que o campo mostra.
                    widget.setCurrentIndex(0)
                    setattr(config, spec.name, valores[0])
        finally:
            widget.blockSignals(False)

    # ── Edição ─────────────────────────────────────────────────────────────

    def _on_value(self, spec: StepParamSpec, valor: Any):
        if self._syncing or self._config is None:
            return
        setattr(self._config, spec.name, spec.clamp(valor))
        self._config.normalize()
        self.refresh()
        self.config_changed.emit(self._config)
