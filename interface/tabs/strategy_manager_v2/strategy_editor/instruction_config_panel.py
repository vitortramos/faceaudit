# interface/tabs/strategy_manager_v2/strategy_editor/instruction_config_panel.py

from __future__ import annotations

from typing import Any, Dict, List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

_MAX_FIELD_WIDTH = 340


class InstructionConfigPanel(QtWidgets.QWidget):
    """
    Painel de configuração da instrução selecionada na lista.

    Renderiza dinamicamente os campos de configuração da instrução:
    - name, description
    - input_resource_ids (dropdowns com recursos disponíveis no contexto)
    - output_resource_ids (campos de texto livres — o usuário nomeia os outputs)
    - parameters (campos genéricos chave/valor para a instrução específica)

    Sinais:
        config_changed(StrategyInstructionDefinition) — emitido quando o usuário
            confirma alterações no formulário
    """

    config_changed = QtCore.pyqtSignal(object)
    validate_requested = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instruction: Optional[StrategyInstructionDefinition] = None
        self._available_resources: List[StrategyResource] = []
        self._init_ui()
        self.show_empty()

    def _init_ui(self) -> None:
        self._root_layout = QtWidgets.QVBoxLayout(self)
        self._root_layout.setContentsMargins(8, 8, 8, 8)
        self._root_layout.setSpacing(8)

        # Stack: empty ↔ form
        self.stack = QtWidgets.QStackedWidget()
        self._root_layout.addWidget(self.stack)

        # Página 0 — vazia
        empty_page = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty_page)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl = QtWidgets.QLabel("Selecione uma instrução para configurar.")
        lbl.setStyleSheet("color: #888; font-size: 13px;")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(lbl)
        self.stack.addWidget(empty_page)   # index 0

        # Página 1 — formulário
        self.form_page = QtWidgets.QWidget()
        form_layout = QtWidgets.QVBoxLayout(self.form_page)
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setSpacing(6)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.form_container = QtWidgets.QWidget()
        self.form_container.setMaximumWidth(500)
        self.form_inner = QtWidgets.QFormLayout(self.form_container)
        self.form_inner.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.form_inner.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.form_inner.setSpacing(6)
        scroll.setWidget(self.form_container)
        form_layout.addWidget(scroll, 1)

        # Botões Aplicar + Validar
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_apply = QtWidgets.QPushButton("Aplicar alterações")
        self.btn_apply.setMaximumWidth(200)
        self.btn_apply.setStyleSheet("""
            QPushButton { border: 1px solid #2e7d32; border-radius: 4px;
                          background: #e8f5e9; color: #2e7d32;
                          padding: 4px 12px; font-weight: bold; }
            QPushButton:hover { background: #c8e6c9; }
        """)
        self.btn_apply.clicked.connect(self._on_apply)
        btn_row.addWidget(self.btn_apply)

        self.btn_validate = QtWidgets.QPushButton("Validar")
        self.btn_validate.setMaximumWidth(200)
        self.btn_validate.setStyleSheet("""
            QPushButton { border: 1px solid #1565c0; border-radius: 4px;
                          background: #e3f2fd; color: #1565c0;
                          padding: 4px 12px; font-weight: bold; }
            QPushButton:hover { background: #bbdefb; }
        """)
        self.btn_validate.clicked.connect(self._on_validate)
        btn_row.addWidget(self.btn_validate)

        form_layout.addLayout(btn_row)

        self.stack.addWidget(self.form_page)   # index 1

    # ── Public API ────────────────────────────────────────────────────────────

    def set_available_resources(self, resources: List[StrategyResource]) -> None:
        """Define os recursos disponíveis para popular os dropdowns de input.
        Se já há uma instrução carregada, reconstrói o formulário para refletir os novos recursos."""
        self._available_resources = list(resources)
        if self._instruction is not None:
            self._rebuild_form()

    def load_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """Carrega a instrução no formulário."""
        self._instruction = instruction
        self._rebuild_form()
        self.stack.setCurrentIndex(1)

    def show_empty(self) -> None:
        self._instruction = None
        self.stack.setCurrentIndex(0)

    # ── Form building ─────────────────────────────────────────────────────────

    def _rebuild_form(self) -> None:
        instr = self._instruction
        if not instr:
            return

        # Bloqueia sinais dos combos antigos para evitar efeitos colaterais durante remoção
        for combo in getattr(self, '_choice_field_widgets', {}).values():
            combo.blockSignals(True)

        # Limpa o formulário anterior
        while self.form_inner.rowCount():
            self.form_inner.removeRow(0)

        # Header compacto: tipo + ID na mesma linha
        header = QtWidgets.QLabel(
            f"<span style='font-size:11px; color:#2c3e50;'>"
            f"<b>{instr.instruction_type.value}</b>"
            f"<span style='color:#888;'>  {instr.instruction_id}</span></span>"
        )
        self.form_inner.addRow(header)

        # Nome
        self._field_name = QtWidgets.QLineEdit(instr.name)
        self._field_name.setMaximumWidth(_MAX_FIELD_WIDTH)
        self.form_inner.addRow("Nome:", self._field_name)

        # Descrição
        self._field_desc = QtWidgets.QPlainTextEdit(instr.description or "")
        self._field_desc.setMaximumHeight(50)
        self._field_desc.setMaximumWidth(_MAX_FIELD_WIDTH)
        self.form_inner.addRow("Descrição:", self._field_desc)

        self.form_inner.addRow(self._make_separator())

        # Campos de escolha (enums) — ex: pipeline_mode LINEAR/DUAL
        self._choice_field_widgets: Dict[str, QtWidgets.QComboBox] = {}
        self._choice_specs: Dict[str, object] = {}  # field_name → ChoiceFieldSpec
        choice_specs = instr.get_choice_field_specs()
        for cspec in choice_specs:
            combo = QtWidgets.QComboBox()
            combo.setMaximumWidth(_MAX_FIELD_WIDTH)
            combo.addItems(cspec.choices)

            # Resolve valor atual do campo (pode ser enum ou string)
            raw_val = getattr(instr, cspec.field_name, None)
            if raw_val is not None:
                # Tenta .name (enum), senão str() com normalização
                if hasattr(raw_val, "name"):
                    current_str = raw_val.name      # ex: PipelineMode.DUAL → "DUAL"
                else:
                    current_str = str(raw_val)
                    current_str = current_str.split(".")[-1] if "." in current_str else current_str

                # Comparação case-insensitive
                choices_upper = {c.upper(): c for c in cspec.choices}
                match = choices_upper.get(current_str.upper())
                if match:
                    combo.setCurrentText(match)

            self._choice_field_widgets[cspec.field_name] = combo
            self._choice_specs[cspec.field_name] = cspec
            self.form_inner.addRow(f"{cspec.label}:", combo)
            # Ao mudar o modo, aplica imediatamente e reconstrói o formulário
            combo.currentTextChanged.connect(
                lambda _text, fn=cspec.field_name: self._on_choice_changed(fn)
            )

        if choice_specs:
            self.form_inner.addRow(self._make_separator())

        # Campos de recurso tipados via get_resource_field_specs()
        specs = instr.get_resource_field_specs()
        self._resource_field_widgets: Dict[str, QtWidgets.QWidget] = {}

        if specs:
            for spec in specs:
                if spec.is_output:
                    widget = QtWidgets.QLineEdit(getattr(instr, spec.field_name, "") or "")
                    widget.setPlaceholderText("nome do recurso de saída")
                    widget.setMaximumWidth(_MAX_FIELD_WIDTH)
                else:
                    widget = QtWidgets.QComboBox()
                    widget.setMaximumWidth(_MAX_FIELD_WIDTH)
                    widget.addItem("(nenhum)", userData=None)
                    for res in self._available_resources:
                        if res.resource_type in spec.accepted_types:
                            display = res.name if res.name else res.resource_id
                            widget.addItem(display, userData=res.resource_id)
                            widget.setItemData(
                                widget.count() - 1,
                                f"ID: {res.resource_id}\nTipo: {res.resource_type.value}",
                                QtCore.Qt.ItemDataRole.ToolTipRole,
                            )
                    current_val = getattr(instr, spec.field_name, None)
                    if current_val:
                        for idx in range(widget.count()):
                            if widget.itemData(idx) == current_val:
                                widget.setCurrentIndex(idx)
                                break

                self._resource_field_widgets[spec.field_name] = widget
                self.form_inner.addRow(f"{spec.label}:", widget)

            self.form_inner.addRow(self._make_separator())
        else:
            # Fallback genérico para instruções sem specs
            resource_ids = [r.resource_id for r in self._available_resources]
            self._input_combos: List[QtWidgets.QComboBox] = []
            for i, inp_id in enumerate(instr.input_resource_ids):
                combo = QtWidgets.QComboBox()
                combo.addItems(["(nenhum)"] + resource_ids)
                if inp_id in resource_ids:
                    combo.setCurrentText(inp_id)
                self._input_combos.append(combo)
                self.form_inner.addRow(f"Input {i+1}:", combo)

            self._output_fields: List[QtWidgets.QLineEdit] = []
            for i, out_id in enumerate(instr.output_resource_ids):
                field = QtWidgets.QLineEdit(out_id)
                field.setPlaceholderText("nome do recurso de saída")
                self._output_fields.append(field)
                self.form_inner.addRow(f"Output {i+1}:", field)

            if instr.input_resource_ids or instr.output_resource_ids:
                self.form_inner.addRow(self._make_separator())

        # Parâmetros
        self._param_fields: Dict[str, QtWidgets.QWidget] = {}
        for key, val in instr.parameters.items():
            if isinstance(val, bool):
                widget = QtWidgets.QCheckBox()
                widget.setChecked(val)
            elif isinstance(val, (int, float)):
                widget = QtWidgets.QDoubleSpinBox()
                widget.setRange(-1e9, 1e9)
                widget.setValue(float(val))
                widget.setMaximumWidth(160)
            else:
                widget = QtWidgets.QLineEdit(str(val) if val is not None else "")
                widget.setMaximumWidth(_MAX_FIELD_WIDTH)
            self._param_fields[key] = widget
            self.form_inner.addRow(f"{key}:", widget)

    def _on_apply(self) -> None:
        if not self._instruction:
            return

        # Atualiza campos simples
        self._instruction.name = self._field_name.text().strip() or self._instruction.name
        self._instruction.description = self._field_desc.toPlainText().strip() or None

        # IMPORTANTE: atualizar recurso fields ANTES dos choice fields, pois os
        # specs dependem do estado atual do choice (ex: pipeline_mode DUAL → 4 campos).
        # Se atualizássemos o choice primeiro, get_resource_field_specs() poderia
        # retornar specs diferentes dos widgets que existem no formulário.

        # Atualiza campos de recurso tipados (usando specs do estado ATUAL)
        specs = self._instruction.get_resource_field_specs()
        if specs and hasattr(self, '_resource_field_widgets'):
            for spec in specs:
                widget = self._resource_field_widgets.get(spec.field_name)
                if widget is None:
                    continue
                if spec.is_output:
                    val = widget.text().strip() or None
                else:
                    val = widget.currentData()
                setattr(self._instruction, spec.field_name, val)

            if hasattr(self._instruction, '_sync_declared_io'):
                self._instruction._sync_declared_io()
        else:
            # Fallback genérico
            if hasattr(self, '_input_combos'):
                for i, combo in enumerate(self._input_combos):
                    val = combo.currentText()
                    if val != "(nenhum)" and i < len(self._instruction.input_resource_ids):
                        self._instruction.input_resource_ids[i] = val

            if hasattr(self, '_output_fields'):
                for i, field in enumerate(self._output_fields):
                    val = field.text().strip()
                    if val and i < len(self._instruction.output_resource_ids):
                        self._instruction.output_resource_ids[i] = val

        # Atualiza campos de escolha (enums) — DEPOIS dos resource fields
        for field_name, combo in getattr(self, '_choice_field_widgets', {}).items():
            new_val = combo.currentText()
            new_val = self._convert_choice_value(field_name, new_val)
            setattr(self._instruction, field_name, new_val)

        # Atualiza parâmetros
        for key, widget in self._param_fields.items():
            if isinstance(widget, QtWidgets.QCheckBox):
                self._instruction.parameters[key] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QDoubleSpinBox):
                self._instruction.parameters[key] = widget.value()
            elif isinstance(widget, QtWidgets.QLineEdit):
                self._instruction.parameters[key] = widget.text().strip()

        # Sincroniza io novamente após choice fields (modo pode ter mudado)
        if hasattr(self._instruction, '_sync_declared_io'):
            self._instruction._sync_declared_io()

        self.config_changed.emit(self._instruction)

    def _rebuild_and_notify(self) -> None:
        """Reconstrói o formulário e emite config_changed. Chamado via QTimer para evitar crash."""
        if not self._instruction:
            return
        self._rebuild_form()
        self.config_changed.emit(self._instruction)

    def _on_choice_changed(self, field_name: str) -> None:
        """Aplica o valor do campo de escolha e agenda reconstrução do formulário."""
        if not self._instruction:
            return
        combo = self._choice_field_widgets.get(field_name)
        if combo is None:
            return
        new_val = combo.currentText()
        # Converte para o tipo correto usando enum_type do ChoiceFieldSpec
        new_val = self._convert_choice_value(field_name, new_val)
        setattr(self._instruction, field_name, new_val)
        if hasattr(self._instruction, "_sync_declared_io"):
            self._instruction._sync_declared_io()
        # Adia o rebuild para depois do processamento do sinal atual terminar,
        # evitando crash por destruição do widget que originou o sinal.
        QtCore.QTimer.singleShot(0, self._rebuild_and_notify)

    def _convert_choice_value(self, field_name: str, text: str) -> Any:
        """
        Converte o texto do combo para o enum correto usando o enum_type
        registrado no ChoiceFieldSpec. Garante que o campo NUNCA fique como
        string se um enum_type foi declarado.
        """
        cspec = getattr(self, '_choice_specs', {}).get(field_name)
        if cspec is not None and cspec.enum_type is not None:
            enum_cls = cspec.enum_type
            if hasattr(enum_cls, "from_str"):
                return enum_cls.from_str(text)
        return text

    def _on_validate(self) -> None:
        if self._instruction:
            self._on_apply()
            self.validate_requested.emit(self._instruction)

    @staticmethod
    def _make_separator() -> QtWidgets.QFrame:
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #e0e0e0; margin: 4px 0;")
        return sep
