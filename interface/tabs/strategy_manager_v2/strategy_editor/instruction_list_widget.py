# interface/tabs/strategy_manager_v2/strategy_editor/instruction_list_widget.py

from __future__ import annotations

from typing import List, Optional, Set

from PyQt6 import QtWidgets, QtCore, QtGui

from interface.theme import apply_themed_stylesheet, current_token_fn

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition

from .section_header import ColumnHeader


_INSTRUCTION_ICONS = {
    StrategyInstructionType.PIPELINE_TRAINING: "🔧",
    StrategyInstructionType.PIPELINE_APPLICATION: "⚙",
    StrategyInstructionType.DATASET_SPLIT: "✂",
    StrategyInstructionType.PAIRED_DATASET_SPLIT: "🔗",
    StrategyInstructionType.DATASET_MERGE: "🔀",
    StrategyInstructionType.PAIR_MINING: "⛏",
    StrategyInstructionType.SIMILARITY_MODEL_TRAINING: "📐",
    StrategyInstructionType.SIMILARITY_SCORING: "📊",
    StrategyInstructionType.CALIBRATION_MODEL_TRAINING: "🎯",
    StrategyInstructionType.SCORE_CALIBRATION: "🎚️",
    StrategyInstructionType.SCORE_VECTORIZATION: "🧬",
    StrategyInstructionType.SCORE_INTERACTION_FEATURE: "🪢",
    StrategyInstructionType.LR_ARCHITECTURE_VECTORIZATION: "🧩",
    StrategyInstructionType.STRATEGY_EVALUATION: "📈",
    StrategyInstructionType.CASEWORK_VISUALIZATION: "🎯",
}


# Instruções de TREINAMENTO — exclusivas da fase Preparação. Não devem
# aparecer no menu "Adicionar" das fases Validação ou Análise de Caso.
_TRAINING_ONLY_TYPES = {
    StrategyInstructionType.PIPELINE_TRAINING,
    StrategyInstructionType.SIMILARITY_MODEL_TRAINING,
    StrategyInstructionType.CALIBRATION_MODEL_TRAINING,
    StrategyInstructionType.TYPICALITY_MODEL_TRAINING,
}

# Instruções exclusivas da fase Validação. Não devem aparecer em
# Preparação nem em Análise de Caso.
_VALIDATION_ONLY_TYPES = {
    StrategyInstructionType.STRATEGY_EVALUATION,
}

# Instruções exclusivas da fase Análise de Caso. Não devem aparecer em
# Preparação nem em Validação — produzem ou consomem datasets de casework
# que não fazem sentido fora desse contexto.
_CASEWORK_ONLY_TYPES = {
    StrategyInstructionType.CASEWORK_VISUALIZATION,
}


def _allowed_types_for_phase(phase: Optional[str]) -> List[StrategyInstructionType]:
    """
    Retorna a lista de tipos de instrução que o menu "Adicionar" deve listar
    para a fase informada.

    - preparation: tudo MENOS exclusivas de Validação E casework.
    - validation:  tudo MENOS exclusivas de Preparação (treinamento) E casework.
    - casework:    tudo MENOS treinamento E exclusivas de Validação. INCLUI
                   instruções _CASEWORK_ONLY_TYPES.
    - None:        comportamento legado — lista todos os tipos.
    """
    if phase is None:
        return list(StrategyInstructionType)
    if phase == "preparation":
        return [
            t for t in StrategyInstructionType
            if t not in _VALIDATION_ONLY_TYPES and t not in _CASEWORK_ONLY_TYPES
        ]
    if phase == "validation":
        return [
            t for t in StrategyInstructionType
            if t not in _TRAINING_ONLY_TYPES and t not in _CASEWORK_ONLY_TYPES
        ]
    # casework: sem treinamento e sem validação; inclui CASEWORK_ONLY.
    return [
        t for t in StrategyInstructionType
        if t not in _TRAINING_ONLY_TYPES and t not in _VALIDATION_ONLY_TYPES
    ]


class _InstructionItemWidget(QtWidgets.QWidget):
    """
    Widget visual de um item da lista de instruções.

    Layout vertical com:
    - linha principal: ícone de estado/tipo + nome + [tipo]
    - sub-linha (opcional): indicador de outputs declarados vs nomeados
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # WA_DontShowOnScreen impede o Qt de exibir este widget como
        # janela top-level enquanto está sendo configurado. Sem esse
        # cinto-de-segurança, mesmo quando passamos parent=list_widget
        # no construtor, em algumas plataformas (Windows especialmente)
        # o widget pode flashear como janela flutuante durante o ciclo
        # entre criação dos children e o setItemWidget que finaliza o
        # parent. A flag é automaticamente "desligada" pelo Qt quando o
        # widget é efetivamente mostrado dentro de um parent visível.
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)

        self._lbl_main = QtWidgets.QLabel()
        self._lbl_main.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._lbl_main)

        self._lbl_outputs = QtWidgets.QLabel()
        apply_themed_stylesheet(
            self._lbl_outputs,
            lambda t: (
                "font-size: 9px; color: " + t("TEXT_MUTED")
                + "; font-style: italic; padding-left: 18px;"
            ),
        )
        self._lbl_outputs.setVisible(False)
        layout.addWidget(self._lbl_outputs)

    def showEvent(self, event):
        # Remove a flag uma vez que o widget está visível dentro do
        # parent legítimo — comportamento normal a partir daqui.
        if self.testAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen):
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DontShowOnScreen, False)
        super().showEvent(event)

    def set_main(self, text: str, color: QtGui.QColor) -> None:
        self._lbl_main.setText(text)
        self._lbl_main.setStyleSheet(
            f"font-size: 11px; color: {color.name()};"
        )

    def set_outputs_indicator(self, total: int, named: int) -> None:
        if total <= 0:
            self._lbl_outputs.setVisible(False)
            self._lbl_outputs.setText("")
            return

        self._lbl_outputs.setVisible(True)
        if named == total:
            self._lbl_outputs.setText(f"→ produz {total} recurso(s)")
        elif named == 0:
            self._lbl_outputs.setText(
                f"→ produzirá {total} recurso(s) (nenhum nomeado)"
            )
        else:
            self._lbl_outputs.setText(
                f"→ produzirá {total} recurso(s) ({named} nomeado(s))"
            )


class InstructionListWidget(QtWidgets.QWidget):
    """
    Lista reordenável de instruções da estratégia em edição.

    Sinais:
        instruction_selected(StrategyInstructionDefinition)
        add_instruction_requested(StrategyInstructionType)
        remove_instruction_requested(StrategyInstructionDefinition)
        order_changed(List[StrategyInstructionDefinition])  — emitido após drag-drop
        run_instruction_requested(StrategyInstructionDefinition)
    """

    instruction_selected = QtCore.pyqtSignal(object)
    add_instruction_requested = QtCore.pyqtSignal(object)   # StrategyInstructionType
    remove_instruction_requested = QtCore.pyqtSignal(object)
    order_changed = QtCore.pyqtSignal(list)
    run_instruction_requested = QtCore.pyqtSignal(object)   # StrategyInstructionDefinition

    def __init__(self, parent=None, phase: Optional[str] = None):
        super().__init__(parent)
        self._instructions: List[StrategyInstructionDefinition] = []
        self._runnable_ids: Set[str] = set()
        self._executed_ids: Set[str] = set()
        # Fase à qual esta lista pertence ("preparation"/"validation"/"casework").
        # Usada para filtrar quais tipos de instrução aparecem no menu Adicionar.
        self._phase: Optional[str] = phase
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Cabeçalho principal da coluna — padrão unificado com as colunas
        # de Configurações e Saídas (mesmo widget ColumnHeader).
        layout.addWidget(ColumnHeader("Instruções"))

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setMinimumWidth(160)
        # Espaçamento confortável para itens com sub-linha.
        self.list_widget.setSpacing(1)
        # Cor de seleção clara — o texto do item pode ser verde (executado),
        # cinza (locked) ou preto (runnable). O fundo azul escuro padrão do
        # Qt deixa o texto verde ilegível; trocamos por um azul bem suave
        # com borda lateral marcadora para indicar seleção sem reduzir
        # contraste com qualquer cor de texto.
        apply_themed_stylesheet(
            self.list_widget,
            lambda t: (
                "QListWidget::item:selected {"
                " background: " + t("PRIMARY_BG") + "; color: " + t("TEXT") + ";"
                " border-left: 3px solid " + t("PRIMARY") + ";"
                "}"
                "QListWidget::item:selected:!active {"
                " background: " + t("PRIMARY_BG") + "; color: " + t("TEXT") + ";"
                "}"
                "QListWidget::item:hover:!selected {"
                " background: " + t("SURFACE_ALT") + ";"
                "}"
            ),
        )
        layout.addWidget(self.list_widget, 1)

        # Botões
        btn_row = QtWidgets.QHBoxLayout()

        self.btn_add = QtWidgets.QPushButton("➕ Adicionar")
        self.btn_add.setToolTip("Adicionar nova instrução à estratégia")
        self.btn_add.clicked.connect(self._show_add_menu)

        self.btn_remove = QtWidgets.QPushButton("🗑 Remover")
        self.btn_remove.setToolTip("Remover instrução selecionada")
        self.btn_remove.setEnabled(False)
        self.btn_remove.clicked.connect(self._on_remove)

        self.btn_run = QtWidgets.QPushButton("▶ Executar")
        self.btn_run.setToolTip("Executar instrução selecionada")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)

        def _list_btn_style(t):
            return (
                "QPushButton { border: 1px solid " + t("BORDER") + "; border-radius: 3px;"
                " background: " + t("SURFACE") + "; padding: 3px 8px; font-size: 11px; }"
                "QPushButton:hover { background: " + t("PRIMARY_BG") + "; }"
                "QPushButton:disabled { color: " + t("TEXT_MUTED") + "; }"
            )
        for btn in (self.btn_add, self.btn_remove, self.btn_run):
            apply_themed_stylesheet(btn, _list_btn_style)
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_strategy(self, instructions: List[StrategyInstructionDefinition]) -> None:
        """Carrega as instruções de uma estratégia na lista."""
        self._instructions = list(instructions)
        self.list_widget.clear()
        for instr in self._instructions:
            self._add_list_item(instr)
        self._update_buttons()

    def clear(self) -> None:
        self._instructions.clear()
        self.list_widget.clear()
        self._update_buttons()

    def current_order(self) -> List[StrategyInstructionDefinition]:
        """Retorna as instruções na ordem atual da lista (após reordenações)."""
        result = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            instr = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if instr:
                result.append(instr)
        return result

    def selected_instruction(self) -> Optional[StrategyInstructionDefinition]:
        item = self.list_widget.currentItem()
        return item.data(QtCore.Qt.ItemDataRole.UserRole) if item else None

    def update_states(self, runnable_ids: Set[str], executed_ids: Set[str]) -> None:
        """Atualiza visual de cada item: bloqueado / executável / executado."""
        self._runnable_ids = runnable_ids
        self._executed_ids = executed_ids
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            instr = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if not instr:
                continue
            widget = self.list_widget.itemWidget(item)
            if widget is None:
                continue

            t = current_token_fn()
            iid = instr.instruction_id
            icon = _INSTRUCTION_ICONS.get(instr.instruction_type, "▪")
            if iid in executed_ids:
                prefix = "✅"
                color = QtGui.QColor(t("SUCCESS"))
            elif iid in runnable_ids:
                prefix = icon
                color = QtGui.QColor(t("TEXT"))
            else:
                prefix = "🔒"
                color = QtGui.QColor(t("TEXT_MUTED"))

            text = f"{prefix}  {instr.name}  [{instr.instruction_type.label_pt}]"
            widget.set_main(text, color)
            # Recalcula indicador de outputs — a contagem nomeada pode ter
            # mudado entre chamadas (ex: usuário digitou nome de output e o
            # estado foi atualizado por refresh).
            total, named = self._count_outputs(instr)
            widget.set_outputs_indicator(total, named)
        self._update_buttons()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_list_item(self, instr: StrategyInstructionDefinition) -> None:
        icon = _INSTRUCTION_ICONS.get(instr.instruction_type, "▪")
        text = f"{icon}  {instr.name}  [{instr.instruction_type.label_pt}]"

        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.ItemDataRole.UserRole, instr)
        item.setToolTip(instr.description or "")
        self.list_widget.addItem(item)

        # IMPORTANTE: parent já no construtor + setItemWidget ANTES de
        # qualquer chamada que possa disparar setVisible(True) num filho.
        # Antes a ordem era:
        #   widget = _InstructionItemWidget()  # parent=None
        #   widget.set_outputs_indicator(...)   # pode chamar setVisible(True)
        #   self.list_widget.setItemWidget(item, widget)  # parenteia agora
        # Com widget parentless, setVisible(True) num filho fazia o Qt
        # exibi-lo brevemente como janela flutuante (bug "janelinha
        # piscando" ao selecionar instrução/estratégia).
        widget = _InstructionItemWidget(parent=self.list_widget)
        widget.set_main(text, QtGui.QColor(current_token_fn()("TEXT")))
        total, named = self._count_outputs(instr)
        widget.set_outputs_indicator(total, named)

        # sizeHint precisa refletir a altura do widget customizado (1 ou 2 linhas).
        item.setSizeHint(widget.sizeHint())
        self.list_widget.setItemWidget(item, widget)

    @staticmethod
    def _count_outputs(instr: StrategyInstructionDefinition) -> tuple:
        """Retorna (total_outputs_declarados, total_nomeados) na instrução."""
        try:
            specs = instr.get_resource_field_specs()
        except Exception:
            return (0, 0)

        total = 0
        named = 0
        for spec in specs:
            if not spec.is_output:
                continue
            total += 1
            value = getattr(instr, spec.field_name, None)
            if value:
                named += 1
        return (total, named)

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        instr = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if instr:
            self.instruction_selected.emit(instr)
        self._update_buttons()

    def _on_remove(self) -> None:
        instr = self.selected_instruction()
        if instr:
            self.remove_instruction_requested.emit(instr)

    def _on_run(self) -> None:
        instr = self.selected_instruction()
        if instr:
            self.run_instruction_requested.emit(instr)

    def _on_rows_moved(self, *_args) -> None:
        self.order_changed.emit(self.current_order())

    def _update_buttons(self) -> None:
        instr = self.selected_instruction()
        self.btn_remove.setEnabled(instr is not None)
        # Botão Executar fica habilitado sempre que houver instrução
        # selecionada — permite re-execução forçada sobrescrevendo o
        # recurso já produzido. O gate de dependência (inputs faltando)
        # é verificado em runtime pelo controller via
        # `can_execute_instruction` antes de disparar o worker.
        self.btn_run.setEnabled(instr is not None)

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        for instr_type in _allowed_types_for_phase(self._phase):
            icon = _INSTRUCTION_ICONS.get(instr_type, "▪")
            action = menu.addAction(f"{icon}  {instr_type.label_pt}")
            action.setData(instr_type)
            action.setToolTip(instr_type.description)
        chosen = menu.exec(self.btn_add.mapToGlobal(
            self.btn_add.rect().bottomLeft()
        ))
        if chosen:
            self.add_instruction_requested.emit(chosen.data())
