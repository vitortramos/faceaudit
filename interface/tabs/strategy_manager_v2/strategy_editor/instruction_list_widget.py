# interface/tabs/strategy_manager_v2/strategy_editor/instruction_list_widget.py

from __future__ import annotations

from typing import List, Optional, Set

from PyQt6 import QtWidgets, QtCore, QtGui

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition


_INSTRUCTION_ICONS = {
    StrategyInstructionType.PIPELINE_TRAINING: "🔧",
    StrategyInstructionType.PIPELINE_APPLICATION: "⚙",
    StrategyInstructionType.DATASET_SPLIT: "✂",
    StrategyInstructionType.DATASET_MERGE: "🔀",
    StrategyInstructionType.PAIR_MINING: "⛏",
    StrategyInstructionType.SIMILARITY_MODEL_TRAINING: "📐",
    StrategyInstructionType.SIMILARITY_SCORING: "📊",
    StrategyInstructionType.CALIBRATION_MODEL_TRAINING: "🎯",
}


class InstructionListWidget(QtWidgets.QWidget):
    """
    Lista reordenável de instruções da estratégia em edição.

    Sinais:
        instruction_selected(StrategyInstructionDefinition)
        add_instruction_requested(StrategyInstructionType)
        remove_instruction_requested(StrategyInstructionDefinition)
        order_changed(List[StrategyInstructionDefinition])  — emitido após drag-drop
    """

    instruction_selected = QtCore.pyqtSignal(object)
    add_instruction_requested = QtCore.pyqtSignal(object)   # StrategyInstructionType
    remove_instruction_requested = QtCore.pyqtSignal(object)
    order_changed = QtCore.pyqtSignal(list)
    run_instruction_requested = QtCore.pyqtSignal(object)   # StrategyInstructionDefinition

    def __init__(self, parent=None):
        super().__init__(parent)
        self._instructions: List[StrategyInstructionDefinition] = []
        self._runnable_ids: Set[str] = set()
        self._executed_ids: Set[str] = set()
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QtWidgets.QLabel("Instruções")
        header.setStyleSheet("font-weight: bold; font-size: 11px; color: #444;")
        layout.addWidget(header)

        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.list_widget.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.list_widget.model().rowsMoved.connect(self._on_rows_moved)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.setMinimumWidth(200)
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

        self.btn_run = QtWidgets.QPushButton("▶ Run")
        self.btn_run.setToolTip("Executar instrução selecionada")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._on_run)

        for btn in (self.btn_add, self.btn_remove, self.btn_run):
            btn.setStyleSheet("""
                QPushButton { border: 1px solid #ccc; border-radius: 3px;
                              background: white; padding: 3px 8px; font-size: 11px; }
                QPushButton:hover { background: #e8f0fe; }
                QPushButton:disabled { color: #aaa; }
            """)
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
            iid = instr.instruction_id
            icon = _INSTRUCTION_ICONS.get(instr.instruction_type, "▪")
            if iid in executed_ids:
                prefix = "✅"
                color = QtGui.QColor("#2e7d32")
            elif iid in runnable_ids:
                prefix = icon
                color = QtGui.QColor("#1a1a1a")
            else:
                prefix = "🔒"
                color = QtGui.QColor("#999")
            item.setText(f"{prefix}  {instr.name}  [{instr.instruction_type.value}]")
            item.setForeground(color)
        self._update_buttons()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_list_item(self, instr: StrategyInstructionDefinition) -> None:
        icon = _INSTRUCTION_ICONS.get(instr.instruction_type, "▪")
        text = f"{icon}  {instr.name}  [{instr.instruction_type.value}]"
        item = QtWidgets.QListWidgetItem(text)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, instr)
        item.setToolTip(instr.description or "")
        self.list_widget.addItem(item)

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
        if (
            instr
            and instr.instruction_id in self._runnable_ids
            and instr.instruction_id not in self._executed_ids
        ):
            self.btn_run.setEnabled(True)
        else:
            self.btn_run.setEnabled(False)

    def _show_add_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        for instr_type in StrategyInstructionType:
            icon = _INSTRUCTION_ICONS.get(instr_type, "▪")
            action = menu.addAction(f"{icon}  {instr_type.value}")
            action.setData(instr_type)
        chosen = menu.exec(self.btn_add.mapToGlobal(
            self.btn_add.rect().bottomLeft()
        ))
        if chosen:
            self.add_instruction_requested.emit(chosen.data())
