# interface/tabs/strategy_manager_v2/strategy_editor/strategy_editor_widget.py

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set, Tuple

from PyQt6 import QtWidgets, QtCore

from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)

from .instruction_list_widget import InstructionListWidget
from .instruction_config_panel import InstructionConfigPanel

_PHASES: List[Tuple[str, str]] = [
    ("preparation", "Preparação"),
    ("validation", "Validação"),
    ("casework", "Análise de Caso"),
]


class StrategyEditorWidget(QtWidgets.QWidget):
    """
    Área principal de edição de uma LRStrategyDefinition.

    QTabWidget com 3 abas (Preparação, Validação, Análise de Caso), cada uma
    com InstructionListWidget + InstructionConfigPanel em splitter horizontal.

    Sinais:
        strategy_modified(LRStrategyDefinition) — qualquer alteração aplicada
        run_instruction_requested(StrategyInstructionDefinition) — execução individual
        run_all_requested() — execução de todas as fases em sequência
    """

    strategy_modified = QtCore.pyqtSignal(object)
    run_instruction_requested = QtCore.pyqtSignal(object)
    run_all_requested = QtCore.pyqtSignal()
    validate_instruction_requested = QtCore.pyqtSignal(object)
    view_trajectory_requested = QtCore.pyqtSignal(object)
    view_score_distribution_requested = QtCore.pyqtSignal(object)
    view_validation_result_requested = QtCore.pyqtSignal(object)
    view_casework_visualization_requested = QtCore.pyqtSignal(object)
    resource_edit_requested = QtCore.pyqtSignal(str)
    instruction_applied = QtCore.pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._definition: Optional[LRStrategyDefinition] = None
        self._phase_widgets: Dict[str, Tuple[InstructionListWidget, InstructionConfigPanel]] = {}
        self._init_ui()
        self.show_empty()

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(0)

        # Stack: empty ↔ editor
        self.stack = QtWidgets.QStackedWidget()
        root.addWidget(self.stack)

        # Página 0 — vazia
        empty_page = QtWidgets.QWidget()
        lbl = QtWidgets.QLabel("Abra ou crie uma estratégia para começar a editar.")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 14px;")
        QtWidgets.QVBoxLayout(empty_page).addWidget(lbl)
        self.stack.addWidget(empty_page)   # index 0

        # Página 1 — editor
        editor_page = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        # Header: nome da estratégia + combobox de EmbeddingType
        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(12)

        self.lbl_strategy_name = QtWidgets.QLabel()
        self.lbl_strategy_name.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #2c3e50; padding: 6px 8px;"
        )
        header_row.addWidget(self.lbl_strategy_name, 1)

        lbl_emb = QtWidgets.QLabel("Embedding:")
        lbl_emb.setStyleSheet("font-size: 11px; color: #555; padding-right: 2px;")
        header_row.addWidget(lbl_emb)

        self.combo_embedding_type = QtWidgets.QComboBox()
        self.combo_embedding_type.setStyleSheet("font-size: 11px;")
        self.combo_embedding_type.setMinimumWidth(180)
        self.combo_embedding_type.addItem("(auto-detect)", userData=None)
        for emb in EmbeddingType:
            self.combo_embedding_type.addItem(
                f"{emb.display_name}  ({emb.dimensions}d)", userData=emb
            )
        self.combo_embedding_type.currentIndexChanged.connect(
            self._on_embedding_type_changed
        )
        header_row.addWidget(self.combo_embedding_type)

        editor_layout.addLayout(header_row)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ddd;")
        editor_layout.addWidget(sep)

        # QTabWidget com 3 abas
        self._tabs = QtWidgets.QTabWidget()

        for phase_key, phase_label in _PHASES:
            instr_list = InstructionListWidget(phase=phase_key)
            config_panel = InstructionConfigPanel()

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(instr_list)
            splitter.addWidget(config_panel)
            splitter.setSizes([240, 460])

            self._tabs.addTab(splitter, phase_label)
            self._phase_widgets[phase_key] = (instr_list, config_panel)

            # Conexões — default args capturam valores corretamente
            instr_list.instruction_selected.connect(
                lambda instr, cp=config_panel: cp.load_instruction(instr)
            )
            instr_list.add_instruction_requested.connect(
                lambda itype, pk=phase_key: self._on_add_instruction(itype, pk)
            )
            instr_list.remove_instruction_requested.connect(
                lambda instr, pk=phase_key: self._on_remove_instruction(instr, pk)
            )
            instr_list.order_changed.connect(
                lambda new_order, pk=phase_key: self._on_order_changed(new_order, pk)
            )
            config_panel.config_changed.connect(self._on_config_changed)
            config_panel.validate_requested.connect(self._on_validate_instruction_requested)
            config_panel.view_trajectory_requested.connect(self._on_view_trajectory_requested)
            config_panel.view_score_distribution_requested.connect(self._on_view_score_distribution_requested)
            config_panel.view_validation_result_requested.connect(self._on_view_validation_result_requested)
            config_panel.view_casework_visualization_requested.connect(
                self._on_view_casework_visualization_requested
            )
            config_panel.resource_edit_requested.connect(
                self._on_resource_edit_requested
            )
            config_panel.instruction_applied.connect(self._on_instruction_applied)
            instr_list.run_instruction_requested.connect(self._on_run_instruction_requested)

        editor_layout.addWidget(self._tabs, 1)
        self.stack.addWidget(editor_page)  # index 1

    # ── Public API ────────────────────────────────────────────────────────────

    def load_strategy(
        self,
        definition: LRStrategyDefinition,
        available_resources: Optional[List[StrategyResource]] = None,
    ) -> None:
        self._definition = definition
        self.lbl_strategy_name.setText(f"Estratégia: {definition.name}")

        for phase_key, _ in _PHASES:
            instr_list, config_panel = self._phase_widgets[phase_key]
            config_panel.set_available_resources(available_resources or [])
            instr_list.load_strategy(definition._get_phase_list(phase_key))
            config_panel.show_empty()

        # Selecionar embedding_type correto
        self.combo_embedding_type.blockSignals(True)
        idx = 0
        if definition.embedding_type:
            for i in range(self.combo_embedding_type.count()):
                if self.combo_embedding_type.itemData(i) == definition.embedding_type:
                    idx = i
                    break
        self.combo_embedding_type.setCurrentIndex(idx)
        self.combo_embedding_type.blockSignals(False)

        self.stack.setCurrentIndex(1)

    def set_strategy_name(self, name: str) -> None:
        self.lbl_strategy_name.setText(f"Estratégia: {name}")

    def update_available_resources(self, resources: List[StrategyResource]) -> None:
        for _, (_, config_panel) in self._phase_widgets.items():
            config_panel.set_available_resources(resources)

    def show_empty(self) -> None:
        self._definition = None
        for _, (instr_list, config_panel) in self._phase_widgets.items():
            instr_list.clear()
            config_panel.show_empty()
        self.combo_embedding_type.blockSignals(True)
        self.combo_embedding_type.setCurrentIndex(0)
        self.combo_embedding_type.blockSignals(False)
        self.stack.setCurrentIndex(0)

    def update_instruction_states(
        self,
        runnable_ids: Set[str],
        executed_ids: Set[str],
    ) -> None:
        """Atualiza os indicadores visuais de todas as instruções em todas as abas."""
        for _, (instr_list, _) in self._phase_widgets.items():
            instr_list.update_states(runnable_ids, executed_ids)

    def update_produced_resources(
        self,
        definition: LRStrategyDefinition,
        executed_ids: Set[str],
        context: Optional[StrategyExecutionContextProtocol] = None,
    ) -> None:
        """Atualiza o estado runtime das Saídas no InstructionConfigPanel de cada aba."""
        for phase_key, _ in _PHASES:
            _, config_panel = self._phase_widgets[phase_key]
            config_panel.update_runtime_state(definition, executed_ids, context)

    # ── Internal slots ────────────────────────────────────────────────────────

    def _on_add_instruction(self, instr_type: StrategyInstructionType, phase_key: str) -> None:
        if not self._definition:
            return
        new_instr = StrategyInstructionDefinition.from_dict({
            "instruction_id": str(uuid.uuid4())[:8],
            "name": instr_type.value,
            "instruction_type": instr_type.value,
        })
        self._definition._get_phase_list(phase_key).append(new_instr)
        instr_list, _ = self._phase_widgets[phase_key]
        instr_list.load_strategy(self._definition._get_phase_list(phase_key))
        self.strategy_modified.emit(self._definition)

    def _on_remove_instruction(self, instr: StrategyInstructionDefinition, phase_key: str) -> None:
        if not self._definition:
            return
        phase_list = self._definition._get_phase_list(phase_key)
        phase_list[:] = [i for i in phase_list if i.instruction_id != instr.instruction_id]
        instr_list, config_panel = self._phase_widgets[phase_key]
        instr_list.load_strategy(phase_list)
        config_panel.show_empty()
        self.strategy_modified.emit(self._definition)

    def _on_order_changed(self, new_order: List[StrategyInstructionDefinition], phase_key: str) -> None:
        if self._definition:
            phase_list = self._definition._get_phase_list(phase_key)
            phase_list[:] = new_order
            self.strategy_modified.emit(self._definition)

    def _on_config_changed(self, instr: StrategyInstructionDefinition) -> None:
        if self._definition:
            self.strategy_modified.emit(self._definition)

    def _on_run_instruction_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.run_instruction_requested.emit(instr)

    def _on_validate_instruction_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.validate_instruction_requested.emit(instr)

    def _on_view_trajectory_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_trajectory_requested.emit(instr)

    def _on_view_score_distribution_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_score_distribution_requested.emit(instr)

    def _on_view_validation_result_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_validation_result_requested.emit(instr)

    def _on_view_casework_visualization_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_casework_visualization_requested.emit(instr)

    def _on_resource_edit_requested(self, resource_id: str) -> None:
        self.resource_edit_requested.emit(resource_id)

    def _on_instruction_applied(self, instr: StrategyInstructionDefinition) -> None:
        self.instruction_applied.emit(instr)

    def _on_embedding_type_changed(self, index: int) -> None:
        if not self._definition:
            return
        self._definition.embedding_type = self.combo_embedding_type.itemData(index)
        self.strategy_modified.emit(self._definition)
