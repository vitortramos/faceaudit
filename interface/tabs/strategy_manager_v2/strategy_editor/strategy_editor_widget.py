# interface/tabs/strategy_manager_v2/strategy_editor/strategy_editor_widget.py

from __future__ import annotations

import uuid
from typing import Dict, List, Optional, Set, Tuple

from PyQt6 import QtWidgets, QtCore, QtGui

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource

from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)

from .instruction_list_widget import InstructionListWidget
from .instruction_config_panel import InstructionConfigPanel
from .instruction_results_panel import InstructionResultsPanel

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
    view_hubness_report_requested = QtCore.pyqtSignal(object)
    view_hubness_ablation_requested = QtCore.pyqtSignal(object)
    # Pedido de hidratação assíncrona de um recurso ainda em disco — emitido
    # pelo botão "Carregar e visualizar" do InstructionResultsPanel quando o
    # usuário pede pra ver gráfico de dataset grande não hidratado.
    hydrate_resource_requested = QtCore.pyqtSignal(object, str)
    resource_edit_requested = QtCore.pyqtSignal(str)
    instruction_applied = QtCore.pyqtSignal(object)
    # Disparado quando o usuário remove uma instrução da lista — o
    # controller usa pra apagar arquivos dos recursos produzidos antes
    # da instrução sumir.
    instruction_removed = QtCore.pyqtSignal(object)
    # Pedido de troca do embedding column. Diferente de strategy_modified,
    # este signal é interceptado pelo controller que pode mostrar diálogo
    # de confirmação (datasets incompatíveis + cascade de derivados) e
    # decidir aceitar ou reverter.
    embedding_column_change_requested = QtCore.pyqtSignal(str, object)
    # (old_value, new_value) — old é "" quando nada estava selecionado.

    def __init__(self, parent=None):
        super().__init__(parent)
        self._definition: Optional[LRStrategyDefinition] = None
        # Tupla expandida pra incluir o painel de resultados (3ª coluna).
        self._phase_widgets: Dict[
            str,
            Tuple[InstructionListWidget, InstructionConfigPanel, InstructionResultsPanel],
        ] = {}
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
        lbl = QtWidgets.QLabel("Selecione ou crie uma estratégia para começar a editar.")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #888; font-size: 14px;")
        QtWidgets.QVBoxLayout(empty_page).addWidget(lbl)
        self.stack.addWidget(empty_page)   # index 0

        # Página 1 — editor
        editor_page = QtWidgets.QWidget()
        editor_layout = QtWidgets.QVBoxLayout(editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)

        # Header: nome da estratégia + combobox da coluna de embedding
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

        # Combo dinâmico — populado via `set_available_embedding_columns()`
        # com a UNIÃO das colunas role=EMBEDDING dos datasets PRIMÁRIOS do
        # projeto. Sem datasets, fica desabilitado com placeholder.
        self.combo_embedding_column = QtWidgets.QComboBox()
        self.combo_embedding_column.setStyleSheet("font-size: 11px;")
        self.combo_embedding_column.setMinimumWidth(220)
        self.combo_embedding_column.setToolTip(
            "Coluna de embedding usada por toda a estratégia. "
            "Apenas datasets que tenham essa coluna ficam disponíveis nas "
            "instruções. Trocar invalida todos os recursos derivados."
        )
        # `activated` (não currentIndexChanged) — só dispara em interação
        # USUÁRIO, ignora mudanças programáticas (`setCurrentIndex` em
        # populate). Evita loops e mudanças "fantasma" durante refresh.
        self.combo_embedding_column.activated.connect(
            self._on_embedding_combo_activated
        )
        # Cache pra evitar repopulações redundantes do combo a cada
        # refresh do controller (`update_available_resources` é chamado
        # com frequência).
        self._available_embedding_columns: List[str] = []
        header_row.addWidget(self.combo_embedding_column)

        editor_layout.addLayout(header_row)

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        sep.setStyleSheet("color: #ddd;")
        editor_layout.addWidget(sep)

        # QTabWidget com 3 abas
        self._tabs = QtWidgets.QTabWidget()

        for phase_key, phase_label in _PHASES:
            instr_list = InstructionListWidget(phase=phase_key)
            # Painel de configuração: outputs + view buttons agora vivem no
            # results_panel; gate via flag pra não duplicar.
            config_panel = InstructionConfigPanel(outputs_in_results_panel=True)
            results_panel = InstructionResultsPanel()

            splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
            splitter.setChildrenCollapsible(False)
            splitter.addWidget(instr_list)
            splitter.addWidget(config_panel)
            splitter.addWidget(results_panel)
            # Tamanhos default: lista compacta / config médio / resultados largo
            # (gráficos precisam de espaço).
            splitter.setSizes([200, 400, 600])

            self._tabs.addTab(splitter, phase_label)
            self._phase_widgets[phase_key] = (instr_list, config_panel, results_panel)

            # Conexões — default args capturam valores corretamente.
            # Cursor de espera durante a troca de instrução: o config_panel
            # reconstrói o form (DTOs aninhados, dropdowns filtrados por
            # tipo de recurso) e o results_panel hidrata/redesenha o card —
            # ambos podem travar a UI por segundos em instruções pesadas.
            instr_list.instruction_selected.connect(
                lambda instr, cp=config_panel, rp=results_panel:
                    self._load_instruction_with_wait_cursor(instr, cp, rp)
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
            config_panel.resource_edit_requested.connect(
                self._on_resource_edit_requested
            )
            config_panel.instruction_applied.connect(self._on_instruction_applied)
            # Sincroniza o nome de output exibido no painel de resultados
            # (3ª coluna, read-only) sempre que o usuário edita o nome no
            # painel de configuração (2ª coluna).
            config_panel.output_name_changed.connect(
                lambda fn, val, rp=results_panel: rp.update_output_name(fn, val)
            )

            # Os botões de visualização agora vivem no painel de resultados.
            results_panel.view_trajectory_requested.connect(
                self._on_view_trajectory_requested
            )
            results_panel.view_score_distribution_requested.connect(
                self._on_view_score_distribution_requested
            )
            results_panel.view_validation_result_requested.connect(
                self._on_view_validation_result_requested
            )
            results_panel.view_casework_visualization_requested.connect(
                self._on_view_casework_visualization_requested
            )
            results_panel.view_hubness_report_requested.connect(
                self._on_view_hubness_report_requested
            )
            results_panel.view_hubness_ablation_requested.connect(
                self._on_view_hubness_ablation_requested
            )
            results_panel.hydrate_requested.connect(
                lambda instr, rid: self.hydrate_resource_requested.emit(instr, rid)
            )

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
        self._refresh_header_label()

        for phase_key, _ in _PHASES:
            instr_list, config_panel, results_panel = self._phase_widgets[phase_key]
            config_panel.set_available_resources(available_resources or [])
            instr_list.load_strategy(definition._get_phase_list(phase_key))
            config_panel.show_empty()
            # Drop dos cards cacheados: estratégia anterior pode ter
            # instruções com IDs que colidiriam (raro mas possível).
            results_panel.invalidate_all()

        # Sincroniza a seleção do combo com `definition.embedding_column`.
        # Se a coluna gravada não está entre as opções disponíveis (datasets
        # ainda não importados), adicionamos como item órfão visível —
        # validation rule depois alerta sobre inconsistência.
        self._sync_embedding_combo_to_definition()

        self.stack.setCurrentIndex(1)

    def set_strategy_name(self, name: str) -> None:
        if self._definition is not None:
            self._definition.name = name
        self._refresh_header_label()

    def _refresh_header_label(self) -> None:
        """Renderiza o header do editor com nome + id entre parênteses.

        O id na header é útil pra distinguir estratégias com nomes
        iguais (ex: "PLDA 2" / "PLDA 2") e pra confirmar visualmente
        qual estratégia está aberta antes de ações destrutivas (remover,
        selar, etc.). Single fonte: ambos `load_strategy` e
        `set_strategy_name` delegam aqui.
        """
        if self._definition is None:
            self.lbl_strategy_name.setText("Estratégia: —")
            return
        self.lbl_strategy_name.setText(
            f"Estratégia: {self._definition.name} "
            f"({self._definition.strategy_id})"
        )

    def update_available_resources(self, resources: List[StrategyResource]) -> None:
        for _, (_, config_panel, _) in self._phase_widgets.items():
            config_panel.set_available_resources(resources)

    def show_empty(self) -> None:
        self._definition = None
        for _, (instr_list, config_panel, results_panel) in self._phase_widgets.items():
            instr_list.clear()
            config_panel.show_empty()
            results_panel.invalidate_all()
        self.combo_embedding_column.blockSignals(True)
        self.combo_embedding_column.clear()
        self.combo_embedding_column.setEnabled(False)
        self.combo_embedding_column.setPlaceholderText(
            "Adicione um dataset primeiro"
        )
        self.combo_embedding_column.blockSignals(False)
        # Reset do cache — sem isso, o próximo `set_available_embedding_columns`
        # vê o mesmo conteúdo do cache pré-show_empty e considera "sem mudança",
        # pulando a repopulação. Resultado visível: combo permanece com
        # placeholder mesmo quando há datasets no projeto.
        self._available_embedding_columns = []
        self.stack.setCurrentIndex(0)

    def set_available_embedding_columns(self, columns: List[str]) -> None:
        """
        Atualiza as opções do combo com a UNIÃO das colunas role=EMBEDDING
        dos datasets PRIMÁRIOS do projeto.

        Sem datasets primários (lista vazia): combo desabilitado com
        placeholder. Com pelo menos uma coluna: combo habilitado, e:

        - Se a estratégia já tem `embedding_column` gravado E ele está nas
          opções → seleciona ele.
        - Se está gravado mas SUMIU das opções → preserva como item órfão
          marcado (validação alerta).
        - Se nada está gravado → auto-seleciona o primeiro item da lista
          e silenciosamente grava em `definition.embedding_column`.
          Sem isso, a estratégia abriria com o combo no primeiro item mas
          a definition continuaria None — confuso e quebra os filtros
          de dropdown.
        """
        normalized = sorted(set(columns or []))
        if normalized == self._available_embedding_columns:
            return  # Sem mudança — evita rebuild redundante.
        self._available_embedding_columns = normalized

        self.combo_embedding_column.blockSignals(True)
        self.combo_embedding_column.clear()
        self.combo_embedding_column.setPlaceholderText("")

        if not normalized:
            self.combo_embedding_column.setEnabled(False)
            self.combo_embedding_column.setPlaceholderText(
                "Adicione um dataset primeiro"
            )
            self.combo_embedding_column.blockSignals(False)
            return

        self.combo_embedding_column.setEnabled(True)
        for col in normalized:
            self.combo_embedding_column.addItem(col, userData=col)

        self._sync_embedding_combo_to_definition()
        self.combo_embedding_column.blockSignals(False)

    def _sync_embedding_combo_to_definition(self) -> None:
        """Alinha o índice selecionado do combo com
        `definition.embedding_column`. Se a definition não tem nada
        gravado e o combo tem opções, auto-seleciona a primeira E grava
        silenciosamente — evita estado intermediário onde o usuário
        "vê" uma seleção mas a definition está None.

        Caller deve ter chamado `blockSignals(True)` antes — esse método
        muta o combo sem disparar `activated`.
        """
        target = getattr(self._definition, "embedding_column", None)

        # Caso 1: definition tem coluna gravada — tenta selecionar.
        if target:
            for i in range(self.combo_embedding_column.count()):
                if self.combo_embedding_column.itemData(i) == target:
                    self.combo_embedding_column.setCurrentIndex(i)
                    return
            # Coluna não está mais nas opções — preserva como item órfão.
            self.combo_embedding_column.addItem(
                f"{target}  (indisponível nos datasets atuais)",
                userData=target,
            )
            self.combo_embedding_column.setCurrentIndex(
                self.combo_embedding_column.count() - 1
            )
            return

        # Caso 2: definition sem coluna + combo tem opções → auto-seleciona
        # a primeira E grava na definition. Não emite signal (mudança
        # programática automática, não interação do usuário).
        if self.combo_embedding_column.count() > 0 and self._definition is not None:
            self.combo_embedding_column.setCurrentIndex(0)
            self._definition.embedding_column = (
                self.combo_embedding_column.itemData(0)
            )

    def update_instruction_states(
        self,
        runnable_ids: Set[str],
        executed_ids: Set[str],
    ) -> None:
        """Atualiza os indicadores visuais de todas as instruções em todas as abas."""
        for _, (instr_list, _, _) in self._phase_widgets.items():
            instr_list.update_states(runnable_ids, executed_ids)

    def update_produced_resources(
        self,
        definition: LRStrategyDefinition,
        executed_ids: Set[str],
        context: Optional[StrategyExecutionContextProtocol] = None,
    ) -> None:
        """
        Atualiza o estado runtime no InstructionResultsPanel de cada aba.

        (Nome do método mantido por compatibilidade com o controller V2,
        que não foi alterado.)
        """
        for phase_key, _ in _PHASES:
            _, _, results_panel = self._phase_widgets[phase_key]
            results_panel.update_runtime_state(definition, executed_ids, context)

    def invalidate_results_for_instruction(self, instruction_id: str) -> None:
        """
        Notifica todas as abas que uma instrução foi re-executada — força o
        InstructionResultsPanel a dropar o card cacheado pra que a próxima
        renderização use dados frescos.
        """
        for phase_key, _ in _PHASES:
            _, _, results_panel = self._phase_widgets[phase_key]
            results_panel.invalidate_instruction(instruction_id)

    def set_resource_loader(self, loader: object) -> None:
        """Propaga o resource loader para os painéis de resultados."""
        for phase_key, _ in _PHASES:
            _, _, results_panel = self._phase_widgets[phase_key]
            results_panel.set_resource_loader(loader)

    def set_project_root(self, project_root: object) -> None:
        """
        Propaga a raiz do projeto para os painéis de configuração. Usada
        pelos info labels de dataset (sob dropdowns de DATASET) para
        localizar o sidecar `<stem>_stats.json` ao lado do parquet.
        """
        for phase_key, _ in _PHASES:
            _, config_panel, _ = self._phase_widgets[phase_key]
            if hasattr(config_panel, "set_project_root"):
                config_panel.set_project_root(project_root)

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
        instr_list, _, _ = self._phase_widgets[phase_key]
        instr_list.load_strategy(self._definition._get_phase_list(phase_key))
        self.strategy_modified.emit(self._definition)

    def _on_remove_instruction(self, instr: StrategyInstructionDefinition, phase_key: str) -> None:
        if not self._definition:
            return
        # Remove do phase_list ANTES de emitir os signals — qualquer
        # save disparado por `_on_instruction_removed` ou
        # `_on_strategy_modified` no controller vê a definition já no
        # estado pós-remoção.
        phase_list = self._definition._get_phase_list(phase_key)
        phase_list[:] = [i for i in phase_list if i.instruction_id != instr.instruction_id]

        # `instruction_removed` carrega o `instr` object (passado por
        # referência) — controller usa pra saber quais outputs apagar
        # do disco e dos placeholders. NÃO depende de phase_list.
        self.instruction_removed.emit(instr)

        instr_list, config_panel, results_panel = self._phase_widgets[phase_key]
        instr_list.load_strategy(phase_list)
        config_panel.show_empty()
        # Drop do card cacheado dessa instrução (se havia).
        results_panel.invalidate_instruction(instr.instruction_id)
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

    def _on_view_hubness_report_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_hubness_report_requested.emit(instr)

    def _on_view_hubness_ablation_requested(self, instr: StrategyInstructionDefinition) -> None:
        self.view_hubness_ablation_requested.emit(instr)

    def _on_resource_edit_requested(self, resource_id: str) -> None:
        self.resource_edit_requested.emit(resource_id)

    def _on_instruction_applied(self, instr: StrategyInstructionDefinition) -> None:
        self.instruction_applied.emit(instr)

    def _on_embedding_combo_activated(self, index: int) -> None:
        """Disparado quando o usuário interage com o combo (não em mudanças
        programáticas). Emite `embedding_column_change_requested(old, new)`
        — o controller decide se aceita (cascade dos derivados) ou se
        cancela (e chama `set_embedding_column_silent` pra reverter).
        """
        if not self._definition:
            return
        old_value = getattr(self._definition, "embedding_column", None) or ""
        new_value = self.combo_embedding_column.itemData(index)
        if old_value == (new_value or ""):
            return  # No-op (mesmo valor) — não notifica.
        self.embedding_column_change_requested.emit(old_value, new_value)

    def set_embedding_column_silent(self, value: Optional[str]) -> None:
        """Sincroniza o combo + a definition com `value` SEM disparar
        signals. Usado pelo controller pra (a) confirmar uma mudança
        aceita ou (b) reverter uma mudança rejeitada pelo usuário no
        diálogo de confirmação."""
        if self._definition is None:
            return
        self._definition.embedding_column = value
        self.combo_embedding_column.blockSignals(True)
        self._sync_embedding_combo_to_definition()
        self.combo_embedding_column.blockSignals(False)

    def _load_instruction_with_wait_cursor(
        self,
        instr: StrategyInstructionDefinition,
        config_panel: InstructionConfigPanel,
        results_panel: InstructionResultsPanel,
    ) -> None:
        """
        Carrega a instrução nos painéis de configuração e resultados com
        cursor de espera ativo. Reconstruir o form (specs dinâmicos, DTOs
        aninhados, dropdowns filtrados) e popular o card de resultados
        (hidratar payload, desenhar gráfico) podem travar a UI por
        segundos em instruções pesadas (SIMILARITY_SCORING com 5M+ pares,
        STRATEGY_EVALUATION com 4 abas matplotlib).
        """
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            QtWidgets.QApplication.processEvents()
            config_panel.load_instruction(instr)
            results_panel.load_instruction(instr)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
