# interface/tabs/strategy_manager_v2/left_panel/strategy_list_widget.py

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtWidgets, QtCore

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition


class StrategyListWidget(QtWidgets.QWidget):
    """
    Painel com lista de estratégias do projeto aberto.

    Sinais:
        strategy_selected(LRStrategyDefinition)  — clique simples
        strategy_opened(LRStrategyDefinition)    — duplo-clique → carrega no editor
        new_strategy_requested
        duplicate_requested(LRStrategyDefinition)
        remove_requested(LRStrategyDefinition)
    """

    strategy_selected = QtCore.pyqtSignal(object)
    strategy_opened = QtCore.pyqtSignal(object)
    new_strategy_requested = QtCore.pyqtSignal()
    duplicate_requested = QtCore.pyqtSignal(object)
    rename_requested = QtCore.pyqtSignal(object)
    remove_requested = QtCore.pyqtSignal(object)
    run_strategy_requested = QtCore.pyqtSignal()
    save_as_template_requested = QtCore.pyqtSignal()
    comparison_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._definitions: List[LRStrategyDefinition] = []
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header com fundo azul (mesmo padrão do header de Recursos do Projeto)
        header = QtWidgets.QLabel("Estratégias")
        header.setStyleSheet("""
            font-weight: bold; font-size: 12px; color: #2c3e50;
            background: #eef2f7;
            border: none;
            border-bottom: 1px solid #c0c8d0;
            padding: 5px 8px;
        """)
        layout.addWidget(header)

        # Lista
        self.list_widget = QtWidgets.QListWidget()
        # AlternatingRowColors desativado: o tom cinza alternado dava a
        # impressão visual de que o item cinza estava selecionado. Como
        # o highlight de seleção é o azul-claro dedicado, a alternância
        # confunde ao invés de ajudar.
        self.list_widget.setAlternatingRowColors(False)
        # Seleção: fundo azul claro + barra lateral azul vibrante. Mantém
        # o highlight visível mesmo quando a janela perde foco (:!active),
        # crítico pra deixar claro qual estratégia está aberta no editor.
        self.list_widget.setStyleSheet("""
            QListWidget { font-size: 11px; border: none; background: white; }
            QListWidget::item:selected {
                background: #cfe7fa; color: inherit;
                border-left: 3px solid #1565c0;
            }
            QListWidget::item:selected:!active {
                background: #cfe7fa; color: inherit;
            }
            QListWidget::item:hover:!selected {
                background: #f0f4f8;
            }
        """)
        self.list_widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.list_widget, 1)

        # Botões
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(4, 3, 4, 3)
        btn_row.setSpacing(4)

        self.btn_new = QtWidgets.QPushButton("+ Nova")
        self.btn_new.setToolTip("Criar nova estratégia")
        self.btn_new.clicked.connect(self.new_strategy_requested)

        self.btn_dup = QtWidgets.QPushButton("Duplicar")
        self.btn_dup.setToolTip("Duplicar estratégia selecionada")
        self.btn_dup.setEnabled(False)
        self.btn_dup.clicked.connect(self._on_duplicate)

        self.btn_del = QtWidgets.QPushButton("Remover")
        self.btn_del.setToolTip("Remover estratégia selecionada")
        self.btn_del.setEnabled(False)
        self.btn_del.clicked.connect(self._on_remove)

        for btn in (self.btn_new, self.btn_dup, self.btn_del):
            btn.setStyleSheet("""
                QPushButton { border: 1px solid #ccc; border-radius: 3px;
                              background: white; padding: 2px 6px; font-size: 11px; }
                QPushButton:hover { background: #e8f0fe; }
                QPushButton:disabled { color: #aaa; }
            """)
            btn_row.addWidget(btn)

        layout.addLayout(btn_row)

        # Botões de ação da estratégia selecionada
        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(4, 0, 4, 3)
        action_row.setSpacing(4)

        self.btn_run = QtWidgets.QPushButton("Executar")
        self.btn_run.setToolTip("Executar a estratégia selecionada (todas as fases)")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("""
            QPushButton {
                border: 1px solid #2e7d32; border-radius: 3px;
                background: #e8f5e9; color: #2e7d32;
                padding: 2px 6px; font-size: 11px; font-weight: bold;
            }
            QPushButton:hover { background: #c8e6c9; }
            QPushButton:pressed { background: #a5d6a7; }
            QPushButton:disabled { background: #f0f0f0; color: #aaa; border-color: #ddd; }
        """)
        self.btn_run.clicked.connect(self.run_strategy_requested)
        action_row.addWidget(self.btn_run)

        self.btn_template = QtWidgets.QPushButton("Salvar como Template")
        self.btn_template.setToolTip("Salvar estratégia como template na biblioteca global")
        self.btn_template.setEnabled(False)
        self.btn_template.setStyleSheet("""
            QPushButton { border: 1px solid #ccc; border-radius: 3px;
                          background: white; padding: 2px 6px; font-size: 11px; }
            QPushButton:hover { background: #e8f0fe; }
            QPushButton:disabled { color: #aaa; }
        """)
        self.btn_template.clicked.connect(self.save_as_template_requested)
        action_row.addWidget(self.btn_template)

        # Botão "Comparar" — abre janela top-level dedicada à comparação
        # entre estratégias treinadas. Habilitado apenas quando há ≥2
        # estratégias no projeto (lógica em populate/_update_buttons).
        self.btn_compare = QtWidgets.QPushButton("Comparar")
        self.btn_compare.setToolTip(
            "Abrir janela de comparação entre estratégias treinadas"
        )
        self.btn_compare.setEnabled(False)
        self.btn_compare.setStyleSheet("""
            QPushButton { border: 1px solid #ccc; border-radius: 3px;
                          background: white; padding: 2px 6px; font-size: 11px; }
            QPushButton:hover { background: #e8f0fe; }
            QPushButton:disabled { color: #aaa; }
        """)
        self.btn_compare.clicked.connect(self.comparison_requested)
        action_row.addWidget(self.btn_compare)

        layout.addLayout(action_row)

    # ── Public API ────────────────────────────────────────────────────────────

    def populate(
        self,
        definitions: List[LRStrategyDefinition],
        selected_strategy_id: Optional[str] = None,
    ) -> None:
        """
        Repopula a lista com as definições fornecidas.

        Quando `selected_strategy_id` é informado e bate com alguma das
        definições, esse item fica selecionado visualmente — preserva o
        feedback de "qual estratégia está aberta no editor" através de
        refreshes da lista (após edição, execução, rename etc.).
        """
        self._definitions = list(definitions)
        # blockSignals evita reemitir strategy_selected durante o setCurrentRow
        # programático — só queremos o highlight visual, não um novo evento de
        # abertura que dispararia _on_strategy_opened de novo.
        self.list_widget.blockSignals(True)
        try:
            self.list_widget.clear()
            target_index = -1
            for idx, defn in enumerate(self._definitions):
                item = QtWidgets.QListWidgetItem(defn.name)
                item.setData(QtCore.Qt.ItemDataRole.UserRole, defn)
                self.list_widget.addItem(item)
                if (
                    selected_strategy_id is not None
                    and defn.strategy_id == selected_strategy_id
                ):
                    target_index = idx
            if target_index >= 0:
                self.list_widget.setCurrentRow(target_index)
        finally:
            self.list_widget.blockSignals(False)
        self._update_buttons()

    def select_strategy_by_id(self, strategy_id: Optional[str]) -> None:
        """
        Restaura a seleção visual da estratégia com o id informado.
        No-op se id é None ou não está na lista. Não dispara signals.
        """
        if not strategy_id:
            return
        self.list_widget.blockSignals(True)
        try:
            for idx in range(self.list_widget.count()):
                item = self.list_widget.item(idx)
                defn = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if defn is not None and defn.strategy_id == strategy_id:
                    self.list_widget.setCurrentRow(idx)
                    break
        finally:
            self.list_widget.blockSignals(False)
        self._update_buttons()

    def clear(self) -> None:
        self._definitions.clear()
        self.list_widget.clear()
        self._update_buttons()

    def selected_definition(self) -> Optional[LRStrategyDefinition]:
        item = self.list_widget.currentItem()
        if item:
            return item.data(QtCore.Qt.ItemDataRole.UserRole)
        return None

    # ── Internal slots ────────────────────────────────────────────────────────

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        defn = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self._update_buttons()
        self.strategy_selected.emit(defn)

    def _on_item_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        defn = item.data(QtCore.Qt.ItemDataRole.UserRole)
        self.strategy_opened.emit(defn)

    def _on_duplicate(self) -> None:
        defn = self.selected_definition()
        if defn:
            self.duplicate_requested.emit(defn)

    def _on_remove(self) -> None:
        defn = self.selected_definition()
        if defn:
            self.remove_requested.emit(defn)

    def set_strategy_actions_enabled(self, enabled: bool) -> None:
        """Habilita/desabilita botões de ação (Executar, Template)."""
        self.btn_run.setEnabled(enabled)
        self.btn_template.setEnabled(enabled)

    def _update_buttons(self) -> None:
        has_selection = self.selected_definition() is not None
        self.btn_dup.setEnabled(has_selection)
        self.btn_del.setEnabled(has_selection)
        # Comparar exige pelo menos 1 estratégia. Mesmo com uma só vale
        # abrir a janela: uma estratégia pode ter múltiplos resultados
        # (ex: 2 STRATEGY_EVALUATION com VALIDATION_RESULTs distintos —
        # tuning de calibrador vs validação final) que valem comparar.
        self.btn_compare.setEnabled(self.list_widget.count() >= 1)

    def _show_context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.list_widget.itemAt(pos)
        if not item:
            return
        defn = item.data(QtCore.Qt.ItemDataRole.UserRole)
        menu = QtWidgets.QMenu(self)
        menu.addAction("Abrir no editor", lambda: self.strategy_opened.emit(defn))
        menu.addAction("Renomear", lambda: self.rename_requested.emit(defn))
        menu.addAction("Duplicar", lambda: self.duplicate_requested.emit(defn))
        menu.addSeparator()
        menu.addAction("Remover", lambda: self.remove_requested.emit(defn))
        menu.exec(self.list_widget.mapToGlobal(pos))
