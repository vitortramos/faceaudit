# interface/tabs/strategy_manager_v2/comparison/comparison_left_panel.py

"""
Painel esquerdo da ComparisonWindow.

Topo: lista de tipos de comparação (3 itens fixos, single-select).
Baixo: árvore com 2 níveis (estratégia → recursos checkable).

Sinais:
    kind_changed(ComparisonKind)
    selection_changed(List[ComparisonItem])
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6 import QtCore, QtWidgets

from interface.theme import current_token_fn
from .availability_resolver import ComparisonItem
from .comparison_kind import ComparisonKind


class ComparisonLeftPanel(QtWidgets.QWidget):
    kind_changed = QtCore.pyqtSignal(object)         # ComparisonKind
    selection_changed = QtCore.pyqtSignal(object)    # List[ComparisonItem]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        # State: items disponíveis indexados por (strategy_id, resource_id).
        # Reconstruído a cada `populate_items`.
        self._items_by_key: Dict[tuple, ComparisonItem] = {}

        t = current_token_fn()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle { background: " + t("BORDER") + "; }"
            "QSplitter::handle:hover { background: " + t("TEXT_MUTED") + "; }"
        )

        # ── Topo: kind selector ──────────────────────────────────────────
        kind_container = QtWidgets.QWidget()
        kind_layout = QtWidgets.QVBoxLayout(kind_container)
        kind_layout.setContentsMargins(0, 0, 0, 0)
        kind_layout.setSpacing(0)

        kind_header = QtWidgets.QLabel("O QUE COMPARAR")
        kind_header.setStyleSheet(
            "background: " + t("HEADER_BG") + "; color: " + t("HEADER_TEXT") + "; font-size: 11px;"
            " font-weight: bold; letter-spacing: 1px; padding: 6px 10px;"
        )
        kind_layout.addWidget(kind_header)

        self._kind_list = QtWidgets.QListWidget()
        self._kind_list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._kind_list.setStyleSheet(
            "QListWidget { font-size: 12px; border: none; background: " + t("SURFACE") + "; }"
            "QListWidget::item { padding: 8px 10px; }"
            "QListWidget::item:selected {"
            "    background: " + t("PRIMARY_BG") + "; color: " + t("TEXT") + ";"
            "    border-left: 3px solid " + t("PRIMARY") + ";"
            "}"
            "QListWidget::item:selected:!active {"
            "    background: " + t("PRIMARY_BG") + "; color: " + t("TEXT") + ";"
            "}"
            "QListWidget::item:hover:!selected { background: " + t("SURFACE_RAISED") + "; }"
        )
        for kind in ComparisonKind:
            item = QtWidgets.QListWidgetItem(kind.label)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, kind)
            item.setToolTip(kind.description)
            self._kind_list.addItem(item)
        self._kind_list.itemSelectionChanged.connect(self._on_kind_selection)
        kind_layout.addWidget(self._kind_list, 1)

        splitter.addWidget(kind_container)

        # ── Baixo: items tree ────────────────────────────────────────────
        items_container = QtWidgets.QWidget()
        items_layout = QtWidgets.QVBoxLayout(items_container)
        items_layout.setContentsMargins(0, 0, 0, 0)
        items_layout.setSpacing(0)

        items_header = QtWidgets.QLabel("ITENS A COMPARAR")
        items_header.setStyleSheet(
            "background: " + t("HEADER_BG") + "; color: " + t("HEADER_TEXT") + "; font-size: 11px;"
            " font-weight: bold; letter-spacing: 1px; padding: 6px 10px;"
        )
        items_layout.addWidget(items_header)

        self._items_tree = QtWidgets.QTreeWidget()
        self._items_tree.setHeaderHidden(True)
        self._items_tree.setStyleSheet(
            "QTreeWidget { font-size: 11px; border: none; background: " + t("SURFACE") + "; }"
            "QTreeWidget::item { padding: 2px 0px; }"
            "QTreeWidget::item:selected { background: transparent; }"
            "QTreeWidget::item:hover:!selected { background: " + t("SURFACE_RAISED") + "; }"
        )
        self._items_tree.itemChanged.connect(self._on_item_check_changed)
        items_layout.addWidget(self._items_tree, 1)

        # Estado vazio
        self._empty_lbl = QtWidgets.QLabel(
            "Selecione um tipo de comparação acima."
        )
        self._empty_lbl.setStyleSheet(
            "color: " + t("TEXT_MUTED") + "; font-style: italic; font-size: 11px; padding: 10px;"
        )
        self._empty_lbl.setWordWrap(True)
        self._empty_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        items_layout.addWidget(self._empty_lbl)

        splitter.addWidget(items_container)
        splitter.setSizes([180, 400])

        layout.addWidget(splitter)

    # ── Public API ────────────────────────────────────────────────────────────

    def selected_kind(self) -> Optional[ComparisonKind]:
        item = self._kind_list.currentItem()
        if item is None:
            return None
        return item.data(QtCore.Qt.ItemDataRole.UserRole)

    def selected_items(self) -> List[ComparisonItem]:
        """Retorna apenas os ComparisonItem com checkbox marcado."""
        out: List[ComparisonItem] = []
        for i in range(self._items_tree.topLevelItemCount()):
            strategy_node = self._items_tree.topLevelItem(i)
            for j in range(strategy_node.childCount()):
                child = strategy_node.child(j)
                if child.checkState(0) == QtCore.Qt.CheckState.Checked:
                    key = child.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    if key in self._items_by_key:
                        out.append(self._items_by_key[key])
        return out

    def populate_items(self, items: List[ComparisonItem]) -> None:
        """
        Repovoa a árvore com os items disponíveis para o kind atual.

        Itens já marcados anteriormente cuja chave persiste continuam marcados
        (preserva seleção quando o usuário só clica em "Atualizar"). Novos
        itens entram desmarcados.
        """
        previously_checked = {
            key for key, _ in self._items_by_key.items()
            if self._is_key_checked(key)
        }

        self._items_tree.blockSignals(True)
        try:
            self._items_tree.clear()
            self._items_by_key = {}

            # Agrupa por estratégia (nome).
            by_strategy: Dict[str, List[ComparisonItem]] = {}
            for it in items:
                by_strategy.setdefault(it.strategy_name, []).append(it)

            for strategy_name in sorted(by_strategy.keys(), key=str.lower):
                group = by_strategy[strategy_name]
                strategy_node = QtWidgets.QTreeWidgetItem([
                    f"{strategy_name}  ({len(group)})"
                ])
                # Estilo do header do grupo: cinza, bold, não-clicável.
                font = strategy_node.font(0)
                font.setBold(True)
                strategy_node.setFont(0, font)
                strategy_node.setForeground(
                    0, QtCore.Qt.GlobalColor.darkGray
                )
                # Sem checkbox no header — ele é meramente agrupador.
                self._items_tree.addTopLevelItem(strategy_node)
                strategy_node.setExpanded(True)

                for it in group:
                    child = QtWidgets.QTreeWidgetItem([it.resource_name])
                    child.setFlags(
                        child.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                    )
                    key = (it.strategy_id, it.resource_id)
                    child.setData(0, QtCore.Qt.ItemDataRole.UserRole, key)
                    initial = (
                        QtCore.Qt.CheckState.Checked
                        if key in previously_checked
                        else QtCore.Qt.CheckState.Unchecked
                    )
                    child.setCheckState(0, initial)
                    child.setToolTip(0, str(it.artifact_path or ""))
                    strategy_node.addChild(child)
                    self._items_by_key[key] = it
        finally:
            self._items_tree.blockSignals(False)

        # Mostra/esconde o label de vazio.
        has_any = bool(self._items_by_key)
        self._items_tree.setVisible(has_any)
        if has_any:
            kind = self.selected_kind()
            self._empty_lbl.setText("")
            self._empty_lbl.setVisible(False)
        else:
            self._empty_lbl.setText(
                "Nenhum recurso disponível para o tipo selecionado.\n"
                "Execute as instruções correspondentes nas estratégias."
            )
            self._empty_lbl.setVisible(True)

        # Re-emite seleção (pode ter mudado se itens previamente marcados
        # sumiram).
        self.selection_changed.emit(self.selected_items())

    def show_empty_kind_state(self) -> None:
        """Exibido quando nenhum kind está selecionado ainda."""
        self._items_tree.blockSignals(True)
        try:
            self._items_tree.clear()
            self._items_by_key = {}
        finally:
            self._items_tree.blockSignals(False)
        self._items_tree.setVisible(False)
        self._empty_lbl.setText(
            "Selecione um tipo de comparação acima."
        )
        self._empty_lbl.setVisible(True)
        self.selection_changed.emit([])

    # ── Internal slots ────────────────────────────────────────────────────────

    def _on_kind_selection(self) -> None:
        kind = self.selected_kind()
        if kind is not None:
            self.kind_changed.emit(kind)

    def _on_item_check_changed(self, _item, _col) -> None:
        self.selection_changed.emit(self.selected_items())

    def _is_key_checked(self, key: tuple) -> bool:
        """Verifica se um item específico está marcado na árvore atual."""
        for i in range(self._items_tree.topLevelItemCount()):
            strategy_node = self._items_tree.topLevelItem(i)
            for j in range(strategy_node.childCount()):
                child = strategy_node.child(j)
                if child.data(0, QtCore.Qt.ItemDataRole.UserRole) == key:
                    return child.checkState(0) == QtCore.Qt.CheckState.Checked
        return False
