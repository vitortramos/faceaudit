# interface/tabs/strategy_manager_v2/comparison/comparison_window.py

"""
Janela top-level de comparação entre resultados de estratégias treinadas.

Layout:
    [toolbar com botão Atualizar]
    [QSplitter horizontal]
        [ComparisonLeftPanel]  [ComparisonContentArea]

Estado:
    - kind selecionado (qual tipo de comparação)
    - itens marcados (pares estratégia/recurso)

Ao trocar de kind: recarrega a lista de itens via availability_resolver.
Ao trocar a seleção: re-renderiza o conteúdo central.
Ao clicar Atualizar: drop dos caches + re-resolve availability + re-render.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition

from .availability_resolver import ComparisonItem, list_available_items
from .comparison_kind import ComparisonKind
from .comparison_left_panel import ComparisonLeftPanel
from .views import (
    CalibrationDistributionView,
    CaseworkResultsView,
    ValidationResultsView,
)


class ComparisonWindow(QtWidgets.QDialog):
    """
    Janela não-modal de comparação. Mantida com referência forte no
    controller pra não ser GC enquanto visível.
    """

    def __init__(
        self,
        definitions: List[LRStrategyDefinition],
        project_root: Optional[Path] = None,
        resource_loader: Any = None,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._definitions = list(definitions)
        self._project_root = project_root
        self._resource_loader = resource_loader

        # State runtime.
        self._current_kind: Optional[ComparisonKind] = None
        self._current_view: Optional[QtWidgets.QWidget] = None

        self.setWindowTitle("Comparação de Estratégias")
        self.setModal(False)
        # Window + min/max/close buttons explicitamente. O QDialog default
        # vem só com close — sem isso o usuário não tem como maximizar a
        # janela pra ocupar toda a tela como pediu.
        self.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowMinimizeButtonHint
            | QtCore.Qt.WindowType.WindowMaximizeButtonHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        # Tamanho default grande (caso o caller não faça showMaximized).
        self.resize(1500, 900)

        self._init_ui()

    # ── UI build ──────────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar topo ────────────────────────────────────────────────────
        toolbar = QtWidgets.QFrame()
        toolbar.setStyleSheet(
            "QFrame { background: #f5f7fa; border-bottom: 1px solid #c0c8d0; }"
        )
        tb_layout = QtWidgets.QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 4, 8, 4)
        tb_layout.setSpacing(4)

        self.btn_refresh = QtWidgets.QPushButton("🔄 Atualizar")
        self.btn_refresh.setToolTip(
            "Recarrega a lista de itens disponíveis e re-renderiza o "
            "conteúdo. Útil se você executou uma instrução no editor "
            "principal e quer ver o resultado novo aqui."
        )
        self.btn_refresh.setStyleSheet(
            "QPushButton { border: 1px solid #bbb; border-radius: 3px;"
            " background: white; padding: 4px 12px; font-size: 11px; }"
            "QPushButton:hover { background: #e8f0fe; }"
        )
        self.btn_refresh.clicked.connect(self._on_refresh_clicked)
        tb_layout.addWidget(self.btn_refresh)
        tb_layout.addStretch(1)

        root.addWidget(toolbar)

        # ── Splitter horizontal: left panel | conteúdo ─────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #dcdde1; }"
            "QSplitter::handle:hover { background: #a0a0a0; }"
        )

        self._left_panel = ComparisonLeftPanel()
        self._left_panel.setMinimumWidth(220)
        self._left_panel.setMaximumWidth(400)
        self._left_panel.kind_changed.connect(self._on_kind_changed)
        self._left_panel.selection_changed.connect(self._on_selection_changed)

        self._content_area = QtWidgets.QStackedWidget()
        self._content_area.setStyleSheet(
            "QStackedWidget { background: #fafdf6; }"
        )
        # Página 0: estado vazio (nenhum kind selecionado).
        empty = QtWidgets.QWidget()
        empty_layout = QtWidgets.QVBoxLayout(empty)
        empty_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_lbl = QtWidgets.QLabel(
            "Selecione um tipo de comparação à esquerda."
        )
        empty_lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        empty_lbl.setStyleSheet(
            "color: #888; font-style: italic; font-size: 14px;"
        )
        empty_lbl.setWordWrap(True)
        empty_layout.addWidget(empty_lbl)
        self._content_area.addWidget(empty)  # index 0

        splitter.addWidget(self._left_panel)
        splitter.addWidget(self._content_area)
        splitter.setSizes([280, 920])

        root.addWidget(splitter, 1)

        # Estado inicial.
        self._left_panel.show_empty_kind_state()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_kind_changed(self, kind: ComparisonKind) -> None:
        """Trocou tipo de comparação — repopula a tree + re-renderiza."""
        self._current_kind = kind
        with self._wait_cursor():
            self._reload_items()
            self._render_content([])

    def _on_selection_changed(self, items: List[ComparisonItem]) -> None:
        with self._wait_cursor():
            self._render_content(items)

    def _on_refresh_clicked(self) -> None:
        """Drop caches + re-resolve availability + re-render."""
        with self._wait_cursor():
            if self._current_kind is None:
                self._left_panel.show_empty_kind_state()
                self._render_content([])
                return
            self._reload_items()
            self._render_content(self._left_panel.selected_items())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reload_items(self) -> None:
        if self._current_kind is None or self._project_root is None:
            self._left_panel.populate_items([])
            return
        items = list_available_items(
            self._current_kind, self._definitions, Path(self._project_root)
        )
        self._left_panel.populate_items(items)

    def _render_content(self, items: List[ComparisonItem]) -> None:
        # Drop view anterior (limpa figures matplotlib pra liberar memória).
        if self._current_view is not None:
            idx = self._content_area.indexOf(self._current_view)
            if idx >= 0:
                self._content_area.removeWidget(self._current_view)
            self._current_view.deleteLater()
            self._current_view = None

        if self._current_kind is None:
            self._content_area.setCurrentIndex(0)
            return

        new_view = self._build_view_for_kind(self._current_kind, items)
        self._current_view = new_view
        self._content_area.addWidget(new_view)
        self._content_area.setCurrentWidget(new_view)

    def _build_view_for_kind(
        self,
        kind: ComparisonKind,
        items: List[ComparisonItem],
    ) -> QtWidgets.QWidget:
        if kind == ComparisonKind.CALIBRATION_DATASET_DISTRIBUTION:
            return CalibrationDistributionView(
                items=items,
                resource_loader=self._resource_loader,
                project_root=self._project_root,
            )
        if kind == ComparisonKind.VALIDATION_RESULTS:
            return ValidationResultsView(
                items=items, definitions=self._definitions,
                project_root=self._project_root,
            )
        if kind == ComparisonKind.CASEWORK_RESULTS:
            return CaseworkResultsView(
                items=items,
                resource_loader=self._resource_loader,
                project_root=self._project_root,
            )
        # Fallback (não deveria acontecer)
        lbl = QtWidgets.QLabel(f"Tipo não suportado: {kind}")
        lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return lbl

    # ── Wait cursor ───────────────────────────────────────────────────────────

    @contextmanager
    def _wait_cursor(self):
        """Cursor de espera durante reload/render — pickles e plots travam UI."""
        QtWidgets.QApplication.setOverrideCursor(
            QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor)
        )
        try:
            QtWidgets.QApplication.processEvents()
            yield
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
