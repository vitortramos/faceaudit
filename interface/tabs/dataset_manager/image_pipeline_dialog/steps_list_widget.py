"""StepsListWidget — lista drag-reorder de steps + botão Adicionar
(PR I2).

Coluna esquerda do editor. Mostra os steps configurados na ordem em
que serão aplicados, com um resumo dos parâmetros de cada um (dois
"Redimensionar" na mesma cadeia precisam ser distinguíveis). Drag-reorder
via `QListWidget.InternalMove`. Cada item tem botão de remover. O botão
`+ Adicionar` lista os tipos registrados na factory, agrupados.

Sinais emitidos pro controller:
  - `selection_changed(idx)` — usuário selecionou outro step (idx ou -1)
  - `step_added(step_type)` — usuário clicou no menu Adicionar
  - `step_removed(idx)` — usuário clicou no botão remover de um item
  - `steps_reordered(new_order)` — usuário arrastou; lista com o
    índice ANTERIOR de cada passo na nova ordem
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.models.image_processing.image_processing_step_config import (
    ImageProcessingStepConfig,
)


def _texto_da_linha(idx: int, config: ImageProcessingStepConfig) -> str:
    resumo = config.summary_pt()
    base = f"{idx + 1}.  {config.step_type.label_pt}"
    return f"{base}  —  {resumo}" if resumo else base


class _StepRow(QtWidgets.QWidget):
    """Widget custom de uma linha — label do step + botão remover."""

    remove_clicked = QtCore.pyqtSignal()

    def __init__(self, config: ImageProcessingStepConfig, idx: int, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self.lbl = QtWidgets.QLabel(_texto_da_linha(idx, config), parent=self)
        self.lbl.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl, 1)

        self.btn_remove = QtWidgets.QToolButton(parent=self)
        self.btn_remove.setText("✕")
        self.btn_remove.setToolTip("Remover step")
        apply_themed_stylesheet(
            self.btn_remove,
            lambda t: (
                "QToolButton { color: " + t("DANGER") + "; border: none; "
                "padding: 2px 6px; font-weight: bold; } "
                "QToolButton:hover { background: " + t("DANGER_BG") + "; "
                "border-radius: 3px; }"
            ),
        )
        self.btn_remove.clicked.connect(self.remove_clicked.emit)
        layout.addWidget(self.btn_remove)

    def update_text(self, idx: int, config: ImageProcessingStepConfig):
        self.lbl.setText(_texto_da_linha(idx, config))


class StepsListWidget(QtWidgets.QWidget):
    """Lista drag-reorder + botão Adicionar."""

    selection_changed = QtCore.pyqtSignal(int)
    step_added = QtCore.pyqtSignal(object)  # ImageProcessingStepType
    step_removed = QtCore.pyqtSignal(int)
    steps_reordered = QtCore.pyqtSignal(list)  # nova ordem (indices originais)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Botão Adicionar com menu
        self.btn_add = QtWidgets.QToolButton()
        self.btn_add.setText("➕ Adicionar")
        self.btn_add.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup,
        )
        apply_themed_stylesheet(
            self.btn_add,
            lambda t: (
                "QToolButton { padding: 6px 12px; background: "
                + t("SUCCESS") + "; "
                "color: " + t("HIGHLIGHT_TEXT") + "; border-radius: 4px; "
                "font-weight: bold; } "
                "QToolButton:hover { background: " + t("SUCCESS_BG") + "; "
                "color: " + t("SUCCESS") + "; }"
            ),
        )
        menu = QtWidgets.QMenu(self.btn_add)
        categoria_atual = None
        for step_type in ImageProcessorFactory.available_step_types():
            if step_type.category_pt != categoria_atual:
                categoria_atual = step_type.category_pt
                menu.addSection(categoria_atual)
            act = menu.addAction(step_type.label_pt)
            act.triggered.connect(
                lambda _checked, st=step_type: self.step_added.emit(st)
            )
        self.btn_add.setMenu(menu)
        layout.addWidget(self.btn_add)

        # Lista
        self.list = QtWidgets.QListWidget()
        self.list.setDragDropMode(
            QtWidgets.QAbstractItemView.DragDropMode.InternalMove,
        )
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection,
        )
        self.list.currentRowChanged.connect(self._on_selection_changed)
        self.list.model().rowsMoved.connect(self._on_rows_moved)
        layout.addWidget(self.list, 1)

    # ── API pra controller ─────────────────────────────────────────────────

    def populate(self, configs: List[ImageProcessingStepConfig]):
        """Reconstrói a lista com os passos dados, na ordem."""
        self.list.blockSignals(True)
        self.list.clear()
        for idx, config in enumerate(configs):
            self._insert_row(idx, config)
        self.list.blockSignals(False)

    def refresh_texts(self, configs: List[ImageProcessingStepConfig]):
        """Atualiza número e resumo de cada linha sem reconstruir."""
        for idx, config in enumerate(configs):
            item = self.list.item(idx)
            row = self.list.itemWidget(item) if item is not None else None
            if isinstance(row, _StepRow):
                row.update_text(idx, config)

    def _insert_row(self, idx: int, config: ImageProcessingStepConfig):
        item = QtWidgets.QListWidgetItem()
        # Guarda a posição do passo no pipeline — é por ela, e não pelo
        # tipo, que a reordenação acha o config de cada linha (dois passos
        # do mesmo tipo trocavam de configuração ao arrastar).
        item.setData(QtCore.Qt.ItemDataRole.UserRole, idx)
        self.list.addItem(item)
        row_widget = _StepRow(config, idx, parent=self.list)
        row_widget.remove_clicked.connect(
            lambda i=item: self._on_remove_clicked(i),
        )
        item.setSizeHint(row_widget.sizeHint())
        self.list.setItemWidget(item, row_widget)

    def select_index(self, idx: int):
        """Seleciona programaticamente. -1 limpa seleção."""
        if idx < 0 or idx >= self.list.count():
            self.list.clearSelection()
            return
        self.list.setCurrentRow(idx)

    def current_index(self) -> int:
        return self.list.currentRow()

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_selection_changed(self, row: int):
        self.selection_changed.emit(row)

    def _on_remove_clicked(self, item: QtWidgets.QListWidgetItem):
        idx = self.list.row(item)
        if idx >= 0:
            self.step_removed.emit(idx)

    def _on_rows_moved(self, _parent, start, _end, _dest, dest_row):
        """Drag-reorder: emite a nova ordem como lista de índices
        anteriores. Ex.: [2, 0, 1] = o antigo 2 foi para a posição 0, o
        antigo 0 para a 1 e o antigo 1 para a 2. O controller reordena
        e chama `populate`, que renumera as linhas.
        """
        new_order = [
            int(self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole))
            for i in range(self.list.count())
        ]
        self.steps_reordered.emit(new_order)
