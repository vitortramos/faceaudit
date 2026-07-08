"""StepsListWidget — lista drag-reorder de steps + botão Adicionar
(PR I2).

Coluna esquerda do editor. Mostra os steps configurados na ordem em
que serão aplicados. Drag-reorder via `QListWidget.InternalMove`.
Cada item tem botão de remover. Botão `+ Adicionar` mostra menu com
os 5 step types.

Sinais emitidos pro controller:
  - `selection_changed(idx)` — usuário selecionou outro step (idx ou -1)
  - `step_added(step_type)` — usuário clicou no menu Adicionar
  - `step_removed(idx)` — usuário clicou no botão remover de um item
  - `steps_reordered(new_order_indices)` — usuário arrastou
"""

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from interface.theme import apply_themed_stylesheet
from faceaudit.enums.image_processing_step_type import ImageProcessingStepType


# Step types disponíveis no menu "Adicionar". Ordem reflete a do MVP.
_AVAILABLE_TYPES = [
    ImageProcessingStepType.BLUR,
    ImageProcessingStepType.JPEG,
    ImageProcessingStepType.WEBP,
    ImageProcessingStepType.CROP,
    ImageProcessingStepType.OCCLUSION,
]


class _StepRow(QtWidgets.QWidget):
    """Widget custom de uma linha — label do step + botão remover."""

    remove_clicked = QtCore.pyqtSignal()

    def __init__(self, step_type: ImageProcessingStepType, idx: int):
        super().__init__()
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(6)

        self.lbl = QtWidgets.QLabel(f"{idx + 1}.  {step_type.label_pt}")
        self.lbl.setStyleSheet("font-size: 11px;")
        layout.addWidget(self.lbl, 1)

        self.btn_remove = QtWidgets.QToolButton()
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

    def update_index(self, idx: int, step_type: ImageProcessingStepType):
        self.lbl.setText(f"{idx + 1}.  {step_type.label_pt}")


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
        for step_type in _AVAILABLE_TYPES:
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

    def populate(self, step_types: List[ImageProcessingStepType]):
        """Reconstrói a lista com os step_types dados na ordem."""
        self.list.blockSignals(True)
        self.list.clear()
        for idx, step_type in enumerate(step_types):
            self._insert_row(idx, step_type)
        self.list.blockSignals(False)

    def _insert_row(self, idx: int, step_type: ImageProcessingStepType):
        item = QtWidgets.QListWidgetItem()
        # Guarda step_type pra reordenamento
        item.setData(QtCore.Qt.ItemDataRole.UserRole, step_type)
        self.list.addItem(item)
        row_widget = _StepRow(step_type, idx)
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
        """Drag-reorder: emite a nova ordem como lista de indices
        originais. Ex: se a ordem virou [2, 0, 1], significa que o
        antigo idx 2 foi pra posição 0, antigo 0 pra 1, antigo 1 pra 2.
        """
        # Reconstrói a ordem lendo userData de cada item
        # (step_types) e mapeando pra indices da ordem anterior — mas
        # como `populate` é chamado pelo controller após cada
        # mudança, basta sinalizar a ordem atual visível.
        new_order = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            new_order.append(
                item.data(QtCore.Qt.ItemDataRole.UserRole)
            )
        # Emite o delta (inicio, dest_row) pra controller fazer
        # reorder no pipeline. Convertendo pra simples sinal de
        # "lista mudou de ordem" — controller refaz com base em
        # populate→sync.
        self.steps_reordered.emit(new_order)
