"""Delegate que troca o editor de célula por um QComboBox quando a coluna
tem vocabulário fechado (ColumnSpec.vocabulary != None).

Aplicado pelo `ManagerTableController` quando o model recebe column_specs
via `set_column_specs`. Funciona junto com o guard em `setData` do model:
o delegate evita o erro de UX (usuário digitando valor inválido); o guard
do model é a defesa final.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from PyQt6 import QtCore, QtWidgets


class VocabularyComboDelegate(QtWidgets.QStyledItemDelegate):
    """Delegate que renderiza um QComboBox como editor de célula.

    Recebe o vocabulário (tupla de strings) para popular o dropdown.
    Quando aplicado a uma coluna específica do model, todas as células
    daquela coluna ficam editáveis via combo box em vez de QLineEdit.
    """

    def __init__(self, vocabulary: Tuple[str, ...], parent=None):
        super().__init__(parent)
        self._vocabulary = tuple(vocabulary)

    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(self._vocabulary)
        # Editable=False força o usuário a escolher do dropdown
        combo.setEditable(False)
        return combo

    def setEditorData(self, editor: QtWidgets.QComboBox, index):
        current = index.data(QtCore.Qt.ItemDataRole.EditRole)
        if current is None:
            return
        text = str(current)
        idx = editor.findText(text)
        if idx >= 0:
            editor.setCurrentIndex(idx)

    def setModelData(self, editor: QtWidgets.QComboBox, model, index):
        value = editor.currentText()
        model.setData(index, value, QtCore.Qt.ItemDataRole.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


def install_vocabulary_delegates(view: QtWidgets.QTableView, model) -> None:
    """Instala VocabularyComboDelegate em todas as colunas do model que
    têm `ColumnSpec.vocabulary` não-nulo. Idempotente — chamável após cada
    mudança de specs.
    """
    if not hasattr(model, "visible_columns") or not hasattr(model, "column_specs"):
        return

    specs: Dict[str, Any] = getattr(model, "column_specs", {}) or {}
    if not specs:
        # Sem specs (fallback enum) → nada a instalar.
        return

    for col_idx, col_view in enumerate(model.visible_columns):
        spec = specs.get(col_view.key)
        vocab = getattr(spec, "vocabulary", None) if spec else None
        if vocab:
            view.setItemDelegateForColumn(col_idx, VocabularyComboDelegate(vocab, view))
