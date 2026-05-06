import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt

from faceaudit.enums.dataset_columns import DatasetColumns


@dataclass(frozen=True)
class _ColumnView:
    """Adaptador interno: representa uma coluna visível na tabela com os
    três atributos que o model usa (key, label, editable). Construído tanto
    a partir de ColumnSpec (caminho novo, schema-by-role) quanto do enum
    DatasetColumns (fallback legacy).
    """
    key: str
    label: str
    editable: bool
    order: int = 0  # apenas pra ordenação determinística do fallback


def _columns_from_specs(specs: Dict[str, Any]) -> List[_ColumnView]:
    """Constrói _ColumnView a partir de um dict de ColumnSpec.

    Filtra por `visible_default=True`, preserva a ordem de inserção do dict
    (que reflete o catálogo + customizações do usuário).
    """
    out: List[_ColumnView] = []
    for i, (name, spec) in enumerate(specs.items()):
        if not getattr(spec, "visible_default", False):
            continue
        out.append(_ColumnView(
            key=name,
            label=getattr(spec, "label", name),
            editable=bool(getattr(spec, "editable", False)),
            order=i,
        ))
    return out


def _columns_from_enum() -> List[_ColumnView]:
    """Fallback: deriva _ColumnView do enum DatasetColumns.

    Usado quando o FaceDataset carregado não tem column_specs (parquet
    legado pré-PR 3 ou dataset construído na mão).
    """
    return [
        _ColumnView(key=c.key, label=c.label, editable=c.editable, order=c.order)
        for c in DatasetColumns.get_visible_columns()
    ]


class DatasetTableModel(QtCore.QAbstractTableModel):
    """Modelo de tabela schema-by-role.

    A lista de colunas visíveis pode vir de duas fontes (em ordem de
    prioridade):
      1. `column_specs` injetados via `set_column_specs(specs)` — caminho
         novo; pós-PR 3 todo FaceDataset carregado tem este dict populado.
      2. `DatasetColumns` enum — fallback para parquets legados ou
         datasets criados em memória sem specs.

    O resto do contrato (rowCount/columnCount/data/setData/sort) lê dos
    `_ColumnView` resolvidos no momento da escolha.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None):
        super().__init__()
        self._df = data if data is not None else pd.DataFrame()
        self._column_specs: Dict[str, Any] = {}
        self.visible_columns: List[_ColumnView] = _columns_from_enum()

    def set_column_specs(self, specs: Dict[str, Any]) -> None:
        """Atualiza a lista de colunas visíveis a partir do dict de specs.

        Quando `specs` é vazio (None ou {}), volta ao fallback do enum.
        Chamar isto sempre que um dataset novo for carregado.
        """
        self.beginResetModel()
        self._column_specs = dict(specs) if specs else {}
        if self._column_specs:
            self.visible_columns = _columns_from_specs(self._column_specs)
        else:
            self.visible_columns = _columns_from_enum()
        self.endResetModel()

    @property
    def column_specs(self) -> Dict[str, Any]:
        return self._column_specs

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return len(self.visible_columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        col = self.visible_columns[index.column()]
        key = col.key

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if key not in self._df.columns:
                return ""

            val = self._df.iloc[index.row()][key]

            if isinstance(val, (list, tuple, np.ndarray)):
                if hasattr(val, "shape"):
                    return f"[Array {val.shape}]"
                return f"[List len={len(val)}]"

            if pd.isna(val) or val is None:
                return ""

            if isinstance(val, (float, np.floating)):
                return f"{val:.2f}"

            return str(val)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                if 0 <= section < len(self.visible_columns):
                    return self.visible_columns[section].label
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        col = self.visible_columns[index.column()]
        if col.editable:
            base_flags |= Qt.ItemFlag.ItemIsEditable
        return base_flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """Permite edição da célula (ex: renomear identidade)."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        col = self.visible_columns[index.column()]
        if not col.editable:
            return False

        # Vocabulário fechado (post-PR 4): se o spec da coluna define
        # `vocabulary`, valida que o novo valor pertence à lista. Valor
        # inválido é rejeitado silenciosamente — a UI deveria usar
        # ComboBoxDelegate pra evitar isso, mas a guarda é defensiva.
        spec = self._column_specs.get(col.key) if self._column_specs else None
        vocab = getattr(spec, "vocabulary", None) if spec else None
        if vocab is not None and value not in (None, "") and str(value) not in vocab:
            print(f"⚠ Valor '{value}' fora do vocabulário de '{col.key}': {list(vocab)}")
            return False

        try:
            self._df.iloc[index.row(), self._df.columns.get_loc(col.key)] = value
            self.dataChanged.emit(
                index, index,
                [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole],
            )
            return True
        except Exception as e:
            print(f"Erro ao editar célula: {e}")
            return False

    def sort(self, column, order):
        """Ordenação baseada no _ColumnView selecionado."""
        self.layoutAboutToBeChanged.emit()

        col = self.visible_columns[column]
        key = col.key
        ascending = (order == Qt.SortOrder.AscendingOrder)

        if key in self._df.columns:
            try:
                self._df = self._df.sort_values(by=key, ascending=ascending)
            except Exception as e:
                print(f"Aviso: Falha ao ordenar por {key}: {e}")

        self.layoutChanged.emit()

    def get_row_data(self, row_idx):
        """Retorna a linha completa (Series) do DataFrame subjacente."""
        if 0 <= row_idx < self.rowCount():
            return self._df.iloc[row_idx]
        return None

    def get_dataframe(self):
        return self._df

    def load_dataframe(self, df):
        """Método helper para trocar o DF inteiro."""
        self.beginResetModel()
        self._df = df if df is not None else pd.DataFrame()
        self.endResetModel()
