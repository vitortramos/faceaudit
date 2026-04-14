import pandas as pd
import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from faceaudit.enums.dataset_columns import DatasetColumns

class DatasetTableModel(QtCore.QAbstractTableModel):
    """
    Modelo de Tabela baseado no Schema (Enum DatasetColumn).
    Mostra apenas colunas configuradas como visíveis e permite edição controlada.
    """
    def __init__(self, data=pd.DataFrame()):
        super().__init__()
        self._df = data
        # Carrega a "Receita" de colunas visíveis
        self.visible_columns = DatasetColumns.get_visible_columns()

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        # O número de colunas agora é definido pelo Enum, não pelo DataFrame
        return len(self.visible_columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        
        # Identifica qual coluna do Enum estamos renderizando
        col_enum = self.visible_columns[index.column()]
        key = col_enum.key

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # Se a coluna não existir no DF
            if key not in self._df.columns:
                return ""

            val = self._df.iloc[index.row()][key]

            # 1. Tratamento de Tipos Complexos (Arrays/Listas)
            if isinstance(val, (list, tuple, np.ndarray)):
                if hasattr(val, 'shape'): return f"[Array {val.shape}]"
                return f"[List len={len(val)}]"

            # 2. Tratamento de Nulos
            if pd.isna(val) or val is None:
                return ""
            
            # 3. --- FORMATAÇÃO DE DECIMAIS (A MUDANÇA ESTÁ AQUI) ---
            # Verifica se é float (Python ou Numpy) e corta para 2 casas
            if isinstance(val, (float, np.floating)):
                return f"{val:.2f}"

            return str(val)

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                # Usa o Label amigável do Enum (ex: "Idade (Est.)")
                return self.visible_columns[section].label
            
            if orientation == Qt.Orientation.Vertical:
                return str(section + 1)
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags

        base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        
        # Verifica se a coluna é editável no Enum
        col_enum = self.visible_columns[index.column()]
        if col_enum.editable:
            base_flags |= Qt.ItemFlag.ItemIsEditable
            
        return base_flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        """Permite edição da célula (ex: renomear identidade)."""
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        col_enum = self.visible_columns[index.column()]
        
        # Segurança extra: confere se é editável
        if not col_enum.editable:
            return False

        # Atualiza o DataFrame
        try:
            self._df.iloc[index.row(), self._df.columns.get_loc(col_enum.key)] = value
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole])
            return True
        except Exception as e:
            print(f"Erro ao editar célula: {e}")
            return False

    def sort(self, column, order):
        """Ordenação baseada no Enum."""
        self.layoutAboutToBeChanged.emit()
        
        col_enum = self.visible_columns[column]
        key = col_enum.key
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