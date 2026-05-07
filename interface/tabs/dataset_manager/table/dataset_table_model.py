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

    def __init__(self, data: Optional[pd.DataFrame] = None,
                 tooltip_cache: Optional[Any] = None):
        super().__init__()
        self._df = data if data is not None else pd.DataFrame()
        self._column_specs: Dict[str, Any] = {}
        self.visible_columns: List[_ColumnView] = _columns_from_enum()
        # PR 15.1: cache opcional pra hover-thumb tooltip. Quando None,
        # os tooltips são strings normais (path completo).
        self._tooltip_cache = tooltip_cache

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

    def set_column_visibility(self, name: str, visible: bool) -> bool:
        """Toggle visibility de uma coluna do dict de specs.

        Funciona apenas no caminho schema-by-role (specs populados). No
        fallback do enum, retorna False sem fazer nada — visibilidade do
        enum é estática.

        Modifica `column_specs` IN-PLACE (sem recriar o spec) — propaga
        para o `FaceDataset.column_specs` que o controller mantém. Após
        a mudança, dispara reset do model.
        """
        if not self._column_specs:
            return False
        spec = self._column_specs.get(name)
        if spec is None:
            return False
        # ColumnSpec é frozen (dataclass) → reconstrói com o novo valor.
        # Usa o `dataclasses.replace` para não duplicar lógica.
        try:
            import dataclasses
            self._column_specs[name] = dataclasses.replace(spec, visible_default=bool(visible))
        except Exception:
            return False

        self.beginResetModel()
        self.visible_columns = _columns_from_specs(self._column_specs)
        self.endResetModel()
        return True

    def update_column_spec(self, name: str, new_spec: Any) -> bool:
        """Substitui o spec de uma coluna existente pelo novo.

        Usado pelo ColumnPropertiesDialog. Reseta o model porque label
        (cabeçalho) ou vocabulary (delegate) podem ter mudado. Não toca
        no DataFrame — só nos metadados.
        """
        if not self._column_specs or name not in self._column_specs:
            return False
        if new_spec is None:
            return False
        self._column_specs[name] = new_spec

        self.beginResetModel()
        self.visible_columns = _columns_from_specs(self._column_specs)
        self.endResetModel()
        return True

    # Roles cujo valor não pode sumir do parquet sem quebrar a leitura
    # como FaceDataset/EmbeddingDataset (ColumnRole.IDENTITY/IMAGE_PATH são
    # marcadas como "singleton, obrigatória" no column_catalog).
    _ESSENTIAL_ROLES = frozenset({"identity", "image_path"})

    def can_remove_column(self, name: str) -> bool:
        """Indica se a coluna é removível pela UI.

        - Essenciais (identity, image_path) → False, nunca removíveis.
        - Demais (CUSTOM, EMBEDDING, QUALITY, POSE, DEMOGRAPHIC, etc.) → True.
          A consequência semântica de remover (ex: estratégias quebrarem
          por falta da embedding_column) é capturada pelas regras de
          validação na execução, não pelo bloqueio de UI.
        """
        if not self._column_specs or name not in self._column_specs:
            return False
        spec = self._column_specs[name]
        role = getattr(spec, "role", None)
        return getattr(role, "value", None) not in self._ESSENTIAL_ROLES

    def is_canonical_column(self, name: str) -> bool:
        """Retorna True se a coluna não é CUSTOM — usada pela UI pra
        mostrar confirmação reforçada antes de apagar."""
        spec = self._column_specs.get(name) if self._column_specs else None
        role = getattr(spec, "role", None) if spec else None
        role_value = getattr(role, "value", None)
        return bool(role_value) and role_value != "custom"

    def remove_column(self, name: str) -> bool:
        """Remove uma coluna do dataset (spec + DataFrame).

        Bloqueia colunas essenciais (IDENTITY, IMAGE_PATH). Aceita
        CUSTOM e canônicas (FF/OFIQ/SERFIQ/embeddings) — a checagem
        semântica de impacto em estratégias é feita pela
        StrategyEmbeddingConsistencyRule e amigas em tempo de execução.
        """
        if not self.can_remove_column(name):
            return False

        self._column_specs.pop(name, None)
        if name in self._df.columns:
            self._df = self._df.drop(columns=[name])

        self.beginResetModel()
        self.visible_columns = _columns_from_specs(self._column_specs)
        self.endResetModel()
        return True

    # Retrocompat — antigos callers que chamavam apenas pra colunas custom
    # continuam tendo o mesmo comportamento (rejeita canônicas).
    def remove_custom_column(self, name: str) -> bool:
        if not self._column_specs or name not in self._column_specs:
            return False
        spec = self._column_specs[name]
        role = getattr(spec, "role", None)
        if getattr(role, "value", None) != "custom":
            return False
        return self.remove_column(name)

    def add_custom_column(
        self,
        name: str,
        label: str,
        dtype: str = "str",
        default_value: Any = None,
        editable: bool = True,
    ) -> bool:
        """Adiciona uma nova coluna custom ao dataset.

        Cria um ColumnSpec com role=custom e popula a coluna no DataFrame
        com o `default_value`. Se a coluna já existir, retorna False.

        Apenas no caminho schema-by-role (specs populados). No fallback do
        enum, retorna False — não faz sentido adicionar coluna a um schema
        legado sem metadata.
        """
        if not self._column_specs:
            return False
        if not name or name in self._column_specs:
            return False

        from faceaudit.models.column_catalog import (
            ColumnSpec, ColumnRole, ColumnSource, ColumnDType, CATEGORY_CUSTOM,
        )

        # Mapeia string → ColumnDType (defensivo)
        try:
            dtype_enum = ColumnDType(dtype)
        except ValueError:
            dtype_enum = ColumnDType.STR

        spec = ColumnSpec(
            name=name,
            label=label or name,
            role=ColumnRole.CUSTOM,
            source=ColumnSource.CUSTOM,
            dtype=dtype_enum,
            editable=editable,
            visible_default=True,
            category=CATEGORY_CUSTOM,
            default_value=default_value,
        )
        self._column_specs[name] = spec

        # Adiciona a coluna ao DataFrame com default_value em todas as linhas.
        if name not in self._df.columns:
            self._df[name] = default_value

        self.beginResetModel()
        self.visible_columns = _columns_from_specs(self._column_specs)
        self.endResetModel()
        return True

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

        # PR 15.1: hover preview da face. Qualquer célula da linha mostra
        # a thumbnail, com fallback pro path quando o cache não conhece
        # ou o arquivo está quebrado. Renderização rich-text pelo Qt
        # automaticamente quando a string começa com `<`.
        if role == Qt.ItemDataRole.ToolTipRole:
            if "image_path" not in self._df.columns:
                return None
            try:
                path = self._df.iloc[index.row()]["image_path"]
            except (IndexError, KeyError):
                return None
            if not path or pd.isna(path):
                return None
            path_str = str(path)
            if self._tooltip_cache is not None:
                html = self._tooltip_cache.get_html(path_str)
                if html:
                    return html
            # Fallback: quando o cache falha (path quebrado) ou não foi
            # injetado, mostra o path puro — comportamento de tooltip
            # padrão pré-PR 15.1.
            return path_str

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
