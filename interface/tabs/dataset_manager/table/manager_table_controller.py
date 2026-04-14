from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMenu, QInputDialog, QMessageBox
from PyQt6.QtCore import Qt

# Imports locais
from .manager_table_view import ManagerTableView
from .dataset_table_model import DatasetTableModel
from .replace_dialog.replace_dialog_controller import ReplaceDialogController

class ManagerTableController(QtCore.QObject):
    """
    Controlador responsável pela Tabela de Datasets.
    Gerencia seleção por CÉLULA (Excel-like), menu de contexto e edição.
    """
    
    face_selected = QtCore.pyqtSignal(object) 

    def __init__(self, parent_layout=None):
        super().__init__()
        self.view = ManagerTableView()
        self.model = DatasetTableModel()
        self.view.setModel(self.model)

        # --- MUDANÇA 1: PERMITIR SELEÇÃO DE CÉLULAS INDIVIDUAIS ---
        # Antes estava SelectRows (Linha inteira). Agora é SelectItems (Célula).
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        
        # Mantém ExtendedSelection (Ctrl+Click, Shift+Click, Arrastar mouse)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        # Context Menu
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._open_context_menu)

        if parent_layout:
            parent_layout.addWidget(self.view)

        self._connect_signals()

    def _connect_signals(self):
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    # =========================================================================
    # LÓGICA DE DADOS
    # =========================================================================

    def load_dataframe(self, df):
        self.model.load_dataframe(df)
        self.view.resizeColumnsToContents()

        # Reconecta sinais (necessário após setModel)
        try:
            self.view.selectionModel().selectionChanged.disconnect()
        except TypeError:
            pass
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    def get_current_dataframe(self):
        return self.model.get_dataframe()

    def _on_selection_changed(self, selected=None, deselected=None):
        """
        Disparado quando o usuário clica/navega na tabela.
        Agora adaptado para seleção de células.
        """
        # --- MUDANÇA 2: PEGAR LINHA A PARTIR DA CÉLULA ---
        # selectedRows() retorna vazio se o modo for SelectItems e a linha não estiver TODA selecionada.
        # Por isso usamos selectedIndexes() e pegamos a linha do primeiro item.
        indexes = self.view.selectionModel().selectedIndexes()
        
        if indexes:
            # Pega a linha do último item clicado (current) ou do primeiro da lista
            current_idx = self.view.currentIndex()
            if current_idx.isValid():
                row_idx = current_idx.row()
            else:
                row_idx = indexes[0].row()
            
            # Atualiza o Painel de Detalhes
            row_data = self.model.get_row_data(row_idx)
            self.face_selected.emit(row_data)
        else:
            self.face_selected.emit(None)

    # =========================================================================
    # CONTEXT MENU
    # =========================================================================

    def _open_context_menu(self, position):
        if self.model.rowCount() == 0: return

        # Índice da coluna clicada
        col_idx = self.view.columnAt(position.x())
        if col_idx == -1: return 

        col_enum = self.model.visible_columns[col_idx]
        is_editable = col_enum.editable
        col_label = col_enum.label

        menu = QMenu()
        
        # Ação de selecionar coluna inteira
        action_sel_col = menu.addAction(f"Selecionar Toda Coluna '{col_label}'")
        action_sel_col.triggered.connect(lambda: self._action_select_whole_column(col_idx))

        menu.addSeparator()

        if is_editable:
            # Mostra a quantidade de células selecionadas para dar feedback ao usuário
            indexes = self.view.selectedIndexes()
            count_sel = len(indexes)
            
            txt_set = f"✏️ Definir Valor ({count_sel} células)..."
            txt_rep = f"🔄 Substituir ({count_sel} células)..."

            action_set_val = menu.addAction(txt_set)
            action_set_val.triggered.connect(self._action_set_value)

            action_replace = menu.addAction(txt_rep)
            action_replace.triggered.connect(self._action_replace_value)
        else:
            act = menu.addAction(f"🔒 Coluna '{col_label}' não é editável")
            act.setEnabled(False)

        menu.exec(self.view.viewport().mapToGlobal(position))

    # --- AÇÕES ---

    def _action_select_whole_column(self, col_idx):
        """Seleciona verticalmente todas as células daquela coluna."""
        from PyQt6.QtCore import QItemSelection, QItemSelectionModel
        
        top_left = self.model.index(0, col_idx)
        bottom_right = self.model.index(self.model.rowCount() - 1, col_idx)
        
        selection = QItemSelection(top_left, bottom_right)
        self.view.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _action_set_value(self):
        indexes = self.view.selectedIndexes()
        if not indexes: return

        # Confirma qual coluna estamos editando (baseado na primeira célula)
        col_enum = self.model.visible_columns[indexes[0].column()]
        
        # Verifica se o usuário selecionou células de múltiplas colunas (pode ser perigoso)
        unique_cols = set(i.column() for i in indexes)
        if len(unique_cols) > 1:
            QMessageBox.warning(self.view, "Atenção", "Você selecionou células de colunas diferentes.\nA edição será aplicada em TODAS elas.")

        new_val, ok = QInputDialog.getText(
            self.view, 
            "Definir Valor", 
            f"Novo valor para {len(indexes)} células:"
        )
        if ok:
            self._apply_batch_edit(indexes, new_val)

    def _action_replace_value(self):
        indexes = self.view.selectedIndexes()
        if not indexes: return

        dialog_ctrl = ReplaceDialogController(parent_view=self.view)
        
        if dialog_ctrl.exec():
            find_text, replace_text, exact_match = dialog_ctrl.get_data()
            self._apply_batch_replace(indexes, find_text, replace_text, exact_match)

    # --- WORKERS ---

    def _apply_batch_edit(self, indexes, new_value):
        self.model.layoutAboutToBeChanged.emit()
        count = 0
        for idx in indexes:
            # Tenta editar. Se a coluna da célula não for editável (validação no Model), retorna False.
            if self.model.setData(idx, new_value, Qt.ItemDataRole.EditRole):
                count += 1
        self.model.layoutChanged.emit()
        
    def _apply_batch_replace(self, indexes, find_text, replace_text, exact_match):
        self.model.layoutAboutToBeChanged.emit()
        count = 0
        
        for idx in indexes:
            # Pega valor visual atual
            current_val = str(idx.data(Qt.ItemDataRole.DisplayRole))
            should_replace = False
            final_val = current_val

            if exact_match:
                if current_val == find_text:
                    should_replace = True
                    final_val = replace_text
            else:
                if find_text in current_val:
                    should_replace = True
                    final_val = current_val.replace(find_text, replace_text)
            
            if should_replace:
                if self.model.setData(idx, final_val, Qt.ItemDataRole.EditRole):
                    count += 1
                    
        self.model.layoutChanged.emit()
        
        if count > 0:
            QMessageBox.information(self.view, "Substituição", f"{count} registros atualizados.")
        else:
            QMessageBox.information(self.view, "Substituição", "Nenhuma ocorrência encontrada.")