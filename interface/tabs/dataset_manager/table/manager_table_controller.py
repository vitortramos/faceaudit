from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import QMenu, QInputDialog, QMessageBox
from PyQt6.QtCore import Qt

# Imports locais
from .manager_table_view import ManagerTableView
from .dataset_table_model import DatasetTableModel
from .replace_dialog.replace_dialog_controller import ReplaceDialogController
from .tooltip_thumb_cache import TooltipThumbCache
from .vocabulary_combo_delegate import install_vocabulary_delegates
from interface.wait_cursor import wait_cursor


class ManagerTableController(QtCore.QObject):
    """
    Controlador responsável pela Tabela de Datasets.
    Gerencia seleção por CÉLULA (Excel-like), menu de contexto e edição.
    """

    face_selected = QtCore.pyqtSignal(object)
    # Sinal emitido quando o schema do dataset muda em runtime (visibilidade
    # de coluna, adição de coluna custom, etc.). O controller pai escuta
    # para marcar dirty + atualizar `FaceDataset.column_specs`.
    schema_changed = QtCore.pyqtSignal()
    # PR 15.2: usuário pediu comparação de par via context menu da tabela
    # (apenas quando 2 linhas estão selecionadas). Emite os dois face_idx
    # (índices do dataframe = mesma indexação de FaceDataset.faces).
    compare_pair_requested = QtCore.pyqtSignal(int, int)

    def __init__(self, parent_layout=None):
        super().__init__()
        self.view = ManagerTableView()
        # PR 15.1: cache de tooltip-thumb compartilhado pelo model;
        # invalidado em cada `load_dataframe` quando o dataset muda.
        self._tooltip_cache = TooltipThumbCache()
        self.model = DatasetTableModel(tooltip_cache=self._tooltip_cache)
        self.view.setModel(self.model)

        # --- MUDANÇA 1: PERMITIR SELEÇÃO DE CÉLULAS INDIVIDUAIS ---
        self.view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.view.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        # Context menu (células)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._open_context_menu)

        # Context menu do header (show/hide colunas) — schema-by-role
        header = self.view.horizontalHeader()
        header.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        header.customContextMenuRequested.connect(self._open_header_context_menu)

        # PR 20 (Tema 4.1) — drag-reorder. `sectionMoved` dispara várias
        # vezes durante um drag (visual updates intermediários). Usamos
        # `_reorder_pending` + `QTimer.singleShot(0, ...)` pra debounçar
        # e processar uma única vez no fim do drag, fora do callback.
        self._reorder_pending = False
        header.sectionMoved.connect(self._on_section_moved)

        if parent_layout:
            parent_layout.addWidget(self.view)

        self._connect_signals()

    def set_dataset(self, dataset):
        """Propaga dataset pro model — necessário pra resolver paths
        em modo RELATIVE antes do hover-tooltip carregar a thumbnail.
        """
        if hasattr(self.model, "set_dataset"):
            self.model.set_dataset(dataset)

    def apply_column_specs(self, specs):
        """Hook chamado pelo controller pai quando um dataset com specs
        é carregado. Atualiza o model + instala os delegates de vocabulário.
        """
        self.model.set_column_specs(specs)
        install_vocabulary_delegates(self.view, self.model)
        self.view.resizeColumnsToContents()

    def _connect_signals(self):
        self.view.selectionModel().selectionChanged.connect(self._on_selection_changed)

    # =========================================================================
    # LÓGICA DE DADOS
    # =========================================================================

    def load_dataframe(self, df):
        # PR 15.1: dataset trocou — esvazia o cache de tooltip-thumbs.
        # Cache é populado lazy em hover; nada urgente aqui, só evita
        # memória stale acumulada de datasets anteriores.
        self._tooltip_cache.clear()

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

    # =========================================================================
    # HEADER CONTEXT MENU — show/hide colunas (PR 8)
    # =========================================================================

    def _open_header_context_menu(self, position):
        """Menu de contexto do header da tabela.

        Quando o schema-by-role está ativo (model.column_specs não vazio),
        mostra:
          1. Ações pra coluna clicada (Propriedades, Remover se custom)
          2. Sub-menu "Mostrar/Ocultar" agrupado por categoria
        Em fallback ao enum, mostra mensagem explicativa.
        """
        specs = self.model.column_specs
        header = self.view.horizontalHeader()
        if not specs:
            menu = QMenu()
            act = menu.addAction("(legacy) Schema fixo — sem column_specs")
            act.setEnabled(False)
            menu.exec(header.mapToGlobal(position))
            return

        # Identifica a coluna clicada (logical index dentro de visible_columns)
        clicked_logical = header.logicalIndexAt(position)
        clicked_view = (
            self.model.visible_columns[clicked_logical]
            if 0 <= clicked_logical < len(self.model.visible_columns)
            else None
        )

        menu = QMenu()

        if clicked_view is not None:
            spec = specs.get(clicked_view.key)
            label = getattr(spec, "label", clicked_view.key) if spec else clicked_view.key
            head = menu.addAction(f"🎯 Coluna: {label}")
            head.setEnabled(False)

            act_props = menu.addAction("⚙️  Propriedades…")
            act_props.triggered.connect(
                lambda _checked=False, n=clicked_view.key: self._on_open_properties(n)
            )

            # Remover: aceita CUSTOM e canônicas (FF/OFIQ/SERFIQ/embeddings).
            # IDENTITY e IMAGE_PATH ficam protegidas porque sem elas o
            # parquet não carrega como FaceDataset.
            can_remove = self.model.can_remove_column(clicked_view.key)
            act_remove = menu.addAction("🗑️  Remover Coluna")
            act_remove.setEnabled(can_remove)
            if not can_remove:
                act_remove.setToolTip(
                    "Colunas essenciais (Identidade, Caminho da Imagem) não "
                    "podem ser removidas — sem elas o dataset não pode ser "
                    "carregado."
                )
            act_remove.triggered.connect(
                lambda _checked=False, n=clicked_view.key: self._on_remove_column(n)
            )

            menu.addSeparator()

        # Sub-menu de visibilidade agrupado por categoria
        vis_menu = menu.addMenu("📋 Mostrar/Ocultar Colunas")

        category_order = []
        by_category = {}
        for name, spec in specs.items():
            cat = getattr(spec, "category", None) or "Outras"
            if cat not in by_category:
                by_category[cat] = []
                category_order.append(cat)
            by_category[cat].append((name, spec))

        for cat in category_order:
            cat_header = vis_menu.addAction(f"— {cat} —")
            cat_header.setEnabled(False)
            for name, spec in by_category[cat]:
                visible = bool(getattr(spec, "visible_default", True))
                act = vis_menu.addAction(getattr(spec, "label", name))
                act.setCheckable(True)
                act.setChecked(visible)
                act.toggled.connect(
                    lambda checked, n=name: self._on_visibility_toggled(n, checked)
                )
            vis_menu.addSeparator()

        menu.exec(header.mapToGlobal(position))

    def _on_open_properties(self, name: str):
        """Abre o diálogo de propriedades pra coluna `name` e aplica o
        resultado via model.update_column_spec."""
        from ..column_dialogs.column_properties_dialog import ColumnPropertiesDialog

        spec = self.model.column_specs.get(name)
        if spec is None:
            return
        dialog = ColumnPropertiesDialog(spec, parent=self.view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        new_spec = dialog.apply_to_spec(spec)
        if self.model.update_column_spec(name, new_spec):
            install_vocabulary_delegates(self.view, self.model)
            self.view.resizeColumnsToContents()
            self.schema_changed.emit()

    def _on_remove_column(self, name: str):
        """Confirma e remove uma coluna do dataset.

        Confirmação simples para CUSTOM, reforçada para canônicas
        (FF/OFIQ/SERFIQ/embedding) que podem ser referenciadas por
        estratégias.
        """
        from PyQt6.QtWidgets import QMessageBox
        spec = self.model.column_specs.get(name)
        label = getattr(spec, "label", name) if spec else name

        if self.model.is_canonical_column(name):
            text = (
                f"Remover a coluna '{label}'?\n\n"
                "⚠ Esta é uma coluna canônica (vinda de ForensicFace/OFIQ/"
                "SER-FIQ ou embedding). Estratégias que dependam dela vão "
                "sinalizar erro de validação até serem reconfiguradas para "
                "outra coluna ou o dataset ser re-enriquecido.\n\n"
                "Os dados serão perdidos. Esta ação só persiste ao salvar "
                "o dataset."
            )
        else:
            text = (
                f"Remover a coluna '{label}'?\n\n"
                "Os dados desta coluna serão perdidos. Esta ação só será "
                "persistida ao salvar o dataset."
            )

        confirm = QMessageBox.question(
            self.view, "Remover Coluna", text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            defaultButton=QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        if self.model.remove_column(name):
            install_vocabulary_delegates(self.view, self.model)
            self.view.resizeColumnsToContents()
            self.schema_changed.emit()

    def _on_visibility_toggled(self, name: str, checked: bool):
        """Callback do checkbox no header context menu."""
        if self.model.set_column_visibility(name, checked):
            # Reinstala delegates pq column_specs mudou (visible reset → idx changed)
            install_vocabulary_delegates(self.view, self.model)
            self.view.resizeColumnsToContents()
            self.schema_changed.emit()

    # =========================================================================
    # DRAG-REORDER (PR 20 — Tema 4.1)
    # =========================================================================

    def _on_section_moved(self, _logical_index: int, _old_visual: int, _new_visual: int):
        """Disparado pelo header sempre que uma coluna é arrastada.

        Debouncing via `_reorder_pending`: drags emitem múltiplos
        `sectionMoved` em sequência (snap visual). Processamos só uma vez
        no próximo tick do event loop, fora do callback do drag — evita
        re-entrância no `beginResetModel/endResetModel`.

        No fallback do enum (sem column_specs), drag continua funcionando
        visualmente mas não é persistido — o `reorder_columns` retorna
        False sem fazer nada.
        """
        if not self.model.column_specs:
            return
        if self._reorder_pending:
            return
        self._reorder_pending = True
        QtCore.QTimer.singleShot(0, self._apply_column_reorder)

    def _apply_column_reorder(self):
        """Lê a ordem visual atual do header e reordena `column_specs`.

        Após o reset do model, a ordem dict↔visual volta a coincidir
        (logical = visual) — drags subsequentes começam de um estado
        limpo.
        """
        self._reorder_pending = False
        header = self.view.horizontalHeader()
        n = self.model.columnCount()
        visible = self.model.visible_columns

        new_order = []
        for visual in range(n):
            logical = header.logicalIndex(visual)
            if 0 <= logical < len(visible):
                new_order.append(visible[logical].key)

        if not self.model.reorder_columns(new_order):
            return
        install_vocabulary_delegates(self.view, self.model)
        self.view.resizeColumnsToContents()
        self.schema_changed.emit()

    # =========================================================================
    # CELL CONTEXT MENU
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

        # PR 15.2: comparar par habilitado quando exatamente 2 linhas
        # estão selecionadas. Independe da editabilidade da coluna —
        # qualquer célula da linha serve como hover-target.
        rows_selected = sorted({i.row() for i in self.view.selectedIndexes()})
        menu.addSeparator()
        if len(rows_selected) == 2:
            row_a, row_b = rows_selected
            act_cmp = menu.addAction("⚖️ Comparar este par lado a lado…")
            act_cmp.triggered.connect(
                lambda _checked=False, a=row_a, b=row_b:
                    self.compare_pair_requested.emit(a, b)
            )
        else:
            act_cmp = menu.addAction(
                f"⚖️ Comparar par (selecione 2 linhas — {len(rows_selected)})"
            )
            act_cmp.setEnabled(False)

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
            # WaitCursor cobre o batch — em N células grande
            # (coluna inteira de 1M rows) leva segundos. Sem isso,
            # o clique no OK do dialog parece "engasgado".
            with wait_cursor():
                self._apply_batch_edit(indexes, new_val)

    def _action_replace_value(self):
        indexes = self.view.selectedIndexes()
        if not indexes: return

        dialog_ctrl = ReplaceDialogController(parent_view=self.view)

        if dialog_ctrl.exec():
            find_text, replace_text, exact_match = dialog_ctrl.get_data()
            with wait_cursor():
                count = self._apply_batch_replace(
                    indexes, find_text, replace_text, exact_match,
                )
            # QMessageBox fora do `with` — mensagem aparece com
            # cursor normal e modal interativo.
            if count > 0:
                QMessageBox.information(
                    self.view, "Substituição",
                    f"{count} registros atualizados.",
                )
            else:
                QMessageBox.information(
                    self.view, "Substituição",
                    "Nenhuma ocorrência encontrada.",
                )

    # --- WORKERS ---

    def _apply_batch_edit(self, indexes, new_value):
        self.model.layoutAboutToBeChanged.emit()
        count = 0
        for idx in indexes:
            # Tenta editar. Se a coluna da célula não for editável (validação no Model), retorna False.
            if self.model.setData(idx, new_value, Qt.ItemDataRole.EditRole):
                count += 1
        self.model.layoutChanged.emit()

    def _apply_batch_replace(self, indexes, find_text, replace_text, exact_match) -> int:
        """Aplica substituição em batch. Retorna a contagem de células
        modificadas (caller exibe QMessageBox com o resultado FORA do
        wait_cursor pra não congelar o cursor durante a interação)."""
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
        return count