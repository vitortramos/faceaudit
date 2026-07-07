import os
from pathlib import Path
from typing import Optional
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QApplication

# --- COMPONENTES VISUAIS (SUB-CONTROLLERS) ---
from .toolbar.manager_toolbar_controller import ManagerToolbarController
from .table.manager_table_controller import ManagerTableController
from .detail_panel.detail_panel_controller import DetailPanelController

# --- DIÁLOGOS DE FUNCIONALIDADE ---
from .new_dataset_dialog.new_dataset_controller import NewDatasetController
from .filter_dialog.filter_dialog_controller import FilterDialogController
from .stats_dialog.stats_dialog_controller import StatsDialogController
from .compare_dialog.compare_dialog_controller import CompareDialogController
from .enrich_dataset_dialog.enrich_dataset_controller import EnrichDatasetController
from .split_dataset_dialog.split_dataset_dialog_controller import SplitDatasetDialogController

# [NOVO] Controlador do Diálogo de Análise Forense
from .dataset_analysis.dataset_analysis_controller import DatasetAnalysisController

# --- WORKERS (PROCESSAMENTO) ---
from .workers.load_dataset_worker import LoadDatasetWorker
from .workers.modify_paths_worker import ModifyPathsWorker  # PR L1d
from .workers.filter_worker import FilterWorker
from .workers.new_dataset_worker import NewDatasetWorker
from .workers.enrich_worker import EnrichWorker
from .workers.worker_progress_runner import WorkerProgressRunner

# --- MODELOS ---
from faceaudit.models import FaceDataset
from faceaudit.models.filter import FilterConfig
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig

# PR 25 (Tema 5.4) — audit log forense / chain of custody
from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.enums.localization_mode import LocalizationMode
from faceaudit.enums.path_mode import PathMode
from faceaudit.services.image_path_resolver import (
    normalize_stored_path,
    resolve_image_path,
)

# Feedback visual em operações síncronas pesadas
from interface.wait_cursor import wait_cursor
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.hash_service import HashService


class DatasetManagerController(QtCore.QObject):
    """
    Controlador Mestre do Gerenciamento de Datasets.
    """
    
    dataset_changed = QtCore.pyqtSignal()
    # Save concluído sobrescrevendo o path atual (fluxo_salvar). Path é
    # absoluto. Subscriber natural: Strategy Manager V2 — quando o path
    # casa com um primary do projeto aberto, atualiza hash + cascade.
    dataset_saved_to_project = QtCore.pyqtSignal(str)
    # PR 17: emitido quando o usuário pede pra enviar o dataset corrente
    # pra V2. Carrega o caminho absoluto do parquet salvo. Capturado pelo
    # MainGUIController, que troca de aba e chama a V2.
    send_to_strategy_requested = QtCore.pyqtSignal(str)

    # RECEBE O CONTAINER AQUI
    def __init__(
        self,
        view,
        pipeline,
        config,
        container,
        ff_available: bool = True,
        ofiq_available: bool = True,
    ):
        super().__init__()
        self.view = view
        self.pipeline = pipeline
        self.config = config
        self.container = container # Guardamos a referência do container para injeção explícita
        # Flags de disponibilidade dos serviços externos. Propagadas aos
        # diálogos Novo/Enriquecer Dataset pra desabilitar checkboxes
        # quando o respectivo serviço não pôde ser inicializado (ex:
        # modelo ForensicFace não instalado, ou OFIQ exe não encontrado).
        # O resto da aba (carregar, filtrar, comparar, balancear, pipeline
        # de imagens, exportar) funciona normalmente sem esses serviços.
        self.ff_available = bool(ff_available)
        self.ofiq_available = bool(ofiq_available)

        # Estado Global
        self.dataset = None
        self.dataset_original = None
        self.current_path = None
        self.current_filter_config = FilterConfig()
        self._is_dirty = False

        # Ref forte para o runner ativo — o WorkerProgressRunner gerencia o
        # próprio QThread + QProgressDialog; só precisamos manter vivo até
        # ele finalizar (quando finished/error disparam, é descartado).
        self._active_runner: WorkerProgressRunner | None = None

        # --- 1. SUB-CONTROLADORES ---
        self.toolbar_ctrl = ManagerToolbarController(parent_layout=None)
        self.table_ctrl = ManagerTableController(parent_layout=None)
        self.detail_ctrl = DetailPanelController(parent_layout=None)

        # PR 15: galeria embutida em QStackedWidget junto com a tabela.
        # Toggle na toolbar troca o índice (0=tabela, 1=galeria).
        from .gallery.gallery_widget import GalleryWidget
        self.gallery_widget = GalleryWidget()
        self._view_stack = QtWidgets.QStackedWidget()
        self._view_stack.addWidget(self.table_ctrl.view)   # index 0
        self._view_stack.addWidget(self.gallery_widget)    # index 1

        if hasattr(self.view, 'toolbar_layout'):
            self.view.toolbar_layout.addWidget(self.toolbar_ctrl.view)
        if hasattr(self.view, 'table_layout'):
            self.view.table_layout.addWidget(self._view_stack)
        if hasattr(self.view, 'detail_layout'):
            self.view.detail_layout.addWidget(self.detail_ctrl.view)

        # --- 2. SINAIS ---
        self._connect_submodules()

    def _connect_submodules(self):
        self.toolbar_ctrl.action_new.connect(self.fluxo_novo_dataset)
        self.toolbar_ctrl.action_new_from_video.connect(self.fluxo_novo_dataset_video)
        self.toolbar_ctrl.action_import_video_run.connect(self.fluxo_importar_run_video)
        self.toolbar_ctrl.action_load.connect(self.fluxo_abrir_dataset)
        self.toolbar_ctrl.action_close.connect(self.fluxo_fechar_dataset)
        self.toolbar_ctrl.action_save.connect(self.fluxo_salvar)
        self.toolbar_ctrl.action_save_as.connect(self.fluxo_salvar_como)
        self.toolbar_ctrl.action_merge.connect(self.fluxo_merge)
        self.toolbar_ctrl.action_split.connect(self.fluxo_split)
        self.toolbar_ctrl.action_compare.connect(self.fluxo_comparar)

        self.toolbar_ctrl.action_stats.connect(self.fluxo_estatisticas)
        # [NOVO] Conecta o botão Analisar ao novo fluxo
        self.toolbar_ctrl.action_analyze.connect(self.fluxo_analise_dataset)
        # PR 24 (Tema 5.3 + 5.5) — relatório de schema + export CSV/Excel
        self.toolbar_ctrl.action_schema_report.connect(self.fluxo_schema_report)
        # PR 25 (Tema 5.4) — audit log forense
        if hasattr(self.toolbar_ctrl, "action_audit_log"):
            self.toolbar_ctrl.action_audit_log.connect(self.fluxo_audit_log)
        # PR L1d — Modificação de caminhos das imagens
        if hasattr(self.toolbar_ctrl, "action_modify_paths"):
            self.toolbar_ctrl.action_modify_paths.connect(self.fluxo_modificar_caminhos)
        # PR I2 — Editor de pipeline de imagens
        if hasattr(self.toolbar_ctrl, "action_image_pipeline_editor"):
            self.toolbar_ctrl.action_image_pipeline_editor.connect(
                self.fluxo_image_pipeline_editor
            )
        # PR I5 — Aplicar pipeline em batch
        if hasattr(self.toolbar_ctrl, "action_image_pipeline_apply"):
            self.toolbar_ctrl.action_image_pipeline_apply.connect(
                self.fluxo_aplicar_image_pipeline
            )

        self.toolbar_ctrl.action_filter.connect(self.fluxo_filtrar)
        self.toolbar_ctrl.action_clear_filter.connect(self.fluxo_limpar_filtro)
        self.table_ctrl.face_selected.connect(self.detail_ctrl.load_face_data)

        # Edits no modelo da tabela marcam dirty; sem isso, alterações de célula
        # ficavam invisíveis até um flush manual e podiam ser perdidas no fechamento.
        if hasattr(self.table_ctrl, "model"):
            self.table_ctrl.model.dataChanged.connect(self._on_table_data_changed)

        # PR 8 — Spreadsheet UX
        if hasattr(self.toolbar_ctrl, "action_add_column"):
            self.toolbar_ctrl.action_add_column.connect(self.fluxo_adicionar_coluna)
        if hasattr(self.toolbar_ctrl, "action_metadata"):
            self.toolbar_ctrl.action_metadata.connect(self.fluxo_editar_metadados)
        if hasattr(self.table_ctrl, "schema_changed"):
            self.table_ctrl.schema_changed.connect(self._on_schema_changed)

        # PR 15 — Modo galeria
        if hasattr(self.toolbar_ctrl, "action_toggle_gallery"):
            self.toolbar_ctrl.action_toggle_gallery.connect(self._on_toggle_gallery)
        # Selection sync: galeria → detail panel + tabela
        self.gallery_widget.face_selected.connect(self._on_gallery_face_selected)

        # PR 15.2 — Comparar par lado a lado (tabela + galeria emitem
        # com a mesma assinatura: compare_pair_requested(int, int))
        if hasattr(self.table_ctrl, "compare_pair_requested"):
            self.table_ctrl.compare_pair_requested.connect(self._on_compare_pair)
        self.gallery_widget.compare_pair_requested.connect(self._on_compare_pair)

        # Ações de seleção do menu de contexto da galeria (N faces).
        self.gallery_widget.remove_faces_requested.connect(self._on_gallery_remove_faces)
        self.gallery_widget.create_dataset_from_selection_requested.connect(
            self._on_gallery_create_dataset_from_selection)
        self.gallery_widget.set_column_value_requested.connect(
            self._on_gallery_set_column_value)

        # PR 16 — Adicionar à Biblioteca Global
        if hasattr(self.toolbar_ctrl, "action_save_to_library"):
            self.toolbar_ctrl.action_save_to_library.connect(self.fluxo_salvar_biblioteca)

        # PR 17 — Enviar pra Estratégia (V2)
        if hasattr(self.toolbar_ctrl, "action_send_to_strategy"):
            self.toolbar_ctrl.action_send_to_strategy.connect(self.fluxo_enviar_para_estrategia)

        if hasattr(self.toolbar_ctrl, 'action_enrich'):
            self.toolbar_ctrl.action_enrich.connect(self.fluxo_enriquecer)

        if hasattr(self.toolbar_ctrl, 'action_balance'):
            self.toolbar_ctrl.action_balance.connect(self.fluxo_balancear)

    def _on_table_data_changed(self, *_args):
        if self.dataset is not None:
            self.set_dirty(True)

    # ── PR 15: Modo galeria ──────────────────────────────────────────

    def _on_toggle_gallery(self, is_gallery: bool) -> None:
        """Toggle entre tabela (index 0) e galeria (index 1)."""
        if is_gallery:
            # Repopula a galeria com as faces atuais — barato pra cache hit
            # (mesmo dataset, mesmos paths). Garante que edições recentes
            # na tabela aparecem no grid.
            faces = list(self.dataset.faces) if self.dataset else []
            self.gallery_widget.populate(faces)
            self._view_stack.setCurrentIndex(1)
        else:
            self._view_stack.setCurrentIndex(0)

    def _on_gallery_face_selected(self, face_idx: int) -> None:
        """Click no grid → atualiza detail panel (mesma semântica que tabela).

        A tabela e a galeria compartilham o mesmo dataframe via
        `table_ctrl.model`, então reusamos `get_row_data(face_idx)` pra
        propagar pra detail panel — mesmo path que o slot da tabela.
        """
        if self.dataset is None:
            return
        if hasattr(self.table_ctrl, "model"):
            row_data = self.table_ctrl.model.get_row_data(face_idx)
            if row_data is not None:
                self.detail_ctrl.load_face_data(row_data)

    # ── PR 15.2: comparar par de faces lado a lado ────────────────────

    def _on_compare_pair(self, idx_a: int, idx_b: int) -> None:
        """Abre o ComparePairController com as duas faces selecionadas.

        Disparado tanto pelo context menu da tabela quanto da galeria.
        Recebe índices no `dataset.faces` (mesma indexação que o
        dataframe, por construção). Usa `comparison_service` quando
        disponível pra calcular cosine similarity.
        """
        if self.dataset is None:
            return
        n = len(self.dataset.faces)
        if not (0 <= idx_a < n and 0 <= idx_b < n):
            return
        if idx_a == idx_b:
            QMessageBox.warning(self.view, "Comparar par",
                                "Os dois índices apontam pra mesma face.")
            return

        face_a = self.dataset.faces[idx_a]
        face_b = self.dataset.faces[idx_b]

        # comparison_service vem do container (já usado em outras flows
        # como `_abrir_analise_real`); ausência é tolerada — o dialog
        # só não mostra similaridade calculada.
        comp_svc = None
        try:
            comp_svc = self.container.comparison_service()
        except Exception:
            pass

        from .compare_pair_dialog.compare_pair_controller import ComparePairController
        cmp_ctrl = ComparePairController(
            face_a=face_a, face_b=face_b,
            comparison_service=comp_svc,
            parent_view=self.view,
            dataset=self.dataset,
        )
        cmp_ctrl.exec()

    # ── Ações de seleção do menu de contexto da galeria ───────────────
    #
    # As ações recebem face_idx (índices em dataset.faces na ordem em que a
    # galeria foi populada). Ordenar a TABELA reordena o dataframe — e o
    # `sync_changes_from_table` reconstrói dataset.faces a partir dele —, então
    # índices posicionais podem deixar de bater após um sync. Por isso cada
    # handler captura os `image_path` ALVO ANTES de sincronizar e, após o sync,
    # recupera as faces por caminho (mesma chave que o próprio sync usa).
    #
    # IMPORTANTE — NÃO-DESTRUTIVO: "Remover" só tira a face do dataset (em
    # memória, efetivado ao Salvar). NENHUM arquivo de imagem é apagado do
    # disco, em nenhum modo. `path_mode==RELATIVE` NÃO garante que as imagens
    # sejam cópias do dataset (REWRITE_RELATIVE referencia os ORIGINAIS; o
    # parquet_dir pode ser None → cwd), então apagar arquivo arriscaria destruir
    # evidência original. Cópias órfãs num autocontido apenas permanecem em disco.

    def _gallery_target_paths(self, idxs) -> list:
        """`image_path` das faces selecionadas, em ordem crescente de índice de
        dataset.faces (a seleção já vem ordenada da galeria). Capturar ANTES de
        qualquer sync — depois a ordem de dataset.faces pode mudar."""
        if not self.dataset:
            return []
        faces = self.dataset.faces
        n = len(faces)
        out = []
        for i in sorted(set(idxs)):
            if isinstance(i, int) and 0 <= i < n:
                p = getattr(faces[i], "image_path", None)
                if p:
                    out.append(p)
        return out

    def _gallery_match_after_sync(self, idxs, action_label: str):
        """Captura os caminhos da seleção, sincroniza a tabela e devolve as
        faces correspondentes em self.dataset.faces. Aborta (devolve None) se um
        caminho casar com MAIS faces do que as selecionadas — sinal de
        `image_path` duplicado (raro; ex.: localização HASH com conteúdo
        idêntico). Operar por caminho aí afetaria faces NÃO selecionadas, então
        paramos por segurança em vez de adivinhar."""
        paths = self._gallery_target_paths(idxs)
        if not paths:
            return None
        n_selected = len(paths)
        self.sync_changes_from_table()              # não perder edições da tabela
        path_set = set(paths)
        matched = [f for f in self.dataset.faces
                   if getattr(f, "image_path", None) in path_set]
        if len(matched) > n_selected:
            QMessageBox.warning(
                self.view, action_label,
                "Algumas faces selecionadas compartilham o caminho de imagem com "
                "faces NÃO selecionadas (imagens de conteúdo idêntico podem "
                "apontar para o mesmo arquivo). Para não afetar faces que você "
                "não escolheu, a operação foi cancelada.")
            return None
        return matched

    def _refresh_after_structural_change(self) -> None:
        """Reconstrói tabela + galeria a partir de self.dataset (fonte de
        verdade) e marca modificado. Reseta a ordenação da tabela para a ordem
        do dataset (efeito colateral aceitável de uma mutação estrutural)."""
        self.table_ctrl.load_dataframe(self.dataset.to_pandas())
        self.gallery_widget.populate(list(self.dataset.faces))
        self.detail_ctrl.load_face_data(None)
        self.set_dirty(True)

    def _on_gallery_remove_faces(self, idxs) -> None:
        matched = self._gallery_match_after_sync(idxs, "Remover do dataset")
        if not matched:
            return
        n = len(matched)
        res = QMessageBox.question(
            self.view, "Remover do dataset",
            f"Remover {n} face(s) do dataset?\n\n"
            "Só remove do dataset — os arquivos de imagem NÃO são apagados. "
            "Efetivado ao Salvar.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if res != QMessageBox.StandardButton.Yes:
            return
        matched_ids = {id(f) for f in matched}      # remove EXATAMENTE as faces casadas
        self.dataset.faces = [f for f in self.dataset.faces if id(f) not in matched_ids]
        self._refresh_after_structural_change()
        QMessageBox.information(
            self.view, "Removido",
            f"{n} face(s) removida(s) do dataset. Salve para efetivar.")

    def _on_gallery_create_dataset_from_selection(self, idxs) -> None:
        import copy as _copy
        from PyQt6.QtWidgets import QInputDialog
        matched = self._gallery_match_after_sync(idxs, "Criar dataset a partir da seleção")
        if not matched:
            return
        suggested = f"{self.dataset.name}_sel"
        name, ok = QInputDialog.getText(
            self.view, "Criar dataset a partir da seleção",
            f"Nome do novo dataset ({len(matched)} face(s)):", text=suggested)
        if not ok or not name.strip():
            return
        name = name.strip()
        if self.is_dirty():
            res = QMessageBox.question(
                self.view, "Dataset modificado",
                "O dataset atual tem alterações não salvas; criar o subset "
                "substitui o que está na tela.\nContinuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if res != QMessageBox.StandardButton.Yes:
                return
        subset = FaceDataset(
            name=name, description=getattr(self.dataset, "description", "") or "",
            faces=[_copy.copy(f) for f in matched],
            column_specs=dict(self.dataset.column_specs) if self.dataset.column_specs else {},
            path_mode=self.dataset.path_mode, path_root=self.dataset.path_root,
            is_aligned=self.dataset.is_aligned,
        )
        subset._parquet_dir = self.dataset._parquet_dir
        subset.hash = None
        # O subset SUBSTITUI o dataset na tela — zera o estado de filtro do
        # anterior, senão "Limpar filtro" restauraria o dataset antigo por cima
        # do subset (dataset_original ficaria obsoleto).
        self.dataset_original = None
        self.current_filter_config = FilterConfig()
        self.carregar_dataset_na_tela(subset, path=None)
        self.set_dirty(True)
        QMessageBox.information(
            self.view, "Subset criado",
            f"Dataset '{name}' com {len(subset.faces)} face(s) carregado na tela.\n"
            "Use Salvar Como para gravá-lo num arquivo.")

    def _on_gallery_set_column_value(self, idxs) -> None:
        matched = self._gallery_match_after_sync(idxs, "Definir valor de coluna")
        if not matched:
            return
        # Alinha a ordem da tabela à de dataset.faces (df row == faces index).
        self.table_ctrl.load_dataframe(self.dataset.to_pandas())
        model = self.table_ctrl.model
        # Colunas EDITÁVEIS (label, key, vocabulary, dtype) para o diálogo —
        # dtype faz o campo de valor se adaptar (bool→combo, numérico→validação).
        cols = []
        for c in model.visible_columns:
            if getattr(c, "editable", False):
                spec = (self.dataset.column_specs or {}).get(getattr(c, "key", None))
                vocab = getattr(spec, "vocabulary", None) if spec else None
                dtype = getattr(spec, "dtype", None) if spec else None
                dtype_str = getattr(dtype, "value", None) or (dtype if isinstance(dtype, str) else None)
                cols.append((getattr(c, "label", None) or c.key, c.key, vocab, dtype_str))
        if not cols:
            QMessageBox.information(
                self.view, "Definir valor de coluna",
                "Este dataset não tem colunas editáveis.")
            return
        matched_ids = {id(f) for f in matched}
        rows = [r for r, f in enumerate(self.dataset.faces) if id(f) in matched_ids]
        if not rows:
            return
        from .set_column_value_dialog import SetColumnValueDialog
        dlg = SetColumnValueDialog(cols, len(rows), parent=self.view)
        if not dlg.exec():
            return
        result = dlg.get_result()
        if not result:
            return
        key, value = result
        n_ok = self._apply_column_value_to_rows(rows, key, value)
        if n_ok == 0:
            QMessageBox.information(
                self.view, "Definir valor de coluna",
                f"Nenhuma face alterada — '{value}' pode estar fora do "
                "vocabulário da coluna.")
            return
        # Propaga df→dataset e repinta a galeria (caption pode mudar — ex. identidade).
        self.sync_changes_from_table()
        self.gallery_widget.populate(list(self.dataset.faces))
        self.set_dirty(True)
        msg = f"'{key}' = '{value}' aplicado a {n_ok} face(s)."
        if n_ok < len(rows):
            msg += (f"\n{len(rows) - n_ok} rejeitada(s) — valor fora do "
                    "vocabulário da coluna?")
        QMessageBox.information(self.view, "Valor definido", msg)

    def _apply_column_value_to_rows(self, rows, key, value) -> int:
        """Seta `value` na coluna `key` para as `rows` (índices do model, já
        alinhados a dataset.faces) via model.setData — que valida vocabulário.
        Retorna quantas foram aceitas."""
        model = self.table_ctrl.model
        col_idx = next((i for i, c in enumerate(model.visible_columns)
                        if getattr(c, "key", None) == key), None)
        if col_idx is None:
            return 0
        model.layoutAboutToBeChanged.emit()
        n_ok = 0
        for r in rows:
            if 0 <= r < model.rowCount():
                if model.setData(model.index(r, col_idx), value, Qt.ItemDataRole.EditRole):
                    n_ok += 1
        model.layoutChanged.emit()
        return n_ok

    # ── PR 8: Spreadsheet UX (add column + show/hide) ─────────────────

    def _on_schema_changed(self):
        """Disparado quando o usuário muda visibilidade/adiciona coluna via
        header context-menu ou toolbar. Propaga o novo column_specs (que o
        model já mutou in-place) para o FaceDataset e marca dirty.
        """
        if self.dataset is None:
            return
        # column_specs é o mesmo objeto referenciado pelo model — mas
        # garantimos a atribuição explícita em caso de refactor futuro.
        self.dataset.column_specs = dict(self.table_ctrl.model.column_specs)
        self.set_dirty(True)

    def fluxo_editar_metadados(self):
        """Abre o DatasetMetadataEditorDialog (visão global).

        Aplica em sucesso:
        - Campos do dataset (name, description, is_aligned).
        - column_specs alterados → `update_column_spec(name, new_spec)` por coluna.

        Em seguida: reinstala vocabulary delegates + emite schema_changed
        (sincroniza specs do model com o dataset + marca dirty).
        """
        if self.dataset is None:
            return
        from PyQt6.QtWidgets import QMessageBox

        if not self.table_ctrl.model.column_specs:
            QMessageBox.warning(
                self.view, "Schema legado",
                "Este dataset foi carregado de um parquet sem column_specs "
                "na metadata. Salve-o primeiro (Salvar) para atualizar para "
                "o novo formato — depois você pode editar metadados.",
            )
            return

        from .column_dialogs.dataset_metadata_editor_dialog import (
            DatasetMetadataEditorDialog,
        )
        from .table.vocabulary_combo_delegate import install_vocabulary_delegates

        dialog = DatasetMetadataEditorDialog(self.dataset, parent=self.view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        ds_changes, col_changes = dialog.get_result()

        # Aplica dataset-level
        for field_name, new_value in ds_changes.items():
            setattr(self.dataset, field_name, new_value)

        # Aplica por-coluna via model (mantém modelo em sync e reset adequado)
        applied: List[str] = []
        for name, new_spec in col_changes.items():
            if self.table_ctrl.model.update_column_spec(name, new_spec):
                applied.append(name)

        # Re-instala delegates (vocabularies podem ter mudado), refresca
        # specs no dataset e marca dirty.
        install_vocabulary_delegates(self.table_ctrl.view, self.table_ctrl.model)
        self.table_ctrl.view.resizeColumnsToContents()
        self._on_schema_changed()

        # PR 25 (Tema 5.4) — registra a edição se houve mudanças.
        if ds_changes or applied:
            summary_parts = []
            if ds_changes:
                summary_parts.append(
                    "dataset: " + ", ".join(sorted(ds_changes.keys()))
                )
            if applied:
                summary_parts.append(f"colunas: {', '.join(sorted(applied))}")
            self._audit(
                AuditOperation.EDIT_SCHEMA,
                "Metadados editados — " + "; ".join(summary_parts),
                metadata={
                    "dataset_fields": sorted(ds_changes.keys()),
                    "columns": sorted(applied),
                },
            )

    def fluxo_adicionar_coluna(self):
        """Abre o AddColumnDialog. Se aceito, adiciona a coluna ao model
        (via add_custom_column do DatasetTableModel) e propaga ao dataset.
        """
        if self.dataset is None:
            return
        if not self.table_ctrl.model.column_specs:
            QMessageBox.warning(
                self.view, "Schema legado",
                "Este dataset foi carregado de um parquet sem column_specs "
                "na metadata. Salve-o primeiro (Salvar) para atualizar para "
                "o novo formato — depois você pode adicionar colunas custom.",
            )
            return

        from .column_dialogs.add_column_dialog import AddColumnDialog
        existing = set(self.table_ctrl.model.column_specs.keys())
        dialog = AddColumnDialog(existing_column_names=existing, parent=self.view)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        name, label, dtype, default, computed_expr = dialog.get_result()
        ok = self.table_ctrl.model.add_custom_column(
            name=name, label=label, dtype=dtype, default_value=default,
            editable=True, computed_expr=computed_expr,
        )
        if not ok:
            QMessageBox.warning(self.view, "Erro", "Falha ao adicionar coluna.")
            return

        # Re-propaga + marca dirty
        self._on_schema_changed()
        self.table_ctrl.view.resizeColumnsToContents()
        # PR 25 (Tema 5.4) — registra ADD_COLUMN.
        self._audit(
            AuditOperation.ADD_COLUMN,
            f"Coluna '{name}' adicionada (rótulo: '{label}', dtype: {dtype}).",
            metadata={
                "column": name,
                "label": label,
                "dtype": str(dtype),
                "computed_expr": computed_expr,
            },
        )

    def _get_initial_dir(self) -> str:
        if self.pipeline and hasattr(self.pipeline, 'storage_path'):
            path = str(self.pipeline.storage_path)
            if os.path.isdir(path): return path
        return ""

    # ── PR 25 (Tema 5.4) — Audit log forense / chain of custody ──────────
    def _audit(
        self,
        operation: AuditOperation,
        description: str,
        *,
        metadata: dict | None = None,
        path_override: str | None = None,
        hash_file: bool = False,
    ) -> None:
        """Append best-effort no audit log do dataset corrente.

        - `path_override`: para SAVE_AS, onde `current_path` ainda
          aponta para o path antigo no momento da chamada.
        - `hash_file`: quando True, calcula SHA-256 do parquet via
          `HashService.calculate_file_hash` e grava em `dataset_hash`.
          Pular em arquivos > 500 MB (nesses casos grava
          `metadata['hash_skipped']=True` com o tamanho).
        - Falhas no append ficam em stderr e NÃO interrompem o fluxo
          chamado — auditoria é observabilidade, não bloqueio.
        """
        path = path_override or self.current_path
        if not path:
            return  # dataset em memória ainda — sem sidecar pra escrever
        try:
            ds_hash = None
            extra_meta = dict(metadata or {})
            if hash_file:
                try:
                    p = Path(path)
                    if p.exists():
                        size_mb = p.stat().st_size / (1024 * 1024)
                        if size_mb > 500.0:
                            extra_meta["hash_skipped"] = True
                            extra_meta["file_size_mb"] = round(size_mb, 1)
                        else:
                            ds_hash = HashService().calculate_file_hash(p)
                except Exception as e:
                    extra_meta["hash_error"] = str(e)
            AuditLogService.append(
                path,
                operation,
                description,
                metadata=extra_meta,
                dataset_hash=ds_hash,
            )
        except Exception as e:
            # Best-effort — não abortar o fluxo principal
            import sys as _sys
            print(f"[AuditLog] Falha ao registrar {operation.value}: {e}",
                  file=_sys.stderr)

    def carregar_dataset_na_tela(self, dataset, path=None):
        self.dataset = dataset
        self.current_path = path
        # Propaga referência pra todos os widgets que precisam resolver
        # paths em modo RELATIVE (sem isso, datasets do Image Pipeline
        # mostram "Imagem não encontrada" / thumbnails vazias).
        self.detail_ctrl.set_dataset(dataset)
        if hasattr(self.gallery_widget, "set_dataset"):
            self.gallery_widget.set_dataset(dataset)
        if hasattr(self.table_ctrl, "set_dataset"):
            self.table_ctrl.set_dataset(dataset)

        # Schema-by-role (PR 6): se o FaceDataset trouxe column_specs (parquet
        # pós-PR 3), o table_ctrl propaga pra model + instala delegates de
        # vocabulário. Caso contrário (parquet legado ou dataset criado em
        # memória), o model cai no enum estático.
        column_specs = getattr(self.dataset, "column_specs", None) or {}
        if hasattr(self.table_ctrl, "apply_column_specs"):
            self.table_ctrl.apply_column_specs(column_specs)
        elif hasattr(self.table_ctrl, "model") and hasattr(self.table_ctrl.model, "set_column_specs"):
            self.table_ctrl.model.set_column_specs(column_specs)

        self.table_ctrl.load_dataframe(self.dataset.to_pandas())

        # Reseta o painel direito — sem isso, ao trocar de dataset
        # (ou limpar filtro) a face selecionada antes continua
        # exibida, mesmo podendo não estar mais no dataset corrente.
        self.detail_ctrl.load_face_data(None)

        # PR 15: invalida cache da galeria (dataset trocou) e re-popula só
        # se o usuário estiver no modo galeria. No modo tabela, a galeria
        # popula on-demand quando o usuário ativar o toggle.
        self.gallery_widget.clear_cache()
        if self._view_stack.currentIndex() == 1:
            self.gallery_widget.populate(list(self.dataset.faces))
        else:
            # Repopula com lista vazia pra resetar o estado interno do widget
            # (evita seleção velha quando o user trocar pra modo galeria).
            self.gallery_widget.populate([])

        self.set_dirty(False)

        text, tooltip = self._build_dataset_status(filtered=False)
        self.toolbar_ctrl.set_dataset_loaded_state(True, text, tooltip)

    def sync_changes_from_table(self):
        if not self.dataset: return
        df_edited = self.table_ctrl.get_current_dataframe()
        if df_edited is None or df_edited.empty: return

        dataset_visual_atualizado = FaceDataset.from_dataframe(df_edited, name=self.dataset.name)
        # PR 21 (Tema 4.2) — `from_dataframe` constrói um FaceDataset novo
        # sem column_specs (default {}). Preserva os specs vindos do
        # parquet/UI pra que custom columns continuem visíveis no filter
        # dialog e na próxima save (parquet metadata).
        if self.dataset.column_specs:
            dataset_visual_atualizado.column_specs = dict(self.dataset.column_specs)
        # `from_dataframe` também não restaura path_mode/path_root/
        # _parquet_dir/is_aligned — sem isso, datasets RELATIVE viravam
        # ABSOLUTE silenciosamente e enrich falhava (cv2.imread em paths
        # relativos contra CWD), gravando metadata corrompida no save
        # subsequente. is_aligned tem o mesmo problema: ao trocar de
        # view (galeria/tabela) ou editar qualquer célula, o flag se
        # perdia e o status bar passava a mostrar "fotos originais".
        dataset_visual_atualizado.path_mode = self.dataset.path_mode
        dataset_visual_atualizado.path_root = self.dataset.path_root
        dataset_visual_atualizado._parquet_dir = self.dataset._parquet_dir
        dataset_visual_atualizado.is_aligned = self.dataset.is_aligned
        self.dataset = dataset_visual_atualizado

        if self.dataset_original:
            print(f"🔄 Sync: Propagando {len(self.dataset)} registros editados para o Dataset Original...")
            mapa_edicoes = {face.image_path: face for face in self.dataset.faces}
            count_updates = 0
            for i, face_orig in enumerate(self.dataset_original.faces):
                if face_orig.image_path in mapa_edicoes:
                    self.dataset_original.faces[i] = mapa_edicoes[face_orig.image_path]
                    count_updates += 1
            print(f"✅ Sync Concluído: {count_updates} faces atualizadas no backup.")
        else:
            print(f"✅ Sync Simples: Dataset atualizado ({len(self.dataset)} registros).")
        # NOTE: dirty é marcado por _on_table_data_changed quando o usuário edita
        # uma célula. Sync apenas propaga; não inventa estado dirty quando o df
        # estiver sem alterações reais.

    def is_dirty(self) -> bool:
        return self._is_dirty

    def set_dirty(self, state: bool):
        self._is_dirty = state
        if self.dataset:
            text, tooltip = self._build_dataset_status(filtered=False)
            self.toolbar_ctrl.set_dataset_loaded_state(True, text, tooltip)

    # ── Status bar: linha principal + tooltip rico ───────────────────
    def _build_dataset_status(
        self, *, filtered: bool = False,
    ) -> tuple[str, str]:
        """Constrói (texto-HTML, tooltip-HTML) pra status bar.

        Linha principal compacta:
          📂 nome★ | 🖼️ N Imagens | 👤 N IDs | 🎯 alinhado |
          🧬 N FF · 📐 N OFIQ · 🎲 N SER-FIQ | ⚖ verdict (min–max/id) |
          ⚠️ N erros

        Badges omitidos quando 0 ou inaplicáveis. ★ aparece se dirty.

        Tooltip rich-text detalha: path do parquet, descrição,
        path_mode, hash, mtime, cobertura de enriquecimento e métricas
        de balanceamento (CV, ratio, min/max/mean/median).
        """
        if self.dataset is None:
            return ("Nenhum dataset carregado.", "")
        ds = self.dataset
        n_imgs = len(ds)
        n_ids = ds.number_of_identities
        nome = ds.name or "Sem Nome"
        star = "*" if self._is_dirty else ""
        suffix = " (Filtrado)" if filtered else ""

        parts: list[str] = [
            f"📂 <b>{nome}{star}</b>",
            f"🖼️ {n_imgs} Imagens",
            f"👤 {n_ids} IDs",
        ]

        # is_aligned — só mostra quando True (a maioria é "originais").
        if getattr(ds, "is_aligned", False):
            parts.append("🎯 alinhado")

        # Coberturas de enriquecimento (count com *_info populado).
        n_ff = sum(1 for f in ds.faces if f.ff_info is not None)
        n_ofiq = sum(1 for f in ds.faces if f.ofiq_info is not None)
        n_serfiq = sum(1 for f in ds.faces if f.serfiq_info is not None)
        enrich_badges = []
        if n_ff:
            enrich_badges.append(f"🧬 {n_ff} FF")
        if n_ofiq:
            enrich_badges.append(f"📐 {n_ofiq} OFIQ")
        if n_serfiq:
            enrich_badges.append(f"🎲 {n_serfiq} SER-FIQ")
        if enrich_badges:
            parts.append(" · ".join(enrich_badges))

        # Balanceamento — usa BalanceMetrics consistente com o resto
        # do app (CV + imbalance ratio, worst-of-two).
        balance_main, balance_detail = self._compute_balance_for_status(ds)
        if balance_main:
            parts.append(balance_main)

        # Erros (do tracking de creation/enrich).
        n_err = sum(1 for f in ds.faces if getattr(f, "error", False))
        if n_err:
            parts.append(f"⚠️ {n_err} erro(s)")

        text = " | ".join(parts) + suffix

        # ── Tooltip rich-text ──
        tooltip_lines: list[str] = []
        if self.current_path:
            tooltip_lines.append(f"<b>Path:</b> {self.current_path}")
        if ds.description:
            tooltip_lines.append(f"<b>Descrição:</b> {ds.description}")
        tooltip_lines.append(
            f"<b>Modo de path:</b> {ds.path_mode.value}"
            + (f" (root={ds.path_root})" if ds.path_root else "")
        )
        # Sempre mostra o estado de alinhamento no tooltip — útil quando
        # o usuário cria com "Alinhar durante criação" e quer confirmar
        # que o dataset ficou marcado.
        if getattr(ds, "is_aligned", False):
            tooltip_lines.append("<b>Conteúdo:</b> faces alinhadas 112×112")
        else:
            tooltip_lines.append("<b>Conteúdo:</b> fotos originais (não alinhadas)")
        # Coverage % por estágio (mais informativo que count).
        if n_imgs > 0:
            tooltip_lines.append(
                f"<b>Enriquecimento:</b> "
                f"FF {n_ff}/{n_imgs} ({100*n_ff//n_imgs}%), "
                f"OFIQ {n_ofiq}/{n_imgs} ({100*n_ofiq//n_imgs}%), "
                f"SER-FIQ {n_serfiq}/{n_imgs} ({100*n_serfiq//n_imgs}%)"
            )
        if balance_detail:
            tooltip_lines.append(f"<b>Balanceamento:</b> {balance_detail}")
        if n_err:
            tooltip_lines.append(
                f"<b>Erros:</b> {n_err} face(s) marcadas — "
                f"filtre por <code>error=True</code> na tabela."
            )
        # File stats — best effort (parquet pode não existir em
        # datasets em memória).
        if self.current_path:
            try:
                p = Path(self.current_path)
                if p.exists():
                    from datetime import datetime
                    size_mb = p.stat().st_size / (1024 * 1024)
                    mtime = datetime.fromtimestamp(
                        p.stat().st_mtime,
                    ).strftime("%Y-%m-%d %H:%M")
                    tooltip_lines.append(
                        f"<b>Tamanho:</b> {size_mb:.1f} MB · "
                        f"<b>Modificado:</b> {mtime}"
                    )
            except Exception:
                pass

        tooltip = "<br>".join(tooltip_lines)
        return (text, tooltip)

    @staticmethod
    def _compute_balance_for_status(ds) -> tuple[str, str]:
        """Calcula verdict de balanceamento + detalhe.
        Retorna ("", "") quando inaplicável (≤1 identidade).
        """
        from collections import Counter
        from faceaudit.models.dataset_stats import DatasetStats
        ids = [f.identity for f in ds.faces if f.identity]
        counts = Counter(ids)
        if len(counts) < 2:
            return ("", "")
        # Reusa a lógica de BalanceMetrics via DatasetStats inline.
        stub = DatasetStats(
            n_imgs=len(ds), n_ids=len(counts),
            samples_per_identity=dict(counts),
        )
        bm = stub.balance_metrics
        if bm is None:
            return ("", "")
        main = (
            f"{bm.verdict} ({bm.min_samples}–{bm.max_samples}/id)"
        )
        detail = (
            f"{bm.verdict} · "
            f"CV={bm.cv:.2f} · max/min={bm.imbalance_ratio:.1f}:1 · "
            f"min={bm.min_samples}, max={bm.max_samples}, "
            f"mean={bm.mean_samples:.1f}, median={bm.median_samples:.0f}"
        )
        return (main, detail)

    def fluxo_abrir_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Abrir Dataset", self._get_initial_dir(), "Parquet (*.parquet)"
        )
        if not path:
            return
        self.open_dataset_from_path(path)

    def open_dataset_from_path(self, path: str) -> None:
        """Carrega um dataset diretamente de um path absoluto, pulando o
        QFileDialog. Entry point usado por chamadas cross-tab (ex:
        Strategy Manager V2 "Abrir no Gerenciador de Datasets")."""
        if not path:
            return
        self._run_worker(
            LoadDatasetWorker(path),
            label="Lendo arquivo Parquet...",
            allow_cancel=False,
            determinate=False,
            wait_cursor=True,
            on_finished=lambda ds: self._on_load_success(ds, path),
        )

    def _run_worker(
        self,
        worker,
        *,
        label,
        allow_cancel=False,
        determinate=True,
        wait_cursor=False,
        use_log_dialog=False,
        on_finished=None,
    ):
        """Encapsula a criação do WorkerProgressRunner + ref forte.

        `use_log_dialog=True` ativa o LogProgressDialog (área de texto
        rolando os logs) — adequado pra fluxos multi-etapa onde o
        usuário precisa ver cada fase (ingestão de dataset, enriquecimento).
        Default é `False` (QProgressDialog compacto).

        Só permite uma operação ativa de cada vez — se o usuário disparar
        outra antes da anterior terminar, a anterior é cancelada
        cooperativamente.
        """
        if self._active_runner is not None:
            self._active_runner.stop_and_wait()
            self._active_runner = None

        def _wrap_finished(payload):
            self._active_runner = None
            if on_finished is not None:
                on_finished(payload)

        def _wrap_error(msg):
            self._active_runner = None
            QMessageBox.critical(self.view, "Erro", msg)

        runner = WorkerProgressRunner(
            self.view,
            worker,
            label=label,
            allow_cancel=allow_cancel,
            determinate=determinate,
            wait_cursor=wait_cursor,
            use_log_dialog=use_log_dialog,
            on_finished=_wrap_finished,
            on_error=_wrap_error,
        )
        self._active_runner = runner
        runner.start()
        return runner

    def _on_load_success(self, dataset, path):
        try:
            self.carregar_dataset_na_tela(dataset, path)
            # PR 25 (Tema 5.4) — registra OPEN no audit log do dataset.
            # hash_file=True calcula SHA-256 do parquet (eficiente via
            # hashlib.file_digest) — anchor da cadeia de custódia.
            self._audit(
                AuditOperation.OPEN,
                f"Dataset '{dataset.name}' aberto.",
                metadata={
                    "rows": len(dataset),
                    "identities": dataset.number_of_identities,
                },
                hash_file=True,
            )
        except Exception as e:
            QMessageBox.critical(self.view, "Erro de Renderização", f"Falha ao exibir tabela: {e}")

    def fluxo_salvar(self):
        if not self.dataset: return
        self.sync_changes_from_table()

        if self.current_path:
            try:
                self.dataset.to_parquet(self.current_path)
                QMessageBox.information(self.view, "Salvo", "Dataset salvo com sucesso.")
                self.set_dirty(False)
                # PR 25 (Tema 5.4) — registra SAVE com hash do arquivo
                # recém-escrito.
                self._audit(
                    AuditOperation.SAVE,
                    f"Dataset salvo em '{Path(self.current_path).name}'.",
                    metadata={
                        "rows": len(self.dataset),
                        "identities": self.dataset.number_of_identities,
                    },
                    hash_file=True,
                )
                # Notifica subscribers (cross-tab) que o arquivo no path
                # atual foi sobrescrito. Strategy Manager V2 usa pra
                # detectar mudança em primary do projeto e propagar
                # hash + cascade.
                self.dataset_saved_to_project.emit(str(self.current_path))
            except Exception as e:
                QMessageBox.critical(self.view, "Erro", f"Falha ao salvar: {e}")
        else:
            self.fluxo_salvar_como()

    def fluxo_salvar_como(self):
        if not self.dataset: return
        self.sync_changes_from_table()

        default_name = f"{self.dataset.name}.parquet"
        init_path = os.path.join(self._get_initial_dir(), default_name)

        path, _ = QFileDialog.getSaveFileName(self.view, "Salvar Como", init_path, "Parquet (*.parquet)")
        if path:
            old_path = self.current_path
            self.dataset.name = Path(path).stem
            self.dataset.to_parquet(path)
            # PR 25 (Tema 5.4) — preserva o sidecar do dataset original
            # quando o usuário faz Save As: copia o `.audit.json` antigo
            # para o novo path antes de adicionar a entrada SAVE_AS,
            # mantendo a cadeia de custódia ininterrupta.
            if old_path and old_path != path:
                try:
                    AuditLogService.migrate_sidecar(old_path, path)
                except Exception:
                    pass  # se a migração falhar, segue com chain nova
            self.current_path = path
            self.set_dirty(False)
            self.carregar_dataset_na_tela(self.dataset, path)
            self._audit(
                AuditOperation.SAVE_AS,
                f"Dataset salvo em novo arquivo '{Path(path).name}'.",
                metadata={
                    "previous_path": old_path,
                    "rows": len(self.dataset),
                    "identities": self.dataset.number_of_identities,
                },
                path_override=path,
                hash_file=True,
            )

    # ── PR 16: Adicionar dataset à Biblioteca Global ──────────────────────

    def fluxo_salvar_biblioteca(self):
        """Copia o parquet salvo do dataset atual pra biblioteca global,
        ficando reusável entre projetos do Strategy Manager V2.

        Pré-condições:
          - Dataset carregado e salvo em disco (current_path setado).
          - Se houver edições não-salvas, prompt pra salvar antes (a
            biblioteca só recebe o que está em disco).
          - Library path configurado (`config.paths.library`).
        """
        if not self.dataset:
            return

        # 1. Garante que o dataset foi salvo primeiro — a biblioteca
        # global aceita só arquivos em disco; edições em memória ficam de fora.
        self.sync_changes_from_table()
        if self._is_dirty or not self.current_path or not Path(self.current_path).exists():
            res = QMessageBox.question(
                self.view, "Salvar antes",
                "O dataset precisa estar salvo em disco antes de ir pra "
                "biblioteca global. Salvar agora?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Save:
                return
            self.fluxo_salvar()
            if self._is_dirty or not self.current_path:
                return  # falhou ou usuário cancelou o save dialog

        # 2. Resolve o root da biblioteca via app_config.paths.library
        library_root = getattr(getattr(self.config, "paths", None), "library", None)
        if not library_root:
            QMessageBox.warning(
                self.view, "Biblioteca não configurada",
                "Configure 'paths.library' no config.json antes de "
                "adicionar datasets à biblioteca global.",
            )
            return

        # 3. Pede o dataset_id (default = nome atual). Permite versionar
        # ao salvar (ex: "passaportes_2025_01" sem afetar o nome interno
        # do dataset).
        from PyQt6.QtWidgets import QInputDialog
        default_id = self.dataset.name or Path(self.current_path).stem
        dataset_id, ok = QInputDialog.getText(
            self.view, "Adicionar à Biblioteca",
            "Identificador (sem extensão) — será usado como nome do "
            "arquivo na biblioteca:",
            text=default_id,
        )
        if not ok:
            return
        dataset_id = (dataset_id or "").strip()
        if not dataset_id:
            QMessageBox.warning(self.view, "Inválido",
                                "O identificador não pode ser vazio.")
            return

        # 4. Confirma overwrite
        target_path = Path(library_root) / "datasets" / f"{dataset_id}.parquet"
        if target_path.exists():
            res = QMessageBox.question(
                self.view, "Já existe",
                f"Já existe um dataset '{dataset_id}.parquet' na biblioteca. "
                "Sobrescrever?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                defaultButton=QMessageBox.StandardButton.No,
            )
            if res != QMessageBox.StandardButton.Yes:
                return

        # 5. Copia via GlobalLibraryStorage — preserva schema metadata
        # (PR 3) e calcula hash SHA-256 automaticamente.
        try:
            from faceaudit.storage.global_library_storage import GlobalLibraryStorage
            from faceaudit.enums.strategy_artifact_type import StrategyArtifactType

            storage = GlobalLibraryStorage(library_root=library_root)
            storage.create_library_structure()  # idempotente
            ref = storage.copy_artifact_to_project(
                source_path=self.current_path,
                artifact_type=StrategyArtifactType.GLOBAL_DATASET,
                dataset_id=dataset_id,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self.view, "Erro",
                f"Falha ao copiar pra biblioteca: {e}",
            )
            return

        QMessageBox.information(
            self.view, "Adicionado à biblioteca",
            f"Dataset salvo na biblioteca global como '{dataset_id}.parquet'.\n\n"
            f"Path: {target_path}\n"
            f"Hash: {ref.hash[:16]}…",
        )

    # ── PR 17: Enviar dataset pra Strategy Manager V2 ─────────────────────

    def fluxo_enviar_para_estrategia(self):
        """Garante dataset salvo, então emite `send_to_strategy_requested`
        com o path absoluto. O `MainGUIController` captura, troca de aba
        e delega o import pra V2 (criando/abrindo projeto se necessário).

        Pré-condições:
          - Dataset carregado e salvo em disco (current_path setado).
          - Edições não salvas → prompt pra salvar antes; aborta se cancelar.
        """
        if not self.dataset:
            return

        self.sync_changes_from_table()
        if self._is_dirty or not self.current_path or not Path(self.current_path).exists():
            res = QMessageBox.question(
                self.view, "Salvar antes",
                "O dataset precisa estar salvo em disco antes de ir pra V2. "
                "Salvar agora?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Cancel,
            )
            if res != QMessageBox.StandardButton.Save:
                return
            self.fluxo_salvar()
            if self._is_dirty or not self.current_path:
                return

        # Emite o signal pro MainGUIController fazer a ponte cross-tab
        abs_path = str(Path(self.current_path).resolve())
        self.send_to_strategy_requested.emit(abs_path)

    def request_close(self) -> bool:
        if not self.dataset:
            return True

        # Flush pending in-table edits para self.dataset antes de avaliar dirty —
        # caso contrário, edições só no QAbstractTableModel são perdidas silenciosamente.
        self.sync_changes_from_table()

        if self.is_dirty():
            res = QMessageBox.question(self.view, "Salvar Alterações?", 
                                       "O dataset possui alterações não salvas. Deseja salvar antes de fechar?", 
                                       QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            
            if res == QMessageBox.StandardButton.Cancel:
                return False
            if res == QMessageBox.StandardButton.Save:
                self.fluxo_salvar()
                if self.is_dirty(): return False

        # PR 25 (Tema 5.4) — registra CLOSE antes de zerar current_path.
        # Sem hash do arquivo (não muda no fechamento) — só timestamp/operator.
        self._audit(
            AuditOperation.CLOSE,
            f"Dataset '{self.dataset.name}' fechado.",
        )

        self.dataset = None
        self.current_path = None
        self.dataset_original = None
        self.table_ctrl.load_dataframe(None)
        # Limpa a galeria também — senão, ao fechar em modo galeria, as
        # miniaturas do dataset anterior continuam aparecendo.
        self.gallery_widget.clear_cache()
        self.gallery_widget.populate([])
        self.detail_ctrl.load_face_data(None)
        self.toolbar_ctrl.set_dataset_loaded_state(False, "Nenhum dataset carregado.")
        self.set_dirty(False)
        return True

    def fluxo_fechar_dataset(self):
        self.request_close()

    def fluxo_novo_dataset_video(self):
        """🎬 Abre o fluxo de criação de dataset a partir de vídeo."""
        self._video_ctrl_lazy()  # cursor de espera no import pesado da 1ª vez
        self._video_ctrl.run_new_from_video()

    def fluxo_importar_run_video(self):
        """📥 Importa um run de extração de vídeo já produzido (revisar + dataset)."""
        self._video_ctrl_lazy()
        self._video_ctrl.run_import_video_run()

    def _video_ctrl_lazy(self):
        """Cria o VideoDatasetController sob demanda. O import puxa cv2 +
        FFService + serviços de vídeo — pesado na 1ª vez —, então mostra o
        cursor de ocupado para dar feedback ao clicar no botão."""
        if getattr(self, "_video_ctrl", None) is not None:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # paint imediato do cursor
        try:
            from .video_dataset_dialog.video_dataset_controller import VideoDatasetController
            self._video_ctrl = VideoDatasetController(self)
        finally:
            QApplication.restoreOverrideCursor()

    def fluxo_novo_dataset(self):
        templates_path = Path(self.config.paths.dir_templates)
        default_dataset_dir = str(getattr(self.config.paths, "datasets", "") or "")
        ff_cfg = getattr(self.config, "ff", None)
        dialog = NewDatasetController(
            self.view,
            templates_path=templates_path,
            ff_available=self.ff_available,
            ofiq_available=self.ofiq_available,
            default_dataset_dir=default_dataset_dir,
            default_use_gpu=bool(getattr(ff_cfg, "use_gpu", False)),
            default_batch_size=int(getattr(ff_cfg, "batch_size", 32)),
        )
        
        if dialog.exec():
            ingestion_config: DatasetIngestionConfig = dialog.get_config()
            storage_path = Path(self.pipeline.storage_path)
            cache_file = storage_path / f"{ingestion_config.dataset_name}.parquet"
            
            if cache_file.exists():
                msg = f"O dataset '{ingestion_config.dataset_name}' já existe. Sobrescrever?"
                res = QMessageBox.question(self.view, "Confirmar", msg)
                if res == QMessageBox.StandardButton.No: return

            self._iniciar_worker_criacao(ingestion_config)

    def _iniciar_worker_criacao(self, config):
        self._run_worker(
            NewDatasetWorker(self.pipeline, config),
            label="Criando Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_criacao_concluida,
        )

    def _on_criacao_concluida(self, dataset):
        if dataset and len(dataset) > 0:
            path_cache = self.pipeline.storage_path / f"{dataset.name}.parquet"
            self.carregar_dataset_na_tela(dataset, path=str(path_cache))
            # PR 25 (Tema 5.4) — registra CREATE como primeira entrada
            # da chain. hash_file=True dá um anchor inicial: o hash do
            # parquet recém-ingerido.
            self._audit(
                AuditOperation.CREATE,
                f"Dataset '{dataset.name}' criado a partir de ingestão.",
                metadata={
                    "rows": len(dataset),
                    "identities": dataset.number_of_identities,
                },
                hash_file=True,
            )
            QMessageBox.information(self.view, "Sucesso", "Dataset Criado!")
        else:
            QMessageBox.warning(self.view, "Aviso", "Processo finalizado sem dados.")

    def fluxo_stats(self): pass 

    def fluxo_estatisticas(self):
        if not self.dataset: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(50, self._abrir_estatisticas_real)

    def _abrir_estatisticas_real(self):
        try:
            self.sync_changes_from_table()
            stats_ctrl = StatsDialogController(self.dataset, parent_view=self.view)
            QApplication.restoreOverrideCursor()
            stats_ctrl.exec()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao gerar estatísticas: {e}")

    # =================================================================================
    # PR 24 (Tema 5.3 + 5.5) — Relatório de schema + Export CSV/Excel
    # =================================================================================
    def fluxo_schema_report(self):
        """Abre o SchemaReportDialog com a cobertura por coluna do
        dataset atual. O dialog inclui um botão de export CSV/Excel."""
        if not self.dataset:
            return
        # Sincroniza edições in-table antes — garante que custom columns
        # editadas ali já apareçam no relatório. WaitCursor cobre todo
        # o setup pesado (sync + computação de stats de cobertura no
        # construtor do dialog); restaurado antes do `exec()` pra que
        # o usuário interaja com cursor normal.
        from .schema_report_dialog import SchemaReportDialog

        # PR 25 (Tema 5.4) — callback de audit pro dialog.
        def _audit_export(meta: dict) -> None:
            self._audit(
                AuditOperation.EXPORT,
                f"Dataset exportado para {meta.get('format', 'arquivo')} "
                f"({meta.get('rows', '?')} linhas, "
                f"{meta.get('cols', '?')} colunas).",
                metadata=meta,
            )

        # Callback de visibilidade — chamado a cada toggle do checkbox
        # da coluna "Visível" no dialog. Aplica imediatamente no model
        # da tabela (mesmo caminho do menu de contexto do header).
        def _on_visibility_change(col_name: str, visible: bool) -> None:
            try:
                if self.table_ctrl.model.set_column_visibility(col_name, visible):
                    from .table.vocabulary_combo_delegate import (
                        install_vocabulary_delegates,
                    )
                    install_vocabulary_delegates(
                        self.table_ctrl.view, self.table_ctrl.model,
                    )
                    self.table_ctrl.view.resizeColumnsToContents()
                    self.set_dirty(True)
            except Exception as e:
                print(f"[warn] visibility toggle falhou: {e}")

        with wait_cursor():
            self.sync_changes_from_table()
            dialog = SchemaReportDialog(
                self.dataset, parent=self.view,
                on_export=_audit_export,
                on_visibility_change=_on_visibility_change,
            )
        dialog.exec()

    def fluxo_audit_log(self):
        """Abre o AuditLogDialog com a chain de custódia do dataset
        atual. Requer dataset salvo em disco — log é per-arquivo."""
        if not self.dataset:
            return
        if not self.current_path:
            QMessageBox.information(
                self.view, "Audit log indisponível",
                "Salve o dataset primeiro — a cadeia de custódia é "
                "anexada ao arquivo parquet (sidecar .audit.json).",
            )
            return
        from .audit_log_dialog import AuditLogDialog
        dialog = AuditLogDialog(self.current_path, parent=self.view)
        dialog.exec()

    # =================================================================================
    # PR L1d — FLUXO "MODIFICAR CAMINHOS" (corrige path_root / relativiza / copia)
    # =================================================================================
    def fluxo_modificar_caminhos(self):
        """Abre o ModifyPathsDialog. `FIX_PATH_ROOT` é aplicado direto
        (só-metadados, sem mover arquivos); as migrações pesadas
        (REWRITE/COPY/REBASE) disparam o worker e, ao final, substituem
        o dataset corrente + propõem Save As (consistência com
        Enrich/Filter)."""
        if not self.dataset:
            return

        # sync + detect_path_prefix + construção do dialog são síncronos
        # e podem demorar em datasets grandes → wait cursor cobre tudo
        # até o dialog estar pronto pra interagir.
        with wait_cursor():
            self.sync_changes_from_table()

            # Sugere target_dir = <parquet_dir_atual>/<dataset_name>__localized/
            if self.current_path:
                base = Path(self.current_path).parent
            else:
                base = Path(self._get_initial_dir() or os.getcwd())
            suggested = base / f"{self.dataset.name}__localized"

            # Auto-detecta o prefixo comum dos image_paths atuais — só
            # é útil pro modo REBASE_PREFIX, mas computar antes evita
            # uma round-trip do dialog ao service.
            from faceaudit.services.dataset_localization_service import (
                DatasetLocalizationService,
            )
            detected = DatasetLocalizationService.detect_path_prefix(self.dataset)

            # Pasta do parquet: ancora o picker de subpasta e a validação
            # de path_root. `_parquet_dir` é runtime (from_parquet); cai
            # no diretório do current_path quando ausente.
            parquet_dir = getattr(self.dataset, "_parquet_dir", None)
            if parquet_dir is None and self.current_path:
                parquet_dir = Path(self.current_path).parent

            from .modify_paths_dialog import ModifyPathsDialog, PathOp
            dialog = ModifyPathsDialog(
                dataset_name=self.dataset.name,
                suggested_target_dir=suggested,
                parent=self.view,
                detected_prefix=detected,
                path_mode=self.dataset.path_mode,
                path_root=self.dataset.path_root,
                parquet_dir=parquet_dir,
                dataset=self.dataset,
            )
        # exec() fora do wait_cursor — dialog precisa de cursor normal
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        try:
            result = dialog.get_result()
        except ValueError as e:
            QMessageBox.warning(self.view, "Configuração inválida", str(e))
            return

        # Correção leve de path_root — só-metadados, aplicada em memória
        # sem worker (não move nenhum arquivo).
        if result["op"] == PathOp.FIX_PATH_ROOT:
            self._aplicar_path_root(result["path_root"])
            return

        self._iniciar_worker_modificar_caminhos(
            target_dir=result["target_dir"],
            mode=result["mode"],
            naming=result["naming"],
            skip_existing=result["skip_existing"],
            old_prefix=result.get("old_prefix"),
        )

    # ── Correção de path_root (só-metadados, sem mover arquivos) ─────────
    def _aplicar_path_root(self, new_root: str):
        """Repontar a subpasta (`path_root`) de um dataset RELATIVE sem
        mover arquivos. Valida por amostragem; se falhar, exige
        confirmação explícita antes de aplicar."""
        if not self.dataset:
            return
        old_root = self.dataset.path_root or ""
        new_root = normalize_stored_path((new_root or "").strip())

        # "", ".", None são todos equivalentes à raiz — early-out se
        # não houve mudança efetiva.
        if self._canon_root(new_root) == self._canon_root(old_root):
            QMessageBox.information(
                self.view, "Sem alteração",
                "A subpasta (path_root) já está com esse valor.",
            )
            return

        ok, msg = self._validate_path_root(new_root)
        if not ok:
            resp = QMessageBox.question(
                self.view, "Validação falhou",
                f"{msg}\n\nAplicar mesmo assim?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        # Aplica em memória. path_mode defensivamente RELATIVE (path_root
        # só faz sentido em RELATIVE; o combo só oferece FIX nesse modo,
        # mas garantimos coerência).
        self.dataset.path_root = new_root or None
        self.dataset.path_mode = PathMode.RELATIVE

        # Refresh mínimo ANTES do set_dirty (o refresh não toca no dirty;
        # carregar_dataset_na_tela tocaria — por isso NÃO é usado aqui).
        self._refresh_after_path_root_change()
        self.set_dirty(True)

        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"path_root alterado de '{old_root or '(raiz)'}' para "
                    f"'{new_root or '(raiz)'}' (edição de metadados, sem "
                    "mover arquivos)."
                ),
                metadata={
                    "operation": "fix_path_root",
                    "old_path_root": old_root,
                    "new_path_root": new_root,
                },
            )
        except Exception:
            pass  # audit é best-effort

        QMessageBox.information(
            self.view, "Subpasta atualizada",
            "O path_root foi atualizado em memória. Use 'Salvar' pra "
            "gravar a mudança no arquivo do dataset.",
        )

    @staticmethod
    def _canon_root(root: Optional[str]) -> str:
        """Canonicaliza um path_root pra comparação: '', '.', None → ''."""
        r = (root or "").strip()
        return "" if r in ("", ".") else normalize_stored_path(r)

    def _validate_path_root(self, candidate: str) -> tuple:
        """Valida um path_root candidato resolvendo a 1ª face com imagem,
        SEM mutar o dataset. Retorna (ok: bool, msg: str). Indisponível
        (retorna ok=True) quando o dataset ainda não tem pasta em disco."""
        parquet_dir = getattr(self.dataset, "_parquet_dir", None)
        if parquet_dir is None and self.current_path:
            parquet_dir = Path(self.current_path).parent
        if parquet_dir is None:
            return True, "Validação indisponível (dataset ainda não salvo)."
        faces = getattr(self.dataset, "faces", None) or []
        sample = next(
            (f for f in faces if getattr(f, "image_path", None)), None,
        )
        if sample is None:
            return True, "Nenhuma face com caminho de imagem para validar."
        resolved = resolve_image_path(
            sample,
            parquet_dir=parquet_dir,
            path_mode=PathMode.RELATIVE,
            path_root=candidate,
        )
        if resolved.exists():
            return True, f"OK — o arquivo de exemplo existe: {resolved}"
        return False, f"O arquivo de exemplo NÃO existe: {resolved}"

    def _refresh_after_path_root_change(self):
        """Refresh mínimo após mudar path_root: re-resolve imagens (só a
        RESOLUÇÃO mudou, não as células). NÃO chama
        `carregar_dataset_na_tela` — este termina em `set_dirty(False)` e
        apagaria o estado dirty que acabamos de marcar."""
        if not self.dataset:
            return
        self.detail_ctrl.set_dataset(self.dataset)
        if hasattr(self.gallery_widget, "set_dataset"):
            self.gallery_widget.set_dataset(self.dataset)
        if hasattr(self.table_ctrl, "set_dataset"):
            self.table_ctrl.set_dataset(self.dataset)
        # Reseta o painel direito (a face antes selecionada pode agora
        # resolver pra um arquivo diferente).
        self.detail_ctrl.load_face_data(None)
        # Invalida o cache de miniaturas — thumbnails são resolvidas via
        # path e mudaram de origem.
        self.gallery_widget.clear_cache()
        if self._view_stack.currentIndex() == 1:
            self.gallery_widget.populate(list(self.dataset.faces))

    def _iniciar_worker_modificar_caminhos(
        self,
        target_dir: Path,
        mode,
        naming,
        skip_existing: bool,
        old_prefix: Optional[str] = None,
    ):
        worker = ModifyPathsWorker(
            dataset=self.dataset,
            target_dir=target_dir,
            mode=mode,
            naming=naming,
            skip_existing=skip_existing,
            old_prefix=old_prefix,
        )
        # Guarda contexto pra usar no on_finished
        self._localize_target_dir = target_dir
        self._localize_mode = mode
        self._localize_naming = naming
        self._run_worker(
            worker,
            label=f"Modificando caminhos: {self.dataset.name}",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,  # feedback constante (decisão #3 do usuário)
            on_finished=self._on_modificar_caminhos_concluido,
        )

    def _on_modificar_caminhos_concluido(self, result):
        """Result é `LocalizationResult` ou None (cancelado/erro)."""
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Modificação de caminhos cancelada ou finalizada sem "
                "resultado.",
            )
            return

        # Resumo na barra de status / popup
        if result.n_failed:
            failures_preview = "\n".join(
                f"  • {p}: {msg}" for p, msg in list(result.failures)[:5]
            )
            extra = (
                f"\n\n⚠ {result.n_failed} falha(s):\n{failures_preview}"
                + ("\n  …" if result.n_failed > 5 else "")
            )
        else:
            extra = ""

        # PR 25 (Tema 5.4) — registra no audit log do dataset original
        # ANTES de trocar o dataset corrente. Anchor da cadeia de
        # custódia: depois de localizar, o original perde 'rastreio
        # vivo' (continua existindo no disco mas o usuário fica com o
        # localizado). O audit no original preserva essa transição.
        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"Modificação de caminhos concluída: {result.summary()}. "
                    f"Destino: {result.target_dir}"
                ),
                metadata={
                    "mode": result.mode.value,
                    "naming": result.naming.value,
                    "target_dir": str(result.target_dir),
                    "n_processed": result.n_processed,
                    "n_copied": result.n_copied,
                    "n_skipped": result.n_skipped,
                    "n_failed": result.n_failed,
                    "duration_seconds": round(result.duration_seconds, 3),
                },
            )
        except Exception:
            pass  # audit é best-effort

        # Substitui o dataset corrente pelo localizado.
        # Em REBASE_PREFIX o arquivo conceitualmente é o mesmo (só os
        # paths foram reescritos), então preservamos current_path —
        # assim "Salvar" sobrescreve o parquet original e "Salvar Como"
        # permite gravar como cópia. Nos demais modos (COPY_*, REWRITE),
        # imagens foram movidas/transformadas e o original perde
        # significado — current_path=None força Save As.
        is_rebase = result.mode == LocalizationMode.REBASE_PREFIX
        self.dataset = result.dataset
        self.dataset_original = None
        if not is_rebase:
            self.current_path = None
        self.set_dirty(True)
        self.carregar_dataset_na_tela(self.dataset, path=self.current_path)

        QMessageBox.information(
            self.view, "Caminhos modificados",
            f"Dataset atualizado em memória.\n\n{result.summary()}{extra}\n\n"
            "Use 'Salvar' pra sobrescrever o arquivo original ou "
            "'Salvar Como' pra gravar com outro nome.",
        )

    # =================================================================================
    # PR I2 — FLUXO DE EDITOR DE PIPELINE DE IMAGENS
    # =================================================================================
    def fluxo_image_pipeline_editor(self):
        """Abre o ImagePipelineDialogController em modal.

        Imagem inicial: face única selecionada na tabela, ou aleatória
        quando 0 ou múltiplas seleções (decisão #2 do tema).
        """
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.information(
                self.view, "Sem dataset",
                "Abra ou crie um dataset antes de usar o editor de "
                "pipeline de imagens.",
            )
            return

        # Determina face inicial baseado na seleção da tabela.
        # Decisão #2 do tema: 1 face selecionada → abre com ela;
        # 0 ou múltiplas → aleatória (passamos -1 e o controller do
        # editor escolhe).
        initial_face_idx = -1
        try:
            selection_model = self.table_ctrl.view.selectionModel()
            if selection_model is not None:
                rows = {idx.row() for idx in selection_model.selectedIndexes()}
                if len(rows) == 1:
                    initial_face_idx = next(iter(rows))
        except Exception:
            pass  # API mudou ou estado não inicializado — cai pro aleatório

        # FFService do pipeline pai pra suportar ALIGNED_FACE +
        # CropMode.FACE_DETECTED.
        ff_svc = (
            getattr(self.pipeline, "ff_service", None)
            if self.pipeline
            else None
        )

        # PR I4 — library_root pra Salvar/Carregar pipelines da
        # biblioteca global. None desabilita esses 2 botões.
        library_root = getattr(
            getattr(self.config, "paths", None), "library", None,
        )

        # Construção do controller é pesada (carrega face inicial,
        # inicializa pipeline + view com preview). WaitCursor cobre.
        from .image_pipeline_dialog import ImagePipelineDialogController
        with wait_cursor():
            ctrl = ImagePipelineDialogController(
                dataset=self.dataset,
                initial_face_index=initial_face_idx,
                ff_service=ff_svc,
                library_root=library_root,
                parent=self.view,
            )
        ctrl.exec()

    # =================================================================================
    # PR I5 — FLUXO DE BATCH PROCESSING DE PIPELINE DE IMAGENS
    # =================================================================================
    def fluxo_aplicar_image_pipeline(self):
        """Abre o ApplyImagePipelineDialog. Após confirmação, dispara
        worker que aplica pipeline em todo dataset."""
        if not self.dataset or len(self.dataset) == 0:
            QMessageBox.information(
                self.view, "Sem dataset",
                "Abra ou crie um dataset antes de aplicar pipelines.",
            )
            return

        library_root = getattr(
            getattr(self.config, "paths", None), "library", None,
        )
        if not library_root:
            QMessageBox.warning(
                self.view, "Biblioteca indisponível",
                "Configure 'paths.library' no config.json antes de "
                "aplicar pipelines de imagens.",
            )
            return

        # Default destino: <parquet_dir>/<dataset>__<pipeline>/
        if self.current_path:
            base = Path(self.current_path).parent
        else:
            base = Path(self._get_initial_dir() or os.getcwd())

        from .apply_image_pipeline_dialog import ApplyImagePipelineDialog
        # Construção lista pipelines disponíveis da biblioteca — I/O.
        with wait_cursor():
            dialog = ApplyImagePipelineDialog(
                library_root=Path(library_root),
                dataset_name=self.dataset.name,
                suggested_target_parent=base,
                parent=self.view,
            )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        pipeline_cfg = dialog.selected_pipeline()
        if pipeline_cfg is None:
            QMessageBox.warning(
                self.view, "Pipeline inválido",
                "Nenhum pipeline foi selecionado.",
            )
            return

        target_dir = dialog.target_dir()
        if not target_dir:
            QMessageBox.warning(
                self.view, "Destino vazio",
                "Pasta destino não pode estar vazia.",
            )
            return

        skip_existing = dialog.skip_existing()
        output_filename = dialog.output_filename()
        open_new_after = dialog.open_new_after()

        # Se o usuário quer abrir o derivado e o dataset atual está
        # dirty, oferece salvar antes (mesmo padrão do close).
        if open_new_after and self._is_dirty:
            res = QMessageBox.question(
                self.view, "Salvar antes?",
                "O dataset atual tem alterações não salvas. "
                "Deseja salvar antes de abrir o derivado?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if res == QMessageBox.StandardButton.Cancel:
                return
            if res == QMessageBox.StandardButton.Save:
                self.fluxo_salvar()
                if self._is_dirty:
                    return  # save falhou

        self._iniciar_worker_aplicar_image_pipeline(
            pipeline_cfg, target_dir, skip_existing,
            output_filename, open_new_after,
        )

    def _iniciar_worker_aplicar_image_pipeline(
        self, pipeline_cfg, target_dir, skip_existing: bool,
        output_filename: str, open_new_after: bool,
    ):
        from .workers.apply_image_pipeline_worker import (
            ApplyImagePipelineWorker,
        )

        ff_svc = (
            getattr(self.pipeline, "ff_service", None)
            if self.pipeline
            else None
        )

        # Mesmo padrão do "Novo Dataset com Alinhamento": parquet vai
        # pra biblioteca global de datasets, imagens vão pra pasta
        # escolhida pelo usuário.
        dataset_output_dir = Path(self.pipeline.storage_path)

        worker = ApplyImagePipelineWorker(
            dataset=self.dataset,
            pipeline_config=pipeline_cfg,
            images_dir=Path(target_dir),
            dataset_output_dir=dataset_output_dir,
            ff_service=ff_svc,
            skip_existing=skip_existing,
            output_filename=output_filename or None,
        )
        # Guarda contexto pra usar no on_finished
        self._batch_target_dir = Path(target_dir)
        self._batch_pipeline_name = pipeline_cfg.name
        self._batch_open_new_after = open_new_after
        self._run_worker(
            worker,
            label=(
                f"Aplicando pipeline '{pipeline_cfg.name}' em "
                f"'{self.dataset.name}'…"
            ),
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,  # feedback constante
            on_finished=self._on_aplicar_image_pipeline_concluido,
        )

    def _on_aplicar_image_pipeline_concluido(self, result):
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Aplicação de pipeline cancelada ou finalizada sem "
                "resultado.",
            )
            return

        # Audit log (PR 25) — registra COPY_IMAGES no parquet original
        try:
            self._audit(
                AuditOperation.COPY_IMAGES,
                (
                    f"Pipeline de imagens '{result.pipeline_name}' "
                    f"aplicado em todo o dataset. {result.summary()}"
                ),
                metadata={
                    "operation": "image_pipeline_batch",
                    "pipeline_name": result.pipeline_name,
                    "images_dir": str(result.images_dir),
                    "parquet_path": str(result.parquet_path),
                    "n_processed": result.n_processed,
                    "n_succeeded": result.n_succeeded,
                    "n_skipped": result.n_skipped,
                    "n_failed": result.n_failed,
                    "duration_seconds": round(result.duration_seconds, 3),
                },
            )
        except Exception:
            pass  # audit é best-effort

        # Resumo + falhas
        if result.n_failed:
            failures_preview = "\n".join(
                f"  • {p}: {msg}"
                for p, msg in list(result.failures)[:5]
            )
            extra = (
                f"\n\n⚠ {result.n_failed} falha(s):\n{failures_preview}"
                + ("\n  …" if result.n_failed > 5 else "")
            )
        else:
            extra = ""

        error_line = ""
        if result.error_parquet_path is not None:
            error_line = (
                f"\nDataset de erros: {result.error_parquet_path}\n"
                f"  → abra como dataset normal pra inspecionar a "
                f"coluna `error_message`."
            )

        QMessageBox.information(
            self.view, "Pipeline aplicado",
            f"{result.summary()}{extra}\n\n"
            f"Dataset derivado: {result.parquet_path}\n"
            f"Imagens: {result.images_dir}\n"
            f"Manifest forense: {result.images_dir / 'manifest.json'}"
            f"{error_line}",
        )

        # Abre o derivado quando o usuário pediu. Fecha o atual sem
        # prompt (já tratamos dirty antes de iniciar o worker).
        if getattr(self, "_batch_open_new_after", False):
            try:
                # Reusa o caminho normal de open — fluxo idêntico a
                # clicar Abrir Dataset apontando pro novo parquet.
                self._run_worker(
                    LoadDatasetWorker(str(result.parquet_path)),
                    label="Abrindo dataset derivado…",
                    allow_cancel=False,
                    determinate=False,
                    wait_cursor=True,
                    on_finished=(
                        lambda ds: self._on_load_success(
                            ds, str(result.parquet_path),
                        )
                    ),
                )
            except Exception as e:
                QMessageBox.warning(
                    self.view, "Não foi possível abrir",
                    f"O derivado foi salvo, mas falhou abrir: {e}\n"
                    f"Use 'Abrir Dataset' apontando pra "
                    f"{result.parquet_path}.",
                )
            finally:
                self._batch_open_new_after = False

    # =================================================================================
    # [NOVO] FLUXO DE ANÁLISE DO DATASET (AUDITORIA E MINERAÇÃO BIOMÉTRICA)
    # =================================================================================
    def fluxo_analise_dataset(self):
        if not self.dataset: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_analise_real)
        
    def _abrir_analise_real(self):
        try:
            # Salva qualquer edição de rótulo antes de rodar a auditoria
            self.sync_changes_from_table()

            # Pós-PR 11: o controller delega pro DatasetAuditService (cheap audits
            # automáticos no abrir) e o ComparisonService (hard mining sob demanda).
            comp_svc = self.container.comparison_service()
            audit_svc = self.container.dataset_audit_service()

            analysis_ctrl = DatasetAnalysisController(
                dataset=self.dataset,
                audit_service=audit_svc,
                comparison_service=comp_svc,
                parent_view=self.view,
            )
            # Quando o usuário usa as ações de sanitização (excluir, ...),
            # o dialog muta self.dataset em memória. Refrescamos a tabela
            # principal e marcamos dirty.
            analysis_ctrl.dataset_modified.connect(self._on_dataset_modified_by_audit)

            QApplication.restoreOverrideCursor()
            analysis_ctrl.exec()

        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao abrir Auditoria: {e}")

    def _on_dataset_modified_by_audit(self) -> None:
        """Hook chamado pelo DatasetAnalysisController quando faces foram
        deletadas/modificadas via o painel de auditoria."""
        if self.dataset is None:
            return
        self.table_ctrl.load_dataframe(self.dataset.to_pandas())
        self.set_dirty(True)

    def fluxo_comparar(self):
        if not self.dataset: return
        path, _ = QFileDialog.getOpenFileName(self.view, "Comparar com...", self._get_initial_dir(), "Parquet (*.parquet)")
        if path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            QTimer.singleShot(50, lambda: self._abrir_comparacao_real(path))

    def _abrir_comparacao_real(self, path):
        try:
            self.sync_changes_from_table()
            other = FaceDataset.from_parquet(path)
            QApplication.restoreOverrideCursor()
            
            # Buscamos os Singletons diretamente do Container
            stats_svc = self.container.stats_service()
            comp_svc = self.container.dataset_comparison_service()
            
            # Injetamos manualmente no construtor do Diálogo
            cmp_ctrl = CompareDialogController(
                dataset_a=self.dataset, 
                dataset_b=other, 
                stats_service=stats_svc, 
                comparison_service=comp_svc, 
                parent_view=self.view
            )
            cmp_ctrl.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha na comparação: {e}")

    def fluxo_merge(self):
        if not self.dataset: return
        self.sync_changes_from_table()
        path, _ = QFileDialog.getOpenFileName(self.view, "Juntar com...", self._get_initial_dir(), "Parquet (*.parquet)")
        if path:
            with wait_cursor():
                other = FaceDataset.from_parquet(path)
                new_ds = self.dataset.concat(other)
                self.carregar_dataset_na_tela(new_ds)
            QMessageBox.information(self.view, "Merge", f"Datasets unidos. Total: {len(new_ds)} imagens.")
        self.set_dirty(True)

    def fluxo_split(self):
        if not self.dataset: return
        self.sync_changes_from_table()
        
        split_dialog = SplitDatasetDialogController(self.dataset, parent_view=self.view)
        
        if split_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            config = split_dialog.get_split_config()
            
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                if config["mode"] == "ratio":
                    ds_a, ds_b = self.dataset.split_by_identity(
                        split_ratio=config["ratio"]
                    )
                else:
                    ds_a, ds_b = self.dataset.split_by_identity_list(
                        ids_for_dataset_b=config["manual_ids_b"]
                    )
                
                ds_a.name = config["name_a"]
                ds_b.name = config["name_b"]
                
                path_a = self.pipeline.storage_path / f"{ds_a.name}.parquet"
                path_b = self.pipeline.storage_path / f"{ds_b.name}.parquet"
                
                ds_a.to_parquet(path_a)
                ds_b.to_parquet(path_b)
                
                QApplication.restoreOverrideCursor()
                
                QMessageBox.information(
                    self.view, 
                    "Sucesso", 
                    f"Divisão concluída com sucesso!\n\nSalvos em:\n- {path_a.name}\n- {path_b.name}"
                )

                action = config["post_action"]
                if action == 1:
                    self.carregar_dataset_na_tela(ds_a, path=str(path_a))
                elif action == 2:
                    self.carregar_dataset_na_tela(ds_b, path=str(path_b))
                
            except Exception as e:
                QApplication.restoreOverrideCursor()
                QMessageBox.critical(self.view, "Erro", f"Erro durante a divisão do dataset:\n{e}")

    def fluxo_filtrar(self):
        if not self.dataset: return
        # IMPORTANTE: setOverrideCursor PRIMEIRO + processEvents pra
        # forçar o repaint do cursor ANTES de qualquer trabalho síncrono.
        # `sync_changes_from_table()` em datasets grandes (1M+ rows)
        # bloqueia a main thread por segundos — sem o defer abaixo, o
        # cursor só virava ampulheta lá no fim, parecendo que o clique
        # não foi recebido.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # repaint imediato do cursor
        # Defer TODO o trabalho síncrono pro próximo tick — sync + abrir
        # dialog vão acontecer com a ampulheta já visível.
        QTimer.singleShot(0, self._abrir_dialogo_filtro_continuar)

    def _abrir_dialogo_filtro_continuar(self):
        # PR 21 (Tema 4.2) — sincroniza edições in-table (incluindo
        # colunas custom no model._df) pra `dataset.faces[i].extra_data`
        # antes de abrir o dialog. Sem isso, custom columns adicionadas/
        # editadas via UI ficariam só no model e seriam invisíveis pra
        # `apply_filter`.
        self.sync_changes_from_table()
        if not self.dataset_original:
            self.dataset_original = self.dataset
        self._abrir_dialogo_filtro_real()

    def _abrir_dialogo_filtro_real(self):
        try:
            # library_root vem de app_config.paths.library — usado pelo
            # filter_dialog pra Salvar/Abrir filtros na biblioteca global.
            library_root = getattr(
                getattr(self.config, "paths", None), "library", None,
            )
            filter_ctrl = FilterDialogController(
                self.dataset_original,
                self.current_filter_config,
                parent=self.view,
                library_root=library_root,
            )
            QApplication.restoreOverrideCursor()
            if filter_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                QApplication.restoreOverrideCursor()
                self.current_filter_config = filter_ctrl.get_config()
                self._iniciar_worker_filtro()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Erro ao abrir filtros: {e}")

    def _iniciar_worker_filtro(self):
        self._run_worker(
            FilterWorker(self.dataset_original, self.current_filter_config),
            label="Aplicando Filtros...",
            allow_cancel=True,
            determinate=True,
            wait_cursor=True,
            on_finished=self._on_filtro_concluido,
        )

    def _on_filtro_concluido(self, filtered_ds):
        if filtered_ds is None:
            # Cancelamento cooperativo do usuário.
            return
        try:
            if len(filtered_ds) > 0:
                self.dataset = filtered_ds
                self.table_ctrl.load_dataframe(self.dataset.to_pandas())
                # Propaga o NOVO dataset filtrado pros widgets que
                # resolvem paths/atributos contra ele. Sem isso, o
                # detail_panel/gallery continuam apontando pro dataset
                # original (e a face selecionada antes do filtro pode
                # nem estar mais no resultado).
                self.detail_ctrl.set_dataset(self.dataset)
                # Reseta a face exibida no painel direito — sem isso
                # o usuário vê uma face que pode nem estar no
                # resultado filtrado (ex: filtrou só M e ficou
                # mostrando a F que estava selecionada antes).
                self.detail_ctrl.load_face_data(None)
                # Idem pra galeria
                if hasattr(self.gallery_widget, "set_dataset"):
                    self.gallery_widget.set_dataset(self.dataset)
                self.gallery_widget.clear_cache()
                if self._view_stack.currentIndex() == 1:
                    self.gallery_widget.populate(list(self.dataset.faces))
                else:
                    self.gallery_widget.populate([])
                text, tooltip = self._build_dataset_status(filtered=True)
                self.toolbar_ctrl.update_filter_status(
                    text, is_filtered=True, tooltip=tooltip,
                )
            else:
                QMessageBox.warning(self.view, "Filtro", "Nenhum resultado encontrado com estes critérios.")
        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao aplicar filtro: {e}")

    def fluxo_limpar_filtro(self):
        if not self.dataset_original:
            return
        # Recarregar a tabela com o dataset original pode levar segundos
        # em datasets grandes (BFW 20k → 1-2s; webface 1M → minutos).
        # WaitCursor cobre o trabalho síncrono na main thread.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.dataset = self.dataset_original
            self.dataset_original = None
            self.current_filter_config = FilterConfig()
            self.carregar_dataset_na_tela(self.dataset, self.current_path)

            text, tooltip = self._build_dataset_status(filtered=False)
            self.toolbar_ctrl.update_filter_status(
                text, is_filtered=False, tooltip=tooltip,
            )
        finally:
            QApplication.restoreOverrideCursor()

    def fluxo_enriquecer(self):
        if not self.dataset: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_enrich_real)

    def _abrir_enrich_real(self):
        try:
            self.sync_changes_from_table()
            
            ff_cfg = getattr(self.config, "ff", None)
            enrich_ctrl = EnrichDatasetController(
                self.dataset,
                parent=self.view,
                ff_available=self.ff_available,
                ofiq_available=self.ofiq_available,
                default_use_gpu=bool(getattr(ff_cfg, "use_gpu", False)),
                default_batch_size=int(getattr(ff_cfg, "batch_size", 32)),
            )
            
            QApplication.restoreOverrideCursor()
            
            if enrich_ctrl.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                config = enrich_ctrl.get_config()
                self._iniciar_worker_enrich(config)
                
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Erro ao abrir janela de enriquecimento: {e}")

    def _iniciar_worker_enrich(self, config: DatasetIngestionConfig):
        self._run_worker(
            EnrichWorker(self.pipeline, self.dataset, config),
            label="Enriquecendo Dataset",
            allow_cancel=True,
            determinate=True,
            use_log_dialog=True,
            on_finished=self._on_enrich_concluido,
        )

    def _on_enrich_concluido(self, dataset):
        if dataset:
            # Reusa `carregar_dataset_na_tela` pra refresh completo:
            # propaga column_specs atualizados (que agora incluem
            # embedding_v2/fiqa_score/ofiq_*/serfiq_score) pro table
            # model + delegates, repopula gallery, e propaga o dataset
            # pros widgets que resolvem paths em modo RELATIVE. Sem
            # isso, o "Mostrar/Ocultar Colunas" ficava cego pras
            # colunas adicionadas pelo enrichment.
            self.carregar_dataset_na_tela(dataset, self.current_path)
            # PR 25 (Tema 5.4) — registra ENRICH (sem hash; o parquet
            # já foi resalvo pelo enrich_existing_dataset, mas a próxima
            # save manual rehasha).
            self._audit(
                AuditOperation.ENRICH,
                f"Enriquecimento aplicado em '{dataset.name}'.",
                metadata={
                    "rows": len(self.dataset),
                    "identities": self.dataset.number_of_identities,
                },
            )
            # `enrich_existing_dataset` SOBRESCREVE o parquet em disco
            # diretamente — não passa por `fluxo_salvar`. Pra que o
            # Strategy Manager V2 (cross-tab) detecte a mudança e
            # propague atualização de hash + cascade dos derivados,
            # disparamos o signal aqui manualmente. Crítico pro fluxo
            # de enriquecimento (caso mais comum de "dataset modificado
            # externamente"): sem isso, derivados continuariam ✅
            # apontando pra entrada de uma versão pré-enriquecimento.
            if self.current_path:
                self.dataset_saved_to_project.emit(str(self.current_path))
            QMessageBox.information(self.view, "Sucesso", "Enriquecimento concluído com sucesso!")
            self.set_dirty(True)
        else:
            QMessageBox.warning(self.view, "Cancelado", "Operação cancelada ou finalizada sem dados.")

    # ── Balanceamento por identidade (botão "Balancear" da toolbar) ──
    def fluxo_balancear(self):
        if not self.dataset:
            return
        with wait_cursor():
            self.sync_changes_from_table()
            from .balance_dataset_dialog import BalanceDatasetDialog
            dialog = BalanceDatasetDialog(
                dataset_name=self.dataset.name or "Sem Nome",
                parent=self.view,
            )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        from .workers.balance_dataset_worker import BalanceDatasetWorker
        worker = BalanceDatasetWorker(
            dataset=self.dataset,
            mode=dialog.mode(),
            target_n=dialog.target_n(),
            selection=dialog.selection(),
            random_seed=dialog.random_seed(),
            quality_metric=dialog.quality_metric(),
            output_name=None,  # service usa default <name>_Balanced
        )
        self._balance_replace_current = dialog.replace_current()
        self._run_worker(
            worker,
            label="Balanceando dataset…",
            allow_cancel=True,
            determinate=False,
            use_log_dialog=False,
            on_finished=self._on_balance_concluido,
        )

    def _on_balance_concluido(self, result):
        if result is None:
            QMessageBox.warning(
                self.view, "Cancelado",
                "Balanceamento cancelado ou finalizado sem resultado.",
            )
            return

        # `result` é um DatasetBalancingResult; `.dataset` é o
        # FaceDataset balanceado.
        balanced = result.dataset
        summary_extra = []
        if result.discarded_identities > 0:
            summary_extra.append(
                f"{result.discarded_identities} ID(s) descartada(s) "
                f"(FLOOR_AND_CAP)"
            )
        if result.quality_fallback_count > 0:
            summary_extra.append(
                f"{result.quality_fallback_count} sample(s) sem métrica "
                "(usados em ordem original)"
            )
        extra_msg = (
            "\n\n⚠ " + " · ".join(summary_extra) if summary_extra else ""
        )

        replace = getattr(self, "_balance_replace_current", False)
        try:
            self._balance_replace_current = False
            if replace:
                # Substitui o dataset atual pelo balanceado, marcando
                # dirty (perito decide se Salva ou Salva Como).
                self.carregar_dataset_na_tela(
                    balanced, path=None,
                )
                self.set_dirty(True)
                QMessageBox.information(
                    self.view, "Balanceamento aplicado",
                    f"Dataset substituído pelo balanceado.\n\n"
                    f"{result.summary()}{extra_msg}\n\n"
                    f"Use 'Salvar Como' pra gravar como novo parquet "
                    f"(o original em disco continua intacto)."
                )
            else:
                # Modo "manter atual": só mostra o resumo, não muta a
                # tela. Útil pra ver o impacto sem comprometer o
                # dataset corrente.
                QMessageBox.information(
                    self.view, "Pré-visualização de balanceamento",
                    f"Balanceamento simulado (não aplicado ao dataset "
                    f"corrente):\n\n{result.summary()}{extra_msg}\n\n"
                    f"Marque 'Substituir o dataset corrente' no "
                    f"diálogo se quiser aplicar.",
                )
        except Exception as e:
            QMessageBox.critical(
                self.view, "Erro ao aplicar balanceamento", str(e),
            )