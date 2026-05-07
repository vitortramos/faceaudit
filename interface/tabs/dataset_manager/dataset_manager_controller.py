import os
from pathlib import Path
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
from .workers.filter_worker import FilterWorker
from .workers.new_dataset_worker import NewDatasetWorker
from .workers.enrich_worker import EnrichWorker
from .workers.worker_progress_runner import WorkerProgressRunner

# --- MODELOS ---
from faceaudit.models import FaceDataset
from faceaudit.models.filter_config import FilterConfig
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig


class DatasetManagerController(QtCore.QObject):
    """
    Controlador Mestre do Gerenciamento de Datasets.
    """
    
    dataset_changed = QtCore.pyqtSignal() 

    # RECEBE O CONTAINER AQUI
    def __init__(self, view, pipeline, config, container):
        super().__init__()
        self.view = view
        self.pipeline = pipeline
        self.config = config
        self.container = container # Guardamos a referência do container para injeção explícita

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

        if hasattr(self.toolbar_ctrl, 'action_enrich'):
            self.toolbar_ctrl.action_enrich.connect(self.fluxo_enriquecer)

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
        )
        cmp_ctrl.exec()

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

        name, label, dtype, default = dialog.get_result()
        ok = self.table_ctrl.model.add_custom_column(
            name=name, label=label, dtype=dtype, default_value=default, editable=True,
        )
        if not ok:
            QMessageBox.warning(self.view, "Erro", "Falha ao adicionar coluna.")
            return

        # Re-propaga + marca dirty
        self._on_schema_changed()
        self.table_ctrl.view.resizeColumnsToContents()

    def _get_initial_dir(self) -> str:
        if self.pipeline and hasattr(self.pipeline, 'storage_path'):
            path = str(self.pipeline.storage_path)
            if os.path.isdir(path): return path
        return ""

    def carregar_dataset_na_tela(self, dataset, path=None):
        self.dataset = dataset
        self.current_path = path

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
        
        qtd_imgs = len(self.dataset)
        qtd_ids = self.dataset.number_of_identities
        nome = self.dataset.name if self.dataset.name else "Sem Nome"
        status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
        self.toolbar_ctrl.set_dataset_loaded_state(True, status_txt)

    def sync_changes_from_table(self):
        if not self.dataset: return
        df_edited = self.table_ctrl.get_current_dataframe()
        if df_edited is None or df_edited.empty: return

        dataset_visual_atualizado = FaceDataset.from_dataframe(df_edited, name=self.dataset.name)
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
            qtd_imgs = len(self.dataset)
            qtd_ids = self.dataset.number_of_identities
            nome = self.dataset.name if self.dataset.name else "Sem Nome"
            asterisco = "*" if state else ""
            status_txt = f"📂 <b>{nome}{asterisco}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
            self.toolbar_ctrl.set_dataset_loaded_state(True, status_txt)

    def fluxo_abrir_dataset(self):
        path, _ = QFileDialog.getOpenFileName(
            self.view, "Abrir Dataset", self._get_initial_dir(), "Parquet (*.parquet)"
        )
        if not path:
            return
        self._run_worker(
            LoadDatasetWorker(path),
            label="Lendo arquivo Parquet...",
            allow_cancel=False,
            determinate=False,  # pyarrow não emite progresso
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
            self.dataset.name = Path(path).stem
            self.dataset.to_parquet(path)
            self.current_path = path
            self.set_dirty(False)
            self.carregar_dataset_na_tela(self.dataset, path)

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

        self.dataset = None
        self.current_path = None
        self.dataset_original = None
        self.table_ctrl.load_dataframe(None)
        self.detail_ctrl.load_face_data(None)
        self.toolbar_ctrl.set_dataset_loaded_state(False, "Nenhum dataset carregado.")
        self.set_dirty(False)
        return True

    def fluxo_fechar_dataset(self):
        self.request_close()

    def fluxo_novo_dataset(self):
        templates_path = Path(self.config.paths.dir_templates)
        dialog = NewDatasetController(self.view, templates_path=templates_path)
        
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
        if not self.dataset_original: self.dataset_original = self.dataset
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_dialogo_filtro_real)

    def _abrir_dialogo_filtro_real(self):
        try:
            filter_ctrl = FilterDialogController(
                self.dataset_original, 
                self.current_filter_config, 
                parent=self.view
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
                qtd_imgs = len(self.dataset)
                qtd_ids = self.dataset.number_of_identities
                nome = self.dataset.name if self.dataset.name else "Sem Nome"
                status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs (Filtrado)"
                self.toolbar_ctrl.update_filter_status(status_txt, is_filtered=True)
            else:
                QMessageBox.warning(self.view, "Filtro", "Nenhum resultado encontrado com estes critérios.")
        except Exception as e:
            QMessageBox.critical(self.view, "Erro", f"Erro ao aplicar filtro: {e}")

    def fluxo_limpar_filtro(self):
        if self.dataset_original:
            self.dataset = self.dataset_original
            self.dataset_original = None
            self.current_filter_config = FilterConfig() 
            self.carregar_dataset_na_tela(self.dataset, self.current_path)
            
            qtd_imgs = len(self.dataset)
            qtd_ids = self.dataset.number_of_identities
            nome = self.dataset.name
            status_txt = f"📂 <b>{nome}</b> | 🖼️ {qtd_imgs} Imagens | 👤 {qtd_ids} IDs"
            self.toolbar_ctrl.update_filter_status(status_txt, is_filtered=False)

    def fluxo_enriquecer(self):
        if not self.dataset: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QTimer.singleShot(10, self._abrir_enrich_real)

    def _abrir_enrich_real(self):
        try:
            self.sync_changes_from_table()
            
            enrich_ctrl = EnrichDatasetController(self.dataset, parent=self.view)
            
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
            self.dataset = dataset
            self.table_ctrl.load_dataframe(self.dataset.to_pandas())
            QMessageBox.information(self.view, "Sucesso", "Enriquecimento concluído com sucesso!")
            self.set_dirty(True)
        else:
            QMessageBox.warning(self.view, "Cancelado", "Operação cancelada ou finalizada sem dados.")