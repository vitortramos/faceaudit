import os
from pathlib import Path
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QProgressDialog, QApplication

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

        # --- 1. SUB-CONTROLADORES ---
        self.toolbar_ctrl = ManagerToolbarController(parent_layout=None)
        self.table_ctrl = ManagerTableController(parent_layout=None)
        self.detail_ctrl = DetailPanelController(parent_layout=None)

        if hasattr(self.view, 'toolbar_layout'):
            self.view.toolbar_layout.addWidget(self.toolbar_ctrl.view)
        if hasattr(self.view, 'table_layout'):
            self.view.table_layout.addWidget(self.table_ctrl.view)
        if hasattr(self.view, 'detail_layout'):
            self.view.detail_layout.addWidget(self.detail_ctrl.view)

        # --- 2. SINAIS ---
        self._connect_submodules()

        # --- 3. PRÉ-ALOCAÇÃO ---
        self._init_loading_dialog()

    def _init_loading_dialog(self):
        self.progress = QProgressDialog("Inicializando...", None, 0, 0, parent=self.view)
        self.progress.setWindowTitle("Aguarde")
        self.progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setCancelButton(None)
        self.progress.setRange(0, 0) 
        self.progress.setValue(0)
        self.progress.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint | Qt.WindowType.WindowTitleHint)
        self.progress.close() 

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

        if hasattr(self.toolbar_ctrl, 'action_enrich'):
            self.toolbar_ctrl.action_enrich.connect(self.fluxo_enriquecer)

    def _on_table_data_changed(self, *_args):
        if self.dataset is not None:
            self.set_dirty(True)

    def _get_initial_dir(self) -> str:
        if self.pipeline and hasattr(self.pipeline, 'storage_path'):
            path = str(self.pipeline.storage_path)
            if os.path.isdir(path): return path
        return ""

    def carregar_dataset_na_tela(self, dataset, path=None):
        self.dataset = dataset
        self.current_path = path
        self.table_ctrl.load_dataframe(self.dataset.to_pandas())
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
        self.progress.setLabelText("Aguardando seleção do arquivo...")
        self.progress.show()
        
        ag = self.view.geometry()
        dg = self.progress.geometry()
        self.progress.move(ag.center().x() - dg.width() // 2, ag.center().y() - dg.height() // 2)
        
        QApplication.processEvents()

        path, _ = QFileDialog.getOpenFileName(
            self.view, "Abrir Dataset", self._get_initial_dir(), "Parquet (*.parquet)"
        )
        
        self.progress.raise_()
        self.progress.activateWindow()
        QApplication.processEvents()
        
        if not path:
            self.progress.hide()
            return

        self.progress.setLabelText("Lendo arquivo Parquet...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        self._iniciar_worker_load(path)

    def _iniciar_worker_load(self, path):
        try:
            self.thread = QtCore.QThread()
            self.worker = LoadDatasetWorker(path)
            self.worker.moveToThread(self.thread)
            
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(lambda ds: self._on_load_success(ds, path))
            self.worker.error.connect(self._on_error)
            
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            
            self.thread.start()

        except Exception as e:
            QApplication.restoreOverrideCursor()
            if hasattr(self, 'progress'): self.progress.hide()
            raise e

    def _on_load_success(self, dataset, path):
        if hasattr(self, 'progress'):
            self.progress.setLabelText("Construindo tabela...")
            QApplication.processEvents()

        try:
            self.carregar_dataset_na_tela(dataset, path)
        except Exception as e:
            QMessageBox.critical(self.view, "Erro de Renderização", f"Falha ao exibir tabela: {e}")
        finally:
            if hasattr(self, 'progress'):
                self.progress.hide()
            QApplication.restoreOverrideCursor()

    def _on_error(self, msg):
        if hasattr(self, 'progress'): 
            self.progress.hide()
        QApplication.restoreOverrideCursor()
        QMessageBox.critical(self.view, "Erro", msg)

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
        templates_path = Path(self.config.paths.templates)
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
        self.progress_create = QProgressDialog("Processando Ingestão...", "Cancelar", 0, 100, self.view)
        self.progress_create.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_create.setMinimumDuration(0)
        
        self.thread = QtCore.QThread()
        self.worker = NewDatasetWorker(self.pipeline, config)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(lambda msg, pct: self.progress_create.setLabelText(msg))
        self.worker.progress_update.connect(lambda msg, pct: self.progress_create.setValue(pct))
        
        self.worker.finished.connect(self._on_criacao_concluida)
        self.worker.error.connect(self._on_error)
        self.progress_create.canceled.connect(self.worker.stop)
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def _on_criacao_concluida(self, dataset):
        if hasattr(self, 'progress_create'): self.progress_create.close()
        
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
            
            # Busca o serviço diretamente do container, livre de erros de namespace
            comp_svc = self.container.comparison_service()
            
            # Inicializa a injeção explicitamente
            analysis_ctrl = DatasetAnalysisController(
                dataset=self.dataset, 
                comparison_service=comp_svc, 
                parent_view=self.view
            )
            QApplication.restoreOverrideCursor()
            analysis_ctrl.exec()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.view, "Erro", f"Falha ao abrir Auditoria: {e}")

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
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.progress_filter = QProgressDialog("Aplicando Filtros...", None, 0, 100, self.view)
        self.progress_filter.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_filter.setCancelButton(None)

        self.thread = QtCore.QThread()
        self.worker = FilterWorker(self.dataset_original, self.current_filter_config)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_filter.setValue)

        self.worker.finished.connect(self._on_filtro_concluido)

        self.worker.error.connect(self._on_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()
        self.progress_filter.exec()

    def _on_filtro_concluido(self, filtered_ds):
        if hasattr(self, 'progress_filter'): self.progress_filter.close()
        
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
        finally:
            QApplication.restoreOverrideCursor()

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
        self.progress_enrich = QProgressDialog("Enriquecendo Dataset...", "Cancelar", 0, 100, self.view)
        self.progress_enrich.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        self.progress_enrich.setMinimumDuration(0)
        
        self.thread = QtCore.QThread()
        
        self.worker = EnrichWorker(self.pipeline, self.dataset, config)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress_update.connect(lambda msg, pct: self.progress_enrich.setLabelText(msg))
        self.worker.progress_update.connect(lambda msg, pct: self.progress_enrich.setValue(pct))
        
        self.worker.finished.connect(self._on_enrich_concluido)
        self.worker.error.connect(self._on_error)
        self.progress_enrich.canceled.connect(self.worker.stop) 
        
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.thread.start()

    def _on_enrich_concluido(self, dataset):
        if hasattr(self, 'progress_enrich'): self.progress_enrich.close()
        
        if dataset:
            self.dataset = dataset
            self.table_ctrl.load_dataframe(self.dataset.to_pandas())
            QMessageBox.information(self.view, "Sucesso", "Enriquecimento concluído com sucesso!")
            self.set_dirty(True)
        else:
            QMessageBox.warning(self.view, "Cancelado", "Operação cancelada ou finalizada sem dados.")