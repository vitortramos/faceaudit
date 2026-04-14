from PyQt6 import QtCore, QtWidgets
from pathlib import Path

# --- WORKERS & DIALOGS ---
from .workers.project_woker import ProjectWorker
from .log_progress_dialog.log_progress_dialog import LogProgressDialog
from .new_project_dialog.new_project_controller import NewProjectController
from .save_project_dialog.save_project_controller import SaveProjectController
from .new_mining_config_dialog.new_mining_config_dialog_controller import NewMiningConfigDialogController
from .pipeline_creator.pipeline_creator_controller import PipelineCreatorController 
from .new_split_config_dialog.new_split_config_dialog_controller import NewSplitConfigDialogController # <--- [NOVO IMPORT]

# --- TOOLBAR & PANELS ---
from .toolbar.strategy_toolbar_controller import StrategyToolbarController
from .panels.preparation.preparation_controller import PreparationController
from .panels.validation.validation_controller import ValidationController
from .panels.analysis.analysis_controller import AnalysisController

# --- ENUMS ---
from faceaudit.enums.project_file_type import ProjectFileType 

class StrategyManagerController(QtCore.QObject):
    def __init__(self, view, config_service, strategies_manager, project_manager, validation_manager): 
        super().__init__()
        self.view = view
        self.config_service = config_service
        self.strategies_manager = strategies_manager
        self.project_manager = project_manager 
        self.validation_manager = validation_manager 

        current_config = self.config_service.load()

        # 1. Toolbar
        self.toolbar_ctrl = StrategyToolbarController(parent_layout=self.view.toolbar_layout)

        # 2. Sub-controllers
        self.ctrl_prep = PreparationController(self.view.view_prep, current_config, self.strategies_manager)
        
        # Injeta o ValidationManager diretamente no Controller da Arena
        self.ctrl_val = ValidationController(self.view.view_val, current_config, self.strategies_manager, self.validation_manager)
        
        self.ctrl_ana = AnalysisController(self.view.view_ana, current_config)
        # 3. Propagação Interna de Sinais
        self.ctrl_prep.strategy_created.connect(self.ctrl_val.on_strategy_created)
        self.ctrl_prep.strategy_removed.connect(self.ctrl_val.on_strategy_removed)
        self.ctrl_prep.workspace_cleared.connect(self.ctrl_val.on_workspace_cleared)
        self.ctrl_prep.strategy_ready.connect(self._propagate_strategy)

        # 4. Conexões da Toolbar
        self._connect_toolbar()

        # Estado inicial
        self._set_project_open(False)
        self._active_worker = None

    def _connect_toolbar(self):
        self.toolbar_ctrl.action_create.connect(self._handle_new_project)
        self.toolbar_ctrl.action_add_strategy.connect(self.ctrl_prep.add_new_strategy)
        self.toolbar_ctrl.action_close.connect(self._handle_close_project)
        self.toolbar_ctrl.action_save.connect(self._handle_save_project)
        self.toolbar_ctrl.action_open.connect(self._handle_load_project)
        
        self.toolbar_ctrl.action_new_mining_config.connect(self._handle_new_mining_config)
        self.toolbar_ctrl.action_new_pipeline.connect(self._handle_new_pipeline) 
        self.toolbar_ctrl.action_new_split.connect(self._handle_new_split) # <--- [NOVO] Conexão do Split

    # =========================================================================
    # --- FLUXO DE CRIAÇÃO DE ATIVOS GLOBAIS (MINERAÇÃO, PIPELINE, SPLIT) ---
    # =========================================================================
    
    def _handle_new_mining_config(self):
        """Abre o dialog para criar uma regra de mineração global."""
        dialog = NewMiningConfigDialogController(self.view)
        
        if dialog.exec():
            try:
                # 1. Pega o DTO montado da tela
                config_dto = dialog.get_config()
                
                # 2. Delega para o ProjectManager salvar no ambiente global
                saved_path = self.project_manager.save_global_asset(
                    file_type=ProjectFileType.MINING_CONFIG,
                    data=config_dto,
                    logger_callback=lambda msg: print(msg), 
                    config_name=config_dto.name 
                )
                
                # 3. Atualiza os ComboBoxes dos Cards abertos
                if hasattr(self, 'ctrl_prep'):
                    self.ctrl_prep.force_sync_ui_to_managers() 
                    for ctrl in self.ctrl_prep.get_active_controllers():
                        if hasattr(ctrl, '_init_data') and hasattr(ctrl, '_sync_ui_from_config'):
                            ctrl._init_data() 
                            ctrl._sync_ui_from_config() 

                # 4. Feedback visual
                QtWidgets.QMessageBox.information(
                    self.view, 
                    "Sucesso", 
                    f"Estratégia de Mineração criada com sucesso!\n\nEla já está disponível para uso nos cards do KDE.\n\nSalvo em: {saved_path.name}"
                )
                self.toolbar_ctrl.set_status_text(f"⛏️ Regra global '{config_dto.name}' salva.")
                
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self.view, "Erro ao Salvar", f"Falha ao salvar a regra de mineração:\n\n{e}"
                )

    def _handle_new_pipeline(self):
        """Abre o dialog para criar um pipeline de transformação global."""
        dialog = PipelineCreatorController(self.view)
        
        new_config = dialog.run() 
        
        if new_config:
            try:
                # O PipelineCreator higieniza o nome internamente, mas vamos garantir
                clean_name = "".join(x for x in new_config.name if x.isalnum() or x in ('_', '-'))
                new_config.name = clean_name

                # Salva usando o ProjectManager
                saved_path = self.project_manager.save_global_asset(
                    file_type=ProjectFileType.PIPELINE_CONFIG,
                    data=new_config,
                    logger_callback=lambda msg: print(msg),
                    pipeline_name=new_config.name 
                )

                # Atualiza os ComboBoxes de todos os cards
                if hasattr(self, 'ctrl_prep'):
                    self.ctrl_prep.force_sync_ui_to_managers() 
                    for ctrl in self.ctrl_prep.get_active_controllers():
                        if hasattr(ctrl, '_init_data') and hasattr(ctrl, '_sync_ui_from_config'):
                            ctrl._init_data() 
                            ctrl._sync_ui_from_config() 

                # Feedback visual
                QtWidgets.QMessageBox.information(
                    self.view, 
                    "Sucesso", 
                    f"Pipeline de Transformação criado com sucesso!\n\nEle já está disponível para uso nos cards do PLDA.\n\nSalvo em: {saved_path.name}"
                )
                self.toolbar_ctrl.set_status_text(f"🛠️ Pipeline global '{new_config.name}' salvo.")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self.view, "Erro ao Criar", f"Falha ao salvar o Pipeline:\n\n{e}"
                )

    def _handle_new_split(self):
        """[NOVO] Abre o dialog para criar uma regra de corte de dados (Split)."""
        dialog = NewSplitConfigDialogController(self.view)
        
        new_config = dialog.exec()
        
        if new_config:
            try:
                # Higieniza o nome do arquivo para o disco
                clean_name = "".join(x for x in new_config.name if x.isalnum() or x in ('_', '-'))
                
                # Salva usando o ProjectManager como um JSON global
                saved_path = self.project_manager.save_global_asset(
                    file_type=ProjectFileType.SPLIT_CONFIG, # Usa o enum correto
                    data=new_config,
                    logger_callback=lambda msg: print(msg),
                    config_name=clean_name # Nomeia o arquivo JSON
                )

                # Atualiza os ComboBoxes de todos os cards abertos
                if hasattr(self, 'ctrl_prep'):
                    self.ctrl_prep.force_sync_ui_to_managers() 
                    for ctrl in self.ctrl_prep.get_active_controllers():
                        if hasattr(ctrl, '_init_data') and hasattr(ctrl, '_sync_ui_from_config'):
                            ctrl._init_data() 
                            ctrl._sync_ui_from_config() 

                # Feedback visual de sucesso
                QtWidgets.QMessageBox.information(
                    self.view, 
                    "Sucesso", 
                    f"Configuração de Split criada com sucesso!\n\nEla já está disponível para uso nos cards do PLDA.\n\nSalvo em: {saved_path.name}"
                )
                self.toolbar_ctrl.set_status_text(f"✂️ Split global '{new_config.name}' salvo.")

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self.view, "Erro ao Criar", f"Falha ao salvar a Configuração de Split:\n\n{e}"
                )

    # =========================================================================
    # --- HANDLERS DE PERSISTÊNCIA (ORQUESTRAÇÃO ASSÍNCRONA) ---
    # =========================================================================

    def _handle_save_project(self):
        current_root = self.project_manager.fs_service.current_project_root
        
        if not current_root:
            QtWidgets.QMessageBox.warning(self.view, "Atenção", "Crie ou Abra um projeto primeiro antes de salvar.")
            return
            
        fap_path = current_root / f"{current_root.name}.fap"

        dialog_ctrl = SaveProjectController(self.view)
        if dialog_ctrl.exec():
            wants_deep_copy, wants_cleanup = dialog_ctrl.get_options()
            
            # Antes de invocar o worker de disco, nós OBRIGAMOS todos os cards
            # a vomitarem seus DTOs atualizados para dentro dos seus respectivos Managers.
            self.ctrl_prep.force_sync_ui_to_managers()

            self._run_project_task(
                operation="save", 
                file_path=str(fap_path), 
                copy_images=wants_deep_copy, 
                cleanup_orphans=wants_cleanup, 
                title="Salvando Projeto..."
            )

    def _handle_load_project(self):
        if self.ctrl_prep.get_active_controllers() or self.project_manager.is_dirty():
            if not self.request_close():
                return

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.view, "Abrir Projeto FaceAudit", "", "FaceAudit Project (*.fap)"
        )
        if file_path:
            self._run_project_task(
                operation="load", 
                file_path=file_path, 
                copy_images=False, 
                cleanup_orphans=False, 
                title="Carregando Projeto..."
            )

    def _run_project_task(self, operation: str, file_path: str, copy_images: bool, cleanup_orphans: bool, title: str):
        """Orquestra o Worker e o Dialog de progresso para operações de disco."""
        self.log_dialog = LogProgressDialog(title=title, parent=self.view)
        
        self._active_worker = ProjectWorker(
            self.project_manager, 
            operation, 
            file_path, 
            copy_images=copy_images, 
            cleanup_orphans=cleanup_orphans
        )
        
        self._active_worker.progress_log.connect(self.log_dialog.append_log)
        self._active_worker.finished_success.connect(lambda msg: self._on_task_finished(True, msg, operation))
        self._active_worker.finished_error.connect(lambda msg: self._on_task_finished(False, msg, operation))
        self.log_dialog.cancellation_requested.connect(self._active_worker.cancel)

        self._active_worker.start()
        self.log_dialog.exec()

    def _on_task_finished(self, success: bool, message: str, operation: str):
        """Finaliza o diálogo e aplica as mudanças de estado na UI."""
        self.log_dialog.finish(success)
        
        if success:
            self.toolbar_ctrl.set_status_text(message)
            
            if operation == "load":
                self._set_project_open(True)
                self.ctrl_prep.sync_with_backend()
                self.ctrl_val.sync_ui_from_backend() 
            elif operation == "save":
                self.project_manager.reset_dirty_state() 
                self._set_project_open(True)
                self.ctrl_prep.refresh_all_cards() 
                self.ctrl_val.sync_ui_from_backend() 
        else:
            QtWidgets.QMessageBox.critical(self.view, "Erro na Operação", message)
            if operation == "load":
                self._set_project_open(False)

    # =========================================================================
    # --- MÉTODOS DE ESTADO (CRIAÇÃO E FECHAMENTO DE PROJETO) ---
    # =========================================================================

    def _handle_new_project(self):
        """Abre o diálogo de criação de projeto e inicia o workspace vazio."""
        if self.project_manager.current_project_path or self.project_manager.is_dirty():
            if not self.request_close():
                return
                
        dialog = NewProjectController(self.view)
        if dialog.exec():
            fap_path = dialog.get_fap_path()
            wants_deep_copy = dialog.wants_deep_copy()
            
            self.ctrl_prep.clear_workspace() 
            
            self._run_project_task(
                operation="save", 
                file_path=str(fap_path), 
                copy_images=wants_deep_copy, 
                cleanup_orphans=False, 
                title="Criando Novo Projeto..."
            )

    def request_close(self) -> bool:
        """
        Tenta fechar o projeto (Workspace) atual. 
        Retorna True se fechou com sucesso (ou se estava vazio), e False se o usuário cancelou.
        """
        if not self.ctrl_prep.get_active_controllers() and not self.project_manager.is_dirty():
            self.ctrl_prep.clear_workspace()
            self._set_project_open(False)
            return True

        if self.project_manager.is_dirty():
            res = QtWidgets.QMessageBox.question(
                self.view, 
                "Salvar Alterações?", 
                "O projeto atual possui modificações não salvas. Deseja salvar antes de fechar?", 
                QtWidgets.QMessageBox.StandardButton.Save | 
                QtWidgets.QMessageBox.StandardButton.Discard | 
                QtWidgets.QMessageBox.StandardButton.Cancel
            )
            
            if res == QtWidgets.QMessageBox.StandardButton.Cancel: 
                return False
                
            if res == QtWidgets.QMessageBox.StandardButton.Save: 
                self._handle_save_project() 
                if self.project_manager.is_dirty(): 
                    return False

        self.ctrl_prep.clear_workspace()
        self._set_project_open(False)
        self.project_manager.reset_dirty_state()
        return True

    def _handle_close_project(self):
        """Método conectado ao botão 'Fechar' da Toolbar"""
        self.request_close()

    def _set_project_open(self, is_open: bool):
        self.toolbar_ctrl.set_workspace_open(is_open)
        if is_open:
            self.view.show_workspace_screen()
        else:
            self.view.show_welcome_screen()
            self.toolbar_ctrl.set_status_text("Nenhum projeto ativo.")

    def _propagate_strategy(self, card_controller):
        manager = card_controller.manager
        if hasattr(self.ctrl_val, 'set_current_manager'):
            self.ctrl_val.set_current_manager(manager)
        if hasattr(self.ctrl_ana, 'set_current_manager'):
            self.ctrl_ana.set_current_manager(manager)

    def on_config_changed(self):
        new_config = self.config_service.load()
        self.ctrl_prep.update_config(new_config)
        self.ctrl_val.update_config(new_config)
        self.ctrl_ana.update_config(new_config)