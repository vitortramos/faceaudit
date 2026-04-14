import sys
from pathlib import Path
from PyQt6 import QtWidgets

# Import do serviço e modelos
from faceaudit.services.config_loader_service import ConfigLoaderService
from faceaudit.containers import ApplicationContainer

# Imports de Componentes (MVC) - CORRIGIDOS COM 'interface.'
from interface.config_dialog import ConfigDialog
from interface.main_gui_view import MainGuiView
from interface.tabs.dataset_manager.dataset_manager_view import DatasetManagerView
from interface.tabs.dataset_manager.dataset_manager_controller import DatasetManagerController
from interface.tabs.strategy_manager.strategy_manager_controller import StrategyManagerController
from interface.tabs.strategy_manager.strategy_manager_view import StrategyManagerView

class MainGuiController:
    """
    Controlador mestre da interface gráfica (Maestro da Aplicação).
    """

    def __init__(self, view: 'MainGuiView', container: ApplicationContainer, config_file: str):
        self.view = view
        self.container = container
        
        # 1. Inicializa o serviço de configuração
        self.config_service = ConfigLoaderService(config_file)
        self.app_config = self.config_service.load()

        # 2. Sincroniza Container (Backend) com Config (JSON)
        self._sync_container_config()

        # 3. Inicializa as abas
        self._init_dataset_manager()
        self._init_strategy_manager()
        self._setup_connections()

    def _sync_container_config(self):
        """
        Sincroniza o AppConfig (JSON) com o ApplicationContainer (Backend).
        """
        cfg = self.app_config

        # 1. Mapeamento Forensic Face
        self.container.config.forensic_face.model.from_value(cfg.ff.model)
        self.container.config.forensic_face.det_size.from_value(cfg.ff.det_size)

        # 2. Mapeamento de Sistema
        self.container.config.system.gpu_enabled.from_value(cfg.ff.use_gpu)

        # 3. Mapeamento de Caminhos OFIQ
        self.container.config.paths.ofiq.exe_path.from_value(cfg.ofiq.ofiq_exe_path)
        self.container.config.paths.ofiq.data_path.from_value(cfg.ofiq.ofiq_data_path)
        
        # 4. Mapeamento de Caminhos Gerais
        self.container.config.paths.datasets.from_value(cfg.paths.datasets)

        # 5. Mapeamento SERFIQ
        self.container.config.strategies.serfiq.t_augmentations.from_value(cfg.serfiq.t)
        self.container.config.strategies.serfiq.alpha.from_value(cfg.serfiq.alpha)

        print("✅ [MainGuiController] Sincronização de dependências concluída.")

    def _init_dataset_manager(self):
        """
        Inicializa a aba de gerenciamento de datasets.
        """
        pipeline_extracao = self.container.feature_extraction_pipeline()
        
        self.manager_view = DatasetManagerView()
        
        self.manager_controller = DatasetManagerController(
            view=self.manager_view, 
            pipeline=pipeline_extracao,
            config=self.app_config,
            container=self.container # <--- PASSAMOS O CONTAINER AQUI
        )
        
        self.view.add_tab(0, self.manager_view, "Gerenciador de Datasets")

    # No método _init_strategy_manager()

    def _init_strategy_manager(self):
        """
        Inicializa a aba de gerenciamento de estratégias.
        """
        self.strategy_view = StrategyManagerView(parent=self.view)
        
        # 1. Recupera os Singletons do Container
        strategies_manager_backend = self.container.strategies_manager()
        project_manager_backend = self.container.project_manager()
        validation_manager_backend = self.container.validation_manager() # <--- NOVO

        # 2. Injeta no Controller
        self.strategy_controller = StrategyManagerController(
            view=self.strategy_view,
            config_service=self.config_service,
            strategies_manager=strategies_manager_backend,
            project_manager=project_manager_backend,
            validation_manager=validation_manager_backend # <--- PASSANDO ADIANTE
        )
        self.view.add_tab(1, self.strategy_view, "Gerenciador de Estratégias")

    def _setup_connections(self):
        """Estabelece a comunicação via sinais."""
        self.view.config_requested.connect(self.abrir_dialogo_configuracoes)
        
        if hasattr(self.manager_controller, 'request_new_dataset'):
            self.manager_controller.request_new_dataset.connect(
                self.manager_controller.abrir_dialog_novo_dataset
            )
        # Sequestra o botão "X" da janela principal
        self.view.closeEvent = self.handle_app_close

    def abrir_dialogo_configuracoes(self):
        """Executa o diálogo de configurações."""
        current_config = self.config_service.load()
        dialog = ConfigDialog(current_config, parent=self.view)
        
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.config_service.save(current_config)
            self.app_config = current_config
            self._sync_container_config()
            self._propagate_config_change()

    def _propagate_config_change(self):
        """Notifica sub-controladores sobre alterações."""
        if hasattr(self, 'manager_controller'):
            self.manager_controller.config = self.app_config
            if hasattr(self.manager_controller.pipeline, 'storage_path'):
                self.manager_controller.pipeline.storage_path = Path(self.app_config.paths.datasets)
        
        if hasattr(self.strategy_controller, 'on_config_changed'):
            self.strategy_controller.on_config_changed()
    
    def handle_app_close(self, event):
        """
        Gatekeeper Final: Intercepta o fechamento do aplicativo inteiro.
        Pergunta a cada aba se elas podem ser fechadas.
        """
        # 1. Pergunta para a Aba 1 (Dataset)
        if hasattr(self, 'manager_controller'):
            pode_fechar_dataset = self.manager_controller.request_close()
            if not pode_fechar_dataset:
                event.ignore()
                return

        # 2. Pergunta para a Aba 2 (Estratégias / Projeto)
        if hasattr(self, 'strategy_controller'):
            pode_fechar_projeto = self.strategy_controller.request_close()
            if not pode_fechar_projeto:
                event.ignore()
                return

        # 3. Se todo mundo retornou True (está tudo salvo ou vazio), deixa a janela fechar
        event.accept()