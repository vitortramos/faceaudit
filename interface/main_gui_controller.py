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
        self._init_strategy_manager_v2()
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
        try:
            pipeline_extracao = self.container.feature_extraction_pipeline()

            self.manager_view = DatasetManagerView()

            self.manager_controller = DatasetManagerController(
                view=self.manager_view,
                pipeline=pipeline_extracao,
                config=self.app_config,
                container=self.container # <--- PASSAMOS O CONTAINER AQUI
            )

            self.view.add_tab(0, self.manager_view, "Gerenciador de Datasets")
        except Exception as e:
            print(f"⚠ Aba 'Gerenciador de Datasets' desativada (FFService falhou): {e}")
            self.manager_controller = None

    # No método _init_strategy_manager()

    def _init_strategy_manager(self):
        """
        Inicializa a aba de gerenciamento de estratégias.
        """
        try:
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
        except Exception as e:
            print(f"⚠ Aba 'Gerenciador de Estratégias' (legada) desativada: {e}")
            self.strategy_controller = None

    def _init_strategy_manager_v2(self):
        """
        Inicializa a aba Strategy Manager V2 (nova arquitetura declarativa).
        Convive com a aba antiga durante a migração.
        """
        from interface.tabs.strategy_manager_v2.strategy_manager_v2_controller import (
            StrategyManagerV2Controller,
        )
        from faceaudit.storage.global_library_storage import GlobalLibraryStorage
        from pathlib import Path

        # Tenta construir GlobalLibraryStorage:
        # 1. preferência: config.json → paths.library
        # 2. fallback: ~/.faceaudit/library (criada automaticamente se não existir)
        # Sem o storage, o botão "Salvar como Template" e a importação da
        # biblioteca ficam indisponíveis silenciosamente.
        global_storage = None
        try:
            library_root = getattr(self.app_config.paths, "library", None)
            if not library_root:
                library_root = str(Path.home() / ".faceaudit" / "library")
            global_storage = GlobalLibraryStorage(library_root=Path(library_root))
            global_storage.create_library_structure()
        except Exception:
            # Biblioteca opcional — a aba funciona sem ela; save_as_template
            # exibirá uma mensagem clara ao usuário.
            global_storage = None

        self.strategy_manager_v2 = StrategyManagerV2Controller(
            parent=self.view,
            global_storage=global_storage,
            app_config=self.app_config,
        )
        self.view.add_tab(2, self.strategy_manager_v2.view, "Estratégias (Nova Arch)")

    def _setup_connections(self):
        """Estabelece a comunicação via sinais."""
        self.view.config_requested.connect(self.abrir_dialogo_configuracoes)
        
        if self.manager_controller is not None and hasattr(self.manager_controller, 'request_new_dataset'):
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
        if getattr(self, 'manager_controller', None) is not None:
            self.manager_controller.config = self.app_config
            if hasattr(self.manager_controller.pipeline, 'storage_path'):
                self.manager_controller.pipeline.storage_path = Path(self.app_config.paths.datasets)

        if getattr(self, 'strategy_controller', None) is not None and hasattr(self.strategy_controller, 'on_config_changed'):
            self.strategy_controller.on_config_changed()
    
    def handle_app_close(self, event):
        """
        Gatekeeper Final: Intercepta o fechamento do aplicativo inteiro.
        Pergunta a cada aba se elas podem ser fechadas.
        """
        # 1. Pergunta para a Aba 1 (Dataset)
        if getattr(self, 'manager_controller', None) is not None:
            pode_fechar_dataset = self.manager_controller.request_close()
            if not pode_fechar_dataset:
                event.ignore()
                return

        # 2. Pergunta para a Aba 2 (Estratégias / Projeto) — arch antiga
        if getattr(self, 'strategy_controller', None) is not None:
            pode_fechar_projeto = self.strategy_controller.request_close()
            if not pode_fechar_projeto:
                event.ignore()
                return

        # 3. Pergunta para a Aba 3 (Estratégias V2 / Nova Arch)
        if hasattr(self, 'strategy_manager_v2'):
            pode_fechar_v2 = self.strategy_manager_v2.request_close()
            if not pode_fechar_v2:
                event.ignore()
                return

        # 4. Se todo mundo retornou True (está tudo salvo ou vazio), deixa a janela fechar
        event.accept()