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
        self._init_strategy_manager_v2()
        self._setup_connections()

    def _sync_container_config(self):
        """Sincroniza o SystemConfig (JSON) com o ApplicationContainer.

        O escopo é mínimo agora: só campos que efetivamente são lidos pelos
        providers do container. Parâmetros operacionais (det_size, T, alpha,
        modelo de embedding) saíram daqui e vivem no DatasetIngestionConfig
        por execução.
        """
        cfg = self.app_config

        # GPU capability + batch size — lidos pelo FFService Singleton.
        self.container.config.ff.use_gpu.from_value(cfg.ff.use_gpu)
        self.container.config.ff.batch_size.from_value(cfg.ff.batch_size)

        # OFIQ executable + data path — lidos pelo OFIQService Singleton.
        self.container.config.ofiq.ofiq_exe_path.from_value(cfg.ofiq.ofiq_exe_path)
        self.container.config.ofiq.ofiq_data_path.from_value(cfg.ofiq.ofiq_data_path)

        # Library root — `_datasets_dir` provider deriva
        # `library/datasets` para o storage_path do feature_extraction_pipeline.
        self.container.config.paths.library.from_value(cfg.paths.library)

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

    def _init_strategy_manager_v2(self):
        """Inicializa a aba 'Gerenciador de Estratégias' (nova arquitetura
        declarativa). A tab antiga foi removida — esta agora é a única."""
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
            library_root = self.app_config.paths.library or str(
                Path.home() / ".faceaudit" / "library"
            )
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
        self.view.add_tab(1, self.strategy_manager_v2.view, "Gerenciador de Estratégias")

    def _setup_connections(self):
        """Estabelece a comunicação via sinais."""
        self.view.config_requested.connect(self.abrir_dialogo_configuracoes)

        if self.manager_controller is not None and hasattr(self.manager_controller, 'request_new_dataset'):
            self.manager_controller.request_new_dataset.connect(
                self.manager_controller.abrir_dialog_novo_dataset
            )

        # PR 17 — cross-tab: Dataset Manager → Strategy Manager V2.
        # Quando o usuário clica "🎯 → Estratégia" no Dataset Manager,
        # trocamos pra aba V2 e delegamos o import. Só conectamos se
        # ambos os lados existem (V2 pode estar desativada se faltar
        # storage; Dataset Manager pode estar desativado se FFService
        # falhou).
        if (
            self.manager_controller is not None
            and hasattr(self.manager_controller, "send_to_strategy_requested")
            and getattr(self, "strategy_manager_v2", None) is not None
        ):
            self.manager_controller.send_to_strategy_requested.connect(
                self._on_send_dataset_to_strategy
            )

        # Strategy Manager V2 → Dataset Manager: clicou em "Abrir no
        # Gerenciador" no painel de propriedades de um DATASET primário.
        # Troca de aba e carrega o parquet do projeto. Save no Dataset
        # Manager dispara `dataset_saved_to_project` que volta pro V2
        # via `_on_dataset_saved_to_project` pra atualizar hash + cascade.
        if (
            getattr(self, "strategy_manager_v2", None) is not None
            and self.manager_controller is not None
            and hasattr(self.manager_controller, "open_dataset_from_path")
        ):
            self.strategy_manager_v2.open_dataset_in_manager_requested.connect(
                self._on_open_dataset_in_manager_requested
            )
        if (
            self.manager_controller is not None
            and hasattr(self.manager_controller, "dataset_saved_to_project")
            and getattr(self, "strategy_manager_v2", None) is not None
        ):
            self.manager_controller.dataset_saved_to_project.connect(
                self.strategy_manager_v2.on_external_dataset_saved
            )

        # Sequestra o botão "X" da janela principal
        self.view.closeEvent = self.handle_app_close

    def _on_send_dataset_to_strategy(self, parquet_path: str) -> None:
        """Hook chamado pelo Dataset Manager via signal (PR 17). Troca de
        aba pro V2 e delega o import — V2 cuida do prompt pra abrir/criar
        projeto se necessário."""
        try:
            self.view.main_tabs.setCurrentWidget(self.strategy_manager_v2.view)
        except Exception:
            # View pode ter sido destruída em algum cenário inesperado;
            # tentamos por índice como fallback.
            try:
                self.view.main_tabs.setCurrentIndex(1)
            except Exception:
                pass
        self.strategy_manager_v2.import_dataset_from_path(parquet_path)

    def _on_open_dataset_in_manager_requested(self, parquet_path: str) -> None:
        """Hook reverso: Strategy Manager V2 → Dataset Manager. Troca de
        aba e carrega o parquet via `open_dataset_from_path`."""
        if not parquet_path or self.manager_controller is None:
            return
        try:
            self.view.main_tabs.setCurrentWidget(self.manager_controller.view)
        except Exception:
            try:
                self.view.main_tabs.setCurrentIndex(0)
            except Exception:
                pass
        try:
            self.manager_controller.open_dataset_from_path(parquet_path)
        except Exception:
            pass  # falha de load mostra erro pelo próprio Dataset Manager

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

    def handle_app_close(self, event):
        """Gatekeeper final: intercepta o fechamento da aplicação e pergunta
        a cada aba se ela pode ser fechada (cada uma pode salvar antes)."""
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        # 1. Aba Gerenciador de Datasets
        if getattr(self, 'manager_controller', None) is not None:
            if not self.manager_controller.request_close():
                event.ignore()
                return

        # 2. Aba Gerenciador de Estratégias (nova arch)
        if hasattr(self, 'strategy_manager_v2'):
            if not self.strategy_manager_v2.request_close():
                event.ignore()
                return

        # WaitCursor cobre o teardown síncrono que vem depois do
        # accept() — Qt libera widgets, datasets em memória (1M+ faces
        # podem demorar segundos pra deallocate), etc. SEM restore —
        # o cursor morre junto com a app, e não queremos "piscar" pra
        # o cursor normal antes do shutdown completar.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # força repaint imediato do cursor
        event.accept()