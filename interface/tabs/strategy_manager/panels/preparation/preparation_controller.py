# interface/tabs/strategy_manager/panels/preparation/preparation_controller.py
from PyQt6 import QtWidgets, QtCore
from pathlib import Path
from typing import List, Union, TYPE_CHECKING

# Imports do Domínio
from faceaudit.enums.strategy_type import StrategyType
from faceaudit.enums.lr_model_type import LRModelType

# Imports dos Controladores de Cards
from .cards.plda_card_controller import PLDACardController
from .cards.kde_card_controller import KDECardController

if TYPE_CHECKING:
    from faceaudit.manager.strategies_manager import StrategiesManager

class PreparationController(QtCore.QObject):
    """
    Controlador da aba de Preparação (Coluna 1).
    Gerencia o ciclo de vida dos Cards e notifica o sistema sobre o estado dos dados.
    """
    
    # --- SINAIS ---
    strategy_ready = QtCore.pyqtSignal(object) 
    strategy_created = QtCore.pyqtSignal(str, object)  # (strategy_id, StrategyType)
    strategy_removed = QtCore.pyqtSignal(str, object)  # (strategy_id, StrategyType)
    workspace_cleared = QtCore.pyqtSignal()

    def __init__(self, view, config, strategies_manager: 'StrategiesManager'):
        super().__init__()
        self.view = view
        self.config = config 
        self.strategies_manager = strategies_manager
        
        self._active_card_controllers: List[Union[PLDACardController, KDECardController]] = []
        self._card_counter = 0 

        self._update_internal_paths()

        # Conexões da View
        if hasattr(self.view, "add_strategy_requested"):
            self.view.add_strategy_requested.connect(self.add_new_strategy)

        print(f"[PreparationController] Iniciado e vinculado ao StrategiesManager.")

    def update_config(self, new_config):
        """Atualiza preferências globais nos cards."""
        self.config = new_config
        self._update_internal_paths()
        
        for card_ctrl in self._active_card_controllers:
            if hasattr(card_ctrl, 'update_global_config'):
                card_ctrl.update_global_config(new_config)

    def _update_internal_paths(self):
        self.datasets_dir = Path(self.config.paths.datasets)
        self.pipelines_dir = Path(self.config.paths.pipelines)
        self.strategies_dir = Path(self.config.paths.strategies)

    # =========================================================================
    # GESTÃO DE CARDS (UI x BACKEND)
    # =========================================================================

    def sync_with_backend(self):
        """
        Reconstrói a interface visual baseada no estado atual do backend.
        Chamado pelo ProjectManager após carregar um arquivo .fap.
        """
        for ctrl in self._active_card_controllers:
            ctrl.deleteLater()
        self._active_card_controllers.clear()
        self._card_counter = 0
        self.view.clear_cards()

        active_managers = self.strategies_manager.get_all_managers()
        
        for strat_id, manager_instance in active_managers.items():
            config = manager_instance.get_config()
            if not config:
                continue

            self._card_counter += 1
            card_ctrl = None
            
            if config.type == LRModelType.PLDA_STRATEGY:
                card_ctrl = PLDACardController(
                    view_idx=self._card_counter, 
                    strategy_id=strat_id,
                    manager=manager_instance,
                    global_config=self.config
                )
                card_ctrl.strategy_type = StrategyType.PLDA

            elif config.type == LRModelType.KDE_STRATEGY:
                card_ctrl = KDECardController(
                    view_idx=self._card_counter,
                    strategy_id=strat_id,
                    manager=manager_instance,
                    global_config=self.config
                )
                card_ctrl.strategy_type = StrategyType.KDE

            if card_ctrl:
                card_ctrl.preparation_finished.connect(lambda ctrl=card_ctrl: self.strategy_ready.emit(ctrl))
                card_ctrl.request_destruction.connect(self._remove_strategy_card)
                
                self._active_card_controllers.append(card_ctrl)
                self.view.add_card_widget(card_ctrl.get_widget())
                self.strategy_created.emit(strat_id, card_ctrl.strategy_type)

    def add_new_strategy(self, strat_type_str: str):
        """Cria uma nova estratégia no backend e desenha o card na tela."""
        self._card_counter += 1
        card_ctrl = None

        try:
            strat_enum = StrategyType(strat_type_str)
            strategy_id = self.strategies_manager.create_strategy(strat_enum)
            manager_instance = self.strategies_manager.get_manager(strategy_id)

            if strat_enum == StrategyType.PLDA:
                card_ctrl = PLDACardController(
                    view_idx=self._card_counter, 
                    strategy_id=strategy_id,
                    manager=manager_instance,
                    global_config=self.config
                )
            elif strat_enum == StrategyType.KDE:
                card_ctrl = KDECardController(
                    view_idx=self._card_counter,
                    strategy_id=strategy_id,
                    manager=manager_instance,
                    global_config=self.config
                )

            if card_ctrl:
                card_ctrl.strategy_type = strat_enum
                
                card_ctrl.preparation_finished.connect(lambda ctrl=card_ctrl: self.strategy_ready.emit(ctrl))
                card_ctrl.request_destruction.connect(self._remove_strategy_card)
                
                self._active_card_controllers.append(card_ctrl)
                self.view.add_card_widget(card_ctrl.get_widget())
                self.strategy_created.emit(strategy_id, strat_enum)

        except Exception as e:
            import traceback
            print("\n" + "="*50)
            print(f"ERRO NO PREPARATION CONTROLLER (Card ID: {self._card_counter})")
            traceback.print_exc()
            print("="*50 + "\n")
            QtWidgets.QMessageBox.critical(self.view, "Erro", f"Falha ao criar estratégia: {e}")

    def _remove_strategy_card(self, card_controller):
        """Remove do backend e do visual (Apaga do HD pois o usuário clicou no X)."""
        if card_controller in self._active_card_controllers:
            sid = getattr(card_controller, 'strategy_id', None)
            stype = getattr(card_controller, "strategy_type", None)
            
            if sid:
                # Garante que os arquivos sejam excluídos da pasta do projeto
                self.strategies_manager.remove_strategy(sid, wipe_disk=True)
            
            self._active_card_controllers.remove(card_controller)
            self.view.remove_card_widget(card_controller.get_widget())
            
            if sid and stype:
                self.strategy_removed.emit(sid, stype)
            
            card_controller.deleteLater()

    def clear_workspace(self):
        """Chamado ao fechar um projeto. Apenas descarrega a RAM e a Tela."""
        for controller in list(self._active_card_controllers):
            # wipe_disk=False garante que os arquivos físicos sejam mantidos
            self.strategies_manager.remove_strategy(controller.strategy_id, wipe_disk=False)
            
            controller.get_widget().setParent(None)
            controller.get_widget().deleteLater()
            controller.deleteLater()
            
        self._active_card_controllers.clear()
        self.workspace_cleared.emit()

    def get_active_controllers(self):
        return self._active_card_controllers

    # --- NOVO MÉTODO PARA GARANTIR A FONTE DA VERDADE NO SALVAMENTO ---
    def force_sync_ui_to_managers(self):
        """
        Varre todos os cards da UI, extrai o DTO atualizado e força o Manager
        a atualizar seu estado interno antes de salvar.
        """
        for card_ctrl in self._active_card_controllers:
            try:
                # 1. Pega o DTO limpo da UI (get_config ao invés de get_config_dict)
                current_dto = card_ctrl.get_config()
                # 2. Força a atualização no Backend de forma silenciosa
                card_ctrl.manager.update_config(current_dto, logger_callback=lambda x: None)
            except Exception as e:
                print(f"Aviso: Não foi possível sincronizar o card {card_ctrl.strategy_id} antes de salvar: {e}")
    
    def refresh_all_cards(self):
        """
        Atualiza as listas de arquivos e a interface de todos os cards ativos.
        Utilizado após o salvamento para que os datasets/pipelines globais que
        foram copiados (Deep Copy) apareçam na seção 'Projeto Ativo (Local)'.
        """
        for card_ctrl in self._active_card_controllers:
            # Pede para o card vasculhar o disco novamente (agora achará os arquivos locais)
            if hasattr(card_ctrl, '_init_data'):
                card_ctrl._init_data()
            
            # Reconecta as opções que estavam selecionadas no DTO
            if hasattr(card_ctrl, '_sync_ui_from_config'):
                card_ctrl._sync_ui_from_config()