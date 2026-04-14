from typing import Dict, Optional, Callable

from faceaudit.enums.strategy_type import StrategyType
from faceaudit.protocols.strategy_manager_protocol import StrategyManagerProtocol

class StrategiesManager:
    """
    Gerenciador Central de Estratégias (Single Source of Truth).
    Mantém o registro de todas as instâncias de estratégias ativas e gera IDs sequenciais.
    """
    
    def __init__(self):
        # Armazena as instâncias ativas: { 'plda_1': manager_instance }
        self._active_managers: Dict[str, StrategyManagerProtocol] = {}
        
        # Contadores para geração de IDs bonitos (ex: plda_1, kde_2)
        self._counters: Dict[StrategyType, int] = {
            StrategyType.PLDA: 0,
            StrategyType.KDE: 0
        }
        
        # O Fio do Alarme Global
        self._global_dirty_callback: Optional[Callable[[], None]] = None

    def set_global_dirty_callback(self, callback: Callable[[], None]) -> None:
        """Define a função que será chamada quando qualquer estratégia for criada, alterada ou removida."""
        self._global_dirty_callback = callback

    def create_strategy(self, strategy_type: StrategyType, force_id: Optional[str] = None) -> str:
        """
        Cria uma nova estratégia e gera um ID amigável.
        Se `force_id` for fornecido (ex: ao carregar um projeto), ele é utilizado.
        """
        # 1. Obtém a classe do Enum e instancia
        ManagerClass = strategy_type.strategy_class
        new_manager = ManagerClass()
        
        # --- A MÁGICA AQUI: Conecta o fio do alarme na nova estratégia ---
        if self._global_dirty_callback and hasattr(new_manager, 'set_dirty_callback'):
            new_manager.set_dirty_callback(self._global_dirty_callback)
            
            # Se for uma criação nova (não originada de um load de projeto via force_id), o projeto já nasce sujo
            if not force_id:
                self._global_dirty_callback()
        
        # 2. Determina o ID
        if force_id:
            strategy_id = force_id
            # Atualiza o contador para evitar colisões futuras se o usuário adicionar mais cards
            try:
                num = int(force_id.split("_")[-1])
                if num > self._counters[strategy_type]:
                    self._counters[strategy_type] = num
            except (ValueError, IndexError):
                pass
        else:
            self._counters[strategy_type] += 1
            strategy_id = f"{strategy_type.name.lower()}_{self._counters[strategy_type]}"
        
        # 3. Registra
        self._active_managers[strategy_id] = new_manager
        
        return strategy_id

    def remove_strategy(self, strategy_id: str, wipe_disk: bool = False) -> None:
        """Remove a estratégia da memória. Se wipe_disk=True, apaga os arquivos."""
        if strategy_id in self._active_managers:
            manager = self._active_managers[strategy_id]
            
            # Repassa a instrução para o Manager específico
            manager.clear_state(wipe_disk=wipe_disk) 
            del self._active_managers[strategy_id]
            
            # --- Notifica que o projeto foi alterado (perdeu uma estratégia) ---
            if wipe_disk and self._global_dirty_callback:
                self._global_dirty_callback()

    def get_manager(self, strategy_id: str) -> Optional[StrategyManagerProtocol]:
        return self._active_managers.get(strategy_id)

    def get_all_managers(self) -> Dict[str, StrategyManagerProtocol]:
        return self._active_managers