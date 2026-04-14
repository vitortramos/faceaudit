from typing import List, Protocol, Any, Dict, Optional, Tuple, Callable
from pathlib import Path

from faceaudit.models.face_dataset import FaceDataset

class StrategyManagerProtocol(Protocol):
    """
    Define o contrato para gerenciadores de estratégia (ex: PLDAManager, KDEManager).
    Evoluído para suportar Workspace com Lazy Loading, Paths Relativos e 
    Gestão de Memória (RAM vs DISK).
    """

    # --- 0. Notificações de Estado ---
    
    def set_dirty_callback(self, callback: Callable[[], None]) -> None:
        """Injeta a função que será chamada quando o estado do manager mudar."""
        ...

    # --- 1. Gestão de Configuração e Estado ---
    
    def update_config(self, config: Any) -> None:
        """Atualiza a configuração (o DTO do Card) e invalida caches se necessário."""
        ...

    def get_config(self) -> Any:
        """Retorna a configuração (o DTO do Card) atual."""
        ...

    # --- 2. Validação (Check de Sanidade da Configuração) ---
    
    def validate_all(self) -> Any:
        """
        Executa a bateria de checks de sanidade (datasets, pipelines, parâmetros).
        Retorna o DTO de resultado (KDEValidationResult ou PLDAValidationResult).
        """
        ...

    # --- 3. I/O de Dados de Treino (Preparação) ---

    def load_input_datasets(self) -> None:
        """Carrega fisicamente os arquivos de imagem/vetores para a memória do Manager."""
        ...

    def run_preparation(self, logger_callback: Any = print) -> Dict[str, Any]:
        """Executa pipelines e agregações. Retorna estatísticas pós-preparo."""
        ...

    def is_prepared(self) -> bool:
        """Informa se os dados/pipelines estão prontos para o treinamento."""
        ...

    # --- 4. Ciclo de Vida do Modelo (RAM vs DISCO) ---

    def run_training(self, logger_callback: Any = print) -> str:
        """
        Treina a estratégia, salva no diretório base e mantém a instância carregada na RAM.
        (Removido o parâmetro output_path explícito pois o manager usa o manifesto interno).
        """
        ...

    def get_model_path(self) -> Optional[Path]:
        """Retorna o caminho onde o modelo está (ou deveria estar) no disco."""
        ...

    def set_model_path(self, path: Path) -> None:
        """
        Define o caminho do modelo (usado pelo Workspace ao carregar projetos).
        Isso não carrega o modelo imediatamente (Lazy).
        """
        ...

    def has_model_on_disk(self) -> bool:
        """Verifica se existe um arquivo físico no caminho registrado."""
        ...

    def is_model_loaded(self) -> bool:
        """Informa se a instância da estratégia está na RAM."""
        ...

    def ensure_model_loaded(self) -> None:
        """
        Garante que o modelo esteja pronto para uso. 
        Se não estiver na RAM, tenta carregar do disco usando o path registrado.
        Levanta erro se o arquivo não existir.
        """
        ...

    def unload_model(self) -> None:
        """Remove a estratégia da RAM (seta para None), preservando apenas o path no disco."""
        ...

    # --- 5. Limpeza e Reset ---

    def clear_state(self, wipe_disk: bool = False, logger_callback: Any = print) -> None:
        """
        Limpa TUDO: dados de treino carregados, pipelines processados e modelos em RAM.
        Reseta o manager para o estado inicial.
        """
        ...
    
    # --- 6. Execução e Análise (Coluna 3 da UI) ---

    def run_validation_inference(self, 
                                 probe_dataset: FaceDataset, 
                                 reference_dataset: FaceDataset, 
                                 logger_callback: Callable[[str], None] = print) -> Tuple[List[float], List[float]]:
        """
        Gera LRs para a Arena de Validação. 
        Mapeia identidades e devolve (genuine_lrs, impostor_lrs).
        """
        ...

    def run_inference(self, 
                      probe_dataset: FaceDataset, 
                      reference_dataset: FaceDataset, 
                      logger_callback: Callable[[str], None] = print) -> List[float]:
        """
        Gera LRs para a Aplicação Forense (Coluna 3). 
        Devolve uma lista flat de scores na ordem dos pares gerados.
        """
        ...
    
    def get_training_datasets(self) -> List[str]:
        """
        Retorna a lista de nomes dos datasets utilizados para treinar/calibrar esta estratégia.
        Usado pelo ValidationManager para auditar Data Leakage.
        """
        ...