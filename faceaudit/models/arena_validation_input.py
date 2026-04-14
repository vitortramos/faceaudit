from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Ajuste este import para o caminho real do seu DTO de estatísticas
from faceaudit.models.dataset_stats import DatasetStats 

@dataclass
class ArenaValidationInput:
    """
    Pacote de dados contendo apenas os metadados (estatísticas e inventários)
    necessários para auditar a Arena sem carregar imagens na RAM.
    """
    probe_stats: Optional[DatasetStats] = None
    reference_stats: Optional[DatasetStats] = None
    
    # Mapeia: ID da Estratégia -> Lista de Stats dos datasets usados no treino dela
    strategies_history_stats: Dict[str, List[DatasetStats]] = field(default_factory=dict)