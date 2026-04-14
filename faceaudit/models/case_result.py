from dataclasses import dataclass
from typing import Optional
# Ajuste o import conforme a estrutura do seu projeto. 

from .embedding_info import EmbeddingInfo 

@dataclass
class CaseResult:
    """
    Representa o resultado forense de uma comparação 1:1.
    """
    strategy_name: str
    probe: 'EmbeddingInfo'      # Aspas para evitar erro de referência circular
    reference: 'EmbeddingInfo'
    
    # Resultados Matemáticos
    score: float           # Score bruto (Cosseno ou Latente)
    lr: float              # Likelihood Ratio
    
    # A Escala Verbal é armazenada aqui como texto fixo
    verbal_support: str    
    
    # Ground Truth (Opcional)
    ground_truth: Optional[bool] = None