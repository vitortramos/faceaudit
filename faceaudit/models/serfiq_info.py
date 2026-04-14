from dataclasses import dataclass, asdict
from typing import Dict, Any

@dataclass
class SERFIQInfo:
    """
    Dataclass para armazenar o resultado do algoritmo SER-FIQ 
    (Score baseado na estabilidade estocástica dos embeddings).
    """
    score: float   # O score final de qualidade (0-100)
    t: int         # Número de variações estocásticas (T) utilizadas
    alpha: float   # Parâmetro de calibração (logística) - OBRIGATÓRIO

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SERFIQInfo':
        """
        Cria a instância a partir de um dicionário.
        """
        data_clean = {k.lower(): v for k, v in data.items()}
        
        # Se o dado não existir, o .get() retornará o valor padrão 
        # para não quebrar a reconstrução de arquivos antigos
        return cls(
            score=data_clean.get('score', 0.0),
            t=data_clean.get('t', 10),
            alpha=data_clean.get('alpha', 2.0) # Valor padrão sugerido para calibração
        )