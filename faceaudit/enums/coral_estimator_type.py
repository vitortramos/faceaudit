from __future__ import annotations
from enum import Enum
from typing import Optional

class CoralEstimatorType(Enum):
    """
    Define os métodos de estimativa de covariância para o algoritmo CORAL.
    Contém metadados para UI (Label e Tooltip).
    """
    
    STANDARD = (
        "standard", 
        "Padrão (Grid Search)", 
        "Método clássico do CORAL. Executa uma busca (Grid Search) para encontrar o valor fixo de regularização que maximiza a separação das classes.\n\nRecomendado quando há dados de calibração suficientes."
    )
    
    LEDOIT_WOLF = (
        "ledoit_wolf", 
        "Ledoit-Wolf (Robusto)", 
        "Estimador de covariância com regularização analítica (Shrinkage). Minimiza o erro quadrático médio (MSE) matematicamente.\n\nIdeal quando o Dataset Target é pequeno (poucas fotos), pois evita distorções geométricas severas."
    )
    
    OAS = (
        "oas", 
        "OAS (Oracle Shrinkage)", 
        "Oracle Approximating Shrinkage. Uma evolução do Ledoit-Wolf que converge iterativamente para uma estimativa melhor.\n\nGeralmente supera o Ledoit-Wolf quando o número de amostras é muito menor que a dimensão do embedding."
    )

    def __new__(cls, value: str, label: str, description: str) -> CoralEstimatorType:
        """Construtor que define múltiplos atributos por item sem herdar de str."""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.label = label
        obj.description = description
        return obj

    def __str__(self) -> str:
        """Retorna o label amigável para exibição em combos e logs."""
        return self.label

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Converte para a chave string (ex: 'standard') para o JSON."""
        return self.value

    @classmethod
    def from_str(cls, value: str) -> CoralEstimatorType:
        """
        Reconstrói o Enum a partir da chave string. 
        Substitui o antigo 'get_by_value' por uma implementação nativa e segura.
        """
        try:
            # cls(value) busca automaticamente pelo atributo ._value_ definido no __new__
            return cls(value.lower() if value else "")
        except ValueError:
            return cls.STANDARD