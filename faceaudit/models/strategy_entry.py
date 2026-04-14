from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.strategy_type import StrategyType
from faceaudit.models.plda_config import PLDAConfig
from faceaudit.models.kde_config import KDEConfig

T = TypeVar("T", bound="StrategyEntry")

@dataclass
class StrategyEntry(JSONSerializableMixin):
    """
    Representa uma instância de estratégia persistida dentro do Workspace.
    Gerencia o polimorfismo entre diferentes tipos de modelos (PLDA/KDE),
    vinculando a configuração ao arquivo de pesos treinados.
    """
    strategy_id: str
    strategy_type: StrategyType
    model_config: Union[PLDAConfig, KDEConfig]
    
    # Caminho relativo ou absoluto para o arquivo .pkl do modelo treinado
    trained_model_path: Optional[str] = None
    
    # Metadados de auditoria (ex: data de treino, autor, métricas rápidas)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói a entrada de estratégia, resolvendo o polimorfismo do model_config
        baseado no strategy_type identificado no JSON.
        """
        if not data:
            return None

        # 1. Resolve o Enum de Tipo de forma resiliente usando o padrão from_str
        s_type = StrategyType.from_str(data.get('strategy_type'))
        data['strategy_type'] = s_type

        # 2. Resolve o model_config baseado no tipo (PLDA ou KDE)
        config_data = data.get('model_config')
        
        # A mágica do polimorfismo: o Enum StrategyType possui a propriedade 
        # 'config_class' que retorna PLDAConfig ou KDEConfig.
        if isinstance(config_data, dict) and s_type:
            target_config_class = s_type.config_class
            # Chamada recursiva para o from_dict da classe de configuração específica
            data['model_config'] = target_config_class.from_dict(config_data)

        return cls(**data)