from dataclasses import dataclass, field
from typing import Any, Optional, Type, TypeVar

# Mixin de Serialização
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Importa o Enum Rico
from faceaudit.enums.transformation_type import TransformationType

T = TypeVar("T", bound="TransformatorConfig")

@dataclass
class TransformatorConfig(JSONSerializableMixin):
    """
    Representa a configuração de um transformador individual dentro de um pipeline de transformação.
    """
    
    type: TransformationType = field(metadata={"help": "O tipo do transformador (ex: PCA, L2_NORM)."})
    
    # Sai o "params: Dict", entra o "config: Any" que armazenará os DTOs polimórficos
    config: Any = field(default=None, metadata={"help": "Instância do DTO de configuração específico."})
    
    name: Optional[str] = field(default=None, metadata={"help": "Nome opcional para identificação na UI."})

    def __post_init__(self):
        """Preenche o nome de exibição se o usuário não tiver fornecido um."""
        if self.name is None and self.type:
            self.name = self.type.name if hasattr(self.type, 'name') else str(self.type)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o transformador garantindo a conversão do Enum e 
        usando o Factory Pattern para hidratar o DTO correto.
        """
        if not data:
            return None
            
        t_type = None
        if 'type' in data:
            t_type = TransformationType.from_str(data.get('type'))
            
        # Busca o dicionário cru. Suporte legado para projetos que usavam a chave 'params'
        raw_config = data.get('config')
        if raw_config is None and 'params' in data:
            raw_config = data.get('params')
            
        parsed_config = None
        if t_type and t_type.dto_class and raw_config is not None:
            # A Mágica: Delega a reconstrução para o DTO específico (ex: WhiteningConfig.from_dict)
            parsed_config = t_type.dto_class.from_dict(raw_config)
        elif raw_config is not None:
            # Fallback para transformadores sem DTO mapeado ou genéricos
            parsed_config = raw_config

        return cls(
            type=t_type, 
            config=parsed_config, 
            name=data.get('name')
        )