from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.whitening_style import WhiteningStyle

T = TypeVar("T", bound="WhiteningConfig")

@dataclass
class WhiteningConfig(JSONSerializableMixin):
    """
    Configuração para a etapa de Whitening (Branqueamento/PCA).
    """
    n_components: Optional[int] = field(default=None)
    style: WhiteningStyle = field(default=WhiteningStyle.PCA)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
            
        processed_data = data.copy()
        
        # Retrocompatibilidade: Se existir 'pca_components' de um arquivo JSON antigo
        if 'pca_components' in processed_data:
            if 'n_components' not in processed_data:
                processed_data['n_components'] = processed_data.pop('pca_components')
            else:
                processed_data.pop('pca_components')

        # Saneamento seguro do Enum
        if 'style' in processed_data and processed_data['style']:
            processed_data['style'] = WhiteningStyle.from_str(processed_data['style'])
            
        return cls(**processed_data)