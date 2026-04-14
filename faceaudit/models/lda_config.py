from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

T = TypeVar("T", bound="LDAConfig")

@dataclass
class LDAConfig(JSONSerializableMixin):
    """
    Configuração para a etapa de Linear Discriminant Analysis (LDA).
    """
    n_components: Optional[int] = field(default=None)
    regularization: float = field(default=1e-4)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
        # O LDA trabalha com primitivos puros, a instigação é direta
        return cls(**data)