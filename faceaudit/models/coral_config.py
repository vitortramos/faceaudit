from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.coral_estimator_type import CoralEstimatorType

T = TypeVar("T", bound="CoralConfig")

@dataclass
class CoralConfig(JSONSerializableMixin):
    """
    Configuração para a etapa de Correlation Alignment (CORAL).
    """
    regularization: float = field(default=1e-6)
    use_grid_search: bool = field(default=True)
    estimator: CoralEstimatorType = field(default=CoralEstimatorType.STANDARD)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        if not data:
            return cls()
            
        processed_data = data.copy()
        
        # Saneamento seguro do Enum
        if 'estimator' in processed_data and processed_data['estimator']:
            processed_data['estimator'] = CoralEstimatorType.from_str(processed_data['estimator'])
            
        return cls(**processed_data)