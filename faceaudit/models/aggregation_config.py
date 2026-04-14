from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Importação dos Enums "Ricos"
from faceaudit.enums.aggregation_method import AggregationMethod
from faceaudit.enums.quality_score_type import QualityScoreType

T = TypeVar("T", bound="AggregationConfig")

@dataclass
class AggregationConfig(JSONSerializableMixin):
    """
    Configuração independente para a etapa de Agregação de Embeddings (Smart Enrollment).
    Define como múltiplos vetores de uma mesma identidade serão combinados
    para formar um template sintético de maior qualidade.
    """
    
    # Método matemático de fusão (Ex: Média, Média Ponderada, COMBINATORIAL)
    method: AggregationMethod = field(default=AggregationMethod.CENTROID)
    
    # Se o método for COMBINATORIAL, quantos vetores formarão cada subgrupo?
    combination_group_size: int = field(default=3)
    
    # Limite máximo de templates sintéticos gerados por identidade (evita explosão combinatória)
    max_synthetic_templates_per_id: int = field(default=100)
    
    # Habilita a ponderação matemática usando scores de qualidade da face
    use_quality_weighting: bool = field(default=False)
    
    # Se a ponderação estiver habilitada, qual score usar (FIQA, OFIQ, etc)?
    quality_weight_type: Optional[QualityScoreType] = field(default=QualityScoreType.FIQA)

    @property
    def quality_key_str(self) -> Optional[str]:
        """Helper rápido para extrair a string da chave ('fiqa', 'ofiq') para o MathService."""
        if self.use_quality_weighting and self.quality_weight_type:
            # Acessa o atributo do Enum (considerando que você tem a estrutura de Enum Rico implementada)
            if hasattr(self.quality_weight_type, 'key_name'):
                return self.quality_weight_type.key_name
            return self.quality_weight_type.name.lower()
        return None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO de Agregação, garantindo o fallback e a conversão correta dos Enums.
        """
        if not data:
            return None
            
        # Puxa os dados para um novo dict para não mutar a entrada original
        processed_data = data.copy()

        # Resiliência: Converte strings de volta para Enums usando 'from_str'
        if 'method' in processed_data:
            processed_data['method'] = AggregationMethod.from_str(processed_data['method'])
            
        if 'quality_weight_type' in processed_data and processed_data['quality_weight_type']:
            processed_data['quality_weight_type'] = QualityScoreType.from_str(processed_data['quality_weight_type'])
            
        return cls(**processed_data)