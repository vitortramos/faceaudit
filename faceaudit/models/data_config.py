from dataclasses import dataclass, field
from typing import Optional, Union, Dict, Any, Type, TypeVar
from pathlib import Path
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Imports de domínio
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.transformator_config import TransformatorConfig
from faceaudit.enums.transformation_type import TransformationType

T = TypeVar("T", bound="DataConfig")

@dataclass
class DataConfig(JSONSerializableMixin):
    """
    Define a configuração e o estado dos datasets utilizados em um caso ou treinamento.
    """
    name: str = field(metadata={"help": "Nome único para esta configuração de dados."})
    description: Optional[str] = field(default=None, metadata={"help": "Descrição da configuração de dados."})

    # Configuração do pipeline de adaptação
    adaptation_pipeline_config: TransformationPipelineConfig = field( 
        default_factory=lambda: TransformationPipelineConfig(
            name="DefaultAdaptationPipeline",
            description="Pipeline de adaptação padrão (CORAL + Whitening + L2)",
            transformators=[
                TransformatorConfig(type=TransformationType.CORAL_TRANSFORMATION, params={'regularization': 1e-6, 'use_grid_search': True}, name="CoralDefault"),
                TransformatorConfig(type=TransformationType.WHITENING_TRANSFORMATION, params={'n_components': None}, name="WhiteningDefault"),
                TransformatorConfig(type=TransformationType.L2_NORMALIZATION, name="L2NormDefault")
            ]
        ),
        metadata={"help": "Configuração do pipeline de transformação de embeddings."}
    )

    # Paths para os datasets brutos
    raw_source_path: Optional[str] = field(default=None)
    raw_target_path: Optional[str] = field(default=None)
    raw_external_test_source_path: Optional[str] = field(default=None)
    raw_external_test_target_path: Optional[str] = field(default=None)

    # Configurações de split
    test_ratio: float = field(default=0.2)
    calibration_ratio: float = field(default=0.2)
    random_seed: int = field(default=42)

    # Objetos de Dataset (não serializados diretamente)
    plda_train_dataset: Optional[EmbeddingDataset] = field(default=None, init=False, repr=False)
    calibration_dataset: Optional[EmbeddingDataset] = field(default=None, init=False, repr=False)
    test_reference_dataset: Optional[EmbeddingDataset] = field(default=None, init=False, repr=False)
    test_probe_dataset: Optional[EmbeddingDataset] = field(default=None, init=False, repr=False)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Sobrescreve a reconstrução para inflar os objetos aninhados e Enums.
        """
        if not data:
            return None

        # 1. Trata o pipeline aninhado (que por sua vez tratará seus próprios Enums)
        if 'adaptation_pipeline_config' in data and isinstance(data['adaptation_pipeline_config'], dict):
            # Chamamos o from_dict do TransformationPipelineConfig (que vamos ajustar a seguir)
            data['adaptation_pipeline_config'] = TransformationPipelineConfig.from_dict(
                data['adaptation_pipeline_config']
            )
        
        return cls(**data)

    def __str__(self):
        raw_source_name = Path(self.raw_source_path).name if self.raw_source_path else 'N/A'
        return f"DataConfig(name='{self.name}', source='{raw_source_name}', pipeline='{self.adaptation_pipeline_config.name}')"

    def __repr__(self):
        return self.__str__()