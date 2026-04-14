# faceaudit/pipelines/transformation_pipeline.py
import json
import pickle
from typing import List, Optional, Union, Type, Dict, Callable
from pathlib import Path

# Models e Enums
from faceaudit.models import EmbeddingDataset
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.step_type import StepType

# Serviços
from faceaudit.services.embedding_math_service import EmbeddingMathService

# Transformadores Concretos
from faceaudit.transformators.l2_normalization import L2Normalization
from faceaudit.transformators.whitening_transformation import WhiteningTransformation
from faceaudit.transformators.coral_transformation import CoralTransformation
from faceaudit.transformators.identity_aggregation import IdentityAggregation
from faceaudit.transformators.lda_transformation import LdaTransformation
from faceaudit.transformators.centering_transformation import CenteringTransformation

class TransformationPipeline:
    """
    Entidade que representa uma sequência executável de transformações.
    
    Responsabilidade:
    - Manter a lista de transformadores instanciados.
    - Executar a transformação sequencial (transform).
    - Serializar/Deserializar (save/load).
    
    Nota: A lógica de treinamento (fit) e sincronia Source/Target foi movida 
    para o PLDATransformationPipelinesController.
    """

    _TRANSFORMATOR_MAP: Dict[TransformationType, Type[EmbeddingTransformationProtocol]] = {
        TransformationType.L2_NORMALIZATION: L2Normalization,
        TransformationType.WHITENING_TRANSFORMATION: WhiteningTransformation,
        TransformationType.CORAL_TRANSFORMATION: CoralTransformation,
        TransformationType.IDENTITY_AGGREGATION: IdentityAggregation,
        TransformationType.LDA_TRANSFORMATION: LdaTransformation,
        TransformationType.CENTERING_TRANSFORMATION: CenteringTransformation
    }

    def __init__(self,
                 config: TransformationPipelineConfig,
                 math_service: Optional[EmbeddingMathService] = None):
        
        self.config = config
        self.math_service = math_service or EmbeddingMathService()
        
        # Lista ordenada de passos executáveis
        self.steps: List[EmbeddingTransformationProtocol] = []
        self._build()

    def _build(self):
        """Instancia os objetos baseados na configuração."""
        self.steps = []
        for t_config in self.config.transformators:
            t_class = self._TRANSFORMATOR_MAP.get(t_config.type)
            if not t_class:
                raise ValueError(f"Tipo desconhecido: {t_config.type}")
            
            # Instanciação Limpa: Passamos o DTO diretamente!
            # Se o transformador for stateless (como L2), o t_config.config será None, 
            # e a classe simplesmente o ignorará (pois ela nem exige isso no __init__).
            transformator = t_class(
                math_service=self.math_service,
                name=t_config.name,
                config=t_config.config
            )
            self.steps.append(transformator)

    def transform(self, dataset: EmbeddingDataset, checkpoint_dir: Optional[Path] = None) -> EmbeddingDataset:
        current_data = dataset
        
        # 1. Salva 0_Raw (Estado Inicial)
        if checkpoint_dir:
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                path = checkpoint_dir / "0_Raw.parquet"
                
                # AQUI: Usamos StepType.RAW explicitamente
                raw_type_str = StepType.RAW.value 
                
                meta = {
                    "step_type": raw_type_str,
                    "step_metrics_json": json.dumps({
                        "step_type": raw_type_str,
                        "description": "Original embeddings",
                        "n_samples": len(current_data.items)
                    })
                }
                current_data.to_parquet(path, extra_metadata=meta)
            except Exception as e:
                print(f"[Pipeline] ⚠️ Erro ao salvar 0_Raw: {e}")

        # 2. Loop de Transformações
        for i, step in enumerate(self.steps):
            try:
                current_data = step.transform(current_data)
            except Exception as e:
                raise e

            if checkpoint_dir:
                try:
                    # --- A MÁGICA ACONTECE AQUI ---
                    # O Enum TransformationType já sabe qual é o seu StepType correspondente.
                    # Acessamos .value para pegar a string (ex: "CORAL", "WHITENING")
                    step_enum_val = step.transformation_type.step_type.value
                    
                    filename = f"{i+1}_{step_enum_val}.parquet"
                    path = checkpoint_dir / filename
                    
                    # Coleta métricas do passo e injeta o tipo
                    metrics = step.get_execution_metrics()
                    metrics["step_type"] = step_enum_val 
                    
                    meta = {
                        "step_type": step_enum_val,
                        "step_metrics_json": json.dumps(metrics)
                    }
                    
                    current_data.to_parquet(path, extra_metadata=meta)
                    
                except Exception as e:
                    print(f"[Pipeline] ⚠️ Erro ao salvar checkpoint {i+1}: {e}")

        return current_data

    def save(self, path: Union[str, Path]):
        """Serializa o pipeline treinado."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TransformationPipeline':
        """Carrega do disco."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Pipeline não encontrado: {p}")
        with open(p, 'rb') as f:
            return pickle.load(f)