# faceaudit/transformators/l2_normalization.py
import pickle
from typing import Optional, Union, Dict, Any
from pathlib import Path

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType

class L2Normalization(EmbeddingTransformationProtocol):
    """
    Transformador para aplicar Normalização L2 (Unit Length) em embeddings.
    Projeta os vetores na superfície de uma hiperesfera unitária para garantir 
    que a magnitude do vetor não influencie na comparação de cosseno.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 config: Optional[Any] = None, 
                 name: str = "L2Normalization"):
        """
        Args:
            math_service: Serviço para operações matemáticas.
            config: DTO de configuração (incluído para compatibilidade de contrato).
            name: Identificador da transformação.
        """
        self._name = name
        self.math_service = math_service
        self.config = config
        
        # L2 é stateless, mas mantemos as métricas para a Cadeia de Custódia Forense.
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str: 
        return self._name
    
    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.L2_NORMALIZATION
    
    @property
    def requires_target_dataset(self) -> bool: 
        return False

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        L2 Normalization não aprende parâmetros, mas registra a execução no manifesto.
        """
        t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)

        self._metrics = {
            "step_type": t_type_val,
            "operation": "Unit Sphere Projection (L2)",
            "target_norm": 1.0,
            "stateless": True
        }

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """
        Delega a normalização L2 para o serviço matemático.
        """
        return self.math_service.apply_l2_normalization(dataset)

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica a normalização L2 a um único embedding.
        """
        temp_dataset = EmbeddingDataset(name="temp_l2", items=[embedding_info])
        transformed_dataset = self.transform(temp_dataset)
        return transformed_dataset.items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Retorna os metadados da operação."""
        return self._metrics if self._metrics else {"status": "stateless_not_executed"}

    def save(self, path: str) -> None:
        """
        Persiste as métricas e a configuração para integridade do pipeline.
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'metrics': self._metrics,
            'config_dict': self.config.to_dict() if hasattr(self.config, 'to_dict') else None
        }
        
        with open(file_path, 'wb') as f: 
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Carrega as métricas de forma resiliente."""
        file_path = Path(path)
        if not file_path.exists():
            return

        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
                self._metrics = state.get('metrics', {})
        except Exception:
            # Falha no carregamento de métricas não impede a operação do L2
            pass

    def get_state(self) -> Dict[str, Any]:
        """Captura o estado para sincronização."""
        return {
            "_metrics": self._metrics,
            "config": self.config
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura o estado vindo de outro objeto."""
        self._metrics = state.get("_metrics", {})
        self.config = state.get("config")