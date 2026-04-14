# faceaudit/transformators/centering_transformation.py
import pickle
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType

class CenteringTransformation(EmbeddingTransformationProtocol):
    """
    Transformador que centraliza os dados subtraindo o vetor médio (Mean Subtraction).
    Refatorado para suportar o padrão de DTOs e Injeção de Dependência do Pipeline.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 config: Optional[Any] = None, 
                 name: str = "CenteringTransformation"):
        """
        Args:
            math_service: Serviço para operações matemáticas vetorizadas.
            config: DTO de configuração (pode ser CenteringConfig ou None).
            name: Identificador da transformação.
        """
        self._name = name
        self.math_service = math_service
        self.config = config # Armazena o DTO para manter consistência arquitetural
        
        self._mean_vector: Optional[np.ndarray] = None
        self._fitted: bool = False
        
        # Métricas de execução (para Auditoria/Manifesto)
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str: 
        return self._name

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.CENTERING_TRANSFORMATION
    
    @property
    def requires_target_dataset(self) -> bool: 
        return False

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        Calcula o vetor médio do dataset de origem.
        """
        if not source_dataset.items:
            raise ValueError(f"[{self.name}] Dataset de origem vazio para treinamento.")

        self._mean_vector = self.math_service.calculate_centering_parameters(source_dataset)
        
        # --- CÁLCULO DE MÉTRICAS PARA CADEIA DE CUSTÓDIA ---
        try:
            shift_magnitude = np.linalg.norm(self._mean_vector)
            dims = self._mean_vector.shape[0]
            
            t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)

            self._metrics = {
                "step_type": t_type_val,
                "input_dimension": int(dims),
                "shift_magnitude": float(shift_magnitude), 
                "description": "Mean vector subtraction (Centering)"
            }
            print(f"[{self.name}] Treino concluído. Magnitude do ajuste médio: {shift_magnitude:.4f}")
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Aviso ao calcular métricas: {e}")
            self._metrics = {"error": str(e)}

        self._fitted = True

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado. Chame 'fit()' primeiro.")
        
        return self.math_service.apply_centering_transformation(dataset, self._mean_vector)

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado.")
        
        # Cria dataset temporário para reutilizar lógica vetorizada
        temp = EmbeddingDataset("temp", [embedding_info])
        return self.transform(temp).items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save(self, path: str) -> None:
        """Persiste o estado treinado e as configurações no disco."""
        if not self._fitted: return
        
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'mean': self._mean_vector,
            'metrics': self._metrics,
            'config_dict': self.config.to_dict() if hasattr(self.config, 'to_dict') else None
        }
        
        with open(file_path, 'wb') as f: 
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Recupera o estado e reconstrói o transformador."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"[{self.name}] Estado não encontrado: {file_path}")
            
        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            self._mean_vector = state['mean']
            self._metrics = state.get('metrics', {})
            
            # Nota: Se precisar reconstruir o DTO config a partir do config_dict, 
            # adicione a lógica aqui (como feito em CoralTransformation)
            
            self._fitted = True
    
    def get_state(self) -> Dict[str, Any]:
        """Usado para sincronização entre pipelines Source/Target."""
        return {
            "_mean_vector": self._mean_vector,
            "_fitted": self._fitted,
            "_metrics": self._metrics,
            "config": self.config
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restaura o estado vindo de outro transformador ou cache."""
        self._mean_vector = state.get("_mean_vector")
        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))
        self.config = state.get("config")