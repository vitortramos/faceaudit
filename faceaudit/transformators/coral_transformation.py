import pickle
from typing import Optional, Union, Dict, Any
from pathlib import Path
import numpy as np

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.coral_estimator_type import CoralEstimatorType

# --- INJEÇÃO DO DTO ---
from faceaudit.models.coral_config import CoralConfig

class CoralTransformation(EmbeddingTransformationProtocol):
    """
    Transformador para aplicar CORAL (Correlation Alignment) em embeddings.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 name: str = "CoralTransformation", 
                 config: Optional[CoralConfig] = None):
        
        self._name = name
        self.math_service = math_service
        
        # Se nenhum config for passado, usamos os defaults do DTO
        self.config = config if config is not None else CoralConfig()
        
        self._coral_matrix: Optional[np.ndarray] = None
        self._final_regularization: Optional[float] = None
        self._fitted: bool = False
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.CORAL_TRANSFORMATION
    
    @property
    def requires_target_dataset(self) -> bool:
        return True

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        Ajusta os parâmetros de CORAL.
        """
        print(f"[{self.name}] Iniciando treino (fit) usando estimador: {self.config.estimator.label}...")
        
        if target_dataset is None:
            raise ValueError(f"[{self.name}] O 'target_dataset' é obrigatório para treinar a CoralTransformation.")

        if not source_dataset.items or not target_dataset.items:
            raise ValueError(f"[{self.name}] Datasets source ou target vazios para treinamento de Coral.")

        self._coral_matrix, self._final_regularization = self.math_service.calculate_coral_parameters(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            regularization=self.config.regularization,
            use_grid_search=self.config.use_grid_search,
            estimator=self.config.estimator
        )
        
        # --- CÁLCULO DE MÉTRICAS (AUDITORIA) ---
        try:
            src_sample = source_dataset.matrix
            if len(src_sample) > 2000:
                 indices = np.random.choice(len(src_sample), 2000, replace=False)
                 src_sample = src_sample[indices]
            
            transformed_src = np.dot(src_sample, self._coral_matrix)
            
            tgt_sample = target_dataset.matrix
            if len(tgt_sample) > 2000:
                 indices = np.random.choice(len(tgt_sample), 2000, replace=False)
                 tgt_sample = tgt_sample[indices]

            cov_src_new = np.cov(transformed_src, rowvar=False)
            cov_tgt = np.cov(tgt_sample, rowvar=False)
            
            diff_norm = np.linalg.norm(cov_src_new - cov_tgt, ord='fro')
            
            self._metrics = {
                "step_type": self.transformation_type.value,
                "estimator_used": self.config.estimator.value,
                "frobenius_distance": float(diff_norm),
                "regularization_initial": self.config.regularization,
                "regularization_used": self._final_regularization,
                "use_grid_search": self.config.use_grid_search
            }
            print(f"[{self.name}] Métricas: Distância Cov={diff_norm:.4f}, Reg Usada={self._final_regularization:.1e}")

        except Exception as e:
            print(f"[{self.name}] ⚠️ Erro ao calcular métricas: {e}")
            self._metrics = {"error": str(e)}

        self._fitted = True

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Transformador não treinado.")

        return self.math_service.apply_coral_transformation(
            dataset=dataset,
            coral_matrix=self._coral_matrix
        )

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if not self._fitted:
            raise RuntimeError(f"[{self.name}] Transformador não treinado.")
        
        temp_dataset = EmbeddingDataset(name="temp", items=[embedding_info])
        transformed_dataset = self.transform(temp_dataset)
        return transformed_dataset.items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save(self, path: str) -> None:
        if not self._fitted:
            return

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'coral_matrix': self._coral_matrix,
            'final_regularization': self._final_regularization,
            'config_dict': self.config.to_dict(), # Salva o DTO de forma segura
            'metrics': self._metrics
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"[{self.name}] Arquivo não encontrado: {file_path}")

        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            self._coral_matrix = state['coral_matrix']
            self._final_regularization = state.get('final_regularization', None)
            self._metrics = state.get('metrics', {}) 
            
            # Reconstrói o DTO a partir do dicionário ou legado
            config_dict = state.get('config_dict')
            if config_dict:
                self.config = CoralConfig.from_dict(config_dict)
            else:
                # Suporte legado para carregar pickles gerados antes desta refatoração
                self.config = CoralConfig(
                    regularization=state.get('regularization_config', 1e-6),
                    use_grid_search=state.get('use_grid_search', True),
                    estimator=CoralEstimatorType.from_str(state.get('estimator_value', 'STANDARD'))
                )
            
            self._fitted = True

    def get_state(self) -> Dict[str, Any]:
        return {
            "_coral_matrix": self._coral_matrix,
            "_final_regularization": self._final_regularization,
            "config_dict": self.config.to_dict(),
            "_metrics": self._metrics,
            "_fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._coral_matrix = state.get("_coral_matrix")
        self._final_regularization = state.get("_final_regularization", None)
        
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = CoralConfig.from_dict(config_dict)
        else:
             # Suporte legado no set_state (sync com o Target no PipelineController)
             self.config = CoralConfig(
                 regularization=state.get("regularization_config", 1e-6),
                 use_grid_search=state.get("use_grid_search", True),
                 estimator=CoralEstimatorType.from_str(state.get("estimator_value", "STANDARD"))
             )

        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))