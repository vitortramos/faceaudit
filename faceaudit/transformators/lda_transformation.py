# faceaudit/transformators/lda_transformation.py
import pickle
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType

# --- INJEÇÃO DO DTO ---
from faceaudit.models.lda_config import LDAConfig

class LdaTransformation(EmbeddingTransformationProtocol):
    """
    Transformador SUPERVISIONADO: Linear Discriminant Analysis.
    Aprende uma projeção que maximiza a distância entre identidades diferentes.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 name: str = "LdaTransformation", 
                 config: Optional[LDAConfig] = None):
        
        self._name = name
        self.math_service = math_service
        
        # Se nenhum config for passado, instanciamos com os defaults do DTO
        self.config = config if config is not None else LDAConfig()
        
        self._mean_vector: Optional[np.ndarray] = None
        self._lda_matrix: Optional[np.ndarray] = None
        self._fitted: bool = False
        
        # Métricas de execução (para auditoria)
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str: 
        return self._name

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.LDA_TRANSFORMATION
    
    @property
    def requires_target_dataset(self) -> bool: 
        return False

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        Treina o LDA para encontrar o espaço de projeção que melhor separa as identidades.
        """
        print(f"[{self.name}] Treinando LDA...")
        
        # Validação de identidades (necessária para variância entre classes)
        ids = set(item.identity for item in source_dataset.items)
        n_classes = len(ids)
        
        if n_classes < 2:
            raise ValueError(f"[{self.name}] Erro: LDA requer dataset com pelo menos 2 identidades. Encontrado: {n_classes}")
            
        # Executa o cálculo matemático via serviço usando os parâmetros do DTO
        self._mean_vector, self._lda_matrix = self.math_service.calculate_lda_parameters(
            dataset=source_dataset,
            n_components=self.config.n_components,
            regularization=self.config.regularization
        )
        
        # --- CÁLCULO DE MÉTRICAS (AUDITORIA) ---
        try:
            input_dim = source_dataset.matrix.shape[1]
            output_dim = self._lda_matrix.shape[1]
            
            # SANEAMENTO: Extração segura do valor da string do Enum para o relatório
            t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)

            self._metrics = {
                "step_type": t_type_val,
                "n_classes_found": int(n_classes),
                "input_dimension": int(input_dim),
                "output_dimension": int(output_dim), 
                "regularization": self.config.regularization
            }
            
            print(f"[{self.name}] Métricas: {n_classes} classes, Redução {input_dim}->{output_dim}")
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Erro ao calcular métricas: {e}")
            self._metrics = {"error": str(e)}

        self._fitted = True

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado. Chame 'fit()' primeiro.")
        return self.math_service.apply_lda_transformation(
            dataset, self._mean_vector, self._lda_matrix
        )

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado.")
        temp = EmbeddingDataset("temp", [embedding_info])
        return self.transform(temp).items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Retorna as métricas calculadas durante o Fit."""
        if not self._metrics:
            t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)
            return {"step_type": t_type_val, "status": "not_fitted"}
        return self._metrics

    def save(self, path: str) -> None:
        if not self._fitted: return
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'mean': self._mean_vector,
            'matrix': self._lda_matrix,
            'config_dict': self.config.to_dict(), # Salva o DTO encapsulado
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
            self._mean_vector = state['mean']
            self._lda_matrix = state['matrix']
            self._metrics = state.get('metrics', {})
            
            # Reconstrói o DTO a partir do dicionário salvo ou carrega legados
            config_dict = state.get('config_dict')
            if config_dict:
                self.config = LDAConfig.from_dict(config_dict)
            else:
                self.config = LDAConfig(
                    n_components=state.get('n_components'),
                    regularization=state.get('regularization', 1e-4)
                )
                
            self._fitted = True
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "_mean_vector": self._mean_vector,
            "_lda_matrix": self._lda_matrix,
            "config_dict": self.config.to_dict(),
            "_metrics": self._metrics,
            "_fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._mean_vector = state.get("_mean_vector")
        self._lda_matrix = state.get("_lda_matrix")
        
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = LDAConfig.from_dict(config_dict)
        else:
             self.config = LDAConfig(
                 n_components=state.get("n_components"),
                 regularization=state.get("regularization", 1e-4)
             )
             
        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))