# faceaudit/transformators/whitening_transformation.py
import pickle
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType
from faceaudit.enums.whitening_style import WhiteningStyle

# --- INJEÇÃO DO DTO ---
from faceaudit.models.whitening_config import WhiteningConfig

class WhiteningTransformation(EmbeddingTransformationProtocol):
    """
    Transformador para aplicar Whitening (Branqueamento).
    Suporta PCA (Principal Component Analysis) e ZCA.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 name: str = "WhiteningTransformation", 
                 config: Optional[WhiteningConfig] = None):
        
        self._name = name
        self.math_service = math_service
        
        # Se nenhum config for passado, instanciamos com os defaults do DTO
        self.config = config if config is not None else WhiteningConfig()
        
        self._whitening_mean: Optional[np.ndarray] = None
        self._whitening_matrix: Optional[np.ndarray] = None
        self._fitted: bool = False
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str: 
        return self._name

    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.WHITENING_TRANSFORMATION
    
    @property
    def requires_target_dataset(self) -> bool: 
        return False

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        print(f"[{self.name}] Treinando Whitening ({self.config.style.value})...")
        
        self._whitening_mean, self._whitening_matrix = self.math_service.calculate_whitening_parameters(
            dataset=source_dataset,
            n_components=self.config.n_components,
            style=self.config.style
        )
        
        try:
            X = source_dataset.matrix
            n_features = X.shape[1]
            
            cov = np.cov(X, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov) 
            eigvals = eigvals[::-1]
            eigvals[eigvals < 0] = 0 
            
            total_variance = np.sum(eigvals)
            n_kept = self.config.n_components if self.config.n_components and self.config.n_components < n_features else n_features
            
            kept_variance = np.sum(eigvals[:n_kept])
            ratio = kept_variance / total_variance if total_variance > 0 else 0.0
            
            # SANEAMENTO: Extração segura do tipo para as métricas
            t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)

            self._metrics = {
                "step_type": t_type_val,
                "original_dimension": int(n_features),
                "output_dimension": int(n_kept),
                "explained_variance_ratio": float(ratio),
                "explained_variance_percent": f"{ratio:.2%}",
                "style": self.config.style.value
            }
            print(f"[{self.name}] Métricas: {self._metrics['explained_variance_percent']} variância preservada.")
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Erro ao calcular métricas: {e}")
            self._metrics = {"error": str(e)}

        self._fitted = True

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado. Chame 'fit()' primeiro.")
        return self.math_service.apply_whitening_transformation(
            dataset, self._whitening_mean, self._whitening_matrix
        )

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        if not self._fitted: 
            raise RuntimeError(f"[{self.name}] Não treinado.")
        temp = EmbeddingDataset("temp", [embedding_info])
        return self.transform(temp).items[0]

    def get_execution_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def save(self, path: str) -> None:
        if not self._fitted: return
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            'mean': self._whitening_mean,
            'matrix': self._whitening_matrix,
            'config_dict': self.config.to_dict(), # Salva o DTO de forma segura
            'metrics': self._metrics
        }
        with open(file_path, 'wb') as f: 
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"[{self.name}] Arquivo de estado não encontrado: {file_path}")

        with open(file_path, 'rb') as f:
            state = pickle.load(f)
            self._whitening_mean = state['mean']
            self._whitening_matrix = state['matrix']
            self._metrics = state.get('metrics', {})
            
            # Reconstrói o DTO a partir do dicionário ou legado
            config_dict = state.get('config_dict')
            if config_dict:
                self.config = WhiteningConfig.from_dict(config_dict)
            else:
                # Suporte legado
                style_val = state.get('style', 'PCA')
                self.config = WhiteningConfig(
                    n_components=state.get('n_components'),
                    style=WhiteningStyle.from_str(style_val) if hasattr(WhiteningStyle, 'from_str') else WhiteningStyle(style_val)
                )
                
            self._fitted = True
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "_whitening_mean": self._whitening_mean,
            "_whitening_matrix": self._whitening_matrix,
            "config_dict": self.config.to_dict(),
            "_metrics": self._metrics,
            "_fitted": self._fitted,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._whitening_mean = state.get("_whitening_mean")
        self._whitening_matrix = state.get("_whitening_matrix")
        
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = WhiteningConfig.from_dict(config_dict)
        else:
             style_val = state.get("style", "PCA")
             self.config = WhiteningConfig(
                 n_components=state.get("n_components"),
                 style=WhiteningStyle.from_str(style_val) if hasattr(WhiteningStyle, 'from_str') else WhiteningStyle(style_val)
             )
             
        self._metrics = state.get("_metrics", {})
        self._fitted = bool(state.get("_fitted", False))