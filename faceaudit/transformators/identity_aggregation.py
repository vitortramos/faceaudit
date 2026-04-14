# faceaudit/transformators/identity_aggregation.py
import pickle
from typing import Optional, Dict, Any
from pathlib import Path

from faceaudit.protocols.embedding_transformation_protocol import EmbeddingTransformationProtocol
from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.services.embedding_math_service import EmbeddingMathService
from faceaudit.enums.transformation_type import TransformationType

# --- INJEÇÃO DO DTO ---
from faceaudit.models.aggregation_config import AggregationConfig

class IdentityAggregation(EmbeddingTransformationProtocol):
    """
    Transformador que agrega múltiplos embeddings de uma mesma identidade.
    Agora suporta Smart Enrollment (Combinações, Ponderação por Qualidade, etc)
    através do AggregationConfig.
    """

    def __init__(self, 
                 math_service: EmbeddingMathService, 
                 name: str = "IdentityAggregation", 
                 config: Optional[AggregationConfig] = None):
        
        self._name = name
        self.math_service = math_service
        
        # Se nenhum config for passado, instanciamos com os defaults (Centroid simples)
        self.config = config if config is not None else AggregationConfig()
        
        # Mantemos métricas para auditoria, mesmo sendo stateless
        self._metrics: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def transformation_type(self) -> TransformationType:
        return TransformationType.IDENTITY_AGGREGATION

    @property
    def requires_target_dataset(self) -> bool:
        return False

    def fit(self, source_dataset: EmbeddingDataset, target_dataset: Optional[EmbeddingDataset] = None) -> None:
        """
        Stateless: Não requer treinamento de parâmetros matemáticos.
        Apenas registra metadados baseados no DTO de configuração para a Cadeia de Custódia.
        """
        t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)

        self._metrics = {
            "step_type": t_type_val,
            "operation": "Smart Aggregation",
            "method": self.config.method.value if hasattr(self.config.method, 'value') else str(self.config.method),
            "quality_weighted": self.config.use_quality_weighting,
            "stateless": True
        }

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """
        Delega a agregação inteligente para o serviço matemático, 
        agora passando o DTO com as regras completas.
        """
        return self.math_service.apply_smart_aggregation(dataset, config=self.config)

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Identidade: A agregação de 1 item é ele mesmo.
        """
        return embedding_info

    # --- IMPLEMENTAÇÃO DO PROTOCOLO ---
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Retorna metadados para auditoria."""
        if not self._metrics:
            t_type_val = self.transformation_type.value if hasattr(self.transformation_type, 'value') else str(self.transformation_type)
            return {"step_type": t_type_val, "status": "stateless"}
        return self._metrics
    # ----------------------------------

    def save(self, path: str) -> None:
        """Salva as métricas e o DTO de Configuração para manter consistência do pipeline."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'metrics': self._metrics,
            'config_dict': self.config.to_dict() # Serialização segura do DTO
        }
        with open(file_path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        """Carrega o estado e reconstrói o DTO de forma resiliente."""
        file_path = Path(path)
        if not file_path.exists():
            return

        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
                self._metrics = state.get('metrics', {})
                
                # Reconstrói o DTO a partir do dicionário salvo
                config_dict = state.get('config_dict')
                if config_dict:
                    self.config = AggregationConfig.from_dict(config_dict)
                    
        except Exception as e:
            print(f"[{self.name}] Aviso ao carregar estado: {e}")

    def get_state(self) -> Dict[str, Any]:
        return {
            "_metrics": self._metrics,
            "config_dict": self.config.to_dict()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self._metrics = state.get("_metrics", {})
        
        config_dict = state.get("config_dict")
        if config_dict:
            self.config = AggregationConfig.from_dict(config_dict)
        else:
            self.config = AggregationConfig()