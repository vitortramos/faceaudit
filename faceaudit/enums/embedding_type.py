from __future__ import annotations
from enum import Enum
from typing import Callable, Any, Optional
import numpy as np

class DistanceMetric(Enum):
    """
    Define as métricas de distância suportadas para comparação de vetores.
    """
    COSINE = "Similaridade de Cosseno"
    EUCLIDEAN = "Distância Euclidiana"

    def to_str(self) -> str:
        return self.name

    @classmethod
    def from_str(cls, value: str) -> Optional[DistanceMetric]:
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.COSINE

class EmbeddingType(Enum):
    """
    Enumerador Inteligente (Smart Enum) que define as fontes de embeddings suportadas.
    Encapsula metadados descritivos, parâmetros matemáticos e lógica de extração.
    """

    # --- Definições dos Membros ---
    # Formato: (display_name, architecture, dimensions, distance_metric, default_threshold, extractor)

    SEPAEL_V2 = (
        "Sepael V2 (Estável)",       
        "AdaFace (ResNet100)",       
        512,                         
        DistanceMetric.COSINE,       
        0.40,                        
        lambda face: face.ff_info.embedding_sepaelv2 if face.ff_info else None
    )

    SEPAEL_V4 = (
        "Sepael V4 (Experimental)",  
        "MagFace (ResNet100)",       
        512,                         
        DistanceMetric.COSINE,       
        0.50,
        lambda face: face.ff_info.embedding_sepaelv4 if face.ff_info else None
    )

    def __init__(self, 
                 display_name: str, 
                 architecture: str, 
                 dimensions: int, 
                 distance_metric: DistanceMetric,
                 default_threshold: float,
                 extractor: Callable[[Any], Any]):
        self.display_name = display_name
        self.architecture = architecture
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.default_threshold = default_threshold
        self._extractor = extractor

    def extract_from(self, face_info: Any) -> Optional[np.ndarray]:
        """Executa a extração segura do vetor biométrico a partir de um objeto FaceInfo."""
        try:
            val = self._extractor(face_info)
            if val is not None and not isinstance(val, np.ndarray):
                return np.array(val, dtype=np.float32)
            return val
        except Exception:
            return None

    def is_available_in(self, face_info: Any) -> bool:
        """Verifica se um registro possui o embedding deste tipo calculado."""
        return self.extract_from(face_info) is not None

    def __str__(self):
        return self.display_name

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna a chave do Enum (ex: 'SEPAEL_V2') para persistência."""
        return self.name

    @classmethod
    def from_str(cls, value: str) -> EmbeddingType:
        """Reconstrói o Enum a partir da chave string com fallback seguro."""
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.SEPAEL_V2