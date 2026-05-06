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

    @property
    def parquet_column(self) -> str:
        """Nome canônico da coluna no parquet (formato pós-PR 3).

        Antes do PR 3, SEPAEL_V2 ficava em uma coluna `embedding` (alias) que
        duplicava `embedding_v2`. Após a migração, só `embedding_v2` existe.
        """
        _map = {
            "SEPAEL_V2": "embedding_v2",
            "SEPAEL_V4": "embedding_v4",
        }
        return _map[self.name]

    @property
    def model_name(self) -> str:
        """Nome do modelo no `ColumnSpec.model_name` (pós-PR 3)."""
        return {
            "SEPAEL_V2": "sepaelv2",
            "SEPAEL_V4": "sepaelv4",
        }[self.name]

    def column_name_in(self, column_specs: dict) -> Optional[str]:
        """Procura no dict de ColumnSpecs (do parquet) qual coluna tem
        role=embedding e model_name correspondente a este tipo.

        Útil quando o caller já tem os specs em mãos (ex: FaceDataset com
        column_specs populada por from_parquet) e quer saber qual coluna
        usar sem hardcodar nomes.
        """
        target = self.model_name
        for name, spec in (column_specs or {}).items():
            role = getattr(spec, "role", None)
            mname = getattr(spec, "model_name", None)
            role_value = getattr(role, "value", role)
            if role_value == "embedding" and mname == target:
                return name
        return None

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