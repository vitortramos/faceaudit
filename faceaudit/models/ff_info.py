from dataclasses import dataclass, field, asdict
from typing import Optional, Union, Any, List
import numpy as np

@dataclass
class FFInfo:
    """
    Dataclass que mapeia o output da biblioteca ForensicFace.
    Armazena métricas de qualidade, pose, demografia e embeddings.
    """
    # Métricas Físicas e Qualidade
    ipd: Optional[float] = None           # Inter-pupillary distance
    fiqa_score: Optional[float] = None    # Face Image Quality Assessment
    det_score: Optional[float] = None     # Detection confidence score
    
    # Demografia Estimada
    gender: Optional[Union[str, float]] = None
    age: Optional[float] = None
    
    # Pose (Head Pose Estimation)
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    
    # Geometria (Arrays)
    # repr=False evita imprimir a matriz inteira no terminal ao debugar
    bbox: Optional[np.ndarray] = None
    keypoints: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Embeddings (Vetores de características)
    embedding_sepaelv2: Optional[np.ndarray] = field(default=None, repr=False)
    embedding_sepaelv4: Optional[np.ndarray] = field(default=None, repr=False)
    
    # Imagem Processada
    aligned_face: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self, serialize_numpy: bool = False, exclude_heavy: bool = True) -> dict:
        """
        Gera dicionário dos dados.
        exclude_heavy: Remove embeddings e arrays para exportação tabular leve.
        """
        data = asdict(self)
        
        # Remove campos pesados/técnicos se solicitado (para Pandas/Parquet)
        if exclude_heavy:
            keys_to_remove = [k for k in data.keys() if 'embedding' in k or k in ['keypoints', 'aligned_face', 'bbox']]
            for k in keys_to_remove:
                data.pop(k, None)
        
        if serialize_numpy:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    data[key] = value.tolist()
                    
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'FFInfo':
        """
        Cria instância a partir do dicionário cru do ForensicFace.
        Garante que apenas as chaves esperadas sejam passadas.
        """
        # Filtra o dicionário de entrada para pegar apenas chaves que existem na dataclass
        # Isso evita erro de "unexpected keyword argument" se a lib atualizar no futuro
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)