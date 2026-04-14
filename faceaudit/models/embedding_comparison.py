from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd

# Import relativo para evitar dependência circular
from .embedding_info import EmbeddingInfo

@dataclass
class EmbeddingComparison:
    """
    Representa o resultado de uma comparação 1:1 entre dois embeddings,
    contendo o score de cosseno e o ground truth (same_person).
    """
    embedding_1: EmbeddingInfo
    embedding_2: EmbeddingInfo
    
    # Métrica de Comparação principal
    cosine_similarity: float

    # Ground Truth
    same_person: Optional[bool] = None

    def __post_init__(self):
        """
        Define automaticamente same_person baseando-se na identidade dos embeddings.
        """
        if self.same_person is None:
            id1 = self.embedding_1.identity
            id2 = self.embedding_2.identity
            
            # Mesma pessoa apenas se IDs forem iguais e não nulos
            if id1 is not None and id2 is not None:
                self.same_person = (id1 == id2)
            else:
                self.same_person = False

    def to_dict(self) -> Dict[str, Any]:
        """Retorna dicionário achatado para DataFrames."""
        return {
            "file_name_1": self.embedding_1.file_name,
            "image_path_1": self.embedding_1.image_path, # <--- NOVO
            "face_hash_1": self.embedding_1.face_hash,
            "identity_1": self.embedding_1.identity,
            
            "file_name_2": self.embedding_2.file_name,
            "image_path_2": self.embedding_2.image_path, # <--- NOVO
            "face_hash_2": self.embedding_2.face_hash,
            "identity_2": self.embedding_2.identity,
            
            "same_person": self.same_person,
            "cosine_similarity": self.cosine_similarity
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.to_dict()])