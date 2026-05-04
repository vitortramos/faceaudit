from __future__ import annotations
from enum import Enum
from typing import Callable, Any, Optional, Tuple

import numpy as np


def _embedding_norm_from_face(f: Any) -> Optional[float]:
    """
    Extrai a norma L2 do embedding primário (sepaelv2) de um FaceInfo.
    Retorna None quando o embedding não está disponível.
    """
    try:
        emb = f.ff_info.embedding_sepaelv2 if f.ff_info else None
        if emb is None:
            return None
        return float(np.linalg.norm(emb))
    except Exception:
        return None


class QualityScoreType(Enum):
    """
    Define as métricas de qualidade suportadas e suas estratégias de extração.
    Atua como um Smart Enum que sabe como recuperar valores de objetos FaceInfo.
    """

    FIQA = (
        "ForensicFace (FIQA)",
        (0.0, 1.0),
        lambda f: f.ff_info.fiqa_score if f.ff_info else None
    )

    OFIQ = (
        "ISO/OFIQ (Unificado)",
        (0.0, 100.0),
        lambda f: f.ofiq_info.unified_quality_score_scalar if f.ofiq_info else None
    )

    SERFIQ = (
        "SER-FIQ (Estocástico)",
        (0.0, 100.0),
        lambda f: f.serfiq_info.score if f.serfiq_info else None
    )

    EMBEDDING_NORM = (
        "Norma do Embedding (AdaFace)",
        (0.0, float("inf")),
        _embedding_norm_from_face
    )

    def __init__(self, display_name: str, value_range: Tuple[float, float], extractor: Callable[[Any], Any]):
        """
        Inicializa o membro com metadados ricos.
        Note: O valor do Enum (self.value) será a tupla inteira.
        """
        self.display_name = display_name
        self.value_range = value_range
        self._extractor = extractor

    def __str__(self) -> str:
        return self.display_name

    def extract_from(self, face_info: Any) -> Optional[float]:
        """Extrai o valor numérico do objeto FaceInfo de forma segura."""
        try:
            val = self._extractor(face_info)
            if val is not None and val != "":
                return float(val)
            return None
        except Exception:
            return None

    @property
    def key_name(self) -> str:
        """Retorna a chave usada em dicionários (ex: 'fiqa')."""
        return self.name.lower()

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o nome do membro para persistência (ex: 'FIQA')."""
        return self.name

    @classmethod
    def from_str(cls, value: str) -> QualityScoreType:
        """Reconstrói o Enum a partir do nome da constante com fallback para FIQA."""
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.FIQA