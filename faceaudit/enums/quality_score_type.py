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
        # label_pt alias — `display_name` já está em PT-BR. Mantido por
        # consistência de API com outros enums do projeto.
        self.label_pt = display_name
        self.value_range = value_range
        self._extractor = extractor

    def __str__(self) -> str:
        return self.label_pt

    # Nomes procurados via getattr pelo editor inline de DTO aninhado
    # (`NestedDTOInlineWidget`) para o texto do combo e a linha de ajuda.
    # Sem eles o combo exibiria "FIQA"/"OFIQ" crus, sem descrição.
    @property
    def label(self) -> str:
        return self.display_name

    @property
    def description(self) -> str:
        lo, hi = self.value_range
        return f"{self.display_name} — faixa esperada {lo:g} a {hi:g}."

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

    @property
    def metric_name(self) -> Optional[str]:
        """Nome da métrica em ColumnSpec.metric_name (pós-PR 3).

        Retorna None para EMBEDDING_NORM porque a norma é calculada
        on-the-fly e não corresponde a uma coluna persistida.
        """
        return {
            "FIQA": "fiqa",
            "OFIQ": "ofiq",
            "SERFIQ": "serfiq",
            "EMBEDDING_NORM": None,
        }.get(self.name)

    def column_name_in(self, column_specs: dict) -> Optional[str]:
        """Procura no dict de ColumnSpecs qual coluna tem role=quality e
        metric_name correspondente. Útil para localizar a coluna primária
        de quality sem hardcodar nomes (FIQA→`fiqa_score`, etc.)."""
        target = self.metric_name
        if target is None:
            return None
        for name, spec in (column_specs or {}).items():
            role = getattr(spec, "role", None)
            mname = getattr(spec, "metric_name", None)
            role_value = getattr(role, "value", role)
            if role_value == "quality" and mname == target:
                return name
        return None

    # --- Interface de Serialização ---

    def to_str(self) -> str:
        """Retorna o nome do membro para persistência (ex: 'FIQA')."""
        return self.name

    @classmethod
    def from_str(cls, value: str) -> QualityScoreType:
        """Reconstrói o Enum a partir do nome da constante com fallback para FIQA."""
        # Idempotência: sem este guard, passar um membro já hidratado caía no
        # `except AttributeError` e voltava FIQA SILENCIOSAMENTE — trocando a
        # métrica escolhida pelo usuário sem aviso nenhum. Alcançável via
        # `AggregationConfig.from_dict` sobre dict já processado, cujo guard é
        # só um teste de truthiness (membro de Enum é truthy).
        if isinstance(value, cls):
            return value
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.FIQA