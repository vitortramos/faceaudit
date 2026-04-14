from __future__ import annotations
from dataclasses import dataclass
from typing import List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class HubnessQualityRange(JSONSerializableMixin):
    """Métricas de saúde (Qualidade e Norma) para uma faixa de pose."""
    label: str
    sample_count: int
    
    # Métricas de Qualidade (FIQA/Score do Modelo)
    avg_quality: float
    median_quality: float
    std_quality: float
    quality_values: List[float]
    
    # Métricas de Magnitude (Norma L2)
    avg_norm: float
    median_norm: float
    std_norm: float
    norm_values: List[float]

@dataclass
class HubnessQualityDiagnosticsInfo(JSONSerializableMixin):
    """Container para o estudo de covariáveis (Qualidade vs Pose)."""
    ranges: List[HubnessQualityRange]

    @classmethod
    def from_dict(cls: type[HubnessQualityDiagnosticsInfo], data: dict) -> HubnessQualityDiagnosticsInfo:
        ranges_data = data.pop("ranges", [])
        ranges = [HubnessQualityRange.from_dict(r) for r in ranges_data]
        return cls(ranges=ranges, **data)