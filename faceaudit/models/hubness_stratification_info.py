from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class HubnessStratificationInfo(JSONSerializableMixin):
    """
    Representa métricas agregadas de hubness e risco divididas por faixas de pose.
    Utilizado para gráficos de comparação entre grupos (Boxplots/Violin).
    """
    label: str               # Ex: "Frontal (|Yaw| < 15°)"
    min_yaw: float
    max_yaw: float
    sample_count: int
    
    # Médias do grupo
    avg_nk: float
    avg_norm: float
    avg_mean_impostor_score: float
    avg_max_impostor_score: float
    
    # Valores brutos para permitir a plotagem de distribuições (KDE/Boxplot)
    nk_scores: list[int]
    norms: list[float]
    max_impostor_scores: list[float]