from __future__ import annotations
from dataclasses import dataclass
from typing import List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class HubnessOptimizationSample(JSONSerializableMixin):
    """Dados de uma identidade em três estágios para testar a zona de conforto."""
    identity: str
    
    # 1. Estrito (< 8°) - Referência de Laboratório
    s_nk: int
    s_norm: float
    
    # 2. Leve (15-30°) - Hipótese de Zona de Conforto (In the Wild)
    l_nk: int
    l_norm: float
    l_yaw: float
    
    # 3. Perfil (> 45°) - Degradação Geométrica
    p_nk: int
    p_norm: float
    
    # Deltas em relação ao Estrito (Referência)
    delta_nk_optimization: int     # (Leve - Estrito) -> Negativo indica ganho
    delta_nk_degradation: int      # (Perfil - Estrito) -> Positivo indica perda
    delta_norm_optimization: float  # (Leve - Estrito) -> Positivo indica mais confiança

@dataclass
class HubnessOptimizationStudyInfo(JSONSerializableMixin):
    """Estudo focado no viés de treinamento em faixas intermediárias."""
    label: str
    samples: List[HubnessOptimizationSample]

    @classmethod
    def from_dict(cls: type[HubnessOptimizationStudyInfo], data: dict) -> HubnessOptimizationStudyInfo:
        samples_data = data.pop("samples", [])
        samples = [HubnessOptimizationSample.from_dict(s) for s in samples_data]
        return cls(samples=samples, **data)