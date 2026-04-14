# faceaudit/models/hubness_analysis_info.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.hubness_sample_info import HubnessSampleInfo
from faceaudit.models.hubness_stratification_info import HubnessStratificationInfo
from faceaudit.models.hubness_paired_study_info import HubnessPairedStudyInfo

@dataclass
class HubnessAnalysisInfo(JSONSerializableMixin):
    """Resultado agregado da análise de hubness com capacidades de persistência."""
    k_neighbors: int
    sample_count: int
    valid_identity_count: int
    samples: list[HubnessSampleInfo]

    max_nk: Optional[int] = None
    mean_nk: Optional[float] = None
    std_nk: Optional[float] = None
    
    # --- Estudos Adicionais (Opcionais) ---
    stratification: List[HubnessStratificationInfo] = field(default_factory=list)
    paired_study: Optional[HubnessPairedStudyInfo] = None

    # --- Estatísticas Globais ---
    pearson_hubness_vs_mean_impostor: Optional[float] = None
    pearson_hubness_vs_mean_impostor_p_value: Optional[float] = None
    spearman_hubness_vs_mean_impostor: Optional[float] = None
    spearman_hubness_vs_mean_impostor_p_value: Optional[float] = None
    
    pearson_norm_vs_hubness: Optional[float] = None
    pearson_norm_vs_hubness_p_value: Optional[float] = None
    spearman_norm_vs_hubness: Optional[float] = None
    spearman_norm_vs_hubness_p_value: Optional[float] = None

    @classmethod
    def from_dict(cls: type[HubnessAnalysisInfo], data: dict) -> HubnessAnalysisInfo:
        """
        Sobrescrita necessária para instanciar recursivamente a hierarquia de DTOs.
        """
        # 1. Converte Amostras Individuais
        samples_data = data.pop("samples", [])
        samples = [HubnessSampleInfo.from_dict(s) for s in samples_data]
        
        # 2. Converte Estratificação (Lista de objetos)
        strat_data = data.pop("stratification", [])
        stratification = [HubnessStratificationInfo.from_dict(s) for s in strat_data]
        
        # 3. Converte Estudo Pareado (Objeto Único)
        paired_data = data.pop("paired_study", None)
        paired_study = HubnessPairedStudyInfo.from_dict(paired_data) if paired_data else None
        
        # Instancia a classe principal com as listas tipadas
        return cls(
            samples=samples, 
            stratification=stratification, 
            paired_study=paired_study, 
            **data
        )