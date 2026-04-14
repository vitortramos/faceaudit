from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class HubnessPairedSample(JSONSerializableMixin):
    """Dados pareados de uma única identidade em duas poses distintas."""
    identity: str
    
    # Dados da foto Frontal (Referência)
    f_nk: int
    f_norm: float
    f_max_impostor: float
    
    # Dados da foto Não-Frontal (Desvio)
    nf_nk: int
    nf_norm: float
    nf_max_impostor: float
    nf_yaw: float
    
    # Deltas calculados (NF - F)
    delta_nk: int
    delta_norm: float
    delta_max_impostor: float

@dataclass
class HubnessPairedStudyInfo(JSONSerializableMixin):
    """
    Resultado do estudo pareado por identidade.
    Focado em isolar o efeito da pose removendo a variabilidade entre indivíduos.
    """
    comparison_label: str  # Ex: "Frontal vs Moderada"
    paired_samples: list[HubnessPairedSample]
    
    @property
    def sample_count(self) -> int:
        return len(self.paired_samples)

    @classmethod
    def from_dict(cls: type[HubnessPairedStudyInfo], data: dict) -> HubnessPairedStudyInfo:
        """Sobrescrita para instanciar recursivamente a lista de HubnessPairedSample."""
        samples_data = data.pop("paired_samples", [])
        paired_samples = [HubnessPairedSample.from_dict(s) for s in samples_data]
        return cls(paired_samples=paired_samples, **data)