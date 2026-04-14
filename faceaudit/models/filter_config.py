from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Type, TypeVar
from pathlib import Path
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

T = TypeVar("T", bound="FilterConfig")

@dataclass
class FilterConfig(JSONSerializableMixin):
    """
    Define critérios de filtragem para um FaceDataset.
    Centraliza ranges de qualidade, pose e seleções demográficas.
    """
    name: str = "Novo Filtro"
    description: Optional[str] = None

    # --- FILTROS DE QUALIDADE (Scores) ---
    min_fiqa_score: Optional[float] = None
    max_fiqa_score: Optional[float] = None
    fiqa_score: Optional[float] = None
    abs_fiqa_score: bool = False

    min_ofiq_score: Optional[float] = None
    max_ofiq_score: Optional[float] = None
    ofiq_score: Optional[float] = None
    abs_ofiq_score: bool = False
    
    min_serfiq_score: Optional[float] = None
    max_serfiq_score: Optional[float] = None
    abs_serfiq_score: bool = False

    # --- FILTROS DE POSE (OFIQ) ---
    min_ofiq_yaw: Optional[float] = None
    max_ofiq_yaw: Optional[float] = None
    ofiq_yaw: Optional[float] = None
    abs_ofiq_yaw: bool = False

    min_ofiq_pitch: Optional[float] = None
    max_ofiq_pitch: Optional[float] = None
    ofiq_pitch: Optional[float] = None
    abs_ofiq_pitch: bool = False

    min_ofiq_roll: Optional[float] = None
    max_ofiq_roll: Optional[float] = None
    ofiq_roll: Optional[float] = None
    abs_ofiq_roll: bool = False

    # --- FILTROS DE POSE (ForensicFace - FF) ---
    min_ff_yaw: Optional[float] = None
    max_ff_yaw: Optional[float] = None
    ff_yaw: Optional[float] = None
    abs_ff_yaw: bool = False

    min_ff_pitch: Optional[float] = None
    max_ff_pitch: Optional[float] = None
    ff_pitch: Optional[float] = None
    abs_ff_pitch: bool = False

    min_ff_roll: Optional[float] = None
    max_ff_roll: Optional[float] = None
    ff_roll: Optional[float] = None
    abs_ff_roll: bool = False

    # --- FILTROS DEMOGRÁFICOS E METADADOS ---
    gender: Optional[str] = None
    gender_list: List[str] = field(default_factory=list)

    ethnicity: Optional[str] = None
    ethnicity_list: List[str] = field(default_factory=list)

    identity: Optional[str] = None
    identity_list: List[str] = field(default_factory=list)

    source: Optional[str] = None
    source_list: List[str] = field(default_factory=list)

    # --- OUTROS ---
    min_age: Optional[float] = None
    max_age: Optional[float] = None
    abs_age: bool = False
    
    min_ipd: Optional[float] = None
    max_ipd: Optional[float] = None # Adicionado para simetria
    abs_ipd: bool = False

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói a configuração garantindo que campos nulos 
        do JSON sejam tratados corretamente.
        """
        if not data:
            return None
        
        # O Mixin base já faz o grosso do trabalho. 
        # Filtramos apenas para garantir compatibilidade com versões antigas do dicionário.
        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return cls(**filtered_data)