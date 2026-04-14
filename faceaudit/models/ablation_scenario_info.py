# faceaudit/models/ablation_scenario_info.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class AblationScenarioInfo(JSONSerializableMixin):
    """Dados serializáveis de um cenário do estudo de ablação."""
    scenario_name: str
    image_count_per_identity: Optional[int]
    max_nk: int
    nk_scores: list[int]