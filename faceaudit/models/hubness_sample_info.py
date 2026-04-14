# faceaudit/models/hubness_sample_info.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

@dataclass
class HubnessSampleInfo(JSONSerializableMixin):
    """Representação persistente dos dados de hubness para uma única face."""
    index: int
    identity: Optional[str]
    norm: float
    quality_score: float
    serfiq: float
    yaw: float
    pitch: float
    abs_pose: float
    nk_score: int
    mean_impostor_score: float
    max_impostor_score: float
    mean_delta_pose_to_neighbors: float
    t_x: Optional[float] = None