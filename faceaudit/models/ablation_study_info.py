from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.ablation_scenario_info import AblationScenarioInfo

@dataclass
class AblationStudyInfo(JSONSerializableMixin):
    """Estudo completo de ablação persistível."""
    k_neighbors: int
    scenarios: list[AblationScenarioInfo]

    @classmethod
    def from_dict(cls: type[AblationStudyInfo], data: dict) -> AblationStudyInfo:
        """Sobrescrita para instanciar recursivamente os cenários de ablação."""
        scenarios_data = data.pop("scenarios", [])
        scenarios = [AblationScenarioInfo.from_dict(s) for s in scenarios_data]
        return cls(scenarios=scenarios, **data)