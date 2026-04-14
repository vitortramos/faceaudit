from dataclasses import dataclass, field
from typing import List
from faceaudit.models.plda_validation_result import ValidationSeverity

@dataclass
class StrategyAuditResult:
    """
    Sub-resultado detalhando os vazamentos (Data/Identity Leakage) 
    de uma única estratégia na Arena.
    """
    is_pure: bool = True
    severity: ValidationSeverity = ValidationSeverity.OK
    messages: List[str] = field(default_factory=list)
    leakage_count_ids: int = 0
    leakage_count_hashes: int = 0