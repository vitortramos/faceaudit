from dataclasses import dataclass, field
from typing import Dict, List
from faceaudit.models.plda_validation_result import ValidationSeverity
from faceaudit.models.strategy_audit_result import StrategyAuditResult

@dataclass
class ArenaValidationResult:
    """
    O Selo de Auditoria Final da Arena.
    Contém a sanidade geral do teste e os resultados individuais dos competidores.
    """
    severity: ValidationSeverity = ValidationSeverity.NEUTRAL
    is_pure: bool = True 
    
    # Mensagens gerais sobre o Gabarito (Probe vs Ref)
    arena_messages: List[str] = field(default_factory=list)
    
    # Dicionário com o laudo individual de cada competidor (Key = Strategy ID)
    strategy_audits: Dict[str, StrategyAuditResult] = field(default_factory=dict)