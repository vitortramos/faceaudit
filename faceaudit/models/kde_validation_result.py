from dataclasses import dataclass, field
from typing import List, Dict, Any

from faceaudit.enums.validation_severity import ValidationSeverity


@dataclass
class KDEValidationResult:
    """
    Resultado da auditoria de configuração do KDE.
    """
    # Veredito Geral
    severity: ValidationSeverity
    messages: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Detalhamento por Seção (para a UI colorir abas específicas)
    data_severity: ValidationSeverity = ValidationSeverity.OK
    params_severity: ValidationSeverity = ValidationSeverity.OK