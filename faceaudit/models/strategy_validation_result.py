from dataclasses import dataclass
from faceaudit.models.validation_metrics import ValidationMetrics
from faceaudit.models.validation_plot_data import ValidationPlotData

@dataclass
class StrategyValidationResult:
    """
    Encapsula o resultado completo da validação de uma única estratégia 
    (seja ela KDE, PLDA, etc.) contra um dataset específico.
    """
    strategy_id: str
    strategy_name: str         # Nome legível (ex: "PLDA 128D Whitened")
    strategy_type: str         # "KDE" ou "PLDA" (Para colorir ou agrupar os gráficos)
    
    n_genuines: int            # Número de comparações de mesma pessoa processadas
    n_impostors: int           # Número de comparações de pessoas diferentes processadas
    
    metrics: ValidationMetrics
    plot_data: ValidationPlotData