from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationMetrics:
    """
    Armazena os indicadores absolutos de desempenho de um modelo biométrico forense.
    Separado dos dados de plotagem para facilitar a exportação de laudos (PDF/CSV) e a exibição em tabelas.
    """
    eer: float                 # Equal Error Rate (Poder de Discriminação bruto)
    eer_threshold: float       # O valor exato de LR/Score onde o EER ocorre
    
    cllr: float                # Cost of Log-Likelihood Ratio (Desempenho Forense Real)
    cllr_min: float            # Cllr Mínimo possível (Desempenho ótimo se perfeitamente calibrado)
    
    auc: Optional[float] = None # Area Under the Curve (ROC) - Opcional, comum em biometria comercial

    @property
    def cal_loss(self) -> float:
        """
        Perda de Calibração (Calibration Loss).
        Diferença entre o desempenho real e o desempenho ótimo.
        Se for muito alto, significa que o modelo discrimina bem, mas 'mente' na certeza do LR.
        """
        # Garantia contra flutuações de ponto flutuante
        return max(0.0, self.cllr - self.cllr_min)

    @property
    def is_well_calibrated(self) -> bool:
        """Heurística forense: Uma perda de calibração menor que 0.1 indica calibração aceitável."""
        return self.cal_loss < 0.1