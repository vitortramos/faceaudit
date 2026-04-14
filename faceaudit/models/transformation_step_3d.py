from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
from faceaudit.enums.step_type import StepType 

@dataclass
class TransformationStep3D:
    """
    Representa um passo projetado em 3D para visualização.
    """
    step_name: str
    source_points: np.ndarray
    target_points: Optional[np.ndarray]
    variance: float
    original_dimension: int
    
    # Adicionamos " = StepType.UNKNOWN" para evitar o erro de ordenação de argumentos
    step_type: StepType = StepType.UNKNOWN
    
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Guarda as matrizes de rotação (Autovetores) para os wireframes de forma independente.
    # Se None, o visualizador calcula a orientação dinamicamente. 
    # Se preenchido, o visualizador obedece a essa matriz (cura do Paradoxo da Esfera).
    source_orientation_matrix: Optional[np.ndarray] = None
    target_orientation_matrix: Optional[np.ndarray] = None