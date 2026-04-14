from dataclasses import dataclass
from typing import List

# Import do novo modelo
from .transformation_step_3d import TransformationStep3D

@dataclass
class TransformationTrajectoryData:
    """
    DTO Agregador para a animação de trajetória.
    
    Substitui a estrutura antiga baseada em dicionários por uma lista
    sequencial e tipada de passos 3D.
    """
    # Lista ordenada contendo os pares sincronizados (Source/Target) para cada momento
    steps: List[TransformationStep3D]