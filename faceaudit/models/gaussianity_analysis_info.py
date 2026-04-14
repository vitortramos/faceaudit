from dataclasses import dataclass
import numpy as np
from typing import Type

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin

# Certifique-se de importar o seu Mixin aqui
# from faceaudit.utils.mixins import JSONSerializableMixin 

@dataclass
class GaussianityAnalysisInfo(JSONSerializableMixin):
    """
    DTO que armazena os resultados da análise de normalidade multivariada.
    Herda de JSONSerializableMixin para persistência de laudos em disco.
    """
    dataset_name: str
    dimensions: int
    sample_size: int
    
    # Distâncias de Mahalanobis ao quadrado reais da sua base (NumPy Array)
    mahalanobis_sq_distances: np.ndarray
    
    # Quantis teóricos esperados se a base fosse uma Gaussiana Perfeita (NumPy Array)
    theoretical_chi2_quantiles: np.ndarray
    
    # Uma métrica de "saúde": % de pontos que caem dentro da elipse teórica de 95%
    pct_inside_95_ellipse: float

    def to_dict(self) -> dict:
        """
        Sobrescreve o método do Mixin para lidar com a conversão de numpy.ndarray 
        para list, garantindo que o json.dumps não quebre.
        """
        # 1. Pega o dicionário base higienizado pelo Mixin
        data = super().to_dict()
        
        # 2. Converte os arrays do NumPy para listas nativas
        data['mahalanobis_sq_distances'] = self.mahalanobis_sq_distances.tolist()
        data['theoretical_chi2_quantiles'] = self.theoretical_chi2_quantiles.tolist()
        
        return data

    @classmethod
    def from_dict(cls: Type['GaussianityAnalysisInfo'], data: dict) -> 'GaussianityAnalysisInfo':
        """
        Reconstrói o objeto a partir de um dicionário, garantindo que as listas 
        voltem a ser instâncias de numpy.ndarray para cálculos matemáticos.
        """
        # Clona o dicionário para não mutar a entrada original acidentalmente
        processed_data = data.copy()
        
        # Reconstrói os arrays
        if 'mahalanobis_sq_distances' in processed_data:
            processed_data['mahalanobis_sq_distances'] = np.array(processed_data['mahalanobis_sq_distances'])
            
        if 'theoretical_chi2_quantiles' in processed_data:
            processed_data['theoretical_chi2_quantiles'] = np.array(processed_data['theoretical_chi2_quantiles'])
            
        return cls(**processed_data)