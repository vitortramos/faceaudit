# faceaudit/parameters/kde_parameters.py
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
# ModelParametersProtocol deve ser mantido para garantir a interface do MathService
from faceaudit.protocols.model_parameters import ModelParametersProtocol

# Importação dos Enums "Ricos"
from faceaudit.enums.score_metric import ScoreMetric
from faceaudit.enums.kde_kernel_type import KDEKernelType

T = TypeVar("T", bound="KDEParameters")

@dataclass
class KDEParameters(ModelParametersProtocol, JSONSerializableMixin):
    """
    Hiperparâmetros para a estratégia KDE (Kernel Density Estimation).
    Define o comportamento do Mining e do algoritmo matemático.
    """
    
    # --- 1. ESTRATÉGIA DE CURADORIA E MINERAÇÃO (Mining Strategy) ---
    # Aponta para o arquivo .json (ex: "VIANA") criado via Interface
    mining_config_name: str = ""
    
    # Travas de Segurança (Mínimos necessários para a matemática não gerar uma distribuição anômala)
    impostors_min_count: int = 100
    genuines_min_count: int = 100
    
    # --- 2. ALGORITMO KDE ---
    score_metric: ScoreMetric = ScoreMetric.COSINE
    kernel: KDEKernelType = KDEKernelType.GAUSSIAN
    
    # --- 3. OTIMIZAÇÃO (GridSearch & Downsampling) ---
    use_gridsearch: bool = True
    bandwidth_genuines: float = 0.1 
    bandwidth_impostors: float = 0.1 
    
    # Amostras usadas para o cálculo do GridSearch (Performance)
    max_samples_for_gridsearch: int = 5_000
    
    # Limite de pontos salvos no modelo final (Cadeia de Custódia)
    max_stored_samples: int = 20_000

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o KDEParameters utilizando os métodos from_str dos Enums
        para garantir resiliência e fallback.
        """
        if not data:
            return None

        # Reconstrução segura dos Enums remanescentes
        data['score_metric'] = ScoreMetric.from_str(data.get('score_metric'))
        data['kernel'] = KDEKernelType.from_str(data.get('kernel'))
            
        return cls(**data)