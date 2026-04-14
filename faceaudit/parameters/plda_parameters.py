from dataclasses import dataclass, field, asdict
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.protocols.model_parameters import ModelParametersProtocol
from faceaudit.enums.plda_mode import PLDAMode

T = TypeVar("T", bound="PLDAParameters")

@dataclass
class PLDAParameters(ModelParametersProtocol, JSONSerializableMixin):
    """
    Hiperparâmetros estritos para a estratégia PLDA (Probabilistic Linear Discriminant Analysis).
    
    Categorias:
    1. Algoritmo EM: Configura as iterações do Expectation-Maximization.
    2. Dimensionalidade: Define se o rank será cheio ou reduzido (Low Rank).
    3. Calibração: Parâmetro da Regressão Logística. 
       (A mineração e proporção de impostores foram delegadas para o MiningConfig e SplitConfig).
    """
    
    # --- 1. Algoritmo EM (Expectation-Maximization) ---
    em_iterations: int = 20
    
    # --- 2. Controle de Dimensionalidade (SpeechBrain) ---
    plda_mode: PLDAMode = PLDAMode.FULL_RANK
    latent_dim: int = 128  # Aplicado apenas em LOW_RANK
    
    # --- 3. Calibração (Logistic Regression) ---
    logreg_C: float = 1.0 
    
    # --- 4. Performance e Sistema ---
    score_block_size: int = 512
    seed: int = 42

    # --- Feature Flags de Pipeline (Override) ---
    use_pca: bool = False 
    pca_dim: int = 128

    def __post_init__(self):
        """Sanitização de valores para evitar erros matemáticos no solver."""
        self.em_iterations = max(1, self.em_iterations)
        self.latent_dim = max(1, self.latent_dim)
        self.score_block_size = max(32, self.score_block_size)

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """
        Reconstrói o objeto garantindo a conversão resiliente de strings para Enums.
        """
        if not data:
            return None

        # Como removemos os campos legados (min_impostors, etc), 
        # o mixin base ignoraria chaves extras do dict se estiver no Python >3.10. 
        # Em versões anteriores, podemos filtrar para não quebrar:
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        if 'plda_mode' in filtered_data:
            # Usando o padrão de resiliência que estabelecemos
            filtered_data['plda_mode'] = PLDAMode.from_str(filtered_data['plda_mode'])
            
        return cls(**filtered_data)
    
    def to_dict(self) -> dict:
        """
        Converte o dataclass para dicionário, garantindo que objetos complexos (Enums)
        sejam convertidos para tipos primitivos seguros para JSON.
        """
        data = asdict(self)
        
        # Converte o Enum para a sua representação em String usando a nossa padronização
        if 'plda_mode' in data and isinstance(data['plda_mode'], PLDAMode):
            data['plda_mode'] = data['plda_mode'].to_str()
            
        return data