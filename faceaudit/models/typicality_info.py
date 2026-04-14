# faceaudit/models/typicality_info.py

from dataclasses import dataclass, field
from typing import Optional, List

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


@dataclass
class TypicalityInfo(JSONSerializableMixin):
    """
    DTO com as informações de typicality retornadas pelo modelo treinado.

    Convenções:
    - quanto maior o log_likelihood, mais típica é a face para a população/modelo;
    - quanto maior o percentile, mais típica é a face dentro da distribuição de treino;
    - em modelos de mistura, alguns campos referem-se ao componente dominante.
    """

    # -------------------------------------------------------------------------
    # Identificação do modelo
    # -------------------------------------------------------------------------
    model_name: str = field(default="TypicalityModel")
    population_label: Optional[str] = field(default=None)

    # -------------------------------------------------------------------------
    # Informações gerais do embedding / score
    # -------------------------------------------------------------------------
    embedding_dim: int = field(default=0)
    log_likelihood: float = field(default=0.0)
    percentile: float = field(default=0.0)          # 0 a 100
    typicality_score: float = field(default=0.0)    # alias útil; geralmente igual ao percentile

    # -------------------------------------------------------------------------
    # Informações do modelo ajustado
    # -------------------------------------------------------------------------
    sample_count: int = field(default=0)
    is_fitted: bool = field(default=False)
    
    # Observabilidade e Treino
    avg_log_likelihood_train: Optional[float] = field(default=None)
    converged: Optional[bool] = field(default=None)

    # -------------------------------------------------------------------------
    # Transformações aplicadas antes do ajuste / inferência
    # -------------------------------------------------------------------------
    center_embeddings_used: bool = field(default=False)
    mean_vector_norm: float = field(default=0.0)

    # -------------------------------------------------------------------------
    # Informações sobre mistura
    # -------------------------------------------------------------------------
    n_components: int = field(default=1)

    # -------------------------------------------------------------------------
    # Estatísticas do componente dominante
    # -------------------------------------------------------------------------
    dominant_component_index: Optional[int] = field(default=None)
    dominant_component_posterior: Optional[float] = field(default=None)
    dominant_component_weight: Optional[float] = field(default=None)
    dominant_component_kappa: Optional[float] = field(default=None)
    cosine_to_dominant_mean_direction: Optional[float] = field(default=None)

    # -------------------------------------------------------------------------
    # Compatibilidade retroativa com a versão anterior
    # -------------------------------------------------------------------------
    kappa: float = field(default=0.0)
    cosine_to_mean_direction: float = field(default=0.0)

    # -------------------------------------------------------------------------
    # Informações opcionais detalhadas da mistura
    # -------------------------------------------------------------------------
    component_posteriors: Optional[List[float]] = field(default=None)
    component_weights: Optional[List[float]] = field(default=None)
    component_kappas: Optional[List[float]] = field(default=None)

    # -------------------------------------------------------------------------
    # Campo livre
    # -------------------------------------------------------------------------
    notes: Optional[str] = field(default=None)