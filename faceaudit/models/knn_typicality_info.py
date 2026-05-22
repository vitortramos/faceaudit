# faceaudit/models/knn_typicality_info.py

from dataclasses import dataclass, field
from typing import List, Optional

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


@dataclass
class KNNTypicalityInfo(JSONSerializableMixin):
    """
    Métricas de tipicidade não-paramétrica para um embedding facial
    avaliado contra uma população de referência via k-NN esférico.

    Convenções:
    - `typicality_percentile`: 0 a 100. Quanto maior, mais típico
      é o query em relação à distribuição da população de referência.
    - `mean_knn_similarity`: cosseno médio aos `k` vizinhos mais próximos.
      Cresce monotonicamente com a densidade local.
    - `log_density_knn`: estimador de Loftsgaarden-Quesenberry em S^{d-1}.
      Pode entrar direto como p(x | H_d) num LR generativo.
    - `angular_radius_to_kth_nn`: raio angular da menor calota esférica
      que contém os `k` vizinhos.
    """

    # -------------------------------------------------------------------------
    # Identificação do modelo
    # -------------------------------------------------------------------------
    model_name: str = field(default="KNNTypicalityModel")
    population_label: Optional[str] = field(default=None)

    # -------------------------------------------------------------------------
    # Metadata da população de referência
    # -------------------------------------------------------------------------
    embedding_dim: int = field(default=0)
    k: int = field(default=0)
    sample_count: int = field(default=0)

    # -------------------------------------------------------------------------
    # Scores principais (essas três linhas são o "resultado" do modelo)
    # -------------------------------------------------------------------------
    mean_knn_similarity: float = field(default=0.0)
    typicality_percentile: float = field(default=0.0)   # 0..100
    log_density_knn: float = field(default=0.0)

    # -------------------------------------------------------------------------
    # Diagnóstico geométrico/OOD
    # -------------------------------------------------------------------------
    angular_radius_to_kth_nn: float = field(default=0.0)   # radianos
    is_in_distribution: bool = field(default=True)

    # -------------------------------------------------------------------------
    # Opcional: top-k similaridades (útil pra debug/visualização, custoso pra
    # serialização em massa — pode ficar None em batch processing).
    # -------------------------------------------------------------------------
    knn_similarities: Optional[List[float]] = field(default=None)

    # -------------------------------------------------------------------------
    # Campo livre
    # -------------------------------------------------------------------------
    notes: Optional[str] = field(default=None)
