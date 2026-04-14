from dataclasses import dataclass
from typing import Optional, Dict, Any

from faceaudit.models.plda_config import PLDAConfig
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.models.split_simulation_result import SplitSimulationResult
from .dataset_stats import DatasetStats

@dataclass
class PLDAValidationInput:
    """
    DTO para o Serviço de Validação PLDA.
    """

    # Configuração do Card
    config: PLDAConfig

    # Stats de domínio (antes do split)
    train_stats: Optional[DatasetStats] = None
    calib_stats: Optional[DatasetStats] = None

    # Pipelines
    source_pipeline_config: Optional[TransformationPipelineConfig] = None
    target_pipeline_config: Optional[TransformationPipelineConfig] = None

    # Dimensão final real (após pipeline)
    final_embedding_dim: Optional[int] = None

    # Indica se preparação já foi executada
    is_prepared: bool = False

    # Métricas reais do pipeline (pós-preparação)
    post_prep_stats: Optional[Dict[str, Any]] = None

    # Métricas precisas da simulação de split
    split_simulation: Optional[SplitSimulationResult] = None 

    # -------------------------
    # Conveniências
    # -------------------------

    @property
    def has_domain_data(self) -> bool:
        return (
            self.train_stats is not None
            and self.calib_stats is not None
            and self.train_stats.n_imgs > 0
            and self.calib_stats.n_imgs > 0
        )

    @property
    def has_data(self) -> bool:
        return self.has_domain_data