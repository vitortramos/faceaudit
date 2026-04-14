from dataclasses import dataclass
from typing import Optional

from faceaudit.models.kde_config import KDEConfig
# Assumindo que DatasetStats já existe no seu projeto (usado no PLDA também)
# Se não existir, avise que criamos um mock ou interface.
from faceaudit.models.dataset_stats import DatasetStats 

@dataclass
class KDEValidationInput:
    """
    Encapsula os dados necessários para validar uma configuração KDE.
    """
    config: KDEConfig
    probe_stats: Optional[DatasetStats] = None
    reference_stats: Optional[DatasetStats] = None

    @property
    def has_data(self) -> bool:
        return self.probe_stats is not None and self.reference_stats is not None