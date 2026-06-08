# faceaudit/models/typicality/typicality_model_summary.py

"""DTO leve para `TypicalityScoringModelProtocol.summary()`.

Substitui o dict ad-hoc usado pelos modelos hoje. Estrutura comum +
`backend_details` extensível para diagnósticos específicos de cada
backend (weights/kappas no MoVMF; k/loo_quantiles no k-NN).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType


@dataclass(frozen=True)
class TypicalityModelSummary:
    """Resumo de auditoria de um modelo de tipicidade treinado.

    Estrutura comum entre k-NN e MoVMF; `backend_details` carrega o que
    é específico de cada backend.
    """

    model_type: TypicalityModelType
    model_name: str
    embedding_dim: int
    sample_count: int
    enabled_score_types: Tuple[TypicalityScoreType, ...]

    health_status: str                                  # "OK" / "WARNING" / "CRITICAL"
    warnings: Tuple[str, ...] = ()
    backend_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Defensive copy: garante imutabilidade efetiva da tupla e do dict.
        object.__setattr__(self, "enabled_score_types", tuple(self.enabled_score_types))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        # backend_details: mantém Mapping (pode ser MappingProxyType se quiser
        # imutabilidade estrita, mas frozendict não é padrão; aceitamos dict).
