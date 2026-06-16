# faceaudit/services/typicality_model_backend_registry.py

"""Registry único da família de modelos de tipicidade.

Ponto único de verdade para:
- enum value <-> classe concreta
- `potential_score_types` (conceitual; subset efetivo da instância está em
  `enabled_score_types` da própria instância)
- badge (`STABLE`/`EXPERIMENTAL`)
- descrição UI

Análogo a `LR_ARCHITECTURE_SPECS` para arquiteturas LR. Em PR 1b o
registry é populado por hooks no próprio módulo de inicialização — sem
discovery automático. Quando um novo backend for adicionado, basta
chamar `register(...)` no módulo da implementação.

`config_factory` e `trainer` ficam stubs em PR 1b para o MoVMF (sem
trainer exposto). Em PR 2 os configs reais são plugados; em PR 4 o
trainer MoVMF passa por `ReferenceScoreCalibrationService`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Type

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType


@dataclass(frozen=True)
class TypicalityBackendSpec:
    """Especificação registrada por backend."""

    model_type: TypicalityModelType
    model_class: type
    """Classe concreta (`KNNTypicalityModel`, `MoVMFTypicalityModel`)."""

    potential_score_types: frozenset[TypicalityScoreType]
    """Score types que o backend pode CONCEITUALMENTE produzir.

    Distinto de `enabled_score_types` da instância: a instância pode ter
    subset efetivo (ex.: MoVMF treinado sem CDF não tem PERCENTILE).
    UI antes do treinamento mostra o potencial; executor valida o
    efetivo.
    """

    badge: str = "STABLE"
    """`STABLE` ou `EXPERIMENTAL`."""

    label_pt: str = ""
    description: str = ""

    config_class: Optional[type] = None
    """DTO de config (plugado no PR 2)."""

    trainer: Optional[Callable] = None
    """Callable (config, embeddings, identity_ids) -> trained model.
    Plugado no PR 2; no PR 4 passa por ReferenceScoreCalibrationService."""


class TypicalityModelBackendRegistry:
    """Ponto único de verdade para backends de tipicidade."""

    _registry: Dict[TypicalityModelType, TypicalityBackendSpec] = {}

    @classmethod
    def register(cls, spec: TypicalityBackendSpec) -> None:
        if spec.model_type in cls._registry:
            existing = cls._registry[spec.model_type]
            if existing.model_class is spec.model_class:
                return  # idempotente
            raise ValueError(
                f"TypicalityModelBackendRegistry: backend "
                f"{spec.model_type.to_str()} já registrado com "
                f"{existing.model_class.__name__}; tentou registrar "
                f"{spec.model_class.__name__}."
            )
        cls._registry[spec.model_type] = spec

    @classmethod
    def get(cls, model_type: TypicalityModelType) -> TypicalityBackendSpec:
        if model_type not in cls._registry:
            raise KeyError(
                f"TypicalityModelBackendRegistry: backend "
                f"{model_type.to_str()} não registrado. Backends "
                f"disponíveis: {sorted(t.to_str() for t in cls._registry)}."
            )
        return cls._registry[model_type]

    @classmethod
    def resolve_class(cls, model_type: TypicalityModelType) -> type:
        return cls.get(model_type).model_class

    @classmethod
    def all_specs(cls) -> Dict[TypicalityModelType, TypicalityBackendSpec]:
        return dict(cls._registry)

    @classmethod
    def _clear_for_testing(cls) -> None:
        """Usado apenas em testes — não chamar em código de produção."""
        cls._registry.clear()


def _register_default_backends() -> None:
    """Popula o registry com os backends que vivem no projeto.

    Chamado uma vez por `__init__` do módulo do registry — idempotente
    via `register`. `config_class` é plugado em PR 2; `trainer` é
    plugado em PR 4 (o executor unificado faz dispatch inline em PR 2).
    """
    from faceaudit.models.typicality.knn_typicality_model_config import (
        KNNTypicalityModelConfig,
    )
    from faceaudit.models.typicality.movmf_typicality_model_config import (
        MoVMFTypicalityModelConfig,
    )
    from faceaudit.services.knn_typicality_model import KNNTypicalityModel
    from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel

    TypicalityModelBackendRegistry.register(
        TypicalityBackendSpec(
            model_type=TypicalityModelType.KNN,
            model_class=KNNTypicalityModel,
            potential_score_types=frozenset({
                TypicalityScoreType.MEAN_KNN_SIMILARITY,
                TypicalityScoreType.TYPICALITY_PERCENTILE,
                TypicalityScoreType.LOG_DENSITY_ESTIMATE,
            }),
            badge="STABLE",
            label_pt=TypicalityModelType.KNN.label_pt,
            description=TypicalityModelType.KNN.description,
            config_class=KNNTypicalityModelConfig,
            trainer=None,        # PR 4
        )
    )

    TypicalityModelBackendRegistry.register(
        TypicalityBackendSpec(
            model_type=TypicalityModelType.MOVMF,
            model_class=MoVMFTypicalityModel,
            potential_score_types=frozenset({
                TypicalityScoreType.TYPICALITY_PERCENTILE,
                TypicalityScoreType.LOG_DENSITY_ESTIMATE,
            }),
            badge="EXPERIMENTAL",
            label_pt=TypicalityModelType.MOVMF.label_pt,
            description=TypicalityModelType.MOVMF.description,
            config_class=MoVMFTypicalityModelConfig,
            trainer=None,        # PR 4
        )
    )


_register_default_backends()
