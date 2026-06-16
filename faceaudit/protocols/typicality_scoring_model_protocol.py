# faceaudit/protocols/typicality_scoring_model_protocol.py

"""Protocolos da família de modelos de tipicidade.

Dois protocolos hierárquicos:

- `TypicalityScoringModelProtocol`: contrato mínimo consumido por
  `TYPICALITY_PAIR_SCORING` e por consumidores read-only do modelo
  treinado. Não exige capacidade de `fit` — uma instância carregada via
  `TypicalityModelFactory.from_dict` pode ser usada apenas para scoring.

- `FittableTypicalityModelProtocol(TypicalityScoringModelProtocol)`:
  adiciona `fit(...)`. Consumido pelo executor de
  `TYPICALITY_MODEL_TRAINING`.

A separação evita acoplamento desnecessário: quem só pontua não precisa
saber que o modelo é treinável.

`runtime_checkable` permite `isinstance(payload, ...)` como sanity check,
mas **não substitui validação explícita** do executor (assinaturas,
shapes, fingerprints — ver `compute_scores` consumer).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.typicality.typicality_model_summary import (
    TypicalityModelSummary,
)
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)


@runtime_checkable
class TypicalityScoringModelProtocol(Protocol):
    """Contrato de scoring (read-only após carregamento)."""

    @property
    def model_type(self) -> TypicalityModelType:
        ...

    @property
    def enabled_score_types(self) -> frozenset[TypicalityScoreType]:
        """Score types que esta instância concreta pode produzir agora.

        Pode ser subset de `potential_score_types` do registry: ex., um
        MoVMF treinado sem CDF de referência não pode produzir
        `TYPICALITY_PERCENTILE`. UI antes do treinamento mostra o
        potencial; executor valida o efetivo via esta propriedade.
        """
        ...

    @property
    def embedding_dim(self) -> int:
        ...

    @property
    def sample_count(self) -> int:
        ...

    @property
    def reference_score_basis_type(self) -> TypicalityScoreType:
        """Grandeza-base usada para o cálculo interno de percentil e OOD.

        P0-R2-2 (revisão externa 2026-06-09 round 2): cada backend tem
        uma grandeza canônica para a qual sua distribuição de
        referência (`_train_log_likelihoods` no MoVMF,
        `loo_mean_knn_sorted` no KNN) está calibrada. O
        `ReferenceScoreCalibrationService` consulta esta propriedade
        para pedir os scores OOF na escala correta — sem isso o
        serviço escolhia silenciosamente `LOG_DENSITY_ESTIMATE` do
        KNN como referência e instalava-os em um state cujo
        `get_percentiles` consulta `T_k`, gerando todos os percentis
        em 100%.

        - KNN → `MEAN_KNN_SIMILARITY`
        - MoVMF → `LOG_DENSITY_ESTIMATE`
        """
        ...

    @property
    def model_state_fingerprint(self) -> str:
        """SHA-256 do estado numérico (parâmetros + CDF + threshold)."""
        ...

    @property
    def model_contract_fingerprint(self) -> str:
        """SHA-256 do contrato semântico (model_type +
        score_semantics_ids + população + upstream embedding fingerprint).

        Este é o fingerprint que entra no
        `vectorization_contract_fingerprint` downstream — INHERIT compara
        este, não o `state`.
        """
        ...

    def compute_scores(
        self,
        embeddings: np.ndarray,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch:
        """Aplica o modelo aos embeddings e retorna o batch tipado.

        Levanta `UnsupportedTypicalityScoreTypeError` se
        `score_type not in self.enabled_score_types`.
        """
        ...

    def is_in_distribution(self, embeddings: np.ndarray) -> np.ndarray:
        """Booleano por embedding usando o threshold OOD do modelo."""
        ...

    def summary(self) -> TypicalityModelSummary:
        """Resumo de auditoria + health checks."""
        ...


@runtime_checkable
class FittableTypicalityModelProtocol(
    TypicalityScoringModelProtocol, Protocol
):
    """Contrato de treinamento (estende o de scoring)."""

    def fit(
        self,
        embeddings: np.ndarray,
        *,
        identity_ids: Optional[tuple[str, ...]] = None,
    ) -> "FittableTypicalityModelProtocol":
        """Treina o modelo. `identity_ids` é obrigatório quando o config
        usa modo populacional baseado em identidade (centroides ou
        ponderação inversa).
        """
        ...


class UnsupportedTypicalityScoreTypeError(ValueError):
    """Levantado quando o `score_type` solicitado não está em
    `enabled_score_types` da instância.
    """

    def __init__(
        self,
        *,
        requested: TypicalityScoreType,
        enabled: frozenset[TypicalityScoreType],
        model_type: TypicalityModelType,
    ) -> None:
        msg = (
            f"Score type {requested.to_str()!r} não suportado pela "
            f"instância de {model_type.to_str()!r}. "
            f"Enabled: {sorted(s.to_str() for s in enabled)}."
        )
        super().__init__(msg)
        self.requested = requested
        self.enabled = enabled
        self.model_type = model_type
