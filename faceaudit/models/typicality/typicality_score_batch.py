# faceaudit/models/typicality/typicality_score_batch.py

"""
DTO tipado para o retorno de `compute_scores` em modelos de tipicidade.

Carrega `scores` (vetor numérico) + proveniência completa: tipo de score,
backend, fingerprints do modelo de origem, base do percentil (quando
aplicável). Substitui o uso de `np.ndarray` isolado, garantindo que a
semântica não se perca ao propagar para `TYPICALITY_PAIR_SCORING` e,
downstream, para `LR_ARCHITECTURE_VECTORIZATION`.

O array `scores` é tornado **imutável** no `__post_init__`
(`.copy() + setflags(write=False)`), porque `frozen=True` no dataclass
congela apenas as referências, não o conteúdo do buffer numpy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType


@dataclass(frozen=True)
class TypicalityScoreBatch:
    """Retorno tipado de `TypicalityScoringModelProtocol.compute_scores`.

    Campos:
        scores: vetor (N,) float64 imutável (write=False após
            `__post_init__`).
        n: tamanho do batch.
        embedding_dim: dimensão dos embeddings que produziram os scores.
        score_type: enum identificando o tipo de score
            (`MEAN_KNN_SIMILARITY`, `TYPICALITY_PERCENTILE`,
            `LOG_DENSITY_ESTIMATE`/`LOG_DENSITY_KNN` legado, etc.).
        score_semantics_id: identificador estável da semântica matemática
            (ex.: `"LOG_DENSITY_ESTIMATE_LQ_SPHERICAL"`,
            `"LOG_LIKELIHOOD_MOVMF"`, `"MEAN_KNN_SIMILARITY_SPHERICAL"`).
            Usado para hash e auditoria; ID de máquina, não label
            amigável.
        percentile_basis_score_type: quando `score_type ==
            TYPICALITY_PERCENTILE`, identifica sobre qual grandeza
            subjacente o percentil foi calculado. None nos demais casos.
        reference_score_distribution_fingerprint: hash da distribuição
            empírica de referência usada para o percentile lookup. None
            quando o score não é um percentil.
        model_type: backend que gerou os scores.
        model_state_fingerprint: fingerprint do estado numérico do modelo.
        model_contract_fingerprint: fingerprint do contrato semântico do
            modelo (entra no `vectorization_contract_fingerprint`
            downstream).
        batch_fingerprint: hash deste batch específico (scores +
            metadata). Permite detecção de divergência exata.
    """

    # Dados (imutáveis após __post_init__)
    scores: np.ndarray
    n: int
    embedding_dim: int

    # Identidade do score
    score_type: TypicalityScoreType
    score_semantics_id: str

    # Quando score_type == TYPICALITY_PERCENTILE
    percentile_basis_score_type: Optional[TypicalityScoreType] = None
    reference_score_distribution_fingerprint: Optional[str] = None

    # Proveniência do modelo
    model_type: TypicalityModelType = field(default=TypicalityModelType.KNN)
    model_state_fingerprint: str = ""
    model_contract_fingerprint: str = ""

    # Identidade do próprio batch
    batch_fingerprint: str = ""

    def __post_init__(self) -> None:
        # P0-6 (revisão externa 2026-06-09): ordem canônica do
        # `__post_init__` é
        #   1. validar (shape, n)
        #   2. normalizar (float64 C-contíguo + write=False)
        #   3. remover campos semanticamente irrelevantes (percentile_basis
        #      e reference_distribution_fingerprint só existem em scores
        #      TYPICALITY_PERCENTILE)
        #   4. computar fingerprint
        #   5. validar fingerprint fornecido externamente (não confiar
        #      cegamente)
        # Antes da correção o fingerprint era calculado em (4) ANTES de
        # (3), então dois batches com mesmo conteúdo observável mas
        # metadata supérflua diferente tinham fingerprints divergentes.

        # 1 + 2 — normalização do array
        arr = np.ascontiguousarray(np.asarray(self.scores, dtype=np.float64))
        arr = arr.copy()
        arr.setflags(write=False)
        object.__setattr__(self, "scores", arr)

        if arr.ndim != 1:
            raise ValueError(
                f"TypicalityScoreBatch.scores deve ser 1D; recebido "
                f"shape={arr.shape}."
            )
        if int(self.n) != int(arr.shape[0]):
            raise ValueError(
                f"TypicalityScoreBatch: n={self.n} difere de "
                f"len(scores)={arr.shape[0]}."
            )

        # P1-R2-9 (revisão externa 2026-06-09 round 2): invariantes
        # mais estritos. Este DTO é fronteira de proveniência —
        # deixar passar batch malformado contamina toda a cadeia
        # downstream (LR architecture, score vectorization, etc.).
        if not np.all(np.isfinite(arr)):
            n_bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"TypicalityScoreBatch.scores tem {n_bad}/{arr.size} valores "
                "não finitos (NaN ou inf). Score batch é fronteira de "
                "proveniência — não pode propagar invariantes quebrados."
            )
        if int(self.embedding_dim) <= 0:
            raise ValueError(
                f"TypicalityScoreBatch.embedding_dim deve ser > 0; "
                f"recebido {self.embedding_dim}."
            )
        if not self.model_state_fingerprint:
            raise ValueError(
                "TypicalityScoreBatch.model_state_fingerprint é obrigatório."
            )
        if not self.model_contract_fingerprint:
            raise ValueError(
                "TypicalityScoreBatch.model_contract_fingerprint é obrigatório."
            )

        # 3 — coerência semântica + limpeza de campos irrelevantes ANTES
        # do fingerprint.
        if self.score_type == TypicalityScoreType.TYPICALITY_PERCENTILE:
            if self.percentile_basis_score_type is None:
                raise ValueError(
                    "TypicalityScoreBatch: score_type=TYPICALITY_PERCENTILE "
                    "requer percentile_basis_score_type explícito."
                )
            # P1-R2-9: reference distribution fingerprint é obrigatório
            # em percentile — o número 47.3% só faz sentido referente a
            # uma distribuição específica.
            if not self.reference_score_distribution_fingerprint:
                raise ValueError(
                    "TypicalityScoreBatch: score_type=TYPICALITY_PERCENTILE "
                    "requer reference_score_distribution_fingerprint não "
                    "vazio (proveniência da CDF que define o percentile)."
                )
            # Percentis devem estar em [0, 100]; tolerância pequena para
            # arredondamento.
            if arr.size > 0 and (arr.min() < -1e-6 or arr.max() > 100.0 + 1e-6):
                raise ValueError(
                    f"TypicalityScoreBatch: percentile fora de [0, 100]: "
                    f"min={arr.min():.4f}, max={arr.max():.4f}."
                )
        else:
            if self.percentile_basis_score_type is not None:
                object.__setattr__(self, "percentile_basis_score_type", None)
            if self.reference_score_distribution_fingerprint is not None:
                object.__setattr__(
                    self, "reference_score_distribution_fingerprint", None
                )

        # 4 — calcula o fingerprint canônico
        provided_fp = self.batch_fingerprint
        canonical_fp = self._compute_batch_fingerprint(arr)
        object.__setattr__(self, "batch_fingerprint", canonical_fp)

        # 5 — valida fingerprint fornecido externamente (anti-spoof
        # leve; não confiamos cegamente em hash externo).
        if provided_fp and provided_fp != canonical_fp:
            raise ValueError(
                "TypicalityScoreBatch: batch_fingerprint fornecido "
                f"({provided_fp[:16]}…) diverge do canônico computado "
                f"({canonical_fp[:16]}…). Caller passou um hash que não "
                "corresponde ao conteúdo do batch — possível mutação "
                "silenciosa de metadata."
            )

    def _compute_batch_fingerprint(self, arr: np.ndarray) -> str:
        """SHA-256 dos scores + metadados de proveniência.

        Não inclui `embedding_dim` porque o batch já é derivado do shape
        do array de entrada — `embedding_dim` está implícito na proveniência
        via `model_state_fingerprint`.
        """
        h = hashlib.sha256()
        # Bytes do array (já é float64 C-contíguo)
        h.update(arr.tobytes(order="C"))
        # Metadata
        meta_parts = [
            self.score_type.to_str(),
            self.score_semantics_id,
            self.percentile_basis_score_type.to_str()
            if self.percentile_basis_score_type else "",
            self.reference_score_distribution_fingerprint or "",
            self.model_type.to_str(),
            self.model_state_fingerprint,
            self.model_contract_fingerprint,
        ]
        h.update("\n".join(meta_parts).encode("utf-8"))
        return h.hexdigest()
