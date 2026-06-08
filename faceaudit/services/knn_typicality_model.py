# faceaudit/services/knn_typicality_model.py

"""
Modelo de tipicidade não-paramétrico baseado em k-NN esférico.

Filosofia:
  Em vez de assumir uma família distribucional (vMF, Power Spherical),
  esse modelo carrega a população de referência inteira e computa,
  para cada query, três métricas complementares de tipicidade:

    1. mean_knn_similarity: T_k(x) = (1/k) * sum_{i=1..k} cos(x, y_(i))
       Cosseno médio aos k vizinhos mais próximos. Cresce com a densidade
       local. Intuitivo e cheap de computar.

    2. typicality_percentile: rank empírico de T_k(x) na distribuição
       leave-one-out de T_k(y_j) sobre a referência. 0..100.
       Defensível em corte como estatística não-paramétrica de ranking
       ("X é mais típico que 73% da população de referência").

    3. log_density_knn: estimador de Loftsgaarden-Quesenberry na esfera.
       log p̂(x) ≈ log k - log N - (d-1) * log(theta_k)
       Pode entrar como p(x | H_d) num LR generativo.

Modos de referência (reference_mode):
  - PER_IDENTITY (default): a referência tem 1 ponto por identidade
    (centroide L2-renormalizado dos embeddings dela). Tipicidade mede
    "esta pessoa é típica na população", insensível a desbalanceamento
    de fotos por pessoa. Identidades singleton mantidas — centroide
    é a própria face única.
  - PER_FACE: a referência tem 1 ponto por face. Mede "esta foto é
    típica no conjunto de fotos", vulnerável a hubness intra-identidade.

Vantagens vs MoVMF:
  - Sem suposição de simetria rotacional (que hubness viola em FR).
  - Sem inversão de Bessel/Newton.
  - Hubness vira sinal: faces em região densa têm T_k alto.

Custo:
  - O "modelo" carrega a referência inteira na memória/disco.
  - Para PER_IDENTITY com 5 fotos/id médio, a referência fica 5× menor
    que PER_FACE — modelos compactos sem perda de poder forense.
  - Inferência: O(N·D) por query. Para uso forense exato e auditável,
    é o preço justo. Backends aproximados (FAISS) ficam para uma V2 se
    N >> 100k tornar isso problemático.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.knn_typicality_info import KNNTypicalityInfo
from faceaudit.models.typicality.typicality_model_summary import (
    TypicalityModelSummary,
)
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)


ArrayLike = Union[np.ndarray, Iterable[float], Iterable[Iterable[float]]]


@dataclass
class _KNNRefState:
    """Estado interno congelado pós-fit. Imutável conceitualmente."""
    reference: np.ndarray              # (N_ref, D), L2-normalizado, float32
    loo_mean_knn_sorted: np.ndarray    # (N_ref,), float64, ordem crescente
    ood_threshold: float
    k: int
    dim: int
    reference_mode: TypicalityReferenceMode
    # Diagnóstico — quantas faces de origem foram agrupadas no PER_IDENTITY,
    # ou shape original em PER_FACE.
    n_input_faces: int
    n_singleton_identities: int  # só relevante em PER_IDENTITY


class KNNTypicalityModel:
    """
    Tipicidade não-paramétrica via k-NN esférico.

    Uso:
        model = KNNTypicalityModel(k=50, reference_mode=TypicalityReferenceMode.PER_IDENTITY)
        model.fit(reference_embeddings, identities=reference_identities)
        info = model.get_typicality_info(query_embedding)
        print(info.typicality_percentile, info.log_density_knn)
    """

    def __init__(
        self,
        k: int = 50,
        ood_quantile: float = 0.01,
        reference_mode: TypicalityReferenceMode = TypicalityReferenceMode.PER_IDENTITY,
        model_name: str = "KNNTypicalityModel",
        eps: float = 1e-12,
    ):
        if k < 1:
            raise ValueError("k deve ser >= 1.")
        if not (0.0 <= ood_quantile < 0.5):
            raise ValueError("ood_quantile deve estar em [0, 0.5).")

        self.k = int(k)
        self.ood_quantile = float(ood_quantile)
        self.reference_mode = TypicalityReferenceMode.from_str(reference_mode) \
            if not isinstance(reference_mode, TypicalityReferenceMode) \
            else reference_mode
        self.model_name = model_name
        self.eps = float(eps)

        # Estado pós-fit. None até `fit()` ser chamado.
        self._state: Optional[_KNNRefState] = None

    # =========================================================================
    # API PÚBLICA
    # =========================================================================

    def fit(
        self,
        embeddings: ArrayLike,
        *,
        identities: Optional[Sequence[str]] = None,
        identity_ids: Optional[Sequence[str]] = None,
    ) -> "KNNTypicalityModel":
        """
        Armazena a referência (faces ou centroides por identidade) e
        pré-computa a CDF empírica leave-one-out de T_k pra suporte a
        percentile lookup.

        Args:
            embeddings: (N, D). Será L2-normalizado se ainda não estiver.
            identities: sequência paralela a embeddings com o rótulo de
                identidade de cada face. Obrigatório em PER_IDENTITY,
                ignorado em PER_FACE.
            identity_ids: alias canônico de `identities` definido pelo
                `FittableTypicalityModelProtocol` (PR 1b). Quando
                fornecido sem `identities`, é usado como tal. Erro
                quando ambos forem fornecidos com valores divergentes.
        """
        if identity_ids is not None and identities is not None and identity_ids != identities:
            raise ValueError(
                "KNNTypicalityModel.fit recebeu `identities` e "
                "`identity_ids` divergentes — use apenas um dos dois."
            )
        if identities is None and identity_ids is not None:
            identities = identity_ids
        x_raw = self._validate_and_normalize_2d(embeddings)
        n_input, d = x_raw.shape

        if self.reference_mode is TypicalityReferenceMode.PER_IDENTITY:
            if identities is None:
                raise ValueError(
                    "reference_mode=PER_IDENTITY exige `identities` "
                    "paralelo a `embeddings`."
                )
            if len(identities) != n_input:
                raise ValueError(
                    f"len(identities)={len(identities)} difere de "
                    f"len(embeddings)={n_input}."
                )
            reference, n_singletons = self._aggregate_by_identity(x_raw, identities)
        else:
            reference = x_raw.copy()
            n_singletons = 0

        n_ref = reference.shape[0]
        if n_ref < self.k + 1:
            raise ValueError(
                f"k+1 ({self.k + 1}) pontos de referência são necessários para LOO. "
                f"Obtidos: {n_ref} (modo={self.reference_mode.to_str()})."
            )

        # Similaridade N_ref × N_ref. Exclui self via diagonal = -inf.
        sims = reference @ reference.T
        np.fill_diagonal(sims, -np.inf)

        top_k_idx = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
        top_k_sim = np.take_along_axis(sims, top_k_idx, axis=1)
        loo_T_k = top_k_sim.mean(axis=1).astype(np.float64)

        if not np.all(np.isfinite(loo_T_k)):
            raise RuntimeError(
                "Cálculo LOO produziu valores não finitos. Verifique se a "
                "referência tem pontos degenerados (norma zero)."
            )

        loo_sorted = np.sort(loo_T_k)
        ood_threshold = (
            float(np.quantile(loo_sorted, self.ood_quantile))
            if self.ood_quantile > 0.0
            else float(-np.inf)
        )

        self._state = _KNNRefState(
            reference=reference.astype(np.float32),
            loo_mean_knn_sorted=loo_sorted,
            ood_threshold=ood_threshold,
            k=self.k,
            dim=int(d),
            reference_mode=self.reference_mode,
            n_input_faces=int(n_input),
            n_singleton_identities=int(n_singletons),
        )
        return self

    def is_fitted(self) -> bool:
        return self._state is not None

    def score_samples(self, embeddings: ArrayLike) -> np.ndarray:
        """Retorna log_density_knn pra cada embedding (batch)."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.log_density_knn for i in infos], dtype=np.float64)

    def get_percentiles(self, embeddings: ArrayLike) -> np.ndarray:
        """Retorna o percentile (0..100) pra cada embedding (batch)."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.typicality_percentile for i in infos], dtype=np.float64)

    def is_in_distribution(self, embeddings: ArrayLike) -> np.ndarray:
        """Boolean por embedding: T_k >= ood_threshold."""
        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        return np.array([i.is_in_distribution for i in infos], dtype=bool)

    def get_typicality_info(
        self, embedding: ArrayLike, *, include_top_k: bool = True,
    ) -> KNNTypicalityInfo:
        """Avalia um único embedding e empacota em KNNTypicalityInfo."""
        self._ensure_fitted()
        x = self._validate_and_normalize_1d(embedding)
        return self._compute_info_single(x, include_top_k=include_top_k)

    def get_typicality_infos(
        self, embeddings: ArrayLike, *, include_top_k: bool = False,
    ) -> List[KNNTypicalityInfo]:
        """
        Batch. Default include_top_k=False pra economizar memória — em
        scoring de pares (~milhões), guardar 50 floats por face explode.
        """
        self._ensure_fitted()
        x = self._validate_and_normalize_2d(embeddings)
        return [
            self._compute_info_single(x[i], include_top_k=include_top_k)
            for i in range(x.shape[0])
        ]

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    def to_dict(self) -> dict:
        """Exporta o modelo treinado para um dict JSON-serializável."""
        self._ensure_fitted()
        st = self._state
        assert st is not None
        return {
            "format_version": 2,
            "hyperparameters": {
                "k": self.k,
                "ood_quantile": self.ood_quantile,
                "reference_mode": self.reference_mode.to_str(),
                "model_name": self.model_name,
                "eps": self.eps,
            },
            "learned_state": {
                "reference": st.reference.tolist(),
                "loo_mean_knn_sorted": st.loo_mean_knn_sorted.tolist(),
                "ood_threshold": st.ood_threshold,
                "k": st.k,
                "dim": st.dim,
                "reference_mode": st.reference_mode.to_str(),
                "n_input_faces": st.n_input_faces,
                "n_singleton_identities": st.n_singleton_identities,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KNNTypicalityModel":
        version = data.get("format_version", 1)
        if version not in (1, 2):
            raise ValueError(f"Versão de formato não suportada: {version}")

        hp = dict(data["hyperparameters"])

        # Migração v1 → v2: campos removidos (population_label) + novo
        # campo reference_mode default PER_FACE (era o comportamento
        # implícito antes desse toggle existir).
        if version == 1:
            hp.pop("population_label", None)
            hp.setdefault("reference_mode", TypicalityReferenceMode.PER_FACE.to_str())

        model = cls(
            k=int(hp["k"]),
            ood_quantile=float(hp["ood_quantile"]),
            reference_mode=TypicalityReferenceMode.from_str(
                hp.get("reference_mode", TypicalityReferenceMode.PER_IDENTITY.to_str())
            ),
            model_name=hp.get("model_name", "KNNTypicalityModel"),
            eps=float(hp.get("eps", 1e-12)),
        )

        ls = data["learned_state"]
        reference = np.asarray(ls["reference"], dtype=np.float32)
        loo = np.asarray(ls["loo_mean_knn_sorted"], dtype=np.float64)

        if reference.ndim != 2:
            raise ValueError("reference deve ser 2D.")
        if reference.shape[0] != loo.shape[0]:
            raise ValueError(
                f"Inconsistência: reference tem {reference.shape[0]} "
                f"pontos, loo_mean_knn_sorted tem {loo.shape[0]}."
            )
        norms = np.linalg.norm(reference, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError(
                f"reference não está L2-normalizada (norms min={norms.min()}, "
                f"max={norms.max()})."
            )

        # v1 não carregava reference_mode no state — inferimos PER_FACE.
        state_mode = TypicalityReferenceMode.from_str(
            ls.get("reference_mode", TypicalityReferenceMode.PER_FACE.to_str())
        )

        model._state = _KNNRefState(
            reference=reference,
            loo_mean_knn_sorted=loo,
            ood_threshold=float(ls["ood_threshold"]),
            k=int(ls["k"]),
            dim=int(ls["dim"]),
            reference_mode=state_mode,
            n_input_faces=int(ls.get("n_input_faces", reference.shape[0])),
            n_singleton_identities=int(ls.get("n_singleton_identities", 0)),
        )
        return model

    # =========================================================================
    # DIAGNÓSTICO / OBSERVABILIDADE
    # =========================================================================

    def summary(self) -> dict:
        """Estatísticas de auditoria do modelo treinado."""
        if not self.is_fitted():
            return {"status": "unfitted"}
        st = self._state
        assert st is not None
        loo = st.loo_mean_knn_sorted
        return {
            "status": "fitted",
            "model_name": self.model_name,
            "reference_mode": st.reference_mode.to_str(),
            "k": st.k,
            "sample_count": st.reference.shape[0],
            "n_input_faces": st.n_input_faces,
            "n_singleton_identities": st.n_singleton_identities,
            "embedding_dim": st.dim,
            "ood_quantile": self.ood_quantile,
            "ood_threshold": st.ood_threshold,
            "loo_T_k_min": float(loo[0]),
            "loo_T_k_max": float(loo[-1]),
            "loo_T_k_median": float(np.median(loo)),
            "loo_T_k_mean": float(np.mean(loo)),
            "loo_T_k_std": float(np.std(loo)),
        }

    # =========================================================================
    # PROPRIEDADES READ-ONLY (útil pra UI / inspeção)
    # =========================================================================

    @property
    def sample_count(self) -> Optional[int]:
        """Tamanho da referência (faces em PER_FACE, identidades em PER_IDENTITY)."""
        return int(self._state.reference.shape[0]) if self._state is not None else None

    @property
    def n_input_faces(self) -> Optional[int]:
        """Quantas faces foram passadas pro fit (antes de qualquer agregação)."""
        return self._state.n_input_faces if self._state is not None else None

    @property
    def n_singleton_identities(self) -> Optional[int]:
        """Em PER_IDENTITY: quantas identidades têm uma única face."""
        return self._state.n_singleton_identities if self._state is not None else None

    @property
    def embedding_dim(self) -> Optional[int]:
        return self._state.dim if self._state is not None else None

    @property
    def ood_threshold(self) -> Optional[float]:
        return self._state.ood_threshold if self._state is not None else None

    @property
    def loo_distribution(self) -> Optional[np.ndarray]:
        """Cópia da CDF empírica de T_k — útil pra plot."""
        if self._state is None:
            return None
        return self._state.loo_mean_knn_sorted.copy()

    # =========================================================================
    # TypicalityScoringModelProtocol (PR 1b)
    # =========================================================================

    @property
    def model_type(self) -> TypicalityModelType:
        return TypicalityModelType.KNN

    @property
    def enabled_score_types(self) -> frozenset[TypicalityScoreType]:
        """Quais score types esta instância está pronta para produzir.

        k-NN treinado disponibiliza os três: T_k bruto, percentil
        empírico (sobre T_k) e log-densidade L-Q esférica. Antes do `fit`,
        nenhum.
        """
        if not self.is_fitted():
            return frozenset()
        return frozenset({
            TypicalityScoreType.MEAN_KNN_SIMILARITY,
            TypicalityScoreType.TYPICALITY_PERCENTILE,
            TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        })

    @property
    def embedding_dim(self) -> int:
        self._ensure_fitted()
        return int(self._state.dim)  # type: ignore[union-attr]

    @property
    def sample_count(self) -> int:
        self._ensure_fitted()
        return int(self._state.reference.shape[0])  # type: ignore[union-attr]

    @property
    def model_state_fingerprint(self) -> str:
        """SHA-256 do estado numérico (reference matrix + CDF LOO + threshold).

        Recalculado sob demanda; não é cacheado (consistente com
        movmf/PLDA fingerprints existentes em outros pontos do projeto).
        """
        self._ensure_fitted()
        import hashlib
        st = self._state
        assert st is not None
        h = hashlib.sha256()
        # Referência canonicaliza para float32 C-contígua (formato persistido)
        ref = np.ascontiguousarray(st.reference, dtype=np.float32)
        h.update(ref.tobytes(order="C"))
        loo = np.ascontiguousarray(st.loo_mean_knn_sorted, dtype=np.float64)
        h.update(loo.tobytes(order="C"))
        h.update(f"{st.ood_threshold:.17g}".encode("utf-8"))
        h.update(f"k={st.k};dim={st.dim};rmode={st.reference_mode.to_str()}".encode("utf-8"))
        return h.hexdigest()

    @property
    def model_contract_fingerprint(self) -> str:
        """SHA-256 do contrato semântico (sem o estado bruto).

        Inclui o tipo do modelo, score_semantics_ids habilitados,
        reference_mode, k, ood_quantile. Excluímos `eps` e `model_name`
        porque são cosmeticamente relevantes mas não mudam a matemática.
        """
        self._ensure_fitted()
        import hashlib
        st = self._state
        assert st is not None
        semantics_ids = sorted([
            self._score_semantics_id_for(t) for t in self.enabled_score_types
        ])
        contract_parts = [
            f"model_type={TypicalityModelType.KNN.to_str()}",
            f"semantics_ids={','.join(semantics_ids)}",
            f"reference_mode={st.reference_mode.to_str()}",
            f"k={st.k}",
            f"dim={st.dim}",
            f"ood_quantile={self.ood_quantile:.17g}",
        ]
        h = hashlib.sha256()
        h.update("\n".join(contract_parts).encode("utf-8"))
        return h.hexdigest()

    @staticmethod
    def _score_semantics_id_for(t: TypicalityScoreType) -> str:
        if t == TypicalityScoreType.MEAN_KNN_SIMILARITY:
            return "MEAN_KNN_SIMILARITY_SPHERICAL"
        if t == TypicalityScoreType.TYPICALITY_PERCENTILE:
            return "PERCENTILE_OF_MEAN_KNN_SIMILARITY"
        if t == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            return "LOG_DENSITY_ESTIMATE_LQ_SPHERICAL"
        raise ValueError(f"score_type sem semantics id: {t}")

    def compute_scores(
        self,
        embeddings: ArrayLike,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch:
        """Aplica o k-NN aos embeddings e retorna o batch tipado.

        Levanta `UnsupportedTypicalityScoreTypeError` se o score_type não
        estiver em `enabled_score_types`.
        """
        from faceaudit.protocols.typicality_scoring_model_protocol import (
            UnsupportedTypicalityScoreTypeError,
        )

        self._ensure_fitted()
        if score_type not in self.enabled_score_types:
            raise UnsupportedTypicalityScoreTypeError(
                requested=score_type,
                enabled=self.enabled_score_types,
                model_type=self.model_type,
            )

        infos = self.get_typicality_infos(embeddings, include_top_k=False)
        if score_type == TypicalityScoreType.MEAN_KNN_SIMILARITY:
            arr = np.array([i.mean_knn_similarity for i in infos], dtype=np.float64)
            percentile_basis = None
            ref_dist_fp = None
        elif score_type == TypicalityScoreType.TYPICALITY_PERCENTILE:
            arr = np.array([i.typicality_percentile for i in infos], dtype=np.float64)
            # Base histórica do k-NN: o percentil é sobre MEAN_KNN_SIMILARITY
            # (LOO T_k da referência).
            percentile_basis = TypicalityScoreType.MEAN_KNN_SIMILARITY
            # Fingerprint da CDF de referência (loo_mean_knn_sorted).
            ref_dist_fp = self._reference_score_distribution_fingerprint()
        elif score_type == TypicalityScoreType.LOG_DENSITY_ESTIMATE:
            arr = np.array([i.log_density_knn for i in infos], dtype=np.float64)
            percentile_basis = None
            ref_dist_fp = None
        else:
            raise ValueError(f"score_type desconhecido: {score_type}")

        return TypicalityScoreBatch(
            scores=arr,
            n=int(arr.shape[0]),
            embedding_dim=self.embedding_dim,
            score_type=score_type,
            score_semantics_id=self._score_semantics_id_for(score_type),
            percentile_basis_score_type=percentile_basis,
            reference_score_distribution_fingerprint=ref_dist_fp,
            model_type=self.model_type,
            model_state_fingerprint=self.model_state_fingerprint,
            model_contract_fingerprint=self.model_contract_fingerprint,
        )

    def _reference_score_distribution_fingerprint(self) -> str:
        """SHA-256 da CDF empírica usada como referência para percentiles."""
        import hashlib
        self._ensure_fitted()
        st = self._state
        assert st is not None
        loo = np.ascontiguousarray(st.loo_mean_knn_sorted, dtype=np.float64)
        return hashlib.sha256(loo.tobytes(order="C")).hexdigest()

    def summary_protocol(self) -> TypicalityModelSummary:
        """Implementação do `summary()` do protocolo.

        Nome distinto do método legado `summary()` (que retorna dict) para
        não quebrar consumidores existentes. PR 2 unifica para `summary()`.
        """
        self._ensure_fitted()
        st = self._state
        assert st is not None
        loo = st.loo_mean_knn_sorted

        warnings_list: list[str] = []
        health = "OK"
        if st.n_singleton_identities > 0 and st.reference_mode is TypicalityReferenceMode.PER_IDENTITY:
            warnings_list.append(
                f"{st.n_singleton_identities} identidades singleton (centroide == face única)"
            )

        return TypicalityModelSummary(
            model_type=TypicalityModelType.KNN,
            model_name=self.model_name,
            embedding_dim=int(st.dim),
            sample_count=int(st.reference.shape[0]),
            enabled_score_types=tuple(sorted(self.enabled_score_types, key=lambda t: t.value)),
            health_status=health,
            warnings=tuple(warnings_list),
            backend_details={
                "k": int(st.k),
                "reference_mode": st.reference_mode.to_str(),
                "ood_quantile": float(self.ood_quantile),
                "ood_threshold": float(st.ood_threshold),
                "n_input_faces": int(st.n_input_faces),
                "n_singleton_identities": int(st.n_singleton_identities),
                "loo_T_k_min": float(loo[0]),
                "loo_T_k_max": float(loo[-1]),
                "loo_T_k_median": float(np.median(loo)),
                "loo_T_k_mean": float(np.mean(loo)),
                "loo_T_k_std": float(np.std(loo)),
            },
        )

    # =========================================================================
    # AGREGAÇÃO POR IDENTIDADE
    # =========================================================================

    def _aggregate_by_identity(
        self, embeddings: np.ndarray, identities: Sequence[str],
    ) -> Tuple[np.ndarray, int]:
        """
        Soma os embeddings de cada identidade e L2-renormaliza para obter
        o centroide esférico (direção média na hiperesfera unitária).

        Identidades singleton viram centroide = a face única (renormalizada,
        mantém norma 1). Não há descarte — toda identidade pesa igual no
        espaço de referência.

        Retorna:
            (matriz_centroides, n_singletons_count)
        """
        ids_arr = np.asarray(identities)
        unique_ids, inverse_idx = np.unique(ids_arr, return_inverse=True)

        n_ids = unique_ids.shape[0]
        d = embeddings.shape[1]
        sums = np.zeros((n_ids, d), dtype=np.float64)
        counts = np.zeros(n_ids, dtype=np.int64)

        np.add.at(sums, inverse_idx, embeddings)
        np.add.at(counts, inverse_idx, 1)

        # L2-renormaliza cada centroide. Identidades singleton já estão
        # normalizadas (sums = a face única), mas a divisão segura cobre
        # tudo.
        norms = np.linalg.norm(sums, axis=1, keepdims=True)
        centroids = sums / np.maximum(norms, self.eps)

        if not np.all(np.isfinite(centroids)):
            raise RuntimeError(
                "Centroides por identidade contêm valores não finitos. "
                "Verifique embeddings degenerados."
            )

        n_singletons = int(np.sum(counts == 1))
        return centroids.astype(np.float64), n_singletons

    # =========================================================================
    # CORE COMPUTATION
    # =========================================================================

    def _compute_info_single(
        self, x_normalized: np.ndarray, *, include_top_k: bool,
    ) -> KNNTypicalityInfo:
        """
        x_normalized: (D,) L2-normalizado.
        """
        st = self._state
        assert st is not None

        if x_normalized.shape[0] != st.dim:
            raise ValueError(
                f"Dimensão do query ({x_normalized.shape[0]}) difere da "
                f"referência ({st.dim})."
            )

        sims = st.reference @ x_normalized.astype(st.reference.dtype)  # (N,)

        # Top-k. Tratamento de self-match: se o query for um ponto da referência,
        # vai aparecer com cosseno ≈ 1.0. Pega top k+1 e descarta o ≈1.0 mais alto
        # quando isso acontece — garante consistência com a CDF LOO usada no fit.
        n = sims.shape[0]
        kp = min(self.k + 1, n)
        top_idx = np.argpartition(-sims, kp - 1)[:kp]
        top_sim = sims[top_idx]
        top_sim = np.sort(top_sim)[::-1]  # desc

        if top_sim[0] > 1.0 - 1e-6 and kp > self.k:
            top_sim = top_sim[1 : self.k + 1]
        else:
            top_sim = top_sim[: self.k]

        T_k = float(np.mean(top_sim))
        theta_k = float(np.arccos(np.clip(top_sim[-1], -1.0, 1.0)))

        rank = float(np.searchsorted(st.loo_mean_knn_sorted, T_k, side="left"))
        rank_right = float(np.searchsorted(st.loo_mean_knn_sorted, T_k, side="right"))
        mid_rank = 0.5 * (rank + rank_right)
        percentile = 100.0 * mid_rank / len(st.loo_mean_knn_sorted)

        d = st.dim
        theta_safe = max(theta_k, 1e-12)
        log_density = (
            np.log(self.k)
            - np.log(st.reference.shape[0])
            - (d - 1) * np.log(theta_safe)
        )

        return KNNTypicalityInfo(
            model_name=self.model_name,
            embedding_dim=d,
            k=self.k,
            sample_count=st.reference.shape[0],
            mean_knn_similarity=T_k,
            typicality_percentile=percentile,
            log_density_knn=float(log_density),
            angular_radius_to_kth_nn=theta_k,
            is_in_distribution=bool(T_k >= st.ood_threshold),
            knn_similarities=top_sim.tolist() if include_top_k else None,
            notes=None,
        )

    # =========================================================================
    # HELPERS DE VALIDAÇÃO / NORMALIZAÇÃO
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self.is_fitted():
            raise RuntimeError("KNNTypicalityModel não foi treinado. Chame fit() antes.")

    def _validate_and_normalize_2d(self, embeddings: ArrayLike) -> np.ndarray:
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2:
            raise ValueError("Formato esperado: (N, D) ou (D,).")
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        out = x / np.maximum(norms, self.eps)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("Renormalização gerou valores não finitos.")
        return out

    def _validate_and_normalize_1d(self, embedding: ArrayLike) -> np.ndarray:
        x = np.asarray(embedding, dtype=np.float64)
        if x.ndim == 2:
            if x.shape[0] != 1:
                raise ValueError("Esperado um único embedding (1, D) ou (D,).")
            x = x[0]
        if x.ndim != 1:
            raise ValueError("Formato esperado: vetor 1D.")
        norm = max(float(np.linalg.norm(x)), self.eps)
        return x / norm
