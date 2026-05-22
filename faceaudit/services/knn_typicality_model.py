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

from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.models.knn_typicality_info import KNNTypicalityInfo


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
        """
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
