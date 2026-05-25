# faceaudit/models/similarity/plda_similarity_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np
from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB, Ndx, fast_PLDA_scoring

from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.enums.plda_variant import PLDAVariant
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.protocols.similarity_model_protocol import SimilarityModelProtocol

# `two_covariance_plda_service` é importado APENAS para type-checking em tempo
# top-level — em runtime os símbolos são resolvidos via `import` local dentro
# das funções que os usam. Razão: `faceaudit.services.__init__` carrega
# FileSystemService → ProjectConfig → StrategyEntry → PLDAConfig →
# PLDAParameters durante a importação top-level, colidindo com a init de
# `faceaudit.parameters` em ordens específicas (ex: via `faceaudit.enums` em
# testes). Imports locais quebram o ciclo sem afetar comportamento.
if TYPE_CHECKING:
    from faceaudit.services.two_covariance_plda_service import (
        TwoCovariancePLDAState,
    )


class PLDASimilarityModel(SimilarityModelProtocol):
    """
    Modelo de similaridade PLDA puro (sem calibração) usando backend SpeechBrain.

    Responsabilidades:
    - treinar o PLDA a partir de um EmbeddingDataset
    - produzir scores brutos de similaridade (LLR bruto)
    - persistir e restaurar o estado treinado

    Esta classe NÃO faz:
    - calibração
    - regressão logística
    - mineração de pares
    - geração de LR final
    """

    MODEL_VERSION = 2

    def __init__(
        self,
        config: Optional[PLDASimilarityModelConfig] = None,
        name: str = "PLDA Similarity Model",
    ):
        self.config = config or PLDASimilarityModelConfig()
        self._name = name

        # Backend SpeechBrain (variante SIMPLE)
        self.plda: Optional[PLDA] = None

        # Backend numpy puro (variante TWO_COVARIANCE) — type via TYPE_CHECKING
        self.two_cov_state: Optional["TwoCovariancePLDAState"] = None
        # Kernels Q/P/const pré-computados para scoring rápido em batch.
        # Cacheados em fit() e load(), reutilizados em score_*.
        self._two_cov_kernels: Optional[Tuple[np.ndarray, np.ndarray, float]] = None

        # Estado treinado
        self._is_fitted = False
        self.trained_rank: Optional[int] = None
        self.trained_dim: Optional[int] = None

        # RNG para reprodutibilidade futura, quando aplicável
        self._rng = np.random.default_rng(self.config.random_seed)

    # =========================================================================
    # PROPRIEDADES
    # =========================================================================

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def requires_training(self) -> bool:
        return True

    @property
    def input_dimension(self) -> Optional[int]:
        """
        Dimensão dos embeddings esperados na entrada (D), gravada durante o
        fit como `trained_dim`. Property unificada com CosineSimilarityModel
        para validação cross-arch (ex: SIMILARITY_SCORING checa se o
        dataset bate com a dim do modelo).
        """
        return self.trained_dim

    # =========================================================================
    # TREINO
    # =========================================================================

    def fit(
        self,
        dataset: EmbeddingDataset,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Treina o modelo PLDA a partir de um EmbeddingDataset, ramificando
        para o backend correspondente à variante configurada.

        Regras (válidas para ambas as variantes):
        - embeddings devem estar presentes
        - identity=None é removido
        - apenas IDs com >= 2 embeddings entram no treino
        """
        if dataset is None:
            raise ValueError("dataset não pode ser nulo.")

        if len(dataset) == 0:
            raise ValueError("Dataset vazio. Não é possível treinar PLDA.")

        # Pré-processamento comum: extrai X/y filtrando identity=None e
        # IDs com < 2 amostras.
        X_train, y_train = self._extract_training_arrays(dataset)
        D = int(X_train.shape[1])
        C = int(len(np.unique(y_train)))
        N = int(X_train.shape[0])

        # Ramificação por variante.
        if self.config.uses_shrinkage_two_covariance_variant:
            self._fit_shrinkage_two_covariance(
                X_train=X_train,
                y_train=y_train,
                D=D, C=C, N=N,
                logger_callback=logger_callback,
            )
        elif self.config.uses_two_covariance_variant:
            self._fit_two_covariance(
                X_train=X_train,
                y_train=y_train,
                D=D, C=C, N=N,
                logger_callback=logger_callback,
            )
        else:
            self._fit_simple(
                X_train=X_train,
                y_train=y_train,
                D=D, C=C, N=N,
                logger_callback=logger_callback,
            )

        self.trained_dim = D
        self._is_fitted = True

        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

    def _extract_training_arrays(
        self, dataset: EmbeddingDataset
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filtragem comum: remove identity=None e identidades singleton.
        Retorna (X_train, y_train) prontos para os backends.
        """
        X_all = dataset.matrix.astype(np.float64)
        y_all = np.array([item.identity for item in dataset.items], dtype=object)

        if X_all.size == 0:
            raise ValueError("Dataset sem embeddings válidos.")

        mask_labeled = np.array([identity is not None for identity in y_all], dtype=bool)
        X = X_all[mask_labeled]
        y = np.array([str(identity) for identity in y_all[mask_labeled]], dtype=str)

        if X.shape[0] < 2:
            raise ValueError("Dados insuficientes após remover identity=None.")

        unique_ids, counts = np.unique(y, return_counts=True)
        valid_ids = set(unique_ids[counts > 1])
        mask_multi = np.array([identity in valid_ids for identity in y], dtype=bool)

        X_train = X[mask_multi]
        y_train = y[mask_multi]

        if X_train.shape[0] < 2:
            raise ValueError(
                "Dados insuficientes para treinar PLDA "
                "(são necessárias identidades com >= 2 embeddings)."
            )

        return X_train, y_train

    def _fit_simple(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        D: int, C: int, N: int,
        logger_callback: Callable[[str], None],
    ) -> None:
        """Backend SpeechBrain (formulação simplificada com subspace F)."""
        logger_callback(
            f"[{self.name}] 🧠 Treinando PLDA SIMPLE "
            f"(EM={self.config.em_iterations} | "
            f"Mode={self.config.plda_mode.value} | "
            f"latent_dim={self.config.latent_dim} | "
            f"seed={self.config.random_seed})..."
        )

        rank_f = self._decide_rank_f(D=D, C=C)
        self.trained_rank = rank_f

        logger_callback(
            f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D} | rank_f={rank_f}"
        )

        train_stat = self._make_stat_object(X_train, y_train)
        self.plda = PLDA(rank_f=rank_f, nb_iter=self.config.em_iterations)
        self.plda.plda(train_stat, whiten=False)

    def _fit_two_covariance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        D: int, C: int, N: int,
        logger_callback: Callable[[str], None],
    ) -> None:
        """Backend Brümmer 2010 (forma fechada, sem EM)."""
        # Import preguiçoso — vide nota no topo do arquivo.
        from faceaudit.services.two_covariance_plda_service import (
            fit_two_covariance,
            _compute_scoring_kernels,
        )

        logger_callback(
            f"[{self.name}] 🧠 Treinando PLDA TWO_COVARIANCE "
            f"(forma fechada | regularization={self.config.regularization} | "
            f"seed={self.config.random_seed})..."
        )

        logger_callback(
            f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D}"
        )

        self.two_cov_state = fit_two_covariance(
            X=X_train,
            y=y_train,
            regularization=self.config.regularization,
        )

        # Pré-computa kernels de scoring para reuso em score_*.
        self._two_cov_kernels = _compute_scoring_kernels(self.two_cov_state)

        # `trained_rank` não se aplica em duas covariâncias; usamos D como
        # marcador (covariância full-rank).
        self.trained_rank = D

    def _fit_shrinkage_two_covariance(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        D: int, C: int, N: int,
        logger_callback: Callable[[str], None],
    ) -> None:
        """Backend Brümmer 2010 + shrinkage de W (Ledoit-Wolf ou manual)."""
        # Imports preguiçosos — vide nota no topo do arquivo.
        from faceaudit.services.shrinkage_two_covariance_plda_service import (
            fit_shrinkage_two_covariance,
        )
        from faceaudit.services.two_covariance_plda_service import (
            _compute_scoring_kernels,
        )

        mode = self.config.shrinkage_mode
        intensity_log = (
            f" | intensity={self.config.shrinkage_intensity:g}"
            if mode.value == "MANUAL" else ""
        )
        logger_callback(
            f"[{self.name}] 🧠 Treinando PLDA SHRINKAGE_TWO_COVARIANCE "
            f"(forma fechada | shrinkage={mode.value}{intensity_log} | "
            f"regularization={self.config.regularization} | "
            f"seed={self.config.random_seed})..."
        )

        logger_callback(
            f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D}"
        )

        self.two_cov_state = fit_shrinkage_two_covariance(
            X=X_train,
            y=y_train,
            shrinkage_mode=mode,
            shrinkage_intensity=self.config.shrinkage_intensity,
            regularization=self.config.regularization,
        )

        alpha_w = self.two_cov_state.shrinkage_alpha_w
        if alpha_w is not None:
            logger_callback(
                f"[{self.name}] 🎯 α_w aplicado a W: {alpha_w:.4f}"
            )

        self._two_cov_kernels = _compute_scoring_kernels(self.two_cov_state)
        self.trained_rank = D

    # =========================================================================
    # SCORING
    # =========================================================================

    def score_pair(
        self,
        probe: EmbeddingInfo,
        reference: EmbeddingInfo,
    ) -> float:
        """
        Calcula o score bruto PLDA entre dois EmbeddingInfo.
        """
        if probe is None or reference is None:
            raise ValueError("probe e reference não podem ser nulos.")

        return self.score_vectors(probe.embedding, reference.embedding)

    def score_vectors(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> float:
        """
        Calcula o score bruto PLDA (LLR bruto) entre dois vetores.
        Ramifica para o backend correspondente à variante.
        """
        self._ensure_fitted()

        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)

        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError("score_vectors exige vetores 1D.")

        if v1.shape[0] != v2.shape[0]:
            raise ValueError("Os vetores devem ter a mesma dimensionalidade.")

        if self.config.uses_any_two_covariance_variant:
            from faceaudit.services.two_covariance_plda_service import (
                score_two_covariance_pair,
            )
            return score_two_covariance_pair(self.two_cov_state, v1, v2)

        # Backend SpeechBrain (SIMPLE).
        probe_ids = np.array(["p_00000001"], dtype="|O")
        ref_ids = np.array(["r_00000001"], dtype="|O")

        probe_stat = self._make_stat_object(v1.reshape(1, -1), probe_ids)
        ref_stat = self._make_stat_object(v2.reshape(1, -1), ref_ids)

        ndx = Ndx(models=probe_ids, testsegs=ref_ids)
        ndx.trialmask = np.ones((1, 1), dtype=bool)

        scores = fast_PLDA_scoring(
            probe_stat,
            ref_stat,
            ndx,
            self.plda.mean,
            self.plda.F,
            self.plda.Sigma,
        )

        return float(scores.scoremat[0, 0])

    def score_matrix(
        self,
        probe_matrix: np.ndarray,
        ref_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula a matriz completa de scores brutos PLDA. Ramifica para o
        backend correspondente à variante.
        """
        self._ensure_fitted()

        P = np.asarray(probe_matrix, dtype=np.float64)
        R = np.asarray(ref_matrix, dtype=np.float64)

        if P.ndim != 2 or R.ndim != 2:
            raise ValueError("score_matrix exige matrizes 2D.")

        if P.shape[1] != R.shape[1]:
            raise ValueError("probe_matrix e ref_matrix devem ter a mesma dimensão D.")

        n_p, n_r = P.shape[0], R.shape[0]

        if n_p == 0 or n_r == 0:
            return np.empty((n_p, n_r), dtype=np.float64)

        if self.config.uses_any_two_covariance_variant:
            from faceaudit.services.two_covariance_plda_service import (
                score_two_covariance_matrix,
            )
            return score_two_covariance_matrix(
                self.two_cov_state, P, R, kernels=self._two_cov_kernels
            )

        # Backend SpeechBrain (SIMPLE).
        probe_ids = np.array([f"p_{i:08d}" for i in range(n_p)], dtype="|O")
        ref_ids = np.array([f"r_{i:08d}" for i in range(n_r)], dtype="|O")

        probe_stat = self._make_stat_object(P, probe_ids)
        ref_stat = self._make_stat_object(R, ref_ids)

        ndx = Ndx(models=probe_ids, testsegs=ref_ids)
        ndx.trialmask = np.ones((n_p, n_r), dtype=bool)

        scores = fast_PLDA_scoring(
            probe_stat,
            ref_stat,
            ndx,
            self.plda.mean,
            self.plda.F,
            self.plda.Sigma,
        )

        return np.asarray(scores.scoremat, dtype=np.float64)

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Persiste o estado do modelo em disco. O payload inclui o estado
        de ambas as variantes — apenas a relevante estará populada.
        """
        if self.config.uses_shrinkage_two_covariance_variant:
            backend = "shrinkage_two_covariance"
        elif self.config.uses_two_covariance_variant:
            backend = "two_covariance"
        else:
            backend = "speechbrain"

        # Estado das variantes two-cov serializado como dict (cópia rasa do
        # dataclass frozen, JSON/pickle-friendly). Inclui metadados de
        # shrinkage — None para TWO_COVARIANCE puro.
        two_cov_serialized = None
        if self.two_cov_state is not None:
            two_cov_serialized = {
                "mu": self.two_cov_state.mu,
                "B": self.two_cov_state.B,
                "W": self.two_cov_state.W,
                "n_classes": self.two_cov_state.n_classes,
                "n_samples": self.two_cov_state.n_samples,
                "D": self.two_cov_state.D,
                "shrinkage_mode": getattr(self.two_cov_state, "shrinkage_mode", None),
                "shrinkage_target": getattr(self.two_cov_state, "shrinkage_target", None),
                "shrinkage_alpha_w": getattr(self.two_cov_state, "shrinkage_alpha_w", None),
                "shrinkage_alpha_b": getattr(self.two_cov_state, "shrinkage_alpha_b", None),
            }

        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "plda_similarity_model",
            "backend": backend,
            "name": self._name,
            "config": self.config.to_dict(),
            # Variante SIMPLE (SpeechBrain).
            "plda": self.plda,
            # Variante TWO_COVARIANCE.
            "two_cov_state": two_cov_serialized,
            "trained_rank": self.trained_rank,
            "trained_dim": self.trained_dim,
            "is_fitted": self._is_fitted,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """
        Restaura o estado do modelo a partir do disco.

        Retrocompat: payloads de model_version=1 (apenas SpeechBrain) são
        carregados como variante SIMPLE — comportamento idêntico ao anterior.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo do modelo não encontrado: {path_obj}")

        with open(path_obj, "rb") as f:
            payload = pickle.load(f)

        config_dict = payload.get("config", {})
        self.config = PLDASimilarityModelConfig.from_dict(config_dict)

        self._name = payload.get("name", self._name)
        self.plda = payload.get("plda")
        self.trained_rank = payload.get("trained_rank")
        self.trained_dim = payload.get("trained_dim")

        # Hidrata estado TWO_COVARIANCE se presente no payload (model_version
        # ≥ 2). Em payloads antigos (v1), o campo não existe — ignora.
        two_cov_dict = payload.get("two_cov_state")
        if two_cov_dict is not None:
            from faceaudit.services.two_covariance_plda_service import (
                TwoCovariancePLDAState,
                _compute_scoring_kernels,
            )
            self.two_cov_state = TwoCovariancePLDAState(
                mu=np.asarray(two_cov_dict["mu"], dtype=np.float64),
                B=np.asarray(two_cov_dict["B"], dtype=np.float64),
                W=np.asarray(two_cov_dict["W"], dtype=np.float64),
                n_classes=int(two_cov_dict["n_classes"]),
                n_samples=int(two_cov_dict["n_samples"]),
                D=int(two_cov_dict["D"]),
                shrinkage_mode=two_cov_dict.get("shrinkage_mode"),
                shrinkage_target=two_cov_dict.get("shrinkage_target"),
                shrinkage_alpha_w=two_cov_dict.get("shrinkage_alpha_w"),
                shrinkage_alpha_b=two_cov_dict.get("shrinkage_alpha_b"),
            )
            # Recomputa kernels — não persistidos para evitar bloat e desync
            # quando regularização muda.
            self._two_cov_kernels = _compute_scoring_kernels(self.two_cov_state)
        else:
            self.two_cov_state = None
            self._two_cov_kernels = None

        # is_fitted: válido se qualquer dos backends tem estado.
        has_simple = self.plda is not None
        has_two_cov = self.two_cov_state is not None
        self._is_fitted = bool(
            payload.get("is_fitted", has_simple or has_two_cov)
        )

        self._rng = np.random.default_rng(self.config.random_seed)

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Modelo PLDASimilarityModel não treinado. Execute fit() primeiro."
            )

        if self.config.uses_any_two_covariance_variant:
            if self.two_cov_state is None:
                raise RuntimeError(
                    f"Variante {self.config.plda_variant.value} marcada como "
                    "treinada mas sem estado interno (two_cov_state)."
                )
        else:
            if self.plda is None:
                raise RuntimeError(
                    "Variante SIMPLE marcada como treinada mas sem backend "
                    "SpeechBrain (plda)."
                )

    def _make_stat_object(self, X: np.ndarray, ids: np.ndarray) -> StatObject_SB:
        X = np.asarray(X, dtype=np.float64)
        ids = np.asarray(ids).astype(str)

        n = X.shape[0]
        stat0 = np.ones((n, 1), dtype=np.float64)
        none_arr = np.array([None] * n)

        return StatObject_SB(
            modelset=ids.astype("|O"),
            segset=ids.astype("|O"),
            start=none_arr,
            stop=none_arr,
            stat0=stat0,
            stat1=X,
        )

    def _decide_rank_f(self, D: int, C: int) -> int:
        """
        Decide o rank latente com base no modo configurado.
        """
        max_allowed = max(1, min(D, C - 1))

        if self.config.plda_mode == PLDAMode.FULL_RANK:
            return max_allowed

        return max(1, min(self.config.latent_dim, max_allowed))