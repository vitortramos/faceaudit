# faceaudit/models/similarity/plda_similarity_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from speechbrain.processing.PLDA_LDA import PLDA, StatObject_SB, Ndx, fast_PLDA_scoring

from faceaudit.enums.plda_mode import PLDAMode
from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.similarity.plda_similarity_model_config import (
    PLDASimilarityModelConfig,
)
from faceaudit.protocols.similarity_model_protocol import SimilarityModelProtocol


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

    MODEL_VERSION = 1

    def __init__(
        self,
        config: Optional[PLDASimilarityModelConfig] = None,
        name: str = "PLDA Similarity Model",
    ):
        self.config = config or PLDASimilarityModelConfig()
        self._name = name

        # Backend SpeechBrain
        self.plda: Optional[PLDA] = None

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
        Treina o modelo PLDA puro a partir de um EmbeddingDataset.

        Regras:
        - embeddings devem estar presentes
        - identity=None é removido
        - apenas IDs com >= 2 embeddings entram no treino
        """
        if dataset is None:
            raise ValueError("dataset não pode ser nulo.")

        if len(dataset) == 0:
            raise ValueError("Dataset vazio. Não é possível treinar PLDA.")

        logger_callback(
            f"[{self.name}] 🧠 Treinando PLDA puro "
            f"(EM={self.config.em_iterations} | "
            f"Mode={self.config.plda_mode.value} | "
            f"latent_dim={self.config.latent_dim} | "
            f"seed={self.config.random_seed})..."
        )

        X_all = dataset.matrix.astype(np.float64)
        y_all = np.array([item.identity for item in dataset.items], dtype=object)

        if X_all.size == 0:
            raise ValueError("Dataset sem embeddings válidos.")

        # Remove identity=None
        mask_labeled = np.array([identity is not None for identity in y_all], dtype=bool)
        X = X_all[mask_labeled]
        y = np.array([str(identity) for identity in y_all[mask_labeled]], dtype=str)

        if X.shape[0] < 2:
            raise ValueError("Dados insuficientes após remover identity=None.")

        # Mantém apenas IDs com >= 2 amostras
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

        D = int(X_train.shape[1])
        C = int(len(np.unique(y_train)))
        N = int(X_train.shape[0])

        rank_f = self._decide_rank_f(D=D, C=C)
        self.trained_rank = rank_f
        self.trained_dim = D

        logger_callback(
            f"[{self.name}] 📊 Treino: N={N} | Classes={C} | D={D} | rank_f={rank_f}"
        )

        train_stat = self._make_stat_object(X_train, y_train)

        self.plda = PLDA(rank_f=rank_f, nb_iter=self.config.em_iterations)
        self.plda.plda(train_stat, whiten=False)

        self._is_fitted = True

        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

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
        """
        self._ensure_fitted()

        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)

        if v1.ndim != 1 or v2.ndim != 1:
            raise ValueError("score_vectors exige vetores 1D.")

        if v1.shape[0] != v2.shape[0]:
            raise ValueError("Os vetores devem ter a mesma dimensionalidade.")

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
        Calcula a matriz completa de scores brutos PLDA.
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
        Persiste o estado do modelo em disco.
        """
        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "plda_similarity_model",
            "backend": "speechbrain",
            "name": self._name,
            "config": self.config.to_dict(),
            "plda": self.plda,
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
        self._is_fitted = bool(payload.get("is_fitted", self.plda is not None))

        self._rng = np.random.default_rng(self.config.random_seed)

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._is_fitted or self.plda is None:
            raise RuntimeError("Modelo PLDASimilarityModel não treinado. Execute fit() primeiro.")

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