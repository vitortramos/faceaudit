# faceaudit/models/similarity/cosine_similarity_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np

from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.protocols.similarity_model_protocol import SimilarityModelProtocol


class CosineSimilarityModel(SimilarityModelProtocol):
    """
    Modelo de similaridade cosseno da nova arquitetura.

    Responsabilidades:
    - preparar o modelo de forma leve a partir de um EmbeddingDataset
    - produzir scores brutos de similaridade cosseno
    - persistir e restaurar seu estado

    Observação:
    Este modelo não exige treino estatístico real. O método fit() existe para
    manter compatibilidade com o protocolo e para registrar metadados úteis.
    """

    MODEL_VERSION = 1

    def __init__(self, name: str = "Cosine Similarity Model"):
        self._name = name
        # Cosine não tem estado treinável — fica "fitted" desde a construção.
        # fit() pode ser chamado opcionalmente para registrar metadados, mas
        # scoring funciona sem ele.
        self._is_fitted = True
        self.input_dimension: int | None = None
        self.n_samples_seen: int = 0

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
        return False

    # =========================================================================
    # PREPARAÇÃO
    # =========================================================================

    def fit(
        self,
        dataset: EmbeddingDataset,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Prepara o modelo de similaridade cosseno.

        Não há treino real. Apenas validamos o dataset e registramos metadados
        do espaço de entrada para fins de rastreabilidade.
        """
        if dataset is None:
            raise ValueError("dataset não pode ser nulo.")

        if len(dataset) == 0:
            raise ValueError("Dataset vazio. Não é possível preparar o modelo cosseno.")

        matrix = dataset.matrix
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise ValueError("EmbeddingDataset inválido para preparo do modelo cosseno.")

        self.input_dimension = int(matrix.shape[1])
        self.n_samples_seen = int(matrix.shape[0])
        self._is_fitted = True

        logger_callback(
            f"[{self.name}] ✅ Modelo cosseno preparado "
            f"(N={self.n_samples_seen} | D={self.input_dimension})."
        )

    # =========================================================================
    # SCORING
    # =========================================================================

    def score_pair(
        self,
        probe: EmbeddingInfo,
        reference: EmbeddingInfo,
    ) -> float:
        """
        Calcula o score bruto de similaridade cosseno entre dois EmbeddingInfo.
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
        Calcula a similaridade cosseno entre dois vetores 1D.
        """
        self._ensure_fitted()

        a = np.asarray(v1, dtype=np.float64)
        b = np.asarray(v2, dtype=np.float64)

        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("score_vectors exige vetores 1D.")

        if a.shape[0] != b.shape[0]:
            raise ValueError("Os vetores devem ter a mesma dimensionalidade.")

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a <= 0.0 or norm_b <= 0.0:
            raise ValueError("Não é possível calcular cosseno com vetor de norma zero.")

        return float(np.dot(a, b) / (norm_a * norm_b))

    def score_matrix(
        self,
        probe_matrix: np.ndarray,
        ref_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula a matriz completa de similaridades cosseno entre probes e referências.
        """
        self._ensure_fitted()

        P = np.asarray(probe_matrix, dtype=np.float64)
        R = np.asarray(ref_matrix, dtype=np.float64)

        if P.ndim != 2 or R.ndim != 2:
            raise ValueError("score_matrix exige matrizes 2D.")

        if P.shape[1] != R.shape[1]:
            raise ValueError("probe_matrix e ref_matrix devem ter a mesma dimensão D.")

        if P.shape[0] == 0 or R.shape[0] == 0:
            return np.empty((P.shape[0], R.shape[0]), dtype=np.float64)

        p_norms = np.linalg.norm(P, axis=1, keepdims=True)
        r_norms = np.linalg.norm(R, axis=1, keepdims=True)

        if np.any(p_norms <= 0.0) or np.any(r_norms <= 0.0):
            raise ValueError("Não é possível calcular cosseno com vetores de norma zero.")

        Pn = P / p_norms
        Rn = R / r_norms

        return np.asarray(Pn @ Rn.T, dtype=np.float64)

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Persiste o estado do modelo em disco.
        """
        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "cosine_similarity_model",
            "name": self._name,
            "is_fitted": self._is_fitted,
            "input_dimension": self.input_dimension,
            "n_samples_seen": self.n_samples_seen,
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

        self._name = payload.get("name", self._name)
        self._is_fitted = bool(payload.get("is_fitted", False))
        self.input_dimension = payload.get("input_dimension")
        self.n_samples_seen = int(payload.get("n_samples_seen", 0))

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Modelo CosineSimilarityModel não preparado. Execute fit() primeiro."
            )