# faceaudit/protocols/similarity_model_protocol.py

from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable
import numpy as np

from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo


@runtime_checkable
class SimilarityModelProtocol(Protocol):
    """
    Protocolo para modelos de similaridade da nova arquitetura.

    Um modelo de similaridade é responsável por:
    - opcionalmente treinar em um EmbeddingDataset
    - produzir scores brutos de similaridade
    - persistir e restaurar seu estado treinado

    Observação:
    O score retornado por este protocolo é um score bruto do modelo.
    No caso de PLDA, isso corresponde ao LLR bruto não calibrado.
    """

    @property
    def name(self) -> str:
        """
        Nome amigável do modelo.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """
        Retorna True se o modelo já foi treinado/preparado.
        """
        ...

    @property
    def requires_training(self) -> bool:
        """
        Retorna True se o modelo exige treino explícito.
        """
        ...

    def fit(
        self,
        dataset: EmbeddingDataset,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Treina/prepara o modelo usando um EmbeddingDataset.
        """
        ...

    def score_pair(
        self,
        probe: EmbeddingInfo,
        reference: EmbeddingInfo,
    ) -> float:
        """
        Calcula o score bruto de similaridade entre dois embeddings.
        """
        ...

    def score_vectors(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
    ) -> float:
        """
        Calcula o score bruto de similaridade entre dois vetores.
        """
        ...

    def score_matrix(
        self,
        probe_matrix: np.ndarray,
        ref_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Calcula a matriz completa de scores brutos entre probes e referências.
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o estado do modelo em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o estado do modelo a partir do disco.
        """
        ...