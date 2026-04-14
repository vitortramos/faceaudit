# faceaudit/protocols/calibration_model_protocol.py

from __future__ import annotations
from typing import Protocol, runtime_checkable, Callable
import numpy as np


@runtime_checkable
class CalibrationModelProtocol(Protocol):
    """
    Protocolo para modelos de calibração da nova arquitetura.

    Um modelo de calibração é responsável por:
    - treinar a partir de scores brutos e rótulos binários
    - produzir scores/probabilidades calibradas
    - persistir e restaurar seu estado treinado

    Convenção de rótulos:
    - 1 = genuíno / mesma identidade / hipótese Hp
    - 0 = impostor / identidades diferentes / hipótese Hd
    """

    @property
    def name(self) -> str:
        """
        Nome amigável do calibrador.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        """
        Retorna True se o calibrador já foi treinado.
        """
        ...

    @property
    def requires_training(self) -> bool:
        """
        Retorna True se o calibrador exige treino explícito.
        """
        ...

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Treina o calibrador a partir de scores brutos e labels binários.
        """
        ...

    def predict_proba(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna probabilidades calibradas para as classes [Hd, Hp].

        Espera-se retorno com shape (N, 2), compatível com scikit-learn:
        - coluna 0: P(Hd)
        - coluna 1: P(Hp)
        """
        ...

    def calibrate_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna os scores calibrados em forma de LLR calibrado,
        quando aplicável ao modelo.
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o estado do calibrador em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o estado do calibrador a partir do disco.
        """
        ...