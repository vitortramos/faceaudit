# faceaudit/models/calibration/logistic_regression_calibration_model.py

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.protocols.calibration_model_protocol import CalibrationModelProtocol


class LogisticRegressionCalibrationModel(CalibrationModelProtocol):
    """
    Modelo de calibração por Regressão Logística.

    Responsabilidades:
    - treinar a partir de scores brutos e rótulos binários
    - produzir probabilidades calibradas
    - produzir LLR calibrado
    - persistir e restaurar seu estado
    """

    MODEL_VERSION = 1

    def __init__(
        self,
        config: LogisticRegressionCalibrationModelConfig | None = None,
        name: str = "Logistic Regression Calibration Model",
    ):
        self.config = config or LogisticRegressionCalibrationModelConfig()
        self._name = name
        self._is_fitted = False

        self.logreg = LogisticRegression(
            C=float(self.config.C),
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )

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

    # =========================================================================
    # TREINO
    # =========================================================================

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        logger_callback: Callable[[str], None] = print,
    ) -> None:
        """
        Treina o calibrador a partir de scores brutos e labels binários.
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int32)

        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            X = scores_arr
        elif scores_arr.ndim == 1:
            X = scores_arr.reshape(-1, 1)
        else:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")

        if labels_arr.ndim != 1:
            raise ValueError("labels deve ser vetor 1D.")

        if X.shape[0] != labels_arr.shape[0]:
            raise ValueError("scores e labels devem ter o mesmo número de amostras.")

        unique_labels = set(np.unique(labels_arr).tolist())
        if not unique_labels.issubset({0, 1}):
            raise ValueError("labels deve conter apenas 0 e 1.")

        if len(unique_labels) < 2:
            raise ValueError("O treino do calibrador requer amostras das duas classes (0 e 1).")

        n_hp = int(np.sum(labels_arr == 1))
        n_hd = int(np.sum(labels_arr == 0))

        logger_callback(
            f"[{self.name}] 🧠 Treinando calibrador LogReg "
            f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | C={self.config.C})..."
        )

        self.logreg.fit(X, labels_arr)
        self._is_fitted = True

        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

    # =========================================================================
    # INFERÊNCIA
    # =========================================================================

    def predict_proba(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna probabilidades calibradas para [Hd, Hp].
        """
        self._ensure_fitted()

        scores_arr = np.asarray(scores, dtype=np.float64)

        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            X = scores_arr
        elif scores_arr.ndim == 1:
            X = scores_arr.reshape(-1, 1)
        else:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, 1).")

        probs = self.logreg.predict_proba(X)
        return np.asarray(probs, dtype=np.float64)

    def calibrate_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna LLR calibrado a partir das probabilidades previstas.
        """
        probs = self.predict_proba(scores)

        eps = 1e-15
        p_hd = np.clip(probs[:, 0], eps, 1.0)
        p_hp = np.clip(probs[:, 1], eps, 1.0)

        llr = np.log(p_hp) - np.log(p_hd)
        return np.asarray(llr, dtype=np.float64)

    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================

    def save(self, path: str) -> None:
        """
        Persiste o estado do calibrador em disco.
        """
        payload = {
            "model_version": self.MODEL_VERSION,
            "model_type": "logistic_regression_calibration_model",
            "name": self._name,
            "config": self.config.to_dict(),
            "logreg": self.logreg,
            "is_fitted": self._is_fitted,
        }

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        """
        Restaura o estado do calibrador a partir do disco.
        """
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Arquivo do calibrador não encontrado: {path_obj}")

        with open(path_obj, "rb") as f:
            payload = pickle.load(f)

        config_dict = payload.get("config", {})
        self.config = LogisticRegressionCalibrationModelConfig.from_dict(config_dict)

        self._name = payload.get("name", self._name)
        self.logreg = payload.get("logreg", self.logreg)
        self._is_fitted = bool(payload.get("is_fitted", False))

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Modelo LogisticRegressionCalibrationModel não treinado. Execute fit() primeiro."
            )