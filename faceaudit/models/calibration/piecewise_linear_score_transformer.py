# faceaudit/models/calibration/piecewise_linear_score_transformer.py

"""Transformer sklearn-compatible que aprende `hinge_anchor_s0` no
treino e aplica expansão piecewise-linear.

**Crítico para correção estatística**: ao inserir este transformer
dentro de uma `sklearn.pipeline.Pipeline`, o GridSearchCV re-treina
o `fit()` (que aprende `s0`) em cada fold usando APENAS os dados
do treino do fold. Sem isso, o `s0` estimado globalmente vazaria
informação do holdout pra cada fold do CV, inflando otimisticamente
o Cllr CV.

Pipeline canônico:

    Pipeline([
        ("piecewise", PiecewiseLinearScoreTransformer(config)),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(...)),
    ])

Transform:

    z = score − hinge_anchor_s0_
    features = [z, max(z − offset_j, 0)  for offset_j in knot_offsets]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from faceaudit.enums.hinge_anchor_mode import HingeAnchorMode
from faceaudit.models.calibration.logreg_s0_config import LogRegS0Config
from faceaudit.models.calibration.piecewise_linear_calibration_config import (
    PiecewiseLinearCalibrationConfig,
)


# Coeficiente do LogReg auxiliar abaixo deste limite trata-se como
# degenerado — o score não discrimina classes no fold. Provavelmente
# config ou dado quebrado.
_AUX_COEF_ZERO_EPSILON = 1e-6


class PiecewiseLinearScoreTransformer(BaseEstimator, TransformerMixin):
    """Transformer fold-aware que produz features piecewise-linear.

    Atributos aprendidos no `fit`:
    - `hinge_anchor_s0_`: float — âncora da basis function.
    - `knots_absolute_`: list[float] — knots absolutos (s0 + offset).
    - `aux_coef_`, `aux_intercept_`, `aux_C_`: diagnóstico da LogReg
      auxiliar (None em modo MANUAL).
    - `n_train_samples_`: número de samples no treino do fit.
    - `score_min_`, `score_max_`: range observado no treino (pra detectar
      extrapolação no transform).

    O LogReg auxiliar (`AUTO_LOGREG`) é re-treinado em cada `fit` —
    portanto cada fold do GridSearchCV tem seu próprio `s0`. Não há
    estado compartilhado entre folds.
    """

    def __init__(
        self,
        config: Optional[PiecewiseLinearCalibrationConfig] = None,
        s0_logreg_config: Optional[LogRegS0Config] = None,
    ):
        # Importante: sklearn BaseEstimator espera que __init__ apenas
        # atribua params, sem coerções. As coerções acontecem em fit.
        self.config = config
        self.s0_logreg_config = s0_logreg_config

    # ─────────────────────── helpers internos ───────────────────────

    def _effective_config(self) -> PiecewiseLinearCalibrationConfig:
        """Retorna o config a usar — default se não fornecido."""
        return self.config or PiecewiseLinearCalibrationConfig()

    def _effective_s0_config(self) -> LogRegS0Config:
        """Retorna o config do LogReg auxiliar — default se não fornecido."""
        return self.s0_logreg_config or LogRegS0Config()

    def _coerce_X(self, X) -> np.ndarray:
        """Aceita 1D ou 2D-com-1-coluna; retorna vetor 1D float64."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim == 2:
            if arr.shape[1] != 1:
                raise ValueError(
                    "PiecewiseLinearScoreTransformer aceita apenas "
                    f"score 1D na V1. Recebido shape={arr.shape}. "
                    "Para múltiplas features use SCORE_INTERACTION_FEATURE "
                    "externa e CALIBRATION_MODEL_TRAINING sem "
                    "piecewise interno."
                )
            return arr.reshape(-1)
        if arr.ndim != 1:
            raise ValueError(
                f"PiecewiseLinearScoreTransformer espera 1D ou (N,1); "
                f"recebido ndim={arr.ndim}."
            )
        return arr

    def _estimate_s0_auto(
        self,
        score: np.ndarray,
        y: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Estima `s0 = −intercept/coef` via LogReg auxiliar univariado.

        Retorna `(s0, coef, intercept, C)`. Levanta `ValueError` em
        casos degenerados (uma única classe, coef ≈ 0, coef < 0 — esse
        último indica direção do score invertida).
        """
        labels = np.asarray(y, dtype=np.int32)
        if labels.size == 0:
            raise ValueError(
                "PiecewiseLinearScoreTransformer.fit: y vazio — não dá "
                "pra estimar hinge_anchor_s0."
            )
        unique = np.unique(labels)
        if unique.size < 2:
            raise ValueError(
                f"AUTO_LOGREG: apenas uma classe presente (labels={list(unique)}). "
                "Para fit fold-aware, cada fold do treino precisa de gen+imp."
            )

        cfg = self._effective_s0_config()
        clf = LogisticRegression(
            C=cfg.C,
            solver=cfg.solver,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
        )
        X = score.reshape(-1, 1)
        clf.fit(X, labels)
        coef = float(clf.coef_[0, 0])
        intercept = float(clf.intercept_[0])

        if abs(coef) < _AUX_COEF_ZERO_EPSILON:
            raise ValueError(
                f"AUTO_LOGREG: calibrador auxiliar degenerou "
                f"(coef={coef:.3e} ≈ 0). O score não discrimina classes "
                "neste fold."
            )
        if coef < 0:
            raise ValueError(
                f"AUTO_LOGREG: coef negativo (coef={coef:.6g}). "
                "Direção do score invertida — maior score deveria "
                "corresponder a maior evidência Hp."
            )

        s0 = -intercept / coef
        return s0, coef, intercept, cfg.C

    # ─────────────────────── sklearn API ───────────────────────

    def fit(self, X, y=None):
        """Aprende `hinge_anchor_s0_` no treino do fold.

        Args:
            X: score 1D (ou matrix N×1).
            y: labels {0=Hd, 1=Hp}. Obrigatório em AUTO_LOGREG; ignorado
                em MANUAL.

        Returns:
            self.
        """
        config = self._effective_config()
        score = self._coerce_X(X)

        # Range do treino — usado em transform pra detectar extrapolação.
        self.score_min_ = float(score.min()) if score.size else float("nan")
        self.score_max_ = float(score.max()) if score.size else float("nan")
        self.n_train_samples_ = int(score.size)

        # Resolve s0.
        if config.hinge_anchor_mode is HingeAnchorMode.MANUAL:
            if config.hinge_anchor_manual_value is None:
                raise ValueError(
                    "hinge_anchor_mode=MANUAL requer "
                    "hinge_anchor_manual_value não nulo (já deveria ter "
                    "sido validado no DTO)."
                )
            self.hinge_anchor_s0_ = float(config.hinge_anchor_manual_value)
            self.aux_coef_ = None
            self.aux_intercept_ = None
            self.aux_C_ = None
            self.training_method_ = "manual"
        else:
            if y is None:
                raise ValueError(
                    "AUTO_LOGREG: `fit(X, y)` precisa de labels y. "
                    "Em pipeline sklearn, garanta que y é propagado."
                )
            s0, coef, intercept, C = self._estimate_s0_auto(score, y)
            self.hinge_anchor_s0_ = float(s0)
            self.aux_coef_ = float(coef)
            self.aux_intercept_ = float(intercept)
            self.aux_C_ = float(C)
            self.training_method_ = "auto_logreg"

        self.knots_absolute_ = config.knots_absolute(self.hinge_anchor_s0_)
        self.knot_offsets_ = list(config.knot_offsets)
        self.n_features_out_ = 1 + len(self.knot_offsets_)
        return self

    def transform(self, X) -> np.ndarray:
        """Aplica a expansão `[z, max(z − offset_j, 0)]` ao score.

        Returns:
            Matrix (N, 1 + n_knots) com `z = score − s0` na 1ª coluna e
            `max(z − offset_j, 0)` nas demais.
        """
        if not hasattr(self, "hinge_anchor_s0_"):
            raise RuntimeError(
                "PiecewiseLinearScoreTransformer.transform chamado antes "
                "do fit. Certifique-se de chamar fit(X, y) primeiro."
            )
        score = self._coerce_X(X)
        z = score - self.hinge_anchor_s0_

        columns = [z]
        for offset in self.knot_offsets_:
            columns.append(np.maximum(z - offset, 0.0))

        return np.column_stack(columns).astype(np.float64)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Nomes das features expandidas para inspeção/debug."""
        if not hasattr(self, "knot_offsets_"):
            raise RuntimeError("Chame fit antes de get_feature_names_out.")
        names = ["z=score-s0"]
        for offset in self.knot_offsets_:
            if offset == 0.0:
                names.append("max(z, 0)")
            elif offset > 0.0:
                names.append(f"max(z - {offset:+.4f}, 0)")
            else:
                names.append(f"max(z - ({offset:+.4f}), 0)")
        return np.array(names, dtype=object)
