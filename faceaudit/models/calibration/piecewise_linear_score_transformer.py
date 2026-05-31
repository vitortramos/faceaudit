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
from faceaudit.enums.knot_placement import KnotPlacement
from faceaudit.enums.knot_scale import KnotScale
from faceaudit.models.calibration.logreg_s0_config import LogRegS0Config
from faceaudit.models.calibration.piecewise_linear_calibration_config import (
    PiecewiseLinearCalibrationConfig,
)


# AUTO_FROM_MEDIANS: 2 knots fixos em [-log_lr_Hp_P50, -log_lr_Hd_P50].
# A escolha entre 1 vs 2 knots não tem critério confiável só a partir
# das medianas (validado empiricamente em cosseno e PLDA na sessão de
# 2026-05). Em cosseno-like com forte assimetria entre as magnitudes
# das medianas, 1 knot em [0.0] manual pode ser marginalmente melhor
# — esses casos ficam pro modo MANUAL. AUTO entrega 2 knots como
# heurística "boa em todos os casos, ótima nos PLDA-like".


# Coeficiente do LogReg auxiliar abaixo deste limite trata-se como
# degenerado — o score não discrimina classes no fold. Provavelmente
# config ou dado quebrado.
_AUX_COEF_ZERO_EPSILON = 1e-6


class PiecewiseLinearScoreTransformer(BaseEstimator, TransformerMixin):
    """Transformer fold-aware que produz features piecewise-linear.

    Atributos aprendidos no `fit`:
    - `hinge_anchor_s0_`: float — âncora da basis function.
    - `knot_offsets_user_`: list[float] — offsets na unidade escolhida
      pelo usuário (`config.knot_scale`). É o que o usuário escreveu.
    - `knot_offsets_raw_`: list[float] — offsets convertidos para escala
      bruta (usados internamente em `transform` e em `knots_absolute_`).
      Em `KnotScale.RAW` coincide com `knot_offsets_user_`; em
      `KnotScale.LOG_LR` é `[log_lr_j / aux_coef_ for log_lr_j in user]`.
    - `knot_scale_`: KnotScale — eco da escala usada (auditável no pickle).
    - `knots_absolute_`: list[float] — knots absolutos em escala bruta
      (s0 + offset_raw). Sempre na escala do score.
    - `aux_coef_`, `aux_intercept_`, `aux_C_`: diagnóstico da LogReg
      auxiliar (None em modo MANUAL).
    - `n_train_samples_`: número de samples no treino do fit.
    - `score_min_`, `score_max_`: range observado no treino (pra detectar
      extrapolação no transform).

    O LogReg auxiliar (`AUTO_LOGREG`) é re-treinado em cada `fit` —
    portanto cada fold do GridSearchCV tem seu próprio `s0` e seu próprio
    `aux_coef`, o que implica que cada fold também tem sua própria
    conversão `log-LR → raw` quando `knot_scale=LOG_LR`. Sem leakage.
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

        # Modo AUTO_FROM_MEDIANS: o transformer popula `knot_offsets`
        # no próprio fit baseado nas medianas log-LR de cada classe.
        # Fold-aware porque acontece dentro deste fit (cada fold do
        # GridSearchCV chega aqui com seu próprio subset de treino).
        # Sempre LOG_LR scale (consequência natural da fórmula).
        self.knot_placement_ = config.knot_placement
        if config.knot_placement is KnotPlacement.AUTO_FROM_MEDIANS:
            if self.aux_coef_ is None or self.aux_coef_ <= 0:
                raise RuntimeError(
                    "knot_placement=AUTO_FROM_MEDIANS exige aux_coef do "
                    f"LogReg auxiliar > 0. Obtido aux_coef={self.aux_coef_!r}. "
                    "Provável erro de invariante — DTO deveria ter bloqueado."
                )
            if y is None:
                raise ValueError(
                    "AUTO_FROM_MEDIANS: `fit(X, y)` precisa de labels y."
                )
            labels = np.asarray(y, dtype=np.int32)
            score_hp = score[labels == 1]
            score_hd = score[labels == 0]
            if score_hp.size == 0 or score_hd.size == 0:
                raise ValueError(
                    "AUTO_FROM_MEDIANS: precisa de ambas as classes no "
                    f"fold (Hp={score_hp.size}, Hd={score_hd.size})."
                )
            # log-LR predito pelo LogReg auxiliar:
            #   log_lr_aux(score) = aux_coef · (score − s0)
            # Medianas log-LR por classe (no subset de treino do fold):
            log_lr_hp_p50 = float(
                self.aux_coef_ * (np.median(score_hp) - self.hinge_anchor_s0_)
            )
            log_lr_hd_p50 = float(
                self.aux_coef_ * (np.median(score_hd) - self.hinge_anchor_s0_)
            )
            self.auto_log_lr_hp_p50_ = log_lr_hp_p50
            self.auto_log_lr_hd_p50_ = log_lr_hd_p50
            # 2 knots fixos em [-log_lr_Hp_P50, -log_lr_Hd_P50],
            # ordenados. Em log-LR scale (sempre).
            # Colapso defensivo: se as duas medianas estão muito próximas
            # (caso degenerado de classes não separáveis no fold), colapsa
            # pra 1 knot na média — evita slope colinear nas duas hinges.
            auto_offsets_raw = sorted([-log_lr_hp_p50, -log_lr_hd_p50])
            if abs(auto_offsets_raw[1] - auto_offsets_raw[0]) < 1e-6:
                auto_offsets = [
                    (auto_offsets_raw[0] + auto_offsets_raw[1]) / 2.0
                ]
            else:
                auto_offsets = auto_offsets_raw
            # Override do DTO: substitui knot_offsets/knot_scale pelos
            # valores computados. Mantém o `config` original imutável.
            self.knot_offsets_user_ = list(auto_offsets)
            self.knot_scale_ = KnotScale.LOG_LR
        else:
            self.knot_offsets_user_ = list(config.knot_offsets)
            self.knot_scale_ = config.knot_scale
            self.auto_log_lr_hp_p50_ = None
            self.auto_log_lr_hd_p50_ = None

        # Resolve a escala dos offsets. `knot_offsets_user_` preserva
        # o que o usuário escreveu (na unidade `knot_scale_`).
        # `knot_offsets_raw_` é a versão sempre em escala bruta usada
        # internamente por transform/knots_absolute/monotonicity check.
        if self.knot_scale_ is KnotScale.LOG_LR:
            # Conversão exige aux_coef positivo do LogReg auxiliar.
            # Como `knot_scale=LOG_LR` é mutuamente exclusivo com MANUAL
            # (validado no DTO), `self.aux_coef_` está garantidamente
            # disponível e > 0 nesse ramo.
            if self.aux_coef_ is None or self.aux_coef_ <= 0:
                raise RuntimeError(
                    "knot_scale=LOG_LR requer aux_coef do LogReg "
                    f"auxiliar > 0. Obtido aux_coef={self.aux_coef_!r}. "
                    "Provável erro de invariante — DTO deveria ter "
                    "bloqueado essa combinação."
                )
            # Após dedup do DTO, knot_offsets_user_ está ordenado e
            # finito. Como aux_coef_ > 0, dividir por ele preserva
            # ordem — knot_offsets_raw_ também sai ordenado.
            self.knot_offsets_raw_ = [
                float(offset) / self.aux_coef_
                for offset in self.knot_offsets_user_
            ]
        else:
            self.knot_offsets_raw_ = list(self.knot_offsets_user_)

        # Backward-compat: `knot_offsets_` historicamente apontava para
        # os offsets já em escala bruta (consumidores externos esperam
        # essa semântica). Mantém o alias.
        self.knot_offsets_ = list(self.knot_offsets_raw_)
        self.knots_absolute_ = [
            self.hinge_anchor_s0_ + off for off in self.knot_offsets_raw_
        ]
        self.n_features_out_ = 1 + len(self.knot_offsets_raw_)
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
        for offset in self.knot_offsets_raw_:
            columns.append(np.maximum(z - offset, 0.0))

        return np.column_stack(columns).astype(np.float64)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Nomes das features expandidas para inspeção/debug."""
        if not hasattr(self, "knot_offsets_raw_"):
            raise RuntimeError("Chame fit antes de get_feature_names_out.")
        names = ["z=score-s0"]
        for offset in self.knot_offsets_raw_:
            if offset == 0.0:
                names.append("max(z, 0)")
            elif offset > 0.0:
                names.append(f"max(z - {offset:+.4f}, 0)")
            else:
                names.append(f"max(z - ({offset:+.4f}), 0)")
        return np.array(names, dtype=object)
