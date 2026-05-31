# faceaudit/models/calibration/logistic_regression_calibration_model.py

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from faceaudit.enums.calibration_basis_mode import CalibrationBasisMode
from faceaudit.enums.monotonicity_policy import MonotonicityPolicy
from faceaudit.models.calibration.logistic_regression_calibration_model_config import (
    LogisticRegressionCalibrationModelConfig,
)
from faceaudit.models.calibration.piecewise_linear_score_transformer import (
    PiecewiseLinearScoreTransformer,
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

    MODEL_VERSION = 3  # +pipeline (piecewise-linear interno) na V3

    def __init__(
        self,
        config: LogisticRegressionCalibrationModelConfig | None = None,
        name: str = "Logistic Regression Calibration Model",
    ):
        self.config = config or LogisticRegressionCalibrationModelConfig()
        self._name = name
        self._is_fitted = False
        # Preenchido após fit() quando use_gridsearch=True. Útil para
        # diagnóstico (log + execution_metadata) e para reprodução.
        self.best_C_: float | None = None

        # Standardização de features (ativada via config.standardize_features
        # quando X tem shape (N, D) com D > 1 — caso típico de score
        # vectorization). μ/σ aprendidos no fit são persistidos pra que
        # calibrate_scores aplique a mesma transformação em validação e
        # casework. None quando standardização não foi usada.
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None

        # Bucket de telemetria leve. Preenchido pelo executor com
        # calibrator_cllr_train / _cv / cv_gap / cv_split_mode etc.
        self.metadata: Dict[str, Any] = {}

        # Pipeline sklearn quando `piecewise_linear_config.is_enabled`
        # — encapsula [PiecewiseLinearScoreTransformer → StandardScaler →
        # LogReg] e é fold-aware no GridSearchCV. Quando None, calibrador
        # opera no caminho legado direto sobre `self.logreg`.
        self._pipeline: Optional[Pipeline] = None

        self.logreg = self._build_estimator(self.config.C)

    def _build_estimator(self, c_value: float) -> LogisticRegression:
        return LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
        )

    # =========================================================================
    # PIECEWISE-LINEAR INTERNO (V1)
    # =========================================================================

    def _fit_piecewise(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        n_hp: int,
        n_hd: int,
        logger_callback: Callable[[str], None],
    ) -> None:
        """Fit fold-aware via Pipeline [PWL, StandardScaler, LogReg].

        Restrito a score 1D na V1: rejeita multivariado com erro claro
        (multivariado deve usar SCORE_INTERACTION_FEATURE externa).

        Estado pós-fit:
        - `self._pipeline`: Pipeline treinada
        - `self.logreg`: alias pra `self._pipeline.named_steps['logreg']`
          (backward-compat com diagnósticos que leem `self.logreg`)
        - `self._feature_mean`, `self._feature_std`: μ/σ das features
          EXPANDIDAS (não do score bruto) — preserva contrato de
          standardização da scaler.
        - `self.best_C_`: melhor C do GridSearch (se ativo).
        - `self.metadata`: anotações de diagnóstico (s0, knots, slopes,
          monotonicidade).
        """
        pwl_cfg = self.config.piecewise_linear_config

        if X.shape[1] != 1:
            raise ValueError(
                "piecewise_linear_config.mode=PIECEWISE_LINEAR só aceita "
                f"score 1D na V1. Recebido X com shape={X.shape}. "
                "Para multivariado use SCORE_INTERACTION_FEATURE externa "
                "+ CALIBRATION_MODEL_TRAINING sem piecewise interno."
            )

        # Pre-fit validation: a UI permite mudanças de campo via setattr
        # sem re-rodar __post_init__ (caso típico: usuário muda
        # `hinge_anchor_mode` de AUTO_LOGREG → MANUAL mas esquece de
        # preencher `hinge_anchor_manual_value`). Sem essa checagem o
        # erro só apareceria fundo no sklearn GridSearchCV ("All 25
        # fits failed"), ofuscando a causa real. Re-rodamos
        # `__post_init__` explicitamente pra surfacear o problema ANTES
        # de entrar no sklearn.
        try:
            pwl_cfg.__post_init__()
        except ValueError as exc:
            raise ValueError(
                f"Configuração piecewise-linear inválida no calibrador: "
                f"{exc}. Confira: mode={pwl_cfg.mode.to_str()}, "
                f"hinge_anchor_mode={pwl_cfg.hinge_anchor_mode.to_str()}, "
                f"hinge_anchor_manual_value={pwl_cfg.hinge_anchor_manual_value}, "
                f"knot_offsets={list(pwl_cfg.knot_offsets)}."
            ) from exc

        # Pipeline: o transformer aprende s0 fold-aware no GridSearchCV.
        score_1d = X.reshape(-1)

        def _make_pipeline(c_value: float) -> Pipeline:
            return Pipeline([
                ("pwl", PiecewiseLinearScoreTransformer(config=pwl_cfg)),
                ("scaler", StandardScaler()),
                ("logreg", self._build_estimator(c_value)),
            ])

        if self.config.use_gridsearch:
            min_class_size = min(n_hp, n_hd)
            cv_folds = max(2, min(self.config.cv_folds, min_class_size))
            if cv_folds < self.config.cv_folds:
                logger_callback(
                    f"[{self.name}] ⚠️ cv_folds reduzido de "
                    f"{self.config.cv_folds} para {cv_folds} — classe "
                    f"minoritária tem {min_class_size} amostras."
                )

            scoring_str = (
                self.config.scoring.value
                if hasattr(self.config.scoring, "value")
                else str(self.config.scoring)
            )

            from faceaudit.enums.knot_placement import KnotPlacement
            if pwl_cfg.knot_placement is KnotPlacement.AUTO_FROM_MEDIANS:
                knots_disp = "AUTO (medianas log-LR, fold-aware)"
            else:
                knots_disp = (
                    f"{pwl_cfg.knot_offsets} ({pwl_cfg.knot_scale.unit_label})"
                )
            logger_callback(
                f"[{self.name}] 🧠 GridSearch piecewise-linear "
                f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | "
                f"knots={knots_disp} | "
                f"c_grid={self.config.c_grid} | cv={cv_folds})..."
            )

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=_make_pipeline(self.config.C),
                param_grid={"logreg__C": list(self.config.c_grid)},
                cv=cv,
                scoring=scoring_str,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(score_1d, labels)
            self._pipeline = grid.best_estimator_
            self.best_C_ = float(
                grid.best_params_.get("logreg__C", self.config.C)
            )
            logger_callback(
                f"[{self.name}] ✅ Melhor C: {self.best_C_} "
                f"(score={grid.best_score_:.4f})."
            )
        else:
            from faceaudit.enums.knot_placement import KnotPlacement
            if pwl_cfg.knot_placement is KnotPlacement.AUTO_FROM_MEDIANS:
                knots_disp = "AUTO (medianas log-LR)"
            else:
                knots_disp = (
                    f"{pwl_cfg.knot_offsets} ({pwl_cfg.knot_scale.unit_label})"
                )
            logger_callback(
                f"[{self.name}] 🧠 Treinando piecewise-linear "
                f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | "
                f"knots={knots_disp} | "
                f"C={self.config.C})..."
            )
            pipe = _make_pipeline(self.config.C)
            pipe.fit(score_1d, labels)
            self._pipeline = pipe
            self.best_C_ = float(self.config.C)

        # Backward-compat: aponta self.logreg pra o LogReg interno da
        # pipeline (vários diagnósticos legados acessam `.logreg.coef_`).
        self.logreg = self._pipeline.named_steps["logreg"]

        # Standardização persistida pelo wrapper antigo NÃO se aplica aqui
        # — a pipeline tem sua própria StandardScaler. Mantemos None
        # explicitamente pra deixar claro qual caminho está ativo.
        self._feature_mean = None
        self._feature_std = None

        # Validação de monotonicidade na escala bruta.
        self._validate_monotonicity_and_persist_diagnostics(
            logger_callback=logger_callback,
            train_scores=score_1d,
            train_labels=labels,
        )

    def _validate_monotonicity_and_persist_diagnostics(
        self,
        logger_callback: Callable[[str], None],
        train_scores: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
    ) -> None:
        """Valida monotonicidade da curva piecewise-linear na escala bruta
        e persiste diagnósticos no `self.metadata`.

        Coeficientes raw são derivados desfazendo a standardização aplicada
        pela StandardScaler interna da pipeline.
        """
        pwl_cfg = self.config.piecewise_linear_config
        pwl_t: PiecewiseLinearScoreTransformer = (
            self._pipeline.named_steps["pwl"]
        )
        scaler: StandardScaler = self._pipeline.named_steps["scaler"]
        logreg: LogisticRegression = self._pipeline.named_steps["logreg"]

        # Coeficientes em escala STANDARDIZADA.
        beta_std = logreg.coef_[0].astype(np.float64)
        intercept_std = float(logreg.intercept_[0])

        # μ e σ usados na standardização das features expandidas.
        mu = scaler.mean_.astype(np.float64)
        sigma = scaler.scale_.astype(np.float64)

        # Coeficientes em escala BRUTA: β_raw_j = β_std_j / σ_j.
        beta_raw = beta_std / sigma
        # intercept_raw = b_std − Σ β_std_j · μ_j / σ_j
        intercept_raw = intercept_std - float(np.sum(beta_std * mu / sigma))

        # As features expandidas, em ordem, são:
        # [0]: z = score − s0
        # [j ≥ 1]: max(z − offset_{j−1}, 0)
        # Inclinação por trecho (escala bruta):
        # - Abaixo do menor offset: slope = β_z
        # - Depois de cada knot acumula β do hinge correspondente.
        offsets_sorted = list(pwl_t.knot_offsets_)  # já ordenado pelo DTO
        beta_z = float(beta_raw[0])
        beta_hinge_per_knot = [float(b) for b in beta_raw[1:]]

        segment_slopes = [beta_z]
        running = beta_z
        for b in beta_hinge_per_knot:
            running += b
            segment_slopes.append(running)

        # Monotonicidade: todas as inclinações devem ser > min_allowed_slope.
        is_monotonic = all(s > pwl_cfg.min_allowed_slope for s in segment_slopes)

        # Diagnóstico trecho-a-trecho com suporte estatístico (quantos
        # pares Hp/Hd caem em cada trecho). Necessário pra dar pra o
        # usuário pista de qual knot está em região com pouco suporte.
        knots_abs = list(pwl_t.knots_absolute_)  # já ordenado
        segment_bounds: List[tuple] = []  # [(low, high)] por trecho
        for j in range(len(segment_slopes)):
            low = knots_abs[j - 1] if j > 0 else float("-inf")
            high = knots_abs[j] if j < len(knots_abs) else float("+inf")
            segment_bounds.append((low, high))
        # Conta pares Hp/Hd em cada trecho (quando temos os dados de treino).
        segment_support: List[tuple] = []  # [(n_hp, n_hd)] por trecho
        if train_scores is not None and train_labels is not None:
            for low, high in segment_bounds:
                in_seg = (train_scores >= low) & (train_scores < high)
                n_hp = int(((train_labels == 1) & in_seg).sum())
                n_hd = int(((train_labels == 0) & in_seg).sum())
                segment_support.append((n_hp, n_hd))
        else:
            segment_support = [(None, None) for _ in segment_slopes]

        def _fmt_segment(j: int, slope: float) -> str:
            low, high = segment_bounds[j]
            n_hp, n_hd = segment_support[j]
            ok = slope > pwl_cfg.min_allowed_slope
            marker = "✓" if ok else "❌"
            # Bound display: -inf → "−∞", +inf → "+∞", senão valor.
            low_str = "−∞" if low == float("-inf") else f"{low:+.4g}"
            high_str = "+∞" if high == float("+inf") else f"{high:+.4g}"
            sup = ""
            if n_hp is not None:
                sup = f", suporte: {n_hp} Hp + {n_hd} Hd"
            return (
                f"    Trecho {j+1} (score ∈ [{low_str}, {high_str}){sup}): "
                f"slope = {slope:+.6f} {marker}"
            )

        if pwl_cfg.require_monotonicity and not is_monotonic:
            # Lista os trechos que violam, identifica os com menor suporte
            # (provável causa raiz).
            offenders = [
                (j, s) for j, s in enumerate(segment_slopes)
                if s <= pwl_cfg.min_allowed_slope
            ]
            offender_ids = [j + 1 for j, _ in offenders]
            seg_lines = [_fmt_segment(j, s) for j, s in enumerate(segment_slopes)]
            # Causa provável: trecho ofensor com menor suporte total
            cause_line = ""
            if segment_support[0][0] is not None:
                offenders_with_sup = [
                    (j + 1, segment_support[j][0] + segment_support[j][1])
                    for j, _ in offenders
                ]
                if offenders_with_sup:
                    j_worst, n_worst = min(offenders_with_sup, key=lambda x: x[1])
                    # Knots que delimitam o trecho j_worst (1-indexed).
                    # Trecho J em [knot[J-2], knot[J-1]) — exceto extremos.
                    neighbor_hints = []
                    user_offsets = list(pwl_t.knot_offsets_user_)
                    scale_unit = pwl_cfg.knot_scale.unit_label
                    if 1 < j_worst:
                        left_idx = j_worst - 2  # knot que fecha o trecho à esquerda
                        neighbor_hints.append(
                            f"knot_offsets[{left_idx}]={user_offsets[left_idx]:+.4g}"
                        )
                    if j_worst <= len(user_offsets):
                        right_idx = j_worst - 1
                        neighbor_hints.append(
                            f"knot_offsets[{right_idx}]={user_offsets[right_idx]:+.4g}"
                        )
                    neighbor_str = " ou ".join(neighbor_hints) if neighbor_hints else "knot vizinho"
                    cause_line = (
                        f"\n  Causa provável: trecho {j_worst} tem só "
                        f"{n_worst} pares — sobreajuste em região esparsa. "
                        f"Ajuste {neighbor_str} (escala {scale_unit}) "
                        f"pra mover a fronteira pra região com mais suporte "
                        f"OU aumente a regularização (c_grid menor)."
                    )
            # Apresenta os offsets na escala do usuário (LOG_LR ou RAW)
            # E nos absolutos em raw, pra ele identificar qual knot
            # ajustar no DTO.
            user_offsets_disp = [
                round(o, 4) for o in pwl_t.knot_offsets_user_
            ]
            raw_offsets_disp = [
                round(o, 4) for o in pwl_t.knot_offsets_raw_
            ]
            scale_unit = pwl_cfg.knot_scale.unit_label
            offsets_section = (
                f"  knot_offsets (escala do usuário, {scale_unit}) = "
                f"{user_offsets_disp}\n"
            )
            if pwl_cfg.knot_scale is KnotScale.LOG_LR:
                offsets_section += (
                    f"  knot_offsets (equivalente em score bruto) = "
                    f"{raw_offsets_disp}  [conversão via aux_coef="
                    f"{pwl_t.aux_coef_:.6g}]\n"
                )
            msg = (
                f"[{self.name}] curva piecewise-linear NÃO monotônica na "
                f"escala bruta.\n"
                + offsets_section +
                f"  knots_absolute (score bruto) = "
                f"{[round(k, 4) for k in knots_abs]}\n"
                f"  inclinações por trecho:\n" +
                "\n".join(seg_lines) +
                f"\n  min_allowed_slope = {pwl_cfg.min_allowed_slope}\n"
                f"  trechos ofensores (slope ≤ 0): {offender_ids}"
                + cause_line + "\n"
                "  Calibrador inválido forensicamente (score_maior → "
                "log_LR_menor em algum trecho)."
            )
            if pwl_cfg.monotonicity_policy is MonotonicityPolicy.ERROR:
                raise ValueError(msg)
            else:
                logger_callback(f"⚠️ WARNING: {msg}")

        # Cruzamento efetivo com log-LR = 0 — não é o mesmo que s0 quando
        # houver mais de 1 knot ou intercept não compensável.
        # Para 1 knot em z=0: logit(z) = β_z·z + β_max·max(z,0) + b_raw.
        #   z ≤ 0: logit = β_z·z + b_raw = 0 → z = −b_raw/β_z (se válido)
        #   z > 0: logit = (β_z+β_max)·z + b_raw = 0 → z = −b_raw/(β_z+β_max)
        # Pra múltiplos knots ficaria mais elaborado; reportamos o valor
        # do trecho à direita do último knot por padrão (cobre o caso típico).
        last_segment_slope = segment_slopes[-1]
        if abs(last_segment_slope) > pwl_cfg.min_allowed_slope:
            z_zero = -intercept_raw / last_segment_slope
            effective_zero_crossing = float(
                pwl_t.hinge_anchor_s0_ + z_zero
            )
        else:
            effective_zero_crossing = None

        # Persiste diagnósticos pra auditoria forense.
        diagnostics = {
            "calibration_basis_mode": pwl_cfg.mode.to_str(),
            "piecewise_enabled": True,
            "piecewise_knot_offsets": list(pwl_t.knot_offsets_),  # raw (compat)
            "piecewise_knot_offsets_user": list(pwl_t.knot_offsets_user_),
            "piecewise_knot_offsets_raw": list(pwl_t.knot_offsets_raw_),
            "piecewise_knot_scale": pwl_t.knot_scale_.to_str(),
            "piecewise_knot_placement": pwl_t.knot_placement_.to_str(),
            "piecewise_auto_log_lr_hp_p50": pwl_t.auto_log_lr_hp_p50_,
            "piecewise_auto_log_lr_hd_p50": pwl_t.auto_log_lr_hd_p50_,
            "piecewise_knots_absolute": list(pwl_t.knots_absolute_),
            "hinge_anchor_s0": float(pwl_t.hinge_anchor_s0_),
            "hinge_anchor_mode": pwl_cfg.hinge_anchor_mode.to_str(),
            "hinge_anchor_aux_coef": (
                None if pwl_t.aux_coef_ is None else float(pwl_t.aux_coef_)
            ),
            "hinge_anchor_aux_intercept": (
                None if pwl_t.aux_intercept_ is None
                else float(pwl_t.aux_intercept_)
            ),
            "hinge_anchor_aux_C": (
                None if pwl_t.aux_C_ is None else float(pwl_t.aux_C_)
            ),
            "hinge_anchor_training_method": pwl_t.training_method_,
            "hinge_anchor_inside_training_range": bool(
                pwl_t.score_min_ <= pwl_t.hinge_anchor_s0_ <= pwl_t.score_max_
            ),
            "calibrator_coef_standardized": [float(b) for b in beta_std],
            "calibrator_intercept_standardized": float(intercept_std),
            "calibrator_coef_raw": [float(b) for b in beta_raw],
            "calibrator_intercept_raw": float(intercept_raw),
            "segment_slopes_raw": [float(s) for s in segment_slopes],
            "is_monotonic": bool(is_monotonic),
            "min_allowed_slope": float(pwl_cfg.min_allowed_slope),
            "effective_log_lr_zero_crossing": effective_zero_crossing,
            "pipeline_feature_names": [
                str(n) for n in pwl_t.get_feature_names_out()
            ],
            "log_lr_computation_method": "decision_function",
            "positive_class_label": int(logreg.classes_[-1]),
        }
        self.metadata.update(diagnostics)

        logger_callback(
            f"[{self.name}] 📐 hinge_anchor_s0={pwl_t.hinge_anchor_s0_:.6g}  "
            f"slopes_raw={[round(s, 6) for s in segment_slopes]}  "
            f"monotônica={is_monotonic}  "
            f"effective_zero_crossing={effective_zero_crossing}"
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

        Suporta:
        - vetor 1D `(N,)` — calibração univariada de score → log-LR
        - matriz 2D `(N, D)` — calibração multivariada (alimentada por
          SCORE_VECTORIZATION): D features por amostra, LogReg aprende
          um peso por feature (Brümmer & du Preez 2006 score-level
          combination).
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        labels_arr = np.asarray(labels, dtype=np.int32)

        if scores_arr.ndim == 2:
            X = scores_arr   # (N, 1) ou (N, D) — sklearn aceita
        elif scores_arr.ndim == 1:
            X = scores_arr.reshape(-1, 1)
        else:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, D).")

        # Standardização opcional (só aplica quando D > 1 e config ativa).
        # Aprende μ, σ por feature no treino e persiste pra aplicar
        # exatamente os mesmos parâmetros em calibrate_scores (validação
        # / casework). Sem isso, a L2 do sklearn penaliza pesos
        # uniformemente, distorcendo features de escalas dispares.
        self._feature_mean = None
        self._feature_std = None
        if self.config.standardize_features and X.shape[1] > 1:
            self._feature_mean = X.mean(axis=0)
            self._feature_std = X.std(axis=0, ddof=1)
            # Defensivo: features constantes (std=0) viram divisão por
            # zero. Substitui 1.0 — feature constante não contribui pra
            # discriminação mesmo, então o peso aprendido pelo LogReg
            # vai ser próximo de zero.
            zero_std = self._feature_std == 0
            if np.any(zero_std):
                logger_callback(
                    f"[{self.name}] ⚠️ {int(zero_std.sum())} feature(s) "
                    "com std=0 no treino — substituindo por 1.0 (não "
                    "contribuirão pra discriminação)."
                )
                self._feature_std = np.where(zero_std, 1.0, self._feature_std)
            X = (X - self._feature_mean) / self._feature_std
            logger_callback(
                f"[{self.name}] 📐 Standardização ativa (D={X.shape[1]}). "
                f"μ={self._feature_mean.round(4).tolist()}, "
                f"σ={self._feature_std.round(4).tolist()}"
            )

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

        # Caminho piecewise-linear: 1D-only, sklearn Pipeline fold-aware.
        # Sai do método cedo após ajustar a pipeline e validar
        # monotonicidade — caminho legado (`self.logreg.fit(X)`) é pulado.
        if self.config.piecewise_linear_config.is_enabled:
            self._fit_piecewise(
                X=X,
                labels=labels_arr,
                n_hp=n_hp,
                n_hd=n_hd,
                logger_callback=logger_callback,
            )
            self._is_fitted = True
            logger_callback(
                f"[{self.name}] ✅ Treinamento concluído (piecewise-linear)."
            )
            return

        if self.config.use_gridsearch:
            # Garante que o número de folds não excede a classe minoritária —
            # StratifiedKFold exige pelo menos `cv_folds` amostras por classe.
            min_class_size = min(n_hp, n_hd)
            cv_folds = max(2, min(self.config.cv_folds, min_class_size))
            if cv_folds < self.config.cv_folds:
                logger_callback(
                    f"[{self.name}] ⚠️ cv_folds reduzido de "
                    f"{self.config.cv_folds} para {cv_folds} — classe "
                    f"minoritária tem {min_class_size} amostras."
                )

            # scoring é um LogRegGridScoringMetric enum — passar o .value
            # ("neg_log_loss" ou "roc_auc") para o sklearn.
            scoring_str = (
                self.config.scoring.value
                if hasattr(self.config.scoring, "value")
                else str(self.config.scoring)
            )

            logger_callback(
                f"[{self.name}] 🧠 GridSearch LogReg "
                f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | "
                f"c_grid={self.config.c_grid} | cv={cv_folds} | "
                f"scoring={scoring_str})..."
            )

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            grid = GridSearchCV(
                estimator=self._build_estimator(self.config.C),
                param_grid={"C": list(self.config.c_grid)},
                cv=cv,
                scoring=scoring_str,
                n_jobs=-1,
                refit=True,
            )
            grid.fit(X, labels_arr)

            best_estimator = grid.best_estimator_
            self.logreg = best_estimator
            self.best_C_ = float(grid.best_params_.get("C", self.config.C))

            logger_callback(
                f"[{self.name}] ✅ Melhor C: {self.best_C_} "
                f"(score={grid.best_score_:.4f})."
            )
        else:
            logger_callback(
                f"[{self.name}] 🧠 Treinando calibrador LogReg "
                f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | C={self.config.C})..."
            )
            self.logreg.fit(X, labels_arr)
            self.best_C_ = float(self.config.C)

        self._is_fitted = True
        logger_callback(f"[{self.name}] ✅ Treinamento concluído.")

    # =========================================================================
    # INFERÊNCIA
    # =========================================================================

    def _features_to_X(self, scores: np.ndarray) -> np.ndarray:
        """Converte `scores` em matrix X de features standardizadas.

        Helper compartilhado entre `predict_proba` e `calibrate_scores` —
        garante que ambos vejam os mesmos features pré-processados.

        Crítica: usa μ/σ do treino. NÃO recomputa nos scores novos —
        senão recalibra silenciosamente com distribuição diferente em
        validação/casework.

        Não usado no caminho `self._pipeline is not None` (piecewise),
        que tem sua própria pipeline encadeada de transformações.
        """
        self._ensure_fitted()

        scores_arr = np.asarray(scores, dtype=np.float64)

        if scores_arr.ndim == 2:
            X = scores_arr
        elif scores_arr.ndim == 1:
            X = scores_arr.reshape(-1, 1)
        else:
            raise ValueError("scores deve ser vetor 1D ou matriz (N, D).")

        if self._feature_mean is not None and self._feature_std is not None:
            if X.shape[1] != self._feature_mean.shape[0]:
                raise ValueError(
                    f"Dimensão das features mudou desde o treino: "
                    f"esperado D={self._feature_mean.shape[0]}, recebido "
                    f"D={X.shape[1]}. Re-treine o calibrador."
                )
            X = (X - self._feature_mean) / self._feature_std
        return X

    def _coerce_scalar_input(self, scores: np.ndarray) -> np.ndarray:
        """Converte input pra vetor 1D — usado no caminho piecewise.

        Aceita 1D direto ou 2D com 1 coluna. Rejeita multivariado com
        mensagem clara (a pipeline piecewise foi treinada em 1D).
        """
        scores_arr = np.asarray(scores, dtype=np.float64)
        if scores_arr.ndim == 1:
            return scores_arr
        if scores_arr.ndim == 2 and scores_arr.shape[1] == 1:
            return scores_arr.reshape(-1)
        raise ValueError(
            f"Calibrador piecewise espera score 1D na inferência. "
            f"Recebido shape={scores_arr.shape}. Use o caminho legado "
            "(piecewise_linear_config.mode=NONE) para input multivariado."
        )

    def predict_proba(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Retorna probabilidades calibradas para [Hd, Hp].
        """
        self._ensure_fitted()
        if self._pipeline is not None:
            # Piecewise: pipeline aplica PWL + scaler + LogReg em sequência.
            score_1d = self._coerce_scalar_input(scores)
            return np.asarray(
                self._pipeline.predict_proba(score_1d), dtype=np.float64
            )
        X = self._features_to_X(scores)
        probs = self.logreg.predict_proba(X)
        return np.asarray(probs, dtype=np.float64)

    def calibrate_scores(
        self,
        scores: np.ndarray,
    ) -> np.ndarray:
        """Retorna LLR calibrado a partir do logit bruto do LogReg.

        Bugfix: a implementação anterior usava `np.log(p_hp) − np.log(p_hd)`
        com `eps = 1e-15` em `np.clip`, o que criava um teto numérico em
        `±log(1/1e-15) ≈ ±34.5388`. Em features de alta amplitude (modelos
        com tipicidade), até 87% dos genuínos batiam no teto, mascarando
        a verdadeira escala do calibrador e tornando as métricas Cllr/ECE/
        slope agregadas sob log-LR clipados.

        `decision_function` retorna `X · coef.T + intercept` direto, que
        para regressão logística binária É exatamente o LLR (logit de
        P(Hp)). Sem clipping, sem aproximação numérica. A magnitude
        reportada agora reflete a extrapolação real do calibrador —
        cabe ao consumidor (relatório forense) impor limites por
        suporte empírico, não ao calibrador escondê-los.
        """
        self._ensure_fitted()
        if self._pipeline is not None:
            score_1d = self._coerce_scalar_input(scores)
            llr = self._pipeline.decision_function(score_1d)
            return np.asarray(llr, dtype=np.float64)
        X = self._features_to_X(scores)
        llr = self.logreg.decision_function(X)
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
            "best_C_": self.best_C_,
            # Standardização (None quando não foi usada).
            "feature_mean": self._feature_mean,
            "feature_std": self._feature_std,
            # Pipeline sklearn (PWL + scaler + LogReg) quando piecewise
            # ativo. None no caminho legado. Carrega TUDO necessário pra
            # inferência — transformer aprendido com s0 congelado, scaler
            # com μ/σ das features expandidas, LogReg final.
            "pipeline": self._pipeline,
            # Telemetria de CV / Cllr in-sample (vazio se executor não setou).
            "metadata": dict(self.metadata) if self.metadata else {},
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
        best_c = payload.get("best_C_")
        self.best_C_ = float(best_c) if best_c is not None else None
        # Standardização (pickles antigos sem essas keys retornam None,
        # equivalente a "não usou standardização" — comportamento legado).
        fm = payload.get("feature_mean")
        fs = payload.get("feature_std")
        self._feature_mean = np.asarray(fm, dtype=np.float64) if fm is not None else None
        self._feature_std = np.asarray(fs, dtype=np.float64) if fs is not None else None
        # Pipeline piecewise (None em pickles V2 — backward-compat).
        self._pipeline = payload.get("pipeline", None)
        md = payload.get("metadata") or {}
        self.metadata = dict(md) if isinstance(md, dict) else {}

    # =========================================================================
    # FACTORY DE FOLD (CV diagnóstico)
    # =========================================================================

    def _make_frozen_factory(self) -> Callable[[], "LogisticRegressionCalibrationModel"]:
        """Factory pra k-fold CV diagnóstico: clona o config com hyperparams
        congelados (C=best_C_, sem GridSearch) e devolve callable que cria
        instâncias novas. Standardização (μ/σ) é re-aprendida por fold
        dentro do fit — sem data leak entre folds."""
        self._ensure_fitted()
        frozen_config = copy.deepcopy(self.config)
        frozen_config.use_gridsearch = False
        frozen_config.C = float(self.best_C_) if self.best_C_ is not None else float(self.config.C)
        frozen_config.k_folds = 0  # impede recursão se algum caller olhar

        base_name = self._name

        def _make() -> "LogisticRegressionCalibrationModel":
            return LogisticRegressionCalibrationModel(
                config=copy.deepcopy(frozen_config),
                name=f"{base_name} [CV-fold]",
            )

        return _make

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _ensure_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Modelo LogisticRegressionCalibrationModel não treinado. Execute fit() primeiro."
            )