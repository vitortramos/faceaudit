# faceaudit/models/calibration/logistic_regression_calibration_model.py

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from faceaudit.enums.calibration_basis_mode import CalibrationBasisMode
from faceaudit.enums.knot_scale import KnotScale
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

            # Resolve scoring: NEG_CLLR_BALANCED vira o callable customizado
            # (balanceado por classe, equivalente a -Cllr); demais viram
            # string nativa do sklearn.
            scoring_obj = self._resolve_scoring(self.config.scoring)
            scoring_label = (
                self.config.scoring.value
                if hasattr(self.config.scoring, "value")
                else str(self.config.scoring)
            )

            from faceaudit.enums.knot_placement import KnotPlacement
            if pwl_cfg.knot_placement is KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS:
                knots_disp = "AUTO (medianas log-LR, fold-aware)"
            else:
                knots_disp = (
                    f"{pwl_cfg.knot_offsets} ({pwl_cfg.knot_scale.unit_label})"
                )
            logger_callback(
                f"[{self.name}] 🧠 GridSearch piecewise-linear "
                f"(N={X.shape[0]} | Hp={n_hp} | Hd={n_hd} | "
                f"knots={knots_disp} | "
                f"c_grid={self.config.c_grid} | cv={cv_folds} | "
                f"scoring={scoring_label})..."
            )

            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

            # Seleção de C sob restrição de monotonicidade.
            # Antes (V2.4): GridSearchCV escolhia por CV score; só depois
            # o fit final podia ser rejeitado por não-monotonicidade.
            # Agora (V2.5): pra cada C ordenado por CV score, faz fit
            # completo e checa monotonicidade. O primeiro C cuja curva é
            # monotônica é o escolhido. Persiste lista completa de
            # candidatos válidos e rejeitados pra auditoria.
            selection = self._select_monotonic_C(
                score_1d=score_1d,
                labels=labels,
                cv=cv,
                c_grid=list(self.config.c_grid),
                scoring=scoring_obj,
                pwl_cfg=pwl_cfg,
                make_pipeline=_make_pipeline,
                logger_callback=logger_callback,
            )
            self._pipeline = selection["selected_pipeline"]
            self.best_C_ = selection["selected_C"]
            # Persiste decisões da seleção pra auditoria (acessível antes
            # do `_validate_monotonicity_and_persist_diagnostics` rodar).
            self.metadata["piecewise_valid_c_values"] = selection["valid_C_values"]
            self.metadata["piecewise_valid_c_cv_scores"] = (
                selection["valid_C_cv_scores"]
            )
            self.metadata["piecewise_invalid_c_values"] = selection["invalid_C_values"]
            self.metadata["piecewise_invalid_c_reasons"] = (
                selection["invalid_C_reasons"]
            )
            self.metadata["piecewise_selected_c"] = selection["selected_C"]
            self.metadata["piecewise_selection_policy"] = (
                "best_valid_candidate_by_cv_score"
            )
            # Função objetivo do CV — auditável no laudo.
            from faceaudit.enums.logreg_grid_scoring_metric import LogRegGridScoringMetric
            self.metadata["piecewise_cv_scoring"] = scoring_label
            if self.config.scoring is LogRegGridScoringMetric.NEG_CLLR_BALANCED:
                self.metadata["piecewise_cv_scoring_log_base"] = "bits"
                self.metadata["piecewise_cv_scoring_class_aggregation"] = (
                    "equal_class_weight"
                )
            else:
                self.metadata["piecewise_cv_scoring_log_base"] = "natural"
                self.metadata["piecewise_cv_scoring_class_aggregation"] = (
                    "per_sample"
                )
            # Contrato de reprodutibilidade do CV.
            self.metadata["cv_splitter_type"] = "StratifiedKFold"
            self.metadata["cv_splitter_random_seed"] = 42
            self.metadata["cv_fold_count"] = int(cv_folds)

            logger_callback(
                f"[{self.name}] ✅ Melhor C válido (monotônico): "
                f"{self.best_C_} (CV score={selection['cv_score_of_selected']:.4f}). "
                f"Candidatos válidos: {len(selection['valid_C_values'])}; "
                f"rejeitados por não-monotonicidade: "
                f"{len(selection['invalid_C_values'])}."
            )

            # CV diagnóstico: refit em cada fold do cv com C selecionado
            # pra coletar s0/aux_coef/knots por fold. Persiste estabilidade
            # da heurística (especialmente útil para AUTO_FROM_MIRRORED_MEDIANS).
            try:
                cv_diag = self._compute_cv_fold_diagnostics(
                    score_1d=score_1d,
                    labels=labels,
                    cv=cv,
                    c_value=self.best_C_,
                    make_pipeline=_make_pipeline,
                )
                self.metadata.update(cv_diag)
            except Exception as exc:
                logger_callback(
                    f"[{self.name}] ⚠️ Falha ao computar diagnósticos CV "
                    f"({type(exc).__name__}: {exc})."
                )
        else:
            from faceaudit.enums.knot_placement import KnotPlacement
            if pwl_cfg.knot_placement is KnotPlacement.AUTO_FROM_MIRRORED_MEDIANS:
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

    # ─────────────────────── helpers internos ───────────────────────

    @staticmethod
    def _neg_cllr_balanced_scorer(estimator, X, y) -> float:
        """Scorer sklearn-compatible que retorna `-Cllr` balanceado por classe.

        Diferente do `neg_log_loss` do sklearn (média por amostra), esta
        métrica calcula a média da log-loss em base 2 por classe e agrega
        com peso igual entre Hp (y=1) e Hd (y=0):

            Cllr = 0.5 · [ mean_{i ∈ Hp} log2(1 + exp(-ℓ_i))
                          + mean_{j ∈ Hd} log2(1 + exp(+ℓ_j)) ]

        onde ℓ_k = decision_function(x_k). Equivalente ao Cllr de
        Brümmer & du Preez (2006) calculado a partir do estimador
        (não precisa de `predict_proba` + clipping).

        Em datasets fortemente desbalanceados (N_Hd >> N_Hp), o
        `neg_log_loss` padrão fica dominado pela classe majoritária e
        pode selecionar um C ótimo para impostores médios mas mal
        calibrado para os genuínos difíceis. Esta versão balanceada
        evita esse viés. Retorna `-Cllr` (scorer maximiza).
        """
        llr = np.asarray(estimator.decision_function(X), dtype=np.float64)
        y_arr = np.asarray(y)
        hp = llr[y_arr == 1]
        hd = llr[y_arr == 0]
        if hp.size == 0 or hd.size == 0:
            # Fold sem ambas as classes — retorna pior pontuação possível
            # pra que esse C não seja escolhido.
            return float("-inf")
        # log-loss em base 2; `np.logaddexp(0, x) = log(1 + exp(x))` é
        # numericamente estável pra valores grandes.
        cllr_hp = float(np.mean(np.logaddexp(0.0, -hp))) / np.log(2.0)
        cllr_hd = float(np.mean(np.logaddexp(0.0, hd))) / np.log(2.0)
        cllr = 0.5 * (cllr_hp + cllr_hd)
        return -cllr

    @classmethod
    def _resolve_scoring(cls, scoring_enum):
        """Converte o enum `LogRegGridScoringMetric` num objeto aceito
        pelo `GridSearchCV.scoring` (string ou callable).

        Para `NEG_CLLR_BALANCED`, retorna a função `_neg_cllr_balanced_scorer`
        diretamente — o GridSearchCV aceita callables com assinatura
        `(estimator, X, y) -> float`. Para as demais, retorna a string
        nativa do sklearn.
        """
        from faceaudit.enums.logreg_grid_scoring_metric import LogRegGridScoringMetric
        if scoring_enum is LogRegGridScoringMetric.NEG_CLLR_BALANCED:
            return cls._neg_cllr_balanced_scorer
        if hasattr(scoring_enum, "value"):
            return scoring_enum.value
        return str(scoring_enum)

    @staticmethod
    def _compute_pipeline_slopes(
        pipeline: Pipeline,
    ) -> Dict[str, Any]:
        """Extrai slopes da pipeline já treinada (sem persistir, sem raise).

        Retorna dict com:
        - `beta_raw`: coeficientes em escala bruta (np.ndarray)
        - `intercept_raw`: intercepto em escala bruta (float)
        - `segment_slopes`: lista de inclinações por trecho (List[float])
        - `is_monotonic_strict`: bool — todos os slopes > 1e-8
        - `offending_indices`: lista de índices (0-based) com slope ≤ 1e-8
        - `knot_offsets_raw`: lista de offsets em escala bruta (do transformer)
        - `pwl_t`: referência ao PiecewiseLinearScoreTransformer

        Usado por (a) `_fit_piecewise` no fluxo GridSearch pra filtrar
        candidatos não-monotônicos durante a seleção de C, e (b)
        `_validate_monotonicity_and_persist_diagnostics` pra persistir
        os mesmos valores no fit final.
        """
        pwl_t = pipeline.named_steps["pwl"]
        scaler = pipeline.named_steps["scaler"]
        logreg = pipeline.named_steps["logreg"]

        beta_std = logreg.coef_[0].astype(np.float64)
        intercept_std = float(logreg.intercept_[0])
        mu = scaler.mean_.astype(np.float64)
        sigma = scaler.scale_.astype(np.float64)
        beta_raw = beta_std / sigma
        intercept_raw = intercept_std - float(np.sum(beta_std * mu / sigma))

        offsets_sorted = list(pwl_t.knot_offsets_)
        beta_z = float(beta_raw[0])
        beta_hinge_per_knot = [float(b) for b in beta_raw[1:]]

        segment_slopes: List[float] = [beta_z]
        running = beta_z
        for b in beta_hinge_per_knot:
            running += b
            segment_slopes.append(running)

        _MIN_SLOPE = 1e-8
        offending_indices = [
            j for j, s in enumerate(segment_slopes) if s <= _MIN_SLOPE
        ]
        return {
            "beta_raw": beta_raw,
            "intercept_raw": intercept_raw,
            "segment_slopes": segment_slopes,
            "is_monotonic_strict": len(offending_indices) == 0,
            "offending_indices": offending_indices,
            "knot_offsets_raw": offsets_sorted,
            "pwl_t": pwl_t,
        }

    def _select_monotonic_C(
        self,
        score_1d: np.ndarray,
        labels: np.ndarray,
        cv: StratifiedKFold,
        c_grid: List[float],
        scoring,
        pwl_cfg: Any,
        make_pipeline: Callable[[float], Pipeline],
        logger_callback: Callable[[str], None],
    ) -> Dict[str, Any]:
        """Varredura de C com filtro de monotonicidade (política Opção A).

        Para cada C em `c_grid`, faz fit completo do pipeline e checa
        monotonicidade. **Avalia todos** os candidatos (não para no
        primeiro válido) pra que o metadata persista a lista completa
        de aprovações e rejeições — útil pra auditoria. O candidato
        selecionado é o primeiro válido na ordenação por CV score
        decrescente.

        `scoring` aceita string sklearn ("neg_log_loss", "roc_auc") ou
        callable `(estimator, X, y) -> float` (caso do scorer
        `_neg_cllr_balanced_scorer` — recomendado para calibração
        forense em datasets desbalanceados, ver §9.3 do doc).

        Retorna dict com:
        - `selected_pipeline`: Pipeline já treinada (best valid candidate)
        - `selected_C`: float — C escolhido
        - `valid_C_values`: List[float]
        - `invalid_C_values`: List[float]
        - `invalid_C_reasons`: List[str] (paralela a invalid_C_values)
        - `cv_score_of_selected`: float
        """
        # Primeira fase: GridSearchCV(refit=False) pra obter scores CV
        grid = GridSearchCV(
            estimator=make_pipeline(c_grid[0]),
            param_grid={"logreg__C": list(c_grid)},
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=False,
        )
        grid.fit(score_1d, labels)
        cv_scores = grid.cv_results_["mean_test_score"]
        c_values_from_grid = [
            float(p["logreg__C"]) for p in grid.cv_results_["params"]
        ]
        # Ordena C por CV score (decrescente — maior score = melhor)
        ranked = sorted(
            range(len(c_values_from_grid)),
            key=lambda i: -float(cv_scores[i]),
        )

        # Segunda fase: avalia TODOS os candidatos (Opção A do veredito)
        # e seleciona o primeiro válido por ranking. Custo: c_grid fits
        # adicionais sobre o pipeline (grids tipicamente ≤ 6 elementos —
        # custo aceitável em troca de auditoria completa).
        valid_C: List[float] = []
        valid_C_cv_scores: List[float] = []
        invalid_C: List[float] = []
        invalid_C_reasons: List[str] = []
        selected_pipeline: Optional[Pipeline] = None
        selected_C: Optional[float] = None
        selected_cv_score: Optional[float] = None

        for idx in ranked:
            c_val = c_values_from_grid[idx]
            cv_score = float(cv_scores[idx])
            pipe = make_pipeline(c_val)
            try:
                pipe.fit(score_1d, labels)
            except Exception as exc:
                invalid_C.append(c_val)
                invalid_C_reasons.append(
                    f"fit final falhou: {type(exc).__name__}: {exc}"
                )
                continue
            slope_info = self._compute_pipeline_slopes(pipe)
            if slope_info["is_monotonic_strict"]:
                valid_C.append(c_val)
                valid_C_cv_scores.append(cv_score)
                if selected_pipeline is None:
                    selected_pipeline = pipe
                    selected_C = c_val
                    selected_cv_score = cv_score
            else:
                offending = [j + 1 for j in slope_info["offending_indices"]]
                slopes_disp = [
                    round(s, 6) for s in slope_info["segment_slopes"]
                ]
                invalid_C.append(c_val)
                invalid_C_reasons.append(
                    f"slopes={slopes_disp}, trechos não-monotônicos "
                    f"(slope ≤ 1e-8): {offending}"
                )

        if selected_pipeline is None:
            # Nenhum candidato válido — erro detalhado com a lista
            # de todos os C testados e seus motivos.
            lines = [
                f"[{self.name}] nenhum candidato no c_grid produziu "
                f"curva piecewise-linear monotônica.",
                f"  c_grid testado: {list(c_grid)}",
                f"  candidatos rejeitados:",
            ]
            for c_val, reason in zip(invalid_C, invalid_C_reasons):
                lines.append(f"    C={c_val}: {reason}")
            lines.append(
                "  Recomendação: reduzir c_grid pra valores menores "
                "(regularização mais forte), reposicionar knots manualmente "
                "pra regiões com contraste por classe, OU usar "
                "knot_offsets=[0.0] (uma única dobra em s0)."
            )
            msg = "\n".join(lines)
            if pwl_cfg.monotonicity_policy is MonotonicityPolicy.ERROR:
                raise ValueError(msg)
            else:
                logger_callback(f"⚠️ WARNING: {msg}")
                # Em WARNING, devolve o pipeline com melhor CV score
                # mesmo não-monotônico — mantém comportamento legado.
                best_idx = ranked[0]
                c_val = c_values_from_grid[best_idx]
                selected_pipeline = make_pipeline(c_val)
                selected_pipeline.fit(score_1d, labels)
                selected_C = c_val
                selected_cv_score = float(cv_scores[best_idx])

        return {
            "selected_pipeline": selected_pipeline,
            "selected_C": float(selected_C),
            "valid_C_values": valid_C,
            "valid_C_cv_scores": valid_C_cv_scores,
            "invalid_C_values": invalid_C,
            "invalid_C_reasons": invalid_C_reasons,
            "cv_score_of_selected": float(selected_cv_score),
        }

    def _compute_cv_fold_diagnostics(
        self,
        score_1d: np.ndarray,
        labels: np.ndarray,
        cv: StratifiedKFold,
        c_value: float,
        make_pipeline: Callable[[float], Pipeline],
    ) -> Dict[str, Any]:
        """Roda cv.split com C selecionado e coleta s0/aux_coef/knots/slopes
        por fold pra avaliar estabilidade da heurística (especialmente
        relevante para `AUTO_FROM_MIRRORED_MEDIANS`).

        Custo: `cv_folds` fits adicionais sobre o pipeline. Aceitável
        porque já há cv_folds fits no GridSearch — duplica esse custo.
        """
        s0_per_fold: List[float] = []
        aux_coef_per_fold: List[Optional[float]] = []
        knots_absolute_per_fold: List[List[float]] = []
        segment_slopes_per_fold: List[List[float]] = []

        for train_idx, _ in cv.split(score_1d, labels):
            try:
                pipe = make_pipeline(c_value)
                pipe.fit(score_1d[train_idx], labels[train_idx])
                pwl_t = pipe.named_steps["pwl"]
                s0_per_fold.append(float(pwl_t.hinge_anchor_s0_))
                aux_coef_per_fold.append(
                    None if pwl_t.aux_coef_ is None else float(pwl_t.aux_coef_)
                )
                knots_absolute_per_fold.append(list(pwl_t.knots_absolute_))
                slope_info = self._compute_pipeline_slopes(pipe)
                segment_slopes_per_fold.append(
                    [float(s) for s in slope_info["segment_slopes"]]
                )
            except Exception:
                # Se algum fold falhar, registra placeholder None.
                s0_per_fold.append(float("nan"))
                aux_coef_per_fold.append(None)
                knots_absolute_per_fold.append([])
                segment_slopes_per_fold.append([])

        def _array_stats(values: List[float]):
            arr = np.array(values, dtype=np.float64)
            mask = np.isfinite(arr)
            if not mask.any():
                return None, None
            return float(arr[mask].mean()), float(arr[mask].std(ddof=0))

        s0_mean, s0_std = _array_stats(s0_per_fold)
        aux_coef_finite = [v for v in aux_coef_per_fold if v is not None]
        ac_mean, ac_std = _array_stats(aux_coef_finite) if aux_coef_finite else (None, None)

        # Estatísticas por knot (média e std de cada knot ao longo dos folds).
        n_knots_per_fold = [len(k) for k in knots_absolute_per_fold]
        knot_means: List[Optional[float]] = []
        knot_stds: List[Optional[float]] = []
        if n_knots_per_fold and all(n == n_knots_per_fold[0] for n in n_knots_per_fold) and n_knots_per_fold[0] > 0:
            n = n_knots_per_fold[0]
            for j in range(n):
                col = [k[j] for k in knots_absolute_per_fold]
                m, s = _array_stats(col)
                knot_means.append(m)
                knot_stds.append(s)

        # Estatísticas por segmento (média e std de cada slope ao longo dos folds).
        n_segs_per_fold = [len(s) for s in segment_slopes_per_fold]
        seg_means: List[Optional[float]] = []
        seg_stds: List[Optional[float]] = []
        if n_segs_per_fold and all(n == n_segs_per_fold[0] for n in n_segs_per_fold) and n_segs_per_fold[0] > 0:
            n = n_segs_per_fold[0]
            for r in range(n):
                col = [s[r] for s in segment_slopes_per_fold]
                m, s = _array_stats(col)
                seg_means.append(m)
                seg_stds.append(s)

        return {
            "cv_hinge_anchor_s0_values": s0_per_fold,
            "cv_s0_mean": s0_mean,
            "cv_s0_std": s0_std,
            "cv_aux_coef_values": aux_coef_per_fold,
            "cv_aux_coef_mean": ac_mean,
            "cv_aux_coef_std": ac_std,
            "cv_piecewise_knots_absolute_values": knots_absolute_per_fold,
            "cv_knot_means": knot_means,
            "cv_knot_stds": knot_stds,
            "cv_segment_slopes_raw_values": segment_slopes_per_fold,
            "cv_segment_slopes_mean": seg_means,
            "cv_segment_slopes_std": seg_stds,
        }

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

        # Contagem de identidades únicas por trecho (quando o executor
        # forneceu probe/reference identities via self.metadata antes do
        # fit). Pares são correlacionados por identidade — mesma face pode
        # aparecer em vários pares. Identidades únicas é uma medida de
        # suporte estatístico mais informativa que contagem de pares.
        train_probe_ids = self.metadata.get("train_probe_identities")
        train_reference_ids = self.metadata.get("train_reference_identities")
        segment_unique_identities_hp: List[Optional[int]] = []
        segment_unique_identities_hd: List[Optional[int]] = []
        if (
            train_scores is not None
            and train_labels is not None
            and train_probe_ids is not None
            and train_reference_ids is not None
            and len(train_probe_ids) == len(train_scores)
            and len(train_reference_ids) == len(train_scores)
        ):
            probe_arr = np.asarray(train_probe_ids)
            ref_arr = np.asarray(train_reference_ids)
            for low, high in segment_bounds:
                in_seg = (train_scores >= low) & (train_scores < high)
                hp_mask = in_seg & (train_labels == 1)
                hd_mask = in_seg & (train_labels == 0)
                # Hp: probe_id == reference_id; só uma identidade por par.
                hp_ids = set(probe_arr[hp_mask].tolist())
                # Hd: probe ≠ reference; conta união das duas identidades.
                hd_ids = (
                    set(probe_arr[hd_mask].tolist())
                    | set(ref_arr[hd_mask].tolist())
                )
                segment_unique_identities_hp.append(len(hp_ids))
                segment_unique_identities_hd.append(len(hd_ids))
        else:
            segment_unique_identities_hp = [None for _ in segment_slopes]
            segment_unique_identities_hd = [None for _ in segment_slopes]

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
            if pwl_cfg.knot_scale is KnotScale.AUX_LOG_LR:
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

        # Cruzamento efetivo com log-LR = 0 — algoritmo geral por segmento.
        #
        # Decomposição da função no segmento r (todas as hinges j com δ_j < z
        # estão ativas; chamamos esse conjunto de A_r):
        #
        #   f_r(z) = c_r + m_r · z
        #
        # onde:
        #   m_r = β_z + Σ_{j ∈ A_r} β_j                 (inclinação do segmento)
        #   c_r = β_0_raw − Σ_{j ∈ A_r} β_j · δ_j       (intercepto efetivo)
        #
        # Raiz candidata em z: z_r = −c_r / m_r. Aceita se cai dentro do
        # intervalo do segmento [δ_{r-1}, δ_r) (extremos vão pra ±∞).
        # Sob crescimento estrito (m_r > 0 em todos os segmentos), existe
        # no máximo uma raiz válida.
        #
        # offsets_sorted é a lista de δ_j na escala bruta (já ordenada).
        # beta_hinge_per_knot tem β_j (índices alinhados com offsets_sorted).
        # segment_slopes tem m_r (já calculado acima).
        effective_zero_crossing: Optional[float] = None
        effective_zero_crossing_segment_index: Optional[int] = None
        effective_zero_crossing_inside_training_range: Optional[bool] = None

        # Soma cumulativa de β_j · δ_j conforme entram as hinges.
        cum_beta_delta = 0.0
        for r, m_r in enumerate(segment_slopes):
            # c_r usa as hinges ativas no segmento r (todas com índice < r).
            c_r = intercept_raw - cum_beta_delta
            if abs(m_r) > pwl_cfg.min_allowed_slope:
                z_r = -c_r / m_r
                # Intervalo do segmento em z. Os offsets já estão em escala
                # bruta (offsets_sorted vem de pwl_t.knot_offsets_).
                z_lo = offsets_sorted[r - 1] if r > 0 else float("-inf")
                z_hi = offsets_sorted[r] if r < len(offsets_sorted) else float("+inf")
                if z_lo <= z_r < z_hi:
                    effective_zero_crossing = float(
                        pwl_t.hinge_anchor_s0_ + z_r
                    )
                    effective_zero_crossing_segment_index = r
                    s0_eff = effective_zero_crossing
                    effective_zero_crossing_inside_training_range = bool(
                        pwl_t.score_min_ <= s0_eff <= pwl_t.score_max_
                    )
                    break
            # Avança a cumulativa: ao passar pro segmento r+1, a hinge δ_r
            # entra em A (se houver knot nessa posição).
            if r < len(offsets_sorted):
                cum_beta_delta += beta_hinge_per_knot[r] * offsets_sorted[r]

        # === Diagnósticos adicionais por trecho e auditoria forense ===
        #
        # Suporte de Hp/Hd já foi calculado em segment_support (acima);
        # extraímos contagens totais e fração da amostra de treino que
        # cai em cada trecho. Necessário pra avaliar se uma inclinação
        # extrema num trecho está apoiada por amostra suficiente.
        segment_counts_total: List[Optional[int]] = []
        segment_counts_hp: List[Optional[int]] = []
        segment_counts_hd: List[Optional[int]] = []
        segment_fractions_total: List[Optional[float]] = []
        segment_support_warnings: List[str] = []
        n_train_total = pwl_t.n_train_samples_
        # Limite empírico abaixo do qual a inclinação do trecho é
        # considerada estatisticamente frágil. 30 amostras totais ou
        # 5 de cada classe. Configurável no DTO no futuro se necessário.
        _MIN_TOTAL_SUPPORT = 30
        _MIN_PER_CLASS_SUPPORT = 5
        for j, (n_hp, n_hd) in enumerate(segment_support):
            if n_hp is None or n_hd is None:
                segment_counts_total.append(None)
                segment_counts_hp.append(None)
                segment_counts_hd.append(None)
                segment_fractions_total.append(None)
                continue
            total = n_hp + n_hd
            segment_counts_total.append(total)
            segment_counts_hp.append(n_hp)
            segment_counts_hd.append(n_hd)
            segment_fractions_total.append(
                total / n_train_total if n_train_total > 0 else None
            )
            if total < _MIN_TOTAL_SUPPORT:
                segment_support_warnings.append(
                    f"trecho {j + 1}: suporte total ({total}) abaixo "
                    f"de {_MIN_TOTAL_SUPPORT} — inclinação pouco "
                    f"confiável."
                )
            elif n_hp < _MIN_PER_CLASS_SUPPORT or n_hd < _MIN_PER_CLASS_SUPPORT:
                segment_support_warnings.append(
                    f"trecho {j + 1}: cobertura por classe assimétrica "
                    f"(Hp={n_hp}, Hd={n_hd}); inclinação pode estar "
                    f"dominada por uma classe."
                )

        # Diagnóstico por knot: posição absoluta cai dentro do range de
        # treino? Fração da amostra acima de cada knot? Útil pra detectar
        # knots inertes (fora do range = features colineares ou nulas).
        knot_inside_training_range: List[bool] = []
        fraction_scores_above_each_knot: List[Optional[float]] = []
        for k_abs in knots_abs:
            knot_inside_training_range.append(
                bool(pwl_t.score_min_ <= k_abs <= pwl_t.score_max_)
            )
            if train_scores is not None and n_train_total > 0:
                frac = float((train_scores > k_abs).sum()) / n_train_total
                fraction_scores_above_each_knot.append(frac)
            else:
                fraction_scores_above_each_knot.append(None)

        # Range de score e de log-LR observados no treino (escala bruta).
        # Permite documentar até onde a curva foi estimada vs onde está
        # extrapolando em validação ou casework.
        if train_scores is not None and n_train_total > 0:
            raw_score_min_training = float(train_scores.min())
            raw_score_max_training = float(train_scores.max())
            # log-LR estimado pelo próprio calibrador piecewise nos extremos
            # do score de treino (apenas referência — extrapolação dos
            # segmentos lineares).
            train_z = train_scores - pwl_t.hinge_anchor_s0_
            train_log_lr = (
                beta_z * train_z + intercept_raw
                + np.sum(
                    [
                        b * np.maximum(train_z - off, 0.0)
                        for b, off in zip(beta_hinge_per_knot, offsets_sorted)
                    ],
                    axis=0,
                ) if beta_hinge_per_knot else (beta_z * train_z + intercept_raw)
            )
            raw_log_lr_min_training = float(train_log_lr.min())
            raw_log_lr_max_training = float(train_log_lr.max())
        else:
            raw_score_min_training = None
            raw_score_max_training = None
            raw_log_lr_min_training = None
            raw_log_lr_max_training = None

        # Prior efetivo no treino. Com class_weight="balanced", o sklearn
        # pondera cada amostra por n_total / (n_classes · n_classe_dela),
        # resultando em soma de pesos igual entre Hp e Hd → prior efetivo
        # de 0.5 para cada classe. Nesse caso, a decision_function do LogReg
        # retorna log-odds idêntico ao log-LR (correção de prior = 0). Esse
        # campo deixa explícito que estamos usando essa propriedade.
        cw = self.logreg.class_weight
        if cw == "balanced":
            n_pos = int(np.sum(np.asarray(self.logreg.classes_) == 1))  # eficiência irrelevante
            # class_weight="balanced" sempre gera pesos que igualam soma por classe.
            effective_training_prior_hp = 0.5
            effective_training_prior_hd = 0.5
            prior_log_odds_correction = 0.0
        else:
            # Outros class_weight: prior efetivo precisa de aproximação ou
            # documentação extra. Aqui marcamos como None pra forçar revisão
            # antes de tratar decision_function como log-LR.
            effective_training_prior_hp = None
            effective_training_prior_hd = None
            prior_log_odds_correction = None

        # Persiste diagnósticos pra auditoria forense.
        diagnostics = {
            # === Config geral ===
            "calibration_basis_mode": pwl_cfg.mode.to_str(),
            "piecewise_enabled": True,
            "knot_placement_protocol_version": (
                # Nome semântico atual; "auto_from_medians_v1" continua
                # aceito como tag legada em pickles V2.0-V2.5.
                "auto_from_mirrored_medians_v1"
                if pwl_t.knot_placement_.to_str() == "AUTO_FROM_MEDIANS"
                else "manual_v1"
            ),

            # === Knots ===
            "piecewise_knot_offsets": list(pwl_t.knot_offsets_),  # raw (compat)
            "piecewise_knot_offsets_user": list(pwl_t.knot_offsets_user_),
            "piecewise_knot_offsets_raw": list(pwl_t.knot_offsets_raw_),
            "piecewise_knot_scale": pwl_t.knot_scale_.to_str(),
            "piecewise_knot_placement": pwl_t.knot_placement_.to_str(),
            "piecewise_auto_log_lr_hp_p50": pwl_t.auto_log_lr_hp_p50_,
            "piecewise_auto_log_lr_hd_p50": pwl_t.auto_log_lr_hd_p50_,
            "piecewise_knots_absolute": list(pwl_t.knots_absolute_),
            "knot_inside_training_range": knot_inside_training_range,
            "fraction_scores_above_each_knot": fraction_scores_above_each_knot,

            # === Âncora s0 ===
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

            # === Coeficientes ===
            "calibrator_coef_standardized": [float(b) for b in beta_std],
            "calibrator_intercept_standardized": float(intercept_std),
            "calibrator_coef_raw": [float(b) for b in beta_raw],
            "calibrator_intercept_raw": float(intercept_raw),

            # === Slopes e monotonicidade ===
            "segment_slopes_raw": [float(s) for s in segment_slopes],
            "is_monotonic": bool(is_monotonic),
            "min_allowed_slope": float(pwl_cfg.min_allowed_slope),
            "monotonicity_check": "strict_increasing_post_fit",

            # === Suporte por segmento ===
            "segment_counts_total": segment_counts_total,
            "segment_counts_hp": segment_counts_hp,
            "segment_counts_hd": segment_counts_hd,
            "segment_fractions_total": segment_fractions_total,
            "segment_support_warnings": segment_support_warnings,
            # Identidades únicas por trecho — None se executor não passou
            # train_probe_identities/train_reference_identities via self.metadata.
            "segment_unique_identities_hp": segment_unique_identities_hp,
            "segment_unique_identities_hd": segment_unique_identities_hd,

            # === Ponto de cruzamento efetivo de log-LR = 0 ===
            "effective_log_lr_zero_crossing": effective_zero_crossing,
            "effective_log_lr_zero_crossing_segment_index": (
                effective_zero_crossing_segment_index
            ),
            "effective_log_lr_zero_crossing_inside_training_range": (
                effective_zero_crossing_inside_training_range
            ),

            # === Range observado no treino ===
            "raw_score_min_training": raw_score_min_training,
            "raw_score_max_training": raw_score_max_training,
            "raw_log_lr_min_training": raw_log_lr_min_training,
            "raw_log_lr_max_training": raw_log_lr_max_training,

            # === Prior e método de cálculo ===
            "log_lr_base": "natural",
            "log_lr_computation_method": "decision_function",
            "positive_class_label": int(logreg.classes_[-1]),
            "effective_training_prior_hp": effective_training_prior_hp,
            "effective_training_prior_hd": effective_training_prior_hd,
            "prior_log_odds_correction": prior_log_odds_correction,

            # === Auxiliar ===
            "pipeline_feature_names": [
                str(n) for n in pwl_t.get_feature_names_out()
            ],
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