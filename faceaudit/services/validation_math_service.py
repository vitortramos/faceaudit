import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics import roc_curve, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from faceaudit.models.validation_metrics import ValidationMetrics
from faceaudit.models.validation_plot_data import ValidationPlotData
from faceaudit.models.strategy_validation_result import StrategyValidationResult


_LN2 = float(np.log(2.0))


class ValidationMathService:
    """
    Serviço matemático puro para cálculo de métricas biométricas e forenses.

    Esta classe opera como um motor agnóstico. Ela não sabe se os dados vieram
    de um modelo PLDA, KDE ou de um pipeline de Deep Learning. Sua única
    responsabilidade é receber arrays de scores (tratados como Log-Likelihood Ratios
    na base neperiana 'e') e extrair métricas de performance rigorosas exigidas
    pela ENFSI (European Network of Forensic Science Institutes).
    """

    def calculate_metrics(self,
                          strategy_id: str,
                          strategy_name: str,
                          strategy_type: str,
                          genuine_scores: List[float],
                          impostor_scores: List[float],
                          genuine_raw_scores: List[float] = None,
                          impostor_raw_scores: List[float] = None,
                          genuine_score_vectors: Optional[np.ndarray] = None,
                          impostor_score_vectors: Optional[np.ndarray] = None,
                          score_vector_axis_labels: Optional[List[str]] = None,
                          calibrator_coef_raw: Optional[List[float]] = None,
                          calibrator_intercept_raw: Optional[float] = None,
                          piecewise_enabled: bool = False,
                          hinge_anchor_s0: Optional[float] = None,
                          piecewise_knots_absolute: Optional[List[float]] = None,
                          calibrator_cllr_train: float = None,
                          calibrator_cllr_train_gen: float = None,
                          calibrator_cllr_train_imp: float = None,
                          calibrator_cllr_cv: float = None,
                          calibrator_cllr_cv_gen: float = None,
                          calibrator_cllr_cv_imp: float = None) -> StrategyValidationResult:
        """
        Orquestra o cálculo de todas as métricas de validação e gera os vetores
        necessários para a plotagem dos gráficos de desempenho.

        Os parâmetros `calibrator_cllr_*` são opcionais e são apenas propagados
        do dataset.metadata (computados em CALIBRATION_MODEL_TRAINING) para o
        ValidationMetrics. Não há cálculo novo aqui — só transporte para a UI.
        """
        if not genuine_scores or not impostor_scores:
            raise ValueError("As listas de scores genuínos e impostores não podem estar vazias.")

        gen_arr = np.array(genuine_scores, dtype=np.float64)
        imp_arr = np.array(impostor_scores, dtype=np.float64)

        # 1. ROC / DET
        y_true = np.concatenate([np.ones_like(gen_arr), np.zeros_like(imp_arr)])
        y_scores = np.concatenate([gen_arr, imp_arr])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        roc_auc = float(auc(fpr, tpr))

        # 2. EER
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer_value = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        eer_thresh = float(thresholds[eer_idx])

        # 3. Cllr decomposto: total + per-hypothesis
        cllr_value, cllr_gen, cllr_imp = self._calculate_cllr_detailed(gen_arr, imp_arr)

        # 4. PAVA → Cllr_min + Cllr_min per-hypothesis
        gen_cal, imp_cal = self._get_calibrated_llrs(gen_arr, imp_arr)
        min_cllr_value, cllr_min_gen, cllr_min_imp = self._calculate_cllr_detailed(gen_cal, imp_cal)

        # 5. ECE (vetores + métricas escalares numéricas)
        ece_x, ece_y, ece_min_y = self._calculate_ece_data(gen_arr, imp_arr, gen_cal, imp_cal)
        ece_loss_area, ece_max_gap, ece_argmax_prior = self._compute_ece_metrics(
            ece_x, ece_y, ece_min_y,
        )

        # 6. Wrong-sign rates (Tier 1 forense)
        wrong_sign_gen_rate = float(np.mean(gen_arr < 0.0))
        wrong_sign_imp_rate = float(np.mean(imp_arr > 0.0))

        # 7. Affine decomposition (Brümmer-Doddington)
        cllr_affine, calib_slope, calib_intercept = self._compute_affine_calibration(
            gen_arr, imp_arr,
        )

        # Raw scores opcionais (necessários pro grid de curva de calibração)
        gen_raw_arr = (
            np.asarray(genuine_raw_scores, dtype=np.float64)
            if genuine_raw_scores is not None else np.array([])
        )
        imp_raw_arr = (
            np.asarray(impostor_raw_scores, dtype=np.float64)
            if impostor_raw_scores is not None else np.array([])
        )

        # 7b. Grid pré-computado da curva de calibração.
        # D = 1 (input escalar): plano raw_score → log-LR, mesma lógica
        #   tradicional (calibrador 1D LogReg/KDE/BiGauss em UM score).
        # D = 2 (SCORE_VECTORIZATION com 2 sistemas): superfície log-LR
        #   no plano (x₀, x₁) — heatmap + scatter + contornos.
        # D ≥ 3: visualização suprimida (raw_score → log-LR é enganoso;
        #   placeholder textual aponta pro Diagnóstico Afim).
        # slope/intercept (do passo 7) operam sobre o log-LR atual,
        # qualquer que seja D.
        cal_dim = self._infer_cal_curve_dim(
            genuine_score_vectors, impostor_score_vectors,
        )
        if cal_dim == 2:
            cal_curve = self._compute_calibration_curve_2d_grid(
                genuine_score_vectors, impostor_score_vectors,
                gen_arr, imp_arr,
                calibrator_coef_raw=calibrator_coef_raw,
                calibrator_intercept_raw=calibrator_intercept_raw,
            )
            cal_curve["cal_curve_dim"] = 2
            cal_curve["cal_curve_axis_labels"] = list(
                score_vector_axis_labels or ["x₀", "x₁"]
            )[:2]
            cal_curve["cal_curve_2d_feature_names"] = list(
                cal_curve["cal_curve_axis_labels"]
            )
        elif cal_dim >= 3:
            cal_curve = self._empty_calibration_curve_grid()
            cal_curve["cal_curve_dim"] = cal_dim
            cal_curve["cal_curve_axis_labels"] = list(
                score_vector_axis_labels or []
            )
        else:
            cal_curve = self._compute_calibration_curve_grid(
                gen_raw_arr, imp_raw_arr, gen_arr, imp_arr,
                calib_slope, calib_intercept,
                piecewise_enabled=piecewise_enabled,
                hinge_anchor_s0=hinge_anchor_s0,
                piecewise_knots_absolute=piecewise_knots_absolute,
                calibrator_coef_raw=calibrator_coef_raw,
                calibrator_intercept_raw=calibrator_intercept_raw,
            )
            cal_curve["cal_curve_dim"] = 1
            cal_curve["cal_curve_axis_labels"] = []

        # 8. Tail / outliers (Tier 5)
        tail = self._compute_tail_metrics(gen_arr, imp_arr)

        # Montagem do DTO de Métricas Escalares
        metrics = ValidationMetrics(
            eer=eer_value,
            eer_threshold=eer_thresh,
            cllr=cllr_value,
            cllr_min=min_cllr_value,
            auc=roc_auc,
            wrong_sign_gen_rate=wrong_sign_gen_rate,
            wrong_sign_imp_rate=wrong_sign_imp_rate,
            cllr_gen=cllr_gen,
            cllr_imp=cllr_imp,
            cllr_min_gen=cllr_min_gen,
            cllr_min_imp=cllr_min_imp,
            cllr_affine=cllr_affine,
            calibration_slope=calib_slope,
            calibration_intercept=calib_intercept,
            ece_calibration_loss_area=ece_loss_area,
            ece_max_gap=ece_max_gap,
            ece_argmax_prior_log_odds=ece_argmax_prior,
            loss_gen_q95=tail["loss_gen_q95"],
            loss_imp_q95=tail["loss_imp_q95"],
            max_loss_gen=tail["max_loss_gen"],
            max_loss_imp=tail["max_loss_imp"],
            top_1pct_loss_share=tail["top_1pct_loss_share"],
            top_5pct_loss_share=tail["top_5pct_loss_share"],
            n_top_1pct=tail["n_top_1pct"],
            # Telemetria CV do calibrador — transporte do metadata
            # do CALIBRATED_SCORES, sem cálculo aqui.
            calibrator_cllr_train=calibrator_cllr_train,
            calibrator_cllr_train_gen=calibrator_cllr_train_gen,
            calibrator_cllr_train_imp=calibrator_cllr_train_imp,
            calibrator_cllr_cv=calibrator_cllr_cv,
            calibrator_cllr_cv_gen=calibrator_cllr_cv_gen,
            calibrator_cllr_cv_imp=calibrator_cllr_cv_imp,
        )

        plot_data = ValidationPlotData(
            genuine_log_lrs=gen_arr,
            impostor_log_lrs=imp_arr,
            fpr_array=fpr,
            fnr_array=fnr,
            ece_prior_log_odds=ece_x,
            ece_values=ece_y,
            ece_min_values=ece_min_y,
            genuine_raw_scores=gen_raw_arr,
            impostor_raw_scores=imp_raw_arr,
            **cal_curve,
        )

        return StrategyValidationResult(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            n_genuines=len(gen_arr),
            n_impostors=len(imp_arr),
            metrics=metrics,
            plot_data=plot_data,
        )

    # =========================================================================
    # CLLR (TOTAL + PER-HYPOTHESIS)
    # =========================================================================

    def _calculate_cllr(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> float:
        """
        Cllr empírico em bits. Mantido pra retrocompat com callers externos
        (ex: BiGaussianizedCalibrationModel). Implementado em termos de
        `_calculate_cllr_detailed`.
        """
        cllr, _, _ = self._calculate_cllr_detailed(gen_scores, imp_scores)
        return cllr

    def _calculate_cllr_detailed(
        self,
        gen_scores: np.ndarray,
        imp_scores: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Calcula Cllr e seus termos per-hypothesis.

        Cllr = 0.5 * (cllr_gen + cllr_imp), onde:
            cllr_gen = mean over gen de log2(1 + e^(-llr))
            cllr_imp = mean over imp de log2(1 + e^(+llr))

        Retorna (cllr, cllr_gen, cllr_imp), todos em bits.
        """
        term_gen = float(np.mean(np.logaddexp(0, -gen_scores)) / _LN2)
        term_imp = float(np.mean(np.logaddexp(0, imp_scores)) / _LN2)
        cllr = 0.5 * (term_gen + term_imp)
        return cllr, term_gen, term_imp

    def _calculate_min_cllr(self, gen_scores: np.ndarray, imp_scores: np.ndarray) -> float:
        """Mantido pra retrocompat externa. Use diretamente os fields novos de
        `ValidationMetrics` quando possível."""
        gen_cal_llrs, imp_cal_llrs = self._get_calibrated_llrs(gen_scores, imp_scores)
        return self._calculate_cllr(gen_cal_llrs, imp_cal_llrs)

    # =========================================================================
    # PAVA CALIBRATION (Cllr_min)
    # =========================================================================

    def _get_calibrated_llrs(
        self,
        gen_scores: np.ndarray,
        imp_scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PAVA → log-LRs otimamente calibrados (isotônica), com correção
        de prior odds empírico."""
        y_true = np.concatenate([np.ones_like(gen_scores), np.zeros_like(imp_scores)])
        y_scores = np.concatenate([gen_scores, imp_scores])

        ir = IsotonicRegression(out_of_bounds='clip')
        p_calibrated = ir.fit_transform(y_scores, y_true)

        epsilon = 1e-14
        p_calibrated = np.clip(p_calibrated, epsilon, 1 - epsilon)

        prior_log_odds = np.log(len(gen_scores) / len(imp_scores))
        calibrated_llrs = np.log(p_calibrated / (1 - p_calibrated)) - prior_log_odds

        return calibrated_llrs[:len(gen_scores)], calibrated_llrs[len(gen_scores):]

    # =========================================================================
    # AFFINE CALIBRATION (Brümmer-Doddington)
    # =========================================================================

    def _compute_affine_calibration(
        self,
        gen_scores: np.ndarray,
        imp_scores: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Ajusta a recalibração affine ótima: log-LR_recal = a · log-LR + b.

        Os parâmetros (a, b) são estimados via regressão logística 1D com
        class_weight='balanced' (equivale a prior 0.5 — convenção Brümmer).
        Diagnóstico de calibração:
          - a ≈ 1, b ≈ 0  → sistema bem calibrado affinely
          - a > 1         → LRs over-confident (precisam ser puxados pra 0)
          - a < 1         → LRs under-confident
          - b ≠ 0         → viés sistemático

        Retorna (cllr_affine, slope a, intercept b).

        Se o fit falhar (ex: separação perfeita), retorna (cllr_original, 1, 0)
        — comportamento neutro.
        """
        try:
            x = np.concatenate([gen_scores, imp_scores]).reshape(-1, 1)
            y = np.concatenate([
                np.ones(len(gen_scores), dtype=np.int32),
                np.zeros(len(imp_scores), dtype=np.int32),
            ])
            lr = LogisticRegression(
                solver='lbfgs',
                class_weight='balanced',
                max_iter=1000,
                C=1e6,  # mínima regularização — queremos o MLE puro
            )
            lr.fit(x, y)
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
            # Aplica affine e calcula Cllr resultante.
            gen_aff = slope * gen_scores + intercept
            imp_aff = slope * imp_scores + intercept
            cllr_affine, _, _ = self._calculate_cllr_detailed(gen_aff, imp_aff)
            return cllr_affine, slope, intercept
        except Exception:
            # Fallback neutro — não bloqueia o cálculo das outras métricas.
            cllr_original, _, _ = self._calculate_cllr_detailed(gen_scores, imp_scores)
            return cllr_original, 1.0, 0.0

    # =========================================================================
    # CALIBRATION CURVE GRID (pós-afim Brümmer + Δ no plano raw × logLR)
    # =========================================================================

    @staticmethod
    def _infer_cal_curve_dim(
        gen_vectors: Optional[np.ndarray],
        imp_vectors: Optional[np.ndarray],
    ) -> int:
        """Inferre dimensionalidade do calibrador a partir dos vetores
        de score por classe. None ou shape (N,) → 1. (N,D) → D.

        Quando uma classe tem D=K e a outra D≠K, considera 1 (fallback
        defensivo — não devemos cair em 2D heatmap com classes
        inconsistentes).
        """
        def _d(arr):
            if arr is None:
                return None
            a = np.asarray(arr)
            if a.ndim == 1 or a.size == 0:
                return 1
            return int(a.shape[1])
        dg = _d(gen_vectors)
        di = _d(imp_vectors)
        if dg is None and di is None:
            return 1
        if dg is None:
            return di
        if di is None:
            return dg
        if dg != di:
            return 1
        return dg

    @staticmethod
    def _empty_calibration_curve_grid() -> dict:
        """Placeholder pra todos os campos cal_curve_* quando o plot 1D
        não se aplica (D ≥ 3 ou sem dados). O draw cai em texto
        explicativo."""
        return {
            "cal_curve_grid_x": np.array([]),
            "cal_curve_llr_y": np.array([]),
            "cal_curve_affine_y": np.array([]),
            "cal_curve_affine_delta_y": np.array([]),
            "cal_curve_zero_raw_score": None,
            "cal_curve_rug_gen_x": np.array([]),
            "cal_curve_rug_imp_x": np.array([]),
            "cal_curve_2d_grid_x0": np.array([]),
            "cal_curve_2d_grid_x1": np.array([]),
            "cal_curve_2d_grid_log_lr": np.array([]),
            "cal_curve_2d_gen_x0": np.array([]),
            "cal_curve_2d_gen_x1": np.array([]),
            "cal_curve_2d_imp_x0": np.array([]),
            "cal_curve_2d_imp_x1": np.array([]),
            "cal_curve_2d_coef_raw": np.array([]),
            "cal_curve_2d_coef_std": np.array([]),
            "cal_curve_2d_intercept": None,
            "cal_curve_2d_feature_names": [],
            "cal_curve_2d_density_grid": np.array([]),
            "cal_curve_2d_envelope_levels": np.array([]),
            "cal_curve_2d_density_grid_gen": np.array([]),
            "cal_curve_2d_density_grid_imp": np.array([]),
            "cal_curve_2d_envelope_level_gen": None,
            "cal_curve_2d_envelope_level_imp": None,
        }

    def _compute_calibration_curve_2d_grid(
        self,
        gen_vectors: np.ndarray,
        imp_vectors: np.ndarray,
        gen_log_lr: np.ndarray,
        imp_log_lr: np.ndarray,
        *,
        calibrator_coef_raw: Optional[List[float]] = None,
        calibrator_intercept_raw: Optional[float] = None,
        n_grid: int = 100,
        scatter_cap_gen: int = 5000,
        scatter_cap_imp: int = 2000,
        kde_fit_cap: int = 10000,
    ) -> dict:
        """Pré-computa heatmap de log-LR no plano (x₀, x₁) pra calibrador
        2D (típico: SCORE_VECTORIZATION combinando 2 sistemas) + envelope
        de suporte dos dados + coefs raw/std do calibrador.

        Superfície log-LR:
          - Se `calibrator_coef_raw` + `calibrator_intercept_raw` dados
            (caso LogReg 2D): avaliação ANALÍTICA do grid via
            `intercept + β·x` — fronteira logLR=0 vira reta perfeita.
          - Senão (KDE/BiGauss 2D): fallback de interpolação Delaunay
            via `LinearNDInterpolator` dos observados (x₀, x₁, log_lr).
            Pontos fora do casco convexo viram NaN, masked no plot.

        Envelope de suporte (sempre computado):
          - `gaussian_kde` em (gen ∪ imp), subamostrado pra `kde_fit_cap`
            pontos (Scott's rule pra bandwidth).
          - Densidade avaliada no mesmo grid 100×100. Normalizada pelo max
            pra colorbar log-LR não interferir.
          - `envelope_levels` = (p5, p1) da densidade nos pontos observados
            — contornos nesses níveis encerram 95% e 99% da massa observada
            respectivamente. Outer (95%) é envelope "principal", inner
            (99%) é envelope "permissivo".

        Coefs padronizados:
          - β_std[i] = β_raw[i] · σ(x_i_observado). Mudança esperada no
            log-LR por 1 desvio-padrão da feature. Resposta direta a
            "qual sistema domina?". None quando β_raw não disponível.

        Scatter cap diferenciado: 5000 genuínos (geralmente classe menor),
        2000 impostores (saturariam visualmente acima disso).

        Returns:
            dict completo com chaves de `_empty_calibration_curve_grid`.
        """
        from scipy.interpolate import LinearNDInterpolator
        from scipy.stats import gaussian_kde

        out = self._empty_calibration_curve_grid()

        gen_vectors = np.asarray(gen_vectors, dtype=np.float64)
        imp_vectors = np.asarray(imp_vectors, dtype=np.float64)
        if gen_vectors.ndim != 2 or imp_vectors.ndim != 2:
            return out
        if gen_vectors.shape[1] != 2 or imp_vectors.shape[1] != 2:
            return out
        if gen_vectors.shape[0] != gen_log_lr.shape[0]:
            return out
        if imp_vectors.shape[0] != imp_log_lr.shape[0]:
            return out

        all_x = np.vstack([gen_vectors, imp_vectors])

        # Bounds do grid: span observado com padding 5%.
        x0_lo, x0_hi = float(all_x[:, 0].min()), float(all_x[:, 0].max())
        x1_lo, x1_hi = float(all_x[:, 1].min()), float(all_x[:, 1].max())
        x0_pad = (x0_hi - x0_lo) * 0.05 if x0_hi > x0_lo else 1.0
        x1_pad = (x1_hi - x1_lo) * 0.05 if x1_hi > x1_lo else 1.0

        x0_grid = np.linspace(x0_lo - x0_pad, x0_hi + x0_pad, n_grid)
        x1_grid = np.linspace(x1_lo - x1_pad, x1_hi + x1_pad, n_grid)
        xx, yy = np.meshgrid(x0_grid, x1_grid)

        # ─── Superfície log-LR: ANALÍTICA se coefs dados, senão interp ────
        coef_raw_arr = np.array([])
        intercept_val: Optional[float] = None
        used_analytical = False
        if calibrator_coef_raw is not None and calibrator_intercept_raw is not None:
            try:
                coef_arr = np.asarray(calibrator_coef_raw, dtype=np.float64).ravel()
                if coef_arr.shape[0] == 2:
                    intercept_val = float(calibrator_intercept_raw)
                    grid_log_lr = intercept_val + coef_arr[0] * xx + coef_arr[1] * yy
                    coef_raw_arr = coef_arr
                    used_analytical = True
            except Exception:
                # Cai no fallback de interpolação.
                pass

        if not used_analytical:
            all_y = np.concatenate([gen_log_lr, imp_log_lr])
            try:
                interp = LinearNDInterpolator(all_x, all_y, fill_value=np.nan)
                grid_log_lr = interp(xx, yy)
            except Exception:
                # Defensivo: pontos colineares quebram Delaunay → cai em
                # 1D fallback (grid vazio sinaliza pro draw).
                return out

        # Coefs padronizados: β_std[i] = β_raw[i] · σ(x_i observado).
        # Sempre computamos σ — usado pra normalizar mesmo quando coef_raw
        # não disponível (fica em [] e a equação no plot não aparece).
        feature_sigma = np.array([
            float(np.std(all_x[:, 0], ddof=1)),
            float(np.std(all_x[:, 1], ddof=1)),
        ])
        coef_std_arr = np.array([])
        if coef_raw_arr.size == 2 and np.all(feature_sigma > 0):
            coef_std_arr = coef_raw_arr * feature_sigma

        # ─── Density envelopes PER-CLASSE via KDE 2D ─────────────────────
        # Decisão de design: 2 KDEs separadas (uma por classe) em vez
        # de 1 KDE na união gen+imp. Motivo: datasets forenses são
        # tipicamente muito desbalanceados (80k impostores × 1.5k
        # genuínos, ratio ~50:1). Numa KDE pooled, a classe minoritária
        # vira ruído sob a maior; o envelope 95% mass-based ignora os
        # genuínos completamente — não é o que o analista quer ver.
        #
        # Per-classe: cada envelope encerra 95% da massa DA PRÓPRIA
        # classe. Plot renderiza 2 contours coloridos (verde+vermelho).
        # Fora dos dois = extrapolação genuína. Dentro de um só = suporte
        # parcial.
        grid_pts = np.vstack([xx.ravel(), yy.ravel()])
        rng_kde = np.random.default_rng(42)

        def _class_envelope(class_x: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
            """KDE 2D + nível 95% mass-based pra uma classe. Devolve
            (density_grid normalizada, threshold) ou (empty, None) se
            falhar (covariância singular, < 5 pontos, etc)."""
            n = class_x.shape[0]
            if n < 5:
                return np.array([]), None
            try:
                if n > kde_fit_cap:
                    idx = rng_kde.choice(n, size=kde_fit_cap, replace=False)
                    pts = class_x[idx].T
                else:
                    pts = class_x.T
                kde = gaussian_kde(pts)  # Scott's rule auto
                density = kde(grid_pts).reshape(xx.shape)
                d_max = float(density.max())
                if d_max > 0:
                    density = density / d_max
                dens_at_obs = kde(pts) / (d_max if d_max > 0 else 1.0)
                # Percentil 5 da densidade nos observados → contorno nesse
                # nível encerra 95% da massa observada.
                level_95 = float(np.percentile(dens_at_obs, 5))
                return density, level_95
            except Exception:
                return np.array([]), None

        density_grid_gen, envelope_level_gen = _class_envelope(gen_vectors)
        density_grid_imp, envelope_level_imp = _class_envelope(imp_vectors)

        # ─── Scatter subsampled (caps diferenciados) ─────────────────────
        rng = np.random.default_rng(0)

        def _sample(vec: np.ndarray, cap: int) -> Tuple[np.ndarray, np.ndarray]:
            n = vec.shape[0]
            if n <= cap:
                return vec[:, 0], vec[:, 1]
            idx = rng.choice(n, size=cap, replace=False)
            return vec[idx, 0], vec[idx, 1]

        gen_x0_s, gen_x1_s = _sample(gen_vectors, scatter_cap_gen)
        imp_x0_s, imp_x1_s = _sample(imp_vectors, scatter_cap_imp)

        out["cal_curve_2d_grid_x0"] = x0_grid
        out["cal_curve_2d_grid_x1"] = x1_grid
        out["cal_curve_2d_grid_log_lr"] = grid_log_lr
        out["cal_curve_2d_gen_x0"] = gen_x0_s
        out["cal_curve_2d_gen_x1"] = gen_x1_s
        out["cal_curve_2d_imp_x0"] = imp_x0_s
        out["cal_curve_2d_imp_x1"] = imp_x1_s
        out["cal_curve_2d_coef_raw"] = coef_raw_arr
        out["cal_curve_2d_coef_std"] = coef_std_arr
        out["cal_curve_2d_intercept"] = intercept_val
        # Campos POOLED ficam vazios — caminho legacy preservado pra
        # pickles antigos pelo plot, mas executor novo não popula.
        out["cal_curve_2d_density_grid_gen"] = density_grid_gen
        out["cal_curve_2d_density_grid_imp"] = density_grid_imp
        out["cal_curve_2d_envelope_level_gen"] = envelope_level_gen
        out["cal_curve_2d_envelope_level_imp"] = envelope_level_imp
        return out

    def _compute_calibration_curve_grid(
        self,
        gen_raw: np.ndarray,
        imp_raw: np.ndarray,
        gen_log_lr: np.ndarray,
        imp_log_lr: np.ndarray,
        slope: float,
        intercept: float,
        *,
        n_grid: int = 200,
        rug_cap: int = 2000,
        piecewise_enabled: bool = False,
        hinge_anchor_s0: Optional[float] = None,
        piecewise_knots_absolute: Optional[List[float]] = None,
        calibrator_coef_raw: Optional[List[float]] = None,
        calibrator_intercept_raw: Optional[float] = None,
    ) -> dict:
        """Pré-computa grid uniforme do calibrador f(x) e da recalibração afim.

        O calibrador é tratado como função monotônica de raw_score em log-LR.
        Sem acesso direto ao calibrador treinado aqui, aproximamos f via
        interpolação linear dos pontos observados (sorted by raw). A
        recalibração afim Brümmer-Doddington opera sobre o log-LR já
        calibrado:

            affine(x) = slope · f(x) + intercept
            delta(x)  = affine(x) − f(x) = (slope−1)·f(x) + intercept

        Em ambos os casos `x` é raw_score; a "pós-afim" só é uma reta no
        plano raw×logLR quando f é linear (caso LogReg). Pra KDE/BiGauss,
        é curva.

        Zero-crossing (raw_score onde affine cruza log-LR=0) é achado por
        busca da primeira mudança de sinal — robusto a slope negativo
        (caso patológico) e a affine não estritamente monotônica.

        Retorna dict pronto pra splat em ValidationPlotData. Quando raw
        não está disponível (pickle pre-curva), todos os campos ficam
        vazios.
        """
        out = {
            "cal_curve_grid_x": np.array([]),
            "cal_curve_llr_y": np.array([]),
            "cal_curve_affine_y": np.array([]),
            "cal_curve_affine_delta_y": np.array([]),
            "cal_curve_zero_raw_score": None,
            "cal_curve_rug_gen_x": np.array([]),
            "cal_curve_rug_imp_x": np.array([]),
        }

        # Sem raw_scores não há grid a computar (pickle pre-curva).
        if gen_raw.size == 0 and imp_raw.size == 0:
            return out

        # Junta tudo e ordena por raw_score — observed (x, y) pairs.
        all_raw = np.concatenate([gen_raw, imp_raw])
        all_lr = np.concatenate([gen_log_lr, imp_log_lr])

        if all_raw.size < 2:
            return out

        order = np.argsort(all_raw)
        sorted_raw = all_raw[order]
        sorted_lr = all_lr[order]

        # Remove duplicates em raw (np.interp tolera, mas pode produzir
        # platôs estranhos quando o calibrador é descontínuo).
        unique_raw, unique_idx = np.unique(sorted_raw, return_index=True)
        unique_lr = sorted_lr[unique_idx]

        if unique_raw.size < 2:
            return out

        # Caminho piecewise-linear analítico: quando o calibrador foi
        # treinado com mode=PIECEWISE_LINEAR, computamos f(x) DIRETO a
        # partir dos coeficientes em escala bruta — sem np.interp dos
        # observados. np.interp colapsa múltiplos knots numa única
        # curvatura visual quando os knots estão próximos em relação ao
        # range total (caso típico: PLDA com s0=-22.65 e knots em
        # [-27.65, -22.65, -17.65] dentro de range [-250, +70]).
        use_piecewise = (
            piecewise_enabled
            and hinge_anchor_s0 is not None
            and piecewise_knots_absolute is not None
            and calibrator_coef_raw is not None
            and calibrator_intercept_raw is not None
            and len(calibrator_coef_raw) == 1 + len(piecewise_knots_absolute)
        )

        if use_piecewise:
            # Grid mais denso pra capturar as quebras com nitidez. Cobre
            # range observado + 5% de folga em cada lado pra dar contexto
            # visual do comportamento extrapolante além dos dados.
            x_min, x_max = float(unique_raw[0]), float(unique_raw[-1])
            span = x_max - x_min
            grid_x = np.linspace(x_min - 0.05 * span, x_max + 0.05 * span, n_grid * 2)

            # logLR(x) = β_z·(x − s0) + Σ β_j·max(x − knot_j, 0) + intercept
            s0 = float(hinge_anchor_s0)
            beta_z = float(calibrator_coef_raw[0])
            beta_hinges = [float(b) for b in calibrator_coef_raw[1:]]
            intercept_raw = float(calibrator_intercept_raw)
            z = grid_x - s0
            llr_y = beta_z * z + intercept_raw
            for beta_j, knot_abs in zip(beta_hinges, piecewise_knots_absolute):
                llr_y = llr_y + beta_j * np.maximum(grid_x - float(knot_abs), 0.0)
        else:
            grid_x = np.linspace(unique_raw[0], unique_raw[-1], n_grid)
            llr_y = np.interp(grid_x, unique_raw, unique_lr)

        affine_y = slope * llr_y + intercept
        delta_y = affine_y - llr_y  # = (slope-1)·llr_y + intercept

        zero_raw = self._find_zero_crossing(grid_x, affine_y)

        rng = np.random.default_rng(0)

        def _maybe_sample(arr: np.ndarray) -> np.ndarray:
            if arr.size <= rug_cap:
                return arr
            idx = rng.choice(arr.size, size=rug_cap, replace=False)
            return arr[idx]

        out["cal_curve_grid_x"] = grid_x
        out["cal_curve_llr_y"] = llr_y
        out["cal_curve_affine_y"] = affine_y
        out["cal_curve_affine_delta_y"] = delta_y
        out["cal_curve_zero_raw_score"] = zero_raw
        out["cal_curve_rug_gen_x"] = _maybe_sample(gen_raw)
        out["cal_curve_rug_imp_x"] = _maybe_sample(imp_raw)

        # Diagnósticos piecewise pro plot: knots + inclinações por trecho.
        # UI usa pra desenhar linhas verticais nos knots + exibir slopes
        # no subtítulo. Vazio quando o caminho não é piecewise.
        if use_piecewise:
            out["cal_curve_piecewise_knots_absolute"] = [
                float(k) for k in piecewise_knots_absolute
            ]
            # Inclinações por trecho na escala bruta: segment[0] = β_z;
            # segment[j] = β_z + cumulative sum of β_hinges até j.
            beta_z = float(calibrator_coef_raw[0])
            betas_hinge = [float(b) for b in calibrator_coef_raw[1:]]
            slopes = [beta_z]
            running = beta_z
            for b in betas_hinge:
                running += b
                slopes.append(running)
            out["cal_curve_piecewise_segment_slopes"] = slopes
        return out

    @staticmethod
    def _find_zero_crossing(
        grid_x: np.ndarray, affine_y: np.ndarray,
    ) -> Optional[float]:
        """Acha o primeiro x onde affine_y cruza zero, via mudança de sinal.

        Mais robusto que np.interp (que exige xp monotônico). Retorna
        None quando não há cruzamento na faixa observada.
        """
        if affine_y.size < 2:
            return None
        signs = np.sign(affine_y)
        changes = np.where(np.diff(signs) != 0)[0]
        if changes.size == 0:
            return None
        i = int(changes[0])
        y0, y1 = float(affine_y[i]), float(affine_y[i + 1])
        x0, x1 = float(grid_x[i]), float(grid_x[i + 1])
        if y1 == y0:
            return x0
        return x0 - y0 * (x1 - x0) / (y1 - y0)

    # =========================================================================
    # ECE NUMERICAL METRICS
    # =========================================================================

    def _compute_ece_metrics(
        self,
        ece_x: np.ndarray,
        ece_y: np.ndarray,
        ece_min_y: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Métricas escalares da curva ECE.

        - area: ∫(ECE − ECE_min) d(prior_log10_odds), via regra trapezoidal
        - max_gap: max(ECE − ECE_min) ao longo do domínio
        - argmax_prior: o prior_log10_odds onde max_gap acontece

        ece_x está em log10-odds (não em ln), conforme convenção do plot.
        """
        gap = np.asarray(ece_y, dtype=np.float64) - np.asarray(ece_min_y, dtype=np.float64)
        gap_nonneg = np.clip(gap, 0.0, None)
        area = float(np.trapezoid(gap_nonneg, ece_x))
        idx_max = int(np.argmax(gap_nonneg))
        max_gap = float(gap_nonneg[idx_max])
        argmax_prior = float(ece_x[idx_max])
        return area, max_gap, argmax_prior

    # =========================================================================
    # TAIL / OUTLIERS
    # =========================================================================

    def _compute_tail_metrics(
        self,
        gen_scores: np.ndarray,
        imp_scores: np.ndarray,
    ) -> dict:
        """
        Métricas de cauda / concentração de loss.

        Retorna dict com:
        - loss_*_q95: percentil 95 da perda por amostra.
        - max_loss_gen / max_loss_imp: worst case por classe (em bits). Importante
          quando 1 sample extremo domina o Cllr.
        - top_1pct_loss_share / top_5pct_loss_share: fração do Cllr total
          contribuída pelos top N% das amostras ponderadas.
        - n_top_1pct: contagem absoluta de samples no top 1% (interpretável
          mesmo sem total).
        """
        loss_gen = np.logaddexp(0, -gen_scores) / _LN2  # per-sample, em bits
        loss_imp = np.logaddexp(0, imp_scores) / _LN2

        loss_gen_q95 = float(np.percentile(loss_gen, 95))
        loss_imp_q95 = float(np.percentile(loss_imp, 95))
        max_loss_gen = float(loss_gen.max()) if loss_gen.size else 0.0
        max_loss_imp = float(loss_imp.max()) if loss_imp.size else 0.0

        # Cllr = 0.5/N_gen · sum(loss_gen) + 0.5/N_imp · sum(loss_imp).
        # Pra calcular share dos top N%, ponderamos cada sample pela sua
        # contribuição efetiva ao Cllr.
        w_gen = 0.5 / max(len(gen_scores), 1)
        w_imp = 0.5 / max(len(imp_scores), 1)
        contrib_gen = loss_gen * w_gen
        contrib_imp = loss_imp * w_imp
        all_contrib = np.concatenate([contrib_gen, contrib_imp])
        total = float(all_contrib.sum())
        sorted_desc = np.sort(all_contrib)[::-1]  # decrescente

        def _top_share(pct: float) -> float:
            if total <= 0.0:
                return 0.0
            n_top = max(1, int(len(sorted_desc) * pct))
            return float(sorted_desc[:n_top].sum() / total)

        top_1pct_share = _top_share(0.01)
        top_5pct_share = _top_share(0.05)
        n_top_1pct = max(1, len(all_contrib) // 100) if total > 0 else 0

        return {
            "loss_gen_q95": loss_gen_q95,
            "loss_imp_q95": loss_imp_q95,
            "max_loss_gen": max_loss_gen,
            "max_loss_imp": max_loss_imp,
            "top_1pct_loss_share": top_1pct_share,
            "top_5pct_loss_share": top_5pct_share,
            "n_top_1pct": int(n_top_1pct),
        }

    # =========================================================================
    # ECE PLOT VECTORS
    # =========================================================================

    def _calculate_ece_data(
        self,
        gen_scores: np.ndarray,
        imp_scores: np.ndarray,
        gen_cal: np.ndarray,
        imp_cal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vetores para o gráfico de Empirical Cross-Entropy (ECE).
        Eixo X: prior log10-odds em [-3.5, +3.5], 100 pontos.
        """
        prior_log10_odds = np.linspace(-3.5, 3.5, 100)
        prior_ln_odds = prior_log10_odds * np.log(10)
        P = 1 / (1 + np.exp(-prior_ln_odds))

        gen_term_real = np.mean(np.logaddexp(0, -(prior_ln_odds[:, None] + gen_scores[None, :])), axis=1) / _LN2
        imp_term_real = np.mean(np.logaddexp(0, prior_ln_odds[:, None] + imp_scores[None, :]), axis=1) / _LN2
        ece_real = P * gen_term_real + (1 - P) * imp_term_real

        gen_term_min = np.mean(np.logaddexp(0, -(prior_ln_odds[:, None] + gen_cal[None, :])), axis=1) / _LN2
        imp_term_min = np.mean(np.logaddexp(0, prior_ln_odds[:, None] + imp_cal[None, :]), axis=1) / _LN2
        ece_min = P * gen_term_min + (1 - P) * imp_term_min

        return prior_log10_odds, ece_real, ece_min
