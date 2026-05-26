import numpy as np
from typing import List, Tuple
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

        # 8. Tail / outliers (Tier 5)
        loss_gen_q95, loss_imp_q95, top_1pct_share = self._compute_tail_metrics(
            gen_arr, imp_arr,
        )

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
            loss_gen_q95=loss_gen_q95,
            loss_imp_q95=loss_imp_q95,
            top_1pct_loss_share=top_1pct_share,
            # Telemetria CV do calibrador — transporte do metadata
            # do CALIBRATED_SCORES, sem cálculo aqui.
            calibrator_cllr_train=calibrator_cllr_train,
            calibrator_cllr_train_gen=calibrator_cllr_train_gen,
            calibrator_cllr_train_imp=calibrator_cllr_train_imp,
            calibrator_cllr_cv=calibrator_cllr_cv,
            calibrator_cllr_cv_gen=calibrator_cllr_cv_gen,
            calibrator_cllr_cv_imp=calibrator_cllr_cv_imp,
        )

        # Raw scores opcionais
        gen_raw_arr = (
            np.asarray(genuine_raw_scores, dtype=np.float64)
            if genuine_raw_scores is not None else np.array([])
        )
        imp_raw_arr = (
            np.asarray(impostor_raw_scores, dtype=np.float64)
            if impostor_raw_scores is not None else np.array([])
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
    ) -> Tuple[float, float, float]:
        """
        Métricas de cauda / concentração de loss.

        - loss_*_q95: percentil 95 da perda por amostra. Mostra worst-case
          em cada classe sem ser dominado por 1 outlier extremo.
        - top_1pct_loss_share: fração do Cllr total (com pesos 0.5/N por
          classe) contribuída pelo top 1% das amostras com maior loss
          ponderada. Identifica leverage de outliers.
        """
        loss_gen = np.logaddexp(0, -gen_scores) / _LN2  # per-sample, em bits
        loss_imp = np.logaddexp(0, imp_scores) / _LN2

        loss_gen_q95 = float(np.percentile(loss_gen, 95))
        loss_imp_q95 = float(np.percentile(loss_imp, 95))

        # Cllr = 0.5/N_gen · sum(loss_gen) + 0.5/N_imp · sum(loss_imp).
        # Pra calcular share dos top 1%, ponderamos cada sample pela sua
        # contribuição efetiva ao Cllr.
        w_gen = 0.5 / max(len(gen_scores), 1)
        w_imp = 0.5 / max(len(imp_scores), 1)
        contrib_gen = loss_gen * w_gen
        contrib_imp = loss_imp * w_imp
        all_contrib = np.concatenate([contrib_gen, contrib_imp])
        total = float(all_contrib.sum())
        if total <= 0.0:
            top_1pct_share = 0.0
        else:
            n_top = max(1, len(all_contrib) // 100)
            top_sum = float(np.sort(all_contrib)[-n_top:].sum())
            top_1pct_share = top_sum / total

        return loss_gen_q95, loss_imp_q95, top_1pct_share

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
