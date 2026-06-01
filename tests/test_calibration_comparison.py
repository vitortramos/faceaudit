"""Testes estruturais de comparação entre LogReg, BiGauss e LogReg-piecewise
em dados sintéticos com log-LR oráculo analítico.

Cobre propriedades determinísticas (fórmulas dos oráculos, normalização das
densidades transformadas, monotonicidade) e relações estatísticas estáveis
sobre K seeds pequeno. Asserções estatísticas mais sensíveis ficam marcadas
`slow` e são consumidas pelo script exploratório
`experiments/calibration/compare_calibration_synthetic.py`.

Plano: docs piecewise §0.4, §1.4 e a revisão final do plano gravada em
`/root/.claude/plans/jaunty-prancing-boole.md`.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate, stats

from experiments.calibration._synthetic_calibration_helpers import (
    SCENARIOS,
    cllr_from_logLR,
    cllr_population,
    fit_eval,
    log_pdf_s_given_hd,
    log_pdf_s_given_hp,
    make_calibrators,
    oracle_cllr_holdout,
    oracle_cllr_population,
    oracle_log_lr,
    sample,
)


SEEDS_FAST = [0, 1, 2]
N_CAL_FAST = 2000
N_VAL_FAST = 20000
N_VAL_BIG = 20000


# ──────────────────────────── oráculos vs razão de densidades ────────

@pytest.mark.parametrize("scen", list(SCENARIOS.keys()))
def test_oracle_matches_density_ratio(scen):
    """Para uma malha em região de massa, logLR*(s) ≈ log p(s|Hp) − log p(s|Hd).

    No cenário C, o oráculo usa `atanh_epsilon=1e-4` para estabilidade de
    scores observados, mas a densidade usa clip muito menor (1e-12). Para
    evitar discrepância artificial nas bordas, o grid de teste evita a
    região saturada (Q5–Q95 em vez de Q1–Q99) no cenário C.
    """
    s_hp, s_hd = sample(scen, 5000, seed=0)
    union = np.concatenate([s_hp, s_hd])
    quantiles = (0.05, 0.95) if scen == "C_SMOOTH_TANH" else (0.01, 0.99)
    lo, hi = np.quantile(union, list(quantiles))
    grid = np.linspace(lo, hi, 200)
    by_ratio = log_pdf_s_given_hp(scen, grid) - log_pdf_s_given_hd(scen, grid)
    by_formula = oracle_log_lr(scen, grid)
    np.testing.assert_allclose(by_ratio, by_formula, atol=1e-10)


def test_transformed_density_integrates_to_one_population_B():
    """Para o cenário B, p_s(s|H) integra a 1 com jacobiano correto."""
    mass_hp, _ = integrate.quad(
        lambda s: float(np.exp(log_pdf_s_given_hp("B_EXACT_ONE_KNOT",
                                                    np.array([s])))[0]),
        -10.0, 10.0, limit=200,
    )
    mass_hd, _ = integrate.quad(
        lambda s: float(np.exp(log_pdf_s_given_hd("B_EXACT_ONE_KNOT",
                                                    np.array([s])))[0]),
        -10.0, 10.0, limit=200,
    )
    assert abs(mass_hp - 1.0) < 1e-6, mass_hp
    assert abs(mass_hd - 1.0) < 1e-6, mass_hd


def test_transformed_density_integrates_to_one_population_C():
    """Para o cenário C, p_s(s|H) integra a 1 sobre (−1, 1).

    O clip interno da densidade é 1e-12 (essencialmente exato), diferente do
    `atanh_epsilon=1e-4` usado no oráculo para scores observados. A
    quadratura adaptativa converge rapidamente.
    """
    mass_hp, _ = integrate.quad(
        lambda s: float(np.exp(log_pdf_s_given_hp("C_SMOOTH_TANH",
                                                    np.array([s])))[0]),
        -1.0, 1.0, limit=400,
    )
    mass_hd, _ = integrate.quad(
        lambda s: float(np.exp(log_pdf_s_given_hd("C_SMOOTH_TANH",
                                                    np.array([s])))[0]),
        -1.0, 1.0, limit=400,
    )
    assert abs(mass_hp - 1.0) < 1e-3, mass_hp
    assert abs(mass_hd - 1.0) < 1e-3, mass_hd


def test_student_t_oracle_is_non_monotonic():
    """O oráculo de E muda de sinal da derivada nas caudas."""
    nu = SCENARIOS["E_NON_MONOTONIC_STUDENT_T"]["nu"]
    d = SCENARIOS["E_NON_MONOTONIC_STUDENT_T"]["d"]
    # zero da derivada: ν + d² − t² = 0 → t = ±√(ν+d²)
    t_zero = float(np.sqrt(nu + d * d))
    # logLR(t_zero + 1) < logLR(t_zero - 1) ? (derivada já caiu)
    llr = oracle_log_lr("E_NON_MONOTONIC_STUDENT_T",
                        np.array([t_zero - 1.0, t_zero, t_zero + 1.0]))
    # Não-monotônico depois do ponto crítico
    assert llr[1] > llr[0], "derivada deveria ser positiva ainda antes do crítico"
    assert llr[2] < llr[1], "derivada deveria virar negativa após o crítico"


# ──────────────────────────── saúde dos calibradores ─────────────────


@pytest.mark.parametrize("scen", ["A_AFFINE", "B_EXACT_ONE_KNOT", "D_LEPTOKURTIC_SMALL_N"])
def test_all_calibrators_emit_finite_loglr(scen):
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=0)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=10)
    for spec in make_calibrators():
        r = fit_eval(spec, scen, 0, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        assert r.fit_status == "OK", f"{spec.name} falhou em {scen}: {r.fit_failure_reason}"
        assert r.llr_val_hp is not None
        assert np.all(np.isfinite(r.llr_val_hp))
        assert np.all(np.isfinite(r.llr_val_hd))


@pytest.mark.parametrize("scen", ["A_AFFINE", "B_EXACT_ONE_KNOT", "D_LEPTOKURTIC_SMALL_N"])
def test_calibrated_outputs_are_monotonic_non_decreasing(scen):
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=0)
    # Avalia numa malha densa de scores (não nas amostras, para detectar bem
    # diferenças por classe)
    scores_grid = np.linspace(np.percentile(np.concatenate([s_cal_hp, s_cal_hd]), 1),
                              np.percentile(np.concatenate([s_cal_hp, s_cal_hd]), 99),
                              1000)
    scores_cal = np.concatenate([s_cal_hp, s_cal_hd])
    labels_cal = np.concatenate([np.ones_like(s_cal_hp, dtype=np.int64),
                                  np.zeros_like(s_cal_hd, dtype=np.int64)])
    for spec in make_calibrators():
        calibrator = spec.fresh()
        calibrator.fit(scores_cal, labels_cal, logger_callback=lambda *_a, **_k: None)
        llr = np.asarray(calibrator.calibrate_scores(scores_grid), dtype=np.float64)
        diffs = np.diff(llr)
        tol = 1e-12 if "logreg" in spec.name else 1e-6  # BiGauss aceita platôs
        assert np.all(diffs >= -tol), (
            f"{spec.name} em {scen}: monotonicidade não-decrescente quebrada "
            f"(min diff = {diffs.min():.2e}, tol = {tol:.0e})"
        )


def test_piecewise_segment_slopes_are_positive():
    scen = "B_EXACT_ONE_KNOT"
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=0)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=10)
    for spec in make_calibrators(include_manual_true_knot=True):
        if "piecewise" not in spec.name:
            continue
        r = fit_eval(spec, scen, 0, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        assert r.fit_status == "OK", r.fit_failure_reason
        slopes = r.piecewise_segment_slopes_raw
        assert slopes is not None and len(slopes) >= 2, slopes
        assert all(m > 0 for m in slopes), (
            f"{spec.name}: slopes {slopes} contêm não-positivos"
        )


# ─────────────────────────── cenário A — parcimônia ──────────────────


def test_regime_A_no_large_overflexibility_penalty():
    """Mediana sobre K seeds: |Cllr_piecewise_auto − Cllr_linear| pequeno."""
    diffs = []
    for seed in SEEDS_FAST:
        s_cal_hp, s_cal_hd = sample("A_AFFINE", N_CAL_FAST, seed=seed)
        s_val_hp, s_val_hd = sample("A_AFFINE", N_VAL_FAST, seed=seed + 10_000)
        per = {}
        for spec in make_calibrators():
            r = fit_eval(spec, "A_AFFINE", seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
            assert r.fit_status == "OK"
            per[spec.name] = r.cllr_population
        diffs.append(abs(per["logreg_piecewise_auto_s0"] - per["logreg_linear"]))
    margin = 5e-3  # generosa: parcimônia em N grande
    assert float(np.median(diffs)) < margin, f"Δmediana = {np.median(diffs):.4f}"


# ─────────────────────────── cenário B — basis estrutural ────────────


def test_regime_B_manual_true_knot_has_low_population_regret():
    """Com knot manual em zero, C=1e4 e N grande, o regret populacional
    deve ser muito pequeno (a basis representa exatamente a verdade)."""
    scen = "B_EXACT_ONE_KNOT"
    seed = 0
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
    s_val_hp, s_val_hd = sample(scen, N_VAL_FAST, seed=seed + 10_000)
    cllr_oracle_pop, _ = oracle_cllr_population(scen)
    spec = next(s for s in make_calibrators(include_manual_true_knot=True)
                if s.name == "logreg_piecewise_manual_true_knot")
    r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
    assert r.fit_status == "OK", r.fit_failure_reason
    regret = r.cllr_population - cllr_oracle_pop
    assert regret < 5e-3, f"regret_pop = {regret:.4f}"


def test_regime_B_manual_true_knot_beats_linear():
    """Mediana sobre K seeds: piecewise manual reduz Cllr vs linear."""
    margins = []
    for seed in SEEDS_FAST:
        s_cal_hp, s_cal_hd = sample("B_EXACT_ONE_KNOT", N_CAL_FAST, seed=seed)
        s_val_hp, s_val_hd = sample("B_EXACT_ONE_KNOT", N_VAL_FAST, seed=seed + 10_000)
        per = {}
        for spec in make_calibrators(include_manual_true_knot=True):
            r = fit_eval(spec, "B_EXACT_ONE_KNOT", seed,
                          s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
            assert r.fit_status == "OK"
            per[spec.name] = r.cllr_population
        margins.append(per["logreg_linear"]
                       - per["logreg_piecewise_manual_true_knot"])
    # Margem realista observada ≈ 0.006; usamos 0.002 com folga
    assert float(np.median(margins)) > 2e-3, (
        f"Δmediana(linear − manual) = {np.median(margins):.4f}"
    )


def test_regime_B_piecewise_second_slope_is_higher():
    """No cenário B, verdade tem slopes (2d, 2d/c) → slope direito > esquerdo."""
    scen = "B_EXACT_ONE_KNOT"
    seed = 0
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=seed + 10_000)
    spec = next(s for s in make_calibrators(include_manual_true_knot=True)
                if s.name == "logreg_piecewise_manual_true_knot")
    r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
    assert r.fit_status == "OK"
    slopes = r.piecewise_segment_slopes_raw
    assert slopes is not None and len(slopes) == 2
    assert slopes[1] > slopes[0], f"slopes = {slopes}, esperado slope[1] > slope[0]"


# ─────────────────── Cllr_min preservado sob remap monotônico ────────


def test_cllr_min_approximately_invariant_under_monotonic_remap():
    """Em A (todos monotônicos), Cllr_min deve ser ≈ entre os 3
    calibradores operacionais."""
    scen = "A_AFFINE"
    seed = 0
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
    s_val_hp, s_val_hd = sample(scen, N_VAL_BIG, seed=seed + 10_000)

    # Usa ValidationMathService para Cllr_min via PAVA — mesma fonte da prod
    from faceaudit.services.validation_math_service import ValidationMathService
    svc = ValidationMathService()

    cllr_mins = []
    for spec in make_calibrators():
        calibrator = spec.fresh()
        scores_cal = np.concatenate([s_cal_hp, s_cal_hd])
        labels_cal = np.concatenate([np.ones_like(s_cal_hp, dtype=np.int64),
                                      np.zeros_like(s_cal_hd, dtype=np.int64)])
        calibrator.fit(scores_cal, labels_cal, logger_callback=lambda *_a, **_k: None)
        llr_hp = np.asarray(calibrator.calibrate_scores(s_val_hp), dtype=np.float64)
        llr_hd = np.asarray(calibrator.calibrate_scores(s_val_hd), dtype=np.float64)
        result = svc.calculate_metrics(
            strategy_id=spec.name, strategy_name=spec.name, strategy_type="synthetic",
            genuine_scores=list(map(float, llr_hp)),
            impostor_scores=list(map(float, llr_hd)),
        )
        cllr_mins.append(result.metrics.cllr_min)
    spread = float(np.max(cllr_mins) - np.min(cllr_mins))
    assert spread < 1e-2, f"Cllr_min spread = {spread:.4f}, cllr_mins = {cllr_mins}"


# ─────────────────── sanidade da integração populacional ─────────────


@pytest.mark.parametrize("scen", ["A_AFFINE", "B_EXACT_ONE_KNOT", "D_LEPTOKURTIC_SMALL_N"])
def test_population_oracle_regret_is_non_negative_with_numeric_tolerance(scen):
    """O oráculo minimiza Cllr esperado → qualquer calibrador tem Cllr_pop ≥ Cllr_oracle.

    Em cenários monotônicos com basis adequada, isso vira `≥ −1e-3` (tol numérica).
    O calibrador linear é mal-especificado em B, então tem regret claramente > 0.
    """
    cllr_oracle, health = oracle_cllr_population(scen, order=128)
    assert abs(health["population_mass_hp"] - 1.0) < 1e-6
    assert abs(health["population_mass_hd"] - 1.0) < 1e-6

    seed = 0
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=seed + 10_000)
    for spec in make_calibrators():
        r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        if r.fit_status != "OK" or r.cllr_population is None:
            continue
        regret = r.cllr_population - cllr_oracle
        assert regret >= -1e-3, (
            f"{spec.name} em {scen}: regret_pop = {regret:.6f} < 0 "
            f"(falha de sanidade da integração)"
        )


# ───────────────────────────────── slow ───────────────────────────────


@pytest.mark.slow
def test_regime_B_auto_s0_operational_behavior():
    """Diagnóstico: registra o erro do anchor automático sem exigir s0=0."""
    s0_errors = []
    regrets = []
    cllr_oracle_pop, _ = oracle_cllr_population("B_EXACT_ONE_KNOT")
    for seed in range(5):
        s_cal_hp, s_cal_hd = sample("B_EXACT_ONE_KNOT", N_CAL_FAST, seed=seed)
        s_val_hp, s_val_hd = sample("B_EXACT_ONE_KNOT", N_VAL_FAST, seed=seed + 10_000)
        spec = next(s for s in make_calibrators()
                    if s.name == "logreg_piecewise_auto_s0")
        r = fit_eval(spec, "B_EXACT_ONE_KNOT", seed,
                      s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        assert r.fit_status == "OK"
        s0_errors.append(abs(r.piecewise_s0 - 0.0))
        regrets.append(r.cllr_population - cllr_oracle_pop)
    print(f"\nAUTO_LOGREG s0 errors: median={np.median(s0_errors):.3f}, "
          f"max={np.max(s0_errors):.3f}; regret median={np.median(regrets):.4f}")


@pytest.mark.slow
def test_regime_D_bigauss_variability_small_n():
    """Em N=50, IQR do regret do BiGauss não deve ser menor que da família LogReg."""
    scen = "D_LEPTOKURTIC_SMALL_N"
    cllr_oracle_pop, _ = oracle_cllr_population(scen)
    regrets = {"logreg_linear": [], "logreg_piecewise_auto_s0": [],
               "bigauss_logreg_kde_cllr": []}
    for seed in range(30):
        s_cal_hp, s_cal_hd = sample(scen, 50, seed=seed)
        s_val_hp, s_val_hd = sample(scen, 5000, seed=seed + 10_000)
        for spec in make_calibrators():
            r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, 50)
            if r.fit_status == "OK" and r.cllr_population is not None:
                regrets[spec.name].append(r.cllr_population - cllr_oracle_pop)
    iqr = {name: float(np.percentile(v, 75) - np.percentile(v, 25))
           for name, v in regrets.items() if v}
    print(f"\nIQR(regret_pop) por calibrador em N=50: {iqr}")
    # Direcional: BiGauss não deve ter IQR claramente menor que LogReg linear
    assert iqr["bigauss_logreg_kde_cllr"] >= 0.5 * iqr["logreg_linear"], (
        f"BiGauss IQR ({iqr['bigauss_logreg_kde_cllr']:.4f}) < 0.5 × LogReg IQR "
        f"({iqr['logreg_linear']:.4f}) — comportamento inesperado"
    )


@pytest.mark.slow
def test_regime_E_non_monotonic_truth_behavior():
    """Caracterização: registra failure rate e regret no cenário não-monotônico."""
    scen = "E_NON_MONOTONIC_STUDENT_T"
    cllr_oracle_pop, _ = oracle_cllr_population(scen)
    statuses = {"logreg_linear": [], "logreg_piecewise_auto_s0": [],
                "bigauss_logreg_kde_cllr": []}
    regrets = {name: [] for name in statuses}
    for seed in range(10):
        s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
        s_val_hp, s_val_hd = sample(scen, 5000, seed=seed + 10_000)
        for spec in make_calibrators():
            r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
            statuses[spec.name].append(r.fit_status)
            if r.fit_status == "OK" and r.cllr_population is not None:
                regrets[spec.name].append(r.cllr_population - cllr_oracle_pop)
    failure_rates = {name: sum(1 for s in v if s != "OK") / len(v)
                     for name, v in statuses.items()}
    median_regrets = {name: float(np.median(v)) if v else float("nan")
                      for name, v in regrets.items()}
    print(f"\nE: failure rates = {failure_rates}")
    print(f"E: median regret_pop = {median_regrets}")
