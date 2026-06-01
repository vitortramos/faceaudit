"""Testes estruturais de comparação entre LogReg, BiGauss e LogReg-piecewise
em dados sintéticos com log-LR oráculo analítico.

Cobre propriedades determinísticas (fórmulas dos oráculos, normalização das
densidades transformadas, monotonicidade) e relações estatísticas estáveis
sobre K seeds pequeno. Asserções estatísticas mais sensíveis ficam marcadas
`slow` e são consumidas pelo script exploratório
`experiments/calibration/compare_calibration_synthetic.py`.

Referência conceitual: `docs/piecewise_linear_calibration.md` e o relatório
`experiments/calibration/RELATORIO_comparacao_calibradores.md`.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from experiments.calibration._synthetic_calibration_helpers import (
    SCENARIOS,
    cllr_from_logLR,
    fit_eval,
    log_pdf_s_given_hd,
    log_pdf_s_given_hp,
    make_calibrators,
    oracle_cllr_population,
    oracle_log_lr,
    sample,
)


SEEDS_FAST = [0, 1, 2]
N_CAL_FAST = 2000
N_VAL_FAST = 20000
N_VAL_BIG = 20000


# ──────────────────────────── oráculos vs razão de densidades ────────

def test_cllr_helper_matches_production():
    """O Cllr local (`cllr_from_logLR`) deve ser idêntico ao `compute_cllr`
    de produção — o helper delega para ele, este teste trava a equivalência."""
    from faceaudit.services.cllr_math import compute_cllr
    rng = np.random.default_rng(0)
    for _ in range(5):
        llr_hp = rng.normal(2.0, 3.0, 1000)
        llr_hd = rng.normal(-2.0, 3.0, 1000)
        a = cllr_from_logLR(llr_hp, llr_hd)
        b = compute_cllr(llr_hp, llr_hd)
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_scenario_C_oracle_is_not_clipped_in_mass_region():
    """O oráculo de C deve ser ≈ 2d·t (sem platô artificial) na faixa de massa.

    Regressão do bug do `atanh_epsilon=1e-4`, que clipava a LR verdadeira em
    ~2.5% das amostras. Com 1e-12, logLR*(tanh(t/T)) = 2d·t exatamente.
    """
    d = SCENARIOS["C_SMOOTH_TANH"]["d"]
    T = SCENARIOS["C_SMOOTH_TANH"]["T"]
    # latente cobrindo ±4σ (inclui a região antes saturada/clipada)
    t = np.linspace(-4.0, 4.0, 400)
    s = np.tanh(t / T)
    llr = oracle_log_lr("C_SMOOTH_TANH", s)
    np.testing.assert_allclose(llr, 2.0 * d * t, atol=1e-6)


@pytest.mark.parametrize("scen", list(SCENARIOS.keys()))
def test_oracle_matches_density_ratio(scen):
    """Para uma malha em região de massa, logLR*(s) ≈ log p(s|Hp) − log p(s|Hd).

    Oráculo e densidade usam o mesmo clip (1e-12) em C, então batem em toda a
    faixa central das amostras.
    """
    s_hp, s_hd = sample(scen, 5000, seed=0)
    union = np.concatenate([s_hp, s_hd])
    lo, hi = np.quantile(union, [0.01, 0.99])
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

    Oráculo e densidade usam o mesmo clip 1e-12 (essencialmente exato). A
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


def test_tanh_oracle_matches_density_ratio_near_supported_edges():
    """Regressão do bug do clipping: oráculo de C deve casar a razão de
    densidades mesmo perto das bordas saturadas ±0.999999."""
    scen = "C_SMOOTH_TANH"
    grid = np.array([-0.999999, -0.999, 0.0, 0.999, 0.999999])
    by_ratio = log_pdf_s_given_hp(scen, grid) - log_pdf_s_given_hd(scen, grid)
    by_formula = oracle_log_lr(scen, grid)
    np.testing.assert_allclose(by_ratio, by_formula, atol=1e-8)


def test_tilted_mixture_scenario_D_internals():
    """Cenário D: pesos inclinados π'_k somam 1, Z finito, massas ≈ 1.

    Localiza regressões futuras na fórmula de π'_k = π_k·exp(a·μ_k+a²σ_k²/2)/Z.
    """
    from experiments.calibration._synthetic_calibration_helpers import (
        scenario_d_health, oracle_cllr_population,
    )
    h = scenario_d_health()
    assert np.isfinite(h["log_Z"])
    assert abs(sum(h["tilted_component_weights_hp"]) - 1.0) < 1e-12
    assert abs(sum(h["base_component_weights_hd"]) - 1.0) < 1e-12
    _, health = oracle_cllr_population("D_LEPTOKURTIC_SMALL_N")
    assert abs(health["population_mass_hp"] - 1.0) < 1e-6
    assert abs(health["population_mass_hd"] - 1.0) < 1e-6


@pytest.mark.parametrize("scen", ["A_AFFINE", "B_EXACT_ONE_KNOT",
                                   "C_SMOOTH_TANH", "D_LEPTOKURTIC_SMALL_N"])
def test_population_integration_is_stable_across_orders(scen):
    """A integração populacional do oráculo deve ser estável entre ordens
    Gauss-Hermite (64 vs 128) — garante que a malha já convergiu."""
    c64, _ = oracle_cllr_population(scen, order=64)
    c128, _ = oracle_cllr_population(scen, order=128)
    assert abs(c64 - c128) < 1e-6, f"{scen}: |c64−c128| = {abs(c64-c128):.2e}"


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


# Cenário C incluído: a saturação de tanh perto de ±1 é exatamente onde
# clipping indevido, platôs e NaN/Inf apareceriam.
_HEALTH_SCENARIOS = ["A_AFFINE", "B_EXACT_ONE_KNOT", "C_SMOOTH_TANH",
                     "D_LEPTOKURTIC_SMALL_N"]


@pytest.mark.parametrize("scen", _HEALTH_SCENARIOS)
def test_all_calibrators_emit_finite_loglr(scen):
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=0)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=10)
    for spec in make_calibrators():
        r = fit_eval(spec, scen, 0, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        assert r.fit_status == "OK", f"{spec.name} falhou em {scen}: {r.fit_failure_reason}"
        assert r.llr_val_hp is not None
        assert np.all(np.isfinite(r.llr_val_hp))
        assert np.all(np.isfinite(r.llr_val_hd))


def test_non_monotonic_truth_calibrators_emit_finite_outputs_when_fit_succeeds():
    """Cenário E: a verdade é não-monotônica, mas todo calibrador que retorna
    OK ainda deve produzir log-LRs finitos (sem NaN/Inf)."""
    scen = "E_NON_MONOTONIC_STUDENT_T"
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=0)
    s_val_hp, s_val_hd = sample(scen, 2000, seed=10)
    for spec in make_calibrators():
        r = fit_eval(spec, scen, 0, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        if r.fit_status != "OK":
            continue
        assert np.all(np.isfinite(r.llr_val_hp))
        assert np.all(np.isfinite(r.llr_val_hd))


@pytest.mark.parametrize("scen", _HEALTH_SCENARIOS)
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
        # Tolerância explícita por calibrador (NÃO inferida por substring do
        # nome — `bigauss_logreg_kde_cllr` contém "logreg").
        tol = spec.monotonicity_tol
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


def test_regime_B_manual_true_knot_recovers_expected_slopes():
    """A variante manual deve recuperar os slopes verdadeiros [2d, 2d/c],
    não apenas slope[1] > slope[0]. Verdade: [1.0, 2.222] para d=0.5, c=0.45."""
    scen = "B_EXACT_ONE_KNOT"
    d = SCENARIOS[scen]["d"]
    c = SCENARIOS[scen]["c"]
    expected = np.array([2.0 * d, 2.0 * d / c])
    # mediana sobre poucas seeds para estabilidade
    recovered = []
    for seed in SEEDS_FAST:
        s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
        s_val_hp, s_val_hd = sample(scen, 2000, seed=seed + 10_000)
        spec = next(s for s in make_calibrators(include_manual_true_knot=True)
                    if s.name == "logreg_piecewise_manual_true_knot")
        r = fit_eval(spec, scen, seed, s_cal_hp, s_cal_hd, s_val_hp, s_val_hd, N_CAL_FAST)
        assert r.fit_status == "OK"
        recovered.append(r.piecewise_segment_slopes_raw)
    median_slopes = np.median(np.array(recovered), axis=0)
    # tolerância 12% (observado [0.999, 2.188] vs [1.0, 2.222])
    np.testing.assert_allclose(median_slopes, expected, rtol=0.12, atol=0.05)


# ─────────────────── Cllr_min preservado sob remap monotônico ────────


def test_cllr_min_approximately_invariant_under_monotonic_remap():
    """Em A (todos monotônicos), Cllr_min deve ser ≈ entre os 3 calibradores
    operacionais E ≈ ao Cllr_min do score bruto.

    A tese é: um remapeamento monotônico não altera a ordenação disponível no
    score original, então o Cllr_min (que o PAVA extrai dessa ordenação)
    permanece invariante. O score bruto é a referência explícita.
    """
    scen = "A_AFFINE"
    seed = 0
    s_cal_hp, s_cal_hd = sample(scen, N_CAL_FAST, seed=seed)
    s_val_hp, s_val_hd = sample(scen, N_VAL_BIG, seed=seed + 10_000)

    # Usa ValidationMathService para Cllr_min via PAVA — mesma fonte da prod
    from faceaudit.services.validation_math_service import ValidationMathService
    svc = ValidationMathService()

    def _cllr_min(gen, imp):
        result = svc.calculate_metrics(
            strategy_id="x", strategy_name="x", strategy_type="synthetic",
            genuine_scores=list(map(float, gen)),
            impostor_scores=list(map(float, imp)),
        )
        return result.metrics.cllr_min

    # Referência: Cllr_min do score bruto (sem calibração)
    cllr_min_raw = _cllr_min(s_val_hp, s_val_hd)

    cllr_mins = [cllr_min_raw]
    for spec in make_calibrators():
        calibrator = spec.fresh()
        scores_cal = np.concatenate([s_cal_hp, s_cal_hd])
        labels_cal = np.concatenate([np.ones_like(s_cal_hp, dtype=np.int64),
                                      np.zeros_like(s_cal_hd, dtype=np.int64)])
        calibrator.fit(scores_cal, labels_cal, logger_callback=lambda *_a, **_k: None)
        llr_hp = np.asarray(calibrator.calibrate_scores(s_val_hp), dtype=np.float64)
        llr_hd = np.asarray(calibrator.calibrate_scores(s_val_hd), dtype=np.float64)
        cllr_mins.append(_cllr_min(llr_hp, llr_hd))
    spread = float(np.max(cllr_mins) - np.min(cllr_mins))
    assert spread < 1e-2, (
        f"Cllr_min spread (incl. raw) = {spread:.4f}, cllr_mins = {cllr_mins}"
    )


# ─────────────────── sanidade da integração populacional ─────────────


@pytest.mark.parametrize("scen", ["A_AFFINE", "B_EXACT_ONE_KNOT",
                                   "C_SMOOTH_TANH", "D_LEPTOKURTIC_SMALL_N"])
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
    """Diagnóstico do anchor automático em B, com limites muito generosos.

    Não exige s0=0 (a LogReg auxiliar é mal-especificada quando a verdade é
    piecewise). Apenas trava uma regressão grosseira: o regret mediano deve
    permanecer pequeno e o erro do anchor não pode explodir."""
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
    # Limites generosos (guarda contra regressão grosseira, não asserção fina):
    assert np.median(regrets) < 5e-3
    assert np.median(s0_errors) < 0.5


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
    # Proteção mínima contra uma inversão grosseira do comportamento observado;
    # a comparação estatística completa pertence ao script exploratório.
    assert iqr["bigauss_logreg_kde_cllr"] >= 0.5 * iqr["logreg_linear"], (
        f"BiGauss IQR ({iqr['bigauss_logreg_kde_cllr']:.4f}) < 0.5 × LogReg IQR "
        f"({iqr['logreg_linear']:.4f}) — comportamento inesperado"
    )


@pytest.mark.slow
def test_regime_E_non_monotonic_truth_characterization():
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
