"""Validação numérica da implementação do BiGaussianizedCalibrationModel
contra o paper Morrison (2024, "Bi-Gaussianized calibration of likelihood
ratios", Law, Probability and Risk 23/1, mgae004).

Reproduz três experimentos do paper:

  1. **Relações fundamentais (Eq. 6, 7, 8, 10)** — para σ alvo conhecido,
     gera amostras bi-gaussianas perfeitas e verifica se nossa fórmula
     `_sigma_sq_from_cllr` (Eq. 8) e `_sigma_sq_from_eer` (Eq. 10)
     recuperam o σ original a partir do Cllr/EER calculado.

  2. **§2.6 Tabela 1** — Monte Carlo de 100 trials para σ_true ∈
     {1, 2, 3, 4} (paper usa 1000, reduzimos para acelerar). Cada trial
     gera 100 same-source + 4950 different-source de N(±σ²/2, σ²) e
     mede o σ alvo recuperado pelos variants EER, LogReg e KDE.
     Compara RMS contra valores tabelados do paper.

  3. **§3.2 reprodução qualitativa** — para σ_true=3, gera 1 trial
     grande, treina BiGauss(LOGREG) e verifica se o output do BiGauss
     em dados perfeitamente calibrados é aproximadamente a função
     identidade (paper §3.2 Fig. 12).

Uso::

    .venv/Scripts/python.exe -m faceaudit.scripts.validate_bigauss_against_paper

Não tem flags. Saída tabular no stdout. Tempo típico: ~2-3 min.
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# Força stdout em UTF-8 — o terminal Windows default (cp1252) não renderiza
# σ, ², Φ etc. usados nos labels da tabela.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace",
    )

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from faceaudit.models.calibration.bi_gaussianized_calibration_model import (
    BiGaussianizedCalibrationModel,
    _CLLR_FIT_B,
    _CLLR_FIT_C,
)
from faceaudit.models.calibration.bi_gaussianized_calibration_model_config import (
    BiGaussianizedCalibrationModelConfig,
    BiGaussInitialCalibrator,
    BiGaussInversionMethod,
    BiGaussTargetMetric,
)


# Tabela 1 do paper (RMS errors do σ_target relativo ao σ_true).
PAPER_TABLE_1_RMS = {
    1: {"EER": 0.133, "LogReg": 0.113, "KDE": 0.122},
    2: {"EER": 0.153, "LogReg": 0.127, "KDE": 0.131},
    3: {"EER": 0.203, "LogReg": 0.162, "KDE": 0.159},
    4: {"EER": 0.309, "LogReg": 0.253, "KDE": 0.232},
}


@dataclass
class TrialResult:
    sigma_true: float
    sigma_est_eer: Optional[float]
    sigma_est_logreg: Optional[float]
    sigma_est_kde: Optional[float]


def _generate_bi_gaussian_sample(
    sigma_true: float,
    n_same: int,
    n_diff: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Amostra de um sistema bi-gaussiano perfeitamente calibrado:
    Hd ~ N(-σ²/2, σ²), Hp ~ N(+σ²/2, σ²)."""
    sigma_sq = sigma_true * sigma_true
    mu_d = -sigma_sq / 2.0
    mu_s = +sigma_sq / 2.0
    same_scores = rng.normal(loc=mu_s, scale=sigma_true, size=n_same)
    diff_scores = rng.normal(loc=mu_d, scale=sigma_true, size=n_diff)
    scores = np.concatenate([same_scores, diff_scores])
    labels = np.concatenate([
        np.ones(n_same, dtype=np.int32),
        np.zeros(n_diff, dtype=np.int32),
    ])
    return scores, labels


def _train_one_variant(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_metric: BiGaussTargetMetric,
    initial_calibrator: BiGaussInitialCalibrator,
) -> Optional[float]:
    """Treina BiGauss e retorna o σ_target recuperado, ou None se falhar."""
    cfg = BiGaussianizedCalibrationModelConfig(
        target_metric=target_metric,
        initial_calibrator=initial_calibrator,
        inversion_method=BiGaussInversionMethod.INTERP1D,
        sigma_shrinkage_factor=1.0,
    )
    model = BiGaussianizedCalibrationModel(config=cfg)
    try:
        model.fit(scores=scores, labels=labels, logger_callback=lambda _msg: None)
    except Exception as exc:
        print(f"  ⚠ trial falhou: {exc}")
        return None
    return float(model.sigma_target)


# ───────────────────────────────────────────────────────────────────────────
# Experimento 1: Relações fundamentais (Eq. 8 e Eq. 10)
# ───────────────────────────────────────────────────────────────────────────

def experiment_1_fundamental_equations() -> None:
    """Verifica que para σ conhecido, Cllr-via-Eq-6 invertido pela Eq-8
    recupera σ². Análogo para EER via Eq-10.
    """
    print("=" * 76)
    print("EXPERIMENTO 1 — Relações fundamentais (paper §2.3 Eq. 6-8 / §2.5 Eq. 10)")
    print("=" * 76)

    print("\nPara cada σ_true, computamos Cllr/EER analítico do sistema "
          "perfeitamente calibrado e invertemos via Eq. 8 / Eq. 10:\n")

    print(
        f"{'σ_true':>8} {'σ²_true':>10} | "
        f"{'Cllr (Eq. 6)':>15} {'σ_recov (Eq. 8)':>18} {'erro %':>10} | "
        f"{'EER':>9} {'σ_recov (Eq. 10)':>18} {'erro %':>10}"
    )
    print("-" * 130)

    for sigma_true in [1.0, 2.0, 3.0, 4.0, 5.0]:
        sigma_sq = sigma_true * sigma_true
        mu_s = sigma_sq / 2.0

        # Cllr analítico via Eq. 6: integrar f(x|μ_s, σ)·ln(1+e^(-x))/ln(2)
        def integrand(x: float) -> float:
            return norm.pdf(x, loc=mu_s, scale=sigma_true) * np.log1p(np.exp(-x))
        cllr, _ = quad(integrand, -50.0, 50.0)
        cllr /= np.log(2.0)

        # Inverter via nossa _sigma_sq_from_cllr (Eq. 8)
        sigma_sq_recov = BiGaussianizedCalibrationModel._sigma_sq_from_cllr(cllr)
        sigma_recov_cllr = float(np.sqrt(sigma_sq_recov))
        err_cllr = 100.0 * abs(sigma_recov_cllr - sigma_true) / sigma_true

        # EER analítico: f(-σ/2 | 0,1) onde f = pdf normal
        eer = float(norm.cdf(-sigma_true / 2.0))

        # Inverter via _sigma_sq_from_eer (Eq. 10)
        sigma_sq_recov_eer = BiGaussianizedCalibrationModel._sigma_sq_from_eer(eer)
        sigma_recov_eer = float(np.sqrt(sigma_sq_recov_eer))
        err_eer = 100.0 * abs(sigma_recov_eer - sigma_true) / sigma_true

        print(
            f"{sigma_true:8.2f} {sigma_sq:10.2f} | "
            f"{cllr:15.6f} {sigma_recov_cllr:18.4f} {err_cllr:9.3f}% | "
            f"{eer:9.4f} {sigma_recov_eer:18.4f} {err_eer:9.3f}%"
        )

    print("\nConclusão esperada:")
    print("  - σ via Eq. 8 (CLLR): erro ~1-2% devido ao fit numérico (b=17.7, c=9.33e-3).")
    print("  - σ via Eq. 10 (EER): erro essencialmente zero (forma analítica fechada).")


# ───────────────────────────────────────────────────────────────────────────
# Experimento 2: Tabela 1 do paper (Monte Carlo)
# ───────────────────────────────────────────────────────────────────────────

def experiment_2_table1_reproduction(n_trials: int = 100, seed: int = 42) -> None:
    """Reproduz §2.6 Tabela 1 com Monte Carlo reduzido (paper usa 1000 trials).

    Para cada σ_true ∈ {1,2,3,4}, gera n_trials amostras de 100 same +
    4950 diff de N(±σ²/2, σ²), treina os 3 variants (EER, LogReg, KDE) e
    mede RMS error do σ_target estimado vs σ_true.
    """
    print("\n" + "=" * 76)
    print(f"EXPERIMENTO 2 — Tabela 1 do paper (Monte Carlo, n_trials={n_trials})")
    print("=" * 76)
    print("\nEstrutura idêntica ao paper §2.6:")
    print("  - 100 same-source + 4950 different-source por trial")
    print("  - σ_true ∈ {1, 2, 3, 4}")
    print("  - Treina 3 variants: EER, LogReg, KDE")
    print(f"  - {n_trials} trials por σ_true (paper usa 1000)\n")

    rng = np.random.default_rng(seed)

    print(
        f"{'σ_true':>8} {'variant':>8} | "
        f"{'mean(σ_est)':>13} {'std(σ_est)':>13} {'RMS_ours':>10} | "
        f"{'RMS_paper':>10} {'Δ':>8}"
    )
    print("-" * 90)

    for sigma_true in [1.0, 2.0, 3.0, 4.0]:
        results: List[TrialResult] = []
        t0 = time.time()
        for _ in range(n_trials):
            scores, labels = _generate_bi_gaussian_sample(
                sigma_true=sigma_true, n_same=100, n_diff=4950, rng=rng,
            )
            sigma_eer = _train_one_variant(
                scores, labels,
                target_metric=BiGaussTargetMetric.EER,
                initial_calibrator=BiGaussInitialCalibrator.NONE,
            )
            sigma_logreg = _train_one_variant(
                scores, labels,
                target_metric=BiGaussTargetMetric.CLLR,
                initial_calibrator=BiGaussInitialCalibrator.LOGREG,
            )
            sigma_kde = _train_one_variant(
                scores, labels,
                target_metric=BiGaussTargetMetric.CLLR,
                initial_calibrator=BiGaussInitialCalibrator.KDE,
            )
            results.append(TrialResult(
                sigma_true=sigma_true,
                sigma_est_eer=sigma_eer,
                sigma_est_logreg=sigma_logreg,
                sigma_est_kde=sigma_kde,
            ))
        elapsed = time.time() - t0

        for variant_label, attr in [
            ("EER", "sigma_est_eer"),
            ("LogReg", "sigma_est_logreg"),
            ("KDE", "sigma_est_kde"),
        ]:
            ests = np.array(
                [getattr(r, attr) for r in results if getattr(r, attr) is not None],
                dtype=np.float64,
            )
            if ests.size == 0:
                print(
                    f"{sigma_true:8.0f} {variant_label:>8} | "
                    f"{'(falhou)':>13} {'(falhou)':>13}"
                )
                continue
            mean_est = float(ests.mean())
            std_est = float(ests.std(ddof=1)) if ests.size > 1 else 0.0
            rms = float(np.sqrt(np.mean((ests - sigma_true) ** 2)))
            paper_rms = PAPER_TABLE_1_RMS[int(sigma_true)][variant_label]
            delta = rms - paper_rms
            sign = "+" if delta >= 0 else ""
            print(
                f"{sigma_true:8.0f} {variant_label:>8} | "
                f"{mean_est:13.4f} {std_est:13.4f} {rms:10.4f} | "
                f"{paper_rms:10.4f} {sign}{delta:7.4f}"
            )
        print(f"  ({elapsed:.1f}s para σ_true={sigma_true:.0f})")
        print()

    print("Conclusão esperada:")
    print("  - Mean(σ_est) deve ficar próximo de σ_true (com leve viés positivo,")
    print("    relatado no paper §2.6 página 12).")
    print("  - RMS_ours deve estar dentro de ~30% de RMS_paper (dispersão de")
    print("    Monte Carlo com n=100 vs n=1000 do paper).")
    print("  - Δ negativo seria suspeitamente bom; Δ positivo grande seria bug.")


# ───────────────────────────────────────────────────────────────────────────
# Experimento 3: Sanity check do mapeamento F⁻¹(G(x))
# ───────────────────────────────────────────────────────────────────────────

def experiment_3_perfect_input_identity(seed: int = 42) -> None:
    """Para input perfeitamente bi-gaussiano com σ=3, o output do BiGauss
    deve ser aproximadamente a função identidade (paper §3.2)."""
    print("\n" + "=" * 76)
    print("EXPERIMENTO 3 — Mapping em input perfeitamente calibrado (paper §3.2)")
    print("=" * 76)
    print("\nGera 1000 same + 49500 different de bi-gauss σ=3 perfeito.")
    print("Treina BiGauss(LOGREG init), mapeia o próprio treino, mede:")
    print("  - σ_target recuperado (esperado: ≈ 3)")
    print("  - Cllr antes vs depois (esperado: aproximadamente igual)")
    print("  - Output ≈ identidade nos extremos do range")
    print()

    rng = np.random.default_rng(seed)
    sigma_true = 3.0
    scores, labels = _generate_bi_gaussian_sample(
        sigma_true=sigma_true, n_same=1000, n_diff=49500, rng=rng,
    )

    cfg = BiGaussianizedCalibrationModelConfig(
        target_metric=BiGaussTargetMetric.CLLR,
        initial_calibrator=BiGaussInitialCalibrator.LOGREG,
        inversion_method=BiGaussInversionMethod.INTERP1D,
        sigma_shrinkage_factor=1.0,
    )
    model = BiGaussianizedCalibrationModel(config=cfg)
    model.fit(scores=scores, labels=labels, logger_callback=lambda _msg: None)

    sigma_est = float(model.sigma_target)
    print(f"σ_target recuperado: {sigma_est:.4f}  (esperado: ≈ {sigma_true:.1f})")
    print(f"Erro relativo:       {100.0 * abs(sigma_est - sigma_true) / sigma_true:.2f}%")

    # Cllr antes/depois
    cllr_before = BiGaussianizedCalibrationModel._compute_cllr(
        scores[labels == 1], scores[labels == 0]
    )
    calibrated = model.calibrate_scores(scores)
    cllr_after = BiGaussianizedCalibrationModel._compute_cllr(
        calibrated[labels == 1], calibrated[labels == 0]
    )
    print(f"\nCllr antes (input perfeito):  {cllr_before:.6f}")
    print(f"Cllr depois (output BiGauss): {cllr_after:.6f}")
    print(f"Razão after/before:           {cllr_after / cllr_before:.4f}")
    print("Esperado: razão ≈ 1.0 (input já perfeito → BiGauss não deve degradar)")

    # Mapeamento ≈ identidade
    test_xs = np.linspace(-15.0, 15.0, 7)
    mapped = model.calibrate_scores(test_xs)
    print(f"\n{'x':>10} {'BiGauss(x)':>15} {'|Δ| de y=x':>15}")
    print("-" * 45)
    for x, y in zip(test_xs, mapped):
        print(f"{x:10.3f} {y:15.4f} {abs(y - x):15.4f}")
    print("Esperado: BiGauss(x) ≈ x para x dentro do range do treino.")


# ───────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────

def main() -> int:
    print("Validação numérica BiGaussianized vs Morrison (2024)")
    print(f"  Constantes do fit: b={_CLLR_FIT_B}, c={_CLLR_FIT_C}")
    print()

    t0 = time.time()
    experiment_1_fundamental_equations()
    experiment_2_table1_reproduction(n_trials=100)
    experiment_3_perfect_input_identity()
    print(f"\nTempo total: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
