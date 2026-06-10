"""Regression tests para os 5 P0 da segunda rodada da revisão externa
(2026-06-09 round 2).

P0-R2-1: razão de Bessel/κ em alta dimensão (d=512, d=1024).
P0-R2-2: cross-fit usa `reference_score_basis_type` por backend.
P0-R2-3: IDENTITY_CENTROIDS agrega EMBEDDINGS por id, não scores.
P0-R2-4: reference distribution fingerprint inclui pesos + CDF + threshold.
P0-R2-5: kNN state fingerprint inclui reference_score_weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)


def _make_data(n_id=12, per=15, d=16, seed=0):
    rng = np.random.default_rng(seed)
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d)
        c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.18 * rng.normal(size=d)
            v /= np.linalg.norm(v)
            X.append(v)
            ids.append(f"id_{i}")
    return np.asarray(X), tuple(ids)


# ─── P0-R2-1 ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("d,kappa,expected_ratio", [
    (512, 20,    0.0390),
    (512, 100,   0.1884),
    (512, 341,   0.4999),
    (512, 500,   0.6117),
    (512, 1000,  0.7765),
    (512, 1400,  0.8340),
])
def test_A_d_high_dim_matches_perron_ratio(d, kappa, expected_ratio):
    """`_A_d` recupera a razão correta de I_{ν+1}/I_ν em d=512 mesmo
    onde scipy.ive faz underflow."""
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
    )
    a = m._A_d(float(kappa), d)
    assert abs(a - expected_ratio) < 1e-3, (
        f"_A_d(κ={kappa}, d={d}) = {a}, esperado ≈ {expected_ratio}"
    )


def test_estimate_kappa_high_dim_does_not_collapse():
    """`_estimate_kappa_robust` não colapsa em d=512: r̄ distintos
    produzem κ distintos."""
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
    )
    d = 512
    kappas = [m._estimate_kappa_robust(r, d) for r in [0.01, 0.1, 0.3, 0.5, 0.7]]
    # Pelo menos 4 valores distintos (antes colapsava em 1)
    assert len(set(round(k, 2) for k in kappas)) >= 4


def test_estimate_kappa_d_1024_scales():
    """Em d=1024 a recuperação também não colapsa."""
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
    )
    d = 1024
    kappas = [m._estimate_kappa_robust(r, d) for r in [0.05, 0.2, 0.5, 0.8]]
    assert len(set(round(k, 2) for k in kappas)) == 4


def test_log_iv_robust_does_not_underflow_in_high_dim():
    """log I_ν(κ) em d=512 com κ pequeno NÃO retorna -inf."""
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
    )
    nu = 256.0 - 1
    for kappa in [10.0, 50.0, 100.0, 500.0]:
        log_iv = m._log_iv_robust(nu, kappa)
        assert np.isfinite(log_iv), (
            f"log I_ν({nu}, {kappa}) = {log_iv} — Olver fallback falhou"
        )


# ─── P0-R2-2 ─────────────────────────────────────────────────────────────


def test_reference_score_basis_type_knn():
    m = KNNTypicalityModel(k=5)
    assert m.reference_score_basis_type == TypicalityScoreType.MEAN_KNN_SIMILARITY


def test_reference_score_basis_type_movmf():
    m = MoVMFTypicalityModel(n_components=1, centering="none")
    assert m.reference_score_basis_type == TypicalityScoreType.LOG_DENSITY_ESTIMATE


def test_cross_fit_knn_uses_mean_knn_similarity_not_log_density():
    """Em KNN, a CDF OOF é construída sobre T_k (MEAN_KNN_SIMILARITY),
    não LOG_DENSITY_ESTIMATE. Antes, todos os percentis caíam em 100%
    porque a escala instalada era incompatível com a consultada pelo
    `_compute_info_single`."""
    X, ids = _make_data(n_id=12, per=8)

    def trainer(seed=None):
        return KNNTypicalityModel(
            k=5, ood_quantile=0.1,
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.KNN,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=3, random_state=0,
    )
    # T_k vive em [-1, +1]; LOG_DENSITY (que era escolhido antes) vive
    # em ~[-50, +50]. Se range OOF está em [-1, +1] → base correta.
    assert result.oof_cdf.min() >= -1.0
    assert result.oof_cdf.max() <= 1.0

    # Percentis no domínio do treino devem ser bem distribuídos —
    # NÃO todos em 100% (que era o sintoma do bug).
    refit = result.refit_model
    percentiles = refit.get_percentiles(X[:30])
    assert percentiles.min() < 99.0, (
        f"percentil mínimo {percentiles.min()} — bug do round 1 reaparece"
    )
    # Cobertura razoável de valores
    assert percentiles.max() - percentiles.min() > 10.0


# ─── P0-R2-3 ─────────────────────────────────────────────────────────────


def test_centroid_oof_aggregates_embeddings_not_scores():
    """Em IDENTITY_CENTROIDS, o número de scores na CDF OOF é exatamente
    o número de identidades únicas (1 score por id, não média de scores
    por id)."""
    X, ids = _make_data(n_id=10, per=6)
    n_unique = len(set(ids))

    def trainer(seed=None):
        return MoVMFTypicalityModel(
            n_components=1, centering="none",
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            # R7: factory deve propagar a seed para passar pela
            # validação `EffectiveSeedMismatchError`.
            random_state=seed if seed is not None else 0,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=5, random_state=0,
    )
    assert len(result.oof_cdf) == n_unique


def test_centroid_oof_differs_from_score_mean_aggregation():
    """O score do centroide NÃO é igual à média dos scores das faces
    (verificação numérica direta) — confirma que P0-R2-3 mudou
    semântica."""
    X, ids = _make_data(n_id=8, per=5)

    # Modelo único pra calibrar
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    # Identidade alvo: pega faces dela
    target_id = ids[0]
    mask = np.array([i == target_id for i in ids])
    faces = X[mask]

    # Centroide L2-normalizado
    centroid = faces.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    centroid = centroid.reshape(1, -1)

    score_centroid = m.score_samples(centroid)[0]
    score_face_mean = m.score_samples(faces).mean()

    # Em corpora não-triviais a diferença é mensurável
    assert abs(score_centroid - score_face_mean) > 1e-3


# ─── P0-R2-4 ─────────────────────────────────────────────────────────────


def test_movmf_reference_dist_fingerprint_includes_weights():
    """Mesmos scores, pesos diferentes → fingerprints diferentes."""
    X, ids = _make_data(n_id=8, per=8)
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    same_scores = np.linspace(-5, 5, 50)
    w_uniform = np.ones(50)
    w_skewed = np.linspace(0.1, 3.0, 50)

    m.install_reference_score_calibration(
        reference_scores=same_scores, reference_weights=w_uniform,
        ood_threshold=-3.0,
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
    )
    fp_uniform = m._reference_score_distribution_fingerprint()

    m.install_reference_score_calibration(
        reference_scores=same_scores, reference_weights=w_skewed,
        ood_threshold=-3.0,
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
    )
    fp_skewed = m._reference_score_distribution_fingerprint()

    assert fp_uniform != fp_skewed


def test_movmf_reference_dist_fingerprint_includes_calibration_mode():
    """Mesma distribuição instalada com modos diferentes → fingerprints
    diferentes."""
    X, ids = _make_data(n_id=8, per=8)
    m = MoVMFTypicalityModel(
        n_components=1, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)

    scores = np.linspace(-5, 5, 30)
    weights = np.ones(30)

    m.install_reference_score_calibration(
        reference_scores=scores, reference_weights=weights, ood_threshold=-3.0,
        calibration_mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
    )
    fp_cross = m._reference_score_distribution_fingerprint()

    m.install_reference_score_calibration(
        reference_scores=scores, reference_weights=weights, ood_threshold=-3.0,
        calibration_mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
    )
    fp_in_sample = m._reference_score_distribution_fingerprint()

    assert fp_cross != fp_in_sample


# ─── P0-R2-5 ─────────────────────────────────────────────────────────────


def test_knn_state_fingerprint_includes_reference_score_weights():
    """Alterar `reference_score_weights` muda o state fingerprint do
    KNN. Antes ficava igual apesar de T_k ponderado divergir."""
    from dataclasses import replace as dataclass_replace

    X, ids = _make_data(n_id=10, per=8)
    m = KNNTypicalityModel(
        k=5, ood_quantile=0.1,
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)

    fp_orig = m.model_state_fingerprint
    orig_weights = m._state.reference_score_weights.copy()

    # Muda só os pesos (mantém tudo o resto) — diretamente no state
    new_weights = orig_weights * 2.0
    m._state = dataclass_replace(m._state, reference_score_weights=new_weights)
    fp_new = m.model_state_fingerprint

    assert fp_orig != fp_new
