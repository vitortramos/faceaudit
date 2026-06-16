"""Regression tests para os 12 patches da terceira rodada da revisão
externa (2026-06-09 round 3).

P0-R3-1: `percentile_basis_score_type` propagado ao runtime + DTO rejeita
         valores não suportados pelo backend.
P0-R3-2: `ReferenceScoreCalibrationService` valida backend declarado vs
         instância produzida pela factory.
P0-R3-3: helper único `_build_model(seed=...)` substitui `trainer_factory()`
         direto em peek/fold/refit; `_factory_accepts_seed` via inspect.
P0-R3-4: `n_folds >= 2` rejeitado; `identity_ids` normalizado a `str`.
P0-R3-5: `validate_reference_score_distribution` exige weights+cdf
         pareados e CDF = cumsum(weights)/sum.
P0-R3-6: telemetria de normalização separada por escopo
         (`_fit_unit_norm_telemetry` vs `_inference_unit_norm_telemetry_last`).
P1-R3-7: arrays internos do state são read-only após fit / install /
         from_dict (k-NN + MoVMF).
P1-R3-8: `_log_iv_robust` falha não cai em uniforme — `RuntimeError`.
P1-R3-9: `ReferenceScoreCalibrationResult.oof_cdf` renomeado para
         `oof_reference_scores_sorted` + novos `oof_reference_weights_sorted`
         e `oof_reference_cdf`; alias deprecated com warning; arrays
         expostos no result são read-only.
P1-R3-10: `apply_unit_norm_policy` rejeita `tolerance`/`eps` não finitos
          ou <= 0.
P1-R3-11: default de `reference_score_calibration_mode` no DTO k-NN é
          `CROSS_FIT_IDENTITY` para estratégias novas.
P1-R3-12: docstrings e comentário do k-means++ atualizados (cobertura
          formal, não testável; cobre-se via lint manual).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_percentile_basis import TypicalityPercentileBasis
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.models.typicality.knn_typicality_model_config import (
    KNNTypicalityModelConfig,
)
from faceaudit.models.typicality.movmf_typicality_model_config import (
    MoVMFTypicalityModelConfig,
)
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import (
    MoVMFTypicalityModel,
    _MoVMFParams,
)
from faceaudit.services.reference_score_calibration_service import (
    BackendMismatchError,
    ReferenceScoreCalibrationResult,
    ReferenceScoreCalibrationService,
)
from faceaudit.services.reference_score_distribution_validator import (
    InvalidReferenceDistributionError,
    validate_reference_score_distribution,
)
from faceaudit.services.unit_norm_validator import apply_unit_norm_policy


# ─── fixtures ──────────────────────────────────────────────────────────


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


# ─── P0-R3-1 ───────────────────────────────────────────────────────────


def test_knn_config_rejects_log_density_estimate_as_percentile_basis():
    """k-NN só implementa MEAN_KNN_SIMILARITY como base de percentil."""
    with pytest.raises(ValueError, match="MEAN_KNN_SIMILARITY"):
        KNNTypicalityModelConfig(
            percentile_basis_score_type=TypicalityPercentileBasis.LOG_DENSITY_ESTIMATE,
        )


def test_movmf_config_rejects_mean_knn_similarity_as_percentile_basis():
    """MoVMF só implementa LOG_DENSITY_ESTIMATE como base de percentil."""
    with pytest.raises(ValueError):
        MoVMFTypicalityModelConfig(
            percentile_basis_score_type=TypicalityPercentileBasis.MEAN_KNN_SIMILARITY,
        )


# ─── P0-R3-2 ───────────────────────────────────────────────────────────


def test_cross_fit_rejects_factory_producing_mismatched_backend():
    """Serviço sinaliza erro quando factory devolve backend diferente do
    declarado (P0-R3-2)."""
    X, ids = _make_data()

    def bad_factory(**_):  # declara MOVMF mas devolve k-NN
        return KNNTypicalityModel(k=3, ood_quantile=0.0)

    with pytest.raises(BackendMismatchError):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.MOVMF,
            trainer_factory=bad_factory,
            embeddings=X,
            identity_ids=ids,
            n_folds=3,
            random_state=0,
        )


# ─── P0-R3-3 + P1-R3-12 ────────────────────────────────────────────────


def test_cross_fit_passes_seed_to_factory_when_accepted():
    """Helper `_build_model` injeta seed quando factory aceita
    (verificado via `inspect.signature` — P0-R3-3)."""
    X, ids = _make_data()
    captured_seeds = []

    def factory(*, seed=None):
        captured_seeds.append(seed)
        return MoVMFTypicalityModel(
            n_components=1, random_state=seed if seed is not None else 0,
            ood_quantile=0.0,
        )

    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.MOVMF,
        trainer_factory=factory,
        embeddings=X,
        identity_ids=ids,
        n_folds=3,
        random_state=10,
    )
    # Fold seeds usados foram propagados.
    assert {10, 11, 12}.issubset(set(captured_seeds))
    assert result.refit_model is not None


# ─── P0-R3-4 ───────────────────────────────────────────────────────────


def test_cross_fit_rejects_n_folds_below_two():
    X, ids = _make_data()
    factory = lambda **_: KNNTypicalityModel(k=3, ood_quantile=0.0)  # noqa: E731
    with pytest.raises(ValueError, match="n_folds"):
        ReferenceScoreCalibrationService().calibrate(
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            backend=TypicalityModelType.KNN,
            trainer_factory=factory,
            embeddings=X,
            identity_ids=ids,
            n_folds=1,
            random_state=0,
        )


def test_cross_fit_normalizes_identity_ids_to_str():
    """ids inteiros passam — são normalizados a `str` antes do fit
    (P0-R3-4). Não pode levantar TypeError."""
    rng = np.random.default_rng(0)
    n_id, per, d = 6, 12, 12
    X, ids = [], []
    for i in range(n_id):
        c = rng.normal(size=d); c /= np.linalg.norm(c)
        for _ in range(per):
            v = c + 0.2 * rng.normal(size=d); v /= np.linalg.norm(v)
            X.append(v); ids.append(i)  # int, não str
    X = np.asarray(X); ids = tuple(ids)
    factory = lambda **_: KNNTypicalityModel(k=3, ood_quantile=0.0)  # noqa: E731
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    assert result.refit_model is not None


# ─── P0-R3-5 ───────────────────────────────────────────────────────────


def test_validator_rejects_weights_without_cdf():
    scores = np.array([0.1, 0.5, 0.9])
    weights = np.array([1.0, 2.0, 3.0])
    with pytest.raises(InvalidReferenceDistributionError):
        validate_reference_score_distribution(
            scores=scores, weights=weights, cdf=None,
        )


def test_validator_rejects_cdf_incoherent_with_weights():
    scores = np.array([0.1, 0.5, 0.9])
    weights = np.array([1.0, 2.0, 3.0])
    # CDF errada: deveria ser cumsum/6 = [1/6, 3/6, 6/6]
    cdf_wrong = np.array([0.3, 0.6, 1.0])
    with pytest.raises(InvalidReferenceDistributionError, match="incoerente"):
        validate_reference_score_distribution(
            scores=scores, weights=weights, cdf=cdf_wrong,
        )


def test_validator_accepts_coherent_weights_and_cdf():
    scores = np.array([0.1, 0.5, 0.9])
    weights = np.array([1.0, 2.0, 3.0])
    total = weights.sum()
    cdf_ok = np.cumsum(weights) / total
    # Deve passar silenciosamente.
    validate_reference_score_distribution(
        scores=scores, weights=weights, cdf=cdf_ok,
    )


# ─── P0-R3-6 ───────────────────────────────────────────────────────────


def test_knn_fit_and_inference_telemetry_are_independent_dicts():
    """fit() popula `_fit_unit_norm_telemetry`; chamadas de scoring
    populam `_inference_unit_norm_telemetry_last`. Não devem
    sobrescrever-se mutuamente (P0-R3-6)."""
    X, ids = _make_data(d=16)
    m = KNNTypicalityModel(
        k=3, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    fit_tel = dict(m._fit_unit_norm_telemetry or {})
    # Inferência com input fora-da-esfera para popular tel de inferência.
    probe = np.random.default_rng(0).normal(size=(5, 16))
    m.get_percentiles(probe)
    infer_tel = dict(m._inference_unit_norm_telemetry_last or {})
    # Os dicts são separados — fit_tel não muta com inferência.
    assert m._fit_unit_norm_telemetry == fit_tel
    # E ambos têm conteúdo (n_total > 0).
    assert fit_tel.get("n_total", 0) > 0
    assert infer_tel.get("n_total", 0) > 0


# ─── P1-R3-7 ───────────────────────────────────────────────────────────


def test_knn_state_arrays_are_read_only_after_fit():
    X, _ = _make_data()
    m = KNNTypicalityModel(
        k=3, ood_quantile=0.0,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m.fit(X)
    st = m._state
    with pytest.raises(ValueError, match="read-only"):
        st.reference[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        st.loo_mean_knn_sorted[0] = 999.0


def test_movmf_params_are_read_only_after_fit():
    X, _ = _make_data()
    m = MoVMFTypicalityModel(n_components=2, random_state=0)
    m.fit(X)
    p = m._params
    with pytest.raises(ValueError, match="read-only"):
        p.mus[0, 0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        p.kappas[0] = 999.0
    with pytest.raises(ValueError, match="read-only"):
        p.weights[0] = 0.0


def test_movmf_state_arrays_read_only_after_from_dict():
    X, _ = _make_data()
    m = MoVMFTypicalityModel(n_components=2, random_state=0)
    m.fit(X)
    payload = m.to_dict()
    m2 = MoVMFTypicalityModel.from_dict(payload)
    with pytest.raises(ValueError, match="read-only"):
        m2._params.mus[0, 0] = 0.0
    if m2._train_log_likelihoods is not None:
        with pytest.raises(ValueError, match="read-only"):
            m2._train_log_likelihoods[0] = 0.0


# ─── P1-R3-9 ───────────────────────────────────────────────────────────


def test_cross_fit_result_exposes_new_oof_reference_fields():
    """Result expõe `oof_reference_scores_sorted` +
    `oof_reference_weights_sorted` + `oof_reference_cdf`; `oof_cdf` é
    alias deprecated (P1-R3-9)."""
    X, ids = _make_data()
    factory = lambda **_: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    # Novos campos presentes e não-None em CROSS_FIT.
    assert result.oof_reference_scores_sorted is not None
    assert result.oof_reference_weights_sorted is not None
    assert result.oof_reference_cdf is not None
    # Coerência: cumsum(weights)/sum == cdf.
    w = result.oof_reference_weights_sorted
    expected = np.cumsum(w) / w.sum()
    np.testing.assert_allclose(result.oof_reference_cdf, expected, atol=1e-12)
    # Scores sorted ascending.
    assert np.all(np.diff(result.oof_reference_scores_sorted) >= 0)
    # Arrays expostos no result são read-only.
    with pytest.raises(ValueError, match="read-only"):
        result.oof_reference_scores_sorted[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.oof_reference_weights_sorted[0] = 0.0
    with pytest.raises(ValueError, match="read-only"):
        result.oof_reference_cdf[0] = 0.0


def test_oof_cdf_alias_emits_deprecation_warning():
    X, ids = _make_data()
    factory = lambda **_: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=3, random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = result.oof_cdf
    assert any(
        issubclass(w.category, DeprecationWarning) and "oof_cdf" in str(w.message)
        for w in caught
    )


def test_in_sample_legacy_result_has_no_oof_fields():
    X, ids = _make_data()
    factory = lambda **_: KNNTypicalityModel(  # noqa: E731
        k=3, ood_quantile=0.0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    result = ReferenceScoreCalibrationService().calibrate(
        mode=ReferenceScoreCalibrationMode.IN_SAMPLE_LEGACY,
        backend=TypicalityModelType.KNN,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=0, random_state=0,
    )
    assert result.oof_reference_scores_sorted is None
    assert result.oof_reference_weights_sorted is None
    assert result.oof_reference_cdf is None


# ─── P1-R3-10 ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad_tolerance", [0.0, -1e-6, np.nan, np.inf])
def test_apply_unit_norm_policy_rejects_invalid_tolerance(bad_tolerance):
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="tolerance"):
        apply_unit_norm_policy(
            x, policy=UnitNormPolicy.ERROR, tolerance=bad_tolerance,
        )


@pytest.mark.parametrize("bad_eps", [0.0, -1e-6, np.nan, np.inf])
def test_apply_unit_norm_policy_rejects_invalid_eps(bad_eps):
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="eps"):
        apply_unit_norm_policy(
            x, policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            tolerance=1e-4, eps=bad_eps,
        )


# ─── P1-R3-11 ──────────────────────────────────────────────────────────


def test_knn_config_default_calibration_mode_is_cross_fit():
    """Novas estratégias k-NN nascem em CROSS_FIT_IDENTITY (P1-R3-11)."""
    cfg = KNNTypicalityModelConfig()
    assert (
        cfg.reference_score_calibration_mode
        == ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY
    )
