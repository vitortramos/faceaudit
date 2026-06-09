"""Regression tests for P1 robustness items from external review (2026-06-09).

Cobertura:
- P1-7: _pending_sample_weights eliminado (variável local em fit).
- P1-9: unit_norm_validator rejeita NaN/inf/zero-norm em todas as policies.
- P1-10: enum from_str strict para strings desconhecidas.
- P1-11: contract fingerprint inclui unit_norm_tolerance + population_label.
- P1-12: fold_seeds são realmente passados ao trainer_factory.

P1-8 (frechet documentation) é só docstring, sem cobertura de teste.
"""
from __future__ import annotations

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
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)
from faceaudit.services.unit_norm_validator import (
    InvalidEmbeddingError,
    apply_unit_norm_policy,
)


def _make_data(n_id=8, per=12, d=16, seed=0):
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


# ─── P1-7 ────────────────────────────────────────────────────────────────


def test_movmf_no_pending_sample_weights_leak():
    """Não há campo `_pending_sample_weights` persistente; pesos são
    locais ao fit()."""
    X, ids = _make_data()
    m = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    # Campo eliminado em P1-7 — não persiste no objeto
    assert getattr(m, "_pending_sample_weights", None) is None


def test_movmf_fit_with_inv_freq_then_face_level_does_not_leak():
    """Segundo fit() em modo FACE_LEVEL_RAW após primeiro fit() em
    INVERSE_IDENTITY_FREQUENCY não herda pesos."""
    X, ids = _make_data()
    m1 = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m1.fit(X, identity_ids=ids)
    assert m1._train_log_likelihoods_weights is not None

    # Modelo separado em FACE_LEVEL_RAW não deve ter pesos.
    m2 = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m2.fit(X)
    assert m2._train_log_likelihoods_weights is None


# ─── P1-9 ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("policy", [
    UnitNormPolicy.ERROR,
    UnitNormPolicy.NORMALIZE_WITH_WARNING,
    UnitNormPolicy.NORMALIZE_SILENTLY,
])
def test_unit_norm_rejects_nan_in_all_policies(policy):
    x = np.array([[1.0, 0.0], [np.nan, 0.5]])
    with pytest.raises(InvalidEmbeddingError) as ei:
        apply_unit_norm_policy(x, policy=policy, tolerance=1e-3)
    assert ei.value.kind == "nonfinite"


@pytest.mark.parametrize("policy", [
    UnitNormPolicy.ERROR,
    UnitNormPolicy.NORMALIZE_WITH_WARNING,
    UnitNormPolicy.NORMALIZE_SILENTLY,
])
def test_unit_norm_rejects_inf_in_all_policies(policy):
    x = np.array([[1.0, 0.0], [np.inf, 0.5]])
    with pytest.raises(InvalidEmbeddingError) as ei:
        apply_unit_norm_policy(x, policy=policy, tolerance=1e-3)
    assert ei.value.kind == "nonfinite"


@pytest.mark.parametrize("policy", [
    UnitNormPolicy.ERROR,
    UnitNormPolicy.NORMALIZE_WITH_WARNING,
    UnitNormPolicy.NORMALIZE_SILENTLY,
])
def test_unit_norm_rejects_near_zero_norm_in_all_policies(policy):
    x = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(InvalidEmbeddingError) as ei:
        apply_unit_norm_policy(x, policy=policy, tolerance=1e-3)
    assert ei.value.kind == "near_zero_norm"


# ─── P1-10 ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("enum_cls,garbage", [
    (TypicalityModelType, "IDENTITY_CENTROD"),
    (TypicalityScoreType, "LOG_DENSITY_KNNN"),
    (PopulationWeightingMode, "GARBAGE"),
    (UnitNormPolicy, "NORMALIZE_NOPE"),
    (ReferenceScoreCalibrationMode, "ZZZ"),
    (TypicalityPercentileBasis, "WAT"),
])
def test_enum_from_str_strict_on_unknown(enum_cls, garbage):
    """P1-10: from_str levanta em string desconhecida em vez de cair
    em default silencioso."""
    with pytest.raises(ValueError, match="desconhecido"):
        enum_cls.from_str(garbage)


@pytest.mark.parametrize("enum_cls", [
    TypicalityModelType, TypicalityScoreType, PopulationWeightingMode,
    UnitNormPolicy, ReferenceScoreCalibrationMode, TypicalityPercentileBasis,
])
def test_enum_from_str_empty_returns_default(enum_cls):
    """from_str(None) e from_str('') ainda retornam default — back-compat
    de JSONs antigos sem o campo."""
    # Não levanta
    _ = enum_cls.from_str(None)
    _ = enum_cls.from_str("")


# ─── P1-11 ───────────────────────────────────────────────────────────────


def test_movmf_contract_fingerprint_sensitive_to_unit_norm_tolerance():
    """Dois modelos identicos exceto pela tolerance → contract fingerprints diferentes."""
    X, ids = _make_data()

    def make(tol):
        m = MoVMFTypicalityModel(
            n_components=2, n_init=2, centering="none",
            unit_norm_policy="NORMALIZE_SILENTLY",
            unit_norm_tolerance=tol,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )
        m.fit(X, identity_ids=ids)
        return m

    m1 = make(1e-4)
    m2 = make(1e-3)
    assert m1.model_contract_fingerprint != m2.model_contract_fingerprint


def test_movmf_contract_fingerprint_sensitive_to_population_label():
    X, ids = _make_data()
    m1 = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        population_label="FRONTAL",
    )
    m1.fit(X, identity_ids=ids)
    m2 = MoVMFTypicalityModel(
        n_components=2, n_init=2, centering="none",
        unit_norm_policy="NORMALIZE_SILENTLY",
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        population_label="PROFILE",
    )
    m2.fit(X, identity_ids=ids)
    assert m1.model_contract_fingerprint != m2.model_contract_fingerprint


def test_knn_contract_fingerprint_sensitive_to_unit_norm_tolerance():
    X, ids = _make_data()
    m1 = KNNTypicalityModel(
        k=5, ood_quantile=0.1, unit_norm_policy="NORMALIZE_SILENTLY",
        unit_norm_tolerance=1e-4,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m1.fit(X, identity_ids=ids)
    m2 = KNNTypicalityModel(
        k=5, ood_quantile=0.1, unit_norm_policy="NORMALIZE_SILENTLY",
        unit_norm_tolerance=1e-3,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m2.fit(X, identity_ids=ids)
    assert m1.model_contract_fingerprint != m2.model_contract_fingerprint


# ─── P1-12 ───────────────────────────────────────────────────────────────


def test_cross_fit_passes_fold_seed_to_trainer_factory():
    """Trainer factory recebe `seed=fold_seed` quando aceita o kwarg.

    P0-R3-3 (round 3): helper único `_build_model` instancia peek,
    folds e refit. Peek e refit recebem `seed=random_state`
    principal (10); folds recebem `random_state + fold_idx`
    (10, 11, 12 para n_folds=3).
    """
    X, ids = _make_data(n_id=10, per=8)
    captured_seeds = []

    def trainer(seed=None):
        captured_seeds.append(seed)
        return MoVMFTypicalityModel(
            n_components=1, centering="none",
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
            random_state=seed if seed is not None else 42,
        )

    svc = ReferenceScoreCalibrationService()
    result = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=3, random_state=10,
    )
    # Esperado: peek (10) + 3 folds (10, 11, 12) + refit (10).
    # Todas as chamadas via helper repassam seed (não há mais
    # chamada sem seed).
    assert None not in captured_seeds
    # Seeds dos folds presentes no conjunto.
    assert {10, 11, 12}.issubset(set(captured_seeds))
    assert result.fold_seeds == (10, 11, 12)


def test_cross_fit_back_compat_with_factory_without_seed():
    """Factory sem kwarg `seed` ainda funciona (back-compat)."""
    X, ids = _make_data(n_id=10, per=8)

    def trainer():
        return MoVMFTypicalityModel(
            n_components=1, centering="none",
            unit_norm_policy="NORMALIZE_SILENTLY",
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )

    svc = ReferenceScoreCalibrationService()
    # Não levanta
    _ = svc.calibrate(
        backend=TypicalityModelType.MOVMF,
        trainer_factory=trainer,
        embeddings=X, identity_ids=ids,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        n_folds=3, random_state=10,
    )
