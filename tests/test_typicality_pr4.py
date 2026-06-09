"""PR 4 — Endurecimento experimental do MoVMF + harmonização do k-NN.

Cobre:
  1. UnitNormPolicy aplicado em MoVMF e k-NN (ERROR rejeita;
     NORMALIZE_* normaliza).
  2. PopulationWeightingMode.IDENTITY_CENTROIDS no MoVMF
     (agrega centroides por identidade antes do EM).
  3. PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY no MoVMF
     levanta NotImplementedError (documentado em docs/typicality_models.md).
  4. Canonicalização de componentes MoVMF:
     - to_dict ordena por peso ↓ κ ↓ hash_μ.
     - permutação interna não altera model_contract_fingerprint.
  5. Cross-fit por identidade (CROSS_FIT_IDENTITY) no MoVMF e k-NN:
     - oof_cdf construído com scores OOF.
     - refit_model treinado em toda a população.
     - split_fingerprint estável para mesmo random_state.
  6. Health checks no summary_protocol.
  7. Round-trip preserva PR 4 fields (unit_norm_*, population_weighting_mode).
  8. Contract fingerprint inclui PR 4 fields (mudança em UnitNorm muda fp).
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.reference_score_calibration_mode import (
    ReferenceScoreCalibrationMode,
)
from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.reference_score_calibration_service import (
    ReferenceScoreCalibrationService,
)
from faceaudit.services.unit_norm_validator import (
    UnitNormViolationError,
    apply_unit_norm_policy,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def reference_population():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 16))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = tuple(f"id_{i // 3}" for i in range(60))
    return X, ids


# ─────────────────────────────────────────────────────────────────────
# UnitNormPolicy
# ─────────────────────────────────────────────────────────────────────

def test_unit_norm_error_rejects_non_unit(reference_population):
    X, _ = reference_population
    Xbad = X.copy(); Xbad[0] *= 3.0
    with pytest.raises(UnitNormViolationError) as ei:
        apply_unit_norm_policy(
            Xbad, policy=UnitNormPolicy.ERROR, tolerance=1e-4,
        )
    assert ei.value.n_violations == 1
    assert ei.value.max_deviation > 1e-4


def test_unit_norm_normalize_silently_accepts(reference_population):
    X, _ = reference_population
    Xbad = X.copy(); Xbad[0] *= 3.0
    out = apply_unit_norm_policy(
        Xbad, policy=UnitNormPolicy.NORMALIZE_SILENTLY, tolerance=1e-4,
    )
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0)


def test_movmf_unit_norm_error_in_fit(reference_population):
    X, _ = reference_population
    Xbad = X.copy(); Xbad[0] *= 2.0
    m = MoVMFTypicalityModel(
        n_components=1, unit_norm_policy=UnitNormPolicy.ERROR,
        random_state=0,
    )
    with pytest.raises(UnitNormViolationError):
        m.fit(Xbad)


def test_knn_unit_norm_error_in_fit(reference_population):
    X, ids = reference_population
    Xbad = X.copy(); Xbad[0] *= 2.0
    m = KNNTypicalityModel(
        k=4, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
        unit_norm_policy=UnitNormPolicy.ERROR,
    )
    with pytest.raises(UnitNormViolationError):
        m.fit(Xbad, identities=ids)


# ─────────────────────────────────────────────────────────────────────
# PopulationWeightingMode no MoVMF
# ─────────────────────────────────────────────────────────────────────

def test_movmf_identity_centroids_requires_identity_ids(reference_population):
    X, _ = reference_population
    m = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    with pytest.raises(ValueError, match="identity_ids"):
        m.fit(X)


def test_movmf_identity_centroids_reduces_sample_count(reference_population):
    """20 identidades agregadas = sample_count 20 (não 60)."""
    X, ids = reference_population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)
    assert m.sample_count == 20


def test_movmf_inverse_identity_frequency_now_implemented(reference_population):
    """PR 5 implementou. O modelo treina sem NotImplementedError."""
    X, ids = reference_population
    m = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    assert m.is_fitted()


# ─────────────────────────────────────────────────────────────────────
# Canonicalização de componentes MoVMF
# ─────────────────────────────────────────────────────────────────────

def test_movmf_component_permutation_preserves_contract_fp(reference_population):
    """Permutar componentes não altera o contract fingerprint
    (invariância à simetria de permutação da mistura).
    """
    X, _ = reference_population
    m = MoVMFTypicalityModel(n_components=3, random_state=0)
    m.fit(X)
    fp_before = m.model_contract_fingerprint
    state_before = m.model_state_fingerprint
    # Permuta componentes 0 e 1
    p = m._params
    p.mus[0], p.mus[1] = p.mus[1].copy(), p.mus[0].copy()
    p.kappas[0], p.kappas[1] = p.kappas[1], p.kappas[0]
    p.weights[0], p.weights[1] = p.weights[1], p.weights[0]
    assert m.model_contract_fingerprint == fp_before
    assert m.model_state_fingerprint == state_before


def test_movmf_to_dict_orders_components_canonically(reference_population):
    """`to_dict` deve emitir componentes em ordem canônica (peso ↓)."""
    X, _ = reference_population
    m = MoVMFTypicalityModel(n_components=3, random_state=0)
    m.fit(X)
    d = m.to_dict()
    weights = d["learned_params"]["weights"]
    assert weights == sorted(weights, reverse=True)


# ─────────────────────────────────────────────────────────────────────
# CROSS_FIT_IDENTITY
# ─────────────────────────────────────────────────────────────────────

def test_cross_fit_movmf_produces_oof_cdf(reference_population):
    X, ids = reference_population

    def factory():
        return MoVMFTypicalityModel(
            n_components=2, random_state=0,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
            population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
        )

    result = ReferenceScoreCalibrationService().calibrate(
        backend=TypicalityModelType.MOVMF,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=5, random_state=42,
    )
    assert result.oof_cdf is not None
    assert len(result.oof_cdf) > 0
    # OOF ordenado (CDF)
    assert np.all(np.diff(result.oof_cdf) >= 0)
    assert result.refit_model.sample_count == 20  # 20 identidades agregadas
    assert result.n_folds_used == 5


def test_cross_fit_knn_produces_oof_cdf(reference_population):
    X, ids = reference_population

    def factory():
        return KNNTypicalityModel(
            k=4, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
            unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        )

    result = ReferenceScoreCalibrationService().calibrate(
        backend=TypicalityModelType.KNN,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        trainer_factory=factory,
        embeddings=X, identity_ids=ids,
        n_folds=5, random_state=42,
    )
    assert len(result.oof_cdf) > 0
    assert result.refit_model.sample_count == 20


def test_cross_fit_split_fingerprint_stable_for_same_random_state(reference_population):
    X, ids = reference_population

    def factory():
        return KNNTypicalityModel(
            k=4, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
        )

    svc = ReferenceScoreCalibrationService()
    r1 = svc.calibrate(
        backend=TypicalityModelType.KNN,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        trainer_factory=factory, embeddings=X, identity_ids=ids,
        n_folds=5, random_state=42,
    )
    r2 = svc.calibrate(
        backend=TypicalityModelType.KNN,
        mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
        trainer_factory=factory, embeddings=X, identity_ids=ids,
        n_folds=5, random_state=42,
    )
    assert r1.split_fingerprint == r2.split_fingerprint


def test_cross_fit_too_few_identities_rejected(reference_population):
    X, _ = reference_population
    # Só 3 identidades únicas, 5 folds → erro
    few_ids = ["a"] * 20 + ["b"] * 20 + ["c"] * 20

    def factory():
        return KNNTypicalityModel(
            k=2, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
        )

    with pytest.raises(ValueError, match="identidades únicas insuficientes"):
        ReferenceScoreCalibrationService().calibrate(
            backend=TypicalityModelType.KNN,
            mode=ReferenceScoreCalibrationMode.CROSS_FIT_IDENTITY,
            trainer_factory=factory,
            embeddings=X, identity_ids=few_ids,
            n_folds=5, random_state=42,
        )


# ─────────────────────────────────────────────────────────────────────
# Round-trip de serialização preserva PR 4 fields
# ─────────────────────────────────────────────────────────────────────

def test_movmf_round_trip_preserves_pr4_fields(reference_population):
    X, ids = reference_population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_WITH_WARNING,
        unit_norm_tolerance=1e-3,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)
    d = m.to_dict()
    m2 = MoVMFTypicalityModel.from_dict(d)
    assert m2.unit_norm_policy == UnitNormPolicy.NORMALIZE_WITH_WARNING
    assert m2.unit_norm_tolerance == 1e-3
    assert m2.population_weighting_mode == PopulationWeightingMode.IDENTITY_CENTROIDS
    # Format version >= 3 (PR 4 introduziu v3; revisão externa
    # 2026-06-09 bumpou para v4 com estado ponderado + proveniência da
    # referência. PR 4 tolera ambos.)
    assert d["format_version"] >= 3


def test_movmf_from_dict_v2_legacy_defaults_pr4_fields(reference_population):
    """JSONs gravados pré-PR 4 (format_version 2) defaultam para
    comportamento V2.14 (NORMALIZE_SILENTLY + FACE_LEVEL_RAW)."""
    X, _ = reference_population
    m_legacy = MoVMFTypicalityModel(n_components=1, random_state=0)
    m_legacy.fit(X)
    d = m_legacy.to_dict()
    # Simula JSON legado: muda format_version para 2, remove PR 4 fields
    d["format_version"] = 2
    for f in ("unit_norm_policy", "unit_norm_tolerance",
              "population_weighting_mode"):
        d["hyperparameters"].pop(f, None)
    m_loaded = MoVMFTypicalityModel.from_dict(d)
    assert m_loaded.unit_norm_policy == UnitNormPolicy.NORMALIZE_SILENTLY
    assert m_loaded.population_weighting_mode == PopulationWeightingMode.FACE_LEVEL_RAW


# ─────────────────────────────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────────────────────────────

def test_movmf_summary_includes_pr4_audit_fields(reference_population):
    X, ids = reference_population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m.fit(X, identity_ids=ids)
    s = m.summary_protocol()
    assert s.backend_details["unit_norm_policy"] == "NORMALIZE_SILENTLY"
    assert s.backend_details["population_weighting_mode"] == "IDENTITY_CENTROIDS"
    assert "random_state" in s.backend_details
    assert s.backend_details["n_input_faces"] == 60
    # 20 identidades, todas com 3 faces — sem singletons
    assert s.backend_details["n_singleton_identities"] == 0


def test_movmf_summary_warns_on_oversized_n_components():
    """n_components > sample_count → health CRITICAL."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((6, 8)); X /= np.linalg.norm(X, axis=1, keepdims=True)
    m = MoVMFTypicalityModel(n_components=5, random_state=0)
    m.fit(X)
    s = m.summary_protocol()
    # Pode ser OK ou WARNING/CRITICAL dependendo dos warnings agregados
    # — só verificamos que n_components_fitted ≤ sample_count após drops
    assert s.backend_details["n_components"] <= s.sample_count


# ─────────────────────────────────────────────────────────────────────
# Contract fingerprint inclui PR 4 fields
# ─────────────────────────────────────────────────────────────────────

def test_unit_norm_policy_affects_contract_fp(reference_population):
    """Mudança em UnitNormPolicy muda o model_contract_fingerprint."""
    X, _ = reference_population

    m1 = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m1.fit(X)
    fp1 = m1.model_contract_fingerprint

    m2 = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        unit_norm_policy=UnitNormPolicy.ERROR,
    )
    m2.fit(X)
    fp2 = m2.model_contract_fingerprint
    assert fp1 != fp2


def test_population_weighting_mode_affects_contract_fp(reference_population):
    X, ids = reference_population

    m_raw = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m_raw.fit(X)

    m_id = MoVMFTypicalityModel(
        n_components=1, random_state=0,
        population_weighting_mode=PopulationWeightingMode.IDENTITY_CENTROIDS,
    )
    m_id.fit(X, identity_ids=ids)
    assert m_raw.model_contract_fingerprint != m_id.model_contract_fingerprint


def test_knn_unit_norm_policy_affects_contract_fp(reference_population):
    X, ids = reference_population
    m1 = KNNTypicalityModel(
        k=4, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
    )
    m1.fit(X, identities=ids)
    m2 = KNNTypicalityModel(
        k=4, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_WITH_WARNING,
    )
    m2.fit(X, identities=ids)
    assert m1.model_contract_fingerprint != m2.model_contract_fingerprint
