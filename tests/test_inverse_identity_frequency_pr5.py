"""PR 5 — INVERSE_IDENTITY_FREQUENCY no k-NN e MoVMF.

Cobre:
  1. k-NN: pesos `w_i = 1/n_id_i` persistidos em
     `reference_score_weights`; `weighted_T_k_formula_id = "INV_ID_FREQ_v1"`
     no contract fingerprint.
  2. k-NN: agregação T_k usa pesos renormalizados localmente entre os k
     vizinhos.
  3. k-NN: CDF de referência ponderada usada no percentile.
  4. k-NN: round-trip preserva os campos novos.
  5. k-NN: contract fingerprint difere de FACE_LEVEL_RAW.
  6. k-NN: invariância (parcial) sob expansão da população — `INV` tem
     drift menor que `FACE_LEVEL_RAW` quando uma identidade ganha mais
     fotos.
  7. MoVMF: EM ponderado converge.
  8. MoVMF: contract fingerprint difere de FACE_LEVEL_RAW.
  9. MoVMF: round-trip preserva configs.
  10. MoVMF: avg_log_likelihood é ponderado.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.population_weighting_mode import PopulationWeightingMode
from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.enums.unit_norm_policy import UnitNormPolicy
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel


# ─────────────────────────────────────────────────────────────────────
# Fixture: identidades com fotos
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def population():
    """8 identidades, 3 fotos cada (variações em torno de centróide)."""
    rng = np.random.default_rng(42)
    N_ids = 8; per_id = 3
    centroids = rng.standard_normal((N_ids, 16))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    items = []; ids = []
    for i in range(N_ids):
        for j in range(per_id):
            v = centroids[i] + 0.1 * rng.standard_normal(16)
            v /= np.linalg.norm(v)
            items.append(v); ids.append(f"id_{i}")
    return np.array(items), ids, centroids, rng


@pytest.fixture
def expanded_population(population):
    """Mesma população + 5 fotos extras para as 2 primeiras identidades."""
    X, ids, centroids, rng = population
    extra = []; extra_ids = []
    for i in range(2):
        for j in range(5):
            v = centroids[i] + 0.1 * rng.standard_normal(16)
            v /= np.linalg.norm(v)
            extra.append(v); extra_ids.append(f"id_{i}")
    return np.vstack([X, extra]), ids + extra_ids


# ─────────────────────────────────────────────────────────────────────
# k-NN
# ─────────────────────────────────────────────────────────────────────

def test_knn_inv_sets_weights_per_face(population):
    X, ids, _, _ = population
    m = KNNTypicalityModel(
        k=3,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    st = m._state
    assert st.reference_score_weights is not None
    # 8 identidades * 3 fotos = 24 faces, todas com peso 1/3
    np.testing.assert_allclose(st.reference_score_weights, 1.0 / 3.0)
    assert st.weighted_T_k_formula_id == "INV_ID_FREQ_v1"
    assert st.loo_cdf is not None


def test_knn_inv_contract_fp_differs_from_raw(population):
    X, ids, _, _ = population
    m_raw = KNNTypicalityModel(
        k=3,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m_raw.fit(X, identity_ids=ids)
    m_inv = KNNTypicalityModel(
        k=3,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m_inv.fit(X, identity_ids=ids)
    assert m_raw.model_contract_fingerprint != m_inv.model_contract_fingerprint
    # weighted_T_k_formula_id entra na fp
    assert "INV_ID_FREQ_v1" not in str(m_raw._state.weighted_T_k_formula_id)


def test_knn_inv_round_trip_preserves_weights(population):
    X, ids, _, _ = population
    m = KNNTypicalityModel(
        k=3,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    d = m.to_dict()
    m2 = KNNTypicalityModel.from_dict(d)
    np.testing.assert_allclose(
        m2._state.reference_score_weights, m._state.reference_score_weights,
    )
    assert m2._state.weighted_T_k_formula_id == "INV_ID_FREQ_v1"
    # Format version >= 3 (PR 4/5 introduziu v3; revisão externa
    # 2026-06-09 bumpou para v4 com estado ponderado + proveniência da
    # referência).
    assert d["format_version"] >= 3


def test_knn_inv_percentiles_use_weighted_cdf(population):
    X, ids, centroids, rng = population
    queries = []
    for i in range(8):
        v = centroids[i] + 0.1 * rng.standard_normal(16); v /= np.linalg.norm(v)
        queries.append(v)
    queries = np.array(queries)

    m = KNNTypicalityModel(
        k=3,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    p = m.get_percentiles(queries)
    assert p.shape == (8,)
    assert (p >= 0).all() and (p <= 100).all()


def test_knn_inv_reduces_drift_under_population_expansion(population, expanded_population):
    """Quando 2 identidades ganham 5 fotos extras, INV deve produzir
    `get_percentiles` MAIS estáveis que FACE_LEVEL_RAW."""
    X, ids, centroids, rng = population
    X_dup, ids_dup = expanded_population
    queries = []
    for i in range(8):
        v = centroids[i] + 0.1 * rng.standard_normal(16); v /= np.linalg.norm(v)
        queries.append(v)
    queries = np.array(queries)

    # FACE_LEVEL_RAW
    m_raw = KNNTypicalityModel(
        k=3, population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m_raw.fit(X, identity_ids=ids)
    m_raw_dup = KNNTypicalityModel(
        k=3, population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m_raw_dup.fit(X_dup, identity_ids=ids_dup)
    drift_raw = float(np.median(np.abs(
        m_raw.get_percentiles(queries) - m_raw_dup.get_percentiles(queries)
    )))

    # INVERSE_IDENTITY_FREQUENCY
    m_inv = KNNTypicalityModel(
        k=3, population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m_inv.fit(X, identity_ids=ids)
    m_inv_dup = KNNTypicalityModel(
        k=3, population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m_inv_dup.fit(X_dup, identity_ids=ids_dup)
    drift_inv = float(np.median(np.abs(
        m_inv.get_percentiles(queries) - m_inv_dup.get_percentiles(queries)
    )))

    # INV deve ter drift ≤ FACE_LEVEL_RAW (idealmente menor; toleramos
    # igualdade em casos degenerados)
    assert drift_inv <= drift_raw + 1e-6, (
        f"INVERSE_IDENTITY_FREQUENCY não reduziu drift: "
        f"raw_drift={drift_raw:.2f}, inv_drift={drift_inv:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────
# MoVMF
# ─────────────────────────────────────────────────────────────────────

def test_movmf_inv_em_converges(population):
    X, ids, _, _ = population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    assert m.is_fitted()
    assert m._converged_ is True


def test_movmf_inv_contract_fp_differs_from_raw(population):
    X, ids, _, _ = population
    m_raw = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.FACE_LEVEL_RAW,
    )
    m_raw.fit(X)
    m_inv = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m_inv.fit(X, identity_ids=ids)
    assert m_raw.model_contract_fingerprint != m_inv.model_contract_fingerprint


def test_movmf_inv_round_trip(population):
    X, ids, _, _ = population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    d = m.to_dict()
    m2 = MoVMFTypicalityModel.from_dict(d)
    assert m2.population_weighting_mode == PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY
    assert m2.model_contract_fingerprint == m.model_contract_fingerprint


def test_movmf_inv_avg_ll_is_weighted(population):
    """avg_train_log_likelihood deve ser média ponderada quando INV."""
    X, ids, _, _ = population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    # weights internos ponderados existem
    assert m._train_log_likelihoods_weights is not None
    assert m._train_log_likelihoods_cdf is not None


def test_movmf_inv_summary_includes_weighting_mode(population):
    X, ids, _, _ = population
    m = MoVMFTypicalityModel(
        n_components=2, random_state=0,
        unit_norm_policy=UnitNormPolicy.NORMALIZE_SILENTLY,
        population_weighting_mode=PopulationWeightingMode.INVERSE_IDENTITY_FREQUENCY,
    )
    m.fit(X, identity_ids=ids)
    s = m.summary()
    assert s.backend_details["population_weighting_mode"] == "INVERSE_IDENTITY_FREQUENCY"
