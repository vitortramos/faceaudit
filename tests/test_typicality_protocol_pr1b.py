"""PR 1b — Testes dos contratos: protocolo, batch, summary, registry, factory.

Cobre:
  1. TypicalityModelType enum (from_str, to_str, choices)
  2. TypicalityScoreBatch (imutabilidade, fingerprint, validações)
  3. TypicalityModelSummary (estrutura)
  4. Ambos os modelos satisfazem o TypicalityScoringModelProtocol
  5. enabled_score_types vazio antes de fit; populado depois
  6. compute_scores produz batch correto em ambos os backends
  7. UnsupportedTypicalityScoreTypeError quando score_type incompatível
  8. Fingerprints determinísticos (mesma instância → mesmo hash)
  9. Backend registry: KNN + MOVMF registrados; potential_score_types
  10. Factory round-trip preserva scores e fingerprints
  11. Factory detecta fingerprint mismatch
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.typicality_model_type import TypicalityModelType
from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.enums.typicality_score_type import TypicalityScoreType
from faceaudit.models.typicality.typicality_model_summary import (
    TypicalityModelSummary,
)
from faceaudit.models.typicality.typicality_score_batch import (
    TypicalityScoreBatch,
)
from faceaudit.protocols.typicality_scoring_model_protocol import (
    FittableTypicalityModelProtocol,
    TypicalityScoringModelProtocol,
    UnsupportedTypicalityScoreTypeError,
)
from faceaudit.services.knn_typicality_model import KNNTypicalityModel
from faceaudit.services.movmf_typicality_model import MoVMFTypicalityModel
from faceaudit.services.typicality_model_backend_registry import (
    TypicalityModelBackendRegistry,
)
from faceaudit.services.typicality_model_factory import (
    EnvelopeFingerprintMismatch,
    TypicalityModelFactory,
)


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture
def reference_data():
    rng = np.random.default_rng(20260608)
    X = rng.standard_normal((60, 16))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    ids = tuple(f"id{i // 3}" for i in range(60))
    return X, ids


@pytest.fixture
def trained_knn(reference_data):
    X, ids = reference_data
    m = KNNTypicalityModel(
        k=8, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    )
    m.fit(X, identities=ids)
    return m, X


@pytest.fixture
def trained_movmf(reference_data):
    X, _ = reference_data
    m = MoVMFTypicalityModel(n_components=2, random_state=0)
    m.fit(X)
    return m, X


# ─────────────────────────────────────────────────────────────────────
# TypicalityModelType enum
# ─────────────────────────────────────────────────────────────────────

def test_typicality_model_type_from_str():
    assert TypicalityModelType.from_str("KNN") == TypicalityModelType.KNN
    assert TypicalityModelType.from_str("knn") == TypicalityModelType.KNN
    assert TypicalityModelType.from_str("MOVMF") == TypicalityModelType.MOVMF
    assert TypicalityModelType.from_str("movmf") == TypicalityModelType.MOVMF
    # Default em string vazia
    assert TypicalityModelType.from_str("") == TypicalityModelType.KNN
    # Identity-aware
    assert TypicalityModelType.from_str(TypicalityModelType.MOVMF) == TypicalityModelType.MOVMF


def test_typicality_model_type_choices_with_description():
    rows = TypicalityModelType.choices_with_description()
    assert len(rows) == 2
    for value, label, descr in rows:
        assert isinstance(value, str)
        assert isinstance(label, str)
        assert isinstance(descr, str)


# ─────────────────────────────────────────────────────────────────────
# TypicalityScoreBatch
# ─────────────────────────────────────────────────────────────────────

def test_score_batch_immutable_scores():
    b = TypicalityScoreBatch(
        scores=np.array([1.0, 2.0, 3.0]),
        n=3,
        embedding_dim=8,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        score_semantics_id="MEAN_KNN_SIMILARITY_SPHERICAL",
        model_type=TypicalityModelType.KNN,
        model_state_fingerprint="abc",
        model_contract_fingerprint="def",
    )
    assert b.scores.flags.writeable is False


def test_score_batch_auto_fingerprint():
    b = TypicalityScoreBatch(
        scores=np.array([1.0, 2.0, 3.0]),
        n=3,
        embedding_dim=8,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        score_semantics_id="X",
        model_type=TypicalityModelType.KNN,
    )
    assert b.batch_fingerprint != ""
    assert len(b.batch_fingerprint) == 64  # SHA-256 hex


def test_score_batch_size_mismatch_raises():
    with pytest.raises(ValueError, match="difere de"):
        TypicalityScoreBatch(
            scores=np.array([1.0, 2.0]),
            n=99,
            embedding_dim=8,
            score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
            score_semantics_id="X",
            model_type=TypicalityModelType.KNN,
        )


def test_score_batch_percentile_requires_basis():
    with pytest.raises(ValueError, match="percentile_basis_score_type"):
        TypicalityScoreBatch(
            scores=np.array([1.0]),
            n=1,
            embedding_dim=8,
            score_type=TypicalityScoreType.TYPICALITY_PERCENTILE,
            score_semantics_id="X",
            model_type=TypicalityModelType.KNN,
            # falta percentile_basis_score_type
        )


def test_score_batch_basis_cleared_when_not_percentile():
    b = TypicalityScoreBatch(
        scores=np.array([1.0]),
        n=1,
        embedding_dim=8,
        score_type=TypicalityScoreType.MEAN_KNN_SIMILARITY,
        score_semantics_id="X",
        percentile_basis_score_type=TypicalityScoreType.LOG_DENSITY_ESTIMATE,
        reference_score_distribution_fingerprint="abc",
        model_type=TypicalityModelType.KNN,
    )
    # Limpa campos que não fazem sentido em scores não-percentil.
    assert b.percentile_basis_score_type is None
    assert b.reference_score_distribution_fingerprint is None


# ─────────────────────────────────────────────────────────────────────
# Protocol satisfação estrutural
# ─────────────────────────────────────────────────────────────────────

def test_knn_satisfies_scoring_protocol(trained_knn):
    m, _ = trained_knn
    assert isinstance(m, TypicalityScoringModelProtocol)


def test_movmf_satisfies_scoring_protocol(trained_movmf):
    m, _ = trained_movmf
    assert isinstance(m, TypicalityScoringModelProtocol)


def test_knn_satisfies_fittable_protocol(trained_knn):
    m, _ = trained_knn
    assert isinstance(m, FittableTypicalityModelProtocol)


def test_movmf_satisfies_fittable_protocol(trained_movmf):
    m, _ = trained_movmf
    assert isinstance(m, FittableTypicalityModelProtocol)


# ─────────────────────────────────────────────────────────────────────
# enabled_score_types
# ─────────────────────────────────────────────────────────────────────

def test_knn_enabled_score_types_empty_before_fit():
    m = KNNTypicalityModel(k=5)
    assert m.enabled_score_types == frozenset()


def test_movmf_enabled_score_types_empty_before_fit():
    m = MoVMFTypicalityModel(n_components=1)
    assert m.enabled_score_types == frozenset()


def test_knn_enabled_score_types_after_fit(trained_knn):
    m, _ = trained_knn
    assert m.enabled_score_types == frozenset({
        TypicalityScoreType.MEAN_KNN_SIMILARITY,
        TypicalityScoreType.TYPICALITY_PERCENTILE,
        TypicalityScoreType.LOG_DENSITY_ESTIMATE,
    })


def test_movmf_enabled_score_types_after_fit(trained_movmf):
    m, _ = trained_movmf
    # MoVMF não suporta MEAN_KNN_SIMILARITY
    assert m.enabled_score_types == frozenset({
        TypicalityScoreType.TYPICALITY_PERCENTILE,
        TypicalityScoreType.LOG_DENSITY_ESTIMATE,
    })


# ─────────────────────────────────────────────────────────────────────
# compute_scores
# ─────────────────────────────────────────────────────────────────────

def test_knn_compute_scores_mean_knn_similarity(trained_knn):
    m, X = trained_knn
    b = m.compute_scores(X, TypicalityScoreType.MEAN_KNN_SIMILARITY)
    assert b.n == X.shape[0]
    assert b.score_type == TypicalityScoreType.MEAN_KNN_SIMILARITY
    assert b.score_semantics_id == "MEAN_KNN_SIMILARITY_SPHERICAL"
    assert b.model_type == TypicalityModelType.KNN
    assert b.percentile_basis_score_type is None


def test_knn_compute_scores_percentile_has_basis(trained_knn):
    m, X = trained_knn
    b = m.compute_scores(X, TypicalityScoreType.TYPICALITY_PERCENTILE)
    assert b.percentile_basis_score_type == TypicalityScoreType.MEAN_KNN_SIMILARITY
    assert b.reference_score_distribution_fingerprint is not None
    assert len(b.reference_score_distribution_fingerprint) == 64


def test_movmf_compute_scores_log_density(trained_movmf):
    m, X = trained_movmf
    b = m.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    assert b.score_semantics_id == "LOG_LIKELIHOOD_MOVMF"
    assert b.model_type == TypicalityModelType.MOVMF


def test_movmf_compute_scores_percentile_has_basis(trained_movmf):
    m, X = trained_movmf
    b = m.compute_scores(X, TypicalityScoreType.TYPICALITY_PERCENTILE)
    assert b.score_semantics_id == "PERCENTILE_OF_LOG_LIKELIHOOD_MOVMF"
    assert b.percentile_basis_score_type == TypicalityScoreType.LOG_DENSITY_ESTIMATE
    assert b.reference_score_distribution_fingerprint is not None


def test_movmf_rejects_mean_knn_similarity(trained_movmf):
    m, X = trained_movmf
    with pytest.raises(UnsupportedTypicalityScoreTypeError) as ei:
        m.compute_scores(X, TypicalityScoreType.MEAN_KNN_SIMILARITY)
    err = ei.value
    assert err.requested == TypicalityScoreType.MEAN_KNN_SIMILARITY
    assert err.model_type == TypicalityModelType.MOVMF


def test_knn_compute_scores_immutable(trained_knn):
    m, X = trained_knn
    b = m.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    with pytest.raises(ValueError):
        b.scores[0] = 999.0


# ─────────────────────────────────────────────────────────────────────
# Fingerprints determinísticos
# ─────────────────────────────────────────────────────────────────────

def test_knn_fingerprints_deterministic(trained_knn):
    m, _ = trained_knn
    fp1 = m.model_state_fingerprint
    fp2 = m.model_state_fingerprint
    assert fp1 == fp2
    cp1 = m.model_contract_fingerprint
    cp2 = m.model_contract_fingerprint
    assert cp1 == cp2


def test_movmf_fingerprints_deterministic(trained_movmf):
    m, _ = trained_movmf
    fp1 = m.model_state_fingerprint
    fp2 = m.model_state_fingerprint
    assert fp1 == fp2


def test_knn_contract_changes_with_k(reference_data):
    X, ids = reference_data
    m_a = KNNTypicalityModel(k=8).fit(X, identities=ids)
    m_b = KNNTypicalityModel(k=12).fit(X, identities=ids)
    assert m_a.model_contract_fingerprint != m_b.model_contract_fingerprint


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────

def test_knn_summary_protocol(trained_knn):
    m, _ = trained_knn
    s = m.summary()
    assert isinstance(s, TypicalityModelSummary)
    assert s.model_type == TypicalityModelType.KNN
    assert s.health_status in ("OK", "WARNING", "CRITICAL")
    assert "k" in s.backend_details


def test_movmf_summary_protocol(trained_movmf):
    m, _ = trained_movmf
    s = m.summary()
    assert isinstance(s, TypicalityModelSummary)
    assert s.model_type == TypicalityModelType.MOVMF
    assert "n_components" in s.backend_details


# ─────────────────────────────────────────────────────────────────────
# Backend registry
# ─────────────────────────────────────────────────────────────────────

def test_registry_has_both_backends():
    specs = TypicalityModelBackendRegistry.all_specs()
    assert TypicalityModelType.KNN in specs
    assert TypicalityModelType.MOVMF in specs


def test_registry_knn_badge_stable():
    spec = TypicalityModelBackendRegistry.get(TypicalityModelType.KNN)
    assert spec.badge == "STABLE"


def test_registry_movmf_badge_experimental():
    spec = TypicalityModelBackendRegistry.get(TypicalityModelType.MOVMF)
    assert spec.badge == "EXPERIMENTAL"


def test_registry_potential_score_types():
    knn = TypicalityModelBackendRegistry.get(TypicalityModelType.KNN)
    movmf = TypicalityModelBackendRegistry.get(TypicalityModelType.MOVMF)
    assert TypicalityScoreType.MEAN_KNN_SIMILARITY in knn.potential_score_types
    assert TypicalityScoreType.MEAN_KNN_SIMILARITY not in movmf.potential_score_types
    assert TypicalityScoreType.TYPICALITY_PERCENTILE in knn.potential_score_types
    assert TypicalityScoreType.TYPICALITY_PERCENTILE in movmf.potential_score_types


def test_registry_resolve_class():
    cls_k = TypicalityModelBackendRegistry.resolve_class(TypicalityModelType.KNN)
    cls_m = TypicalityModelBackendRegistry.resolve_class(TypicalityModelType.MOVMF)
    assert cls_k is KNNTypicalityModel
    assert cls_m is MoVMFTypicalityModel


# ─────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────

def test_factory_round_trip_knn(trained_knn):
    m, X = trained_knn
    envelope = TypicalityModelFactory.to_dict(m)
    assert envelope["schema_version"] == 1
    assert envelope["model_type"] == "KNN"

    m2 = TypicalityModelFactory.from_dict(envelope)
    assert isinstance(m2, KNNTypicalityModel)

    b1 = m.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    b2 = m2.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    np.testing.assert_allclose(b1.scores, b2.scores)
    assert b1.batch_fingerprint == b2.batch_fingerprint


def test_factory_round_trip_movmf(trained_movmf):
    m, X = trained_movmf
    envelope = TypicalityModelFactory.to_dict(m)
    assert envelope["model_type"] == "MOVMF"

    m2 = TypicalityModelFactory.from_dict(envelope)
    assert isinstance(m2, MoVMFTypicalityModel)

    b1 = m.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    b2 = m2.compute_scores(X, TypicalityScoreType.LOG_DENSITY_ESTIMATE)
    np.testing.assert_allclose(b1.scores, b2.scores)


def test_factory_detects_state_fingerprint_mismatch(trained_knn):
    m, _ = trained_knn
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["fingerprints"]["model_state_fingerprint"] = "0" * 64
    with pytest.raises(EnvelopeFingerprintMismatch) as ei:
        TypicalityModelFactory.from_dict(envelope)
    assert ei.value.kind == "state"


def test_factory_detects_contract_fingerprint_mismatch(trained_knn):
    m, _ = trained_knn
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["fingerprints"]["model_contract_fingerprint"] = "0" * 64
    with pytest.raises(EnvelopeFingerprintMismatch) as ei:
        TypicalityModelFactory.from_dict(envelope)
    assert ei.value.kind == "contract"


def test_factory_rejects_wrong_schema_version(trained_knn):
    m, _ = trained_knn
    envelope = TypicalityModelFactory.to_dict(m)
    envelope["schema_version"] = 999
    with pytest.raises(ValueError, match="schema_version"):
        TypicalityModelFactory.from_dict(envelope)


def test_factory_rejects_unfitted_model():
    m = KNNTypicalityModel(k=5)
    with pytest.raises(RuntimeError, match="não treinado"):
        TypicalityModelFactory.to_dict(m)
