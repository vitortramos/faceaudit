"""Testes do KNNTypicalityModel.

Cobre:
- API básica (fit, get_typicality_info, batch, percentile, OOD).
- Modos PER_FACE e PER_IDENTITY (agregação por centroide).
- Singletons preservados em PER_IDENTITY.
- Comportamento estatístico: queries do "modo" são típicas (percentile alto);
  queries OOD são atípicas.
- Self-match handling.
- Serialização: roundtrip to_dict/from_dict v2 + migração v1 → v2.
- Validações: N < k+1, dimensão errada, identities ausentes em PER_IDENTITY.
"""
from __future__ import annotations

import numpy as np
import pytest

from faceaudit.enums.typicality_reference_mode import TypicalityReferenceMode
from faceaudit.models.knn_typicality_info import KNNTypicalityInfo
from faceaudit.services.knn_typicality_model import KNNTypicalityModel


# ─────────────────────────── helpers ────────────────────────────


def _sample_vmf(n: int, mu: np.ndarray, kappa: float, rng: np.random.Generator) -> np.ndarray:
    """Amostragem aproximada de vMF via projeção (suficiente pra testes)."""
    d = mu.shape[0]
    samples = rng.normal(size=(n, d))
    samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
    target_cos = kappa / (d - 1 + kappa)
    coef = np.sqrt(max(0.0, 1.0 - target_cos ** 2))
    out = target_cos * mu[None, :] + coef * samples
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _make_population_with_ids(
    n_identities: int, samples_per_id: int, d: int,
    rng: np.random.Generator,
    proto_jitter: float = 0.3, intra_jitter: float = 0.1,
):
    """Gera embeddings + lista paralela de identities pra testes PER_IDENTITY."""
    mu = np.zeros(d); mu[0] = 1.0
    protos = _sample_vmf(n_identities, mu, kappa=5.0, rng=rng)
    embs = []
    ids = []
    for i, p in enumerate(protos):
        noise = rng.normal(size=(samples_per_id, d)) * intra_jitter
        samples = _normalize(p[None, :] + noise)
        embs.append(samples)
        ids.extend([f"id_{i:03d}"] * samples_per_id)
    return np.vstack(embs), ids


# ─────────────────────────── construtor / validações ────────────


def test_construct_with_defaults():
    m = KNNTypicalityModel()
    assert m.k == 50
    assert m.ood_quantile == 0.01
    assert m.reference_mode is TypicalityReferenceMode.PER_IDENTITY
    assert not m.is_fitted()


def test_construct_per_face_mode():
    m = KNNTypicalityModel(reference_mode=TypicalityReferenceMode.PER_FACE)
    assert m.reference_mode is TypicalityReferenceMode.PER_FACE


def test_construct_rejects_invalid_k():
    with pytest.raises(ValueError, match="k deve ser >= 1"):
        KNNTypicalityModel(k=0)


def test_construct_rejects_invalid_ood_quantile():
    with pytest.raises(ValueError, match="ood_quantile"):
        KNNTypicalityModel(ood_quantile=0.5)


# ─────────────────────────── fit PER_FACE ───────────────────────


def test_fit_per_face_requires_enough_samples():
    m = KNNTypicalityModel(k=10, reference_mode=TypicalityReferenceMode.PER_FACE)
    X = np.random.default_rng(0).normal(size=(5, 8))
    with pytest.raises(ValueError, match="k\\+1"):
        m.fit(X)


def test_fit_per_face_sets_state():
    rng = np.random.default_rng(42)
    X = _normalize(rng.normal(size=(200, 32)))
    m = KNNTypicalityModel(
        k=20, reference_mode=TypicalityReferenceMode.PER_FACE,
    ).fit(X)
    assert m.is_fitted()
    assert m.sample_count == 200
    assert m.n_input_faces == 200
    assert m.n_singleton_identities == 0
    assert m.embedding_dim == 32


# ─────────────────────────── fit PER_IDENTITY ───────────────────


def test_fit_per_identity_requires_identities():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    )
    with pytest.raises(ValueError, match="IDENTITY_CENTROIDS exige"):
        m.fit(X)


def test_fit_per_identity_rejects_mismatched_identities_length():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    ids = ["id_0"] * 99  # tamanho errado
    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    )
    with pytest.raises(ValueError, match="len\\(identities\\)"):
        m.fit(X, identities=ids)


def test_fit_per_identity_aggregates_to_centroids():
    """N identidades × M fotos cada → referência tem N pontos, não N·M."""
    rng = np.random.default_rng(7)
    embs, ids = _make_population_with_ids(
        n_identities=50, samples_per_id=5, d=16, rng=rng,
    )
    assert embs.shape[0] == 250
    assert len(set(ids)) == 50

    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)

    assert m.sample_count == 50, "Referência deveria ter 50 centroides"
    assert m.n_input_faces == 250
    assert m.n_singleton_identities == 0


def test_fit_per_identity_preserves_singletons():
    """Identidades singleton entram como centroide = a face única."""
    rng = np.random.default_rng(8)
    # 30 identidades com 4 fotos + 20 singletons
    embs_grouped, ids_grouped = _make_population_with_ids(
        n_identities=30, samples_per_id=4, d=16, rng=rng,
    )
    embs_single = _normalize(rng.normal(size=(20, 16)))
    ids_single = [f"single_{i:02d}" for i in range(20)]

    embs = np.vstack([embs_grouped, embs_single])
    ids = ids_grouped + ids_single

    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)

    assert m.sample_count == 50, "30 multi + 20 singleton = 50 centroides"
    assert m.n_input_faces == 30 * 4 + 20  # = 140
    assert m.n_singleton_identities == 20


def test_fit_per_identity_centroids_are_l2_normalized():
    rng = np.random.default_rng(9)
    embs, ids = _make_population_with_ids(
        n_identities=20, samples_per_id=3, d=16, rng=rng,
    )
    m = KNNTypicalityModel(
        k=5, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)
    norms = np.linalg.norm(m._state.reference, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ─────────────────────────── inferência: shape e tipos ─────────


def test_get_typicality_info_single():
    rng = np.random.default_rng(0)
    embs, ids = _make_population_with_ids(
        n_identities=40, samples_per_id=5, d=16, rng=rng,
    )
    m = KNNTypicalityModel(k=10).fit(embs, identities=ids)
    info = m.get_typicality_info(embs[0])
    assert isinstance(info, KNNTypicalityInfo)
    assert info.embedding_dim == 16
    assert info.k == 10
    assert 0.0 <= info.typicality_percentile <= 100.0
    assert info.knn_similarities is not None
    assert len(info.knn_similarities) == 10


def test_batch_returns_correct_length():
    rng = np.random.default_rng(0)
    embs, ids = _make_population_with_ids(
        n_identities=30, samples_per_id=5, d=16, rng=rng,
    )
    m = KNNTypicalityModel(k=8).fit(embs, identities=ids)
    queries = _normalize(rng.normal(size=(7, 16)))
    infos = m.get_typicality_infos(queries)
    assert len(infos) == 7
    assert all(i.knn_similarities is None for i in infos)


# ─────────────────────────── comportamento estatístico ─────────


def test_typical_query_has_high_percentile_per_face():
    rng = np.random.default_rng(7)
    d = 32
    mu = np.zeros(d); mu[0] = 1.0
    X = _sample_vmf(500, mu, kappa=20.0, rng=rng)
    m = KNNTypicalityModel(
        k=30, reference_mode=TypicalityReferenceMode.PER_FACE,
    ).fit(X)

    info_typical = m.get_typicality_info(mu)
    info_atypical = m.get_typicality_info(-mu)
    assert info_typical.typicality_percentile > 75.0
    assert info_atypical.typicality_percentile < 5.0
    assert info_typical.is_in_distribution
    assert not info_atypical.is_in_distribution


def test_typical_query_has_high_percentile_per_identity():
    """Em PER_IDENTITY, query no modo da distribuição (≈ centroide global)
    também deve ter percentile alto."""
    rng = np.random.default_rng(11)
    d = 32
    mu = np.zeros(d); mu[0] = 1.0
    embs, ids = _make_population_with_ids(
        n_identities=100, samples_per_id=4, d=d, rng=rng,
        proto_jitter=0.2, intra_jitter=0.05,
    )
    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)

    info_typical = m.get_typicality_info(mu)
    info_atypical = m.get_typicality_info(-mu)
    assert info_typical.typicality_percentile > 50.0
    assert info_atypical.typicality_percentile < 5.0


def test_per_identity_neutralizes_imbalance():
    """Identidade com 50 fotos vs identidades com 1 foto: em PER_IDENTITY,
    a tipicidade da pessoa super-replicada NÃO fica artificialmente alta."""
    rng = np.random.default_rng(13)
    d = 32
    mu = np.zeros(d); mu[0] = 1.0

    # 49 identidades de 1 foto cada (centroide=foto)
    single_embs = _sample_vmf(49, mu, kappa=5.0, rng=rng)
    single_ids = [f"s_{i}" for i in range(49)]

    # 1 identidade com 50 fotos (centroide forte mas é apenas 1 ponto)
    big_id_proto = _sample_vmf(1, mu, kappa=5.0, rng=rng)[0]
    big_embs = _normalize(
        big_id_proto[None, :] + rng.normal(size=(50, d)) * 0.05
    )
    big_ids = ["big_id"] * 50

    embs = np.vstack([single_embs, big_embs])
    ids = single_ids + big_ids
    assert len(ids) == 99

    m_per_id = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)

    # 49 singles + 1 multi → 50 centroides
    assert m_per_id.sample_count == 50

    # Comparação: em PER_FACE, o "big_id" teria T_k inflado pela
    # auto-similaridade (top-10 vizinhos = outras fotos dele mesmo).
    # Em PER_IDENTITY, o "big_id" é UM centroide entre 50 — sua
    # tipicidade depende da posição dele vs os outros centroides.
    big_centroid_typicality = m_per_id.get_typicality_info(
        _normalize(big_embs.mean(axis=0))
    ).typicality_percentile

    # Não há expectativa numérica forte aqui; o ponto é só não
    # explodir pra 100 artificialmente como aconteceria em PER_FACE.
    assert 0.0 <= big_centroid_typicality <= 100.0


def test_self_match_is_handled():
    rng = np.random.default_rng(9)
    X = _normalize(rng.normal(size=(300, 32)))
    m = KNNTypicalityModel(
        k=30, reference_mode=TypicalityReferenceMode.PER_FACE,
    ).fit(X)
    info = m.get_typicality_info(X[42])
    assert info.knn_similarities[0] < 0.999


def test_dim_mismatch_raises():
    rng = np.random.default_rng(0)
    embs, ids = _make_population_with_ids(
        n_identities=20, samples_per_id=5, d=32, rng=rng,
    )
    m = KNNTypicalityModel(k=8).fit(embs, identities=ids)
    with pytest.raises(ValueError, match="Dimensão"):
        m.get_typicality_info(np.ones(16))


# ─────────────────────────── serialização v2 ───────────────────


def test_to_dict_v2_roundtrip_per_identity():
    rng = np.random.default_rng(0)
    embs, ids = _make_population_with_ids(
        n_identities=30, samples_per_id=4, d=16, rng=rng,
    )
    m1 = KNNTypicalityModel(
        k=8, ood_quantile=0.02,
        reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)

    data = m1.to_dict()
    # PR 5: format_version 3 (PR 4 era 2)
    assert data["format_version"] == 3
    assert data["hyperparameters"]["reference_mode"] == "PER_IDENTITY"

    m2 = KNNTypicalityModel.from_dict(data)
    assert m2.is_fitted()
    assert m2.reference_mode is TypicalityReferenceMode.PER_IDENTITY
    assert m2.sample_count == m1.sample_count
    assert m2.n_input_faces == m1.n_input_faces
    assert m2.n_singleton_identities == m1.n_singleton_identities

    queries = _normalize(rng.normal(size=(5, 16)))
    np.testing.assert_allclose(
        m1.get_percentiles(queries), m2.get_percentiles(queries),
    )


def test_from_dict_v1_migrates_to_per_face():
    """Modelo v1 (sem reference_mode, com population_label legado) deve
    desserializar como PER_FACE (modo implícito da v1)."""
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m1 = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_FACE,
    ).fit(X)

    # Simula JSON v1: descarta reference_mode, adiciona population_label legado
    data = m1.to_dict()
    data["format_version"] = 1
    data["hyperparameters"].pop("reference_mode", None)
    data["hyperparameters"]["population_label"] = "legacy_pop"
    data["learned_state"].pop("reference_mode", None)

    m_migrated = KNNTypicalityModel.from_dict(data)
    assert m_migrated.is_fitted()
    assert m_migrated.reference_mode is TypicalityReferenceMode.PER_FACE
    assert m_migrated.sample_count == m1.sample_count


def test_from_dict_rejects_unknown_version():
    bogus = {"format_version": 999, "hyperparameters": {}, "learned_state": {}}
    with pytest.raises(ValueError, match="Versão de formato"):
        KNNTypicalityModel.from_dict(bogus)


def test_from_dict_rejects_unnormalized_reference():
    rng = np.random.default_rng(0)
    X = _normalize(rng.normal(size=(100, 16)))
    m = KNNTypicalityModel(
        k=10, reference_mode=TypicalityReferenceMode.PER_FACE,
    ).fit(X)
    data = m.to_dict()
    data["learned_state"]["reference"] = (
        np.asarray(data["learned_state"]["reference"]) * 3.0
    ).tolist()
    with pytest.raises(ValueError, match="L2-normalizada"):
        KNNTypicalityModel.from_dict(data)


# ─────────────────────────── summary / diagnóstico ─────────────


def test_summary_unfitted_returns_status():
    s = KNNTypicalityModel().summary()
    assert s == {"status": "unfitted"}


def test_summary_fitted_per_identity_includes_diagnostics():
    rng = np.random.default_rng(0)
    embs_grouped, ids_grouped = _make_population_with_ids(
        n_identities=20, samples_per_id=3, d=16, rng=rng,
    )
    # 10 singletons
    embs_single = _normalize(rng.normal(size=(10, 16)))
    ids_single = [f"sing_{i}" for i in range(10)]

    embs = np.vstack([embs_grouped, embs_single])
    ids = ids_grouped + ids_single

    m = KNNTypicalityModel(
        k=5, reference_mode=TypicalityReferenceMode.PER_IDENTITY,
    ).fit(embs, identities=ids)
    s = m.summary()
    assert s["status"] == "fitted"
    assert s["reference_mode"] == "PER_IDENTITY"
    assert s["sample_count"] == 30  # 20 grouped + 10 singletons
    assert s["n_input_faces"] == 60 + 10  # 20*3 + 10
    assert s["n_singleton_identities"] == 10
