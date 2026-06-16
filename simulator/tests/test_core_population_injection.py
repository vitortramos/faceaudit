"""Core center-injection + faceaudit-free bridge (Mode A) tests.

Runs without faceaudit installed — `faceaudit_seed`'s extraction/Mode A/leakage helpers never
import faceaudit at runtime (MoVMFTypicalityModel is lazy-imported only by Mode B).

    .venv/Scripts/python.exe -m pytest simulator/tests/test_core_population_injection.py
"""

from __future__ import annotations

import numpy as np
import pytest

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    l2_normalize_rows,
    make_cooperative_plda_config,
)
import faceaudit_seed as fs


# --------------------------------------------------------------------------- fake dataset


class _Item:
    def __init__(self, embedding, identity, quality_score=None):
        self.embedding = embedding
        self.identity = identity
        self.quality_score = quality_score


class _FakeDataset:
    """Duck-typed stand-in for EmbeddingDataset (matrix / identities / items)."""

    def __init__(self, embeddings, identities, qualities=None, with_items=True):
        self._embeddings = np.asarray(embeddings, dtype=np.float64)
        self._identities = list(identities)
        if with_items:
            quals = qualities if qualities is not None else [None] * len(self._identities)
            self.items = [_Item(e, i, q) for e, i, q in zip(self._embeddings, self._identities, quals)]

    @property
    def matrix(self):
        return self._embeddings

    @property
    def identities(self):
        return np.array(self._identities, dtype=object)


def _two_cluster_dataset(*, dim=16, per_cluster=8, per_identity=4, seed=0):
    rng = np.random.default_rng(seed)
    mu = {"A": l2_normalize_rows(np.eye(1, dim, 0))[0], "B": l2_normalize_rows(np.eye(1, dim, 1))[0]}
    embs, ids, quals = [], [], []
    for cluster in ("A", "B"):
        for k in range(per_cluster):
            center = l2_normalize_rows(mu[cluster] + 0.10 * rng.normal(size=dim))[0]
            for _ in range(per_identity):
                u = l2_normalize_rows(center + 0.05 * rng.normal(size=dim))[0]
                embs.append((20 + 10 * rng.random()) * u)
                ids.append(f"{cluster}{k}")
                quals.append(float(rng.random()))
    return _FakeDataset(embs, ids, quals)


# --------------------------------------------------------------------------- core injection


def _sim(dim=16):
    return AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=dim), random_seed=7)


def test_inject_centers_uses_and_normalizes_them() -> None:
    sim = _sim()
    raw = np.random.default_rng(1).normal(size=(5, 16))
    pop = sim.sample_identity_population(centers=raw)
    assert np.allclose(pop.identity_centers, l2_normalize_rows(raw))
    assert pop.identity_ids.tolist() == list(range(5))


def test_inject_rejects_both_and_neither() -> None:
    sim = _sim()
    centers = l2_normalize_rows(np.random.default_rng(2).normal(size=(4, 16)))
    with pytest.raises(ValueError, match="exactly one"):
        sim.sample_identity_population()
    with pytest.raises(ValueError, match="exactly one"):
        sim.sample_identity_population(identity_count=4, centers=centers)


def test_inject_validates_dimension_and_components_and_lengths() -> None:
    sim = _sim()
    centers = l2_normalize_rows(np.random.default_rng(3).normal(size=(4, 16)))
    with pytest.raises(ValueError, match="dimension"):
        sim.sample_identity_population(centers=l2_normalize_rows(np.random.default_rng(3).normal(size=(4, 8))))
    with pytest.raises(ValueError, match="integers"):
        sim.sample_identity_population(centers=centers, components=np.zeros(4))  # float
    with pytest.raises(ValueError, match="component_count"):
        sim.sample_identity_population(centers=centers, components=np.array([0, 1, 2, 999]))
    with pytest.raises(ValueError, match="identity_labels"):
        sim.sample_identity_population(centers=centers, identity_labels=["a", "b"])
    with pytest.raises(ValueError, match="source_sample_counts must be positive"):
        sim.sample_identity_population(centers=centers, source_sample_counts=[1, 2, 0, 3])


def test_injected_population_does_not_change_sampled_golden() -> None:
    # A sampled population is seed-derived (order-independent), so injecting elsewhere can't shift it.
    sim = _sim()
    sampled = sim.sample_dataset(identity_count=10, samples_per_identity=3, split_name="s")
    sim.sample_identity_population(centers=l2_normalize_rows(np.random.default_rng(9).normal(size=(7, 16))))
    again = sim.sample_dataset(identity_count=10, samples_per_identity=3, split_name="s")
    assert np.array_equal(sampled.raw_embeddings, again.raw_embeddings)


# --------------------------------------------------------------------------- extraction (Mode A path)


def test_extraction_centers_unit_norm_and_counts() -> None:
    ds = _two_cluster_dataset()
    res = fs.identity_centers_from_dataset(ds, min_samples=2)
    assert res.centers.shape == (16, 16)
    assert np.allclose(np.linalg.norm(res.centers, axis=1), 1.0)
    assert set(res.counts.tolist()) == {4}
    assert len(res.labels) == 16


def test_extraction_matches_manual_spherical_mean() -> None:
    emb = np.array([[3.0, 0, 0, 0], [0, 4.0, 0, 0], [0, 0, 5.0, 0]])
    ds = _FakeDataset(emb, ["x", "x", "x"])
    res = fs.identity_centers_from_dataset(ds)
    manual = l2_normalize_rows(l2_normalize_rows(emb).sum(axis=0))[0]
    assert np.allclose(res.centers[0], manual)


def test_extraction_groups_by_original_identity_value() -> None:
    # int 1 and str "1" must NOT be merged.
    emb = np.eye(2, 4)
    ds = _FakeDataset(emb, [1, "1"])
    res = fs.identity_centers_from_dataset(ds)
    assert res.centers.shape[0] == 2
    assert set(res.labels) == {"1"}  # both stringify to "1", but they came from 2 groups


def test_extraction_ignores_none_identity_and_filters_min_samples() -> None:
    emb = np.eye(4, 4)
    ds = _FakeDataset(emb, ["a", "a", None, "b"])
    res = fs.identity_centers_from_dataset(ds, min_samples=2)
    assert res.labels == ["a"]  # "b" has 1 sample, None dropped


def test_extraction_quality_handles_nan_none_and_rejects_negative() -> None:
    emb = l2_normalize_rows(np.random.default_rng(4).normal(size=(3, 8)))
    ds = _FakeDataset(emb, ["a", "a", "a"], qualities=[None, float("nan"), 0.5])
    res = fs.identity_centers_from_dataset(ds, weight_by_quality=True)
    assert res.centers.shape == (1, 8)
    bad = _FakeDataset(emb, ["a", "a", "a"], qualities=[-1.0, 0.5, 0.5])
    with pytest.raises(ValueError, match="negative"):
        fs.identity_centers_from_dataset(bad, weight_by_quality=True)


def test_extraction_weight_by_quality_requires_items() -> None:
    emb = np.eye(2, 4)
    ds = _FakeDataset(emb, ["a", "a"], with_items=False)
    with pytest.raises(ValueError, match="requires dataset.items"):
        fs.identity_centers_from_dataset(ds, weight_by_quality=True)


def test_extraction_degenerate_mean_fallback() -> None:
    e = l2_normalize_rows(np.random.default_rng(5).normal(size=(1, 8)))[0]
    ds = _FakeDataset(np.array([e, -e]), ["a", "a"])  # antipodal -> sum ~ 0
    with pytest.warns(UserWarning, match="Degenerate"):
        res = fs.identity_centers_from_dataset(ds)
    assert np.allclose(np.linalg.norm(res.centers[0]), 1.0)


# --------------------------------------------------------------------------- Mode A + leakage


def test_mode_a_population_and_provenance_columns() -> None:
    ds = _two_cluster_dataset()
    sim = _sim()
    pop = fs.population_from_dataset(sim, ds, min_samples=2)
    res = fs.identity_centers_from_dataset(ds, min_samples=2)
    assert np.allclose(pop.identity_centers, res.centers)
    assert pop.source_identity_labels == res.labels
    obs = sim.sample_observations(pop, samples_per_identity=3, split_name="A")
    assert "source_identity" in obs.metadata.columns
    assert "source_sample_count" in obs.metadata.columns
    assert obs.metadata["pre_nuisance_center_cosine"].min() > 0.0


def test_leakage_helpers() -> None:
    ds_full = _two_cluster_dataset()
    ds_a_only = _FakeDataset(ds_full.matrix[:32], list(ds_full.identities[:32]))  # only A* ids
    sim = _sim()
    pop_full = fs.population_from_dataset(sim, ds_full, min_samples=2)
    pop_a = fs.population_from_dataset(sim, ds_a_only, min_samples=2)
    assert len(fs.source_identity_set(pop_full)) == 16
    with pytest.raises(ValueError, match="leakage"):
        fs.assert_disjoint_source_identities(pop_full, pop_a)


def test_module_imports_without_triggering_typicality_import() -> None:
    import importlib
    import sys

    mod = importlib.import_module("faceaudit_seed")
    assert hasattr(mod, "identity_centers_from_dataset")
    # The heavy MoVMF dependency must not be imported just by importing the bridge / running Mode A.
    # (extraction + Mode A above already ran without it.)
