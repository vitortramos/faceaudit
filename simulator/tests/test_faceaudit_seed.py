"""Mode B (MoVMF) integration tests — require faceaudit (TypicalityModel + EmbeddingDataset).

    .venv/Scripts/python.exe -m pytest simulator/tests/test_faceaudit_seed.py
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faceaudit")

from adaface_embedding_simulator import (  # noqa: E402
    AdaFaceEmbeddingSimulator,
    config_from_dict,
    config_to_dict,
    l2_normalize_rows,
    make_cooperative_plda_config,
)
import faceaudit_seed as fs  # noqa: E402
from faceaudit_concrete_adapter import to_embedding_dataset  # noqa: E402


class _Item:
    def __init__(self, embedding, identity, quality_score=None):
        self.embedding, self.identity, self.quality_score = embedding, identity, quality_score


class _FakeDataset:
    def __init__(self, embeddings, identities, qualities=None):
        self._embeddings = np.asarray(embeddings, dtype=np.float64)
        self._identities = list(identities)
        quals = qualities if qualities is not None else [None] * len(self._identities)
        self.items = [_Item(e, i, q) for e, i, q in zip(self._embeddings, self._identities, quals)]

    @property
    def matrix(self):
        return self._embeddings

    @property
    def identities(self):
        return np.array(self._identities, dtype=object)


def _two_cluster_dataset(*, dim=16, per_cluster=10, per_identity=4, norm=25.0, jitter_norm=True, seed=0):
    rng = np.random.default_rng(seed)
    mu = {"A": l2_normalize_rows(np.eye(1, dim, 0))[0], "B": l2_normalize_rows(np.eye(1, dim, 1))[0]}
    embs, ids = [], []
    for cluster in ("A", "B"):
        for k in range(per_cluster):
            center = l2_normalize_rows(mu[cluster] + 0.08 * rng.normal(size=dim))[0]
            for _ in range(per_identity):
                u = l2_normalize_rows(center + 0.04 * rng.normal(size=dim))[0]
                scale = float(rng.uniform(0.7, 1.4)) if jitter_norm else 1.0
                embs.append(norm * scale * u)
                ids.append(f"{cluster}{k}")
    return _FakeDataset(embs, ids)


def _nn_cos(centers):
    gram = centers @ centers.T
    np.fill_diagonal(gram, -np.inf)
    return float(gram.max(axis=1).mean())


def test_mode_b_fixed_k_builds_valid_config() -> None:
    ds = _two_cluster_dataset()
    config, summary = fs.simulator_config_from_dataset(ds, n_components=2, random_state=1)
    config.validate()
    assert config.profile_name == "FACEAUDIT_MOVMF"
    assert config.population.component_count == 2
    assert np.allclose(np.linalg.norm(config.population.component_means, axis=1), 1.0)
    assert abs(float(np.sum(config.population.component_weights)) - 1.0) < 1e-9
    assert np.all(np.asarray(config.population.identity_center_kappas) > 0)
    assert summary["selected_k"] == 2
    assert summary["n_identity_centers"] == 20
    assert summary["norm_summary"]["unit_norm_warning"] is False
    assert abs(summary["norm_summary"]["norm_mean"] - 25.0) < 2.0


def test_mode_b_auto_k_selects_within_candidates() -> None:
    ds = _two_cluster_dataset()
    config, summary = fs.simulator_config_from_dataset(
        ds, component_candidates=(1, 2, 4), max_k=4, random_state=1
    )
    assert summary["selected_k"] in {1, 2, 4}
    assert set(summary["criterion_values"]) <= {1, 2, 4}


def test_mode_b_warns_on_unit_norms() -> None:
    ds = _two_cluster_dataset(norm=1.0, jitter_norm=False)  # already on the sphere
    with pytest.warns(UserWarning, match="L2-normalized"):
        _, summary = fs.simulator_config_from_dataset(ds, n_components=2)
    assert summary["norm_summary"]["unit_norm_warning"] is True


def test_mode_b_config_roundtrips() -> None:
    ds = _two_cluster_dataset()
    config, _ = fs.simulator_config_from_dataset(ds, n_components=2)
    assert config_to_dict(config_from_dict(config_to_dict(config))) == config_to_dict(config)


def test_mode_b_ghost_centers_resemble_real_more_than_generic() -> None:
    ds = _two_cluster_dataset()
    real_centers = fs.identity_centers_from_dataset(ds).centers
    config, _ = fs.simulator_config_from_dataset(ds, n_components=2, fit_norm=False, random_state=1)
    ghost = AdaFaceEmbeddingSimulator(config, random_seed=3).sample_identity_population(identity_count=200)
    generic = AdaFaceEmbeddingSimulator(
        make_cooperative_plda_config(dimension=16), random_seed=3
    ).sample_identity_population(identity_count=200)
    real_nn = _nn_cos(real_centers)
    ghost_nn = _nn_cos(ghost.identity_centers)
    generic_nn = _nn_cos(generic.identity_centers)
    # Ghost population (fitted to real centres) is closer to the real clustering than a generic one.
    assert abs(ghost_nn - real_nn) < abs(generic_nn - real_nn)


def test_mode_b_requires_two_centers() -> None:
    ds = _FakeDataset(np.eye(2, 8) * 25.0, ["solo", "solo"])  # single identity
    with pytest.raises(ValueError, match="At least two identity centers"):
        fs.simulator_config_from_dataset(ds, n_components=1)


def test_roundtrip_with_real_embedding_dataset() -> None:
    # Generate synthetic data, adapt to a REAL EmbeddingDataset, then seed back through the bridge.
    sim = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=32), random_seed=7)
    synthetic = sim.sample_dataset(identity_count=40, samples_per_identity=4, split_name="real")
    real_ds = to_embedding_dataset(synthetic, embedding_view="raw")

    pop = fs.population_from_dataset(sim, real_ds, min_samples=2)
    assert pop.source_identity_labels is not None
    assert len(pop.source_identity_labels) == 40

    config, summary = fs.simulator_config_from_dataset(real_ds, component_candidates=(1, 2, 4, 8))
    config.validate()
    assert summary["dimension"] == 32
    assert config.population.component_count == summary["selected_k"]
