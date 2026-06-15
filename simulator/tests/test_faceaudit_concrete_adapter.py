"""Tests for the concrete FaceAudit adapter (imports the real DTOs).

Run from the repo root::

    .venv/Scripts/python.exe -m pytest simulator/tests/test_faceaudit_concrete_adapter.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)
from faceaudit_concrete_adapter import (
    cross_condition_to_embedding_datasets,
    to_embedding_dataset,
)

from faceaudit.models.embedding_dataset import EmbeddingDataset


@pytest.fixture(scope="module")
def dataset():
    simulator = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=48), random_seed=7)
    return simulator.sample_dataset(
        identity_count=40, samples_per_identity=(3, 6), split_name="dev", identity_offset=1000
    )


def test_basic_mapping(dataset) -> None:
    ds = to_embedding_dataset(dataset)
    md = dataset.metadata.reset_index(drop=True)

    assert isinstance(ds, EmbeddingDataset)
    assert len(ds) == len(md)
    assert ds.matrix.shape == (len(md), 48)
    assert ds.matrix.dtype == np.float32

    for i, item in enumerate(ds.items):
        assert item.identity == str(int(md.loc[i, "identity_id"]))
        assert item.quality_score == pytest.approx(float(md.loc[i, "latent_quality"]))
        assert item.source == str(md.loc[i, "acquisition_domain"])
        assert item.origin == "adaface_simulator"
        assert item.pose == {}


def test_raw_preserves_norm_and_normalized_is_unit(dataset) -> None:
    raw = to_embedding_dataset(dataset, embedding_view="raw")
    norm = to_embedding_dataset(dataset, embedding_view="normalized")
    assert np.allclose(np.linalg.norm(raw.matrix, axis=1), dataset.metadata["embedding_norm"], rtol=1e-4)
    assert np.allclose(np.linalg.norm(norm.matrix, axis=1), 1.0, atol=1e-4)


def test_face_hashes_unique_and_provenance(dataset) -> None:
    ds = to_embedding_dataset(dataset)
    hashes = [it.face_hash for it in ds.items]
    assert len(set(hashes)) == len(hashes)

    meta = ds.loaded_metadata
    assert meta["generator"] == "adaface_embedding_simulator"
    assert meta["profile_name"] == "COOPERATIVE_PLDA_BASELINE"
    assert meta["split"] == "dev"
    assert meta["dimension"] == "48"
    assert meta["n_samples"] == str(len(ds))


def test_domain_filter(dataset) -> None:
    full = to_embedding_dataset(dataset)
    controlled = to_embedding_dataset(dataset, domain_filter="controlled")
    assert {it.source for it in controlled.items} == {"controlled"}
    assert len(controlled) < len(full)
    assert controlled.loaded_metadata["domain_filter"] == "controlled"


def test_parquet_roundtrip(dataset, tmp_path: Path) -> None:
    ds = to_embedding_dataset(dataset)
    path = tmp_path / "synthetic_dev.parquet"
    ds.to_parquet(path)
    reloaded = EmbeddingDataset.from_parquet(path)

    assert len(reloaded) == len(ds)
    assert np.allclose(reloaded.matrix, ds.matrix, rtol=1e-5, atol=1e-6)
    assert [it.identity for it in reloaded.items] == [it.identity for it in ds.items]
    assert [it.source for it in reloaded.items] == [it.source for it in ds.items]


def test_cross_condition_shares_identities() -> None:
    simulator = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=32), random_seed=13)
    aligned = simulator.sample_cross_condition_datasets(
        identity_count=30,
        samples_per_identity_by_domain={"controlled": 3, "surveillance": 4},
        population_random_seed=88,
    )
    mapped = cross_condition_to_embedding_datasets(aligned)

    assert set(mapped) == {"controlled", "surveillance"}
    ids_a = {it.identity for it in mapped["controlled"].items}
    ids_b = {it.identity for it in mapped["surveillance"].items}
    assert ids_a == ids_b
    assert {it.source for it in mapped["controlled"].items} == {"controlled"}
    assert {it.source for it in mapped["surveillance"].items} == {"surveillance"}
