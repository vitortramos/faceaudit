"""Manifest `environment` block — software-version capture for reproducibility audit.

Faceaudit-free: the block records python/numpy/pandas always and faceaudit only when
importable (degrades to None otherwise), so these tests run with or without faceaudit.

    .venv/Scripts/python.exe -m pytest simulator/tests/test_manifest_environment.py
"""

from __future__ import annotations

import json
from pathlib import Path

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    config_from_manifest,
    config_to_dict,
    make_cooperative_plda_config,
)


def _tiny(seed: int = 3):
    sim = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=16), random_seed=seed)
    return sim, sim.sample_dataset(identity_count=6, samples_per_identity=2, split_name="env")


def test_manifest_has_environment_block(tmp_path: Path) -> None:
    _, ds = _tiny()
    manifest = json.loads(Path(ds.save_manifest(tmp_path / "m.json")).read_text(encoding="utf-8"))

    assert "environment" in manifest
    env = manifest["environment"]
    assert set(env) == {"python", "numpy", "pandas", "faceaudit"}
    # Runtime numeric stack is always present; faceaudit may be None when absent.
    assert env["python"] and env["numpy"] and env["pandas"]


def test_environment_is_top_level_not_inside_config(tmp_path: Path) -> None:
    # Must be a sibling of `config`, never nested — otherwise config_from_dict would reject it.
    _, ds = _tiny()
    manifest = json.loads(Path(ds.save_manifest(tmp_path / "m.json")).read_text(encoding="utf-8"))
    assert "environment" not in manifest["config"]


def test_environment_block_does_not_break_config_roundtrip(tmp_path: Path) -> None:
    sim, ds = _tiny()
    reloaded = config_from_manifest(ds.save_manifest(tmp_path / "m.json"))
    assert config_to_dict(reloaded) == config_to_dict(sim.config)
