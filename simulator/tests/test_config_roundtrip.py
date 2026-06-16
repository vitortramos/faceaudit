"""Round-trip tests for config_to_dict <-> config_from_dict (v0.2).

Run from the repo root::

    .venv/Scripts/python.exe -m pytest simulator/tests/test_config_roundtrip.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaface_embedding_simulator import (
    AcquisitionDomainConfig,
    AdaFaceEmbeddingSimulator,
    NuisanceStructureMode,
    PopulationConfig,
    SimulatorConfig,
    TangentUpdateMode,
    config_from_dict,
    config_from_manifest,
    config_to_dict,
    make_cooperative_plda_config,
    make_cross_condition_stress_config,
    make_idperturb_cap_config,
    make_spherical_baseline_config,
)

_PROFILES = [
    make_spherical_baseline_config,
    make_cooperative_plda_config,
    make_cross_condition_stress_config,
    make_idperturb_cap_config,
]


@pytest.mark.parametrize("factory", _PROFILES)
def test_dict_roundtrip_is_idempotent_and_json_safe(factory) -> None:
    config = factory(dimension=64)
    first = config_to_dict(config)
    json.dumps(first)  # must not raise
    second = config_to_dict(config_from_dict(first))
    assert first == second


@pytest.mark.parametrize("factory", _PROFILES)
def test_functional_roundtrip_matches_bit_for_bit(factory) -> None:
    config = factory(dimension=64)
    reloaded = config_from_dict(config_to_dict(config))

    a = AdaFaceEmbeddingSimulator(config, random_seed=7).sample_dataset(
        identity_count=25, samples_per_identity=(2, 5), split_name="t"
    )
    b = AdaFaceEmbeddingSimulator(reloaded, random_seed=7).sample_dataset(
        identity_count=25, samples_per_identity=(2, 5), split_name="t"
    )
    assert np.array_equal(a.raw_embeddings, b.raw_embeddings)
    pd.testing.assert_frame_equal(a.metadata, b.metadata)


def test_enum_fields_survive_as_string_values() -> None:
    config = make_cross_condition_stress_config(dimension=32)
    payload = config_to_dict(config)
    assert payload["directional"]["tangent_update_mode"] == TangentUpdateMode.EXPONENTIAL_MAP.value
    assert payload["directional"]["nuisance_structure_mode"] == NuisanceStructureMode.HIERARCHICAL.value

    reloaded = config_from_dict(payload)
    # The reconstructed config accepts the string value and still drives generation.
    reloaded.validate()
    AdaFaceEmbeddingSimulator(reloaded, random_seed=1).sample_dataset(
        identity_count=5, samples_per_identity=2, split_name="t"
    )


def test_numpy_array_fields_serialize_and_reproduce() -> None:
    config = SimulatorConfig(
        dimension=16,
        population=PopulationConfig(
            component_count=3,
            component_weights=np.array([0.5, 0.3, 0.2]),
            identity_center_kappas=np.array([40.0, 60.0, 80.0]),
            component_mean_global_kappa=10.0,
        ),
        acquisition_domains=(AcquisitionDomainConfig(name="default"),),
    )
    payload = config_to_dict(config)
    json.dumps(payload)
    assert payload["population"]["component_weights"] == [0.5, 0.3, 0.2]
    assert payload["population"]["identity_center_kappas"] == [40.0, 60.0, 80.0]

    reloaded = config_from_dict(payload)
    a = AdaFaceEmbeddingSimulator(config, random_seed=3).sample_dataset(
        identity_count=12, samples_per_identity=3, split_name="t"
    )
    b = AdaFaceEmbeddingSimulator(reloaded, random_seed=3).sample_dataset(
        identity_count=12, samples_per_identity=3, split_name="t"
    )
    assert np.allclose(a.raw_embeddings, b.raw_embeddings)


def test_config_from_manifest_roundtrip(tmp_path: Path) -> None:
    config = make_cooperative_plda_config(dimension=48)
    dataset = AdaFaceEmbeddingSimulator(config, random_seed=9).sample_dataset(
        identity_count=8, samples_per_identity=2, split_name="m"
    )
    manifest_path = dataset.save_manifest(tmp_path / "m_manifest.json")

    reloaded = config_from_manifest(manifest_path)
    assert config_to_dict(reloaded) == config_to_dict(config)


def test_missing_keys_fall_back_to_defaults() -> None:
    payload = config_to_dict(make_spherical_baseline_config(dimension=16))
    payload.pop("coupling")  # optional block
    payload.pop("profile_name")
    reloaded = config_from_dict(payload)
    assert reloaded.profile_name == "custom"
    assert reloaded.coupling.quality_affects_intra_kappa is True


def test_unknown_key_is_rejected() -> None:
    payload = config_to_dict(make_spherical_baseline_config(dimension=16))
    payload["radial"]["bogus_field"] = 1.0
    with pytest.raises(TypeError):
        config_from_dict(payload)
