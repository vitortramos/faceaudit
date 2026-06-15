from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np

from adaface_embedding_simulator import (
    AcquisitionDomainConfig,
    AdaFaceEmbeddingSimulator,
    CouplingConfig,
    DirectionalConfig,
    PairSamplingStrategy,
    PopulationConfig,
    RadialConfig,
    SimulatorConfig,
    SyntheticEmbeddingDataset,
    make_cooperative_plda_config,
    make_cross_condition_stress_config,
    sample_pair_table,
    sample_uniform_directions,
    sample_vmf,
)


def _small_config() -> SimulatorConfig:
    return replace(make_cooperative_plda_config(dimension=32), directional=replace(
        make_cooperative_plda_config(dimension=32).directional,
        anisotropic_rank=4,
    ))


def test_sample_vmf_unit_norm_and_mean_direction_alignment() -> None:
    rng = np.random.default_rng(1)
    mean = np.zeros(16)
    mean[0] = 1.0
    samples = sample_vmf(np.repeat(mean[np.newaxis, :], 5000, axis=0), 80.0, rng)
    assert np.allclose(np.linalg.norm(samples, axis=1), 1.0, atol=1e-10)
    empirical_direction = samples.mean(axis=0)
    empirical_direction /= np.linalg.norm(empirical_direction)
    assert float(empirical_direction @ mean) > 0.995


def test_vmf_resultant_length_increases_with_kappa() -> None:
    rng = np.random.default_rng(2)
    means = np.repeat(np.eye(1, 12, 0), 3000, axis=0)
    low = sample_vmf(means, 5.0, rng)
    high = sample_vmf(means, 100.0, rng)
    assert np.linalg.norm(high.mean(axis=0)) > np.linalg.norm(low.mean(axis=0))


def test_dataset_embeddings_and_metadata_are_consistent() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=10)
    dataset = simulator.sample_dataset(identity_count=30, samples_per_identity=4, split_name="test")
    assert np.allclose(np.linalg.norm(dataset.normalized_embeddings, axis=1), 1.0, atol=1e-10)
    assert np.allclose(np.linalg.norm(dataset.raw_embeddings, axis=1), dataset.metadata["embedding_norm"])
    assert "combined_tangent_step_norm" in dataset.metadata


def test_split_sampling_is_independent_of_call_order() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=11)
    first = simulator.sample_dataset(identity_count=20, samples_per_identity=3, split_name="dev", identity_offset=100)
    _ = simulator.sample_dataset(identity_count=10, samples_per_identity=2, split_name="inserted", identity_offset=500)
    second = simulator.sample_dataset(identity_count=20, samples_per_identity=3, split_name="dev", identity_offset=100)
    assert np.array_equal(first.raw_embeddings, second.raw_embeddings)
    assert first.metadata.equals(second.metadata)


def test_explicit_seed_is_reproducible() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=12)
    first = simulator.sample_dataset(identity_count=10, samples_per_identity=3, split_name="a", random_seed=90, population_random_seed=91)
    second = simulator.sample_dataset(identity_count=10, samples_per_identity=3, split_name="b", random_seed=90, population_random_seed=91)
    assert np.array_equal(first.raw_embeddings, second.raw_embeddings)


def test_cross_condition_datasets_share_identity_population() -> None:
    simulator = AdaFaceEmbeddingSimulator(make_cross_condition_stress_config(dimension=24), random_seed=13)
    aligned = simulator.sample_cross_condition_datasets(
        identity_count=25,
        samples_per_identity_by_domain={"controlled": 2, "surveillance": 3},
        split_name="aligned",
        population_random_seed=88,
    )
    controlled = aligned["controlled"]
    surveillance = aligned["surveillance"]
    assert np.array_equal(controlled.identity_centers, surveillance.identity_centers)
    assert set(controlled.metadata["identity_id"]) == set(surveillance.metadata["identity_id"])
    assert set(controlled.metadata["acquisition_domain"]) == {"controlled"}
    assert set(surveillance.metadata["acquisition_domain"]) == {"surveillance"}


def test_cross_condition_profile_uses_distinct_domain_bases() -> None:
    simulator = AdaFaceEmbeddingSimulator(make_cross_condition_stress_config(dimension=24), random_seed=14)
    left = simulator.domain_anisotropic_bases["controlled"]
    right = simulator.domain_anisotropic_bases["surveillance"]
    assert left is not None and right is not None
    assert not np.array_equal(left, right)


def test_radial_regime_probabilities_match_configuration_when_decoupled() -> None:
    config = SimulatorConfig(
        dimension=16,
        population=PopulationConfig(component_count=1),
        radial=RadialConfig(),
        directional=DirectionalConfig(),
        coupling=CouplingConfig(quality_affects_regime_probabilities=False),
        acquisition_domains=(AcquisitionDomainConfig(
            name="default",
            central_regime_probability=0.70,
            low_tail_probability=0.20,
            high_tail_probability=0.10,
        ),),
    )
    simulator = AdaFaceEmbeddingSimulator(config, random_seed=15)
    dataset = simulator.sample_dataset(identity_count=2000, samples_per_identity=10, split_name="regimes")
    frequencies = dataset.metadata["radial_regime"].value_counts(normalize=True)
    assert abs(float(frequencies["central"]) - 0.70) < 0.02
    assert abs(float(frequencies["low_tail"]) - 0.20) < 0.02
    assert abs(float(frequencies["high_tail"]) - 0.10) < 0.02


def test_pair_sampler_labels_are_correct() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=16)
    dataset = simulator.sample_dataset(identity_count=30, samples_per_identity=4, split_name="pairs")
    pairs = sample_pair_table(dataset, genuine_pair_count=300, random_impostor_pair_count=300, random_seed=7)
    genuine = pairs[pairs["is_genuine"] == 1]
    impostor = pairs[pairs["is_genuine"] == 0]
    assert np.all(genuine["left_identity_id"] == genuine["right_identity_id"])
    assert np.all(impostor["left_identity_id"] != impostor["right_identity_id"])


def test_pair_sampler_balanced_observation_mode() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=17)
    dataset = simulator.sample_dataset(identity_count=20, samples_per_identity=(2, 5), split_name="pairs")
    pairs = sample_pair_table(
        dataset,
        genuine_pair_count=100,
        random_impostor_pair_count=100,
        random_seed=8,
        strategy=PairSamplingStrategy.BALANCED_OBSERVATION,
    )
    assert len(pairs) == 200


def test_norm_diagnostics_reports_actual_bins() -> None:
    simulator = AdaFaceEmbeddingSimulator(_small_config(), random_seed=18)
    dataset = simulator.sample_dataset(identity_count=20, samples_per_identity=3, split_name="bins")
    result = dataset.norm_bin_diagnostics(bin_count=5)
    assert set(["requested_bin_count", "actual_bin_count"]).issubset(result.columns)
    assert set(result["requested_bin_count"]) == {5}
