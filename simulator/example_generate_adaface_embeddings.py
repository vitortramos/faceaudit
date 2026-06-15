"""Example: generate reproducible disjoint splits and aligned cross-condition data."""

from pathlib import Path

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
    make_cross_condition_stress_config,
    sample_pair_table,
)


def main() -> None:
    output_dir = Path("synthetic_adaface_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    simulator = AdaFaceEmbeddingSimulator(
        make_cooperative_plda_config(dimension=512),
        random_seed=2026,
    )
    train = simulator.sample_dataset(
        identity_count=2_000,
        samples_per_identity=(5, 12),
        split_name="train",
        identity_offset=0,
        random_seed=101,
        population_random_seed=1001,
    )
    dev = simulator.sample_dataset(
        identity_count=500,
        samples_per_identity=(4, 10),
        split_name="dev",
        identity_offset=100_000,
        random_seed=102,
        population_random_seed=1002,
    )
    test = simulator.sample_dataset(
        identity_count=1_000,
        samples_per_identity=(4, 10),
        split_name="test",
        identity_offset=200_000,
        random_seed=103,
        population_random_seed=1003,
    )

    for dataset, split_name in [(train, "train"), (dev, "dev"), (test, "test")]:
        dataset.save_bundle(output_dir, split_name)

    sample_pair_table(
        dev,
        genuine_pair_count=20_000,
        random_impostor_pair_count=20_000,
        random_seed=201,
    ).to_csv(output_dir / "dev_pairs.csv", index=False)
    sample_pair_table(
        test,
        genuine_pair_count=50_000,
        random_impostor_pair_count=50_000,
        random_seed=202,
    ).to_csv(output_dir / "test_pairs.csv", index=False)

    stress = AdaFaceEmbeddingSimulator(
        make_cross_condition_stress_config(dimension=512),
        random_seed=3030,
    )
    aligned = stress.sample_cross_condition_datasets(
        identity_count=800,
        identity_offset=300_000,
        split_name="cross_condition_test",
        samples_per_identity_by_domain={"controlled": (3, 6), "surveillance": (3, 6)},
        sessions_per_identity=(1, 3),
        population_random_seed=4001,
        observation_random_seeds={"controlled": 4002, "surveillance": 4003},
    )
    for domain_name, dataset in aligned.items():
        dataset.save_bundle(output_dir, f"cross_condition_{domain_name}")

    print(f"Files created in: {output_dir.resolve()}")
    print("Train shape:", train.raw_embeddings.shape)
    print("Dev shape:  ", dev.raw_embeddings.shape)
    print("Test shape: ", test.raw_embeddings.shape)
    print("Aligned identities:", aligned["controlled"].metadata["identity_id"].nunique())
    print(test.norm_bin_diagnostics(bin_count=10).to_string(index=False))


if __name__ == "__main__":
    main()
