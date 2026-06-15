from dataclasses import dataclass

from adaface_embedding_simulator import AdaFaceEmbeddingSimulator, make_cooperative_plda_config
from faceaudit_adapter import dataset_to_embedding_records, to_faceaudit_embedding_dataset


@dataclass
class DummyInfo:
    embedding: object
    identity: str
    metadata: dict


@dataclass
class DummyDataset:
    embeddings: list


def test_plain_records_and_factory_adapter() -> None:
    simulator = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=16), random_seed=21)
    dataset = simulator.sample_dataset(identity_count=4, samples_per_identity=2, split_name="adapter")
    records = dataset_to_embedding_records(dataset, embedding_view="normalized")
    assert len(records) == 8
    converted = to_faceaudit_embedding_dataset(
        dataset,
        embedding_info_factory=DummyInfo,
        embedding_dataset_factory=DummyDataset,
        dataset_factory_keyword="embeddings",
    )
    assert len(converted.embeddings) == 8
