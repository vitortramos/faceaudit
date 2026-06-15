"""Reference adapter helpers for integrating synthetic data into FaceAudit.

The concrete FaceAudit constructors are intentionally injected because their
import path and DTO signatures belong to the host project, not to this small
standalone simulator package.
"""

from __future__ import annotations

from typing import Any, Callable, Literal
import numpy as np

from adaface_embedding_simulator import SyntheticEmbeddingDataset

EmbeddingView = Literal["raw", "normalized"]


def dataset_to_embedding_records(
    dataset: SyntheticEmbeddingDataset,
    *,
    embedding_view: EmbeddingView = "normalized",
    domain_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Convert a synthetic dataset to plain records suitable for DTO factories."""
    if embedding_view not in {"raw", "normalized"}:
        raise ValueError("embedding_view must be 'raw' or 'normalized'.")
    frame = dataset.metadata.reset_index(drop=True)
    mask = np.ones(len(frame), dtype=bool)
    if domain_filter is not None:
        mask = frame["acquisition_domain"].to_numpy() == domain_filter
    rows = np.flatnonzero(mask)
    matrix = dataset.raw_embeddings if embedding_view == "raw" else dataset.normalized_embeddings
    output: list[dict[str, Any]] = []
    for row_index in rows:
        metadata = frame.iloc[int(row_index)].to_dict()
        output.append(
            {
                "embedding": matrix[int(row_index)].copy(),
                "identity": str(metadata["identity_id"]),
                "metadata": metadata,
            }
        )
    return output


def to_faceaudit_embedding_dataset(
    dataset: SyntheticEmbeddingDataset,
    *,
    embedding_info_factory: Callable[..., Any],
    embedding_dataset_factory: Callable[..., Any],
    embedding_view: EmbeddingView = "normalized",
    domain_filter: str | None = None,
    dataset_factory_keyword: str | None = None,
    include_metadata: bool = True,
) -> Any:
    """Create FaceAudit DTOs using injected constructors.

    ``embedding_info_factory`` is normally ``EmbeddingInfo`` and must accept
    ``embedding`` and ``identity``. When ``include_metadata=True`` it must also
    accept ``metadata``. ``embedding_dataset_factory`` is normally
    ``EmbeddingDataset``. If its constructor expects a keyword such as
    ``embeddings``, pass that name in ``dataset_factory_keyword``.
    """
    records = dataset_to_embedding_records(
        dataset,
        embedding_view=embedding_view,
        domain_filter=domain_filter,
    )
    infos = []
    for record in records:
        kwargs = {"embedding": record["embedding"], "identity": record["identity"]}
        if include_metadata:
            kwargs["metadata"] = record["metadata"]
        infos.append(embedding_info_factory(**kwargs))
    if dataset_factory_keyword is None:
        return embedding_dataset_factory(infos)
    return embedding_dataset_factory(**{dataset_factory_keyword: infos})
