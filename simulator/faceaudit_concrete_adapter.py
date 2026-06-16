"""Adapter: ``SyntheticEmbeddingDataset`` -> FaceAudit ``EmbeddingDataset``.

The real bridge to the host project: it imports the FaceAudit DTOs and maps the simulator's
metadata columns onto the actual ``EmbeddingInfo`` fields (``EmbeddingInfo(embedding, face_hash,
image_path, identity, file_name, ...)``).

Field mapping (forensically actionable covariates kept; embeddings cast to float32 to match
real extracted datasets):

* ``embedding``  <- ``raw_embeddings`` (default; norm preserved) or ``normalized_embeddings``.
* ``identity``   <- ``str(identity_id)``.
* ``quality_scores[quality_key]`` <- ``latent_quality`` (``"generic"`` bucket by default,
  which FaceAudit's ``quality_score`` property and BY_QUALITY balancing read).
* ``source``     <- ``acquisition_domain`` (per-sample handle for cross-condition splits).
* ``origin``     <- provenance tag (default ``"adaface_simulator"``).
* ``face_hash``  <- deterministic SHA-256 of ``(split, identity_id, sample_index, session_index)``.
* ``image_path`` / ``file_name`` <- deterministic synthetic paths.
* ``pose``       <- left empty (the simulator models nuisance, not literal head pose).

Continuous covariates not mapped here (``reference_scaled_log_norm_z``, ``intra_kappa``,
``radial_regime``, tangent-step diagnostics, ...) remain in the simulator metadata; the
simulator's own ``save_bundle`` persists them as CSV sidecars for full traceability. The L2
norm is recoverable from a raw embedding, so nothing essential for stratification is lost.
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np

from adaface_embedding_simulator import SyntheticEmbeddingDataset, __version__

from faceaudit.models.embedding_dataset import EmbeddingDataset
from faceaudit.models.embedding_info import EmbeddingInfo

EmbeddingView = Literal["raw", "normalized"]

_DEFAULT_ORIGIN = "adaface_simulator"


def _face_hash(split: str, identity_id: int, sample_index: int, session_index: int) -> str:
    payload = f"{split}|{identity_id}|{sample_index}|{session_index}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_embedding_dataset(
    dataset: SyntheticEmbeddingDataset,
    *,
    name: str | None = None,
    embedding_view: EmbeddingView = "raw",
    quality_key: str = "generic",
    origin: str = _DEFAULT_ORIGIN,
    domain_filter: str | None = None,
    dtype: np.dtype | type = np.float32,
) -> EmbeddingDataset:
    """Convert a synthetic dataset into a real FaceAudit :class:`EmbeddingDataset`.

    Args:
        dataset: Output of ``sample_observations`` / ``sample_dataset``.
        name: Dataset name. Defaults to ``f"synthetic_{split}"``.
        embedding_view: ``"raw"`` (norm preserved, default) or ``"normalized"``.
        quality_key: Key under which ``latent_quality`` is stored in ``quality_scores``.
        origin: Value written to every item's ``origin`` field.
        domain_filter: If set, keep only rows whose ``acquisition_domain`` equals it.
        dtype: NumPy dtype for stored embeddings (``float32`` by default).
    """
    if embedding_view == "raw":
        source_matrix = dataset.raw_embeddings
    elif embedding_view == "normalized":
        source_matrix = dataset.normalized_embeddings
    else:  # pragma: no cover - guarded by Literal at type-check time
        raise ValueError(f"embedding_view must be 'raw' or 'normalized', got {embedding_view!r}.")

    metadata = dataset.metadata.reset_index(drop=True)
    if domain_filter is not None:
        keep = metadata["acquisition_domain"].to_numpy() == domain_filter
        row_indices = np.flatnonzero(keep)
    else:
        row_indices = np.arange(len(metadata), dtype=np.int64)

    embeddings = np.asarray(source_matrix, dtype=dtype)
    split = str(metadata["split"].iloc[0]) if len(metadata) else "synthetic"

    identity_ids = metadata["identity_id"].to_numpy()
    sample_indices = metadata["sample_index"].to_numpy()
    session_indices = (
        metadata["session_index"].to_numpy()
        if "session_index" in metadata.columns
        else np.zeros(len(metadata), dtype=np.int64)
    )
    domains = metadata["acquisition_domain"].to_numpy()
    qualities = metadata["latent_quality"].to_numpy()

    items: list[EmbeddingInfo] = []
    for row in row_indices:
        row = int(row)
        identity_id = int(identity_ids[row])
        sample_index = int(sample_indices[row])
        session_index = int(session_indices[row])
        items.append(
            EmbeddingInfo(
                embedding=np.array(embeddings[row], dtype=dtype),
                face_hash=_face_hash(split, identity_id, sample_index, session_index),
                image_path=f"synthetic/{split}/{identity_id}/{sample_index}.png",
                identity=str(identity_id),
                file_name=f"{identity_id}_{sample_index}.png",
                source=str(domains[row]),
                origin=origin,
                face_image=None,
                quality_scores={quality_key: float(qualities[row])},
            )
        )

    dataset_name = name if name is not None else f"synthetic_{split}"
    loaded_metadata = {
        "generator": "adaface_embedding_simulator",
        "simulator_version": __version__,
        "profile_name": str(dataset.config.profile_name),
        "split": split,
        "embedding_view": embedding_view,
        "dimension": str(int(embeddings.shape[1]) if len(metadata) else dataset.config.dimension),
        "n_samples": str(int(row_indices.size)),
        "n_identities": str(int(np.unique(identity_ids[row_indices]).size) if row_indices.size else 0),
    }
    if domain_filter is not None:
        loaded_metadata["domain_filter"] = domain_filter

    return EmbeddingDataset(
        name=dataset_name,
        items=items,
        description=f"Synthetic AdaFace-like dataset ({embedding_view}) from split {split!r}.",
        loaded_metadata=loaded_metadata,
    )


def cross_condition_to_embedding_datasets(
    datasets_by_domain: dict[str, SyntheticEmbeddingDataset],
    *,
    embedding_view: EmbeddingView = "raw",
    quality_key: str = "generic",
    origin: str = _DEFAULT_ORIGIN,
    dtype: np.dtype | type = np.float32,
) -> dict[str, EmbeddingDataset]:
    """Map the output of ``sample_cross_condition_datasets`` to FaceAudit datasets.

    The returned datasets keep identical ``identity`` values across conditions, so they
    form a paired cross-condition set (each identity yields genuine cross-condition pairs).
    """
    return {
        domain_name: to_embedding_dataset(
            dataset,
            name=f"synthetic_{domain_name}",
            embedding_view=embedding_view,
            quality_key=quality_key,
            origin=origin,
            dtype=dtype,
        )
        for domain_name, dataset in datasets_by_domain.items()
    }
