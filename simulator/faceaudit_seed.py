"""Anchor the simulator population in a real FaceAudit ``EmbeddingDataset``.

Sampled identity centres come from a generic vMF mixture and may not respect the fine empirical
geometry of real face embeddings. This bridge feeds the population from real data, in two modes:

* **Mode A (anchored)** — :func:`population_from_dataset`: per-identity spherical means of the real
  embeddings become the identity centres. Identities are real; only the within-identity dispersion
  (and the norm) is synthetic. NOTE: Mode A is NOT anonymization — it generates synthetic variation
  around real biometric anchors.
* **Mode B (distribution-matched)** — :func:`simulator_config_from_dataset`: fit a MoVMF to the real
  per-identity centres (FaceAudit's ``TypicalityModel``) and use μ_k/κ_k/π_k as a ``PopulationConfig``;
  the simulator then samples NEW ghost centres from an MoVMF approximation of the empirical centre
  distribution (a per-component isotropic mixture — first-order angular density, not the full
  manifold). Optionally also fits the radial (norm) reference.

Leakage caveat: the dataset used to anchor/fit must be DISJOINT in identities from any evaluation
set (use :func:`assert_disjoint_source_identities`). ``faceaudit`` is imported lazily inside the
functions that need it, so ``import simulator.faceaudit_seed`` works without faceaudit installed.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np

from adaface_embedding_simulator import (
    FloatArray,
    IntArray,
    PopulationConfig,
    RadialConfig,
    RadialReference,
    SimulatorConfig,
    SyntheticIdentityPopulation,
    l2_normalize_rows,
    make_cooperative_plda_config,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from faceaudit.models.embedding_dataset import EmbeddingDataset


def _import_typicality_model():
    from faceaudit.services.typicality_model import TypicalityModel

    return TypicalityModel


# ---------------------------------------------------------------------------
# Per-identity spherical centres
# ---------------------------------------------------------------------------


@dataclass
class IdentityCenterExtractionResult:
    """Real per-identity spherical-mean centres extracted from a dataset."""

    centers: FloatArray          # (M, D) unit-norm directions
    labels: list[str]            # original FaceAudit identity label per centre
    counts: IntArray             # number of real images that formed each centre
    quality_weighted: bool


def identity_centers_from_dataset(
    dataset: Any,
    *,
    min_samples: int = 1,
    weight_by_quality: bool = False,
    eps: float = 1e-12,
) -> IdentityCenterExtractionResult:
    """Compute one spherical-mean centre per identity: ``ĉ_i = normalize(Σ_j w_j u_ij)``.

    Groups by the ORIGINAL ``identity`` value (not ``str(...)`` — avoids merging ``1`` and
    ``"1"``), preserving first-occurrence order; ``None`` identities are ignored; identities with
    ``< min_samples`` are dropped. ``weight_by_quality`` weights each image by its
    ``quality_score`` (None/NaN -> weight 1; negative -> error; all-zero -> unweighted mean);
    a degenerate (near-zero) mean falls back to the best-quality (or first) image with a warning.

    ``min_samples`` default 1 is flexible; prefer 2-3 (1 = noisy anchor, 2+ stabler, 5+ best).
    """
    embeddings = np.asarray(dataset.matrix, dtype=np.float64)
    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("dataset.matrix must be a non-empty (N, D) array.")
    sample_count = embeddings.shape[0]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    unit = embeddings / np.maximum(norms, eps)

    identities = list(dataset.identities)
    if len(identities) != sample_count:
        raise ValueError("dataset.identities length does not match dataset.matrix rows.")

    if weight_by_quality:
        if not hasattr(dataset, "items"):
            raise ValueError("weight_by_quality=True requires dataset.items with quality_score.")
        weights = np.empty(sample_count, dtype=np.float64)
        for index, item in enumerate(dataset.items):
            quality = getattr(item, "quality_score", None)
            if quality is None or (isinstance(quality, float) and np.isnan(quality)):
                weights[index] = 1.0
            else:
                value = float(quality)
                if value < 0:
                    raise ValueError("quality weights cannot be negative.")
                weights[index] = value
    else:
        weights = np.ones(sample_count, dtype=np.float64)

    groups: dict[Any, list[int]] = {}
    for index, identity in enumerate(identities):
        if identity is None:
            continue
        groups.setdefault(identity, []).append(index)

    centers: list[FloatArray] = []
    labels: list[str] = []
    counts: list[int] = []
    for identity, rows in groups.items():
        if len(rows) < min_samples:
            continue
        row_idx = np.asarray(rows, dtype=np.int64)
        block = unit[row_idx]
        block_weights = weights[row_idx]
        if np.all(block_weights == 0):
            mean = block.mean(axis=0)
        else:
            mean = (block * block_weights[:, np.newaxis]).sum(axis=0)
        mean_norm = float(np.linalg.norm(mean))
        if mean_norm <= eps:
            warnings.warn(
                f"Degenerate spherical mean for identity {identity!r}; "
                "falling back to the best-quality (or first) image.",
                UserWarning,
                stacklevel=2,
            )
            best = row_idx[int(np.argmax(block_weights))] if np.any(block_weights > 0) else row_idx[0]
            center = unit[best]
        else:
            center = mean / mean_norm
        centers.append(center)
        labels.append(str(identity))
        counts.append(len(rows))

    if not centers:
        raise ValueError(f"No identity has >= {min_samples} samples.")

    return IdentityCenterExtractionResult(
        centers=np.asarray(centers, dtype=np.float64),
        labels=labels,
        counts=np.asarray(counts, dtype=np.int64),
        quality_weighted=bool(weight_by_quality),
    )


# ---------------------------------------------------------------------------
# Mode A — anchored population
# ---------------------------------------------------------------------------


def population_from_dataset(
    simulator: Any,
    dataset: Any,
    *,
    min_samples: int = 1,
    identity_offset: int = 0,
    random_seed: int | None = None,
    weight_by_quality: bool = False,
    stream_name: str = "dataset_seeded",
) -> SyntheticIdentityPopulation:
    """Build a population whose centres ARE the real per-identity spherical means (Mode A).

    The returned identities are real. Mode A is NOT anonymization: it produces synthetic variation
    around real biometric anchors. Use only for augmentation / methodological validation / internal
    experiments on an authorized base, and keep the anchor set disjoint from any evaluation set.
    """
    extraction = identity_centers_from_dataset(
        dataset, min_samples=min_samples, weight_by_quality=weight_by_quality
    )
    return simulator.sample_identity_population(
        centers=extraction.centers,
        identity_labels=extraction.labels,
        source_sample_counts=extraction.counts,
        identity_offset=identity_offset,
        random_seed=random_seed,
        stream_name=stream_name,
    )


# ---------------------------------------------------------------------------
# Mode B — distribution-matched population (MoVMF fitted to real centres)
# ---------------------------------------------------------------------------


def population_config_from_dataset(
    dataset: Any,
    *,
    n_components: int | None = None,
    component_candidates: Iterable[int] = (1, 2, 4, 8, 16),
    max_k: int | None = None,
    criterion: str = "bic",
    min_samples: int = 1,
    weight_by_quality: bool = False,
    random_state: int = 42,
    n_init: int = 5,
) -> tuple[PopulationConfig, dict]:
    """Fit a MoVMF to the real per-identity centres and return a ``PopulationConfig`` + summary.

    Fits on the per-identity CENTRES (not all observations — would mix intra/inter variance). With
    ``n_components`` set, fits that fixed K; otherwise selects K by ``criterion`` ('bic'/'aic') over
    ``component_candidates`` clamped to ``[1, min(M, max_k)]``.
    """
    TypicalityModel = _import_typicality_model()

    extraction = identity_centers_from_dataset(
        dataset, min_samples=min_samples, weight_by_quality=weight_by_quality
    )
    centers = extraction.centers
    n_centers, dimension = centers.shape
    if n_centers < 2:
        raise ValueError("At least two identity centers are required to fit a MoVMF.")

    common = dict(centering="none", random_state=random_state, n_init=n_init)
    criterion = criterion.lower()
    if criterion not in {"aic", "bic"}:
        raise ValueError("criterion must be 'aic' or 'bic'.")

    criterion_values: dict[int, float | None] = {}
    candidate_k: list[int] | None = None
    if n_components is not None:
        model = TypicalityModel(n_components=int(n_components), **common).fit(centers)
    else:
        effective_max_k = int(max_k) if max_k is not None else max(component_candidates)
        candidate_k = sorted(
            {int(k) for k in component_candidates if 1 <= int(k) <= min(n_centers, effective_max_k)}
        )
        if not candidate_k:
            candidate_k = [1]
        best = None
        best_score = float("inf")
        for k in candidate_k:
            try:
                candidate = TypicalityModel(n_components=k, **common).fit(centers)
                score = candidate.bic_ if criterion == "bic" else candidate.aic_
            except Exception as exc:  # noqa: BLE001 - numerical/EM failures per K
                criterion_values[k] = None
                continue
            finite = score is not None and np.isfinite(score)
            criterion_values[k] = float(score) if finite else None
            if finite and score < best_score:
                best_score = float(score)
                best = candidate
        if best is None:
            raise RuntimeError(f"No MoVMF converged. Per-K criterion: {criterion_values}.")
        model = best

    mus = np.asarray(model.component_mus, dtype=np.float64)
    kappas = np.asarray(model.component_kappas, dtype=np.float64)
    weights = np.asarray(model.component_weights, dtype=np.float64)
    fitted_k = int(mus.shape[0])

    population = PopulationConfig(
        component_count=fitted_k,
        component_means=mus,
        component_weights=weights,
        identity_center_kappas=kappas,
    )
    entropy = float(-(weights * np.log(np.maximum(weights, 1e-12))).sum())
    fit_summary = {
        "population_source": "faceaudit_movmf",
        "selected_k": fitted_k,
        "n_components_fixed": int(n_components) if n_components is not None else None,
        "candidate_k": candidate_k,
        "criterion": criterion,
        "criterion_values": criterion_values or None,
        "fit_seed": int(random_state),
        "n_identity_centers": int(n_centers),
        "dimension": int(dimension),
        "n_degenerate": len(model.get_degenerate_components()),
        "min_component_weight": float(weights.min()),
        "max_component_kappa": float(kappas.max()),
        "component_weight_entropy": entropy,
        "weight_by_quality": bool(weight_by_quality),
        "min_samples": int(min_samples),
    }
    return population, fit_summary


def simulator_config_from_dataset(
    dataset: Any,
    *,
    base_config: SimulatorConfig | None = None,
    fit_norm: bool = True,
    embedding_view_for_norm: str = "raw",
    warn_if_unit_norms: bool = True,
    norm_tol: float = 1e-3,
    **movmf_kwargs: Any,
) -> tuple[SimulatorConfig, dict]:
    """Full Mode-B config: MoVMF population fitted to real centres + optional radial norm fit.

    Starts from ``base_config`` (default ``make_cooperative_plda_config`` at the dataset's
    dimension), replaces the population, and — when ``fit_norm`` — replaces the radial reference
    with one fitted from the real norms (``RadialReference.fit`` accepts the (N, D) matrix). Warns
    if the dataset looks L2-normalized (``|mean(‖x‖)−1| < tol`` and ``std(‖x‖) < tol``), which would
    make the radial fit uninformative.
    """
    if embedding_view_for_norm != "raw":
        raise ValueError("embedding_view_for_norm currently supports only 'raw' (dataset.matrix).")

    matrix = np.asarray(dataset.matrix, dtype=np.float64)
    dimension = matrix.shape[1]
    base = base_config if base_config is not None else make_cooperative_plda_config(dimension=dimension)
    if base.dimension != dimension:
        raise ValueError(f"base_config.dimension {base.dimension} != dataset dimension {dimension}.")

    population, fit_summary = population_config_from_dataset(dataset, **movmf_kwargs)
    config = replace(base, population=population, profile_name="FACEAUDIT_MOVMF")

    if fit_norm:
        norms = np.linalg.norm(matrix, axis=1)
        if np.any(norms <= 0):
            raise ValueError("dataset has zero-norm embeddings; cannot assess radial reference.")
        # A robust radial reference needs the log-norms to vary (scaled MAD > 0). Constant or
        # L2-normalized norms are degenerate: warn and keep the base radial instead of crashing.
        log_norms = np.log(norms)
        robust_scale = 1.4826 * float(np.median(np.abs(log_norms - np.median(log_norms))))
        unit_norm_warning = bool(
            abs(float(norms.mean()) - 1.0) < norm_tol and float(norms.std()) < norm_tol
        )
        radial_fitted = robust_scale > 1e-9
        if not radial_fitted and warn_if_unit_norms:
            warnings.warn(
                "dataset norms are (near-)constant (possibly L2-normalized); keeping the base "
                "radial config and skipping the radial fit.",
                UserWarning,
                stacklevel=2,
            )
        if radial_fitted:
            config = replace(config, radial=RadialConfig.from_reference(RadialReference.fit(matrix)))
        fit_summary["norm_summary"] = {
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "unit_norm_warning": unit_norm_warning,
            "radial_fitted": radial_fitted,
        }
    else:
        fit_summary["norm_summary"] = None

    return config, fit_summary


# ---------------------------------------------------------------------------
# Leakage helpers (faceaudit-free)
# ---------------------------------------------------------------------------


def source_identity_set(population: SyntheticIdentityPopulation) -> set[str]:
    """Set of original identity labels a Mode-A population was anchored on (empty if sampled)."""
    if population.source_identity_labels is None:
        return set()
    return set(population.source_identity_labels)


def assert_disjoint_source_identities(
    population_a: SyntheticIdentityPopulation,
    population_b: SyntheticIdentityPopulation,
) -> None:
    """Raise if two anchored populations share any source identity (anchor↔eval leakage)."""
    common = source_identity_set(population_a) & source_identity_set(population_b)
    if common:
        sample = sorted(common)[:5]
        raise ValueError(
            f"Populations share {len(common)} source identities (leakage), e.g. {sample}."
        )
