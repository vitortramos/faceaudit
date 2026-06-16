"""Synthetic AdaFace-like embedding generator for PLDA experiments.

This module does not implement the AdaFace neural network.  It implements a
configurable synthetic world for controlled experiments with embedding backends
such as cosine similarity, LDA, whitening, PLDA and spherical alternatives.

Version 0.2 adds:

* deterministic named RNG streams and explicit split seeds;
* separate identity-population and observation sampling APIs;
* aligned cross-condition datasets with shared latent identities;
* global, per-domain and hierarchical nuisance structures;
* fixed domain bias, domain-by-identity bias and session bias;
* retraction and exponential-map tangent updates;
* coupling switches for causal ablation studies;
* vectorized balanced pair sampling with optional deduplication;
* explicit qcut diagnostics when fewer bins can be created;
* manifest serialization helpers for reproducible experiments.

The default configuration remains an illustrative cooperative PLDA baseline.
For scientific claims, calibrate parameters against representative real
embeddings and report which synthetic profile was used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from hashlib import blake2b, sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This module requires pandas. Install it with: pip install pandas"
    ) from exc


__version__ = "0.3.0"

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StrArray = NDArray[np.str_]


# ---------------------------------------------------------------------------
# Enums and numerical helpers
# ---------------------------------------------------------------------------


class NuisanceStructureMode(str, Enum):
    """How anisotropic nuisance directions are shared across samples."""

    GLOBAL_SHARED = "global_shared"
    PER_DOMAIN = "per_domain"
    HIERARCHICAL = "hierarchical"


class TangentUpdateMode(str, Enum):
    """How a tangent-space nuisance step is mapped back to the hypersphere."""

    RETRACTION = "retraction"
    EXPONENTIAL_MAP = "exponential_map"


class PairSamplingStrategy(str, Enum):
    """Sampling weights used for random trial generation."""

    BALANCED_IDENTITY = "balanced_identity"
    BALANCED_OBSERVATION = "balanced_observation"


class IntraClassSamplingMode(str, Enum):
    """How within-identity directions are dispersed around the identity centre.

    * ``VMF`` — von Mises-Fisher concentration ``kappa`` (the v0.2 behaviour): a
      directional distribution with no hard angular floor.
    * ``IDPERTURB_CAP`` — cosine-uniform spherical cap (Boutros et al., IDperturb,
      CVPR 2026): the cosine to the centre is drawn ``s ~ U[lb, 1]``, giving an
      interpretable *maximum* angular excursion controlled by the lower bound ``lb``.
      NOTE: this is uniform *in cosine*, NOT uniform over the cap area in high dim.
    * ``HYBRID_VMF_CAP`` — vMF body with a hard cap floor: sample vMF, then redraw (or
      fall back to the cap) any sample whose cosine to the centre falls below ``lb``.
    """

    VMF = "vmf"
    IDPERTURB_CAP = "idperturb_cap"
    HYBRID_VMF_CAP = "hybrid_vmf_cap"


def _enum_value(value: str | Enum) -> str:
    return value.value if isinstance(value, Enum) else str(value)


def _as_float_array(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def l2_normalize_rows(values: ArrayLike, *, eps: float = 1e-12) -> FloatArray:
    """Normalize the rows of a 1D or 2D array to unit Euclidean norm."""
    array = _as_float_array(values, name="values")
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2:
        raise ValueError("values must be a one-dimensional or two-dimensional array.")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= eps):
        raise ValueError("Cannot normalize vectors whose norm is zero or too small.")
    return array / norms


def sample_uniform_directions(
    sample_count: int,
    dimension: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Sample vectors uniformly from the surface of the unit hypersphere."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if dimension < 2:
        raise ValueError("dimension must be at least 2.")
    return l2_normalize_rows(rng.normal(size=(sample_count, dimension)))


def _validate_probability_vector(values: ArrayLike, *, name: str) -> FloatArray:
    probabilities = _as_float_array(values, name=name)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.any(probabilities < 0):
        raise ValueError(f"{name} cannot contain negative values.")
    total = float(probabilities.sum())
    if total <= 0:
        raise ValueError(f"{name} must have a positive sum.")
    return probabilities / total


def _stable_seed(root_seed: int, *parts: object) -> int:
    """Derive a deterministic 64-bit seed independent of call order."""
    payload = "|".join([str(int(root_seed)), *(str(part) for part in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _orthonormal_basis(
    dimension: int,
    rank: int,
    rng: np.random.Generator,
) -> FloatArray | None:
    if rank == 0:
        return None
    raw = rng.normal(size=(dimension, rank))
    basis, _ = np.linalg.qr(raw, mode="reduced")
    return np.asarray(basis, dtype=np.float64)


def _householder_rotate_from_e1(
    base_samples: FloatArray,
    mean_directions: FloatArray,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Rotate samples centred around e1 so that e1 maps to each requested mean."""
    samples = _as_float_array(base_samples, name="base_samples")
    means = l2_normalize_rows(mean_directions)
    if samples.shape != means.shape:
        raise ValueError("base_samples and mean_directions must have identical shapes.")

    e1 = np.zeros(samples.shape[1], dtype=np.float64)
    e1[0] = 1.0
    reflectors = e1[np.newaxis, :] - means
    reflector_norms = np.linalg.norm(reflectors, axis=1, keepdims=True)
    unchanged = reflector_norms[:, 0] <= eps
    safe_norms = np.where(reflector_norms <= eps, 1.0, reflector_norms)
    reflectors = reflectors / safe_norms
    rotated = samples - 2.0 * np.sum(samples * reflectors, axis=1, keepdims=True) * reflectors
    rotated[unchanged] = samples[unchanged]
    return l2_normalize_rows(rotated)


def sample_vmf(
    mean_directions: ArrayLike,
    kappas: ArrayLike | float,
    rng: np.random.Generator,
    *,
    max_rejection_rounds: int = 10_000,
) -> FloatArray:
    """Sample row-wise von Mises-Fisher observations with varying means/kappas.

    The implementation uses the rejection sampler described by Wood (1994) for
    dimensions >= 3 and a vectorized Householder rotation. Each returned row is
    a unit vector. ``kappas`` may be a scalar or one value per row. ``kappa=0``
    yields a uniform direction.
    """
    means = l2_normalize_rows(mean_directions)
    sample_count, dimension = means.shape
    if dimension < 3:
        raise ValueError("sample_vmf currently supports dimensions >= 3.")

    kappa_array = np.asarray(kappas, dtype=np.float64)
    if kappa_array.ndim == 0:
        kappa_array = np.full(sample_count, float(kappa_array), dtype=np.float64)
    if kappa_array.shape != (sample_count,):
        raise ValueError("kappas must be a scalar or one value per mean direction.")
    if not np.all(np.isfinite(kappa_array)):
        raise ValueError("kappas contains non-finite values.")
    if np.any(kappa_array < 0):
        raise ValueError("kappas cannot contain negative values.")

    result = np.empty((sample_count, dimension), dtype=np.float64)
    uniform_mask = kappa_array <= 1e-12
    concentrated_mask = ~uniform_mask
    if np.any(uniform_mask):
        result[uniform_mask] = sample_uniform_directions(int(uniform_mask.sum()), dimension, rng)
    if not np.any(concentrated_mask):
        return result

    active_means = means[concentrated_mask]
    active_kappas = kappa_array[concentrated_mask]
    active_count = active_means.shape[0]
    m = float(dimension - 1)
    b = m / (2.0 * active_kappas + np.sqrt(4.0 * active_kappas**2 + m**2))
    x0 = (1.0 - b) / (1.0 + b)
    c = active_kappas * x0 + m * np.log1p(-(x0**2))

    w = np.empty(active_count, dtype=np.float64)
    pending = np.ones(active_count, dtype=bool)
    alpha = 0.5 * m
    for _ in range(max_rejection_rounds):
        pending_indices = np.flatnonzero(pending)
        if pending_indices.size == 0:
            break
        z = rng.beta(alpha, alpha, size=pending_indices.size)
        b_pending = b[pending_indices]
        x0_pending = x0[pending_indices]
        candidates = (1.0 - (1.0 + b_pending) * z) / (1.0 - (1.0 - b_pending) * z)
        log_acceptance = (
            active_kappas[pending_indices] * candidates
            + m * np.log1p(-x0_pending * candidates)
            - c[pending_indices]
        )
        accepted = log_acceptance >= np.log(rng.uniform(size=pending_indices.size))
        accepted_indices = pending_indices[accepted]
        w[accepted_indices] = candidates[accepted]
        pending[accepted_indices] = False
    else:  # pragma: no cover
        raise RuntimeError(
            "The vMF rejection sampler exceeded max_rejection_rounds. "
            "Inspect the kappa range or increase the limit."
        )

    tangent = l2_normalize_rows(rng.normal(size=(active_count, dimension - 1)))
    radial_factor = np.sqrt(np.maximum(0.0, 1.0 - w**2))
    base_samples = np.column_stack((w, radial_factor[:, np.newaxis] * tangent))
    result[concentrated_mask] = _householder_rotate_from_e1(base_samples, active_means)
    return result


def _as_lower_bounds(lower_bounds: ArrayLike | float, sample_count: int) -> FloatArray:
    lb = np.asarray(lower_bounds, dtype=np.float64)
    if lb.ndim == 0:
        lb = np.full(sample_count, float(lb), dtype=np.float64)
    if lb.shape != (sample_count,):
        raise ValueError("lower_bounds must be a scalar or one value per mean direction.")
    if not np.all(np.isfinite(lb)):
        raise ValueError("lower_bounds contains non-finite values.")
    if np.any(lb < -1.0) or np.any(lb > 1.0):
        raise ValueError("lower_bounds must lie in [-1, 1].")
    return lb


def _deterministic_orthogonal(means: FloatArray) -> FloatArray:
    """A deterministic unit vector orthogonal to each row of ``means``.

    Uses the least-aligned canonical axis per row (argmin |mean_i|), which is well
    conditioned for the Gram-Schmidt step.  Fallback for the (vanishingly rare) case
    where a random tangent draw is numerically parallel to the mean.
    """
    row_count, dimension = means.shape
    axis = np.argmin(np.abs(means), axis=1)
    canonical = np.zeros((row_count, dimension), dtype=np.float64)
    canonical[np.arange(row_count), axis] = 1.0
    ortho = canonical - np.sum(canonical * means, axis=1, keepdims=True) * means
    return l2_normalize_rows(ortho)


def _sample_tangent_directions(
    means: FloatArray,
    rng: np.random.Generator,
    *,
    eps: float = 1e-12,
    max_resample: int = 8,
) -> FloatArray:
    """Unit vectors uniform on the tangent hypersphere at each row of ``means``."""
    row_count, dimension = means.shape
    noise = rng.normal(size=(row_count, dimension))
    proj = noise - np.sum(noise * means, axis=1, keepdims=True) * means
    norms = np.linalg.norm(proj, axis=1)
    for _ in range(max_resample):
        degenerate = norms <= eps
        if not np.any(degenerate):
            break
        replacement = rng.normal(size=(int(degenerate.sum()), dimension))
        sub_means = means[degenerate]
        replacement -= np.sum(replacement * sub_means, axis=1, keepdims=True) * sub_means
        proj[degenerate] = replacement
        norms[degenerate] = np.linalg.norm(replacement, axis=1)
    degenerate = norms <= eps
    if np.any(degenerate):
        proj[degenerate] = _deterministic_orthogonal(means[degenerate])
        norms[degenerate] = 1.0
    return proj / np.maximum(norms, eps)[:, np.newaxis]


def sample_spherical_cap(
    mean_directions: ArrayLike,
    lower_bounds: ArrayLike | float,
    rng: np.random.Generator,
    *,
    eps: float = 1e-12,
) -> FloatArray:
    """Sample a cosine-uniform spherical cap around each mean direction.

    For each row: draw a target cosine ``s ~ U[lb, 1]`` and a tangent direction ``u``
    orthogonal to the mean ``mu``, then build ``v = s*mu + sqrt(1-s^2)*u`` (IDperturb,
    Boutros et al. 2026, Eq. 5). Guarantees ``||v|| = 1`` and ``<v, mu> = s >= lb``.

    NOTE: ``s`` is uniform *in cosine*, which is NOT the area-uniform distribution on
    the cap in high dimension. ``lb = 1`` yields ``v = mu`` (no perturbation).
    """
    means = l2_normalize_rows(mean_directions)
    sample_count, _ = means.shape
    lb = _as_lower_bounds(lower_bounds, sample_count)
    cosines = rng.uniform(lb, 1.0)
    radial = np.sqrt(np.maximum(0.0, 1.0 - cosines**2))
    tangent = _sample_tangent_directions(means, rng, eps=eps)
    # Where radial ~ 0 (s ~ 1) the tangent coefficient vanishes -> v collapses to mu.
    perturbed = cosines[:, np.newaxis] * means + radial[:, np.newaxis] * tangent
    return l2_normalize_rows(perturbed)


def sample_hybrid_vmf_cap(
    mean_directions: ArrayLike,
    kappas: ArrayLike | float,
    lower_bounds: ArrayLike | float,
    rng: np.random.Generator,
    *,
    max_rounds: int = 8,
) -> tuple[FloatArray, IntArray, NDArray[np.bool_]]:
    """vMF body with a hard cap floor at ``lb``.

    Sample vMF; rows whose cosine to the centre falls below ``lb`` are redrawn from vMF
    up to ``max_rounds`` times; rows still violating fall back to :func:`sample_spherical_cap`
    at ``lb`` (guaranteeing the floor). Returns the samples plus, per row, the number of
    vMF redraws and whether the cap fallback was used (diagnostics: high fallback rate
    means the result is dominated by the cap rather than the vMF body).
    """
    means = l2_normalize_rows(mean_directions)
    sample_count, _ = means.shape
    lb = _as_lower_bounds(lower_bounds, sample_count)
    kappa_array = np.asarray(kappas, dtype=np.float64)
    if kappa_array.ndim == 0:
        kappa_array = np.full(sample_count, float(kappa_array), dtype=np.float64)

    samples = sample_vmf(means, kappa_array, rng)
    cosines = np.sum(samples * means, axis=1)
    resample_count = np.zeros(sample_count, dtype=np.int64)
    used_cap_fallback = np.zeros(sample_count, dtype=bool)

    for _ in range(max_rounds):
        violating = np.flatnonzero(cosines < lb)
        if violating.size == 0:
            break
        samples[violating] = sample_vmf(means[violating], kappa_array[violating], rng)
        cosines[violating] = np.sum(samples[violating] * means[violating], axis=1)
        resample_count[violating] += 1

    violating = np.flatnonzero(cosines < lb)
    if violating.size > 0:
        samples[violating] = sample_spherical_cap(means[violating], lb[violating], rng)
        used_cap_fallback[violating] = True
    return samples, resample_count, used_cap_fallback


# ---------------------------------------------------------------------------
# Configuration objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RadialReference:
    """Reference location and scale for standardized log norms.

    ``fit`` estimates a robust reference using the median and scaled MAD by
    default.  The simulator stores the standardized value under the explicit
    name ``reference_scaled_log_norm_z`` and retains ``robust_norm_z`` as a
    backward-compatible alias.
    """

    log_norm_location: float
    log_norm_scale: float
    method: str = "median_scaled_mad"

    @classmethod
    def fit(cls, raw_embeddings_or_norms: ArrayLike, *, eps: float = 1e-12) -> "RadialReference":
        values = _as_float_array(raw_embeddings_or_norms, name="raw_embeddings_or_norms")
        if values.ndim == 2:
            norms = np.linalg.norm(values, axis=1)
        elif values.ndim == 1:
            norms = values
        else:
            raise ValueError("Expected a one-dimensional norm vector or a 2D embedding matrix.")
        if np.any(norms <= 0):
            raise ValueError("All norms must be positive.")
        log_norms = np.log(norms)
        location = float(np.median(log_norms))
        mad = float(np.median(np.abs(log_norms - location)))
        scale = 1.4826 * mad
        if scale <= eps:
            raise ValueError("Cannot fit a robust radial reference: scaled MAD is zero.")
        return cls(location, scale)


@dataclass(frozen=True)
class AcquisitionDomainConfig:
    """Acquisition-domain parameters used to create controlled domain shifts."""

    name: str
    weight: float = 1.0
    quality_beta_alpha: float = 4.0
    quality_beta_beta: float = 2.0

    central_regime_probability: float = 0.86
    low_tail_probability: float = 0.08
    high_tail_probability: float = 0.06
    low_tail_quality_logit_slope: float = 2.0
    high_tail_quality_logit_slope: float = 0.0

    log_norm_shift: float = 0.0
    log_norm_sigma_multiplier: float = 1.0
    log_kappa_shift: float = 0.0

    anisotropic_noise_multiplier: float = 1.0
    domain_bias_scale: float = 0.0
    domain_identity_bias_scale: float = 0.0
    session_bias_scale: float = 0.0

    # Additive per-domain shift of the cap lower bound `lb` (cosine-cap intra-class
    # modes). Any sign: positive tightens identity (controlled), negative loosens it
    # (surveillance). Mirrors `log_kappa_shift` for the cap parametrization.
    cap_lower_bound_offset: float = 0.0

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Each acquisition domain must have a non-empty name.")
        if self.weight <= 0:
            raise ValueError(f"Domain {self.name!r}: weight must be positive.")
        if self.quality_beta_alpha <= 0 or self.quality_beta_beta <= 0:
            raise ValueError(f"Domain {self.name!r}: beta parameters must be positive.")
        if self.log_norm_sigma_multiplier <= 0:
            raise ValueError(f"Domain {self.name!r}: log_norm_sigma_multiplier must be positive.")
        if self.anisotropic_noise_multiplier < 0:
            raise ValueError(f"Domain {self.name!r}: anisotropic_noise_multiplier cannot be negative.")
        for field_name in ["domain_bias_scale", "domain_identity_bias_scale", "session_bias_scale"]:
            if getattr(self, field_name) < 0:
                raise ValueError(f"Domain {self.name!r}: {field_name} cannot be negative.")
        _validate_probability_vector(
            [self.central_regime_probability, self.low_tail_probability, self.high_tail_probability],
            name=f"radial regime probabilities for domain {self.name!r}",
        )


@dataclass(frozen=True)
class PopulationConfig:
    """Distribution of synthetic identity centres on the unit hypersphere."""

    component_count: int = 8
    component_weights: ArrayLike | None = None
    identity_center_kappas: ArrayLike | float = 70.0
    component_means: ArrayLike | None = None
    component_mean_global_kappa: float = 20.0
    global_mean_direction: ArrayLike | None = None

    def validate(self, dimension: int) -> None:
        if self.component_count <= 0:
            raise ValueError("component_count must be positive.")
        if self.component_mean_global_kappa < 0:
            raise ValueError("component_mean_global_kappa cannot be negative.")
        if self.component_weights is not None:
            weights = _validate_probability_vector(self.component_weights, name="population component_weights")
            if weights.shape != (self.component_count,):
                raise ValueError("population component_weights must have one value per component.")
        kappas = np.asarray(self.identity_center_kappas, dtype=np.float64)
        if kappas.ndim > 1:
            raise ValueError("identity_center_kappas must be scalar or one-dimensional.")
        if kappas.ndim == 1 and kappas.shape != (self.component_count,):
            raise ValueError("identity_center_kappas must be scalar or have one value per component.")
        if np.any(~np.isfinite(kappas)) or np.any(kappas < 0):
            raise ValueError("identity_center_kappas must contain finite non-negative values.")
        if self.component_means is not None:
            means = l2_normalize_rows(self.component_means)
            if means.shape != (self.component_count, dimension):
                raise ValueError("component_means must have shape (component_count, dimension).")
        if self.global_mean_direction is not None:
            global_mean = l2_normalize_rows(self.global_mean_direction)
            if global_mean.shape != (1, dimension):
                raise ValueError("global_mean_direction must contain exactly dimension values.")


@dataclass(frozen=True)
class RadialConfig:
    """Non-monotonic norm model with central, low-tail and high-tail regimes."""

    reference_log_norm_mean: float = float(np.log(25.0))
    reference_log_norm_scale: float = 0.24
    reference_method: str = "configured_reference"
    quality_log_norm_slope: float = 0.12
    central_offset: float = 0.0
    central_sigma: float = 0.10
    low_tail_offset: float = -0.70
    low_tail_sigma: float = 0.16
    high_tail_offset: float = 0.70
    high_tail_sigma: float = 0.16
    minimum_norm: float = 1e-6
    maximum_norm: float = 1e6

    @classmethod
    def from_reference(cls, reference: RadialReference, **kwargs: Any) -> "RadialConfig":
        return cls(
            reference_log_norm_mean=reference.log_norm_location,
            reference_log_norm_scale=reference.log_norm_scale,
            reference_method=reference.method,
            **kwargs,
        )

    def validate(self) -> None:
        if self.reference_log_norm_scale <= 0:
            raise ValueError("reference_log_norm_scale must be positive.")
        if min(self.central_sigma, self.low_tail_sigma, self.high_tail_sigma) <= 0:
            raise ValueError("All log-norm sigmas must be positive.")
        if self.minimum_norm <= 0 or self.maximum_norm <= self.minimum_norm:
            raise ValueError("Norm limits must satisfy 0 < minimum_norm < maximum_norm.")


@dataclass(frozen=True)
class CouplingConfig:
    """Switches used for causal ablations of quality/radius/direction coupling."""

    quality_affects_regime_probabilities: bool = True
    quality_affects_log_norm: bool = True
    quality_affects_intra_kappa: bool = True
    norm_z_affects_intra_kappa: bool = True


@dataclass(frozen=True)
class DirectionalConfig:
    """Angular reliability model and optional tangent-space nuisance model."""

    peak_intra_kappa: float = 1_400.0
    quality_log_kappa_slope: float = 0.90
    optimal_reference_scaled_log_norm_z: float = 0.0
    low_tail_quadratic_penalty: float = 0.55
    high_tail_quadratic_penalty: float = 0.75
    minimum_intra_kappa: float = 20.0
    maximum_intra_kappa: float = 10_000.0

    anisotropic_rank: int = 0
    anisotropic_base_scale: float = 0.0
    anisotropic_quality_slope: float = 1.0
    nuisance_structure_mode: NuisanceStructureMode | str = NuisanceStructureMode.GLOBAL_SHARED
    global_nuisance_weight: float = 1.0
    domain_nuisance_weight: float = 0.0
    identity_specific_nuisance_rank: int = 0
    identity_specific_nuisance_scale: float = 0.0
    tangent_update_mode: TangentUpdateMode | str = TangentUpdateMode.RETRACTION

    # Intra-class dispersion mode and cosine-cap (IDperturb) parameters. The cap lower
    # bound `lb` is modulated by the SAME covariates as kappa (quality, radial position,
    # domain), reusing the CouplingConfig switches `quality_affects_intra_kappa` and
    # `norm_z_affects_intra_kappa` (in cap modes they gate intra-class dispersion, despite
    # the historical kappa-centric name). Penalties act on cosine scale, NOT on log-kappa.
    intra_class_sampling_mode: IntraClassSamplingMode | str = IntraClassSamplingMode.VMF
    cap_base_lower_bound: float = 0.6
    cap_quality_lower_bound_slope: float = 0.10  # any sign; higher quality -> higher lb
    cap_low_tail_penalty: float = 0.0
    cap_high_tail_penalty: float = 0.0
    cap_minimum_lower_bound: float = 0.0
    cap_maximum_lower_bound: float = 0.999
    cap_overlap_guard: bool = True
    hybrid_max_resample_rounds: int = 8

    @property
    def optimal_robust_norm_z(self) -> float:
        """Backward-compatible alias for version 0.1 documentation."""
        return self.optimal_reference_scaled_log_norm_z

    def validate(self, dimension: int) -> None:
        if self.peak_intra_kappa <= 0 or self.minimum_intra_kappa <= 0:
            raise ValueError("Kappa values must be positive.")
        if self.maximum_intra_kappa < self.minimum_intra_kappa:
            raise ValueError("maximum_intra_kappa must be >= minimum_intra_kappa.")
        if self.low_tail_quadratic_penalty < 0 or self.high_tail_quadratic_penalty < 0:
            raise ValueError("Quadratic tail penalties cannot be negative.")
        if not 0 <= self.anisotropic_rank < dimension:
            raise ValueError("anisotropic_rank must satisfy 0 <= rank < dimension.")
        if not 0 <= self.identity_specific_nuisance_rank < dimension:
            raise ValueError("identity_specific_nuisance_rank must satisfy 0 <= rank < dimension.")
        for name in [
            "anisotropic_base_scale", "anisotropic_quality_slope", "global_nuisance_weight",
            "domain_nuisance_weight", "identity_specific_nuisance_scale",
        ]:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative.")
        try:
            NuisanceStructureMode(_enum_value(self.nuisance_structure_mode))
        except ValueError as exc:
            raise ValueError("Invalid nuisance_structure_mode.") from exc
        try:
            TangentUpdateMode(_enum_value(self.tangent_update_mode))
        except ValueError as exc:
            raise ValueError("Invalid tangent_update_mode.") from exc
        try:
            IntraClassSamplingMode(_enum_value(self.intra_class_sampling_mode))
        except ValueError as exc:
            raise ValueError("Invalid intra_class_sampling_mode.") from exc
        for name in ["cap_base_lower_bound", "cap_minimum_lower_bound", "cap_maximum_lower_bound"]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
        if self.cap_minimum_lower_bound > self.cap_maximum_lower_bound:
            raise ValueError("cap_minimum_lower_bound must be <= cap_maximum_lower_bound.")
        if self.cap_low_tail_penalty < 0 or self.cap_high_tail_penalty < 0:
            raise ValueError("Cap tail penalties cannot be negative.")
        if self.hybrid_max_resample_rounds < 1:
            raise ValueError("hybrid_max_resample_rounds must be >= 1.")


@dataclass(frozen=True)
class SimulatorConfig:
    """Top-level configuration for the synthetic embedding generator."""

    dimension: int = 512
    profile_name: str = "custom"
    population: PopulationConfig = field(default_factory=PopulationConfig)
    radial: RadialConfig = field(default_factory=RadialConfig)
    directional: DirectionalConfig = field(default_factory=DirectionalConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    acquisition_domains: tuple[AcquisitionDomainConfig, ...] = field(
        default_factory=lambda: (AcquisitionDomainConfig(name="default"),)
    )

    def validate(self) -> None:
        if self.dimension < 3:
            raise ValueError("dimension must be at least 3.")
        if not self.profile_name.strip():
            raise ValueError("profile_name must be non-empty.")
        if not self.acquisition_domains:
            raise ValueError("At least one acquisition domain is required.")
        names: set[str] = set()
        for domain in self.acquisition_domains:
            domain.validate()
            if domain.name in names:
                raise ValueError(f"Duplicated acquisition-domain name: {domain.name!r}.")
            names.add(domain.name)
        self.population.validate(self.dimension)
        self.radial.validate()
        self.directional.validate(self.dimension)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def config_to_dict(config: SimulatorConfig) -> dict[str, Any]:
    """Convert a simulator configuration to JSON-ready nested dictionaries."""
    return _json_ready(asdict(config))


def config_from_dict(data: Mapping[str, Any]) -> SimulatorConfig:
    """Reconstruct a :class:`SimulatorConfig` from :func:`config_to_dict` output.

    Closes the serialization round-trip: ``config_to_dict`` writes JSON-ready dicts
    (enums as their string values, arrays as lists) and this reads them back. Array
    fields stay as Python lists — every consumer calls ``np.asarray`` internally, so
    generation reproduces. Enum-valued fields stay as their string value, which the
    config DTOs and the simulator both accept. The result is validated before return.

    Missing keys fall back to dataclass defaults (forward-compatible with configs
    written by an older version); unknown keys raise, surfacing version drift early.
    """
    config = SimulatorConfig(
        dimension=int(data["dimension"]),
        profile_name=str(data.get("profile_name", "custom")),
        population=PopulationConfig(**data["population"]),
        radial=RadialConfig(**data["radial"]),
        directional=DirectionalConfig(**data["directional"]),
        coupling=CouplingConfig(**data.get("coupling", {})),
        acquisition_domains=tuple(
            AcquisitionDomainConfig(**domain) for domain in data["acquisition_domains"]
        ),
    )
    config.validate()
    return config


def config_from_manifest(path: str | Path) -> SimulatorConfig:
    """Read the ``config`` block of a manifest written by ``save_manifest``."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if "config" not in manifest:
        raise ValueError(f"Manifest at {path!s} has no 'config' block.")
    return config_from_dict(manifest["config"])


# ---------------------------------------------------------------------------
# Population and dataset objects
# ---------------------------------------------------------------------------


@dataclass
class SyntheticIdentityPopulation:
    """Latent identities that may be observed under one or more conditions."""

    identity_ids: IntArray
    population_components: IntArray
    identity_centers: FloatArray
    identity_specific_nuisance_bases: FloatArray | None
    domain_identity_bias_vectors: Mapping[str, FloatArray]
    sampling_seed: int
    stream_name: str
    # Lazily-computed bisector floor per identity (cosine of half-angle to nearest other
    # centre), cached for reuse across cross-condition observations. NOT serialized: it is
    # never passed to asdict()/save_manifest/save_npz; keep it that way (a large vector).
    overlap_lower_bound_floor: FloatArray | None = field(
        default=None, repr=False, compare=False
    )

    def validate(self) -> None:
        identity_count, dimension = self.identity_centers.shape
        if self.identity_ids.shape != (identity_count,):
            raise ValueError("identity_ids shape mismatch.")
        if self.population_components.shape != (identity_count,):
            raise ValueError("population_components shape mismatch.")
        if not np.allclose(np.linalg.norm(self.identity_centers, axis=1), 1.0, atol=1e-8):
            raise ValueError("identity_centers contains non-unit vectors.")
        if self.identity_specific_nuisance_bases is not None:
            if self.identity_specific_nuisance_bases.shape[:2] != (identity_count, dimension):
                raise ValueError("identity_specific_nuisance_bases shape mismatch.")
        for name, vectors in self.domain_identity_bias_vectors.items():
            if np.asarray(vectors).shape != (identity_count, dimension):
                raise ValueError(f"Domain identity bias shape mismatch for {name!r}.")


@dataclass
class SyntheticEmbeddingDataset:
    """Embeddings and metadata produced by :class:`AdaFaceEmbeddingSimulator`."""

    raw_embeddings: FloatArray
    normalized_embeddings: FloatArray
    identity_centers: FloatArray
    metadata: pd.DataFrame
    population_component_means: FloatArray
    anisotropic_basis: FloatArray | None
    domain_anisotropic_bases: Mapping[str, FloatArray | None]
    domain_bias_vectors: Mapping[str, FloatArray]
    config: SimulatorConfig
    structural_seed: int
    population_sampling_seed: int
    observation_sampling_seed: int
    stream_name: str

    def validate(self) -> None:
        row_count, dimension = self.raw_embeddings.shape
        if self.normalized_embeddings.shape != (row_count, dimension):
            raise ValueError("normalized_embeddings shape mismatch.")
        if len(self.metadata) != row_count:
            raise ValueError("metadata row count mismatch.")
        if not np.allclose(np.linalg.norm(self.normalized_embeddings, axis=1), 1.0, atol=1e-8):
            raise ValueError("normalized_embeddings contains non-unit vectors.")
        observed_norms = np.linalg.norm(self.raw_embeddings, axis=1)
        if not np.allclose(observed_norms, self.metadata["embedding_norm"], rtol=1e-8):
            raise ValueError("Raw embedding norms do not match metadata.")

    def save_npz(self, path: str | Path) -> Path:
        """Save arrays and metadata columns in a compressed NumPy archive."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        domain_names = np.asarray(sorted(self.domain_bias_vectors), dtype=np.str_)
        domain_bias_matrix = np.vstack([self.domain_bias_vectors[name] for name in domain_names])
        np.savez_compressed(
            output_path,
            raw_embeddings=self.raw_embeddings,
            normalized_embeddings=self.normalized_embeddings,
            identity_centers=self.identity_centers,
            population_component_means=self.population_component_means,
            anisotropic_basis=(
                self.anisotropic_basis
                if self.anisotropic_basis is not None
                else np.empty((self.config.dimension, 0), dtype=np.float64)
            ),
            domain_names=domain_names,
            domain_bias_vectors=domain_bias_matrix,
            metadata_records=self.metadata.to_records(index=False),
        )
        return output_path

    def save_metadata_csv(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata.to_csv(output_path, index=False)
        return output_path

    def save_manifest(self, path: str | Path, *, extra: Mapping[str, Any] | None = None) -> Path:
        """Write a JSON manifest with configuration, seeds and source hash."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        source_path = Path(__file__)
        source_hash = sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else None
        manifest = {
            "simulator_version": __version__,
            "profile_name": self.config.profile_name,
            "structural_seed": self.structural_seed,
            "population_sampling_seed": self.population_sampling_seed,
            "observation_sampling_seed": self.observation_sampling_seed,
            "stream_name": self.stream_name,
            "row_count": int(self.raw_embeddings.shape[0]),
            "dimension": int(self.raw_embeddings.shape[1]),
            "identity_count": int(self.metadata["identity_id"].nunique()),
            "source_sha256": source_hash,
            "config": config_to_dict(self.config),
            "extra": _json_ready(dict(extra or {})),
        }
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path

    def save_bundle(self, output_dir: str | Path, stem: str) -> dict[str, Path]:
        """Save NPZ, metadata CSV, diagnostics CSV and JSON manifest."""
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "npz": self.save_npz(directory / f"{stem}_embeddings.npz"),
            "metadata": self.save_metadata_csv(directory / f"{stem}_metadata.csv"),
            "manifest": self.save_manifest(directory / f"{stem}_manifest.json"),
        }
        diagnostics_path = directory / f"{stem}_norm_diagnostics.csv"
        self.norm_bin_diagnostics(bin_count=10).to_csv(diagnostics_path, index=False)
        paths["norm_diagnostics"] = diagnostics_path
        return paths

    def norm_bin_diagnostics(self, *, bin_count: int = 10) -> pd.DataFrame:
        """Summarize the expected inverted-U pattern across norm quantiles.

        If duplicate quantile edges prevent creation of all requested bins, a
        warning is emitted and both requested and actual counts are included.
        """
        if bin_count < 2:
            raise ValueError("bin_count must be at least 2.")
        frame = self.metadata.copy()
        frame["norm_quantile_bin"] = pd.qcut(frame["embedding_norm"], q=bin_count, duplicates="drop")
        actual_bin_count = int(frame["norm_quantile_bin"].cat.categories.size)
        if actual_bin_count < bin_count:
            warnings.warn(
                f"Requested {bin_count} norm bins, but only {actual_bin_count} could be "
                "created because quantile edges were duplicated.",
                UserWarning,
                stacklevel=2,
            )
        result = (
            frame.groupby("norm_quantile_bin", observed=True)
            .agg(
                sample_count=("embedding_norm", "size"),
                norm_mean=("embedding_norm", "mean"),
                reference_scaled_log_norm_z_mean=("reference_scaled_log_norm_z", "mean"),
                latent_quality_mean=("latent_quality", "mean"),
                intra_kappa_mean=("intra_kappa", "mean"),
                own_center_cosine_mean=("own_center_cosine", "mean"),
                own_center_cosine_median=("own_center_cosine", "median"),
                tangent_step_norm_mean=("combined_tangent_step_norm", "mean"),
            )
            .reset_index()
        )
        result.insert(0, "requested_bin_count", bin_count)
        result.insert(1, "actual_bin_count", actual_bin_count)
        return result


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class AdaFaceEmbeddingSimulator:
    """Generate AdaFace-like synthetic embeddings for controlled PLDA studies."""

    _REGIME_NAMES = np.asarray(["central", "low_tail", "high_tail"])

    def __init__(self, config: SimulatorConfig, *, random_seed: int = 42) -> None:
        config.validate()
        self.config = config
        self.structural_seed = int(random_seed)
        self._domains = tuple(config.acquisition_domains)
        self._domain_names = np.asarray([domain.name for domain in self._domains])
        self._domain_weights = _validate_probability_vector(
            [domain.weight for domain in self._domains], name="acquisition-domain weights"
        )
        self._domain_index_by_name = {domain.name: index for index, domain in enumerate(self._domains)}
        structural_rng = self._rng("structure")
        self.population_component_means = self._create_population_component_means(structural_rng)
        self.population_component_weights = self._get_component_weights()
        self.population_identity_kappas = self._get_identity_center_kappas()
        self.anisotropic_basis = self._create_anisotropic_basis(structural_rng)
        self.domain_anisotropic_bases = self._create_domain_anisotropic_bases(structural_rng)
        self.domain_bias_vectors = self._create_domain_bias_vectors(structural_rng)

    def _rng(self, *parts: object, explicit_seed: int | None = None) -> np.random.Generator:
        seed = int(explicit_seed) if explicit_seed is not None else _stable_seed(self.structural_seed, *parts)
        return np.random.default_rng(seed)

    def _resolved_seed(self, *parts: object, explicit_seed: int | None = None) -> int:
        return int(explicit_seed) if explicit_seed is not None else _stable_seed(self.structural_seed, *parts)

    def _get_component_weights(self) -> FloatArray:
        population = self.config.population
        if population.component_weights is None:
            return np.full(population.component_count, 1.0 / population.component_count, dtype=np.float64)
        return _validate_probability_vector(population.component_weights, name="population component_weights")

    def _get_identity_center_kappas(self) -> FloatArray:
        kappas = np.asarray(self.config.population.identity_center_kappas, dtype=np.float64)
        if kappas.ndim == 0:
            return np.full(self.config.population.component_count, float(kappas), dtype=np.float64)
        return kappas.copy()

    def _create_population_component_means(self, rng: np.random.Generator) -> FloatArray:
        population = self.config.population
        dimension = self.config.dimension
        if population.component_means is not None:
            return l2_normalize_rows(population.component_means)
        global_mean = (
            sample_uniform_directions(1, dimension, rng)
            if population.global_mean_direction is None
            else l2_normalize_rows(population.global_mean_direction)
        )
        if population.component_mean_global_kappa <= 1e-12:
            return sample_uniform_directions(population.component_count, dimension, rng)
        repeated = np.repeat(global_mean, population.component_count, axis=0)
        kappas = np.full(population.component_count, population.component_mean_global_kappa)
        return sample_vmf(repeated, kappas, rng)

    def _create_anisotropic_basis(self, rng: np.random.Generator) -> FloatArray | None:
        directional = self.config.directional
        if directional.anisotropic_rank == 0 or directional.anisotropic_base_scale == 0:
            return None
        return _orthonormal_basis(self.config.dimension, directional.anisotropic_rank, rng)

    def _create_domain_anisotropic_bases(self, rng: np.random.Generator) -> dict[str, FloatArray | None]:
        directional = self.config.directional
        result: dict[str, FloatArray | None] = {}
        for domain in self._domains:
            if directional.anisotropic_rank == 0 or directional.anisotropic_base_scale == 0:
                result[domain.name] = None
            else:
                result[domain.name] = _orthonormal_basis(self.config.dimension, directional.anisotropic_rank, rng)
        return result

    def _create_domain_bias_vectors(self, rng: np.random.Generator) -> dict[str, FloatArray]:
        result: dict[str, FloatArray] = {}
        for domain in self._domains:
            if domain.domain_bias_scale == 0:
                result[domain.name] = np.zeros(self.config.dimension, dtype=np.float64)
            else:
                result[domain.name] = sample_uniform_directions(1, self.config.dimension, rng)[0] * domain.domain_bias_scale
        return result

    def _sample_identity_centers(self, identity_count: int, rng: np.random.Generator) -> tuple[IntArray, FloatArray]:
        components = rng.choice(
            self.config.population.component_count,
            size=identity_count,
            p=self.population_component_weights,
        ).astype(np.int64)
        centers = sample_vmf(
            self.population_component_means[components],
            self.population_identity_kappas[components],
            rng,
        )
        return components, centers

    def _sample_identity_specific_bases(self, identity_count: int, rng: np.random.Generator) -> FloatArray | None:
        rank = self.config.directional.identity_specific_nuisance_rank
        if rank == 0 or self.config.directional.identity_specific_nuisance_scale == 0:
            return None
        bases = np.empty((identity_count, self.config.dimension, rank), dtype=np.float64)
        for index in range(identity_count):
            bases[index] = _orthonormal_basis(self.config.dimension, rank, rng)
        return bases

    def sample_identity_population(
        self,
        *,
        identity_count: int,
        identity_offset: int = 0,
        random_seed: int | None = None,
        stream_name: str = "population",
    ) -> SyntheticIdentityPopulation:
        """Sample latent identities once so they can be observed in many domains."""
        if identity_count <= 0:
            raise ValueError("identity_count must be positive.")
        seed = self._resolved_seed("population", stream_name, identity_offset, identity_count, explicit_seed=random_seed)
        rng = np.random.default_rng(seed)
        components, centers = self._sample_identity_centers(identity_count, rng)
        identity_ids = np.arange(identity_count, dtype=np.int64) + int(identity_offset)
        identity_bases = self._sample_identity_specific_bases(identity_count, rng)
        domain_identity_biases: dict[str, FloatArray] = {}
        for domain in self._domains:
            if domain.domain_identity_bias_scale == 0:
                domain_identity_biases[domain.name] = np.zeros((identity_count, self.config.dimension), dtype=np.float64)
            else:
                domain_identity_biases[domain.name] = (
                    sample_uniform_directions(identity_count, self.config.dimension, rng)
                    * domain.domain_identity_bias_scale
                )
        population = SyntheticIdentityPopulation(
            identity_ids=identity_ids,
            population_components=components,
            identity_centers=centers,
            identity_specific_nuisance_bases=identity_bases,
            domain_identity_bias_vectors=domain_identity_biases,
            sampling_seed=seed,
            stream_name=stream_name,
        )
        population.validate()
        return population

    @staticmethod
    def _expand_samples_per_identity(
        identity_count: int,
        samples_per_identity: int | tuple[int, int],
        rng: np.random.Generator,
    ) -> IntArray:
        if isinstance(samples_per_identity, int):
            if samples_per_identity <= 0:
                raise ValueError("samples_per_identity must be positive.")
            return np.full(identity_count, samples_per_identity, dtype=np.int64)
        if len(samples_per_identity) != 2:
            raise ValueError("samples_per_identity range must be a tuple (minimum, maximum).")
        minimum, maximum = samples_per_identity
        if minimum <= 0 or maximum < minimum:
            raise ValueError("samples_per_identity range must satisfy 0 < minimum <= maximum.")
        return rng.integers(minimum, maximum + 1, size=identity_count, dtype=np.int64)

    @staticmethod
    def _expand_session_indices(
        counts: IntArray,
        sessions_per_identity: int | tuple[int, int],
        rng: np.random.Generator,
    ) -> IntArray:
        identity_count = counts.size
        session_counts = AdaFaceEmbeddingSimulator._expand_samples_per_identity(identity_count, sessions_per_identity, rng)
        output: list[IntArray] = []
        for sample_count, session_count in zip(counts, session_counts, strict=True):
            output.append(rng.integers(0, max(1, int(session_count)), size=int(sample_count), dtype=np.int64))
        return np.concatenate(output)

    def _sample_domains(
        self,
        sample_count: int,
        rng: np.random.Generator,
        *,
        forced_domain: str | None,
    ) -> tuple[IntArray, StrArray]:
        if forced_domain is not None:
            if forced_domain not in self._domain_index_by_name:
                raise ValueError(f"Unknown acquisition domain: {forced_domain!r}.")
            indices = np.full(sample_count, self._domain_index_by_name[forced_domain], dtype=np.int64)
        else:
            indices = rng.choice(len(self._domains), size=sample_count, p=self._domain_weights).astype(np.int64)
        return indices, self._domain_names[indices]

    def _sample_latent_quality(self, domain_indices: IntArray, rng: np.random.Generator) -> FloatArray:
        qualities = np.empty(domain_indices.size, dtype=np.float64)
        for domain_index, domain in enumerate(self._domains):
            mask = domain_indices == domain_index
            if np.any(mask):
                qualities[mask] = rng.beta(domain.quality_beta_alpha, domain.quality_beta_beta, size=int(mask.sum()))
        return qualities

    def _sample_radial_regimes(
        self,
        domain_indices: IntArray,
        latent_quality: FloatArray,
        rng: np.random.Generator,
    ) -> tuple[IntArray, StrArray]:
        regime_indices = np.empty(domain_indices.size, dtype=np.int64)
        coupling = self.config.coupling
        for domain_index, domain in enumerate(self._domains):
            mask = domain_indices == domain_index
            if not np.any(mask):
                continue
            quality = latent_quality[mask]
            base = _validate_probability_vector(
                [domain.central_regime_probability, domain.low_tail_probability, domain.high_tail_probability],
                name=f"radial regime probabilities for domain {domain.name!r}",
            )
            logits = np.repeat(np.log(np.maximum(base, 1e-300))[np.newaxis, :], quality.size, axis=0)
            if coupling.quality_affects_regime_probabilities:
                logits[:, 1] += domain.low_tail_quality_logit_slope * (0.5 - quality)
                logits[:, 2] += domain.high_tail_quality_logit_slope * (0.5 - quality)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            uniforms = rng.uniform(size=quality.size)
            cumulative = np.cumsum(probabilities, axis=1)
            cumulative[:, -1] = 1.0
            regime_indices[mask] = (uniforms[:, np.newaxis] > cumulative).sum(axis=1)
        return regime_indices, self._REGIME_NAMES[regime_indices]

    def _sample_log_norms(
        self,
        domain_indices: IntArray,
        regime_indices: IntArray,
        latent_quality: FloatArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        radial = self.config.radial
        offsets = np.asarray([radial.central_offset, radial.low_tail_offset, radial.high_tail_offset])
        sigmas = np.asarray([radial.central_sigma, radial.low_tail_sigma, radial.high_tail_sigma])
        shifts = np.asarray([domain.log_norm_shift for domain in self._domains])
        sigma_multipliers = np.asarray([domain.log_norm_sigma_multiplier for domain in self._domains])
        quality_effect = radial.quality_log_norm_slope * (latent_quality - 0.5) if self.config.coupling.quality_affects_log_norm else 0.0
        means = radial.reference_log_norm_mean + quality_effect + offsets[regime_indices] + shifts[domain_indices]
        scales = sigmas[regime_indices] * sigma_multipliers[domain_indices]
        return rng.normal(loc=means, scale=scales)

    def _calculate_intra_kappas(
        self,
        domain_indices: IntArray,
        latent_quality: FloatArray,
        reference_scaled_log_norm_z: FloatArray,
    ) -> FloatArray:
        directional = self.config.directional
        coupling = self.config.coupling
        low_distance = np.maximum(0.0, directional.optimal_reference_scaled_log_norm_z - reference_scaled_log_norm_z)
        high_distance = np.maximum(0.0, reference_scaled_log_norm_z - directional.optimal_reference_scaled_log_norm_z)
        quality_effect = directional.quality_log_kappa_slope * (latent_quality - 0.5) if coupling.quality_affects_intra_kappa else 0.0
        radial_effect = (
            - directional.low_tail_quadratic_penalty * low_distance**2
            - directional.high_tail_quadratic_penalty * high_distance**2
            if coupling.norm_z_affects_intra_kappa
            else 0.0
        )
        shifts = np.asarray([domain.log_kappa_shift for domain in self._domains])
        log_kappas = np.log(directional.peak_intra_kappa) + quality_effect + shifts[domain_indices] + radial_effect
        return np.clip(np.exp(log_kappas), directional.minimum_intra_kappa, directional.maximum_intra_kappa)

    @staticmethod
    def _overlap_lower_bound_floor(centers: FloatArray, *, chunk_size: int = 2048) -> FloatArray:
        """Per-identity cosine floor from the bisector rule (Boutros et al. 2026, Eq. 6).

        ``floor_i = sqrt((1 + max_{j!=i} c_i.c_j) / 2) = cos(angle_to_nearest / 2)`` — keeps a
        perturbation closer to its own centre than to the nearest other identity. Computed in
        row blocks so the full C x C cosine matrix is never materialized (O(C^2 d) time,
        O(chunk*C) memory). Singletons get floor 0.
        """
        count = centers.shape[0]
        if count < 2:
            return np.zeros(count, dtype=np.float64)
        nearest = np.full(count, -np.inf, dtype=np.float64)
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            block = centers[start:stop] @ centers.T
            block[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # exclude self
            nearest[start:stop] = block.max(axis=1)
        return np.sqrt(np.maximum(0.0, (1.0 + nearest) / 2.0))

    def _calculate_cap_lower_bounds(
        self,
        domain_indices: IntArray,
        latent_quality: FloatArray,
        reference_scaled_log_norm_z: FloatArray,
        repeated_local_ids: IntArray,
        population: "SyntheticIdentityPopulation",
    ) -> FloatArray:
        """Per-sample cap lower bound `lb`, mirroring `_calculate_intra_kappas`.

        Modulation (cosine scale): base + per-domain offset + quality term + non-monotonic
        norm penalties, gated by the CouplingConfig switches. Then clip to the configured
        range; the overlap guard is applied AFTER the clip and may push the effective `lb`
        above ``cap_maximum_lower_bound`` (anti-collision has priority). Physical cap at 1.
        """
        directional = self.config.directional
        coupling = self.config.coupling
        low_distance = np.maximum(0.0, directional.optimal_reference_scaled_log_norm_z - reference_scaled_log_norm_z)
        high_distance = np.maximum(0.0, reference_scaled_log_norm_z - directional.optimal_reference_scaled_log_norm_z)
        quality_effect = (
            directional.cap_quality_lower_bound_slope * (latent_quality - 0.5)
            if coupling.quality_affects_intra_kappa
            else 0.0
        )
        radial_effect = (
            - directional.cap_low_tail_penalty * low_distance**2
            - directional.cap_high_tail_penalty * high_distance**2
            if coupling.norm_z_affects_intra_kappa
            else 0.0
        )
        domain_offsets = np.asarray([domain.cap_lower_bound_offset for domain in self._domains])
        lb = directional.cap_base_lower_bound + domain_offsets[domain_indices] + quality_effect + radial_effect
        lb = np.clip(lb, directional.cap_minimum_lower_bound, directional.cap_maximum_lower_bound)
        if directional.cap_overlap_guard:
            if population.overlap_lower_bound_floor is None:
                population.overlap_lower_bound_floor = self._overlap_lower_bound_floor(population.identity_centers)
            lb = np.maximum(lb, population.overlap_lower_bound_floor[repeated_local_ids])
        return np.minimum(lb, 1.0)

    def _sample_session_bias_matrix(
        self,
        repeated_identity_ids: IntArray,
        domain_indices: IntArray,
        session_indices: IntArray,
        rng: np.random.Generator,
    ) -> FloatArray:
        output = np.zeros((len(repeated_identity_ids), self.config.dimension), dtype=np.float64)
        keys = np.column_stack((repeated_identity_ids, domain_indices, session_indices))
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        for key_index, (_, domain_index, _) in enumerate(unique_keys):
            scale = self._domains[int(domain_index)].session_bias_scale
            if scale > 0:
                vector = sample_uniform_directions(1, self.config.dimension, rng)[0] * scale
                output[inverse == key_index] = vector
        return output

    def _apply_anisotropic_nuisance(
        self,
        directions: FloatArray,
        identity_centers: FloatArray,
        domain_indices: IntArray,
        latent_quality: FloatArray,
        repeated_local_ids: IntArray,
        repeated_identity_ids: IntArray,
        session_indices: IntArray,
        population: SyntheticIdentityPopulation,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, dict[str, FloatArray]]:
        directional = self.config.directional
        sample_count = directions.shape[0]
        multipliers = np.asarray([domain.anisotropic_noise_multiplier for domain in self._domains])
        scales = directional.anisotropic_base_scale * multipliers[domain_indices] * (
            1.0 + directional.anisotropic_quality_slope * (1.0 - latent_quality)
        )
        mode = NuisanceStructureMode(_enum_value(directional.nuisance_structure_mode))
        anisotropic_noise = np.zeros_like(directions)

        if self.anisotropic_basis is not None and mode in {NuisanceStructureMode.GLOBAL_SHARED, NuisanceStructureMode.HIERARCHICAL}:
            coefficients = rng.normal(size=(sample_count, self.anisotropic_basis.shape[1]))
            anisotropic_noise += directional.global_nuisance_weight * (coefficients @ self.anisotropic_basis.T)

        if mode in {NuisanceStructureMode.PER_DOMAIN, NuisanceStructureMode.HIERARCHICAL}:
            for domain_index, domain in enumerate(self._domains):
                mask = domain_indices == domain_index
                basis = self.domain_anisotropic_bases[domain.name]
                if basis is not None and np.any(mask):
                    coefficients = rng.normal(size=(int(mask.sum()), basis.shape[1]))
                    weight = directional.domain_nuisance_weight if mode is NuisanceStructureMode.HIERARCHICAL else 1.0
                    anisotropic_noise[mask] += weight * (coefficients @ basis.T)

        identity_bases = population.identity_specific_nuisance_bases
        if identity_bases is not None and directional.identity_specific_nuisance_scale > 0:
            rank = identity_bases.shape[2]
            coefficients = rng.normal(size=(sample_count, rank))
            selected = identity_bases[repeated_local_ids]
            identity_noise = np.einsum("ndr,nr->nd", selected, coefficients)
            anisotropic_noise += directional.identity_specific_nuisance_scale * identity_noise

        anisotropic_noise *= scales[:, np.newaxis]
        fixed_domain_bias = np.vstack([self.domain_bias_vectors[self._domains[index].name] for index in domain_indices])
        domain_identity_bias = np.vstack(
            [population.domain_identity_bias_vectors[self._domains[index].name][local_id] for index, local_id in zip(domain_indices, repeated_local_ids, strict=True)]
        )
        session_bias = self._sample_session_bias_matrix(repeated_identity_ids, domain_indices, session_indices, rng)
        raw_step = anisotropic_noise + fixed_domain_bias + domain_identity_bias + session_bias

        update_mode = TangentUpdateMode(_enum_value(directional.tangent_update_mode))
        tangent_reference = identity_centers if update_mode is TangentUpdateMode.RETRACTION else directions
        tangent_step = raw_step - np.sum(raw_step * tangent_reference, axis=1, keepdims=True) * tangent_reference
        step_norms = np.linalg.norm(tangent_step, axis=1)

        if update_mode is TangentUpdateMode.RETRACTION:
            updated = l2_normalize_rows(directions + tangent_step)
        else:
            safe = np.where(step_norms <= 1e-12, 1.0, step_norms)
            updated = (
                np.cos(step_norms)[:, np.newaxis] * directions
                + (np.sin(step_norms) / safe)[:, np.newaxis] * tangent_step
            )
            updated[step_norms <= 1e-12] = directions[step_norms <= 1e-12]
            updated = l2_normalize_rows(updated)

        diagnostics = {
            "anisotropic_noise_norm": np.linalg.norm(anisotropic_noise, axis=1),
            "fixed_domain_bias_norm": np.linalg.norm(fixed_domain_bias, axis=1),
            "domain_identity_bias_norm": np.linalg.norm(domain_identity_bias, axis=1),
            "session_bias_norm": np.linalg.norm(session_bias, axis=1),
            "combined_tangent_step_norm": step_norms,
        }
        return updated, diagnostics

    def sample_observations(
        self,
        population: SyntheticIdentityPopulation,
        *,
        samples_per_identity: int | tuple[int, int],
        split_name: str = "synthetic",
        forced_domain: str | None = None,
        sessions_per_identity: int | tuple[int, int] = 1,
        random_seed: int | None = None,
        stream_name: str | None = None,
    ) -> SyntheticEmbeddingDataset:
        """Observe an existing latent population under mixed or forced domains."""
        population.validate()
        if not split_name.strip():
            raise ValueError("split_name must be non-empty.")
        stream = stream_name or split_name
        seed = self._resolved_seed("observations", stream, forced_domain or "mixed", explicit_seed=random_seed)
        rng = np.random.default_rng(seed)
        identity_count = population.identity_ids.size
        counts = self._expand_samples_per_identity(identity_count, samples_per_identity, rng)
        repeated_local_ids = np.repeat(np.arange(identity_count, dtype=np.int64), counts)
        repeated_identity_ids = population.identity_ids[repeated_local_ids]
        repeated_components = population.population_components[repeated_local_ids]
        repeated_centers = population.identity_centers[repeated_local_ids]
        sample_indices = np.concatenate([np.arange(count, dtype=np.int64) for count in counts])
        session_indices = self._expand_session_indices(counts, sessions_per_identity, rng)

        domain_indices, domain_names = self._sample_domains(len(repeated_identity_ids), rng, forced_domain=forced_domain)
        latent_quality = self._sample_latent_quality(domain_indices, rng)
        regime_indices, regime_names = self._sample_radial_regimes(domain_indices, latent_quality, rng)
        log_norms = self._sample_log_norms(domain_indices, regime_indices, latent_quality, rng)
        radial = self.config.radial
        embedding_norms = np.clip(np.exp(log_norms), radial.minimum_norm, radial.maximum_norm)
        log_norms = np.log(embedding_norms)
        reference_scaled_z = (log_norms - radial.reference_log_norm_mean) / radial.reference_log_norm_scale
        intra_kappas = self._calculate_intra_kappas(domain_indices, latent_quality, reference_scaled_z)

        directional = self.config.directional
        sampling_mode = IntraClassSamplingMode(_enum_value(directional.intra_class_sampling_mode))
        sample_total = len(repeated_identity_ids)
        cap_lower_bound_effective = np.full(sample_total, np.nan, dtype=np.float64)
        hybrid_resample_count = np.zeros(sample_total, dtype=np.int64)
        hybrid_used_cap_fallback = np.zeros(sample_total, dtype=bool)

        if sampling_mode is IntraClassSamplingMode.VMF:
            normalized_embeddings = sample_vmf(repeated_centers, intra_kappas, rng)
        else:
            cap_lower_bound_effective = self._calculate_cap_lower_bounds(
                domain_indices, latent_quality, reference_scaled_z, repeated_local_ids, population
            )
            if sampling_mode is IntraClassSamplingMode.IDPERTURB_CAP:
                normalized_embeddings = sample_spherical_cap(repeated_centers, cap_lower_bound_effective, rng)
            else:  # HYBRID_VMF_CAP
                normalized_embeddings, hybrid_resample_count, hybrid_used_cap_fallback = sample_hybrid_vmf_cap(
                    repeated_centers,
                    intra_kappas,
                    cap_lower_bound_effective,
                    rng,
                    max_rounds=directional.hybrid_max_resample_rounds,
                )

        # Floor `lb` is guaranteed on the PRE-nuisance direction; the tangent nuisance step
        # below can push the realized (post-nuisance) own_center_cosine slightly below `lb`.
        pre_nuisance_center_cosines = np.sum(normalized_embeddings * repeated_centers, axis=1)

        normalized_embeddings, nuisance_diagnostics = self._apply_anisotropic_nuisance(
            normalized_embeddings,
            repeated_centers,
            domain_indices,
            latent_quality,
            repeated_local_ids,
            repeated_identity_ids,
            session_indices,
            population,
            rng,
        )
        raw_embeddings = embedding_norms[:, np.newaxis] * normalized_embeddings
        own_center_cosines = np.sum(normalized_embeddings * repeated_centers, axis=1)

        metadata = pd.DataFrame(
            {
                "split": split_name,
                "identity_id": repeated_identity_ids,
                "local_identity_index": repeated_local_ids,
                "sample_index": sample_indices,
                "session_index": session_indices,
                "population_component": repeated_components,
                "acquisition_domain": domain_names,
                "radial_regime": regime_names,
                "latent_quality": latent_quality,
                "embedding_norm": embedding_norms,
                "log_embedding_norm": log_norms,
                "reference_scaled_log_norm_z": reference_scaled_z,
                "robust_norm_z": reference_scaled_z,  # backward-compatible alias
                "radial_reference_method": radial.reference_method,
                "intra_kappa": intra_kappas,
                "own_center_cosine": own_center_cosines,
                "pre_nuisance_center_cosine": pre_nuisance_center_cosines,
                "intra_class_sampling_mode": sampling_mode.value,
                "cap_lower_bound_effective": cap_lower_bound_effective,
                "hybrid_resample_count": hybrid_resample_count,
                "hybrid_used_cap_fallback": hybrid_used_cap_fallback,
                "tangent_update_mode": _enum_value(self.config.directional.tangent_update_mode),
                **nuisance_diagnostics,
            }
        )
        dataset = SyntheticEmbeddingDataset(
            raw_embeddings=raw_embeddings,
            normalized_embeddings=normalized_embeddings,
            identity_centers=population.identity_centers,
            metadata=metadata,
            population_component_means=self.population_component_means,
            anisotropic_basis=self.anisotropic_basis,
            domain_anisotropic_bases=self.domain_anisotropic_bases,
            domain_bias_vectors=self.domain_bias_vectors,
            config=self.config,
            structural_seed=self.structural_seed,
            population_sampling_seed=population.sampling_seed,
            observation_sampling_seed=seed,
            stream_name=stream,
        )
        dataset.validate()
        return dataset

    def sample_dataset(
        self,
        *,
        identity_count: int,
        samples_per_identity: int | tuple[int, int],
        split_name: str = "synthetic",
        identity_offset: int = 0,
        sessions_per_identity: int | tuple[int, int] = 1,
        random_seed: int | None = None,
        population_random_seed: int | None = None,
    ) -> SyntheticEmbeddingDataset:
        """Generate one split with disjoint identities and call-order-independent RNG."""
        population = self.sample_identity_population(
            identity_count=identity_count,
            identity_offset=identity_offset,
            random_seed=population_random_seed,
            stream_name=f"{split_name}:population",
        )
        return self.sample_observations(
            population,
            samples_per_identity=samples_per_identity,
            split_name=split_name,
            sessions_per_identity=sessions_per_identity,
            random_seed=random_seed,
            stream_name=f"{split_name}:observations",
        )

    def sample_cross_condition_datasets(
        self,
        *,
        identity_count: int,
        samples_per_identity_by_domain: Mapping[str, int | tuple[int, int]],
        split_name: str = "cross_condition",
        identity_offset: int = 0,
        sessions_per_identity: int | tuple[int, int] = 1,
        population_random_seed: int | None = None,
        observation_random_seeds: Mapping[str, int] | None = None,
    ) -> dict[str, SyntheticEmbeddingDataset]:
        """Generate aligned datasets that share identities but differ by domain."""
        population = self.sample_identity_population(
            identity_count=identity_count,
            identity_offset=identity_offset,
            random_seed=population_random_seed,
            stream_name=f"{split_name}:population",
        )
        output: dict[str, SyntheticEmbeddingDataset] = {}
        for domain_name, samples_per_identity in samples_per_identity_by_domain.items():
            explicit_seed = None if observation_random_seeds is None else observation_random_seeds.get(domain_name)
            output[domain_name] = self.sample_observations(
                population,
                samples_per_identity=samples_per_identity,
                split_name=f"{split_name}:{domain_name}",
                forced_domain=domain_name,
                sessions_per_identity=sessions_per_identity,
                random_seed=explicit_seed,
                stream_name=f"{split_name}:observations:{domain_name}",
            )
        return output


# ---------------------------------------------------------------------------
# Pair generation and scoring helpers
# ---------------------------------------------------------------------------


def cosine_scores(embeddings: ArrayLike, left_indices: ArrayLike, right_indices: ArrayLike) -> FloatArray:
    """Compute cosine similarity scores for selected embedding pairs."""
    normalized = l2_normalize_rows(embeddings)
    left = np.asarray(left_indices, dtype=np.int64)
    right = np.asarray(right_indices, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("left_indices and right_indices must have identical shapes.")
    return np.sum(normalized[left] * normalized[right], axis=1)


def _group_rows_by_identity(metadata: pd.DataFrame) -> tuple[IntArray, list[IntArray]]:
    grouped = metadata.groupby("identity_id", sort=True).indices
    identities = np.asarray(list(grouped), dtype=np.int64)
    rows = [np.asarray(grouped[int(identity)], dtype=np.int64) for identity in identities]
    return identities, rows


def _sample_rows_from_groups(
    groups: list[IntArray],
    selected_group_positions: IntArray,
    rng: np.random.Generator,
) -> IntArray:
    lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
    offsets = np.concatenate(([0], np.cumsum(lengths[:-1])))
    flat = np.concatenate(groups)
    selected_lengths = lengths[selected_group_positions]
    within = (rng.random(len(selected_group_positions)) * selected_lengths).astype(np.int64)
    return flat[offsets[selected_group_positions] + within]


def _sample_balanced_identity_pairs(
    metadata: pd.DataFrame,
    genuine_pair_count: int,
    impostor_pair_count: int,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray, IntArray]:
    identities, groups = _group_rows_by_identity(metadata)
    eligible_positions = np.asarray([index for index, rows in enumerate(groups) if len(rows) >= 2], dtype=np.int64)
    if genuine_pair_count > 0 and eligible_positions.size == 0:
        raise ValueError("No identity has at least two samples for genuine-pair sampling.")
    if impostor_pair_count > 0 and len(groups) < 2:
        raise ValueError("At least two identities are needed for impostor-pair sampling.")

    if genuine_pair_count:
        selected = rng.choice(eligible_positions, size=genuine_pair_count)
        group_lengths = np.asarray([len(group) for group in groups], dtype=np.int64)
        group_offsets = np.concatenate(([0], np.cumsum(group_lengths[:-1])))
        flat_rows = np.concatenate(groups)
        selected_lengths = group_lengths[selected]
        first_pos = (rng.random(genuine_pair_count) * selected_lengths).astype(np.int64)
        second_pos = (rng.random(genuine_pair_count) * (selected_lengths - 1)).astype(np.int64)
        second_pos += second_pos >= first_pos
        genuine_left = flat_rows[group_offsets[selected] + first_pos]
        genuine_right = flat_rows[group_offsets[selected] + second_pos]
    else:
        genuine_left = np.empty(0, dtype=np.int64)
        genuine_right = np.empty(0, dtype=np.int64)

    if impostor_pair_count:
        left_group = rng.integers(0, len(groups), size=impostor_pair_count, dtype=np.int64)
        right_group = rng.integers(0, len(groups) - 1, size=impostor_pair_count, dtype=np.int64)
        right_group += right_group >= left_group
        impostor_left = _sample_rows_from_groups(groups, left_group, rng)
        impostor_right = _sample_rows_from_groups(groups, right_group, rng)
    else:
        impostor_left = np.empty(0, dtype=np.int64)
        impostor_right = np.empty(0, dtype=np.int64)

    left = np.concatenate([genuine_left, impostor_left])
    right = np.concatenate([genuine_right, impostor_right])
    labels = np.concatenate([np.ones(genuine_pair_count, dtype=np.int64), np.zeros(impostor_pair_count, dtype=np.int64)])
    return left, right, labels


def _sample_balanced_observation_impostors(
    metadata: pd.DataFrame,
    pair_count: int,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    if pair_count == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    identity_ids = metadata["identity_id"].to_numpy()
    if np.unique(identity_ids).size < 2:
        raise ValueError("At least two identities are needed for impostor-pair sampling.")
    row_count = len(metadata)
    left_parts: list[IntArray] = []
    right_parts: list[IntArray] = []
    remaining = pair_count
    while remaining > 0:
        batch = max(remaining * 2, 1024)
        left = rng.integers(0, row_count, size=batch, dtype=np.int64)
        right = rng.integers(0, row_count, size=batch, dtype=np.int64)
        mask = identity_ids[left] != identity_ids[right]
        left_parts.append(left[mask][:remaining])
        right_parts.append(right[mask][:remaining])
        remaining -= min(int(mask.sum()), remaining)
    return np.concatenate(left_parts), np.concatenate(right_parts)


def _deduplicate_pairs(
    left: IntArray,
    right: IntArray,
    labels: IntArray,
) -> tuple[IntArray, IntArray, IntArray]:
    low = np.minimum(left, right)
    high = np.maximum(left, right)
    frame = pd.DataFrame({"left": low, "right": high, "label": labels})
    frame = frame.drop_duplicates(ignore_index=True)
    return frame["left"].to_numpy(np.int64), frame["right"].to_numpy(np.int64), frame["label"].to_numpy(np.int64)


def sample_pair_table(
    dataset: SyntheticEmbeddingDataset,
    *,
    genuine_pair_count: int,
    random_impostor_pair_count: int,
    random_seed: int = 123,
    strategy: PairSamplingStrategy | str = PairSamplingStrategy.BALANCED_IDENTITY,
    unique_pairs: bool = False,
) -> pd.DataFrame:
    """Sample genuine and random-impostor pairs with cosine scores.

    ``BALANCED_IDENTITY`` assigns approximately equal trial weight to each
    identity. ``BALANCED_OBSERVATION`` samples impostors approximately in
    proportion to the number of observed embeddings. Genuine pairs remain
    identity-balanced in both modes.
    """
    if genuine_pair_count < 0 or random_impostor_pair_count < 0:
        raise ValueError("Pair counts cannot be negative.")
    strategy_value = PairSamplingStrategy(_enum_value(strategy))
    rng = np.random.default_rng(random_seed)
    metadata = dataset.metadata.reset_index(drop=True)
    left, right, labels = _sample_balanced_identity_pairs(metadata, genuine_pair_count, random_impostor_pair_count, rng)
    if strategy_value is PairSamplingStrategy.BALANCED_OBSERVATION and random_impostor_pair_count:
        impostor_left, impostor_right = _sample_balanced_observation_impostors(metadata, random_impostor_pair_count, rng)
        left[genuine_pair_count:] = impostor_left
        right[genuine_pair_count:] = impostor_right
    if unique_pairs:
        left, right, labels = _deduplicate_pairs(left, right, labels)
        if len(labels) < genuine_pair_count + random_impostor_pair_count:
            warnings.warn(
                "unique_pairs=True removed duplicated trials; the returned table contains fewer rows than requested.",
                UserWarning,
                stacklevel=2,
            )
    scores = cosine_scores(dataset.normalized_embeddings, left, right)
    result = pd.DataFrame(
        {
            "left_row": left,
            "right_row": right,
            "is_genuine": labels,
            "cosine_score": scores,
            "left_identity_id": metadata.loc[left, "identity_id"].to_numpy(),
            "right_identity_id": metadata.loc[right, "identity_id"].to_numpy(),
            "left_norm": metadata.loc[left, "embedding_norm"].to_numpy(),
            "right_norm": metadata.loc[right, "embedding_norm"].to_numpy(),
            "left_reference_scaled_log_norm_z": metadata.loc[left, "reference_scaled_log_norm_z"].to_numpy(),
            "right_reference_scaled_log_norm_z": metadata.loc[right, "reference_scaled_log_norm_z"].to_numpy(),
            "left_robust_norm_z": metadata.loc[left, "robust_norm_z"].to_numpy(),
            "right_robust_norm_z": metadata.loc[right, "robust_norm_z"].to_numpy(),
            "left_quality": metadata.loc[left, "latent_quality"].to_numpy(),
            "right_quality": metadata.loc[right, "latent_quality"].to_numpy(),
        }
    )
    result.attrs["sampling_strategy"] = strategy_value.value
    result.attrs["unique_pairs"] = bool(unique_pairs)
    return result


# ---------------------------------------------------------------------------
# Ready-to-use profiles
# ---------------------------------------------------------------------------


def make_spherical_baseline_config(*, dimension: int = 512) -> SimulatorConfig:
    """Return an isotropic spherical baseline for cosine/PLDA/PSDA comparisons."""
    return SimulatorConfig(
        dimension=dimension,
        profile_name="SPHERICAL_BASELINE",
        population=PopulationConfig(component_count=8, identity_center_kappas=70.0, component_mean_global_kappa=20.0),
        directional=DirectionalConfig(anisotropic_rank=0, anisotropic_base_scale=0.0),
        acquisition_domains=(AcquisitionDomainConfig(name="default"),),
    )


def make_cooperative_plda_config(*, dimension: int = 512) -> SimulatorConfig:
    """Return the version-0.1-style cooperative PLDA baseline."""
    return SimulatorConfig(
        dimension=dimension,
        profile_name="COOPERATIVE_PLDA_BASELINE",
        population=PopulationConfig(component_count=8, identity_center_kappas=70.0, component_mean_global_kappa=20.0),
        radial=RadialConfig(
            reference_log_norm_mean=float(np.log(25.0)), reference_log_norm_scale=0.24,
            quality_log_norm_slope=0.12, central_sigma=0.10,
            low_tail_offset=-0.70, low_tail_sigma=0.16,
            high_tail_offset=0.70, high_tail_sigma=0.16,
        ),
        directional=DirectionalConfig(
            peak_intra_kappa=1_400.0, quality_log_kappa_slope=0.90,
            low_tail_quadratic_penalty=0.55, high_tail_quadratic_penalty=0.75,
            minimum_intra_kappa=20.0, maximum_intra_kappa=10_000.0,
            anisotropic_rank=min(24, dimension - 1), anisotropic_base_scale=0.010,
            anisotropic_quality_slope=1.2,
            nuisance_structure_mode=NuisanceStructureMode.GLOBAL_SHARED,
            tangent_update_mode=TangentUpdateMode.RETRACTION,
        ),
        acquisition_domains=(
            AcquisitionDomainConfig(
                name="controlled", weight=0.65, quality_beta_alpha=5.0, quality_beta_beta=2.0,
                central_regime_probability=0.91, low_tail_probability=0.05, high_tail_probability=0.04,
                low_tail_quality_logit_slope=2.0, log_kappa_shift=0.05,
                anisotropic_noise_multiplier=0.7,
            ),
            AcquisitionDomainConfig(
                name="surveillance", weight=0.35, quality_beta_alpha=2.0, quality_beta_beta=4.5,
                central_regime_probability=0.72, low_tail_probability=0.17, high_tail_probability=0.11,
                low_tail_quality_logit_slope=2.8, high_tail_quality_logit_slope=0.6,
                log_norm_shift=-0.04, log_norm_sigma_multiplier=1.25,
                log_kappa_shift=-0.30, anisotropic_noise_multiplier=1.7,
                domain_bias_scale=0.008,
            ),
        ),
    )


def make_cross_condition_stress_config(*, dimension: int = 512) -> SimulatorConfig:
    """Return a heterogeneous cross-condition profile for backend stress tests."""
    return SimulatorConfig(
        dimension=dimension,
        profile_name="CROSS_CONDITION_HETEROSCEDASTIC",
        population=PopulationConfig(component_count=8, identity_center_kappas=70.0, component_mean_global_kappa=20.0),
        radial=RadialConfig(
            reference_log_norm_mean=float(np.log(25.0)), reference_log_norm_scale=0.24,
            quality_log_norm_slope=0.12, central_sigma=0.10,
            low_tail_offset=-0.70, low_tail_sigma=0.16,
            high_tail_offset=0.70, high_tail_sigma=0.16,
        ),
        directional=DirectionalConfig(
            peak_intra_kappa=1_400.0, quality_log_kappa_slope=0.90,
            low_tail_quadratic_penalty=0.55, high_tail_quadratic_penalty=0.75,
            minimum_intra_kappa=20.0, maximum_intra_kappa=10_000.0,
            anisotropic_rank=min(20, dimension - 1), anisotropic_base_scale=0.010, anisotropic_quality_slope=1.2,
            nuisance_structure_mode=NuisanceStructureMode.HIERARCHICAL,
            global_nuisance_weight=0.55, domain_nuisance_weight=0.90,
            identity_specific_nuisance_rank=min(2, dimension - 1), identity_specific_nuisance_scale=0.40,
            tangent_update_mode=TangentUpdateMode.EXPONENTIAL_MAP,
        ),
        acquisition_domains=(
            AcquisitionDomainConfig(
                name="controlled", weight=0.5, quality_beta_alpha=5.0, quality_beta_beta=2.0,
                central_regime_probability=0.91, low_tail_probability=0.05, high_tail_probability=0.04,
                low_tail_quality_logit_slope=2.0, log_kappa_shift=0.05,
                anisotropic_noise_multiplier=0.65, domain_identity_bias_scale=0.002,
                session_bias_scale=0.001,
            ),
            AcquisitionDomainConfig(
                name="surveillance", weight=0.5, quality_beta_alpha=2.0, quality_beta_beta=4.5,
                central_regime_probability=0.70, low_tail_probability=0.18, high_tail_probability=0.12,
                low_tail_quality_logit_slope=2.8, high_tail_quality_logit_slope=0.6,
                log_norm_shift=-0.04, log_norm_sigma_multiplier=1.25,
                log_kappa_shift=-0.35, anisotropic_noise_multiplier=1.85,
                domain_bias_scale=0.006, domain_identity_bias_scale=0.012,
                session_bias_scale=0.006,
            ),
        ),
    )


def make_idperturb_cap_config(*, dimension: int = 512, lower_bound: float = 0.6) -> SimulatorConfig:
    """Cooperative PLDA baseline using the cosine-uniform spherical cap (IDperturb).

    ``lower_bound`` (``lb``) is the base cosine floor for within-identity dispersion. The
    value ``0.6`` is INSPIRED by the IDperturb experimental regime (Boutros et al. 2026),
    NOT an optimum for AdaFace/PLDA/FaceAudit — calibrate empirically before drawing
    conclusions. The overlap guard is on, so the *effective* ``lb`` can exceed
    ``lower_bound`` for identities in dense regions of the population.
    """
    base = make_cooperative_plda_config(dimension=dimension)
    directional = replace(
        base.directional,
        intra_class_sampling_mode=IntraClassSamplingMode.IDPERTURB_CAP,
        cap_base_lower_bound=lower_bound,
        cap_quality_lower_bound_slope=0.10,
        cap_low_tail_penalty=0.15,
        cap_high_tail_penalty=0.20,
        cap_overlap_guard=True,
    )
    return replace(base, profile_name="IDPERTURB_CAP", directional=directional)


def make_example_config(*, dimension: int = 512) -> SimulatorConfig:
    """Backward-compatible alias for the cooperative PLDA baseline."""
    return make_cooperative_plda_config(dimension=dimension)
