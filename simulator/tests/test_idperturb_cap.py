"""Tests for the IDperturb cosine-uniform spherical-cap intra-class sampling.

Run from the repo root::

    .venv/Scripts/python.exe -m pytest simulator/tests/test_idperturb_cap.py
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from adaface_embedding_simulator import (
    AcquisitionDomainConfig,
    AdaFaceEmbeddingSimulator,
    CouplingConfig,
    DirectionalConfig,
    IntraClassSamplingMode,
    PopulationConfig,
    SimulatorConfig,
    config_from_dict,
    config_to_dict,
    make_cooperative_plda_config,
    make_idperturb_cap_config,
    sample_spherical_cap,
    sample_uniform_directions,
)
from adaface_embedding_simulator import (
    _deterministic_orthogonal,
    _sample_tangent_directions,
)


# --------------------------------------------------------------------------- cap function


def test_cap_unit_norm_and_preserves_sampled_cosine() -> None:
    rng = np.random.default_rng(0)
    means = sample_uniform_directions(3000, 64, rng)
    v = sample_spherical_cap(means, 0.6, rng)
    cos = np.sum(v * means, axis=1)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10)
    assert cos.min() >= 0.6 - 1e-9                      # >= lb
    assert abs(cos.mean() - (1 + 0.6) / 2) < 0.01       # cosine-uniform: E[s] = (1+lb)/2
    assert cos.max() <= 1.0 + 1e-9


def test_cap_lb_one_returns_mean_direction() -> None:
    rng = np.random.default_rng(1)
    means = sample_uniform_directions(200, 32, rng)
    v = sample_spherical_cap(means, 1.0, rng)
    assert np.allclose(v, means, atol=1e-9)


def test_cap_vector_lower_bounds_respected_per_row() -> None:
    rng = np.random.default_rng(2)
    means = sample_uniform_directions(500, 48, rng)
    lb = rng.uniform(0.2, 0.95, size=500)
    v = sample_spherical_cap(means, lb, rng)
    cos = np.sum(v * means, axis=1)
    assert np.all(cos >= lb - 1e-9)


def test_deterministic_orthogonal_and_tangent_are_unit_and_orthogonal() -> None:
    rng = np.random.default_rng(3)
    means = sample_uniform_directions(100, 40, rng)
    ortho = _deterministic_orthogonal(means)
    tangent = _sample_tangent_directions(means, rng)
    for vecs in (ortho, tangent):
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-10)
        assert np.allclose(np.sum(vecs * means, axis=1), 0.0, atol=1e-9)


# --------------------------------------------------------------------------- lb modulation


def _single_domain_cap_config(*, coupling_on: bool) -> SimulatorConfig:
    return SimulatorConfig(
        dimension=32,
        population=PopulationConfig(component_count=2, identity_center_kappas=70.0),
        directional=DirectionalConfig(
            intra_class_sampling_mode=IntraClassSamplingMode.IDPERTURB_CAP,
            cap_base_lower_bound=0.6,
            cap_quality_lower_bound_slope=0.30,
            cap_low_tail_penalty=0.20,
            cap_high_tail_penalty=0.20,
            cap_overlap_guard=False,
        ),
        coupling=CouplingConfig(
            quality_affects_intra_kappa=coupling_on,
            norm_z_affects_intra_kappa=coupling_on,
        ),
        acquisition_domains=(AcquisitionDomainConfig(name="default"),),
    )


def test_cap_lb_modulation_follows_quality_and_norm_when_coupled() -> None:
    sim = AdaFaceEmbeddingSimulator(_single_domain_cap_config(coupling_on=True), random_seed=5)
    md = sim.sample_dataset(identity_count=200, samples_per_identity=8, split_name="m").metadata
    lb = md["cap_lower_bound_effective"].to_numpy()
    quality = md["latent_quality"].to_numpy()
    norm_z_abs = np.abs(md["reference_scaled_log_norm_z"].to_numpy())
    assert np.corrcoef(quality, lb)[0, 1] > 0.3            # higher quality -> higher lb
    assert np.corrcoef(norm_z_abs, lb)[0, 1] < -0.3        # norm extremes -> lower lb


def test_cap_lb_is_flat_when_decoupled() -> None:
    sim = AdaFaceEmbeddingSimulator(_single_domain_cap_config(coupling_on=False), random_seed=5)
    md = sim.sample_dataset(identity_count=100, samples_per_identity=8, split_name="m").metadata
    lb = md["cap_lower_bound_effective"].to_numpy()
    assert np.allclose(lb, 0.6, atol=1e-12)


def test_domain_offset_shifts_lb() -> None:
    config = SimulatorConfig(
        dimension=32,
        directional=DirectionalConfig(
            intra_class_sampling_mode=IntraClassSamplingMode.IDPERTURB_CAP,
            cap_base_lower_bound=0.6,
            cap_overlap_guard=False,
        ),
        coupling=CouplingConfig(quality_affects_intra_kappa=False, norm_z_affects_intra_kappa=False),
        acquisition_domains=(
            AcquisitionDomainConfig(name="hi", weight=0.5, cap_lower_bound_offset=0.10),
            AcquisitionDomainConfig(name="lo", weight=0.5, cap_lower_bound_offset=-0.15),
        ),
    )
    md = AdaFaceEmbeddingSimulator(config, random_seed=6).sample_dataset(
        identity_count=200, samples_per_identity=6, split_name="d"
    ).metadata
    by_domain = md.groupby("acquisition_domain")["cap_lower_bound_effective"].mean()
    assert abs(by_domain["hi"] - 0.70) < 1e-9
    assert abs(by_domain["lo"] - 0.45) < 1e-9


# --------------------------------------------------------------------------- overlap guard


def test_overlap_floor_excludes_self_and_grows_with_density() -> None:
    floor = AdaFaceEmbeddingSimulator._overlap_lower_bound_floor
    # Two orthogonal centres: nearest cosine 0 -> floor = sqrt(0.5); self (cos 1) excluded.
    ortho = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.allclose(floor(ortho), np.sqrt(0.5))
    # Denser pair -> higher floor.
    close = np.array([[1.0, 0.0, 0.0], [np.cos(0.2), np.sin(0.2), 0.0]])
    assert floor(close)[0] > floor(ortho)[0]
    # Singleton -> floor 0.
    assert np.allclose(floor(ortho[:1]), 0.0)


def test_overlap_floor_chunked_matches_full() -> None:
    rng = np.random.default_rng(7)
    centers = sample_uniform_directions(300, 24, rng)
    floor = AdaFaceEmbeddingSimulator._overlap_lower_bound_floor
    assert np.allclose(floor(centers, chunk_size=37), floor(centers, chunk_size=10_000))


def test_guard_keeps_perturbation_closer_to_own_center() -> None:
    # Bisector guarantee: with cos(v,c_i) >= floor_i, v stays closer to c_i than to the
    # nearest other centre (pre-nuisance).
    rng = np.random.default_rng(8)
    centers = sample_uniform_directions(150, 32, rng)
    floor = AdaFaceEmbeddingSimulator._overlap_lower_bound_floor(centers)
    v = sample_spherical_cap(centers, floor, rng)
    own = np.sum(v * centers, axis=1)
    cross = v @ centers.T
    np.fill_diagonal(cross, -np.inf)
    nearest_other = cross.max(axis=1)
    assert np.all(own >= nearest_other - 1e-9)


def test_effective_lb_can_exceed_configured_max_under_guard() -> None:
    # Dense population + low nominal max: the guard should lift the effective lb above it.
    config = replace(
        make_idperturb_cap_config(dimension=32),
        population=PopulationConfig(component_count=1, identity_center_kappas=300.0),  # tight cluster
    )
    config = replace(config, directional=replace(config.directional, cap_maximum_lower_bound=0.5, cap_base_lower_bound=0.4))
    md = AdaFaceEmbeddingSimulator(config, random_seed=9).sample_dataset(
        identity_count=400, samples_per_identity=2, split_name="g"
    ).metadata
    eff = md["cap_lower_bound_effective"].to_numpy()
    assert eff.max() > 0.5  # exceeded the configured maximum via the guard
    # Percentiles are reportable (sanity-sweep requirement).
    assert np.percentile(eff, 90) >= np.percentile(eff, 10)


# --------------------------------------------------------------------------- dispatch / invariants


def _no_nuisance_cap_config() -> SimulatorConfig:
    config = make_idperturb_cap_config(dimension=48)
    directional = replace(
        config.directional,
        anisotropic_base_scale=0.0,
        anisotropic_rank=0,
        identity_specific_nuisance_scale=0.0,
        cap_overlap_guard=False,
    )
    domains = tuple(
        replace(d, domain_bias_scale=0.0, domain_identity_bias_scale=0.0, session_bias_scale=0.0)
        for d in config.acquisition_domains
    )
    return replace(config, directional=directional, acquisition_domains=domains)


def test_cap_without_nuisance_floor_is_exact() -> None:
    md = AdaFaceEmbeddingSimulator(_no_nuisance_cap_config(), random_seed=7).sample_dataset(
        identity_count=60, samples_per_identity=4, split_name="cap"
    ).metadata
    # No nuisance -> post-nuisance cosine equals pre-nuisance, both >= lb.
    assert np.allclose(md["own_center_cosine"], md["pre_nuisance_center_cosine"], atol=1e-9)
    assert bool((md["pre_nuisance_center_cosine"] >= md["cap_lower_bound_effective"] - 1e-9).all())


def test_nuisance_can_push_post_cosine_below_lb_but_pre_holds() -> None:
    # Default idperturb profile HAS anisotropic nuisance; pre-nuisance floor always holds,
    # post-nuisance may dip below.
    md = AdaFaceEmbeddingSimulator(make_idperturb_cap_config(dimension=48), random_seed=11).sample_dataset(
        identity_count=120, samples_per_identity=6, split_name="cap"
    ).metadata
    assert bool((md["pre_nuisance_center_cosine"] >= md["cap_lower_bound_effective"] - 1e-9).all())
    below = (md["own_center_cosine"] < md["cap_lower_bound_effective"] - 1e-6).mean()
    assert below > 0.0  # nuisance does shift some samples below the geometric floor


def test_overlap_floor_not_serialized_in_manifest(tmp_path: Path) -> None:
    ds = AdaFaceEmbeddingSimulator(make_idperturb_cap_config(dimension=48), random_seed=7).sample_dataset(
        identity_count=30, samples_per_identity=4, split_name="cap"
    )
    manifest = ds.save_manifest(tmp_path / "m.json")
    text = json.loads(manifest.read_text(encoding="utf-8"))
    assert "overlap_lower_bound_floor" not in json.dumps(text)


# --------------------------------------------------------------------------- hybrid


def test_hybrid_all_samples_respect_floor() -> None:
    config = replace(
        make_idperturb_cap_config(dimension=48),
        directional=replace(
            make_idperturb_cap_config(dimension=48).directional,
            intra_class_sampling_mode=IntraClassSamplingMode.HYBRID_VMF_CAP,
            cap_base_lower_bound=0.5,
            cap_overlap_guard=False,
        ),
    )
    md = AdaFaceEmbeddingSimulator(config, random_seed=3).sample_dataset(
        identity_count=40, samples_per_identity=4, split_name="hyb"
    ).metadata
    assert bool((md["pre_nuisance_center_cosine"] >= md["cap_lower_bound_effective"] - 1e-9).all())


def _hybrid_config(*, lower_bound: float, peak_kappa: float) -> SimulatorConfig:
    return SimulatorConfig(
        dimension=24,
        directional=DirectionalConfig(
            intra_class_sampling_mode=IntraClassSamplingMode.HYBRID_VMF_CAP,
            peak_intra_kappa=peak_kappa,
            minimum_intra_kappa=10.0,
            maximum_intra_kappa=peak_kappa,
            cap_base_lower_bound=lower_bound,
            cap_overlap_guard=False,
            anisotropic_base_scale=0.0,
            anisotropic_rank=0,
        ),
        coupling=CouplingConfig(quality_affects_intra_kappa=False, norm_z_affects_intra_kappa=False),
        acquisition_domains=(AcquisitionDomainConfig(name="default"),),
    )


def test_hybrid_fallback_rate_rises_with_high_lb_low_kappa() -> None:
    hard = AdaFaceEmbeddingSimulator(_hybrid_config(lower_bound=0.9, peak_kappa=15.0), random_seed=4).sample_dataset(
        identity_count=80, samples_per_identity=4, split_name="h"
    ).metadata
    easy = AdaFaceEmbeddingSimulator(_hybrid_config(lower_bound=0.3, peak_kappa=2000.0), random_seed=4).sample_dataset(
        identity_count=80, samples_per_identity=4, split_name="e"
    ).metadata
    assert hard["hybrid_used_cap_fallback"].mean() > easy["hybrid_used_cap_fallback"].mean()
    assert easy["hybrid_used_cap_fallback"].mean() < 0.05  # body ~ vMF when lb low / kappa high


# --------------------------------------------------------------------------- determinism / golden


@pytest.mark.parametrize("mode", list(IntraClassSamplingMode))
def test_each_mode_is_deterministic(mode) -> None:
    base = make_idperturb_cap_config(dimension=32)
    config = replace(base, directional=replace(base.directional, intra_class_sampling_mode=mode))
    a = AdaFaceEmbeddingSimulator(config, random_seed=2).sample_dataset(identity_count=20, samples_per_identity=4, split_name="d")
    b = AdaFaceEmbeddingSimulator(config, random_seed=2).sample_dataset(identity_count=20, samples_per_identity=4, split_name="d")
    assert np.array_equal(a.raw_embeddings, b.raw_embeddings)


def test_golden_vmf_core_arrays_unchanged() -> None:
    # VMF output must remain bit-for-bit identical to v0.2 (fingerprint over the core
    # arrays only — new metadata columns intentionally change the full bundle).
    ds = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=64), random_seed=7).sample_dataset(
        identity_count=25, samples_per_identity=4, split_name="golden", identity_offset=0
    )
    digests = {
        "raw": "38e327b737678c6c5cca9c0bb8e26675d613ee0ba88cc4d4cc3be0a48a8e7bdb",
        "normalized": "062f9bd3fb1573ea9a19b8d3f0ac3b16844cc38f986941ccce9deb5f78bf4672",
        "centers": "dfae80133c19d48fb5eead8b88ef056af7aff48b6036627cf2709a1170be42c2",
    }
    assert hashlib.sha256(ds.raw_embeddings.tobytes()).hexdigest() == digests["raw"]
    assert hashlib.sha256(ds.normalized_embeddings.tobytes()).hexdigest() == digests["normalized"]
    assert hashlib.sha256(ds.identity_centers.tobytes()).hexdigest() == digests["centers"]


def test_idperturb_profile_config_roundtrips() -> None:
    config = make_idperturb_cap_config(dimension=64)
    assert config_to_dict(config_from_dict(config_to_dict(config))) == config_to_dict(config)
