"""Distribuição de escores com série degenerada + aviso de pares idênticos.

Cenário real: probe e referência compartilhando as mesmas imagens → todos
os pares genuínos comparam a imagem consigo mesma (cosseno = 1,0, desvio
~1e-16). A KDE desses genuínos virava um pico de ~1e13 que achatava os
impostores no gráfico, e nada avisava sobre os pares idênticos.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from faceaudit.executors.instructions.similarity_scoring_instruction_executor import (
    SimilarityScoringInstructionExecutor,
)
from faceaudit.models.embedding_info import EmbeddingInfo
from faceaudit.models.similarity.distribution_plot import draw_distribution_from_summary
from faceaudit.models.similarity.kde_curve import (
    is_renderable_density_curve,
    kde_curve,
)
from faceaudit.models.similarity.similarity_score_dataset import SimilarityScoreDataset
from faceaudit.models.similarity.similarity_score_sample import SimilarityScoreSample

_RNG = np.random.default_rng(0)
# Genuínos "mesma imagem": 1,0 com ruído de ponto flutuante.
_DEGENERATE = 1.0 + _RNG.normal(0.0, 3e-16, 118)
_IMPOSTORS = _RNG.normal(0.07, 0.08, 5000)
_XS = np.linspace(-0.2, 1.0, 400)


def _emb(face_hash: str, identity: str) -> EmbeddingInfo:
    return EmbeddingInfo(
        embedding=np.zeros(4), face_hash=face_hash, identity=identity,
        image_path=face_hash, file_name=face_hash,
    )


# ── kde_curve ────────────────────────────────────────────────────────────────

def test_kde_curve_returns_none_for_near_constant_series():
    assert kde_curve(_DEGENERATE, _XS) is None


def test_kde_curve_returns_none_for_exactly_constant_series():
    assert kde_curve(np.ones(50), _XS) is None


def test_kde_curve_returns_none_for_too_few_values():
    assert kde_curve([0.3], _XS) is None
    assert kde_curve([np.nan, 0.3], _XS) is None


def test_kde_curve_integrates_to_about_one_for_regular_series():
    ys = kde_curve(_IMPOSTORS, _XS)
    assert ys is not None
    assert abs(np.trapezoid(ys, _XS) - 1.0) < 0.02


# ── is_renderable_density_curve (sidecars antigos) ───────────────────────────

def test_renderable_rejects_raw_scipy_spike():
    from scipy.stats import gaussian_kde
    # O pico só aparece quando cai sobre um ponto da grade (1,0 é o extremo).
    spike = gaussian_kde(_DEGENERATE)(_XS)
    assert spike.max() > 1e6
    assert not is_renderable_density_curve(_XS, spike)


def test_renderable_accepts_regular_curve_and_rejects_bad_input():
    assert is_renderable_density_curve(_XS, kde_curve(_IMPOSTORS, _XS))
    assert not is_renderable_density_curve(_XS, None)
    assert not is_renderable_density_curve(None, np.zeros(400))
    assert not is_renderable_density_curve(_XS, np.zeros(10))
    assert not is_renderable_density_curve(_XS, np.full(400, np.inf))


# ── Sidecar + desenho ────────────────────────────────────────────────────────

def _degenerate_score_dataset() -> SimilarityScoreDataset:
    items = [
        SimilarityScoreSample(
            probe=_emb(f"g{i}", f"id{i}"), reference=_emb(f"g{i}", f"id{i}"),
            raw_score=float(s), same_person=True,
        )
        for i, s in enumerate(_DEGENERATE)
    ]
    items += [
        SimilarityScoreSample(
            probe=_emb(f"p{i}", f"a{i}"), reference=_emb(f"r{i}", f"b{i}"),
            raw_score=float(s), same_person=False,
        )
        for i, s in enumerate(_IMPOSTORS)
    ]
    return SimilarityScoreDataset(name="degenerado", items=items)


def test_summary_drops_degenerate_kde_but_keeps_impostor_kde():
    summary = _degenerate_score_dataset()._compute_distribution_summary()
    assert summary["kde_gen_y"] is None
    assert summary["kde_imp_y"] is not None
    assert summary["n_gen"] == 118


def test_old_sidecar_with_spike_draws_with_readable_y_axis():
    summary = _degenerate_score_dataset()._compute_distribution_summary()
    # Simula sidecar gravado antes da correção: KDE crua do scipy.
    from scipy.stats import gaussian_kde
    kde_x = np.asarray(summary["kde_x"])
    summary["kde_gen_y"] = gaussian_kde(_DEGENERATE)(kde_x).tolist()

    fig = Figure()
    draw_distribution_from_summary(fig, summary)
    y_top = fig.axes[0].get_ylim()[1]
    # Histograma dos genuínos num único bin de largura ~0,02 → densidade ~50.
    assert y_top < 1e3


# ── Contagem de pares com a mesma imagem ─────────────────────────────────────

_count = SimilarityScoringInstructionExecutor._count_identical_image_pairs


def test_count_identical_pairs_cross_comparison():
    probe = [_emb("a", "1"), _emb("b", "2"), _emb("c", "3")]
    reference = [_emb("a", "1"), _emb("a", "1"), _emb("c", "3"), _emb("z", "9")]
    # a: 1×2, c: 1×1
    assert _count(probe, reference) == 3


def test_count_identical_pairs_self_comparison_counts_duplicates():
    dataset = [_emb("a", "1"), _emb("a", "1"), _emb("a", "1"), _emb("b", "2")]
    # C(3, 2) para "a"
    assert _count(dataset) == 3


def test_count_identical_pairs_ignores_missing_hashes_and_disjoint_sets():
    assert _count([_emb("", "1"), _emb("x", "2")], [_emb("", "1")]) == 0
    assert _count([_emb("x", "1")], [_emb("y", "1")]) == 0
