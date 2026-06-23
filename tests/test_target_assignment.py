"""Testes da Fase 4 — módulo puro de atribuição/tracking."""

from __future__ import annotations

import math

import numpy as np

from faceaudit.services.target_assignment import (
    Candidate, TargetState, assign_frame, cosine, iou, predict_bbox,
)


def _unit_cos(c):
    """Vetor 2D unitário com cosseno `c` em relação a [1,0]."""
    return np.array([c, math.sqrt(max(0.0, 1.0 - c * c))], dtype=float)


def test_cosine_and_iou_basics():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 1.0
    assert abs(cosine(np.array([1.0, 0.0]), np.array([0.0, 1.0]))) < 1e-9
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_single_target_above_threshold_assigns_embedding():
    t = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.5)
    cands = [Candidate(det_index=0, bbox=(0, 0, 10, 10), embedding=np.array([1.0, 0.0]))]
    asg = assign_frame([t], cands, frame_index=0)
    assert len(asg) == 1
    assert asg[0].det_index == 0
    assert asg[0].assigned_by == "embedding"
    assert asg[0].margin_to_second_best is None
    assert t.last_frame_index == 0  # track atualizado


def test_below_threshold_no_assignment():
    t = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.9)
    cands = [Candidate(0, (0, 0, 10, 10), embedding=np.array([0.0, 1.0]))]  # sim 0
    assert assign_frame([t], cands, frame_index=0) == []


def test_two_candidates_picks_higher_sim_and_reports_margin():
    t = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.5)
    cands = [
        Candidate(0, (0, 0, 10, 10), embedding=_unit_cos(0.95)),
        Candidate(1, (50, 50, 60, 60), embedding=_unit_cos(0.80)),
    ]
    asg = assign_frame([t], cands, frame_index=0)
    assert len(asg) == 1 and asg[0].det_index == 0
    assert asg[0].margin_to_second_best == pytest_approx(0.15)


def test_spatial_tiebreak_changes_winner():
    # B tem sim levemente menor mas casa com a trajetória → vence com peso espacial
    t = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.5,
                    last_bbox=(50, 50, 60, 60), last_frame_index=0, velocity=(0.0, 0.0))
    cands = [
        Candidate(0, (0, 0, 10, 10), embedding=_unit_cos(0.90)),     # longe da predição
        Candidate(1, (50, 50, 60, 60), embedding=_unit_cos(0.88)),   # casa a predição
    ]
    asg = assign_frame([t], cands, frame_index=1, spatial_weight=0.1)
    assert asg[0].det_index == 1
    assert asg[0].assigned_by == "tiebreak_track"
    # sem peso espacial, venceria o de maior similaridade (det 0)
    t2 = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.5,
                     last_bbox=(50, 50, 60, 60), last_frame_index=0, velocity=(0.0, 0.0))
    asg2 = assign_frame([t2], cands, frame_index=1, spatial_weight=0.0)
    assert asg2[0].det_index == 0 and asg2[0].assigned_by == "embedding"


def test_track_recovery_after_gap():
    t = TargetState("alvo", np.array([1.0, 0.0]), threshold=0.5,
                    last_bbox=(0, 0, 10, 10), last_frame_index=0, velocity=(0.0, 0.0))
    cands = [Candidate(0, (0, 0, 10, 10), embedding=np.array([1.0, 0.0]))]
    asg = assign_frame([t], cands, frame_index=20, gap_tolerance=5)
    assert asg[0].assigned_by == "track_recovery"


def test_one_candidate_not_assigned_to_two_targets():
    t1 = TargetState("a", np.array([1.0, 0.0]), threshold=0.5)
    t2 = TargetState("b", np.array([1.0, 0.0]), threshold=0.5)
    cands = [Candidate(0, (0, 0, 10, 10), embedding=np.array([1.0, 0.0]))]
    asg = assign_frame([t1, t2], cands, frame_index=0)
    assert len(asg) == 1  # só um alvo leva a detecção


def test_two_targets_two_candidates_one_to_one():
    t1 = TargetState("a", np.array([1.0, 0.0]), threshold=0.5)
    t2 = TargetState("b", np.array([0.0, 1.0]), threshold=0.5)
    cands = [
        Candidate(0, (0, 0, 10, 10), embedding=np.array([1.0, 0.0])),
        Candidate(1, (50, 50, 60, 60), embedding=np.array([0.0, 1.0])),
    ]
    asg = assign_frame([t1, t2], cands, frame_index=0)
    by_label = {a.target_label: a.det_index for a in asg}
    assert by_label == {"a": 0, "b": 1}


# pequeno helper local (evita depender do pytest.approx import top-level)
def pytest_approx(value, abs_tol=1e-6):
    import pytest
    return pytest.approx(value, abs=abs_tol)
