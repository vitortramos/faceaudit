"""Modo de combinação das âncoras: CENTROID (média) vs ANY (qualquer uma).

Cenário discriminante: duas âncoras ortogonais (A1, A2) e uma face = A1.
- CENTROID: cosseno com a média ≈ 0.707 (cai abaixo de um limiar 0.9).
- ANY: melhor cosseno entre as âncoras = 1.0 (passa).
Cobre o modelo (`VideoTarget.similarity_to`), o scoring puro
(`target_assignment.score`/`assign_frame`) e o seletor na UI (`_AnchorsDialog`).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from faceaudit.models.video.video_target import VideoTarget
from faceaudit.services.target_assignment import (
    Candidate, TargetState, assign_frame, score,
)


def _orthos():
    a1 = np.zeros(512, np.float64); a1[0] = 1.0
    a2 = np.zeros(512, np.float64); a2[1] = 1.0
    mean = a1 + a2
    mean = mean / np.linalg.norm(mean)
    return a1, a2, mean


# ───────────────────────── modelo ─────────────────────────
def test_similarity_to_centroid_vs_any():
    a1, a2, _ = _orthos()
    t = VideoTarget(label="x")
    t.add_anchor(a1, {"type": "video", "frame_index": 1})
    t.add_anchor(a2, {"type": "video", "frame_index": 2})
    cand = a1.copy()
    t.anchor_match_mode = "CENTROID"
    s_centroid = t.similarity_to(cand)
    t.anchor_match_mode = "ANY"
    s_any = t.similarity_to(cand)
    assert abs(s_centroid - (1.0 / np.sqrt(2))) < 1e-3
    assert abs(s_any - 1.0) < 1e-6
    assert s_any > s_centroid


def test_default_mode_is_centroid():
    assert VideoTarget(label="x").anchor_match_mode == "CENTROID"


def test_has_near_duplicate_anchor():
    t = VideoTarget(label="x")
    a = np.zeros(512, np.float32); a[0] = 1.0
    t.add_anchor(a, {"type": "video", "frame_index": 1})
    assert t.has_near_duplicate_anchor(a.copy())     # idêntico
    assert t.has_near_duplicate_anchor(2.0 * a)      # mesma direção (paralelo)
    b = np.zeros(512, np.float32); b[1] = 1.0
    assert not t.has_near_duplicate_anchor(b)        # ortogonal → não é duplicata


# ───────────────────────── scoring puro ─────────────────────────
def test_assignment_score_modes():
    a1, a2, mean = _orthos()
    t = TargetState(label="x", reference_embedding=mean, threshold=0.9,
                    anchor_embeddings=[a1, a2], match_mode="CENTROID")
    cand = a1.copy()
    assert score(t, cand) < 0.9            # centroide ≈ 0.707
    t.match_mode = "ANY"
    assert score(t, cand) >= 0.9           # qualquer uma = 1.0


def test_assign_frame_gate_differs_by_mode():
    a1, a2, mean = _orthos()
    cand = Candidate(det_index=0, bbox=(0, 0, 10, 10), embedding=a1.copy())
    t_centroid = TargetState(label="x", reference_embedding=mean, threshold=0.9,
                             anchor_embeddings=[a1, a2], match_mode="CENTROID")
    assert assign_frame([t_centroid], [cand], 0) == []       # centroide rejeita
    t_any = TargetState(label="x", reference_embedding=mean, threshold=0.9,
                        anchor_embeddings=[a1, a2], match_mode="ANY")
    res = assign_frame([t_any], [cand], 0)
    assert len(res) == 1 and res[0].target_label == "x"      # qualquer uma aceita


# ───────────────────────── UI (seletor) ─────────────────────────
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt6")
    from PyQt6 import QtWidgets
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def test_anchors_dialog_mode_selector(app):
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        _AnchorsDialog,
    )
    t = VideoTarget(label="x")
    t.add_anchor(np.ones(512, np.float32), {"type": "video", "frame_index": 1})
    d = _AnchorsDialog(t, None)
    assert d.cmb_mode.currentData() == "CENTROID"
    d.cmb_mode.setCurrentIndex(d.cmb_mode.findData("ANY"))
    assert t.anchor_match_mode == "ANY"
