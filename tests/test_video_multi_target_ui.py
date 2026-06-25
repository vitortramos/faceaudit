"""Fase 7 — multi-alvo na UI (setup dialog + filtro da revisão).

Testes de lógica (offscreen) sobre as janelas: o painel de alvos do setup
dialog e o filtro por alvo da revisão. Não tocam ForensicFace/GPU — as
âncoras são `VideoTarget` fabricados; a listagem de faces da revisão é
monkeypatchada. Cobre o que destrava o multi-alvo: `get_targets()` com N,
limiar por alvo, remoção mantendo ≥1, e escopo visível do filtro.
"""

from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.models.video.video_target import VideoTarget  # noqa: E402
from faceaudit.services.video_review_service import (  # noqa: E402
    VideoReviewService, ReviewFace,
)


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _setup_dialog():
    from interface.tabs.dataset_manager.video_dataset_dialog.video_dataset_setup_dialog import (
        VideoDatasetSetupDialog,
    )
    return VideoDatasetSetupDialog(None, use_gpu=False, model="adaface",
                                   datasets_dir=".", output_dir="C:/tmp/runs")


def _fake_anchor(row, frame_index, det_index, label):
    vt = VideoTarget(label=label, anchor_embeddings=[np.ones(512, dtype=np.float32)],
                     similarity_threshold=row.threshold(), anchor_frame_index=frame_index)
    row.set_anchor(vt, det_index)


# ───────────────────────── setup dialog ─────────────────────────
def test_seeds_one_target(app):
    dlg = _setup_dialog()
    assert len(dlg._target_rows) == 1
    assert dlg._active_row is dlg._target_rows[0]
    assert dlg.get_targets() == []
    assert not dlg.btn_extract.isEnabled()


def test_add_targets_unique_labels_and_colors(app):
    dlg = _setup_dialog()
    dlg._add_target()
    dlg._add_target()
    assert [r.label_text() for r in dlg._target_rows] == ["alvo", "alvo_2", "alvo_3"]
    assert len({r.color for r in dlg._target_rows}) == 3
    assert dlg._active_row is dlg._target_rows[2]


def test_get_targets_returns_anchored_in_order(app):
    dlg = _setup_dialog()
    dlg._add_target()
    dlg._add_target()
    _fake_anchor(dlg._target_rows[0], 10, 1, "alvo")
    _fake_anchor(dlg._target_rows[2], 20, 0, "alvo_3")
    assert [t.label for t in dlg.get_targets()] == ["alvo", "alvo_3"]
    assert dlg._any_anchor()  # gate do botão Extrair
    # back-compat: get_target() devolve o primeiro ancorado
    assert dlg.get_target().label == "alvo"


def test_per_target_threshold_independent(app):
    dlg = _setup_dialog()
    dlg._add_target()
    dlg._target_rows[0].spn_thr.setValue(0.25)
    dlg._target_rows[1].spn_thr.setValue(0.55)
    _fake_anchor(dlg._target_rows[0], 1, 0, "alvo")
    _fake_anchor(dlg._target_rows[1], 1, 1, "alvo_2")
    dlg._on_target_threshold_changed(dlg._target_rows[1])
    assert abs(dlg._target_rows[1].video_target.similarity_threshold - 0.55) < 1e-9
    # limiar representativo (manifesto) = primeiro alvo ancorado
    assert abs(dlg._representative_threshold() - 0.25) < 1e-9


def test_label_edit_propagates_to_anchored_target(app):
    dlg = _setup_dialog()
    _fake_anchor(dlg._target_rows[0], 1, 0, "alvo")
    dlg._target_rows[0].txt_label.setText("joao")
    dlg._on_target_label_changed(dlg._target_rows[0])
    assert dlg._target_rows[0].video_target.label == "joao"


def test_remove_keeps_at_least_one(app):
    dlg = _setup_dialog()
    _fake_anchor(dlg._target_rows[0], 1, 0, "alvo")
    dlg._remove_target(dlg._target_rows[0])
    assert len(dlg._target_rows) == 1
    assert not dlg._target_rows[0].has_anchor()


def test_remove_middle_target_reactivates(app):
    dlg = _setup_dialog()
    dlg._add_target()
    dlg._add_target()
    middle = dlg._target_rows[1]
    dlg._remove_target(middle)
    assert len(dlg._target_rows) == 2
    assert dlg._active_row in dlg._target_rows


# ───────────────────────── review dialog ─────────────────────────
def _faces(labels_counts):
    faces, i = [], 0
    for label, n in labels_counts:
        for _ in range(n):
            kw = {}
            for f in dataclasses.fields(ReviewFace):
                nm = f.name
                if nm == "image_file":
                    kw[nm] = f"{label}_{i}.png"
                elif nm == "label":
                    kw[nm] = label
                elif nm == "crop_path":
                    kw[nm] = None
                elif nm == "frame_index":
                    kw[nm] = i
                elif nm in ("det_score", "fiqa_score", "similarity_to_anchor"):
                    kw[nm] = 0.5
                elif nm == "human_review_status":
                    kw[nm] = "pending"
                elif nm == "review_reasons":
                    kw[nm] = []
                else:
                    kw[nm] = None
            faces.append(ReviewFace(**kw))
            i += 1
    return faces


def _review_dialog(monkeypatch, faces):
    monkeypatch.setattr(VideoReviewService, "list_faces", lambda self, run_dir: faces)
    from interface.tabs.dataset_manager.video_dataset_dialog import video_review_dialog as mod
    return mod.VideoReviewDialog(None, "C:/tmp/fake", operator="perito")


def test_review_filter_scopes_selection(app, monkeypatch):
    dlg = _review_dialog(monkeypatch, _faces([("alvo_1", 3), ("alvo_2", 2)]))
    assert len(dlg._thumbs) == 5
    assert dlg.cmb_filter.count() == 3  # todos + 2 rótulos
    assert dlg._label_colors["alvo_1"] != dlg._label_colors["alvo_2"]

    dlg.cmb_filter.setCurrentIndex(2)  # alvo_2
    vis = dlg._visible_thumbs()
    assert [t.label for t in vis] == ["alvo_2", "alvo_2"]
    assert all(not t.isVisible() for t in dlg._thumbs if t.label == "alvo_1")

    dlg._set_all(False)
    assert all(not t.chk.isChecked() for t in vis)
    assert all(t.chk.isChecked() for t in dlg._thumbs if t.label == "alvo_1")


def test_review_no_filter_with_single_label(app, monkeypatch):
    dlg = _review_dialog(monkeypatch, _faces([("alvo", 3)]))
    assert not hasattr(dlg, "cmb_filter")  # filtro só aparece com ≥2 alvos
    assert len(dlg._visible_thumbs()) == 3
