"""Testes da Fase 3 — revisão/cull de um run de extração de vídeo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.hash_service import HashService
from faceaudit.services.video_review_service import VideoReviewService
from faceaudit.services.video_run_importer import VideoRunImporter


def _write_pending_face(run_dir, meta_dir, hs, label, frame, *, reasons=None):
    crop = np.full((40, 40, 3), (frame * 9) % 256, np.uint8)
    crop_name = f"{label}_f{frame:07d}.png"
    crop_path = run_dir / label / crop_name
    imwrite_unicode_safe(crop_path, crop)
    sha = hs.calculate_file_hash(crop_path)
    prov = FaceProvenance(
        run_id="run9", run_manifest="_run_run9.json", identity_label=label,
        image_file=crop_name, image_sha256=sha, frame_native_index=frame,
        det_score=0.9, fiqa_score=0.8, similarity_to_anchor=0.8,
        auto_decision="accepted", review_reasons=list(reasons or []),
        human_review_status="pending",
    )
    (meta_dir / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    return crop_name


@pytest.fixture
def run_dir(tmp_path):
    rd = tmp_path / "clip_rev0000001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    _write_pending_face(rd, meta, hs, "alvo", 1)
    _write_pending_face(rd, meta, hs, "alvo", 2, reasons=["low_fiqa"])
    _write_pending_face(rd, meta, hs, "alvo", 3)
    manifest = {"run_id": "run9", "recognition_model": "sepaelv2", "extraction_status": "completed",
                "review_status": "not_started", "dataset_status": "not_imported",
                "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}
    (meta / "_run_run9.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rd


def test_list_faces_and_summary(run_dir):
    svc = VideoReviewService()
    faces = svc.list_faces(run_dir)
    assert len(faces) == 3
    assert all(f.human_review_status == "pending" for f in faces)
    f2 = next(f for f in faces if f.frame_index == 2)
    assert "low_fiqa" in f2.review_reasons
    assert svc.review_summary(run_dir) == {"pending": 3, "accepted": 0, "rejected": 0}


def test_reject_moves_crop_to_rejected_and_excludes_from_import(run_dir):
    svc = VideoReviewService()
    rf = svc.set_face_decision(run_dir, "alvo_f0000001.png", status="rejected",
                               operator="perito", reason="false_positive")
    assert rf.human_review_status == "rejected"
    assert not (Path(run_dir) / "alvo" / "alvo_f0000001.png").exists()
    assert (Path(run_dir) / "_rejected" / "alvo_f0000001.png").exists()
    # aceita as outras duas e importa → rejeitado fora
    svc.set_face_decision(run_dir, "alvo_f0000002.png", status="accepted", operator="perito")
    svc.set_face_decision(run_dir, "alvo_f0000003.png", status="accepted", operator="perito")
    ds = VideoRunImporter().import_run(run_dir)
    assert len(ds) == 2
    frames = sorted(f.extra_data["frame_index"] for f in ds.faces)
    assert frames == [2, 3]


def test_re_accept_moves_crop_back(run_dir):
    svc = VideoReviewService()
    svc.set_face_decision(run_dir, "alvo_f0000001.png", status="rejected", operator="p")
    assert (Path(run_dir) / "_rejected" / "alvo_f0000001.png").exists()
    svc.set_face_decision(run_dir, "alvo_f0000001.png", status="accepted", operator="p")
    assert (Path(run_dir) / "alvo" / "alvo_f0000001.png").exists()
    assert not (Path(run_dir) / "_rejected" / "alvo_f0000001.png").exists()


def test_commit_review_completed(run_dir):
    svc = VideoReviewService()
    svc.set_face_decision(run_dir, "alvo_f0000001.png", status="accepted", operator="p")
    svc.set_face_decision(run_dir, "alvo_f0000002.png", status="rejected", operator="p")
    block = svc.commit_review(run_dir, operator="p", skipped=False,
                              now_utc=lambda: "2026-01-01T00:00:00Z")
    assert block["skipped"] is False
    manifest = json.loads(next((Path(run_dir) / "_meta").glob("_run_*.json")).read_text(encoding="utf-8"))
    assert manifest["review_status"] == "completed"
    assert manifest["review"]["n_accepted"] == 1
    assert manifest["review"]["n_rejected"] == 1
    assert manifest["review"]["n_pending"] == 1


def test_commit_review_skipped_override_is_audited(run_dir):
    svc = VideoReviewService()
    svc.commit_review(run_dir, operator="p", skipped=True, now_utc=lambda: "2026-01-01T00:00:00Z")
    manifest = json.loads(next((Path(run_dir) / "_meta").glob("_run_*.json")).read_text(encoding="utf-8"))
    assert manifest["review_status"] == "skipped"
    assert manifest["review"]["skipped"] is True
    assert manifest["review"]["operator"] == "p"


def test_set_face_decision_invalid_status_raises(run_dir):
    with pytest.raises(ValueError):
        VideoReviewService().set_face_decision(run_dir, "alvo_f0000001.png", status="maybe")
