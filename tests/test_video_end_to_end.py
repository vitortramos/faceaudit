"""Testes da Fase 6 — ponta a ponta (enrich) + audit op + invariante."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.enums.audit_operation import AuditOperation
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.hash_service import HashService
from faceaudit.services.video_review_service import VideoReviewService
from faceaudit.services.video_run_importer import VideoRunImporter


class _FakePipeline:
    """Simula o FeatureExtractionPipeline.enrich_existing_dataset salvando o
    parquet em _parquet_dir/<name>.parquet (como o real faz)."""
    def __init__(self):
        self.called = False

    def enrich_existing_dataset(self, ds, cfg):
        self.called = True
        ds.to_parquet(Path(ds._parquet_dir) / f"{ds.name}.parquet")
        return ds


def _write_accepted_face(run_dir, meta_dir, hs, label, frame):
    crop = np.full((40, 40, 3), (frame * 11) % 256, np.uint8)
    crop_name = f"{label}_f{frame:07d}.png"
    crop_path = run_dir / label / crop_name
    imwrite_unicode_safe(crop_path, crop)
    sha = hs.calculate_file_hash(crop_path)
    prov = FaceProvenance(
        run_id="r6", run_manifest="_run_r6.json", identity_label=label,
        image_file=crop_name, image_sha256=sha, frame_native_index=frame,
        det_score=0.9, fiqa_score=0.8, similarity_to_anchor=0.8,
        auto_decision="accepted", human_review_status="accepted",
    )
    (meta_dir / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path):
    rd = tmp_path / "clip_e2e0000001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    _write_accepted_face(rd, meta, hs, "alvo", 1)
    _write_accepted_face(rd, meta, hs, "alvo", 2)
    manifest = {"run_id": "r6", "recognition_model": "sepaelv2",
                "source_video": {"sha256": "vid6"}, "extraction_status": "completed",
                "review_status": "not_started", "dataset_status": "not_imported",
                "summary": {"n_frames_sampled": 5, "n_accepted_auto": 2},
                "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}
    (meta / "_run_r6.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rd


def test_audit_operation_video_extract():
    op = AuditOperation.VIDEO_EXTRACT
    assert op.value == "video_extract"
    assert op.emoji == "🎬"
    assert op.label_pt == "Extração de vídeo"


def test_enrich_blocked_before_review(run_dir):
    with pytest.raises(RuntimeError):
        VideoRunImporter().import_run(run_dir, enrich=True, pipeline=_FakePipeline())


def test_enrich_requires_pipeline(run_dir):
    VideoReviewService().commit_review(run_dir, operator="p")  # review_status=completed
    with pytest.raises(ValueError):
        VideoRunImporter().import_run(run_dir, enrich=True, pipeline=None)


def test_enrich_after_review_finalizes_manifest(run_dir):
    VideoReviewService().commit_review(run_dir, operator="p", now_utc=lambda: "2026-01-01T00:00:00Z")
    pipe = _FakePipeline()
    ds = VideoRunImporter().import_run(run_dir, enrich=True, pipeline=pipe)
    assert pipe.called is True
    assert len(ds) == 2
    parquet = Path(run_dir) / f"{Path(run_dir).name}.parquet"
    assert parquet.exists()
    manifest = json.loads(next((Path(run_dir) / "_meta").glob("_run_*.json")).read_text(encoding="utf-8"))
    assert manifest["dataset_status"] == "enriched"
    assert manifest["final_dataset"]["n_faces"] == 2
    assert manifest["final_dataset"]["path"].endswith(".parquet")
    assert manifest["final_dataset"]["sha256"]
    # dataset enriquecido preserva o aviso de viés
    ds2 = FaceDataset.from_parquet(parquet)
    assert ds2.provenance_metadata["dataset_origin_type"] == "VIDEO_TARGET_EXTRACTION"


def test_skipped_review_allows_enrich(run_dir):
    VideoReviewService().commit_review(run_dir, operator="p", skipped=True)
    pipe = _FakePipeline()
    VideoRunImporter().import_run(run_dir, enrich=True, pipeline=pipe)
    assert pipe.called is True


def test_audit_metadata_from_manifest(run_dir):
    VideoReviewService().commit_review(run_dir, operator="p")
    meta = VideoRunImporter().audit_metadata(run_dir)
    assert meta["run_id"] == "r6"
    assert meta["video_sha256"] == "vid6"
    assert meta["n_frames_sampled"] == 5
    assert meta["n_accepted_after_review"] == 2
