"""Testes da Fase 2 — importador de run de vídeo + provenance_metadata.

Constrói um run sintético no disco (crops + sidecars + manifesto), sem
precisar de vídeo/ForensicFace, e valida o importador e o round-trip do
parquet (aviso de viés + colunas video_* com column_specs).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.hash_service import HashService
from faceaudit.services.video_run_importer import VideoRunImporter, build_video_column_specs


def _write_face(run_dir, meta_dir, hs, label, frame, status, det=0.9, fiqa=0.8):
    crop = np.full((40, 40, 3), (frame * 7) % 256, np.uint8)
    crop_name = f"{label}_f{frame:07d}.png"
    crop_path = run_dir / label / crop_name
    imwrite_unicode_safe(crop_path, crop)
    sha = hs.calculate_file_hash(crop_path)
    prov = FaceProvenance(
        run_id="run123", run_manifest="_run_run123.json", identity_label=label,
        image_file=crop_name, image_sha256=sha, image_width_px=40, image_height_px=40,
        crop_margin_factor=2.0, source_video_filename="clip.avi", source_video_sha256="vidsha256",
        frame_native_index=frame, timestamp_s_from_fps=frame / 5.0,
        det_score=det, fiqa_score=fiqa, det_score_decision="passed", det_score_policy="TAG_ONLY",
        fiqa_score_decision="passed", fiqa_score_policy="TAG_ONLY",
        similarity_to_anchor=0.8, anchor_embedding_fingerprint="sha256:abc",
        auto_decision="accepted", human_review_status=status,
    )
    (meta_dir / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    return crop_path


@pytest.fixture
def run_dir(tmp_path):
    rd = tmp_path / "clip_abc1234567"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    _write_face(rd, meta, hs, "alvo", 1, "accepted", 0.9, 0.8)
    _write_face(rd, meta, hs, "alvo", 2, "accepted", 0.7, 0.4)
    _write_face(rd, meta, hs, "alvo", 3, "rejected", 0.9, 0.9)
    manifest = {"run_id": "run123", "recognition_model": "sepaelv2",
                "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}
    (meta / "_run_run123.json").write_text(json.dumps(manifest), encoding="utf-8")
    return rd


def test_build_video_column_specs_keys():
    specs = build_video_column_specs()
    for k in ("target_label", "similarity_to_anchor", "video_extraction_det_score",
              "video_extraction_fiqa_score", "video_det_score_policy", "source_video_sha256",
              "frame_index", "anchor_embedding_fingerprint"):
        assert k in specs
        assert specs[k].category == "Vídeo (extração)"


def test_import_run_excludes_rejected_and_promotes_provenance(run_dir):
    ds = VideoRunImporter().import_run(run_dir)
    assert len(ds) == 2  # rejeitado excluído
    assert ds.path_mode == PathMode.RELATIVE and ds.path_root == "."
    f0 = ds.faces[0]
    assert f0.identity == "alvo"
    assert f0.extra_data["source_video_sha256"] == "vidsha256"
    assert "video_extraction_det_score" in f0.extra_data
    assert "det_score" not in f0.extra_data  # namespaced, sem colisão
    # proveniência de viés no dataset
    assert ds.provenance_metadata["dataset_origin_type"] == "VIDEO_TARGET_EXTRACTION"
    assert ds.provenance_metadata["selection_anchor_model"] == "sepaelv2"
    assert "selection_warning" in ds.provenance_metadata
    # column specs das colunas de vídeo
    assert "video_extraction_fiqa_score" in ds.column_specs


def test_import_run_can_include_pending(run_dir):
    ds = VideoRunImporter().import_run(run_dir, accepted_statuses=("accepted", "pending"))
    assert len(ds) == 2  # ainda 2 (não há pending neste fixture; rejeitado fora)


def test_import_run_hash_mismatch_skips_face(run_dir):
    # adultera um recorte aceito após o sidecar → hash diverge
    tampered = run_dir / "alvo" / "alvo_f0000001.png"
    imwrite_unicode_safe(tampered, np.full((40, 40, 3), 200, np.uint8))
    ds = VideoRunImporter().import_run(run_dir, validate_hashes=True)
    assert len(ds) == 1  # só o f2 sobra
    assert ds.faces[0].extra_data["frame_index"] == 2


def test_parquet_roundtrip_preserves_bias_and_video_columns(run_dir, tmp_path):
    ds = VideoRunImporter().import_run(run_dir)
    out = tmp_path / "video_ds.parquet"
    ds.to_parquet(out)
    ds2 = FaceDataset.from_parquet(out)
    # aviso de viés sobreviveu ao round-trip
    assert ds2.provenance_metadata["dataset_origin_type"] == "VIDEO_TARGET_EXTRACTION"
    assert "selection_warning" in ds2.provenance_metadata
    assert ds2.provenance_metadata["selection_conditioned_on_embedding"] == "true"
    # colunas video_* presentes no df e com column_specs
    df = ds2.to_pandas()
    assert "video_extraction_det_score" in df.columns
    assert "video_fiqa_score_decision" in df.columns
    assert "similarity_to_anchor" in df.columns
    assert "video_extraction_det_score" in ds2.column_specs


def _write_face_with_aligned(run_dir, meta_dir, hs, label, frame):
    crop_name = f"{label}_f{frame:07d}.png"
    imwrite_unicode_safe(run_dir / label / crop_name, np.full((24, 24, 3), 50, np.uint8))
    aligned_path = run_dir / "_aligned" / label / crop_name
    imwrite_unicode_safe(aligned_path, np.full((112, 112, 3), 80, np.uint8))
    prov = FaceProvenance(
        run_id="ra", run_manifest="_run_ra.json", identity_label=label, image_file=crop_name,
        image_sha256=hs.calculate_file_hash(run_dir / label / crop_name),
        aligned_image_file=f"_aligned/{label}/{crop_name}",
        aligned_image_sha256=hs.calculate_file_hash(aligned_path),
        frame_native_index=frame, det_score=0.9, fiqa_score=0.8, similarity_to_anchor=0.8,
        auto_decision="accepted", human_review_status="accepted",
    )
    (meta_dir / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")


def _write_face_full(run_dir, meta_dir, hs, label, frame):
    """Sidecar com pose + gender + age (capturados na extração, extended=True)."""
    crop_name = f"{label}_f{frame:07d}.png"
    cp = run_dir / label / crop_name
    imwrite_unicode_safe(cp, np.full((40, 40, 3), 90, np.uint8))
    prov = FaceProvenance(
        run_id="rf", run_manifest="_run_rf.json", identity_label=label, image_file=crop_name,
        image_sha256=hs.calculate_file_hash(cp), source_video_filename="clip.avi",
        frame_native_index=frame, det_score=0.91, fiqa_score=0.77,
        pose={"yaw": 12.5, "pitch": -4.0, "roll": 1.2}, gender="M", age=33.0,
        similarity_to_anchor=0.8, auto_decision="accepted", human_review_status="accepted",
    )
    (meta_dir / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")


def test_import_promotes_pose_gender_origin_to_canonical_columns(tmp_path):
    rd = tmp_path / "clip_full0001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    _write_face_full(rd, meta, hs, "alvo", 1)
    (meta / "_run_rf.json").write_text(json.dumps(
        {"run_id": "rf", "recognition_model": "sepaelv2",
         "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}), encoding="utf-8")
    ds = VideoRunImporter().import_run(rd)
    assert len(ds) == 1
    f = ds.faces[0]
    # FFInfo populado a partir do sidecar (genderage + pose capturados na extração)
    assert f.ff_info is not None
    assert f.ff_info.yaw == 12.5 and f.ff_info.pitch == -4.0 and f.ff_info.roll == 1.2
    assert f.ff_info.gender == "M" and f.ff_info.age == 33.0
    # origem identifica o vídeo de origem (análogo ao campo Origem das imagens)
    assert f.origin == "Vídeo: clip.avi"
    # colunas canônicas no parquet (não vazias)
    row = ds.to_pandas().iloc[0]
    assert row["yaw"] == 12.5 and row["pitch"] == -4.0
    assert row["gender_est"] == "M"   # estimativa do genderage → gender_est
    assert row["age_est"] == 33.0
    assert row["origin"] == "Vídeo: clip.avi"


def test_import_run_prefers_aligned_and_sets_is_aligned(tmp_path):
    rd = tmp_path / "clip_aligned0001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    _write_face_with_aligned(rd, meta, hs, "alvo", 1)
    _write_face_with_aligned(rd, meta, hs, "alvo", 2)
    (meta / "_run_ra.json").write_text(json.dumps(
        {"run_id": "ra", "recognition_model": "sepaelv2",
         "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}), encoding="utf-8")
    ds = VideoRunImporter().import_run(rd)
    assert len(ds) == 2
    assert ds.is_aligned is True  # usou as faces alinhadas → enrich não re-detecta
    assert all(f.image_path.startswith("_aligned/") for f in ds.faces)
