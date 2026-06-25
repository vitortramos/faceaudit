"""Finalização do dataset de vídeo na pasta de datasets.

Cobre o `VideoImportWorker._finalize_to_dataset_dir`: autocontido (copia os
recortes alinhados, paths relativos, portátil) vs caminho absoluto
(referencia o run, sem copiar). Verifica que embeddings/pose/gênero e o
aviso de viés sobrevivem, e o round-trip do parquet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("PyQt6")
from PyQt6 import QtWidgets  # noqa: E402

from faceaudit.models.face_dataset import FaceDataset  # noqa: E402
from faceaudit.models.video.face_provenance import FaceProvenance  # noqa: E402
from faceaudit.services.image_io import imwrite_unicode_safe  # noqa: E402
from faceaudit.services.hash_service import HashService  # noqa: E402
from faceaudit.services.video_run_importer import VideoRunImporter  # noqa: E402
from interface.tabs.dataset_manager.workers.video_import_worker import VideoImportWorker  # noqa: E402


@pytest.fixture(scope="module")
def app():
    a = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield a


def _build_run(tmp_path):
    rd = tmp_path / "clip_fin0001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    for frame in (1, 2):
        crop_name = f"alvo_f{frame:07d}.png"
        imwrite_unicode_safe(rd / "alvo" / crop_name, np.full((24, 24, 3), 50, np.uint8))
        aligned = rd / "_aligned" / "alvo" / crop_name
        imwrite_unicode_safe(aligned, np.full((112, 112, 3), 80, np.uint8))
        prov = FaceProvenance(
            run_id="rf", run_manifest="_run_rf.json", identity_label="alvo", image_file=crop_name,
            image_sha256=hs.calculate_file_hash(rd / "alvo" / crop_name),
            aligned_image_file=f"_aligned/alvo/{crop_name}",
            aligned_image_sha256=hs.calculate_file_hash(aligned),
            source_video_filename="clip.avi", frame_native_index=frame,
            det_score=0.9, fiqa_score=0.8, pose={"yaw": 5.0, "pitch": -2.0, "roll": 0.5},
            gender="M", age=30.0, similarity_to_anchor=0.8,
            auto_decision="accepted", human_review_status="accepted",
        )
        (meta / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    (meta / "_run_rf.json").write_text(json.dumps(
        {"run_id": "rf", "recognition_model": "sepaelv2",
         "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}), encoding="utf-8")
    return rd


def test_finalize_self_contained_copies_crops(app, tmp_path):
    run_dir = _build_run(tmp_path)
    ds = VideoRunImporter().import_run(run_dir)  # sem enrich: ds tem ff_info + is_aligned + viés
    assert ds.is_aligned is True
    dataset_dir = tmp_path / "datasets"
    w = VideoImportWorker(str(run_dir), enrich=False, dataset_dir=str(dataset_dir), self_contained=True)
    final = w._finalize_to_dataset_dir(ds)

    assert (dataset_dir / f"{ds.name}.parquet").exists()
    copied = list((dataset_dir / "images").glob("*.png"))
    assert len(copied) == 2  # recortes copiados → autocontido
    assert all(not Path(f.image_path).is_absolute() for f in final.faces)
    assert final.provenance_metadata.get("dataset_origin_type") == "VIDEO_TARGET_EXTRACTION"

    ds2 = FaceDataset.from_parquet(dataset_dir / f"{ds.name}.parquet")
    assert len(ds2) == 2
    row = ds2.to_pandas().iloc[0]
    assert row["yaw"] == 5.0 and row["gender_est"] == "M"  # ff_info clonado


def test_finalize_absolute_references_run(app, tmp_path):
    run_dir = _build_run(tmp_path)
    ds = VideoRunImporter().import_run(run_dir)
    dataset_dir = tmp_path / "datasets_abs"
    w = VideoImportWorker(str(run_dir), enrich=False, dataset_dir=str(dataset_dir), self_contained=False)
    final = w._finalize_to_dataset_dir(ds)

    assert (dataset_dir / f"{ds.name}.parquet").exists()
    assert not (dataset_dir / "images").exists()  # sem cópia
    assert all(Path(f.image_path).is_absolute() for f in final.faces)
    assert all("_aligned" in f.image_path.replace("\\", "/") for f in final.faces)
