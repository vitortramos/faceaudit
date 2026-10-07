"""KPRPE (sepaelv6): keypoints alinhados persistidos e disponíveis no enrich.

O modelo KPRPE exige os 5 keypoints no sistema 112×112 como segundo input.
A extração os captura (detect_and_align), o sidecar os guarda, o importador os
coloca em FaceInfo (memória) + extra_data (parquet), e o caminho alinhado do
FFService os repassa ao batch. Testa o helper + o round-trip de persistência.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.hash_service import HashService
from faceaudit.services.video_run_importer import VideoRunImporter
from faceaudit.services.ff_service import _aligned_keypoints_from_face

_KPS = [[30.0, 40.0], [80.0, 41.0], [55.0, 65.0], [35.0, 85.0], [78.0, 86.0]]


def test_pontos_no_sistema_do_dataset_nao_servem_ao_kprpe():
    """O KPRPE lê SÓ a casa própria dos pontos alinhados.

    A premissa anterior era que bastava haver pontos em
    `ff_info.keypoints`. Mas aquele slot é, por invariante, o sistema de
    coordenadas da imagem que o dataset guarda: num dataset de recortes
    vindo de fotos são as coordenadas da foto original, e alimentar o
    modelo com elas produzia embedding errado sem erro aparente. Seguir
    sem keypoints, que o encoder sabe recusar, é melhor que seguir com os
    errados.
    """
    fi = FaceInfo(image_path="x.png", ff_info=FFInfo(keypoints=_KPS))

    assert _aligned_keypoints_from_face(fi) is None


def test_helper_reads_extra_data_json():
    fi = FaceInfo(image_path="x.png")
    fi.extra_data = {"aligned_keypoints_112": json.dumps(_KPS)}
    arr = _aligned_keypoints_from_face(fi)
    assert arr is not None and arr.shape == (5, 2)
    assert np.allclose(arr, _KPS)


def test_helper_none_when_absent():
    assert _aligned_keypoints_from_face(FaceInfo(image_path="x.png")) is None


def test_importer_sets_keypoints_and_persists(tmp_path):
    rd = tmp_path / "clip_kps0001"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    crop_name = "alvo_f0000001.png"
    cp = rd / "alvo" / crop_name
    imwrite_unicode_safe(cp, np.full((40, 40, 3), 90, np.uint8))
    prov = FaceProvenance(
        run_id="rk", run_manifest="_run_rk.json", identity_label="alvo", image_file=crop_name,
        image_sha256=hs.calculate_file_hash(cp), source_video_filename="clip.avi",
        frame_native_index=1, det_score=0.9, fiqa_score=0.8,
        aligned_keypoints=_KPS, similarity_to_anchor=0.8,
        auto_decision="accepted", human_review_status="accepted",
    )
    (meta / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    (meta / "_run_rk.json").write_text(json.dumps(
        {"run_id": "rk", "recognition_model": "sepaelv2",
         "targets": [{"label": "alvo", "similarity_threshold": 0.5}]}), encoding="utf-8")

    ds = VideoRunImporter().import_run(rd)
    f = ds.faces[0]
    # E2E: keypoints disponíveis em memória + persistidos em extra_data
    assert _aligned_keypoints_from_face(f) is not None
    assert "aligned_keypoints_112" in f.extra_data

    # Re-enrich: sobrevivem ao round-trip do parquet (ff_info.keypoints não
    # persiste; vêm do extra_data).
    out = tmp_path / "ds.parquet"
    ds.to_parquet(out)
    f2 = FaceDataset.from_parquet(out).faces[0]
    arr = _aligned_keypoints_from_face(f2)
    assert arr is not None and arr.shape == (5, 2)
    assert np.allclose(arr, _KPS)
