"""Enriquecimento com idempotência por ATRIBUTO + caminho "atributos-só".

Cobre o roteamento (embedding-faltando → full, atributo-de-detecção-faltando →
attrs-only, fiqa-faltando → full), o merge seletivo de detect_and_align (sem
tocar no embedding) e a resolução dos atributos pedidos. Sem GPU/ForensicFace —
o engine é um fake que devolve o dict de detect_and_align.
"""
from __future__ import annotations

import numpy as np

from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline as FP

_ALL_DA = {"yaw", "pitch", "roll", "gender", "age", "det_score", "ipd"}


def _face(*, emb=True, **ff_kwargs):
    ff = FFInfo(**ff_kwargs)
    if emb:
        ff.embedding_sepaelv2 = np.ones(4, np.float32)
    return FaceInfo(image_path="x.jpg", identity="1", ff_info=ff)


class _FakeDS:
    def __init__(self, faces):
        self.faces = faces

    def resolve_image_path(self, face):
        return face.image_path


# ───────────────────────── roteamento ─────────────────────────
def test_route_missing_embedding_goes_full():
    face = _face(emb=False)   # sem embedding
    full, attrs = FP._route_ff_faces(_FakeDS([face]), ["sepaelv2"], _ALL_DA, True, False)
    assert full == [face] and attrs == []


def test_route_missing_attr_goes_attrs_only():
    face = _face(yaw=None)    # tem embedding, falta pose
    full, attrs = FP._route_ff_faces(_FakeDS([face]), ["sepaelv2"], _ALL_DA, False, False)
    assert full == [] and attrs == [face]


def test_route_missing_fiqa_goes_full():
    face = _face(yaw=1.0, pitch=2.0, roll=3.0, gender="M", age=30.0,
                 det_score=0.9, ipd=64.0, fiqa_score=None)   # só falta fiqa
    full, attrs = FP._route_ff_faces(_FakeDS([face]), ["sepaelv2"], _ALL_DA, True, False)
    assert full == [face] and attrs == []   # fiqa exige o caminho de reconhecimento


def test_route_complete_face_skipped():
    face = _face(yaw=1.0, pitch=2.0, roll=3.0, gender="M", age=30.0,
                 det_score=0.9, ipd=64.0, fiqa_score=0.8)   # tem tudo
    full, attrs = FP._route_ff_faces(_FakeDS([face]), ["sepaelv2"], _ALL_DA, True, False)
    assert full == [] and attrs == []


def test_route_force_puts_all_in_full():
    a, b = _face(yaw=1.0), _face(emb=False)
    full, attrs = FP._route_ff_faces(_FakeDS([a, b]), ["sepaelv2"], _ALL_DA, True, True)
    assert full == [a, b] and attrs == []


def test_route_aligned_no_da_attrs_skips():
    # Em alinhadas wanted_da vem vazio → face com embedding não vira attrs_only.
    face = _face(yaw=None)
    full, attrs = FP._route_ff_faces(_FakeDS([face]), ["sepaelv2"], set(), False, False)
    assert full == [] and attrs == []


# ───────────────────────── merge seletivo ─────────────────────────
def test_merge_detect_attrs_only_requested():
    ff = FFInfo(embedding_sepaelv2=np.ones(4, np.float32))
    res = {"det_score": 0.9, "gender": "M", "age": 33.0,
           "pose": (5.0, 12.0, 2.0),   # pitch, yaw, roll (cru)
           "keypoints": np.array([[0, 0], [10, 0], [5, 5], [2, 8], [8, 8]], float)}
    FP._merge_detect_attrs(ff, res, {"yaw", "pitch", "roll", "ipd"})
    assert ff.yaw == 12.0 and ff.pitch == 5.0 and ff.roll == 2.0
    assert ff.ipd == 10.0                     # dist entre kps[0] e kps[1]
    assert ff.gender is None and ff.age is None   # não pedidos → não mesclados
    assert ff.det_score is None
    assert ff.embedding_sepaelv2 is not None      # embedding intacto


# ───────────────────────── caminho atributos-só ─────────────────────────
class _FakeEngine:
    def detect_and_align(self, img, single_face=True):
        return {"det_score": 0.9, "gender": "M", "age": 33.0,
                "pose": (5.0, 12.0, 2.0),
                "keypoints": np.array([[0, 0], [10, 0], [5, 5], [2, 8], [8, 8]], float)}


class _FakeSvc:
    def __init__(self):
        self.app = _FakeEngine()


def test_enrich_attributes_only_keeps_embedding(monkeypatch):
    monkeypatch.setattr("faceaudit.services.image_io.imread_unicode_safe",
                        lambda p: np.zeros((64, 64, 3), np.uint8))
    emb = np.ones(4, np.float32)
    face = _face(fiqa_score=0.8)   # tem embedding + fiqa, faltam atributos DA
    face.ff_info.embedding_sepaelv2 = emb.copy()
    n = FP._enrich_attributes_only([face], _FakeDS([face]), _FakeSvc(), _ALL_DA)
    assert n == 1
    assert face.ff_info.yaw == 12.0 and face.ff_info.pitch == 5.0
    assert face.ff_info.gender == "M" and face.ff_info.age == 33.0
    assert face.ff_info.det_score == 0.9 and face.ff_info.ipd == 10.0
    assert face.ff_info.embedding_sepaelv2.tolist() == emb.tolist()   # não recomputado
    assert face.ff_info.fiqa_score == 0.8                             # preservado


def test_enrich_attributes_only_no_face_detected_skips(monkeypatch):
    monkeypatch.setattr("faceaudit.services.image_io.imread_unicode_safe",
                        lambda p: np.zeros((64, 64, 3), np.uint8))

    class _NoFace:
        app = type("E", (), {"detect_and_align": lambda self, img, single_face=True: None})()

    face = _face(yaw=None)
    n = FP._enrich_attributes_only([face], _FakeDS([face]), _NoFace(), _ALL_DA)
    assert n == 0 and face.ff_info.yaw is None   # nada mesclado, embedding intacto
    assert face.ff_info.embedding_sepaelv2 is not None


# ───────────────────────── resolução dos atributos pedidos ─────────────────────────
def test_resolve_wanted_attrs_kept_priority():
    cfg = DatasetIngestionConfig(input_path=".", dataset_name="d",
                                 ff_kept_attributes=["yaw", "fiqa"], ff_extended=True)
    assert FP._resolve_wanted_ff_attrs(cfg) == {"yaw", "fiqa"}


def test_resolve_wanted_attrs_extended_fallback():
    cfg_on = DatasetIngestionConfig(input_path=".", dataset_name="d",
                                    ff_kept_attributes=None, ff_extended=True)
    assert FP._resolve_wanted_ff_attrs(cfg_on) == {
        "fiqa", "yaw", "pitch", "roll", "gender", "age", "det_score", "ipd"}
    cfg_off = DatasetIngestionConfig(input_path=".", dataset_name="d",
                                     ff_kept_attributes=None, ff_extended=False)
    assert FP._resolve_wanted_ff_attrs(cfg_off) == set()
