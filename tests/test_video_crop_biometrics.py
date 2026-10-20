"""Dataset de vídeo em modo "Face recortada": biometria da extração, sem redetectar.

O caso que motivou: importar o run com o recorte com margem como imagem do
dataset fazia o enriquecimento detectar a face DE NOVO dentro do recorte. Em
face pequena de vigilância (~50 px) a confiança do detector caía ~0,25–0,30
e a face ficava abaixo do limiar — 572 de 1024 faces saíram sem embedding,
com `error=False`; e parte das restantes ganhou embedding de outra região.

Aqui se verifica, sem modelo nenhum:
  - a geometria gravada é a da extração, no sistema do RECORTE (conferida
    contra os pixels do quadro);
  - runs antigos, sem `crop_box_xyxy`, chegam à mesma origem;
  - o enriquecimento recebe a face alinhada da extração como cache;
  - o FFService manda as faces com cache pelo caminho sem detecção mesmo
    quando só parte do dataset tem cache;
  - o pipeline marca erro na face que sai sem o embedding pedido, ainda que
    ela já tivesse FFInfo.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from forensicface.geometry import extend_bbox

from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.video.face_provenance import FaceProvenance
from faceaudit.services.hash_service import HashService
from faceaudit.services.image_io import imwrite_unicode_safe
from faceaudit.services.video_run_importer import VideoRunImporter

MARGEM = 2.0
BBOX = [101.7, 62.4, 131.2, 99.9]
KPS = [[110.5, 75.25], [122.0, 76.0], [116.0, 84.5], [111.0, 92.0], [121.5, 92.75]]


def _quadro():
    quadro = np.random.RandomState(0).randint(0, 200, (240, 320, 3), dtype=np.uint8)
    # Marca cada ponto com uma cor que não existe no ruído (canal B = 255),
    # para conferir depois que o ponto transladado cai sobre a marca.
    for x, y in KPS:
        quadro[int(y), int(x)] = (255, 255, 255)
    return quadro


def _escrever_run(tmp_path, *, gravar_caixa=True, alinhada_adulterada=False):
    rd = tmp_path / "run"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    hs = HashService()
    quadro = _quadro()
    x1, y1, x2, y2 = extend_bbox(np.asarray(BBOX), quadro.shape, margin_factor=MARGEM)
    recorte = quadro[y1:y2, x1:x2]
    nome = "alvo_v00_f0000007.png"
    imwrite_unicode_safe(rd / "alvo" / nome, recorte)
    alinhada = np.random.RandomState(1).randint(0, 255, (112, 112, 3), dtype=np.uint8)
    caminho_alinhada = rd / "_aligned" / "alvo" / nome
    imwrite_unicode_safe(caminho_alinhada, alinhada)
    sha_alinhada = hs.calculate_file_hash(caminho_alinhada)
    if alinhada_adulterada:
        imwrite_unicode_safe(caminho_alinhada, 255 - alinhada)
    prov = FaceProvenance(
        run_id="r1", run_manifest="_run_r1.json", identity_label="alvo",
        image_file=nome, image_sha256=hs.calculate_file_hash(rd / "alvo" / nome),
        image_width_px=recorte.shape[1], image_height_px=recorte.shape[0],
        crop_margin_factor=MARGEM,
        crop_box_xyxy=[x1, y1, x2, y2] if gravar_caixa else None,
        aligned_image_file=f"_aligned/alvo/{nome}", aligned_image_sha256=sha_alinhada,
        frame_native_index=7, bbox_xyxy=list(BBOX), keypoints_5pt=KPS,
        aligned_keypoints=[[38.3, 51.7], [73.5, 51.5], [56.0, 71.7], [41.5, 92.4], [70.7, 92.2]],
        det_score=0.55, fiqa_score=0.9, similarity_to_anchor=0.6,
        auto_decision="accepted", human_review_status="accepted",
    )
    (meta / prov.sidecar_filename()).write_text(
        json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    (meta / "_run_r1.json").write_text(json.dumps(
        {"run_id": "r1", "recognition_model": "sepaelv2", "review_status": "completed",
         "targets": [{"label": "alvo", "similarity_threshold": 0.3}]}), encoding="utf-8")
    return rd, (x1, y1), recorte, cv2.cvtColor(alinhada, cv2.COLOR_BGR2RGB)


# ── Geometria no sistema do recorte ──────────────────────────────────────

@pytest.mark.parametrize("gravar_caixa", [True, False], ids=["run_novo", "run_antigo"])
def test_geometria_da_extracao_no_sistema_do_recorte(tmp_path, gravar_caixa):
    rd, (ox, oy), recorte, _ = _escrever_run(tmp_path, gravar_caixa=gravar_caixa)
    ds = VideoRunImporter().import_run(rd, image_kind="crop")
    ff = ds.faces[0].ff_info

    esperado = np.asarray(BBOX, np.float32) - np.asarray([ox, oy, ox, oy], np.float32)
    np.testing.assert_allclose(ff.bbox, esperado, atol=1e-4)
    kps = np.asarray(ff.keypoints).reshape(5, 2)
    np.testing.assert_allclose(kps, np.asarray(KPS) - [ox, oy], atol=1e-4)
    # Contra os PIXELS: cada ponto transladado cai sobre a marca branca que
    # foi pintada no quadro antes de recortar.
    for x, y in kps:
        assert tuple(recorte[int(y), int(x)]) == (255, 255, 255)
    assert ds.provenance_metadata.get("geometry_source") == "video_extraction_detection"


def test_modo_alinhado_nao_grava_geometria(tmp_path):
    rd, *_ = _escrever_run(tmp_path)
    ds = VideoRunImporter().import_run(rd, image_kind="aligned")
    ff = ds.faces[0].ff_info
    assert ff.bbox is None and ff.keypoints is None
    assert "geometry_source" not in ds.provenance_metadata


def test_geometria_sobrevive_ao_parquet(tmp_path):
    rd, (ox, oy), *_ = _escrever_run(tmp_path)
    ds = VideoRunImporter().import_run(rd, image_kind="crop")
    destino = rd / "ds.parquet"
    ds.to_parquet(destino)
    relido = FaceDataset.from_parquet(destino)
    np.testing.assert_allclose(
        relido.faces[0].ff_info.bbox,
        np.asarray(BBOX, np.float32) - [ox, oy, ox, oy], atol=1e-4)
    assert relido.provenance_metadata.get("geometry_source") == "video_extraction_detection"


# ── Biometria da face alinhada da extração ───────────────────────────────

class _PipelineEspiao:
    """Registra o que o enriquecimento recebeu, sem rodar modelo."""

    def __init__(self):
        self.recebido = []

    def enrich_existing_dataset(self, ds, cfg, progress_callback=None):
        self.recebido = [
            None if f.ff_info.aligned_face is None else f.ff_info.aligned_face.copy()
            for f in ds.faces
        ]
        self.is_aligned = cfg.is_aligned
        return ds


def test_enriquecimento_recebe_a_face_alinhada_da_extracao(tmp_path):
    rd, _, _, alinhada_rgb = _escrever_run(tmp_path)
    espiao = _PipelineEspiao()
    ds = VideoRunImporter().import_run(
        rd, image_kind="crop", enrich=True, pipeline=espiao,
        ingestion_config=DatasetIngestionConfig(input_path=rd, dataset_name="x"),
    )
    assert len(espiao.recebido) == 1
    np.testing.assert_array_equal(espiao.recebido[0], alinhada_rgb)
    # O dataset continua sendo de recortes — a imagem é a de evidência.
    assert espiao.is_aligned is False
    assert ds.faces[0].image_path.startswith("alvo/")
    # Cache liberado depois do uso e origem declarada.
    assert ds.faces[0].ff_info.aligned_face is None
    assert ds.provenance_metadata["embedding_source"] == "video_extraction_aligned_face"


def test_face_alinhada_adulterada_nao_entra(tmp_path):
    rd, *_ = _escrever_run(tmp_path, alinhada_adulterada=True)
    ds = VideoRunImporter().import_run(
        rd, image_kind="crop", enrich=True, pipeline=_PipelineEspiao(),
        ingestion_config=DatasetIngestionConfig(input_path=rd, dataset_name="x"),
    )
    assert len(ds) == 0


# ── FFService: partição cache × detecção ─────────────────────────────────

def _ff_service_espiao():
    from faceaudit.services.ff_service import FFService
    svc = object.__new__(FFService)   # sem carregar modelo nenhum
    svc._batch_size = 4
    svc.chamadas = {"cache": [], "align": []}
    svc._process_dataset_cached_aligned_batched = (
        lambda ds, n, *a: svc.chamadas["cache"].append([f.file_name for f in ds]))
    svc._process_dataset_align_then_batch = (
        lambda ds, n, *a, **k: svc.chamadas["align"].append([f.file_name for f in ds]))
    svc._apply_identity_gender_consensus = lambda ds: None
    return svc


def _face_com_ff(nome, *, cache):
    ff = FFInfo(det_score=0.5)
    if cache:
        ff.aligned_face = np.zeros((112, 112, 3), np.uint8)
    return FaceInfo(image_path=nome, file_name=nome, identity="A", hash=nome, ff_info=ff)


def test_faces_com_cache_nao_vao_para_a_redeteccao_mesmo_em_dataset_misto():
    svc = _ff_service_espiao()
    ds = FaceDataset(name="d", faces=[
        _face_com_ff("a", cache=True), _face_com_ff("b", cache=False),
        _face_com_ff("c", cache=True),
    ])
    svc.process_dataset(ds, models=["sepaelv2"])
    assert svc.chamadas["cache"] == [["a", "c"]]
    assert svc.chamadas["align"] == [["b"]]


def test_todas_com_cache_vao_inteiras_pelo_cache():
    svc = _ff_service_espiao()
    ds = FaceDataset(name="d", faces=[_face_com_ff("a", cache=True), _face_com_ff("b", cache=True)])
    svc.process_dataset(ds, models=["sepaelv2"])
    assert svc.chamadas["cache"] == [["a", "b"]]
    assert svc.chamadas["align"] == []


# ── Pipeline: falha marcada mesmo com FFInfo preexistente ────────────────

def test_face_com_ffinfo_que_sai_sem_embedding_e_marcada_como_erro(tmp_path):
    from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline

    ff_service = MagicMock()
    ff_service.process_dataset.side_effect = lambda ds, **kw: ds  # não produz nada
    pipe = FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=ff_service,
        ofiq_service=MagicMock(), serfiq_service=MagicMock(),
        storage_path=Path(tmp_path),
    )
    # FFInfo já existe (como no dataset de vídeo, vindo do sidecar).
    face = FaceInfo(image_path="a.png", file_name="a.png", identity="A", hash="h",
                    ff_info=FFInfo(det_score=0.6, yaw=1.0))
    ds = FaceDataset(name="video", faces=[face])
    ds._parquet_dir = Path(tmp_path)
    pipe.enrich_existing_dataset(ds, DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="video", compute_ff=True,
        compute_ofiq=False, compute_serfiq=False, ff_models=["sepaelv2"],
    ))
    assert face.error is True
    assert face.error_stage == "ff"
    assert "sepaelv2" in face.error_message
