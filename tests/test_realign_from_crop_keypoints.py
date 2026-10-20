"""Realinhar a partir do recorte + 5 pontos, sem detectar de novo.

Contexto: acrescentar um modelo a um dataset de recortes de vídeo detectava
a face de novo no recorte — o que falha em face pequena. Guardar a face
alinhada no dataset dobraria o tamanho das imagens. A alternativa é
refazer o alinhamento pelos pontos da extração, que o dataset já guarda.

Medido num run real (1024 faces): com a margem 2,0 a janela do gabarito
(um quadrado GIRADO) escapava do recorte em 232 faces de cabeça inclinada,
e a parte de fora virava preto — pior cosseno 0,788 com o embedding da
extração. Daí as duas metades desta mudança, cobertas aqui:
  - a extração faz o recorte cobrir a janela inteira;
  - o enriquecimento realinha pelos pontos e mede, por face, quanto da
    janela ficou sem pixel (o que diz se o realinhamento reproduz o
    alinhamento original).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from forensicface.geometry import extend_bbox
from forensicface.insightface import face_align

from faceaudit.models.dataset_ingestion_config import DatasetIngestionConfig
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.services.face_alignment_window_service import (
    ALIGNMENT_MISSING_PIXELS_COLUMN,
    caixa_do_recorte,
    fracao_sem_pixel,
    janela_de_alinhamento,
    realinhar,
)
from faceaudit.services.image_io import imwrite_unicode_safe

MARGEM = 2.0


def _pontos_inclinados(cx=160.0, cy=120.0, escala=0.45, graus=35.0):
    """5 pontos do gabarito, girados e escalados — cabeça inclinada."""
    ref = face_align.arcface_dst.astype(np.float64) - [56.0, 71.7]
    t = np.deg2rad(graus)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (ref @ rot.T) * escala + [cx, cy]


def _caixa_da_face(pontos):
    x1, y1 = pontos.min(axis=0) - 6
    x2, y2 = pontos.max(axis=0) + 6
    return np.array([x1, y1, x2, y2])


def _quadro(h=240, w=320):
    # Gradiente + ruído: preto (o preenchimento da borda) não aparece por acaso.
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], axis=2)
    return np.clip(base + rng.randint(20, 60, (h, w, 3)), 0, 255).astype(np.uint8)


def _recortar(quadro, caixa):
    x1, y1, x2, y2 = (int(v) for v in caixa)
    return quadro[y1:y2, x1:x2], (x1, y1)


# ── Geometria da janela ──────────────────────────────────────────────────

def test_realinhar_e_o_norm_crop_da_forensicface():
    quadro, pontos = _quadro(), _pontos_inclinados()
    alinhada, _ = realinhar(quadro, pontos)
    np.testing.assert_array_equal(alinhada, face_align.norm_crop(quadro, pontos))


def test_pontos_alinhados_caem_no_gabarito():
    _, alinhados = realinhar(_quadro(), _pontos_inclinados())
    np.testing.assert_allclose(alinhados, face_align.arcface_dst, atol=0.05)


def test_janela_contem_toda_amostra_do_alinhamento():
    pontos = _pontos_inclinados()
    x1, y1, x2, y2 = janela_de_alinhamento(pontos)
    inv = cv2.invertAffineTransform(face_align.estimate_norm(pontos))
    yy, xx = np.mgrid[0:112, 0:112]
    src = np.stack([xx.ravel(), yy.ravel()], 1) @ inv[:, :2].T + inv[:, 2]
    assert src[:, 0].min() >= x1 - 1e-6 and src[:, 0].max() <= x2 + 1e-6
    assert src[:, 1].min() >= y1 - 1e-6 and src[:, 1].max() <= y2 + 1e-6


def test_margem_antiga_nao_cobre_cabeca_inclinada_e_a_nova_cobre():
    quadro, pontos = _quadro(), _pontos_inclinados()
    margem = extend_bbox(_caixa_da_face(pontos), quadro.shape, MARGEM)
    nova = caixa_do_recorte(margem, pontos, quadro.shape[1], quadro.shape[0])

    for caixa, deve_faltar in ((margem, True), (nova, False)):
        recorte, (ox, oy) = _recortar(quadro, caixa)
        no_recorte = pontos - [ox, oy]
        falta = fracao_sem_pixel(no_recorte, recorte.shape[1], recorte.shape[0])
        refeita, _ = realinhar(recorte, no_recorte)
        original = face_align.norm_crop(quadro, pontos)
        dif = np.abs(refeita.astype(int) - original.astype(int))
        if deve_faltar:
            assert falta > 0.01            # o caso medido no run real
            assert dif.max() > 50          # preto onde o quadro tinha pixel
        else:
            assert falta == 0.0
            # Só arredondamento da interpolação (origem deslocada).
            assert dif.max() <= 2
            assert np.mean(dif > 0) < 0.05


def test_caixa_nova_contem_a_margem_pedida_e_respeita_o_quadro():
    quadro = _quadro()
    pontos = _pontos_inclinados(cx=20.0, cy=20.0)   # encostada no canto
    margem = extend_bbox(_caixa_da_face(pontos), quadro.shape, MARGEM)
    x1, y1, x2, y2 = caixa_do_recorte(margem, pontos, quadro.shape[1], quadro.shape[0])
    assert x1 <= margem[0] and y1 <= margem[1] and x2 >= margem[2] and y2 >= margem[3]
    assert x1 >= 0 and y1 >= 0 and x2 <= quadro.shape[1] and y2 <= quadro.shape[0]


def test_falta_alem_da_borda_do_quadro_nao_conta():
    """Recorte preso na borda: a extração também não tinha pixel ali."""
    quadro = _quadro()
    pontos = _pontos_inclinados(cx=20.0, cy=20.0)
    margem = extend_bbox(_caixa_da_face(pontos), quadro.shape, MARGEM)
    caixa = caixa_do_recorte(margem, pontos, quadro.shape[1], quadro.shape[0])
    recorte, (ox, oy) = _recortar(quadro, caixa)
    no_recorte = pontos - [ox, oy]
    h, w = recorte.shape[:2]
    limite_superior = fracao_sem_pixel(no_recorte, w, h)
    exata = fracao_sem_pixel(no_recorte, w, h,
                             existente=(-ox, -oy, quadro.shape[1] - ox, quadro.shape[0] - oy))
    assert limite_superior > 0     # sai do recorte...
    assert exata == 0.0            # ...mas só onde o quadro também acabava


def test_pontos_degenerados_sao_recusados():
    with pytest.raises((ValueError, RuntimeError)):
        realinhar(_quadro(), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        realinhar(_quadro(), np.full((5, 2), np.nan))


# ── Extração: o recorte gravado cobre a janela ───────────────────────────

def test_extracao_grava_recorte_que_cobre_a_janela(tmp_path):
    from tests.test_video_frame_extraction import (
        EMB_TARGET, _FakeEngine, _FakeFFService, _cfg, _face, _make_video, _target,
    )
    from faceaudit.services.video_frame_face_extraction_service import (
        VideoFrameFaceExtractionService,
    )
    video = tmp_path / "clip.avi"
    if not _make_video(video, size=(320, 240)):
        pytest.skip("cv2.VideoWriter MJPG indisponível neste ambiente")
    pontos = _pontos_inclinados()
    det = _face(0, _caixa_da_face(pontos), 0.9)
    det["keypoints"] = pontos
    svc = VideoFrameFaceExtractionService(
        _FakeFFService(_FakeEngine(lambda call: [det]), {0: (EMB_TARGET, 0.8)}))
    res = svc.extract(_cfg(video, tmp_path), [_target(threshold=0.5)])
    side = json.loads(next((Path(res.run_dir) / "_meta").glob("alvo_v*_f*.json"))
                      .read_text(encoding="utf-8"))
    x1, y1, x2, y2 = side["artifact"]["crop_box_xyxy"]
    jx1, jy1, jx2, jy2 = janela_de_alinhamento(pontos)
    assert x1 <= jx1 and y1 <= jy1 and x2 >= jx2 and y2 >= jy2
    margem = extend_bbox(_caixa_da_face(pontos), (240, 320, 3), MARGEM)
    assert x1 <= margem[0] and x2 >= margem[2]   # a margem pedida continua lá
    # E o recorte gravado tem exatamente o tamanho da caixa registrada.
    assert side["artifact"]["image_size_px"] == [x2 - x1, y2 - y1]


# ── Importação: a fração sem pixel por face ──────────────────────────────

def _run_com_margem_antiga(tmp_path, *, com_tamanho_do_quadro=True):
    from faceaudit.models.video.face_provenance import FaceProvenance
    from faceaudit.services.hash_service import HashService

    rd = tmp_path / "run"
    meta = rd / "_meta"
    meta.mkdir(parents=True)
    quadro, pontos = _quadro(), _pontos_inclinados()
    caixa = _caixa_da_face(pontos)
    recorte, _ = _recortar(quadro, extend_bbox(caixa, quadro.shape, MARGEM))
    imwrite_unicode_safe(rd / "alvo" / "alvo_v00_f0000001.png", recorte)
    prov = FaceProvenance(
        run_id="r", run_manifest="_run_r.json", identity_label="alvo",
        image_file="alvo_v00_f0000001.png",
        image_sha256=HashService().calculate_file_hash(rd / "alvo" / "alvo_v00_f0000001.png"),
        image_width_px=recorte.shape[1], image_height_px=recorte.shape[0],
        crop_margin_factor=MARGEM, source_video_sha256="vsha",
        frame_native_index=1, bbox_xyxy=caixa.tolist(), keypoints_5pt=pontos.tolist(),
        det_score=0.6, auto_decision="accepted", human_review_status="accepted",
    )
    (meta / prov.sidecar_filename()).write_text(json.dumps(prov.to_sidecar_dict()), encoding="utf-8")
    manifesto = {"run_id": "r", "recognition_model": "sepaelv2",
                 "targets": [{"label": "alvo", "similarity_threshold": 0.3}]}
    if com_tamanho_do_quadro:
        manifesto["source_videos"] = [{"sha256": "vsha", "resolution_effective": [320, 240]}]
    (meta / "_run_r.json").write_text(json.dumps(manifesto), encoding="utf-8")
    return rd


def test_importacao_registra_a_fracao_sem_pixel(tmp_path):
    from faceaudit.services.video_run_importer import VideoRunImporter
    ds = VideoRunImporter().import_run(_run_com_margem_antiga(tmp_path), image_kind="crop")
    falta = ds.faces[0].extra_data[ALIGNMENT_MISSING_PIXELS_COLUMN]
    assert falta > 0.01
    # Sobrevive ao parquet e tem rótulo no catálogo.
    destino = tmp_path / "ds.parquet"
    ds.to_parquet(destino)
    relido = FaceDataset.from_parquet(destino)
    assert relido.faces[0].extra_data[ALIGNMENT_MISSING_PIXELS_COLUMN] == pytest.approx(falta)
    assert ALIGNMENT_MISSING_PIXELS_COLUMN in relido.column_specs


# ── FFService: roteamento para o realinhamento ───────────────────────────

def _ff_service_espiao():
    from faceaudit.services.ff_service import FFService
    svc = object.__new__(FFService)
    svc._batch_size = 4
    svc._default_model = "sepaelv2"
    svc.chamadas = {"cache": [], "realinhar": [], "align": []}
    svc._process_dataset_cached_aligned_batched = (
        lambda ds, n, *a: svc.chamadas["cache"].append([f.file_name for f in ds]))
    svc._process_dataset_realign_from_keypoints = (
        lambda ds, n, *a: svc.chamadas["realinhar"].append([f.file_name for f in ds]))
    svc._process_dataset_align_then_batch = (
        lambda ds, n, *a, **k: svc.chamadas["align"].append([f.file_name for f in ds]))
    svc._apply_identity_gender_consensus = lambda ds: None
    return svc


def _face_info(nome, *, pontos=None, cache=False, image_path=None):
    ff = FFInfo(det_score=0.5)
    if pontos is not None:
        ff.keypoints = np.asarray(pontos, np.float32)
    if cache:
        ff.aligned_face = np.zeros((112, 112, 3), np.uint8)
    return FaceInfo(image_path=image_path or nome, file_name=nome, identity="A",
                    hash=nome, ff_info=ff)


def test_com_o_pedido_faces_com_pontos_vao_ao_realinhamento():
    svc = _ff_service_espiao()
    p = _pontos_inclinados()
    ds = FaceDataset(name="d", faces=[
        _face_info("a", pontos=p), _face_info("b"), _face_info("c", pontos=p, cache=True)])
    svc.process_dataset(ds, models=["sepaelv2"], realign_from_keypoints=True)
    assert svc.chamadas == {"cache": [["c"]], "realinhar": [["a"]], "align": [["b"]]}


def test_sem_o_pedido_os_pontos_nao_sao_usados():
    svc = _ff_service_espiao()
    ds = FaceDataset(name="d", faces=[_face_info("a", pontos=_pontos_inclinados())])
    svc.process_dataset(ds, models=["sepaelv2"])
    assert svc.chamadas["realinhar"] == [] and svc.chamadas["align"] == [["a"]]


def test_caminho_de_realinhamento_entrega_a_face_refeita_ao_modelo(tmp_path):
    from faceaudit.services.ff_service import FFService
    quadro, pontos = _quadro(), _pontos_inclinados()
    imwrite_unicode_safe(tmp_path / "a.png", quadro)
    svc = object.__new__(FFService)
    svc._batch_size = 4
    svc._default_model = "sepaelv2"
    recebido = {}

    def extrair(lote, *, models=None, aligned_keypoints_batch=None, error_callback=None):
        recebido["lote"], recebido["kps"] = lote.copy(), aligned_keypoints_batch.copy()
        return [FFInfo(fiqa_score=1.5, embedding_sepaelv2=np.ones(8, np.float32))
                for _ in range(len(lote))]
    svc.extract_embeddings_from_aligned_faces_batch = extrair
    face = _face_info("a.png", pontos=pontos, image_path=str(tmp_path / "a.png"))
    ds = FaceDataset(name="d", faces=[face])
    svc._process_dataset_realign_from_keypoints(ds, 1, None, ["sepaelv2"], None)

    # Os pontos guardados na face (float32), não os de precisão total.
    guardados = np.asarray(face.ff_info.keypoints, np.float64)
    np.testing.assert_array_equal(recebido["lote"][0], face_align.norm_crop(quadro, guardados))
    np.testing.assert_allclose(recebido["kps"][0], face_align.arcface_dst, atol=0.05)
    assert face.ff_info.fiqa_score == 1.5
    assert face.ff_info.det_score == 0.5      # o resto do FFInfo fica


# ── Pipeline: decide pela proveniência e registra a origem ───────────────

def _pipeline(tmp_path, ff_service):
    from faceaudit.pipelines.feature_extraction_pipeline import FeatureExtractionPipeline
    return FeatureExtractionPipeline(
        hash_service=MagicMock(), ff_service=ff_service, ofiq_service=MagicMock(),
        serfiq_service=MagicMock(), storage_path=Path(tmp_path))


@pytest.mark.parametrize("origem, espera", [
    ("video_extraction_detection", True),
    ("backfill_detector", False),     # detecção posterior: pode ser outra face
    (None, False),
])
def test_pipeline_so_realinha_pontos_da_extracao(tmp_path, origem, espera):
    quadro, pontos = _quadro(), _pontos_inclinados()
    imwrite_unicode_safe(tmp_path / "a.png", quadro)
    ff_service = MagicMock()
    ff_service.process_dataset.side_effect = lambda ds, **kw: ds
    face = _face_info("a.png", pontos=pontos, image_path=str(tmp_path / "a.png"))
    ds = FaceDataset(name="video", faces=[face])
    ds._parquet_dir = Path(tmp_path)
    if origem:
        ds.provenance_metadata["geometry_source"] = origem
    _pipeline(tmp_path, ff_service).enrich_existing_dataset(ds, DatasetIngestionConfig(
        input_path=tmp_path, dataset_name="video", compute_ff=True,
        compute_ofiq=False, compute_serfiq=False, ff_models=["sepaelv4"]))
    kwargs = ff_service.process_dataset.call_args.kwargs
    assert kwargs["realign_from_keypoints"] is espera
    if espera:
        fontes = json.loads(ds.provenance_metadata["embedding_sources_by_model"])
        assert fontes == {"sepaelv4": "realigned_from_crop_keypoints"}
        # Face importada antes da medida: recebe o limite superior aqui.
        assert ALIGNMENT_MISSING_PIXELS_COLUMN in face.extra_data


def test_record_embedding_source_acumula_por_modelo():
    ds = FaceDataset(name="d", faces=[])
    ds.record_embedding_source(["sepaelv2"], "video_extraction_aligned_face")
    ds.record_embedding_source(["sepaelv4"], "realigned_from_crop_keypoints")
    assert json.loads(ds.provenance_metadata["embedding_sources_by_model"]) == {
        "sepaelv2": "video_extraction_aligned_face",
        "sepaelv4": "realigned_from_crop_keypoints",
    }
