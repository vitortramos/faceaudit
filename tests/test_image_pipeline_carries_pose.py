"""O Processar (pipeline de imagens) leva a pose da cabeça para o derivado.

A cabeça não gira quando a imagem é degradada, e nenhum passo do
pipeline gira ou espelha a imagem: yaw, pitch e roll continuam
verdadeiros. As demais medidas do FF (IPD, idade e gênero estimados,
det_score, geometria, embeddings) foram medidas na imagem de origem e
ficam de fora.

O derivado sai com um FFInfo só com a pose, e isso não pode enganar o
enriquecimento: tratado como "já extraído", ele iria pelo caminho que
só calcula embedding e o derivado ficaria sem IPD, idade e det_score
sem erro nenhum. E a pose herdada prevalece sobre a estimada na imagem
degradada.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.ff_service import FFService, _tem_extracao_previa
from faceaudit.services.image_pipeline_batch_service import (
    POSE_LEVADA,
    ImagePipelineBatchService,
)


@pytest.fixture
def origem(tmp_path):
    pasta = tmp_path / "Documents"
    (pasta / "images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    faces = []
    for i in range(3):
        nome = f"ID00{i}_CNH.png"
        cv2.imencode(".png", rng.integers(0, 256, (60, 50, 3), np.uint8))[1].tofile(
            str(pasta / "images" / nome))
        ff = FFInfo(ipd=40.0 + i, det_score=0.9, age=30.0 + i, gender="M",
                    yaw=-10.0 + i, pitch=5.0, roll=1.5 * i, fiqa_score=0.8,
                    embedding_sepaelv2=rng.standard_normal(512).astype(np.float32))
        faces.append(FaceInfo(image_path=nome, identity=f"ID00{i}", gender="M",
                              ff_info=ff, extra_data={"idade": 30 + i}))
    ds = FaceDataset(name="Documents", faces=faces, path_mode=PathMode.RELATIVE,
                     path_root="images")
    ds._parquet_dir = pasta
    parquet = pasta / "Documents.parquet"
    ds.to_parquet(parquet)
    return FaceDataset.from_parquet(str(parquet)), parquet


def _processar(origem, destino):
    ds, parquet = origem
    cfg = ImageProcessingPipelineConfig.from_dict({"name": "xerox", "steps": [
        {"step_type": "grayscale"},
        {"step_type": "halftone", "mode": "error_diffusion"},
        {"step_type": "resize", "mode": "factor", "factor": 0.5},
    ]})
    return ImagePipelineBatchService().run(ds, cfg, destino, dataset_name="adaptado",
                                           source_parquet_path=parquet)


def test_pose_acompanha_e_o_resto_do_ff_nao(origem, tmp_path):
    r = _processar(origem, tmp_path / "saida")
    derivado = FaceDataset.from_parquet(str(r.parquet_path))
    origem_por_id = {f.identity: f for f in origem[0].faces}
    for f in derivado.faces:
        o = origem_por_id[f.identity].ff_info
        assert (f.ff_info.yaw, f.ff_info.pitch, f.ff_info.roll) == pytest.approx(
            (o.yaw, o.pitch, o.roll))
        for medida in ("ipd", "det_score", "age", "gender", "bbox", "keypoints"):
            assert getattr(f.ff_info, medida) is None, medida
        assert f.gender == "M" and f.extra_data["idade"] == o.age  # declarados ficam

    import pyarrow.parquet as pq
    colunas = pq.read_schema(r.parquet_path).names
    assert {"yaw", "pitch", "roll"} <= set(colunas)
    assert not {"ipd", "det_score", "age_est", "fiqa_score", "embedding_v2"} & set(colunas)

    # Abrir → salvar não pode apagar a pose (o FFInfo só com pose tem de
    # sobreviver à leitura).
    derivado.to_parquet(r.parquet_path)
    relido = FaceDataset.from_parquet(str(r.parquet_path))
    assert all(f.ff_info is not None and f.ff_info.yaw is not None for f in relido.faces)


def test_manifesto_e_custodia_dizem_que_a_pose_veio_da_origem(origem, tmp_path):
    r = _processar(origem, tmp_path / "saida")
    manifesto = json.loads(r.manifest_path.read_text(encoding="utf-8"))
    assert manifesto["pose_carried"] == list(POSE_LEVADA)
    (entrada,) = AuditLogService.load(r.parquet_path)
    assert entrada.metadata["pose_carried"] == list(POSE_LEVADA)


def test_ff_so_com_pose_nao_e_extracao_previa():
    assert not _tem_extracao_previa(None)
    assert not _tem_extracao_previa(FFInfo(yaw=1.0, pitch=2.0, roll=3.0))
    assert _tem_extracao_previa(FFInfo(yaw=1.0, det_score=0.9))
    assert _tem_extracao_previa(FFInfo(embedding_sepaelv2=np.ones(512)))


class _MotorFalso:
    """Engine do ForensicFace falso: detecta sempre, com pose diferente
    da herdada (como faria numa imagem degradada)."""

    def process_image(self, _img):
        return {"ipd": 25.0, "det_score": 0.7, "age": 41.0, "gender": "F",
                "yaw": 33.0, "pitch": -20.0, "roll": 9.0,
                "embedding": np.ones(512, np.float32)}


def _ff_falso():
    svc = FFService.__new__(FFService)
    svc._default_model = "sepaelv2"
    svc._resolve_engine = lambda m: ("sepaelv2", _MotorFalso())
    svc._slot_for_model = lambda m: "embedding_sepaelv2"
    svc._safe_read_image = lambda p: np.zeros((40, 40, 3), np.uint8)
    svc._normalize_process_image_result = lambda r: r
    svc._detectar_tentando_rotacoes = (
        lambda engine, img, rot, path, cb: (engine.process_image(img), 0))
    return svc


def test_enriquecer_derivado_calcula_o_resto_e_mantem_a_pose_herdada():
    face = FaceInfo(image_path="x.png", ff_info=FFInfo(yaw=-10.0, pitch=5.0, roll=1.0))
    _ff_falso().process_face(face)
    ff = face.ff_info
    assert (ff.yaw, ff.pitch, ff.roll) == (-10.0, 5.0, 1.0)  # herdada prevalece
    assert ff.ipd == 25.0 and ff.det_score == 0.7 and ff.age == 41.0
    assert ff.embedding_sepaelv2 is not None


def test_reextracao_normal_continua_atualizando_a_pose():
    face = FaceInfo(image_path="x.png",
                    ff_info=FFInfo(yaw=-10.0, pitch=5.0, roll=1.0, det_score=0.9))
    _ff_falso().process_face(face)
    assert face.ff_info.yaw == 33.0
