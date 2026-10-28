"""Pipeline de imagens, Fase 3: a questionada como alvo.

- As medidas são feitas na região da face e acusam quando não há face.
- O perfil da questionada (nome, SHA-256, medidas) vive no pipeline e
  sobrevive ao JSON.
- O Processar mede todas as imagens adaptadas e grava, no manifesto e na
  cadeia de custódia, a comparação com a questionada.
- O editor mostra a questionada e as três colunas de medidas.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.image_processing.image_face_geometry import ImageFaceGeometry
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.image_degradation_profile_service import (
    MEASURES_BY_KEY,
    REGIAO_FACE,
    REGIAO_IMAGEM,
    build_questioned_profile,
    compare,
    measure,
)
from faceaudit.services.image_pipeline_batch_service import ImagePipelineBatchService

FACE = ImageFaceGeometry(
    bbox=[20, 20, 80, 100],
    keypoints=[[35, 50], [65, 50], [50, 65], [38, 82], [62, 82]],
)


def _imagem_metade_preta():
    """Fundo branco; dentro da caixa da face, metade preta e metade cinza."""
    img = np.full((140, 120, 3), 255, np.uint8)
    img[20:100, 20:50] = 0
    img[20:100, 50:80] = 128
    return img


class _DetectorFalso:
    """FFService falso: devolve a face FACE para qualquer imagem."""

    class app:  # noqa: N801
        class backend:  # noqa: N801
            @staticmethod
            def detect_faces(_img):
                class F:
                    bbox = np.asarray(FACE.bbox)
                    kps = np.asarray(FACE.keypoints)
                return [F()]


# ── Medidas ────────────────────────────────────────────────────────────────

def test_medidas_na_regiao_da_face():
    m = measure(_imagem_metade_preta(), FACE)
    assert m.region == REGIAO_FACE
    assert m.black_fraction == pytest.approx(0.5)
    assert m.midtone_fraction == pytest.approx(0.5)
    assert m.white_fraction == 0.0  # o fundo branco ficou de fora
    assert m.ipd_px == pytest.approx(30.0)
    assert m.is_grayscale and m.distinct_levels == 2


def test_sem_face_mede_a_imagem_inteira_e_diz_isso():
    m = measure(_imagem_metade_preta())
    assert m.region == REGIAO_IMAGEM and m.ipd_px is None
    assert m.white_fraction > 0.5


def test_comparacao_usa_a_tolerancia_de_cada_medida():
    alvo = measure(_imagem_metade_preta(), FACE)
    perto = alvo.__class__(**{**alvo.to_dict(), "black_fraction": alvo.black_fraction + 0.04,
                              "grain": alvo.grain * 1.2, "ipd_px": 32.0})
    longe = alvo.__class__(**{**alvo.to_dict(), "black_fraction": alvo.black_fraction + 0.2,
                              "grain": alvo.grain * 3 + 1, "ipd_px": 40.0})
    por = lambda linhas: {l.key: l.close for l in linhas}  # noqa: E731
    assert por(compare(alvo, perto))["black_fraction"] is True
    assert por(compare(alvo, perto))["ipd_px"] is True
    assert por(compare(alvo, longe))["black_fraction"] is False
    assert por(compare(alvo, longe))["grain"] is False
    assert por(compare(alvo, longe))["ipd_px"] is False
    assert por(compare(None, perto))["black_fraction"] is None
    assert MEASURES_BY_KEY["black_fraction"].format(0.1875) == "18,8%"


# ── Perfil ─────────────────────────────────────────────────────────────────

def test_perfil_da_questionada_e_gravado_no_pipeline(tmp_path):
    q = tmp_path / "Q1.png"
    cv2.imencode(".png", _imagem_metade_preta())[1].tofile(str(q))
    perfil, img = build_questioned_profile(q, _DetectorFalso())
    assert perfil.sha256 == hashlib.sha256(q.read_bytes()).hexdigest()
    assert perfil.face_source == "detector" and perfil.measures.region == REGIAO_FACE
    assert img.shape == (140, 120, 3)

    cfg = ImageProcessingPipelineConfig(name="x", questioned=perfil)
    relido = ImageProcessingPipelineConfig.from_json_string(cfg.to_json_string())
    assert relido.questioned.to_dict() == perfil.to_dict()
    assert "questioned" not in ImageProcessingPipelineConfig(name="y").to_dict()


def test_perfil_sem_detector_mede_a_imagem_inteira(tmp_path):
    q = tmp_path / "Q1.png"
    cv2.imencode(".png", _imagem_metade_preta())[1].tofile(str(q))
    perfil, _ = build_questioned_profile(q, None)
    assert perfil.face is None and perfil.face_source is None
    assert perfil.measures.region == REGIAO_IMAGEM


# ── Processar ──────────────────────────────────────────────────────────────

@pytest.fixture
def dataset(tmp_path):
    pasta = tmp_path / "Documents"
    (pasta / "images").mkdir(parents=True)
    faces = []
    rng = np.random.default_rng(0)
    for i in range(4):
        nome = f"ID00{i}_CNH.png"
        img = cv2.GaussianBlur(rng.integers(0, 256, (140, 120, 3), dtype=np.uint8), (0, 0), 3)
        cv2.imencode(".png", img)[1].tofile(str(pasta / "images" / nome))
        ff = FFInfo()
        ff.bbox = np.asarray(FACE.bbox, dtype=np.float64)
        ff.keypoints = np.asarray(FACE.keypoints, dtype=np.float64)
        faces.append(FaceInfo(image_path=nome, identity=f"ID00{i}", ff_info=ff))
    # Em memória: o parquet só grava a geometria junto com os embeddings,
    # e este dataset de teste não tem embedding.
    ds = FaceDataset(name="Documents", faces=faces, path_mode=PathMode.RELATIVE,
                     path_root="images")
    ds._parquet_dir = pasta
    return ds


def _xerox(perfil=None):
    cfg = ImageProcessingPipelineConfig.from_dict({"name": "xerox", "steps": [
        {"step_type": "grayscale"},
        {"step_type": "tone_curve", "black_point": 90, "white_point": 160},
        {"step_type": "halftone", "mode": "error_diffusion"},
        {"step_type": "resize", "mode": "target_ipd", "target_ipd_px": 20},
    ]})
    cfg.questioned = perfil
    return cfg


def test_processar_mede_e_compara_com_a_questionada(dataset, tmp_path):
    q = tmp_path / "Q1.png"
    cv2.imencode(".png", _imagem_metade_preta())[1].tofile(str(q))
    perfil, _ = build_questioned_profile(q, _DetectorFalso())

    r = ImagePipelineBatchService().run(dataset, _xerox(perfil), tmp_path / "saida",
                                        dataset_name="adaptado")
    deg = r.degradation
    assert deg["n_measured"] == 4
    assert deg["summary"]["regions"] == {REGIAO_FACE: 4}
    assert deg["summary"]["ipd_px"]["median"] == pytest.approx(20, abs=0.5)
    linhas = {l["measure"]: l for l in deg["comparison"]}
    assert linhas["ipd_px"]["questioned"] == pytest.approx(30.0)
    assert linhas["ipd_px"]["close"] is False  # 20 px contra 30 px
    assert linhas["is_grayscale"]["close"] is True

    manifesto = json.loads(r.manifest_path.read_text(encoding="utf-8"))
    assert manifesto["degradation_measures"]["questioned"]["sha256"] == perfil.sha256
    (entrada,) = AuditLogService.load(r.parquet_path)
    assert entrada.metadata["questioned_sha256"] == perfil.sha256
    assert entrada.metadata["degradation_comparison"] == deg["comparison"]


def test_processar_sem_questionada_ainda_mede_o_conjunto(dataset, tmp_path):
    r = ImagePipelineBatchService().run(dataset, _xerox(), tmp_path / "saida",
                                        dataset_name="adaptado")
    assert r.degradation["questioned"] is None
    assert r.degradation["summary"]["black_fraction"]["n"] == 4
    assert all(l["close"] is None for l in r.degradation["comparison"])
    (entrada,) = AuditLogService.load(r.parquet_path)
    assert "questioned_sha256" not in entrada.metadata


def test_reprocessar_na_mesma_pasta_ainda_mede(dataset, tmp_path):
    ImagePipelineBatchService().run(dataset, _xerox(), tmp_path / "saida", dataset_name="a")
    r = ImagePipelineBatchService().run(dataset, _xerox(), tmp_path / "saida", dataset_name="a")
    assert r.n_skipped == 4 and r.degradation["n_measured"] == 4


# ── Editor ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_editor_mostra_a_questionada_e_as_tres_colunas(app, dataset, tmp_path):
    from interface.tabs.dataset_manager.image_pipeline_dialog import (
        ImagePipelineDialogController,
    )
    from faceaudit.factories.image_processor_factory import ImageProcessorFactory

    q = tmp_path / "Q1.png"
    cv2.imencode(".png", _imagem_metade_preta())[1].tofile(str(q))
    perfil, img = build_questioned_profile(q, _DetectorFalso())

    c = ImagePipelineDialogController(dataset, initial_face_index=0)
    painel = c.view.degradation_measures
    assert painel._original is not None and painel._original.region == REGIAO_FACE
    assert c.view.image_compare.pane_questioned.isHidden()

    c.apply_questioned(perfil, img)
    assert not c.view.image_compare.pane_questioned.isHidden()
    assert "Q1.png" in c.view.lbl_questioned_info.text()
    assert c._is_dirty

    c._pipeline_config.steps = _xerox().steps
    c._on_apply()
    assert painel._processada is not None
    assert painel._processada.ipd_px == pytest.approx(20, abs=0.5)
    assert "medidas dentro da tolerância" in painel.lbl_status.text()

    # Pipeline salvo e reaberto: a questionada volta pelo caminho gravado.
    salvo = ImageProcessingPipelineConfig.from_json_string(c._pipeline_config.to_json_string())
    c._pipeline_config = salvo
    c._reload_questioned_from_config()
    assert c._questioned_image is not None and c._questioned_warning is None

    # Arquivo alterado depois: as medidas gravadas continuam, com aviso.
    cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tofile(str(q))
    c._reload_questioned_from_config()
    assert c._questioned_image is None
    assert "SHA-256 diferente" in c.view.lbl_questioned_info.text()

    c._on_questioned_clear()
    assert c._pipeline_config.questioned is None
    assert c.view.image_compare.pane_questioned.isHidden()
