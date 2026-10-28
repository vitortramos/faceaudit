"""Concordância imagem a imagem com a questionada — não só a mediana.

Caso real que motivou: a mediana de uma base adaptada acertou as 8
medidas da questionada enquanto só 5% das imagens, uma a uma, se
pareciam com ela. Aqui:
- `agreement` conta imagem a imagem;
- o Processar grava a concordância no manifesto e na cadeia;
- a avaliação na amostra (editor) faz exatamente a mesma conta;
- o botão "Medir na base" preenche a coluna da amostra.
"""

from __future__ import annotations

import json
import os

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
from faceaudit.models.image_processing.questioned_profile import QuestionedProfile
from faceaudit.services.audit_log_service import AuditLogService
from faceaudit.services.image_degradation_profile_service import (
    DegradationMeasures,
    agreement,
    compare_summary,
    measure,
    summarize,
)
from faceaudit.services.image_pipeline_batch_service import ImagePipelineBatchService
from faceaudit.services.image_pipeline_evaluation_service import (
    evaluate,
    load_sample,
    sample_indices,
)

FACE = ImageFaceGeometry(
    bbox=[20, 20, 80, 100],
    keypoints=[[35, 50], [65, 50], [50, 65], [38, 82], [62, 82]],
)


def _m(**kw):
    base = dict(width=100, height=100, region="face", is_grayscale=True, ipd_px=50.0,
                black_fraction=0.2, white_fraction=0.4, midtone_fraction=0.2,
                distinct_levels=170, grain=20000.0, mean_level=145.0)
    base.update(kw)
    return DegradationMeasures(**base)


def test_mediana_perfeita_com_imagens_que_nao_se_parecem():
    """Uma escura e uma clara: a mediana cai no alvo, nenhuma imagem cai."""
    alvo = _m()
    escura = _m(black_fraction=0.5, white_fraction=0.1, mean_level=90.0)
    clara = _m(black_fraction=0.0, white_fraction=0.7, mean_level=200.0)
    medidas = [escura, clara, escura, clara, alvo]
    resumo = summarize(medidas)
    linhas = {l["measure"]: l for l in compare_summary(alvo, resumo, medidas)}
    assert linhas["black_fraction"]["close"] is True          # pela mediana, ok
    assert linhas["black_fraction"]["images_close_fraction"] == pytest.approx(0.2)

    c = agreement(alvo, medidas)
    assert c["n_images"] == 5
    assert c["all_close_fraction"] == pytest.approx(0.2)
    assert c["per_measure"]["black_fraction"] == pytest.approx(0.2)
    assert c["per_measure"]["ipd_px"] == 1.0
    assert sum(c["close_count_histogram"].values()) == 5
    assert agreement(None, medidas) is None and agreement(alvo, []) is None


def test_medida_que_a_questionada_nao_tem_fica_de_fora():
    alvo = _m(ipd_px=None)
    c = agreement(alvo, [_m()])
    assert "ipd_px" not in c["measures"] and c["all_close_fraction"] == 1.0


@pytest.fixture
def dataset(tmp_path):
    pasta = tmp_path / "Documents"
    (pasta / "images").mkdir(parents=True)
    faces = []
    for i, nivel in enumerate([40, 90, 130, 170, 220, 120]):
        nome = f"ID00{i}.png"
        rng = np.random.default_rng(i)
        img = np.clip(rng.normal(nivel, 30, (140, 120, 3)), 0, 255).astype(np.uint8)
        cv2.imencode(".png", img)[1].tofile(str(pasta / "images" / nome))
        ff = FFInfo()
        ff.bbox = np.asarray(FACE.bbox, dtype=np.float64)
        ff.keypoints = np.asarray(FACE.keypoints, dtype=np.float64)
        faces.append(FaceInfo(image_path=nome, identity=f"ID00{i}", ff_info=ff))
    ds = FaceDataset(name="Documents", faces=faces, path_mode=PathMode.RELATIVE,
                     path_root="images")
    ds._parquet_dir = pasta
    return ds


def _config():
    cfg = ImageProcessingPipelineConfig.from_dict({"name": "xerox", "seed": 3, "steps": [
        {"step_type": "grayscale"},
        {"step_type": "tone_curve", "black_point": 90, "white_point": 160},
        {"step_type": "halftone"},
        {"step_type": "resize", "mode": "target_ipd", "target_ipd_px": 20},
    ]})
    cfg.questioned = QuestionedProfile(
        file_name="Q1.png", source_path="", sha256="ab" * 32, measured_at="",
        face=FACE, face_source="detector",
        measures=_m(ipd_px=20.0, distinct_levels=200, grain=15000.0))
    return cfg


def test_processar_grava_a_concordancia(dataset, tmp_path):
    r = ImagePipelineBatchService().run(dataset, _config(), tmp_path / "saida",
                                        dataset_name="adaptado")
    conc = r.degradation["agreement"]
    assert conc["n_images"] == 6
    assert all(l["images_close_fraction"] is not None for l in r.degradation["comparison"])
    manifesto = json.loads(r.manifest_path.read_text(encoding="utf-8"))
    assert manifesto["degradation_measures"]["agreement"] == conc
    (entrada,) = AuditLogService.load(r.parquet_path)
    assert entrada.metadata["degradation_agreement"] == conc


def test_amostra_do_editor_faz_a_mesma_conta_que_o_processar(dataset, tmp_path):
    cfg = _config()
    r = ImagePipelineBatchService().run(dataset, cfg, tmp_path / "saida", dataset_name="a")
    amostra = load_sample(dataset, cfg.input_source, n=len(dataset))
    av = evaluate(amostra, cfg)
    assert av.agreement == r.degradation["agreement"]
    assert av.comparison == r.degradation["comparison"]
    assert av.n_failed == 0


def test_amostra_cobre_a_base_inteira_e_conta_falhas(dataset):
    assert sample_indices(271, 40)[0] == 0 and sample_indices(271, 40)[-1] == 270
    assert len(sample_indices(271, 40)) == 40
    assert sample_indices(5, 40) == [0, 1, 2, 3, 4]
    cfg = _config()
    amostra = load_sample(dataset, cfg.input_source, n=3)
    sem_face = [a.__class__(a.image, None, a.jitter, a.file_name) for a in amostra]
    av = evaluate(sem_face, cfg)  # Redimensionar por IPD sem face → falha
    assert av.n_failed == 3 and av.n_measured == 0 and av.agreement is None


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_botao_medir_na_base_preenche_a_coluna_da_amostra(app, dataset):
    from PyQt6 import QtCore, QtTest
    from interface.tabs.dataset_manager.image_pipeline_dialog import (
        ImagePipelineDialogController,
    )

    c = ImagePipelineDialogController(dataset, initial_face_index=0)
    cfg = _config()
    c._pipeline_config.steps = cfg.steps
    c.apply_questioned(cfg.questioned, None)
    painel = c.view.degradation_measures

    c._on_sample_requested()
    for _ in range(200):
        if painel._amostra is not None:
            break
        QtTest.QTest.qWait(25)
    assert painel._amostra is not None and painel._amostra.n_measured == 6
    assert "amostra de 6 imagens" in painel.lbl_status.text()
    assert "%" in painel.table.item(1, 4).text()

    # Mudou o pipeline: a coluna fica marcada como desatualizada.
    c._on_step_config_changed(c._pipeline_config.steps[0])
    assert painel._amostra_desatualizada
    assert "meça de novo" in painel.lbl_status.text()
    # Segunda medição reaproveita a amostra carregada.
    assert c._sample_cache is not None
    for _ in range(40):
        QtTest.QTest.qWait(25)
        QtCore.QCoreApplication.processEvents()
