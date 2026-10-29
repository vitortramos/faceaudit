"""Curva de tons por percentis da face, IPD da questionada, ajuste
automático à questionada e o modelo "Xerox de documento".

O motivo de tudo: com curva fixa, a exposição de cada foto decidia se a
cópia saía escura ou clara demais (correlação 0,96 entre o nível médio
original e o processado num caso real), e só ~5% das imagens se pareciam
com a questionada. A curva por percentis absorve a exposição; o ajuste
procura os valores olhando as imagens uma a uma e confere em imagens que
não usou.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from faceaudit.enums.path_mode import PathMode
from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.image_processors.image_step_base import FaceGeometryRequiredError
from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ff_info import FFInfo
from faceaudit.models.image_processing.image_face_geometry import ImageFaceGeometry
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.models.image_processing.pipeline_templates import BUILTIN_TEMPLATES
from faceaudit.models.image_processing.questioned_profile import QuestionedProfile
from faceaudit.pipelines.image_processing_pipeline import ImageProcessingPipeline
from faceaudit.services.image_degradation_profile_service import measure
from faceaudit.services.image_pipeline_evaluation_service import load_sample
from faceaudit.services.image_pipeline_fit_service import (
    FitCancelled,
    fit_to_questioned,
    split_indices,
    tunable_params,
)

FACE = ImageFaceGeometry(
    bbox=[20, 20, 100, 120],
    keypoints=[[40, 55], [80, 55], [60, 75], [45, 100], [75, 100]],
)  # IPD 40


def _foto(nivel: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.normal(0, 1, (140, 120)), (0, 0), 4)
    base = (base - base.mean()) / (base.std() + 1e-9)
    g = np.clip(nivel + 35 * base, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _cfg(steps, questionada=None):
    c = ImageProcessingPipelineConfig.from_dict({"name": "t", "steps": steps})
    c.questioned = questionada
    return c


# ── Curva por percentis ────────────────────────────────────────────────────

def test_curva_por_percentis_absorve_a_exposicao():
    escura, clara = _foto(80, 1), _foto(170, 1)
    fixa = _cfg([{"step_type": "tone_curve", "black_point": 100, "white_point": 150}])
    adapt = _cfg([{"step_type": "tone_curve", "mode": "face_percentiles",
                   "black_percentile": 20, "white_percentile": 60}])
    pretos = lambda c, img: measure(  # noqa: E731
        ImageProcessingPipeline(c).process_image(img, face=FACE), FACE).black_fraction
    # Fixa: a escura vira quase toda preta e a clara quase nada.
    assert pretos(fixa, escura) - pretos(fixa, clara) > 0.5
    # Por percentis: ~20% de preto nas duas.
    for img in (escura, clara):
        assert pretos(adapt, img) == pytest.approx(0.20, abs=0.03)


def test_curva_por_percentis_precisa_da_face():
    passo = ImageProcessorFactory.build(ImageProcessorFactory.config_from_dict(
        {"step_type": "tone_curve", "mode": "face_percentiles"}))
    assert passo.requires_face
    with pytest.raises(FaceGeometryRequiredError):
        passo.apply_step(_foto(120, 0))


def test_campo_arredonda_as_casas_do_formulario():
    cfg = ImageProcessorFactory.config_from_dict({"step_type": "tone_curve"})
    spec = next(s for s in cfg.param_specs() if s.name == "gamma")
    assert spec.clamp(1.156249) == 1.16


# ── IPD da questionada ─────────────────────────────────────────────────────

def _perfil(ipd=20.0, **medidas):
    m = measure(_foto(120, 9), FACE)
    m = m.__class__(**{**m.to_dict(), "ipd_px": ipd, **medidas})
    return QuestionedProfile("Q1.png", "", "ab" * 32, "", FACE, "detector", m)


def test_redimensionar_pelo_ipd_da_questionada():
    passos = [{"step_type": "resize", "mode": "target_ipd", "use_questioned_ipd": True,
               "target_ipd_px": 999}]
    out = ImageProcessingPipeline(_cfg(passos, _perfil(ipd=25.0))).process(_foto(120, 0), face=FACE)
    assert out.face.ipd == pytest.approx(25.0, abs=0.3)
    with pytest.raises(ValueError):
        ImageProcessingPipeline(_cfg(passos))
    cfg = _cfg(passos)
    assert cfg.steps[0].summary_pt() == "IPD da questionada"
    assert cfg.steps[0].target_ipd_px == 999  # o config não é alterado


# ── Ajuste automático ──────────────────────────────────────────────────────

@pytest.fixture
def dataset(tmp_path):
    pasta = tmp_path / "Documents"
    (pasta / "images").mkdir(parents=True)
    faces = []
    for i, nivel in enumerate(np.linspace(70, 190, 12)):
        nome = f"ID{i:02d}.png"
        cv2.imencode(".png", _foto(float(nivel), i))[1].tofile(str(pasta / "images" / nome))
        ff = FFInfo()
        ff.bbox = np.asarray(FACE.bbox, dtype=np.float64)
        ff.keypoints = np.asarray(FACE.keypoints, dtype=np.float64)
        faces.append(FaceInfo(image_path=nome, identity=f"ID{i:02d}", ff_info=ff))
    ds = FaceDataset(name="Documents", faces=faces, path_mode=PathMode.RELATIVE,
                     path_root="images")
    ds._parquet_dir = pasta
    return ds


def _alvo_e_inicio(dataset):
    """Questionada = uma imagem da base processada com valores conhecidos;
    o ajuste parte de valores errados."""
    certos = [{"step_type": "grayscale"},
              {"step_type": "tone_curve", "mode": "face_percentiles",
               "black_percentile": 30, "white_percentile": 55},
              {"step_type": "halftone"},
              {"step_type": "resize", "mode": "target_ipd", "use_questioned_ipd": True}]
    perfil0 = _perfil(ipd=20.0)
    img = load_sample(dataset, _cfg([]).input_source, indices=[5])[0]
    out = ImageProcessingPipeline(_cfg(certos, perfil0)).process(img.image, face=img.face)
    perfil = QuestionedProfile("Q1.png", "", "ab" * 32, "", out.face, "detector",
                               measure(out.image, out.face))
    errados = [dict(p) for p in certos]
    errados[1].update(black_percentile=5, white_percentile=95)
    return _cfg(errados, perfil)


def test_ajuste_melhora_e_confere_em_imagens_nao_usadas(dataset):
    inicio = _alvo_e_inicio(dataset)
    r = fit_to_questioned(inicio, dataset, n_train=6, n_validation=6, rounds=2)
    from faceaudit.services.image_pipeline_fit_service import objective
    antes = objective(r.train_before, inicio, 6)
    depois = objective(r.train_after, r.config, 6)
    assert depois < antes
    assert {c.param for c in r.changes} <= {"black_percentile", "white_percentile", "gamma"}
    assert r.changes and r.validation_after is not None
    assert r.validation_after.agreement["median_distance"] < \
        r.validation_before.agreement["median_distance"]
    # Ordem e técnica intactas; questionada e config original intactos.
    assert [s.step_type for s in r.config.steps] == [s.step_type for s in inicio.steps]
    assert r.config.steps[2].mode == inicio.steps[2].mode
    assert inicio.steps[1].black_percentile == 5
    assert r.to_dict()["n_evaluations"] == r.n_evaluations


def test_amostras_de_treino_e_validacao_sao_disjuntas():
    treino, val = split_indices(271, 40, 40)
    assert len(treino) == 40 and len(val) == 40 and not set(treino) & set(val)
    treino, val = split_indices(271, 40)  # padrão: todo o resto valida
    assert len(val) == 231 and not set(treino) & set(val)
    treino, val = split_indices(10, 40, 40)
    assert len(treino) == 10 and val == []


def test_ajuste_cancelado_e_sem_questionada(dataset):
    inicio = _alvo_e_inicio(dataset)
    with pytest.raises(FitCancelled):
        fit_to_questioned(inicio, dataset, n_train=4, n_validation=0,
                          should_stop=lambda: True)
    with pytest.raises(ValueError):
        fit_to_questioned(_cfg(inicio.to_dict()["steps"]), dataset)
    sem_numero = _cfg([{"step_type": "grayscale"}], inicio.questioned)
    assert tunable_params(sem_numero) == []
    with pytest.raises(ValueError):
        fit_to_questioned(sem_numero, dataset)


def test_so_campos_visiveis_e_numericos_sao_ajustados():
    cfg = _cfg([{"step_type": "tone_curve", "mode": "face_percentiles"},
                {"step_type": "resize", "mode": "target_ipd", "use_questioned_ipd": True},
                {"step_type": "halftone", "mode": "error_diffusion"}])
    nomes = {(i, s.name) for i, s in tunable_params(cfg)}
    assert nomes == {(0, "black_percentile"), (0, "white_percentile"), (0, "gamma")}


# ── Modelo ─────────────────────────────────────────────────────────────────

def test_modelo_xerox_de_documento():
    cfg = BUILTIN_TEMPLATES["Xerox de documento"]()
    assert cfg.questioned is None
    assert ImageProcessingPipelineConfig.from_json_string(cfg.to_json_string()).to_dict() \
        == cfg.to_dict()
    assert cfg.steps[4].use_questioned_ipd
    assert tunable_params(cfg)


# ── Editor ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_editor_aplica_o_ajuste_e_registra_na_descricao(app, dataset):
    from interface.tabs.dataset_manager.image_pipeline_dialog import (
        ImagePipelineDialogController,
    )

    inicio = _alvo_e_inicio(dataset)
    r = fit_to_questioned(inicio, dataset, n_train=6, n_validation=6, rounds=1)
    c = ImagePipelineDialogController(dataset, initial_face_index=0)
    c._pipeline_config = inicio
    c._mark_clean()
    c.apply_fit(r)
    assert c._pipeline_config.steps[1].black_percentile == r.config.steps[1].black_percentile
    assert "Ajustado à questionada Q1.png" in c._pipeline_config.description
    assert c._is_dirty
    assert c.view.degradation_measures._amostra is r.validation_after


def test_objetivo_so_conta_o_que_esta_fora_da_tolerancia():
    """A distância média melhorava enquanto a parcela de imagens dentro da
    tolerância caía (caso real 18% → 12%). O objetivo novo não premia
    aproximar o que já está dentro."""
    from faceaudit.services.image_degradation_profile_service import DegradationMeasures
    from faceaudit.services.image_pipeline_evaluation_service import SampleEvaluation
    from faceaudit.services.image_pipeline_fit_service import excess_objective

    alvo = _perfil(ipd=20.0)
    m0 = alvo.measures
    dentro = m0.__class__(**{**m0.to_dict(), "black_fraction": m0.black_fraction + 0.04})
    exato = m0
    fora = m0.__class__(**{**m0.to_dict(), "black_fraction": m0.black_fraction + 0.15})
    ev = lambda ms: SampleEvaluation(ms, {}, [], None, 0, [])  # noqa: E731
    cfg = _cfg([], alvo)
    assert excess_objective(ev([dentro]), cfg, 1) == excess_objective(ev([exato]), cfg, 1) == 0
    assert excess_objective(ev([fora]), cfg, 1) == pytest.approx(2.0)
    assert excess_objective(ev([]), cfg, 1) > excess_objective(ev([fora]), cfg, 1)


def test_sem_melhora_na_validacao_o_editor_recomenda_nao_aplicar(app, dataset):
    from PyQt6 import QtCore, QtWidgets
    from interface.tabs.dataset_manager.image_pipeline_dialog import (
        ImagePipelineDialogController,
    )
    from faceaudit.services.image_pipeline_fit_service import FitResult, ParamChange

    inicio = _alvo_e_inicio(dataset)
    r = fit_to_questioned(inicio, dataset, n_train=6, n_validation=6, rounds=1)
    pior = FitResult(r.config, r.changes or [ParamChange(1, "Curva", "gamma", "Gama", 1.0, 1.1)],
                     r.train_before, r.train_after, r.validation_before, r.validation_after,
                     r.n_evaluations, validation_objective_before=1.0,
                     validation_objective_after=2.0)
    assert pior.improved_on_validation is False

    c = ImagePipelineDialogController(dataset, initial_face_index=0)
    c._pipeline_config = inicio
    vistos = {}

    def _responder(botao):
        caixa = QtWidgets.QApplication.activeModalWidget()
        vistos["texto"] = caixa.text()
        vistos["padrao"] = caixa.standardButton(caixa.defaultButton())
        caixa.button(botao).click()

    QtCore.QTimer.singleShot(0, lambda: _responder(QtWidgets.QMessageBox.StandardButton.No))
    c._on_fit_done(pior)
    assert "NÃO melhorou" in vistos["texto"]
    assert vistos["padrao"] == QtWidgets.QMessageBox.StandardButton.No
    assert c._pipeline_config is inicio and not c._is_dirty

    # Quem insiste e clica Sim, aplica.
    QtCore.QTimer.singleShot(0, lambda: _responder(QtWidgets.QMessageBox.StandardButton.Yes))
    c._on_fit_done(pior)
    assert c._is_dirty and "Ajustado à questionada" in c._pipeline_config.description


def test_ponto_de_partida_pelas_medidas_da_questionada():
    """A busca fica presa perto de onde começa; a segunda partida vem das
    medidas: a curva da copiadora põe em preto e branco as frações da
    questionada, a do scanner parte neutra com os níveis distintos dela."""
    from faceaudit.services.image_degradation_profile_service import DegradationMeasures
    from faceaudit.services.image_pipeline_fit_service import start_from_questioned
    cfg = BUILTIN_TEMPLATES["Xerox de documento"]()
    cfg.questioned = QuestionedProfile(
        file_name="q.png", source_path="", sha256="0" * 64, measured_at="",
        face=None, face_source=None,
        measures=DegradationMeasures(
            width=172, height=230, region="face", is_grayscale=True, ipd_px=48.0,
            black_fraction=0.032, white_fraction=0.524, midtone_fraction=0.128,
            distinct_levels=150, grain=18079.0, mean_level=218.6))
    p = start_from_questioned(cfg)
    copiadora, scanner = p.steps[1], p.steps[-1]
    assert (copiadora.black_percentile, copiadora.white_percentile) == (3.2, 47.6)
    assert (scanner.black_point, scanner.white_point, scanner.input_levels) == (0, 255, 150)
    assert copiadora.gamma == scanner.gamma == 1.0
    # O pipeline do perito não é tocado.
    assert cfg.steps[1].black_percentile == 17.5 and cfg.steps[-1].input_levels == 256
    # Sem passo que saiba partir das medidas: só a partida atual.
    so_resize = _cfg([{"step_type": "resize", "mode": "factor", "factor": 0.5}],
                     cfg.questioned)
    assert start_from_questioned(so_resize) is None


def test_ajuste_diz_de_que_partida_veio_o_resultado(dataset):
    r = fit_to_questioned(_alvo_e_inicio(dataset), dataset, n_train=6,
                          n_validation=6, rounds=1)
    assert r.start in ("pipeline atual", "medidas da questionada")
    assert r.to_dict()["start"] == r.start
