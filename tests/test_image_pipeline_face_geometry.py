"""Pipeline de imagens, Fase 1: geometria da face, formulário declarativo
e sorteio reprodutível.

- A geometria entra uma vez (a do dataset) e cada passo a converte: o
  recorte pela face e a oclusão dos olhos não redetectam na imagem já
  degradada.
- Sem geometria, passo que precisa dela FALHA em vez de cair no centro.
- O sorteio não depende do `hash()` do Python, que muda por execução.
- Todo passo registrado declara campos que existem no config, e o painel
  monta o formulário de cada um.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from faceaudit.enums.image_processing_step_type import ImageProcessingStepType
from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.image_processors.image_step_base import FaceGeometryRequiredError
from faceaudit.models.image_processing.image_face_geometry import ImageFaceGeometry
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.models.image_processing.step_param_spec import StepParamKind
from faceaudit.pipelines.image_processing_pipeline import (
    ImageProcessingPipeline,
    stable_rng,
)

RAIZ = Path(__file__).resolve().parents[1]

FACE = ImageFaceGeometry(
    bbox=[40, 50, 120, 150],
    keypoints=[[60, 80], [100, 80], [80, 100], [65, 125], [95, 125]],
)


def _pipe(*steps, seed=42):
    return ImageProcessingPipeline(
        ImageProcessingPipelineConfig.from_dict({"seed": seed, "steps": list(steps)})
    )


def _imagem():
    return np.full((200, 160, 3), 200, np.uint8)


class _SemDetector:
    """FFService falso que falha se alguém tentar detectar."""

    class app:  # noqa: N801
        class backend:  # noqa: N801
            @staticmethod
            def detect_faces(_img):
                raise AssertionError("não devia redetectar")


# ── Geometria ──────────────────────────────────────────────────────────────

def test_escala_e_translacao_da_geometria():
    g = FACE.scaled(0.5, 0.5)
    assert g.ipd == pytest.approx(FACE.ipd * 0.5)
    np.testing.assert_allclose(g.keypoints[0], [(60.5) * 0.5 - 0.5, (80.5) * 0.5 - 0.5])
    t = FACE.translated(-40, -50)
    np.testing.assert_allclose(t.bbox, [0, 0, 80, 100])
    assert ImageFaceGeometry.from_values(None, None) is None
    assert ImageFaceGeometry.from_values([np.nan] * 4, None) is None


def test_recorte_pela_face_usa_a_geometria_e_a_converte():
    pipe = ImageProcessingPipeline(
        ImageProcessingPipelineConfig.from_dict({"steps": [
            {"step_type": "crop", "mode": "face_detected", "margin_factor": 1.0},
        ]}),
        ff_service=_SemDetector(),
    )
    out = pipe.process(_imagem(), face=FACE)
    assert out.image.shape[:2] == (100, 80)
    np.testing.assert_allclose(out.face.keypoints[0], [20, 30])


def test_oclusao_dos_olhos_depois_do_recorte_cai_sobre_os_olhos():
    out = _pipe(
        {"step_type": "crop", "mode": "face_detected", "margin_factor": 1.2},
        {"step_type": "occlusion", "kind": "circle_black", "position": "eyes",
         "radius_pct": 0.05},
    ).process(_imagem(), face=FACE)
    for x, y in out.face.eyes:
        assert tuple(out.image[int(round(y)), int(round(x))]) == (0, 0, 0)
    assert tuple(out.image[2, 2]) == (200, 200, 200)


def test_passo_que_precisa_da_face_falha_sem_ela_em_vez_de_ir_para_o_centro():
    pipe = _pipe({"step_type": "occlusion", "kind": "circle_black", "position": "eyes"})
    with pytest.raises(FaceGeometryRequiredError):
        pipe.process(_imagem())


def test_passo_de_pixel_preserva_geometria_e_o_desconhecido_a_descarta():
    out = _pipe({"step_type": "blur"}, {"step_type": "jpeg", "quality": 80}).process(
        _imagem(), face=FACE)
    assert out.face is FACE
    out = _pipe({"step_type": "crop", "mode": "centered",
                 "target_width": 100, "target_height": 100}).process(_imagem(), face=FACE)
    np.testing.assert_allclose(out.face.bbox, np.asarray(FACE.bbox) - [30, 50, 30, 50])


def test_margem_do_recorte_editada_tem_efeito():
    """O campo antigo editava `face_padding`, que o processador não lia."""
    cfg = ImageProcessingPipelineConfig.from_dict({"steps": [
        {"step_type": "crop", "mode": "face_detected", "margin_factor": 1.0},
    ]})
    antes = ImageProcessingPipeline(cfg).process(_imagem(), face=FACE).image.shape
    cfg.steps[0].margin_factor = 1.5
    depois = ImageProcessingPipeline(cfg).process(_imagem(), face=FACE).image.shape
    assert depois[0] > antes[0] and depois[1] > antes[1]
    assert ImageProcessorFactory.config_from_dict(cfg.steps[0].to_dict()).margin_factor == 1.5


# ── Sorteio ────────────────────────────────────────────────────────────────

_SCRIPT_SORTEIO = (
    "import numpy as np;"
    "from faceaudit.models.image_processing.image_processing_pipeline_config "
    "import ImageProcessingPipelineConfig as C;"
    "from faceaudit.pipelines.image_processing_pipeline import ImageProcessingPipeline as P;"
    "c=C.from_dict({'seed':7,'steps':[{'step_type':'occlusion','kind':'rectangle_black',"
    "'position':'random','width_pct':0.1,'height_pct':0.1}]});"
    "o=P(c).process_image(np.full((300,300,3),255,np.uint8),seed_jitter='x/a.jpg');"
    "ys,xs=np.where(o[...,0]==0);print(int(xs.min()),int(ys.min()))"
)


def test_oclusao_aleatoria_cai_no_mesmo_lugar_em_execucoes_diferentes():
    posicoes = set()
    for semente_hash in ("1", "2", "999"):
        env = dict(os.environ, PYTHONHASHSEED=semente_hash)
        r = subprocess.run([sys.executable, "-c", _SCRIPT_SORTEIO], cwd=RAIZ, env=env,
                           capture_output=True, text=True, check=True)
        posicoes.add(r.stdout.strip())
    assert len(posicoes) == 1, posicoes


def test_sorteio_muda_com_semente_e_com_a_imagem():
    a = stable_rng(1, "a").random()
    assert a == stable_rng(1, "a").random()
    assert a != stable_rng(2, "a").random()
    assert a != stable_rng(1, "b").random()


# ── Declaração dos campos ──────────────────────────────────────────────────

@pytest.mark.parametrize("tipo", ImageProcessorFactory.available_step_types())
def test_campos_declarados_existem_e_o_config_sobrevive_ao_json(tipo):
    cfg = ImageProcessorFactory.new_config(tipo)
    for spec in cfg.param_specs():
        if spec.kind != StepParamKind.ACTION:
            assert hasattr(cfg, spec.name), (tipo, spec.name)
        if spec.kind == StepParamKind.CHOICE:
            valores = [v for v, _ in spec.choices_for(cfg)]
            assert getattr(cfg, spec.name) in valores, (tipo, spec.name)
    copia = ImageProcessorFactory.config_from_dict(cfg.to_dict())
    assert copia.to_dict() == cfg.to_dict()
    assert isinstance(cfg.summary_pt(), str)
    assert tipo.category_pt


# ── Painel ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.mark.parametrize("tipo", ImageProcessorFactory.available_step_types())
def test_painel_monta_o_formulario_de_todo_passo(app, tipo):
    from interface.tabs.dataset_manager.image_pipeline_dialog.step_config_panel import (
        StepConfigPanel,
    )
    painel = StepConfigPanel()
    cfg = ImageProcessorFactory.new_config(tipo)
    painel.show_config(cfg)
    assert len(painel._rows) == len(cfg.param_specs())


def test_painel_edita_o_config_e_reavalia_visibilidade(app):
    from interface.tabs.dataset_manager.image_pipeline_dialog.step_config_panel import (
        StepConfigPanel,
    )
    from faceaudit.enums.crop_mode import CropMode

    painel = StepConfigPanel()
    cfg = ImageProcessorFactory.new_config(ImageProcessingStepType.CROP)
    painel.show_config(cfg)
    linhas = {s.name: (lbl, w) for s, lbl, w in painel._rows}
    assert linhas["margin_factor"][1].isHidden()

    mudou = []
    painel.config_changed.connect(mudou.append)
    combo = linhas["mode"][1]
    combo.setCurrentIndex([combo.itemData(i) for i in range(combo.count())]
                          .index(CropMode.FACE_DETECTED))
    assert cfg.mode == CropMode.FACE_DETECTED and mudou
    assert not linhas["margin_factor"][1].isHidden()
    assert linhas["target_width"][1].isHidden()

    linhas["margin_factor"][1].setValue(99.0)  # acima da faixa
    assert cfg.margin_factor == 5.0


def test_reordenar_dois_passos_do_mesmo_tipo_leva_cada_config_junto(app):
    from interface.tabs.dataset_manager.image_pipeline_dialog.steps_list_widget import (
        StepsListWidget,
    )
    from PyQt6 import QtCore

    lista = StepsListWidget()
    a = ImageProcessorFactory.config_from_dict({"step_type": "blur", "sigma": 1.0})
    b = ImageProcessorFactory.config_from_dict({"step_type": "blur", "sigma": 9.0})
    lista.populate([a, b])
    ordens = []
    lista.steps_reordered.connect(ordens.append)
    # Simula o arrasto: troca os dados das linhas como o Qt faz.
    item0, item1 = lista.list.item(0), lista.list.item(1)
    d0 = item0.data(QtCore.Qt.ItemDataRole.UserRole)
    item0.setData(QtCore.Qt.ItemDataRole.UserRole, item1.data(QtCore.Qt.ItemDataRole.UserRole))
    item1.setData(QtCore.Qt.ItemDataRole.UserRole, d0)
    lista._on_rows_moved(None, 0, 0, None, 2)
    assert ordens == [[1, 0]]
