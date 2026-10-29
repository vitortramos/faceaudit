"""Pipeline de imagens, Fase 2: passos de impressão, cópia e digitalização.

Cada passo conferido pelo efeito que reproduz, e a cadeia da xerox de
documento conferida de ponta a ponta: a face sai com a distância entre
os olhos pedida, em preto e branco com grão.
"""

from __future__ import annotations

from random import Random

import cv2
import numpy as np
import pytest

from faceaudit.enums.halftone_mode import HalftoneMode
from faceaudit.factories.image_processor_factory import ImageProcessorFactory
from faceaudit.image_processors.copy_degradation_processors import (
    halftone,
    tone_curve_lut,
)
from faceaudit.image_processors.image_step_base import FaceGeometryRequiredError
from faceaudit.models.image_processing.image_face_geometry import ImageFaceGeometry
from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.pipelines.image_processing_pipeline import ImageProcessingPipeline

FACE = ImageFaceGeometry(
    bbox=[100, 120, 300, 380],
    keypoints=[[150, 200], [250, 200], [200, 260], [160, 320], [240, 320]],
)  # IPD 100 px


def _passo(d):
    return ImageProcessorFactory.build(ImageProcessorFactory.config_from_dict(d))


def _gradiente(h=200, w=256):
    g = np.tile(np.arange(w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def _foto(h=500, w=400, seed=0):
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), (0, 0), 6)
    return cv2.normalize(base, None, 0, 255, cv2.NORM_MINMAX)


# ── Redimensionar ──────────────────────────────────────────────────────────

def test_redimensionar_leva_a_distancia_entre_os_olhos_ao_alvo():
    out, g = _passo({"step_type": "resize", "mode": "target_ipd",
                     "target_ipd_px": 48.8}).apply_step(_foto(), face=FACE)
    assert g.ipd == pytest.approx(48.8, abs=0.5)
    assert out.shape[:2] == (round(500 * 0.488), round(400 * 0.488))
    # A geometria convertida aponta para o mesmo lugar da face na imagem.
    assert g.bbox[0] == pytest.approx(100 * 0.488, abs=1)


def test_redimensionar_por_ipd_sem_pontos_falha():
    with pytest.raises(FaceGeometryRequiredError):
        _passo({"step_type": "resize", "mode": "target_ipd"}).apply_step(_foto())
    so_caixa = ImageFaceGeometry(bbox=[0, 0, 10, 10])
    with pytest.raises(RuntimeError):
        _passo({"step_type": "resize", "mode": "target_ipd"}).apply_step(_foto(), face=so_caixa)


def test_redimensionar_por_fator_e_volta_ao_tamanho():
    img = _foto()
    out, g = _passo({"step_type": "resize", "mode": "factor", "factor": 0.25,
                     "restore_size": True}).apply_step(img, face=FACE)
    assert out.shape == img.shape
    assert g.ipd == pytest.approx(FACE.ipd, abs=0.5)
    # Perdeu detalhe: a variação de alta frequência caiu.
    assert cv2.Laplacian(out, cv2.CV_64F).var() < cv2.Laplacian(img, cv2.CV_64F).var()


def test_redimensionar_recusa_imagem_gigante():
    with pytest.raises(ValueError):
        _passo({"step_type": "resize", "mode": "target_ipd",
                "target_ipd_px": 2000}).apply_step(_foto(), face=FACE.scaled(0.01, 0.01))


# ── Tom ────────────────────────────────────────────────────────────────────

def test_tons_de_cinza_iguala_os_canais():
    out = _passo({"step_type": "grayscale"}).apply(_foto())
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_curva_de_tons():
    lut = tone_curve_lut(40, 200, 1.0)
    assert lut[0] == lut[40] == 0 and lut[200] == lut[255] == 255
    assert tone_curve_lut(0, 255, 2.0)[128] < tone_curve_lut(0, 255, 1.0)[128]
    out = _passo({"step_type": "tone_curve", "black_point": 100,
                  "white_point": 150}).apply(_gradiente())
    assert (out[:, :100] == 0).all() and (out[:, 150:] == 255).all()
    # Ponto de branco abaixo do de preto é corrigido, não quebra.
    cfg = ImageProcessorFactory.config_from_dict(
        {"step_type": "tone_curve", "black_point": 200, "white_point": 100})
    assert cfg.white_point > cfg.black_point


# ── Impressão e cópia ──────────────────────────────────────────────────────

def test_ganho_de_toner_expande_o_escuro():
    img = np.full((60, 60, 3), 255, np.uint8)
    img[28:32, 28:32] = 0
    pretos = lambda a: int((a[..., 0] == 0).sum())  # noqa: E731
    assert pretos(_passo({"step_type": "toner_gain", "radius_px": 2}).apply(img)) > pretos(img)
    assert np.array_equal(_passo({"step_type": "toner_gain", "radius_px": 0}).apply(img), img)


@pytest.mark.parametrize("modo", list(HalftoneMode))
def test_meio_tom_e_binario_e_preserva_o_tom_medio(modo):
    g = cv2.cvtColor(_gradiente(256, 256), cv2.COLOR_BGR2GRAY)
    b = halftone(g, modo, 6.0, 45.0)
    assert set(np.unique(b)) <= {0, 255}
    # Média local preservada: claro continua claro, escuro continua escuro.
    assert b[:, :32].mean() < 60 and b[:, -32:].mean() > 195
    assert abs(b.mean() - g.mean()) < 20


def test_ruido_de_toner_e_deterministico_e_respeita_a_densidade():
    img = np.full((200, 200, 3), 255, np.uint8)
    passo = _passo({"step_type": "toner_noise", "density": 0.01})
    a = passo.apply(img, rng=Random(1))
    assert np.array_equal(a, passo.apply(img, rng=Random(1)))
    assert not np.array_equal(a, passo.apply(img, rng=Random(2)))
    assert 0.007 < (a[..., 0] == 0).mean() <= 0.01


# ── Cadeia da xerox ────────────────────────────────────────────────────────

def test_cadeia_da_xerox_de_documento():
    cfg = ImageProcessingPipelineConfig.from_dict({"steps": [
        {"step_type": "grayscale"},
        {"step_type": "tone_curve", "black_point": 90, "white_point": 160},
        {"step_type": "resize", "mode": "target_ipd", "target_ipd_px": 70},
        {"step_type": "toner_gain", "radius_px": 1},
        {"step_type": "halftone", "mode": "error_diffusion"},
        {"step_type": "toner_noise", "density": 0.001},
        {"step_type": "resize", "mode": "target_ipd", "target_ipd_px": 48.8},
    ]})
    out = ImageProcessingPipeline(cfg).process(_foto(), seed_jitter="a.png", face=FACE)
    assert out.face.ipd == pytest.approx(48.8, abs=0.5)
    canais = out.image
    assert np.array_equal(canais[..., 0], canais[..., 2])
    # Contraste estourado: bem mais preto e branco que a entrada.
    cinza = canais[..., 0]
    entrada = cv2.cvtColor(_foto(), cv2.COLOR_BGR2GRAY)
    for extremo in (lambda a: a <= 20, lambda a: a >= 235):
        assert extremo(cinza).mean() > 0.05
        assert extremo(cinza).mean() > 3 * extremo(entrada).mean()
    # Reprodutível.
    de_novo = ImageProcessingPipeline(cfg).process(_foto(), seed_jitter="a.png", face=FACE)
    assert np.array_equal(out.image, de_novo.image)


def test_curva_com_niveis_antes_do_contraste_reproduz_o_pente():
    """Contraste esticado por último sobre imagem de 8 bits de pouco
    contraste: toda a faixa 0–255 ocupada, com buracos regulares. Sem o
    parâmetro, o número de níveis só mudaria movendo preto e branco."""
    cheio = tone_curve_lut(24, 223, 1.09)
    assert len(np.unique(cheio)) == 200  # (branco − preto + 1)
    pente = tone_curve_lut(24, 223, 1.09, input_levels=150)
    niveis = np.unique(pente)
    assert len(niveis) == 150 and niveis[0] == 0 and niveis[-1] == 255
    assert (np.diff(niveis) > 1).any()  # buracos, não faixa estreita
    # Preto e branco não mudam com o número de níveis.
    assert pente[24] == 0 and pente[223] == 255
    # 256 = todos: a curva é exatamente a de antes (datasets já gerados
    # continuam reproduzíveis pixel a pixel).
    assert np.array_equal(tone_curve_lut(24, 223, 1.09, input_levels=256), cheio)


def test_niveis_antes_do_contraste_viaja_no_json_e_tem_padrao():
    cfg = ImageProcessorFactory.config_from_dict(
        {"step_type": "tone_curve", "black_point": 24, "white_point": 223,
         "input_levels": 150})
    assert cfg.input_levels == 150 and "150 níveis" in cfg.summary_pt()
    assert ImageProcessorFactory.config_from_dict(cfg.to_dict()).input_levels == 150
    antigo = ImageProcessorFactory.config_from_dict({"step_type": "tone_curve"})
    assert antigo.input_levels == 256 and "níveis" not in antigo.summary_pt()
    spec = next(s for s in antigo.param_specs() if s.name == "input_levels")
    assert spec.tunable and spec.is_visible(antigo)
    out = _passo({"step_type": "tone_curve", "black_point": 0, "white_point": 255,
                  "input_levels": 16}).apply(_gradiente())
    assert len(np.unique(out)) <= 16
