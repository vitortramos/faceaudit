"""Nona medida do perfil: textura por faixa de tom.

Meio-tom concentra a textura nos tons médios e deixa preto e branco
chapados limpos; ruído de toner põe textura nas áreas claras; uma imagem
sem textura tem o formato do meio-tom mas força bem menor. A medida
compara FORMATO e NÍVEL e tem de separar os três casos. Perfil gravado
antes de a medida existir continua abrindo; o editor completa a medida
quando a questionada no caminho gravado tem o mesmo SHA-256.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from faceaudit.models.image_processing.image_processing_pipeline_config import (
    ImageProcessingPipelineConfig,
)
from faceaudit.services.image_degradation_profile_service import (
    FAIXAS_DE_TOM,
    MEASURES_BY_KEY,
    DegradationMeasures,
    agreement,
    build_questioned_profile,
    complete_questioned_measures,
    measure,
    summarize,
)

SPEC = MEASURES_BY_KEY["texture_by_tone"]
N_FAIXAS = len(FAIXAS_DE_TOM) - 1


def _rampa(largura=480, altura=160):
    """Tom contínuo de 0 a 255 da esquerda para a direita."""
    return np.tile(np.linspace(0, 255, largura), (altura, 1))


def _meio_tom(seed):
    """Pontos pretos com probabilidade = escuridão, depois reduzidos à
    metade (o scanner junta os pontos em cinzas)."""
    rng = np.random.default_rng(seed)
    tom = cv2.resize(_rampa(), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    pontos = np.where(rng.random(tom.shape) < tom / 255.0, 255, 0).astype(np.float32)
    return cv2.resize(pontos, None, fx=0.5, fy=0.5,
                      interpolation=cv2.INTER_AREA).astype(np.uint8)


def _ruido_de_toner(seed):
    """Tom contínuo com pontos pretos isolados espalhados por toda parte."""
    rng = np.random.default_rng(seed)
    img = _rampa().copy()
    img[rng.random(img.shape) < 0.04] = 0
    return img.astype(np.uint8)


def _perfil(img):
    return measure(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)).texture_by_tone


def test_meio_tom_concentra_a_textura_nos_tons_medios():
    p = _perfil(_meio_tom(0))
    assert len(p) == N_FAIXAS and all(x is not None for x in p)
    meio = p[N_FAIXAS // 2]
    assert p[0] < 0.6 * meio and p[-1] < 0.6 * meio


def test_ruido_de_toner_poe_textura_no_claro():
    p = _perfil(_ruido_de_toner(0))
    assert p[-1] > p[0]


def test_distancia_separa_meio_tom_de_ruido_e_de_imagem_lisa():
    alvo = _perfil(_meio_tom(0))
    assert SPEC.is_close(alvo, _perfil(_meio_tom(1)))  # outra semente, mesmo processo
    assert not SPEC.is_close(alvo, _perfil(_ruido_de_toner(0)))  # formato errado
    assert not SPEC.is_close(alvo, _perfil(_rampa().astype(np.uint8)))  # sem textura


def test_poucas_faixas_em_comum_nao_compara():
    so_duas = (None, None, None, 10.0, 12.0)
    assert SPEC.distance(so_duas, so_duas) is None
    assert SPEC.is_close(so_duas, so_duas) is None


def test_formato_e_tolerancia_na_tela():
    assert SPEC.format((22.4, 69.6, None, 66.7, 24.9)) == "22 · 70 · — · 67 · 25"
    assert SPEC.tolerance_text() == "formato ± 0,12; nível ± 40%"


def test_json_ida_e_volta():
    m = measure(cv2.cvtColor(_meio_tom(0), cv2.COLOR_GRAY2BGR))
    relida = DegradationMeasures.from_dict(m.to_dict())
    assert isinstance(relida.texture_by_tone, tuple)
    assert relida.texture_by_tone == pytest.approx(m.texture_by_tone, abs=1e-3)


def test_perfil_antigo_abre_e_a_medida_fica_de_fora():
    m = measure(cv2.cvtColor(_meio_tom(0), cv2.COLOR_GRAY2BGR)).to_dict()
    del m["texture_by_tone"]
    antigo = DegradationMeasures.from_dict(m)
    assert antigo.texture_by_tone is None
    conc = agreement(antigo, [measure(cv2.cvtColor(_meio_tom(1), cv2.COLOR_GRAY2BGR))])
    assert "texture_by_tone" not in conc["measures"]


def test_resumo_por_faixa():
    medidas = [measure(cv2.cvtColor(_meio_tom(s), cv2.COLOR_GRAY2BGR)) for s in range(3)]
    r = summarize(medidas)["texture_by_tone"]
    assert r["n"] == 3 and len(r["median"]) == N_FAIXAS
    assert all(lo <= med <= hi for lo, med, hi in zip(r["p5"], r["median"], r["p95"]))


def test_completar_perfil_antigo_remede_so_o_que_falta(tmp_path):
    q = tmp_path / "Q1.png"
    cv2.imencode(".png", cv2.cvtColor(_meio_tom(0), cv2.COLOR_GRAY2BGR))[1].tofile(str(q))
    perfil, img = build_questioned_profile(q, None)
    import dataclasses
    antigo = dataclasses.replace(
        perfil, measures=dataclasses.replace(perfil.measures, texture_by_tone=None,
                                             mean_level=123.0))
    completo = complete_questioned_measures(antigo, img)
    assert completo.measures.texture_by_tone == pytest.approx(perfil.measures.texture_by_tone)
    assert completo.measures.mean_level == 123.0  # o que estava gravado não muda
    assert complete_questioned_measures(completo, img) is completo


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6 import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_editor_completa_a_questionada_so_com_o_sha_conferido(app, tmp_path):
    import dataclasses

    from faceaudit.models.face_dataset import FaceDataset
    from interface.tabs.dataset_manager.image_pipeline_dialog import (
        ImagePipelineDialogController,
    )

    q = tmp_path / "Q1.png"
    cv2.imencode(".png", cv2.cvtColor(_meio_tom(0), cv2.COLOR_GRAY2BGR))[1].tofile(str(q))
    perfil, _ = build_questioned_profile(q, None)
    antigo = dataclasses.replace(
        perfil, measures=dataclasses.replace(perfil.measures, texture_by_tone=None))

    c = ImagePipelineDialogController(FaceDataset(name="vazio", faces=[]))
    c._pipeline_config = ImageProcessingPipelineConfig(name="x", questioned=antigo)
    assert c._reload_questioned_from_config() is True
    assert c._pipeline_config.questioned.measures.texture_by_tone is not None

    # Arquivo trocado no caminho gravado: não remede (seria outro alvo).
    cv2.imencode(".png", cv2.cvtColor(_ruido_de_toner(0), cv2.COLOR_GRAY2BGR))[1].tofile(str(q))
    c._pipeline_config = ImageProcessingPipelineConfig(name="x", questioned=antigo)
    assert c._reload_questioned_from_config() is False
    assert c._pipeline_config.questioned.measures.texture_by_tone is None
    assert "MUDOU" in c._questioned_warning
