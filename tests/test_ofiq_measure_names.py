"""Nomes do OFIQ → campos do OFIQInfo, nas duas versões que o FaceAudit lê.

As fixtures são linhas REAIS do `OFIQSampleApp` da v1.0.3 e da v1.2.0 sobre a
mesma imagem de conformidade do BSI (`b-01-smile.png`). A v1.2.0 mudou os
nomes — `.native` no valor bruto e oito medidas renomeadas — e a tabela antiga
perdia essas colunas em silêncio.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pytest

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ofiq_info import OFIQInfo
from faceaudit.models.ofiq_measure_names import (
    MEASURE_FIELD_PREFIX,
    RENAMED_MEASURES,
    field_for_column,
)

FIXTURES = Path(__file__).parent / "fixtures" / "ofiq"
CSV_103 = FIXTURES / "ofiqsampleapp_v1_0_3_b-01-smile.csv"
CSV_120 = FIXTURES / "ofiqsampleapp_v1_2_0_b-01-smile.csv"

# Os 28 nomes que `OFIQ.quality_measure_name` devolve no wrapper Python da
# v1.2.0 (copiados da execução real) — sem sufixo, que é o valor bruto.
NOMES_DO_WRAPPER_120 = [
    "UnifiedQualityScore", "BackgroundUniformity", "IlluminationUniformity",
    "LuminanceMean", "LuminanceVariance", "UnderExposurePrevention",
    "OverExposurePrevention", "DynamicRange", "Sharpness",
    "NoCompressionArtifacts", "NaturalColour", "SingleFacePresent", "EyesOpen",
    "MouthClosed", "EyesVisible", "MouthOcclusionPrevention",
    "FaceOcclusionPrevention", "InterEyeDistance", "HeadSize",
    "LeftwardCropOfFace", "RightwardCropOfFace", "MarginAboveTheFace",
    "MarginBelowTheFace", "HeadPoseYawFrontal", "HeadPosePitchFrontal",
    "HeadPoseRollFrontal", "ExpressionNeutrality", "NoHeadCoverings",
]

# Únicos valores que mudam entre as versões nesta imagem (medido também nas
# 28 imagens de conformidade: só esses dois brutos, nenhuma nota 0–100).
_DIFEREM_ENTRE_VERSOES = {"background_uniformity_raw", "no_head_coverings_raw"}


def _linha(caminho: Path) -> dict:
    with open(caminho, encoding="utf-8", newline="") as fh:
        return next(csv.DictReader(fh, delimiter=";"))


def _medidas(info: OFIQInfo) -> dict:
    return {
        k: v for k, v in info.to_dict().items()
        if k.endswith("_raw") or k.endswith("_scalar")
    }


@pytest.mark.parametrize("caminho", [CSV_103, CSV_120], ids=["v1.0.3", "v1.2.0"])
def test_toda_coluna_real_tem_campo(caminho):
    colunas = [c for c in _linha(caminho) if c and c.strip()]
    sem_campo = [c for c in colunas if field_for_column(c) is None]
    assert sem_campo == []


@pytest.mark.parametrize("caminho", [CSV_103, CSV_120], ids=["v1.0.3", "v1.2.0"])
def test_as_56_medidas_chegam_ao_ofiqinfo_e_nada_sobra(caminho):
    info = OFIQInfo.from_dict(_linha(caminho))
    medidas = _medidas(info)
    assert len(medidas) == 2 * len(MEASURE_FIELD_PREFIX) == 56
    faltando = sorted(k for k, v in medidas.items() if v is None)
    assert faltando == [], f"medidas perdidas na tradução: {faltando}"
    assert not info._extra_metrics


def test_as_duas_versoes_dao_o_mesmo_ofiqinfo():
    """Mesma imagem, versões diferentes: só os dois brutos conhecidos mudam."""
    a = _medidas(OFIQInfo.from_dict(_linha(CSV_103)))
    b = _medidas(OFIQInfo.from_dict(_linha(CSV_120)))
    diferentes = {k for k in a if a[k] != pytest.approx(b[k], abs=1e-9)}
    assert diferentes <= _DIFEREM_ENTRE_VERSOES


def test_angulos_da_cabeca_da_v120_nao_somem():
    """O caso que motivou a tabela: `HeadPoseYawFrontal.native` caía na sobra,
    e `ofiq_yaw/pitch/roll` ficavam vazios em datasets da v1.2.0."""
    info = OFIQInfo.from_dict(_linha(CSV_120))
    assert info.head_pose_yaw_raw is not None
    assert info.head_pose_pitch_raw is not None
    assert info.head_pose_roll_raw is not None


def test_nomes_do_wrapper_viram_valor_bruto():
    for nome in NOMES_DO_WRAPPER_120:
        campo = field_for_column(nome)
        assert campo is not None and campo.endswith("_raw"), nome
        assert field_for_column(nome + ".scalar").endswith("_scalar")


def test_toda_renomeacao_aponta_para_medida_conhecida():
    assert set(RENAMED_MEASURES.values()) <= set(MEASURE_FIELD_PREFIX)


def test_coluna_desconhecida_vai_para_extras_e_avisa(caplog):
    with caplog.at_level(logging.WARNING, logger="faceaudit.models.ofiq_info"):
        info = OFIQInfo.from_dict({"MedidaInventada2027.native": "3.5"})
    assert info._extra_metrics == {"MedidaInventada2027.native": 3.5}
    assert "MedidaInventada2027.native" in caplog.text


def test_versao_e_modo_ficam_texto():
    """"1.2" não pode virar o número 1.2."""
    info = OFIQInfo.from_dict({"version": "1.2", "backend": "library"})
    assert info.version == "1.2"
    assert info.backend == "library"


def test_versao_e_modo_sobrevivem_ao_parquet(tmp_path):
    face = FaceInfo(image_path="/fotos/a.png", identity="a")
    face.ofiq_info = OFIQInfo.from_dict(_linha(CSV_103))
    face.ofiq_info.version = "1.0.3"
    face.ofiq_info.backend = "executable"
    caminho = tmp_path / "ds.parquet"
    FaceDataset(name="ds", faces=[face]).to_parquet(caminho)

    relida = FaceDataset.from_parquet(caminho).faces[0].ofiq_info
    assert relida.version == "1.0.3"
    assert relida.backend == "executable"
    assert relida.head_pose_yaw_raw == pytest.approx(face.ofiq_info.head_pose_yaw_raw)


def test_colunas_novas_tem_spec(tmp_path):
    """Coluna sem spec fica invisível na tabela."""
    face = FaceInfo(image_path="/fotos/a.png", identity="a")
    face.ofiq_info = OFIQInfo(unified_quality_score_scalar=50.0,
                              version="1.0.3", backend="executable")
    caminho = tmp_path / "ds.parquet"
    FaceDataset(name="ds", faces=[face]).to_parquet(caminho)
    relido = FaceDataset.from_parquet(caminho)
    assert "ofiq_version" in relido.column_specs
    assert "ofiq_backend" in relido.column_specs


def test_angulos_brutos_voltam_ao_reabrir_o_dataset(tmp_path):
    """`to_row` grava `ofiq_yaw/pitch/roll` sem `_raw`; a leitura não conhecia
    esses nomes e os mandava para os extras — `head_pose_*_raw` voltava
    vazio a cada abertura do dataset."""
    face = FaceInfo(image_path="/fotos/a.png", identity="a")
    face.ofiq_info = OFIQInfo(head_pose_yaw_raw=-1.2, head_pose_pitch_raw=3.4,
                              head_pose_roll_raw=0.5)
    caminho = tmp_path / "ds.parquet"
    FaceDataset(name="ds", faces=[face]).to_parquet(caminho)

    relida = FaceDataset.from_parquet(caminho).faces[0].ofiq_info
    assert (relida.head_pose_yaw_raw, relida.head_pose_pitch_raw,
            relida.head_pose_roll_raw) == (-1.2, 3.4, 0.5)
    assert not relida._extra_metrics
