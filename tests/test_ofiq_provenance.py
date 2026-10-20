"""Proveniência do OFIQ: versão, modelos e o carimbo em cada face."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from faceaudit.models.face_dataset import FaceDataset
from faceaudit.models.face_info import FaceInfo
from faceaudit.models.ofiq_info import OFIQInfo
from faceaudit.services import ofiq_service as ofiq_service_module
from faceaudit.services.ofiq_provenance_service import (
    detect_ofiq_version,
    mixed_versions_message,
    models_fingerprint,
    parse_version_file,
    summarize_ofiq_versions,
)
from faceaudit.services.ofiq_service import OFIQService

FIXTURES = Path(__file__).parent / "fixtures" / "ofiq"

# Formato real do `Version.txt` do OFIQ (v1.0.3).
VERSION_TXT_103 = """# This file is used to set the version numbers for the cmake project and conan.
# DO NOT change the "VERSION_*" keywords, as this file gets parsed by the conanfile.py and CMakeLists.txt
# keep SUFFIX empty for release builds
VERSION_MAJOR 1
VERSION_MINOR 0
VERSION_PATCH 3
VERSION_SUFFIX
"""


def _instalacao_ofiq(raiz: Path, version_txt: str = VERSION_TXT_103) -> Path:
    """Árvore mínima de uma instalação do OFIQ; devolve o caminho do exe."""
    bin_dir = raiz / "install_x86_64" / "Release" / "bin"
    bin_dir.mkdir(parents=True)
    exe = bin_dir / "OFIQSampleApp.exe"
    exe.write_bytes(b"")
    (raiz / "Version.txt").write_text(version_txt, encoding="utf-8")
    modelos = raiz / "data" / "models" / "sharpness"
    modelos.mkdir(parents=True)
    (modelos / "face_sharpness_rtree.xml.gz").write_bytes(b"modelo")
    (modelos / "LICENSE.md").write_text("MIT", encoding="utf-8")
    (raiz / "data" / "ofiq_config.jaxn").write_text("{}", encoding="utf-8")
    return exe


# ── versão ───────────────────────────────────────────────────────────

def test_parse_version_file():
    assert parse_version_file(VERSION_TXT_103) == "1.0.3"
    assert parse_version_file(VERSION_TXT_103.replace("VERSION_SUFFIX", "VERSION_SUFFIX rc2")) == "1.0.3-rc2"
    assert parse_version_file("lixo") is None


def test_versao_vem_do_version_txt_da_instalacao(tmp_path):
    exe = _instalacao_ofiq(tmp_path / "OFIQ-1.0.3")
    assert detect_ofiq_version(str(exe)) == "1.0.3"


def test_executavel_solto_tem_versao_desconhecida(tmp_path):
    assert detect_ofiq_version("OFIQSampleApp.exe") is None
    solto = tmp_path / "OFIQSampleApp.exe"
    solto.write_bytes(b"")
    assert detect_ofiq_version(str(solto)) is None


# ── modelos ──────────────────────────────────────────────────────────

def test_impressao_digital_dos_modelos(tmp_path):
    exe = _instalacao_ofiq(tmp_path / "a")
    data = tmp_path / "a" / "data"
    fp = models_fingerprint(str(data))
    assert fp and len(fp) == 64
    # Aceita a pasta ou o .jaxn, como o `-c` do OFIQSampleApp.
    assert models_fingerprint(str(data / "ofiq_config.jaxn")) == fp
    # Licença/README não entram.
    (data / "models" / "sharpness" / "LICENSE.md").write_text("outra", encoding="utf-8")
    assert models_fingerprint(str(data)) == fp
    # Trocar um modelo muda.
    (data / "models" / "sharpness" / "face_sharpness_rtree.xml.gz").write_bytes(b"outro")
    assert models_fingerprint(str(data)) != fp


def test_pasta_de_modelos_vazia_nao_tem_impressao_digital(tmp_path):
    """Foi o estado da instalação copiada de outra máquina sem os modelos."""
    (tmp_path / "data" / "models" / "sharpness").mkdir(parents=True)
    assert models_fingerprint(str(tmp_path / "data")) is None


# ── resumo e mistura ─────────────────────────────────────────────────

def _face(version=None, backend=None, com_ofiq=True):
    f = FaceInfo(image_path="/x.png", identity="a")
    if com_ofiq:
        f.ofiq_info = OFIQInfo(unified_quality_score_scalar=1.0, version=version, backend=backend)
    return f


def test_resumo_por_origem():
    faces = [_face("1.0.3", "executable"), _face("1.0.3", "executable"),
             _face(), _face(com_ofiq=False)]
    assert summarize_ofiq_versions(faces) == {
        "1.0.3 (executable)": 2, "desconhecida": 1,
    }


def test_mistura_gera_aviso_e_origem_unica_nao():
    assert mixed_versions_message({"1.0.3 (executable)": 5}) is None
    aviso = mixed_versions_message({"1.0.3 (executable)": 5, "desconhecida": 2})
    assert aviso and "desconhecida" in aviso and "1.0.3" in aviso


# ── carimbo no serviço ───────────────────────────────────────────────

def test_servico_carimba_versao_e_modo_em_cada_face(tmp_path, monkeypatch):
    exe = _instalacao_ofiq(tmp_path / "OFIQ-1.0.3")
    servico = OFIQService(str(exe), str(tmp_path / "OFIQ-1.0.3" / "data"))
    assert servico.version == "1.0.3"

    imagem = tmp_path / "foto.png"
    imagem.write_bytes(b"png")
    linha = (FIXTURES / "ofiqsampleapp_v1_0_3_b-01-smile.csv").read_text(encoding="utf-8")

    def _ofiq_falso(cmd, **_kwargs):
        # O serviço copia a imagem como img_00000.png; o CSV responde por ela.
        saida = Path(cmd[cmd.index("-o") + 1])
        saida.write_text(linha.replace("b-01-smile.png", "img_00000.png"), encoding="utf-8")

    monkeypatch.setattr(ofiq_service_module.subprocess, "run", _ofiq_falso)
    face = FaceInfo(image_path=str(imagem), identity="a")
    servico.process_batch([face])

    assert face.ofiq_info is not None
    assert face.ofiq_info.version == "1.0.3"
    assert face.ofiq_info.backend == "executable"
    assert face.ofiq_info.head_pose_yaw_raw is not None

    prov = servico.provenance()
    assert prov["ofiq_version"] == "1.0.3"
    assert prov["ofiq_backend"] == "executable"
    assert prov["ofiq_models_sha256"] == models_fingerprint(str(tmp_path / "OFIQ-1.0.3" / "data"))


@pytest.mark.skipif(not Path(r"C:\OFIQ-1.0.3\Version.txt").is_file(),
                    reason="instalação real do OFIQ 1.0.3 ausente")
def test_instalacao_real_103():
    exe = r"C:\OFIQ-1.0.3\install_x86_64\Release\bin\OFIQSampleApp.exe"
    assert detect_ofiq_version(exe) == "1.0.3"
