"""A cópia que vai ao OFIQ tem extensão que o OFIQSampleApp aceita.

Ele escolhe os arquivos da pasta pela extensão — medido na 1.0.3: só
`.jpg`, `.jpeg` e `.png`, em minúsculas — e pula os outros sem aviso. Um
dataset com 290 `.jfif` saiu com 290 faces "OFIQ não produziu resultado";
`.JPG` em maiúsculas (comum em câmera) teria o mesmo destino.

O executável é simulado com o mesmo filtro, para o teste rodar sem OFIQ.
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import pytest

from faceaudit.models.face_info import FaceInfo
from faceaudit.services import ofiq_service as mod
from faceaudit.services.ofiq_service import OFIQService, extensao_para_ofiq

_ACEITAS_PELO_OFIQ = (".jpg", ".jpeg", ".png")  # sensível a maiúsculas


@pytest.mark.parametrize("nome, esperado", [
    ("a.jpg", (".jpg", False)), ("a.JPG", (".jpg", False)),
    ("a.jpeg", (".jpg", False)), ("a.jfif", (".jpg", False)),
    ("a.JFIF", (".jpg", False)), ("a.jpe", (".jpg", False)),
    ("a.png", (".png", False)), ("a.PNG", (".png", False)),
    ("a.bmp", (".png", True)), ("a.webp", (".png", True)),
    ("a.tif", (".png", True)), ("a.tiff", (".png", True)),
])
def test_extensao_para_ofiq(nome, esperado):
    assert extensao_para_ofiq(nome) == esperado
    assert esperado[0] in _ACEITAS_PELO_OFIQ


def _ofiq_falso(vistos):
    """Imita o OFIQSampleApp: lista a pasta, filtra pela extensão, grava CSV."""
    def run(cmd, **kwargs):
        pasta, saida = cmd[cmd.index("-i") + 1], cmd[cmd.index("-o") + 1]
        with open(saida, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Filename", "UnifiedQualityScore"])
            for nome in sorted(os.listdir(pasta)):
                if os.path.splitext(nome)[1] not in _ACEITAS_PELO_OFIQ:
                    continue
                img = cv2.imdecode(np.fromfile(os.path.join(pasta, nome), np.uint8),
                                   cv2.IMREAD_COLOR)
                vistos[nome] = None if img is None else img.copy()
                w.writerow([os.path.join(pasta, nome), "50"])
    return run


def _servico():
    s = OFIQService.__new__(OFIQService)
    s.exe_path, s.config_path = "ofiq.exe", "data"
    s.version, s.backend = "1.0.3", mod.BACKEND_EXECUTABLE
    return s


def test_todas_as_extensoes_aceitas_pela_ingestao_chegam_ao_ofiq(tmp_path, monkeypatch):
    from faceaudit.models.image_file_extensions import EXTENSOES_DE_IMAGEM

    original = np.zeros((40, 30, 3), np.uint8)
    original[5:15, 5:20] = (10, 200, 90)
    jpeg = cv2.imencode(".jpg", original)[1].tobytes()
    faces = []
    for ext in EXTENSOES_DE_IMAGEM + (".JPG", ".JFIF"):
        p = tmp_path / f"face{ext.replace('.', '_')}{ext}"
        if ext.lower() in mod._EXTENSOES_JPEG:
            p.write_bytes(jpeg)
        else:
            formato = ".tiff" if ext.lower() == ".tif" else ext.lower()
            p.write_bytes(cv2.imencode(formato, original)[1].tobytes())
        faces.append(FaceInfo(image_path=str(p), identity="x"))

    vistos = {}
    monkeypatch.setattr(mod.subprocess, "run", _ofiq_falso(vistos))
    _servico().process_batch(faces)

    sem = [os.path.basename(f.image_path) for f in faces if f.ofiq_info is None]
    assert not sem, f"faces sem OFIQ: {sem}"
    assert all(f.ofiq_info.version == "1.0.3" for f in faces)


def test_conversao_para_png_preserva_os_pixels(tmp_path, monkeypatch):
    original = np.random.default_rng(0).integers(0, 256, (24, 18, 3), dtype=np.uint8)
    p = tmp_path / "doc.bmp"
    cv2.imencode(".bmp", original)[1].tofile(str(p))

    vistos = {}
    monkeypatch.setattr(mod.subprocess, "run", _ofiq_falso(vistos))
    face = FaceInfo(image_path=str(p), identity="x")
    _servico().process_batch([face])

    (nome, img), = vistos.items()
    assert nome.endswith(".png")
    assert np.array_equal(img, original)
    assert face.ofiq_info is not None


def test_o_arquivo_do_dataset_nao_e_tocado(tmp_path, monkeypatch):
    p = tmp_path / "ID004_RENACH2016_39.jfif"
    p.write_bytes(cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes())
    antes = p.read_bytes()
    monkeypatch.setattr(mod.subprocess, "run", _ofiq_falso({}))
    _servico().process_batch([FaceInfo(image_path=str(p), identity="x")])
    assert p.exists() and p.read_bytes() == antes
