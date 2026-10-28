"""A interface mostra a imagem na mesma orientação em que a detecção a viu.

Foto de câmera segurada de lado é gravada deitada, com `Orientation` no
EXIF. O OpenCV aplica a etiqueta — caixa, pontos e OFIQ ficam no sistema
da foto de pé —, e `QPixmap(caminho)` não: um dataset inteiro aparecia
deitado na galeria, com a caixa da face fora do rosto.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtWidgets  # noqa: E402

from interface.oriented_image import (  # noqa: E402
    ler_imagem_orientada,
    ler_pixmap_orientado,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _foto_deitada(caminho: Path, orientacao: int) -> None:
    """60 de largura × 40 de altura gravados, marca vermelha no canto
    superior esquerdo do arquivo, EXIF dizendo como girar."""
    pixels = np.full((40, 60, 3), 255, np.uint8)
    pixels[:8, :8] = (255, 0, 0)  # RGB
    exif = Image.Exif()
    exif[0x0112] = orientacao
    Image.fromarray(pixels).save(caminho, "JPEG", quality=95, exif=exif)


@pytest.mark.parametrize("orientacao", [1, 3, 6, 8])
def test_exibicao_coincide_com_o_que_o_opencv_ve(app, tmp_path, orientacao):
    p = tmp_path / f"foto_{orientacao}.JPG"
    _foto_deitada(p, orientacao)

    visto = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    exibido = ler_imagem_orientada(p)
    assert (exibido.width(), exibido.height()) == (visto.shape[1], visto.shape[0])

    # A marca está no mesmo canto nas duas leituras.
    alto, largo = visto.shape[:2]
    for x, y in [(2, 2), (largo - 3, 2), (2, alto - 3), (largo - 3, alto - 3)]:
        vermelho_cv = visto[y, x, 2] > 200 and visto[y, x, 1] < 80
        cor = exibido.pixelColor(x, y)
        vermelho_qt = cor.red() > 200 and cor.green() < 80
        assert vermelho_cv == vermelho_qt, (orientacao, x, y)


def test_foto_com_orientacao_6_aparece_de_pe(app, tmp_path):
    p = tmp_path / "DSCN9675.JPG"
    _foto_deitada(p, 6)
    pix = ler_pixmap_orientado(p)
    assert (pix.width(), pix.height()) == (40, 60)


def test_arquivo_ilegivel_devolve_nulo(app, tmp_path):
    p = tmp_path / "x.jpg"
    p.write_bytes(b"nao e imagem")
    assert ler_pixmap_orientado(p).isNull()
    assert ler_pixmap_orientado(tmp_path / "nao_existe.jpg").isNull()


def test_interface_nao_carrega_imagem_de_arquivo_com_qpixmap_direto():
    """Trava: um widget novo com `QPixmap(caminho)` voltaria a mostrar
    deitada a foto que o resto do sistema vê de pé."""
    raiz = Path(__file__).resolve().parents[1] / "interface"
    padrao = re.compile(r"QPixmap\(\s*(?:str\()?[a-z]")
    achados = []
    for arq in raiz.rglob("*.py"):
        if arq.name == "oriented_image.py":
            continue
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            if padrao.search(linha) and not linha.lstrip().startswith("#"):
                achados.append(f"{arq.relative_to(raiz)}:{n}: {linha.strip()}")
    assert not achados, "use interface.oriented_image:\n" + "\n".join(achados)
