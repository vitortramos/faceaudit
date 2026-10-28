"""Leitura de imagem para EXIBIÇÃO, na orientação em que a imagem foi vista
pelo resto do sistema.

Foto de câmera segurada de lado é gravada deitada, com a etiqueta EXIF
`Orientation` dizendo como girá-la. O OpenCV aplica a etiqueta ao
decodificar — então detecção, caixa da face, pontos, OFIQ e recortes estão
todos no sistema da foto DE PÉ. `QPixmap(caminho)` ignora a etiqueta: a
interface mostrava a foto deitada, e a caixa desenhada sobre ela caía fora
da face. Caso real: as 2515 fotos de um dataset de referência, todas com
`Orientation=6`.

Toda imagem de dataset exibida na interface passa por aqui.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PyQt6 import QtGui


def ler_imagem_orientada(caminho: Union[str, Path]) -> QtGui.QImage:
    """`QImage` com a orientação EXIF aplicada; nula se não der para ler.

    Pode ser chamada fora da thread da interface (QImage, não QPixmap).
    """
    leitor = QtGui.QImageReader(str(caminho))
    leitor.setAutoTransform(True)
    imagem = leitor.read()
    return imagem if not imagem.isNull() else QtGui.QImage()


def ler_pixmap_orientado(caminho: Union[str, Path]) -> QtGui.QPixmap:
    """`QPixmap` com a orientação EXIF aplicada; nulo se não der para ler.

    Só na thread da interface — QPixmap não pode ser criado fora dela.
    """
    imagem = ler_imagem_orientada(caminho)
    return QtGui.QPixmap.fromImage(imagem) if not imagem.isNull() else QtGui.QPixmap()
