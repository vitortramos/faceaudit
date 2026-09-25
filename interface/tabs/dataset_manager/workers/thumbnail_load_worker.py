"""Worker que decodifica miniaturas fora da thread da interface.

As imagens da revisão são fotos originais, de vários megapixels cada.
Decodificar isso na thread da GUI congela a janela justamente enquanto o
usuário tenta olhar as imagens.

Emite `QImage` e não `QPixmap`: `QPixmap` só pode ser criado na thread da
interface. A conversão acontece do outro lado do sinal.

`QImageReader.setScaledSize` faz a redução durante a decodificação — o
formato permite ler direto em escala reduzida, então a foto inteira nunca
chega à memória em tamanho cheio.
"""

from __future__ import annotations

from typing import List, Sequence

from PyQt6 import QtCore, QtGui


class ThumbnailLoadWorker(QtCore.QObject):
    """Decodifica uma lista de caminhos em miniaturas, uma a uma."""

    # (índice na lista original, imagem) — o índice é o que permite ao
    # receptor casar a miniatura com a linha certa mesmo fora de ordem.
    thumbnail_ready = QtCore.pyqtSignal(int, QtGui.QImage)
    finished = QtCore.pyqtSignal()

    def __init__(self, paths: Sequence[str], lado: int = 160) -> None:
        super().__init__()
        self._paths: List[str] = [str(p) for p in paths]
        self._lado = int(lado)
        self._cancelled = False

    @QtCore.pyqtSlot()
    def cancel(self) -> None:
        """Cancelamento cooperativo — a janela pode fechar antes do fim."""
        self._cancelled = True

    @QtCore.pyqtSlot()
    def run(self) -> None:
        for indice, caminho in enumerate(self._paths):
            if self._cancelled:
                break
            imagem = self._ler_reduzida(caminho)
            # Imagem ilegível também é resposta: emitir vazio deixa o
            # receptor trocar o placeholder por "não pôde ser lida", em vez
            # de a linha ficar para sempre no estado "carregando".
            self.thumbnail_ready.emit(indice, imagem)
        self.finished.emit()

    def _ler_reduzida(self, caminho: str) -> QtGui.QImage:
        leitor = QtGui.QImageReader(caminho)
        leitor.setAutoTransform(True)    # respeita a orientação do EXIF
        tamanho = leitor.size()
        if tamanho.isValid() and tamanho.width() > 0 and tamanho.height() > 0:
            escala = min(
                self._lado / tamanho.width(), self._lado / tamanho.height(), 1.0,
            )
            leitor.setScaledSize(
                QtCore.QSize(
                    max(1, int(tamanho.width() * escala)),
                    max(1, int(tamanho.height() * escala)),
                )
            )
        imagem = leitor.read()
        return imagem if not imagem.isNull() else QtGui.QImage()
