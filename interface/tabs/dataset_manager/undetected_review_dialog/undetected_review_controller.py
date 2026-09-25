"""Controlador da revisão das imagens sem face detectada.

Junta três coisas e nada mais: a política que vem do backend, a view, e o
worker que decodifica as miniaturas. A decisão sai como duas listas de
`FaceInfo` — incluir e descartar — para quem chamou aplicar.

Cancelar (fechar a janela pelo X) devolve TODAS como incluídas quando a
inclusão é permitida. Descartar em silêncio o que o usuário não decidiu
seria apagar imagens que ele escolheu ingerir; incluir com a marca de erro
é o comportamento antigo, que ele conhece, e nada se perde.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from faceaudit.services.undetected_review_policy import (
    UndetectedInclusionPolicy,
    resolve_inclusion_policy,
)

from ..workers.thumbnail_load_worker import ThumbnailLoadWorker
from .undetected_review_view import UndetectedReviewView

_LADO_GRANDE = 1000


class UndetectedReviewController(QtCore.QObject):
    """Abre a revisão e devolve a decisão. Sem regra de domínio aqui."""

    def __init__(
        self,
        parent_view: Optional[QtWidgets.QWidget],
        *,
        faces: Sequence,
        config,
        dataset=None,
        n_total: int = 0,
    ) -> None:
        super().__init__(parent_view)
        self._faces = list(faces)
        self._dataset = dataset
        self._politica: UndetectedInclusionPolicy = resolve_inclusion_policy(config)

        # Inclusão bloqueada: ninguém entra, e a view não deixa marcar.
        self._marcadas: List[bool] = [False] * len(self._faces)

        self._caminhos = [self._resolver_caminho(f) for f in self._faces]
        self._grandes: dict = {}

        self.view = UndetectedReviewView(
            parent_view,
            n_imagens=len(self._faces),
            n_total=n_total or len(self._faces),
            inclusao_permitida=self._politica.allowed,
            motivo=self._politica.reason,
        )
        self.view.popular([self._rotulo(c) for c in self._caminhos])
        self.view.atualizar_contador(0, len(self._faces))

        self.view.selecao_mudou.connect(self._ao_selecionar)
        self.view.marcacao_mudou.connect(self._ao_marcar)
        self.view.marcar_todas.connect(self._marcar_todas)
        self.view.desmarcar_todas.connect(self._desmarcar_todas)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ThumbnailLoadWorker] = None
        self._iniciar_miniaturas()

    # ── Ciclo de vida ────────────────────────────────────────────────────

    def exec(self) -> bool:
        try:
            aceito = self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted
        finally:
            self._encerrar_worker()
        if not aceito and self._politica.allowed:
            # Fechou sem concluir: mantém o comportamento antigo (entram
            # todas, marcadas com erro) em vez de descartar por omissão.
            self._marcadas = [True] * len(self._faces)
        return aceito

    def decision(self) -> Tuple[List, List]:
        """(a incluir, a descartar) — nesta ordem."""
        incluir = [f for f, m in zip(self._faces, self._marcadas) if m]
        descartar = [f for f, m in zip(self._faces, self._marcadas) if not m]
        return incluir, descartar

    @property
    def inclusao_permitida(self) -> bool:
        return self._politica.allowed

    @property
    def motivo_da_politica(self) -> str:
        return self._politica.reason

    # ── Miniaturas ───────────────────────────────────────────────────────

    def _iniciar_miniaturas(self) -> None:
        if not self._caminhos:
            return
        self._thread = QtCore.QThread(self)
        self._worker = ThumbnailLoadWorker(self._caminhos)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.thumbnail_ready.connect(self._ao_receber_miniatura)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    @QtCore.pyqtSlot(int, QtGui.QImage)
    def _ao_receber_miniatura(self, indice: int, imagem: QtGui.QImage) -> None:
        self.view.definir_miniatura(indice, imagem)

    def _encerrar_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None

    # ── Reações da view ──────────────────────────────────────────────────

    def _ao_selecionar(self, indice: int) -> None:
        if not (0 <= indice < len(self._caminhos)):
            return
        self.view.mostrar_imagem(
            self._imagem_grande(indice), self._descricao(indice),
        )

    def _ao_marcar(self, indice: int, marcada: bool) -> None:
        if not (0 <= indice < len(self._marcadas)):
            return
        if marcada and not self._politica.allowed:
            return
        self._marcadas[indice] = bool(marcada)
        self.view.definir_marcacao(indice, bool(marcada))
        self._atualizar_contador()

    def _marcar_todas(self) -> None:
        if not self._politica.allowed:
            return
        for indice in range(len(self._marcadas)):
            self._marcadas[indice] = True
            self.view.definir_marcacao(indice, True)
        self._atualizar_contador()

    def _desmarcar_todas(self) -> None:
        for indice in range(len(self._marcadas)):
            self._marcadas[indice] = False
            self.view.definir_marcacao(indice, False)
        self._atualizar_contador()

    def _atualizar_contador(self) -> None:
        self.view.atualizar_contador(sum(self._marcadas), len(self._marcadas))

    # ── Leitura de imagem ────────────────────────────────────────────────

    def _imagem_grande(self, indice: int) -> QtGui.QImage:
        """Lê reduzida e guarda em cache.

        Reduz na decodificação: a foto de 24 MP nunca precisa existir em
        memória em tamanho cheio só para caber num painel de 600 px, e
        navegar com as setas leria de novo a cada passo sem o cache.
        """
        if indice in self._grandes:
            return self._grandes[indice]
        leitor = QtGui.QImageReader(self._caminhos[indice])
        leitor.setAutoTransform(True)
        tamanho = leitor.size()
        if tamanho.isValid() and tamanho.width() > 0 and tamanho.height() > 0:
            escala = min(
                _LADO_GRANDE / tamanho.width(), _LADO_GRANDE / tamanho.height(), 1.0,
            )
            leitor.setScaledSize(
                QtCore.QSize(
                    max(1, int(tamanho.width() * escala)),
                    max(1, int(tamanho.height() * escala)),
                )
            )
        imagem = leitor.read()
        self._grandes[indice] = imagem if not imagem.isNull() else QtGui.QImage()
        return self._grandes[indice]

    # ── Texto ────────────────────────────────────────────────────────────

    def _resolver_caminho(self, face) -> str:
        if self._dataset is not None:
            try:
                return str(self._dataset.resolve_image_path(face))
            except Exception:
                pass
        return str(getattr(face, "image_path", "") or "")

    @staticmethod
    def _rotulo(caminho: str) -> str:
        return Path(caminho).name or caminho

    def _descricao(self, indice: int) -> str:
        caminho = self._caminhos[indice]
        face = self._faces[indice]
        imagem = self._grandes.get(indice)
        partes = [f"<b>{self._rotulo(caminho)}</b>"]
        if imagem is not None and not imagem.isNull():
            partes.append(f"exibida a {imagem.width()}×{imagem.height()}")
        identidade = getattr(face, "identity", None)
        if identidade:
            partes.append(f"identidade: {identidade}")
        partes.append(f"<span>{caminho}</span>")
        return " · ".join(partes[:-1]) + "<br>" + partes[-1]
