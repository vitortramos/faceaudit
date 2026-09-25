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

from faceaudit.enums.undetected_review_mode import UndetectedReviewMode
from faceaudit.services.undetected_review_policy import (
    UndetectedInclusionPolicy,
    resolve_inclusion_policy,
)
from faceaudit.services.undetected_retry_service import (
    RESOLUCOES_DE_RETENTATIVA,
    UndetectedRetryService,
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
        ff_service_factory=None,
        modo: UndetectedReviewMode = UndetectedReviewMode.CREATION,
    ) -> None:
        super().__init__(parent_view)
        self._faces = list(faces)
        self._dataset = dataset
        self._config = config
        self._modo = modo
        self._politica: UndetectedInclusionPolicy = resolve_inclusion_policy(
            config, modo,
        )

        # Inclusão bloqueada: ninguém entra, e a view não deixa marcar.
        self._marcadas: List[bool] = [False] * len(self._faces)
        # Índices recuperados numa segunda tentativa. Recuperada tem caixa da
        # face, logo tem recorte a gerar — a inclusão deixa de estar
        # bloqueada para ELA, mesmo em dataset de recortes. O bloqueio é
        # "enquanto não detectada", não "para sempre".
        self._recuperadas: dict = {}
        self._retry = (
            UndetectedRetryService(ff_service_factory)
            if ff_service_factory is not None else None
        )
        self._baseline_det_size = int(getattr(config, "ff_det_size", 320) or 320)
        self._resolucoes = tuple(
            r for r in RESOLUCOES_DE_RETENTATIVA if r > self._baseline_det_size
        ) if self._retry is not None else ()

        self._caminhos = [self._resolver_caminho(f) for f in self._faces]
        self._grandes: dict = {}

        self.view = UndetectedReviewView(
            parent_view,
            n_imagens=len(self._faces),
            n_total=n_total or len(self._faces),
            inclusao_permitida=self._politica.allowed,
            motivo=self._politica.reason,
            det_size_atual=self._baseline_det_size,
            resolucoes=self._resolucoes,
            modo=modo,
        )
        self.view.popular([self._rotulo(c) for c in self._caminhos])
        self.view.atualizar_contador(0, len(self._faces))

        self.view.selecao_mudou.connect(self._ao_selecionar)
        self.view.marcacao_mudou.connect(self._ao_marcar)
        self.view.marcar_todas.connect(self._marcar_todas)
        self.view.desmarcar_todas.connect(self._desmarcar_todas)
        self.view.tentar_de_novo.connect(self._ao_pedir_retentativa)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[ThumbnailLoadWorker] = None
        self._iniciar_miniaturas()

    # ── Ciclo de vida ────────────────────────────────────────────────────

    def exec(self) -> bool:
        try:
            aceito = self.view.exec() == QtWidgets.QDialog.DialogCode.Accepted
        finally:
            self._encerrar_worker()
            if self._retry is not None:
                # Solta os modelos ONNX carregados para as retentativas.
                self._retry.release()
        if not aceito:
            # Fechou sem concluir: mantém o comportamento antigo (entram
            # todas, marcadas com erro) em vez de descartar por omissão.
            # As recuperadas entram em qualquer caso — elas TÊM biometria, e
            # descartá-las jogaria fora justamente o trabalho da retentativa.
            for indice in range(len(self._faces)):
                if self._politica.allowed or indice in self._recuperadas:
                    self._marcadas[indice] = True
        return aceito

    def decision(self) -> Tuple[List, List]:
        """(a incluir, a descartar) — nesta ordem."""
        incluir = [f for f, m in zip(self._faces, self._marcadas) if m]
        descartar = [f for f, m in zip(self._faces, self._marcadas) if not m]
        return incluir, descartar

    def recovered_faces(self) -> List:
        """Faces que ganharam biometria numa segunda tentativa.

        Quem chamou precisa saber quais são: elas chegaram DEPOIS dos
        estágios de qualidade e de recorte, e têm de passar por eles antes
        da gravação — senão o dataset fica heterogêneo por dentro, com
        embedding mas sem métrica de qualidade, e sem nada na tela dizendo.
        """
        return [
            self._faces[i] for i in sorted(self._recuperadas)
            if self._marcadas[i] or not self._modo.permite_decidir_inclusao
        ]

    def _pode_incluir(self, indice: int) -> bool:
        if not self._modo.permite_decidir_inclusao:
            # A face já está no dataset; não há inclusão a decidir. O que a
            # retentativa muda aqui é a BIOMETRIA dela, não a presença.
            return False
        return self._politica.allowed or indice in self._recuperadas

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
        # A permissão é por imagem: bloqueada de saída, liberada se esta
        # imagem foi recuperada numa segunda tentativa.
        self.view.habilitar_caixa_do_painel(self._pode_incluir(indice))

    def _ao_marcar(self, indice: int, marcada: bool) -> None:
        if not (0 <= indice < len(self._marcadas)):
            return
        if marcada and not self._pode_incluir(indice):
            return
        self._marcadas[indice] = bool(marcada)
        self.view.definir_marcacao(indice, bool(marcada))
        self._atualizar_contador()

    def _marcar_todas(self) -> None:
        marcou = False
        for indice in range(len(self._marcadas)):
            if not self._pode_incluir(indice):
                continue
            self._marcadas[indice] = True
            self.view.definir_marcacao(indice, True)
            marcou = True
        if marcou:
            self._atualizar_contador()

    def _desmarcar_todas(self) -> None:
        for indice in range(len(self._marcadas)):
            self._marcadas[indice] = False
            self.view.definir_marcacao(indice, False)
        self._atualizar_contador()

    def _atualizar_contador(self) -> None:
        self.view.atualizar_contador(sum(self._marcadas), len(self._marcadas))

    # ── Nova tentativa em outra resolução ────────────────────────────────

    def _ao_pedir_retentativa(self, det_size: int, so_selecionada: bool) -> None:
        """Dispara a retentativa em worker e aplica o resultado na grade."""
        if self._retry is None:
            return

        indices = self._indices_para_retentativa(so_selecionada)
        if not indices:
            self.view.definir_aviso_de_retentativa(
                "Nada a tentar: as pendentes já foram recuperadas."
            )
            return

        faces = [self._faces[i] for i in indices]
        self.view.definir_retentativa_habilitada(False)
        self.view.definir_aviso_de_retentativa(
            f"Tentando em {len(faces)} imagem(ns) a {det_size} px…"
        )

        from ..workers.retry_detection_worker import RetryDetectionWorker

        worker = RetryDetectionWorker(
            self._retry, faces,
            det_size=det_size,
            baseline_det_size=self._baseline_det_size,
            models=list(getattr(self._config, "ff_models", None) or []) or None,
            dataset=self._dataset,
        )
        self._executar_retentativa(worker, indices, det_size)

    def _executar_retentativa(self, worker, indices: List[int], det_size: int) -> None:
        """Roda o worker numa thread, com a janela viva e informando.

        A janela continua aberta durante a rodada: é dela que o usuário
        acompanha o progresso, e fechá-la para mostrar uma barra tiraria da
        vista justamente as imagens em questão.
        """
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress_update.connect(
            lambda msg, _pct: self.view.definir_aviso_de_retentativa(msg)
        )
        worker.finished.connect(
            lambda resultado: self._ao_terminar_retentativa(
                resultado, indices, det_size,
            )
        )
        worker.finished.connect(thread.quit)
        worker.error.connect(
            lambda msg: self.view.definir_aviso_de_retentativa(f"Falhou: {msg}")
        )
        # Referências fortes: sem elas, thread e worker são coletados no meio
        # da execução e o Qt derruba o processo.
        self._retry_thread = thread
        self._retry_worker = worker
        thread.start()

    def _ao_terminar_retentativa(
        self, resultado, indices: List[int], det_size: int,
    ) -> None:
        self.view.definir_retentativa_habilitada(True)
        if resultado is None:
            return

        recuperados_ids = {id(f) for f in resultado.recovered}
        n_novas = 0
        for indice in indices:
            if id(self._faces[indice]) not in recuperados_ids:
                continue
            self._recuperadas[indice] = det_size
            self.view.marcar_como_recuperada(indice, det_size)
            # Recuperada entra por padrão: ela tem biometria, e o motivo de
            # estar nesta janela deixou de existir.
            self._marcadas[indice] = True
            self.view.definir_marcacao(indice, True)
            n_novas += 1

        self._atualizar_contador()
        atual = self.view.lista.currentRow()
        if atual >= 0:
            self.view.habilitar_caixa_do_painel(self._pode_incluir(atual))

        restantes = len(indices) - n_novas
        if n_novas:
            self.view.definir_aviso_de_retentativa(
                f"{n_novas} face(s) encontrada(s) a {det_size} px. "
                f"{restantes} continua(m) sem face."
                + (
                    " As recuperadas já podem ser incluídas — agora existe "
                    "recorte a gerar para elas."
                    if self._politica.blocked
                    and self._modo.permite_decidir_inclusao else ""
                )
            )
        else:
            self.view.definir_aviso_de_retentativa(
                f"Nenhuma face encontrada a {det_size} px nas "
                f"{len(indices)} imagem(ns) tentada(s)."
                + (" Vale tentar numa resolução maior." if det_size
                   != max(self._resolucoes, default=det_size) else "")
            )

    def _indices_para_retentativa(self, so_selecionada: bool) -> List[int]:
        """Pendentes a tentar; as já recuperadas ficam de fora."""
        if so_selecionada:
            indice = self.view.lista.currentRow()
            if indice < 0 or indice in self._recuperadas:
                return []
            return [indice]
        return [
            i for i in range(len(self._faces)) if i not in self._recuperadas
        ]

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
