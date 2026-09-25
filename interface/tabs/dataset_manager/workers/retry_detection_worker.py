"""Worker da nova tentativa de detecção em resolução maior.

Duas coisas caras acontecem aqui, e nenhuma pode rodar na thread da
interface: carregar os modelos ONNX na resolução pedida (~5–10 s, uma vez
por resolução) e refazer a detecção nas quatro orientações de cada imagem.

Sem regra própria: só orquestra o `UndetectedRetryService`.
"""

from __future__ import annotations

import traceback
from typing import List, Optional, Sequence

from PyQt6 import QtCore


class RetryDetectionWorker(QtCore.QObject):
    """Repete a detecção nas faces pedidas. Emite `RetryOutcome` (ou None)."""

    progress_update = QtCore.pyqtSignal(str, int)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        servico,
        faces: Sequence,
        *,
        det_size: int,
        baseline_det_size: int,
        models: Optional[List[str]] = None,
        dataset=None,
    ) -> None:
        super().__init__()
        self._servico = servico
        self._faces = list(faces)
        self._det_size = int(det_size)
        self._baseline = int(baseline_det_size)
        self._models = models
        self._dataset = dataset
        self._cancelado = False

    def stop(self) -> None:
        """Cancelamento cooperativo (ligado ao botão Cancelar)."""
        self._cancelado = True

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            if not self._servico.has_service_for(self._det_size):
                # A espera da carga dos modelos precisa estar na tela: sem
                # aviso, o usuário vê a janela parada e conclui que travou.
                self.progress_update.emit(
                    f"Carregando o detector em {self._det_size} px "
                    "(uma vez por resolução)...", 1,
                )

            def _tick(indice: int, total: int, caminho: str) -> None:
                pct = 1 + int((indice / max(total, 1)) * 98)
                nome = caminho.replace("\\", "/").rsplit("/", 1)[-1]
                self.progress_update.emit(
                    f"Tentando a {self._det_size} px ({min(indice + 1, total)}"
                    f"/{total}): {nome}", pct,
                )

            resultado = self._servico.retry(
                self._faces,
                det_size=self._det_size,
                baseline_det_size=self._baseline,
                models=self._models,
                dataset=self._dataset,
                progress_callback=_tick,
                should_cancel=lambda: self._cancelado,
                error_callback=lambda msg: self.progress_update.emit(f"⚠ {msg}", 0),
            )
            self.progress_update.emit(
                f"{len(resultado.recovered)} de {resultado.n_tentadas} "
                f"recuperada(s) a {self._det_size} px.", 100,
            )
            self.finished.emit(resultado)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            # Sem isto a tela de progresso fica aberta para sempre.
            self.finished.emit(None)
