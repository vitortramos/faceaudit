"""FitWorker — ajuste automático do pipeline à questionada, fora da
thread da interface (dezenas de segundos: ~150 avaliações numa amostra).

Recebe uma CÓPIA do config. Cancelável: `stop()` faz o serviço
interromper na próxima avaliação.

Sinais: `progress(str)`, `finished(FitResult)`, `cancelled()`, `error(str)`.
"""

from __future__ import annotations

from PyQt6 import QtCore

from faceaudit.services.image_pipeline_fit_service import (
    FitCancelled,
    fit_to_questioned,
)


class FitWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    cancelled = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, dataset, config, *, ff_service=None):
        super().__init__()
        self._dataset = dataset
        self._config = config
        self._ff_service = ff_service
        self._parar = False

    def stop(self):
        self._parar = True

    def run(self):
        try:
            resultado = fit_to_questioned(
                self._config, self._dataset,
                ff_service=self._ff_service,
                progress=self.progress.emit,
                should_stop=lambda: self._parar,
            )
            self.finished.emit(resultado)
        except FitCancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 — vira mensagem na interface
            self.error.emit(str(e))
