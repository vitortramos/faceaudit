"""Worker da extração de faces de vídeo (Fase 5).

Roda `VideoFrameFaceExtractionService.extract(...)` numa thread, com
progresso por quadro e cancelamento cooperativo (cancel_check).
"""

from __future__ import annotations

import traceback

from PyQt6 import QtCore


class VideoExtractionWorker(QtCore.QObject):
    progress_update = QtCore.pyqtSignal(str, int)   # msg, %
    finished = QtCore.pyqtSignal(object)            # VideoExtractionResult | None
    error = QtCore.pyqtSignal(str)
    cancelled = QtCore.pyqtSignal()

    def __init__(self, service, config, targets):
        super().__init__()
        self._service = service
        self._config = config
        self._targets = targets
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            def cb(msg, pct):
                self.progress_update.emit(str(msg), int(pct))

            self.progress_update.emit("Iniciando extração…", 0)
            result = self._service.extract(
                self._config, self._targets,
                progress_callback=cb, cancel_check=lambda: self._stop,
            )
            if self._stop:
                self.cancelled.emit()
            if result is not None:
                self.progress_update.emit(
                    f"Concluído: {result.n_accepted_auto} face(s) aceita(s) · "
                    f"{result.n_frames_sampled} quadros amostrados · "
                    f"{result.n_no_detection_frames} sem face.", 100)
            self.finished.emit(result)
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))
            self.finished.emit(None)
