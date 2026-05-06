from PyQt6 import QtCore


class FilterWorker(QtCore.QObject):
    """Aplica `FilterConfig` no Dataset em background.
    Suporta cancelamento cooperativo via `stop()`.
    """

    finished = QtCore.pyqtSignal(object)  # dataset filtrado, ou None se cancelado
    progress = QtCore.pyqtSignal(int)     # % de progresso
    error = QtCore.pyqtSignal(str)

    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self._is_running = True

    def stop(self):
        self._is_running = False

    def _progress_callback(self, pct: int):
        if not self._is_running:
            raise InterruptedError("Filtro cancelado pelo usuário.")
        self.progress.emit(pct)

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if hasattr(self.dataset, "apply_filter"):
                result = self.dataset.apply_filter(
                    self.config, callback=self._progress_callback
                )
                self.finished.emit(result)
            else:
                # Fallback de segurança caso o método não exista.
                self.finished.emit(self.dataset)
        except InterruptedError:
            self.finished.emit(None)
        except Exception as e:
            self.error.emit(str(e))
