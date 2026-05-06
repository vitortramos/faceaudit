from typing import Callable, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QProgressDialog, QWidget


class WorkerProgressRunner(QtCore.QObject):
    """Unifica QThread + worker + QProgressDialog num único helper.

    Contrato esperado do worker (todos opcionais exceto `finished`/`error`):
      - `finished(object)`        — payload final (ou None se cancelado)
      - `error(str)`              — mensagem de erro
      - `progress_update(str,int)`— (label, %) — usado pelos workers ricos
      - `progress(int)`           — apenas % — usado pelo FilterWorker legado
      - `stop()`                  — cancelamento cooperativo

    Cada runner gerencia seu próprio dialog/thread e dispara os callbacks
    `on_finished`/`on_error` no thread da UI quando o worker termina.
    """

    def __init__(
        self,
        parent: QWidget,
        worker: QtCore.QObject,
        *,
        label: str,
        allow_cancel: bool = False,
        determinate: bool = True,
        wait_cursor: bool = False,
        on_finished: Optional[Callable[[object], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(parent)
        self._worker = worker
        self._on_finished = on_finished
        self._on_error = on_error
        self._wait_cursor = wait_cursor
        self._cursor_active = False
        self._completed = False

        cancel_text = "Cancelar" if allow_cancel else None
        max_val = 100 if determinate else 0
        self._dialog = QProgressDialog(label, cancel_text, 0, max_val, parent)
        self._dialog.setWindowTitle("Aguarde")
        self._dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._dialog.setMinimumDuration(0)
        if not allow_cancel:
            self._dialog.setCancelButton(None)
        if not determinate:
            self._dialog.setRange(0, 0)
            self._dialog.setValue(0)

        # Thread parented ao runner; runner parented ao parent (view).
        self._thread = QtCore.QThread(self)
        worker.setParent(None)  # vai pra outra thread, não pode ter parent na main
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)

        # Sinais opcionais — só ligados se existirem na classe do worker.
        cls = type(worker)
        if hasattr(cls, "progress_update"):
            worker.progress_update.connect(self._on_progress_update)
        if hasattr(cls, "progress"):
            worker.progress.connect(self._dialog.setValue)

        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)

        # Cancel: amarra o botão à .stop() do worker quando ambos existem.
        if allow_cancel and hasattr(cls, "stop"):
            self._dialog.canceled.connect(worker.stop)

        # Lifecycle: thread termina em qualquer um dos sinais terminais.
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    def start(self) -> None:
        if self._wait_cursor:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self._cursor_active = True
        self._dialog.show()
        QApplication.processEvents()  # paint imediato antes do thread.start
        self._thread.start()

    def stop_and_wait(self, timeout_ms: int = 3000) -> None:
        """Cancela cooperativamente e aguarda a thread terminar. Idempotente."""
        if hasattr(self._worker, "stop"):
            try:
                self._worker.stop()
            except RuntimeError:
                pass
        try:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(timeout_ms)
        except RuntimeError:
            pass

    def _on_progress_update(self, msg: str, pct: int) -> None:
        self._dialog.setLabelText(msg)
        self._dialog.setValue(pct)

    def _on_worker_finished(self, payload) -> None:
        if self._completed:
            return
        self._completed = True
        self._teardown_ui()
        if self._on_finished is not None:
            self._on_finished(payload)

    def _on_worker_error(self, msg: str) -> None:
        if self._completed:
            return
        self._completed = True
        self._teardown_ui()
        if self._on_error is not None:
            self._on_error(msg)

    def _teardown_ui(self) -> None:
        if self._cursor_active:
            QApplication.restoreOverrideCursor()
            self._cursor_active = False
        self._dialog.close()
