from typing import Callable, Optional

from PyQt6 import QtCore
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QProgressDialog, QWidget

# LogProgressDialog vive em strategy_manager (legado) mas é reusável.
# Importa lazy pra não pagar custo quando o caller pede só o
# QProgressDialog simples.
def _import_log_progress_dialog():
    from interface.tabs.strategy_manager.log_progress_dialog.log_progress_dialog import (
        LogProgressDialog,
    )
    return LogProgressDialog


class WorkerProgressRunner(QtCore.QObject):
    """Unifica QThread + worker + progress dialog num único helper.

    Dois sabores de UI:
    - **`QProgressDialog`** (default): janela compacta com barra + label
      + botão Cancelar opcional. Bom pra operações curtas com 1 etapa.
    - **`LogProgressDialog`** (`use_log_dialog=True`): janela maior com
      área de texto rolando os logs + barra indeterminada. Bom pra
      operações multi-etapa onde o usuário precisa ver o que está
      acontecendo (ex: ingestão de dataset que faz Read → Hash → FF →
      OFIQ → SER-FIQ → Save).

    Contrato esperado do worker (todos opcionais exceto
    `finished`/`error`):
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
        use_log_dialog: bool = False,
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
        self._use_log_dialog = use_log_dialog

        if use_log_dialog:
            self._dialog = self._build_log_dialog(parent, label, allow_cancel)
        else:
            self._dialog = self._build_progress_dialog(
                parent, label, allow_cancel, determinate
            )

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
            # Só faz sentido pro QProgressDialog; LogProgressDialog não
            # tem barra determinada.
            if not use_log_dialog:
                worker.progress.connect(self._dialog.setValue)

        worker.finished.connect(self._on_worker_finished)
        worker.error.connect(self._on_worker_error)

        # Cancel: amarra o botão à .stop() do worker quando ambos existem.
        # CRÍTICO: usa `Qt.DirectConnection` pra que `.stop()` execute na
        # main thread imediatamente, em vez de ser enfileirado no event
        # loop do thread do worker. Sem isso, o worker NUNCA processa o
        # slot porque está bloqueado no loop do `run()` (não dá tick no
        # event loop). Cancel fica "preso" até o worker terminar
        # naturalmente — usuário vê "Cancelar não funciona".
        # Seguro pra `stop()` que só seta uma flag bool — atomic em CPython
        # via GIL.
        if allow_cancel and hasattr(cls, "stop"):
            if use_log_dialog:
                # LogProgressDialog usa sinal próprio.
                self._dialog.cancellation_requested.connect(
                    worker.stop, QtCore.Qt.ConnectionType.DirectConnection,
                )
            else:
                self._dialog.canceled.connect(
                    worker.stop, QtCore.Qt.ConnectionType.DirectConnection,
                )

        # Lifecycle: thread termina em qualquer um dos sinais terminais.
        worker.finished.connect(self._thread.quit)
        worker.error.connect(self._thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.error.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    # ──────────────────────────────────────────────────────────────────
    # Construção do dialog
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_progress_dialog(
        parent: QWidget, label: str, allow_cancel: bool, determinate: bool
    ) -> QProgressDialog:
        cancel_text = "Cancelar" if allow_cancel else None
        max_val = 100 if determinate else 0
        dialog = QProgressDialog(label, cancel_text, 0, max_val, parent)
        dialog.setWindowTitle("Aguarde")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(0)
        # Default do Qt fecha o dialog quando setValue(100) é chamado.
        # Como vários callers passam por 100% no meio do fluxo (ex:
        # adapter termina leitura com 100% e depois pipeline continua
        # com FF a partir de 0%), desabilitamos esse comportamento —
        # o dialog só fecha quando _teardown_ui chama close().
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        if not allow_cancel:
            dialog.setCancelButton(None)
        if not determinate:
            dialog.setRange(0, 0)
            dialog.setValue(0)
        return dialog

    @staticmethod
    def _build_log_dialog(
        parent: QWidget, label: str, allow_cancel: bool
    ):
        LogProgressDialog = _import_log_progress_dialog()
        dialog = LogProgressDialog(title=label, parent=parent)
        if not allow_cancel:
            dialog.btn_action.setEnabled(False)
            dialog.btn_action.setVisible(False)
        return dialog

    # ──────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────

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

    # ──────────────────────────────────────────────────────────────────
    # Slots
    # ──────────────────────────────────────────────────────────────────

    def _on_progress_update(self, msg: str, pct: int) -> None:
        if self._use_log_dialog:
            # LogProgressDialog: cada msg vira uma linha. % é informativo
            # mas não tem barra determinada — rola pelo log.
            self._dialog.append_log(f"[{pct:>3}%] {msg}")
        else:
            self._dialog.setLabelText(msg)
            self._dialog.setValue(pct)

    def _on_worker_finished(self, payload) -> None:
        if self._completed:
            return
        self._completed = True
        success = payload is not None
        self._teardown_ui(success=success, auto_close=success)
        if self._on_finished is not None:
            self._on_finished(payload)

    def _on_worker_error(self, msg: str) -> None:
        if self._completed:
            return
        self._completed = True
        self._teardown_ui(success=False, auto_close=False)
        if self._on_error is not None:
            self._on_error(msg)

    def _teardown_ui(self, *, success: bool, auto_close: bool) -> None:
        if self._cursor_active:
            QApplication.restoreOverrideCursor()
            self._cursor_active = False

        if self._use_log_dialog:
            # No LogProgressDialog, transitamos para o estado "Fechar"
            # (botão muda, barra fica em 100%). O usuário fecha quando
            # quiser ler os logs. Não auto-fechamos para preservar o
            # histórico visível.
            try:
                self._dialog.finish(success=success)
            except Exception:
                pass
        elif auto_close:
            # QProgressDialog simples — fecha quando termina sem erro.
            self._dialog.close()
        # Em erro com QProgressDialog, deixamos a janela aberta com a
        # última mensagem; quem chamou exibe QMessageBox em cima.
        else:
            self._dialog.close()
