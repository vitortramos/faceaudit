# interface/tabs/strategy_manager_v2/workers/instruction_execution_worker.py

from __future__ import annotations

from PyQt6 import QtCore


class InstructionExecutionWorker(QtCore.QObject):
    """
    Worker para execução de uma única instrução em thread separada.

    Sinais:
        log_message(str) — mensagens de progresso
        finished(InstructionExecutionResult) — execução bem-sucedida
        error(str) — execução falhou
    """

    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(self, executor, instruction, context, parent=None):
        super().__init__(parent)
        self._executor = executor
        self._instruction = instruction
        self._context = context

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.log_message.emit(f"Executando: {self._instruction.name}")
            result = self._executor.execute_single_instruction(
                self._instruction, self._context
            )
            if result.success:
                self.log_message.emit("Instrução concluída com sucesso.")
                self.finished.emit(result)
            else:
                msg = f"Instrução falhou: {result.errors}"
                self.log_message.emit(msg)
                self.error.emit(msg)
        except Exception as exc:
            msg = f"Erro: {exc}"
            self.log_message.emit(msg)
            self.error.emit(msg)
