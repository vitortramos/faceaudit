# interface/tabs/strategy_manager_v2/workers/strategy_execution_worker.py

from __future__ import annotations

from PyQt6 import QtCore

from faceaudit.executors.strategy_executor import StrategyExecutor
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition


class StrategyExecutionWorker(QtCore.QObject):
    """
    Worker para execução de estratégia em thread separada.

    Padrão: QObject + moveToThread (não herdar de QThread).

    Uso:
        worker = StrategyExecutionWorker(executor, definition)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(thread.deleteLater)
        worker.finished.connect(worker.deleteLater)
        worker.log_message.connect(dialog.append_log)
        worker.finished.connect(lambda model: dialog.finish(True))
        worker.error.connect(lambda msg: dialog.finish(False))
        thread.start()
    """

    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)   # LRStrategyTrainedModel
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        executor: StrategyExecutor,
        definition: LRStrategyDefinition,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._executor = executor
        self._definition = definition

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            self.log_message.emit(f"Iniciando execução da estratégia: {self._definition.name}")
            self.log_message.emit(
                f"Instruções: "
                f"{len(self._definition.preparation_instructions)} prep / "
                f"{len(self._definition.validation_instructions)} val / "
                f"{len(self._definition.casework_instructions)} caso"
            )

            result = self._executor.execute(self._definition)

            self.log_message.emit("Execução concluída com sucesso.")
            self.finished.emit(result)

        except Exception as exc:
            msg = f"Erro durante a execução: {exc}"
            self.log_message.emit(msg)
            self.error.emit(msg)
