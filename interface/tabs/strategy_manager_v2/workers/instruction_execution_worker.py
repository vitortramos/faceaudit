# interface/tabs/strategy_manager_v2/workers/instruction_execution_worker.py

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore


class InstructionExecutionWorker(QtCore.QObject):
    """
    Worker para execução de uma única instrução em thread separada.

    Além de executar a instrução, o worker também persiste em disco os
    recursos produzidos (caso strategy_id seja fornecido). Isso evita que
    operações pesadas como pickle.dump de SimilarityScoreDataset gigantes
    bloqueiem a main thread (a janela de progresso fica congelada se a
    persistência rodar no callback finished).

    Sinais:
        log_message(str) — mensagens de progresso (executor + persistência)
        finished(InstructionExecutionResult) — execução bem-sucedida
        error(str) — execução falhou
    """

    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        executor,
        instruction,
        context,
        strategy_id: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._executor = executor
        self._instruction = instruction
        self._context = context
        self._strategy_id = strategy_id

    @QtCore.pyqtSlot()
    def run(self):
        # Registra listener no contexto para que logs emitidos pelo executor
        # via context.log(...) sejam refletidos em tempo real no dialog.
        listener = self.log_message.emit
        attached = False
        try:
            if self._context is not None and hasattr(self._context, "add_log_listener"):
                self._context.add_log_listener(listener)
                attached = True
        except Exception:
            attached = False

        try:
            self.log_message.emit(f"Executando: {self._instruction.name}")
            result = self._executor.execute_single_instruction(
                self._instruction, self._context
            )
            if result.success:
                # Persiste recursos produzidos AINDA NA THREAD DO WORKER —
                # evita travar a UI quando o pickle.dump é pesado (ex:
                # SimilarityScoreDataset com milhões de pares).
                self._persist_produced_resources_inline(result)

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
        finally:
            if attached:
                try:
                    self._context.remove_log_listener(listener)
                except Exception:
                    pass

    def _persist_produced_resources_inline(self, result) -> None:
        """
        Persiste cada recurso produzido em disco via executor.persist_produced_resource,
        atualizando `resource.artifact_ref` diretamente no recurso do contexto.

        A controller, na main thread, ao receber finished(), apenas sincroniza
        os artifact_refs já gravados (operação O(1) por recurso, sem I/O).
        """
        if self._strategy_id is None or self._executor is None or self._context is None:
            return

        produced_ids = list(getattr(result, "produced_resource_ids", []) or [])
        if not produced_ids:
            return

        for resource_id in produced_ids:
            try:
                resource = self._context.get_resource(resource_id)
            except Exception:
                resource = None
            if resource is None or not resource.has_payload:
                continue

            self.log_message.emit(f"💾 Persistindo recurso '{resource_id}' em disco…")
            try:
                artifact_ref = self._executor.persist_produced_resource(
                    resource=resource,
                    strategy_id=self._strategy_id,
                    progress_callback=self.log_message.emit,
                )
                if artifact_ref is not None:
                    resource.artifact_ref = artifact_ref
                    self.log_message.emit(
                        f"✅ Recurso '{resource_id}' salvo em '{artifact_ref.path}'."
                    )
            except Exception as exc:
                # Falha de persistência não invalida a execução — apenas
                # avisa. O recurso continua disponível em memória no contexto.
                self.log_message.emit(
                    f"⚠️ Falha ao persistir '{resource_id}': {exc}"
                )
