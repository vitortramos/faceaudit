# interface/tabs/strategy_manager_v2/workers/strategy_execution_worker.py

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore

from faceaudit.executors.strategy_executor import StrategyExecutor
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource


class StrategyExecutionWorker(QtCore.QObject):
    """
    Worker para execução de estratégia em thread separada.

    Padrão: QObject + moveToThread (não herdar de QThread).

    Após executar todas as fases, persiste em disco os recursos treinados
    (campos do LRStrategyTrainedModel) ainda na thread do worker — evita
    travar a UI quando o pickle é pesado (ex: SimilarityScoreDataset com
    milhões de pares ou TransformationTrajectoryData volumoso).
    """

    log_message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)   # LRStrategyTrainedModel
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        executor: StrategyExecutor,
        definition: LRStrategyDefinition,
        additional_resources: Optional[List[StrategyResource]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._executor = executor
        self._definition = definition
        self._additional_resources = list(additional_resources or [])

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

            result = self._executor.execute(
                self._definition,
                additional_resources=self._additional_resources or None,
            )

            # Persiste recursos AINDA NA THREAD DO WORKER — evita "Não respondendo"
            # na UI quando o pickle/parquet é grande.
            self._persist_trained_model_resources_inline(result)

            self.log_message.emit("Execução concluída com sucesso.")
            self.finished.emit(result)

        except Exception as exc:
            msg = f"Erro durante a execução: {exc}"
            self.log_message.emit(msg)
            self.error.emit(msg)

    def _persist_trained_model_resources_inline(self, model) -> None:
        """
        Persiste cada recurso do LRStrategyTrainedModel em disco via
        executor.persist_produced_resource, atualizando `resource.artifact_ref`
        diretamente no recurso. A controller, na main thread, ao receber
        finished(), apenas sincroniza com a definição em memória (rápido).
        """
        if self._executor is None or model is None or self._definition is None:
            return

        trained_resources = list(getattr(model, "trained_resources", []) or [])
        if not trained_resources:
            return

        strategy_id = self._definition.strategy_id

        for resource in trained_resources:
            if not getattr(resource, "has_payload", False):
                continue

            # Pular recursos que já estão em disco (project resources como
            # PIPELINE_CONFIG / MINING_CONFIG / DATASET importados, ou
            # qualquer resource que veio com artifact_ref preexistente).
            # Regravar exigiria kwargs específicos por tipo (config_id,
            # dataset_id, etc.) que o caller genérico não tem.
            if getattr(resource, "artifact_ref", None) is not None:
                continue

            # Pular recursos sem mapeamento de artifact (ex: project resources
            # que já estão em disco e não precisam ser regravados).
            try:
                artifact_type = resource.resource_type.to_artifact_type()
            except Exception:
                artifact_type = None
            if artifact_type is None:
                continue

            self.log_message.emit(
                f"💾 Persistindo recurso '{resource.resource_id}' em disco…"
            )
            try:
                artifact_ref = self._executor.persist_produced_resource(
                    resource=resource,
                    strategy_id=strategy_id,
                )
                if artifact_ref is not None:
                    resource.artifact_ref = artifact_ref
                    self.log_message.emit(
                        f"✅ Recurso '{resource.resource_id}' salvo em '{artifact_ref.path}'."
                    )
            except Exception as exc:
                self.log_message.emit(
                    f"⚠️ Falha ao persistir '{resource.resource_id}': {exc}"
                )
