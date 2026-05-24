# interface/tabs/strategy_manager_v2/workers/strategy_execution_worker.py

from __future__ import annotations

from typing import List, Optional

from PyQt6 import QtCore

from faceaudit.executors.strategy_executor import (
    StrategyCancelledError,
    StrategyExecutor,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.strategy_resource import StrategyResource


class StrategyExecutionWorker(QtCore.QObject):
    """
    Worker para execução de estratégia em thread separada.

    Padrão: QObject + moveToThread (não herdar de QThread).

    Após executar todas as fases, persiste em disco os recursos treinados
    (campos do LRStrategyTrainedModel) ainda na thread do worker — evita
    travar a UI quando o pickle é pesado (ex: SimilarityScoreDataset com
    milhões de pares ou TransformationTrajetoryData volumoso).

    Cancelamento cooperativo: caller chama `request_cancel()` (thread-safe
    via flag bool) e o executor consulta o flag entre instruções. Quando
    levanta `StrategyCancelledError`, o worker emite `error` com a
    mensagem amigável (e NÃO `finished`) — caller trata como exit normal.
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
        # Flag de cancelamento — escrita pela main thread via
        # `request_cancel()`, lida pelo executor (na worker thread) via
        # `cancel_checker`. Acessos a bool em Python são atômicos por
        # virtude do GIL, não precisa de lock explícito aqui.
        self._cancel_requested = False

    def request_cancel(self) -> None:
        """Sinaliza pro executor parar entre instruções. Idempotente."""
        self._cancel_requested = True

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
                cancel_checker=lambda: self._cancel_requested,
            )

            # Persiste recursos AINDA NA THREAD DO WORKER — evita "Não respondendo"
            # na UI quando o pickle/parquet é grande.
            self._persist_trained_model_resources_inline(result)

            self.log_message.emit("Execução concluída com sucesso.")
            self.finished.emit(result)

        except StrategyCancelledError as exc:
            # Cancelamento cooperativo — não é erro. Loga amigável e
            # emite error pra que o handler feche o dialog sem mexer em
            # `_executed_instruction_ids` ou sincronizar artifacts
            # parciais (algumas instruções RODARAM antes do cancel, mas
            # a estratégia como um todo NÃO completou).
            msg = f"⏹ {exc}"
            self.log_message.emit(msg)
            self.error.emit(msg)

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
        strategy_name = self._definition.name or ""

        # Acesso ao project_root pra checar existência real dos arquivos
        # referenciados. Sem isso, o worker tratava `artifact_ref` no JSON
        # como prova de que o arquivo está em disco — mas ref órfã
        # (arquivo apagado por cleanup intermediário, JSON manteve o path)
        # passava como "já persistido" e a re-execução não regrava o
        # arquivo. Ref ficava órfã indefinidamente.
        project_root = None
        try:
            project_root = self._executor._resource_loader._storage.project_root
        except Exception:
            project_root = None

        from faceaudit.enums.resource_category import ResourceCategory

        for resource in trained_resources:
            if not getattr(resource, "has_payload", False):
                continue

            # PRIMARY (project resources como PIPELINE_CONFIG /
            # MINING_CONFIG / DATASET importados) sempre são pulados.
            # Regravar exigiria kwargs específicos por tipo (config_id,
            # dataset_id, etc.) que o caller genérico não tem.
            if getattr(resource, "category", None) == ResourceCategory.PRIMARY:
                continue

            # DERIVED com ref preexistente: pula APENAS se o arquivo de
            # fato existe no disco. Ref órfã (arquivo sumiu) força nova
            # persistência. Sem essa checagem, refs órfãs ficavam pra
            # sempre porque "tem artifact_ref" era suficiente pro skip.
            existing_ref = getattr(resource, "artifact_ref", None)
            if existing_ref is not None and project_root is not None:
                ref_path = (existing_ref.path or "").replace("\\", "/")
                if ref_path and (project_root / ref_path).exists():
                    continue
                # Arquivo sumiu — zera o ref pra que persist_produced_resource
                # gere UID novo (em vez de tentar reusar o path antigo).
                # Log explícito pra deixar visível na UI que o fix está
                # tomando conta de uma ref órfã (debug de "por que minhas
                # refs continuam órfãs depois do Run All").
                self.log_message.emit(
                    f"🔁 Recurso '{resource.resource_id}' tem ref pra "
                    f"arquivo apagado ({ref_path}); regravando com UID novo."
                )
                resource.artifact_ref = None

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
                    strategy_name=strategy_name,
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
