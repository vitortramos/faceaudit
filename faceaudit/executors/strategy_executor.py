# faceaudit/executors/strategy_executor.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.executors.instruction_executor_registry import (
    InstructionExecutorRegistry,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.protocols.strategy_instruction_executor_protocol import (
    StrategyInstructionExecutorProtocol,
)
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol
from faceaudit.services.strategy_resource_loader_service import (
    StrategyResourceLoaderService,
)


class StrategyExecutor:
    """
    Executor principal responsável por orquestrar a execução completa de uma estratégia.

    Fluxo:
    1. Cria contexto de execução
    2. Registra recursos iniciais
    3. Registra regras de validação
    4. Valida definição da estratégia
    5. Executa instruções sequencialmente
    6. Constrói modelo treinado final
    """

    def __init__(
        self,
        context_factory: Type[StrategyExecutionContextProtocol],
        instruction_executor_registry: Union[
            InstructionExecutorRegistry,
            Dict[StrategyInstructionType, StrategyInstructionExecutorProtocol],
        ],
        validation_rules: Optional[Dict[str, ValidationRuleProtocol]] = None,
        artifact_storage: Optional[ArtifactStorageProtocol] = None,
    ):
        self.context_factory = context_factory
        self.validation_rules = validation_rules or {}
        self._artifact_storage = artifact_storage
        self._resource_loader = (
            StrategyResourceLoaderService(artifact_storage)
            if artifact_storage is not None
            else None
        )

        if isinstance(instruction_executor_registry, InstructionExecutorRegistry):
            self.instruction_executor_registry = instruction_executor_registry
        else:
            registry = InstructionExecutorRegistry()
            for instruction_type, executor in instruction_executor_registry.items():
                registry.register(
                    instruction_type=instruction_type,
                    executor=executor,
                    overwrite=True,
                )
            self.instruction_executor_registry = registry

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def execute(
        self,
        strategy_definition: LRStrategyDefinition,
    ) -> LRStrategyTrainedModel:
        """
        Executa completamente a estratégia e retorna o modelo treinado.
        """
        if strategy_definition is None:
            raise ValueError("A definição da estratégia não pode ser nula.")

        context = self.prepare_context(strategy_definition)
        self._validate_strategy(strategy_definition, context)

        execution_results = self._execute_all_phases(strategy_definition, context)

        trained_model = self._build_trained_model(
            strategy_definition=strategy_definition,
            context=context,
            execution_results=execution_results,
        )

        self._log(context, "🏁 Execução concluída com sucesso.")

        return trained_model

    def prepare_context(
        self,
        strategy_definition: LRStrategyDefinition,
    ) -> StrategyExecutionContextProtocol:
        """
        Cria e inicializa o contexto: hidrata recursos iniciais e registra regras.
        O contexto retornado persiste entre execuções individuais de instruções.
        """
        context = self.context_factory()
        self._register_initial_resources(strategy_definition, context)
        self._register_validation_rules(context)

        # Armazena dimensões do embedding para regras de validação
        if getattr(strategy_definition, "embedding_type", None) is not None:
            context.set_metadata(
                "embedding_dimensions",
                strategy_definition.embedding_type.dimensions,
            )

        self._log(context, f"🚀 Contexto preparado para '{strategy_definition.name}'")
        return context

    def can_execute_instruction(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> Tuple[bool, List[str]]:
        """
        Verifica se todos os input_resource_ids da instrução estão disponíveis no contexto.
        Retorna (True, []) se executável, ou (False, [missing_ids]) se bloqueada.
        """
        missing = [
            rid for rid in instruction.input_resource_ids
            if not context.has_resource(rid)
        ]
        return (len(missing) == 0, missing)

    def execute_single_instruction(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        """
        Executa uma única instrução dentro do contexto fornecido.
        Os recursos produzidos ficam disponíveis no contexto para instruções futuras.
        """
        executor = self._resolve_executor(instruction)
        self._log(context, f"▶ Executando: {instruction.name} ({instruction.instruction_id})")
        result = executor.execute(instruction=instruction, context=context)
        if not result.success:
            self._log(context, f"❌ Falha: {result.errors}")
        else:
            self._log(context, f"✅ Concluída: {instruction.name}")
        return result

    def persist_produced_resource(
        self,
        resource: StrategyResource,
        strategy_id: str,
    ) -> Optional[StrategyArtifactReference]:
        """
        Persiste um recurso produzido em disco via artifact_storage.
        Retorna o StrategyArtifactReference ou None se não for persistível.
        """
        if not resource.has_payload or self._artifact_storage is None:
            return None

        artifact_type = resource.resource_type.to_artifact_type()
        if artifact_type is None:
            return None

        kwargs = {"strategy_id": strategy_id, "resource_id": resource.resource_id}

        # TRAINED_PIPELINE usa step_id ao invés de resource_id no template
        if artifact_type == StrategyArtifactType.TRAINED_PIPELINE:
            kwargs["step_id"] = resource.resource_id

        return self._artifact_storage.save_artifact(
            artifact_type=artifact_type,
            data=resource.payload,
            **kwargs,
        )

    # =========================================================================
    # ETAPAS INTERNAS
    # =========================================================================

    def _register_initial_resources(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        # Resolve a coluna de embedding da estratégia (None = auto-detect)
        embedding_column: Optional[str] = None
        if getattr(strategy_definition, "embedding_type", None) is not None:
            embedding_column = strategy_definition.embedding_type.parquet_column

        for resource in strategy_definition.resources:
            if self._resource_loader is not None:
                self._resource_loader.hydrate(
                    resource, embedding_column=embedding_column
                )
            context.set_resource(resource)

    def _register_validation_rules(
        self,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        for rule in self.validation_rules.values():
            context.register_validation_rule(rule)

    def _validate_strategy(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        self._log(context, "🔎 Validando estratégia...")

        for rule_id in strategy_definition.validation_rule_ids:
            rule = self.validation_rules.get(rule_id)

            if rule is None:
                raise ValueError(f"Regra de validação não encontrada: '{rule_id}'")

            result = rule.validate(target=strategy_definition, context=context)

            if not result.is_valid:
                raise ValueError(
                    f"Validação falhou ({rule_id}): {result.errors}"
                )

        self._log(context, "✅ Estratégia válida.")

    def _execute_all_phases(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> List[InstructionExecutionResult]:
        results: List[InstructionExecutionResult] = []

        phases = [
            ("Preparação", strategy_definition.preparation_instructions),
            ("Validação", strategy_definition.validation_instructions),
            ("Análise de Caso", strategy_definition.casework_instructions),
        ]

        for phase_name, phase_instructions in phases:
            if not phase_instructions:
                continue
            self._log(
                context,
                f"📋 Fase: {phase_name} ({len(phase_instructions)} instruções)",
            )
            for instruction in phase_instructions:
                result = self.execute_single_instruction(instruction, context)
                if not result.success:
                    raise RuntimeError(
                        f"Instrução falhou ({instruction.instruction_id}): {result.errors}"
                    )
                results.append(result)

        return results

    def _build_trained_model(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
        execution_results: List[InstructionExecutionResult],
    ) -> LRStrategyTrainedModel:
        model = LRStrategyTrainedModel(strategy_id=strategy_definition.strategy_id)

        for resource in context.list_resources():
            model.add_trained_resource(resource)

        for artifact in context.list_artifacts():
            model.add_artifact(artifact)

        model.set_execution_metadata("strategy_id", strategy_definition.strategy_id)
        model.set_execution_metadata("strategy_name", strategy_definition.name)
        model.set_execution_metadata(
            "num_instructions",
            len(strategy_definition.instructions),
        )
        model.set_execution_metadata(
            "num_resources",
            len(model.trained_resources),
        )
        model.set_execution_metadata(
            "instruction_results",
            [r.metadata for r in execution_results],
        )
        model.set_execution_metadata(
            "execution_logs",
            list(context.execution_logs),
        )
        model.set_execution_metadata(
            "context_metadata",
            dict(context.execution_metadata),
        )

        return model

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _resolve_executor(
        self,
        instruction: StrategyInstructionDefinition,
    ) -> StrategyInstructionExecutorProtocol:
        return self.instruction_executor_registry.resolve(instruction.instruction_type)

    def _log(self, context: StrategyExecutionContextProtocol, message: str) -> None:
        context.log(message)