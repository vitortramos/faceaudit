# faceaudit/protocols/strategy_instruction_executor_protocol.py

from __future__ import annotations
from typing import Protocol, runtime_checkable

from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)


@runtime_checkable
class StrategyInstructionExecutorProtocol(Protocol):
    """
    Protocolo para executores concretos de instruções da estratégia.

    Um executor é responsável por:
    1. Validar a instrução no contexto atual.
    2. Executar a instrução quando válida.
    3. Produzir um resultado estruturado da execução.
    """

    def validate(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        """
        Valida a instrução no contexto informado antes da execução.
        """
        ...

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        """
        Executa a instrução no contexto informado.
        """
        ...