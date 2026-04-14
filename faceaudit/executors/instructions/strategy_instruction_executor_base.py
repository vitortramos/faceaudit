# faceaudit/executors/instructions/strategy_instruction_executor_base.py

from __future__ import annotations
from typing import Optional

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
from faceaudit.protocols.strategy_instruction_executor_protocol import (
    StrategyInstructionExecutorProtocol,
)


class StrategyInstructionExecutorBase(StrategyInstructionExecutorProtocol):
    """
    Classe base para executores concretos de instrução.

    Responsabilidades desta classe base:
    - validar as validation_rule_ids declaradas na instrução
    - agregar os resultados das regras de validação
    - oferecer helpers para logging e criação de resultados

    A validação específica da instrução concreta pode ser feita sobrescrevendo
    `_validate_instruction_specifics()`.
    """

    def validate(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        """
        Executa a validação base da instrução:
        1. valida as rules declaradas pela própria instrução
        2. executa validações específicas do executor concreto
        """
        result = ValidationResult.neutral()

        if instruction is None:
            return ValidationResult.critical("A instrução não pode ser nula.")

        if context is None:
            return ValidationResult.critical("O contexto de execução não pode ser nulo.")

        for rule_id in instruction.validation_rule_ids:
            if not context.has_validation_rule(rule_id):
                result.add_error(
                    f"A instrução '{instruction.name}' ({instruction.instruction_id}) "
                    f"referencia uma validation_rule não registrada no contexto: '{rule_id}'."
                )
                continue

            rule = context.get_validation_rule(rule_id)
            if rule is None:
                result.add_error(
                    f"Falha ao recuperar a validation_rule '{rule_id}' no contexto."
                )
                continue

            rule_result = rule.validate(target=instruction, context=context)
            result.merge(rule_result)

        specific_result = self._validate_instruction_specifics(
            instruction=instruction,
            context=context,
        )
        result.merge(specific_result)

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def execute(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        """
        Método base defensivo.

        Executores concretos devem sobrescrever este método.
        """
        validation = self.validate(instruction=instruction, context=context)
        if not validation.is_valid:
            return InstructionExecutionResult.failure(errors=validation.errors)

        return InstructionExecutionResult.failure(
            errors=[
                f"O executor '{self.__class__.__name__}' não implementou o método execute()."
            ]
        )

    def _validate_instruction_specifics(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> ValidationResult:
        """
        Extensão para validações específicas do executor concreto.

        Por padrão, não adiciona nenhuma restrição extra.
        """
        return ValidationResult.ok()

    def _log(
        self,
        context: StrategyExecutionContextProtocol,
        message: str,
    ) -> None:
        """
        Helper centralizado para logging no contexto.
        """
        if context is not None and message:
            context.log(message)

    def _success_result(
        self,
        metadata: Optional[dict] = None,
    ) -> InstructionExecutionResult:
        """
        Helper para construir resultado de sucesso.
        """
        return InstructionExecutionResult.ok(metadata=metadata or {})

    def _failure_result(
        self,
        *errors: str,
        metadata: Optional[dict] = None,
    ) -> InstructionExecutionResult:
        """
        Helper para construir resultado de falha.
        """
        return InstructionExecutionResult.failure(
            errors=[msg for msg in errors if msg],
            metadata=metadata or {},
        )