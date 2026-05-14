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
    - validar as `validation_rule_configs` declaradas na instrução
      (instanciadas via `ValidationRuleFactory` em runtime)
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
        Pipeline de validação de instrução — 4 estágios sequenciais.

        Cada estágio agrega seus resultados no mesmo ValidationResult.
        Se o estágio 1 (null checks) falha, os demais não executam.

        Estágios:
            1. Structural  — null checks em instruction e context
            2. Pluggable   — regras instanciadas a partir de
                             validation_rule_configs via ValidationRuleFactory
            3. Dynamic     — regras construídas pelo DTO (build_validation_rules)
            4. Executor    — validações hard-coded do executor concreto
                             (_validate_instruction_specifics)

        Nota: a verificação de dependências (input_resource_ids presentes no
        contexto) ocorre ANTES deste pipeline, via can_execute_instruction()
        no StrategyExecutor — é um gate de execução, não de validação.
        """
        result = ValidationResult.neutral()

        # ── Estágio 1: Structural ──────────────────────────────────
        if instruction is None:
            return ValidationResult.critical("A instrução não pode ser nula.")

        if context is None:
            return ValidationResult.critical("O contexto de execução não pode ser nulo.")

        # ── Pré-estágio: Re-hidratação preguiçosa de inputs ────────
        # Antes da validação propriamente dita, garante que cada input
        # declarado tenha o payload carregado quando possível. Sem isso,
        # validators que checam isinstance(payload, ...) falhariam logo após
        # abrir um projeto (recursos no contexto vêm com payload=None e
        # artifact_ref preenchido — payload só era hidratado no
        # execute_single_instruction). Idempotente: se payload já está em
        # memória, ensure_payload retorna direto sem custo.
        ensure_payload = getattr(context, "ensure_payload", None)
        if callable(ensure_payload):
            for rid in getattr(instruction, "input_resource_ids", []) or []:
                try:
                    ensure_payload(rid)
                except Exception:
                    # Falha não-fatal — o validador concreto reportará
                    # com mensagem específica (payload=None / load do disco).
                    pass

        # ── Estágio 2: Pluggable (validation_rule_configs) ─────────
        # As regras agora são auto-contidas: cada `ValidationRuleConfig`
        # carrega rule_type + params suficientes para o factory instanciar
        # a regra concreta em runtime. Não dependemos mais do contexto
        # ter as regras pré-registradas.
        from faceaudit.factories.validation_rule_factory import ValidationRuleFactory

        for cfg in getattr(instruction, "validation_rule_configs", []) or []:
            try:
                rule = ValidationRuleFactory.create(
                    config=cfg, instruction_id=instruction.instruction_id
                )
            except ValueError as exc:
                # Config malformado (params ausentes/inválidos, tipo
                # desconhecido). Reporta erro específico mas segue tentando
                # as demais — falha de uma regra não bloqueia o pipeline
                # de validação restante.
                result.add_error(
                    f"[{instruction.name}] Falha ao instanciar regra de "
                    f"validação: {exc}"
                )
                continue

            rule_result = rule.validate(target=instruction, context=context)
            result.merge(rule_result)

        # ── Estágio 3: Dynamic (build_validation_rules) ────────────
        for rule in instruction.build_validation_rules():
            rule_result = rule.validate(target=instruction, context=context)
            result.merge(rule_result)

        # ── Estágio 4: Executor (_validate_instruction_specifics) ──
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