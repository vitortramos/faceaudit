# faceaudit/validators/strategy/strategy_definition_validator.py

from __future__ import annotations
from typing import Set

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_definition_validator_protocol import StrategyDefinitionValidatorProtocol


class StrategyDefinitionValidator(StrategyDefinitionValidatorProtocol):
    """
    Implementação concreta do validador global da definição da estratégia.

    Esta versão inicial valida a consistência estrutural básica da estratégia:
    - unicidade de ids de recursos
    - unicidade de ids de instruções
    - existência de recursos de entrada referenciados
    - conflitos de recursos de saída entre instruções
    - existência de regras de validação referenciadas pelas instruções
    """

    def validate(self, strategy_definition: "LRStrategyDefinition") -> ValidationResult:
        result = ValidationResult.neutral()

        if strategy_definition is None:
            return ValidationResult.critical("A definição da estratégia não pode ser nula.")

        self._validate_resource_ids(strategy_definition, result)
        self._validate_instruction_ids(strategy_definition, result)
        self._validate_instruction_inputs(strategy_definition, result)
        self._validate_instruction_outputs(strategy_definition, result)
        self._validate_instruction_validation_rules(strategy_definition, result)

        if result.is_valid and not result.has_warnings:
            result.mark_ok()

        return result

    def _validate_resource_ids(
        self,
        strategy_definition: "LRStrategyDefinition",
        result: ValidationResult
    ) -> None:
        """
        Verifica duplicidade de resource_id.
        """
        seen: Set[str] = set()

        for resource in strategy_definition.resources:
            if not resource.resource_id:
                result.add_error("Existe um recurso sem resource_id definido.")
                continue

            if resource.resource_id in seen:
                result.add_error(
                    f"Resource_id duplicado encontrado na estratégia: '{resource.resource_id}'."
                )
            else:
                seen.add(resource.resource_id)

    def _validate_instruction_ids(
        self,
        strategy_definition: "LRStrategyDefinition",
        result: ValidationResult
    ) -> None:
        """
        Verifica duplicidade de instruction_id.
        """
        seen: Set[str] = set()

        for instruction in strategy_definition.instructions:
            if not instruction.instruction_id:
                result.add_error("Existe uma instrução sem instruction_id definido.")
                continue

            if instruction.instruction_id in seen:
                result.add_error(
                    f"Instruction_id duplicado encontrado na estratégia: '{instruction.instruction_id}'."
                )
            else:
                seen.add(instruction.instruction_id)

    def _validate_instruction_inputs(
        self,
        strategy_definition: "LRStrategyDefinition",
        result: ValidationResult
    ) -> None:
        """
        Verifica se todos os recursos de entrada referenciados existem como:
        - recurso declarado na estratégia
        - ou saída produzida por alguma instrução

        Nesta primeira versão, validamos a existência declarativa, não a ordem real
        de execução.
        """
        declared_resource_ids = {resource.resource_id for resource in strategy_definition.resources}
        declared_output_ids = {
            output_id
            for instruction in strategy_definition.instructions
            for output_id in instruction.output_resource_ids
        }

        valid_ids = declared_resource_ids | declared_output_ids

        for instruction in strategy_definition.instructions:
            for input_id in instruction.input_resource_ids:
                if input_id not in valid_ids:
                    result.add_error(
                        f"A instrução '{instruction.name}' ({instruction.instruction_id}) "
                        f"referencia um recurso de entrada inexistente: '{input_id}'."
                    )

    def _validate_instruction_outputs(
        self,
        strategy_definition: "LRStrategyDefinition",
        result: ValidationResult
    ) -> None:
        """
        Verifica se há conflito entre output_resource_ids produzidos por instruções diferentes.
        """
        produced_outputs: Set[str] = set()

        for instruction in strategy_definition.instructions:
            for output_id in instruction.output_resource_ids:
                if not output_id:
                    result.add_error(
                        f"A instrução '{instruction.name}' ({instruction.instruction_id}) "
                        "possui um output_resource_id vazio."
                    )
                    continue

                if output_id in produced_outputs:
                    result.add_error(
                        f"Conflito de saída detectado: o recurso '{output_id}' é produzido "
                        "por mais de uma instrução."
                    )
                else:
                    produced_outputs.add(output_id)

    def _validate_instruction_validation_rules(
        self,
        strategy_definition: "LRStrategyDefinition",
        result: ValidationResult
    ) -> None:
        """
        Verifica se as validation_rule_ids referenciadas pelas instruções existem
        entre as regras registradas na estratégia.
        """
        available_rule_ids = set(strategy_definition.validation_rule_ids)

        if not available_rule_ids:
            for instruction in strategy_definition.instructions:
                if instruction.validation_rule_ids:
                    result.add_warning(
                        f"A estratégia não possui validation_rule_ids registradas, mas a instrução "
                        f"'{instruction.name}' ({instruction.instruction_id}) referencia regras de validação."
                    )
            return

        for instruction in strategy_definition.instructions:
            for rule_id in instruction.validation_rule_ids:
                if rule_id not in available_rule_ids:
                    result.add_error(
                        f"A instrução '{instruction.name}' ({instruction.instruction_id}) "
                        f"referencia uma validation_rule inexistente: '{rule_id}'."
                    )