# faceaudit/models/strategy/lr_strategy_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource


T = TypeVar("T", bound="LRStrategyDefinition")


@dataclass
class LRStrategyDefinition(JSONSerializableMixin):
    """
    Representa a definição declarativa e editável de uma estratégia de cálculo de LR.

    Esta classe descreve a estratégia como configuração:
    - metadados
    - recursos declarados
    - instruções declaradas
    - regras de validação registradas por identificador

    Ela não executa a estratégia diretamente. A execução/preparação ficará a cargo
    de serviços/executores apropriados na nova arquitetura.
    """

    strategy_id: str = field(
        metadata={"help": "Identificador único da estratégia."}
    )
    name: str = field(
        metadata={"help": "Nome amigável da estratégia."}
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Descrição opcional da finalidade da estratégia."}
    )

    resources: List[StrategyResource] = field(
        default_factory=list,
        metadata={"help": "Lista de recursos declarados da estratégia."}
    )
    instructions: List[StrategyInstructionDefinition] = field(
        default_factory=list,
        metadata={"help": "Lista ordenada de instruções declaradas da estratégia."}
    )

    validation_rule_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de regras de validação registradas na estratégia."}
    )

    version: str = field(
        default="1.0",
        metadata={"help": "Versão/schema da definição da estratégia."}
    )

    @property
    def has_resources(self) -> bool:
        """
        Retorna True se a estratégia possui recursos declarados.
        """
        return bool(self.resources)

    @property
    def has_instructions(self) -> bool:
        """
        Retorna True se a estratégia possui instruções declaradas.
        """
        return bool(self.instructions)

    @property
    def has_validation_rules(self) -> bool:
        """
        Retorna True se a estratégia possui regras de validação registradas.
        """
        return bool(self.validation_rule_ids)

    def get_resource_by_id(self, resource_id: str) -> Optional[StrategyResource]:
        """
        Retorna o recurso correspondente ao resource_id informado.
        """
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        return None

    def get_instruction_by_id(
        self,
        instruction_id: str
    ) -> Optional[StrategyInstructionDefinition]:
        """
        Retorna a instrução correspondente ao instruction_id informado.
        """
        for instruction in self.instructions:
            if instruction.instruction_id == instruction_id:
                return instruction
        return None

    def add_resource(self, resource: StrategyResource) -> None:
        """
        Adiciona um recurso à estratégia.

        Lança erro se já existir um recurso com o mesmo resource_id.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")

        if self.get_resource_by_id(resource.resource_id) is not None:
            raise ValueError(
                f"Já existe um recurso com resource_id '{resource.resource_id}'."
            )

        self.resources.append(resource)

    def update_resource(self, resource: StrategyResource) -> None:
        """
        Atualiza um recurso já existente na estratégia.

        Lança erro se o recurso não existir.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")

        for index, current in enumerate(self.resources):
            if current.resource_id == resource.resource_id:
                self.resources[index] = resource
                return

        raise ValueError(
            f"Não foi encontrado recurso com resource_id '{resource.resource_id}' para atualização."
        )

    def remove_resource(self, resource_id: str) -> None:
        """
        Remove um recurso da estratégia pelo identificador.

        Se o recurso não existir, não faz nada.
        """
        self.resources = [
            resource for resource in self.resources
            if resource.resource_id != resource_id
        ]

    def add_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Adiciona uma instrução à estratégia.

        Lança erro se já existir uma instrução com o mesmo instruction_id.
        """
        if instruction is None:
            raise ValueError("A instrução não pode ser nula.")

        if self.get_instruction_by_id(instruction.instruction_id) is not None:
            raise ValueError(
                f"Já existe uma instrução com instruction_id '{instruction.instruction_id}'."
            )

        self.instructions.append(instruction)

    def update_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Atualiza uma instrução já existente na estratégia.

        Lança erro se a instrução não existir.
        """
        if instruction is None:
            raise ValueError("A instrução não pode ser nula.")

        for index, current in enumerate(self.instructions):
            if current.instruction_id == instruction.instruction_id:
                self.instructions[index] = instruction
                return

        raise ValueError(
            f"Não foi encontrada instrução com instruction_id "
            f"'{instruction.instruction_id}' para atualização."
        )

    def remove_instruction(self, instruction_id: str) -> None:
        """
        Remove uma instrução da estratégia pelo identificador.

        Se a instrução não existir, não faz nada.
        """
        self.instructions = [
            instruction for instruction in self.instructions
            if instruction.instruction_id != instruction_id
        ]

    def add_validation_rule_id(self, rule_id: str) -> None:
        """
        Registra um identificador de regra de validação na estratégia.
        """
        if rule_id and rule_id not in self.validation_rule_ids:
            self.validation_rule_ids.append(rule_id)

    def remove_validation_rule_id(self, rule_id: str) -> None:
        """
        Remove um identificador de regra de validação da estratégia.
        """
        self.validation_rule_ids = [
            current_rule_id
            for current_rule_id in self.validation_rule_ids
            if current_rule_id != rule_id
        ]

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói a definição da estratégia, hidratando corretamente:
        - resources
        - instructions

        Observação:
        validation_rule_ids permanece como lista simples de strings.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "resources" in processed_data and isinstance(processed_data["resources"], list):
            processed_data["resources"] = [
                StrategyResource.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["resources"]
            ]

        if "instructions" in processed_data and isinstance(processed_data["instructions"], list):
            processed_data["instructions"] = [
                StrategyInstructionDefinition.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["instructions"]
            ]

        return cls(**processed_data)