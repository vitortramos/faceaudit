# faceaudit/protocols/lr_strategy_definition_protocol.py

from __future__ import annotations
from typing import List, Optional, Protocol, runtime_checkable


from faceaudit.models.instructions.strategy_instruction_definition import StrategyInstructionDefinition
from faceaudit.protocols.strategy_resource_protocol import StrategyResourceProtocol


@runtime_checkable
class LRStrategyDefinitionProtocol(Protocol):
    """
    Protocolo para a definição editável de uma estratégia de cálculo de LR.

    A definição da estratégia é a representação declarativa do workflow e
    contém:
    - metadados básicos
    - recursos declarados
    - instruções declaradas
    - regras de validação registradas por identificador
    """

    @property
    def strategy_id(self) -> str:
        """
        Identificador único da estratégia.
        """
        ...

    @property
    def name(self) -> str:
        """
        Nome amigável da estratégia.
        """
        ...

    @property
    def description(self) -> Optional[str]:
        """
        Descrição opcional da estratégia.
        """
        ...

    @property
    def resources(self) -> List[StrategyResourceProtocol]:
        """
        Lista de recursos declarados da estratégia.
        """
        ...

    @property
    def instructions(self) -> List[StrategyInstructionDefinition]:
        """
        Lista ordenada de instruções declaradas da estratégia.
        """
        ...

    @property
    def validation_rule_ids(self) -> List[str]:
        """
        Lista de regras de validação registradas na estratégia.
        """
        ...

    @property
    def version(self) -> str:
        """
        Versão/schema da definição da estratégia.
        """
        ...

    @property
    def has_resources(self) -> bool:
        """
        Retorna True se a estratégia possui recursos declarados.
        """
        ...

    @property
    def has_instructions(self) -> bool:
        """
        Retorna True se a estratégia possui instruções declaradas.
        """
        ...

    @property
    def has_validation_rules(self) -> bool:
        """
        Retorna True se a estratégia possui regras de validação registradas.
        """
        ...

    def get_resource_by_id(self, resource_id: str) -> Optional[StrategyResourceProtocol]:
        """
        Retorna o recurso correspondente ao identificador informado.
        """
        ...

    def get_instruction_by_id(
        self,
        instruction_id: str
    ) -> Optional[StrategyInstructionDefinition]:
        """
        Retorna a instrução correspondente ao identificador informado.
        """
        ...

    def add_resource(self, resource: StrategyResourceProtocol) -> None:
        """
        Adiciona um recurso à estratégia.
        """
        ...

    def update_resource(self, resource: StrategyResourceProtocol) -> None:
        """
        Atualiza um recurso já existente na estratégia.
        """
        ...

    def remove_resource(self, resource_id: str) -> None:
        """
        Remove um recurso da estratégia pelo identificador.
        """
        ...

    def add_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Adiciona uma instrução à estratégia.
        """
        ...

    def update_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Atualiza uma instrução já existente na estratégia.
        """
        ...

    def remove_instruction(self, instruction_id: str) -> None:
        """
        Remove uma instrução da estratégia pelo identificador.
        """
        ...

    def add_validation_rule_id(self, rule_id: str) -> None:
        """
        Registra um identificador de regra de validação na estratégia.
        """
        ...

    def remove_validation_rule_id(self, rule_id: str) -> None:
        """
        Remove um identificador de regra de validação da estratégia.
        """
        ...