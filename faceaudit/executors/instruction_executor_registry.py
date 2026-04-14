# faceaudit/executors/instruction_executor_registry.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.protocols.strategy_instruction_executor_protocol import (
    StrategyInstructionExecutorProtocol,
)


@dataclass
class InstructionExecutorRegistry:
    """
    Registry responsável por mapear tipos de instrução para seus respectivos executores.

    Ele atua como uma camada simples de resolução, permitindo que o StrategyExecutor
    obtenha o executor correto para cada StrategyInstructionType sem depender de um
    dicionário montado manualmente em múltiplos pontos do sistema.
    """

    _executors: Dict[StrategyInstructionType, StrategyInstructionExecutorProtocol] = field(
        default_factory=dict
    )

    def register(
        self,
        instruction_type: StrategyInstructionType,
        executor: StrategyInstructionExecutorProtocol,
        overwrite: bool = False,
    ) -> None:
        """
        Registra um executor para um tipo de instrução.

        Args:
            instruction_type:
                Tipo de instrução associado ao executor.
            executor:
                Executor concreto responsável por esse tipo.
            overwrite:
                Se True, permite sobrescrever um executor já registrado.

        Raises:
            ValueError:
                Se o tipo ou o executor forem inválidos, ou se já houver um
                executor registrado e overwrite=False.
        """
        if instruction_type is None:
            raise ValueError("instruction_type não pode ser nulo.")

        if executor is None:
            raise ValueError("executor não pode ser nulo.")

        if not overwrite and instruction_type in self._executors:
            raise ValueError(
                f"Já existe executor registrado para instruction_type '{instruction_type}'."
            )

        self._executors[instruction_type] = executor

    def resolve(
        self,
        instruction_type: StrategyInstructionType,
    ) -> StrategyInstructionExecutorProtocol:
        """
        Resolve o executor correspondente ao tipo de instrução informado.

        Args:
            instruction_type:
                Tipo de instrução a ser resolvido.

        Returns:
            StrategyInstructionExecutorProtocol:
                Executor registrado para o tipo.

        Raises:
            ValueError:
                Se não existir executor registrado para o tipo informado.
        """
        executor = self._executors.get(instruction_type)
        if executor is None:
            raise ValueError(
                f"Nenhum executor registrado para instruction_type '{instruction_type}'."
            )
        return executor

    def get(
        self,
        instruction_type: StrategyInstructionType,
    ) -> Optional[StrategyInstructionExecutorProtocol]:
        """
        Retorna o executor correspondente ao tipo, se existir.
        """
        return self._executors.get(instruction_type)

    def has(
        self,
        instruction_type: StrategyInstructionType,
    ) -> bool:
        """
        Retorna True se houver executor registrado para o tipo informado.
        """
        return instruction_type in self._executors

    def unregister(
        self,
        instruction_type: StrategyInstructionType,
    ) -> None:
        """
        Remove o executor associado ao tipo informado, se existir.
        """
        self._executors.pop(instruction_type, None)

    def clear(self) -> None:
        """
        Remove todos os executores registrados.
        """
        self._executors.clear()

    def to_dict(self) -> Dict[StrategyInstructionType, StrategyInstructionExecutorProtocol]:
        """
        Retorna uma cópia do mapa interno de executores.
        """
        return dict(self._executors)