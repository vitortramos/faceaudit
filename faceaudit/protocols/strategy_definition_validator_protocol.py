# faceaudit/protocols/strategy_definition_validator_protocol.py

from __future__ import annotations
from typing import Protocol, runtime_checkable

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.validation.validation_result import ValidationResult


@runtime_checkable
class StrategyDefinitionValidatorProtocol(Protocol):
    """
    Protocolo para validadores globais da definição da estratégia.
    """

    def validate(self, strategy_definition: "LRStrategyDefinition") -> ValidationResult:
        """
        Valida a definição completa da estratégia.
        """
        ...