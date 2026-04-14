# faceaudit/protocols/validation_rule_protocol.py

from __future__ import annotations
from typing import Any, Optional, Protocol, runtime_checkable

from faceaudit.enums.validation_severity import ValidationSeverity
from faceaudit.models.validation.validation_result import ValidationResult
from faceaudit.protocols.strategy_execution_context_protocol import StrategyExecutionContextProtocol


@runtime_checkable
class ValidationRuleProtocol(Protocol):
    """
    Protocolo para regras de validação reutilizáveis da nova arquitetura.

    Uma regra de validação pode ser aplicada a diferentes alvos, por exemplo:
    - uma instrução
    - a definição completa da estratégia
    - um recurso específico
    - outro objeto de domínio relevante

    O contexto de execução é sempre fornecido como apoio à validação.
    """

    @property
    def rule_id(self) -> str:
        """
        Identificador único da regra de validação.
        """
        ...

    @property
    def name(self) -> str:
        """
        Nome amigável da regra de validação.
        """
        ...

    @property
    def description(self) -> Optional[str]:
        """
        Descrição opcional do objetivo da regra.
        """
        ...

    @property
    def default_severity(self) -> ValidationSeverity:
        """
        Severidade padrão da regra quando a validação falha.
        """
        ...

    def validate(
        self,
        target: Any,
        context: "StrategyExecutionContextProtocol",
    ) -> ValidationResult:
        """
        Executa a validação da regra sobre um alvo de validação no contexto informado.

        Args:
            target:
                Objeto alvo da validação. Pode ser, por exemplo, uma instrução,
                uma estratégia, um recurso ou outro objeto do domínio.
            context:
                Contexto atual de execução da estratégia.

        Returns:
            ValidationResult:
                Resultado estruturado da validação.
        """
        ...