# faceaudit/protocols/strategy_definition_repository_protocol.py

from __future__ import annotations
from typing import List, Optional, Protocol, runtime_checkable

from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition


@runtime_checkable
class StrategyDefinitionRepositoryProtocol(Protocol):
    """
    Protocolo para persistência da definição declarativa da estratégia.

    A implementação concreta deve ser capaz de salvar, carregar e listar
    estratégias editáveis dentro do contexto do projeto.
    """

    def save_definition(self, strategy_definition: "LRStrategyDefinition") -> None:
        """
        Persiste a definição da estratégia.

        Args:
            strategy_definition:
                Definição declarativa da estratégia a ser salva.
        """
        ...

    def load_definition(self, strategy_id: str) -> Optional["LRStrategyDefinition"]:
        """
        Carrega a definição da estratégia correspondente ao strategy_id.

        Args:
            strategy_id:
                Identificador único da estratégia.

        Returns:
            Optional[LRStrategyDefinition]:
                Estratégia carregada, se encontrada.
        """
        ...

    def list_definitions(self) -> List["LRStrategyDefinition"]:
        """
        Lista todas as definições de estratégia disponíveis no repositório.

        Returns:
            List[LRStrategyDefinition]:
                Lista de estratégias declarativas.
        """
        ...