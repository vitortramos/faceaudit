# faceaudit/protocols/strategy_trained_model_repository_protocol.py

from __future__ import annotations
from typing import List, Optional, Protocol, runtime_checkable

from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel


@runtime_checkable
class StrategyTrainedModelRepositoryProtocol(Protocol):
    """
    Protocolo para persistência do modelo treinado final da estratégia.

    A implementação concreta deve ser capaz de salvar, carregar e listar
    modelos treinados de estratégias dentro do contexto do projeto.
    """

    def save_model(self, trained_model: "LRStrategyTrainedModel") -> None:
        """
        Persiste o modelo treinado da estratégia.

        Args:
            trained_model:
                Modelo treinado da estratégia a ser salvo.
        """
        ...

    def load_model(self, strategy_id: str) -> Optional["LRStrategyTrainedModel"]:
        """
        Carrega o modelo treinado correspondente ao strategy_id.

        Args:
            strategy_id:
                Identificador único da estratégia.

        Returns:
            Optional[LRStrategyTrainedModel]:
                Modelo treinado carregado, se encontrado.
        """
        ...

    def list_models(self) -> List["LRStrategyTrainedModel"]:
        """
        Lista todos os modelos treinados disponíveis no repositório.

        Returns:
            List[LRStrategyTrainedModel]:
                Lista de modelos treinados de estratégias.
        """
        ...