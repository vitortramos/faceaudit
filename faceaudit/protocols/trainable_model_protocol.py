# faceaudit/protocols/trainable_model_protocol.py

from __future__ import annotations
from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class TrainableModelProtocol(Protocol):
    """
    Protocolo genérico para modelos treináveis da nova arquitetura.

    Um modelo treinável é capaz de:
    - ajustar seus parâmetros internos a partir de dados
    - expor métricas de treinamento
    - serializar/restaurar seu estado
    - persistir-se em disco
    """

    def fit(self, *args, **kwargs) -> None:
        """
        Treina o modelo.

        A assinatura concreta depende da natureza do modelo.
        Exemplos:
        - pipeline linear: fit(dataset)
        - pipeline dual: fit_dual(source_dataset, target_dataset, policy)
        - calibrador: fit(scores, labels)
        - modelo de similaridade: fit(dataset)
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o modelo treinado em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o modelo treinado a partir do disco.
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        Retorna um dicionário serializável com o estado treinado do modelo.
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaura o estado treinado do modelo a partir de um dicionário.
        """
        ...

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas e metadados relevantes do treinamento.
        """
        ...