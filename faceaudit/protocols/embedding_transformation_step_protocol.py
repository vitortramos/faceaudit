# faceaudit/protocols/embedding_transformation_step_protocol.py

from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, runtime_checkable

from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.enums.transformation_type import TransformationType


@runtime_checkable
class EmbeddingTransformationStepProtocol(Protocol):
    """
    Protocolo para steps do novo EmbeddingTransformationPipeline.

    Um step de transformação de embeddings pode:
    - ser treinado (fit) com um dataset ou com source/target
    - transformar um dataset completo
    - transformar um único embedding
    - expor métricas de treinamento/execução
    - serializar/restaurar seu estado treinado
    """

    @property
    def name(self) -> str:
        """
        Nome amigável/identificador do step.
        """
        ...

    @property
    def transformation_type(self) -> TransformationType:
        """
        Tipo enumerado da transformação.
        """
        ...

    @property
    def requires_target_dataset(self) -> bool:
        """
        Indica se o step exige target_dataset para treinamento.
        Exemplo típico: CORAL.
        """
        ...

    def fit(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: Optional[EmbeddingDataset] = None
    ) -> None:
        """
        Ajusta os parâmetros internos do step.

        Args:
            source_dataset:
                Dataset principal de treino.
            target_dataset:
                Dataset secundário opcional, usado por steps que exigem
                adaptação de domínio ou treinamento dual.
        """
        ...

    def transform(self, dataset: EmbeddingDataset) -> EmbeddingDataset:
        """
        Aplica a transformação a um dataset completo.

        Deve retornar um novo EmbeddingDataset, sem modificar o original.
        """
        ...

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica a transformação a um único EmbeddingInfo.

        Deve retornar um novo EmbeddingInfo, sem modificar o original.
        """
        ...

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas e metadados relevantes do treinamento/ajuste do step.
        """
        ...

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas e metadados relevantes da execução/aplicação do step.
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o estado treinado do step em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o estado treinado do step a partir do disco.
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        Retorna um dicionário serializável com o estado treinado do step.
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaura o estado treinado do step a partir de um dicionário.
        """
        ...