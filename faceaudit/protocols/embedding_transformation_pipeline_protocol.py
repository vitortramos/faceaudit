# faceaudit/protocols/embedding_transformation_pipeline_protocol.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from faceaudit.models import EmbeddingDataset, EmbeddingInfo
from faceaudit.enums.pipeline_mode import PipelineMode
from faceaudit.models.pipeline.pipeline_training_policy import PipelineTrainingPolicy
from faceaudit.models.pipeline.pipeline_application_policy import PipelineApplicationPolicy
from faceaudit.models.transformation_pipeline_config import TransformationPipelineConfig
from faceaudit.protocols.embedding_transformation_step_protocol import (
    EmbeddingTransformationStepProtocol,
)


@runtime_checkable
class EmbeddingTransformationPipelineProtocol(Protocol):
    """
    Protocolo para o novo pipeline de transformação de embeddings.

    O pipeline pode operar em dois modos:
    - LINEAR: um único fluxo de transformação
    - DUAL: dois fluxos (Source e Target), tipicamente em cenários com
      adaptação de domínio
    """

    @property
    def mode(self) -> PipelineMode:
        """
        Modo estrutural do pipeline.
        """
        ...

    @property
    def config(self) -> Optional[TransformationPipelineConfig]:
        """
        Configuração do pipeline linear, quando aplicável.
        """
        ...

    @property
    def source_config(self) -> Optional[TransformationPipelineConfig]:
        """
        Configuração do ramo Source no modo DUAL.
        """
        ...

    @property
    def target_config(self) -> Optional[TransformationPipelineConfig]:
        """
        Configuração do ramo Target no modo DUAL.
        """
        ...

    @property
    def steps(self) -> List[EmbeddingTransformationStepProtocol]:
        """
        Lista de steps do pipeline linear.
        """
        ...

    @property
    def source_steps(self) -> List[EmbeddingTransformationStepProtocol]:
        """
        Lista de steps do ramo Source no modo DUAL.
        """
        ...

    @property
    def target_steps(self) -> List[EmbeddingTransformationStepProtocol]:
        """
        Lista de steps do ramo Target no modo DUAL.
        """
        ...

    def fit(
        self,
        dataset: EmbeddingDataset,
        training_policy: Optional[PipelineTrainingPolicy] = None,
    ) -> None:
        """
        Treina o pipeline em modo LINEAR.

        Args:
            dataset:
                Dataset de treino.
            training_policy:
                Política opcional de treino. Em pipelines lineares, normalmente
                será a política padrão linear.
        """
        ...

    def fit_dual(
        self,
        source_dataset: EmbeddingDataset,
        target_dataset: EmbeddingDataset,
        training_policy: Optional[PipelineTrainingPolicy] = None,
    ) -> None:
        """
        Treina o pipeline em modo DUAL.

        Args:
            source_dataset:
                Dataset do ramo Source.
            target_dataset:
                Dataset do ramo Target.
            training_policy:
                Política opcional de treino dual.
        """
        ...

    def transform(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        """
        Aplica o pipeline linear a um dataset completo.
        """
        ...

    def transform_source(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        """
        Aplica o ramo Source do pipeline dual a um dataset.
        """
        ...

    def transform_target(
        self,
        dataset: EmbeddingDataset,
        application_policy: Optional[PipelineApplicationPolicy] = None,
    ) -> EmbeddingDataset:
        """
        Aplica o ramo Target do pipeline dual a um dataset.
        """
        ...

    def transform_single(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica o pipeline linear a um único EmbeddingInfo.
        """
        ...

    def transform_single_source(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica o ramo Source do pipeline dual a um único EmbeddingInfo.
        """
        ...

    def transform_single_target(self, embedding_info: EmbeddingInfo) -> EmbeddingInfo:
        """
        Aplica o ramo Target do pipeline dual a um único EmbeddingInfo.
        """
        ...

    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas agregadas do treinamento do pipeline.
        """
        ...

    def get_execution_metrics(self) -> Dict[str, Any]:
        """
        Retorna métricas agregadas da última aplicação do pipeline.
        """
        ...

    def save(self, path: str) -> None:
        """
        Persiste o pipeline treinado em disco.
        """
        ...

    def load(self, path: str) -> None:
        """
        Restaura o pipeline treinado a partir do disco.
        """
        ...

    def get_state(self) -> Dict[str, Any]:
        """
        Retorna um dicionário serializável com o estado treinado do pipeline.
        """
        ...

    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restaura o estado treinado do pipeline a partir de um dicionário.
        """
        ...