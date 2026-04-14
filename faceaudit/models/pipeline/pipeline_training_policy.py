# faceaudit/models/pipeline/pipeline_training_policy.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="PipelineTrainingPolicy")


@dataclass
class PipelineTrainingPolicy(JSONSerializableMixin):
    """
    Define a política neutra de treinamento do EmbeddingTransformationPipeline.

    Essa política controla como os steps do pipeline utilizarão os datasets
    disponíveis durante o treinamento, especialmente no modo DUAL.

    Ela foi desenhada para ser genérica e desacoplada de PLDA, permitindo
    reutilização por diferentes estratégias de preparação.

    Atributos:
        use_target_for_coral:
            Se True, permite que o step CORAL utilize o target_dataset no fit.
        use_target_for_whitening:
            Se True, permite que o Whitening seja treinado no espaço combinado
            entre Source e Target.
        use_target_for_lda:
            Se True, permite que o LDA considere também o target_dataset no fit.
        copy_shared_state_to_target:
            Se True, steps compartilhados poderão propagar seu estado treinado
            para o ramo Target no modo DUAL.
    """

    use_target_for_coral: bool = field(
        default=True,
        metadata={"help": "Permite uso do target_dataset no treino do CORAL."}
    )
    use_target_for_whitening: bool = field(
        default=False,
        metadata={"help": "Permite uso do target_dataset no treino do Whitening."}
    )
    use_target_for_lda: bool = field(
        default=False,
        metadata={"help": "Permite uso do target_dataset no treino do LDA."}
    )
    copy_shared_state_to_target: bool = field(
        default=True,
        metadata={"help": "Permite copiar o estado treinado de steps compartilhados para o Target."}
    )

    @classmethod
    def default_linear(cls: Type[T]) -> T:
        """
        Fábrica para a política padrão de pipelines lineares.
        """
        return cls(
            use_target_for_coral=False,
            use_target_for_whitening=False,
            use_target_for_lda=False,
            copy_shared_state_to_target=False
        )

    @classmethod
    def default_dual(cls: Type[T]) -> T:
        """
        Fábrica para a política padrão de pipelines duais com adaptação de domínio.
        """
        return cls(
            use_target_for_coral=True,
            use_target_for_whitening=False,
            use_target_for_lda=False,
            copy_shared_state_to_target=True
        )

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO a partir de dicionário.

        Como este objeto trabalha apenas com tipos primitivos, a hidratação
        é direta.
        """
        if not data:
            return cls()

        return cls(**data)