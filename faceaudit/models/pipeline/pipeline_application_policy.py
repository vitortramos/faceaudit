# faceaudit/models/pipeline/pipeline_application_policy.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin


T = TypeVar("T", bound="PipelineApplicationPolicy")


@dataclass
class PipelineApplicationPolicy(JSONSerializableMixin):
    """
    Define a política de aplicação do EmbeddingTransformationPipeline.

    Essa política controla aspectos operacionais da fase de aplicação, como:
    - geração de checkpoints intermediários
    - persistência do dataset transformado
    - registro da trajetória da transformação

    Ela é neutra em relação ao domínio de PLDA e pode ser reutilizada por
    diferentes estratégias e workflows.
    """

    save_checkpoints: bool = field(
        default=True,
        metadata={"help": "Indica se checkpoints intermediários devem ser persistidos."}
    )
    save_transformed_dataset: bool = field(
        default=True,
        metadata={"help": "Indica se o dataset final transformado deve ser persistido."}
    )
    save_trajectory_data: bool = field(
        default=True,
        metadata={"help": "Indica se os dados da trajetória da transformação devem ser preservados."}
    )

    @classmethod
    def default(cls: Type[T]) -> T:
        """
        Fábrica para a política padrão de aplicação (todos os artefatos persistidos).
        """
        return cls(
            save_checkpoints=True,
            save_transformed_dataset=True,
            save_trajectory_data=True
        )

    @classmethod
    def verbose(cls: Type[T]) -> T:
        """
        Fábrica para uma política mais detalhada, adequada para auditoria
        e inspeção do processo de transformação.
        """
        return cls(
            save_checkpoints=True,
            save_transformed_dataset=True,
            save_trajectory_data=True
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