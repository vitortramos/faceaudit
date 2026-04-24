# faceaudit/models/strategy/strategy_resource.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Any

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference


T = TypeVar("T", bound="StrategyResource")


@dataclass
class StrategyResource(JSONSerializableMixin):
    """
    Representa um recurso lógico da estratégia.

    Um recurso pode encapsular:
    - um dataset
    - uma configuração
    - um modelo treinado
    - um artefato intermediário
    - qualquer outro objeto reutilizável dentro da estratégia

    Atributos:
        resource_id:
            Identificador único do recurso dentro da estratégia.
        name:
            Nome amigável para exibição e referência humana.
        resource_type:
            Tipo lógico do recurso.
        payload:
            Objeto em memória associado ao recurso. Pode ser um DTO, modelo,
            configuração ou outro objeto do domínio.
        artifact_ref:
            Referência opcional ao artefato físico persistido em disco.
    """

    resource_id: str = field(
        metadata={"help": "Identificador único do recurso dentro da estratégia."}
    )
    name: str = field(
        metadata={"help": "Nome amigável do recurso."}
    )
    resource_type: StrategyResourceType = field(
        metadata={"help": "Tipo lógico do recurso."}
    )
    payload: Any = field(
        default=None,
        metadata={"help": "Objeto em memória associado a este recurso."}
    )
    artifact_ref: Optional[StrategyArtifactReference] = field(
        default=None,
        metadata={"help": "Referência opcional ao artefato físico persistido."}
    )

    @property
    def has_payload(self) -> bool:
        """
        Retorna True se o recurso possui objeto em memória associado.
        """
        return self.payload is not None

    @property
    def has_artifact(self) -> bool:
        """
        Retorna True se o recurso possui referência a artefato persistido.
        """
        return self.artifact_ref is not None

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, hidratando corretamente:
        - o enum resource_type
        - a referência de artefato, quando presente

        Observação:
        O payload é mantido como veio do JSON/dict. A hidratação forte do payload
        ficará a cargo de camadas superiores, pois ele pode apontar para muitos
        tipos diferentes de objeto.
        """
        if not data:
            return None

        processed_data = data.copy()

        if "resource_type" in processed_data and processed_data["resource_type"]:
            processed_data["resource_type"] = StrategyResourceType.from_str(
                processed_data["resource_type"]
            )

        if "artifact_ref" in processed_data and isinstance(processed_data["artifact_ref"], dict):
            processed_data["artifact_ref"] = StrategyArtifactReference.from_dict(
                processed_data["artifact_ref"]
            )

        # Retrocompat: campo removido — embedding_type agora vive na LRStrategyDefinition
        processed_data.pop("embedding_column", None)

        return cls(**processed_data)