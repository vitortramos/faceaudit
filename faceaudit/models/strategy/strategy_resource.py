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

    def to_dict(self) -> dict:
        """
        Serializa o recurso para JSON omitindo o campo `payload`.

        O payload é estritamente runtime — modelos treinados, datasets em
        memória, pipelines hidratados — e nem sempre é JSON-serializável
        (ex: EmbeddingTransformationPipeline). A persistência do conteúdo
        físico do recurso é responsabilidade do `artifact_ref`.

        ATENÇÃO PERFORMANCE: NÃO chamamos `super().to_dict()` aqui porque
        a implementação do mixin usa `dataclasses.asdict(self)` que recursa
        em TODOS os campos do dataclass — incluindo `payload`. Para um
        SimilarityScoreDataset com 5M+ samples (cada com EmbeddingInfo
        aninhado + ndarray), o asdict aloca e itera ~70M+ objetos Python
        antes de o `pop("payload")` ter chance de remover. O save da
        definition ficava cada vez mais lento à medida que recursos com
        payloads gigantes apareciam na lista. Construímos o dicionário
        diretamente sem tocar no payload.
        """
        artifact_ref_data = (
            self.artifact_ref.to_dict()
            if (self.artifact_ref is not None and hasattr(self.artifact_ref, "to_dict"))
            else self.artifact_ref
        )
        resource_type_data = (
            self.resource_type.to_str()
            if hasattr(self.resource_type, "to_str")
            else (
                self.resource_type.value
                if hasattr(self.resource_type, "value")
                else self.resource_type
            )
        )
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "resource_type": resource_type_data,
            "artifact_ref": artifact_ref_data,
            # payload é runtime-only, intencionalmente omitido.
        }

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