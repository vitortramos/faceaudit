# faceaudit/models/strategy/strategy_resource.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Any

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.resource_category import ResourceCategory
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
    category: ResourceCategory = field(
        default=ResourceCategory.DERIVED,
        metadata={
            "help": (
                "Origem do recurso: PRIMARY se foi importado pelo usuário "
                "via 'Adicionar Recurso', DERIVED se foi produzido por "
                "execução de instrução. Default DERIVED — os fluxos de "
                "importação setam PRIMARY explicitamente."
            ),
        },
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
            "category": self.category.value,
            # payload é runtime-only, intencionalmente omitido.
        }

    def to_artifact_type(self):
        """Tipo de artefato físico pra persistência, ciente da categoria.

        Recursos PRIMARY não são re-persistidos pelo executor (já vivem
        em disco — foram importados; substituir destruiria o original).
        Pra DERIVED, delega ao `StrategyResourceType.to_artifact_type()`.

        Retorna None pra (PRIMARY) ou tipos sem mapeamento de artefato.
        """
        if self.category == ResourceCategory.PRIMARY:
            return None
        return self.resource_type.to_artifact_type()

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

        # Captura a string original ANTES da normalização — usada pra
        # detectar JSONs legados onde `resource_type="INTERMEDIATE_DATASET"`
        # (tipo extinto, agora é DATASET + category=DERIVED).
        raw_resource_type = (
            processed_data.get("resource_type") or ""
        ).upper().strip()

        if "resource_type" in processed_data and processed_data["resource_type"]:
            processed_data["resource_type"] = StrategyResourceType.from_str(
                processed_data["resource_type"]
            )

        if "artifact_ref" in processed_data and isinstance(processed_data["artifact_ref"], dict):
            processed_data["artifact_ref"] = StrategyArtifactReference.from_dict(
                processed_data["artifact_ref"]
            )

        # Migração de categoria: JSONs antigos não tinham o campo. Infere
        # a partir do `resource_type` original e dos defaults conhecidos:
        #   - "INTERMEDIATE_DATASET" legado → DATASET + DERIVED
        #   - "DATASET" legado → assumimos PRIMARY (era usado só pra
        #     datasets importados pelo usuário antes da unificação)
        #   - Demais tipos → DERIVED (default sensato; trained models,
        #     scores, validation_result etc. nunca são primários)
        if "category" in processed_data and processed_data["category"]:
            processed_data["category"] = ResourceCategory.from_str(
                processed_data["category"]
            )
        else:
            if raw_resource_type == "INTERMEDIATE_DATASET":
                processed_data["category"] = ResourceCategory.DERIVED
            elif raw_resource_type == "DATASET":
                processed_data["category"] = ResourceCategory.PRIMARY
            else:
                # Configs primários (PIPELINE_CONFIG/SPLIT_CONFIG/MINING_CONFIG)
                # — tudo importado via "Adicionar Recurso". Demais derivados.
                if raw_resource_type in {
                    "PIPELINE_CONFIG", "SPLIT_CONFIG", "MINING_CONFIG",
                    "MERGE_CONFIG", "SIMILARITY_MODEL_CONFIG",
                    "CALIBRATION_MODEL_CONFIG",
                }:
                    processed_data["category"] = ResourceCategory.PRIMARY
                else:
                    processed_data["category"] = ResourceCategory.DERIVED

        # Retrocompat: campo removido — embedding_type agora vive na LRStrategyDefinition
        processed_data.pop("embedding_column", None)

        return cls(**processed_data)