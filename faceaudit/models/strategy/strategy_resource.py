# faceaudit/models/strategy/strategy_resource.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar, Any

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.enums.resource_category import ResourceCategory
from faceaudit.enums.strategy_resource_type import StrategyResourceType
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.resource_id_generator import (
    generate_resource_id,
    is_valid_resource_id,
    migrate_resource_id,
)


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
        id:
            Identificador único técnico do recurso (formato `DS-8f3a1b2c`).
            Gerado automaticamente quando o recurso é criado pela primeira vez.
            Estável: nunca muda mesmo se `name` for renomeado pelo usuário.
            Migração de projetos antigos: derivado deterministicamente do
            `resource_id` legado (hash SHA-256[:8]) — garante mesmo id em
            qualquer máquina onde o projeto seja aberto.
        resource_id:
            **DEPRECATED — manter por retrocompat até a Fase 4 da migração.**
            Antigo identificador único do recurso. Hoje é usado em paralelo
            com `id` durante a transição. Pode ser igual ao `name` ou ao
            stem do arquivo. Novos códigos devem usar `id`.
        name:
            Nome amigável para exibição e referência humana. EDITÁVEL pelo
            usuário sem afetar `id`. Pode haver dois recursos com mesmo
            `name` (em pastas/contextos diferentes) — IDs garantem unicidade.
        resource_type:
            Tipo lógico do recurso.
        payload:
            Objeto em memória associado ao recurso. Pode ser um DTO, modelo,
            configuração ou outro objeto do domínio.
        artifact_ref:
            Referência opcional ao artefato físico persistido em disco.
    """

    resource_id: str = field(
        metadata={
            "help": (
                "DEPRECATED desde Fase 2 da migração id/name. Use `id` "
                "para chave técnica e `name` para display. Mantido por "
                "retrocompat — será removido na Fase 4."
            ),
        }
    )
    name: str = field(
        metadata={"help": "Nome amigável do recurso (editável)."}
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
    id: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Identificador único técnico no formato `DS-8f3a1b2c`. "
                "Gerado automaticamente em __post_init__ se não informado. "
                "Estável (não muda em rename do `name`). "
                "Em migração de projeto antigo, derivado deterministicamente "
                "do `resource_id` legado via hash SHA-256."
            ),
        },
    )

    def __post_init__(self) -> None:
        """Garante que `id` esteja preenchido.

        Três cenários:

        1. **Recurso novo com `id` explícito**: aceita o valor (caller é
           responsável por usar `generate_resource_id` ou similar).
        2. **Recurso novo sem `id`**: gera aleatório baseado em `resource_type`.
        3. **Migração de JSON antigo** (id ausente, resource_id presente):
           gera id determinístico a partir do `resource_id` + tipo, garantindo
           consistência cross-machine.
        """
        if self.id is None or not is_valid_resource_id(self.id):
            if self.resource_id:
                # Migração: deriva id determinístico do legacy resource_id.
                self.id = migrate_resource_id(self.resource_id, self.resource_type)
            else:
                # Recurso novo sem resource_id — gera aleatório.
                self.id = generate_resource_id(self.resource_type)

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
            "id": self.id,                       # Fase 2: id técnico estável
            "resource_id": self.resource_id,     # DEPRECATED — mantido até Fase 4
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

        Score datasets têm 3 subtipos físicos distinguidos via metadata
        do payload (`is_mined`/`is_score_vector`) — usados pelo
        `StrategyFileType` pra rotear pra subpasta correta:
            scores/<sid>/raw/         → default
            scores/<sid>/mined/       → metadata.is_mined=True
            scores/<sid>/vectorized/  → metadata.is_score_vector=True

        Retorna None pra (PRIMARY) ou tipos sem mapeamento de artefato.
        """
        from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
        from faceaudit.enums.strategy_resource_type import StrategyResourceType

        if self.category == ResourceCategory.PRIMARY:
            return None

        base = self.resource_type.to_artifact_type()
        if (
            self.resource_type == StrategyResourceType.SIMILARITY_SCORES
            and self.payload is not None
        ):
            metadata = getattr(self.payload, "metadata", None) or {}
            if metadata.get("is_score_vector"):
                return StrategyArtifactType.VECTORIZED_SCORES
            if metadata.get("is_mined"):
                return StrategyArtifactType.MINED_SCORES
        return base

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