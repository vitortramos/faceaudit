# faceaudit/models/strategy/lr_strategy_definition.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar

from faceaudit.enums.embedding_type import EmbeddingType
from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.strategy.strategy_resource import StrategyResource


T = TypeVar("T", bound="LRStrategyDefinition")


@dataclass
class LRStrategyDefinition(JSONSerializableMixin):
    """
    Representa a definição declarativa e editável de uma estratégia de cálculo de LR.

    Esta classe descreve a estratégia como configuração:
    - metadados
    - recursos declarados
    - instruções declaradas
    - regras de validação registradas por identificador

    Ela não executa a estratégia diretamente. A execução/preparação ficará a cargo
    de serviços/executores apropriados na nova arquitetura.
    """

    strategy_id: str = field(
        metadata={"help": "Identificador único da estratégia."}
    )
    name: str = field(
        metadata={"help": "Nome amigável da estratégia."}
    )
    description: Optional[str] = field(
        default=None,
        metadata={"help": "Descrição opcional da finalidade da estratégia."}
    )

    resources: List[StrategyResource] = field(
        default_factory=list,
        metadata={"help": "Lista de recursos declarados da estratégia."}
    )

    preparation_instructions: List[StrategyInstructionDefinition] = field(
        default_factory=list,
        metadata={"help": "Instruções de preparação (treino de pipeline, modelos)."}
    )
    validation_instructions: List[StrategyInstructionDefinition] = field(
        default_factory=list,
        metadata={"help": "Instruções de validação (métricas, gráficos)."}
    )
    casework_instructions: List[StrategyInstructionDefinition] = field(
        default_factory=list,
        metadata={"help": "Instruções de análise de caso (escores do caso)."}
    )

    validation_rule_ids: List[str] = field(
        default_factory=list,
        metadata={"help": "Lista de regras de validação registradas na estratégia."}
    )

    embedding_column: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Nome da coluna de embedding (no parquet) que esta estratégia usa. "
                "Todos os datasets envolvidos precisam ter essa coluna com o mesmo "
                "model_name (consistência cross-dataset garantida pela validação). "
                "None = auto-detect quando os datasets têm uma única coluna de "
                "embedding em comum."
            )
        }
    )

    version: str = field(
        default="1.0",
        metadata={"help": "Versão/schema da definição da estratégia."}
    )

    @property
    def instructions(self) -> List[StrategyInstructionDefinition]:
        """
        Retorna todas as instruções de todas as fases em ordem.
        Read-only — para adicionar/remover, use os métodos específicos.
        """
        return (
            self.preparation_instructions
            + self.validation_instructions
            + self.casework_instructions
        )

    @property
    def has_resources(self) -> bool:
        return bool(self.resources)

    @property
    def has_instructions(self) -> bool:
        return bool(
            self.preparation_instructions
            or self.validation_instructions
            or self.casework_instructions
        )

    @property
    def has_validation_rules(self) -> bool:
        """
        Retorna True se a estratégia possui regras de validação registradas.
        """
        return bool(self.validation_rule_ids)

    def get_resource_by_id(self, resource_id: str) -> Optional[StrategyResource]:
        """
        Retorna o recurso correspondente ao resource_id informado.
        """
        for resource in self.resources:
            if resource.resource_id == resource_id:
                return resource
        return None

    def get_instruction_by_id(
        self,
        instruction_id: str
    ) -> Optional[StrategyInstructionDefinition]:
        """
        Retorna a instrução correspondente ao instruction_id informado.
        """
        for instruction in self.instructions:
            if instruction.instruction_id == instruction_id:
                return instruction
        return None

    def add_resource(self, resource: StrategyResource) -> None:
        """
        Adiciona um recurso à estratégia.

        Lança erro se já existir um recurso com o mesmo resource_id.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")

        if self.get_resource_by_id(resource.resource_id) is not None:
            raise ValueError(
                f"Já existe um recurso com resource_id '{resource.resource_id}'."
            )

        self.resources.append(resource)

    def update_resource(self, resource: StrategyResource) -> None:
        """
        Atualiza um recurso já existente na estratégia.

        Lança erro se o recurso não existir.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")

        for index, current in enumerate(self.resources):
            if current.resource_id == resource.resource_id:
                self.resources[index] = resource
                return

        raise ValueError(
            f"Não foi encontrado recurso com resource_id '{resource.resource_id}' para atualização."
        )

    def remove_resource(self, resource_id: str) -> None:
        """
        Remove um recurso da estratégia pelo identificador.

        Se o recurso não existir, não faz nada.
        """
        self.resources = [
            resource for resource in self.resources
            if resource.resource_id != resource_id
        ]

    def add_instruction(
        self,
        instruction: StrategyInstructionDefinition,
        phase: str = "preparation",
    ) -> None:
        """
        Adiciona uma instrução à fase indicada.

        Lança erro se já existir uma instrução com o mesmo instruction_id.
        """
        if instruction is None:
            raise ValueError("A instrução não pode ser nula.")

        if self.get_instruction_by_id(instruction.instruction_id) is not None:
            raise ValueError(
                f"Já existe uma instrução com instruction_id '{instruction.instruction_id}'."
            )

        self._get_phase_list(phase).append(instruction)

    def update_instruction(self, instruction: StrategyInstructionDefinition) -> None:
        """
        Atualiza uma instrução já existente em qualquer fase.

        Lança erro se a instrução não existir.
        """
        if instruction is None:
            raise ValueError("A instrução não pode ser nula.")

        for phase_list in self._phase_lists():
            for index, current in enumerate(phase_list):
                if current.instruction_id == instruction.instruction_id:
                    phase_list[index] = instruction
                    return

        raise ValueError(
            f"Não foi encontrada instrução com instruction_id "
            f"'{instruction.instruction_id}' para atualização."
        )

    def remove_instruction(self, instruction_id: str) -> None:
        """
        Remove uma instrução de qualquer fase pelo identificador.
        """
        for phase_list in self._phase_lists():
            phase_list[:] = [
                i for i in phase_list if i.instruction_id != instruction_id
            ]

    def _phase_lists(self) -> List[List[StrategyInstructionDefinition]]:
        return [
            self.preparation_instructions,
            self.validation_instructions,
            self.casework_instructions,
        ]

    def _get_phase_list(self, phase: str) -> List[StrategyInstructionDefinition]:
        _map = {
            "preparation": self.preparation_instructions,
            "validation": self.validation_instructions,
            "casework": self.casework_instructions,
        }
        if phase not in _map:
            raise ValueError(f"Fase inválida: '{phase}'. Use: {list(_map.keys())}")
        return _map[phase]

    def add_validation_rule_id(self, rule_id: str) -> None:
        """
        Registra um identificador de regra de validação na estratégia.
        """
        if rule_id and rule_id not in self.validation_rule_ids:
            self.validation_rule_ids.append(rule_id)

    def remove_validation_rule_id(self, rule_id: str) -> None:
        """
        Remove um identificador de regra de validação da estratégia.
        """
        self.validation_rule_ids = [
            current_rule_id
            for current_rule_id in self.validation_rule_ids
            if current_rule_id != rule_id
        ]

    def to_dict(self) -> dict:
        """
        Serializa a definição para JSON delegando aos `to_dict()` dos filhos.

        ATENÇÃO: a implementação padrão do JSONSerializableMixin usa
        `dataclasses.asdict()`, que recursa em dataclasses filhos por conta
        própria — SEM chamar o `to_dict()` deles. Como `StrategyResource`
        precisa omitir o campo `payload` (objeto runtime, não JSON-serializável,
        ex: EmbeddingTransformationPipeline) e as instruções concretas podem ter
        DTOs aninhados com Enums que exigem `to_str()`, é necessário despachar
        manualmente para os `to_dict()` dos filhos.
        """
        def _child_dict(obj):
            if obj is None:
                return None
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            return obj

        return {
            "strategy_id": self.strategy_id,
            "name": self.name,
            "description": self.description,
            "resources": [_child_dict(r) for r in self.resources],
            "preparation_instructions": [_child_dict(i) for i in self.preparation_instructions],
            "validation_instructions": [_child_dict(i) for i in self.validation_instructions],
            "casework_instructions": [_child_dict(i) for i in self.casework_instructions],
            "validation_rule_ids": list(self.validation_rule_ids),
            "embedding_column": self.embedding_column,
            "version": self.version,
        }

    @classmethod
    def from_dict(
        cls: Type[T],
        data: dict,
        *,
        allow_legacy_auto_migration: bool = True,
    ) -> Optional[T]:
        """
        Reconstrói a definição da estratégia, hidratando corretamente:
        - resources
        - instructions

        Observação:
        validation_rule_ids permanece como lista simples de strings.

        Migração preservativa (PR 2):
        - Detecta sinais legados (V2.14) via `LegacyStrategyMigrationService`.
        - Converte em memória — NÃO grava em disco; o JSON original
          permanece intacto. A escrita da cópia `*.v2_15.fa` é acionada
          pela UI/CLI via `LegacyStrategyMigrationService.write_side_by_side_copy`.
        - `allow_legacy_auto_migration=False` (modo estrito) levanta
          `LegacyStrategyVersionRejected` ao detectar formato legado.
        """
        if not data:
            return None

        # Migração preservativa (PR 2) — primeira coisa antes do parsing.
        from faceaudit.services.legacy_strategy_migration_service import (
            LegacyStrategyMigrationService,
            LegacyStrategyVersionRejected,
        )
        plan = LegacyStrategyMigrationService.detect(data)
        if plan.is_needed:
            if not allow_legacy_auto_migration:
                raise LegacyStrategyVersionRejected(plan.legacy_signals_found)
            data = LegacyStrategyMigrationService.convert_in_memory(data, plan)

        processed_data = data.copy()
        # Bloco informativo de auditoria gravado lado-a-lado pela cópia
        # migrada — não faz parte do DTO. Filtra antes de `cls(**...)`.
        processed_data.pop("_migration_metadata", None)

        if "resources" in processed_data and isinstance(processed_data["resources"], list):
            processed_data["resources"] = [
                StrategyResource.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["resources"]
            ]

        for key in ("preparation_instructions", "validation_instructions", "casework_instructions"):
            if key in processed_data and isinstance(processed_data[key], list):
                # Itens que retornam None (instruction_type extinto, ex:
                # `SCORE_FUSION` removido) são filtrados — o resto da
                # estratégia carrega normalmente sem essa instrução. Sem
                # esse filter, `cls(**...)` recebe None na lista e quebra.
                rebuilt = []
                for item in processed_data[key]:
                    if isinstance(item, dict):
                        instr = StrategyInstructionDefinition.from_dict(item)
                        if instr is not None:
                            rebuilt.append(instr)
                    else:
                        rebuilt.append(item)
                processed_data[key] = rebuilt

        # Migração: estratégias antigas gravam `embedding_type` (enum
        # SEPAEL_V2/V4). Mapeamos pro nome da coluna parquet correspondente
        # e descartamos o campo legado para que `cls(**...)` não receba
        # kwarg desconhecido.
        if "embedding_type" in processed_data:
            legacy = processed_data.pop("embedding_type")
            if legacy and not processed_data.get("embedding_column"):
                try:
                    enum_val = (
                        legacy if isinstance(legacy, EmbeddingType)
                        else EmbeddingType.from_str(legacy)
                    )
                    processed_data["embedding_column"] = enum_val.parquet_column
                except (KeyError, ValueError, AttributeError):
                    pass

        return cls(**processed_data)