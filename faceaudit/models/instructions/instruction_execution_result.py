# faceaudit/models/instructions/instruction_execution_result.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Type, TypeVar, Optional

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference


T = TypeVar("T", bound="InstructionExecutionResult")


@dataclass
class InstructionExecutionResult(JSONSerializableMixin):
    """
    Representa o resultado estruturado da execução de uma instrução.

    Esse objeto será retornado pelos executores de instrução e poderá ser usado
    para atualizar o contexto de execução, registrar artefatos produzidos e
    armazenar metadados de auditoria.
    """

    success: bool = False
    produced_resource_ids: List[str] = field(default_factory=list)
    produced_artifacts: List[StrategyArtifactReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        """
        Retorna True se houver erros registrados.
        """
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """
        Retorna True se houver avisos registrados.
        """
        return bool(self.warnings)

    def add_error(self, message: str) -> None:
        """
        Adiciona um erro ao resultado e marca a execução como malsucedida.
        """
        if message:
            self.errors.append(message)
            self.success = False

    def add_warning(self, message: str) -> None:
        """
        Adiciona um aviso ao resultado.
        """
        if message:
            self.warnings.append(message)

    def add_resource_id(self, resource_id: str) -> None:
        """
        Registra o identificador de um recurso produzido pela instrução.
        """
        if resource_id and resource_id not in self.produced_resource_ids:
            self.produced_resource_ids.append(resource_id)

    def add_artifact(self, artifact: StrategyArtifactReference) -> None:
        """
        Registra um artefato produzido pela instrução.
        """
        if artifact is not None:
            self.produced_artifacts.append(artifact)

    def merge(self, other: Optional["InstructionExecutionResult"]) -> "InstructionExecutionResult":
        """
        Mescla outro resultado neste objeto.
        """
        if other is None:
            return self

        self.produced_resource_ids.extend(
            rid for rid in other.produced_resource_ids
            if rid not in self.produced_resource_ids
        )
        self.produced_artifacts.extend(other.produced_artifacts)
        self.metadata.update(other.metadata)
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)

        self.success = self.success and other.success if self.success or other.success else False

        return self

    @classmethod
    def ok(
        cls: Type[T],
        produced_resource_ids: Optional[List[str]] = None,
        produced_artifacts: Optional[List[StrategyArtifactReference]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Fábrica para execução bem-sucedida.
        """
        return cls(
            success=True,
            produced_resource_ids=produced_resource_ids or [],
            produced_artifacts=produced_artifacts or [],
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls: Type[T],
        errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        Fábrica para execução malsucedida.
        """
        return cls(
            success=False,
            errors=errors or [],
            metadata=metadata or {},
        )

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o DTO, hidratando os artefatos corretamente.
        """
        if not data:
            return cls()

        processed_data = data.copy()

        if "produced_artifacts" in processed_data and isinstance(processed_data["produced_artifacts"], list):
            processed_data["produced_artifacts"] = [
                StrategyArtifactReference.from_dict(item) if isinstance(item, dict) else item
                for item in processed_data["produced_artifacts"]
            ]

        return cls(**processed_data)