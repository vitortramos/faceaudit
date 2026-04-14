# faceaudit/contexts/strategy_execution_context.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.strategy_execution_context_protocol import StrategyExecutionContextProtocol
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol


T = TypeVar("T", bound="StrategyExecutionContext")


@dataclass
class StrategyExecutionContext(JSONSerializableMixin, StrategyExecutionContextProtocol):
    """
    Implementação concreta do contexto de execução da estratégia.

    Responsabilidades:
    - armazenar recursos disponíveis durante a execução
    - armazenar artefatos produzidos
    - registrar regras de validação disponíveis
    - acumular logs
    - manter metadados auxiliares da execução
    """

    resources: Dict[str, StrategyResource] = field(default_factory=dict)
    artifacts: Dict[str, StrategyArtifactReference] = field(default_factory=dict)
    validation_rules: Dict[str, ValidationRuleProtocol] = field(default_factory=dict)
    execution_logs: List[str] = field(default_factory=list)
    execution_metadata: Dict[str, Any] = field(default_factory=dict)

    # =========================================================================
    # RECURSOS
    # =========================================================================

    def get_resource(self, resource_id: str) -> Optional[StrategyResource]:
        """
        Retorna o recurso correspondente ao identificador informado.
        """
        return self.resources.get(resource_id)

    def set_resource(self, resource: StrategyResource) -> None:
        """
        Registra ou substitui um recurso no contexto.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")
        self.resources[resource.resource_id] = resource

    def has_resource(self, resource_id: str) -> bool:
        """
        Retorna True se o recurso existir no contexto.
        """
        return resource_id in self.resources

    def remove_resource(self, resource_id: str) -> None:
        """
        Remove um recurso do contexto, se existir.
        """
        self.resources.pop(resource_id, None)

    def list_resources(self) -> List[StrategyResource]:
        """
        Lista todos os recursos registrados no contexto.
        """
        return list(self.resources.values())

    # =========================================================================
    # ARTEFATOS
    # =========================================================================

    def get_artifact(self, key: str) -> Optional[StrategyArtifactReference]:
        """
        Retorna o artefato correspondente à chave informada.
        """
        return self.artifacts.get(key)

    def set_artifact(self, key: str, artifact: StrategyArtifactReference) -> None:
        """
        Registra ou substitui um artefato no contexto.
        """
        if not key:
            raise ValueError("A chave do artefato não pode ser vazia.")
        if artifact is None:
            raise ValueError("O artefato não pode ser nulo.")
        self.artifacts[key] = artifact

    def has_artifact(self, key: str) -> bool:
        """
        Retorna True se o artefato existir no contexto.
        """
        return key in self.artifacts

    def remove_artifact(self, key: str) -> None:
        """
        Remove um artefato do contexto, se existir.
        """
        self.artifacts.pop(key, None)

    def list_artifacts(self) -> List[StrategyArtifactReference]:
        """
        Lista todos os artefatos registrados no contexto.
        """
        return list(self.artifacts.values())

    # =========================================================================
    # REGRAS DE VALIDAÇÃO
    # =========================================================================

    def register_validation_rule(self, rule: ValidationRuleProtocol) -> None:
        """
        Registra uma regra de validação disponível no contexto.
        """
        if rule is None:
            raise ValueError("A validation rule não pode ser nula.")
        self.validation_rules[rule.rule_id] = rule

    def get_validation_rule(self, rule_id: str) -> Optional[ValidationRuleProtocol]:
        """
        Retorna a regra de validação correspondente ao identificador informado.
        """
        return self.validation_rules.get(rule_id)

    def has_validation_rule(self, rule_id: str) -> bool:
        """
        Retorna True se a regra de validação existir no contexto.
        """
        return rule_id in self.validation_rules

    def remove_validation_rule(self, rule_id: str) -> None:
        """
        Remove uma regra de validação do contexto, se existir.
        """
        self.validation_rules.pop(rule_id, None)

    def list_validation_rules(self) -> List[ValidationRuleProtocol]:
        """
        Lista todas as regras de validação registradas no contexto.
        """
        return list(self.validation_rules.values())

    # =========================================================================
    # LOGS E METADADOS
    # =========================================================================

    def log(self, message: str) -> None:
        """
        Registra uma mensagem de log no contexto.
        """
        if message:
            self.execution_logs.append(message)

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Registra ou atualiza um metadado de execução.
        """
        self.execution_metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Recupera um metadado de execução.
        """
        return self.execution_metadata.get(key, default)

    # =========================================================================
    # SERIALIZAÇÃO
    # =========================================================================

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o contexto a partir de um dicionário.

        Observação:
        - resources e artifacts são hidratados
        - validation_rules não é reconstruído automaticamente, pois contém
          objetos comportamentais e será normalmente re-registrado em runtime
        """
        if not data:
            return cls()

        processed_data = data.copy()

        if "resources" in processed_data and isinstance(processed_data["resources"], dict):
            processed_data["resources"] = {
                key: StrategyResource.from_dict(value) if isinstance(value, dict) else value
                for key, value in processed_data["resources"].items()
            }

        if "artifacts" in processed_data and isinstance(processed_data["artifacts"], dict):
            processed_data["artifacts"] = {
                key: StrategyArtifactReference.from_dict(value) if isinstance(value, dict) else value
                for key, value in processed_data["artifacts"].items()
            }

        # validation_rules não é serializado/hidratado automaticamente
        if "validation_rules" in processed_data:
            processed_data["validation_rules"] = {}

        return cls(**processed_data)