# faceaudit/protocols/strategy_execution_context_protocol.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable, TYPE_CHECKING

from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.protocols.strategy_resource_protocol import StrategyResourceProtocol

if TYPE_CHECKING:
    from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol


@runtime_checkable
class StrategyExecutionContextProtocol(Protocol):
    """
    Protocolo para o contexto de execução da estratégia.

    O contexto centraliza os recursos disponíveis, artefatos produzidos,
    regras de validação registradas, logs e metadados de execução.
    """

    @property
    def resources(self) -> Dict[str, StrategyResourceProtocol]:
        """
        Dicionário de recursos disponíveis, indexados por resource_id.
        """
        ...

    @property
    def artifacts(self) -> Dict[str, StrategyArtifactReference]:
        """
        Dicionário de artefatos produzidos ou registrados, indexados por chave.
        """
        ...

    @property
    def validation_rules(self) -> Dict[str, "ValidationRuleProtocol"]:
        """
        Dicionário de regras de validação disponíveis, indexadas por rule_id.
        """
        ...

    @property
    def execution_logs(self) -> List[str]:
        """
        Lista de mensagens de log acumuladas durante a execução.
        """
        ...

    @property
    def execution_metadata(self) -> Dict[str, Any]:
        """
        Metadados adicionais da execução.
        """
        ...

    def get_resource(self, resource_id: str) -> Optional[StrategyResourceProtocol]:
        """
        Retorna o recurso correspondente ao identificador informado.
        """
        ...

    def set_resource(self, resource: StrategyResourceProtocol) -> None:
        """
        Registra ou substitui um recurso no contexto.
        """
        ...

    def has_resource(self, resource_id: str) -> bool:
        """
        Retorna True se o recurso existir no contexto.
        """
        ...

    def remove_resource(self, resource_id: str) -> None:
        """
        Remove um recurso do contexto, se existir.
        """
        ...

    def list_resources(self) -> List[StrategyResourceProtocol]:
        """
        Lista todos os recursos registrados no contexto.
        """
        ...

    def get_artifact(self, key: str) -> Optional[StrategyArtifactReference]:
        """
        Retorna o artefato correspondente à chave informada.
        """
        ...

    def set_artifact(self, key: str, artifact: StrategyArtifactReference) -> None:
        """
        Registra ou substitui um artefato no contexto.
        """
        ...

    def has_artifact(self, key: str) -> bool:
        """
        Retorna True se o artefato existir no contexto.
        """
        ...

    def remove_artifact(self, key: str) -> None:
        """
        Remove um artefato do contexto, se existir.
        """
        ...

    def list_artifacts(self) -> List[StrategyArtifactReference]:
        """
        Lista todos os artefatos registrados no contexto.
        """
        ...

    def register_validation_rule(self, rule: "ValidationRuleProtocol") -> None:
        """
        Registra uma regra de validação disponível no contexto.
        """
        ...

    def get_validation_rule(self, rule_id: str) -> Optional["ValidationRuleProtocol"]:
        """
        Retorna a regra de validação correspondente ao identificador informado.
        """
        ...

    def has_validation_rule(self, rule_id: str) -> bool:
        """
        Retorna True se a regra de validação existir no contexto.
        """
        ...

    def remove_validation_rule(self, rule_id: str) -> None:
        """
        Remove uma regra de validação do contexto, se existir.
        """
        ...

    def list_validation_rules(self) -> List["ValidationRuleProtocol"]:
        """
        Lista todas as regras de validação registradas no contexto.
        """
        ...

    def log(self, message: str) -> None:
        """
        Registra uma mensagem de log no contexto.
        """
        ...

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Registra ou atualiza um metadado de execução.
        """
        ...

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Recupera um metadado de execução.
        """
        ...