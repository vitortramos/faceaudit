# faceaudit/contexts/strategy_execution_context.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol


T = TypeVar("T", bound="StrategyExecutionContext")


@dataclass
class StrategyExecutionContext(JSONSerializableMixin):
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

    # Listeners de log — invocados em tempo real a cada chamada de log().
    # Usado pelos workers de UI (PyQt) para refletir progresso no dialog
    # de execução enquanto o executor está rodando. Não persistido no JSON
    # (excluído via to_dict abaixo).
    log_listeners: List[Callable[[str], None]] = field(default_factory=list)

    # Capability para hidratação preguiçosa de payloads. Preenchidos por
    # StrategyExecutor.prepare_context. Não persistidos no JSON.
    # `Any` em vez de StrategyResourceLoaderService evita ciclo de import
    # (loader -> resource -> context -> loader).
    resource_loader: Any = None
    embedding_column: Optional[str] = None

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

        Side effect: invalida o cache de stats associado (`"{rid}_stats"`)
        sempre que o recurso é sobrescrito. Sem isso, regras como Fisher e
        MinIdentitiesWithMultipleSamples continuariam lendo `DatasetStats`
        antigos após uma re-execução de PIPELINE_APPLICATION (ou qualquer
        executor que sobrescreva o dataset com novo payload sob o mesmo id).

        A invalidação é incondicional — recomputar stats é barato (leitura
        de metadata pyarrow) e a comparação de payloads para detectar
        "realmente mudou" seria complexa e propensa a falsos negativos.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")
        rid = resource.resource_id
        stats_id = f"{rid}_stats"
        # Safeguard: ao registrar o próprio recurso de stats (id já termina
        # em `_stats`), não tentamos invalidar uma chave `{stats_id}_stats`
        # que nunca existe — pop seria no-op de qualquer forma, mas o guard
        # deixa a intenção explícita.
        if not rid.endswith("_stats"):
            self.resources.pop(stats_id, None)
        self.resources[rid] = resource

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

    def ensure_payload(self, resource_id: str) -> Optional[StrategyResource]:
        """
        Retorna o resource garantindo que `payload` esteja carregado quando possível.

        Hidrata sob demanda via `resource_loader` se:
        - resource existe no contexto
        - resource.payload is None
        - resource.artifact_ref is not None
        - self.resource_loader is not None

        Idempotente: se payload já está carregado, retorna direto.
        Se não há loader registrado, retorna o resource como está (payload=None) —
        consumidores precisam tratar como erro de validação normal.
        """
        resource = self.resources.get(resource_id)
        if resource is None:
            return None
        if resource.payload is not None:
            return resource
        if resource.artifact_ref is None or self.resource_loader is None:
            return resource

        try:
            self.resource_loader.hydrate(
                resource, embedding_column=self.embedding_column
            )
        except Exception as exc:
            # Falha não-fatal: o resource permanece sem payload e o consumidor
            # reporta erro normal de validação. Loga para visibilidade.
            self.log(
                f"⚠️ Falha ao hidratar '{resource_id}' sob demanda: "
                f"{type(exc).__name__}: {exc}"
            )
            return resource

        # Mesmo sem exceção, o loader pode ter retornado None silenciosamente
        # (ex: pickle incompatível). Expor o motivo via `_last_error` para
        # que o usuário veja no LogProgressDialog em vez de só em stderr.
        if resource.payload is None:
            last_error = getattr(self.resource_loader, "_last_error", None)
            if last_error:
                self.log(
                    f"⚠️ Falha ao carregar '{resource_id}' do disco: {last_error}"
                )
        return resource

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

    # Limite duro para evitar crescimento ilimitado da lista de logs entre
    # múltiplas execuções na mesma sessão (acumulação que se traduzia em
    # pickle/serialização cada vez mais lenta porque `execution_metadata`
    # captura essa lista).
    _MAX_EXECUTION_LOGS = 5000

    def log(self, message: str) -> None:
        """
        Registra uma mensagem de log no contexto e notifica listeners
        registrados (ex: worker UI repassando ao dialog de progresso).

        A lista é truncada (rotação simples) ao ultrapassar
        `_MAX_EXECUTION_LOGS` para evitar inchaço sem limite ao longo de
        sessões com múltiplas execuções.
        """
        if not message:
            return
        self.execution_logs.append(message)
        if len(self.execution_logs) > self._MAX_EXECUTION_LOGS:
            # Mantém os últimos 80% — descarta o terço mais antigo de uma vez
            # (evita custar O(N) a cada append).
            keep = int(self._MAX_EXECUTION_LOGS * 0.8)
            del self.execution_logs[:-keep]
        for listener in self.log_listeners:
            try:
                listener(message)
            except Exception:
                # Listener falho não pode quebrar a execução do executor.
                pass

    def add_log_listener(self, listener: Callable[[str], None]) -> None:
        """Registra um callback que será invocado a cada chamada de log()."""
        if listener is not None and listener not in self.log_listeners:
            self.log_listeners.append(listener)

    def remove_log_listener(self, listener: Callable[[str], None]) -> None:
        """Remove um callback previamente registrado via add_log_listener."""
        try:
            self.log_listeners.remove(listener)
        except ValueError:
            pass

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

    def to_dict(self) -> dict:
        """
        Override do mixin: remove campos runtime-only (não serializáveis
        com sentido pelo asdict): log_listeners (callbacks), resource_loader
        (objeto de serviço), embedding_column (estado de runtime).
        """
        data = super().to_dict()
        for runtime_key in ("log_listeners", "resource_loader", "embedding_column"):
            data.pop(runtime_key, None)
        return data

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> Optional[T]:
        """
        Reconstrói o contexto a partir de um dicionário.

        Observação:
        - resources e artifacts são hidratados
        - validation_rules não é reconstruído automaticamente, pois contém
          objetos comportamentais e será normalmente re-registrado em runtime
        - resource_loader, embedding_column e log_listeners são runtime-only
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

        # Campos runtime-only nunca vêm do JSON
        for runtime_key in ("resource_loader", "embedding_column", "log_listeners"):
            processed_data.pop(runtime_key, None)

        return cls(**processed_data)