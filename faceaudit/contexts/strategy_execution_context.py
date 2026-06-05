# faceaudit/contexts/strategy_execution_context.py

from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from faceaudit.mixins.json_serializable_mixin import JSONSerializableMixin
from faceaudit.models.storage.strategy_artifact_reference import StrategyArtifactReference
from faceaudit.models.strategy.instruction_execution_frame import InstructionExecutionFrame
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

    # Secondary index pra lookup por StrategyResource.id (DS-xxxxxxxx).
    # A chave primária de `resources` continua sendo `resource_id` (legado)
    # pra retrocompat com 50+ call sites que passam resource_id direto.
    # Este index permite que code novo possa fazer `get_resource(res.id)`
    # e cair no mesmo recurso. Populado/mantido pelo `set_resource` /
    # `remove_resource`. Não persistido no JSON.
    _id_to_resource_id_index: Dict[str, str] = field(default_factory=dict)

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

    # Frame de execução corrente — setado pelo StrategyExecutor antes de
    # cada execute() via context.with_frame(frame). Não persistido no JSON
    # (efêmero, varia por instrução). Imutável (`InstructionExecutionFrame`
    # é frozen). Executores legados ignoram; o novo
    # LR_ARCHITECTURE_VECTORIZATION fail-closes quando frame is None.
    current_frame: Optional[InstructionExecutionFrame] = None

    # =========================================================================
    # RECURSOS
    # =========================================================================

    def get_resource(self, key: str) -> Optional[StrategyResource]:
        """
        Retorna o recurso correspondente ao identificador informado.

        Aceita ambos:
        - `resource_id` legado (ex: 'foo', 'meu_dataset').
        - `id` técnico (ex: 'DS-8f3a1b2c').

        O lookup primário usa `resources` indexado por `resource_id`. Se
        não encontrar, consulta `_id_to_resource_id_index` (Fase 4 da
        migração id/name).
        """
        if key in self.resources:
            return self.resources[key]
        # Fallback: lookup por id técnico (DS-xxxxxxxx).
        legacy_key = self._id_to_resource_id_index.get(key)
        if legacy_key is not None:
            return self.resources.get(legacy_key)
        return None

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

        Secondary index: atualiza `_id_to_resource_id_index` pra permitir
        `get_resource(resource.id)` além de `get_resource(resource.resource_id)`.
        """
        if resource is None:
            raise ValueError("O recurso não pode ser nulo.")
        rid = resource.resource_id

        # Antes de sobrescrever, limpa o id do resource antigo do index
        # (evita órfão em `_id_to_resource_id_index` quando o resource
        # no slot é re-criado com um id diferente — caso comum em
        # re-execução de instruções).
        old_resource = self.resources.get(rid)
        if old_resource is not None:
            old_id = getattr(old_resource, "id", None)
            new_id = getattr(resource, "id", None)
            if old_id and old_id != new_id:
                self._id_to_resource_id_index.pop(old_id, None)

        stats_id = f"{rid}_stats"
        # Safeguard: ao registrar o próprio recurso de stats (id já termina
        # em `_stats`), não tentamos invalidar uma chave `{stats_id}_stats`
        # que nunca existe — pop seria no-op de qualquer forma, mas o guard
        # deixa a intenção explícita.
        if not rid.endswith("_stats"):
            self.resources.pop(stats_id, None)
            # Limpa também o índice do stats antigo, se existir.
            old_stats = self.resources.get(stats_id)
            if old_stats is not None and getattr(old_stats, "id", None):
                self._id_to_resource_id_index.pop(old_stats.id, None)
        self.resources[rid] = resource
        # Indexa por id técnico (Fase 4). Quando `resource.id` é igual ao
        # `resource_id` (caso raro de migração), o mapeamento é redundante
        # mas inofensivo.
        if getattr(resource, "id", None):
            self._id_to_resource_id_index[resource.id] = rid

    def has_resource(self, key: str) -> bool:
        """
        Retorna True se o recurso existir no contexto.

        Aceita `resource_id` legado OU `id` técnico (Fase 4).
        """
        if key in self.resources:
            return True
        return key in self._id_to_resource_id_index

    def remove_resource(self, key: str) -> None:
        """
        Remove um recurso do contexto, se existir.

        Aceita `resource_id` legado OU `id` técnico. Quando remove, limpa
        ambos os índices (resources primário + _id_to_resource_id_index).
        """
        # Resolve a chave canônica (resource_id) primeiro, mesmo se foi
        # passado o id técnico.
        if key in self.resources:
            legacy_key = key
        else:
            legacy_key = self._id_to_resource_id_index.get(key)
            if legacy_key is None:
                return
        # Limpa o índice do id técnico associado, se houver.
        res = self.resources.get(legacy_key)
        if res is not None and getattr(res, "id", None):
            self._id_to_resource_id_index.pop(res.id, None)
        self.resources.pop(legacy_key, None)

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
        # Lookup tolerante a id técnico (DS-xxx) ou resource_id legado.
        resource = self.get_resource(resource_id)
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
    # FRAME DE EXECUÇÃO (PR 2 — LR_ARCHITECTURE_VECTORIZATION)
    # =========================================================================

    def with_frame(self, frame: InstructionExecutionFrame) -> "StrategyExecutionContext":
        """
        Retorna uma view do contexto carregando `frame`.

        Implementação compartilha os dicts mutáveis (resources, artifacts,
        logs, metadata, _id_to_resource_id_index, log_listeners) — quaisquer
        mudanças via `set_resource` etc. continuam visíveis para todas as
        views. Apenas `current_frame` é específico desta view.

        Esta semântica permite que o orquestrador crie uma view por
        instrução sem fragmentar o estado global compartilhado, mantendo
        a fase imutável dentro do escopo de uma `executor.execute(...)`.
        """
        return dataclasses.replace(self, current_frame=frame)

    @property
    def frame(self) -> Optional[InstructionExecutionFrame]:
        """Alias legível de `current_frame` para o lado do executor."""
        return self.current_frame

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
        # `_id_to_resource_id_index` é runtime — reconstruído a cada
        # `set_resource`. Persistir seria desperdício e fonte de drift.
        for runtime_key in (
            "log_listeners",
            "resource_loader",
            "embedding_column",
            "_id_to_resource_id_index",
            "current_frame",
        ):
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
        for runtime_key in (
            "resource_loader",
            "embedding_column",
            "log_listeners",
            "_id_to_resource_id_index",
            "current_frame",
        ):
            processed_data.pop(runtime_key, None)

        ctx = cls(**processed_data)
        # Reconstrói o secondary index a partir dos resources carregados
        # (set_resource já popula, mas aqui criamos do zero pra cobrir o
        # caminho de hidratação direta via dataclass kwargs).
        for res in ctx.resources.values():
            if getattr(res, "id", None):
                ctx._id_to_resource_id_index[res.id] = res.resource_id
        return ctx