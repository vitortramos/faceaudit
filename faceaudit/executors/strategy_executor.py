# faceaudit/executors/strategy_executor.py

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class StrategyCancelledError(Exception):
    """Raised por `StrategyExecutor.execute` quando o `cancel_checker`
    retorna True entre instruções — sinaliza cancelamento cooperativo
    pelo usuário, NÃO falha de execução. Caller (worker) trata como
    exit normal com mensagem amigável em vez de erro técnico."""
    pass

from faceaudit.enums.strategy_artifact_type import StrategyArtifactType
from faceaudit.enums.strategy_instruction_type import StrategyInstructionType
from faceaudit.executors.instruction_executor_registry import (
    InstructionExecutorRegistry,
)
from faceaudit.models.instructions.instruction_execution_result import (
    InstructionExecutionResult,
)
from faceaudit.models.instructions.strategy_instruction_definition import (
    StrategyInstructionDefinition,
)
from faceaudit.models.storage.strategy_artifact_reference import (
    StrategyArtifactReference,
)
from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition
from faceaudit.models.strategy.lr_strategy_trained_model import LRStrategyTrainedModel
from faceaudit.models.strategy.strategy_resource import StrategyResource
from faceaudit.protocols.artifact_storage_protocol import ArtifactStorageProtocol
from faceaudit.protocols.strategy_execution_context_protocol import (
    StrategyExecutionContextProtocol,
)
from faceaudit.protocols.strategy_instruction_executor_protocol import (
    StrategyInstructionExecutorProtocol,
)
from faceaudit.protocols.validation_rule_protocol import ValidationRuleProtocol
from faceaudit.services.strategy_resource_loader_service import (
    StrategyResourceLoaderService,
)


class StrategyExecutor:
    """
    Executor principal responsável por orquestrar a execução completa de uma estratégia.

    Fluxo:
    1. Cria contexto de execução
    2. Registra recursos iniciais
    3. Registra regras de validação
    4. Valida definição da estratégia
    5. Executa instruções sequencialmente
    6. Constrói modelo treinado final
    """

    def __init__(
        self,
        context_factory: Type[StrategyExecutionContextProtocol],
        instruction_executor_registry: Union[
            InstructionExecutorRegistry,
            Dict[StrategyInstructionType, StrategyInstructionExecutorProtocol],
        ],
        validation_rules: Optional[Dict[str, ValidationRuleProtocol]] = None,
        artifact_storage: Optional[ArtifactStorageProtocol] = None,
        always_on_rule_ids: Optional[List[str]] = None,
    ):
        self.context_factory = context_factory
        self.validation_rules = validation_rules or {}
        # Regras "sempre rodam" no _validate_strategy independente de
        # constarem em LRStrategyDefinition.validation_rule_ids. Usado pra
        # invariantes globais (ex: consistência de embedding) que devem
        # valer pra TODA estratégia sem precisar migrar JSONs antigos.
        self._always_on_rule_ids: List[str] = list(always_on_rule_ids or [])
        self._artifact_storage = artifact_storage
        self._resource_loader = (
            StrategyResourceLoaderService(artifact_storage)
            if artifact_storage is not None
            else None
        )

        if isinstance(instruction_executor_registry, InstructionExecutorRegistry):
            self.instruction_executor_registry = instruction_executor_registry
        else:
            registry = InstructionExecutorRegistry()
            for instruction_type, executor in instruction_executor_registry.items():
                registry.register(
                    instruction_type=instruction_type,
                    executor=executor,
                    overwrite=True,
                )
            self.instruction_executor_registry = registry

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def execute(
        self,
        strategy_definition: LRStrategyDefinition,
        additional_resources: Optional[List[StrategyResource]] = None,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> LRStrategyTrainedModel:
        """
        Executa completamente a estratégia e retorna o modelo treinado.

        ``additional_resources`` lista recursos de PROJETO (DATASET,
        PIPELINE_CONFIG, MINING_CONFIG, etc.) que não estão declarados em
        ``strategy_definition.resources`` mas são referenciados pelas
        instruções via dropdown. A controller V2 popula isso a partir do
        scan de disco em ``_collect_project_resources()``.

        ``cancel_checker`` callable opcional consultado entre instruções
        (cooperative cancel). Quando retorna True, a execução é abortada
        levantando `StrategyCancelledError` — caller decide se trata
        como erro ou exit normal. Granularidade é por-instrução: uma
        instrução já em execução não é interrompida no meio.
        """
        if strategy_definition is None:
            raise ValueError("A definição da estratégia não pode ser nula.")

        context = self.prepare_context(
            strategy_definition,
            additional_resources=additional_resources,
        )
        self._validate_strategy(strategy_definition, context)

        execution_results = self._execute_all_phases(
            strategy_definition, context, cancel_checker=cancel_checker,
        )

        trained_model = self._build_trained_model(
            strategy_definition=strategy_definition,
            context=context,
            execution_results=execution_results,
        )

        self._log(context, "🏁 Execução concluída com sucesso.")

        return trained_model

    def prepare_context(
        self,
        strategy_definition: LRStrategyDefinition,
        additional_resources: Optional[List[StrategyResource]] = None,
    ) -> StrategyExecutionContextProtocol:
        """
        Cria e inicializa o contexto: registra recursos da definição e regras.
        Os recursos NÃO são hidratados aqui — apenas registrados com seu
        `artifact_ref`. Hidratação acontece sob demanda via
        `context.ensure_payload(rid)` (chamado por validação, executores e
        `_rehydrate_inputs_if_needed`).

        O contexto retornado persiste entre execuções individuais de instruções.
        """
        context = self.context_factory()

        # Injeta capability de hidratação preguiçosa no contexto. Atributos
        # runtime-only (não persistidos no JSON).
        context.resource_loader = self._resource_loader
        embedding_column = getattr(strategy_definition, "embedding_column", None)
        if embedding_column:
            context.embedding_column = embedding_column

        self._register_initial_resources(strategy_definition, context)
        if additional_resources:
            for resource in additional_resources:
                if not context.has_resource(resource.resource_id):
                    context.set_resource(resource)
        self._register_validation_rules(context)

        # Armazena dimensões do embedding (lidas do `column_specs` do
        # primeiro DATASET com a coluna escolhida — substitui o lookup
        # antigo via enum EmbeddingType.dimensions). Se a coluna não
        # estiver setada ou nenhum dataset oferecer specs, simplesmente
        # não popula o metadata — regras que dependem disso (ex:
        # FisherRule) já têm fallback pra `embedding_dim` da DatasetStats.
        emb_dims = self._infer_embedding_dimensions(strategy_definition, embedding_column)
        if emb_dims is not None:
            context.set_metadata("embedding_dimensions", emb_dims)

        # Expõe identificação da estratégia para executores que precisam
        # rotular seus outputs (ex: STRATEGY_EVALUATION → StrategyValidationResult).
        context.set_metadata("strategy_id", strategy_definition.strategy_id)
        context.set_metadata("strategy_name", strategy_definition.name)

        self._log(context, f"🚀 Contexto preparado para '{strategy_definition.name}'")
        return context

    def can_execute_instruction(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> Tuple[bool, List[str]]:
        """
        Verifica se todos os input_resource_ids da instrução estão disponíveis no contexto.
        Retorna (True, []) se executável, ou (False, [missing_ids]) se bloqueada.
        """
        missing = [
            rid for rid in instruction.input_resource_ids
            if not context.has_resource(rid)
        ]
        return (len(missing) == 0, missing)

    def execute_single_instruction(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> InstructionExecutionResult:
        """
        Executa uma única instrução dentro do contexto fornecido.
        Os recursos produzidos ficam disponíveis no contexto para instruções futuras.
        """
        # Reseta a lista de execution_logs entre execuções individuais. O
        # contexto é compartilhado ao longo da sessão, mas os logs do Run
        # ANTERIOR não interessam ao Run atual e estavam fazendo a lista
        # crescer sem limite (visível em pickles cada vez mais lentos).
        try:
            execution_logs = getattr(context, "execution_logs", None)
            if isinstance(execution_logs, list):
                execution_logs.clear()
        except Exception:
            pass

        # Telemetria: contagem de listeners ativos no contexto. Acumulação
        # de listeners zumbis (de Runs anteriores que falharam em remover)
        # se traduz em context.log() lento — cada listener é chamado por
        # mensagem.
        n_listeners = len(getattr(context, "log_listeners", []) or [])
        if n_listeners > 5:
            self._log(
                context,
                f"⚠️ Aviso: {n_listeners} listeners de log ativos no contexto. "
                "Pode indicar acúmulo de dialogs anteriores que não fecharam."
            )

        executor = self._resolve_executor(instruction)
        self._log(context, f"▶ Executando: {instruction.name} ({instruction.instruction_id})")
        # Re-hidratação preguiçosa: se algum input está no contexto mas com
        # payload=None e artifact_ref válido (ex: contexto antigo, ou hidratação
        # inicial falhou silenciosamente), tenta carregar do disco antes de
        # delegar ao executor concreto.
        self._rehydrate_inputs_if_needed(instruction, context)
        result = executor.execute(instruction=instruction, context=context)
        if not result.success:
            self._log(context, f"❌ Falha: {result.errors}")
        else:
            self._log(context, f"✅ Concluída: {instruction.name}")
        return result

    def persist_produced_resource(
        self,
        resource: StrategyResource,
        strategy_id: str,
        progress_callback=None,
    ) -> Optional[StrategyArtifactReference]:
        """
        Persiste um recurso produzido em disco via artifact_storage.
        Retorna o StrategyArtifactReference ou None se não for persistível.

        progress_callback (opcional): callable(str) chamada com mensagens de
        progresso durante a serialização. Útil para refletir no dialog de UI
        casos pesados como SIMILARITY_SCORES com milhões de pares.
        """
        if not resource.has_payload or self._artifact_storage is None:
            return None

        # CRÍTICO: usar `resource.to_artifact_type()` (método do StrategyResource)
        # e não `resource.resource_type.to_artifact_type()` (método do enum).
        # O método do resource inspeciona `payload.metadata` pra rotear
        # `SIMILARITY_SCORES` em MINED_SCORES / VECTORIZED_SCORES quando
        # apropriado — sem isso PAIR_MINING e SCORE_VECTORIZATION caem
        # todos em `scores/<sid>/raw/` errado.
        artifact_type = resource.to_artifact_type()
        if artifact_type is None:
            return None

        kwargs = {
            "strategy_id": strategy_id,
            "resource_id": resource.resource_id,
            # `id` técnico (`SS-abc12345`, `TP-...`, etc) entra no filename
            # como sufixo pra disambiguação contra colisão de nome.
            "id": resource.id,
        }

        # TRAINED_PIPELINE usa step_id ao invés de resource_id no template
        if artifact_type == StrategyArtifactType.TRAINED_PIPELINE:
            kwargs["step_id"] = resource.resource_id

        if progress_callback is not None:
            kwargs["progress_callback"] = progress_callback

        ref = self._artifact_storage.save_artifact(
            artifact_type=artifact_type,
            data=resource.payload,
            **kwargs,
        )

        # Snapshot PNG em `<project>/graphics/` pra tipos cujo painel de
        # resultados mostra o gráfico inline via QPixmap (sem reconstruir
        # matplotlib na main thread). O plot vivo (`🔍 Expandir`) continua
        # sendo construído sob demanda a partir do payload. Falha é
        # best-effort — não bloqueia a persistência principal.
        from faceaudit.enums.strategy_resource_type import StrategyResourceType
        if ref is not None:
            try:
                project_root = getattr(
                    self._artifact_storage, "project_root", None,
                )
                if project_root is None:
                    pass
                elif resource.resource_type in (
                    StrategyResourceType.SIMILARITY_SCORES,
                    StrategyResourceType.CALIBRATED_SCORES,
                ):
                    # CALIBRATED_SCORES (output de SCORE_CALIBRATION) e
                    # SIMILARITY_SCORES (SCORING/PAIR_MINING) usam o mesmo
                    # render — `write_distribution_png` detecta automaticamente
                    # `metadata["log_lr"]` em algum sample e troca pro
                    # eixo "log-LR calibrado". Subfolder diferente preserva
                    # a distinção visual na pasta `graphics/`.
                    from faceaudit.models.similarity.similarity_score_dataset import (
                        SimilarityScoreDataset,
                    )
                    from faceaudit.models.similarity.distribution_plot import (
                        score_distribution_png_path,
                    )
                    if isinstance(resource.payload, SimilarityScoreDataset):
                        is_calibrated = (
                            resource.resource_type
                            == StrategyResourceType.CALIBRATED_SCORES
                        )
                        png_path = score_distribution_png_path(
                            project_root, strategy_id, resource.resource_id,
                            calibrated=is_calibrated,
                        )
                        resource.payload.write_distribution_png(png_path)

                        # CALIBRATED_SCORES também ganha um PNG da curva de
                        # calibração (raw_score × log-LR). Exibida em aba
                        # separada do painel de resultados pra diagnosticar
                        # forma da função: crossover LR=1, saturação,
                        # não-monotonicidade de KDE mal-configurada.
                        if is_calibrated:
                            from faceaudit.models.similarity.calibration_curve_plot import (
                                calibration_curve_png_path,
                            )
                            curve_png_path = calibration_curve_png_path(
                                project_root, strategy_id, resource.resource_id,
                            )
                            resource.payload.write_calibration_curve_png(
                                curve_png_path,
                            )
                elif (
                    resource.resource_type
                    == StrategyResourceType.CASEWORK_VISUALIZATION
                ):
                    from faceaudit.models.casework_visualization_result import (
                        CaseworkVisualizationResult,
                    )
                    from faceaudit.models.casework_visualization_plot import (
                        casework_visualization_png_path,
                        render_casework_png,
                    )
                    if isinstance(resource.payload, CaseworkVisualizationResult):
                        png_path = casework_visualization_png_path(
                            project_root, strategy_id, resource.resource_id,
                        )
                        render_casework_png(resource.payload, png_path)
                elif (
                    resource.resource_type
                    == StrategyResourceType.TRAINED_TYPICALITY_MODEL
                ):
                    # KNNTypicalityModel treinado → PNG da distribuição LOO
                    # de T_k. Mesmo padrão dos demais — gera headless via
                    # FigureCanvasAgg, painel inline mostra via QPixmap
                    # sem reconstruir matplotlib vivo.
                    from faceaudit.services.knn_typicality_model import (
                        KNNTypicalityModel,
                    )
                    from faceaudit.models.typicality_distribution_plot import (
                        render_typicality_distribution_png,
                        typicality_distribution_png_path,
                    )
                    if isinstance(resource.payload, KNNTypicalityModel):
                        png_path = typicality_distribution_png_path(
                            project_root, strategy_id, resource.resource_id,
                        )
                        render_typicality_distribution_png(
                            resource.payload, png_path,
                        )
                elif (
                    resource.resource_type
                    == StrategyResourceType.VALIDATION_RESULT
                ):
                    from faceaudit.models.strategy_validation_result import (
                        StrategyValidationResult,
                    )
                    from faceaudit.models.validation_result_plot import (
                        render_validation_pngs,
                        validation_result_png_paths,
                        write_validation_metrics_sidecar,
                    )
                    if isinstance(resource.payload, StrategyValidationResult):
                        paths = validation_result_png_paths(
                            project_root, strategy_id, resource.resource_id,
                        )
                        render_validation_pngs(resource.payload, paths)
                        # Sidecar de métricas escalares — sobrevive ao
                        # drop do payload em memória quando o usuário troca
                        # de estratégia. Sem ele a tabela do painel de
                        # resultados ficava em branco até reexecutar.
                        if ref.path:
                            from pathlib import Path as _Path
                            pickle_abs = _Path(project_root) / ref.path
                            write_validation_metrics_sidecar(
                                pickle_abs, resource.payload,
                            )
            except Exception:
                pass  # best-effort; perda de PNG não invalida o parquet

        return ref

    # =========================================================================
    # ETAPAS INTERNAS
    # =========================================================================

    def _infer_embedding_dimensions(
        self,
        strategy_definition: LRStrategyDefinition,
        embedding_column: Optional[str],
    ) -> Optional[int]:
        """
        Deriva a dimensionalidade do embedding lendo o `column_specs` do
        primeiro DATASET da estratégia que tenha a coluna escolhida.

        Substitui o lookup antigo via `EmbeddingType.dimensions` — agora
        funciona pra qualquer coluna com role=EMBEDDING (incluindo as
        que vierem de DeepFace/ArcFace no futuro) sem hardcoding de enum.
        """
        if not embedding_column:
            return None

        from faceaudit.enums.strategy_resource_type import StrategyResourceType

        for resource in strategy_definition.resources:
            if resource.resource_type not in (
                StrategyResourceType.DATASET,
                StrategyResourceType.DATASET,
            ):
                continue
            payload = getattr(resource, "payload", None)
            specs = getattr(payload, "column_specs", None) if payload else None
            if not specs:
                continue
            spec = specs.get(embedding_column)
            dim = getattr(spec, "dimensions", None) if spec else None
            if isinstance(dim, int) and dim > 0:
                return dim
        return None

    def _register_initial_resources(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        """
        Registra recursos da definição no contexto SEM hidratar.

        Recursos com payload preexistente em memória (ex: outputs produzidos
        nesta sessão) entram com payload preservado. Recursos vindos do disco
        entram com payload=None e artifact_ref preenchido — a hidratação
        acontece sob demanda via context.ensure_payload(rid) chamado por
        validation rules ou _rehydrate_inputs_if_needed.

        Esta abordagem preguiçosa evita pagar o custo de carregar parquets
        gigantes / pickles de modelos a cada troca de estratégia ou refresh.
        """
        for resource in strategy_definition.resources:
            context.set_resource(resource)

    def _register_validation_rules(
        self,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        for rule in self.validation_rules.values():
            context.register_validation_rule(rule)

    def _validate_strategy(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        self._log(context, "🔎 Validando estratégia...")

        # Junta regras "sempre rodam" com as declaradas pela estratégia,
        # preservando ordem e descartando duplicatas.
        rule_ids: List[str] = []
        for rid in list(self._always_on_rule_ids) + list(strategy_definition.validation_rule_ids):
            if rid not in rule_ids:
                rule_ids.append(rid)

        for rule_id in rule_ids:
            rule = self.validation_rules.get(rule_id)

            if rule is None:
                # Regras "sempre rodam" não falham se não estiverem
                # registradas (testes podem montar o executor sem elas);
                # já as declaradas pela estratégia continuam exigidas.
                if rule_id in self._always_on_rule_ids:
                    continue
                raise ValueError(f"Regra de validação não encontrada: '{rule_id}'")

            result = rule.validate(target=strategy_definition, context=context)

            if not result.is_valid:
                raise ValueError(
                    f"Validação falhou ({rule_id}): {result.errors}"
                )

        self._log(context, "✅ Estratégia válida.")

    def _execute_all_phases(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
        cancel_checker: Optional[Callable[[], bool]] = None,
    ) -> List[InstructionExecutionResult]:
        results: List[InstructionExecutionResult] = []

        phases = [
            ("Preparação", strategy_definition.preparation_instructions),
            ("Validação", strategy_definition.validation_instructions),
            ("Análise de Caso", strategy_definition.casework_instructions),
        ]

        for phase_name, phase_instructions in phases:
            if not phase_instructions:
                continue
            self._log(
                context,
                f"📋 Fase: {phase_name} ({len(phase_instructions)} instruções)",
            )
            for instruction in phase_instructions:
                # Checkpoint de cancelamento ANTES de cada instrução.
                # Granularidade é por-instrução: uma instrução longa em
                # execução não é interrompida no meio, mas o usuário não
                # precisa esperar TODAS as instruções restantes rodarem
                # depois de clicar cancelar.
                if cancel_checker is not None and cancel_checker():
                    self._log(
                        context,
                        f"⏹ Cancelado pelo usuário antes de '{instruction.name}'.",
                    )
                    raise StrategyCancelledError(
                        f"Cancelado pelo usuário antes de '{instruction.name}'."
                    )
                # execute_single_instruction já chama _rehydrate_inputs_if_needed
                # internamente — não precisa duplicar aqui.
                result = self.execute_single_instruction(instruction, context)
                if not result.success:
                    raise RuntimeError(
                        f"Instrução falhou ({instruction.instruction_id}): {result.errors}"
                    )
                results.append(result)

        return results

    def _build_trained_model(
        self,
        strategy_definition: LRStrategyDefinition,
        context: StrategyExecutionContextProtocol,
        execution_results: List[InstructionExecutionResult],
    ) -> LRStrategyTrainedModel:
        model = LRStrategyTrainedModel(strategy_id=strategy_definition.strategy_id)

        for resource in context.list_resources():
            model.add_trained_resource(resource)

        for artifact in context.list_artifacts():
            model.add_artifact(artifact)

        model.set_execution_metadata("strategy_id", strategy_definition.strategy_id)
        model.set_execution_metadata("strategy_name", strategy_definition.name)
        model.set_execution_metadata(
            "num_instructions",
            len(strategy_definition.instructions),
        )
        model.set_execution_metadata(
            "num_resources",
            len(model.trained_resources),
        )
        model.set_execution_metadata(
            "instruction_results",
            [r.metadata for r in execution_results],
        )
        model.set_execution_metadata(
            "execution_logs",
            list(context.execution_logs),
        )
        model.set_execution_metadata(
            "context_metadata",
            dict(context.execution_metadata),
        )

        return model

    # =========================================================================
    # AUXILIARES
    # =========================================================================

    def _rehydrate_inputs_if_needed(
        self,
        instruction: StrategyInstructionDefinition,
        context: StrategyExecutionContextProtocol,
    ) -> None:
        """
        Para cada input declarado da instrução, garante o payload carregado
        via `context.ensure_payload(rid)`. Idempotente — se o payload já
        está em memória, não há custo. Caso contrário, dispara o load do
        disco (parquet/pickle/etc.) sob demanda.

        Cobre:
        - hidratação preguiçosa após `prepare_context` (que não hidrata mais)
        - contexto vivo desincronizado da definição em disco
        - recursos produzidos por instruções anteriores cujo payload foi
          descartado em algum reset/refresh
        """
        input_ids = list(getattr(instruction, "input_resource_ids", []) or [])
        if not input_ids:
            return

        for rid in input_ids:
            if not context.has_resource(rid):
                continue
            resource = context.get_resource(rid)
            if resource is None or resource.payload is not None:
                continue
            if resource.artifact_ref is None:
                continue
            before = resource.payload
            context.ensure_payload(rid)
            if resource.payload is not None and before is None:
                self._log(
                    context,
                    f"♻ Recurso '{rid}' carregado do disco "
                    f"({resource.artifact_ref.path}).",
                )

    def _resolve_executor(
        self,
        instruction: StrategyInstructionDefinition,
    ) -> StrategyInstructionExecutorProtocol:
        return self.instruction_executor_registry.resolve(instruction.instruction_type)

    def _log(self, context: StrategyExecutionContextProtocol, message: str) -> None:
        context.log(message)