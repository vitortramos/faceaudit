# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FaceAudit is a **forensic facial image analysis and comparison platform** for law enforcement. Its goal is to allow users to **create and configure LR (Likelihood Ratio) calculation strategies** for forensic comparisons of AdaFace facial image embeddings, entirely through the application UI — nothing hard-coded.

## Commands

```bash
# Install in development mode
pip install -e .

# Run the application
python interface/run.py

# Run tests
python -m pytest tests/
python -m pytest tests/test_full_pipeline.py
python -m pytest tests/transformation_pipeline_test.py
python -m pytest tests/test_storage_and_repositories.py

# Data validation utility
python tests/check_parquet.py
```

## Configuration

- `config.json` — Runtime paths (models, datasets, output directories)
- `config.yaml` — System-level settings (GPU, random seeds, output paths)

---

## ACTIVE ARCHITECTURE MIGRATION

**The codebase currently contains two architectures in parallel.** The backend (`faceaudit/`) is being migrated to the new architecture. The frontend (`interface/`) has the old architecture tabs plus the new **Strategy Manager V2** tab (`interface/tabs/strategy_manager_v2/`) which is the first frontend component built against the new backend.

### Old Architecture — folders: `manager/`, `controllers/`, `strategies/` (partially)

Three-layer structure with hard-coded execution order:

- **Managers** (`KDEManager`, `PLDAManager`, `StrategiesManager`, `ProjectManager`, `ValidationManager`): Stateful classes that own the full lifecycle of each strategy — config, data loading, preparation, training, persistence, inference. Use dirty-flag callbacks to propagate changes to `ProjectManager`.
- **Controllers** (`KDEPreparationController`, `PLDATrainingController`, `PLDATransformationPipelinesController`): Orchestrate the steps within each phase. The execution order is **implicit and hard-coded** in their methods.
- **Strategies** (`KDEStrategy`, `PLDAStrategy`, `PLDASnormStrategy`): Implement the mathematical models — fit, predict, infer. Implement `LRModelProtocol`.

Problem being solved: execution order, data types used, and transformation steps were fixed in code. Nothing could be reconfigured without changing Python files.

### New Architecture — folders: `contexts/`, `executors/`, `factories/`, `repositories/` and subfolders of `models/` and `protocols/`

The new architecture is **declarative**: a strategy is described as a `LRStrategyDefinition` (list of resources + ordered instructions). The execution engine interprets and runs it — the user defines what happens, not the code.

#### Core execution flow

```
StrategyExecutorFactory.build()
  └─ StrategyExecutor.execute(LRStrategyDefinition)
       ├─ Creates StrategyExecutionContext  ← shared state bus
       ├─ Loads initial resources into context (datasets, configs)
       ├─ Validates strategy via ValidationRules
       └─ For each phase (Preparação → Validação → Análise de Caso):
            For each StrategyInstructionDefinition in phase:
              InstructionExecutorRegistry.resolve(instruction_type)
                └─ executor.validate(instruction, context)
                     ├─ validation_rule_ids (registered rules)
                     ├─ instruction.build_validation_rules() (dynamic rules)
                     └─ _validate_instruction_specifics() (hard-coded checks)
                └─ executor.execute(instruction, context)
                     ├─ Reads needed resources from context
                     ├─ Runs logic (pipeline/split/score/calibrate)
                     └─ Writes results back into context as new StrategyResource
       └─ Packages final context → LRStrategyTrainedModel

Step-by-step execution (used by the UI for per-instruction runs):
  StrategyExecutor.prepare_context(definition) → persistent context
  StrategyExecutor.can_execute_instruction(instr, ctx) → (bool, missing_ids)
  StrategyExecutor.execute_single_instruction(instr, ctx) → result
```

#### New architecture folder map

| Folder | Role |
|---|---|
| `faceaudit/contexts/` | `StrategyExecutionContext` — the shared state bus (resources, artifacts, logs, metadata) |
| `faceaudit/executors/` | `StrategyExecutor` (orchestrator) + `InstructionExecutorRegistry` (type→executor map) |
| `faceaudit/executors/instructions/` | One concrete executor per instruction type, all extending `StrategyInstructionExecutorBase` |
| `faceaudit/factories/` | `StrategyExecutorFactory` — single assembly point for the engine |
| `faceaudit/repositories/` | `StrategyDefinitionRepository` (JSON) + `StrategyTrainedModelRepository` (pickle) — ambos aceitam `ArtifactStorageProtocol`, não `ArtifactStorageService` |
| `faceaudit/validation_rules/` | Implementações concretas de `ValidationRuleProtocol`, todas estendendo `ValidationRuleBase`; 7 regras: `DatasetIsolationRule`, `DatasetOverlapRequiredRule`, `FisherRule`, `MinCalibrationPairsRule`, `HoldoutLeakageRule`, `DualPipelineDimensionCompatibilityRule`, `DualPipelineSubsequenceRule` |
| `faceaudit/storage/` | `LocalProjectArtifactStorage` (projeto autocontido) + `GlobalLibraryStorage` (biblioteca global) — ambos implementam `ArtifactStorageProtocol`; `ArtifactStorageService` é adaptador transitório para arch antiga. `LocalProjectArtifactStorage` expõe também `verify_integrity(refs)` fora do protocolo. |
| `faceaudit/services/project_service.py` | `FaceAuditProjectService` — cria/abre projetos (`.fa`), agrega `LocalProjectArtifactStorage` + repositórios; `open(verify_integrity=True)` executa `check_integrity()` automaticamente. `FaceAuditProject` expõe `integrity_results`, `has_integrity_problems` e `check_integrity()`. |
| `faceaudit/services/strategy_resource_loader_service.py` | `StrategyResourceLoaderService` — hidrata `payload` de `StrategyResource` a partir de `artifact_ref` SOB DEMANDA. Idempotente: se `payload` já existe, retorna sem fazer nada. Loaders concretos (`_load_pickle`, `_load_dataset`, `_load_pipeline_config`, `_load_similarity_scores_from_parquet`) preenchem `self._last_error` em caso de falha (consultado pelo `StrategyExecutionContext.ensure_payload` para expor a exceção real no LogProgressDialog) e também emitem `warnings.warn` para stderr |
| `faceaudit/services/strategy_inference_service.py` | `StrategyInferenceService` — inferência forense: recebe `LRStrategyTrainedModel` + `StrategyResource` probe/reference + `ArtifactStorageProtocol`; executa pipeline → `score_matrix()` → `calibrate_scores()`; retorna `SimilarityScoreDataset` com `raw_score` e `metadata["log_lr"]` por par |
| `faceaudit/services/pipeline_trajectory_service.py` | `PipelineTrajectoryService` — visualização 3D de trajetória: conecta `EmbeddingTransformationPipeline` treinado ao `EmbeddingMathService` + `TrajectoryPlotterService`; `compute_trajectory()` → `TransformationTrajectoryData`; `export_plots()` → PNGs headless |
| `faceaudit/protocols/` | Structural typing contracts — see section below |
| `faceaudit/models/calibration/` | Calibration model configs + implementations (LogReg, KDE) |
| `faceaudit/models/instructions/` | Instruction DTOs — one per instruction type, inherit `StrategyInstructionDefinition` |
| `faceaudit/models/pipeline/` | `PipelineTrainingPolicy`, `PipelineApplicationPolicy` |
| `faceaudit/models/similarity/` | Similarity model configs + implementations (Cosine, PLDA) + `SimilarityScoreDataset` |
| `faceaudit/models/storage/` | `StrategyArtifactReference` — referência lógica com `path` (sempre relativo) + `hash` SHA-256 gravado no momento da escrita. `ArtifactIntegrityResult` — resultado frozen dataclass por artefato verificado: `status`, `expected_hash`, `actual_hash`, `abs_path`, `describe()`. |
| `faceaudit/models/strategy/` | `StrategyResource`, `LRStrategyDefinition`, `LRStrategyTrainedModel` |
| `faceaudit/models/validation/` | `ValidationResult` — used across all validators and executors |
| `faceaudit/enums/artifact_integrity_status.py` | `ArtifactIntegrityStatus` — `OK`, `MODIFIED`, `MISSING`, `HASH_UNAVAILABLE`; `is_problem` True apenas para MODIFIED e MISSING |

#### Protocols (new architecture only)

Old architecture only: `lr_model_protocol.py`, `strategy_manager_protocol.py`
Shared (used by both): `embedding_transformation_protocol.py` — implemented by all 6 concrete transformators in `faceaudit/transformators/`
New architecture: all others.

Key new protocols and what they define:

- `LRStrategyDefinitionProtocol` — the editable blueprint: resources + ordered instructions + validation rule IDs
- `StrategyExecutionContextProtocol` — state bus contract: get/set/list resources, artifacts, rules, logs; `ensure_payload(resource_id)` para hidratação preguiçosa via `resource_loader` runtime
- `StrategyInstructionExecutorProtocol` — `validate(instruction, context)` + `execute(instruction, context)`
- `StrategyResourceProtocol` — any reusable object in a strategy: `resource_id`, `resource_type`, `payload`, `artifact_ref`
- `SimilarityModelProtocol` — produces **raw scores**: `fit(dataset)`, `score_pair`, `score_vectors`, `score_matrix`
- `CalibrationModelProtocol` — converts raw scores to LRs: `fit(scores, labels)`, `predict_proba`, `calibrate_scores`
- `TrainableModelProtocol` / `ApplicableModelProtocol` — generic train/apply contracts
- `EmbeddingTransformationStepProtocol` / `EmbeddingTransformationPipelineProtocol` — pipeline step and pipeline contracts
- `ValidationRuleProtocol` — reusable rule applied to any target: `validate(target, context) → ValidationResult`
- `ArtifactStorageProtocol` — physical file management: `save_artifact`, `resolve_artifact`, `list_artifacts`
- `StrategyDefinitionRepositoryProtocol` / `StrategyTrainedModelRepositoryProtocol` — persistence contracts
- `LRStrategyTrainedModelProtocol` — final execution result: `trained_resources`, `artifacts`, `execution_metadata`

#### Key design decisions in the new architecture

**`LRStrategyDefinition` organizes instructions in three phases.** `preparation_instructions` (pipeline training, model training), `validation_instructions` (metrics, scoring), `casework_instructions` (case analysis). The read-only property `instructions` concatenates all three for backward compatibility with validators, protocols, and `resource_info_widget`. Helper `_get_phase_list(phase)` returns the list for a given phase key.

**`StrategyResource` is the unit of exchange between instructions.** Every instruction reads named resources from the context and writes named resources back. The names are declared in the instruction DTO itself (`input_resource_ids`, `output_resource_ids`), kept in sync via `_sync_declared_io()` in `__post_init__`. Two categories exist: **project resources** (DATASET, SPLIT_CONFIG, MINING_CONFIG, PIPELINE_CONFIG — imported by the user, shared across strategies, always have `artifact_ref`) and **strategy-produced resources** (TRAINED_PIPELINE, INTERMEDIATE_DATASET, TRAINED_SIMILARITY_MODEL, TRAINED_CALIBRATION_MODEL, SIMILARITY_SCORES, CALIBRATED_SCORES, VALIDATION_RESULT — produced by instruction execution). Hyperparameter configs for similarity/calibration models are NOT project resources anymore — they live embedded in their instruction DTOs (`similarity_model_config`, `logreg_config`, `kde_config`) and are persisted as part of the strategy JSON. Strategy-produced resources start as **placeholders** (payload=None, artifact_ref=None) when the user fills in output fields in the instruction config panel. After execution, they are persisted to disk and gain `artifact_ref`. `StrategyResourceType.to_artifact_type()` maps the logical resource type to the physical `StrategyArtifactType` for persistence. `StrategyInstructionDefinition.get_output_resource_placeholders()` creates placeholder `StrategyResource` objects from `ResourceFieldSpec` entries with `is_output=True`. The controller calls `_sync_all_output_placeholders()` on each strategy modification to keep `LRStrategyDefinition.resources` in sync. `StrategyExecutor.persist_produced_resource(resource, strategy_id)` handles disk persistence, returning a `StrategyArtifactReference`.

**`ResourceFieldSpec` enables type-filtered resource dropdowns.** Each instruction DTO implements `get_resource_field_specs()` returning a list of `ResourceFieldSpec(field_name, label, accepted_types, is_output)`. The UI config panel uses these specs to show only resources of the correct `StrategyResourceType` in each dropdown. It's an instance method (not classmethod) because Pipeline Training/Application specs vary by `pipeline_mode` (LINEAR vs DUAL).

**`StrategyInstructionDefinition.from_dict()` is a polymorphic dispatcher.** It reads `instruction_type` and hydrates the correct concrete DTO. This is how the strategy survives JSON serialization/deserialization with full type information.

**Similarity and calibration are separated.** The old `LRModelProtocol` was monolithic (fit + score + calibrate). The new architecture splits it: `SimilarityModelProtocol` produces raw scores; `CalibrationModelProtocol` converts raw scores to LRs. They connect via `SimilarityScoreDataset`.

**`KDECalibrationModel` vs `KDEStrategy`**: In the old architecture, KDE was a scoring strategy (fit on embeddings, produce LRs). In the new architecture, `KDECalibrationModel` is a calibrator only — it receives pre-computed raw scores and fits two KDEs (genuine/impostor) with Fisher transform + adaptive GridSearch bandwidth. The `CosineSimilarityModel` and `PLDASimilarityModel` handle the raw scoring.

**`PipelineTrainingPolicy`** controls the DUAL mode behavior: which transformation steps (`CORAL`, `Whitening`, `LDA`) may use the target dataset during training, and whether shared state is copied to the target branch.

**Filtro de instruções por fase (UI).** O menu "➕ Adicionar" da `InstructionListWidget` (em `interface/tabs/strategy_manager_v2/strategy_editor/instruction_list_widget.py`) filtra os tipos disponíveis pela fase da aba: `_TRAINING_ONLY_TYPES = {PIPELINE_TRAINING, SIMILARITY_MODEL_TRAINING, CALIBRATION_MODEL_TRAINING}` aparece **só na Preparação**; `_VALIDATION_ONLY_TYPES = {STRATEGY_EVALUATION}` aparece **só na Validação**; demais tipos aparecem em todas. A função `_allowed_types_for_phase(phase)` consolida a lógica e cada `InstructionListWidget` recebe `phase` no construtor (a partir do `_PHASES` em `strategy_editor_widget.py`).

**Visualização de resultados forenses (`ValidationResultViewerDialog`).** A instrução `STRATEGY_EVALUATION` produz um `StrategyValidationResult` com 4 conjuntos de dados de plot pré-computados (`genuine_log_lrs`, `impostor_log_lrs`, `fpr_array`, `fnr_array`, `ece_prior_log_odds`, `ece_values`, `ece_min_values`). O `InstructionConfigPanel` mostra um botão `📊 Visualizar Resultado de Validação` quando a instrução é `StrategyEvaluationInstruction` e o recurso está em memória ou em disco — clique abre `ValidationResultViewerDialog` (em `interface/tabs/strategy_manager_v2/strategy_editor/validation_result_viewer_dialog.py`), `QDialog` 1100×780 com header âmbar (métricas escalares) + `QTabWidget` de 4 abas: **Distribuição de LRs** (histograma + KDE de gen/imp em log-LR), **Tippett Plot** (P(LR ≤ x | Hp) vs P(LR > x | Hd)), **Curva DET** (FPR×FNR em escala normit via `scipy.stats.norm.ppf`, com marca no EER) e **Calibração ECE** (curva real + curva mínima PAVA com área de cal_loss pintada entre as duas). Cada aba tem `FigureCanvas` matplotlib + `NavigationToolbar` (zoom/pan/exportar PNG). Padrão de wiring espelha `view_score_distribution_requested` — sinal no painel → reemissor no widget → slot `_on_view_validation_result` no controller, com lazy hydrate via `_resource_loader.hydrate` quando o payload está no disco.

**Tuning anti-freeze no Windows (interface/run.py topo).** Antes de qualquer import pesado: `sys.setswitchinterval(0.001)` reduz o intervalo de troca da GIL de 5ms para 1ms, dando 5× mais turnos à main thread Qt durante operações Python-puras concorrentes (ex: `pickle.dump` de modelos com tensores SpeechBrain/PyTorch); e em `sys.platform=="win32"`, `ctypes.windll.user32.DisableProcessWindowsGhosting()` desabilita o "ghost window" do Windows — a janela continua reagindo assim que a GIL libera, mas SEM o overlay/título "(Não respondendo)" cosmético. `HashService.calculate_file_hash` usa `hashlib.file_digest` em Python ≥ 3.11 (loop em C com gestão de buffer otimizada que libera GIL agressivamente).

**Performance: serialização de `StrategyResource` NÃO recursa no `payload`.** `StrategyResource.to_dict()` (em `faceaudit/models/strategy/strategy_resource.py`) **NÃO chama `super().to_dict()`** do `JSONSerializableMixin`. O mixin usa `dataclasses.asdict(self)` que recursa em **todos** os campos do dataclass, incluindo `payload` — e só depois faz `pop("payload")`. Para um `SimilarityScoreDataset` com 5M+ samples (cada com `EmbeddingInfo` aninhado + `ndarray` + dict de metadata), o `asdict` aloca e itera ~70M+ objetos Python por chamada. Como `_save_current_definition` chama `to_dict()` em **cada recurso da definição** após cada execução de instrução, e a definição acumula `SimilarityScoreDataset` (raw_scores, calibrated_scores) ao longo das fases, o save ficava cumulativamente mais lento — freezando a main thread Qt por dezenas de segundos a cada Run após a 1ª SIMILARITY_SCORING. **Solução**: `to_dict()` constrói o dict manualmente — `{resource_id, name, resource_type (via .to_str()), artifact_ref (delegando .to_dict())}`, intencionalmente omitindo `payload`. Custo cai para ~0.01ms independente do tamanho do payload. **Não reintroduzir `super().to_dict()` aqui sob hipótese alguma.**

**Performance: caching de contagens em `SimilarityScoreDataset`.** As properties `total_genuines` e `total_impostors` (em `faceaudit/models/similarity/similarity_score_dataset.py`) são **cachadas em `self.__dict__`** (`_cached_n_gen`/`_cached_n_imp`) via helper `_ensure_class_counts_cached()` que computa as duas contagens em uma única passagem pelos `items`. O método `add()` invalida o cache. Razão: o `_refresh_produced_resources` da UI chamava ambas as properties a cada refresh; sem cache, cada chamada percorria 5M+ items, travando a main thread por segundos repetidamente.

**Performance: summarizers da UI usam amostragem e leem `items[0]`, não recomputam matrizes.** Em `interface/tabs/strategy_manager_v2/strategy_editor/output_row_widget.py`: `_summarize_dataset` lê `embedding_dim` de `items[0].embedding.shape[0]` (NÃO chama `payload.matrix`, que faria `np.vstack` de TODOS os embeddings — ~410MB para 200k×256D); `_summarize_calibrated_scores` e summarizers análogos com listas grandes (≥50k items) usam **amostragem aleatória** (`random.sample`) para estimar contagens e estatísticas, em vez de iterar em milhões de samples a cada refresh.

**Performance: executores iteram em chunks com micro-pausa.** `ScoreCalibrationInstructionExecutor` e `StrategyEvaluationInstructionExecutor` iteram `score_dataset.items` em **chunks de 50.000** com `time.sleep(0)` entre eles para liberar a GIL e drenar o event loop Qt. `ScoreCalibrationInstructionExecutor` **muta `sample.metadata["log_lr"]` in-place** (NÃO faz `copy.copy(sample)` + cria novo dict de metadata por sample — operação Python-pura que segurava a GIL por dezenas de segundos com 5M+ samples). O input dataset compartilha referências dos samples com o output, mas ambos terão `log_lr` populado, o que é o resultado desejado.

**Performance: `to_parquet` chunkado com `gc.disable`.** `SimilarityScoreDataset.to_parquet` escreve em chunks de 25k linhas com `time.sleep(0.005)` entre eles e desabilita o coletor cíclico do GC durante a escrita (com 5M+ objetos no heap, gen-2 do GC pausa por segundos percorrendo tudo). `progress_callback` opcional emite mensagens legíveis de progresso. Reabilita o GC no `finally`. Cobre 5 colunas obrigatórias + `log_lr` opcional (detectado automaticamente via `metadata["log_lr"]` em algum sample).

**Performance: telemetria em `LocalProjectArtifactStorage.save_artifact`.** Cada chamada emite via `progress_callback`: `⏱  Persistência '{relative_path}': {file_mb:.1f} MB | write={t:.2f}s, hash={t:.2f}s, total={t:.2f}s` — diagnóstico rápido para distinguir gargalos em I/O de write vs hash vs serialização Python upstream.

**Performance: rotação de `execution_logs` no contexto.** `StrategyExecutionContext.log()` mantém no máximo `_MAX_EXECUTION_LOGS = 5000` mensagens (descartadas em FIFO). Sem isso, executando 10+ instruções em sequência, a lista de logs crescia indefinidamente e `to_dict()` da estratégia copiava tudo a cada save. `StrategyExecutor.execute_single_instruction` também faz `execution_logs.clear()` no início de cada Run e avisa se há mais de 5 listeners (sintoma de leak de subscriber por reabertura de dialog).

**Identificação da estratégia no contexto.** `StrategyExecutor.prepare_context` expõe `strategy_id` e `strategy_name` em `context.execution_metadata` (via `context.set_metadata`) — usado por executores que precisam rotular outputs ligados à estratégia, como `STRATEGY_EVALUATION` (que passa o `strategy_id` para o `StrategyValidationResult`). Lê via `context.get_metadata("strategy_id")` com fallback para `instruction.name`/`instruction.instruction_id`.

**Hidratação preguiçosa de recursos.** `StrategyExecutor.prepare_context` apenas registra os `StrategyResource` da definição no contexto via `set_resource()` — **não chama `loader.hydrate`**. Isso evita pagar o custo de carregar parquets gigantes / pickles de modelos a cada troca de estratégia ou refresh da UI. A hidratação acontece sob demanda em quatro pontos: (1) `StrategyExecutor.execute_single_instruction` chama `_rehydrate_inputs_if_needed`, que para cada `input_resource_id` da instrução chama `context.ensure_payload(rid)` antes de delegar ao executor concreto; (2) **`StrategyInstructionExecutorBase.validate()`** (botão Validar da UI) também chama `context.ensure_payload(rid)` para cada input declarado antes dos estágios de validação — sem isso, validators que checam `isinstance(payload, ...)` falhariam logo após reabrir um projeto; (3) `ValidationRuleBase._get_or_compute_stats` e regras concretas (`MinCalibrationPairsRule`, `DualPipelineDimensionCompatibilityRule`, `DualPipelineSubsequenceRule`) chamam `context.ensure_payload(rid)` antes de ler `payload`; (4) UI controllers (visualização de trajetória, distribuição de scores) chamam `loader.hydrate` direto quando o usuário clica para visualizar. O `StrategyExecutionContext` ganhou os atributos runtime `resource_loader` e `embedding_column` (preenchidos por `prepare_context`, excluídos do `to_dict`) e o método `ensure_payload(resource_id) -> Optional[StrategyResource]` que delega ao loader quando `payload is None` e `artifact_ref` é válido. Se o load falhar, o método loga via `context.log` a `_last_error` que o loader registrou — visível no LogProgressDialog. O controller V2 também NÃO hidrata mais em `_register_project_resources_in_context` — recursos do projeto ficam no contexto só com `resource_id`/`resource_type` para popular dropdowns.

**`ValidationResult`** flows through everything. Uses `merge()` to aggregate results from multiple rules. Four factories: `ok()`, `neutral()`, `critical(message)`, `warning(message)`.

**Pipeline de validação — 5 estágios em dois níveis.** A validação ocorre em dois pontos distintos com estágios bem definidos. Cada estágio agrega resultados no mesmo `ValidationResult` via `merge()`.

**Nível 1 — Validação da estratégia** (antes de qualquer instrução, em `StrategyExecutor._validate_strategy()`):

| Estágio | Onde | O que valida |
|---|---|---|
| Strategy Rules | `LRStrategyDefinition.validation_rule_ids` | Regras globais da estratégia (ex: isolamento entre datasets de treino e calibração). Executam uma vez, antes de qualquer fase. |

**Nível 2 — Validação de instrução** (em `StrategyInstructionExecutorBase.validate()`, executado automaticamente antes de cada `execute()` e pelo botão "Validar" da UI):

| Estágio | Onde | O que valida |
|---|---|---|
| 1. Structural | `validate()` — topo | Null checks em `instruction` e `context`. Falha aqui aborta os demais. |
| 2. Pluggable | `instruction.validation_rule_ids` | Regras registradas no contexto por `rule_id`, referenciadas pela instrução. Políticas do laboratório. |
| 3. Dynamic | `instruction.build_validation_rules()` | Regras construídas pelo DTO com base no estado atual dos campos. Ex: `PipelineTrainingInstruction` DUAL retorna 3 regras (overlap, dimensões, subsequência). |
| 4. Executor | `_validate_instruction_specifics()` | Hard-coded no executor concreto: tipo da instrução, campos obrigatórios, payloads corretos, recursos no contexto. |

**Gate de dependência** (separado do pipeline): `StrategyExecutor.can_execute_instruction()` verifica se todos os `input_resource_ids` estão presentes no contexto. É um gate de execução que roda ANTES do pipeline de validação — na UI, impede o botão "Run" de ativar.

**Regras concretas**: 7 implementações em `faceaudit/validation_rules/`, todas estendendo `ValidationRuleBase` (`faceaudit/validation_rules/validation_rule_base.py`). A classe base fornece o helper `_get_or_compute_stats(resource_id, context)`: busca `"{resource_id}_stats"` no contexto ou computa via `DatasetStatsService` e armazena como `DATASET_STATS` resource — custo pago uma vez por dataset por execução. `DatasetStats` inclui `embedding_dim` para suportar a Regra de Fisher. **Não implementar `ValidationRuleProtocol` diretamente — estender sempre `ValidationRuleBase`.**

**`LRStrategyDefinition` e `LRStrategyTrainedModel` NÃO herdam de seus respectivos Protocols.** Em Python, herança explícita de `Protocol` é desnecessária — a tipagem estrutural garante compatibilidade sem declaração. Além disso, as `@property` declaradas no Protocol bloqueavam o `__init__` gerado pelo `@dataclass` (sem setter → `AttributeError`). Ambas as classes herdam apenas de `JSONSerializableMixin`. O `isinstance(obj, LRStrategyDefinitionProtocol)` continua funcionando via `@runtime_checkable` (verifica presença de atributos, não hierarquia). **Não adicionar herança de Protocol em nenhum `@dataclass` concreto da nova arch.**

**Pendrive Paradigm — projeto autocontido e portátil.** Um projeto FaceAudit na nova arquitetura é uma **pasta autocontida** ancorada por um arquivo `.fa` (ex: `caso_01.fa`) que armazena apenas os metadados do projeto (`FaceAuditProjectMetadata`: nome, id, datas). A estrutura de subpastas é criada por `LocalProjectArtifactStorage` (`faceaudit/storage/local_project_artifact_storage.py`) a partir do enum `ProjectFolder` — todas as referências em `StrategyArtifactReference.path` são sempre **relativas** à raiz do projeto. `FaceAuditProjectService` (`faceaudit/services/project_service.py`) é o ponto de entrada para criação e abertura de projetos, agregando storage + repositórios. O `ArtifactStorageService` existente (`faceaudit/storage/artifact_storage_service.py`) é um adaptador transitório para a arquitetura antiga e não deve ser usado na nova arch.

**Sistema de dois níveis — biblioteca global + projeto.** Além dos projetos autocontidos, existe uma **biblioteca global** (pasta configurada na máquina) gerenciada por `GlobalLibraryStorage` (`faceaudit/storage/global_library_storage.py`) com as subpastas definidas por `GlobalLibraryFolder`. A biblioteca global armazena recursos reutilizáveis entre projetos: datasets (`.parquet`), split configs (`.json`), mining configs (`.json`) e templates de estratégia (`LRStrategyDefinition` sem recursos vinculados, `.json`). Ao montar uma estratégia, o usuário seleciona peças da biblioteca e elas são copiadas para o projeto via `copy_artifact_to_project()`, garantindo que o projeto permaneça autocontido. Templates de estratégia são `LRStrategyDefinition` com instruções configuradas mas `artifact_ref=None` nos recursos — ao instanciar para um projeto, os recursos são vinculados e copiados.

**Verificação de integridade é automática na abertura do projeto.** `FaceAuditProjectService.open(verify_integrity=True)` chama `project.check_integrity()` antes de retornar, que carrega todas as `LRStrategyDefinition` do repositório, coleta os `artifact_ref` dos recursos e delega a `LocalProjectArtifactStorage.verify_integrity(refs)`. O resultado fica em `project.integrity_results`; `project.has_integrity_problems` retorna True apenas para `MISSING` e `MODIFIED` — `HASH_UNAVAILABLE` (arquivo sem hash registrado) não bloqueia. `verify_integrity()` não faz parte do `ArtifactStorageProtocol` — é uma capacidade exclusiva da implementação local, não exposta via protocolo para não forçar implementações externas a suportá-la.

**`StrategyInferenceService` é o ponto de entrada para inferência forense na nova arch.** Recebe `LRStrategyTrainedModel` (pickle, modelos completamente hidratados) + `StrategyResource` probe/reference (com `artifact_ref` para carregar do disco se necessário) + `ArtifactStorageProtocol` do projeto. Auto-detecta pipeline, similarity model e calibration model por `StrategyResourceType`. Modo LINEAR: `pipeline.transform()` nos dois datasets; DUAL: `transform_source(probe)` + `transform_target(reference)`. Score via `score_matrix()` → calibração via `calibrate_scores()` → `log_lr` em `sample.metadata["log_lr"]`. Isolamento por estratégia garantido: `model_repository.load(strategy_id)` carrega sempre os modelos corretos daquela estratégia.

**`faceaudit/transformators/` is the shared math layer between the two pipeline implementations.** Both `TransformationPipeline` (old arch) and `EmbeddingTransformationPipeline` (new arch) instantiate the same 6 transformator classes from this folder. The transformators themselves are already new-arch compatible: they implement `EmbeddingTransformationProtocol`, receive config DTOs via constructor injection, delegate all math to `EmbeddingMathService`, and expose `get_state()`/`set_state()` for the DUAL mode state synchronization. The architectural difference is entirely in the orchestrating pipeline, not in the transformators.

#### The 11 instruction types

| Type | Executor | What it does |
|---|---|---|
| `PIPELINE_TRAINING` | `PipelineTrainingInstructionExecutor` | Trains `EmbeddingTransformationPipeline` (LINEAR or DUAL); writes `TRAINED_PIPELINE` resource |
| `PIPELINE_APPLICATION` | `PipelineApplicationInstructionExecutor` | Applies a trained pipeline to dataset(s); writes `INTERMEDIATE_DATASET` resource(s) |
| `DATASET_SPLIT` | `DatasetSplitInstructionExecutor` | Splits dataset into A and B. Modes: `BY_IDENTITY_RATIO`, `BY_IDENTITY_LIST`, `BY_IDENTITY_ABSOLUTE` (rigid floors `min_calibration_pairs`/`min_calibration_ids`/`min_samples_per_id`), `BY_INDEX_RATIO`, `BY_INDEX_LIST`, `BY_SUPERVISED_ADAPTATION` (receives source+target, prioritizes identities in source∩target for calibration, falls back to target-exclusive identities). Identity-based modes accept `grouping_strategy=PRESERVE_LOOKALIKES` (cluster_size lookalike package, computed via `ComparisonService.get_identity_confusion_matrix`). Advanced modes (ABSOLUTE, SDA, RATIO+LOOKALIKES) delegate to `PLDASplitService` via injection and require `EmbeddingDataset`; trivial modes still work on both `FaceDataset` and `EmbeddingDataset`. Convention for advanced modes: A = training (`plda_set`), B = calibration (`calib_set`). |
| `PAIRED_DATASET_SPLIT` | `PairedDatasetSplitInstructionExecutor` | Splits **two paired datasets simultaneously by identity** — same shuffle is applied to both. Inputs: `primary_dataset_resource_id` + `secondary_dataset_resource_id` (e.g. transformed frontal + transformed lateral views with shared identities). Outputs: 4 datasets — `output_primary_a/b` and `output_secondary_a/b`, where the B splits carry **identical identity sets** (guaranteeing genuine cross-pose pairs for calibration scoring). Only the **intersection** of identities between primary and secondary is the candidate pool for the shuffle; identities exclusive to one side fall into the A split of their own dataset (not discarded, but not paired). Reuses `PLDASplitService._select_ids` over the combined pool restricted to the intersection — supports `PRESERVE_LOOKALIKES` with `cluster_size`, plus `min_calibration_pairs/ids/samples_per_id` floors. Designed for cross-pose forensic strategies where the calibrator must train on the same pair distribution as casework (probe lateral × reference frontal). |
| `DATASET_MERGE` | `DatasetMergeInstructionExecutor` | Concatenates or updates datasets by index list |
| `PAIR_MINING` | `PairMiningInstructionExecutor` | Consumes a `SimilarityScoreDataset` (output of SIMILARITY_SCORING) and produces a curated `SimilarityScoreDataset` according to `ComparisonMiningConfig`. Independent rules for genuines and impostors (HARD/EASY/TYPICAL/RANDOM/MIXED with proportions; focus PAIRS or IDENTITIES; round-robin by pivot or global-order slicing). Sort key is `sample.raw_score` (HARD = top score = hardest impostors / easiest genuines). Output is a single concatenated dataset (impostors first, then genuines) — class separation is preserved per-sample via `same_person`/`label`, so `CALIBRATION_MODEL_TRAINING` consumes it directly. Designed to fix calibrator overconfidence in forensic LR strategies where the calibrator was trained on too-easy mixed-pose pairs — mining top-k hard impostors gives the calibrator a realistic score distribution. `ComparisonMiningConfig` is unchanged from the old arch (DTO is generic — strategies/focus/limits/proportions, no coupling to embedding type). `from_dict` tolerates legacy field names (`comparison_dataset_resource_id`, `output_impostors_resource_id`, `output_genuines_resource_id`). |
| `SIMILARITY_MODEL_TRAINING` | `SimilarityModelTrainingInstructionExecutor` | Prepares/trains a `CosineSimilarityModel` (stateless — no dataset/config) or `PLDASimilarityModel`. PLDA hyperparameters are embedded in the instruction itself via the nested `similarity_model_config: PLDASimilarityModelConfig` field — no external `*_config_resource_id` resource. Writes `TRAINED_SIMILARITY_MODEL` resource |
| `SIMILARITY_SCORING` | `SimilarityScoringInstructionExecutor` | Scores all pairs (self-comparison or cross-comparison); writes `SimilarityScoreDataset` resource |
| `CALIBRATION_MODEL_TRAINING` | `CalibrationModelTrainingInstructionExecutor` | Trains `LogisticRegressionCalibrationModel` or `KDECalibrationModel` from a `SimilarityScoreDataset` or `{scores, labels}` dict. Hyperparameters are embedded inline via two nested fields (`logreg_config`, `kde_config`); only the one matching `calibration_model_type` is used by the executor — the other may stay populated for lossless type-toggling. **LogReg config supports GridSearch for `C`** (default `use_gridsearch=True`) via `c_grid`/`cv_folds`/`scoring: LogRegGridScoringMetric` (enum NEG_LOG_LOSS/ROC_AUC) — when active, fit uses `GridSearchCV` with `StratifiedKFold` (auto-reduces folds to class minority size); the chosen `C` is exposed as `model.best_C_` and emitted in the instruction's `metadata["best_C"]`. KDE config carries 13 fields (kernel, bandwidths, GridSearch toggle, Fisher epsilon, log-LR clipping, etc.) |
| `SCORE_CALIBRATION` | `ScoreCalibrationInstructionExecutor` | Applies a trained `CalibrationModelProtocol` to a `SimilarityScoreDataset` (raw_score) and produces a new dataset with per-pair `log_lr` populated in `sample.metadata["log_lr"]`. Output type is `StrategyResourceType.CALIBRATED_SCORES` (distinct from `SIMILARITY_SCORES`). Reuses the `SimilarityScoreDataset` Python class — `to_parquet` automatically detects `log_lr` in metadata and adds a native `log_lr: float64` column; the loader populates `metadata["log_lr"]` when the column exists (legacy parquets without the column continue to load unchanged). Vectorized via `calibrator.calibrate_scores(scores)`; preserves `raw_score`, `same_person`, `probe`/`reference` from input. No own hyperparameters — calibration is fully embedded in the trained model. |
| `STRATEGY_EVALUATION` | `StrategyEvaluationInstructionExecutor` | **Validation-only** instruction. Reads a `CALIBRATED_SCORES` resource (with `metadata["log_lr"]` populated per sample), separates `gen_log_lrs`/`imp_log_lrs` by `sample.label`, and delegates entirely to `ValidationMathService.calculate_metrics(strategy_id, strategy_name, strategy_type="V2", genuine_scores, impostor_scores)` — which uses `sklearn.isotonic.IsotonicRegression` for PAVA, computes EER + threshold, Cllr (numerically stable via `logaddexp`), Cllr_min, AUC, and ECE plot data via numpy broadcasting. Output is a `StrategyValidationResult` (with `ValidationMetrics` + `ValidationPlotData`) registered as `StrategyResourceType.VALIDATION_RESULT` and persisted as `.pkl` (preserves ndarray). No own hyperparameters — deterministic. |

#### Adding new components — use the project skills

Project skills are available in `.claude/commands/`. Use them instead of creating components manually:

| Command | What it creates |
|---|---|
| `/new-instruction <Name>` | Complete new instruction type: enum + DTO + dispatch + executor + factory registration |
| `/new-similarity-model <Name>` | New similarity model: enum + config DTO + implementation + executor integration |
| `/new-calibration-model <Name>` | New calibration model: enum + config DTO + implementation + executor integration |
| `/new-validation-rule <Name>` | New validation rule extending `ValidationRuleBase`; includes `_get_or_compute_stats()` cache helper for dataset rules |
| `/new-transformation <Name>` | New embedding transformation: `StepType` + config DTO + `TransformationType` member + transformator class + pipeline registration |

---

## Folders with mixed or old-only code

| Folder | Status |
|---|---|
| `faceaudit/manager/` | Old architecture only |
| `faceaudit/controllers/` | Old architecture only |
| `faceaudit/strategies/` | Old architecture only (`KDEStrategy`, `PLDAStrategy`, `PLDASnormStrategy`) |
| `faceaudit/transformators/` | **Shared math layer** — 6 concrete transformators (`CoralTransformation`, `WhiteningTransformation`, `LdaTransformation`, `CenteringTransformation`, `L2Normalization`, `IdentityAggregation`); already refactored for new arch (implement `EmbeddingTransformationProtocol`, inject config DTOs, expose `get_state()`/`set_state()`) |
| `faceaudit/pipelines/` | Mixed — `EmbeddingTransformationPipeline` is new arch (self-contained, LINEAR + DUAL modes, `PipelineTrainingPolicy`); `TransformationPipeline` is old arch (apply-only, fit delegated to `PLDATransformationPipelinesController`); others (`casework_pipeline.py`, `validation_pipeline.py`, `feature_extraction_pipeline.py`) are old arch |
| `faceaudit/services/` | Mixed — old arch: `FileSystemService`, `ConfigLoaderService`, etc.; new arch: `FaceAuditProjectService`, `StrategyResourceLoaderService`, `StrategyInferenceService`, `PipelineTrajectoryService` |
| `faceaudit/protocols/` | Mixed — see protocol list above |
| `faceaudit/models/` | Mixed — subfolders `calibration/`, `instructions/`, `pipeline/`, `similarity/`, `storage/`, `strategy/`, `validation/` are new; root-level models are shared (see notes below) |
| `interface/tabs/strategy_manager_v2/` | **New architecture frontend** — see section below |
| `interface/` (rest) | Old architecture only — not yet migrated |

### Root-level models in `faceaudit/models/` — usage notes

Most root-level models are shared between both architectures. Key points for the new architecture:

- **Used by validation rules**: `dataset_stats.py` (`DatasetStats`) — DTO com `identity_set`, `n_ids`, `embedding_dim` (campo adicionado); `DatasetStatsService.compute_from_embedding_dataset()` computa stats de `EmbeddingDataset` sem recarregar vetores pesados; results cached no contexto como `DATASET_STATS` resource via convenção `"{resource_id}_stats"`.
- **Already in use**: `embedding_info.py`, `embedding_dataset.py`, `face_dataset.py`, `face_info.py`, `transformation_pipeline_config.py`, `transformator_config.py`, `whitening_config.py`, `lda_config.py`, `coral_config.py`, `aggregation_config.py` (used by `IdentityAggregation` via `EmbeddingTransformationPipeline`). Note: `transformation_pipeline_config.py` has a `to_summary()` method that references `PipelineSummary` (old UI) — irrelevant for the new arch but not a blocker.
- **Planned as instruction configs (next)**: `split_config.py` (`SplitConfig`) — será embutido em `DatasetSplitInstructionDefinition`; `comparison_mining_config.py` (`ComparisonMiningConfig`) — será embutido em `PairMiningInstructionDefinition`. Ambos já têm `from_dict()` com hidratação correta de enums.
- **Candidate for future use**: `filter_config.py` (dataset filtering) — DTO bem estruturado, sem instrução correspondente ainda
- **Old-arch replacements** — do not use in new arch: `kde_config.py` (replaced by `models/calibration/kde_calibration_model_config.py`), `plda_config.py` (replaced by `models/similarity/plda_similarity_model_config.py`), `project_config.py` (replaced by `FaceAuditProjectMetadata` + `.fa` file in new arch), `strategy_entry.py` (replaced by `LRStrategyDefinition`)
- **Old-arch UI/analysis only**: `kde_validation_input.py`, `plda_validation_input.py`, `pipeline_summary.py`, `transformation_trajectory_data.py`, `kde_visualization_data.py`, hubness/ablation/gaussianity analysis files

---

## Strategy Manager V2 — Frontend (Nova Arch)

The tab is registered in `interface/main_gui_controller._init_strategy_manager_v2()`. It receives `global_storage` (optional `GlobalLibraryStorage`) and `app_config` (for resolving initial FileDialog directories).

### Folder structure

```
interface/tabs/strategy_manager_v2/
├── strategy_manager_v2_controller.py   ← Central controller (QObject); owns project + execution state
├── strategy_manager_v2_view.py         ← QStackedWidget: welcome screen (0) + workspace (1)
├── toolbar/
│   ├── global_resources_toolbar.py     ← Pipeline / Mining / Split / Manage Library buttons
│   └── project_toolbar.py             ← New / Open / Save / Close / Run / Save as Template
├── left_panel/
│   ├── left_panel_widget.py            ← QSplitter vertical with 3 sections; forwards signals
│   ├── strategy_list_widget.py         ← QListWidget of LRStrategyDefinition; New/Duplicate/Remove
│   ├── resource_tree_widget.py         ← QTreeWidget grouped by StrategyResourceType; "Adicionar Recurso" dropdown
│   └── resource_info_widget.py         ← Metadata + usage + pipeline details + dataset stats of selected resource
├── strategy_editor/
│   ├── strategy_editor_widget.py       ← QTabWidget with 3 tabs (Preparação/Validação/Caso), each with list+config+produced
│   ├── instruction_list_widget.py      ← Drag-and-drop list with ▶ Run button + visual states (🔒/✅)
│   ├── instruction_config_panel.py     ← Dynamic form with type-filtered dropdowns via ResourceFieldSpec + Validate button
│   └── produced_resources_widget.py    ← Shows strategy-produced resources per phase (⬜ placeholder / 🟡 in memory / 🟢 saved)
├── comparison_tab/
│   └── comparison_tab_widget.py        ← Placeholder for future strategy comparison
├── dialogs/
│   ├── new_project_v2/                 ← Name + description + destination folder
│   ├── new_strategy/                   ← From scratch or from GlobalLibraryStorage template
│   └── manage_library/                 ← List/remove resources from GlobalLibraryStorage
└── workers/
    ├── strategy_execution_worker.py    ← QObject + moveToThread; runs StrategyExecutor.execute() (all phases)
    ├── instruction_execution_worker.py ← QObject + moveToThread; runs single instruction via execute_single_instruction()
    └── project_io_worker.py            ← QObject + moveToThread (currently unused; open is sync)
```

### Key design decisions in the V2 frontend

**Project open is synchronous with WaitCursor, not threaded.** On Windows, `ApplicationModal` dialogs prevent queued signal delivery from worker threads, causing `LogProgressDialog` to hang forever. Solution: `FaceAuditProjectService.open()` runs on the main thread with `QApplication.setOverrideCursor(WaitCursor)`. The `ProjectIOWorker` exists but is not used for opening.

**"Adicionar Recurso" lives in the left panel, not the toolbar.** The `ResourceTreeWidget` has a `QToolButton` with `MenuButtonPopup` showing 4 options (Dataset / Split Config / Mining Config / Pipeline Config). `LeftPanelWidget` forwards the 4 signals up to `StrategyManagerV2Controller`. The button is disabled until a project is open.

**Resource discovery = definitions scan + disk scan.** `_collect_project_resources()` in the controller does two passes: (1) resources declared in `LRStrategyDefinition` objects, (2) files on disk found via `storage.list_artifacts()` for each importable type. This ensures imported files appear in the tree even before they are referenced by a strategy.

**`app_config.paths.*` drives FileDialog initial directories.** The controller receives `app_config` from `main_gui_controller`. `_import_resource()` reads `app_config.paths.<attr>` (e.g. `datasets`, `split_configs`, `mining_configs`, `pipelines`) to open the FileDialog in the right folder. Falls back to `Path.home()` if the attribute is missing or `app_config` is None.

**Per-instruction execution with dependency tracking and auto-persistence.** The controller maintains persistent execution state: `_execution_context` (StrategyExecutionContext), `_executed_instruction_ids` (set), `_executor` (StrategyExecutor). When the user clicks "Run" on an instruction, `_ensure_executor_and_context()` lazily creates the executor and context, `can_execute_instruction()` checks dependencies, then `InstructionExecutionWorker` runs the instruction in a separate thread. After success: (1) `_persist_produced_resources()` saves produced resources to disk via `persist_produced_resource()`, (2) `_refresh_instruction_states()` recalculates which instructions are now runnable, (3) `_refresh_produced_resources()` updates the visual state of produced resources. All execution state resets when the strategy is edited, reordered, or switched — but persisted resources (with `artifact_ref`) survive the reset.

**Three-tab strategy editor with produced resources.** The `StrategyEditorWidget` uses a `QTabWidget` with three tabs (Preparação, Validação, Análise de Caso). Each tab has a horizontal splitter: left container (`InstructionListWidget` + `ProducedResourcesWidget` stacked vertically) and right `InstructionConfigPanel`. Visual instruction states: locked (gray), runnable (normal icon), executed (green checkmark). The `ProducedResourcesWidget` shows output resources per phase with three states: placeholder (white), in-memory (yellow), saved to disk (green).

**Automatic placeholder sync and resource persistence.** When the user modifies a strategy (adds/removes/edits instructions), `_sync_all_output_placeholders()` scans all instructions' `get_output_resource_placeholders()`, creates missing placeholder `StrategyResource` objects in `LRStrategyDefinition.resources`, and removes orphaned placeholders (no payload, no artifact_ref). After each successful instruction execution, `_persist_produced_resources()` calls `StrategyExecutor.persist_produced_resource()` for each `produced_resource_id`, saving the artifact to disk and updating the definition's `artifact_ref`. The definition is saved to the repository automatically.

**Type-filtered resource dropdowns.** The `InstructionConfigPanel` uses `get_resource_field_specs()` from each instruction DTO to render dropdowns that only show resources of the correct `StrategyResourceType`. Output fields are rendered as text inputs. Falls back to generic rendering for instructions without specs. The panel has two buttons: "Aplicar alterações" (applies form to DTO) and "Validar" (applies form then runs full validation via the instruction executor, showing errors/warnings in a QMessageBox).

**Reused from old-arch frontend** (imported directly, no copy):
- `LogProgressDialog` — used for strategy execution
- `PipelineCreatorController` — "Pipeline" button in global resources toolbar
- `NewMiningConfigDialogController` — "Mining" button
- `NewSplitConfigDialogController` — "Split" button

### StrategyArtifactType / StrategyFileType — import mappings

When `copy_artifact_to_project()` is called with a global-library type, `StrategyFileType._artifact_map()` maps it to a project-local file type:

| `StrategyArtifactType` | `StrategyFileType` | Destination in project |
|---|---|---|
| `GLOBAL_DATASET` | `SOURCE_DATASET` | `datasets/{dataset_id}.parquet` |
| `SPLIT_CONFIG` | `PROJECT_SPLIT_CONFIG` | `split_configs/{config_id}.json` |
| `MINING_CONFIG` | `PROJECT_MINING_CONFIG` | `mining_configs/{config_id}.json` |
| `PIPELINE_CONFIG` | `PROJECT_PIPELINE_CONFIG` | `pipelines/{config_id}.json` |

`PIPELINE_CONFIG` was added to `StrategyArtifactType` during V2 frontend development. `STRATEGY_TEMPLATE` and `UNKNOWN` intentionally have no project-level mapping.

---

## External Tools (accessed via services)

- **AdaFace** — primary embedding model for facial images
- **Forensic Face / sepaelv2** — face detection and embedding extraction (`FFService`)
- **OFIQ** — presentation attack detection (`OFIQService`)
- **SERFIQ** — image quality assessment (`SERFIQService`)
- **SpeechBrain** — PLDA backend used by `PLDASimilarityModel`
