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
| `faceaudit/services/strategy_resource_loader_service.py` | `StrategyResourceLoaderService` — hidrata `payload` de `StrategyResource` a partir de `artifact_ref`; usado pelo `StrategyExecutor` antes da execução e pelo `StrategyInferenceService` para carregar probe/reference |
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
- `StrategyExecutionContextProtocol` — state bus contract: get/set/list resources, artifacts, rules, logs
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

**`StrategyResource` is the unit of exchange between instructions.** Every instruction reads named resources from the context and writes named resources back. The names are declared in the instruction DTO itself (`input_resource_ids`, `output_resource_ids`), kept in sync via `_sync_declared_io()` in `__post_init__`. Two categories exist: **project resources** (DATASET, *_CONFIG — imported by the user, shared across strategies, always have `artifact_ref`) and **strategy-produced resources** (TRAINED_PIPELINE, INTERMEDIATE_DATASET, TRAINED_SIMILARITY_MODEL, TRAINED_CALIBRATION_MODEL — produced by instruction execution). Strategy-produced resources start as **placeholders** (payload=None, artifact_ref=None) when the user fills in output fields in the instruction config panel. After execution, they are persisted to disk and gain `artifact_ref`. `StrategyResourceType.to_artifact_type()` maps the logical resource type to the physical `StrategyArtifactType` for persistence. `StrategyInstructionDefinition.get_output_resource_placeholders()` creates placeholder `StrategyResource` objects from `ResourceFieldSpec` entries with `is_output=True`. The controller calls `_sync_all_output_placeholders()` on each strategy modification to keep `LRStrategyDefinition.resources` in sync. `StrategyExecutor.persist_produced_resource(resource, strategy_id)` handles disk persistence, returning a `StrategyArtifactReference`.

**`ResourceFieldSpec` enables type-filtered resource dropdowns.** Each instruction DTO implements `get_resource_field_specs()` returning a list of `ResourceFieldSpec(field_name, label, accepted_types, is_output)`. The UI config panel uses these specs to show only resources of the correct `StrategyResourceType` in each dropdown. It's an instance method (not classmethod) because Pipeline Training/Application specs vary by `pipeline_mode` (LINEAR vs DUAL).

**`StrategyInstructionDefinition.from_dict()` is a polymorphic dispatcher.** It reads `instruction_type` and hydrates the correct concrete DTO. This is how the strategy survives JSON serialization/deserialization with full type information.

**Similarity and calibration are separated.** The old `LRModelProtocol` was monolithic (fit + score + calibrate). The new architecture splits it: `SimilarityModelProtocol` produces raw scores; `CalibrationModelProtocol` converts raw scores to LRs. They connect via `SimilarityScoreDataset`.

**`KDECalibrationModel` vs `KDEStrategy`**: In the old architecture, KDE was a scoring strategy (fit on embeddings, produce LRs). In the new architecture, `KDECalibrationModel` is a calibrator only — it receives pre-computed raw scores and fits two KDEs (genuine/impostor) with Fisher transform + adaptive GridSearch bandwidth. The `CosineSimilarityModel` and `PLDASimilarityModel` handle the raw scoring.

**`PipelineTrainingPolicy`** controls the DUAL mode behavior: which transformation steps (`CORAL`, `Whitening`, `LDA`) may use the target dataset during training, and whether shared state is copied to the target branch.

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

#### The 8 instruction types

| Type | Executor | What it does |
|---|---|---|
| `PIPELINE_TRAINING` | `PipelineTrainingInstructionExecutor` | Trains `EmbeddingTransformationPipeline` (LINEAR or DUAL); writes `TRAINED_PIPELINE` resource |
| `PIPELINE_APPLICATION` | `PipelineApplicationInstructionExecutor` | Applies a trained pipeline to dataset(s); writes `INTERMEDIATE_DATASET` resource(s) |
| `DATASET_SPLIT` | `DatasetSplitInstructionExecutor` | Splits dataset into A and B by identity ratio/list or index ratio/list; works on both `FaceDataset` and `EmbeddingDataset` |
| `DATASET_MERGE` | `DatasetMergeInstructionExecutor` | Concatenates or updates datasets by index list |
| `PAIR_MINING` | `PairMiningInstructionExecutor` | Curates training pairs using `ComparisonMiningConfig` (genuine/impostor rules, hard/easy/random/mixed strategies); writes mined `SimilarityScoreDataset` resource |
| `SIMILARITY_MODEL_TRAINING` | `SimilarityModelTrainingInstructionExecutor` | Prepares/trains a `CosineSimilarityModel` or `PLDASimilarityModel`; writes `TRAINED_SIMILARITY_MODEL` resource |
| `SIMILARITY_SCORING` | `SimilarityScoringInstructionExecutor` | Scores all pairs (self-comparison or cross-comparison); writes `SimilarityScoreDataset` resource |
| `CALIBRATION_MODEL_TRAINING` | `CalibrationModelTrainingInstructionExecutor` | Trains `LogisticRegressionCalibrationModel` or `KDECalibrationModel` from a `SimilarityScoreDataset` or `{scores, labels}` dict |

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
