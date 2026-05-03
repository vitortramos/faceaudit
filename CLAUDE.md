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
| `faceaudit/services/two_covariance_plda_service.py` | Implementação numpy pura do PLDA de duas covariâncias (Brümmer & de Villiers 2010): `fit_two_covariance(X, y, regularization)` em forma fechada (sem EM, via scatter matrices intra/entre-classes), `score_two_covariance_pair`/`score_two_covariance_matrix` por log-LR vetorizado, `_compute_scoring_kernels` pré-computa Q/P/const para reuso. Usado pelo `PLDASimilarityModel` quando `config.plda_variant=TWO_COVARIANCE`. |
| `faceaudit/services/hubness_analysis_service.py` | `HubnessAnalysisService` — diagnóstico de viés topológico em alta dimensionalidade. Métodos: `compute_analysis` (KNN exaustivo O(N²) com batching B×C, soma por exclusão, label encoding e top-K contínuo entre chunks), `compute_stratification` (5 faixas de yaw), `compute_paired_study`, `compute_quality_diagnostics`, `compute_optimization_study`, `filter_dataset_by_quality`, `compute_ablation_by_quantity`, `compute_ablation_by_pose`. Aceita `logger_callback: Optional[Callable]` em `compute_analysis`/`_run_ablation_engine` que substitui prints+tqdm por progresso periódico via callback (a cada 10%) — preserva comportamento original quando None. Usado pelos executores `HUBNESS_ANALYSIS` e `HUBNESS_ABLATION`. |
| `faceaudit/services/hubness_plot_service.py` | `HubnessPlotService` — 20 métodos matplotlib/seaborn que consomem os DTOs do `HubnessAnalysisService`. Os métodos criam Figure interna via `plt.figure` e finalizam com `plt.show()`. O `HubnessReportWidget` reutiliza esses métodos via monkey-patch local de `plt.show` + captura de `plt.gcf()` + transplante via PNG buffer — evita refatorar 20 métodos. |
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

**Filtro de instruções por fase (UI).** O menu "➕ Adicionar" da `InstructionListWidget` (em `interface/tabs/strategy_manager_v2/strategy_editor/instruction_list_widget.py`) filtra os tipos disponíveis pela fase da aba: `_TRAINING_ONLY_TYPES = {PIPELINE_TRAINING, SIMILARITY_MODEL_TRAINING, CALIBRATION_MODEL_TRAINING}` aparece **só na Preparação**; `_VALIDATION_ONLY_TYPES = {STRATEGY_EVALUATION}` aparece **só na Validação**; `_CASEWORK_ONLY_TYPES = {CASEWORK_VISUALIZATION}` aparece **só na Análise de Caso**; demais tipos aparecem em todas. A função `_allowed_types_for_phase(phase)` consolida a lógica e cada `InstructionListWidget` recebe `phase` no construtor (a partir do `_PHASES` em `strategy_editor_widget.py`).

**Visualização de resultados forenses (`ValidationResultViewerDialog`).** A instrução `STRATEGY_EVALUATION` produz um `StrategyValidationResult` com 4 conjuntos de dados de plot pré-computados (`genuine_log_lrs`, `impostor_log_lrs`, `fpr_array`, `fnr_array`, `ece_prior_log_odds`, `ece_values`, `ece_min_values`). O `InstructionConfigPanel` mostra um botão `📊 Visualizar Resultado de Validação` quando a instrução é `StrategyEvaluationInstruction` e o recurso está em memória ou em disco — clique abre `ValidationResultViewerDialog` (em `interface/tabs/strategy_manager_v2/strategy_editor/validation_result_viewer_dialog.py`), `QDialog` 1100×780 com header âmbar (métricas escalares) + `QTabWidget` de 4 abas: **Distribuição de LRs** (histograma + KDE de gen/imp em log-LR), **Tippett Plot** (P(LR ≤ x | Hp) vs P(LR > x | Hd)), **Curva DET** (FPR×FNR em escala normit via `scipy.stats.norm.ppf`, com marca no EER) e **Calibração ECE** (curva real + curva mínima PAVA com área de cal_loss pintada entre as duas). Cada aba tem `FigureCanvas` matplotlib + `NavigationToolbar` (zoom/pan/exportar PNG). Padrão de wiring espelha `view_score_distribution_requested` — sinal no painel → reemissor no widget → slot `_on_view_validation_result` no controller, com lazy hydrate via `_resource_loader.hydrate` quando o payload está no disco.

**Visualização forense de caso (`CaseworkVisualizationViewerDialog`).** A instrução `CASEWORK_VISUALIZATION` produz um `CaseworkVisualizationResult` (em `faceaudit/models/casework_visualization_result.py`) com `score_type` (RAW/CALIBRATED), arrays `genuine_scores`/`impostor_scores`/`case_scores` (numpy float64), nomes dos datasets de origem para rastreabilidade, `eer_threshold` opcional (computado on-the-fly em modo RAW via varredura linear de 1000 thresholds — FNR=FPR crossover; `None` em modo CALIBRATED, sinalizando uso de log-LR=0 como referência neutra) e `case_summary` com min/max/mean/median pré-computados. O `InstructionConfigPanel` mostra botão `📊 Visualizar Caso` quando a instrução é `CaseworkVisualizationInstruction` e o recurso está disponível — clique abre `CaseworkVisualizationViewerDialog` (em `interface/tabs/strategy_manager_v2/strategy_editor/casework_visualization_viewer_dialog.py`), `QDialog` 950×720 single-tab com header âmbar (modo, contagens, sumário do caso, EER threshold se RAW), `FigureCanvas` matplotlib + `NavigationToolbar`, e texto de ajuda contextual no rodapé que muda conforme o modo. O plot sobrepõe à distribuição de referência (KDE+histograma de gen/imp, gen=verde / imp=vermelho) o caso forense: KDE azul quando `n_case ≥ 30` ou linhas verticais tracejadas anotadas com valor numérico (rotação 90°) quando `n_case < 30`. Padrão de wiring idêntico ao `view_validation_result_requested` — sinal `view_casework_visualization_requested` no panel → reemissor no editor widget → slot `_on_view_casework_visualization` no controller V2, com lazy hydrate.

**Edição inline de configs editáveis (botão ✎ no `InstructionConfigPanel`).** Dropdowns que selecionam recursos do tipo `PIPELINE_CONFIG` ou `MINING_CONFIG` ganham um botão `✎` à direita do combo. O helper `_wrap_combo_with_edit_button_if_needed(combo, spec)` (em `interface/tabs/strategy_manager_v2/strategy_editor/instruction_config_panel.py`) detecta `_EDITABLE_RESOURCE_TYPES = (PIPELINE_CONFIG, MINING_CONFIG)` em `spec.accepted_types` e envolve o combo num `QHBoxLayout` com `QToolButton`. O botão fica desabilitado enquanto o combo está em "(nenhum)"; emite o sinal genérico `resource_edit_requested(resource_id)` quando clicado. O controller V2 (`_on_resource_edit_requested`) despacha por `resource.resource_type`: PIPELINE_CONFIG → `PipelineCreatorController(existing_config=..., lock_name=True)`; MINING_CONFIG → `NewMiningConfigDialogController(parent=..., lock_name=True) + ctrl.load_config(...)`. Helpers internos compartilhados: `_hydrate_resource_if_needed` (lazy-load do payload via `_resource_loader.hydrate` se necessário) e `_persist_edited_config_resource` (atualiza payload em memória + contexto vivo, re-persiste sobre o mesmo path via `save_artifact(artifact_type, new_config, config_id=Path(old_ref.path).stem)` preservando o nome físico do arquivo, propaga novo `artifact_ref` para qualquer definition que referencie o recurso). Após edição, chama `_left_panel.populate_resources(...)` direto **sem recarregar `self._definitions`** — recarregar do disco descartaria seleções pendentes nos dropdowns do editor.

**`lock_name=True` em editores de config inline.** Tanto `PipelineCreatorController` quanto `NewMiningConfigDialogController` aceitam `lock_name: bool = False` (default preserva comportamento da arch antiga). Quando `True`: `input_name` (ou `txt_name`) recebe `setReadOnly(True)` + tooltip explicativo + stylesheet cinza. Razão: na nova arch, o `resource_id`/nome exibido do recurso é o stem do arquivo no disco (`Path(ref.path).stem`); editar o `config.name` interno **não** muda o nome do arquivo nem o `resource_id` mostrado nos dropdowns/árvore — então o campo é travado para evitar confusão. Renomear será operação dedicada via botão no left panel (não implementada). **Sempre passar `lock_name=True`** quando o dialog é aberto via botão ✎ na nova arch.

**`ValueFieldSpec.triggers_rebuild` para campos que alteram outros specs.** O dataclass `ValueFieldSpec` ganhou o atributo `triggers_rebuild: bool = False`. Quando `True`, o `InstructionConfigPanel` conecta o widget (QCheckBox.toggled, QComboBox.currentTextChanged, QSpinBox/QDoubleSpinBox.valueChanged, QLineEdit.editingFinished) ao slot `_on_value_triggers_rebuild`, que aplica o valor ao DTO, chama `_sync_declared_io()` e dispara rebuild via `QTimer.singleShot(0, _rebuild_and_notify)`. Caso de uso canônico: `PipelineTrainingInstruction.apply_to_training_datasets` (bool com `triggers_rebuild=True`) — quando o usuário marca/desmarca o checkbox, os 3 campos de output condicionais aparecem/somem dinamicamente.

**Recursos do projeto: PIPELINE_CONFIG e MINING_CONFIG.** Ambos são recursos de PROJETO (vivem no disco em `pipelines/*.json` e `mining_configs/*.json`) e **não estão necessariamente declarados em `LRStrategyDefinition.resources`** — só aparecem nos dropdowns via `_collect_project_resources()` (combina recursos de definitions + scan de disco em `_import_scan` que mapeia `MINING_CONFIG → StrategyResourceType.MINING_CONFIG` e `PIPELINE_CONFIG → StrategyResourceType.PIPELINE_CONFIG`). Para edição/lookup, sempre usar `_collect_project_resources()` (mesma fonte que populou o dropdown), não apenas `_current_definition.resources`. `_GLOBAL_TYPES` em `_collect_editor_resources` inclui ambos. Loader: `StrategyResourceLoaderService._load_pipeline_config` carrega `TransformationPipelineConfig.from_dict(...)`; `_load_mining_config` carrega `ComparisonMiningConfig.from_dict(...)` — ambos JSON puro, lazy-loaded. `StrategyResourceType.to_artifact_type()` mapeia ambos para os `StrategyArtifactType` correspondentes.

**Detalhes de modelos no `ResourceInfoWidget`.** O painel de propriedades do recurso (em `interface/tabs/strategy_manager_v2/left_panel/resource_info_widget.py`) ganhou duas seções dedicadas, visíveis apenas para os tipos correspondentes: **(1) Detalhes do Modelo de Similaridade** para `TRAINED_SIMILARITY_MODEL` — mostra `Tipo`, `Dim. entrada` (lê `payload.input_dimension`, property unificada entre Cosine e PLDA — no PLDA mapeia para `trained_dim`), `Dim. latente` (lê `payload.config.latent_dim`, exibe `n/a` para Cosine e para PLDA com `plda_variant=TWO_COVARIANCE`, já que duas covariâncias não usa subspace de baixo rank), e linha `Extra` consolidada (`EM iters`, `modo PLDA`, variante PLDA quando `TWO_COVARIANCE`, `N treino`, `⚠ não treinado`). **(2) Detalhes do Modelo de Calibração** para `TRAINED_CALIBRATION_MODEL` — mostra `Tipo`, `Hyperparâmetros` (LogReg: `C (efetivo)` lido de `best_C_` pós-GridSearch + nota `class_weight=balanced (Brümmer π=0.5)`; KDE: `kernel`, `bw_gen`/`bw_imp` lidos de `kde_genuine.bandwidth`/`kde_impostor.bandwidth`), `Clipping log-LR` (faixa `[min, max]` para KDE, `n/a` para LogReg), e linha `Extra` (LogReg: `slope=coef_[0][0], intercept=intercept_[0]`; KDE: `N_gen/N_imp`, `fisher_eps`). Ambas as seções fazem lazy-load via `_load_pickle(abs_path)` na primeira seleção quando `payload` está vazio. Útil para diagnóstico forense — perito vê imediatamente o C efetivo escolhido pelo GridSearch e a inclinação aprendida do calibrador.

**Detalhes de score datasets no `ResourceInfoWidget`.** Para recursos `SIMILARITY_SCORES` e `CALIBRATED_SCORES` o painel exibe a seção **"Detalhes do Score Dataset"** com 5 campos: `Total de pares`, `Genuínos (Hp)` (com %), `Impostores (Hd)` (com % e razão imp:gen), `Identidades` (total de identidades únicas em `probe ∪ reference`; quando os conjuntos diferem mostra `(probe: N / reference: M)` indicando cross-comparison entre datasets distintos) e `Calibrados (log-LR)` (sim/não, baseado na presença da coluna `log_lr` no parquet). Estratégia de leitura em três caminhos: (1) payload em memória → usa `total_genuines`/`total_impostors` cacheados + iteração para identidades; (2) sidecar JSON `<stem>_scores_stats.json` se existir e estiver mais recente que o parquet; (3) `pq.read_table(path, columns=["same_person", "probe_identity", "reference_identity"])` — leitura colunar do parquet sem hidratar o dataset inteiro, rápida mesmo para 5M+ linhas. Resultado é cacheado no sidecar JSON.

**Métricas de balanceamento de identidade no `ResourceInfoWidget` (DATASET stats).** A seção "Estatísticas do Dataset" (visível para `DATASET` e `INTERMEDIATE_DATASET`) ganhou duas linhas adicionais alimentadas por `DatasetStats.balance_metrics`: **`Balanceamento`** mostra um verdict com emoji (`✅ Balanceado`, `🟢 Aceitável`, `🟡 Moderadamente desbalanceado`, `🔴 Severamente desbalanceado`) seguido de `(CV=0.123, max/min=4.5:1)`; **`Amostras/ID`** mostra `min–max (média X.X, mediana Y.Y)`. Verdict é computado por **worst-of-two** entre o coeficiente de variação CV (std/mean das contagens por identidade) e o imbalance ratio (max/min): tiers são `Balanceado` (CV=0 e ratio=1), `Aceitável` (CV ≤ 0.35 e ratio ≤ 4), `Moderadamente desbalanceado` (CV ≤ 0.75 e ratio ≤ 10), `Severamente desbalanceado` (CV > 0.75 ou ratio > 10). Alimentado pelo novo campo `DatasetStats.samples_per_identity: Dict[str, int]` (populado em todos os 4 caminhos de `DatasetStatsService` — `_read_parquet_stats`, `_read_pickle_stats`, `compute_from_dataset`, `compute_from_embedding_dataset`). O sidecar `<stem>_stats.json` cacheia o dict serializado de `BalanceMetrics`; cache antigo sem chave `balance_metrics` é invalidado e recomputado automaticamente (mesmo padrão de invalidação por mtime usado para `n_identities` em score datasets). Inaplicável (≤ 1 identidade) → exibe `n/a (≤ 1 identidade)`.

**Stale-cache check via mtime nos sidecars de stats.** Tanto `_load_scores_details` quanto `_load_stats` (DATASET / INTERMEDIATE_DATASET) comparam `parquet.mtime` com `cache.mtime` antes de usar o sidecar. Se `parquet.mtime > cache.mtime + 1.0` (folga para clock skew), o cache é invalidado e recomputado. Necessário porque a re-execução de instruções (PAIR_MINING com `max_global_pairs` diferente, SIMILARITY_SCORING reexecutado etc.) reescreve o parquet sobre o mesmo path, e sem essa verificação o widget continua mostrando contagens de execuções anteriores. Também serve como guard contra cache corrompido entre versões do código.

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

#### The 14 instruction types

| Type | Executor | What it does |
|---|---|---|
| `PIPELINE_TRAINING` | `PipelineTrainingInstructionExecutor` | Trains `EmbeddingTransformationPipeline` (LINEAR or DUAL); writes `TRAINED_PIPELINE` resource. Optional flag `apply_to_training_datasets` (default False): when checked, applies the freshly trained pipeline to the training dataset(s) immediately after fit and registers transformed datasets as additional outputs (`output_transformed_dataset_resource_id` in LINEAR; `output_transformed_source_dataset_resource_id` + `output_transformed_target_dataset_resource_id` in DUAL). Avoids the common pattern of chaining a redundant PIPELINE_APPLICATION on the same dataset. The conditional output specs only appear in `get_resource_field_specs()` when the flag is set; the bool field is declared with `triggers_rebuild=True` so the form re-renders immediately on toggle. Default `PipelineApplicationPolicy` is used (no checkpoints/trajectory) — for those artifacts, run a dedicated PIPELINE_APPLICATION afterwards. |
| `PIPELINE_APPLICATION` | `PipelineApplicationInstructionExecutor` | Applies a trained pipeline to dataset(s); writes `INTERMEDIATE_DATASET` resource(s) |
| `DATASET_SPLIT` | `DatasetSplitInstructionExecutor` | Splits dataset into A and B. Modes: `BY_IDENTITY_RATIO`, `BY_IDENTITY_LIST`, `BY_IDENTITY_ABSOLUTE` (rigid floors `min_calibration_pairs`/`min_calibration_ids`/`min_samples_per_id`), `BY_INDEX_RATIO`, `BY_INDEX_LIST`, `BY_SUPERVISED_ADAPTATION` (receives source+target, prioritizes identities in source∩target for calibration, falls back to target-exclusive identities). Identity-based modes accept `grouping_strategy=PRESERVE_LOOKALIKES` (cluster_size lookalike package, computed via `ComparisonService.get_identity_confusion_matrix`). Advanced modes (ABSOLUTE, SDA, RATIO+LOOKALIKES) delegate to `PLDASplitService` via injection and require `EmbeddingDataset`; trivial modes still work on both `FaceDataset` and `EmbeddingDataset`. Convention for advanced modes: A = training (`plda_set`), B = calibration (`calib_set`). |
| `PAIRED_DATASET_SPLIT` | `PairedDatasetSplitInstructionExecutor` | Splits **two paired datasets simultaneously by identity** — same shuffle is applied to both. Inputs: `primary_dataset_resource_id` + `secondary_dataset_resource_id` (e.g. transformed frontal + transformed lateral views with shared identities). Outputs: 4 datasets — `output_primary_a/b` and `output_secondary_a/b`, where the B splits carry **identical identity sets** (guaranteeing genuine cross-pose pairs for calibration scoring). Only the **intersection** of identities between primary and secondary is the candidate pool for the shuffle; identities exclusive to one side fall into the A split of their own dataset (not discarded, but not paired). Reuses `PLDASplitService._select_ids` over the combined pool restricted to the intersection — supports `PRESERVE_LOOKALIKES` with `cluster_size`, plus `min_calibration_pairs/ids/samples_per_id` floors. Designed for cross-pose forensic strategies where the calibrator must train on the same pair distribution as casework (probe lateral × reference frontal). |
| `DATASET_MERGE` | `DatasetMergeInstructionExecutor` | Concatenates or updates datasets by index list |
| `PAIR_MINING` | `PairMiningInstructionExecutor` | Consumes a `SimilarityScoreDataset` (output of SIMILARITY_SCORING) **+ a `MINING_CONFIG` resource** referencing a `ComparisonMiningConfig`, and produces a curated `SimilarityScoreDataset`. The DTO no longer embeds the config inline — it carries `mining_config_resource_id` pointing at a project resource (typed `StrategyResourceType.MINING_CONFIG`), which is editable via the same ✎ button pattern used for PIPELINE_CONFIG. Independent rules for genuines and impostors (HARD/EASY/TYPICAL/RANDOM/MIXED with proportions; focus PAIRS or IDENTITIES; round-robin by pivot or global-order slicing). **Semantics of HARD/EASY are class-aware**: `_sort_pool_by_difficulty(pool, is_genuine_pool)` ensures `pool[0]` is always the epistemically hardest pair regardless of class — for impostors that means top-score (DESC sort), for genuines bottom-score (ASC sort). HARD always means "difficult to classify"; the numeric score position is just the implementation. **Without this normalization, applying HARD to genuines used to pick top-score (trivial same-person pairs), producing degenerate calibration (Cllr ~3.27 was empirically observed)**. Output is a single concatenated dataset (impostors first, then genuines) — class separation preserved per-sample via `same_person`/`label`. Empirical validation in cross-pose forensic case: **100% HARD impostors with `max_imp=20.000` + RANDOM genuines** with `ROUND_ROBIN_BY_PIVOT` reduced cal_loss from 0.107 to 0.033 (below the apparent Cllr_min=0.038 — the calibrator now contributes less informational loss than the discriminator). The varredura {10k, 20k, 25k, 30k, 40k, 50k} for max_imp revealed an **inverted-U at 20-25k** — below subsamples the decision boundary, above 25k the imp:gen ratio breaks `class_weight=balanced`. `from_dict` tolerates legacy field names (`mining_config` embedded, `comparison_dataset_resource_id`, `output_impostors_resource_id`, `output_genuines_resource_id`) — silently dropped. **Bug fix**: `exclude_self_comparisons` now treats empty `face_hash` as missing (post-parquet samples have `face_hash=""` because the column isn't persisted) — the strict equality check `"" == ""` was previously discarding 100% of samples. |
| `SIMILARITY_MODEL_TRAINING` | `SimilarityModelTrainingInstructionExecutor` | Prepares/trains a `CosineSimilarityModel` (stateless — no dataset/config) or `PLDASimilarityModel`. PLDA hyperparameters are embedded in the instruction itself via the nested `similarity_model_config: PLDASimilarityModelConfig` field — no external `*_config_resource_id` resource. Writes `TRAINED_SIMILARITY_MODEL` resource. **PLDA tem duas variantes**, controladas por `PLDASimilarityModelConfig.plda_variant: PLDAVariant`: `SIMPLE` (default, retrocompat) usa o backend SpeechBrain (formulação Ioffe simplificada com subspace `F` entre-classes — usa `em_iterations`, `plda_mode`, `latent_dim`); `TWO_COVARIANCE` usa implementação numpy pura em `faceaudit/services/two_covariance_plda_service.py` (formulação Brümmer & de Villiers 2010, padrão na literatura forense via FoCal toolkit) — covariâncias B e W completas D×D, treino em forma fechada (sem EM iterativo), scoring por log-LR vetorizado com kernels Q/P/const cacheados. `regularization: float ≥ 0` no config soma ε·I a B e W antes da inversão para estabilidade em alta dimensionalidade; `em_iterations`/`plda_mode`/`latent_dim` são IGNORADOS em TWO_COVARIANCE. `MODEL_VERSION=2` no pickle; payloads v1 (apenas `plda` SpeechBrain) carregam como SIMPLE automaticamente. |
| `SIMILARITY_SCORING` | `SimilarityScoringInstructionExecutor` | Scores all pairs (self-comparison or cross-comparison); writes `SimilarityScoreDataset` resource |
| `CALIBRATION_MODEL_TRAINING` | `CalibrationModelTrainingInstructionExecutor` | Trains `LogisticRegressionCalibrationModel` or `KDECalibrationModel` from a `SimilarityScoreDataset` or `{scores, labels}` dict. Hyperparameters are embedded inline via two nested fields (`logreg_config`, `kde_config`); only the one matching `calibration_model_type` is used by the executor — the other may stay populated for lossless type-toggling. **LogReg config supports GridSearch for `C`** (default `use_gridsearch=True`) via `c_grid`/`cv_folds`/`scoring: LogRegGridScoringMetric` (enum NEG_LOG_LOSS/ROC_AUC) — when active, fit uses `GridSearchCV` with `StratifiedKFold` (auto-reduces folds to class minority size); the chosen `C` is exposed as `model.best_C_` and emitted in the instruction's `metadata["best_C"]`. KDE config carries 13 fields (kernel, bandwidths, GridSearch toggle, Fisher epsilon, log-LR clipping, etc.) |
| `SCORE_CALIBRATION` | `ScoreCalibrationInstructionExecutor` | Applies a trained `CalibrationModelProtocol` to a `SimilarityScoreDataset` (raw_score) and produces a new dataset with per-pair `log_lr` populated in `sample.metadata["log_lr"]`. Output type is `StrategyResourceType.CALIBRATED_SCORES` (distinct from `SIMILARITY_SCORES`). Reuses the `SimilarityScoreDataset` Python class — `to_parquet` automatically detects `log_lr` in metadata and adds a native `log_lr: float64` column; the loader populates `metadata["log_lr"]` when the column exists (legacy parquets without the column continue to load unchanged). Vectorized via `calibrator.calibrate_scores(scores)`; preserves `raw_score`, `same_person`, `probe`/`reference` from input. No own hyperparameters — calibration is fully embedded in the trained model. |
| `STRATEGY_EVALUATION` | `StrategyEvaluationInstructionExecutor` | **Validation-only** instruction. Reads a `CALIBRATED_SCORES` resource (with `metadata["log_lr"]` populated per sample), separates `gen_log_lrs`/`imp_log_lrs` by `sample.label`, and delegates entirely to `ValidationMathService.calculate_metrics(strategy_id, strategy_name, strategy_type="V2", genuine_scores, impostor_scores)` — which uses `sklearn.isotonic.IsotonicRegression` for PAVA, computes EER + threshold, Cllr (numerically stable via `logaddexp`), Cllr_min, AUC, and ECE plot data via numpy broadcasting. Output is a `StrategyValidationResult` (with `ValidationMetrics` + `ValidationPlotData`) registered as `StrategyResourceType.VALIDATION_RESULT` and persisted as `.pkl` (preserves ndarray). No own hyperparameters — deterministic. **Validator rejects `is_casework=True` inputs** with explicit message — casework datasets have no ground truth and can't produce EER/Cllr/AUC. |
| `CASEWORK_VISUALIZATION` | `CaseworkVisualizationInstructionExecutor` | **Casework-only** instruction (in `_CASEWORK_ONLY_TYPES`, never appears outside the Análise de Caso phase). Compares the distribution of forensic case scores against a reference population (gen + imp from validation). Choice field `score_type` (CALIBRATED/RAW) drives dropdown filtering: CALIBRATED accepts `CALIBRATED_SCORES` in both reference and case dropdowns; RAW accepts `SIMILARITY_SCORES`. The choice triggers form rebuild automatically (choice fields always do). Validator enforces the semantic contract: reference must NOT be `is_casework=True` (needs gen+imp); case MUST be `is_casework=True`; CALIBRATED mode also requires `log_lr` populated. Executor extracts arrays separated by `same_person`, in RAW mode computes the EER threshold on-the-fly (FNR=FPR crossover via 1000-point linear sweep — `<10ms` for 100k pairs), and packages everything in a `CaseworkVisualizationResult` (numpy arrays + metadata + `case_summary`) registered as `StrategyResourceType.CASEWORK_VISUALIZATION` and persisted as `.pkl`. UI: button `📊 Visualizar Caso` in the InstructionConfigPanel opens `CaseworkVisualizationViewerDialog` (single-tab QDialog with `FigureCanvas` + `NavigationToolbar`) which renders the reference distributions (KDE + histogram, gen=green/imp=red) overlaid with the case: **KDE in blue when n_case ≥ 30**, otherwise **vertical dashed lines with annotated numeric values** (rotated 90° for readability). Reference line is at log-LR=0 (CALIBRATED) or at the computed EER threshold (RAW). |
| `DATASET_BALANCING` | `DatasetBalancingInstructionExecutor` | Balanceia o número de amostras por identidade — produz um novo `INTERMEDIATE_DATASET` preservando o input. Duas dimensões ortogonais. **`balancing_mode: DatasetBalancingMode`**: `CAP_ONLY` (limita a `target_n` por ID; IDs com menos amostras permanecem inalteradas), `FLOOR_AND_CAP` (cap + descarta IDs inteiras com < `target_n` amostras), `AUTO_MIN_BALANCE` (N = min observado em runtime — todas as IDs ficam com a mesma contagem; potencialmente lossy). **`selection_criterion: DatasetBalancingSelection`**: `RANDOM` (sorteio reprodutível via `random_seed`) ou `BY_QUALITY` (top-N por métrica, tie-breaker determinístico via SHA1 de `face_hash`/`file_name`). `quality_metric: QualityScoreType` aparece só quando BY_QUALITY — opções FIQA, OFIQ, SERFIQ ou EMBEDDING_NORM (nova; calculada on-the-fly via `np.linalg.norm` do embedding sepaelv2/AdaFace, não requer coluna no parquet). Direção consistente: maior = melhor para todas as métricas. Executor agnóstico ao tipo do payload (`FaceDataset` via `QualityScoreType.extract_from(face)` ou `EmbeddingDataset` via `item.quality_scores[metric.name.lower()]` + `np.linalg.norm` para norma). `target_n` é ocultado da UI quando `AUTO_MIN_BALANCE` está selecionado. Logs registram N efetivo, IDs descartadas no floor e contagem de fallbacks (samples sem métrica). |
| `HUBNESS_ANALYSIS` | `HubnessAnalysisInstructionExecutor` | Diagnóstico de viés topológico (hubness) em alta dimensionalidade — quantifica quais faces atraem vizinhos errados. Recebe um `EmbeddingDataset`, instancia `HubnessAnalysisService` (com `ComparisonService` injetado pela `StrategyExecutorFactory`), executa `compute_analysis` (O(N²)) + os 4 estudos derivados (stratification, paired_study, quality_diagnostics, optimization_study) e empacota tudo em um `HubnessReport` compósito (em `faceaudit/models/hubness_report.py`) persistido como pickle (preserva ndarrays). Hyperparâmetros: `k_neighbors`, `batch_size`, `chunk_size`, `paired_study_threshold_nf`. O service foi refatorado pra aceitar `logger_callback: Optional[Callable]` que substitui `print + tqdm` por progresso periódico via callback (a cada 10%) — preserva comportamento original quando callback é None. Os métodos `plot_stratified_hubness` e `plot_paired_*` esperam `analysis.stratification` e `analysis.paired_study` populados, então o executor injeta esses 2 estudos de volta no `HubnessAnalysisInfo` antes de empacotar (os outros 2 estudos vão direto pra fields do `HubnessReport`). Validador exige `EmbeddingDataset`, `k_neighbors < n_samples` e avisa quando ID count < 5. Saída: recurso `HUBNESS_REPORT`. |
| `HUBNESS_ABLATION` | `HubnessAblationInstructionExecutor` | Estudo de ablação topológica — recorta progressivamente o dataset (1, 2, 3 fotos por ID + total) seguindo critério escolhido via `ablation_mode: HubnessAblationMode` (`BY_QUANTITY` prioriza melhores fotos por FIQA; `BY_POSE` prioriza mais frontais por menor `abs(yaw)+abs(pitch)`) e mede o hubness em cada cenário. **Custoso:** refaz O(N²) em N subsets — use `HUBNESS_ANALYSIS` para diagnóstico geral barato e use `HUBNESS_ABLATION` apenas quando precisar do estudo formal de evolução do Nk. Saída: `AblationStudyInfo` registrado como recurso `HUBNESS_ABLATION_RESULT` (pickle). Validador avisa quando `BY_QUANTITY` é usado em dataset sem `quality_score`. UI: 3ª coluna do editor instancia `HubnessAblationWidget` (single-plot inline com KDEs sobrepostas dos cenários); botão 🔍 Expandir abre `HubnessAblationViewerDialog` (single-plot full-screen). |

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
| `faceaudit/services/` | Mixed — old arch: `FileSystemService`, `ConfigLoaderService`, etc.; new arch: `FaceAuditProjectService`, `StrategyResourceLoaderService`, `StrategyInferenceService`, `PipelineTrajectoryService`, `two_covariance_plda_service` |
| `faceaudit/protocols/` | Mixed — see protocol list above |
| `faceaudit/models/` | Mixed — subfolders `calibration/`, `instructions/`, `pipeline/`, `similarity/`, `storage/`, `strategy/`, `validation/` are new; root-level models are shared (see notes below) |
| `interface/tabs/strategy_manager_v2/` | **New architecture frontend** — see section below |
| `interface/` (rest) | Old architecture only — not yet migrated |

### Root-level models in `faceaudit/models/` — usage notes

Most root-level models are shared between both architectures. Key points for the new architecture:

- **Used by validation rules**: `dataset_stats.py` (`DatasetStats`) — DTO com `identity_set`, `n_ids`, `embedding_dim`, e `samples_per_identity: Dict[str, int]` (mapeamento identidade → contagem de amostras, populado nos 4 caminhos do `DatasetStatsService`). Property derivada `balance_metrics` retorna um `BalanceMetrics` (frozen dataclass) com `min/max/mean/median/std/cv/imbalance_ratio/verdict` calculados sob demanda — usado pelo `ResourceInfoWidget` para exibir o verdict de balanceamento e disponível para futuras validation rules. `DatasetStatsService.compute_from_embedding_dataset()` computa stats de `EmbeddingDataset` sem recarregar vetores pesados; results cached no contexto como `DATASET_STATS` resource via convenção `"{resource_id}_stats"`.
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
│   ├── strategy_editor_widget.py       ← QTabWidget with 3 tabs (Preparação/Validação/Caso), each with InstructionListWidget + InstructionConfigPanel + InstructionResultsPanel in horizontal splitter (3 columns ~200/400/600)
│   ├── instruction_list_widget.py      ← Drag-and-drop list with ▶ Run button + visual states (🔒/✅)
│   ├── instruction_config_panel.py     ← Dynamic form with type-filtered dropdowns via ResourceFieldSpec + Validate button. Constructor flag `outputs_in_results_panel: bool = False` (default preserves old layout); when True, the "Saídas" section and the 4 view buttons (📊) are hidden — they migrate to InstructionResultsPanel
│   ├── instruction_results_panel.py    ← 3rd column: outputs (OutputRowWidget) + inline plot (when applicable) + 🔍 Expandir button. Per-instruction _ResultsCard cache via QStackedWidget. Lazy hydrate button for big score datasets on disk
│   ├── result_widgets/                 ← Plot widgets reusable inline (results panel) and in dialogs (full-screen):
│   │   ├── score_distribution_widget.py     — histogram + KDE (gen vs imp), use_log_lr flag for SCORE_CALIBRATION
│   │   ├── validation_result_widget.py      — QTabWidget with 4 lazy tabs (LRs / Tippett / DET / ECE), header + help
│   │   ├── casework_visualization_widget.py — single plot, KDE for n_case≥30 or vertical lines for fewer
│   │   ├── hubness_report_widget.py         — QTabWidget with 5 lazy tabs (topology grid / stratification / paired / quality / density). Captures HubnessPlotService output via plt.show monkey-patch + PNG transplant.
│   │   └── hubness_ablation_widget.py       — single-plot KDE overlay of ablation scenarios (1/2/3 photos per ID + total). Same plt.show monkey-patch strategy as the report widget.
│   ├── validation_result_viewer_dialog.py    ← Thin wrapper around ValidationResultWidget (used by 🔍 Expandir)
│   ├── casework_visualization_viewer_dialog.py ← Thin wrapper around CaseworkVisualizationWidget
│   ├── hubness_report_viewer_dialog.py       ← Wrapper around HubnessReportWidget + extra "Galeria Completa" tab with 16-plot scrollable grid
│   ├── hubness_ablation_viewer_dialog.py     ← Thin wrapper around HubnessAblationWidget (single-plot full-screen)
│   ├── graph_viewer_dialog.py          ← Generic Figure dialog (used for trajectory 3D + score distribution from 🔍 Expandir)
│   ├── output_row_widget.py            ← Per-output row used by InstructionResultsPanel showing runtime state (⬜ placeholder / 🟡 in memory / 🟢 saved) and lazy summary of payload
│   └── nested_dto_widget.py            ← Inline editor for nested config DTOs (PLDA / LogReg / KDE / pipeline policies)
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

**Three-tab strategy editor com 3 colunas.** O `StrategyEditorWidget` usa um `QTabWidget` com três abas (Preparação, Validação, Análise de Caso). Cada aba tem um splitter horizontal de **três colunas**: esquerda = `InstructionListWidget` (lista de instruções com botão Run e estados visuais: locked/gray, runnable/normal, executed/green checkmark) — default ~200px; meio = `InstructionConfigPanel` (formulário dinâmico de configuração + botões "Aplicar alterações" / "Validar") — default ~400px; direita = `InstructionResultsPanel` (saídas com estado runtime + gráficos inline) — default ~600px (mais larga porque acomoda os gráficos). Tamanhos via `QSplitter`, redimensionáveis pelo usuário. O `InstructionConfigPanel` aceita uma flag `outputs_in_results_panel: bool = False` (default preserva layout antigo de 2 colunas) — quando o `StrategyEditorWidget` o instancia, passa `True` para que a seção "Saídas" e os botões de visualização (📊) fiquem ocultos no config panel, evitando duplicação com o painel de resultados. O antigo `ProducedResourcesWidget` por aba foi removido faz tempo — recursos produzidos pela estratégia aparecem globalmente no `LeftPanelWidget` (junto com os recursos de projeto). Estado runtime de cada output (⬜ placeholder / 🟡 em memória / 🟢 salvo em disco) é exibido por `OutputRowWidget` agora dentro do `InstructionResultsPanel`.

**Painel de resultados (3ª coluna) — `InstructionResultsPanel`.** Em `interface/tabs/strategy_manager_v2/strategy_editor/instruction_results_panel.py`. Renderiza, para a instrução selecionada na coluna de instruções: lista de outputs (`OutputRowWidget`) + gráfico inline (quando aplicável) + botão `🔍 Expandir` que abre o gráfico no dialog correspondente para visualização em tela cheia. Mapeamento de plot inline por tipo de instrução (via `_instruction_plot_kind`): `SIMILARITY_SCORING`/`PAIR_MINING`/`SCORE_CALIBRATION` → `ScoreDistributionWidget`; `STRATEGY_EVALUATION` → `ValidationResultWidget` (4 abas Lazy: distribuição de LRs, Tippett, DET, ECE); `CASEWORK_VISUALIZATION` → `CaseworkVisualizationWidget`; `PIPELINE_TRAINING`/`PIPELINE_APPLICATION` → **botão de trajetória 3D** (não inline — abre `GraphViewerDialog` por exigir espaço pra rotação/zoom). Demais instruções (DATASET_SPLIT, _MERGE, _BALANCING, PAIRED_DATASET_SPLIT, SIMILARITY_MODEL_TRAINING, CALIBRATION_MODEL_TRAINING) exibem só os outputs. O `InstructionResultsPanel` mantém um cache `_cards: Dict[instruction_id, _ResultsCard]` via `QStackedWidget` interno — ao trocar de instrução, o card já existente é reutilizado (gráficos cacheados); ao re-executar uma instrução, o controller V2 chama `invalidate_results_for_instruction(instruction_id)` que dropa o card e força redraw na próxima exibição. Para datasets grandes (`SIMILARITY_SCORES`/`CALIBRATED_SCORES`) ainda no disco após reabrir um projeto, o card mostra um botão `📊 Carregar e visualizar` em vez de hidratar automaticamente — evita travar a main thread com leituras de parquets de 5M+ pares; emite `hydrate_requested(instruction, resource_id)` que o controller V2 deve atender. Os widgets de plot reutilizáveis (extraídos dos dialogs antigos) ficam em `result_widgets/` — cada um aceita flags `show_header`/`show_toolbar`/`show_help` para alternar entre modo dialog (full) e modo inline (compact).

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
