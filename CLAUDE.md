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

## Architecture

The codebase uses a **declarative strategy engine**: a strategy is described as a `LRStrategyDefinition` (list of resources + ordered instructions). The execution engine interprets and runs it — the user defines what happens, not the code.

### Core execution flow

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
                     ├─ validation_rule_configs (pluggable rules instanciadas pela factory)
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

### Folder map

| Folder | Role |
|---|---|
| `faceaudit/contexts/` | `StrategyExecutionContext` — the shared state bus (resources, artifacts, logs, metadata) |
| `faceaudit/executors/` | `StrategyExecutor` (orchestrator) + `InstructionExecutorRegistry` (type→executor map) |
| `faceaudit/executors/instructions/` | One concrete executor per instruction type, all extending `StrategyInstructionExecutorBase` |
| `faceaudit/factories/` | `StrategyExecutorFactory` (single assembly point for the engine) + `ValidationRuleFactory` (instancia regras a partir de `ValidationRuleConfig`) |
| `faceaudit/repositories/` | `StrategyDefinitionRepository` (JSON) + `StrategyTrainedModelRepository` (pickle) — ambos aceitam `ArtifactStorageProtocol` |
| `faceaudit/validation_rules/` | Implementações concretas de `ValidationRuleProtocol`, todas estendendo `ValidationRuleBase`; 8 regras: `DatasetIsolationRule`, `DatasetOverlapRequiredRule`, `FisherRule`, `MinCalibrationPairsRule`, `MinIdentitiesWithMultipleSamplesRule`, `HoldoutLeakageRule`, `DualPipelineDimensionCompatibilityRule`, `DualPipelineSubsequenceRule` |
| `faceaudit/storage/` | `LocalProjectArtifactStorage` (projeto autocontido) + `GlobalLibraryStorage` (biblioteca global) — ambos implementam `ArtifactStorageProtocol`. `LocalProjectArtifactStorage` expõe também `verify_integrity(refs)` fora do protocolo. |
| `faceaudit/services/project_service.py` | `FaceAuditProjectService` — cria/abre projetos (`.fa`), agrega `LocalProjectArtifactStorage` + repositórios; `open(verify_integrity=True)` executa `check_integrity()` automaticamente. `FaceAuditProject` expõe `integrity_results`, `has_integrity_problems` e `check_integrity()`. |
| `faceaudit/services/strategy_resource_loader_service.py` | `StrategyResourceLoaderService` — hidrata `payload` de `StrategyResource` a partir de `artifact_ref` SOB DEMANDA. Idempotente. Loaders concretos (`_load_pickle`, `_load_dataset`, `_load_pipeline_config`, `_load_similarity_scores_from_parquet`, `_load_mining_config`) preenchem `self._last_error` em caso de falha (consultado por `StrategyExecutionContext.ensure_payload` para expor a exceção real no LogProgressDialog). |
| `faceaudit/services/strategy_inference_service.py` | `StrategyInferenceService` — inferência forense: recebe `LRStrategyTrainedModel` + `StrategyResource` probe/reference + `ArtifactStorageProtocol`; executa pipeline → `score_matrix()` → `calibrate_scores()`; retorna `SimilarityScoreDataset` com `raw_score` e `metadata["log_lr"]` por par |
| `faceaudit/services/pipeline_trajectory_service.py` | `PipelineTrajectoryService` — visualização 3D de trajetória |
| `faceaudit/services/two_covariance_plda_service.py` | Implementação numpy pura do PLDA de duas covariâncias (Brümmer & de Villiers 2010). Usado pelo `PLDASimilarityModel` quando `config.plda_variant=TWO_COVARIANCE`. |
| `faceaudit/services/cross_condition_two_covariance_plda_service.py` | `fit_cross_condition_two_covariance(X_a,y_a,X_b,y_b,...)` — estimador PLDA two-covariance com peso igual por condição. Disparado quando `SimilarityModelTrainingInstruction.secondary_dataset_resource_id` é definido em variantes TWO_COVARIANCE / SHRINKAGE_TWO_COVARIANCE (SIMPLE + secondary → erro). Dois modos via `CrossConditionWithinMode`: `PER_CONDITION` (resíduos centrados em `μ_c^cond`) e `BALANCED_IDENTITY` (default — `μ_c=(μ_c^A+μ_c^B)/2`, mais conservador forensicamente). Singletons entram em B/μ via μ_c balanceada; não contribuem como linha em W_cond. `W = (W_A+W_B)/2`, `B = (1/C) Σ_c (μ_c−μ)(μ_c−μ)ᵀ` — normalizações divergem por design do `fit_two_covariance` (peso igual por condição/classe; W_cond usa `S_w_cond/N_residuals_cond`). Em LEDOIT_WOLF, `α = (α_A+α_B)/2` per-condição. Scoring permanece via kernels Q/P/const do `two_covariance_plda_service`. State retorna em `TwoCovariancePLDAState` com `cross_condition=True`, `n_ids_condition_a/b`, `n_common_identities`, `cross_condition_within_mode` — todos os novos campos são opcionais; UI deve acessar via `getattr` para compat com pickles antigos. |
| `faceaudit/services/hubness_analysis_service.py` | `HubnessAnalysisService` — diagnóstico de viés topológico em alta dimensionalidade. Aceita `logger_callback` para progresso periódico via callback. |
| `faceaudit/services/hubness_plot_service.py` | `HubnessPlotService` — 20 métodos matplotlib/seaborn. O `HubnessReportWidget` reutiliza via monkey-patch local de `plt.show` + captura via PNG buffer. |
| `faceaudit/services/quality_metric_availability_service.py` | `QualityMetricAvailabilityService` — descobre quais `QualityScoreType` (FIQA, OFIQ, SERFIQ, EMBEDDING_NORM) estão disponíveis num dataset sem hidratar o payload inteiro. |
| `faceaudit/services/typicality_model.py` | `TypicalityModel` — modelo de tipicidade paramétrico MoVMF (mistura de von Mises-Fisher) com 3 modos de centralização (`none`, `frechet`, `euclidean_legacy`), EM com restarts e patience, OOD detection via quantile de log-likelihood, AIC/BIC com DOF correto (`K·D − 1`, considerando que μ ∈ S^{D−1}), `from_dict` v1→v2 com migração automática de `center_embeddings` (bool) → `centering` (str). Tier 2 de defensibilidade forense — exige justificar K, κ, μ, centering em corte. |
| `faceaudit/services/knn_typicality_model.py` | `KNNTypicalityModel` — modelo de tipicidade NÃO-paramétrico baseado em k-NN esférico. Tier 1 de defensibilidade forense — não há suposição distribucional, hubness vira sinal (densidade local), defensível como estatística de ranking ("X é mais típico que 73% da população de referência"). Três scores por face: `mean_knn_similarity` (T_k), `typicality_percentile` (0..100), `log_density_knn` (Loftsgaarden-Quesenberry esférico). Self-match handling via top-(k+1) + descarte do ≈1.0. Persiste como pickle (referência N×D em float32 + CDF empírica LOO). |
| `faceaudit/transformators/` | 6 transformators concretos (`CoralTransformation`, `WhiteningTransformation`, `LdaTransformation`, `CenteringTransformation`, `L2Normalization`, `IdentityAggregation`) que implementam `EmbeddingTransformationProtocol`, recebem config DTOs via constructor injection, delegam math ao `EmbeddingMathService`, e expõem `get_state()`/`set_state()` para sincronização DUAL. |
| `faceaudit/pipelines/` | `EmbeddingTransformationPipeline` (LINEAR + DUAL modes, `PipelineTrainingPolicy`). |
| `faceaudit/protocols/` | Structural typing contracts — see section below |
| `faceaudit/models/calibration/` | Calibration model configs + implementations (LogReg, KDE). Inclui `LogRegS0Config` — DTO leve usado como `s0_logreg_config_override` opcional em `SCORE_INTERACTION_FEATURE` (calibrador auxiliar univariado de s0). Defaults: `C=1.0, solver=liblinear, class_weight=balanced, max_iter=1000`. |
| `faceaudit/models/instructions/` | Instruction DTOs — one per instruction type, inherit `StrategyInstructionDefinition` |
| `faceaudit/models/pipeline/` | `PipelineTrainingPolicy`, `PipelineApplicationPolicy` |
| `faceaudit/models/similarity/` | Similarity model configs + implementations (Cosine, PLDA) + `SimilarityScoreDataset` |
| `faceaudit/models/storage/` | `StrategyArtifactReference` — referência lógica com `path` (sempre relativo) + `hash` SHA-256. `ArtifactIntegrityResult` — frozen dataclass por artefato verificado. |
| `faceaudit/models/strategy/` | `StrategyResource`, `LRStrategyDefinition`, `LRStrategyTrainedModel` |
| `faceaudit/models/validation/` | `ValidationResult` (usado por toda validation), `ValidationRuleConfig` (DTO serializável das regras plugáveis) |
| `faceaudit/enums/artifact_integrity_status.py` | `ArtifactIntegrityStatus` — `OK`, `MODIFIED`, `MISSING`, `HASH_UNAVAILABLE`; `is_problem` True apenas para MODIFIED e MISSING |
| `faceaudit/enums/cross_condition_within_mode.py` | `CrossConditionWithinMode` — `PER_CONDITION` e `BALANCED_IDENTITY` (default). Controla o ponto de centragem dos resíduos no PLDA cross-condition (`fit_cross_condition_two_covariance`). Só tem efeito quando há `secondary_dataset_resource_id`. |
| `faceaudit/enums/score_interaction_mode.py` | `ScoreInteractionMode` — `ONE_SIDED_POSITIVE` (default), `LINEAR`, `HINGE_BASE`. Define a fórmula da feature de interação produzida por `SCORE_INTERACTION_FEATURE`. `requires_modifier` property indica se o modo lê a coluna de raridade (False só em `HINGE_BASE`). |
| `faceaudit/enums/s0_mode.py` | `S0Mode` — `AUTO_LOGREG` (default), `MANUAL`, `INHERIT`. Como `SCORE_INTERACTION_FEATURE` resolve `s0` (ponto de evidência neutra logLR=0). AUTO_LOGREG é o calibrador auxiliar univariado; INHERIT é obrigatório em validação/casework para preservar a definição de evidência neutra. |
| `faceaudit/enums/validation_rule_type.py` | `ValidationRuleType` — enum dos 8 tipos de regra plugáveis (FISHER, MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES, MIN_CALIBRATION_PAIRS, DATASET_ISOLATION, DATASET_OVERLAP_REQUIRED, HOLDOUT_LEAKAGE, DUAL_PIPELINE_DIM_COMPAT, DUAL_PIPELINE_SUBSEQUENCE) |

### Protocols

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

### Key design decisions

**`LRStrategyDefinition` organizes instructions in three phases.** `preparation_instructions` (pipeline training, model training), `validation_instructions` (metrics, scoring), `casework_instructions` (case analysis). The read-only property `instructions` concatenates all three for backward compatibility with validators, protocols, and `resource_info_widget`. Helper `_get_phase_list(phase)` returns the list for a given phase key.

**`StrategyResource` is the unit of exchange between instructions.** Every instruction reads named resources from the context and writes named resources back. The names are declared in the instruction DTO itself (`input_resource_ids`, `output_resource_ids`), kept in sync via `_sync_declared_io()` in `__post_init__`. Two categories exist: **project resources** (DATASET, SPLIT_CONFIG, MINING_CONFIG, PIPELINE_CONFIG — imported by the user, shared across strategies, always have `artifact_ref`) and **strategy-produced resources** (TRAINED_PIPELINE, INTERMEDIATE_DATASET, TRAINED_SIMILARITY_MODEL, TRAINED_CALIBRATION_MODEL, TRAINED_TYPICALITY_MODEL, SIMILARITY_SCORES, CALIBRATED_SCORES, VALIDATION_RESULT — produced by instruction execution). Hyperparameter configs for similarity/calibration models live embedded in their instruction DTOs (`similarity_model_config`, `logreg_config`, `kde_config`). Strategy-produced resources start as **placeholders** (payload=None, artifact_ref=None) when the user fills in output fields. After execution, they are persisted to disk and gain `artifact_ref`. `StrategyResourceType.to_artifact_type()` maps to the physical `StrategyArtifactType` for persistence. `StrategyInstructionDefinition.get_output_resource_placeholders()` creates placeholder `StrategyResource` objects from `ResourceFieldSpec` entries with `is_output=True`. The controller calls `_sync_all_output_placeholders()` on each strategy modification.

**`ResourceFieldSpec` enables type-filtered resource dropdowns.** Each instruction DTO implements `get_resource_field_specs()` returning a list of `ResourceFieldSpec(field_name, label, accepted_types, is_output)`. The UI config panel uses these specs to show only resources of the correct `StrategyResourceType` in each dropdown. It's an instance method (not classmethod) because Pipeline Training/Application specs vary by `pipeline_mode` (LINEAR vs DUAL).

**`StrategyInstructionDefinition.from_dict()` is a polymorphic dispatcher.** It reads `instruction_type` and hydrates the correct concrete DTO. This is how the strategy survives JSON serialization/deserialization with full type information.

**Similarity and calibration are separated.** `SimilarityModelProtocol` produces raw scores; `CalibrationModelProtocol` converts raw scores to LRs. They connect via `SimilarityScoreDataset`. `KDECalibrationModel` é um calibrador only — recebe scores pré-computados e ajusta duas KDEs (genuíno/impostor) com Fisher transform + GridSearch adaptativo de bandwidth.

**`PipelineTrainingPolicy`** controla o comportamento do modo DUAL: quais steps (`CORAL`, `Whitening`, `LDA`) podem usar o dataset target durante o treino e se shared state é copiado para o branch target.

**Filtro de instruções por fase (UI).** O menu "➕ Adicionar" da `InstructionListWidget` filtra os tipos disponíveis pela fase da aba: `_TRAINING_ONLY_TYPES = {PIPELINE_TRAINING, SIMILARITY_MODEL_TRAINING, CALIBRATION_MODEL_TRAINING}` só na Preparação; `_VALIDATION_ONLY_TYPES = {STRATEGY_EVALUATION}` só na Validação; `_CASEWORK_ONLY_TYPES = {CASEWORK_VISUALIZATION}` só na Análise de Caso.

**Visualização de resultados forenses (`ValidationResultViewerDialog`).** A instrução `STRATEGY_EVALUATION` produz um `StrategyValidationResult`. O `InstructionConfigPanel` mostra um botão `📊 Visualizar Resultado de Validação`; clique abre um `QDialog` 1100×780 com header âmbar + `QTabWidget` de **7 abas de plot + 1 aba de Métricas**: **Distribuição de LRs**, **Tippett Plot**, **Curva DET** (escala normit com marca no EER), **Calibração ECE** (curva real + curva mínima PAVA in-sample com área de gap diagnóstico), **Curva de Calibração** (2-panel: scatter raw×logLR + pós-afim Brümmer + Δ resíduo), **Diagnóstico Afim em Escala log-LR** (3 camadas: y=x sem recalibração + pós-afim + PAVA oracle in-sample), **Quantis por Classe** (2-panel: percentis P1..P99 per-classe com halos críticos + bottom panel de Contribuição ao Cllr por bin). Cada aba tem `FigureCanvas` + `NavigationToolbar`. Lazy hydrate via `_resource_loader.hydrate` quando o payload está no disco.

**Terminologia PAVA (Morrison 2021).** Toda a UI/help/tooltips foi reframed: `Cllr_min` é "Cllr após recalibração monotônica PAVA ajustada no PRÓPRIO conjunto avaliado" — referência diagnóstica **in-sample**, NÃO medida operacional de calibração nem teto teórico real. `cal_loss = Cllr − Cllr_min` é "gap em relação à PAVA in-sample" — útil em desenvolvimento e comparação inter-estratégias, não deve ser apresentado isoladamente como grau de calibração em laudo. PAVA é chamada de "oracle in-sample; não operacional" nos labels de plot (Diagnóstico Afim e ECE). O subtítulo do Diagnóstico Afim mostra `pava_residual_gap = Cllr_affine − Cllr_min` (renomeação só de string visível; campo Python `nonlinear_cal_loss` em `ValidationMetrics.affine_cal_loss` preservado). A property `is_well_calibrated` (heurística `cal_loss < 0.1`) está marcada DEPRECATED e a linha equivalente foi REMOVIDA da tabela de métricas — Morrison critica explicitamente esse julgamento binário in-sample. A defensibilidade forense da calibração depende sobretudo do modelo de calibração escolhido (LogReg como default parcimonioso) e da representatividade dos dados de calibração, não das métricas PAVA in-sample.

**Visualização forense de caso (`CaseworkVisualizationViewerDialog`).** A instrução `CASEWORK_VISUALIZATION` produz um `CaseworkVisualizationResult` com `score_type` (RAW/CALIBRATED), arrays `genuine_scores`/`impostor_scores`/`case_scores`, `eer_threshold` opcional (computado on-the-fly em modo RAW via FNR=FPR crossover), e `case_summary` pré-computado. Plot: KDE+histograma de gen/imp como referência, sobreposto pelo caso (KDE azul quando `n_case ≥ 30`, ou linhas verticais tracejadas anotadas com valor numérico quando `n_case < 30`).

**Interação cosseno × raridade (`SCORE_INTERACTION_FEATURE`) — s0 congelado e double-counting.** A feature materializa a tese forense de que **raridade NÃO é evidência independente** — ela só modula a *força* de uma similaridade observada. A fórmula `(base − s0)·rarity` exige que `s0` (ponto de evidência neutra, logLR=0) seja **aprendido no domínio de calibração e congelado** para validação/casework. Refit silencioso de `s0` em validação ou casework recalibra a definição de "evidência neutra" e contamina o LR — por isso `S0Mode.AUTO_LOGREG` é **bloqueado em casework** e o protocolo defensável usa `S0Mode.INHERIT` (lê `interaction_s0` do metadata do dataset de calibração). O executor emite WARNING quando o `base_score_type`/`base_similarity_model_resource_id`/`base_embedding_column` herdados divergem do base atual. Em AUTO_LOGREG sobre o mesmo dataset, métricas de **CV interno do calibrador final são otimistas** (s0 estimado sobre todo o conjunto vs cada fold) — usar holdout externo com INHERIT para avaliação honesta. Inversão typicality→rarity é **cosmética no caso bilateral** (termo `C·(base−s0)` absorvido por `{1, base}`) mas **não-cosmética no unilateral com offset não-nulo** (ex.: percentil onde `C=1`); `HINGE_BASE` como feature-companheira torna o modelo `[base, HINGE_BASE, HINGE_BASE·rarity]` plenamente geral, invariante à definição exata de raridade. **Double-counting com PLDA**: B-covariance do PLDA já encoda tipicidade — combinar `PLDA + interação` exige A/B explícito (compare modelos 8 e 9 da janela de Comparação). A feature usa **alinhamento orientado `(probe_hash, reference_hash)`** quando o modifier vem de combiner unilateral (`REFERENCE_ONLY`/`PROBE_ONLY`); senão `frozenset({probe, ref})` (compat simétrico). Detecta chaves duplicadas e valida `same_person` sample-a-sample em modo não-casework — silenciamentos perigosos por sobrescrita em dict ou splits cruzados são explicitamente rejeitados.

**Edição inline de configs editáveis (botão ✎ no `InstructionConfigPanel`).** Dropdowns que selecionam recursos do tipo `PIPELINE_CONFIG` ou `MINING_CONFIG` ganham um botão `✎` à direita do combo. O controller V2 (`_on_resource_edit_requested`) despacha por `resource.resource_type`: PIPELINE_CONFIG → `PipelineCreatorController(existing_config=..., lock_name=True)`; MINING_CONFIG → `NewMiningConfigDialogController(parent=..., lock_name=True) + ctrl.load_config(...)`. Helpers compartilhados: `_hydrate_resource_if_needed` (lazy-load do payload) e `_persist_edited_config_resource` (atualiza payload em memória + contexto vivo, re-persiste sobre o mesmo path preservando o nome físico do arquivo, propaga novo `artifact_ref`, **e dispara cascade transitivo** via `_invalidate_dependents_of`).

**`lock_name=True` em editores de config inline.** Tanto `PipelineCreatorController` quanto `NewMiningConfigDialogController` aceitam `lock_name: bool = False`. Quando `True`: `input_name` recebe `setReadOnly(True)` + tooltip + stylesheet cinza. Razão: o `resource_id` exibido do recurso é o stem do arquivo no disco (`Path(ref.path).stem`); editar o `config.name` interno **não** muda o nome do arquivo nem o `resource_id` mostrado nos dropdowns/árvore. **Sempre passar `lock_name=True`** quando o dialog é aberto via botão ✎.

**`ValueFieldSpec.triggers_rebuild` para campos que alteram outros specs.** O dataclass `ValueFieldSpec` tem `triggers_rebuild: bool = False`. Quando `True`, o `InstructionConfigPanel` conecta o widget ao slot `_on_value_triggers_rebuild`, que aplica o valor ao DTO, chama `_sync_declared_io()` e dispara rebuild. Caso canônico: `PipelineTrainingInstruction.apply_to_training_datasets`.

**`ResourceFieldSpec.triggers_rebuild` análogo para dropdowns de recurso.** Útil quando outros specs dependem do recurso escolhido. Caso canônico: `DatasetBalancingInstruction.dataset_resource_id` é marcado `triggers_rebuild=True` porque o `quality_metric` usa `dynamic_choices` que filtra as métricas pelo dataset escolhido.

**`ChoiceFieldSpec.dynamic_choices` para opções filtradas em runtime.** Hook genérico: callable `(available_resources) → Optional[Tuple[str, ...]]`. **Caso canônico**: `DatasetBalancingInstruction._available_quality_metric_choices` consulta `QualityMetricAvailabilityService` e retorna apenas as métricas com cobertura ≥ 50% no dataset selecionado.

**`get_hidden_fields()` + `REACTIVE_FIELDS` em DTOs aninhados (PLDA).** O `NestedDTOInlineWidget` suporta visibilidade dinâmica de fields via hooks no DTO inline: método `get_hidden_fields() → Set[str]` e atributo `REACTIVE_FIELDS: Tuple[str, ...]`. **Caso canônico**: `PLDASimilarityModelConfig` define `REACTIVE_FIELDS = ("plda_variant",)` e `get_hidden_fields()` retorna `{"regularization"}` em SIMPLE e `{"em_iterations", "plda_mode", "latent_dim"}` em TWO_COVARIANCE.

**Recursos do projeto: PIPELINE_CONFIG e MINING_CONFIG.** Ambos são recursos de PROJETO (vivem no disco em `pipelines/*.json` e `mining_configs/*.json`) e **não estão necessariamente declarados em `LRStrategyDefinition.resources`** — só aparecem nos dropdowns via `_collect_project_resources()` (combina recursos de definitions + scan de disco). Para edição/lookup, sempre usar `_collect_project_resources()`.

**Detalhes de modelos no `ResourceInfoWidget`.** O painel de propriedades do recurso tem seções dedicadas, visíveis apenas para os tipos correspondentes: **Detalhes do Modelo de Similaridade** (Cosine/PLDA — tipo, dim. entrada/latente, EM iters, variante PLDA), **Detalhes do Modelo de Calibração** (LogReg: `C (efetivo)` lido de `best_C_` pós-GridSearch; KDE: kernel, bandwidths, clipping log-LR), **Detalhes do Score Dataset** (`Total de pares`, `Genuínos (Hp)`, `Impostores (Hd)`, `Identidades`, `Calibrados (log-LR)`). Lazy-load via `_load_pickle(abs_path)`.

**Métricas de balanceamento de identidade no `ResourceInfoWidget` (DATASET stats).** Linhas adicionais alimentadas por `DatasetStats.balance_metrics`: **`Balanceamento`** mostra verdict (`✅ Balanceado`, `🟢 Aceitável`, `🟡 Moderadamente desbalanceado`, `🔴 Severamente desbalanceado`); **`Amostras/ID`** mostra `min–max (média, mediana)`. Verdict por **worst-of-two** entre CV (std/mean) e imbalance ratio (max/min). Alimentado por `DatasetStats.samples_per_identity: Dict[str, int]`.

**Stale-cache check via mtime nos sidecars de stats.** Tanto `_load_scores_details` quanto `_load_stats` comparam `parquet.mtime` com `cache.mtime` antes de usar o sidecar. Se `parquet.mtime > cache.mtime + 1.0`, o cache é invalidado e recomputado. Necessário porque re-execução de instruções reescreve o parquet sobre o mesmo path.

**Tuning anti-freeze no Windows (`interface/run.py` topo).** Antes de qualquer import pesado: `sys.setswitchinterval(0.001)` reduz o intervalo de troca da GIL de 5ms para 1ms; e em `sys.platform=="win32"`, `ctypes.windll.user32.DisableProcessWindowsGhosting()` desabilita o "ghost window" do Windows. `HashService.calculate_file_hash` usa `hashlib.file_digest` em Python ≥ 3.11.

**Light palette forçada no startup (`interface/run.py:_apply_light_palette`).** Vários `setStyleSheet(...)` têm hex codes fixos que assumem fundo claro. Solução: imediatamente após `setStyle("Fusion")`, `_apply_light_palette(app)` define todos os `QPalette.ColorRole` com cores claras. Trava a app em modo claro independente do tema do SO. **Workaround temporário.**

**Performance: serialização de `StrategyResource` NÃO recursa no `payload`.** `StrategyResource.to_dict()` **NÃO chama `super().to_dict()`** do `JSONSerializableMixin`. O mixin usaria `dataclasses.asdict(self)` que recursa em TODOS os campos — incluindo `payload`. Para um `SimilarityScoreDataset` com 5M+ samples, isso alocaria ~70M+ objetos Python por chamada. `to_dict()` constrói o dict manualmente: `{resource_id, name, resource_type, artifact_ref}`. **Não reintroduzir `super().to_dict()` aqui sob hipótese alguma.**

**Performance: caching de contagens em `SimilarityScoreDataset`.** As properties `total_genuines` e `total_impostors` são cachadas em `self.__dict__` (`_cached_n_gen`/`_cached_n_imp`) via helper `_ensure_class_counts_cached()`. O método `add()` invalida o cache.

**Performance: summarizers da UI usam amostragem e leem `items[0]`, não recomputam matrizes.** `_summarize_dataset` lê `embedding_dim` de `items[0].embedding.shape[0]` (NÃO chama `payload.matrix` que faria `np.vstack` de TODOS os embeddings). Summarizers com listas grandes (≥50k items) usam **amostragem aleatória** em vez de iterar em milhões de samples a cada refresh.

**Performance: executores iteram em chunks com micro-pausa.** `ScoreCalibrationInstructionExecutor` e `StrategyEvaluationInstructionExecutor` iteram `score_dataset.items` em **chunks de 50.000** com `time.sleep(0)` entre eles para liberar a GIL. `ScoreCalibrationInstructionExecutor` **muta `sample.metadata["log_lr"]` in-place**.

**Performance: `to_parquet` chunkado com `gc.disable`.** `SimilarityScoreDataset.to_parquet` escreve em chunks de 25k linhas com `time.sleep(0.005)` entre eles e desabilita o coletor cíclico do GC durante a escrita. `progress_callback` opcional. Reabilita o GC no `finally`.

**Performance: telemetria em `LocalProjectArtifactStorage.save_artifact`.** Cada chamada emite via `progress_callback`: `⏱  Persistência '{relative_path}': {file_mb:.1f} MB | write={t:.2f}s, hash={t:.2f}s, total={t:.2f}s`.

**Performance: rotação de `execution_logs` no contexto.** `StrategyExecutionContext.log()` mantém no máximo `_MAX_EXECUTION_LOGS = 5000` mensagens (FIFO).

**Identificação da estratégia no contexto.** `StrategyExecutor.prepare_context` expõe `strategy_id` e `strategy_name` em `context.execution_metadata` via `context.set_metadata`.

**`execute()` aceita `additional_resources` para Run All.** Por padrão, `StrategyExecutor.execute(definition)` constrói o contexto via `prepare_context(definition)` que só registra `definition.resources`. Mas as instruções referenciam recursos de PROJETO (DATASET, PIPELINE_CONFIG, MINING_CONFIG) que **não estão declarados em `definition.resources`**. Tanto `execute()` quanto `prepare_context()` aceitam `additional_resources: Optional[List[StrategyResource]] = None`; o controller V2 passa `self._collect_project_resources()`. Execução per-instrução usa caminho separado (`_register_project_resources_in_context`).

**`StrategyExecutionWorker._persist_trained_model_resources_inline` pula recursos com `artifact_ref` preexistente.** Após Run All, o `LRStrategyTrainedModel.trained_resources` inclui também os recursos de projeto. Skip se `resource.artifact_ref is not None` — recurso já existe em disco e não precisa ser regravado.

**Hidratação preguiçosa de recursos.** `StrategyExecutor.prepare_context` apenas registra os `StrategyResource` no contexto via `set_resource()` — **não chama `loader.hydrate`**. Hidratação acontece sob demanda em quatro pontos: (1) `StrategyExecutor.execute_single_instruction` chama `_rehydrate_inputs_if_needed`; (2) `StrategyInstructionExecutorBase.validate()` chama `context.ensure_payload(rid)`; (3) `ValidationRuleBase._get_or_compute_stats` chama `context.ensure_payload(rid)`; (4) UI controllers chamam `loader.hydrate` direto. O `StrategyExecutionContext` tem `resource_loader` e `embedding_column` (preenchidos por `prepare_context`, excluídos do `to_dict`) e o método `ensure_payload(resource_id)`.

**Cache invalidation em `StrategyExecutionContext.set_resource`.** O contexto cacheia `DatasetStats` como recurso com id `"{rid}_stats"`. Quando `set_resource(resource)` é chamado, o método automaticamente **invalida o cache `{rid}_stats`** via `self.resources.pop(stats_id, None)`. Sem isso, ao re-executar PIPELINE_APPLICATION (que sobrescreve um dataset com novo conteúdo sob o mesmo id), regras como Fisher e MinIdentitiesWithMultipleSamples continuariam lendo stats antigos. Safeguard: se o recurso registrado já termina em `_stats`, o método não tenta invalidar — evita popping inútil.

**`ValidationResult`** flows through everything. Uses `merge()` to aggregate results from multiple rules. Four factories: `ok()`, `neutral()`, `critical(message)`, `warning(message)`.

### Sistema de validação — plugável + dinâmica + executor-specific

A validação ocorre em dois pontos com estágios bem definidos.

**Nível 1 — Validação da estratégia** (antes de qualquer instrução, em `StrategyExecutor._validate_strategy()`):

| Estágio | Onde | O que valida |
|---|---|---|
| Strategy Rules | `LRStrategyDefinition.validation_rule_ids` | Regras globais da estratégia (ex: `StrategyEmbeddingConsistencyRule` always-on do factory). Executam uma vez antes de qualquer fase. |

**Nível 2 — Validação de instrução** (em `StrategyInstructionExecutorBase.validate()`):

| Estágio | Onde | O que valida |
|---|---|---|
| 1. Structural | `validate()` — topo | Null checks em `instruction` e `context`. Falha aqui aborta os demais. |
| 2. Pluggable | `instruction.validation_rule_configs` | Regras instanciadas a partir de `ValidationRuleConfig` via `ValidationRuleFactory`. Configs auto-contidas (rule_type + params) — não dependem de pré-registro no contexto. |
| 3. Dynamic | `instruction.build_validation_rules()` | Regras construídas pelo DTO com base no estado atual dos campos. Ex: `PipelineTrainingInstruction` DUAL retorna 3 regras (overlap, dimensões, subsequência). |
| 4. Executor | `_validate_instruction_specifics()` | Hard-coded no executor concreto: tipo da instrução, campos obrigatórios, payloads corretos, recursos no contexto. |

**Gate de dependência** (separado do pipeline): `StrategyExecutor.can_execute_instruction()` verifica se todos os `input_resource_ids` estão presentes no contexto.

#### Pluggable validation rules — Phase 1 + 2

**Modelo de persistência**: cada instrução carrega uma lista `validation_rule_configs: List[ValidationRuleConfig]` (DTO em `faceaudit/models/validation/validation_rule_config.py`) com:
- `rule_type: ValidationRuleType` — enum identifica a classe concreta.
- `params: Dict[str, Any]` — parâmetros de construção (dataset_resource_id, min_genuines, safety_margin_ratio, etc.).

A `ValidationRuleFactory.create(config, instruction_id)` (em `faceaudit/factories/validation_rule_factory.py`) instancia a regra concreta em runtime — não dependemos de pré-registro no contexto. Suporta os **19 tipos** definidos em `ValidationRuleType`.

**Quando usar `_ensure_default_validation_rules` vs `build_validation_rules`**:
- **Persistente (`_ensure_default_validation_rules`)**: regras com thresholds configuráveis que o usuário pode querer ajustar/desabilitar (MIN_IDENTITIES, MIN_SAMPLES, MIN_CALIBRATION_PAIRS, FISHER, QUALITY_METRIC_AVAILABLE, MAX_SAMPLES_WARNING, etc.). Vão em `instruction.validation_rule_configs`, salvas em JSON, deletáveis na UI (🗑).
- **Dinâmica (`build_validation_rules`)**: regras intrínsecas que viram sem-sentido se desabilitadas (MODEL_FITTED, EMBEDDING_DIM_MATCHES_MODEL, SCORE_CALIBRATION_STATE, CASEWORK_STATE, SCORE_TYPE_MATCH, DATASETS_EMBEDDING_COMPAT, DUAL_PIPELINE_*). Construídas em runtime a cada validação, NÃO persistidas, NÃO removíveis pelo usuário.

**Defaults persistentes por DTO** (chamados de `__post_init__` e `_sync_declared_io`):

| DTO | Defaults populados |
|---|---|
| `SimilarityModelTrainingInstruction` (PLDA + dataset definido) | `FISHER` + `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES` |
| `CalibrationModelTrainingInstruction` (input definido) | `MIN_CALIBRATION_PAIRS` + `SCORE_CALIBRATION_STATE(RAW)` |
| `DatasetSplitInstruction` BY_IDENTITY_* | `MIN_IDENTITIES` |
| `DatasetSplitInstruction` BY_INDEX_* | `MIN_SAMPLES` |
| `DatasetSplitInstruction` BY_SUPERVISED_ADAPTATION | `MIN_IDENTITIES(source)` + `MIN_IDENTITIES(target)` + `DATASET_OVERLAP_REQUIRED` |
| `PairedDatasetSplitInstruction` (primary+secondary definidos) | `DATASET_OVERLAP_REQUIRED` |
| `PairMiningInstruction` (input definido) | `MIN_CALIBRATION_PAIRS` |
| `StrategyEvaluationInstruction` (input definido) | `SCORE_CALIBRATION_STATE(CALIBRATED)` + `CASEWORK_STATE(False)` + `MIN_CALIBRATION_PAIRS` |
| `DatasetBalancingInstruction` (RANDOM) | `MIN_IDENTITIES` |
| `DatasetBalancingInstruction` (BY_QUALITY) | `MIN_IDENTITIES` + `QUALITY_METRIC_AVAILABLE` |
| `HubnessAnalysisInstruction` (dataset definido) | `MIN_SAMPLES` + `MAX_SAMPLES_WARNING(100k)` |
| `HubnessAblationInstruction` BY_QUANTITY | `MIN_SAMPLES` + `MAX_SAMPLES_WARNING(50k)` |
| `HubnessAblationInstruction` BY_POSE | `MIN_SAMPLES` + `MAX_SAMPLES_WARNING(50k)` + `HAS_POSE_METADATA` |

**Regras dinâmicas por DTO** (construídas em `build_validation_rules()`, não persistidas):

| DTO | Regras dinâmicas |
|---|---|
| `PipelineTrainingInstruction` (DUAL) | `DATASET_OVERLAP_REQUIRED` + `DUAL_PIPELINE_DIM_COMPAT` + `DUAL_PIPELINE_SUBSEQUENCE` |
| `PipelineApplicationInstruction` LINEAR | `MODEL_FITTED(pipeline)` + `EMBEDDING_DIM_MATCHES_MODEL(dataset, pipeline)` |
| `PipelineApplicationInstruction` DUAL | `MODEL_FITTED(pipeline)` + `EMBEDDING_DIM_MATCHES_MODEL` × 2 (source + target) |
| `DatasetMergeInstruction` | `DATASETS_EMBEDDING_COMPAT(base, secondary)` |
| `SimilarityScoringInstruction` self | `MODEL_FITTED(model)` + `EMBEDDING_DIM_MATCHES_MODEL(probe, model)` |
| `SimilarityScoringInstruction` cross | + `EMBEDDING_DIM_MATCHES_MODEL(reference, model)` |
| `ScoreCalibrationInstruction` | `MODEL_FITTED(calibrator)` + `SCORE_CALIBRATION_STATE(input, RAW)` |
| `CaseworkVisualizationInstruction` | `CASEWORK_STATE(reference, False)` + `CASEWORK_STATE(case, True)` + `SCORE_TYPE_MATCH` |

Defaults persistentes não sobrescrevem escolhas explícitas do usuário (lista não-vazia OU `validation_rules_user_managed=True` é considerada deliberada).

**Flag `validation_rules_user_managed: bool` em `StrategyInstructionDefinition`**. Necessária para preservar **intenção do usuário entre reloads**: quando o usuário deleta uma regra na UI, o flag vira `True`. `_ensure_default_validation_rules` faz early-return se `True`, mesmo com lista vazia. O botão "🔄 Restaurar padrões" reseta para `False` e dispara `_sync_declared_io`, repopulando defaults. Sem esse flag, deletar a única regra default + salvar + reabrir o projeto repopularia o default automaticamente (lista vazia interpretada como "ainda não populei").

**UI Phase 2+3 — Seção "Regras de Validação" no `InstructionConfigPanel`**: seção âmbar abaixo de "Configurações de saída". Dois grupos visuais empilhados:

1. **Regras configuráveis** (persistentes em `instruction.validation_rule_configs`). Cada linha:
   - 🛡️ ícone + nome (label do `ValidationRuleType`) + sumário HTML dos params (`_summarize_rule_params`, cobre os 19 tipos).
   - Tooltip com a descrição da regra.
   - Botão `📤` — abre `SaveValidationRuleTemplateDialog` que salva a regra como `ValidationRuleTemplate` em `<global>/validation_rules/{id}.json` e/ou `<project>/validation_rules/{id}.json`.
   - Botão `🗑` — `_on_delete_validation_rule(index)` remove, marca `user_managed=True`, rebuild, emite `config_changed`.

2. **Regras intrínsecas** (dinâmicas, retornadas por `instruction.build_validation_rules()` a cada render). Cada linha:
   - 🛡️ ícone + nome + descrição completa (já contém os IDs dos recursos via construtor da regra concreta).
   - Indicador `🔒` (não-clicável) — sinaliza "esta regra vai rodar, não é removível".
   - Fundo cinza-azulado distinto do fundo âmbar das configuráveis.
   - Tooltip: "Regra intrínseca — pré-condição matemática da instrução, falha aqui significa impossibilidade de execução".

Botões no header da seção:
- "🔄 Restaurar padrões" — `_on_restore_default_validation_rules` limpa lista + `user_managed=False` + `_sync_declared_io()` + rebuild.
- "➕ Adicionar Regra" — abre `AddValidationRuleDialog` populado com os 11 tipos configuráveis + templates salvos (`_load_validation_rule_templates()` no controller varre `validation_rules/` do global + projeto). Usuário escolhe tipo OU template, preenche params via form dinâmico (`_RULE_PARAM_SCHEMAS` no dialog declara campos `kind=resource|resource_multi|int|float|enum_choice` por tipo). Em aceitar: `editor.append_validation_rule_to_instruction(instr, cfg)` → `panel.append_validation_rule_config(cfg)` → mark `user_managed=True` + rebuild + emit `config_changed`.

**Filtragem de recursos no dropdown do `AddValidationRuleDialog`** (`_resources_available_at_instruction_validation` no controller): apenas recursos **garantidos no contexto no momento de validação da instrução X** entram nos dropdowns. Conjunto:
1. PRIMARIES da estratégia (registrados desde `prepare_context`).
2. `X.input_resource_ids` (gate de dependência garante).
3. `output_resource_ids` de instruções anteriores a X em ordem de execução (preparation → validation → casework; dentro da fase, ordem da lista).

**Excluídos intencionalmente**: outputs da própria X (não existem na hora da validação — X não rodou) e outputs de instruções posteriores (ainda não produzidos). Sem isso o usuário poderia anexar regra que sempre falha com "recurso não encontrado". Implementação itera `definition._phase_lists()` parando ao encontrar X.

**Template-only persistence model**: validation rules continuam embedded em `instruction.validation_rule_configs` (não viraram resources). O `ValidationRuleTemplate(template_id, name, description, rule_config)` em `faceaudit/models/validation/validation_rule_template.py` é só wrapper de metadata pra a feature de Save/Import. `StrategyArtifactType.VALIDATION_RULE_TEMPLATE` + `GlobalLibraryFolder.VALIDATION_RULES` + `ProjectFolder.VALIDATION_RULES` + mapeamentos em `GlobalLibraryFileType` e `StrategyFileType` permitem persistência via `save_artifact` / `list_artifacts`. **Templates NÃO aparecem na árvore de recursos do left panel** — vivem só pra reuso via dialog.

**Always-on no factory** (não removível via UI): `StrategyExecutorFactory.build_default_validation_rules()` retorna `{StrategyEmbeddingConsistencyRule.rule_id: rule}`. Fica fora do `validation_rule_configs` da instrução.

**Backward compat**: JSON antigo com `validation_rule_ids: List[str]` é descartado em silêncio pelo `from_dict` — defaults são repopulados via `__post_init__`.

#### Regras concretas

19 implementações em `faceaudit/validation_rules/`, todas estendendo `ValidationRuleBase` (em `validation_rule_base.py`). A classe base fornece `_get_or_compute_stats(resource_id, context)`: busca `"{resource_id}_stats"` no contexto ou computa via `DatasetStatsService` e armazena como `DATASET_STATS` resource. **Não implementar `ValidationRuleProtocol` diretamente — estender sempre `ValidationRuleBase`.**

**Regras originais (Phase 1):**
- `FisherRule` — `n_identidades > embedding_dim`.
- `MinIdentitiesWithMultipleSamplesRule` — `n_ids com count >= 2 >= min_required` (default 2; warning se < safety_threshold default 10). Cobre o caso do PLDA fit explodir com poucas identidades não-singleton.
- `MinCalibrationPairsRule` — score dataset tem `n_gen >= min_genuines` e `n_imp >= min_impostors`. Aceita só `SimilarityScoreDataset`.
- `DatasetIsolationRule` — dois datasets disjuntos em identidades.
- `DatasetOverlapRequiredRule` — dois datasets compartilham ≥ N identidades.
- `HoldoutLeakageRule` — holdout não compartilha identidades com treino/calibração.
- `DualPipelineDimensionCompatibilityRule` — DUAL: source e target produzem mesma dim final.
- `DualPipelineSubsequenceRule` — DUAL: steps do target são prefixo dos do source.

**Regras de cobertura mínima por instrução (Phase 4):**
- `MinIdentitiesRule` — `n_ids >= min_required` (default 2; warning < safety_threshold default 5). Aplicável a SPLIT, BALANCING e treinos com LDA.
- `MinSamplesRule` — `n_imgs >= min_required` (default 2; warning < safety_threshold default 100). Aplicável a HUBNESS, SPLIT por índice e treinos com Whitening.
- `ModelFittedRule` — verifica `is_fitted=True` em TRAINED_PIPELINE / TRAINED_SIMILARITY_MODEL / TRAINED_CALIBRATION_MODEL / TRAINED_TYPICALITY_MODEL.
- `EmbeddingDimMatchesModelRule` — `dataset.embedding_dim == model.input_dimension`. Modelos sem `input_dimension` declarada (ex: Cosine) passam silenciosamente.
- `ScoreCalibrationStateRule` — detecta via presença de `log_lr` no metadata dos items (amostra 100 itens pra performance). Param `expected` ∈ `{RAW, CALIBRATED}`.
- `CaseworkStateRule` — checa `dataset.metadata.is_casework` contra `expected: bool`.
- `ScoreTypeMatchRule` — referência e caso devem ser ambos RAW ou ambos CALIBRATED (detecção via mesma amostragem do log_lr).
- `DatasetsEmbeddingCompatRule` — múltiplos datasets compartilham coluna de embedding e dimensão. Param opcional `embedding_column` força coluna específica; senão checa interseção não-vazia.
- `QualityMetricAvailableRule` — lê `column_coverage[metric].coverage` do `DatasetStats`; warning se < `min_coverage` (default 0.5). EMBEDDING_NORM é trivialmente OK (computada on-the-fly).
- `HasPoseMetadataRule` — verifica colunas `yaw` e `pitch` presentes com cobertura > 0.
- `MaxSamplesWarningRule` — warning quando `n_imgs > max_samples` (default 100_000). Nunca emite CRITICAL — só aviso de custo computacional.

**`LRStrategyDefinition` e `LRStrategyTrainedModel` NÃO herdam de seus respectivos Protocols.** Em Python, herança explícita de `Protocol` é desnecessária — a tipagem estrutural garante compatibilidade. Além disso, as `@property` declaradas no Protocol bloqueavam o `__init__` gerado pelo `@dataclass`. Ambas as classes herdam apenas de `JSONSerializableMixin`. **Não adicionar herança de Protocol em nenhum `@dataclass` concreto.**

**Pendrive Paradigm — projeto autocontido e portátil.** Um projeto FaceAudit é uma **pasta autocontida** ancorada por um arquivo `.fa` (ex: `caso_01.fa`) que armazena apenas os metadados do projeto (`FaceAuditProjectMetadata`). A estrutura de subpastas é criada por `LocalProjectArtifactStorage` a partir do enum `ProjectFolder` — todas as referências em `StrategyArtifactReference.path` são sempre **relativas** à raiz do projeto. `FaceAuditProjectService` é o ponto de entrada para criação e abertura.

**Sistema de dois níveis — biblioteca global + projeto.** Além dos projetos autocontidos, existe uma **biblioteca global** (pasta configurada na máquina) gerenciada por `GlobalLibraryStorage` com as subpastas definidas por `GlobalLibraryFolder`. Armazena recursos reutilizáveis entre projetos: datasets, split configs, mining configs e templates de estratégia. Ao montar uma estratégia, o usuário seleciona peças da biblioteca e elas são copiadas via `copy_artifact_to_project()`.

**Verificação de integridade é automática na abertura do projeto.** `FaceAuditProjectService.open(verify_integrity=True)` chama `project.check_integrity()` antes de retornar. `has_integrity_problems` retorna True apenas para `MISSING` e `MODIFIED`. `verify_integrity()` não faz parte do `ArtifactStorageProtocol` — é capacidade exclusiva da implementação local.

**Portabilidade Windows ↔ POSIX em `StrategyArtifactReference`.** Projetos criados no Windows gravam `artifact_ref.path` com `\`. No macOS/Linux o pathlib interpreta `\` como caractere literal. `StrategyArtifactReference.from_dict` normaliza automaticamente `\` → `/` ao desserializar; pathlib aceita `/` em todos os SO.

**`EmbeddingDataset.from_parquet` lê schema legado.** Parquets gerados por scripts antigos têm colunas separadas `fiqa_score`/`ofiq_score`/`serfiq_score` + `yaw`/`pitch`. O loader prioriza `quality_scores_json` (formato novo) e cai em fallback para schema legado. Crítico para HUBNESS_ABLATION (BY_QUANTITY) e DATASET_BALANCING (BY_QUALITY).

**`StrategyInferenceService` é o ponto de entrada para inferência forense.** Recebe `LRStrategyTrainedModel` + `StrategyResource` probe/reference + `ArtifactStorageProtocol`. Auto-detecta pipeline, similarity e calibration model por `StrategyResourceType`. Score via `score_matrix()` → calibração via `calibrate_scores()` → `log_lr` em `sample.metadata["log_lr"]`.

### The 19 instruction types

| Type | Executor | What it does |
|---|---|---|
| `PIPELINE_TRAINING` | `PipelineTrainingInstructionExecutor` | Trains `EmbeddingTransformationPipeline` (LINEAR or DUAL); writes `TRAINED_PIPELINE` resource. Optional flag `apply_to_training_datasets` (default False): when checked, applies the freshly trained pipeline to the training dataset(s) immediately after fit and registers transformed datasets as additional outputs. Default `PipelineApplicationPolicy` is used (no checkpoints/trajectory). |
| `PIPELINE_APPLICATION` | `PipelineApplicationInstructionExecutor` | Applies a trained pipeline to dataset(s); writes `INTERMEDIATE_DATASET` resource(s) |
| `DATASET_SPLIT` | `DatasetSplitInstructionExecutor` | Splits dataset into A and B. Modes: `BY_IDENTITY_RATIO`, `BY_IDENTITY_LIST`, `BY_IDENTITY_ABSOLUTE` (rigid floors), `BY_INDEX_RATIO`, `BY_INDEX_LIST`, `BY_SUPERVISED_ADAPTATION`. Identity-based modes accept `grouping_strategy=PRESERVE_LOOKALIKES`. Advanced modes delegam ao `PLDASplitService`. |
| `PAIRED_DATASET_SPLIT` | `PairedDatasetSplitInstructionExecutor` | Splits **two paired datasets simultaneously by identity** — same shuffle is applied to both. Outputs: 4 datasets where B splits carry identical identity sets (guaranteeing genuine cross-pose pairs for calibration). |
| `DATASET_MERGE` | `DatasetMergeInstructionExecutor` | Concatenates or updates datasets by index list |
| `PAIR_MINING` | `PairMiningInstructionExecutor` | Consumes a `SimilarityScoreDataset` + a `MINING_CONFIG` resource referencing `ComparisonMiningConfig`, produces a curated `SimilarityScoreDataset`. Independent rules for genuines and impostors (HARD/EASY/TYPICAL/RANDOM/MIXED). **Semantics of HARD/EASY are class-aware**: `_sort_pool_by_difficulty` ensures pool[0] is sempre o par epistemicamente mais difícil — para impostores top-score (DESC), para genuínos bottom-score (ASC). |
| `SIMILARITY_MODEL_TRAINING` | `SimilarityModelTrainingInstructionExecutor` | Prepares/trains `CosineSimilarityModel` (stateless — sem dataset/config) ou `PLDASimilarityModel`. PLDA hyperparams embebidos via `similarity_model_config: PLDASimilarityModelConfig`. **PLDA tem duas variantes**: `SIMPLE` (default — backend SpeechBrain, usa `em_iterations`/`plda_mode`/`latent_dim`); `TWO_COVARIANCE` (Brümmer & de Villiers 2010, numpy puro — covariâncias completas D×D, treino em forma fechada). `regularization` em TWO_COVARIANCE soma ε·I antes da inversão. **Modo cross-condition**: campo opcional `secondary_dataset_resource_id` (só visível em variantes two-cov) ativa `fit_cross_condition_two_covariance` — peso igual por condição no W, ideal para comparações forenses entre condições assimétricas (ex.: padrão frontal vs questionada lateral). Defaults de validação são estendidos automaticamente quando `secondary_dataset_resource_id` é definido: MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES no secundário (sem FISHER), DATASETS_EMBEDDING_COMPAT e DATASET_OVERLAP_REQUIRED entre primário e secundário. Modo de centragem do W controlado por `config.cross_condition_within_mode` ∈ {PER_CONDITION, BALANCED_IDENTITY default}. |
| `SIMILARITY_SCORING` | `SimilarityScoringInstructionExecutor` | Scores all pairs (self-comparison or cross-comparison); writes `SimilarityScoreDataset` resource |
| `CALIBRATION_MODEL_TRAINING` | `CalibrationModelTrainingInstructionExecutor` | Trains `LogisticRegressionCalibrationModel` or `KDECalibrationModel`. Hyperparams embebidos via `logreg_config`/`kde_config`. **LogReg suporta GridSearch para `C`** via `c_grid`/`cv_folds`/`scoring: LogRegGridScoringMetric`. KDE carries 13 fields (kernel, bandwidths, GridSearch toggle, Fisher epsilon, log-LR clipping). |
| `SCORE_CALIBRATION` | `ScoreCalibrationInstructionExecutor` | Applies a trained `CalibrationModelProtocol` to a `SimilarityScoreDataset` (raw_score) and produces a new dataset with `log_lr` populated em `sample.metadata["log_lr"]`. Output type `CALIBRATED_SCORES`. Reuses `SimilarityScoreDataset` — `to_parquet` detecta `log_lr` em metadata automaticamente. |
| `STRATEGY_EVALUATION` | `StrategyEvaluationInstructionExecutor` | **Validation-only**. Reads a `CALIBRATED_SCORES` resource, delegates to `ValidationMathService.calculate_metrics` — computes EER + threshold, Cllr, **Cllr_min via PAVA in-sample** (oracle, não operacional — Morrison 2021), AUC, Cllr_affine (Brümmer), e ECE plot data. Output `StrategyValidationResult` persistido como `.pkl`. **Validator rejects `is_casework=True` inputs**. |
| `CASEWORK_VISUALIZATION` | `CaseworkVisualizationInstructionExecutor` | **Casework-only**. Choice field `score_type` (CALIBRATED/RAW). Validator: reference must NOT be `is_casework=True`; case MUST be `is_casework=True`. RAW mode computa EER threshold on-the-fly. Output `CaseworkVisualizationResult` (numpy arrays + metadata + `case_summary`). |
| `DATASET_BALANCING` | `DatasetBalancingInstructionExecutor` | Balanceia número de amostras por identidade. **`balancing_mode`**: `CAP_ONLY`/`FLOOR_AND_CAP`/`AUTO_MIN_BALANCE`. **`selection_criterion`**: `RANDOM` ou `BY_QUALITY`. `quality_metric`: FIQA/OFIQ/SERFIQ/EMBEDDING_NORM (calculada on-the-fly via `np.linalg.norm`). |
| `HUBNESS_ANALYSIS` | `HubnessAnalysisInstructionExecutor` | Diagnóstico de hubness em alta dimensionalidade. Executa `compute_analysis` (O(N²)) + 4 estudos derivados (stratification, paired_study, quality_diagnostics, optimization_study). Saída: `HubnessReport` compósito (pickle). |
| `HUBNESS_ABLATION` | `HubnessAblationInstructionExecutor` | Estudo de ablação topológica — recorta progressivamente o dataset por critério `BY_QUANTITY`/`BY_POSE`. **Custoso** — refaz O(N²) em N subsets. Saída: `AblationStudyInfo` (pickle). |
| `KNN_TYPICALITY_MODEL_TRAINING` | `KNNTypicalityModelTrainingInstructionExecutor` | Treina um `KNNTypicalityModel` não-paramétrico a partir de um dataset de referência **já estratificado** pela subpopulação relevante (gênero, etnia, faixa etária). Hiperparâmetros embutidos: `k` (default 50, heurística √N), `ood_quantile`, `population_label`, `embedding_column`. Saída: `TRAINED_TYPICALITY_MODEL` (pickle). Defaults de validação: `MIN_SAMPLES(k+1)` e `MAX_SAMPLES_WARNING(50k)`. |
| `TYPICALITY_PAIR_SCORING` | `TypicalityPairScoringInstructionExecutor` | Aplica modelo(s) de tipicidade treinado(s) a cada face de cada par (self ou cross), combina as duas tipicidades e produz um `SimilarityScoreDataset` com mesma ordem de pares que `SIMILARITY_SCORING`. **Suporta dois modelos** (`typicality_model_probe_resource_id` + `typicality_model_reference_resource_id`) para casos cross-população — ex. cross-pose forense: M_lateral aplicado em probe lateral, M_frontal aplicado em reference frontal. Quando só o modelo do probe é definido, ele é usado em ambas as faces (caso simétrico). **Pareável pelo SCORE_VECTORIZATION** com scores de similaridade pra treinar calibrador LogReg 2D+ que conhece raridade local. Combinadores: simétricos (MEAN/MIN/MAX/PRODUCT/ABS_DIFF) ou **unilaterais** (`REFERENCE_ONLY`/`PROBE_ONLY` — quebram simetria propositalmente para alimentar `SCORE_INTERACTION_FEATURE`; emitem WARNING em self-comparação onde os papéis são convencionais). `score_type` (MEAN_KNN_SIMILARITY/TYPICALITY_PERCENTILE/LOG_DENSITY_KNN). Modo casework zera `same_person` no output. Backward compat: campo legado `trained_typicality_model_resource_id` migrado pra `typicality_model_probe_resource_id` automaticamente. |
| `SCORE_INTERACTION_FEATURE` | `ScoreInteractionFeatureInstructionExecutor` | Constrói feature forense de **interação centrada** entre score base (cosseno/PLDA) e raridade derivada de tipicidade, materializando a tese forense de que raridade NÃO é evidência independente — ela só modula a *força* de uma similaridade observada. Recebe dois `SimilarityScoreDataset` alinháveis por par e produz UM `SimilarityScoreDataset` com `raw_score` 1D (feature derivada) consumível por `SCORE_VECTORIZATION`. Modos: `ONE_SIDED_POSITIVE` (default: `max(base−s0, 0)·rarity` — raridade só amplifica evidência positiva, mais defensável), `LINEAR` (`(base−s0)·rarity`, bilateral), `HINGE_BASE` (`max(base−s0, 0)` — feature-companheira para o modelo unilateral plenamente geral `[base, HINGE_BASE, ONE_SIDED_POSITIVE]`). `s0` (ponto de evidência neutra, logLR=0) resolvido em três modos: `AUTO_LOGREG` (LogReg auxiliar univariado com `LogRegS0Config` override opcional — defaults: C=1.0, solver=liblinear, class_weight=balanced; BLOQUEADO em casework; emite WARNING de CV-leakage), `MANUAL` (valor fixo), `INHERIT` (lê `interaction_s0` do metadata de execução anterior — **obrigatório em validação/casework** para preservar a definição de "evidência neutra"; emite WARNING quando `base_score_type`/`base_similarity_model_resource_id`/`base_embedding_column` divergem entre o herdado e o atual). Inversão typicality→rarity automática por `score_type` (`TYPICALITY_PERCENTILE` → `1 − v/100`; `MEAN_KNN_SIMILARITY`/`LOG_DENSITY_KNN` → `−v`). Alinhamento orientado `(probe_hash, reference_hash)` quando o modifier vem de combiner unilateral (REFERENCE_ONLY/PROBE_ONLY); frozenset simétrico caso contrário. Detecta **chaves duplicadas** e valida consistência sample-a-sample de `same_person` em modo não-casework. AUTO_LOGREG: erro se `coef ≈ 0` ou `coef ≤ 0`; warning se `s0` fora do intervalo de scores. Métricas de auditoria persistidas no metadata (`interaction_s0_coef/intercept/C/coef_sign`, `interaction_s0_inside_score_range`, `fraction_scores_above_s0`, `fraction_hinge_active`, `alignment_orientation`, `rarity_transform`, `modifier_score_type`/`combiner`, `base_score_type/name/similarity_model_resource_id/embedding_column`). **Recomendação**: usar com `LogisticRegressionCalibrationModel.standardize_features=True` — o código denormaliza coeficientes (`calibrator_coef_raw`) para escala bruta na auditoria. |

### Adding new components — use the project skills

Project skills are available in `.claude/commands/`. Use them instead of creating components manually:

| Command | What it creates |
|---|---|
| `/new-instruction <Name>` | Complete new instruction type: enum + DTO + dispatch + executor + factory registration |
| `/new-similarity-model <Name>` | New similarity model: enum + config DTO + implementation + executor integration |
| `/new-calibration-model <Name>` | New calibration model: enum + config DTO + implementation + executor integration |
| `/new-validation-rule <Name>` | New validation rule extending `ValidationRuleBase`; includes `_get_or_compute_stats()` cache helper |
| `/new-transformation <Name>` | New embedding transformation: `StepType` + config DTO + `TransformationType` member + transformator class + pipeline registration |

---

## Strategy Manager V2 — Frontend

The tab is registered in `interface/main_gui_controller._init_strategy_manager_v2()`. It receives `global_storage` (optional `GlobalLibraryStorage`) and `app_config` (for resolving initial FileDialog directories).

### Folder structure

```
interface/tabs/strategy_manager_v2/
├── strategy_manager_v2_controller.py   ← Central controller (QObject); owns project + execution state
├── strategy_manager_v2_view.py         ← QStackedWidget: welcome screen (0) + workspace (1). Workspace é splitter horizontal `[LeftPanel | StrategyEditor]` — a "Comparação" virou janela top-level acessada pelo botão "Comparar" no left panel
├── toolbar/
│   ├── global_resources_toolbar.py     ← Pipeline / Mining / Split / Manage Library buttons
│   └── project_toolbar.py             ← New / Open / Save / Close / Run / Save as Template
├── left_panel/
│   ├── left_panel_widget.py            ← QSplitter vertical com 3 seções; forwards signals
│   ├── strategy_list_widget.py         ← QListWidget of LRStrategyDefinition; New/Duplicate/Remove; action row Executar/Salvar como Template/**Comparar**
│   ├── resource_tree_widget.py         ← QTreeWidget grouped by StrategyResourceType; "Adicionar Recurso" dropdown
│   └── resource_info_widget.py         ← Metadata + usage + pipeline details + dataset stats of selected resource
├── strategy_editor/
│   ├── strategy_editor_widget.py       ← QTabWidget com 3 abas (Preparação/Validação/Caso), cada uma com InstructionListWidget + InstructionConfigPanel + InstructionResultsPanel em horizontal splitter (3 colunas ~200/400/600)
│   ├── instruction_list_widget.py      ← Drag-and-drop list com ▶ Run + visual states (🔒/✅)
│   ├── instruction_config_panel.py     ← Dynamic form com type-filtered dropdowns via ResourceFieldSpec + Validate button + **seção "Regras de Validação"** (lista + restaurar padrões)
│   ├── instruction_results_panel.py    ← 3ª coluna: outputs (OutputRowWidget) + inline plot + 🔍 Expandir. Per-instruction _ResultsCard cache via QStackedWidget. Lazy hydrate button para datasets grandes em disco.
│   ├── result_widgets/                 ← Plot widgets reusáveis inline (results panel) e em dialogs (full-screen):
│   │   ├── score_distribution_widget.py     — histograma + KDE (gen vs imp), use_log_lr para SCORE_CALIBRATION
│   │   ├── validation_result_widget.py      — QTabWidget com 4 abas lazy (LRs/Tippett/DET/ECE)
│   │   ├── casework_visualization_widget.py — single plot, KDE para n_case≥30 ou linhas verticais
│   │   ├── hubness_report_widget.py         — QTabWidget com 5 abas lazy (topology grid / stratification / paired / quality / density). Captura HubnessPlotService via plt.show monkey-patch + PNG transplant.
│   │   └── hubness_ablation_widget.py       — single-plot KDE overlay dos cenários de ablação. Mesmo monkey-patch.
│   ├── validation_result_viewer_dialog.py    ← Thin wrapper around ValidationResultWidget
│   ├── casework_visualization_viewer_dialog.py ← Thin wrapper around CaseworkVisualizationWidget
│   ├── hubness_report_viewer_dialog.py       ← Wrapper around HubnessReportWidget + extra "Galeria Completa" tab
│   ├── hubness_ablation_viewer_dialog.py     ← Thin wrapper around HubnessAblationWidget
│   ├── graph_viewer_dialog.py          ← Generic Figure dialog (trajetória 3D + score distribution from 🔍 Expandir)
│   ├── output_row_widget.py            ← Per-output row mostrando runtime state (⬜/🟡/🟢) + lazy summary
│   └── nested_dto_widget.py            ← Inline editor para nested config DTOs (PLDA / LogReg / KDE / pipeline policies)
├── comparison/                         ← Top-level Strategy Comparison window
│   ├── comparison_window.py            ← QDialog não-modal, aberto pelo botão "Comparar"
│   ├── comparison_kind.py              ← Enum: CALIBRATION_DATASET_DISTRIBUTION / VALIDATION_RESULTS / CASEWORK_RESULTS
│   ├── availability_resolver.py        ← list_available_items(kind, definitions, project_root)
│   ├── comparison_left_panel.py        ← Vertical splitter: kind selector + items tree by strategy
│   ├── flow_layout.py                  ← Canonical Qt FlowLayout port + STRATEGY_COLOR_PALETTE
│   └── views/                          ← calibration_distribution / validation_results / casework_results
├── dialogs/                            ← new_project_v2 / new_strategy / manage_library
└── workers/
    ├── strategy_execution_worker.py    ← QObject + moveToThread; runs StrategyExecutor.execute() (all phases)
    ├── instruction_execution_worker.py ← QObject + moveToThread; runs single instruction
    └── project_io_worker.py            ← Currently unused; open is sync
```

### Key design decisions in the V2 frontend

**Project open is synchronous with WaitCursor, not threaded.** On Windows, `ApplicationModal` dialogs prevent queued signal delivery from worker threads, causing `LogProgressDialog` to hang forever.

**"Adicionar Recurso" lives in the left panel, not the toolbar.** `ResourceTreeWidget` tem `QToolButton` com `MenuButtonPopup` mostrando 4 opções (Dataset / Split Config / Mining Config / Pipeline Config).

**Resource discovery = definitions scan + disk scan.** `_collect_project_resources()` faz duas passagens: (1) resources declarados nas definitions, (2) arquivos em disco via `storage.list_artifacts()`.

**`app_config.paths.*` drives FileDialog initial directories.** O controller lê `app_config.paths.<attr>` (`datasets`, `split_configs`, `mining_configs`, `pipelines`).

**Per-instruction execution com dependency tracking e auto-persistence.** O controller mantém: `_execution_context`, `_executed_instruction_ids`, `_executor`. Run de uma instrução: `_ensure_executor_and_context()` (lazy), `can_execute_instruction()` (gate), `InstructionExecutionWorker` (thread). Após sucesso: `_persist_produced_resources()` + `_refresh_instruction_states()` + `_refresh_produced_resources()`.

**Three-tab strategy editor com 3 colunas.** `QTabWidget` com 3 abas (Preparação, Validação, Análise de Caso). Cada aba: splitter horizontal de 3 colunas — esquerda `InstructionListWidget` (~200px); meio `InstructionConfigPanel` (~400px); direita `InstructionResultsPanel` (~600px). `InstructionConfigPanel` recebe `outputs_in_results_panel=True` quando instanciado pelo `StrategyEditorWidget`.

**Painel de resultados (3ª coluna) — `InstructionResultsPanel`.** Renderiza, para a instrução selecionada: outputs (`OutputRowWidget`) + gráfico inline + botão `🔍 Expandir`. Mapeamento de plot por tipo via `_instruction_plot_kind`. Mantém cache `_cards: Dict[instruction_id, _ResultsCard]` via `QStackedWidget` — re-execução de instrução chama `invalidate_results_for_instruction(instruction_id)`. Datasets grandes em disco mostram botão `📊 Carregar e visualizar` em vez de hidratar automaticamente.

**Automatic placeholder sync e resource persistence.** `_sync_all_output_placeholders()` cria/remove placeholders. `_persist_produced_resources()` chama `StrategyExecutor.persist_produced_resource()` após cada execução.

**Type-filtered resource dropdowns.** `InstructionConfigPanel` usa `get_resource_field_specs()` para mostrar só recursos do `StrategyResourceType` correto. Output fields são text inputs.

**Editor de nome de output na coluna 2 (config), espelhado read-only na coluna 3 (resultados).** Decisão de configuração permanece no config panel. Col 3 usa `OutputRowWidget(read_only=True)` com `QLabel` puro (`TextSelectableByMouse`). Sync via signal `output_name_changed`.

**Painel lateral colapsável (`Cmd+B` / `Ctrl+B`).** `StrategyManagerV2View.toggle_left_panel()` esconde/mostra via `widget.hide()/show()`. Tamanhos memorizados antes do collapse, restaurados na expansão.

**Janela de Comparação de Estratégias (`ComparisonWindow`).** `QDialog` não-modal aberto pelo botão "Comparar". `showMaximized()`. Mantida em `self._comparison_window` para preservar referência forte. Re-clique faz `raise_() + activateWindow()`.

**Três tipos de comparação** (`ComparisonKind`):
- **`CALIBRATION_DATASET_DISTRIBUTION`** — distribuição gen × imp que alimentou cada calibrador. Usa sidecar `<stem>_dist.json` quando disponível.
- **`VALIDATION_RESULTS`** — tabela 11 colunas + 3 plots overlay (DET, Tippett, ECE) + grid de cells de distribuição log-LR.
- **`CASEWORK_RESULTS`** — tabela 9 colunas + grid de plots forenses, cada um com borda esquerda na cor da estratégia.

**`FlowLayout` canônico Qt para reflow responsivo.** Em `comparison/flow_layout.py` — porta direta de `flowlayout.cpp` do qtbase. Usa `hasHeightForWidth() + heightForWidth(width)`. Cells `setFixedSize(640, 480)`. **Não reimplementar manualmente.**

**Paleta `STRATEGY_COLOR_PALETTE` compartilhada.** 12 cores estáveis baseadas em matplotlib tab colors. Usada em swatch da tabela, legendas dos plots overlay, e borda esquerda das cells.

**Plots usam `figure.set_layout_engine("constrained")`.** Constrained layout reserva espaço para ylabel/xlabel/legenda antes de dimensionar o axes. Aplicado em `ScoreDistributionWidget._render` e `CaseworkVisualizationWidget._render`. Widgets compartilhados ganham flag `show_title: bool = True` (False para views de comparação).

**`InstructionConfigPanel._get_field_tooltip()` lê `metadata.help` do dataclass field e aplica como tooltip.** Strings começando com `<html>` são renderizadas como rich text pelo Qt. **Genérico**: qualquer DTO com `metadata={"help": "<html>..."}` ganha tooltip automático.

**Info label sob dropdowns de DATASET.** Cada combo que aceita `StrategyResourceType.DATASET` recebe um QLabel abaixo mostrando `N° Amostras • N° Identidades • Dim. Embedding`. Fonte preferida: sidecar `<stem>_stats.json` ao lado do parquet (gerado on-demand via `DatasetStatsService._read_parquet_stats` se faltar). Fallback: payload em memória. Lazy init via `showEvent` do container (evita flash — ver "Qt patterns" abaixo).

**`OutputRowWidget(read_only=True)` embeda `ResourceInfoWidget(compact=True)` lazy via toggle "▾ Detalhes".** Botão `▸ Detalhes` em cada output row da 3ª coluna. `compact=True` oculta o form básico e a barra de abas — só sobram as seções ricas. `project_root` extraído de `resource_loader._storage.project_root` em `_ResultsCard.update_state`. **Crítico**: o controller V2 chama `editor.set_resource_loader(loader)` logo após construir o executor.

**Handler `_on_hydrate_resource_requested` no controller V2.** Botão "📊 Carregar e visualizar" emite `hydrate_requested` que sobe panel → editor → controller. O handler: garante executor + loader, busca o recurso, hidrata, registra no contexto vivo, invalida o card cacheado e força reconstrução.

**Trajectory button só em PIPELINE_APPLICATION (não PIPELINE_TRAINING).** O `PipelineTrainingInstructionExecutor._apply_to_training_datasets` usa `PipelineApplicationPolicy.default()` (`save_trajectory_data=False`).

**Cascade transitivo após edição inline de PIPELINE_CONFIG / MINING_CONFIG (botão ✎).** Após editar e salvar nova versão do config, `_invalidate_dependents_of(resource_id)` faz BFS sobre o grafo de dependência:
- Itera `self._definitions` (todas as estratégias do projeto).
- Para cada instrução com `input_resource_ids` interseccionando `dirty`: chama `_delete_instruction_output_files` (apaga artefato + sidecars + PNGs), seta `existing.artifact_ref = None`/`payload = None` em `defn.resources`, descarta `instruction_id` de `_executed_instruction_ids`, adiciona outputs em `dirty` e em `touched_strategy_ids`.
- Termina quando dirty estabiliza.
- Salva `save_definition(defn)` para cada strategy em `touched_strategy_ids` (erros surfaceados via QMessageBox — não swallow silencioso).
- Refresh visual: `_refresh_instruction_states` + `_refresh_produced_resources` + `_refresh_left_panel`.

**Gate do cascade**: `if not touched_strategy_ids: return` (NÃO `if not any_invalidated`). A versão anterior com `any_invalidated` retornava cedo se nenhum output tinha `existing` em `defn.resources` (caso raro de placeholder fora de sync), perdendo o refresh visual. Touched-based gate garante que se o BFS atingiu alguma instrução, o save+refresh sempre acontece.

**`_refresh_left_panel` re-aponta `self._current_definition`** para a instância correspondente em `self._definitions` após recarregar do disco. Crítico: antes `self._current_definition` ficava como instância órfã (não estava mais em `self._definitions`); cascade mutava as instâncias frescas, mas `_compute_executed_ids_from_disk` lia da instância antiga não-mutada → ✅ falso persistente na lista de instruções.

**`_refresh_instruction_states` usa REPLACEMENT (`=`), não UNION (`|=`).** `self._executed_instruction_ids = self._compute_executed_ids_from_disk()`. Union podia re-adicionar IDs que o cascade acabara de descartar. Estado de disco é sempre autoritativo após save_definition.

**Defesa adicional em `_persist_edited_config_resource`**: após `_invalidate_dependents_of`, força `self._executed_instruction_ids = self._compute_executed_ids_from_disk()` + atualiza UI diretamente via `update_instruction_states`. Garante que a UI sempre se sincroniza com o estado autoritativo de disco, mesmo se algum caminho intermediário corrompeu o set in-memory.

**Erros de `save_definition` agora são surfaceados** via QMessageBox (antes eram silenciosamente engolidos via `except Exception: pass`). Aplicado em `_save_current_definition`, `_persist_edited_config_resource`, e no save loop de `_invalidate_dependents_of`. Bug histórico: silent failure mascarava casos onde mutações in-memory (novo hash após edição, cascade de invalidação) nunca chegavam ao disco — usuário só percebia no próximo open com integrity warning.

**Reused from outros tabs do app** (imported directly, no copy):
- `LogProgressDialog` — used for strategy execution
- `PipelineCreatorController` — "Pipeline" button in global resources toolbar
- `NewMiningConfigDialogController` — "Mining" button
- `NewSplitConfigDialogController` — "Split" button

### StrategyArtifactType / StrategyFileType — import mappings

Quando `copy_artifact_to_project()` é chamado com um tipo da global-library, `StrategyFileType._artifact_map()` mapeia para um tipo project-local:

| `StrategyArtifactType` | `StrategyFileType` | Destination in project |
|---|---|---|
| `GLOBAL_DATASET` | `SOURCE_DATASET` | `datasets/{dataset_id}.parquet` |
| `SPLIT_CONFIG` | `PROJECT_SPLIT_CONFIG` | `split_configs/{config_id}.json` |
| `MINING_CONFIG` | `PROJECT_MINING_CONFIG` | `mining_configs/{config_id}.json` |
| `PIPELINE_CONFIG` | `PROJECT_PIPELINE_CONFIG` | `pipelines/{config_id}.json` |

`STRATEGY_TEMPLATE` e `UNKNOWN` intencionalmente não têm mapping.

---

## Qt/PyQt patterns e convenções

### Sempre passar `parent=` ao criar QWidgets — anti-flash de janela top-level

**Bug**: ao criar um `QWidget`/`QLabel`/`QFrame`/`FigureCanvas`/etc. **sem parent** (`parent=None` é o default), Qt o trata como **top-level window**. Mesmo que você adicione o widget a um layout logo depois via `addWidget` (que faz reparent), há um intervalo (no event loop seguinte) em que Qt dispara um `Show` event no widget enquanto ele ainda é top-level. Em algumas plataformas (Windows especialmente) isso causa um **flash visível de janelinha** antes do reparent. O sintoma típico: janela pequena (~20×19 px ou tamanho default 640×480) aparece e some rapidamente quando o usuário seleciona algo na UI (lazy widget creation).

**Solução**: sempre passar `parent=` no construtor — mesmo que o widget seja imediatamente adicionado a um layout. O parent provisório pode ser qualquer ancestral legítimo (geralmente `self` do widget que está construindo a UI), Qt reparente normalmente via `addWidget` depois sem efeito colateral.

**Padrão correto**:
```python
class MyWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        # CERTO: parent=self desde o construtor
        self._lbl = QtWidgets.QLabel("texto", parent=self)
        layout.addWidget(self._lbl)
```

**Padrão errado** (causa flash):
```python
self._lbl = QtWidgets.QLabel("texto")  # ← parentless, vira top-level
layout.addWidget(self._lbl)  # reparenteia, mas tarde demais
```

**Para `FigureCanvas` do matplotlib** (que não aceita `parent=` no construtor): chamar `self.canvas.setParent(self)` logo após criar.

**Como diagnosticar**: instalar event filter de debug em `interface/run.py` (já removido do repo, mas pode ser reinstalado se precisar). Pattern:
```python
class _TopLevelShowDebugFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Show:
            if isinstance(obj, QtWidgets.QWidget) and obj.parent() is None:
                # Log classe, texto, geometria, stack trace de criação
                ...
        return False
app.installEventFilter(_TopLevelShowDebugFilter())
```

Combinar com monkey-patch em `QWidget.__init__`/`QLabel.__init__` que anexa stack trace de criação ao widget — quando o filtro detectar o show, mostra exatamente onde no Python o widget foi instanciado.

**Histórico**: o bug afetou múltiplos widgets (`_LazyInitContainer` do dataset info label, `_InstructionItemWidget` da lista de instruções, `OutputRowWidget._status_lbl` ⬜, `ResourceInfoWidget` compact mode, vários `QFrame`/`QWidget` em `_ResultsCard` e plot widgets). Solução foi aplicar `parent=self` em massa via replace_all nos arquivos relevantes.

---

## FFService — CoreML/ANE em Apple Silicon

O `FFService` (faceaudit/services/ff_service.py) carrega N instâncias de `ForensicFace` (uma por modelo de recognition declarado) e expõe APIs per-face e batch. Em macOS, quando `use_gpu=True`, o ORT é configurado pra usar o `CoreMLExecutionProvider`. Esse caminho tem várias arestas que estão consolidadas em um único monkey-patch (`_patch_onnxruntime_for_coreml_static_shapes`), descritas abaixo.

### Patch de configuração CoreML em 3 camadas

Aplicado em `ort.InferenceSession.__init__` no momento em que `FFService(use_gpu=True)` é construído. Idempotente; só faz sentido pra GPU. Camadas:

1. **`RequireStaticInputShapes=1`** em toda sessão. Conserto histórico do crash recorrente do detector `det_10g`:
   ```
   CoreML static output shape ({1,1,1,800,1}) and inferred shape ({3200,1}) have different ranks
   ```
   O CoreML pré-computa shapes em compile time, mas operadores pós-NMS do detector emitem tensores cujo número de elementos varia com a quantidade de candidatas que sobreviveram ao threshold. Esta flag instrui o CoreML a só capturar subgrafos cujas shapes são totalmente estáticas — nodes dinâmicos caem no CPU subgraph automaticamente dentro da mesma sessão.

2. **`ModelFormat=MLProgram`** em toda sessão CoreML. Backend mais novo (Core ML 5+, macOS 12+). Necessário pra encoders ViT (`sepaelv4` adaface_vit_webface4m, `sepaelv6` KPRPE):
   - **No `NeuralNetwork` backend** (default), encoders ViT com a override da camada 3 compilam e particionam bem (~87% cobertura) mas falham em **execução** com `Unable to compute the prediction using a neural network model ... (error code -1)`. Algum op de attention do ViT crasha em runtime no NN backend, mesmo passando o partitioning.
   - **`MLProgram`** tem implementação separada e mais recente, e roda sem esse erro. Bônus: preserva FP32 (ref: https://ym2132.github.io/ONNX_MLProgram_NN_exploration); `NeuralNetwork` faz cast implícito FP32→FP16 em hardware. Importante pra defensibilidade forense — embeddings biométricos não devem variar entre máquinas por causa de precisão silenciosa.

3. **`SessionOptions.add_free_dimension_override_by_name("batch_size", N)`** (e `"batch"`) **só em sessões com path contendo `/recognition/`**. Fixa a batch dim simbólica do ONNX em `N = self._batch_size` (vem do `config.ff.batch_size`) no compile time da sessão. Sem isso, ONNX exports recentes com `dynamic_axes={'input': {0: 'batch_size'}}` (v4/v6) tinham cobertura CoreML pífia: o partitioner não consegue resolver shapes intermediárias com batch dim simbólica e despeja quase tudo na CPU (3/1694 nodes no v6 sem override → 0.2% de CoreML). Com a override, a sessão é compilada como se batch fosse literal `N` e a cobertura sobe pra ~93%.

**Restrição da camada 3:** sessões com override só aceitam exatamente `N` amostras por chamada. Detector/attributes (`1k3d68`/`genderage`)/quality (`cr_fiqa_l`) **NÃO** recebem a override (path não bate o filtro) — continuam aceitando qualquer batch, incluindo batch=1 no path per-face legado. Recognition encoders no caminho per-face (`process_image` da forensicface) **quebram** com `use_gpu=True`, porque a forensicface alimenta batch=1 no encoder com override locked a batch=N. Mitigação atual: o caller deve usar o path align+batch (ver abaixo). Pra primeira extração crua em GPU, rodar com `use_gpu=False` temporariamente.

### Padding do batch parcial em `_run_engine_batch`

Como a sessão de recognition só aceita batch=N, o último flush de `_process_dataset_align_then_batch` (que tipicamente tem `n_real < batch_size`) precisaria falhar. O helper `_run_engine_batch` detecta `self._use_gpu=True and n_real < batch_size`, **preenche com zeros até `batch_size`**, roda a inferência e descarta as `pad_count` linhas excedentes nos arrays de retorno via helper local `_truncate_padding`. Em CPU (`use_gpu=False`) não há override, então não há padding.

### Por que o ANE pode ficar em 0% mesmo com 93% de cobertura CoreML

Limitação arquitetural conhecida do Apple Neural Engine para transformers, **não bug**. Algumas ops do ViT (variantes de `LayerNorm`, `GELU`, reshapes da multi-head attention) não estão implementadas no ANE. Quando o scheduler CoreML monta partições e uma partição contém qualquer op não suportada pelo ANE, a **partição inteira** migra pro GPU/Metal (que é mais permissivo). ViT tem blocos de attention espalhados pelo grafo, então quase tudo cai no GPU. GPU/Metal continua sendo aceleração real (muito acima de CPU); o ideal teórico de ANE-only puro é inalcançável pra essa arquitetura de modelo no Apple Silicon atual.

Validação via `asitop` (instalado em `/Users/perito/.virtualenvs/faceaudit/bin/asitop`, alias `asitop` no `~/.zshrc`): se a barra GPU sobe e ANE fica em 0% durante os flushes de batch, está no padrão esperado. Throughput melhor que CPU-only confirma o ganho.

### Os 4 caminhos de `process_dataset`

Quatro modos selecionados em ordem de prioridade (decisão dentro de `process_dataset`):

| Caminho | Quando | Helper | tqdm desc |
|---|---|---|---|
| `is_aligned=True` (faces 112×112 em disco) | Dataset COPY_ALIGNED | `_process_dataset_aligned_batched` | `... (batch)` |
| Faces com `ff_info.aligned_face` populado em memória | Cache hit (raro pós-reload, `FFInfo.to_dict(exclude_heavy=True)` dropa `aligned_face`) | `_process_dataset_cached_aligned_batched` | `... (cache+batch)` |
| **Todas as faces têm `ff_info` populado, sem cache de aligned_face** | **Enrichment típico — usuário adicionando v4 a dataset que já tem v2** | `_process_dataset_align_then_batch` (alinha per-face com `engine.detect_and_align`, extrai embedding em batch) | `... (align+batch)` |
| Nada do acima | Primeira extração crua | Loop `process_face` per-face via `engine.process_image` | `Processando {name}` |

O fallback per-face usa `process_image` (que faz detect+align+extract em uma chamada) e por isso preserva contrato de metadata completo (gender/age/yaw/pitch/roll/det_score). Os 3 caminhos em batch só preservam `fiqa_score` + slots de embedding via MERGE — os demais campos vêm preservados de execução anterior. **Limitação documentada acima:** com `use_gpu=True`, o fallback per-face quebra nas sessões de recognition (override locked). Pra primeira extração crua em GPU, rodar com `use_gpu=False`.

---

## External Tools (accessed via services)

- **AdaFace** — primary embedding model for facial images
- **Forensic Face / sepaelv2** — face detection and embedding extraction (`FFService`)
- **OFIQ** — presentation attack detection (`OFIQService`)
- **SERFIQ** — image quality assessment (`SERFIQService`)
- **SpeechBrain** — PLDA backend used by `PLDASimilarityModel` (variant SIMPLE)
