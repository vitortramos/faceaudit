# Sistema de validação

> Conteúdo migrado do `CLAUDE.md` da raiz na reestruturação de 2026-11. As regras permanentes continuam resumidas lá; aqui fica o detalhe.

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

#### Regras plugáveis

**Modelo de persistência**: cada instrução carrega uma lista `validation_rule_configs: List[ValidationRuleConfig]` (DTO em `faceaudit/models/validation/validation_rule_config.py`) com:
- `rule_type: ValidationRuleType` — enum identifica a classe concreta.
- `params: Dict[str, Any]` — parâmetros de construção (dataset_resource_id, min_genuines, safety_margin_ratio, etc.).

A `ValidationRuleFactory.create(config, instruction_id)` (em `faceaudit/factories/validation_rule_factory.py`) instancia a regra concreta em runtime — não dependemos de pré-registro no contexto. Suporta todos os tipos definidos em `ValidationRuleType` (23 hoje).

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
| `SimilarityScoringInstruction` self | `EMBEDDINGS_COMPLETE(probe)` + `MODEL_FITTED(model)` + `EMBEDDING_DIM_MATCHES_MODEL(probe, model)` |
| `SimilarityScoringInstruction` cross | + `EMBEDDINGS_COMPLETE(reference)` + `EMBEDDING_DIM_MATCHES_MODEL(reference, model)` |
| `ScoreCalibrationInstruction` | `MODEL_FITTED(calibrator)` + `SCORE_CALIBRATION_STATE(input, RAW)` |
| `CaseworkVisualizationInstruction` | `CASEWORK_STATE(reference, False)` + `CASEWORK_STATE(case, True)` + `SCORE_TYPE_MATCH` |

Defaults persistentes não sobrescrevem escolhas explícitas do usuário (lista não-vazia OU `validation_rules_user_managed=True` é considerada deliberada).

**Flag `validation_rules_user_managed: bool` em `StrategyInstructionDefinition`**. Necessária para preservar **intenção do usuário entre reloads**: quando o usuário deleta uma regra na UI, o flag vira `True`. `_ensure_default_validation_rules` faz early-return se `True`, mesmo com lista vazia. O botão "🔄 Restaurar padrões" reseta para `False` e dispara `_sync_declared_io`, repopulando defaults. Sem esse flag, deletar a única regra default + salvar + reabrir o projeto repopularia o default automaticamente (lista vazia interpretada como "ainda não populei").

**UI — Seção "Regras de Validação" no `InstructionConfigPanel`**: seção âmbar abaixo de "Configurações de saída". Dois grupos visuais empilhados:

1. **Regras configuráveis** (persistentes em `instruction.validation_rule_configs`). Cada linha:
   - 🛡️ ícone + nome (label do `ValidationRuleType`) + sumário HTML dos params (`_summarize_rule_params`; `_RULE_PARAM_SUMMARIZERS` cobre 20 tipos, os demais caem no formato genérico).
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

**Compatibilidade**: JSON antigo de INSTRUÇÃO com `validation_rule_ids: List[str]` é descartado em silêncio pelo `from_dict` — defaults são repopulados via `__post_init__`. Isso vale só para a instrução: `LRStrategyDefinition.validation_rule_ids` continua existindo e é consumido pelo `StrategyExecutor` (regras de nível de estratégia).

#### Regras concretas

24 implementações em `faceaudit/validation_rules/` (as 23 do enum + `StrategyEmbeddingConsistencyRule`, sempre ligada), todas estendendo `ValidationRuleBase` (em `validation_rule_base.py`). A classe base fornece `_get_or_compute_stats(resource_id, context)`: busca `"{resource_id}_stats"` no contexto ou computa via `DatasetStatsService` e armazena como `DATASET_STATS` resource. **Não implementar `ValidationRuleProtocol` diretamente — estender sempre `ValidationRuleBase`.**

**Regras originais:**
- `FisherRule` — `n_identidades > embedding_dim`.
- `MinIdentitiesWithMultipleSamplesRule` — `n_ids com count >= 2 >= min_required` (default 2; warning se < safety_threshold default 10). Cobre o caso do PLDA fit explodir com poucas identidades não-singleton.
- `MinCalibrationPairsRule` — score dataset tem `n_gen >= min_genuines` e `n_imp >= min_impostors`. Aceita só `SimilarityScoreDataset`.
- `DatasetIsolationRule` — dois datasets disjuntos em identidades.
- `DatasetOverlapRequiredRule` — dois datasets compartilham ≥ N identidades.
- `HoldoutLeakageRule` — holdout não compartilha identidades com treino/calibração.
- `DualPipelineDimensionCompatibilityRule` — DUAL: source e target produzem mesma dim final.
- `DualPipelineSubsequenceRule` — DUAL: steps do target são prefixo dos do source.

**Regras de cobertura mínima por instrução:**
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

**Regras que completam o conjunto** (além das listadas acima):
- `EmbeddingsCompleteRule` — toda amostra do dataset tem embedding utilizável (presente, 1-D, finito, com a dimensão da maioria). Existe porque o carregador aceita linha sem embedding em silêncio e o erro só aparecia no empilhamento, sem dizer qual imagem.
- `ScoreVectorPairCompatibilityRule` — N ≥ 2 `SimilarityScoreDataset` compartilham exatamente o mesmo conjunto de pares (chave não ordenada) antes da `SCORE_VECTORIZATION`.
- `ScoreVectorOrderConsistencyRule` — em `SCORE_CALIBRATION` com calibrador multivariado, a ordem das colunas do vetor de escores bate com a do treino (o LogReg aprende coeficientes posicionais).
- `ConditionalMonotonicityRule` — auditoria `∂log_LR/∂s > 0`; ver [`arquiteturas_lr.md`](arquiteturas_lr.md) e `../conditional_monotonicity_rule_design.md`.
- `StrategyEmbeddingConsistencyRule` — sempre ligada pela factory, no nível da estratégia: todos os datasets usam a mesma `embedding_column`, vinda do mesmo modelo.
