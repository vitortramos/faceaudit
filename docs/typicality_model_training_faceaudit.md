# Treinamento de modelo de tipicidade (`TYPICALITY_MODEL_TRAINING`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `TYPICALITY_MODEL_TRAINING`, introduzida na
> V2.15 como substituto unificado de `KNN_TYPICALITY_MODEL_TRAINING`.
> Trata de objetivos, dispatch entre os backends k-NN e MoVMF, ambos os
> DTOs de configuração, validações, fingerprints, migração V2.14 →
> V2.15 e exemplos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Documentos relacionados:** `docs/typicality_models.md` (matemática
> dos dois backends), `docs/tipicidade_pareada_faceaudit.md`
> (`TYPICALITY_PAIR_SCORING` downstream),
> `docs/interacao_score_raridade_faceaudit.md` (consumo do score em
> `SCORE_INTERACTION_FEATURE` / `LR_ARCHITECTURE_VECTORIZATION`).
> **Fontes-código:** `faceaudit/models/instructions/typicality_model_training_instruction.py`,
> `faceaudit/executors/instructions/typicality_model_training_instruction_executor.py`,
> `faceaudit/models/typicality/knn_typicality_model_config.py`,
> `faceaudit/models/typicality/movmf_typicality_model_config.py`,
> `faceaudit/services/typicality_model_backend_registry.py`,
> `faceaudit/services/reference_score_calibration_service.py`,
> `faceaudit/services/legacy_strategy_migration_service.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a instrução se encaixa](#1-onde-a-instrução-se-encaixa)
- [2. Os dois backends](#2-os-dois-backends)
- [3. O dispatch — `TypicalityModelType` e o registry](#3-o-dispatch--typicalitymodeltype-e-o-registry)
- [4. Configurações comuns](#4-configurações-comuns)
- [5. Configuração do backend k-NN — `KNNTypicalityModelConfig`](#5-configuração-do-backend-k-nn--knntypicalitymodelconfig)
- [6. Configuração do backend MoVMF — `MoVMFTypicalityModelConfig`](#6-configuração-do-backend-movmf--movmftypicalitymodelconfig)
- [7. Calibração da CDF de referência — `ReferenceScoreCalibrationService`](#7-calibração-da-cdf-de-referência--referencescorecalibrationservice)
- [8. O fluxo do executor passo a passo](#8-o-fluxo-do-executor-passo-a-passo)
- [9. Validações e guard-rails](#9-validações-e-guard-rails)
- [10. Fingerprints persistidos](#10-fingerprints-persistidos)
- [11. Migração V2.14 → V2.15](#11-migração-v214--v215)
- [12. Casos de uso comparados](#12-casos-de-uso-comparados)
- [13. Boas práticas](#13-boas-práticas)
- [14. Perguntas frequentes](#14-perguntas-frequentes)
- [15. Glossário](#15-glossário)
- [16. Referências](#16-referências)
- [Apêndice A — `TypicalityModelTrainingInstruction`](#apêndice-a--typicalitymodeltraininginstruction)
- [Apêndice B — Tabela de migração V2.14 → V2.15](#apêndice-b--tabela-de-migração-v214--v215)
- [Apêndice C — Checklist operacional](#apêndice-c--checklist-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre `TYPICALITY_MODEL_TRAINING`: o que
ela treina, qual backend escolher, quais hiperparâmetros existem, como
defaults históricos são preservados pela migração da V2.14, quais
fingerprints são persistidos para auditoria, e como o output downstream
consome o modelo.

### 0.2 Diferença em relação à V2.14

A V2.14 expunha uma instrução `KNN_TYPICALITY_MODEL_TRAINING` com
hiperparâmetros embutidos diretamente (`k`, `ood_quantile`,
`reference_mode`). Ela treinava exclusivamente o k-NN.

A V2.15 introduz `TYPICALITY_MODEL_TRAINING` unificada, espelhando o
padrão de `SIMILARITY_MODEL_TRAINING` (Cosine / PLDA):

- Um enum discriminador (`typicality_model_type`) seleciona o backend.
- Um config polimórfico (`typicality_model_config`) carrega os
  hiperparâmetros embutidos. O tipo concreto do config é decidido pelo
  tipo do backend.
- O executor despacha via `TypicalityModelBackendRegistry`.

`KNN_TYPICALITY_MODEL_TRAINING` foi **removido** do runtime. Estratégias
V2.14 são migradas automaticamente pelo
`LegacyStrategyMigrationService` quando o projeto é aberto. Detalhes em
§11.

### 0.3 Por que unificar

Três motivos práticos:

1. **Comparação científica honesta entre backends.** Tipicidade k-NN
   (não-paramétrica) e MoVMF (paramétrica) atacam o mesmo problema com
   modelos matemáticos distintos. Ter a mesma instrução despachando
   para os dois permite ablações com pipeline idêntico ao redor.
2. **Extensibilidade.** Novos backends (Power Spherical,
   FAISS-acelerado, etc.) entram registrando-se no
   `TypicalityModelBackendRegistry` — sem mudança na instrução nem no
   executor.
3. **Contrato declarativo único.** As regras de validação (`MIN_SAMPLES`,
   `MAX_SAMPLES_WARNING`) e o fingerprint downstream
   (`vectorization_contract_fingerprint` no LR) tratam ambos backends
   de modo simétrico.

---

## 1. Onde a instrução se encaixa

`TYPICALITY_MODEL_TRAINING` produz um recurso
`TRAINED_TYPICALITY_MODEL` consumido downstream por
`TYPICALITY_PAIR_SCORING`, que por sua vez gera scores por par que
podem alimentar:

- `SCORE_INTERACTION_FEATURE` — feature de interação entre cosseno e
  raridade derivada da tipicidade.
- `LR_ARCHITECTURE_VECTORIZATION` — vetorizador declarativo das
  arquiteturas LR forenses.
- Vetorização final + treinamento de calibrador
  (`CALIBRATION_MODEL_TRAINING`).

```
TYPICALITY_MODEL_TRAINING
    │
    │ produz TRAINED_TYPICALITY_MODEL (k-NN ou MoVMF)
    ▼
TYPICALITY_PAIR_SCORING
    │
    │ produz SIMILARITY_SCORES (típicidade combinada por par)
    ▼
SCORE_INTERACTION_FEATURE  ou  LR_ARCHITECTURE_VECTORIZATION
    │
    ▼
CALIBRATION_MODEL_TRAINING → SCORE_CALIBRATION → STRATEGY_EVALUATION
```

O modelo treinado é uma **referência populacional**: ele aprende como
os embeddings da população se distribuem na hiperesfera para depois,
em casework, comparar uma face questionada com essa distribuição.

---

## 2. Os dois backends

| Aspecto | k-NN | MoVMF |
|---|---|---|
| Tipo | Não-paramétrico local | Paramétrico (mistura) |
| Modelo matemático | k vizinhos esféricos | Mistura de von Mises-Fisher |
| Estado pós-fit | Referência inteira (N×D) + CDF empírica + threshold OOD | μ_m, κ_m, π_m + CDF empírica + threshold OOD |
| Score nativo | `T_k = (1/k) Σ cos(x, y_(i))` (similaridade média aos k-NN) | `p(x) = Σ π_m C_d(κ_m) exp(κ_m μ_mᵀ x)` |
| Suposições | Nenhuma distribucional; hubness vira sinal | Simetria rotacional por componente; D fixo |
| Defensibilidade forense | Tier 1 (rank empírico) | Tier 2 (densidade paramétrica) |
| Status no FaceAudit | **STABLE** | **EXPERIMENTAL** |
| Custo de inferência por query | O(N·D) | O(K·D) |

O badge `EXPERIMENTAL` do MoVMF é uma sinalização de processo, não de
falha matemática: ele indica que as 6 validações listadas em
`docs/typicality_models.md` §8 ainda não foram concluídas no FaceAudit
para promovê-lo a STABLE.

A matemática de cada backend está em `docs/typicality_models.md` §1–§7.
Esta doc foca na **instrução**: como o usuário declara o treino, quais
campos existem, como a UI se comporta, como a auditoria persistida
permite rastreamento downstream.

---

## 3. O dispatch — `TypicalityModelType` e o registry

### 3.1 O enum

`faceaudit/enums/typicality_model_type.py` define dois valores:

| Valor | Label | Descrição |
|---|---|---|
| `KNN` | k-NN não-paramétrico | Estimador local não-paramétrico baseado em vizinhos esféricos. |
| `MOVMF` | Mistura von Mises-Fisher (EXPERIMENTAL) | Mistura paramétrica de vMF na hiperesfera. |

A UI usa `TypicalityModelType.choices_with_description()` para popular
o dropdown com tooltips.

### 3.2 O registry

`TypicalityModelBackendRegistry` (em
`faceaudit/services/typicality_model_backend_registry.py`) é o **único
ponto de verdade** para o mapeamento backend ↔ classe ↔ config DTO.

```python
@dataclass(frozen=True)
class TypicalityBackendSpec:
    model_type: TypicalityModelType
    model_class: type                                       # KNNTypicalityModel | MoVMFTypicalityModel
    potential_score_types: frozenset[TypicalityScoreType]   # capacidades conceituais
    badge: str                                              # "STABLE" | "EXPERIMENTAL"
    label_pt: str
    description: str
    config_class: Optional[type]                            # KNNTypicalityModelConfig | MoVMFTypicalityModelConfig
    trainer: Optional[Callable]                             # plug-in futuro; PR 4 não usa
```

`potential_score_types` vs `enabled_score_types` (da **instância**) é uma
distinção importante:

- **Potencial** é a capacidade conceitual do backend (k-NN pode produzir
  os três; MoVMF não tem `MEAN_KNN_SIMILARITY`).
- **Efetivo** depende do estado pós-fit da instância (MoVMF treinado
  sem CDF de referência não pode produzir `TYPICALITY_PERCENTILE`).

A UI usa o potencial **antes** do treinamento; o executor de
`TYPICALITY_PAIR_SCORING` valida o efetivo da instância treinada.

### 3.3 Dispatch no executor

O `TypicalityModelTrainingInstructionExecutor` resolve um
`trainer_factory()` consultando o tipo do config no DTO da instrução.
Cada `trainer_factory()` é um callable sem argumentos que retorna uma
instância nova do backend concreto com os hiperparâmetros já fixados.
O serviço de calibração de referência consome esse callable uma ou
mais vezes (uma vez em `IN_SAMPLE_LEGACY`; uma vez por fold + uma vez
no refit final em `CROSS_FIT_IDENTITY`).

---

## 4. Configurações comuns

Os dois configs polimórficos compartilham um conjunto de campos:

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `population_weighting_mode` | `PopulationWeightingMode` | `IDENTITY_CENTROIDS` | Como ponderar a referência populacional (face × identidade). |
| `unit_norm_policy` | `UnitNormPolicy` | `ERROR` | Como reagir a embeddings com norma fora de `‖x‖₂ = 1 ± tolerância`. |
| `unit_norm_tolerance` | `float` | `1e-4` | Tolerância da política. |
| `percentile_basis_score_type` | `TypicalityPercentileBasis` | k-NN: `MEAN_KNN_SIMILARITY`; MoVMF: `LOG_DENSITY_ESTIMATE` | Base sobre a qual o percentil é calculado. |
| `reference_score_calibration_mode` | `ReferenceScoreCalibrationMode` | `IN_SAMPLE_LEGACY` (k-NN); `CROSS_FIT_IDENTITY` (MoVMF) | Como construir CDF de referência e threshold OOD. |
| `cv_n_folds` | `int` | `0` (IN_SAMPLE) ou `5` (CROSS_FIT) | Número de folds do cross-fit; ignorado em `IN_SAMPLE_LEGACY`. |
| `ood_quantile` | `float` | `0.01` (k-NN); `0.001` (MoVMF) | Quantil que define o threshold OOD. |
| `population_label` | `Optional[str]` | `None` | Rótulo opcional da população (vai no fingerprint). |
| `embedding_column` | `Optional[str]` | `None` (usa a da estratégia) | Override da coluna de embedding. |

A semântica detalhada está em `docs/typicality_models.md` §3–§5.

---

## 5. Configuração do backend k-NN — `KNNTypicalityModelConfig`

`faceaudit/models/typicality/knn_typicality_model_config.py`

### 5.1 Campos específicos

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `k` | `int` | `50` | Número de vizinhos. Heurística defensável: `k ≈ √N`. |

Todos os demais campos vêm da tabela §4.

### 5.2 Visibilidade dinâmica

`get_hidden_fields()` esconde `cv_n_folds` quando
`reference_score_calibration_mode != CROSS_FIT_IDENTITY` — o campo só
faz sentido nesse modo.

### 5.3 Validações no `__post_init__`

- `k >= 1`.
- `0 <= ood_quantile < 0.5`.
- `unit_norm_tolerance > 0`.
- `cv_n_folds >= 0`. Em `CROSS_FIT_IDENTITY` com `cv_n_folds = 0` o
  config recebe automaticamente `cv_n_folds = 5` como sentinela.

### 5.4 Exemplo

```json
{
  "k": 50,
  "ood_quantile": 0.01,
  "population_weighting_mode": "IDENTITY_CENTROIDS",
  "unit_norm_policy": "NORMALIZE_SILENTLY",
  "unit_norm_tolerance": 1e-4,
  "percentile_basis_score_type": "MEAN_KNN_SIMILARITY",
  "reference_score_calibration_mode": "IN_SAMPLE_LEGACY",
  "cv_n_folds": 0,
  "population_label": "frontal_brasil_2025",
  "embedding_column": "embedding_v2"
}
```

Esse é o config canônico migrado de uma estratégia V2.14 (defaults
históricos preservados literalmente).

---

## 6. Configuração do backend MoVMF — `MoVMFTypicalityModelConfig`

`faceaudit/models/typicality/movmf_typicality_model_config.py`

### 6.1 Campos específicos (EM e mistura)

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `n_components` | `int` | `1` | Número de componentes da mistura (K). Hiperparâmetro científico delicado. |
| `n_init` | `int` | `1` | Número de inicializações aleatórias do EM. Relevante apenas se `n_components > 1`. |
| `centering` | `"none" \| "frechet" \| "euclidean_legacy"` | `"none"` | Centralização pré-fit. `"frechet"` preserva a métrica esférica (média de Karcher); `"euclidean_legacy"` não preserva (compat). |
| `convergence_patience` | `int` | `3` | Iterações consecutivas abaixo de `tol` para declarar convergência. |
| `max_iter` | `int` | `200` | Iterações máximas do EM por inicialização. |
| `tol` | `float` | `1e-5` | Tolerância relativa de log-likelihood para convergência. |
| `random_state` | `int` | `42` | Semente base; seeds por restart = `random_state + init_idx`. |
| `min_component_weight` | `float` | `1e-6` | Peso mínimo para um componente não ser descartado. |
| `min_kappa` | `float` | `1e-6` | Limite inferior de κ por componente. |
| `max_kappa` | `float` | `1e4` | Limite superior de κ por componente. |

Demais campos vêm da §4.

### 6.2 Visibilidade dinâmica

`get_hidden_fields()` esconde:

- `n_init` quando `n_components <= 1` (só uma componente — `n_init`
  irrelevante).
- `cv_n_folds` quando `reference_score_calibration_mode !=
  CROSS_FIT_IDENTITY`.

### 6.3 Validações

Além das comuns (§5.3 adaptadas), o MoVMF valida:

- `n_components >= 1`, `n_init >= 1`.
- `centering` ∈ {`none`, `frechet`, `euclidean_legacy`}.
- `convergence_patience >= 1`, `max_iter >= 1`.
- `CROSS_FIT_IDENTITY` exige `cv_n_folds >= 2`.

### 6.4 `n_components` como hiperparâmetro científico

O help text do campo deixa explícito:

> A escolha deve ser justificada externamente, não ajustada sobre o
> caso examinado. PR 4 endurece com diagnóstico de componentes
> degenerados; uma iteração futura poderá oferecer seleção automática
> por BIC ou held-out por identidade.

A seleção automática via AIC/BIC existe no método utilitário
`MoVMFTypicalityModel.fit_best_model(...)`, mas hoje não é exposta pela
instrução. Uma estratégia compara `n_components ∈ {1, 2, 3, ...}` com K
fixos por execução, não automaticamente dentro da mesma execução.

---

## 7. Calibração da CDF de referência — `ReferenceScoreCalibrationService`

`faceaudit/services/reference_score_calibration_service.py`

O serviço **separa a orquestração da calibração de referência** (CDF +
threshold OOD) do **fit puro** do backend (EM / armazenamento de
referência). Isso permite que a mesma lógica de cross-fit por
identidade sirva aos dois backends sem replicar código.

### 7.1 `IN_SAMPLE_LEGACY`

Caminho default migratório do k-NN.

1. Treina o backend em todo o dataset.
2. CDF empírica = LOO de imagem (o próprio k-NN nativo faz internamente).
3. Threshold OOD = `np.quantile(loo_T_k_sorted, ood_quantile)`.

**Otimista**: a face que ajudou a definir a CDF aparece com rank
artificialmente "central". Preservado por compatibilidade V2.14.

### 7.2 `CROSS_FIT_IDENTITY`

Procedimento defensável forense de duas etapas:

1. Particiona identidades únicas em `cv_n_folds` folds disjuntos
   (semente `random_state`).
2. Para cada fold:
   - Treina backend nos folds train (faces cujas identidades estão nos
     folds train).
   - Pontua faces do fold test (`LOG_DENSITY_ESTIMATE` quando disponível,
     `MEAN_KNN_SIMILARITY` como fallback). Scores OOF.
3. Concatena scores OOF, ordena → CDF empírica.
4. Threshold OOD = `np.quantile(oof_sorted, ood_quantile)`.
5. **Refit final** em toda a população: o modelo persistido é este
   refit.

Resultado: `ReferenceScoreCalibrationResult` com `refit_model`,
`oof_cdf`, `ood_threshold`, `split_fingerprint`, `fold_seeds`,
`n_folds_used`. O `split_fingerprint` é SHA-256 do mapeamento
identidade → fold; estável para mesmo `random_state`.

**Validações**:

- `identity_ids` obrigatório, mesma length que `embeddings`.
- `unique_identities >= n_folds`. Erro explícito caso contrário.
- Para o k-NN, isso substitui o LOO de imagem por LOIO (leave-one-
  identity-out) real.

---

## 8. O fluxo do executor passo a passo

`TypicalityModelTrainingInstructionExecutor.execute(instruction, context)`

Etapas (estado normal, sem erro):

1. **Validação estrutural** (`validate(...)`):
   - Instrução é do tipo correto.
   - `typicality_model_config` não é None.
   - `input_dataset_resource_id` presente e existe no contexto.
   - `trained_typicality_model_resource_id` presente.
   - Dataset hidratado contém embeddings e (se backend exige
     identidades) `identity` em cada face.
2. **Resolve `embedding_column`**: override do config tem prioridade;
   senão usa a coluna declarada pela estratégia
   (`context.embedding_column`).
3. **Extrai embeddings e identidades** do dataset via
   `_extract_embeddings_and_identities(payload, embedding_column,
   need_identities=...)`. `need_identities` é True quando
   `population_weighting_mode != FACE_LEVEL_RAW`.
4. **Constrói `trainer_factory`** consultando o registry:
   - `KNNTypicalityModelConfig` → callable que instancia
     `KNNTypicalityModel(k, ood_quantile, reference_mode,
     unit_norm_policy, unit_norm_tolerance,
     population_weighting_mode)`.
   - `MoVMFTypicalityModelConfig` → callable que instancia
     `MoVMFTypicalityModel(n_components, centering, ...,
     unit_norm_policy, population_weighting_mode)`.
5. **Chama o serviço de calibração**:
   `ReferenceScoreCalibrationService.calibrate(backend=..., mode=...,
   trainer_factory=..., embeddings=..., identity_ids=...,
   n_folds=..., random_state=...)` →
   `ReferenceScoreCalibrationResult.refit_model`.
6. **Registra o modelo** como recurso `TRAINED_TYPICALITY_MODEL` no
   contexto.
7. **Persiste metadata** no `InstructionExecutionResult`:
   - `instruction_id`, `instruction_name`, identifiers do dataset.
   - `typicality_model_type`, `reference_score_calibration_mode`.
   - `model_contract_fingerprint`, `model_state_fingerprint`.
   - `enabled_score_types` da instância pós-fit.
   - `embedding_column`, dimensões.

O metadata persistido é o que o `TYPICALITY_PAIR_SCORING` consulta
downstream para propagar proveniência.

---

## 9. Validações e guard-rails

### 9.1 Defaults persistentes (`validation_rule_configs`)

Quando o usuário não tomou controle das regras
(`validation_rules_user_managed=False`), o DTO popula automaticamente
em `_sync_declared_io()`:

| Regra | Param | Significado |
|---|---|---|
| `MIN_SAMPLES` | `min_required = max(k+1, 2)` ou `n_components+1` | Requisito do fit (k-NN exige k+1 pontos para LOO; MoVMF exige n_components+1). Warning abaixo de 1000 amostras. |
| `MAX_SAMPLES_WARNING` | `max_samples = 50_000` | Warning de custo computacional (k-NN é O(N²) no fit). |

Regras são deletáveis na UI; clicar "🔄 Restaurar padrões" repopula.

### 9.2 Validações dinâmicas no executor

- `typicality_model_config` não pode ser None — erro com instrução para
  reabrir na UI (que auto-instancia o config certo).
- `input_dataset_resource_id` presente e hidratado.
- Faces com identidade None em modo populacional baseado em identidade
  → erro explícito.
- `embedding_column` resolvível (override ou via estratégia).
- `trained_typicality_model_resource_id` coincidente com o
  `input_dataset_resource_id` → warning (sobrescrita).

### 9.3 Validações dentro do backend

Cada backend faz validações próprias:

- `UnitNormPolicy.ERROR` levanta `UnitNormViolationError` se algum
  embedding violar `|‖x‖ − 1| > tolerance`.
- k-NN em `IDENTITY_CENTROIDS` ou `INVERSE_IDENTITY_FREQUENCY` exige
  `identity_ids` paralelo a `embeddings`.
- MoVMF idem para os modos baseados em identidade.
- k-NN: `n_ref >= k + 1` (necessário para LOO).
- MoVMF: `n_components <= n_samples`.
- `CROSS_FIT_IDENTITY` exige `unique_identities >= n_folds`.

---

## 10. Fingerprints persistidos

A V2.15 endurece a auditoria do modelo treinado com três camadas de
fingerprint (espelha o padrão V2.14 do LR architecture):

| Fingerprint | Conteúdo | Uso downstream |
|---|---|---|
| `model_state_fingerprint` | SHA-256 do estado numérico: parâmetros ajustados (μ/κ/π ordenados canonicamente no MoVMF; reference matrix + CDF no k-NN), threshold OOD, `reference_score_weights` (quando `INVERSE_IDENTITY_FREQUENCY`). | Reproduzibilidade numérica exata. Não comparado entre execuções. |
| `model_contract_fingerprint` | SHA-256 do contrato semântico: `model_type`, score semantics IDs habilitados, `population_weighting_mode`, `unit_norm_policy` (e `weighted_T_k_formula_id` no k-NN), centering (MoVMF). | **Entra no `vectorization_contract_fingerprint`** do `LR_ARCHITECTURE_VECTORIZATION` downstream. Mudança aqui propaga e INHERIT rejeita state divergente. |
| `model_provenance_fingerprint` (futuro) | SHA-256 de resource IDs, timestamps, seeds, commit hash. | Auditoria forense não-comparativa. |

### 10.1 Canonicalização de componentes MoVMF

A mistura tem **simetria de permutação**: trocar a ordem dos
componentes não altera a densidade. Para que o
`model_contract_fingerprint` seja invariante a isso, o
`MoVMFTypicalityModel._canonical_component_order()` ordena por chave
lex composta:

1. Peso decrescente.
2. κ decrescente (entre componentes com mesmo peso).
3. Hash lex de μ (`SHA-256(μ.tobytes(order="C"))`).

`to_dict` emite na ordem canônica; fingerprints leem na ordem canônica.
Resultado validado por teste: permutar componentes manualmente não
altera fingerprint.

---

## 11. Migração V2.14 → V2.15

`LegacyStrategyMigrationService` (em
`faceaudit/services/legacy_strategy_migration_service.py`) implementa
**migração preservativa**:

- **Original intocado**: o JSON `.fa` da V2.14 nunca é mutado em disco.
- **Conversão pura em memória**: `LRStrategyDefinition.from_dict()`
  chama `detect()` + `convert_in_memory()` antes do dispatcher polimórfico.
- **Cópia opt-in**: o writer `write_side_by_side_copy()` é acionado
  pela UI/CLI quando o usuário aciona "Salvar cópia convertida". Gera
  `nome.v2_15.fa` lado-a-lado.
- **Modo estrito**: `LRStrategyDefinition.from_dict(data,
  allow_legacy_auto_migration=False)` levanta
  `LegacyStrategyVersionRejected` quando detecta sinal legado. Útil em
  validação formal.

### 11.1 Sinais detectados

| Sinal | Onde aparece | Ação |
|---|---|---|
| `instruction_type = "KNN_TYPICALITY_MODEL_TRAINING"` | Em qualquer fase | Substituído por `"TYPICALITY_MODEL_TRAINING"` + `typicality_model_type = "KNN"` + config k-NN preenchido. |
| `score_type = "LOG_DENSITY_KNN"` | Em qualquer instrução | Substituído por `"LOG_DENSITY_ESTIMATE"` (rename V2.15). |

### 11.2 Defaults históricos materializados

A migração escreve valores **literais** que reproduzem o comportamento
V2.14 bit-a-bit, sem depender de defaults do código novo:

| Campo antigo | Campo novo | Valor literal |
|---|---|---|
| `reference_mode = "PER_FACE"` | `population_weighting_mode` | `"FACE_LEVEL_RAW"` |
| `reference_mode = "PER_IDENTITY"` | `population_weighting_mode` | `"IDENTITY_CENTROIDS"` |
| (ausente) | `unit_norm_policy` | `"NORMALIZE_SILENTLY"` (comportamento k-NN V2.14) |
| (ausente) | `reference_score_calibration_mode` | `"IN_SAMPLE_LEGACY"` (LOO de imagem, não LOIO) |
| (ausente) | `percentile_basis_score_type` | `"MEAN_KNN_SIMILARITY"` (base histórica do k-NN) |
| (ausente) | `cv_n_folds` | `0` (irrelevante em `IN_SAMPLE_LEGACY`) |
| `k`, `ood_quantile`, `population_label`, `embedding_column` | (preservados) | Valor literal do JSON antigo. |

Tabela completa em §B.

### 11.3 Validação prática

No projeto-base de testes (87 estratégias V2.14), 100% das 61
estratégias com `KNN_TYPICALITY_MODEL_TRAINING` foram migradas
automaticamente sem falha, preservando todos os hiperparâmetros e
produzindo modelos com `model_contract_fingerprint` estável entre
execuções repetidas.

---

## 12. Casos de uso comparados

### 12.1 Cenário típico forense — k-NN

Cenário canônico do FaceAudit:

```
typicality_model_type = KNN
population_weighting_mode = IDENTITY_CENTROIDS
unit_norm_policy = ERROR                          (default forense)
reference_score_calibration_mode = IN_SAMPLE_LEGACY
percentile_basis_score_type = MEAN_KNN_SIMILARITY
k = 50
ood_quantile = 0.01
```

Defensável como ranking não-paramétrico ("X é mais típico que 73% da
população de identidades de referência"). Estatística sem suposições
distribucionais. Preserva comportamento histórico V2.14.

### 12.2 Cenário científico-comparativo — MoVMF EXPERIMENTAL

Para ablação k-NN vs MoVMF com o mesmo pipeline ao redor:

```
typicality_model_type = MOVMF
n_components = 1                                  (vMF puro)
centering = "none"
population_weighting_mode = IDENTITY_CENTROIDS    (mesmo que k-NN)
unit_norm_policy = NORMALIZE_SILENTLY             (compat)
reference_score_calibration_mode = CROSS_FIT_IDENTITY
cv_n_folds = 5
percentile_basis_score_type = LOG_DENSITY_ESTIMATE
ood_quantile = 0.001
```

Cross-fit por identidade evita CDF in-sample; refit final em D
completo é o modelo persistido.

### 12.3 Pesquisa de balanceamento — `INVERSE_IDENTITY_FREQUENCY`

Para investigar como a frequência de coleta por identidade afeta a
LR downstream:

```
typicality_model_type = KNN  (ou MOVMF)
population_weighting_mode = INVERSE_IDENTITY_FREQUENCY
```

Cada face entra com peso `1 / n_id_i`. CDF de referência ponderada;
agregação T_k usa pesos renormalizados localmente. EM ponderado no
MoVMF (`n_k = Σ w_i · r_ik`). Detalhes matemáticos em
`docs/typicality_models.md` §4.1 (k-NN) e §4.2 (MoVMF).

---

## 13. Boas práticas

1. **Sempre declare `population_label`** quando há mais de uma
   subpopulação relevante no projeto (frontal vs lateral, etnia,
   faixa etária). O label entra no `model_contract_fingerprint` e
   permite distinguir modelos de mesma config.
2. **Use `UnitNormPolicy.ERROR` em produção**. O default é
   `NORMALIZE_SILENTLY` apenas porque a migração V2.14 precisa
   preservar o comportamento histórico — mas um pipeline novo que
   declara explicitamente o pré-processamento deve falhar se a
   normalização falhou silenciosamente.
3. **Em casework, prefira `IDENTITY_CENTROIDS`**. É o modo defensável
   forense por construção: uma pessoa, uma observação.
4. **Use `CROSS_FIT_IDENTITY` no MoVMF**. O `IN_SAMPLE_LEGACY` está
   ali por compat; novas estratégias devem usar cross-fit para CDF
   defensável.
5. **Justifique `n_components` externamente**. Não compare K na mesma
   execução escolhendo o melhor — vaze BIC ou held-out por
   identidade em estratégias separadas.
6. **Mantenha o mesmo backend em probe e reference**. Comparar k-NN
   vs MoVMF no mesmo par é matematicamente questionável (escalas
   absolutas diferem). Para comparação científica entre backends,
   gere duas estratégias paralelas que diferem apenas no
   `typicality_model_type`.
7. **Não esqueça do filtro de identidade do split**. Faces de
   calibração e de tipicidade não podem compartilhar identidades —
   use `PAIRED_DATASET_SPLIT` antes para garantir disjunção.

---

## 14. Perguntas frequentes

### 14.1 Por que `MoVMF` é EXPERIMENTAL e `k-NN` é STABLE?

Não é matemática — é processo. As 6 validações científicas listadas em
`docs/typicality_models.md` §8 (estabilidade entre seeds, comparação
controlada k-NN × MoVMF, sensibilidade a K, invariância sob duplicação,
etc.) ainda não foram executadas como benchmark formal. O k-NN passou
por validação científica no relatório principal V2.14; o MoVMF é novo
no fluxo declarativo da V2.15.

### 14.2 Posso usar `MoVMF` em casework agora?

Tecnicamente, sim — o pipeline funciona end-to-end. Defensavelmente
em corte, **não** até que as validações sejam concluídas. Trate
`MoVMF` como ferramenta de pesquisa neste estado.

### 14.3 Mudei `unit_norm_policy` na estratégia. Por que `LR_ARCHITECTURE_VECTORIZATION` recusa carregar com INHERIT?

Esperado. `unit_norm_policy` entra no `model_contract_fingerprint`,
que entra no `vectorization_contract_fingerprint` do calibrador
upstream. INHERIT compara fingerprints. Para reusar o calibrador,
mantenha a config idêntica; para mudar, retreine o calibrador.

### 14.4 Posso rodar `CROSS_FIT_IDENTITY` com 3 identidades únicas?

Não. O serviço rejeita `unique_identities < n_folds` com erro
explícito. Para 3 identidades, baixe `cv_n_folds = 3` (a `LeaveOneOut`
por identidade), ou use `IN_SAMPLE_LEGACY`.

### 14.5 Como migro manualmente uma estratégia V2.14?

Normalmente não precisa: abrir o projeto na UI executa a migração
automática. Caso queira ferramenta explícita, use o serviço:

```python
from faceaudit.services.legacy_strategy_migration_service import (
    LegacyStrategyMigrationService,
)
import json

raw = json.loads(Path(".../strategy.fa").read_text())
plan = LegacyStrategyMigrationService.detect(raw)
if plan.is_needed:
    converted = LegacyStrategyMigrationService.convert_in_memory(raw, plan)
    LegacyStrategyMigrationService.write_side_by_side_copy(
        converted, Path(".../strategy.fa"), plan=plan,
    )
```

---

## 15. Glossário

- **Backend.** Implementação concreta do modelo de tipicidade (k-NN,
  MoVMF).
- **Centróide esférico.** Soma de embeddings da identidade
  L2-renormalizada — uma observação por pessoa.
- **CDF empírica.** Distribuição cumulativa dos scores de referência.
- **CROSS_FIT_IDENTITY.** Calibração da CDF via K-fold por identidade
  + refit final.
- **EM ponderado.** Algoritmo EM em que cada amostra entra com peso
  `w_i` em vez de contagem 1.
- **enabled_score_types.** Score types que a instância concreta pode
  produzir agora. Subset do `potential_score_types`.
- **IDENTITY_CENTROIDS.** Modo populacional em que a referência é
  agregada em centróides por identidade.
- **INVERSE_IDENTITY_FREQUENCY.** Modo populacional em que cada face
  entra com peso `1/n_id_i`.
- **IN_SAMPLE_LEGACY.** Calibração da CDF sobre o próprio conjunto de
  fit (LOO de imagem). Otimista; preservado por compat V2.14.
- **LOO.** Leave-one-out (de imagem, no k-NN).
- **LOIO.** Leave-one-identity-out (todas as faces da identidade).
- **MoVMF.** Mistura de von Mises-Fisher na hiperesfera unitária.
- **potential_score_types.** Score types que o backend pode
  conceitualmente produzir (do registry). Não confundir com
  `enabled_score_types`.
- **UnitNormPolicy.** Política do modelo ao receber embeddings com
  norma fora da tolerância.

---

## 16. Referências

- Banerjee, A. et al. (2005). *Clustering on the unit hypersphere
  using von Mises-Fisher distributions.* JMLR.
- Loftsgaarden, D. O., & Quesenberry, C. P. (1965). *A nonparametric
  estimate of a multivariate density function.* AMS.
- Sra, S. (2012). *A short note on parameter approximation for von
  Mises-Fisher distributions.* Computational Statistics.
- FaceAudit interno:
  - `docs/typicality_models.md` (matemática completa dos dois
    backends).
  - `docs/tipicidade_pareada_faceaudit.md`
    (`TYPICALITY_PAIR_SCORING`).
  - `docs/interacao_score_raridade_faceaudit.md`
    (`SCORE_INTERACTION_FEATURE`).
  - `docs/calibracao_faceaudit.md` (calibração final).

---

## Apêndice A — `TypicalityModelTrainingInstruction`

Definição em
`faceaudit/models/instructions/typicality_model_training_instruction.py`.

### Campos

| Campo | Tipo | Default | Notas |
|---|---|---|---|
| `instruction_id` | `str` | (UUID gerado) | Identificador único. |
| `name` | `str` | `"Typicality Model Training"` | Nome amigável. |
| `instruction_type` | `StrategyInstructionType` | `TYPICALITY_MODEL_TRAINING` | Fixado em `__post_init__`. |
| `typicality_model_type` | `TypicalityModelType` | `KNN` | Discriminador do dispatch. |
| `input_dataset_resource_id` | `Optional[str]` | `None` | Recurso `DATASET` de referência populacional. |
| `typicality_model_config` | `Union[KNN, MoVMF]` | Auto-instanciado | Config embutido; auto-instanciado pelo `__post_init__`. |
| `trained_typicality_model_resource_id` | `Optional[str]` | `None` | Recurso de saída `TRAINED_TYPICALITY_MODEL`. |

### Hooks de DTO

- `__post_init__`: fixa `instruction_type`, chama
  `_reconcile_config_type()` (cria config compatível com o tipo
  escolhido), chama `_sync_declared_io()`.
- `_reconcile_config_type()`: se o tipo escolhido é KNN e o config é
  MoVMF (ou None), troca por `KNNTypicalityModelConfig()` default —
  evita estado inconsistente após mudança de tipo na UI.
- `should_rebuild_on_nested_change(attr, dto)`: retorna True quando
  `attr == "typicality_model_config"` (a UI rebuilda o painel quando
  a config interna muda — ex.: mudar
  `reference_score_calibration_mode` esconde/mostra `cv_n_folds`).
- `prune_unused_fields()`: zera `typicality_model_config` quando o
  tipo escolhido não existe (defensivo).

### Form specs para a UI

- `get_choice_field_specs()`: gera dropdown "Tipo do Backend"
  populado por `TypicalityModelType.choices_with_description()`.
- `get_resource_field_specs()`: gera dropdowns "Dataset de
  Referência" (`DATASET`) e "Modelo Treinado"
  (`TRAINED_TYPICALITY_MODEL`, `is_output=True`).
- `get_value_field_specs()`: (nenhum direto; os valores ficam no
  `NestedDTOInlineWidget` do config).

### `from_dict` dispatcher

Hidrata `instruction_type` e `typicality_model_type` via `from_str`;
hidrata `typicality_model_config` despachando entre
`KNNTypicalityModelConfig.from_dict` e
`MoVMFTypicalityModelConfig.from_dict` com base no
`typicality_model_type`. Backward compat tolera campos extintos da
V2.14 (`k`, `ood_quantile`, `reference_mode`, `embedding_column`,
`population_label`) — apenas para JSONs híbridos que escaparam da
migração.

---

## Apêndice B — Tabela de migração V2.14 → V2.15

Lista completa do que o `LegacyStrategyMigrationService` aplica.

### Instruções

| Antes (V2.14) | Depois (V2.15) | Step kind |
|---|---|---|
| `instruction_type = "KNN_TYPICALITY_MODEL_TRAINING"` | `instruction_type = "TYPICALITY_MODEL_TRAINING"` + `typicality_model_type = "KNN"` + `typicality_model_config = {...}` | `instruction_type_rename` |

### Score types (em qualquer instrução)

| Antes | Depois | Step kind |
|---|---|---|
| `score_type = "LOG_DENSITY_KNN"` | `score_type = "LOG_DENSITY_ESTIMATE"` | `score_type_rename` |

### Defaults históricos materializados (config k-NN)

| Campo antigo | Campo novo | Valor literal | Razão |
|---|---|---|---|
| `reference_mode = "PER_FACE"` | `population_weighting_mode` | `"FACE_LEVEL_RAW"` | Equivalência exata. |
| `reference_mode = "PER_IDENTITY"` | `population_weighting_mode` | `"IDENTITY_CENTROIDS"` | Equivalência exata. |
| `reference_mode` ausente | `population_weighting_mode` | `"IDENTITY_CENTROIDS"` | Default V2.14 era `PER_IDENTITY`. |
| (ausente) | `unit_norm_policy` | `"NORMALIZE_SILENTLY"` | Comportamento histórico do k-NN V2.14. |
| (ausente) | `unit_norm_tolerance` | `1e-4` | Default. |
| (ausente) | `reference_score_calibration_mode` | `"IN_SAMPLE_LEGACY"` | LOO de imagem; comportamento V2.14. |
| (ausente) | `percentile_basis_score_type` | `"MEAN_KNN_SIMILARITY"` | Base histórica. |
| (ausente) | `cv_n_folds` | `0` | Irrelevante em IN_SAMPLE_LEGACY. |
| `k` | `k` | (preservado literal) | Hiperparâmetro. |
| `ood_quantile` | `ood_quantile` | (preservado literal) | Hiperparâmetro. |
| `population_label` | `population_label` | (preservado literal) | Diagnóstico. |
| `embedding_column` | `embedding_column` | (preservado literal) | Override. |

### Metadata adicional (auditoria)

A cópia migrada (`*.v2_15.fa`) ganha um bloco
`_migration_metadata` irmão do envelope da estratégia:

```json
{
  "_migration_metadata": {
    "migrated_from_version": "v2_14",
    "migration_version": "v2_15",
    "migration_signals": ["KNN_TYPICALITY_MODEL_TRAINING", "..."],
    "migration_warnings": []
  }
}
```

O loader ignora esse campo no `cls(**processed_data)` (filtrado
explicitamente). Existe apenas para auditoria humana.

---

## Apêndice C — Checklist operacional

Antes de declarar uma `TYPICALITY_MODEL_TRAINING` válida:

- [ ] Dataset de referência **já estratificado** pela subpopulação
      relevante (frontal, etnia, faixa etária). A instrução não
      estratifica internamente.
- [ ] Dataset com identidade definida em cada face quando
      `population_weighting_mode != FACE_LEVEL_RAW`.
- [ ] `unit_norm_policy = ERROR` quando o pipeline upstream
      garante L2-normalização. Caso contrário, justifique
      `NORMALIZE_*` no campo `notes` da estratégia.
- [ ] `embedding_column` consistente com os datasets downstream
      (`TYPICALITY_PAIR_SCORING` precisa ler a mesma coluna).
- [ ] `k` (k-NN) ou `n_components` (MoVMF) justificado
      externamente — não por escolha pós-hoc no caso atual.
- [ ] Faces de identidade da referência **disjuntas** das de
      calibração e validação. Garanta via `PAIRED_DATASET_SPLIT`
      prévio.
- [ ] `MIN_SAMPLES` e `MAX_SAMPLES_WARNING` revisados (default
      `min_required = max(k+1, 2)`, `safety_threshold = 1000`,
      `max_samples = 50000`).
- [ ] `population_label` preenchido quando há múltiplas
      subpopulações no projeto (frontal vs lateral).

Em casework forense, adicione:

- [ ] Modelo trinado com `unit_norm_policy = ERROR` (rastreabilidade).
- [ ] `reference_score_calibration_mode = CROSS_FIT_IDENTITY`
      (MoVMF) ou `IN_SAMPLE_LEGACY` (k-NN com LOO nativo).
- [ ] `model_contract_fingerprint` registrado no laudo para
      comparação com a re-execução.

---

**Última revisão:** V2.15 (2026-06-08).
