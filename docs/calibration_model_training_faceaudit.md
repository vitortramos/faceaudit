# Treinamento de modelo de calibração (`CALIBRATION_MODEL_TRAINING`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `CALIBRATION_MODEL_TRAINING` — treino e
> preparação de um modelo de calibração que converte score bruto em
> log-LR. Cobre as três famílias suportadas (Regressão Logística,
> KDE, Bi-Gaussianizado), o config aninhado de cada uma, os modos de
> GridSearchCV para `C` em LogReg, a expansão piecewise-linear interna
> V1, o caminho `FIT_PER_FOLD` integrado ao
> `LR_ARCHITECTURE_VECTORIZATION`, a propagação de fingerprints e
> `architecture_state` via `CalibrationModelMetadataService`, a
> telemetria Cllr in-sample / CV diagnóstico, e a auditoria
> `CONDITIONAL_MONOTONICITY` programática V1.5 / V2.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Documentos relacionados:**
> `docs/calibracao_faceaudit.md` (referência guarda-chuva sobre
> calibração),
> `docs/piecewise_linear_calibration.md` (especificação detalhada da
> expansão piecewise-linear interna),
> `docs/conditional_monotonicity_rule_design.md` (auditoria forense
> do calibrador V1/V1.5/V2),
> `docs/lr_architecture_vectorization_faceaudit.md` (substituto
> declarativo da família LR consumido em modo `INHERIT` por validação
> e casework),
> `docs/score_calibration_faceaudit.md` (instrução irmã que aplica o
> calibrador a um dataset de scores brutos).
> **Fontes-código:** `faceaudit/models/instructions/calibration_model_training_instruction.py`,
> `faceaudit/executors/instructions/calibration_model_training_instruction_executor.py`,
> `faceaudit/models/calibration/logistic_regression_calibration_model_config.py`,
> `faceaudit/models/calibration/kde_calibration_model_config.py`,
> `faceaudit/models/calibration/bi_gaussianized_calibration_model_config.py`,
> `faceaudit/models/calibration/piecewise_linear_calibration_config.py`,
> `faceaudit/services/calibration_model_metadata_service.py`,
> `faceaudit/services/architecture_vectorizer_transformer.py`,
> `faceaudit/services/identity_disjoint_pair_kfold.py`,
> `faceaudit/services/calibrator_cross_validation_service.py`,
> `faceaudit/services/cllr_math.py`,
> `faceaudit/enums/calibration_model_type.py`,
> `faceaudit/enums/logreg_gridsearch_cv_mode.py`,
> `faceaudit/enums/logreg_grid_scoring_metric.py`,
> `faceaudit/enums/calibrator_cv_split_mode.py`,
> `faceaudit/enums/calibration_basis_mode.py`,
> `faceaudit/enums/hinge_anchor_mode.py`,
> `faceaudit/enums/knot_placement.py`,
> `faceaudit/enums/knot_scale.py`,
> `faceaudit/enums/monotonicity_policy.py`,
> `faceaudit/validation_rules/conditional_monotonicity_rule.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a instrução se encaixa](#1-onde-a-instrução-se-encaixa)
- [2. Configuração do DTO](#2-configuração-do-dto)
- [3. As três famílias suportadas](#3-as-três-famílias-suportadas)
- [4. Regressão Logística — `LogisticRegressionCalibrationModelConfig`](#4-regressão-logística--logisticregressioncalibrationmodelconfig)
- [5. GridSearchCV interno para `C`](#5-gridsearchcv-interno-para-c)
- [6. Standardização de features (`standardize_features`)](#6-standardização-de-features-standardize_features)
- [7. Expansão piecewise-linear interna V1](#7-expansão-piecewise-linear-interna-v1)
- [8. KDE — `KDECalibrationModelConfig`](#8-kde--kdecalibrationmodelconfig)
- [9. Bi-Gaussianizado — `BiGaussianizedCalibrationModelConfig`](#9-bi-gaussianizado--bigaussianizedcalibrationmodelconfig)
- [10. Caminho `FIT_PER_FOLD`](#10-caminho-fit_per_fold)
- [11. Vínculo dataset ↔ calibrador (`CalibrationModelMetadataService.bind_architecture_state`)](#11-vínculo-dataset--calibrador-calibrationmodelmetadataservicebind_architecture_state)
- [12. Telemetria Cllr in-sample e CV diagnóstico](#12-telemetria-cllr-in-sample-e-cv-diagnóstico)
- [13. Auditoria de monotonicidade pós-refit (V1.5 / V2)](#13-auditoria-de-monotonicidade-pós-refit-v15--v2)
- [14. Regras de validação padrão](#14-regras-de-validação-padrão)
- [15. Casos de uso](#15-casos-de-uso)
- [16. Boas práticas](#16-boas-práticas)
- [17. Perguntas frequentes](#17-perguntas-frequentes)
- [18. Glossário](#18-glossário)
- [19. Referências](#19-referências)
- [Apêndice A — `CalibrationModelTrainingInstruction`](#apêndice-a--calibrationmodeltraininginstruction)
- [Apêndice B — Tabela comparativa das três famílias](#apêndice-b--tabela-comparativa-das-três-famílias)
- [Apêndice C — Checklist operacional](#apêndice-c--checklist-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre `CALIBRATION_MODEL_TRAINING`:
qual modelo escolher entre as três famílias, como configurar cada
uma, como o executor materializa o calibrador, como o GridSearchCV
interno seleciona `C` sob restrição de monotonicidade, como o
caminho `FIT_PER_FOLD` integra com `LR_ARCHITECTURE_VECTORIZATION`
em modo `IdentityDisjointPairKFold`, como `architecture_state` e os
fingerprints viajam para o calibrador, como a telemetria de Cllr é
anexada ao metadata, e como a auditoria pós-refit
`CONDITIONAL_MONOTONICITY` valida o calibrador resultante.

### 0.2 Por que existe

Score bruto não é evidência. O perito reporta um log-LR — razão
entre as probabilidades do score sob `H_p` (mesma identidade) e
`H_d` (identidades distintas). Score cosseno em [-1, +1] ou PLDA em
unidades arbitrárias precisa ser convertido por um modelo treinado
no domínio de calibração.

`CALIBRATION_MODEL_TRAINING` é a instrução que ajusta esse modelo. O
calibrador resultante é serializado, vinculado ao contrato de
vetorização que o produziu (via fingerprints), e aplicado pela
instrução `SCORE_CALIBRATION` a datasets de validação e casework.

### 0.3 Compatibilidade

A V2.14 não obriga migração para arquiteturas declarativas. Quatro
caminhos de uso coexistem:

1. **Calibrador legado** — usuário treina LogReg/KDE/BiGauss
   diretamente sobre score 1D (cosseno ou PLDA escalar), sem
   passar por `LR_ARCHITECTURE_VECTORIZATION`. Funciona
   exatamente como antes — nenhum fingerprint é injetado.
2. **Calibrador sobre vetor** — `SCORE_VECTORIZATION` produz um
   dataset `(N, D)`; o calibrador trata `D > 1` com
   `standardize_features=True`. Compatível com a era
   pré-`LR_ARCHITECTURE_VECTORIZATION`.
3. **Calibrador sobre arquitetura declarativa em
   `FIT_FULL_CALIBRATION_LEGACY`** — `LR_ARCHITECTURE_VECTORIZATION`
   materializa a matriz com s₀/r̄ ajustados em todo D_cal antes da CV
   do calibrador. Funciona, mas tem leakage residual conhecido. O
   calibrador recebe o `vectorization_contract_fingerprint` da
   vetorização e o propaga via `bind_architecture_state`.
4. **Calibrador sobre arquitetura declarativa em `FIT_PER_FOLD`**
   (V2.14, recomendado) — `LR_ARCHITECTURE_VECTORIZATION` emite o
   `LR_ARCHITECTURE_RAW_INPUT` cru e marca o dataset vetorizado com
   `defer_to_calibrator_per_fold=True`. O executor de calibração
   detecta a flag e monta a `Pipeline([ArchitectureVectorizerTransformer
   → StandardScaler → LogReg])` dentro de
   `GridSearchCV(cv=IdentityDisjointPairKFold(...))` — s₀/r̄ são
   ajustados POR FOLD, eliminando o leakage residual. O state final
   reflete o refit em D_cal completo.

---

## 1. Onde a instrução se encaixa

```
[Modelo de similaridade]
        │
        ▼
SIMILARITY_SCORING(probe_dataset, reference_dataset, model)
        │
        ▼
SIMILARITY_SCORES (raw_score por par, same_person rotulado)
        │
        │  ── opcionalmente passa por:
        │     • SCORE_INTERACTION_FEATURE (legado, V2.13)
        │     • SCORE_VECTORIZATION (legado)
        │     • LR_ARCHITECTURE_VECTORIZATION (V2.14 — recomendado)
        ▼
SIMILARITY_SCORES (vetorizado, com `vectorization_contract_fingerprint`)
        │
        ▼
CALIBRATION_MODEL_TRAINING ← você está aqui
   ├── escolhe família (LOGISTIC_REGRESSION / KDE / BI_GAUSSIANIZED)
   ├── ajusta hyperparams via GridSearchCV (LogReg / KDE)
   ├── opcional: piecewise-linear interna (LogReg V1)
   ├── opcional: FIT_PER_FOLD via Pipeline com transformer interno
   ├── computa Cllr in-sample + (opt-in) Cllr CV diagnóstico
   ├── vincula architecture_state + fingerprints via bind_architecture_state
   └── audita monotonicidade pós-refit via CONDITIONAL_MONOTONICITY
        │
        ▼
TRAINED_CALIBRATION_MODEL (pickle)
        │
        ▼
SCORE_CALIBRATION(input_scores, trained_model)
        │
        ▼
CALIBRATED_SCORES (log_lr por par)
        │
        ▼
STRATEGY_EVALUATION → métricas forenses
```

### 1.1 Entrada

Um `StrategyResource` do tipo `SIMILARITY_SCORES` (ou genérico
`ARTIFACT`) contendo:

- `SimilarityScoreDataset` com `raw_score` por par, `same_person`
  rotulado (1 = genuíno, 0 = impostor), e as identidades de cada
  amostra (`probe.identity` / `reference.identity`), que o executor
  extrai e injeta no calibrador como `train_probe_identities` /
  `train_reference_identities` para o splitter do GridSearch e os
  diagnósticos por trecho;
- `metadata` com:
  - `vectorization_contract_fingerprint` (quando vetorizado por
    `LR_ARCHITECTURE_VECTORIZATION` ou `SCORE_VECTORIZATION`);
  - `architecture_state` (quando vetorizado em
    `FIT_FULL_CALIBRATION_LEGACY` ou produzido por
    `LR_ARCHITECTURE_VECTORIZATION` em `FIT_PER_FOLD`);
  - `defer_to_calibrator_per_fold=True` + `raw_input_resource_id`
    (quando vetorizado em `FIT_PER_FOLD`);
  - `score_vector_model_ids` (quando produzido por
    `SCORE_VECTORIZATION`, lista ordenada dos sistemas-fonte).

### 1.2 Saída

Um `StrategyResource` do tipo `TRAINED_CALIBRATION_MODEL` contendo
o calibrador serializado em pickle. O modelo expõe:

- `calibrate_scores(raw) → log_lr` (interface
  `CalibrationModelProtocol`);
- `metadata` com Cllr in-sample, opcionalmente Cllr CV, melhor
  `C`/bandwidths, e — quando aplicável — `vectorization_contract_fingerprint`,
  `architecture_state` completo, `raw_scale_coefficients`,
  `monotonicity_audit`.

### 1.3 Fluxo de execução

```
execute(instruction, context)
  ├── validate(instruction, context)                   (regras pré-execução)
  │
  ├── _extract_scores_and_labels(input_payload)        (scores 1D ou matriz)
  ├── _extract_identities(input_payload)               (probe/reference ids)
  │
  ├── per_fold_active?
  │     └── _train_logreg_per_fold_with_transformer    (FIT_PER_FOLD)
  ├── senão se LOGISTIC_REGRESSION:
  │     └── _train_logistic_regression_calibrator      (caminho clássico)
  ├── senão se KDE:
  │     └── _train_kde_calibrator
  ├── senão se BI_GAUSSIANIZED:
  │     └── _train_bigauss_calibrator
  │
  ├── _attach_calibrator_telemetry                     (Cllr train + CV opt-in)
  ├── _gate_calibrator_capacity                        (capacidade vs D)
  ├── CalibrationModelMetadataService.bind_architecture_state
  │                                                    (vincula contrato)
  ├── _bind_training_raw_input                         (resolve raw_input)
  │
  ├── context.set_resource(TRAINED_CALIBRATION_MODEL)
  │
  └── _run_monotonicity_audit                          (V1.5 — pós-refit)
        └── persiste monotonicity_audit no metadata
```

---

## 2. Configuração do DTO

`CalibrationModelTrainingInstruction` (apêndice A) tem seis campos
relevantes:

| Campo | Tipo | Default | Função |
|---|---|---|---|
| `calibration_model_type` | `CalibrationModelType` | `LOGISTIC_REGRESSION` | Família escolhida |
| `input_resource_id` | `str` | None | Recurso de scores brutos |
| `logreg_config` | `LogisticRegressionCalibrationModelConfig` | auto-instanciado | Hyperparâmetros LogReg |
| `kde_config` | `KDECalibrationModelConfig` | auto-instanciado | Hyperparâmetros KDE |
| `bigauss_config` | `BiGaussianizedCalibrationModelConfig` | auto-instanciado | Hyperparâmetros BiGauss |
| `trained_calibration_model_resource_id` | `str` | None | Recurso de saída |

O `__post_init__` instancia automaticamente o config correspondente
ao `calibration_model_type` escolhido — o usuário não precisa criar
o objeto manualmente. Mudar de família mantém os configs antigos no
DTO (útil para alternar sem perder valores), mas `prune_unused_fields`
(chamado pelo "Apply" do editor de estratégia) zera os configs
inativos antes de persistir o JSON.

### 2.1 As três propriedades booleanas

O DTO expõe `uses_logistic_regression`, `uses_kde`, `uses_bigauss`
para o executor desambiguar a família sem comparar enums
diretamente. Cada propriedade reflete `calibration_model_type`.

---

## 3. As três famílias suportadas

| Família | Quando usar | Vantagens | Limitações |
|---|---|---|---|
| **LOGISTIC_REGRESSION** (default) | Default operacional. Robusto em datasets pequenos a grandes. Defensável forensicamente — parcimonioso, monotonicidade testável. | Único que suporta multi-D, FIT_PER_FOLD, piecewise-linear, GridSearch de C com `IdentityDisjointPairKFold`. | Modelo afim ou afim-piecewise — não captura curvatura complexa. |
| **KDE** | Calibração não-paramétrica univariada. Útil quando a distribuição de scores tem múltiplas modas ou caudas pesadas e há ≥ 50k pares por classe. | Mais flexível que LogReg em score 1D. Adapta-se a distribuições não-gaussianas. | Alta capacidade → overfitting fácil. Cllr CV − in-sample tende a ser grande. Não suporta multi-D nem FIT_PER_FOLD. |
| **BI_GAUSSIANIZED** | Calibração 1D pelo algoritmo de Morrison (2024): mapeia a saída de um calibrador inicial para um sistema bi-gaussiano alvo. | Sem GridSearch de `C`; alvo paramétrico com σ² derivado do Cllr ou do EER. | A CDF empírica das classes é sensível ao N da amostra nas caudas. Apenas score 1D. |

A escolha é declarada via `calibration_model_type`. O default
`LOGISTIC_REGRESSION` é o caminho mais defensável forensicamente
quando combinado com `FIT_PER_FOLD` e a auditoria
`CONDITIONAL_MONOTONICITY` pós-refit.

---

## 4. Regressão Logística — `LogisticRegressionCalibrationModelConfig`

### 4.1 Hiperparâmetro central — `C`

`C` é o inverso da regularização L2 do sklearn. Valor padrão `1.0`
é arbitrário e raramente ótimo; ative `use_gridsearch=True`
(default) para buscar o melhor `C` na grade `c_grid = [0.01, 0.1,
1.0, 10.0, 100.0]` (default). A UI aceita string
`"0.01,0.1,1.0,10.0"` por conveniência — o `from_dict` faz parse
em lista de floats.

### 4.2 Função objetivo do GridSearch — `scoring`

Default `LogRegGridScoringMetric.NEG_CLLR_BALANCED` (V2.6).
Equivalente a `−Cllr` com peso agregado igual entre `H_p` e `H_d`,
independente do número de pares em cada classe. Recomendado para
calibração forense em datasets desbalanceados (`N_Hd >> N_Hp`).

Alternativas:
- `NEG_LOG_LOSS` — média por amostra. Dominada pela classe
  majoritária quando `N_Hd >> N_Hp`. Selecionava `C` subótimo nos
  datasets do projeto pré-V2.6.
- `ROC_AUC` — otimiza separabilidade/ranking, não calibração.
  Inadequado como métrica de calibração.

### 4.3 Standardização — `standardize_features`

Default `False`. Quando `True` E o input `X` for matriz `(N, D)`
com `D > 1`, aplica z-score por feature antes do LogReg.
Necessário em score vectorization quando os sistemas têm escalas
dispares (ex.: Cosseno em [−0.5, +0.5] + PLDA em [−200, +60]).
Sem standardização, a L2 do sklearn penaliza pesos uniformemente,
distorcendo a contribuição relativa das features.

Para `X` 1D (calibração univariada normal), o flag é ignorado —
LogReg vê o score direto.

`μ`/`σ` são persistidos com o modelo em `_feature_mean`/
`_feature_std` e aplicados automaticamente em `score_calibration`
(treino/validação/casework).

### 4.4 CV diagnóstico — `k_folds`

Default `0` (desativado). Quando `≥ 2` (recomendado: 5), roda
folds estratificados após o fit principal usando os hiperparâmetros
do production model (`C` congelado). Anexa `calibrator_cllr_cv`,
`calibrator_cllr_cv_gen`, `calibrator_cllr_cv_imp`,
`calibrator_cv_gap` ao metadata. O Cllr CV é diagnóstico de
generalização DENTRO do domínio de calibração — não substitui
validação externa.

`cv_split_mode`:
- `STRATIFIED_PAIR` (default) — `sklearn.StratifiedKFold`.
  Preserva proporção genuíno/impostor. Pares da mesma identidade
  podem aparecer em ambos os folds → otimista.
- `STRATIFIED_GROUP` — `sklearn.StratifiedGroupKFold` por
  `probe_identity`. Nenhuma identidade de probe usada no treino do
  fold aparece no held-out. Forense-defensável, mais pessimista,
  mais honesto. Recomendado em laudos.

> **Atenção** — `k_folds` é o **CV diagnóstico** opt-in
> (independente do GridSearch). Não confundir com `cv_folds` (folds
> do GridSearch interno para `C`).

### 4.5 GridSearch CV mode — `gridsearch_cv_mode`

Default `LogRegGridSearchCVMode.AUTO`. Governa o splitter usado
pelo `GridSearchCV` interno do LogReg na seleção de `C`. Detalhes
na seção 5.

---

## 5. GridSearchCV interno para `C`

### 5.1 Os quatro modos do `gridsearch_cv_mode`

| Modo | Splitter | Cobertura | Quando usar |
|---|---|---|---|
| `AUTO` (default) | `GROUPED_PROBE` quando há `train_probe_identities` no metadata; senão `PAIR` | Variável | Padrão operacional — escolhe o melhor splitter disponível em runtime |
| `PAIR` | `sklearn.StratifiedKFold` | Nenhuma disjunção | Apenas como ablação; otimista |
| `GROUPED_PROBE` | `sklearn.StratifiedGroupKFold` por `probe_identity` | Probe disjunto, reference pode cruzar | Default V2.8 — bom equilíbrio entre cobertura e custo |
| `STRICT_IDENTITY` | `IdentityDisjointPairKFold` (custom) | Total — folds disjuntos dos dois lados | V2.10+ — defensável forensicamente quando há ≳ 50 identidades em D_cal |

### 5.2 `IdentityDisjointPairKFold` (V2.10)

Splitter customizado que particiona identidades únicas
(`probe ∪ reference`) em K grupos balanceados via LPT scheduling
(Graham 1969) ou round-robin legado. Em cada fold:

- `test = pares com ambas as ids no grupo g`;
- `train = pares com ambas as ids fora de g`;
- **boundary** (uma id dentro, outra fora) **excluído apenas
  daquele fold** — não globalmente, preservando ≈((K−1)/K)² do
  `H_d` no treino.

Aceita `lookalike_clusters` opcional (V2.11.2): identidades de um
cluster ficam no mesmo fold como super-id, com carga LPT agregada.

V2.13: `STRICT_IDENTITY` aplica-se também a entradas multivariadas
(`SCORE_VECTORIZATION` com D > 1) — o splitter é dimension-agnostic,
opera só sobre identidades. O fallback automático para
`GROUPED_PROBE` em multivariado foi removido.

V2.13: quando `standardize_features=True` e D > 1, a
`StandardScaler` é embutida na `sklearn.Pipeline` `[StandardScaler
→ LogReg]` usada como estimator do `GridSearchCV` — `μ`/`σ` por
feature são aprendidos por fold (sem leakage); refit em D_cal
popula `_feature_mean`/`_feature_std` para inferência.

### 5.3 Persistência de proveniência da CV

O calibrador persiste no metadata:

- `cv_split_mode_requested` / `cv_split_mode_applied` /
  `cv_split_mode_fallback_reason`;
- `cv_identity_disjointness_level` (enum textual: `full` /
  `probe_only` / `none`);
- `cv_identity_disjointness_verified` (bool);
- `cv_strict_*` — mapping id→fold, balance loads, contagens por
  fold, `fold_health`.

Esses campos ficam no metadata do calibrador (pickle), para auditoria.
Não são propagados por `SCORE_CALIBRATION` (que repassa só a
telemetria de Cllr, §12.3) nem exibidos pelo `ResourceInfoWidget`.

### 5.4 Seleção de `C` sob restrição de monotonicidade

Só no caminho piecewise-linear (`piecewise_linear_config.mode =
PIECEWISE_LINEAR`, §7). `_select_monotonic_C` (em
`LogisticRegressionCalibrationModel`) roda `GridSearchCV(refit=False)`
e itera o grid em ordem decrescente por CV score. Para cada `C` faz
fit completo da pipeline e verifica crescimento estrito por trecho
(`_compute_pipeline_slopes`, slope > 1e-8). O primeiro `C` válido é
o escolhido — `piecewise_selection_policy =
"best_valid_candidate_by_cv_score"`. O fit final passa ainda por
`_validate_monotonicity_and_persist_diagnostics`.

> Garante que o calibrador final é monotônico na escala bruta,
> mesmo quando o `C` com melhor CV score produz curva não
> monotônica (situação típica em datasets ruidosos onde a
> regularização mais forte é preferível).

No LogReg afim (sem piecewise) o `GridSearchCV` usa `refit=True` e
escolhe `C` só pelo CV score; não há filtro de monotonicidade.

---

## 6. Standardização de features (`standardize_features`)

### 6.1 Quando é necessária

Em score vectorization com `D > 1` e sistemas com escalas
dispares. Sem padronização, a L2 do sklearn favorece coordenadas
de maior magnitude — a feature de "PLDA escalada por 100" recebe
peso desproporcional sobre "cosseno em [-0.5, +0.5]".

### 6.2 Como funciona

O fit aprende `μ`, `σ` por feature em D_cal. A predição transforma
`X_test = (X − μ) / σ` antes de aplicar o LogReg. `μ`/`σ` são
persistidos no calibrador.

Desde a V2.13, com multi-D + `standardize_features=True` (em qualquer
`gridsearch_cv_mode`), a `StandardScaler` vai **dentro** da
`sklearn.Pipeline` usada como estimator do `GridSearchCV`. Cada fold
aprende seu próprio `μ`/`σ` em `X_train`, evitando leakage da
média/desvio do held-out. Sem GridSearch a mesma Pipeline é usada no
fit único.

### 6.3 Auditoria

O calibrador persiste `_feature_mean` / `_feature_std`. O executor
de `SCORE_CALIBRATION` consulta esses atributos para denormalizar
coeficientes z-score em escala bruta (`calibrator_coef_raw`,
`calibrator_intercept_raw`), propagados ao `STRATEGY_EVALUATION`
para o plot 2D analítico.

---

## 7. Expansão piecewise-linear interna V1

### 7.1 Quando ativar

Quando o score bruto tem **comportamento bilateral assimétrico**:
acima de um ponto neutro `s₀`, score alto é evidência crescente
para `H_p`; abaixo, score baixo é evidência para `H_d`. LogReg afim
modela isso linearmente; piecewise-linear introduz uma quebra em `s₀`
que permite slope distinto nos dois lados.

Documentação canônica: `docs/piecewise_linear_calibration.md`.

### 7.2 Configuração via `piecewise_linear_config`

`PiecewiseLinearCalibrationConfig` é um sub-DTO embutido em
`LogisticRegressionCalibrationModelConfig.piecewise_linear_config`.
Default `mode = NONE` (backward-compat: calibrador opera direto
sobre o score sem expansão).

Quando `mode = PIECEWISE_LINEAR`:

```
z = score − hinge_anchor_s0
features = [z, max(z − offset_j, 0)  for offset_j in knot_offsets]
```

O calibrador monta `sklearn.Pipeline([PiecewiseLinearScoreTransformer
→ StandardScaler → LogReg])` e aprende `hinge_anchor_s0` **fold-aware**
no GridSearchCV — sem leakage.

### 7.3 Campos principais

| Campo | Default | Descrição |
|---|---|---|
| `mode` | `NONE` | `NONE` (off) ou `PIECEWISE_LINEAR` (on). |
| `hinge_anchor_mode` | `AUTO_LOGREG` | Como `s₀` é resolvido — auxiliar LogReg univariado (fold-aware) ou MANUAL. |
| `hinge_anchor_manual_value` | None | Valor manual de `s₀` quando `MANUAL`. |
| `knot_offsets` | `[0.0]` | Offsets das dobras em relação a `s₀`, na unidade definida por `knot_scale`. |
| `knot_scale` | `AUX_LOG_LR` | `AUX_LOG_LR` (unidades de log-LR do auxiliar) ou `RAW` (domínio do score). |
| `knot_placement` | `MANUAL` | `MANUAL` ou `AUTO_FROM_MIRRORED_MEDIANS` (heurística experimental — vide `docs/piecewise_linear_calibration.md` §6.2). |
| `require_monotonicity` | `True` | Valida que todas as inclinações cumulativas na escala bruta são > `min_allowed_slope`. |
| `monotonicity_policy` | `ERROR` | `ERROR` (rejeita) ou `WARNING` (alerta com diagnóstico). |
| `min_allowed_slope` | `1e-8` | Inclinação mínima aceita por trecho. |

### 7.4 Restrição V1

Aplicável apenas a score 1D (cosseno escalar, PLDA escalar).
Calibrador multivariado (`SCORE_VECTORIZATION` com `D > 1`)
rejeita `mode = PIECEWISE_LINEAR` — usar `SCORE_INTERACTION_FEATURE`
externa ou `LR_ARCHITECTURE_VECTORIZATION` declarativa.

### 7.5 Validação de monotonicidade

Após o fit final, `_validate_monotonicity_and_persist_diagnostics`
verifica que `∂log_LR/∂score_bruto > min_allowed_slope` em todos
os trechos. Se `require_monotonicity=True` e algum trecho falha:
- `monotonicity_policy=ERROR` → rejeita e levanta exceção
  (default operacional);
- `monotonicity_policy=WARNING` → permite com alerta forte (apenas
  para diagnóstico em desenvolvimento).

### 7.6 Persistência

O calibrador armazena no metadata:

- `piecewise_enabled = True`;
- `hinge_anchor_s0` (aprendido);
- `piecewise_knot_offsets`, `piecewise_knot_offsets_user`,
  `piecewise_knot_offsets_raw`, `piecewise_knot_scale`,
  `piecewise_knot_placement`;
- `piecewise_knots_absolute` (posições absolutas);
- `segment_slopes_raw` (inclinações por trecho);
- `piecewise_auto_log_lr_hp_p50`, `piecewise_auto_log_lr_hd_p50`
  (medianas log-LR do auxiliar — usadas pelo placement
  `AUTO_FROM_MIRRORED_MEDIANS`);
- `calibrator_coef_raw`, `calibrator_intercept_raw` (coeficientes
  pós-denormalização da `StandardScaler` interna — usado pelo plot
  da `STRATEGY_EVALUATION`).

---

## 8. KDE — `KDECalibrationModelConfig`

### 8.1 Quando usar

Calibração não-paramétrica univariada. Apropriado quando:
- distribuição de scores tem múltiplas modas ou caudas pesadas que
  LogReg afim não captura;
- há ≥ 50k pares por classe (KDE é alta-capacidade, requer dados);
- não há restrição de monotonicidade dura (KDE não a impõe
  estruturalmente).

### 8.2 Hiperparâmetros principais

| Campo | Default | Descrição |
|---|---|---|
| `bandwidth_genuine` | `0.1` | Bandwidth do KDE para scores genuínos. |
| `bandwidth_impostor` | `0.1` | Bandwidth do KDE para scores impostores. |
| `use_gridsearch` | `True` | Quando `True`, busca o melhor bandwidth via GridSearch + log-likelihood out-of-fold. |
| `max_samples_for_gridsearch` | `10000` | Subsampling no GridSearch para acelerar. |
| `max_stored_samples` | `50000` | Subsampling no fit final do KDE. |
| `kernel` | `gaussian` | Kernel do `sklearn.neighbors.KernelDensity`. |
| `min_log_lr` | `-20.0` | Limite inferior do log-LR calibrado. |
| `max_log_lr` | `+20.0` | Limite superior do log-LR calibrado. |
| `interpolation_points` | `10000` | Pontos da curva interpolada `score → log-LR`. |
| `max_grid_expansions` | `2` | Tentativas de expandir o grid se o melhor cair na borda. |
| `min_bandwidth_abs` | `1e-3` | Bandwidth mínimo absoluto no espaço transformado. |

### 8.3 Normalização z-score interna

Os scores genuínos e impostores são normalizados por z-score
(`(x − μ) / σ`, com `μ`/`σ` aprendidos sobre todos os scores de
treino concatenados) antes do KDE — funciona para qualquer escala de
score, sem assumir o intervalo [−1, 1] que a transformação de Fisher
usada em versões antigas exigia. `μ`/`σ` são persistidos e aplicados
em validação/casework (`metadata["transform"] = "z_score"`). O log-LR
resultante é computado como `log(f_gen / f_imp)` e **clipado** em
`[min_log_lr, max_log_lr]` para evitar log-LR explosivo nas caudas
onde a densidade KDE é numericamente nula.

### 8.4 CV diagnóstico — `k_folds`

Mesmo padrão do LogReg: `0` (desativado) ou `≥ 2` (recomendado:
5). Cllr CV − in-sample tende a ser **maior** que LogReg —
calibrador de alta capacidade overfita mais. Use o gap como
indicador de generalização limitada.

### 8.5 Limitações

- Apenas score 1D — não suporta multi-D nem matriz `(N, D)`.
- Não suporta `FIT_PER_FOLD` nem `LR_ARCHITECTURE_VECTORIZATION`.
- Não persiste `architecture_state` — a auditoria
  `CONDITIONAL_MONOTONICITY` pula silenciosamente.

---

## 9. Bi-Gaussianizado — `BiGaussianizedCalibrationModelConfig`

### 9.1 Quando usar

Calibração 1D quando se quer que a saída se comporte como a de um
sistema bi-gaussiano perfeitamente calibrado, sem GridSearch de `C`.
Apropriado quando há amostra de treino representativa e grande o
bastante para que a CDF empírica das classes esteja bem resolvida
nas caudas.

### 9.2 Modelo

`BiGaussianizedCalibrationModel` implementa o algoritmo de Morrison
(2024): (1) aplica um calibrador inicial (LogReg, KDE ou nenhum,
quando os scores já são tratados como log-LR); (2) calcula o Cllr ou
o EER dessa saída e resolve o σ² do sistema bi-gaussiano alvo;
(3) constrói a CDF G(x) das duas classes com pesos 0,5/(N_d+1) e
0,5/(N_p+1); (4) mapeia ln(LR) = F⁻¹(G(x)), onde F é a mistura
0,5·N(+σ²/2, σ²) + 0,5·N(−σ²/2, σ²).

### 9.3 Hiperparâmetros principais

Consultar `bi_gaussianized_calibration_model_config.py` para a
lista completa. Campos principais: `target_metric` (CLLR default, ou
EER), `initial_calibrator` (LOGREG default, KDE ou NONE),
`sigma_estimation_mode` (CROSS_FITTED default, ou IN_SAMPLE),
`sigma_shrinkage_factor`, `gx_method` (KDE_SMOOTHED default, ou
EMPIRICAL_LINEAR, a do artigo), `inversion_method` (INTERP1D ou
BRENT_ROOT), clipping opcional de log-LR (`enable_clipping`, default
`False`, com `min_log_lr`/`max_log_lr`) e CV diagnóstico (`k_folds`).

### 9.4 Limitações

Mesmas do KDE — apenas score 1D, não suporta multi-D nem
`FIT_PER_FOLD`, não persiste `architecture_state`.

---

## 10. Caminho `FIT_PER_FOLD`

### 10.1 Quando é ativado

Detecção automática no executor (`per_fold_active`):

```python
per_fold_active = (
    instruction.uses_logistic_regression
    and bool(input_meta.get("defer_to_calibrator_per_fold"))
    and input_meta.get("raw_input_resource_id")
)
```

Acontece quando:
- O modelo é `LOGISTIC_REGRESSION`;
- O input foi produzido por `LR_ARCHITECTURE_VECTORIZATION` em
  modo `FIT_PER_FOLD` (que marca o dataset com a flag e o
  resource_id do raw_input cru).

### 10.2 O que o executor faz

`_train_logreg_per_fold_with_transformer` (l. 630–832):

1. **Carrega o `LR_ARCHITECTURE_RAW_INPUT`** referenciado por
   `raw_input_resource_id` — matriz `(N, K)` em `<f8`, `same_person`,
   `probe_identity_ids`, `reference_identity_ids`, `schema`.
2. **Reconstrói o `LRArchitectureSpec`** da arquitetura herdada do
   metadata (`lr_architecture`).
3. **Recupera proveniência de tipicidade** do `architecture_state`
   anterior — `pair_role_orientation`, `base_score_kind`,
   `rarity_*_score_type`, `s0_logreg_config`.
4. **Monta a Pipeline**:
   ```python
   Pipeline([
       ("arch_vec", ArchitectureVectorizerTransformer(...)),
       ("scaler",   StandardScaler()),
       ("logreg",   LogisticRegression(solver=..., class_weight=...)),
   ])
   ```
   Como `LogisticRegressionCalibrationModelConfig` não tem campos
   `solver`/`class_weight`/`max_iter`, valem os defaults do executor:
   `solver="liblinear"`, `class_weight="balanced"`, `max_iter=1000`.
5. **Instancia o splitter `IdentityDisjointPairKFold`** com
   identidades dos dois lados (probe + reference), `n_splits =
   cv_folds` e `random_state=42`. Neste caminho o splitter é sempre
   esse: `gridsearch_cv_mode`, `use_gridsearch` e
   `standardize_features` do config não são consultados (a
   `StandardScaler` está sempre na Pipeline), `lookalike_clusters`
   não é repassado e não há o filtro de monotonicidade do §5.4.
6. **Roda `GridSearchCV`** com `scoring = neg_cllr_balanced` (ou
   override do config), `refit=True`. Cada fold treina o
   transformer (s₀ via AUTO_LOGREG, r̄ por média) em `X_train`
   apenas — sem leakage de holdout.
7. **Extrai `state_final`** do transformer pós-refit em D_cal
   completo — esse é o `LRArchitectureState` que será propagado
   downstream para `INHERIT` em validação/casework.
8. **Empacota** num `LogisticRegressionCalibrationModel` para
   preservar contrato (consumido pelo `SCORE_CALIBRATION`).

### 10.3 Provenance persistida

O calibrador final recebe no metadata:

```python
{
  "vectorization_contract_fingerprint": state_final.vectorization_contract_fingerprint,
  "lr_architecture":                    state_final.architecture.value,
  "lr_architecture_formula_version":    state_final.formula_version,
  "architecture_spec_fingerprint":      state_final.architecture_spec_fingerprint,
  "architecture_state":                 state_final.to_dict(),
  "fit_per_fold":                       True,
  "inner_cv_splitter":                  "IdentityDisjointPairKFold",
  "inner_cv_fold_count":                cv_folds,
  "strict_identity_disjoint":           True,
  "final_refit_on_full_calibration":    True,
  "selected_C":                         model.best_C_,
  "standardized_coefficients":          logreg_final.coef_.ravel().tolist(),
  "scaler_mean":                        _feature_mean.tolist(),
  "scaler_scale":                       _feature_std.tolist(),
  "raw_scale_coefficients":             (coef / scale).tolist(),
  "raw_scale_intercept":                intercept − Σ(coef · μ / σ),
}
```

`raw_scale_coefficients` é usado pelo `LRArchitectureDerivativeProvider`
para avaliar `∂log_LR/∂s = β_raw · ∂φ/∂s` na auditoria
`CONDITIONAL_MONOTONICITY` — sem precisar reconstruir a
denormalização.

### 10.4 Pré-requisitos do raw_input

Para que `FIT_PER_FOLD` funcione, o `LR_ARCHITECTURE_RAW_INPUT`
precisa ter:
- `same_person` populado (não-casework);
- `probe_identity_ids` e `reference_identity_ids` populados
  (`PairIdentityMetadata`).

Sem isso, `_train_logreg_per_fold_with_transformer` levanta
`RuntimeError` com mensagem clara.

### 10.5 Vantagem sobre `FIT_FULL_CALIBRATION_LEGACY`

No legado, `s₀` e `r̄_F`/`r̄_P` eram estimados sobre TODO `D_cal`
**antes** da CV do calibrador. A CV via Pipeline+GridSearch tinha
leakage residual conhecido — porque `s₀` (aprendido na população
inteira) sangrava informação do held-out para o fold de treino.

No `FIT_PER_FOLD`, `s₀` e `r̄` são reajustados POR FOLD dentro do
`ArchitectureVectorizerTransformer.fit(X_train)`. O fold de teste
vê apenas parâmetros estimados sem ele. Após `refit=True`, o
transformer é reajustado em D_cal completo — o state final
reflete refit em toda a calibração, sem leakage residual.

---

## 11. Vínculo dataset ↔ calibrador (`CalibrationModelMetadataService.bind_architecture_state`)

### 11.1 Por que existe

Quando o calibrador é treinado sobre features vetorizadas (seja
`SCORE_VECTORIZATION` ou `LR_ARCHITECTURE_VECTORIZATION`), o
`SCORE_CALIBRATION` precisa garantir que o dataset de validação ou
casework consome as **mesmas features** sobre as quais o
calibrador foi treinado. Aplicar um calibrador treinado em
BILATERAL_FULL_CENTERED a um dataset vetorizado como
HINGE_ONLY produziria log-LRs sem sentido — features incompatíveis,
mas mesmo tipo `SimilarityScoreDataset`.

A solução é um fingerprint do contrato de vetorização:
`vectorization_contract_fingerprint`. O calibrador carrega o
fingerprint do dataset sobre o qual foi treinado; em
`SCORE_CALIBRATION`, o serviço compara o fingerprint do dataset
atual com o do calibrador.

### 11.2 O que `bind_architecture_state` copia

Chamado pelo executor após o fit do calibrador:

```python
CalibrationModelMetadataService.bind_architecture_state(
    trained_model.metadata,
    getattr(input_payload, "metadata", {}) or {},
)
```

Copia do metadata do dataset para o do calibrador:

| Campo | Função |
|---|---|
| `vectorization_contract_fingerprint` | Comparado em `SCORE_CALIBRATION` |
| `lr_architecture` | Auditoria — qual arquitetura LR |
| `lr_architecture_formula_version` | Auditoria — versão da fórmula |
| `architecture_spec_fingerprint` | Auditoria — fingerprint do spec |
| `training_score_vector_contract_fingerprint` | Auditoria — contrato do vetor de treino |
| `training_score_vector_numeric_fingerprint` | Auditoria — fingerprint numérico |
| `training_raw_input_matrix_fingerprint` | Auditoria — matriz raw |
| `training_pair_key_order_hash` | Auditoria — ordem dos pares |
| `architecture_state` | Snapshot completo — usado por INHERIT em VALIDATION/CASEWORK |

Idempotente. Se o dataset não tem `vectorization_contract_fingerprint`
(calibrador legado), o bind é no-op.

### 11.3 `_bind_training_raw_input`

Para a auditoria `CONDITIONAL_MONOTONICITY` pós-refit, o
calibrador precisa saber qual recurso `LR_ARCHITECTURE_RAW_INPUT`
no contexto bate com seu contract fingerprint. Esse lookup é feito
pelo executor com 4 estratégias em ordem:

1. Match exato por `training_raw_input_matrix_fingerprint`;
2. Match por `vectorization_contract_fingerprint`;
3. Match por `schema.fingerprint()`;
4. Heurística: único `LR_ARCHITECTURE_RAW_INPUT` no contexto.

O resource_id é persistido em
`calibrator.metadata["training_raw_input_resource_id"]` para
resolução automática em runs futuros.

---

## 12. Telemetria Cllr in-sample e CV diagnóstico

### 12.1 In-sample sempre é computado

Após o fit, `_attach_calibrator_telemetry` chama
`calibrate_scores` sobre os scores de treino e computa
`compute_cllr(llrs_gen, llrs_imp)`:

```python
trained_model.metadata.update({
  "calibrator_cllr_train":      cllr_train,
  "calibrator_cllr_train_gen":  cllr_train_gen,
  "calibrator_cllr_train_imp":  cllr_train_imp,
})
```

Otimista — mede ajuste, não generalização. Útil para diagnóstico
rápido.

### 12.2 CV diagnóstico opt-in via `k_folds`

Quando `config.k_folds >= 2`, roda K folds estratificados após o
fit principal com hiperparâmetros congelados (`C` do GridSearch,
bandwidths do KDE, etc.). Anexa:

```python
{
  "calibrator_cllr_cv":            cllr_cv,
  "calibrator_cllr_cv_gen":        cllr_cv_gen,
  "calibrator_cllr_cv_imp":        cllr_cv_imp,
  "calibrator_cv_gap":             cllr_cv − cllr_train,
  "cv_folds_effective":            k_folds_real,
  "cv_split_mode":                 cv_split_mode.value,
  "cv_hyperparameter_mode":        "FROZEN",
  "cv_hyperparameter_selection_scope": "FULL_CALIBRATION_SET",
  # contagens para auditoria: cv_n_gen_total, cv_n_imp_total,
  # cv_fold_train_gen_min, cv_fold_train_imp_min,
  # cv_groups_total, cv_groups_min_test
}
```

`FROZEN` significa que a seleção de hiperparâmetros foi feita no
conjunto de calibração completo antes do CV — não é nested CV. Com
`cv_split_mode = STRATIFIED_GROUP` e alguma amostra sem identidade de
probe, o executor recai em `STRATIFIED_PAIR` e avisa no log.

> O Cllr CV mede generalização **dentro do domínio de calibração**,
> não performance em validação externa. Para LogReg, o gap CV −
> in-sample tende a ser pequeno (calibrador robusto, baixa
> capacidade). Para KDE, o gap tende a ser grande (alta capacidade
> overfita mais).

### 12.3 Propagação downstream

O `SCORE_CALIBRATION` lê esses campos do metadata do calibrador e
copia para o metadata do dataset calibrado:

```python
for key in (
    "calibrator_cllr_train", "calibrator_cllr_train_gen", "calibrator_cllr_train_imp",
    "calibrator_cllr_cv",    "calibrator_cllr_cv_gen",    "calibrator_cllr_cv_imp",
    "calibrator_cv_gap", "cv_folds_effective", "cv_split_mode",
    "cv_hyperparameter_mode",
):
    if key in calibrator_md and calibrator_md[key] is not None:
        ds_metadata[key] = calibrator_md[key]
```

`STRATEGY_EVALUATION` então anexa tudo a `ValidationMetrics` para
exibir na tabela do `ValidationResultViewerDialog`.

---

## 13. Auditoria de monotonicidade pós-refit (V1.5 / V2)

### 13.1 O que é

`ConditionalMonotonicityRule` (V1 / V1.5 / V2) audita que
`∂log_LR/∂s > 0` em todas as regiões do domínio operacional do
calibrador. Sob a tese forense de que raridade modula evidência
mas não é evidência independente, `∂log_LR/∂base > 0` é condição
necessária para o LR ser interpretável: aumentar similaridade
não pode reduzir o log-LR.

Documentação canônica:
`docs/conditional_monotonicity_rule_design.md`.

### 13.2 Quando roda

`_run_monotonicity_audit` é chamado pelo executor após o refit do
calibrador e após `bind_architecture_state`. Roda apenas se o
calibrador tem `architecture_state` no metadata — ou seja, foi
treinado com `LR_ARCHITECTURE_VECTORIZATION` (qualquer
`parameter_mode`). Em calibradores legados (KDE, BiGauss, LogReg
sobre score 1D direto), pula silenciosamente.

### 13.3 Como roda

```python
cfg = ValidationRuleConfig(
    rule_type=ValidationRuleType.CONDITIONAL_MONOTONICITY,
    params={"calibrator_resource_id": calibrator_resource_id},
)
rule = ValidationRuleFactory.create(cfg, instruction_id="post_refit_audit")
result = rule.validate(target=None, context=context)
```

A regra:
1. Resolve o calibrador no contexto;
2. Resolve o `LR_ARCHITECTURE_RAW_INPUT` correspondente
   (via `training_raw_input_resource_id`);
3. Constrói o `LRArchitectureDerivativeProvider` para a arquitetura;
4. Para cada região piecewise (`s > s_0`, `s ≤ s_0` quando há hinge),
   avalia `∂log_LR/∂s = β_raw · ∂φ/∂s` em:
   - todos os pares observados (`observed_rows`);
   - cantos do retângulo `[P5, P95]²` das raridades (`central_box`);
   - cantos do retângulo `reportable_box` (V2 — domínio operacional
     declarado pela política de reportabilidade, quando configurado).
5. Persiste resultado em `calibrator.metadata["monotonicity_audit"]`.

### 13.4 Status reportado

- `OK` 🛡️ — monotonicidade preservada em todas as regiões e
  pontos amostrados.
- `WARNING` ⚠ — slope ≤ 0 apenas em canto da `central_box`
  (P5–P95), com os pares observados monotônicos; ou slope mínimo
  observado abaixo de `operational_min_slope`, quando configurado.
  Não bloqueia, mas exige atenção. Feature degenerada (variância zero
  numa coluna) gera aviso adicional, sem mudar o status.
- `CRITICAL` ❌ — slope ≤ 0 em algum par observado de D_cal
  (`observed_rows`) ou no `reportable_box`. Em V2, violação no
  `reportable_box` é sempre CRITICAL — o domínio operacional foi
  declarado explicitamente pela política de reportabilidade.
- `NOT_AUDITED` — auditoria não realizada (ex.: calibrador sem
  `training_raw_input_resource_id` resolvido); logado, não bloqueia.
  Calibrador sem `architecture_state` nem chega à auditoria (§13.2).

### 13.5 Não bloqueia o treino

A auditoria é informativa — falha não impede a persistência do
calibrador. O perito decide se o status `CRITICAL` justifica
descartar o calibrador. Para bloquear o uso downstream, a regra
`CONDITIONAL_MONOTONICITY` pode ser declarada em
`validation_rule_configs` de uma instrução posterior (param
`calibrator_resource_id`); ela não está entre os tipos oferecidos
pelo diálogo "➕ Adicionar Regra" (`AddValidationRuleDialog`), então
a declaração é feita no JSON da estratégia.

---

## 14. Regras de validação padrão

`_ensure_default_validation_rules` popula `validation_rule_configs`
quando `input_resource_id` está definido E
`validation_rules_user_managed=False`:

| Regra | Função | Quando aciona |
|---|---|---|
| `MIN_CALIBRATION_PAIRS` | Verifica `n_gen ≥ min_genuines` e `n_imp ≥ min_impostors`. | Sempre que input definido. |
| `SCORE_CALIBRATION_STATE(RAW)` | Verifica que o input contém scores RAW, não pré-calibrados. | Sempre que input definido. |

`CONDITIONAL_MONOTONICITY` não está nos defaults — ela depende do
calibrador refitado para avaliar, então roda **programaticamente**
no executor após o refit (vide §13). Quem quiser a regra como
validação pré-execução do `SCORE_CALIBRATION`/`STRATEGY_EVALUATION`
precisa declará-la em `validation_rule_configs` (§13.5) — o diálogo
"➕ Adicionar Regra" não a oferece.

Se o usuário deleta uma regra via UI (🗑), o flag
`validation_rules_user_managed` vira `True` e os defaults param de
se repor. Botão "🔄 Restaurar padrões" reseta o flag para `False`
e dispara `_sync_declared_io`, repopulando.

---

## 15. Casos de uso

### 15.1 Calibrador LogReg simples sobre score cosseno 1D

```
CALIBRATION_MODEL_TRAINING(
    calibration_model_type = LOGISTIC_REGRESSION,
    input_resource_id = "cosseno_D_cal",
    logreg_config = LogisticRegressionCalibrationModelConfig(
        use_gridsearch = True,
        c_grid = [0.01, 0.1, 1.0, 10.0, 100.0],
        cv_folds = 5,
        scoring = NEG_CLLR_BALANCED,
        gridsearch_cv_mode = AUTO,
        k_folds = 5,
        cv_split_mode = STRATIFIED_GROUP,
    ),
    trained_calibration_model_resource_id = "cal_logreg_simples",
)
```

LogReg afim sobre score 1D. GridSearch escolhe `C`. CV diagnóstico
(opt-in) reporta `calibrator_cllr_cv` para a tabela de métricas.

### 15.2 Calibrador LogReg sobre BILATERAL via `FIT_PER_FOLD`

```
LR_ARCHITECTURE_VECTORIZATION(
    architecture = BILATERAL_FULL_CENTERED,
    parameter_mode = FIT_PER_FOLD,
    base_score_resource_id = "cosseno_D_cal",
    rarity_probe_resource_id = "typicality_probe_D_cal",
    rarity_reference_resource_id = "typicality_reference_D_cal",
    output_resource_id = "vec_BILATERAL_cheia",
    output_raw_input_resource_id = "raw_input_BILATERAL_cheia",
)
        │
        ▼
CALIBRATION_MODEL_TRAINING(
    calibration_model_type = LOGISTIC_REGRESSION,
    input_resource_id = "vec_BILATERAL_cheia",
    logreg_config = LogisticRegressionCalibrationModelConfig(
        c_grid = [0.01, 0.1, 1.0, 10.0],
        cv_folds = 5,
        gridsearch_cv_mode = STRICT_IDENTITY,
        standardize_features = True,
    ),
    trained_calibration_model_resource_id = "cal_BILATERAL_cheia",
)
```

O executor detecta `defer_to_calibrator_per_fold=True` no metadata
do dataset vetorizado e despacha para
`_train_logreg_per_fold_with_transformer`. `s₀`, `r̄_F`, `r̄_P` são
ajustados POR FOLD. State final reflete refit em D_cal completo.
Auditoria `CONDITIONAL_MONOTONICITY` roda sobre o resultado. Nesse
caminho `gridsearch_cv_mode` e `standardize_features` não têm efeito
(o splitter é sempre `IdentityDisjointPairKFold` e a `StandardScaler`
está sempre na Pipeline, §10.2); ficam no exemplo apenas como
documentação da intenção.

### 15.3 Calibrador LogReg piecewise-linear sobre score PLDA 1D

```
CALIBRATION_MODEL_TRAINING(
    calibration_model_type = LOGISTIC_REGRESSION,
    input_resource_id = "plda_D_cal",
    logreg_config = LogisticRegressionCalibrationModelConfig(
        use_gridsearch = True,
        c_grid = [0.01, 0.1, 1.0, 10.0],
        cv_folds = 5,
        scoring = NEG_CLLR_BALANCED,
        gridsearch_cv_mode = GROUPED_PROBE,
        piecewise_linear_config = PiecewiseLinearCalibrationConfig(
            mode = PIECEWISE_LINEAR,
            hinge_anchor_mode = AUTO_LOGREG,
            knot_offsets = [0.0],
            knot_scale = AUX_LOG_LR,
            knot_placement = MANUAL,
            require_monotonicity = True,
            monotonicity_policy = ERROR,
        ),
    ),
    trained_calibration_model_resource_id = "cal_plda_piecewise",
)
```

PLDA escalar com dobra única em `s₀` (ponto neutro do LogReg
auxiliar). Slope distinto acima e abaixo de `s₀`. Monotonicidade
validada na escala bruta após o fit final.

### 15.4 Calibrador KDE sobre score cosseno 1D

```
CALIBRATION_MODEL_TRAINING(
    calibration_model_type = KDE,
    input_resource_id = "cosseno_D_cal_large",
    kde_config = KDECalibrationModelConfig(
        bandwidth_genuine = 0.05,
        bandwidth_impostor = 0.05,
        use_gridsearch = True,
        max_samples_for_gridsearch = 10000,
        max_stored_samples = 50000,
        min_log_lr = -20.0,
        max_log_lr = +20.0,
        k_folds = 5,
        cv_split_mode = STRATIFIED_GROUP,
    ),
    trained_calibration_model_resource_id = "cal_kde",
)
```

KDE não-paramétrico. Exige ≥ 50k pares por classe para capacidade
plena. Cllr CV − in-sample geralmente > LogReg.

---

## 16. Boas práticas

### 16.1 Treino

1. **Defaults seguros**: `LOGISTIC_REGRESSION` +
   `use_gridsearch=True` + `scoring=NEG_CLLR_BALANCED`. Único
   padrão defensável em laudo sem ablação ulterior.
2. **CV honesta**: `gridsearch_cv_mode=STRICT_IDENTITY` quando há
   ≳ 50 identidades em D_cal. `GROUPED_PROBE` como fallback. Evite
   `PAIR` exceto para benchmark.
3. **Standardização em multi-D**: ative `standardize_features=True`
   sempre que `D > 1` e os sistemas tiverem escalas dispares. Sem
   isso, a L2 do sklearn distorce a contribuição relativa.
4. **Piecewise-linear consciente**: ative apenas quando há evidência
   empírica de slope assimétrico (vide
   `docs/piecewise_linear_calibration.md`, §6 e §8).
   Default `knot_offsets=[0.0]` (dobra única em `s₀`) é o cenário
   parcimonioso recomendado.
5. **FIT_PER_FOLD**: prefira-o sempre que estiver usando
   `LR_ARCHITECTURE_VECTORIZATION` — elimina o leakage residual do
   `FIT_FULL_CALIBRATION_LEGACY`.

### 16.2 Diagnóstico

1. **Cllr CV diagnóstico**: ative `k_folds=5` +
   `cv_split_mode=STRATIFIED_GROUP` em desenvolvimento. Reporta
   generalização DENTRO do domínio de calibração — não substitui
   validação externa.
2. **Monotonicity audit**: sempre verifique
   `monotonicity_audit.audit_status` após treinar com
   `LR_ARCHITECTURE_VECTORIZATION` — o status sai no log da execução
   (🛡️ / ⚠ / ❌) e fica no metadata do calibrador; o
   `ResourceInfoWidget` não o exibe. Status `CRITICAL` exige
   investigação antes de uso operacional.
3. **Cllr in-sample baixo + Cllr CV alto = overfitting**. Em LogReg,
   tente `C` menor (regularização mais forte). Em KDE, aumente
   bandwidth ou subamostre mais. Em BiGauss, considere maior
   regularização.

### 16.3 Auditoria

1. **Persista `vectorization_contract_fingerprint`** sempre que
   possível — única forma defensável de garantir compatibilidade
   dataset ↔ calibrador em validação/casework downstream.
2. **Inspecione `selected_C`, `fit_per_fold`,
   `strict_identity_disjoint`** no metadata do calibrador antes de
   submeter laudo. Estes campos documentam o protocolo seguido.
3. **`raw_scale_coefficients` para interpretabilidade**: em
   calibradores multi-D, esses coeficientes (denormalizados da
   StandardScaler interna) revelam quanto cada feature contribui
   em escala bruta. Use no plano forense para explicar peso
   relativo de score vs raridade.

---

## 17. Perguntas frequentes

**P1: Posso treinar um calibrador KDE sobre features multivariadas?**

Não. KDE e BiGauss aceitam apenas score 1D. Para multi-D, use
LOGISTIC_REGRESSION com `standardize_features=True`. O executor
levanta erro se você tentar treinar KDE/BiGauss sobre `(N, D > 1)`.

**P2: Quando devo usar `STRICT_IDENTITY` vs `GROUPED_PROBE`?**

`STRICT_IDENTITY` é forensicamente mais defensável — folds
totalmente disjuntos dos dois lados (probe + reference). Requer
≳ 50 identidades em D_cal para manter contagens Hp/Hd saudáveis
por fold (vide `IdentityDisjointPairKFold.get_fold_stats`).
`GROUPED_PROBE` é o default V2.8 — disjunção parcial (probe), boa
relação custo-benefício quando o número de identidades é menor.

**P3: O que acontece se eu mudar de família após treinar?**

`prune_unused_fields` (chamado no Apply) zera os configs inativos.
Mudar de LOGREG para KDE descarta `logreg_config` no JSON. Mas o
calibrador treinado anteriormente (pickle no disco) é
preservado — apenas a próxima execução usará o config novo.

**P4: O calibrador piecewise-linear funciona em multi-D?**

Não na V1. Restrito a score 1D. Para multi-D, use
`LR_ARCHITECTURE_VECTORIZATION` com arquitetura
`HIERARCHICAL_CENTERED` ou `BILATERAL_FULL_CENTERED` — essas já
incorporam quebra em `s₀` via termos hinge `h⁺`, `h⁻`.

**P5: A auditoria `CONDITIONAL_MONOTONICITY` bloqueia o treino?**

Não. Auditoria é informativa — falha não impede a persistência do
calibrador. Para bloquear o uso downstream em validação/casework,
declare a regra `CONDITIONAL_MONOTONICITY` em `validation_rule_configs`
da instrução `SCORE_CALIBRATION` ou `STRATEGY_EVALUATION` (§13.5).

**P6: O que é `defer_to_calibrator_per_fold` e quem o seta?**

É a flag no metadata do dataset vetorizado que `LR_ARCHITECTURE_VECTORIZATION`
em modo `FIT_PER_FOLD` injeta para sinalizar ao calibrador que ele
deve montar a Pipeline+GridSearch+Transformer interna em vez de
treinar sobre a matriz pré-vetorizada (que tem leakage residual).
O calibrador detecta a flag em runtime via
`per_fold_active = uses_logistic_regression and
defer_to_calibrator_per_fold and raw_input_resource_id`.

**P7: Posso usar `FIT_PER_FOLD` com KDE ou BiGauss?**

Não. `FIT_PER_FOLD` exige `LOGISTIC_REGRESSION` — a Pipeline
sklearn assume estimator linear compatível. KDE e BiGauss têm
fit em forma fechada (ou GridSearch separado para bandwidth) que
não casa com o protocolo Pipeline+GridSearchCV+Transformer.

**P8: Como vejo o `best_C` selecionado pelo GridSearch?**

`trained_model.best_C_` (atributo). No metadata do calibrador:
`piecewise_selected_c` (caminho piecewise) ou `selected_C` (caminho
`FIT_PER_FOLD`); o resultado da execução da instrução traz
`best_C`. O `ResourceInfoWidget` exibe como "C (efetivo)" na seção
"Detalhes do Modelo de Calibração".

---

## 18. Glossário

| Termo | Definição |
|---|---|
| `C` | Inverso da regularização L2 do LogReg sklearn. `C` baixo = regularização forte. |
| `c_grid` | Grade de `C` testada pelo GridSearch. |
| `cv_folds` | Folds do GridSearchCV interno para `C` (não confundir com `k_folds` do CV diagnóstico). |
| `k_folds` | Folds do CV diagnóstico opt-in pós-fit. |
| `gridsearch_cv_mode` | Splitter do GridSearchCV interno do LogReg para `C`. |
| `cv_split_mode` | Splitter do CV diagnóstico (`k_folds`). |
| `NEG_CLLR_BALANCED` | Default V2.6 — Cllr balanceado por classe. Forense-defensável em datasets desbalanceados. |
| `standardize_features` | Z-score por feature em multi-D. Necessário quando sistemas têm escalas dispares. |
| `piecewise_linear` | Expansão de feature `[z, max(z−offset, 0), ...]` antes do LogReg. Default V1: 1 dobra em `s₀`. |
| `hinge_anchor_s0` | Ponto neutro da quebra piecewise — onde `log_LR = 0` no calibrador auxiliar. |
| `FIT_PER_FOLD` | Caminho V2.14 onde `s₀`/`r̄` são reajustados POR FOLD via Pipeline+GridSearch+Transformer. |
| `IdentityDisjointPairKFold` | Splitter customizado V2.10 com folds totalmente disjuntos dos dois lados. |
| `vectorization_contract_fingerprint` | Hash do contrato de vetorização. Comparado em `SCORE_CALIBRATION` para garantir compatibilidade dataset↔calibrador. |
| `architecture_state` | Snapshot completo do state da `LR_ARCHITECTURE_VECTORIZATION`. Usado por INHERIT em VALIDATION/CASEWORK. |
| `monotonicity_audit` | Resultado da auditoria pós-refit `CONDITIONAL_MONOTONICITY` persistido no metadata. |
| `raw_scale_coefficients` | Coeficientes do LogReg denormalizados da StandardScaler interna. Usado pela auditoria e pelo plot 2D analítico. |
| `Cllr` | Log-loss em base 2 — métrica primária de calibração forense. |
| `Cllr_train` / `Cllr_cv` | In-sample (otimista) vs out-of-fold (generalização dentro do domínio de calibração). |
| `cv_identity_disjointness_level` | Enum textual `full`/`probe_only`/`none` — documenta a cobertura do splitter usado. |
| `lookalike_clusters` | Identidades agrupadas como super-id no `IdentityDisjointPairKFold` (V2.11.2). |

---

## 19. Referências

### 19.1 Documentos internos

- `docs/calibracao_faceaudit.md` — referência guarda-chuva sobre
  calibração no FaceAudit.
- `docs/piecewise_linear_calibration.md` — especificação da
  expansão piecewise-linear V1 (knots, monotonicidade, leakage
  prevention).
- `docs/conditional_monotonicity_rule_design.md` — auditoria
  forense pós-refit V1/V1.5/V2.
- `docs/lr_architecture_vectorization_faceaudit.md` — substituto
  declarativo das 15 arquiteturas LR consumido em `INHERIT` por
  validação/casework.
- `docs/score_calibration_faceaudit.md` — instrução irmã que
  aplica o calibrador a um dataset de scores brutos.
- `docs/vetorizacao_scores_faceaudit.md` — vetorização explícita
  via `SCORE_VECTORIZATION` (legado).

### 19.2 Bibliografia técnica

- Brümmer, N., & du Preez, J. (2006). *Application-independent
  evaluation of speaker detection*. Computer Speech & Language,
  20(2-3), 230–275. — origem do Cllr.
- Morrison, G. S. (2021). *In the context of forensic casework,
  are there meaningful metrics of the degree of calibration?*.
  FSI: Synergy. — crítica ao uso isolado de PAVA in-sample como
  medida de calibração.
- Brümmer, N., & de Villiers, E. (2010). *The speaker partitioning
  problem*. — fundamentação do PLDA two-covariance.
- Graham, R. L. (1969). *Bounds on multiprocessing timing
  anomalies*. Bell System Technical Journal. — LPT scheduling
  usado pelo `IdentityDisjointPairKFold`.

---

## Apêndice A — `CalibrationModelTrainingInstruction`

Campos do DTO em
`faceaudit/models/instructions/calibration_model_training_instruction.py`:

```python
@dataclass
class CalibrationModelTrainingInstruction(StrategyInstructionDefinition):
    calibration_model_type: CalibrationModelType = LOGISTIC_REGRESSION
    input_resource_id: Optional[str] = None
    logreg_config: Optional[LogisticRegressionCalibrationModelConfig] = None
    kde_config: Optional[KDECalibrationModelConfig] = None
    bigauss_config: Optional[BiGaussianizedCalibrationModelConfig] = None
    trained_calibration_model_resource_id: Optional[str] = None
```

Comportamento do `__post_init__`:
- Auto-instancia o config relevante (`logreg_config` se LOGREG,
  etc.);
- Mantém configs antigos no DTO ao alternar família (útil para
  ajustar valores antes de mudar);
- `prune_unused_fields()` (chamado no Apply) zera os configs
  inativos antes de persistir.

Propriedades:
- `uses_logistic_regression`, `uses_kde`, `uses_bigauss`.

Resource field specs:
- `input_resource_id`: aceita `SIMILARITY_SCORES` ou `ARTIFACT`.
- `trained_calibration_model_resource_id`: produz
  `TRAINED_CALIBRATION_MODEL`.

Choice field spec:
- `calibration_model_type`: 3 opções com descrição.

---

## Apêndice B — Tabela comparativa das três famílias

| Aspecto | LOGISTIC_REGRESSION | KDE | BI_GAUSSIANIZED |
|---|---|---|---|
| Modelo | Afim (com hinge piecewise opt-in) | Não-paramétrico (duas densidades estimadas) | Calibrador inicial + mapeamento para sistema bi-gaussiano alvo (Morrison 2024) |
| Suporta multi-D (`D > 1`) | ✅ | ❌ | ❌ |
| Suporta `FIT_PER_FOLD` | ✅ (V2.14) | ❌ | ❌ |
| Suporta piecewise-linear interna | ✅ (V1, score 1D) | ❌ | ❌ |
| GridSearchCV para hyperparam | ✅ (`C` via `c_grid` + `cv_folds`) | ✅ (bandwidth via GridSearch + expansão) | ❌ (σ² derivado do Cllr/EER) |
| CV diagnóstico (`k_folds`) | ✅ | ✅ | ✅ |
| `standardize_features` | ✅ (multi-D) | n/a | n/a |
| Persiste `architecture_state` | ✅ (quando treinado sobre saída de `LR_ARCHITECTURE_VECTORIZATION`) | ❌ | ❌ |
| Auditoria `CONDITIONAL_MONOTONICITY` pós-refit | ✅ (quando `architecture_state` presente) | ⏭ (skip silencioso) | ⏭ (skip silencioso) |
| Risco de overfitting | Baixo (afim) | Alto (alta capacidade) | Médio |
| Defensibilidade forense | Alta (parcimonioso, monotonicidade auditável) | Média (não-paramétrico, alta capacidade) | Média (sensível ao N nas caudas da CDF empírica) |
| Recomendação padrão | ✅ default operacional | Apenas em score 1D com ≥ 50k pares/classe | Apenas em score 1D quando se quer paramétrico parcimonioso |

---

## Apêndice C — Checklist operacional

Antes de treinar:

- [ ] Escolheu a família com base na natureza do score (1D vs
      multi-D, paramétrico vs não-paramétrico)?
- [ ] Habilitou `use_gridsearch=True` em LogReg/KDE para deixar
      hyperparam fora da escolha manual?
- [ ] Configurou `scoring=NEG_CLLR_BALANCED` (default V2.6) em
      LogReg?
- [ ] Em multi-D, ativou `standardize_features=True`?
- [ ] Em ≳ 50 identidades, ativou
      `gridsearch_cv_mode=STRICT_IDENTITY`? Senão,
      `GROUPED_PROBE`?
- [ ] Em `LR_ARCHITECTURE_VECTORIZATION` upstream, escolheu
      `FIT_PER_FOLD` (e não `FIT_FULL_CALIBRATION_LEGACY`)?
- [ ] Em piecewise-linear, justificou empiricamente a quebra (vide
      `docs/piecewise_linear_calibration.md`)?

Após treinar:

- [ ] Conferiu `best_C` (LogReg) ou bandwidths (KDE) no metadata?
      Valores na borda da grade indicam grid mal dimensionado.
- [ ] Conferiu `calibrator_cllr_train` (in-sample)?
- [ ] Conferiu `calibrator_cllr_cv` quando `k_folds > 0`? Gap
      pequeno = calibrador robusto; gap grande = overfitting.
- [ ] Conferiu `monotonicity_audit.audit_status` quando o
      calibrador veio de `LR_ARCHITECTURE_VECTORIZATION`? `OK`
      libera; `WARNING` exige revisão; `CRITICAL` exige
      investigação séria.
- [ ] Conferiu `vectorization_contract_fingerprint` no metadata
      do calibrador quando vetorizado?
- [ ] Documentou no relatório os campos `selected_C`,
      `fit_per_fold`, `strict_identity_disjoint`,
      `cv_identity_disjointness_level`,
      `gridsearch_cv_mode` para auditoria pericial?

Em validação/casework:

- [ ] Configurou `LR_ARCHITECTURE_VECTORIZATION` downstream em
      `parameter_mode=INHERIT` para preservar a definição de
      "evidência neutra"?
- [ ] Em `SCORE_CALIBRATION`, considerou
      `compatibility_mode=STRICT` (default) ou
      `LEGACY_PERMISSIVE` (opt-in com warning)?
- [ ] Em laudo, justificou a escolha da família e a estratégia de
      CV usada na seleção de hiperparâmetros?
