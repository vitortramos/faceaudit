# Aplicação de calibrador a scores (`SCORE_CALIBRATION`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `SCORE_CALIBRATION` — aplica um
> `TRAINED_CALIBRATION_MODEL` a um `SimilarityScoreDataset` com
> `raw_score`, produzindo um novo dataset com `log_lr` por par.
> Cobre o vínculo dataset ↔ calibrador via
> `vectorization_contract_fingerprint` (modos `STRICT` vs
> `LEGACY_PERMISSIVE`), o roteamento entre score 1D e matriz
> vetorizada `(N, D)`, a propagação de telemetria CV / in-sample +
> coeficientes raw + piecewise state para `STRATEGY_EVALUATION`
> downstream, a preservação do flag `is_casework`, e a estratégia
> de cópia defensiva dos samples (sem mutação in-place do input).
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos
> com conhecimento intermediário.
> **Documentos relacionados:**
> `docs/calibration_model_training_faceaudit.md` (irmã: treina o
> calibrador consumido aqui),
> `docs/calibracao_faceaudit.md` (referência guarda-chuva sobre
> calibração),
> `docs/lr_architecture_vectorization_faceaudit.md` (vetorização
> declarativa que produz o `vectorization_contract_fingerprint`
> consumido aqui),
> `docs/conditional_monotonicity_rule_design.md` (auditoria do
> calibrador, persistida no metadata do próprio calibrador — esta
> instrução não a copia para o output).
> **Fontes-código:** `faceaudit/models/instructions/score_calibration_instruction.py`,
> `faceaudit/executors/instructions/score_calibration_instruction_executor.py`,
> `faceaudit/services/calibration_model_metadata_service.py`,
> `faceaudit/models/similarity/similarity_score_dataset.py`,
> `faceaudit/enums/calibration_compatibility_mode.py`,
> `faceaudit/protocols/calibration_model_protocol.py`,
> `faceaudit/validation_rules/score_calibration_state_rule.py`,
> `faceaudit/validation_rules/score_vector_order_consistency_rule.py`,
> `faceaudit/validation_rules/model_fitted_rule.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a instrução se encaixa](#1-onde-a-instrução-se-encaixa)
- [2. Configuração do DTO](#2-configuração-do-dto)
- [3. Fluxo do executor](#3-fluxo-do-executor)
- [4. Roteamento 1D vs matriz `(N, D)`](#4-roteamento-1d-vs-matriz-n-d)
- [5. Vínculo dataset ↔ calibrador (`validate_compatibility`)](#5-vínculo-dataset--calibrador-validate_compatibility)
- [6. Os dois modos de compatibilidade](#6-os-dois-modos-de-compatibilidade)
- [7. Cópia defensiva dos samples — anti-mutação in-place](#7-cópia-defensiva-dos-samples--anti-mutação-in-place)
- [8. Propagação de telemetria para `STRATEGY_EVALUATION`](#8-propagação-de-telemetria-para-strategy_evaluation)
- [9. Propagação de coeficientes raw + piecewise state](#9-propagação-de-coeficientes-raw--piecewise-state)
- [10. Preservação do flag `is_casework`](#10-preservação-do-flag-is_casework)
- [11. Regras de validação dinâmicas](#11-regras-de-validação-dinâmicas)
- [12. Diagnóstico de log-LR](#12-diagnóstico-de-log-lr)
- [13. Persistência em parquet](#13-persistência-em-parquet)
- [14. Casos de uso](#14-casos-de-uso)
- [15. Boas práticas](#15-boas-práticas)
- [16. Perguntas frequentes](#16-perguntas-frequentes)
- [17. Glossário](#17-glossário)
- [18. Referências](#18-referências)
- [Apêndice A — `ScoreCalibrationInstruction`](#apêndice-a--scorecalibrationinstruction)
- [Apêndice B — Tabela de compatibilidade dataset × calibrador](#apêndice-b--tabela-de-compatibilidade-dataset--calibrador)
- [Apêndice C — Campos propagados no metadata do output](#apêndice-c--campos-propagados-no-metadata-do-output)
- [Apêndice D — Checklist operacional](#apêndice-d--checklist-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre `SCORE_CALIBRATION`: como o
calibrador treinado é aplicado a um dataset de scores brutos, como
o vínculo dataset ↔ calibrador é validado em runtime via
fingerprints, como log-LR é anexado a cada sample, como
propagação de telemetria preserva auditoria forense end-to-end, e
como o flag `is_casework` mantém a semântica "sem ground truth"
através da calibração.

### 0.2 Por que existe

`CALIBRATION_MODEL_TRAINING` produz um calibrador
(`TRAINED_CALIBRATION_MODEL`). Esse calibrador precisa ser
aplicado a um dataset distinto — tipicamente o conjunto de
**validação** (para `STRATEGY_EVALUATION`) ou de **casework**
(para `CASEWORK_VISUALIZATION` / laudo). `SCORE_CALIBRATION` é a
instrução que faz essa aplicação.

A separação `train` / `apply` é deliberada:
- o calibrador é objeto persistente e reusável;
- o dataset de aplicação pode ser maior, menor, ou produzido por
  pipeline distinto;
- a aplicação é o ponto onde o **vínculo de compatibilidade** é
  verificado — `vectorization_contract_fingerprint` do dataset
  precisa bater com o que o calibrador foi treinado para receber.

### 0.3 Sem hyperparâmetros próprios

`SCORE_CALIBRATION` não tem hyperparâmetros configuráveis. Toda a
matemática já está embutida no calibrador treinado. Os campos do
DTO são apenas identificadores de recursos.

---

## 1. Onde a instrução se encaixa

```
CALIBRATION_MODEL_TRAINING
        │
        ▼
TRAINED_CALIBRATION_MODEL (pickle, com metadata + state)
        │
        │              SIMILARITY_SCORING(val_or_case)
        │                     │
        │                     ▼
        │              SIMILARITY_SCORES (raw_score)
        │                     │
        │                     │  ── opcionalmente passa por:
        │                     │     • SCORE_INTERACTION_FEATURE (legado)
        │                     │     • SCORE_VECTORIZATION (legado)
        │                     │     • LR_ARCHITECTURE_VECTORIZATION (V2.14)
        │                     ▼
        │              SIMILARITY_SCORES (vetorizado)
        │                     │
        └─────────┬───────────┘
                  ▼
         SCORE_CALIBRATION ← você está aqui
            ├── valida vínculo via vectorization_contract_fingerprint
            ├── aplica calibrate_scores(raw)
            ├── copia samples + injeta log_lr no metadata
            ├── propaga telemetria CV/in-sample + coefs raw
            └── preserva is_casework
                  │
                  ▼
         CALIBRATED_SCORES (log_lr por par)
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
STRATEGY_EVALUATION   CASEWORK_VISUALIZATION
```

### 1.1 Entrada

Dois `StrategyResource`:

1. `trained_calibration_model_resource_id` — recurso do tipo
   `TRAINED_CALIBRATION_MODEL`, calibrador treinado previamente.
   Payload precisa satisfazer `CalibrationModelProtocol`
   (`fit`, `predict_proba`, `calibrate_scores`).
2. `input_similarity_scores_resource_id` — recurso do tipo
   `SIMILARITY_SCORES`, contendo `SimilarityScoreDataset` com
   `raw_score` populado nos samples.

### 1.2 Saída

Um `StrategyResource` do tipo `CALIBRATED_SCORES` contendo:

- Novo `SimilarityScoreDataset` com cópia rasa dos samples do
  input + `metadata["log_lr"] = float(log_lr)` em cada sample;
- O `raw_score` original é preservado;
- O metadata do dataset é enriquecido com telemetria do calibrador,
  flag `is_casework`, coeficientes raw, piecewise state, e
  `score_vector` metadata quando vetorizado;
- O parquet de saída inclui automaticamente uma coluna `log_lr`
  nativa (`SimilarityScoreDataset.to_parquet` detecta `log_lr` em
  metadata).

---

## 2. Configuração do DTO

`ScoreCalibrationInstruction` (apêndice A) tem três campos
declarativos + um campo cosmético:

| Campo | Tipo | Default | Função |
|---|---|---|---|
| `trained_calibration_model_resource_id` | `str` | None | Recurso do calibrador treinado |
| `input_similarity_scores_resource_id` | `str` | None | Recurso de scores brutos a calibrar |
| `output_calibrated_scores_resource_id` | `str` | None | Recurso de saída |
| `output_calibrated_scores_name` | `str` | None | Nome amigável do dataset gerado (cosmético) |

Sem hyperparâmetros. Sem `compatibility_mode` opcional no DTO atual
— o executor sempre usa o default `STRICT` da
`CalibrationModelMetadataService.validate_compatibility`. (A doc do
serviço descreve o opt-in via parâmetro `compatibility_mode`, mas a
instrução não expõe esse campo nesta versão; ele só pode ser
ativado por chamadores diretos do serviço.)

### 2.1 Auto-sync de I/O

`_sync_declared_io` mantém `input_resource_ids` e
`output_resource_ids` sincronizados com os 3 campos específicos.
Ordem: calibrator primeiro, scores depois.

### 2.2 Field specs

A UI usa `get_resource_field_specs()` para mostrar dropdowns
type-filtered:
- `trained_calibration_model_resource_id`: `TRAINED_CALIBRATION_MODEL`;
- `input_similarity_scores_resource_id`: `SIMILARITY_SCORES`;
- `output_calibrated_scores_resource_id`: `CALIBRATED_SCORES`
  (output, text input).

---

## 3. Fluxo do executor

```
execute(instruction, context)
  ├── validate(instruction, context)              (regras dinâmicas)
  │     ├── ModelFittedRule(calibrator)
  │     ├── ScoreCalibrationStateRule(input, RAW)
  │     └── ScoreVectorOrderConsistencyRule(input, calibrator)
  │
  ├── extrai scores_resource + calibrator_resource do contexto
  ├── score_dataset, calibrator = payloads
  │
  ├── CalibrationModelMetadataService.validate_compatibility(
  │       cal_meta, ds_meta,
  │   )
  │     ├── if error → return failure_result
  │     └── if warning → log + segue
  │
  ├── raw_scores = (
  │       score_dataset.extract_score_vector()  (se is_score_vector)
  │       senão score_dataset.scores            (1D)
  │   )
  ├── log_lrs = calibrator.calibrate_scores(raw_scores)
  │
  ├── output_items = []                          (chunks de 50k + sleep(0))
  │   para cada sample do input:
  │     new_sample = copy.copy(src)              (cópia rasa)
  │     new_sample.metadata = {**old_md, "log_lr": float(log_lrs[j])}
  │   
  ├── ds_metadata = {
  │       source_dataset_name,
  │       calibrator_type,
  │       is_casework,
  │       (telemetria CV/in-sample copiada do calibrator_md),
  │       (is_score_vector + score_vector_source_resources),
  │       (calibrator_coef_raw + calibrator_intercept_raw),
  │       (piecewise_* state),
  │   }
  │
  ├── calibrated_dataset = SimilarityScoreDataset(items, ds_metadata)
  ├── context.set_resource(CALIBRATED_SCORES)
  │
  └── retorna InstructionExecutionResult.ok(
        produced_resource_ids=[output_id],
        metadata={n_samples, n_genuine, n_impostor, mean_log_lr_gen, mean_log_lr_imp},
      )
```

---

## 4. Roteamento 1D vs matriz `(N, D)`

```python
raw_scores = (
    score_dataset.extract_score_vector()
    if score_dataset.is_score_vector
    else score_dataset.scores
)
```

### 4.1 Caso 1D (default)

`score_dataset.scores` retorna um `np.ndarray (N,)` de
`float64` — `raw_score` extraído de cada sample. É o caminho
comum para calibradores LogReg/KDE/BiGauss sobre score cosseno ou
PLDA escalar.

### 4.2 Caso vetorizado `(N, D)`

Quando o dataset foi produzido por `SCORE_VECTORIZATION` (legado)
ou `LR_ARCHITECTURE_VECTORIZATION` (V2.14), o flag
`is_score_vector` está True e a matriz `(N, D)` está em
`metadata[SCORE_VECTOR_MATRIX_KEY]`.

`extract_score_vector()` reconstrói a matriz e retorna-a para o
calibrador. O calibrador (LogReg multivariado, treinado em modo
`standardize_features=True`) consome a matriz diretamente.

### 4.3 Validação de shape

```python
if log_lrs.shape[0] != n_samples:
    return failure_result("Calibrator retornou {X} log-LRs para {Y} samples")
```

Garante que o calibrador retornou um log-LR para cada par. Falha
inesperada (ex.: calibrador 1D recebeu input multi-D) é
capturada aqui antes da escrita.

---

## 5. Vínculo dataset ↔ calibrador (`validate_compatibility`)

### 5.1 O contrato

O calibrador foi treinado com features específicas (cosseno
escalar, ou matriz BILATERAL_FULL_CENTERED `(N, 7)`, ou hinge
piecewise sobre PLDA, etc.). Aplicá-lo a um dataset com features
diferentes produz log-LRs sem sentido — mesmo tipo
`SimilarityScoreDataset`, mas valores numéricos incompatíveis.

A solução é o `vectorization_contract_fingerprint`: hash do
contrato de vetorização que produziu cada dataset. O calibrador
carrega o fingerprint do dataset sobre o qual foi treinado; o
executor de `SCORE_CALIBRATION` compara o fingerprint do dataset
atual com o do calibrador.

### 5.2 A tabela assimétrica

`CalibrationModelMetadataService.validate_compatibility` aplica
esta tabela:

| Dataset | Calibrador | STRICT | LEGACY_PERMISSIVE |
|---|---|---|---|
| sem fingerprint | sem fingerprint | OK + warning | OK + warning |
| com fingerprint | com fingerprint igual | OK | OK |
| com fingerprint | com fingerprint diferente | erro | erro |
| com fingerprint | sem fingerprint | erro | warning forte |
| sem fingerprint | com fingerprint | erro sempre | erro sempre |

A última linha (dataset legado sem fp + calibrador moderno com
fp) é **sempre proibida** — independente do modo. Não há como
verificar que as features atuais correspondem ao que o calibrador
espera, e seguir adiante produziria log-LRs aparentemente válidos
sobre matemática totalmente incompatível.

### 5.3 Retorno

`validate_compatibility` retorna `List[CompatibilityIssue]`. Cada
issue tem:
- `field` (geralmente `vectorization_contract_fingerprint`);
- `expected` / `actual`;
- `severity` (`"error"` ou `"warning"`);
- `message` (texto humanamente legível).

### 5.4 Tratamento no executor

```python
for issue in issues:
    if issue.severity == "error":
        return self._failure_result(
            f"Incompatibilidade dataset↔calibrador "
            f"({issue.field}): {issue.message}"
        )
    elif issue.severity == "warning":
        self._log(context, f"  ⚠ {issue.message}")
```

Erros bloqueiam a calibração com mensagem clara. Warnings são
logados mas não bloqueiam.

---

## 6. Os dois modos de compatibilidade

`CalibrationCompatibilityMode` é o enum que controla a tabela
assimétrica:

### 6.1 `STRICT` (default)

> Vínculo dataset↔calibrador validado integralmente. Dataset sem
> fingerprint + calibrador com fingerprint → erro.

Comportamento defensável forensicamente. Em produção, todas as
combinações exceto "ambos modernos com fingerprints iguais" e
"ambos legados sem fingerprint" são rejeitadas com erro.

### 6.2 `LEGACY_PERMISSIVE`

> Permissivo legado (opt-in). Calibrador novo aceita dataset legado
> sem fingerprint com warning forte. Não tolera mismatch entre
> fingerprints presentes.

Modo de migração. Útil durante a transição da era
pré-`LR_ARCHITECTURE_VECTORIZATION` (sem fingerprints) para a era
V2.14 (com fingerprints). Permite aplicar calibradores antigos a
datasets novos vetorizados com warning visível.

> **Importante**: o DTO `ScoreCalibrationInstruction` desta versão
> **não expõe** um campo `compatibility_mode` na UI. O executor
> usa o default `STRICT` da
> `CalibrationModelMetadataService.validate_compatibility`. Para
> rodar em `LEGACY_PERMISSIVE`, é necessário chamar o serviço
> diretamente (CLI/scripting) ou estender o DTO.

### 6.3 Por que a assimetria

A combinação "dataset sem fp + calibrador com fp" é proibida
mesmo em `LEGACY_PERMISSIVE` porque o calibrador é o **artefato
contratual**. Ele foi treinado com expectativa específica sobre
as features (por exemplo, a matriz vetorizada `(N, 7)` da
`BILATERAL_FULL_CENTERED`, na ordem
`[base, h⁺, h⁻, h⁺·r̃_F, h⁺·r̃_P, h⁻·r̃_F, h⁻·r̃_P]`) e declara
isso no fingerprint. Um dataset sem fingerprint não tem como
provar que entrega essas features — bloquear sempre é a única
opção segura.

A combinação inversa ("dataset com fp + calibrador sem fp") é
erro em `STRICT` e só passa, com warning forte, em
`LEGACY_PERMISSIVE`: o calibrador legado não declara o que espera,
e quem opta pelo modo permissivo assume a responsabilidade de
garantir que as features coincidem.

---

## 7. Cópia defensiva dos samples — anti-mutação in-place

### 7.1 O bug histórico

A versão anterior do executor mutava `sample.metadata["log_lr"]`
**in-place** no input. Premissa errada: "mutação in-place é segura
porque o input não será reusado depois."

Falha porque o Run All passa por
`_persist_trained_model_resources_inline`, que itera
`trained_resources` (TODOS os recursos no contexto, inclusive o
input) e regrava os que ainda não tinham `artifact_ref` no início
da execução. Como a mutação adicionou `log_lr` no metadata dos
items compartilhados, o parquet do input ia parar no disco com a
coluna `log_lr` populada — fazendo a validação
`SCORE_CALIBRATION_STATE(RAW)` falhar na próxima execução
(`expected="RAW"`, encontrado "CALIBRATED").

### 7.2 A solução atual

Shallow copy de cada sample + dict de metadata novo:

```python
output_items: List = [None] * n
CHUNK = 50_000
for chunk_start in range(0, n, CHUNK):
    chunk_end = min(chunk_start + CHUNK, n)
    for j in range(chunk_start, chunk_end):
        src = score_dataset.items[j]
        new_sample = copy.copy(src)
        old_md = src.metadata if isinstance(src.metadata, dict) else {}
        new_sample.metadata = {**old_md, "log_lr": float(log_lrs[j])}
        output_items[j] = new_sample
    time.sleep(0)  # cede GIL à main thread
```

`copy.copy` em ~5M samples leva poucos segundos com chunked
sleep — mesma ordem de grandeza da mutação in-place — e libera a
GIL periodicamente, evitando o freeze da UI.

### 7.3 Não introduzir mutação in-place

> **Regra invariável**: o executor de `SCORE_CALIBRATION` **deve**
> produzir um dataset novo com samples copiados. Nunca mutar o
> input. A degradação de performance é negligenciável; a violação
> de invariante de tipo (`raw_score → log_lr` no mesmo objeto)
> propaga bugs por toda a cadeia downstream.

---

## 8. Propagação de telemetria para `STRATEGY_EVALUATION`

### 8.1 Quais campos

```python
for key in (
    "calibrator_cllr_train",
    "calibrator_cllr_train_gen",
    "calibrator_cllr_train_imp",
    "calibrator_cllr_cv",
    "calibrator_cllr_cv_gen",
    "calibrator_cllr_cv_imp",
    "calibrator_cv_gap",
    "cv_folds_effective",
    "cv_split_mode",
    "cv_hyperparameter_mode",
):
    if key in calibrator_md and calibrator_md[key] is not None:
        ds_metadata[key] = calibrator_md[key]
```

### 8.2 Por que propagar

O `STRATEGY_EVALUATION` consome o `CALIBRATED_SCORES` para
computar métricas forenses (EER, Cllr operacional, Cllr_min PAVA
in-sample, AUC, ECE). Ele **não** tem acesso direto ao calibrador
— recebe apenas o dataset calibrado.

Propagar `calibrator_cllr_train` / `calibrator_cllr_cv` /
`calibrator_cv_gap` para o metadata do dataset permite que a
tabela de métricas do `ValidationResultViewerDialog` exiba:

- **Cllr (operacional)** — computado sobre os log-LRs do dataset
  atual;
- **Cllr_min (PAVA in-sample)** — referência diagnóstica não
  operacional (Morrison 2021);
- **Cllr in-sample do calibrador** (propagado) — referência
  histórica do calibrador;
- **Cllr CV out-of-fold do calibrador** (propagado quando
  `k_folds > 0`) — generalização DENTRO do domínio de calibração.

Sem essa propagação, a tabela perderia a referência cruzada entre
domínio de calibração e domínio de validação.

### 8.3 Resiliente a campos ausentes

Se o calibrador foi treinado com `k_folds=0` (default), os campos
`calibrator_cllr_cv*` não existem — o loop apenas pula.
Calibradores legados sem `metadata` (raros) também passam
silenciosamente.

---

## 9. Propagação de coeficientes raw + piecewise state

### 9.1 Por que propagar

O `STRATEGY_EVALUATION` produz vários plots que **dependem dos
coeficientes do calibrador**:

- **Plot 2D analítico** (heatmap log-LR num plano 2D, eixos
  `x_0`, `x_1`) — quando o calibrador é LogReg afim, a superfície
  log-LR é analítica: `β_0 + β_1·x_0 + β_2·x_1`. Plotar isso
  analiticamente produz superfície exata; interpolar dos pares
  observados produziria escada visível.

- **Plot piecewise-linear** — quando o calibrador tem expansão
  piecewise, a curva log-LR tem dobras. Plotar dos pares
  observados colapsa múltiplas dobras numa só visualmente; plotar
  analiticamente preserva o formato.

- **Equação no rodapé do plot** — exibe a fórmula
  `β_0 + β_1·x_0 + ...` para auditoria pericial.

### 9.2 Como propaga

Dois caminhos:

#### 9.2.1 Calibrador piecewise

```python
if piecewise_active:
    if coef_raw_list is not None and intercept_raw_val is not None:
        ds_metadata["calibrator_coef_raw"]      = list(coef_raw_list)
        ds_metadata["calibrator_intercept_raw"] = float(intercept_raw_val)
    for key in (
        "piecewise_enabled",
        "hinge_anchor_s0",
        "piecewise_knot_offsets",
        "piecewise_knot_offsets_user",
        "piecewise_knot_offsets_raw",
        "piecewise_knot_scale",
        "piecewise_knot_placement",
        "piecewise_auto_log_lr_hp_p50",
        "piecewise_auto_log_lr_hd_p50",
        "piecewise_knots_absolute",
        "segment_slopes_raw",
    ):
        if key in calibrator_md and calibrator_md[key] is not None:
            ds_metadata[key] = calibrator_md[key]
```

O calibrador piecewise já pré-calculou `calibrator_coef_raw` no
`_validate_monotonicity_and_persist_diagnostics`. O executor de
`SCORE_CALIBRATION` apenas copia.

#### 9.2.2 Calibrador LogReg "puro" (sem piecewise)

```python
elif sk_coef is not None and sk_intercept is not None:
    fm = getattr(calibrator, "_feature_mean", None)
    fs = getattr(calibrator, "_feature_std", None)
    if fm is not None and fs is not None:
        # Standardização ativa: coef sklearn em escala Z.
        # Denormaliza para escala bruta:
        coef_raw      = sk_coef / fs_arr
        intercept_raw = sk_intercept − Σ(sk_coef · μ / σ)
    else:
        # Sem standardização: coef sklearn já está em escala bruta.
        coef_raw      = sk_coef
        intercept_raw = sk_intercept
    ds_metadata["calibrator_coef_raw"]      = coef_raw.tolist()
    ds_metadata["calibrator_intercept_raw"] = intercept_raw
```

Importante: quando `standardize_features=True` (multi-D), o
`logreg.coef_` está em escala Z. O executor reverte aplicando
`coef_raw = coef_z / σ` e `intercept_raw = intercept_z − Σ(coef_z · μ / σ)`.

#### 9.2.3 Calibradores não-LogReg (KDE/BiGauss)

Os campos `calibrator_coef_raw` / `calibrator_intercept_raw`
ficam ausentes. `STRATEGY_EVALUATION` cai em fallback de
interpolação dos pares observados.

### 9.3 Tratamento de erro

A denormalização é envolvida em `try/except`:

```python
except Exception as exc:
    self._log(
        context,
        f"⚠️ Falha ao propagar coefs do calibrador "
        f"({type(exc).__name__}: {exc}); plot 2D vai cair em interpolação."
    )
```

Falha na propagação **não bloqueia** a calibração — apenas degrada
o plot. O log-LR no parquet é o número canônico.

---

## 10. Preservação do flag `is_casework`

```python
ds_metadata["is_casework"] = bool(
    score_dataset.metadata.get("is_casework", False)
)
```

### 10.1 Por que importa

Em casework forense, o dataset não tem ground truth — não há
`same_person` rotulado. Esse fato precisa **propagar** através de
todas as instruções downstream:

- `STRATEGY_EVALUATION` **rejeita** datasets `is_casework=True`
  (a regra `CASEWORK_STATE(False)` falha) — não faz sentido
  computar EER sobre dados sem rótulo;
- `CASEWORK_VISUALIZATION` **exige** datasets `is_casework=True`
  para o input "caso" (a regra `CASEWORK_STATE(True)` valida).

Sem a propagação, o usuário poderia acidentalmente passar dataset
de caso para a instrução de validação ou vice-versa. O metadata
preservado permite as regras dinâmicas (`build_validation_rules`)
bloquear o erro em runtime.

### 10.2 Casework + raw_score

Em casework, o calibrador é aplicado normalmente — produz
`log_lr` por par. Apenas o `same_person` permanece ausente. O
`STRATEGY_EVALUATION` não consegue computar Cllr sem rótulos, mas
`CASEWORK_VISUALIZATION` consegue produzir o plot (KDE/histograma)
e o `case_summary` (média, mediana, percentis dos log-LRs do
caso) — diagnóstico forense end-to-end.

---

## 11. Regras de validação dinâmicas

`build_validation_rules` retorna 3 regras (não persistidas):

### 11.1 `ModelFittedRule`

```python
ModelFittedRule(
    rule_id=f"{instruction_id}_calibrator_fitted",
    model_resource_id=trained_calibration_model_resource_id,
)
```

Verifica que o calibrador tem `is_fitted=True`. Calibrador não
treinado seria erro silencioso na hora do `calibrate_scores`.

### 11.2 `ScoreCalibrationStateRule(RAW)`

```python
ScoreCalibrationStateRule(
    rule_id=f"{instruction_id}_scores_raw",
    scores_resource_id=input_similarity_scores_resource_id,
    expected="RAW",
)
```

Verifica que o input contém scores RAW (sem `log_lr` populado).
Detecta dupla calibração — aplicar calibrador a dataset já
calibrado produziria log-LR de log-LR, sem semântica forense.

A regra amostra 100 itens do dataset (performance) — se nenhum
tem `log_lr` no metadata, presume "RAW".

### 11.3 `ScoreVectorOrderConsistencyRule`

```python
ScoreVectorOrderConsistencyRule(
    rule_id=f"{instruction_id}_score_vector_order_consistency",
    scores_resource_id=input_similarity_scores_resource_id,
    calibrator_resource_id=trained_calibration_model_resource_id,
)
```

Aplica-se quando o calibrador foi treinado em score vector
multi-D. A ordem das colunas (sistemas-fonte) precisa ser
preservada entre treino e aplicação — `[cosseno, plda]` no treino
exige `[cosseno, plda]` na aplicação, não `[plda, cosseno]`.

Verifica que `score_dataset.metadata["score_vector_model_ids"]`
bate com `calibrator.score_vector_model_ids` (persistido no fit).

---

## 12. Diagnóstico de log-LR

Após a calibração, o executor computa estatísticas básicas:

```python
labels_arr = score_dataset.labels
mean_log_lr_genuines = (
    float(np.mean(log_lrs[labels_arr == 1]))
    if np.any(labels_arr == 1) else None
)
mean_log_lr_impostors = (
    float(np.mean(log_lrs[labels_arr == 0]))
    if np.any(labels_arr == 0) else None
)
```

Logados no contexto:

```
[Score Calibration] ✅ Calibração concluída.
log-LR médio: gen=2.34, imp=-3.12.
```

Retornados em `InstructionExecutionResult.metadata`:

```python
{
    "n_samples": n,
    "n_genuine":  score_dataset.total_genuines,
    "n_impostor": score_dataset.total_impostors,
    "mean_log_lr_genuines": ...,
    "mean_log_lr_impostors": ...,
    ...
}
```

Diagnóstico rápido — se `mean_log_lr_gen ≤ mean_log_lr_imp`, o
calibrador está invertido (raríssimo) ou o input não tinha
`same_person` correto.

Em casework os pares não têm rótulo (`labels` devolve `-1`), e
`mean_log_lr_genuines` e `mean_log_lr_impostors` ficam ambos
`None`.

---

## 13. Persistência em parquet

`SimilarityScoreDataset.to_parquet` detecta a presença de
`log_lr` em `sample.metadata` e adiciona uma coluna `log_lr`
nativa ao schema parquet — sem precisar declarar manualmente.

Layout típico do parquet de saída:

| Coluna | Tipo | Origem |
|---|---|---|
| `similarity_score_dataset_name` | str | Nome do dataset |
| `similarity_score_dataset_description` | str | Descrição (só quando há) |
| `probe_identity` | str | Sample original |
| `reference_identity` | str | Sample original |
| `probe_face_hash` | str | Sample original |
| `reference_face_hash` | str | Sample original |
| `same_person` | bool | Sample original (nulo em casework) |
| `label` | int8 (1/0, `-1` sem rótulo) | Sample original |
| `raw_score` | float64 | Sample original |
| `log_lr` | float64 | **Adicionado por `SCORE_CALIBRATION`** |
| `score_vector_0` … `score_vector_{D-1}` | float64 | Matriz `(N, D)`, só em dataset vetorizado |

A coluna `log_lr` é o output canônico. Todas as instruções
downstream (`STRATEGY_EVALUATION`, `CASEWORK_VISUALIZATION`) leem
dela.

Carregamento via `SimilarityScoreDataset.from_parquet` reabre os
samples com `metadata["log_lr"]` populado — round-trip
preservado.

---

## 14. Casos de uso

### 14.1 Calibração de score 1D direto

```
SCORE_CALIBRATION(
    trained_calibration_model_resource_id = "cal_logreg_simples",
    input_similarity_scores_resource_id   = "cosseno_D_val",
    output_calibrated_scores_resource_id  = "cal_scores_D_val",
)
```

Calibrador LogReg afim aplicado a dataset 1D de cosseno. O
parquet de saída ganha coluna `log_lr` nativa. Nenhum
fingerprint comparado (dataset e calibrador sem fingerprint —
o serviço emite o warning informativo de §14.4).

### 14.2 Calibração de matriz vetorizada com fingerprint

```
LR_ARCHITECTURE_VECTORIZATION(  ← upstream, produz fingerprint
    architecture           = BILATERAL_FULL_CENTERED,
    parameter_mode         = INHERIT,
    base_score_resource_id = "cosseno_D_val",
    output_resource_id     = "vec_BIL_D_val",
)
        │
        ▼
SCORE_CALIBRATION(
    trained_calibration_model_resource_id = "cal_BILATERAL_cheia",
    input_similarity_scores_resource_id   = "vec_BIL_D_val",
    output_calibrated_scores_resource_id  = "cal_BIL_D_val",
)
```

`validate_compatibility` verifica
`vectorization_contract_fingerprint` do dataset vs do calibrador.
Em `STRICT`, fingerprints diferentes bloqueiam com erro.
Fingerprints iguais (caso operacional: V2.14 declarativa em
`INHERIT`) passam sem warning.

### 14.3 Calibração em casework

```
SCORE_CALIBRATION(
    trained_calibration_model_resource_id = "cal_BILATERAL_cheia",
    input_similarity_scores_resource_id   = "vec_BIL_caso_questionado",
    output_calibrated_scores_resource_id  = "cal_BIL_caso_questionado",
)
```

O input tem `is_casework=True` no metadata. A calibração
funciona normalmente — calibrador é aplicado, log-LR é populado
por par. `is_casework` é propagado para o output.

Downstream:
- `STRATEGY_EVALUATION` rejeita esse output (`CASEWORK_STATE(False)`
  falha);
- `CASEWORK_VISUALIZATION` aceita (`CASEWORK_STATE(True)` valida) e
  produz o plot forense.

### 14.4 Calibração com calibrador legado (sem fingerprint)

```
SCORE_CALIBRATION(
    trained_calibration_model_resource_id = "cal_logreg_legado_2024",
    input_similarity_scores_resource_id   = "cosseno_D_val_legado",
    output_calibrated_scores_resource_id  = "cal_legado_D_val",
)
```

Ambos sem fingerprint. `validate_compatibility` retorna issue
`severity=warning` com mensagem "Dataset e calibrador são legados
(sem fingerprint). Vínculo não pode ser verificado — assumindo
compatível por contexto histórico." Calibração segue.

---

## 15. Boas práticas

### 15.1 Treinamento → aplicação consistentes

1. **Sempre use o mesmo upstream**. Se o calibrador foi treinado
   sobre `LR_ARCHITECTURE_VECTORIZATION(BILATERAL_FULL_CENTERED,
   FIT_PER_FOLD)`, o dataset de validação precisa passar por
   `LR_ARCHITECTURE_VECTORIZATION(BILATERAL_FULL_CENTERED,
   INHERIT)` antes de chegar ao `SCORE_CALIBRATION`. O fingerprint
   garante o vínculo.

2. **`INHERIT` em validação e casework**. Em
   `LR_ARCHITECTURE_VECTORIZATION` downstream, sempre use
   `parameter_mode=INHERIT` — preserva `s₀`, `r̄_F`, `r̄_P`
   aprendidos no domínio de calibração. Refit silencioso em
   validação contamina o LR (vide
   `docs/interacao_score_raridade_faceaudit.md`).

### 15.2 Compatibilidade

3. **Mantenha `STRICT` em produção**. Default seguro. O modo
   `LEGACY_PERMISSIVE` é opt-in explícito por chamadores
   programáticos do serviço durante migração — não usar em laudo
   sem justificativa.

4. **Audite mismatches**. Se `validate_compatibility` retorna erro,
   investigue: foi rebuilt do upstream? Recriação do calibrador?
   Edição manual de pipeline? Não force.

### 15.3 Casework

5. **Marque o caso como `is_casework=True`** desde o
   `SIMILARITY_SCORING`. O flag propaga automaticamente — não
   precisa ser re-injetado em cada etapa.

6. **Não calibre dataset de validação com flag de casework**. A
   regra `CASEWORK_STATE` na `STRATEGY_EVALUATION` vai bloquear
   downstream com mensagem clara.

### 15.4 Performance

7. **Dataset grande?**. O executor já lida com chunks de 50k +
   `sleep(0)` para liberar a GIL. Em datasets ≥ 5M pares, espere
   alguns segundos — mas a UI não congela.

8. **Não mute o input**. O executor produz dataset novo
   intencionalmente (vide §7). Não tente "otimizar" mutando
   in-place — historicamente quebrou o Run All persistindo
   `log_lr` no parquet do input.

---

## 16. Perguntas frequentes

**P1: Posso aplicar um calibrador KDE a um dataset vetorizado
`(N, D)` com `D > 1`?**

Não. KDE espera score 1D. O calibrador tenta extrair scalar,
falha ou produz comportamento indefinido. Use LogReg multivariado
treinado com `standardize_features=True`.

**P2: O calibrador foi treinado em A. Posso aplicá-lo em B sem
passar pelo mesmo upstream?**

Tecnicamente sim, mas é semanticamente errado. Se A tinha
`LR_ARCHITECTURE_VECTORIZATION(BILATERAL_FULL_CENTERED)` e B
tem apenas score cosseno bruto, os shapes batem (`D=1` no caso de
HINGE_ONLY contra cosseno cru, por exemplo), mas a matemática
está incompatível. Em `STRICT`, o fingerprint mismatch bloqueia.
Em modo permissivo, o calibrador vai produzir log-LRs sem
significado.

**P3: Como sei se o calibrador é "moderno" (tem fingerprint) ou
"legado" (sem)?**

Cheque `calibrator.metadata.get("vectorization_contract_fingerprint")`.
Calibradores treinados sobre dataset de
`LR_ARCHITECTURE_VECTORIZATION` expõem o fingerprint em qualquer
`parameter_mode` (inclusive `FIT_FULL_CALIBRATION_LEGACY`).
Calibradores treinados sobre score 1D, sobre
`SCORE_VECTORIZATION` ou antes da V2.14 não.

**P4: O dataset de saída tem `raw_score` preservado?**

Sim. A cópia rasa dos samples preserva todos os campos
originais — `raw_score`, `same_person`, `probe_hash`,
`reference_hash`, `probe_identity`, `reference_identity`, etc.
Apenas o `metadata["log_lr"]` é injetado.

**P5: Quando o dataset tem `is_score_vector=True`, qual é o
`raw_score` propagado?**

`raw_score` em cada sample é o **score original do primeiro
sistema** (geralmente cosseno ou PLDA escalar) — a 1ª coluna da
matriz `(N, D)`. A matriz completa fica em
`metadata[SCORE_VECTOR_MATRIX_KEY]`. O calibrador consome a
matriz; o `raw_score` é mantido por compatibilidade com
visualizações antigas.

**P6: O log-LR persistido no parquet é o número canônico?**

Sim. `SimilarityScoreDataset.to_parquet` detecta `log_lr` em
metadata e cria coluna nativa. `from_parquet` reabre os samples
com `metadata["log_lr"]` populado. Round-trip preservado, não há
perda de precisão.

**P7: O que acontece se o calibrador retornar log-LR `nan`?**

Comportamento indefinido — o executor não filtra `nan` no
output. Downstream (`STRATEGY_EVALUATION`, `CASEWORK_VISUALIZATION`)
pode quebrar ou produzir métricas inválidas. Em prática, o
calibrador piecewise rejeita inclinações zero/negativas já no
ajuste (`monotonicity_policy`); e KDE tem clipping
`min_log_lr`/`max_log_lr`. Mas se você suspeita de `nan`,
verifique o input com `np.isnan(raw_scores).any()`.

**P8: Por que telemetria CV do calibrador é propagada para o
dataset de validação?**

Para que `STRATEGY_EVALUATION` exiba referência histórica do
calibrador na tabela de métricas — Cllr in-sample do treino,
Cllr CV out-of-fold do treino, e Cllr operacional do dataset de
validação ao mesmo tempo. Permite auditoria pericial cruzada
(generalização dentro do domínio de calibração vs performance no
dataset de validação).

---

## 17. Glossário

| Termo | Definição |
|---|---|
| `log_lr` | Logaritmo natural da razão de verossimilhança `log(P(score | H_p) / P(score | H_d))`. Output canônico de `SCORE_CALIBRATION`. |
| `raw_score` | Score bruto do modelo de similaridade — cosseno em `[-1, +1]`, PLDA em unidades arbitrárias, ou matriz `(N, D)` em score vector. |
| `is_score_vector` | Flag no metadata indicando que o dataset tem matriz `(N, D)` em vez de score escalar. |
| `is_casework` | Flag no metadata indicando dataset sem ground truth (`same_person` ausente). Propaga via `SCORE_CALIBRATION`. |
| `vectorization_contract_fingerprint` | Hash do contrato de vetorização. Comparado entre dataset e calibrador via `CalibrationModelMetadataService.validate_compatibility`. |
| `STRICT` | Modo de compatibilidade default. Bloqueia "dataset com fp + calibrador sem fp". |
| `LEGACY_PERMISSIVE` | Modo de compatibilidade opt-in. Aceita calibrador legado em dataset moderno com warning forte. Não tolera mismatch entre fingerprints presentes. |
| `CalibrationModelProtocol` | Interface estrutural exigida do calibrador: `fit`, `predict_proba`, `calibrate_scores`. |
| `calibrate_scores(raw)` | Método canônico do calibrador — converte raw scores em log-LRs. Aceita 1D ou `(N, D)`. |
| `cópia defensiva` | Shallow copy dos samples + dict de metadata novo. Evita mutação in-place do input que historicamente quebrou o Run All. |
| `calibrator_coef_raw` | Coeficientes do LogReg denormalizados da `StandardScaler` interna. Propagado ao output para plots analíticos. |
| `calibrator_intercept_raw` | Intercept denormalizado. Propagado ao output. |
| `piecewise_*` | Campos de state da expansão piecewise-linear (V1). Propagados ao output para plot analítico. |
| `score_vector_source_resources` | Lista ordenada dos recursos-fonte das colunas da matriz `(N, D)`. Propagado ao output para os gráficos do `STRATEGY_EVALUATION`. (A regra `SCORE_VECTOR_ORDER_CONSISTENCY` compara outro campo, `score_vector_model_ids`, no input.) |

---

## 18. Referências

### 18.1 Documentos internos

- `docs/calibration_model_training_faceaudit.md` — instrução irmã
  que treina o calibrador consumido aqui.
- `docs/calibracao_faceaudit.md` — referência guarda-chuva sobre
  calibração.
- `docs/lr_architecture_vectorization_faceaudit.md` — produz
  `vectorization_contract_fingerprint` e `architecture_state`
  consumidos aqui.
- `docs/piecewise_linear_calibration.md` — especificação da
  expansão piecewise-linear (campos de state propagados aqui).
- `docs/conditional_monotonicity_rule_design.md` — auditoria do
  calibrador (metadata `monotonicity_audit`, gravado no calibrador
  pelo `CALIBRATION_MODEL_TRAINING` após o refit).
- `docs/vetorizacao_scores_faceaudit.md` — vetorização explícita
  via `SCORE_VECTORIZATION` (legado).

### 18.2 Bibliografia técnica

- Brümmer, N., & du Preez, J. (2006). *Application-independent
  evaluation of speaker detection*. Computer Speech & Language,
  20(2-3), 230–275. — origem do Cllr propagado aqui.
- Morrison, G. S. (2021). *In the context of forensic casework,
  are there meaningful metrics of the degree of calibration?*.
  FSI: Synergy. — fundamenta a propagação separada de Cllr in-sample
  (referência diagnóstica) vs Cllr operacional.

---

## Apêndice A — `ScoreCalibrationInstruction`

Campos do DTO em
`faceaudit/models/instructions/score_calibration_instruction.py`:

```python
@dataclass
class ScoreCalibrationInstruction(StrategyInstructionDefinition):
    trained_calibration_model_resource_id: Optional[str] = None
    input_similarity_scores_resource_id:   Optional[str] = None
    output_calibrated_scores_resource_id:  Optional[str] = None
    output_calibrated_scores_name:         Optional[str] = None
```

Sem hyperparâmetros — toda configuração está no calibrador
treinado.

Resource field specs:
- `trained_calibration_model_resource_id`: aceita
  `TRAINED_CALIBRATION_MODEL`;
- `input_similarity_scores_resource_id`: aceita
  `SIMILARITY_SCORES`;
- `output_calibrated_scores_resource_id`: produz
  `CALIBRATED_SCORES`.

Regras de validação dinâmicas (não persistidas):
- `ModelFittedRule(calibrator)`;
- `ScoreCalibrationStateRule(input, expected="RAW")`;
- `ScoreVectorOrderConsistencyRule(input, calibrator)`.

---

## Apêndice B — Tabela de compatibilidade dataset × calibrador

| Dataset | Calibrador | `STRICT` | `LEGACY_PERMISSIVE` | Semântica |
|---|---|---|---|---|
| sem fp | sem fp | ✅ OK + warning informativo | ✅ OK + warning informativo | Ambos legados — vínculo não verificável, assumido por contexto histórico |
| com fp | com fp **igual** | ✅ OK | ✅ OK | Caso operacional normal V2.14 |
| com fp | com fp **diferente** | ❌ erro | ❌ erro | Features atuais não correspondem ao calibrador — bloqueia sempre |
| com fp | sem fp | ❌ erro | ⚠ warning forte | Dataset moderno + calibrador legado — opt-in arriscado |
| sem fp | com fp | ❌ erro sempre | ❌ erro sempre | Calibrador moderno + dataset legado — sempre proibido (não há como verificar) |

---

## Apêndice C — Campos propagados no metadata do output

O dataset de saída (`CALIBRATED_SCORES`) ganha no metadata:

### C.1 Campos sempre presentes

| Campo | Origem | Função |
|---|---|---|
| `source_dataset_name` | Input | Nome do dataset original |
| `calibrator_type` | `type(calibrator).__name__` | LogReg/KDE/BiGauss |
| `is_casework` | Input | Preserva semântica "sem ground truth" |

### C.2 Telemetria (quando presente no calibrador)

| Campo | Origem | Significado |
|---|---|---|
| `calibrator_cllr_train` | metadata do calibrador, gravado pelo `CALIBRATION_MODEL_TRAINING` | Cllr in-sample do calibrador |
| `calibrator_cllr_train_gen` / `calibrator_cllr_train_imp` | idem | Componentes |
| `calibrator_cllr_cv` | idem (quando `k_folds > 0`) | Cllr CV out-of-fold |
| `calibrator_cllr_cv_gen` / `calibrator_cllr_cv_imp` | idem | Componentes |
| `calibrator_cv_gap` | idem | `cllr_cv − cllr_train` |
| `cv_folds_effective` | idem | Folds usados |
| `cv_split_mode` | idem | `STRATIFIED_PAIR` ou `STRATIFIED_GROUP` |
| `cv_hyperparameter_mode` | idem | `"FROZEN"` (hiperparâmetros congelados do modelo de produção; não é nested CV) |

### C.3 Score vector (quando `is_score_vector=True`)

| Campo | Origem | Função |
|---|---|---|
| `is_score_vector` | Input | Sinaliza que dataset é multi-D |
| `SCORE_VECTOR_MATRIX_KEY` | Input | Matriz `(N, D)` |
| `score_vector_source_resources` | Input | Lista ordenada dos sistemas-fonte |

### C.4 Coeficientes raw (quando LogReg)

| Campo | Origem | Função |
|---|---|---|
| `calibrator_coef_raw` | LogReg + denormalização ou direto do piecewise | Plot 2D analítico |
| `calibrator_intercept_raw` | idem | Plot 2D analítico |

### C.5 Piecewise state (quando piecewise ativo)

| Campo | Origem | Função |
|---|---|---|
| `piecewise_enabled` | Calibrador | Sinaliza piecewise ativo |
| `hinge_anchor_s0` | Calibrador | Ponto neutro |
| `piecewise_knot_offsets` | Calibrador | Offsets das dobras |
| `piecewise_knot_offsets_user` | Calibrador | Offsets como o usuário declarou |
| `piecewise_knot_offsets_raw` | Calibrador | Offsets em escala bruta após conversão |
| `piecewise_knot_scale` | Calibrador | `AUX_LOG_LR` ou `RAW` |
| `piecewise_knot_placement` | Calibrador | `MANUAL` ou `AUTO_FROM_MIRRORED_MEDIANS` |
| `piecewise_auto_log_lr_hp_p50` | Calibrador | Mediana log-LR genuínos (auxiliar) |
| `piecewise_auto_log_lr_hd_p50` | Calibrador | Mediana log-LR impostores (auxiliar) |
| `piecewise_knots_absolute` | Calibrador | Posições absolutas das dobras |
| `segment_slopes_raw` | Calibrador | Inclinações por trecho |

---

## Apêndice D — Checklist operacional

Antes de aplicar:

- [ ] O calibrador foi treinado sobre o mesmo upstream que produz
      o dataset atual?
- [ ] Em `LR_ARCHITECTURE_VECTORIZATION` upstream do dataset, está
      em `parameter_mode=INHERIT`?
- [ ] O dataset atual é o de **validação** (`is_casework=False`) ou
      o de **caso questionado** (`is_casework=True`)?
- [ ] Verifica `validate_compatibility` retornou OK (sem erro)?

Após aplicar:

- [ ] `log_lr` populado em todos os samples do output?
- [ ] `n_samples = n_genuine + n_impostor` (exceto casework)?
- [ ] `mean_log_lr_genuines > mean_log_lr_impostors` (sanity
      check; em casework, ambos são `None`)?
- [ ] Metadata do output preservou `is_casework` corretamente?
- [ ] Metadata do output propagou `calibrator_cllr_*` quando
      presente no calibrador?
- [ ] Metadata do output propagou `calibrator_coef_raw` /
      `calibrator_intercept_raw` quando LogReg?
- [ ] Parquet de saída tem coluna `log_lr` nativa?

Em laudo:

- [ ] Cita o `vectorization_contract_fingerprint` do dataset e do
      calibrador?
- [ ] Documenta o modo de compatibilidade usado (default
      `STRICT`)?
- [ ] Cita `calibrator_cllr_train` e `calibrator_cllr_cv` do
      calibrador para auditoria de generalização?
- [ ] Cita `monotonicity_audit.audit_status` do calibrador (lido
      do metadata do calibrador; gravado pelo
      `CALIBRATION_MODEL_TRAINING` quando há `architecture_state`)?
