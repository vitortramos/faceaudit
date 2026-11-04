# Vetorização declarativa de arquitetura LR (`LR_ARCHITECTURE_VECTORIZATION`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `LR_ARCHITECTURE_VECTORIZATION` (V2.14),
> declarativa, que substitui o encadeamento manual `SCORE_INTERACTION_FEATURE × N + SCORE_VECTORIZATION`.
> Cobre as 15 arquiteturas (9 two-face + 6 ONE_FACE), os 18 termos
> elementares, os 4 modos de parâmetro (`ArchitectureParameterMode`),
> `LRArchitectureState` com dois fingerprints, payload bruto canônico,
> compatibilidade STRICT vs LEGACY_PERMISSIVE com o calibrador, e a
> integração com `IdentityDisjointPairKFold` no FIT_PER_FOLD.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Documentos relacionados:**
> `docs/interacao_score_raridade_faceaudit.md` (fundamentação da
> interação cosseno × raridade),
> `docs/calibracao_faceaudit.md` (calibração final),
> `docs/conditional_monotonicity_rule_design.md` (auditoria do
> calibrador V1.5/V2),
> `docs/typicality_pair_scoring_faceaudit.md` (proveniência da
> raridade upstream).
> **Fontes-código:** `faceaudit/models/instructions/lr_architecture_vectorization_instruction.py`,
> `faceaudit/executors/instructions/lr_architecture_vectorization_instruction_executor.py`,
> `faceaudit/enums/lr_calibration_architecture.py`,
> `faceaudit/enums/lr_feature_term.py`,
> `faceaudit/enums/architecture_parameter_mode.py`,
> `faceaudit/enums/architecture_fit_scope.py`,
> `faceaudit/enums/s0_resolution_method.py`,
> `faceaudit/enums/rarity_centering_mode.py`,
> `faceaudit/enums/pair_role_orientation.py`,
> `faceaudit/enums/pair_key_mode.py`,
> `faceaudit/enums/calibration_compatibility_mode.py`,
> `faceaudit/models/lr/lr_architecture_spec.py`,
> `faceaudit/models/lr/lr_architecture_state.py`,
> `faceaudit/models/lr/lr_architecture_raw_input_payload.py`,
> `faceaudit/services/architecture_feature_builder.py`,
> `faceaudit/services/architecture_vectorizer_transformer.py`,
> `faceaudit/services/architecture_state_compat_service.py`,
> `faceaudit/services/calibration_model_metadata_service.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a instrução se encaixa](#1-onde-a-instrução-se-encaixa)
- [2. O problema que ela resolve](#2-o-problema-que-ela-resolve)
- [3. As 15 arquiteturas](#3-as-15-arquiteturas)
- [4. Os 18 termos elementares (`LRFeatureTerm`)](#4-os-18-termos-elementares-lrfeatureterm)
- [5. Os 4 modos de parâmetro (`ArchitectureParameterMode`)](#5-os-4-modos-de-parâmetro-architectureparametermode)
- [6. `LRArchitectureState` — o estado declarativo](#6-lrarchitecturestate--o-estado-declarativo)
- [7. `LRArchitectureRawInputPayload` — o payload canônico](#7-lrarchitecturerawinputpayload--o-payload-canônico)
- [8. `ArchitectureFeatureBuilder` — o builder genérico](#8-architecturefeaturebuilder--o-builder-genérico)
- [9. O fluxo do executor passo a passo](#9-o-fluxo-do-executor-passo-a-passo)
- [10. Fingerprints e compatibilidade com o calibrador](#10-fingerprints-e-compatibilidade-com-o-calibrador)
- [11. FIT_PER_FOLD com `ArchitectureVectorizerTransformer`](#11-fit_per_fold-com-architecturevectorizertransformer)
- [12. Validações e guard-rails](#12-validações-e-guard-rails)
- [13. Auditoria de monotonicidade no calibrador](#13-auditoria-de-monotonicidade-no-calibrador)
- [14. Configurações](#14-configurações)
- [15. Casos de uso comuns](#15-casos-de-uso-comuns)
- [16. Boas práticas](#16-boas-práticas)
- [17. Perguntas frequentes](#17-perguntas-frequentes)
- [18. Glossário](#18-glossário)
- [19. Referências](#19-referências)
- [Apêndice A — `LRArchitectureVectorizationInstruction`](#apêndice-a--lrarchitecturevectorizationinstruction)
- [Apêndice B — Tabela completa das 15 arquiteturas](#apêndice-b--tabela-completa-das-15-arquiteturas)
- [Apêndice C — Checklist operacional](#apêndice-c--checklist-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre `LR_ARCHITECTURE_VECTORIZATION`:
o que ela substitui, como declarar uma das 15 arquiteturas LR, como o
executor materializa a matriz de features, como s₀/r̄ são obtidos
(quatro modos), como o estado e os fingerprints viajam para o
calibrador downstream, como o `IdentityDisjointPairKFold` recebe
identidades dos dois lados para CV sem leakage, e como o
`CONDITIONAL_MONOTONICITY` audita o calibrador resultante.

### 0.2 Por que existe

Antes da V2.14, materializar a arquitetura BILATERAL cheia exigia ~13
instruções `SCORE_INTERACTION_FEATURE` + 1 `SCORE_VECTORIZATION` por
estratégia. O usuário reproduzia manualmente a aritmética interna
para cada uma das nove variantes testadas; mudar de arquitetura
significava reescrever a estratégia inteira; cada combinação tinha
risco próprio de erro humano.

`LR_ARCHITECTURE_VECTORIZATION` substitui esse encadeamento por
**uma única instrução declarativa**:

```
LR_ARCHITECTURE_VECTORIZATION(
    architecture = LRCalibrationArchitecture.BILATERAL_FULL_CENTERED,
    parameter_mode = ArchitectureParameterMode.FIT_PER_FOLD,
    base_score_resource_id = "cosseno_D_cal",
    rarity_probe_resource_id = "typicality_probe_D_cal",
    rarity_reference_resource_id = "typicality_reference_D_cal",
    output_resource_id = "vec_BILATERAL_cheia",
    output_raw_input_resource_id = "raw_input_BILATERAL_cheia",
)
```

O executor consulta o `LRArchitectureSpec` daquela arquitetura no
registry `LR_ARCHITECTURE_SPECS` e materializa a matriz de features
via `ArchitectureFeatureBuilder.build(spec, state, base, r_F, r_P)`
genericamente (função de módulo `architecture_feature_builder.build`)
— zero `if-elif` por arquitetura.

### 0.3 Compatibilidade com o fluxo antigo

A V2.14 **não obriga** migração. O encadeamento histórico
`SCORE_INTERACTION_FEATURE` + `SCORE_VECTORIZATION` continua válido com
`legacy_transform_typicality` para preservar tolerância numérica das
estratégias V2.13. Calibradores antigos sem fingerprint aplicados a
dataset também sem fingerprint passam com warning informativo; a
combinação dataset novo + calibrador antigo é rejeitada no
`SCORE_CALIBRATION`, que hoje sempre valida em `STRICT` (o modo
`LEGACY_PERMISSIVE` existe em `CalibrationCompatibilityMode` e no
`CalibrationModelMetadataService`, mas a instrução não expõe campo
para escolhê-lo — vide §10.2).

---

## 1. Onde a instrução se encaixa

```
SIMILARITY_SCORING (cosseno ou PLDA)        TYPICALITY_PAIR_SCORING (probe + reference)
    │                                                       │
    │ produz SIMILARITY_SCORES                              │ produz SIMILARITY_SCORES (raridade)
    │   (base)                                              │
    └───────────────────────┬───────────────────────────────┘
                            │
                            ▼
           LR_ARCHITECTURE_VECTORIZATION
            ├─ architecture: LRCalibrationArchitecture
            ├─ parameter_mode: ArchitectureParameterMode
            └─ Materializa φ(s, r_F, r_P, s₀, r̄_F, r̄_P) → matriz (N, D)
                            │
                            │ produz SIMILARITY_SCORES vetorizado +
                            │ LR_ARCHITECTURE_RAW_INPUT (FIT_PER_FOLD)
                            ▼
            CALIBRATION_MODEL_TRAINING (LogReg) → SCORE_CALIBRATION → STRATEGY_EVALUATION
```

O output `SIMILARITY_SCORES vetorizado` preserva em cada item o
`raw_score` escalar do base (compat com consumidores escalares) e
carrega a matriz de features `(N, D)` no metadata
(`is_score_vector=True`, `score_vector_matrix`, `score_vector_dim`).
O calibrador `LogisticRegressionCalibrationModel`
[`docs/calibracao_faceaudit.md`] consome essa matriz como score
multivariado; com `standardize_features=True` popula o pipeline
`[StandardScaler → LogReg]` internamente.

O output `LR_ARCHITECTURE_RAW_INPUT` (opcional em FIT_FULL/MANUAL;
necessário em FIT_PER_FOLD — vide §5.2) carrega a matriz bruta `[base, r_F,
r_P]` + identidades — é o que o `CALIBRATION_MODEL_TRAINING` consome
no FIT_PER_FOLD para ajustar s₀/r̄ **dentro** de cada fold do
GridSearchCV interno.

---

## 2. O problema que ela resolve

### 2.1 Encadeamento manual antigo

Antes da V2.14, a BILATERAL cheia D=7 precisava de:

| Step | Instrução | Termo computado |
|---|---|---|
| 1 | `SCORE_INTERACTION_FEATURE(HINGE_BASE)` | `h⁺` |
| 2 | `SCORE_INTERACTION_FEATURE(HINGE_BASE_NEGATIVE)` | `h⁻` |
| 3 | `SCORE_INTERACTION_FEATURE(ONE_SIDED_POSITIVE)` × r̃_F | `h⁺·r̃_F` |
| 4 | `SCORE_INTERACTION_FEATURE(ONE_SIDED_POSITIVE)` × r̃_P | `h⁺·r̃_P` |
| 5 | `SCORE_INTERACTION_FEATURE(ONE_SIDED_NEGATIVE)` × r̃_F | `h⁻·r̃_F` |
| 6 | `SCORE_INTERACTION_FEATURE(ONE_SIDED_NEGATIVE)` × r̃_P | `h⁻·r̃_P` |
| 7 | `SCORE_VECTORIZATION` agrupando os 6 termos + base | matriz (N, 7) |

Cada step carregava risco de inconsistência (`s₀` divergente entre
features, ordem de pares desalinhada, transformação de typicality →
rarity diferente entre passos). A escolha de arquitetura era
implícita; um revisor lendo a estratégia precisava reconstruir
mentalmente o que estava sendo materializado.

### 2.2 Instrução declarativa

A V2.14 colapsa tudo em:

```
LR_ARCHITECTURE_VECTORIZATION(
    architecture = BILATERAL_FULL_CENTERED,
    ...
)
```

E garante:

- **`s₀` único** para todos os termos.
- **`r̃_F` e `r̃_P` consistentes** — mesma centragem aplicada em todas
  as features.
- **Ordem de pares alinhada** com `pair_key_order_hash` validado.
- **Transformação `typicality → rarity` única** via
  `RarityTransformRegistry`.
- **State persistido** com fingerprint que segue para o calibrador.

### 2.3 Auditoria

O `LRArchitectureState` persistido (vide §6) é o **registro auditável**
de tudo que foi materializado: termos, ordem, valores de s₀/r̄_F/r̄_P
em hex (preservação numérica exata), fingerprint da matriz de input
bruto, identificadores dos modelos de tipicidade upstream. Em casework
forense isso é o que permite reproduzir o cálculo bit-a-bit.

---

## 3. As 15 arquiteturas

`faceaudit/enums/lr_calibration_architecture.py`

### 3.1 Família two-face (9)

Consomem tipicidade de **duas faces** (probe + reference).

| # | Nome | Termos | D | Centragem | s₀ | r_F | r_P |
|---|---|---|---|---|---|---|---|
| 1 | `COSINE_ONLY` | [s] | 1 | — | não | não | não |
| 2 | `HINGE_ONLY` | [s, h⁺] | 2 | — | sim | não | não |
| 3 | `TWO_SIDED_HINGE_ONLY` | [s, h⁺, h⁻] | 3 | — | sim | não | não |
| 4 | `ONE_SIDED_NON_CENTERED` | [s, h⁺·r_F, h⁺·r_P] | 3 | NONE | sim | sim | sim |
| 5 | `ONE_SIDED_CENTERED` | [s, h⁺·r̃_F, h⁺·r̃_P] | 3 | MEAN | sim | sim | sim |
| 6 | `LINEAR_NON_CENTERED` | [s, (s−s₀)·r_F, (s−s₀)·r_P] | 3 | NONE | sim | sim | sim |
| 7 | `LINEAR_CENTERED` | [s, (s−s₀)·r̃_F, (s−s₀)·r̃_P] | 3 | MEAN | sim | sim | sim |
| 8 | `HIERARCHICAL_CENTERED` | [s, h⁺, h⁺·r̃_F, h⁺·r̃_P] | 4 | MEAN | sim | sim | sim |
| 9 | `BILATERAL_FULL_CENTERED` | [s, h⁺, h⁻, h⁺·r̃_F, h⁺·r̃_P, h⁻·r̃_F, h⁻·r̃_P] | 7 | MEAN | sim | sim | sim |

`TWO_SIDED_HINGE_ONLY` foi **renomeada de `BILATERAL_SIMPLE`** em V2.14
(o nome antigo é traduzido por `from_str`). Reflete que ela usa "duas
hinges puramente derivadas de s" — sem raridade — e portanto não é
"bilateral" no sentido de modular ambos os lados por raridade.

### 3.2 Família ONE_FACE (6, V2.14.1)

Consomem tipicidade de **uma única face** (e.g., só a probe ou só a
reference). Úteis em casework onde apenas uma face é questionada, ou
quando a tipicidade de um dos lados é considerada ruidosa demais.

| # | Nome | Termos | D | Centragem | s₀ | r (single) |
|---|---|---|---|---|---|---|
| 10 | `ONE_SIDED_NON_CENTERED_ONE_FACE` | [s, h⁺·r] | 2 | NONE | sim | sim |
| 11 | `ONE_SIDED_CENTERED_ONE_FACE` | [s, h⁺·r̃] | 2 | MEAN | sim | sim |
| 12 | `LINEAR_NON_CENTERED_ONE_FACE` | [s, (s−s₀)·r] | 2 | NONE | sim | sim |
| 13 | `LINEAR_CENTERED_ONE_FACE` | [s, (s−s₀)·r̃] | 2 | MEAN | sim | sim |
| 14 | `HIERARCHICAL_CENTERED_ONE_FACE` | [s, h⁺, h⁺·r̃] | 3 | MEAN | sim | sim |
| 15 | `BILATERAL_FULL_CENTERED_ONE_FACE` | [s, h⁺, h⁻, h⁺·r̃, h⁻·r̃] | 5 | MEAN | sim | sim |

Em ONE_FACE, `requires_ordered_pair_roles = False` — não há "lado" a
distinguir quando apenas uma face contribui com tipicidade.

### 3.3 Notação

| Símbolo | Significado |
|---|---|
| `s` | Score base (cosseno ou PLDA) por par. |
| `s₀` | Ponto de evidência neutra (logLR = 0); resolvido em FIT/MANUAL/INHERIT. |
| `h⁺(s)` | Hinge positivo: `max(s − s₀, 0)`. |
| `h⁻(s)` | Hinge negativo: `max(s₀ − s, 0)`. Identidade: `h⁺(s) + h⁻(s) = |s − s₀|`. |
| `r_F` | Raridade do probe (rarity probe). |
| `r_P` | Raridade do reference (rarity reference). |
| `r̄_F`, `r̄_P` | Médias de `r_F`, `r_P` em D_cal (usadas para centragem). |
| `r̃_F`, `r̃_P` | Raridade centrada: `r_F − r̄_F`, `r_P − r̄_P`. |
| `r` | Raridade da face única em ONE_FACE. |
| `r̄`, `r̃` | Média e versão centrada de `r` em ONE_FACE. |

---

## 4. Os 18 termos elementares (`LRFeatureTerm`)

`faceaudit/enums/lr_feature_term.py`

Termos atômicos que compõem as 15 arquiteturas. Cada termo tem um
`LRFeatureTermSpec` registrado em `LR_FEATURE_TERM_SPECS`
(`faceaudit/models/lr/lr_feature_term_spec.py`) com:

- `build_fn(ctx: LRFeatureBuildContext) → np.ndarray (N,)` —
  materializa a coluna do termo. O `LRFeatureBuildContext` traz `base`,
  `s0`, `delta`, `hinge_positive`, `hinge_negative` e as raridades
  brutas e centradas (probe, reference, single) já pré-calculadas.
- `derivative_wrt_base_fn(ctx) → np.ndarray (N,)` — derivada
  `∂φ_j/∂s` por par.
- `handles_kink_at_s0` + `left_slope_at_s0_fn` / `right_slope_at_s0_fn`
  (opcionais) — slopes laterais no kink (`s = s₀`) para os termos com
  hinge.
- `formula_version` — entra no fingerprint da arquitetura.

### 4.1 Termos two-face (13)

| Termo | Fórmula | Usa s₀? | Usa r_F? | Usa r_P? |
|---|---|---|---|---|
| `BASE_SCORE` | `s` | não | não | não |
| `HINGE_POSITIVE` | `max(s − s₀, 0)` | sim | não | não |
| `HINGE_NEGATIVE` | `max(s₀ − s, 0)` | sim | não | não |
| `HINGE_POSITIVE_X_RARITY_PROBE_RAW` | `h⁺ · r_F` | sim | sim | não |
| `HINGE_POSITIVE_X_RARITY_REFERENCE_RAW` | `h⁺ · r_P` | sim | não | sim |
| `HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN` | `h⁺ · (r_F − r̄_F)` | sim | sim | não |
| `HINGE_POSITIVE_X_RARITY_REFERENCE_CENTERED_MEAN` | `h⁺ · (r_P − r̄_P)` | sim | não | sim |
| `HINGE_NEGATIVE_X_RARITY_PROBE_CENTERED_MEAN` | `h⁻ · (r_F − r̄_F)` | sim | sim | não |
| `HINGE_NEGATIVE_X_RARITY_REFERENCE_CENTERED_MEAN` | `h⁻ · (r_P − r̄_P)` | sim | não | sim |
| `LINEAR_X_RARITY_PROBE_RAW` | `(s − s₀) · r_F` | sim | sim | não |
| `LINEAR_X_RARITY_REFERENCE_RAW` | `(s − s₀) · r_P` | sim | não | sim |
| `LINEAR_X_RARITY_PROBE_CENTERED_MEAN` | `(s − s₀) · (r_F − r̄_F)` | sim | sim | não |
| `LINEAR_X_RARITY_REFERENCE_CENTERED_MEAN` | `(s − s₀) · (r_P − r̄_P)` | sim | não | sim |

### 4.2 Termos ONE_FACE (5, V2.14.1)

Espelham os two-face para um único `r_single`:

| Termo | Fórmula | Centragem |
|---|---|---|
| `H_POS_TIMES_SINGLE_RARITY` | `h⁺ · r` | NONE |
| `H_POS_TIMES_SINGLE_RARITY_CENTERED` | `h⁺ · (r − r̄)` | MEAN |
| `H_NEG_TIMES_SINGLE_RARITY_CENTERED` | `h⁻ · (r − r̄)` | MEAN |
| `LINEAR_TIMES_SINGLE_RARITY` | `(s − s₀) · r` | NONE |
| `LINEAR_TIMES_SINGLE_RARITY_CENTERED` | `(s − s₀) · (r − r̄)` | MEAN |

### 4.3 Derivadas analíticas

Cada `LRFeatureTermSpec` declara `derivative_wrt_base_fn(ctx)` — a
derivada `∂φ_j/∂s` é consumida pela auditoria de monotonicidade (vide
§13). Por exemplo:

- `BASE_SCORE`: `1` (constante).
- `HINGE_POSITIVE`: `1[s > s₀]` (1 acima de s₀, 0 abaixo; o indicador
  é estrito, então em `s = s₀` exato vale 0).
- `HINGE_NEGATIVE`: `−1[s < s₀]`.
- `HINGE_POSITIVE_X_RARITY_PROBE_CENTERED_MEAN`: `(r_F − r̄_F) ·
  1[s > s₀]`.

A linearidade da derivada em `β_raw · ∂φ/∂s` permite que
`LRArchitectureDerivativeProvider` (auditoria
`CONDITIONAL_MONOTONICITY`) avalie slopes analiticamente em todos os
pares observados em D_cal.

---

## 5. Os 4 modos de parâmetro (`ArchitectureParameterMode`)

`faceaudit/enums/architecture_parameter_mode.py`

### 5.1 `FIT_FULL_CALIBRATION_LEGACY` (default V1)

Ajusta `s₀` via `AUTO_LOGREG` (LogReg auxiliar univariado sobre `(s,
y)`) e `r̄_F`, `r̄_P` como média sobre **todo D_cal** **antes** da CV
interna do calibrador. Mantém compat com estratégias V2.13/V2.14.

**Leakage residual conhecido**: `s₀` é estimado sobre todo D_cal,
portanto cada fold do GridSearchCV vê `s₀` "do futuro" do próprio
fold. Métricas internas do calibrador sob este modo são otimistas;
para avaliação honesta, usar holdout externo com INHERIT.

`requires_inherit_resource = False`, `is_fit_mode = True`.

### 5.2 `FIT_PER_FOLD` (V2.14 / PR 3)

A instrução não faz o ajuste definitivo de `s₀`/r̄. Ela:

1. Emite o `LR_ARCHITECTURE_RAW_INPUT` com a matriz crua `[s, r_F,
   r_P]` + identidades dos dois lados.
2. Marca o dataset vetorizado com `defer_to_calibrator_per_fold = True`
   no metadata.
3. Para preservar o contrato do dataset vetorizado, ainda materializa
   a matriz com `s₀`/r̄ **provisórios**, estimados sobre todo D_cal pelo
   mesmo procedimento do modo legado (o state já sai com `fit_scope =
   PER_FOLD_REFIT_FULL_CALIBRATION`). O calibrador ignora essa matriz e
   usa o raw input.

O `CALIBRATION_MODEL_TRAINING` só assume o caminho por fold quando o
calibrador é LogReg **e** o metadata traz `raw_input_resource_id` — ou
seja, quando `output_raw_input_resource_id` foi preenchido. O DTO não
obriga esse campo: sem ele, o calibrador treina sobre a matriz
provisória como no modo legado, sem aviso.

O `CALIBRATION_MODEL_TRAINING` detecta a flag e monta o pipeline:

```python
Pipeline([
    ("arch_vec", ArchitectureVectorizerTransformer(architecture=..., schema=...)),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(...)),
])
```

Dentro de `GridSearchCV(cv=IdentityDisjointPairKFold(probe_identities=...,
reference_identities=...))` — sempre esse splitter neste caminho,
independentemente de `gridsearch_cv_mode` —, o `arch_vec` ajusta `s₀` (via
LogReg auxiliar fold-aware) e `r̄_F`, `r̄_P` **dentro de cada train
fold** — eliminando o leakage residual do FIT_FULL.

Após `refit=True`, `best_estimator_.named_steps["arch_vec"].state_`
reflete o refit em D_cal completo. O state final é persistido com
`fit_scope = PER_FOLD_REFIT_FULL_CALIBRATION` (vide §11).

### 5.3 `MANUAL`

Usa `manual_s0`, `manual_r0_probe`, `manual_r0_reference`
(ou `manual_r0_probe` reusado como `r0_single` em ONE_FACE com MEAN).
A obrigatoriedade dos campos depende da arquitetura:

| Arquitetura | manual_s0 | manual_r0_probe | manual_r0_reference |
|---|---|---|---|
| `COSINE_ONLY` | — | — | — |
| `HINGE_ONLY`, `TWO_SIDED_HINGE_ONLY` | obrig | — | — |
| `ONE_SIDED_NON_CENTERED`, `LINEAR_NON_CENTERED` | obrig | — | — |
| `*_CENTERED` two-face | obrig | obrig | obrig |
| ONE_FACE NON_CENTERED | obrig | — | — |
| ONE_FACE CENTERED | obrig | obrig (= r̄_single) | — |

### 5.4 `INHERIT`

**Obrigatório em VALIDATION e CASEWORK** (vide §9). Lê o
`LRArchitectureState` completo de um recurso vetorizado já produzido
(de uma execução anterior na PREPARATION). Sem recálculo. Também é
aceito em PREPARATION, sem aviso.

Validações:

- O recurso apontado por `inherit_architecture_state_from_resource_id`
  deve ter `architecture_state` no metadata.
- O `architecture` declarado na instrução deve coincidir com o do
  state herdado.
- `base_score_kind`, `pair_role_orientation` (quando a arquitetura é
  ordenada) e a proveniência da raridade devem ser compatíveis
  (`architecture_state_compat_service.validate_state_compatibility`,
  vide §12.3).

Sem `InstructionExecutionFrame` injetado, o executor levanta
`MissingExecutionFrameError` (fail-closed), em qualquer fase.

---

## 6. `LRArchitectureState` — o estado declarativo

`faceaudit/models/lr/lr_architecture_state.py`

`LRArchitectureState` é **frozen** e construído **exclusivamente** via
`LRArchitectureStateFactory.create(...)`. Carrega:

### 6.1 Parâmetros ajustados

- `s0: Optional[float]` — ponto de evidência neutra.
- `r0_probe`, `r0_reference: Optional[float]` — médias de raridade
  (centragem MEAN).
- `r0_single: Optional[float]` — para arquiteturas ONE_FACE V2.14.1.

Entram nos fingerprints em hex via `hexify_float` (= `float.hex`)
para preservação bit-a-bit (`hexify_float(0.4566)` →
`"0x1.d38ef34d6a162p-2"`).

### 6.2 Proveniência

- `architecture: LRCalibrationArchitecture`.
- `formula_version`, `architecture_spec_fingerprint` — versão e
  fingerprint do `LRArchitectureSpec`.
- `feature_terms: Tuple[LRFeatureTerm, ...]`, `feature_names`,
  `output_dim` — duplicados do spec para validação de integridade.
- `centering_mode: RarityCenteringMode` — `NONE` ou `MEAN`.
- `base_score_kind: str` — lido de `score_kind` no metadata do base;
  o `SIMILARITY_SCORING` grava `"cosine"` ou `"plda"` (default
  `"cosine"` quando ausente).
- `pair_role_orientation: PairRoleOrientation` — semântica do par.
- `pair_key_mode: PairKeyMode` — tratamento da chave do par
  (`ORDERED`/`UNORDERED`).
- `rarity_probe_score_type`, `rarity_reference_score_type`,
  `rarity_single_score_type`: strings (valor do score type da
  tipicidade) propagadas da raridade upstream.
- `rarity_*_combiner`, `rarity_*_transform_formula_version`,
  `rarity_*_transform_fingerprint`, `rarity_*_model_resource_id`.
- `rarity_probe_transform_id`, `rarity_reference_transform_id`,
  `rarity_single_transform_id`: IDs estáveis do
  `RarityTransformRegistry`.
- `rarity_probe_model_fingerprint`,
  `rarity_reference_model_fingerprint`,
  `rarity_single_model_fingerprint`: `model_contract_fingerprint` do
  modelo de tipicidade upstream (PR 3 do plano de tipicidade).
- `fit_scope: ArchitectureFitScope` — `FULL_CALIBRATION_LEGACY`,
  `PER_FOLD_REFIT_FULL_CALIBRATION`, `MANUAL` ou `NOT_APPLICABLE`.
- `s0_resolution_method: S0ResolutionMethod` —
  `AUTO_LOGREG`/`MANUAL`/`NOT_APPLICABLE`. INHERIT não é um método:
  o state herdado mantém o `fit_scope` e o `s0_resolution_method` do
  ajuste original.

### 6.3 Fingerprints (2 camadas)

Calculados em `LRArchitectureStateFactory.create` e validados em
`from_dict` (StateIntegrityError em mismatch — tampering proof).

**`vectorization_contract_fingerprint`** — cobre tudo que altera a
matriz de features, semântica e proveniência dos scores. Hash de:

```
architecture, formula_version, architecture_spec_fingerprint,
feature_terms, centering_mode,
hex(s0), hex(r0_probe), hex(r0_reference), hex(r0_single),
base_score_kind, pair_role_orientation,
rarity_*_score_type, rarity_*_transform_id, rarity_*_transform_formula_version,
rarity_*_transform_fingerprint, rarity_*_combiner,
rarity_*_model_fingerprint
```

**Este é o fingerprint que entra no calibrador downstream** e que
`CalibrationModelMetadataService` compara em STRICT (vide §10).

**`provenance_record_fingerprint`** — auditoria-apenas. Hash de:

```
fit_scope, s0_resolution_method, r0_*_estimator,
fitted_on_resource_id, fitted_on_resource_fingerprint,
n_fitted_pairs, n_fitted_genuine_pairs, n_fitted_impostor_pairs,
fitted_dataset_pair_key_order_hash,
rarity_*_model_resource_id,
s0_logreg_config, s0_fit_summary
```

Não comparado na aplicação — apenas para rastreamento forense.

### 6.4 Enums associados

- `ArchitectureFitScope`: `NOT_APPLICABLE` / `FULL_CALIBRATION_LEGACY`
  / `PER_FOLD_REFIT_FULL_CALIBRATION` / `MANUAL`. O enum tem
  `has_leakage_risk` (True em `FULL_CALIBRATION_LEGACY`); o state tem
  `has_internal_cv_parameter_leakage_risk` (True em
  `FULL_CALIBRATION_LEGACY` quando a arquitetura usa algum parâmetro
  ajustado).
- `S0ResolutionMethod`: `NOT_APPLICABLE` / `AUTO_LOGREG` / `MANUAL`.
- `RarityCenteringMode`: `NONE` / `MEAN`.
- `PairRoleOrientation`: `PROBE_REFERENCE` / `REFERENCE_PROBE` /
  `NOT_APPLICABLE` (arquiteturas simétricas).
- `PairKeyMode`: `ORDERED` / `UNORDERED`. Distinto de
  `PairRoleOrientation`.

---

## 7. `LRArchitectureRawInputPayload` — o payload canônico

`faceaudit/models/lr/lr_architecture_raw_input_payload.py`

DTO frozen construído **exclusivamente** via
`LRArchitectureRawInputPayloadFactory.create(...)`. Carrega a matriz
bruta `[base_score, rarity_probe, rarity_reference]` (ou `[base_score,
rarity_single]` em ONE_FACE V2.14.1) com canonicalização rigorosa.

### 7.1 Estrutura

```python
@dataclass(frozen=True)
class LRArchitectureRawInputPayload:
    matrix: np.ndarray                        # (N, K), <f8 C-contiguous, write=False
    schema: LRArchitectureRawFeatureSchema    # versionada; colunas + tipos
    pair_keys: tuple[tuple[str, str], ...]    # (probe_hash, reference_hash) por par
    same_person: Optional[np.ndarray]         # bool (N,) ou None em casework
    probe_identity_ids: Optional[tuple[str, ...]]
    reference_identity_ids: Optional[tuple[str, ...]]
    pair_role_orientation: PairRoleOrientation
    pair_key_mode: PairKeyMode
    pair_key_order_hash: str                  # SHA-256 dos pair_keys ordenados
    pair_label_hash: Optional[str]            # SHA-256 dos same_person (None em casework)
    raw_input_matrix_fingerprint: str         # SHA-256 do contrato completo
```

### 7.2 Canonicalização

A factory garante:

- `matrix` é convertido para `<f8` (little-endian float64) C-contíguo
  via `np.ascontiguousarray(...).copy()`.
- `matrix.setflags(write=False)` — imutabilidade efetiva.
- `same_person` (quando não-None) é `np.ascontiguousarray(...,
  dtype=np.bool_).copy() + setflags(write=False)`.
- `pair_keys` é `tuple` (não list) — imutável.

### 7.3 `pair_key_order_hash` vs `pair_label_hash`

Separados por design (PR 2 V2.14):

- **`pair_key_order_hash`** = `SHA-256(n_pairs + concat(probe_hash +
  "\x00" + reference_hash + "\n"))`. Capta a estrutura do par
  (`(probe_a, reference_b)` vs `(probe_b, reference_a)` são distintos).
  Necessário para alinhamento.
- **`pair_label_hash`** = `SHA-256(n_pairs + concat("1"/"0" bytes))`.
  Captura os rótulos. **`None` quando `same_person is None`**
  (casework sem ground truth).

Essa separação permite que casework não exija rótulo sem perder a
auditoria do alinhamento.

### 7.4 Identidades dos dois lados

`probe_identity_ids` e `reference_identity_ids` viajam separados —
necessários para `IdentityDisjointPairKFold` no FIT_PER_FOLD (vide
§11). A factory aceita None nos dois em CASEWORK; exige presença em
PREPARATION + VALIDATION.

### 7.5 Schema versionado

`LRArchitectureRawFeatureSchema` declara:

- `schema_version: str` — versão do contrato (hoje `"1.0"`).
- `columns: tuple[str, ...]` — ordem das colunas.
- `base_score_column: int` — índice da coluna do score base.
- `rarity_probe_column`, `rarity_reference_column: Optional[int]` —
  índices das colunas de raridade two-face.
- `rarity_single_column: Optional[int]` — índice da coluna de
  raridade ONE_FACE (V2.14.1). **Mutuamente exclusivo** com
  `rarity_probe_column` + `rarity_reference_column`.
- `fingerprint(): str` — hash do contrato.

Helpers: `schema_for_base_only()` (1 coluna, arquiteturas sem
raridade), `schema_for_base_and_rarities()` (3 colunas two-face) e
`schema_for_base_and_single_rarity()` (2 colunas `[base_score,
rarity_single]` para ONE_FACE).

---

## 8. `ArchitectureFeatureBuilder` — o builder genérico

`faceaudit/services/architecture_feature_builder.py`

Builder **puro**, exposto como funções de módulo (não há classe
`ArchitectureFeatureBuilder`; o nome designa o módulo), sem `if-elif`
por arquitetura. Consulta os
`LRFeatureTermSpec` e materializa a matriz aplicando os termos
declarados no `LRArchitectureSpec`.

### 8.1 API

```python
def build(
    spec: LRArchitectureSpec,
    state: LRArchitectureState,
    base: np.ndarray,
    rarity_probe: Optional[np.ndarray] = None,
    rarity_reference: Optional[np.ndarray] = None,
    rarity_single: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Materializa a matriz (N, D) consumida pelo calibrador."""

def derivative_matrix_wrt_base(
    spec: LRArchitectureSpec,
    state: LRArchitectureState,
    base: np.ndarray,
    rarity_probe: Optional[np.ndarray] = None,
    rarity_reference: Optional[np.ndarray] = None,
    rarity_single: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Matriz (N, D) com ∂φ_j/∂s para auditoria CONDITIONAL_MONOTONICITY."""
```

### 8.2 Aplicação

Para cada `feature_term ∈ spec.feature_terms`:

1. Monta uma vez o `LRFeatureBuildContext` (delta, hinges, raridades
   brutas e centradas) a partir de `state`, `base` e raridades.
2. Consulta `LR_FEATURE_TERM_SPECS[feature_term]` e chama
   `term_spec.build_fn(ctx)` → coluna `(N,)`.
3. Empilha colunas em uma matriz `(N, D)` e confere o shape contra
   `spec.output_dim`.

Result: zero if-elif por arquitetura no executor. Toda a lógica da
matemática vive nos `LRFeatureTermSpec`.

### 8.3 Slopes em `s = s_0`

`derivative_matrix_wrt_base` usa as derivadas pontuais dos termos, com
indicadores estritos (`1[s > s₀]`, `−1[s < s₀]`): em `s = s₀` exato
as duas hinges dão 0. O `LRArchitectureDerivativeProvider` da
auditoria não consulta os slopes laterais — ele separa os pares nas
regiões `s_gt_s0` (`s > s₀`) e `s_le_s0` (`s ≤ s₀`) e avalia cada
uma. Os `left_slope_at_s0_fn` / `right_slope_at_s0_fn` ficam
disponíveis no `LRFeatureTermSpec` para inspeção direta do kink.

---

## 9. O fluxo do executor passo a passo

`faceaudit/executors/instructions/lr_architecture_vectorization_instruction_executor.py`

Etapas (estado normal):

1. **Validação estrutural** (`_validate_instruction_specifics`):
   - Instrução é do tipo correto.
   - `context.frame` presente; ausente → `MissingExecutionFrameError`
     (fail-closed, em qualquer fase).
   - `architecture` registrada em `LR_ARCHITECTURE_SPECS`.
   - Em VALIDATION/CASEWORK: `parameter_mode = INHERIT` obrigatório.
   - `base_score_resource_id` e `output_resource_id` definidos.
   - Resources de raridade exigidos pela arquitetura definidos
     (consulta `spec.requires_typicality_probe`, etc.).
   - Em ONE_FACE: `rarity_resource_id` definido **e**
     `rarity_probe_resource_id` / `rarity_reference_resource_id`
     ausentes (mutuamente exclusivos); fora de ONE_FACE,
     `rarity_resource_id` é rejeitado.
   - MANUAL: campos `manual_*` exigidos pela arquitetura; INHERIT:
     `inherit_architecture_state_from_resource_id` definido.

2. **`InstructionExecutionFrame` no contexto**:
   - O `StrategyExecutor` injeta `InstructionExecutionFrame(phase,
     instruction_id, definition_id)` no `StrategyExecutionContext`
     antes do `execute()`. O executor consulta `context.frame` para
     derivar `StrategyPhase`.

3. **Hidrata recursos**: `context.ensure_payload(...)` no base e nas
   raridades; recurso exigido ausente → falha.

4. **Alinha os inputs** (`_align_inputs`):
   - Lê `raw_score` de cada item do base, na ordem do base.
   - Busca a raridade de cada par por `(probe_hash, reference_hash)`
     orientado; par ausente → erro.
   - Extrai identidades dos dois lados e `same_person` (None se algum
     item não tiver rótulo).

5. **Resolve o state** (s₀, r̄_F, r̄_P ou r̄_single) conforme
   `parameter_mode`, via `LRArchitectureStateFactory.create(...)`
   (que calcula `vectorization_contract_fingerprint` e
   `provenance_record_fingerprint`):
   - `FIT_FULL_CALIBRATION_LEGACY`: chama
     `s0_resolution_service.resolve_s0_auto_logreg` sobre todo D_cal
     (exige `same_person`) + médias das raridades quando
     `centering_mode = MEAN`.
   - `FIT_PER_FOLD`: mesmo cálculo, com valores **provisórios** e
     `fit_scope = PER_FOLD_REFIT_FULL_CALIBRATION`; marca
     `defer_to_calibrator_per_fold=True` no dataset vetorizado. O
     calibrador reestima por fold (vide §5.2).
   - `MANUAL`: usa `manual_s0`, `manual_r0_*` (`fit_scope = MANUAL`).
   - `INHERIT`: `LRArchitectureStateFactory.from_dict` sobre o
     `architecture_state` do recurso herdado +
     `architecture_state_compat_service.validate_state_compatibility`.

6. **Materializa matriz** via
   `architecture_feature_builder.build(spec, state, base, r_F, r_P,
   r_single)`.

7. **Constrói `LRArchitectureRawInputPayload`** via
   `LRArchitectureRawInputPayloadFactory.create(..., phase=phase)`:
   empilha `[base, r_F, r_P]` (ou `[base, r_single]`), canonicaliza
   para `<f8` C-contíguo read-only, computa `pair_key_order_hash`,
   `pair_label_hash` e `raw_input_matrix_fingerprint`, e exige
   `same_person` + identidades fora de CASEWORK.

8. **Emite os outputs**:
   - `SimilarityScoreDataset vetorizado`: mesmos items do base
     (`raw_score` escalar preservado) + metadata com
     `score_vector_matrix`, `architecture_state`,
     `vectorization_contract_fingerprint`,
     `score_vector_numeric_fingerprint`,
     `score_vector_contract_fingerprint`,
     `raw_input_matrix_fingerprint`, `pair_key_order_hash`,
     `raw_input_resource_id` e `defer_to_calibrator_per_fold` se
     FIT_PER_FOLD.
   - `LR_ARCHITECTURE_RAW_INPUT`, só quando
     `output_raw_input_resource_id` está preenchido (necessário em
     FIT_PER_FOLD).

9. **Auditoria runtime** no `CALIBRATION_MODEL_TRAINING` downstream:
   - `_bind_training_raw_input` localiza o raw_input cujo
     `raw_input_matrix_fingerprint` bate com o do calibrador (com
     fallbacks; vide §13).
   - `_run_monotonicity_audit` instancia
     `ConditionalMonotonicityRule` e roda sobre o calibrador
     recém-treinado. Vide §13.

---

## 10. Fingerprints e compatibilidade com o calibrador

`faceaudit/services/calibration_model_metadata_service.py`

### 10.1 `bind_architecture_state`

```python
CalibrationModelMetadataService.bind_architecture_state(
    model_metadata, dataset_metadata
)
```

Chamado pelo executor de `CALIBRATION_MODEL_TRAINING` após o treino
(em todos os caminhos, inclusive FIT_PER_FOLD). No-op se o dataset
não tem `vectorization_contract_fingerprint`. Copia para o
`calibrator.metadata`:

- `architecture_state` (dict completo do `LRArchitectureState` do
  dataset vetorizado).
- `vectorization_contract_fingerprint`, `lr_architecture`,
  `lr_architecture_formula_version`, `architecture_spec_fingerprint`.
- `training_score_vector_contract_fingerprint`,
  `training_score_vector_numeric_fingerprint`,
  `training_raw_input_matrix_fingerprint`,
  `training_pair_key_order_hash` — auditoria-apenas, não comparados
  na aplicação.

### 10.2 `validate_compatibility`

```python
CalibrationModelMetadataService.validate_compatibility(
    model_metadata, dataset_metadata,
    *, compatibility_mode: CalibrationCompatibilityMode = STRICT,
) → List[CompatibilityIssue]
```

Chamado pelo executor de `SCORE_CALIBRATION`; issue `error` bloqueia
a calibração, `warning` vai para o log. Tabela:

| Dataset | Calibrador | Resultado |
|---|---|---|
| sem fingerprint | sem fingerprint | OK com warning informativo (ambos legados) |
| com fingerprint | com fingerprint, igual | OK |
| com fingerprint | com fingerprint, diferente | **erro** em qualquer modo |
| com fingerprint | sem fingerprint | erro em `STRICT`; warning em `LEGACY_PERMISSIVE` |
| sem fingerprint | com fingerprint | **erro** em qualquer modo |

`CalibrationCompatibilityMode`:

- **`STRICT`** (default).
- **`LEGACY_PERMISSIVE`**: só relaxa o caso "dataset novo +
  calibrador legado". **Nota:** o `SCORE_CALIBRATION` chama o serviço
  sem passar o modo e não tem campo `compatibility_mode` — na prática
  a validação é sempre `STRICT`.

### 10.3 INHERIT downstream

Quando uma instrução em VALIDATION/CASEWORK usa `parameter_mode =
INHERIT`, o executor:

1. Lê `architecture_state` do recurso herdado.
2. Reconstrói o state via
   `LRArchitectureStateFactory.from_dict` (recalcula fingerprints —
   `StateIntegrityError` em qualquer mismatch detectando tampering).
3. Valida compatibilidade contra o contexto atual via
   `architecture_state_compat_service.validate_state_compatibility`
   (verifica architecture, spec, base_score_kind,
   pair_role_orientation, centragem, modelos de tipicidade por
   fingerprint, transformações de raridade — vide §12.3).

Issues `error` abortam. O serviço admite severidade `warning`, mas
hoje todas as checagens emitem `error`.

---

## 11. FIT_PER_FOLD com `ArchitectureVectorizerTransformer`

`faceaudit/services/architecture_vectorizer_transformer.py`

### 11.1 Pipeline canônico (V2.14 / PR 3)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("arch_vec", ArchitectureVectorizerTransformer(
        architecture=arch,                       # de metadata["lr_architecture"]
        schema=raw_payload.schema,
        s0_logreg_config=state_dict.get("s0_logreg_config"),  # override persistido no state
        base_score_kind=..., pair_role_orientation=...,
        rarity_probe_score_type=..., rarity_reference_score_type=...,
        rarity_single_score_type=...,
    )),
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(solver=..., class_weight=..., max_iter=...)),
])
```

Dentro de `GridSearchCV` (código em
`CalibrationModelTrainingInstructionExecutor._train_logreg_per_fold_with_transformer`):

```python
gs = GridSearchCV(
    estimator=pipeline,
    param_grid={"logreg__C": list(c_grid)},
    cv=IdentityDisjointPairKFold(
        probe_identities=list(raw_payload.probe_identity_ids),
        reference_identities=list(raw_payload.reference_identity_ids),
        n_splits=cv_folds,          # logreg_config.cv_folds (mínimo 2)
        random_state=42,
    ),
    scoring=scoring_obj,            # logreg_config.scoring; default NEG_CLLR_BALANCED
    refit=True,
    n_jobs=1,
)
gs.fit(raw_payload.matrix, same_person)
```

### 11.2 Como elimina o leakage residual

A cada chamada `Pipeline.fit(X_train, y_train)` (uma vez por fold do
`GridSearchCV`):

1. `arch_vec.fit(X_train, y_train)` resolve **`s₀` apenas com
   X_train** (LogReg auxiliar fold-aware) + **`r̄_F`, `r̄_P` apenas
   sobre X_train** (médias).
2. `arch_vec.transform(X_train)` materializa as features com esses
   s₀/r̄ "do fold".
3. `scaler.fit(X_train_transformed)` aprende μ/σ apenas sobre
   X_train.
4. `logreg.fit(X_train_scaled, y_train)` treina o calibrador.

Em `arch_vec.transform(X_val)` (do fold), aplica os s₀/r̄ do fold
train aos pares de validação — sem leakage.

Após `refit=True`, o `GridSearchCV` chama `pipeline.fit(X_all,
y_all)` com o C escolhido — o `arch_vec.state_` final reflete refit
em D_cal completo. Esse state é gravado no metadata do calibrador com
`fit_scope = PER_FOLD_REFIT_FULL_CALIBRATION`, junto de
`fit_per_fold = True`, `inner_cv_splitter`, `inner_cv_fold_count`,
`final_refit_on_full_calibration = True`, `selected_C` e os
coeficientes em escala bruta (`raw_scale_coefficients`,
`raw_scale_intercept`). Em seguida o `bind_architecture_state` (§10.1)
copia para o metadata o `architecture_state` e o
`vectorization_contract_fingerprint` do dataset vetorizado.

### 11.3 `IdentityDisjointPairKFold` recebe identidades dos dois lados

`faceaudit/services/identity_disjoint_pair_kfold.py`

O splitter exige `probe_identities` **e** `reference_identities`
(separados). Em cada fold:

- **train**: pares com ambas as identidades fora do grupo do fold.
- **test**: pares com ambas as identidades dentro do grupo do fold.
- **boundary**: pares com uma identidade dentro e outra fora —
  **excluídos apenas naquele fold** (não globalmente).

Resultado: train fold preserva ≈((K−1)/K)² do `H_d` original; teste
não contém pares de identidades que apareceram no train. Sem
leakage cross-fold.

No caminho FIT_PER_FOLD o splitter é montado diretamente pelo executor,
sem `lookalike_clusters`. O suporte a clusters de sósias (V2.11.2 —
`metadata["lookalike_clusters"]` propagado como super-ids) existe no
caminho comum do `LogisticRegressionCalibrationModel` com
`gridsearch_cv_mode = STRICT_IDENTITY`. Vide
`docs/piecewise_linear_calibration.md` §4.4.

---

## 12. Validações e guard-rails

### 12.1 Defaults persistentes (`validation_rule_configs`)

`_ensure_default_validation_rules` popula:

| Cenário | Regra | Notas |
|---|---|---|
| `parameter_mode = FIT_FULL_CALIBRATION_LEGACY` + `base_score_resource_id` definido | `MIN_CALIBRATION_PAIRS` | Garante pares mínimos para s₀ via AUTO_LOGREG. |

Demais validações dependem do contexto e ocorrem no executor.

### 12.2 Validações no executor

- `instruction.architecture` registrado em `LR_ARCHITECTURE_SPECS`.
- Recursos de raridade exigidos pela arquitetura estão definidos.
- ONE_FACE rejeita `rarity_probe/reference_resource_id`
  simultaneamente; two-face rejeita `rarity_resource_id`.
- `parameter_mode = INHERIT` em PREPARATION → aceito sem aviso; em
  VALIDATION/CASEWORK → obrigatório.
- `parameter_mode = MANUAL` exige `manual_s0` e `manual_r0_*`
  conforme arquitetura.
- `architecture` que exige `requires_ordered_pair_roles` precisa que
  o metadata do base score tenha `pair_role_orientation` (ou
  `pair_orientation`) explícito; ausência → erro. O executor aceita
  qualquer valor explícito reconhecido pelo enum.
- Em PREPARATION e VALIDATION, `same_person` e as identidades dos dois
  lados são obrigatórios (`LRArchitectureRawInputPayloadFactory`);
  FIT_FULL_CALIBRATION_LEGACY/FIT_PER_FOLD também exigem `same_person`
  para o AUTO_LOGREG.
- Em CASEWORK, `same_person` e identidades podem ser None.

### 12.3 `_validate_state_compatibility` em INHERIT

`architecture_state_compat_service.validate_state_compatibility`
checa (tabela de 20+ cenários em
`tests/test_architecture_state_compat_service.py`):

- `architecture`, `formula_version`, `architecture_spec_fingerprint`,
  `feature_terms`, `output_dim` e `centering_mode` idênticos ao spec.
- `base_score_kind` idêntico.
- `pair_role_orientation` idêntico, só em arquiteturas ordenadas.
- Por lado de raridade exigido (probe, reference ou single): metadata
  presente e `score_type`, `transform_id`,
  `transform_formula_version`, `transform_fingerprint`, `combiner` e
  `model_fingerprint` idênticos.

`pair_key_mode` não é verificado. Todas as divergências são `error` e
abortam o INHERIT.

### 12.4 `MissingExecutionFrameError`

Executor levanta `MissingExecutionFrameError` quando `context.frame` é
`None`, em qualquer fase — sem frame não há como derivar a fase.

Fail-closed: previne execução acidental em fase errada.

---

## 13. Auditoria de monotonicidade no calibrador

`faceaudit/validation_rules/conditional_monotonicity_rule.py`

Após o refit final do calibrador upstream, o
`CALIBRATION_MODEL_TRAINING` executor:

1. Localiza o `LR_ARCHITECTURE_RAW_INPUT` cujo
   `raw_input_matrix_fingerprint` bate com o do treino (fallbacks, nesta
   ordem: `vectorization_contract_fingerprint`, `schema.fingerprint()`,
   único `LR_ARCHITECTURE_RAW_INPUT` no contexto) e grava o id em
   `calibrator.metadata["training_raw_input_resource_id"]`.
2. Instancia `ConditionalMonotonicityRule` via factory.
3. Roda `validate()` sobre o calibrador.

A regra usa `LRArchitectureDerivativeProvider` para avaliar
analiticamente `∂log_LR/∂s = β_raw · ∂φ/∂s` em:

- `observed_rows`: todos os pares observados em D_cal.
- `central_box`: cantos do retângulo `[P5, P95]²` das raridades
  observadas.
- `reportable_box` (V2): domínio operacional explicitamente declarado
  pela política de reportabilidade.

`audit_status`:

- **OK**: slope > 0 em todas as regiões.
- **WARNING**: slope ≤ 0 apenas no `central_box` (P5-P95), ou slope
  mínimo observado abaixo de `operational_min_slope` (quando
  configurado).
- **CRITICAL**: slope ≤ 0 em `observed_rows` ou `reportable_box`.
- **NOT_AUDITED**: auditoria não pôde rodar (ex.: nenhum raw input
  encontrado, sem `training_raw_input_resource_id` no calibrador).
  Calibrador sem `architecture_state` (KDE, BiGauss, LogReg legado)
  nem chega a ser auditado.

Persistido em `calibrator.metadata["monotonicity_audit"]` para
auditoria. Vide `docs/conditional_monotonicity_rule_design.md` para o
roadmap V1/V1.5/V2 + projeto paralelo `TYPICALITY_EFFECT_AUDIT`.

---

## 14. Configurações

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `architecture` | `LRCalibrationArchitecture` | `COSINE_ONLY` | Arquitetura escolhida. |
| `parameter_mode` | `ArchitectureParameterMode` | `FIT_FULL_CALIBRATION_LEGACY` | Como obter s₀/r̄. |
| `base_score_resource_id` | `Optional[str]` | `None` | Score base (cosseno/PLDA). |
| `rarity_probe_resource_id` | `Optional[str]` | `None` | Raridade do probe (two-face). |
| `rarity_reference_resource_id` | `Optional[str]` | `None` | Raridade do reference (two-face). |
| `rarity_resource_id` | `Optional[str]` | `None` | Raridade da face única (ONE_FACE). |
| `inherit_architecture_state_from_resource_id` | `Optional[str]` | `None` | Recurso de origem do state (INHERIT). |
| `manual_s0` | `Optional[float]` | `None` | s₀ manual. |
| `manual_r0_probe` | `Optional[float]` | `None` | r̄_F manual (ou r̄_single em ONE_FACE+MEAN). |
| `manual_r0_reference` | `Optional[float]` | `None` | r̄_P manual. |
| `s0_logreg_config_override` | `Optional[Dict]` | `None` | Override do LogReg auxiliar. |
| `output_resource_id` | `Optional[str]` | `None` | SIMILARITY_SCORES vetorizado de saída. |
| `output_raw_input_resource_id` | `Optional[str]` | `None` | LR_ARCHITECTURE_RAW_INPUT (necessário em FIT_PER_FOLD; não validado pelo DTO). |

### 14.1 Visibilidade dinâmica

`get_hidden_fields()` retorna o conjunto de campos a esconder. UI
rebuilda o form quando `architecture` ou `parameter_mode` muda
(`REACTIVE_FIELDS = ("architecture", "parameter_mode")`).
`s0_logreg_config_override` só aparece em
`FIT_FULL_CALIBRATION_LEGACY` com arquitetura que usa s₀, embora o
valor também seja lido em FIT_PER_FOLD quando definido no JSON. O
seletor de `inherit_architecture_state_from_resource_id` lista apenas
recursos `SIMILARITY_SCORES`.

---

## 15. Casos de uso comuns

### 15.1 PREPARATION — BILATERAL cheia com FIT_PER_FOLD

```python
LR_ARCHITECTURE_VECTORIZATION(
    architecture = BILATERAL_FULL_CENTERED,
    parameter_mode = FIT_PER_FOLD,
    base_score_resource_id = "cosseno_D_cal",
    rarity_probe_resource_id = "typicality_probe_D_cal",
    rarity_reference_resource_id = "typicality_reference_D_cal",
    output_resource_id = "vec_BILATERAL_cheia_D_cal",
    output_raw_input_resource_id = "raw_input_BILATERAL_cheia",
)
```

`CALIBRATION_MODEL_TRAINING` downstream detecta
`defer_to_calibrator_per_fold = True` (e o `raw_input_resource_id`)
e monta o pipeline `Pipeline + GridSearchCV + IdentityDisjointPairKFold`
com `arch_vec.fit` por fold.

### 15.2 VALIDATION — INHERIT do state da preparação

```python
LR_ARCHITECTURE_VECTORIZATION(
    architecture = BILATERAL_FULL_CENTERED,
    parameter_mode = INHERIT,
    base_score_resource_id = "cosseno_D_val",
    rarity_probe_resource_id = "typicality_probe_D_val",
    rarity_reference_resource_id = "typicality_reference_D_val",
    inherit_architecture_state_from_resource_id = "vec_BILATERAL_cheia_D_cal",
    output_resource_id = "vec_BILATERAL_cheia_D_val",
)
```

Reusa `s₀`, `r̄_F`, `r̄_P` aprendidos em D_cal sem recálculo.
Preserva a "definição de evidência neutra" aprendida na calibração.

### 15.3 CASEWORK — ONE_FACE com modelo lateral apenas

Em cross-pose forense onde só a face lateral foi questionada e a
tipicidade da frontal é considerada bem-estabelecida (não é fonte
de evidência):

```python
LR_ARCHITECTURE_VECTORIZATION(
    architecture = HIERARCHICAL_CENTERED_ONE_FACE,
    parameter_mode = INHERIT,
    base_score_resource_id = "cosseno_caso",
    rarity_resource_id = "typicality_lateral_caso",  # ONE_FACE
    inherit_architecture_state_from_resource_id = "vec_HIER_ONE_FACE_D_cal",
    output_resource_id = "vec_HIER_ONE_FACE_caso",
)
```

---

## 16. Boas práticas

1. **Prefira FIT_PER_FOLD em PREPARATION.** Elimina o leakage
   residual do FIT_FULL_CALIBRATION_LEGACY. Estratégias novas em V2.14
   devem usar este modo.
2. **Em VALIDATION/CASEWORK sempre INHERIT.** Recalcular s₀ em
   validação contamina a definição de "evidência neutra".
3. **Mantenha o mesmo modelo de tipicidade upstream em PREPARATION
   e VALIDATION/CASEWORK.** O `rarity_*_model_fingerprint` entra no
   `vectorization_contract_fingerprint`; mudanças invalidam o INHERIT.
4. **Não misture vetor novo com calibrador legado.** O
   `SCORE_CALIBRATION` valida em `STRICT` e rejeita essa combinação;
   `LEGACY_PERMISSIVE` existe no serviço, mas não está exposto na
   instrução.
5. **Registre `vectorization_contract_fingerprint` no laudo
   forense.** Permite re-validação independente da pipeline.
6. **Não troque de arquitetura em meio à validação.** A escolha de
   arquitetura deve ser fixada em PREPARATION e mantida ao longo de
   toda a cadeia.
7. **Para casework com tipicidade ruidosa em um lado, considere
   ONE_FACE.** Por exemplo, cross-pose onde a tipicidade do lado
   lateral tem suporte amostral baixo.

---

## 17. Perguntas frequentes

### 17.1 Posso usar FIT_FULL_CALIBRATION_LEGACY hoje?

Sim, é o default V1 e preserva compat com estratégias V2.13. Mas
ciente do leakage residual: `s₀` estimado sobre todo D_cal antes da
CV interna do calibrador faz com que cada fold veja `s₀` "do
futuro". Avaliação honesta exige holdout externo com INHERIT.

### 17.2 INHERIT recusou meu state. O que mudou?

O `validate_state_compatibility` encontrou divergência entre o state
herdado e o contexto atual (a mensagem de erro nomeia o campo).
Causas comuns:

- Trocou modelo de tipicidade upstream (mudou `unit_norm_policy`,
  `population_weighting_mode`, etc. → mudou
  `model_contract_fingerprint` → mudou `rarity_*_model_fingerprint`).
- Trocou `rarity_*_transform_id` (não esperado se o
  `typicality_score_type` continua o mesmo).
- Trocou o score base (cosseno ↔ PLDA muda `base_score_kind`) ou a
  orientação dos pares numa arquitetura ordenada.
- Re-treinou com `parameter_mode` diferente
  (`FIT_FULL_CALIBRATION_LEGACY` → `FIT_PER_FOLD` muda
  `fit_scope` no state, mas isso entra no
  `provenance_record_fingerprint`, não no
  `vectorization_contract_fingerprint`) — isso, sozinho, não causa
  recusa.

Não há modo permissivo para o INHERIT: corrija a divergência ou
refaça a preparação.

### 17.3 Posso comparar `BILATERAL_FULL_CENTERED` com
`BILATERAL_FULL_CENTERED_ONE_FACE` na mesma estratégia?

Não diretamente. Cada arquitetura produz `vectorization_contract_fingerprint`
distinto, e o calibrador downstream é específico para uma arquitetura.
Para comparação científica, gere duas estratégias paralelas com
arquiteturas diferentes e calibradores próprios. Compare Cllr/EER
finais.

### 17.4 Preciso do `LR_ARCHITECTURE_RAW_INPUT` em FIT_FULL?

Para o cálculo, não: é opcional em FIT_FULL/MANUAL e necessário
apenas em FIT_PER_FOLD. Mas a auditoria `CONDITIONAL_MONOTONICITY`
precisa localizar um raw input no contexto; sem ele o calibrador sai
com `audit_status = NOT_AUDITED`.

### 17.5 Pode usar PLDA como score base?

Sim. `base_score_kind` é propagado do `score_kind` do recurso
`SIMILARITY_SCORES` — o `SIMILARITY_SCORING` grava `cosine` ou `plda`
(sem distinguir a variante PLDA).
**Atenção**: combinar PLDA + interação cosseno × raridade é
double-counting potencial (B-covariance do PLDA já encoda tipicidade
da população). Compare, na janela de Comparação, uma estratégia com
PLDA puro e outra com PLDA + interação antes de adotar.

---

## 18. Glossário

- **`architecture`**. Uma das 15 entradas de `LRCalibrationArchitecture`.
- **`base_score`**. Score de similaridade por par (cosseno ou PLDA).
- **`build`**. Materialização da matriz de features via
  `architecture_feature_builder.build(spec, state, base, r_F, r_P,
  r_single)`.
- **`defer_to_calibrator_per_fold`**. Flag no metadata do dataset
  vetorizado quando `parameter_mode = FIT_PER_FOLD`. Calibrador
  downstream detecta e monta o pipeline com `arch_vec`.
- **`feature_term`**. Termo elementar `LRFeatureTerm`; 18 disponíveis.
- **`fit_scope`**. Como o state foi originalmente ajustado:
  `FULL_CALIBRATION_LEGACY`, `PER_FOLD_REFIT_FULL_CALIBRATION`,
  `MANUAL` ou `NOT_APPLICABLE`.
- **`h⁺`, `h⁻`**. Hinges positivo e negativo: `max(s − s₀, 0)` e
  `max(s₀ − s, 0)`. Identidade: `h⁺ + h⁻ = |s − s₀|`.
- **INHERIT**. `parameter_mode` que lê state de execução anterior.
- **`MissingExecutionFrameError`**. Erro quando executor é chamado sem
  frame injetado (qualquer fase).
- **`pair_key_order_hash`**. SHA-256 dos pair_keys na ordem dos pares.
- **`parameter_mode`**. Como obter s₀/r̄: FIT_FULL_CALIBRATION_LEGACY
  / FIT_PER_FOLD / MANUAL / INHERIT.
- **`r̄_F`, `r̄_P`, `r̄`**. Médias de raridade usadas para centragem MEAN.
- **`r̃_F`, `r̃_P`, `r̃`**. Raridade centrada: `r − r̄`.
- **`s₀`**. Ponto de evidência neutra (logLR = 0).
- **`vectorization_contract_fingerprint`**. SHA-256 do contrato
  semântico do state. Compara dataset ↔ calibrador downstream.

---

## 19. Referências

- FaceAudit interno:
  - `docs/interacao_score_raridade_faceaudit.md` (fundamentação da
    interação cosseno × raridade; `SCORE_INTERACTION_FEATURE`).
  - `docs/calibracao_faceaudit.md` (`LogisticRegressionCalibrationModel`,
    score multivariado).
  - `docs/conditional_monotonicity_rule_design.md` (auditoria
    V1/V1.5/V2).
  - `docs/piecewise_linear_calibration.md` (calibrador piecewise +
    `IdentityDisjointPairKFold`).
  - `docs/typicality_pair_scoring_faceaudit.md` (proveniência da
    raridade upstream em V2.15).
- Referências matemáticas:
  - Tukey, J. W. (1949). *One degree of freedom for non-additivity.*
    Biometrics. (Inspiração do princípio hierárquico das arquiteturas
    HIERARCHICAL_CENTERED e BILATERAL_FULL_CENTERED.)
  - Nelder, J. A. (1977). *A reformulation of linear models.* J. Royal
    Stat. Society. (Centragem MEAN e interações.)
  - Brümmer, N., & de Villiers, E. (2010). *Application-independent
    evaluation of speaker detection.* Computer Speech & Language.
    (Cllr e calibração afim Brümmer.)

---

## Apêndice A — `LRArchitectureVectorizationInstruction`

Definição em
`faceaudit/models/instructions/lr_architecture_vectorization_instruction.py`.

### Campos

Listados em §14. `REACTIVE_FIELDS = ("architecture", "parameter_mode")`.

### Hooks

- `__post_init__`: fixa `instruction_type`, chama
  `_sync_declared_io()` e `_ensure_default_validation_rules()`.
- `get_hidden_fields()`: esconde campos irrelevantes para a
  arquitetura/modo atual (cobre 8 cenários de visibilidade).
- `_sync_declared_io()`: recalcula `input_resource_ids` (base +
  raridades + inherit) e `output_resource_ids` (vetorizado +
  raw_input).
- `_ensure_default_validation_rules()`: popula `MIN_CALIBRATION_PAIRS`
  quando `FIT_FULL_CALIBRATION_LEGACY` + `base_score_resource_id`.

### `from_dict` / `to_dict`

- `from_dict` hidrata `architecture` e `parameter_mode` via `from_str`;
  re-injeta `instruction_type` (exigido pelo dataclass constructor).
- `to_dict` converte enums para `value` (string estável em JSON).

---

## Apêndice B — Tabela completa das 15 arquiteturas

`LR_ARCHITECTURE_SPECS` em `faceaudit/models/lr/lr_architecture_spec.py`.

| ID | Família | D | s₀ | r_F | r_P | r_single | Centragem | Pair roles |
|---|---|---|---|---|---|---|---|---|
| `COSINE_ONLY` | two-face | 1 | — | — | — | — | — | NOT_APPLICABLE |
| `HINGE_ONLY` | two-face | 2 | sim | — | — | — | — | NOT_APPLICABLE |
| `TWO_SIDED_HINGE_ONLY` | two-face | 3 | sim | — | — | — | — | NOT_APPLICABLE |
| `ONE_SIDED_NON_CENTERED` | two-face | 3 | sim | sim | sim | — | NONE | obrigatório |
| `ONE_SIDED_CENTERED` | two-face | 3 | sim | sim | sim | — | MEAN | obrigatório |
| `LINEAR_NON_CENTERED` | two-face | 3 | sim | sim | sim | — | NONE | obrigatório |
| `LINEAR_CENTERED` | two-face | 3 | sim | sim | sim | — | MEAN | obrigatório |
| `HIERARCHICAL_CENTERED` | two-face | 4 | sim | sim | sim | — | MEAN | obrigatório |
| `BILATERAL_FULL_CENTERED` | two-face | 7 | sim | sim | sim | — | MEAN | obrigatório |
| `ONE_SIDED_NON_CENTERED_ONE_FACE` | ONE_FACE | 2 | sim | — | — | sim | NONE | NOT_APPLICABLE |
| `ONE_SIDED_CENTERED_ONE_FACE` | ONE_FACE | 2 | sim | — | — | sim | MEAN | NOT_APPLICABLE |
| `LINEAR_NON_CENTERED_ONE_FACE` | ONE_FACE | 2 | sim | — | — | sim | NONE | NOT_APPLICABLE |
| `LINEAR_CENTERED_ONE_FACE` | ONE_FACE | 2 | sim | — | — | sim | MEAN | NOT_APPLICABLE |
| `HIERARCHICAL_CENTERED_ONE_FACE` | ONE_FACE | 3 | sim | — | — | sim | MEAN | NOT_APPLICABLE |
| `BILATERAL_FULL_CENTERED_ONE_FACE` | ONE_FACE | 5 | sim | — | — | sim | MEAN | NOT_APPLICABLE |

---

## Apêndice C — Checklist operacional

Antes de declarar uma `LR_ARCHITECTURE_VECTORIZATION`:

- [ ] Arquitetura escolhida coerente com a tese forense da
      estratégia.
- [ ] `parameter_mode` apropriado para a fase:
  - PREPARATION → `FIT_PER_FOLD` (V2.14) ou
    `FIT_FULL_CALIBRATION_LEGACY` (compat V2.13).
  - VALIDATION/CASEWORK → `INHERIT` obrigatório.
- [ ] Recursos de raridade exigidos pela arquitetura estão
      disponíveis (probe + reference para two-face; single para
      ONE_FACE).
- [ ] Modelos de tipicidade upstream com `model_contract_fingerprint`
      registrado no laudo.
- [ ] `output_resource_id` definido.
- [ ] `output_raw_input_resource_id` definido em `FIT_PER_FOLD`.
- [ ] `inherit_architecture_state_from_resource_id` aponta para
      recurso vetorizado da PREPARATION em `INHERIT`.
- [ ] `pair_role_orientation` no metadata do base score consistente
      com a arquitetura ordenada (ausência é erro no executor;
      divergência com o state herdado é erro no
      `architecture_state_compat_service`).
- [ ] `MIN_CALIBRATION_PAIRS` ajustado conforme tamanho de D_cal.

Em casework forense:

- [ ] `vectorization_contract_fingerprint` registrado no laudo para
      re-validação independente.
- [ ] Auditoria `monotonicity_audit` do calibrador com `audit_status
      = OK` (ou WARNING justificado em `central_box`).
- [ ] `reportable_box` declarado no `CONDITIONAL_MONOTONICITY` da
      regra V2 quando houver política formal de reportabilidade.

---

**Última revisão:** V2.14 + V2.14.1 + V2.15 (2026-06-08).
