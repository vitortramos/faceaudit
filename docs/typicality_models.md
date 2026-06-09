# Modelos de tipicidade no FaceAudit

Doc canônica dos dois backends da família de tipicidade —
`KNNTypicalityModel` e `MoVMFTypicalityModel` — após a refatoração
V2.15 (PRs 1a, 1b, 2, 3, 4). Cobre matemática, suposições, fingerprints,
política de norma L2, ponderação populacional, calibração de
referência (cross-fit por identidade) e migração de estratégias V2.14.

> **Status do MoVMF: EXPERIMENTAL.** PR 4 introduz o envelope de
> segurança (norma L2, balanceamento, cross-fit, canonicalização,
> health checks). A defensibilidade científica para casework depende de
> validação independente. Veja §8 abaixo para a lista de validações
> pendentes.

## 1. Os dois backends — visão geral

| Aspecto | `KNNTypicalityModel` | `MoVMFTypicalityModel` |
|---|---|---|
| Família | k-NN esférico (Loftsgaarden-Quesenberry) | Mistura de von Mises-Fisher |
| Estado pós-fit | Referência inteira (N×D) + CDF LOO de T_k | μ_m, κ_m, π_m (compacto) + CDF + threshold OOD |
| Suposições | Nenhuma distribucional; hubness vira sinal | Simetria rotacional por componente; dim D fixo |
| Score nativo | T_k = (1/k) Σ cos(x, y_(i)) | p_MoVMF(x) = Σ π_m C_d(κ_m) exp(κ_m μ_mᵀ x) |
| Defensibilidade | Tier 1: ranking não-paramétrico ("X é mais típico que 73 % da população") | Tier 2: exige justificar K, κ, μ, centering em corte |
| Custo de inferência | O(N·D) por query | O(K·D) por query (compacto) |
| Status | STABLE | **EXPERIMENTAL** (PR 4) |

Os dois modelos satisfazem `TypicalityScoringModelProtocol` e
`FittableTypicalityModelProtocol`; `compute_scores(embeddings, score_type)`
retorna um `TypicalityScoreBatch` tipado carregando proveniência completa
(fingerprints + `score_semantics_id` + base do percentil).

## 2. Score types (PR 3)

| `TypicalityScoreType` | k-NN | MoVMF | `score_semantics_id` |
|---|---|---|---|
| `MEAN_KNN_SIMILARITY` | ✅ T_k bruto em [-1, 1] | ❌ (sem análogo paramétrico) | `MEAN_KNN_SIMILARITY_SPHERICAL` |
| `TYPICALITY_PERCENTILE` | ✅ rank empírico de T_k (default histórico) | ✅ rank empírico de log-likelihood (default) | `PERCENTILE_OF_MEAN_KNN_SIMILARITY` (k-NN) / `PERCENTILE_OF_LOG_LIKELIHOOD_MOVMF` (MoVMF) |
| `LOG_DENSITY_ESTIMATE` | ✅ log p̂ via L-Q esférico | ✅ log-likelihood da mistura | `LOG_DENSITY_ESTIMATE_LQ_SPHERICAL` (k-NN) / `LOG_LIKELIHOOD_MOVMF` (MoVMF) |

**As escalas absolutas do `LOG_DENSITY_ESTIMATE` diferem** entre backends.
O nome `ESTIMATE` reconhece isso; o `score_semantics_id` no batch
distingue matematicamente a fórmula efetiva. Auditoria forense deve
comparar k-NN vs MoVMF SOMENTE com o mesmo backend nos dois lados, ou
via percentil que normaliza a escala.

### Base do percentil — `TypicalityPercentileBasis`

Percentil sem base persistida é ambíguo. Para evitar que duas
estratégias com mesma rotulação `TYPICALITY_PERCENTILE` consumam
features semanticamente diferentes, o config do backend declara
`percentile_basis_score_type`, propagado no
`TypicalityScoreBatch.percentile_basis_score_type` e persistido no
metadata do `SimilarityScoreDataset`.

- **k-NN default**: `MEAN_KNN_SIMILARITY` (comportamento V2.14
  histórico).
- **MoVMF default**: `LOG_DENSITY_ESTIMATE` (log-likelihood — k-NN T_k
  não tem análogo paramétrico).

## 3. `UnitNormPolicy` (PR 4)

Modelos esféricos assumem `‖x‖₂ = 1`. Embeddings com norma fora de
tolerância levantam a questão pericial: silenciar normalizando, ou
rejeitar?

| Modo | Default | Comportamento |
|---|---|---|
| `ERROR` | ✅ forense | Levanta `UnitNormViolationError` se `|‖x‖ − 1| > tolerance`. |
| `NORMALIZE_WITH_WARNING` | — | Normaliza e emite warning visível com a divergência máxima observada. |
| `NORMALIZE_SILENTLY` | — (migração) | Normaliza sem aviso. Comportamento histórico do k-NN; usado pela migração V2.14. |

`unit_norm_tolerance` define o limite; default `1e-4`.

A política é avaliada em ambos `fit` e `compute_scores`. Em `fit`, a
política se aplica a todo o dataset de referência; em `compute_scores`,
a queries individuais.

## 4. `PopulationWeightingMode` (PR 4)

Substitui o `TypicalityReferenceMode` (k-NN-only) por um conceito
harmonizado entre k-NN e MoVMF.

| Modo | k-NN | MoVMF | Defensibilidade |
|---|---|---|---|
| `FACE_LEVEL_RAW` | Cada face é um ponto da referência. | EM sobre todas as faces; identidades com mais fotos pesam mais. | Vulnerável a desbalanceamento de coleta. |
| `INVERSE_IDENTITY_FREQUENCY` (PR 5) | Vizinhos contribuem com peso `1/n_faces(identidade)`; CDF de referência também ponderada. | EM ponderado: cada face entra na M-step com peso `1/n_faces(identidade)`; `n_k = Σ_i w_i · r_ik`. | Iguala identidades preservando diversidade intra-identidade. |
| `IDENTITY_CENTROIDS` | Referência tem 1 centróide L2-renormalizado por identidade. | EM treinado sobre centróides. | Default forense; robusto. |

### 4.1 Semântica matemática do `INVERSE_IDENTITY_FREQUENCY` no k-NN

Pré-requisito do plano: "fórmula efetiva persistida no modelo para
auditoria". Implementação V2.15 segue:

Seja `n_i` o número de faces da identidade da face `i`. Definimos
`w_i = 1/n_i` (faces de identidades raras pesam mais que faces de
identidades super-representadas).

**No fit (CDF de referência)**: a estatística T_k de cada face do
treino entra na CDF empírica `loo_mean_knn_sorted` com peso `w_i` em
vez de 1. A CDF passa a ser construída via ordenação por valor + soma
acumulada dos pesos (em vez de contagem). Auxiliar:
`reference_score_weights: np.ndarray (N,) float64`, persistido junto
com `loo_mean_knn_sorted`.

**Na agregação T_k da query**: cada vizinho `y_j` contribui para a
soma com peso `w_j / Σ_{k vizinhos} w_k` (renormalização local). Em
modo `FACE_LEVEL_RAW` isso degenera para o `1/k` tradicional.

**Na inferência de percentil**: o rank usa a CDF ponderada
(`reference_score_weights` cumsum).

**Persistido no modelo**:
- `reference_score_weights` (estado),
- `population_weighting_mode = INVERSE_IDENTITY_FREQUENCY` (config),
- `weighted_T_k_formula_id = "INV_ID_FREQ_v1"` (auditoria, entra no
  `model_contract_fingerprint`).

> **Status (V2.15 / PR 5)**: `INVERSE_IDENTITY_FREQUENCY` **implementado**
> em ambos os backends. Persiste `reference_score_weights` no estado e
> `weighted_T_k_formula_id = "INV_ID_FREQ_v1"` no
> `model_contract_fingerprint` para auditoria.

### 4.2 Semântica no MoVMF

- `IDENTITY_CENTROIDS`: agrega faces de cada identidade em centróide
  L2-renormalizado; treina o MoVMF sobre os centróides (uma observação
  por pessoa). `sample_count` do modelo = nº de identidades.
- `INVERSE_IDENTITY_FREQUENCY` (PR 5, **implementado**): EM ponderado.
  A E-step segue inalterada; a M-step usa pesos `w_i = 1/n_faces(id_i)`
  em vez de contagem unitária — `n_k = Σ_i w_i · r_ik`, e a média
  ponderada da log-likelihood entra em `_train_avg_log_likelihood`.
  CDF de referência também ponderada (cumsum dos pesos ordenados por
  LL). Inicialização do k-means++ permanece não-ponderada na V2.15 —
  P2 conhecido, ver §10.
  **Atenção forense:** AIC/BIC retornam `None` neste modo (P1-R2-6
  da revisão externa round 2): a definição formal de N efetivo
  ponderado ainda está em aberto e o valor ingênuo (N = n_faces)
  inflaria o LL total. **Use a log-likelihood preditiva OOF
  (`oof_predictive_avg_log_likelihood`) como critério primário para
  selecionar K** — vide §5.3.
- `FACE_LEVEL_RAW`: EM clássico, comportamento V2.14 do MoVMF interno.

## 5. `ReferenceScoreCalibrationMode` (PR 4)

| Modo | Default | Comportamento |
|---|---|---|
| `CROSS_FIT_IDENTITY` | ✅ recomendado (MoVMF) | K-fold por identidade; CDF construída com scores OOF; refit final em D. |
| `IN_SAMPLE_LEGACY` | ✅ k-NN (compat V2.14) | CDF construída sobre o mesmo conjunto do fit. Otimista. |

### 5.1 Protocolo `CROSS_FIT_IDENTITY`

Procedimento de duas etapas (formalização explícita do plano):

1. **Splits identitários disjuntos.** Particiona identidades únicas em
   K folds via `sklearn.model_selection.KFold` (ou similar
   determinístico via `random_state`). Persistir mapeamento
   `identity → fold_index` no fingerprint do split.
2. **Para cada fold**: treina o backend nos folds de treino (faces
   cujas identidades estão nos folds train). Pontua identidades do
   fold test (faces cujas identidades estão no fold test). Coleta os
   scores OOF.
3. **CDF + threshold OOD**: concatena os scores OOF de todos os folds
   e ordena. CDF empírica = ordenação; threshold OOD =
   `np.quantile(oof_sorted, ood_quantile)`.
4. **Refit final**: treina o backend em **toda a população de
   referência**. O modelo persistido é este refit; a CDF/threshold
   vêm do passo 3.

**Modelo persistido = refit em D completo.** CDF/threshold ≠ refit.
Auditoria forense deve referenciar ambos.

### 5.2 Implementação V2.15 (PR 4 + revisões R1–R4)

- `ReferenceScoreCalibrationService.calibrate(...)` funcional para
  MoVMF e k-NN.
- Modelo persistido **é o refit em D completo**; a CDF/threshold
  vêm dos scores OOF. O `install_reference_score_calibration` (P0-1
  do round 1) substitui atomicamente a referência IN-SAMPLE do refit
  pela OOF — após o serviço retornar, `get_percentiles()` e
  `is_in_distribution()` consultam a OOF, **não** a in-sample.

#### Campos do `ReferenceScoreCalibrationResult`

Referência OOF (R3 — round 3 corrigiu nomeação enganosa):

| Campo | Significado |
|---|---|
| `refit_model` | Modelo treinado em toda a população, com referência OOF JÁ instalada. |
| `oof_reference_scores_sorted` | Scores OOF ordenados crescente (era `oof_cdf`; renomeado em R3). |
| `oof_reference_weights_sorted` | Pesos por score (mesma ordem). |
| `oof_reference_cdf` | CDF empírica ponderada = `cumsum(weights)/sum`. |
| `ood_threshold` | Quantil ponderado global. |
| `split_fingerprint` | Hash do mapeamento identidade → fold. |
| `fold_seeds` | Seeds efetivamente propagadas ao `trainer_factory`. |
| `n_folds_used` | Número de folds que produziram scores OOF. |

> O alias `result.oof_cdf` segue como property com `DeprecationWarning`
> apontando para `oof_reference_scores_sorted`. Será removido em
> próxima major.

Métricas preditivas OOF (R4 — round 4):

| Campo | Significado |
|---|---|
| `oof_predictive_avg_score` | Média ponderada de TODOS os scores OOF (mesmos pesos da CDF). No **MoVMF** (basis = `LOG_DENSITY_ESTIMATE`) é a **estimativa OOF de E[log p_θ(x)]** — a PLL OOF que o revisor recomenda como critério primário para escolher K. No **k-NN** (basis = `MEAN_KNN_SIMILARITY`) é T_k médio populacional — útil como métrica relativa entre `k`'s, **não** interpretável como log-densidade. |
| `oof_predictive_avg_score_by_fold` | Médias por fold (mesmas regras de ponderação). Dispersão entre folds é o diagnóstico de estabilidade que o revisor pede ao lado da PLL agregada. |
| `oof_predictive_score_type` | `"LOG_DENSITY_ESTIMATE"` ou `"MEAN_KNN_SIMILARITY"` — chave para interpretar `oof_predictive_avg_score`. |
| `n_identities_by_fold` | Identidades únicas pontuadas em cada fold. |
| `n_scores_by_fold` | Quantidade de scores OOF por fold (= n_identidades em `IDENTITY_CENTROIDS`; = n_faces em `FACE_LEVEL_RAW` ou `INVERSE_IDENTITY_FREQUENCY`). |
| `ood_threshold_by_fold` | Threshold OOD do `fold_model` transitório, por fold (estabilidade do quantil). Cada entrada pode ser `None` quando `ood_quantile = 0.0` no config. |

Os arrays expostos nesses campos são **read-only** (P1-R3-7) — auditoria
não pode mutar o snapshot devolvido.

#### Exposição no `summary()`

`summary().backend_details` propaga as métricas preditivas para
auditoria pericial e para a UI:

- **MoVMF** expõe explicitamente o alias log-likelihood:
  - `oof_predictive_avg_log_likelihood` — só populado quando
    `oof_predictive_score_type == "LOG_DENSITY_ESTIMATE"` (o caso
    geral do MoVMF). É **a** métrica forense recomendada para
    selecionar K.
  - `oof_predictive_log_likelihood_by_fold` — análogo por fold.
  - `oof_predictive_avg_score`, `oof_predictive_avg_score_by_fold`,
    `oof_predictive_score_type` — versões genéricas.
- **k-NN** expõe apenas os campos genéricos
  (`oof_predictive_avg_score`, `oof_predictive_avg_score_by_fold`,
  `oof_predictive_score_type`). Não há alias `*_log_likelihood`
  porque `MEAN_KNN_SIMILARITY` não é log-densidade.

#### Persistência

Os campos preditivos sobrevivem ao `to_dict` / `from_dict` em ambos
os backends. Pickles produzidos antes de R4 não trazem esses campos
e restauram-nos como `None` — comportamento back-compat.

#### Validações

- Identidades disjuntas verificadas no `fit` interno.
- `n_folds >= 2` e ≤ identidades únicas → erro explícito (P0-R3-4).
- Backend declarado vs instância retornada pela `trainer_factory`:
  divergência levanta `BackendMismatchError` (P0-R3-2).
- `validate_reference_score_distribution` exige acoplamento
  weights+CDF: ambos presentes → CDF deve ser exatamente
  `cumsum(weights)/sum(weights)` (P0-R3-5).

### 5.3 Selecionando `n_components` (K) no MoVMF

Pergunta forense central: dado um corpus, qual K usar? R4 consolida
a recomendação do revisor.

Critério primário recomendado, em ordem de prioridade:

| Prioridade | Métrica | Onde ler |
|---|---|---|
| 1 | **log-likelihood preditiva OOF por identidade** | `model.summary().backend_details["oof_predictive_avg_log_likelihood"]` (após `CROSS_FIT_IDENTITY` com `population_weighting_mode = IDENTITY_CENTROIDS`) |
| 2 | **dispersão da PLL OOF entre folds** | `model.summary().backend_details["oof_predictive_log_likelihood_by_fold"]` — desvio padrão / range |
| 3 | **saúde dos componentes** | `model.summary().warnings` + `model.summary().backend_details["weights"]/["kappas"]` + método `model.get_degenerate_components()` (separado do summary) |
| 4 | **estabilidade dos percentis e do threshold OOD** | `ReferenceScoreCalibrationResult.ood_threshold_by_fold` (no objeto devolvido pelo serviço cross-fit; o `summary()` do modelo expõe apenas o threshold global em `backend_details["ood_log_likelihood_threshold"]`) |
| 5 | AIC / BIC (apenas diagnóstico auxiliar) | `model.summary().backend_details["aic"]`, `["bic"]` (note as chaves sem underscore) — `None` em `INVERSE_IDENTITY_FREQUENCY` (definição de N efetivo ponderado em aberto, P1-R2-6). Não usar como critério único. |
| 6 | Efeito downstream nas LRs em base de desenvolvimento separada | Métricas do `STRATEGY_EVALUATION` (Cllr, ECE, WS) executado em D_val honesto |

**Recomendação prática**: rodar K ∈ {1, 3, 5} (por exemplo) com o mesmo
split de identidades, e tabular:

| K | PLL OOF média | desvio entre folds | menor peso | maior κ | componentes degenerados | threshold OOD |
|---|---|---|---|---|---|---|

K que maximize PLL **sem aumentar dispersão entre folds** nem produzir
componentes degenerados é o candidato. Mudanças downstream só são
relevantes quando os critérios 1–4 estão empatados.

**Sinais auxiliares baratos** (descarte rápido de K instáveis, antes de
olhar PLL):
- `backend_details["converged"]` — `False` indica EM que não convergiu
  na inicialização final.
- `backend_details["n_iter"]` saturando `max_iter` é sintoma análogo.
- `backend_details["avg_train_log_likelihood"]` − `oof_predictive_avg_log_likelihood`
  quantifica o gap **in-sample vs OOF**; gap crescendo com K é overfit
  explícito.

A justificativa do revisor para escolher PLL OOF acima de AIC/BIC: os
dois critérios de informação só são definidos com tamanho amostral
efetivo bem postado. Em `INVERSE_IDENTITY_FREQUENCY`, AIC/BIC retornam
`None` por essa razão. PLL OOF não precisa dessa definição — é uma
estimativa direta do que o modelo "vai produzir" em identidades não
vistas.

## 6. Fingerprints (PR 4 endurece)

Três camadas por modelo (espelha V2.14 do LR):

| Camada | Conteúdo | Uso |
|---|---|---|
| `model_state_fingerprint` | μ/κ/π (canonicalizados) + CDF + threshold + reference_score_weights | Hash de estado numérico exato |
| `model_contract_fingerprint` | model_type + score_semantics_ids + population_weighting_mode + unit_norm_policy + reference_score_calibration_mode + upstream_embedding_fingerprint | Entra no `vectorization_contract_fingerprint` downstream — INHERIT compara este |
| `model_provenance_fingerprint` | Resource IDs + timestamps + seeds + warnings + commit hash | Auditoria — não comparado na aplicação |

### 6.1 Canonicalização de componentes MoVMF

Mistura tem simetria de permutação: trocar a ordem dos componentes
não altera a densidade. Sem canonicalização, modelos matematicamente
equivalentes geram fingerprints distintos.

PR 4 ordena componentes antes de serializar/hashar via chave
**lexicográfica composta**:
1. **Peso decrescente** (componente mais massivo primeiro).
2. **κ decrescente** (entre componentes com mesmo peso).
3. **Hash lex de μ** (`hashlib.sha256(μ.tobytes(order="C")).hexdigest()`)
   como desempate determinístico.

A ordenação é aplicada em `MoVMFTypicalityModel.to_dict` e nos cálculos
de fingerprint. Validador científico crítico:
permutar componentes → mesmo `model_contract_fingerprint`.

## 7. Health checks MoVMF (PR 4)

`MoVMFTypicalityModel.summary_protocol()` retorna
`TypicalityModelSummary` com `health_status ∈ {OK, WARNING, CRITICAL}`
e `warnings: tuple[str, ...]` cobrindo:

1. `n_components <= effective_population_size`.
2. Cada componente com **suporte mínimo** (faces atribuídas).
3. Nenhum peso abaixo de `min_component_weight` após o ajuste.
4. Nenhum κ saturou `max_kappa` sem warning.
5. log-likelihood finita.
6. Convergência em **pelo menos uma inicialização**
   (`converged_at_least_one_init`).
7. `best_init_index` documentado entre `n_init`.
8. Responsabilidades não degeneradas (não há componente com
   `argmax(r_ik) ≡ vazio`).
9. CDF OOF com tamanho mínimo (`n_oof_samples ≥ min_oof_required`)
   quando `reference_score_calibration_mode = CROSS_FIT_IDENTITY`.
10. Folds OOF com identidades disjuntas (`identity_overlap_between_folds = 0`).

Persiste também: `random_state`, `init_seeds`, `best_init_index`,
`fold_seeds`, `cross_fit_identity_split_fingerprint`.

## 8. Validações pendentes para promover MoVMF a STABLE

Antes de remover o badge `EXPERIMENTAL`:

1. **Estabilidade entre `random_state`**: variância dos coeficientes
   downstream em ≥ 20 seeds, em pelo menos 3 corpora (CFP-FP, CASIA,
   próprio).
2. **Comparação k-NN vs MoVMF em condições controladas**: mesma
   `population_weighting_mode`, mesma `reference_score_calibration_mode`,
   mesmo split, mesmo LR architecture.
3. **Sensibilidade a `n_components`**: tabela documentada de PLL OOF
   (`oof_predictive_avg_log_likelihood`) + dispersão entre folds +
   componentes degenerados para K ∈ {1, 3, 5} em ≥ 3 corpora. AIC/BIC
   apenas como diagnóstico auxiliar (vide §5.3). Heurística defensável
   documentada.
4. **Comportamento sob agulhas de identidade** (paired adversarial):
   adicionar/remover faces de uma identidade deve mover scores
   downstream **menos** que k-NN no mesmo cenário.
5. **Invariância sob duplicação** em `IDENTITY_CENTROIDS`: duplicar
   todas as faces de uma identidade não muda o modelo treinado.
6. **Cross-fit construindo CDF apenas com OOF**: teste técnico que
   nenhum score de identidade aparece em sua própria CDF; modelo
   persistido é o refit completo.

Critério: cada um dos 6 itens com benchmark externo arquivado
documentando o resultado. Recomendado executar via instrumento
indicado em `docs/conditional_monotonicity_rule_design.md` §7
(auditoria runtime no `CALIBRATION_MODEL_TRAINING`).

## 9. Migração de estratégias V2.14 → V2.15

A migração preservativa (`LegacyStrategyMigrationService`) detecta e
converte automaticamente em memória; o JSON original é mantido
intacto e a UI grava uma cópia `*.v2_15.fa` lado a lado sob ação
explícita do usuário.

### Mapeamento de defaults históricos (literais)

| Campo antigo | Campo novo | Valor migrado |
|---|---|---|
| `instruction_type = KNN_TYPICALITY_MODEL_TRAINING` | `instruction_type = TYPICALITY_MODEL_TRAINING` + `typicality_model_type = "KNN"` | (dispatcher) |
| `reference_mode = "PER_FACE"` | `population_weighting_mode` | `"FACE_LEVEL_RAW"` |
| `reference_mode = "PER_IDENTITY"` | `population_weighting_mode` | `"IDENTITY_CENTROIDS"` |
| `score_type = "LOG_DENSITY_KNN"` | `score_type` | `"LOG_DENSITY_ESTIMATE"` |
| (ausente) | `unit_norm_policy` | `"NORMALIZE_SILENTLY"` (comportamento histórico do k-NN) |
| (ausente) | `reference_score_calibration_mode` | `"IN_SAMPLE_LEGACY"` (LOO de imagem) |
| (ausente) | `percentile_basis_score_type` | `"MEAN_KNN_SIMILARITY"` (histórico) |
| (ausente) | `cv_n_folds` | `0` |
| campos preservados | `k`, `ood_quantile`, `population_label`, `embedding_column` | literal do JSON antigo |

Comportamento histórico V2.14 é preservado bit-a-bit.

### Modo estrito

`LRStrategyDefinition.from_dict(data, allow_legacy_auto_migration=False)`
levanta `LegacyStrategyVersionRejected` ao detectar sinais legados.
Usado em validação formal externa onde toda migração deve passar pela
ferramenta explícita.

---

**Última atualização**: V2.15 + R4 (log-likelihood preditiva OOF e
métricas por fold expostas no `ReferenceScoreCalibrationResult` e no
`summary()` dos backends — vide §5.2 e §5.3).
