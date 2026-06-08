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
| `INVERSE_IDENTITY_FREQUENCY` | Vizinhos contribuem com peso `1/n_faces(identidade)`; CDF de referência também ponderada. | EM ponderado: cada face entra na E/M-step com peso `1/n_faces(identidade)`. | Iguala identidades preservando diversidade intra-identidade. |
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

> **Aviso de status (V2.15)**: `INVERSE_IDENTITY_FREQUENCY` ainda **não
> está implementado** em PR 4 — o serviço levanta
> `NotImplementedError` com mensagem apontando esta seção. A
> especificação está fixada para evitar reinterpretação no momento da
> implementação. Próxima iteração materializa.

### 4.2 Semântica no MoVMF

- `IDENTITY_CENTROIDS`: agrega faces de cada identidade em centróide
  L2-renormalizado; treina o MoVMF sobre os centróides (uma observação
  por pessoa). `sample_count` do modelo = nº de identidades.
- `INVERSE_IDENTITY_FREQUENCY` (não implementado em V2.15 — ver §4.1):
  EM ponderado. A E-step e a M-step usam pesos `w_i` em vez de
  contagem; `n_k` na M-step vira `Σ_i w_i · r_ik`.
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

### 5.2 Implementação V2.15 (PR 4)

- `ReferenceScoreCalibrationService.calibrate_via_cross_fit_identity`
  funcional para MoVMF e k-NN.
- Resultado: `ReferenceScoreCalibrationResult` com
  `refit_model`, `oof_cdf`, `ood_threshold`, `split_fingerprint`,
  `fold_seeds`, `n_folds_used`.
- Validações: identidades disjuntas verificadas no `fit` interno;
  folds com menos de K identidades únicas → erro explícito.

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
3. **Sensibilidade a `n_components`**: BIC vs estabilidade ao escolher
   K. Heurística defensável documentada.
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

**Última atualização**: V2.15 (PR 4 — endurecimento experimental).
