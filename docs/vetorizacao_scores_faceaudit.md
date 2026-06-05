# Vetorização de scores no FaceAudit (`SCORE_VECTORIZATION`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `SCORE_VECTORIZATION` — o empilhamento de N (≥2)
> `SimilarityScoreDataset` sobre os mesmos pares numa matriz de features
> multivariadas. Objetivo, fundamentação, alinhamento, configurações e exemplos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** repositório FaceAudit no commit `55c3d8a`
> (branch `claude/read-claude-md-SgzIV`). Campos e algoritmo conferidos em
> `faceaudit/executors/instructions/score_vectorization_instruction_executor.py`,
> `faceaudit/models/instructions/score_vectorization_instruction.py`.
>
> **⚠️ V2.14 — alternativa declarativa para arquiteturas LR canônicas.**
> A partir de V2.14, para as **9 arquiteturas LR forenses canônicas** (do
> `COSINE_ONLY` puro à `BILATERAL_FULL_CENTERED` cheia), o substituto
> ergonômico é a instrução `LR_ARCHITECTURE_VECTORIZATION` — uma única
> instrução declarativa que dispensa o encadeamento manual
> `SCORE_INTERACTION_FEATURE × N + SCORE_VECTORIZATION` e ainda corrige o
> leakage residual de s₀/r̄ via `parameter_mode=FIT_PER_FOLD` (s₀/r̄
> ajustados POR FOLD dentro do `GridSearchCV` do calibrador). Veja
> §1.6 abaixo para a árvore de decisão completa. **O `SCORE_VECTORIZATION`
> permanece como ferramenta geral** para fusões fora do registry (ex.:
> empilhar score de qualidade + cosseno + PLDA arbitrariamente, ou compor
> features que não derivam de raridade × similaridade).

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a vetorização se encaixa](#1-onde-a-vetorização-se-encaixa)
  - [1.6 `SCORE_VECTORIZATION` vs `LR_ARCHITECTURE_VECTORIZATION`](#16-score_vectorization-vs-lr_architecture_vectorization)
- [2. Fundamentação — fusão score-level](#2-fundamentação--fusão-score-level)
- [3. O que a instrução faz](#3-o-que-a-instrução-faz)
- [4. O algoritmo de alinhamento](#4-o-algoritmo-de-alinhamento)
- [5. A saída — matriz de features](#5-a-saída--matriz-de-features)
- [6. Como o calibrador consome a matriz](#6-como-o-calibrador-consome-a-matriz)
- [7. Configurações](#7-configurações)
- [8. Validações e guard-rails](#8-validações-e-guard-rails)
- [9. Exemplo](#9-exemplo)
- [10. Boas práticas resumidas](#10-boas-práticas-resumidas)
- [11. Perguntas frequentes](#11-perguntas-frequentes)
- [12. Glossário](#12-glossário)
- [13. Referências](#13-referências)
- [Apêndice A — ScoreVectorizationInstruction](#apêndice-a--scorevectorizationinstruction)
- [Apêndice B — Chaves de metadata da saída](#apêndice-b--chaves-de-metadata-da-saída)
- [Apêndice C — Checklist de validação operacional](#apêndice-c--checklist-de-validação-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre a instrução `SCORE_VECTORIZATION`: como ela
combina vários sistemas de score (cosseno, PLDA, tipicidade, feature de
interação, qualidade) sobre os mesmos pares num **vetor de features** que o
calibrador multivariado consome.

### 0.2 Escopo

**Incluído:** o empilhamento de N (≥2) `SimilarityScoreDataset`, o alinhamento
de pares por chave simétrica, a matriz de features na saída e a conexão com o
calibrador.

**Fora do escopo:**

- a produção dos scores de entrada (`SIMILARITY_SCORING`,
  `TYPICALITY_PAIR_SCORING`, `SCORE_INTERACTION_FEATURE`) — ver os docs
  correspondentes;
- o **treino do calibrador** multivariado que consome a matriz
  (`CALIBRATION_MODEL_TRAINING`) — ver `docs/calibracao_faceaudit.md`.

### 0.3 Documentos complementares

| Documento | Relação |
|---|---|
| `docs/similaridade_faceaudit.md` | produz scores de similaridade (entrada) |
| `docs/tipicidade_pareada_faceaudit.md` | produz scores de tipicidade (entrada) |
| `docs/interacao_score_raridade_faceaudit.md` | produz a feature de interação (entrada) |
| `docs/vetorizacao_scores_faceaudit.md` (este) | empilha as entradas num vetor |
| `docs/calibracao_faceaudit.md` | consome o vetor no calibrador multivariado |

---

## 1. Onde a vetorização se encaixa

```text
SIMILARITY_SCORING (cosseno)        ┐
SIMILARITY_SCORING (PLDA)           │  N sistemas, mesmos pares
TYPICALITY_PAIR_SCORING             │
SCORE_INTERACTION_FEATURE           ┘
        ↓
SCORE_VECTORIZATION   ← ESTE DOCUMENTO
        ↓
SimilarityScoreDataset com matriz (N_pairs, D) em metadata
        ↓
CALIBRATION_MODEL_TRAINING (LogReg multivariado, aprende um peso por sistema)
        ↓
log-LR contínuo
```

A vetorização é o passo que **prepara** as features multivariadas. O aprendizado
dos pesos por sistema acontece **dentro do calibrador** a jusante — esta
instrução não calibra nada.

### 1.6 `SCORE_VECTORIZATION` vs `LR_ARCHITECTURE_VECTORIZATION`

V2.14 introduziu uma instrução declarativa que cobre exatamente a fatia mais
comum de uso da vetorização — empilhar score base (cosseno/PLDA) + features de
interação derivadas da raridade — sem que o usuário precise reproduzir manualmente
a aritmética de hinges, centragens e produtos. A escolha entre as duas é a
seguinte:

| Cenário | Instrução recomendada |
|---|---|
| Arquitetura LR do registry: `COSINE_ONLY`, `HINGE_ONLY`, `TWO_SIDED_HINGE_ONLY`, `ONE_SIDED_NON_CENTERED`, `ONE_SIDED_CENTERED`, `LINEAR_NON_CENTERED`, `LINEAR_CENTERED`, `HIERARCHICAL_CENTERED`, `BILATERAL_FULL_CENTERED` | **`LR_ARCHITECTURE_VECTORIZATION`** — uma instrução, integrado por construção com o registry de raridade e o calibrador (`vectorization_contract_fingerprint`). Suporta `FIT_PER_FOLD` (s₀/r̄ ajustados por fold) e `INHERIT` em VALIDATION/CASEWORK. |
| Empilhar fontes heterogêneas fora do registry (ex.: cosseno + PLDA + qualidade FIQA, ou cosseno + PLDA + interação cross-pose customizada) | **`SCORE_VECTORIZATION`** (esta instrução). Continua sendo o caminho genérico para fusão score-level multivariada arbitrária. |
| Híbrido: 1 arquitetura LR canônica + 1 feature extra fora do registry | Use `LR_ARCHITECTURE_VECTORIZATION` para produzir o sub-vetor canônico, depois `SCORE_VECTORIZATION` empilhando o output com a feature extra. |

Diferença material (não apenas ergonômica): o caminho declarativo elimina o
leakage residual conhecido do encadeamento manual. No fluxo
`SCORE_INTERACTION_FEATURE + SCORE_VECTORIZATION`, `s₀` é estimado em
`SCORE_INTERACTION_FEATURE` **antes** da CV do calibrador a jusante — vazamento
sutil pra dentro do GridSearch. No fluxo declarativo com
`parameter_mode=FIT_PER_FOLD`, o `CALIBRATION_MODEL_TRAINING` monta
`Pipeline([ArchitectureVectorizerTransformer → StandardScaler → LogReg])` e o
transformer **re-ajusta s₀/r̄ por fold** dentro do `GridSearchCV` — após
`refit=True`, o state final reflete refit sobre D_cal completo. Documentado em
detalhe na seção "Arquitetura LR declarativa" do `CLAUDE.md`.

Compatibilidade: o encadeamento antigo continua válido sem migração obrigatória
(preserva tolerância histórica via `legacy_transform_typicality`). Calibradores
treinados pelo fluxo novo aplicam-se a datasets vetorizados pelo fluxo antigo
somente com `compatibility_mode=LEGACY_PERMISSIVE` opt-in (warning visível).

---

## 2. Fundamentação — fusão score-level

### 2.1 Por que não somar log-LRs

A forma ingênua de combinar dois sistemas seria calibrar cada um separadamente e
somar os log-LRs. Isso só é correto se os sistemas forem **condicionalmente
independentes** dadas as hipóteses — o que tipicamente **não** vale: cosseno,
PLDA e tipicidade compartilham o mesmo embedding base, logo são correlacionados.

A **fusão score-level via LogReg multivariado** (Brümmer & du Preez 2006) aprende
um peso por sistema **respeitando a correlação** entre eles. É a forma defensável
de combinar sistemas que não são independentes.

### 2.2 O papel da vetorização

A `SCORE_VECTORIZATION` é a etapa mecânica que monta a matriz de features
$(N_{\mathrm{pairs}}, D)$ — uma coluna por sistema, uma linha por par — para que
o calibrador receba um vetor por par em vez de um escalar. O termo "vectorization"
enfatiza que ela **prepara** as features; a combinação propriamente dita (os pesos)
é aprendida no calibrador.

---

## 3. O que a instrução faz

Recebe N (≥2) `SimilarityScoreDataset` que descrevem **os mesmos pares** e
produz um novo `SimilarityScoreDataset` onde:

- o vetor de scores é uma **única matriz contígua** $(N_{\mathrm{pairs}}, D)$
  em `metadata["score_vector_matrix"]` ($D$ = número de sistemas);
- o `raw_score` de cada par é o do **primeiro input** (o anchor) — preserva
  utilidade em mining e em visualizações de caso que esperam um score escalar 1D;
- `metadata["is_score_vector"] = True` sinaliza ao calibrador que há features
  multivariadas a extrair.

A matriz vive **inteiramente no metadata do dataset** (uma alocação numpy),
não em `sample.metadata` — decisão de performance: para milhões de pares, N
alocações de `List[float]` por sample geraria pressão de GC enorme; a matriz
contígua é zero-copy para o sklearn.

---

## 4. O algoritmo de alinhamento

1. O **primeiro dataset** (`datasets[0]`) é o **anchor** — define a ordem dos
   pares e doa `probe`/`reference`/`same_person`/`raw_score`.
2. Cada dataset subsequente é alinhado contra o anchor:
   - **fast path posicional**: se os pares já estão na mesma ordem (checagem por
     chave), usa direto (evita construir o índice — relevante para centenas de
     milhares de pares);
   - **fallback por reordenação**: se a ordem difere mas os conjuntos batem,
     reordena via índice `chave → sample`.
3. Concatena as colunas num `column_stack` $(N_{\mathrm{pairs}}, D)$.

### 4.1 A chave simétrica do par

O alinhamento usa `frozenset({probe_hash, reference_hash})` — uma chave
**simétrica**: o par $(F_1, L_1)$ casa com $(L_1, F_1)$. Isso é justificado
porque **todos** os modelos de score (cosseno, PLDA) e os combinadores
simétricos da tipicidade produzem o mesmo valor independente da ordem — então
$(p, r)$ e $(r, p)$ representam o mesmo objeto forense. Resolve o caso comum de o
usuário ter definido probe/reference em ordem trocada entre o `SIMILARITY_SCORING`
e o `TYPICALITY_PAIR_SCORING`.

### 4.2 Defesas

- **Cardinalidade**: erro se algum input tem número de pares diferente do anchor
  (sinaliza `PAIR_MINING` em apenas um lado).
- **Pares ausentes**: erro se algum par do anchor não existe num input (com
  exemplos das chaves faltantes). Pares **extras** num input não-anchor são
  descartados silenciosamente.

> **Nota:** o campo `require_pair_match` é aceito por compat de JSON antigo, mas
> **perdeu efeito prático** — o fast path posicional + o fallback por
> reordenação cobrem ambos os casos sem necessidade de toggle.

---

## 5. A saída — matriz de features

Cada sample do output é a **instância original do anchor** (sem mutação),
preservando `same_person`, identidades, `raw_score` e metadata por sample. No
metadata do dataset:

| Chave | Conteúdo |
|---|---|
| `is_score_vector` | sempre `True` |
| `score_vector_matrix` | a matriz numpy $(N_{\mathrm{pairs}}, D)$ |
| `score_vector_source_resources` | IDs dos inputs, em ordem das colunas |
| `score_vector_n_features` | $D$ (número de sistemas) |
| `score_vector_model_ids` | procedência ordenada (id do modelo de cada coluna; `None` em inputs legados) |

A **ordem das colunas** segue a ordem dos inputs: A, B, depois os extras. Essa
ordem é a procedência auditável de cada feature.

---

## 6. Como o calibrador consome a matriz

O `CALIBRATION_MODEL_TRAINING` detecta `is_score_vector=True` no metadata e
extrai a matriz $(N_{\mathrm{pairs}}, D)$ como features. Com LogReg multivariado,
ele aprende:

$$
\log\mathrm{LR} = \beta_0 + \sum_{d=1}^{D}\beta_d \cdot x_d,
$$

onde $x_d$ é a coluna $d$ (o score do sistema $d$). Cada $\beta_d$ é o peso
aprendido para aquele sistema, respeitando a correlação entre eles.

**Recomendação**: use `standardize_features=True` no calibrador quando as
colunas estão em escalas muito diferentes (ex.: cosseno em $[-1,1]$ ×
`log_density_knn` em escala real) — o LogReg fica mais estável e os coeficientes
ficam comparáveis.

> **Nota V2.14.** Quando o vetor vem de `LR_ARCHITECTURE_VECTORIZATION` com
> `parameter_mode=FIT_PER_FOLD`, o `CALIBRATION_MODEL_TRAINING` detecta a flag
> `defer_to_calibrator_per_fold` no metadata do dataset e despacha para um
> caminho separado — monta `Pipeline([ArchitectureVectorizerTransformer →
> StandardScaler → LogReg])` dentro de `GridSearchCV(cv=IdentityDisjointPairKFold(
> probe_identities=..., reference_identities=...))` e ajusta s₀/r̄ POR FOLD
> a partir do `LR_ARCHITECTURE_RAW_INPUT` cru referenciado pelo dataset
> vetorizado. O caminho legacy descrito nesta seção (LogReg simples sobre
> `score_vector_matrix`) permanece como default para datasets emitidos por
> `SCORE_VECTORIZATION` ou pelo modo `FIT_FULL_CALIBRATION_LEGACY`.

---

## 7. Configurações

Campos do `ScoreVectorizationInstruction`:

| Campo | Tipo | Default | O que faz |
|---|---|---|---|
| `input_a_resource_id` | str | — | sistema A (anchor; ex.: cosseno) |
| `input_b_resource_id` | str | — | sistema B (ex.: PLDA ou interação) |
| `extra_input_resource_ids` | `List[str]` | `[]` | sistemas adicionais (3+); IDs separados por vírgula na UI |
| `output_resource_id` | str | — | id do dataset vetorizado de saída |
| `output_dataset_name` | str? | None | nome amigável |
| `require_pair_match` | bool | True | legado; sem efeito prático (ver §4.2) |

O mínimo são **dois** sistemas distintos (A e B). Para 3+ (ex.: cosseno + PLDA +
qualidade), preencha `extra_input_resource_ids`. O helper `all_input_resource_ids()`
agrega A, B e extras, filtrando vazios e desduplicando preservando ordem.

---

## 8. Validações e guard-rails

| Validação | Severidade |
|---|---|
| A e B definidos e presentes | erro |
| extras presentes (se informados) | erro |
| inputs distintos (sem duplicata; A ≠ B) | erro |
| `output_resource_id` definido | erro |
| todos os payloads são `SimilarityScoreDataset` | erro |

Default persistente: `MIN_CALIBRATION_PAIRS` sobre o input A (protege a
calibração a jusante de pares insuficientes; vale só para A porque a regra
dinâmica garante que B tem o mesmo conjunto).

Regra dinâmica (intrínseca): `SCORE_VECTOR_PAIR_COMPATIBILITY` — todos os inputs
(A, B, extras) compartilham exatamente o mesmo conjunto de pares
`(probe.face_hash, reference.face_hash)`. É o catch antecipado que evita o
`ValueError` tardio do executor.

---

## 9. Exemplo

Combinar cosseno + feature de interação (a "raridade modula a similaridade"):

```text
input_a = scores_cosseno          (raw_score = cosseno por par)
input_b = feature_interacao       (raw_score = (cos − s0)·rarity por par)
output  = scores_vetorizados
```

Saída: um `SimilarityScoreDataset` com `score_vector_matrix` de shape
$(N_{\mathrm{pairs}}, 2)$ — coluna 0 = cosseno, coluna 1 = interação — e
`raw_score` = cosseno (do anchor). O calibrador aprende
$\log\mathrm{LR} = \beta_0 + \beta_1\cdot\mathrm{cos} + \beta_2\cdot\mathrm{interacao}$.

Para o modelo plenamente geral da "dupla interação" (probe + reference):

```text
input_a = scores_cosseno
input_b = interacao_hinge          (HINGE_BASE)
extras  = interacao_probe, interacao_reference
```

→ matriz $(N_{\mathrm{pairs}}, 4)$ → $[\mathrm{cos}, h, h\cdot r_q, h\cdot r_r]$.

---

## 10. Boas práticas resumidas

- **Mesmos pares em todos os inputs**: não minere pares só de um lado; o
  conjunto tem que bater (a regra de compatibilidade pega isso).
- **Anchor = o sistema cujo `raw_score` você quer preservar** (tipicamente o
  cosseno) — é o que sobrevive em mining/visualização.
- **Padronize features de escalas díspares** no calibrador
  (`standardize_features=True`).
- **Ordem das colunas é procedência**: documente qual sistema é cada coluna
  (o metadata guarda, mas registre no laudo).
- **A/B**: compare o modelo vetorizado contra o single-system no holdout — só
  empilhe sistemas que agregam.

---

## 11. Perguntas frequentes

**A vetorização calibra os scores?**
Não. Ela só monta a matriz de features. Os pesos são aprendidos no
`CALIBRATION_MODEL_TRAINING` a jusante.

**Por que não somar log-LRs dos sistemas?**
Porque os sistemas compartilham o embedding base e são correlacionados; somar
log-LRs supõe independência condicional, que não vale. A fusão score-level via
LogReg respeita a correlação.

**Quantos sistemas posso combinar?**
Mínimo 2 (A e B). Para 3+, use `extra_input_resource_ids`. Cada coluna vira uma
feature; mais features exigem mais dados de calibração para não overfitar.

**A ordem dos pares precisa ser idêntica entre os inputs?**
Não obrigatoriamente — o alinhamento reordena por chave simétrica. Mas o
**conjunto** de pares tem que ser o mesmo.

**O que é o anchor?**
O primeiro input. Define a ordem dos pares e doa o `raw_score` escalar do output.

---

## 12. Glossário

- **fusão score-level** — combinar sistemas no nível do score (não do log-LR),
  aprendendo pesos via LogReg multivariado.
- **anchor** — o primeiro input; define a ordem e doa o `raw_score`.
- **chave simétrica do par** — `frozenset({probe_hash, reference_hash})`.
- **vetor de scores / matriz de features** — a matriz $(N_{\mathrm{pairs}}, D)$
  com uma coluna por sistema.
- **procedência** — a ordem das colunas e os IDs dos sistemas que as produziram.

---

## 13. Referências

- Brümmer, N., & du Preez, J. (2006). *Application-independent evaluation of
  speaker detection.* Computer Speech & Language. (Fusão score-level via LogReg.)
- Referências internas do FaceAudit:
  - `docs/calibracao_faceaudit.md` — calibrador multivariado que consome a matriz.
  - `docs/interacao_score_raridade_faceaudit.md` — produz a feature de interação.
  - Docstring de `score_vectorization_instruction_executor.py`.
  - `CLAUDE.md` (seção "Arquitetura LR declarativa") — alternativa V2.14
    `LR_ARCHITECTURE_VECTORIZATION` com `FIT_PER_FOLD` para arquiteturas LR
    canônicas; ver também `faceaudit/services/architecture_vectorizer_
    transformer.py` e `tests/test_per_fold_transformer.py`.

---

## Apêndice A — `ScoreVectorizationInstruction`

Arquivo: `faceaudit/models/instructions/score_vectorization_instruction.py`.

| Campo | Tipo | Default | Observação |
|---|---|---|---|
| `input_a_resource_id` | `Optional[str]` | `None` | obrigatório; anchor |
| `input_b_resource_id` | `Optional[str]` | `None` | obrigatório; distinto de A |
| `extra_input_resource_ids` | `List[str]` | `[]` | sistemas 3+ |
| `output_resource_id` | `Optional[str]` | `None` | obrigatório; saída |
| `output_dataset_name` | `Optional[str]` | `None` | — |
| `require_pair_match` | `bool` | `True` | legado; sem efeito prático |

Helper canônico: `all_input_resource_ids()` agrega A, B e extras (filtra vazios,
desduplica preservando ordem).

---

## Apêndice B — Chaves de metadata da saída

| Chave | Conteúdo |
|---|---|
| `is_score_vector` | `True` |
| `score_vector_matrix` | matriz numpy $(N_{\mathrm{pairs}}, D)$ |
| `score_vector_source_resources` | IDs dos inputs (ordem das colunas) |
| `score_vector_n_features` | $D$ |
| `score_vector_model_ids` | id do modelo de cada coluna (`None` se legado) |

Demais chaves do anchor são preservadas. O `raw_score` de cada sample é o do
anchor.

---

## Apêndice C — Checklist de validação operacional

- [ ] Todos os inputs descrevem o **mesmo conjunto de pares**?
- [ ] Há pelo menos 2 sistemas distintos (A ≠ B, sem duplicatas)?
- [ ] O anchor (input A) é o sistema cujo `raw_score` deve sobreviver?
- [ ] As escalas das colunas são compatíveis, ou o calibrador usa
      `standardize_features=True`?
- [ ] A procedência das colunas está documentada para o laudo?
- [ ] O modelo vetorizado foi comparado (A/B) contra o single-system no holdout?
- [ ] Há massa de calibração suficiente para o número de features (sem overfit)?
