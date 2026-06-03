# Modelo de tipicidade k-NN no FaceAudit (`KNN_TYPICALITY_MODEL_TRAINING`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `KNN_TYPICALITY_MODEL_TRAINING` — o modelo de
> tipicidade não-paramétrico baseado em k-NN esférico. Objetivo,
> fundamentação, as três métricas, configurações, fórmulas e exemplos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** repositório FaceAudit no commit `55c3d8a`
> (branch `claude/read-claude-md-SgzIV`). Campos, fórmulas e enums conferidos
> em `faceaudit/services/knn_typicality_model.py`,
> `faceaudit/executors/instructions/knn_typicality_model_training_instruction_executor.py`,
> `faceaudit/models/instructions/knn_typicality_model_training_instruction.py`,
> `faceaudit/models/knn_typicality_info.py`,
> `faceaudit/enums/typicality_reference_mode.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a tipicidade se encaixa no FaceAudit](#1-onde-a-tipicidade-se-encaixa-no-faceaudit)
- [2. Fundamentação — por que k-NN e não paramétrico](#2-fundamentação--por-que-k-nn-e-não-paramétrico)
- [3. As três métricas de tipicidade](#3-as-três-métricas-de-tipicidade)
- [4. Modo de referência — PER_IDENTITY vs PER_FACE](#4-modo-de-referência--per_identity-vs-per_face)
- [5. O treino passo a passo](#5-o-treino-passo-a-passo)
- [6. A inferência passo a passo](#6-a-inferência-passo-a-passo)
- [7. Configurações](#7-configurações)
- [8. Validações e guard-rails](#8-validações-e-guard-rails)
- [9. Persistência e versionamento](#9-persistência-e-versionamento)
- [10. Exemplo numérico](#10-exemplo-numérico)
- [11. Boas práticas resumidas](#11-boas-práticas-resumidas)
- [12. Perguntas frequentes](#12-perguntas-frequentes)
- [13. Glossário](#13-glossário)
- [14. Referências](#14-referências)
- [Apêndice A — Fórmulas de referência](#apêndice-a--fórmulas-de-referência)
- [Apêndice B — KNNTypicalityModelTrainingInstruction](#apêndice-b--knntypicalitymodeltraininginstruction)
- [Apêndice C — KNNTypicalityInfo](#apêndice-c--knntypicalityinfo)
- [Apêndice D — Checklist de validação operacional](#apêndice-d--checklist-de-validação-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre a instrução
`KNN_TYPICALITY_MODEL_TRAINING`: o que ela treina, as três métricas que o
modelo produz, suas configurações, as fórmulas e a fundamentação forense.

### 0.2 Escopo

**Incluído:** o `KNNTypicalityModel` (k-NN esférico não-paramétrico), as
métricas `mean_knn_similarity`, `typicality_percentile` e `log_density_knn`, os
modos de referência, o treino (CDF leave-one-out, corte OOD), a inferência
(tratamento de self-match) e a persistência.

**Fora do escopo:**

- a **aplicação** do modelo a pares (`TYPICALITY_PAIR_SCORING`) — ver
  `docs/tipicidade_pareada_faceaudit.md`;
- a **combinação** da tipicidade com similaridade
  (`SCORE_INTERACTION_FEATURE`, `SCORE_VECTORIZATION`) — ver
  `docs/interacao_score_raridade_faceaudit.md` e
  `docs/vetorizacao_scores_faceaudit.md`;
- o modelo de tipicidade **paramétrico** MoVMF (`TypicalityModel`).

### 0.3 Documentos complementares

| Documento | Relação |
|---|---|
| `docs/tipicidade_knn_faceaudit.md` (este) | treina o modelo de tipicidade |
| `docs/tipicidade_pareada_faceaudit.md` | aplica o modelo a pares de faces |
| `docs/interacao_score_raridade_faceaudit.md` | usa a tipicidade para modular a similaridade |
| `docs/vetorizacao_scores_faceaudit.md` | empilha tipicidade e similaridade para o calibrador |

---

## 1. Onde a tipicidade se encaixa no FaceAudit

A tipicidade responde a uma pergunta diferente da similaridade. A similaridade
mede *quão parecidas* são duas faces. A tipicidade mede *quão comum ou rara* é
uma face em relação a uma população de referência.

```text
dataset de referência (representativo da população relevante)
  ↓
KNN_TYPICALITY_MODEL_TRAINING   ← ESTE DOCUMENTO
  ↓
modelo de tipicidade treinado (TRAINED_TYPICALITY_MODEL)
  ↓
TYPICALITY_PAIR_SCORING  (aplica o modelo a cada face de cada par)
  ↓
tipicidade pareada → SCORE_INTERACTION_FEATURE / SCORE_VECTORIZATION
  ↓
calibrador multivariado → log-LR
```

O modelo treinado aqui é um **recurso de projeto** do tipo
`TRAINED_TYPICALITY_MODEL`, consumido a jusante pelo `TYPICALITY_PAIR_SCORING`.

### 1.1 A população de referência precisa ser representativa

O modelo mede tipicidade **em relação ao dataset de referência que recebe**. A
instrução **não estratifica por dentro** — quem garante que a referência
representa a população relevante ao caso (gênero, etnia, faixa etária) é quem
monta a estratégia. Um modelo treinado numa população enviesada produz
percentis enviesados.

---

## 2. Fundamentação — por que k-NN e não paramétrico

### 2.1 Tier 1 de defensibilidade forense

O FaceAudit tem dois modelos de tipicidade:

- **paramétrico** (`TypicalityModel`, MoVMF — mistura de von Mises-Fisher):
  assume uma família distribucional na esfera. Exige justificar em corte o
  número de componentes, as concentrações e a centralização — **Tier 2** de
  defensibilidade.
- **não-paramétrico** (`KNNTypicalityModel`, este documento): **não assume
  distribuição**. A tipicidade vira uma **estatística de ranking** — "esta face
  é mais típica que 73% da população de referência". É o **Tier 1**, o mais
  defensável.

### 2.2 Hubness vira sinal, não ruído

Em alta dimensionalidade, alguns pontos viram "hubs" (aparecem como vizinhos de
muitos outros). Modelos paramétricos que assumem simetria rotacional são
distorcidos por esse fenômeno. No k-NN, **hubness é exatamente o sinal**: uma
face numa região densa tem `mean_knn_similarity` alta — ela é, de fato, típica.
Não há suposição de simetria a violar, nem inversão de Bessel/Newton a resolver.

### 2.3 O custo

O "modelo" carrega a **referência inteira** em memória/disco. A inferência é
$O(N \cdot D)$ por query. Para uso forense exato e auditável, é o preço justo.
O modo `PER_IDENTITY` (padrão) reduz a referência a um ponto por pessoa,
deixando o modelo compacto sem perda de poder.

---

## 3. As três métricas de tipicidade

Para cada face avaliada, o modelo produz três escalares complementares
(empacotados em `KNNTypicalityInfo`). Seja $x$ o embedding L2-normalizado da
query e $y_{(1)}, \dots, y_{(k)}$ seus $k$ vizinhos mais próximos na referência.

### 3.1 `mean_knn_similarity` ($T_k$)

O cosseno médio aos $k$ vizinhos mais próximos:

$$
T_k(x) = \frac{1}{k}\sum_{i=1}^{k}\cos\!\bigl(x,\,y_{(i)}\bigr).
$$

Cresce com a densidade local — quanto mais próximos os vizinhos, mais típica a
face. Está em $[-1, 1]$. É intuitivo e barato, mas a **escala depende da
concentração da população** — não é diretamente comparável entre referências
diferentes.

### 3.2 `typicality_percentile`

O rank empírico de $T_k(x)$ na distribuição **leave-one-out** de $T_k$ sobre a
referência, normalizado em $[0, 100]$:

$$
\mathrm{percentile}(x) = 100 \cdot \frac{\mathrm{rank}\bigl(T_k(x)\bigr)}{N_{\mathrm{ref}}}.
$$

O rank usa a média entre as buscas `searchsorted` à esquerda e à direita
(tratamento de empates). É **comparável entre populações** (sempre $0..100$) e
**defensável em corte** como estatística não-paramétrica de ranking. É a métrica
recomendada para alimentar a vetorização com cosseno.

### 3.3 `log_density_knn`

O estimador de densidade de **Loftsgaarden-Quesenberry** na esfera $S^{d-1}$:

$$
\log\hat p(x) \approx \log k - \log N_{\mathrm{ref}} - (d-1)\,\log\theta_k,
$$

onde $\theta_k = \arccos\bigl(\cos(x, y_{(k)})\bigr)$ é o **raio angular** até o
$k$-ésimo vizinho (a menor calota esférica que contém os $k$ vizinhos). É a
única métrica em escala "real" de log-densidade — pode entrar diretamente como
$p(x \mid H_d)$ num LR generativo. Em $D=512$ pode ser muito negativa
(proporcional a $-(d-1)\log\theta_k$); para vetorizar com cosseno, considere
padronizar antes do LogReg.

### 3.4 Diagnóstico OOD

Além das três métricas, o modelo marca `is_in_distribution`: a face é
considerada *in-distribution* quando

$$
T_k(x) \ge \tau_{\mathrm{ood}},
$$

onde $\tau_{\mathrm{ood}}$ é o quantil `ood_quantile` da distribuição LOO de
$T_k$ (ver §5).

---

## 4. Modo de referência — PER_IDENTITY vs PER_FACE

O enum `TypicalityReferenceMode` controla **o que é um ponto** na referência.

| Modo | Um ponto da referência é… | Mede |
|---|---|---|
| `PER_IDENTITY` (default) | o **centroide** L2-renormalizado dos embeddings de cada pessoa | "quão típica é esta **pessoa** na população" |
| `PER_FACE` | cada **face** (foto) individual | "quão típica é esta **foto** no conjunto de fotos" |

### 4.1 `PER_IDENTITY` (default, defensável)

Agrupa por identidade e usa o centroide esférico (direção média na hiperesfera
unitária). A referência tem **um ponto por pessoa**, não por foto:

$$
c_{\mathrm{id}} = \frac{\sum_{j \in \mathrm{id}} x_j}{\bigl\lVert \sum_{j \in \mathrm{id}} x_j \bigr\rVert}.
$$

Isso torna a tipicidade **insensível ao desbalanceamento de fotos por pessoa**
(uma pessoa com 100 fotos não domina a referência). Identidades **singleton**
viram centroide igual à face única — **não são descartadas**. É o modo
defensável em corte: "esta pessoa está no 73º percentil da população".

### 4.2 `PER_FACE`

Cada foto é um ponto. Mede tipicidade da **foto** no conjunto de fotos.
Vulnerável a hubness intra-identidade (fotos da mesma pessoa aparecem como
vizinhas umas das outras, inflando a tipicidade) e a desbalanceamento. Útil para
diagnóstico de fotos atípicas dentro de identidades típicas.

> **Nota:** em `PER_IDENTITY`, todas as faces precisam de identidade definida
> (faces com `identity=None` causam erro).

---

## 5. O treino passo a passo

O método `fit()` do `KNNTypicalityModel`:

1. **Normaliza** os embeddings (L2) e valida o formato $(N, D)$.
2. **Constrói a referência**:
   - `PER_IDENTITY`: agrega por centroide (§4.1), contando singletons;
   - `PER_FACE`: usa as faces diretamente.
3. **Exige** $N_{\mathrm{ref}} \ge k + 1$ (necessário para a CDF leave-one-out).
4. **Computa a matriz de similaridade** $N_{\mathrm{ref}} \times N_{\mathrm{ref}}$
   ($\mathrm{ref} \cdot \mathrm{ref}^\top$), com a **diagonal = $-\infty$**
   (exclui o self).
5. Para cada ponto, pega os $k$ vizinhos mais próximos e calcula $T_k$ —
   formando a **distribuição leave-one-out** (LOO) de $T_k$ sobre a referência.
6. **Ordena** a distribuição LOO (vira a CDF empírica para o lookup de
   percentil) e calcula o **corte OOD**
   $\tau_{\mathrm{ood}} = \mathrm{quantile}(\mathrm{LOO},\ q)$, onde $q$ é o
   `ood_quantile`
   (ou $-\infty$ se `ood_quantile = 0`).
7. Congela o estado: referência (float32), CDF LOO ordenada, $\tau_{\mathrm{ood}}$,
   $k$, $D$, modo, contagens.

A coluna de embedding usada é a **definida no nível da estratégia**
(`context.embedding_column`) — não há default local, e a ausência é erro
explícito.

---

## 6. A inferência passo a passo

Para uma query $x$ (`get_typicality_info`):

1. Normaliza $x$ (L2).
2. Calcula os cossenos a toda a referência.
3. **Tratamento de self-match**: pega o top-$(k+1)$ e, se o maior cosseno for
   $> 1 - 10^{-6}$ (a query é um ponto da referência), **descarta** esse
   $\approx 1{,}0$ e usa os $k$ seguintes. Isso mantém consistência com a CDF
   LOO usada no treino.
4. Calcula $T_k$ (média dos top-$k$), $\theta_k = \arccos(\text{menor cosseno do top-}k)$,
   e a `log_density_knn`.
5. Faz o lookup do percentil na CDF LOO (`searchsorted`, mid-rank para empates).
6. Marca `is_in_distribution` comparando $T_k$ com $\tau_{\mathrm{ood}}$.

O modo batch (`get_typicality_infos`) faz isso por face, com
`include_top_k=False` por padrão (não guarda os 50 cossenos por face — em
scoring de milhões de pares isso explodiria a memória).

---

## 7. Configurações

Campos do `KNNTypicalityModelTrainingInstruction`:

| Campo | Tipo | Default | O que faz |
|---|---|---|---|
| `input_dataset_resource_id` | str | — | dataset de referência (deve ser representativo) |
| `k` | int | `50` | nº de vizinhos. Heurística defensável: $k \approx \sqrt{N}$. Deve ser $\ge 1$ e $< N_{\mathrm{ref}}$ |
| `ood_quantile` | float | `0.01` | quantil de $T_k$ usado como corte OOD; `0` desativa |
| `reference_mode` | `TypicalityReferenceMode` | `PER_IDENTITY` | centroide por pessoa (default) ou por face |
| `trained_typicality_model_resource_id` | str | — | id do recurso de saída (`TRAINED_TYPICALITY_MODEL`) |

A **coluna de embedding** não é campo da instrução — vem do nível da estratégia
(`context.embedding_column`).

### 7.1 Escolha de `k`

A heurística $k \approx \sqrt{N}$ equilibra viés e variância: $k$ pequeno torna
$T_k$ ruidoso; $k$ grande suaviza demais e dilui a noção de vizinhança local.
Em `PER_IDENTITY`, $N$ é o número de identidades (não de fotos).

---

## 8. Validações e guard-rails

Defaults persistentes populados pelo DTO:

| Regra | Critério |
|---|---|
| `MIN_SAMPLES` | pelo menos $k+1$ amostras (requisito do fit LOO); warning abaixo de 1000 |
| `MAX_SAMPLES_WARNING` | avisa se referência $> 50\text{k}$ (custo $O(N^2)$ no fit) |

Em runtime, o `fit` exige $N_{\mathrm{ref}} \ge k+1$ e rejeita centroides ou
$T_k$ não-finitos (embeddings degenerados). Na validação da instrução: `k \ge 1`,
`ood_quantile ∈ [0, 0.5)`, dataset presente e extraível, e o recurso de saída
definido.

> **Nota sobre o `MIN_SAMPLES`**: em `PER_IDENTITY` o $k$ é checado contra o
> número de **faces** (não de identidades) na regra default — é conservador
> (geralmente há menos identidades que faces). A checagem real
> $N_{\mathrm{ref}} \ge k+1$ acontece no `fit`.

---

## 9. Persistência e versionamento

O modelo é serializado via `to_dict`/`from_dict` (`format_version = 2`),
guardando: hiperparâmetros ($k$, `ood_quantile`, `reference_mode`, `eps`) e o
estado aprendido (referência L2-normalizada, CDF LOO ordenada,
$\tau_{\mathrm{ood}}$, $D$, contagens).

**Retrocompat v1 → v2**: payloads antigos não tinham `reference_mode` —
assumidos como `PER_FACE` (o comportamento implícito antes do toggle existir); o
campo removido `population_label` é descartado. O `from_dict` valida que a
referência está L2-normalizada (norma $\approx 1$) e que `reference` e a CDF LOO
têm o mesmo número de pontos.

O `summary()` expõe auditoria: modo, $k$, contagens, $\tau_{\mathrm{ood}}$ e
estatísticas da distribuição LOO de $T_k$ (mín/máx/mediana/média/desvio).

---

## 10. Exemplo numérico

Referência `PER_IDENTITY` com $N_{\mathrm{ref}} = 10\,000$ identidades,
$k = 100$ ($\approx\sqrt{10\,000}$). Duas faces questionadas:

| Face | $T_k$ (cosseno médio aos 100 vizinhos) | rank na CDF LOO | `typicality_percentile` |
|---|---|---|---|
| A (comum) | 0,42 | 7300º | 73 |
| B (rara) | 0,21 | 500º | 5 |

A face A tem vizinhos próximos (região densa) → percentil 73 (típica). A face B
está numa região esparsa → percentil 5 (rara). Em
`SCORE_INTERACTION_FEATURE` com `invert_rarity=True`, a raridade da face B seria
$1 - 0{,}05 = 0{,}95$ (alta) e a da face A, $1 - 0{,}73 = 0{,}27$ (baixa) — a
face rara amplifica a evidência de uma similaridade observada (ver
`docs/interacao_score_raridade_faceaudit.md`).

---

## 11. Boas práticas resumidas

- **Referência representativa**: estratifique a população relevante *antes* de
  treinar — a instrução não estratifica por dentro.
- **`PER_IDENTITY` é o default forense**: robusto a desbalanceamento de fotos;
  defensável como "percentil da pessoa na população".
- **`k ≈ √N`**: ponto de partida defensável; ajuste com cautela.
- **`TYPICALITY_PERCENTILE` para vetorizar**: escala consistente $[0,1]$,
  comparável entre populações.
- **`LOG_DENSITY_KNN` para LR generativo**: padronize antes do LogReg se for
  empilhar com cosseno.
- **Cuidado com referências grandes**: custo $O(N^2)$ no fit; `PER_IDENTITY`
  reduz $N$.

---

## 12. Perguntas frequentes

**A tipicidade é um log-LR?**
Não. É uma estatística de ranking (percentil) ou de densidade. Ela só vira
evidência depois de modular uma similaridade (ver `SCORE_INTERACTION_FEATURE`).

**Por que não usar o MoVMF paramétrico?**
O k-NN não assume distribuição (Tier 1 de defensibilidade), trata hubness como
sinal e dispensa justificar família distribucional em corte. O MoVMF é mais
compacto mas exige defender mais suposições.

**O que acontece com identidades singleton em `PER_IDENTITY`?**
Viram centroide igual à face única — não são descartadas. Toda identidade pesa
igual no espaço de referência.

**Por que descartar o vizinho com cosseno ≈ 1 na inferência?**
Porque significa que a query *é* um ponto da referência (self-match). Descartá-lo
mantém a inferência consistente com a CDF leave-one-out usada no treino.

**Posso treinar com `PER_FACE` e aplicar como percentil?**
Pode, mas a tipicidade medirá "esta foto é típica no conjunto de fotos", não "esta
pessoa é típica na população" — e fica vulnerável a hubness intra-identidade.

---

## 13. Glossário

- **tipicidade** — quão comum é uma face na população de referência.
- **raridade** — o complemento da tipicidade.
- $T_k$ / `mean_knn_similarity` — cosseno médio aos $k$ vizinhos mais próximos.
- **percentil de tipicidade** — rank empírico de $T_k$ na população, $0..100$.
- **log-densidade k-NN** — estimador de Loftsgaarden-Quesenberry na esfera.
- $\theta_k$ — raio angular até o $k$-ésimo vizinho.
- **LOO** — leave-one-out; a distribuição de $T_k$ da referência contra si mesma.
- **OOD** — out-of-distribution; $T_k$ abaixo do corte $\tau_{\mathrm{ood}}$.
- **centroide esférico** — direção média (soma L2-renormalizada) dos embeddings
  de uma identidade.
- **hubness** — fenômeno em alta dimensão onde alguns pontos viram vizinhos de
  muitos outros.

---

## 14. Referências

- Loftsgaarden, D. O., & Quesenberry, C. P. (1965). *A nonparametric estimate
  of a multivariate density function.* Annals of Mathematical Statistics.
- Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010). *Hubs in space:
  popular nearest neighbors in high-dimensional data.* JMLR.
- Referências internas do FaceAudit:
  - `docs/tipicidade_pareada_faceaudit.md` — aplicação do modelo a pares.
  - `docs/interacao_score_raridade_faceaudit.md` — uso da raridade.
  - Docstrings de `knn_typicality_model.py` e `typicality_score_type.py`.

---

## Apêndice A — Fórmulas de referência

**Cosseno médio aos $k$ vizinhos:**

$$
T_k(x) = \frac{1}{k}\sum_{i=1}^{k}\cos\!\bigl(x,\,y_{(i)}\bigr).
$$

**Percentil de tipicidade:**

$$
\mathrm{percentile}(x) = 100\cdot\frac{\mathrm{rank}\bigl(T_k(x)\bigr)}{N_{\mathrm{ref}}}.
$$

**Log-densidade k-NN (Loftsgaarden-Quesenberry esférico):**

$$
\log\hat p(x) \approx \log k - \log N_{\mathrm{ref}} - (d-1)\,\log\theta_k,
\qquad \theta_k = \arccos\bigl(\cos(x, y_{(k)})\bigr).
$$

**Centroide esférico por identidade (`PER_IDENTITY`):**

$$
c_{\mathrm{id}} = \frac{\sum_{j\in\mathrm{id}} x_j}{\bigl\lVert\sum_{j\in\mathrm{id}} x_j\bigr\rVert}.
$$

**Corte OOD:**

$$
\tau_{\mathrm{ood}} = \mathrm{quantile}\bigl(\{T_k^{\mathrm{LOO}}\},\ q\bigr),
\qquad
T_k(x) \ge \tau_{\mathrm{ood}} \ \Rightarrow\ \text{in-distribution},
$$

com $q$ = `ood_quantile`.

---

## Apêndice B — `KNNTypicalityModelTrainingInstruction`

Arquivo: `faceaudit/models/instructions/knn_typicality_model_training_instruction.py`.

| Campo | Tipo | Default | Restrição |
|---|---|---|---|
| `input_dataset_resource_id` | `Optional[str]` | `None` | obrigatório; `DATASET` |
| `k` | `int` | `50` | $\ge 1$ e $< N_{\mathrm{ref}}$ |
| `ood_quantile` | `float` | `0.01` | $[0, 0.5)$ |
| `reference_mode` | `TypicalityReferenceMode` | `PER_IDENTITY` | `PER_IDENTITY` / `PER_FACE` |
| `trained_typicality_model_resource_id` | `Optional[str]` | `None` | obrigatório; saída |

**`TypicalityReferenceMode`** (`faceaudit/enums/typicality_reference_mode.py`):
`PER_FACE`, `PER_IDENTITY` (default). `from_str` defaulta `PER_IDENTITY`.

---

## Apêndice C — `KNNTypicalityInfo`

Arquivo: `faceaudit/models/knn_typicality_info.py`. Resultado por face.

| Campo | Tipo | Significado |
|---|---|---|
| `mean_knn_similarity` | `float` | $T_k$, em $[-1,1]$ |
| `typicality_percentile` | `float` | rank empírico, $0..100$ |
| `log_density_knn` | `float` | log-densidade esférica |
| `angular_radius_to_kth_nn` | `float` | $\theta_k$ (radianos) |
| `is_in_distribution` | `bool` | $T_k \ge \tau_{\mathrm{ood}}$ |
| `knn_similarities` | `Optional[List[float]]` | top-$k$ cossenos (None em batch) |
| `model_name` / `embedding_dim` / `k` / `sample_count` | — | identificação do modelo |

---

## Apêndice D — Checklist de validação operacional

- [ ] A população de referência representa a população relevante ao caso?
- [ ] $N_{\mathrm{ref}} \ge k+1$ e idealmente $\ge 1000$ pontos?
- [ ] $k$ foi escolhido por uma heurística defensável ($\approx\sqrt{N}$)?
- [ ] O modo é `PER_IDENTITY` (default forense), salvo justificativa para `PER_FACE`?
- [ ] Em `PER_IDENTITY`, todas as faces têm identidade definida?
- [ ] A métrica escolhida a jusante é coerente (`TYPICALITY_PERCENTILE` para
      vetorizar; `LOG_DENSITY_KNN` padronizado para LR generativo)?
- [ ] A referência não é grande demais para o custo $O(N^2)$ do fit?
- [ ] O `embedding_column` da estratégia está definido e correto?
