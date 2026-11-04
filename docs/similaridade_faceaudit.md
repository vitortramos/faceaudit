# Modelos de similaridade no FaceAudit

> **Status:** documento técnico de referência interna.
> **Escopo:** modelos de similaridade que produzem o *score bruto* de uma
> comparação facial — cosseno e PLDA (todas as variantes), com foco em PLDA.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** código atual do repositório FaceAudit. Defaults,
> fórmulas e enums conferidos em `faceaudit/models/similarity/`,
> `faceaudit/services/two_covariance_plda_service.py`,
> `faceaudit/services/shrinkage_two_covariance_plda_service.py`,
> `faceaudit/services/cross_condition_two_covariance_plda_service.py`,
> `faceaudit/enums/`, `faceaudit/protocols/similarity_model_protocol.py`,
> `faceaudit/executors/instructions/similarity_model_training_instruction_executor.py`,
> `faceaudit/executors/instructions/similarity_scoring_instruction_executor.py`,
> `faceaudit/services/strategy_inference_service.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde o modelo de similaridade se encaixa no FaceAudit](#1-onde-o-modelo-de-similaridade-se-encaixa-no-faceaudit)
- [2. Conceitos-base e notação](#2-conceitos-base-e-notação)
- [3. Famílias de modelos de similaridade](#3-famílias-de-modelos-de-similaridade)
- [4. Similaridade cosseno](#4-similaridade-cosseno)
- [5. PLDA — fundamentos](#5-plda--fundamentos)
- [6. Variante SIMPLE (SpeechBrain)](#6-variante-simple-speechbrain)
- [7. Variante TWO_COVARIANCE (Brümmer & de Villiers)](#7-variante-two_covariance-brümmer--de-villiers)
- [8. Variante SHRINKAGE_TWO_COVARIANCE](#8-variante-shrinkage_two_covariance)
- [9. Modo cross-condition](#9-modo-cross-condition)
- [10. Scoring: do LLR bruto ao par](#10-scoring-do-llr-bruto-ao-par)
- [11. Pré-requisitos de dados e validação](#11-pré-requisitos-de-dados-e-validação)
- [12. Integração no pipeline do FaceAudit](#12-integração-no-pipeline-do-faceaudit)
- [13. Como escolher a variante](#13-como-escolher-a-variante)
- [14. Relação com a calibração e o double-counting de tipicidade](#14-relação-com-a-calibração-e-o-double-counting-de-tipicidade)
- [15. Persistência, versionamento e rastreabilidade](#15-persistência-versionamento-e-rastreabilidade)
- [16. Boas práticas resumidas](#16-boas-práticas-resumidas)
- [17. Perguntas frequentes](#17-perguntas-frequentes)
- [18. Glossário](#18-glossário)
- [19. Referências](#19-referências)
- [Apêndice A — Fórmulas de referência](#apêndice-a--fórmulas-de-referência)
- [Apêndice B — PLDASimilarityModelConfig](#apêndice-b--pldasimilaritymodelconfig)
- [Apêndice C — Enums](#apêndice-c--enums)
- [Apêndice D — TwoCovariancePLDAState](#apêndice-d--twocovariancepldastate)
- [Apêndice E — Regras de validação aplicáveis](#apêndice-e--regras-de-validação-aplicáveis)
- [Apêndice F — Checklist de revisão documental](#apêndice-f--checklist-de-revisão-documental)
- [Apêndice G — Checklist de validação operacional](#apêndice-g--checklist-de-validação-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica geral sobre os **modelos de similaridade** do
FaceAudit — a etapa que recebe pares de embeddings faciais e produz o *score
bruto* de comparação. Reúne, num único arquivo, os fundamentos matemáticos, as
implementações disponíveis (com ênfase no PLDA e suas variantes), as
configurações operacionais e os controles metodológicos necessários para uso
defensável em laudo.

### 0.2 Público-alvo

Peritos, desenvolvedores e revisores técnicos. O texto é didático na exposição
inicial de cada tópico, formal na formulação matemática e exaustivo nas
tabelas de configuração.

### 0.3 Escopo

**Incluído:**

- o contrato `SimilarityModelProtocol` e o que significa um "score bruto";
- o modelo cosseno (stateless);
- o PLDA em todas as variantes disponíveis: `SIMPLE` (SpeechBrain),
  `TWO_COVARIANCE` (Brümmer & de Villiers 2010) e
  `SHRINKAGE_TWO_COVARIANCE` (Ledoit-Wolf ou α manual sobre W);
- o modo **cross-condition** (treino com dataset secundário);
- o scoring (par, vetores, matriz) e suas propriedades;
- pré-requisitos de dados, regras de validação e dimensionalidade;
- integração com as instruções `SIMILARITY_MODEL_TRAINING` e
  `SIMILARITY_SCORING`, e com a inferência forense;
- persistência, versionamento e rastreabilidade.

**Fora do escopo:**

- engenharia de extração de embeddings (AdaFace, ForensicFace) — ver
  `faceaudit/services/ff_service.py`;
- pipelines de transformação (CORAL, Whitening, LDA, Centering, L2) — ver
  `faceaudit/transformators/` e `faceaudit/pipelines/`;
- **conversão do score bruto em log-LR (calibração)** — coberta em
  `docs/calibracao_faceaudit.md`;
- fusão multivariada score × raridade (`SCORE_INTERACTION_FEATURE`,
  modelos de tipicidade) — tratada apenas no que toca ao double-counting
  com PLDA (§14);
- formulação específica das proposições de cada exame pericial.

### 0.4 Documentos complementares

Este documento é uma **referência guarda-chuva** sobre similaridade. Para
profundidade adjacente, consulte:

| Documento | Escopo | Relação com este |
|---|---|---|
| `docs/similaridade_faceaudit.md` (este arquivo) | modelos de similaridade, PLDA e variantes, scoring, validação, integração | guarda-chuva de similaridade |
| `docs/calibracao_faceaudit.md` | conversão do score bruto em log-LR, métricas, seleção e validação de calibradores | etapa seguinte do fluxo |
| Docstrings dos serviços PLDA (`two_covariance_plda_service.py`, `shrinkage_two_covariance_plda_service.py`, `cross_condition_two_covariance_plda_service.py`) | especificação matemática e decisões de normalização | fonte primária das fórmulas |

O score bruto produzido aqui **não é evidência calibrada**; a transformação em
log-LR é a etapa imediatamente seguinte e está documentada em
`docs/calibracao_faceaudit.md`.

---

## 1. Onde o modelo de similaridade se encaixa no FaceAudit

### 1.1 Fluxo conceitual

O modelo de similaridade é a etapa que recebe **dois embeddings faciais** e
produz **um número** que mede o quão compatíveis eles são com a hipótese de
mesma origem.

```text
imagem
  ↓
embedding facial (AdaFace / ForensicFace)
  ↓
pipeline de transformação (Centering, CORAL, Whitening, LDA, L2…)  ← opcional
  ↓
modelo de similaridade (cosseno | PLDA)   ← ESTE DOCUMENTO
  ↓
score bruto contínuo
  ↓
calibrador
  ↓
log-LR contínuo
  ↓
avaliação, validação e relato
```

### 1.2 Score bruto não é evidência calibrada

O score produzido por um modelo de similaridade preserva a **ordenação
relativa** entre pares — pares mais compatíveis recebem scores maiores. Sua
magnitude, porém, não tem necessariamente interpretação direta como força de
evidência em escala forense.

Isso vale inclusive para o PLDA: embora o PLDA emita um **log-likelihood ratio
bruto** (LLR bruto), esse valor ainda **não é** o log-LR reportável. Ele
depende do conjunto de treino, das suposições gaussianas do modelo e da
representatividade dos dados. A conversão em log-LR defensável é
responsabilidade do calibrador (ver `docs/calibracao_faceaudit.md`).

### 1.3 Similaridade e calibração são responsabilidades separadas

O FaceAudit separa explicitamente:

- **`SimilarityModelProtocol`** — produz scores brutos (`fit`, `score_pair`,
  `score_vectors`, `score_matrix`);
- **`CalibrationModelProtocol`** — converte scores brutos em log-LRs
  (`fit`, `predict_proba`, `calibrate_scores`).

A conexão entre os dois é o `SimilarityScoreDataset`: o modelo de similaridade
preenche o `raw_score` de cada par; o calibrador, numa etapa posterior,
preenche o `metadata["log_lr"]`. Esta fronteira é deliberada e não deve ser
borrada — um modelo de similaridade nunca emite log-LR calibrado diretamente.

### 1.4 O modelo de similaridade não cria informação ausente do embedding

Um modelo de similaridade opera **sobre o espaço de embeddings** que recebe.
Se o pipeline de transformação a montante descartou informação (por exemplo,
truncou dimensões via LDA com poucas classes), nenhum modelo de similaridade
recupera o que foi perdido. Igualmente, o PLDA modela *covariâncias* do espaço
recebido; ele não tem acesso à imagem original nem a metadados de qualidade.

Quando tipicidade ou raridade da face em relação à população forem relevantes,
isso é incorporado por modelagem específica (modelos de tipicidade,
`SCORE_INTERACTION_FEATURE`), com a ressalva de double-counting da §14 — não
por refinamento do score de similaridade.

---

## 2. Conceitos-base e notação

### 2.1 Embedding e dimensionalidade

Um **embedding** é um vetor $x \in \mathbb{R}^D$ que representa uma face. $D$ é
a dimensionalidade (ex.: 512 para AdaFace cru, ou menor após PCA/LDA no
pipeline). Um `EmbeddingDataset` agrega vários `EmbeddingInfo`, cada um com um
`embedding` e uma `identity` (rótulo da pessoa) opcional.

### 2.2 Identidade, genuíno e impostor

Toda comparação ocorre sob duas proposições mutuamente excludentes:

$$
H_p:\text{as imagens comparadas provêm do mesmo indivíduo,}
$$
$$
H_d:\text{as imagens comparadas provêm de indivíduos diferentes.}
$$

| Rótulo interno | Significado |
|---|---|
| `1` | par genuíno (mesma identidade); hipótese $H_p$ |
| `0` | par impostor (identidades diferentes); hipótese $H_d$ |
| `-1` / `None` | ground truth desconhecido (uso em casework) |

### 2.3 Score bruto e LLR bruto

Para um par $(x_1, x_2)$, o modelo de similaridade produz um escalar
$s = \mathrm{score}(x_1, x_2)$.

- No **cosseno**, $s$ é a similaridade cosseno, em $[-1, 1]$.
- No **PLDA**, $s$ é um **log-likelihood ratio bruto** — o log da razão entre
  a verossimilhança do par sob "mesma classe" e sob "classes distintas",
  segundo o modelo gaussiano treinado. É chamado neste documento de **LLR
  bruto** para distingui-lo do log-LR calibrado.

### 2.4 Simetria

Tanto o cosseno quanto o PLDA de duas covariâncias são **simétricos**:
$\mathrm{score}(x_1, x_2) = \mathrm{score}(x_2, x_1)$. No PLDA two-covariance
isso decorre da fórmula de scoring (o termo cruzado usa uma matriz $P$
simétrica aplicada bilinearmente). O modo cross-condition assume um scorer
**homocedástico** (uma única $W$ combinada); o caso heterocedástico
$W_{\text{probe}} \ne W_{\text{reference}}$ está fora do escopo da V1 e
quebraria a simetria.

### 2.5 Matriz de embeddings

`score_matrix(P, R)` recebe duas matrizes — $P \in \mathbb{R}^{n_p \times D}$
(probes) e $R \in \mathbb{R}^{n_r \times D}$ (referências) — e devolve a
matriz de scores $S \in \mathbb{R}^{n_p \times n_r}$. É o caminho vetorizado
usado em produção; `score_pair`/`score_vectors` são o caminho unitário.

---

## 3. Famílias de modelos de similaridade

O FaceAudit suporta duas famílias, identificadas pelo enum
`SimilarityModelType`:

| Tipo | Classe | Treina? | Estado | Score |
|---|---|---|---|---|
| `COSINE` | `CosineSimilarityModel` | não (stateless) | só metadados (`input_dimension`, `n_samples_seen`) | similaridade cosseno em $[-1,1]$ |
| `PLDA` | `PLDASimilarityModel` | sim | covariâncias / subspace treinados | LLR bruto |

Ambas implementam estruturalmente `SimilarityModelProtocol`. A escolha é
explícita no DTO da instrução (`similarity_model_type`), evitando inferência
implícita por presença/ausência de config.

O PLDA, por sua vez, tem **três variantes matemáticas** internas, selecionadas
por `PLDAVariant`:

| Variante | Backend | Treino | Parâmetros próprios |
|---|---|---|---|
| `SIMPLE` | SpeechBrain (Ioffe-style) | EM iterativo | `em_iterations`, `plda_mode`, `latent_dim` |
| `TWO_COVARIANCE` | numpy puro (Brümmer 2010) | forma fechada | `regularization` |
| `SHRINKAGE_TWO_COVARIANCE` | numpy puro + shrinkage de W | forma fechada | `shrinkage_mode`, `shrinkage_intensity`, `regularization` |

As variantes `TWO_COVARIANCE` e `SHRINKAGE_TWO_COVARIANCE` (referidas em
conjunto como *two-covariance*) compartilham o scoring e o estado treinado
(`TwoCovariancePLDAState`); diferem apenas na estimação de $W$.

---

## 4. Similaridade cosseno

### 4.1 Definição

Para dois vetores $a, b \in \mathbb{R}^D$:

$$
\mathrm{cos}(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}\in[-1,1].
$$

### 4.2 Características no FaceAudit

- **Stateless.** `CosineSimilarityModel` nasce com `is_fitted=True`;
  `requires_training` é `False`. O `fit()` existe apenas para registrar
  metadados (`input_dimension`, `n_samples_seen`) de rastreabilidade — o
  scoring funciona sem ele.
- **Sem `input_dimension` obrigatória.** Como não há covariâncias treinadas, a
  regra `EmbeddingDimMatchesModelRule` passa silenciosamente para o cosseno
  (modelos sem `input_dimension` declarada não disparam o check).
- **Norma zero é erro.** Vetores de norma nula levantam `ValueError` (não há
  cosseno definido).
- `score_matrix` normaliza as linhas de $P$ e $R$ e retorna $P_n R_n^\top$.

### 4.3 Quando usar

O cosseno é o baseline: rápido, sem treino, defensável como medida de
similaridade direcional entre embeddings. É o ponto de partida natural antes
de investir num PLDA. Não modela a estrutura entre-classes/intra-classe da
população — toda a "inteligência forense" subsequente fica concentrada no
calibrador.

---

## 5. PLDA — fundamentos

### 5.1 O modelo generativo de duas covariâncias

O PLDA (Probabilistic Linear Discriminant Analysis) na formulação de
duas covariâncias supõe que cada embedding observado é a soma de um termo de
**classe** (a "identidade verdadeira") e um termo de **ruído** (variação
intra-classe: pose, iluminação, expressão, idade):

$$
y_c \sim \mathcal{N}(\mu, B), \qquad x \mid y_c \sim \mathcal{N}(y_c, W),
$$

onde:

- $\mu \in \mathbb{R}^D$ — média global;
- $B$ — covariância **entre-classes** ($D \times D$): o quanto as identidades
  diferem entre si na população. Codifica diretamente a noção de *tipicidade*
  (faces raras estão longe de $\mu$ na métrica de $B$);
- $W$ — covariância **intra-classe** ($D \times D$): o quanto amostras de uma
  mesma pessoa variam.

A força probatória de uma comparação cresce quando a diferença observada entre
os dois embeddings é pequena em relação a $W$ (compatível com mesma pessoa) e,
ao mesmo tempo, a posição no espaço é distintiva em relação a $B$ (a face não
é "média").

### 5.2 Por que PLDA e não só cosseno

O cosseno trata todas as direções do espaço igualmente. O PLDA aprende, a
partir de dados rotulados por identidade, **quais direções carregam
identidade** ($B$) e **quais carregam ruído de captura** ($W$), e pondera a
comparação de acordo. É a formulação padrão na literatura forense de
reconhecimento (toolkit FoCal, área de speaker verification) justamente por
produzir um LLR com interpretação estatística natural.

### 5.3 As três variantes em uma frase

- **SIMPLE** — aprende um *subspace* entre-classes de baixo rank ($F$) por EM;
  regulariza implicitamente por truncamento, mas perde informação fora do
  subspace.
- **TWO_COVARIANCE** — estima $B$ e $W$ **completas** $D \times D$ em forma
  fechada; sem perda por truncamento, mas exige dados suficientes para
  estimar matrizes cheias.
- **SHRINKAGE_TWO_COVARIANCE** — igual à anterior, mas encolhe $W$ na direção
  de um alvo bem-condicionado, estabilizando o modelo quando há poucas
  amostras por identidade ou $D$ alto.

### 5.4 Pré-processamento comum a todas as variantes (single-dataset)

Antes de treinar, `_extract_training_arrays` aplica:

1. converte para `float64`;
2. **remove `identity=None`** (sempre);
3. no caminho single-dataset, **descarta singletons** — identidades com apenas
   1 amostra — porque a variância intra-classe não é estimável a partir de uma
   única amostra. (No cross-condition, singletons são preservados — §9.)

Se restarem menos de 2 amostras, o treino levanta `ValueError`. Nas variantes
two-covariance o serviço de treino exige também pelo menos 2 classes com
$\ge 2$ amostras (`ValueError` caso contrário); na SIMPLE essa checagem não é
feita no modelo — quem a garante antes do treino é a regra
`MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES` (§11).

---

## 6. Variante SIMPLE (SpeechBrain)

### 6.1 Formulação

A variante `SIMPLE` usa o backend `speechbrain.processing.PLDA_LDA`. Ela
modela um **subspace entre-classes** $F \in \mathbb{R}^{D \times r}$ de rank
$r$ (o "espaço latente") mais uma covariância residual $\Sigma$, treinados por
**Expectation-Maximization** iterativo. Direções fora do subspace de rank $r$
são truncadas — o que regulariza implicitamente, ao custo de descartar
informação.

### 6.2 Escolha do rank ($r$)

O rank é decidido por `_decide_rank_f(D, C)`, com $C$ = nº de classes:

$$
r_{\max} = \max\!\bigl(1,\ \min(D,\ C-1)\bigr).
$$

- `PLDAMode.FULL_RANK` → $r = r_{\max}$ (maior rank permitido pelos dados);
- `PLDAMode.LOW_RANK` → $r = \max\!\bigl(1,\ \min(\ell,\ r_{\max})\bigr)$, onde
  $\ell$ é o `latent_dim` configurado.

Em `FULL_RANK`, o campo `latent_dim` é **ignorado em runtime** (a UI o exibe
como read-only para sinalizar isso).

### 6.3 Treino e scoring

```python
self.plda = PLDA(rank_f=rank_f, nb_iter=self.config.em_iterations)
self.plda.plda(train_stat, whiten=False)
```

O scoring usa `fast_PLDA_scoring(probe_stat, ref_stat, ndx, plda.mean,
plda.F, plda.Sigma)`. Tanto `score_vectors` (par único, via matrizes 1×1)
quanto `score_matrix` (lote) constroem `StatObject_SB`/`Ndx` com `trialmask`
toda verdadeira e leem `scoremat`.

### 6.4 Parâmetros e quando ajustar

| Parâmetro | Default | Efeito |
|---|---|---|
| `em_iterations` | `20` | mais iterações ⇒ convergência mais fina do EM (custo linear) |
| `plda_mode` | `LOW_RANK` | `FULL_RANK` preserva mais informação mas exige muitas identidades para estabilizar |
| `latent_dim` | `128` | limita o rank em `LOW_RANK`; menor ⇒ mais estável com poucos dados |

A variante SIMPLE é a mais robusta com **poucas identidades** ou domínio
ruidoso, porque o truncamento de rank é uma regularização forte. O preço é a
perda de informação fora do subspace.

---

## 7. Variante TWO_COVARIANCE (Brümmer & de Villiers)

### 7.1 Estimação em forma fechada

A variante `TWO_COVARIANCE` (`two_covariance_plda_service.fit_two_covariance`)
estima $\mu$, $B$, $W$ **sem EM**, a partir dos scatter matrices. Para classes
$c$ com $n_c \ge 2$ amostras:

$$
\mu = \frac{1}{N}\sum_i x_i,
$$
$$
S_w = \sum_c \sum_{i\in c}(x_i-\mu_c)(x_i-\mu_c)^\top,
\qquad
S_b = \sum_c n_c\,(\mu_c-\mu)(\mu_c-\mu)^\top,
$$
$$
W = \frac{S_w}{N}, \qquad B = \frac{S_b}{N}.
$$

São estimadores de máxima verossimilhança **enviesados** (divisão por $N$) —
convenção da literatura de PLDA. A alternativa não-enviesada introduz ajustes
pequenos que raramente mudam decisões forenses.

### 7.2 Scoring: kernels Q, P e a constante

O LLR bruto de um par $(x_1, x_2)$, com $z = x - \mu$, é:

$$
\log\mathrm{LR}=\tfrac{1}{2}\bigl(z_1^\top Q\,z_1 + z_2^\top Q\,z_2 + z_1^\top P\,z_2\bigr) + \mathrm{const},
$$

com os kernels pré-computados uma única vez em `_compute_scoring_kernels`:

$$
K=(W+B)^{-1},\quad M=(W+2B)^{-1},\quad N_{\!*}=W^{-1},
$$
$$
Q = K - \tfrac{1}{2}(M + N_{\!*}), \qquad P = N_{\!*} - M,
$$
$$
\mathrm{const} = \log\lvert W+B\rvert - \tfrac{1}{2}\log\lvert W+2B\rvert - \tfrac{1}{2}\log\lvert W\rvert.
$$

Os kernels $Q$, $P$ e `const` são cacheados em `fit()`/`load()`
(`self._two_cov_kernels`) e reusados por `score_matrix`, que os aplica
vetorizadamente via `einsum`. O caminho unitário (`score_vectors`, via
`score_two_covariance_pair`) recomputa os kernels a cada chamada — por isso,
em lote, use sempre `score_matrix`. A simetria
$\mathrm{LR}(x_1,x_2)=\mathrm{LR}(x_2,x_1)$ decorre de $P$ ser simétrica.

### 7.3 Regularização (Tikhonov)

O campo `regularization` ($\varepsilon \ge 0$) soma $\varepsilon I$ **a $B$ e a
$W$** antes das inversões, garantindo variância mínima $\varepsilon$ em todas
as direções e evitando singularidade:

$$
W \leftarrow W + \varepsilon I, \qquad B \leftarrow B + \varepsilon I.
$$

> **Nota:** essa é a única variante em que $\varepsilon$ é somado **também em
> $B$**. Na variante de shrinkage (§8) e no cross-condition (§9), a Tikhonov
> final é aplicada **somente em $W$** — para não injetar variância
> entre-classes artificial.

Recomendações de valor (tooltip da UI):

| Faixa de $\varepsilon$ | Uso |
|---|---|
| `0.0` (default) | só com dataset farto (~1000+ identidades × 10+ amostras, $D$ 128–256) |
| `1e-6` a `1e-4` | "leve" — estabiliza numericamente sem alterar log-LRs; default seguro |
| `1e-3` a `1e-2` | amacia decisão (log-LRs menos extremos); pode mascarar problemas reais do pipeline |
| `> 1e-1` | evitar — a identidade $\varepsilon I$ domina a covariância real |

Antes de subir $\varepsilon$: verificar duplicatas exatas, passos de pipeline
excessivos e coerência de $D$ com o nº de classes (uma LDA com 30 classes não
deve sair em 256D). O treino faz um **Cholesky check** de $W$; se $W$ não é
positiva definida, levanta `ValueError` sugerindo aumentar `regularization`.

### 7.4 Quando usar

`TWO_COVARIANCE` é o **padrão recomendado na literatura forense** quando há
dados suficientes para estimar matrizes cheias $D \times D$. Não trunca
informação (ao contrário do SIMPLE), tem treino determinístico em forma
fechada e produz um LLR de interpretação natural.

---

## 8. Variante SHRINKAGE_TWO_COVARIANCE

### 8.1 Motivação

Quando há poucas amostras por identidade ou $D$ é alto, a $W$ empírica fica
**mal-condicionada** (autovalores próximos de zero), o que distorce as
inversões do scoring. O *shrinkage* mistura $W$ empírica com um alvo
bem-condicionado, estabilizando-a.

### 8.2 Fórmula

O shrinkage é aplicado **apenas a $W$** (intra-classe), contra o alvo
*scaled identity* $F = \frac{\mathrm{tr}(W)}{D} I$:

$$
W_{\text{shrunk}} = (1-\alpha)\,W_{\text{emp}} + \alpha\,\frac{\mathrm{tr}(W)}{D}\,I,
\qquad \alpha \in [0,1].
$$

$B$ permanece **empírica** — alterá-la mexeria diretamente na noção de "quão
dispersa é a população de identidades", que é exatamente a quantidade que o
LLR mede. A Tikhonov final (`regularization`), se usada, também incide só em
$W$.

### 8.3 Como $\alpha$ é determinado (`CovarianceShrinkageMode`)

| Modo | Como obtém $\alpha$ | Observação |
|---|---|---|
| `LEDOIT_WOLF` | forma fechada ótima (Ledoit & Wolf 2004) | sem hiperparâmetro — defensável em corte sem justificar valor |
| `MANUAL` (default do config) | $\alpha$ fixo via `shrinkage_intensity` | default `0.10` |
| `NONE` | $\alpha = 0$ | reproduz `TWO_COVARIANCE` puro; uso de ablação |

A fórmula de Ledoit-Wolf (em `ledoit_wolf_alpha`), com $S = R^\top R / N$ sobre
os resíduos intra-classe centrados $R$:

$$
\mu = \tfrac{\mathrm{tr}(S)}{D},\quad
d^2 = \tfrac{\lVert S-\mu I\rVert_F^2}{D},\quad
\bar b^2 = \tfrac{1}{N^2}\sum_n \tfrac{\lVert r_n r_n^\top - S\rVert_F^2}{D},
$$
$$
b^2 = \min(d^2,\ \bar b^2), \qquad \alpha = \mathrm{clip}\!\left(\tfrac{b^2}{d^2},\,0,\,1\right).
$$

> **Nota de calibração.** O default do *config* é `MANUAL` com $\alpha=0.10$,
> não `LEDOIT_WOLF`. O motivo está documentado no código: para a configuração
> típica do FaceAudit (Centering + CORAL + Whitening sobre AdaFace), o
> Ledoit-Wolf tende a escolher $\alpha$ agressivo demais (~0.23 no dataset de
> validação interno), o que prejudica a calibração via LogReg a jusante.
> `LEDOIT_WOLF` é excelente para ablação/comparativo, mas $\alpha=0.10$ manual
> foi o meio-termo empiricamente recomendado. (O *enum* documenta
> `LEDOIT_WOLF` como "padrão defensável sem justificar valor" — a recomendação
> operacional vence o default teórico aqui.)

### 8.4 Auditoria

O $\alpha$ efetivamente aplicado é persistido no estado treinado
(`TwoCovariancePLDAState.shrinkage_alpha_w`), junto com `shrinkage_mode` e
`shrinkage_target` (`SCALED_IDENTITY`). O `ResourceInfoWidget` exibe, por
exemplo, `shrinkage=LEDOIT_WOLF (α_w=0.2314) | target=SCALED_IDENTITY`.

---

## 9. Modo cross-condition

### 9.1 O problema

`fit_two_covariance` clássico junta todas as amostras num único pool. Quando as
condições de captura são **assimétricas** (ex.: 10 fotos frontais e 2 frames
laterais de CFTV por pessoa), a $W$ empírica é dominada pela condição
majoritária — a variabilidade *cross-condição* que importa no caso real fica
subponderada e o LR vira **superconfiante**.

### 9.2 A solução: peso igual por condição

O modo cross-condition é ativado quando a instrução de treino define um
**`secondary_dataset_resource_id`**. Ele exige uma variante two-covariance
(`SIMPLE` + secundário ⇒ `ValueError`). Estima $W$ **por condição** e combina
com peso igual:

$$
W = \tfrac{1}{2}\,(W_A + W_B),
$$
$$
\mu_c = \tfrac{1}{2}\,(\mu_c^A + \mu_c^B)\ \text{(ou a média da condição disponível)},
$$
$$
\mu = \frac{1}{C}\sum_c \mu_c,
\qquad
B = \frac{1}{C}\sum_c (\mu_c-\mu)(\mu_c-\mu)^\top.
$$

As normalizações divergem **por design** do clássico: $W$ usa média
não-ponderada das condições (vs $S_w/N$ total) e $B$ usa peso igual por classe
(vs ponderada por $n_c$). A assimetria do merge é justamente o que o estimador
remove. O scoring permanece **idêntico** (mesmos kernels $Q$, $P$, `const`).

### 9.3 Centragem dos resíduos (`CrossConditionWithinMode`)

| Modo | Centro dos resíduos | Semântica |
|---|---|---|
| `BALANCED_IDENTITY` (default) | $\mu_c$ balanceada | $W$ absorve o deslocamento per-identidade entre condições — **mais conservador** forensicamente |
| `PER_CONDITION` | $\mu_c^{\text{cond}}$ (média da própria condição) | remove o deslocamento per-identidade do $W$ — assume que CORAL/Whitening já eliminaram o shift cross-condição |

O default `BALANCED_IDENTITY` é o defensável: evita LR superconfiante quando
resta interação identidade×condição após as transformações. A recomendação é
comparar empiricamente os dois no projeto real.

### 9.4 Política de singletons

Diferente do single-dataset, o cross-condition **preserva singletons** (só
descarta `identity=None`):

- singletons **entram em $B$ e $\mu$** via $\mu_c$ balanceada;
- singletons **não contribuem como linha em $W_{\text{cond}}$** (não há
  resíduo intra-classe estimável a partir de 1 amostra).

Em `BALANCED_IDENTITY`, um singleton pareado numa condição influencia
*indiretamente* os resíduos da outra: uma identidade com 10 frontais e 1
lateral tem $\mu_c$ balanceada puxada pela única lateral, e os resíduos das 10
frontais ficam centrados nesse ponto puxado. É a propriedade desejada (a
meia-soma puxa para o ponto cross-condição), mas não é óbvia.

### 9.5 Shrinkage em cross-condition

Para não reintroduzir o desbalanceamento, o Ledoit-Wolf é estimado por
condição e combinado por média não-ponderada: $\alpha = \tfrac{1}{2}(\alpha_A
+ \alpha_B)$ — espelhando a lógica de $W = (W_A + W_B)/2$.

### 9.6 Metadados de auditoria

O estado treinado registra `cross_condition=True`, `n_ids_condition_a/b`,
`n_common_identities` e `cross_condition_within_mode`. **Todos os campos
cross-condition e de shrinkage são opcionais** no `TwoCovariancePLDAState` —
a UI deve lê-los via `getattr` para compat com pickles antigos.

---

## 10. Scoring: do LLR bruto ao par

### 10.1 Os três métodos do protocolo

| Método | Entrada | Saída | Uso |
|---|---|---|---|
| `score_pair(probe, reference)` | dois `EmbeddingInfo` | `float` | conveniência; delega a `score_vectors` |
| `score_vectors(v1, v2)` | dois vetores 1D | `float` | par único |
| `score_matrix(P, R)` | duas matrizes 2D | `np.ndarray` $n_p\times n_r$ | lote (produção) |

Todos exigem que o modelo esteja `fitted` (`_ensure_fitted`) e que as
dimensões batam. No PLDA, `score_*` ramifica para o backend da variante:
two-covariance usa os kernels de `two_covariance_plda_service`; SIMPLE usa
`fast_PLDA_scoring`.

### 10.2 Auto vs cross comparison (na instrução `SIMILARITY_SCORING`)

- **Self-comparison** (só `probe_dataset`): gera o **triângulo superior**
  ($i < j$, sem diagonal) — $n(n-1)/2$ pares.
- **Cross-comparison** (`probe` + `reference`): gera **todos** os
  $n_p \times n_r$ pares.

Em ambos os casos o executor chama `model.score_matrix(...)` vetorizadamente e
preenche o `raw_score` de cada `SimilarityScoreSample`.

### 10.3 Modo casework

Quando `is_casework=True`, o executor **zera o `same_person`** de todas as
amostras (não há ground truth no caso real) e marca
`metadata["is_casework"]=True`. Isso impede que o caso seja confundido com um
dataset de validação a jusante (regras `CaseworkStateRule`).

---

## 11. Pré-requisitos de dados e validação

### 11.1 A dimensionalidade é o gargalo do PLDA

Estimar $B$ e $W$ cheias exige $D^2$ parâmetros cada. A regra geométrica
mínima é que **o número de identidades exceda a dimensão dos embeddings**.
Quando não há identidades suficientes, há três saídas, em ordem de
preferência: (1) mais identidades; (2) reduzir $D$ via LDA/PCA no pipeline;
(3) usar shrinkage/regularização (paliativo, não solução).

### 11.2 Regras de validação populadas automaticamente

**`SIMILARITY_MODEL_TRAINING`** (PLDA single-dataset) popula como defaults
persistentes:

| Regra | O que checa | Severidade |
|---|---|---|
| `FISHER` | $n_{\text{ids}} > D$ | CRÍTICA se $D \ge n_{\text{ids}}$; warning se margem estreita |
| `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES` | $\ge 2$ identidades com $\ge 2$ amostras | CRÍTICA se $< 2$; warning se $< 10$ |

Em **cross-condition** (secundário definido + variante two-covariance), os
defaults se estendem:

| Regra | Alvo |
|---|---|
| `FISHER` | dataset primário |
| `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES` | primário **e** secundário |
| `DATASETS_EMBEDDING_COMPAT` | primário × secundário (mesma coluna/dim) |
| `DATASET_OVERLAP_REQUIRED` | primário × secundário (identidades comuns) |

O **cosseno não popula regras** (stateless).

### 11.3 Detalhe das regras críticas do PLDA

**`FisherRule`** (`safety_margin_ratio=0.2` por default):

- $n_{\text{ids}} < 2$ → ERRO (PLDA exige $\ge 2$);
- $D \ge n_{\text{ids}}$ → ERRO de "singularidade de Fisher" (teto de
  dimensões suportadas é $n_{\text{ids}}-1$);
- $n_{\text{ids}} \le D\cdot(1+0{,}2)$ → WARNING de margem estreita;
- caso contrário → OK.

**`MinIdentitiesWithMultipleSamplesRule`** (`min_required=2`,
`safety_threshold=10`):

- lê `stats.samples_per_identity`; conta identidades com $\ge 2$ amostras
  ($n_{\text{multi}}$);
- $n_{\text{multi}} < 2$ → ERRO (singletons serão descartados no fit);
- $n_{\text{multi}} < 10$ → WARNING ("viável mas estatisticamente frágil");
- caso contrário → OK.

### 11.4 Regras dinâmicas do scoring

**`SIMILARITY_SCORING`** constrói em runtime (não persistidas):

- `EmbeddingsCompleteRule` — sobre o probe e, se houver, sobre a referência:
  face sem embedding barra o cálculo, qualquer que seja o modelo (a regra
  existe mesmo antes de o modelo ser escolhido);
- `ModelFittedRule` — o modelo selecionado tem `is_fitted=True`;
- `EmbeddingDimMatchesModelRule` — sobre o probe e, se cross-comparison,
  também sobre a referência. Modelos sem `input_dimension` (cosseno) passam.

A dimensão do modelo PLDA vem de `input_dimension` (que devolve `trained_dim`,
gravado no fit).

---

## 12. Integração no pipeline do FaceAudit

### 12.1 Treino — `SIMILARITY_MODEL_TRAINING`

O `SimilarityModelTrainingInstruction` carrega:

| Campo | Tipo | Uso |
|---|---|---|
| `similarity_model_type` | `SimilarityModelType` | `COSINE` ou `PLDA` |
| `input_dataset_resource_id` | `Optional[str]` | obrigatório para PLDA; nulo para cosseno |
| `secondary_dataset_resource_id` | `Optional[str]` | ativa cross-condition (só two-covariance) |
| `similarity_model_config` | `Optional[PLDASimilarityModelConfig]` | hiperparâmetros PLDA; nulo no cosseno |
| `trained_similarity_model_resource_id` | `Optional[str]` | id do recurso de saída (`TRAINED_SIMILARITY_MODEL`) |

O executor:

1. valida (estrutural + plugável + dinâmica + específica);
2. **cosseno** → instancia `CosineSimilarityModel()` (sem dataset);
3. **PLDA** → carrega o dataset; se há secundário, carrega-o e **emite WARNING
   de proveniência** (assume que ambos vieram do mesmo pipeline, CORAL/
   Whitening compatíveis); instancia `PLDASimilarityModel(config)` e chama
   `fit(dataset, secondary_dataset=...)`;
4. registra o `TRAINED_SIMILARITY_MODEL` no contexto;
5. emite metadados: tipo do modelo, ids de entrada/saída e, em cross-condition,
   `cross_condition`, `n_ids_condition_a/b`, `n_common_identities`,
   `cross_condition_within_mode`.

A UI usa `PLDASimilarityModelConfig.get_hidden_fields()` /
`REACTIVE_FIELDS` / `get_readonly_fields()` para mostrar só os campos
relevantes à variante escolhida (ex.: em `TWO_COVARIANCE`, oculta
`em_iterations`/`plda_mode`/`latent_dim`).

### 12.2 Scoring — `SIMILARITY_SCORING`

Campos principais: `trained_similarity_model_resource_id` (obrigatório),
`probe_dataset_resource_id` (obrigatório), `reference_dataset_resource_id`
(opcional → define self vs cross), `output_similarity_scores_resource_id`
(obrigatório), `is_casework` e `store_metadata_per_sample`. A saída é um
`SimilarityScoreDataset` (tipo `SIMILARITY_SCORES`), pronto para o calibrador.

### 12.3 Inferência forense — `StrategyInferenceService`

Na aplicação a um caso real, o serviço:

1. hidrata os datasets probe/reference;
2. **auto-detecta** o modelo por tipo de recurso
   (`_find_payload_by_type(..., TRAINED_SIMILARITY_MODEL)`);
3. aplica o pipeline (LINEAR: `transform`; DUAL: `transform_source` no probe e
   `transform_target` na referência);
4. chama `similarity_model.score_matrix(probe_matrix, ref_matrix)` →
   `raw_matrix`;
5. se houver `TRAINED_CALIBRATION_MODEL`, calibra (`calibrate_scores`) e grava
   `metadata["log_lr"]` por par;
6. devolve um `SimilarityScoreDataset` com `raw_score` (e `log_lr` quando
   calibrado).

---

## 13. Como escolher a variante

Árvore de decisão prática (premissa: já existe um pipeline de transformação e
um dataset rotulado por identidade):

1. **Baseline / sem treino / poucos dados rotulados** → **Cosseno**. Toda a
   força forense fica no calibrador.

2. **PLDA com dados rotulados:**

   - **Poucas identidades, domínio ruidoso, ou $D$ muito alto sem redução** →
     **`SIMPLE` (LOW_RANK)**. O truncamento de rank regulariza.
   - **Dados fartos (identidades ≫ $D$, várias amostras por pessoa)** →
     **`TWO_COVARIANCE`** com `regularization` leve (`1e-6`–`1e-4`). Padrão
     forense, sem perda por truncamento.
   - **$W$ mal-condicionada (poucas amostras/ID, $D$ alto) mas se quer matriz
     cheia** → **`SHRINKAGE_TWO_COVARIANCE`**. Comece com `MANUAL` $\alpha
     \approx 0.10$; compare com `LEDOIT_WOLF` em ablação.

3. **Comparação entre condições assimétricas** (foto-padrão × CFTV, frontal ×
   lateral) → qualquer variante two-covariance **com `secondary_dataset`**
   (cross-condition), default `BALANCED_IDENTITY`.

Em todos os casos: **valide num holdout externo** e compare estratégias
concorrentes na janela de Comparação antes de escolher. O score bruto só vira
evidência depois da calibração — a escolha da variante deve ser julgada pelas
métricas pós-calibração (Cllr, discriminação), não pelo LLR bruto isolado.

---

## 14. Relação com a calibração e o double-counting de tipicidade

### 14.1 O PLDA já codifica tipicidade

A covariância entre-classes $B$ do PLDA mede a dispersão da população de
identidades. Por construção, faces atípicas (longe de $\mu$ na métrica de $B$)
recebem maior peso probatório quando há similaridade. Ou seja, **o PLDA já
incorpora um componente de tipicidade** no seu LLR bruto.

### 14.2 O risco

Combinar **PLDA + `SCORE_INTERACTION_FEATURE`** (a feature que modula
similaridade por raridade derivada de um modelo de tipicidade) pode **contar a
tipicidade duas vezes**. A documentação do módulo de interação alerta
explicitamente para isso: a combinação exige uma comparação A/B explícita
(PLDA sozinho contra PLDA + interação, lado a lado na janela de Comparação)
para verificar se a feature de
raridade ainda agrega informação além do que o $B$ do PLDA já captura.

Com **cosseno** o risco não existe (o cosseno não modela tipicidade), e a
feature de interação tende a agregar informação genuína.

### 14.3 Fronteira

Este documento cobre apenas o LLR bruto. A modelagem multivariada
score × raridade, os modos de $s_0$ e a tese forense de que "raridade não é
evidência independente" estão nos módulos
`faceaudit/executors/instructions/score_interaction_feature_instruction_executor.py`
e nos enums correspondentes — e a calibração final, em
`docs/calibracao_faceaudit.md`.

---

## 15. Persistência, versionamento e rastreabilidade

### 15.1 Formato

Modelos de similaridade são persistidos como **pickle**. O `PLDASimilarityModel`
usa `MODEL_VERSION = 2`; o `CosineSimilarityModel`, `MODEL_VERSION = 1`.

O payload do PLDA inclui: `backend` (`speechbrain` / `two_covariance` /
`shrinkage_two_covariance`), `config` (dict serializado),
o objeto `plda` do SpeechBrain (variante SIMPLE) **ou** `two_cov_state`
serializado como dict (variantes two-covariance), além de `trained_rank`,
`trained_dim` e `is_fitted`.

### 15.2 Retrocompatibilidade

- `load()` aceita payloads `model_version=1` (só SpeechBrain) e os trata como
  variante SIMPLE — comportamento idêntico ao anterior.
- O `two_cov_state` é reconstruído com `getattr`/`.get` defensivos; campos de
  shrinkage e cross-condition ausentes em pickles antigos viram `None`/`False`.
- **Os kernels Q/P/const não são persistidos** — são recomputados em `load()`.
  Razão: evitar bloat e dessincronização caso a regularização mude.
- O `PLDASimilarityModelConfig.from_dict` defaulta `plda_variant` ausente para
  `SIMPLE`, preservando estratégias já salvas.

### 15.3 O que registrar para defensibilidade

Para um modelo PLDA usado em laudo, registre (a UI já expõe a maioria no
`ResourceInfoWidget`): a **variante**, $D$ de entrada, `regularization`,
`random_seed`; em SIMPLE, `em_iterations`/`plda_mode`/`latent_dim`; em
shrinkage, `shrinkage_mode`/`shrinkage_alpha_w`/`shrinkage_target`; em
cross-condition, `within_mode`, `n_ids_condition_a/b`, `n_common_identities`;
e o dataset de treino (composição por identidade, singletons descartados).

---

## 16. Boas práticas resumidas

- **Separe similaridade de calibração.** O score bruto nunca é a evidência
  final; valide pela métrica pós-calibração.
- **Verifique a regra de Fisher antes de tudo.** $n_{\text{ids}} > D$ é
  condição necessária para two-covariance; se falhar, reduza $D$ no pipeline.
- **Não suba `regularization`/`shrinkage` para esconder problemas de dados.**
  Primeiro investigue duplicatas, excesso de passos de pipeline e coerência de
  $D$ com o nº de classes.
- **Prefira `TWO_COVARIANCE` com dados fartos; `SIMPLE` com poucos.** Use
  shrinkage quando $W$ for mal-condicionada.
- **Em comparações cross-condição, use o modo cross-condition.** Juntar tudo
  num pool único produz LR superconfiante; default `BALANCED_IDENTITY`.
- **Cuidado com double-counting.** PLDA + interação de raridade exige A/B.
- **Compare datasets antes do cross-condition.** Garanta compatibilidade de
  dimensão e overlap de identidades.
- **Registre tudo.** Variante, hiperparâmetros, $\alpha$ efetivo, composição do
  treino — para reprodução e defesa em corte.

---

## 17. Perguntas frequentes

**O score do PLDA já é o log-LR final?**
Não. É um **LLR bruto** sob as suposições gaussianas do modelo treinado. A
evidência reportável vem da calibração (ver `docs/calibracao_faceaudit.md`).

**Posso usar PLDA sem identidades rotuladas?**
Não. O PLDA precisa de identidades com $\ge 2$ amostras para estimar $W$.
Singletons são descartados no treino single-dataset. Sem rótulos, use cosseno.

**Qual a diferença prática entre SIMPLE e TWO_COVARIANCE?**
SIMPLE aprende um subspace de baixo rank por EM (regulariza por truncamento,
perde informação fora do subspace); TWO_COVARIANCE estima covariâncias cheias
em forma fechada (sem perda, mas exige mais dados). Escolha pela quantidade de
dados e dimensionalidade.

**Por que o default de shrinkage é MANUAL e não LEDOIT_WOLF?**
Porque, na configuração típica do FaceAudit, o Ledoit-Wolf escolhe $\alpha$
agressivo demais (~0.23) e isso piora a calibração a jusante. $\alpha=0.10$
manual foi o meio-termo empírico. O enum ainda documenta LEDOIT_WOLF como
default teórico defensável; use-o em ablação.

**Quando preciso do modo cross-condition?**
Quando probe e referência vêm de condições de captura sistematicamente
diferentes e desbalanceadas em amostras/ID (foto-padrão × CFTV). Sem ele, a
$W$ é dominada pela condição majoritária e o LR fica superconfiante.

**`regularization` e `shrinkage_intensity` fazem a mesma coisa?**
Não. `regularization` (Tikhonov) soma $\varepsilon I$; em two-covariance pura
incide em $B$ e $W$, nas demais só em $W$. `shrinkage_intensity` é o $\alpha$
que mistura $W$ empírica com o alvo `tr(W)/D · I`, só na variante de shrinkage.
Podem coexistir (shrinkage primeiro, Tikhonov final em $W$).

**O cosseno precisa ser treinado?**
Não. É stateless (`is_fitted=True` desde a construção). O `fit()` só registra
metadados.

**O LLR do PLDA é simétrico?**
Sim, no two-covariance (kernel $P$ simétrico) e no SIMPLE. O cross-condition
mantém a simetria por usar uma única $W$ combinada (scorer homocedástico).

---

## 18. Glossário

- **Embedding** — vetor $x\in\mathbb{R}^D$ que representa uma face.
- **$D$** — dimensionalidade dos embeddings.
- **$C$ / $n_{\text{ids}}$** — número de classes/identidades distintas.
- **Score bruto** — saída do modelo de similaridade, antes da calibração.
- **LLR bruto** — score bruto do PLDA (log-likelihood ratio sob o modelo).
- **$B$** — covariância entre-classes (dispersão das identidades; tipicidade).
- **$W$** — covariância intra-classe (variação de amostras de uma mesma pessoa).
- **Subspace $F$ / rank latente** — espaço entre-classes de baixo rank na
  variante SIMPLE.
- **Singleton** — identidade com apenas 1 amostra (descartada no treino
  single-dataset; preservada em cross-condition para $B/\mu$).
- **Shrinkage** — encolhimento de $W$ na direção de um alvo bem-condicionado.
- **Ledoit-Wolf** — estimador analítico do coeficiente de shrinkage $\alpha$.
- **Tikhonov / `regularization`** — soma de $\varepsilon I$ para estabilizar
  inversões.
- **Cross-condition** — treino com dois datasets condicionais e peso igual por
  condição.
- **Kernels Q/P/const** — operadores pré-computados do scoring two-covariance.
- **Genuíno / impostor** — par de mesma / diferente identidade ($H_p$/$H_d$).

---

## 19. Referências

- Brümmer, N., & de Villiers, E. (2010). *The speaker partitioning problem.*
  Odyssey Speaker and Language Recognition Workshop. (Formulação two-covariance.)
- Brümmer, N., & du Preez, J. (2006). *Application-independent evaluation of
  speaker detection.* Computer Speech & Language. (Raízes do log-LR.)
- Ioffe, S. (2006). *Probabilistic Linear Discriminant Analysis.* ECCV.
  (Formulação com subspace, base da variante SIMPLE.)
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices.* Journal of Multivariate Analysis.
- Prince, S. J. D., & Elder, J. H. (2007). *Probabilistic Linear Discriminant
  Analysis for Inferences About Identity.* ICCV.
- SpeechBrain — `speechbrain.processing.PLDA_LDA` (backend da variante SIMPLE).
- Referências internas do FaceAudit:
  - `docs/calibracao_faceaudit.md` — calibração do score bruto em log-LR.
  - Docstrings de `two_covariance_plda_service.py`,
    `shrinkage_two_covariance_plda_service.py` e
    `cross_condition_two_covariance_plda_service.py`.

---

## Apêndice A — Fórmulas de referência

**Similaridade cosseno:**

$$
\mathrm{cos}(a,b)=\frac{a^\top b}{\lVert a\rVert\,\lVert b\rVert}.
$$

**Modelo generativo two-covariance:**

$$
y_c \sim \mathcal{N}(\mu, B), \qquad x \mid y_c \sim \mathcal{N}(y_c, W).
$$

**Estimadores (MLE enviesado), classes com $n_c\ge 2$:**

$$
W = \frac{1}{N}\sum_c\sum_{i\in c}(x_i-\mu_c)(x_i-\mu_c)^\top,\quad
B = \frac{1}{N}\sum_c n_c(\mu_c-\mu)(\mu_c-\mu)^\top.
$$

**Regularização Tikhonov (two-covariance pura: em $B$ e $W$):**

$$
W \leftarrow W + \varepsilon I, \qquad B \leftarrow B + \varepsilon I.
$$

**Kernels de scoring:**

$$
K=(W+B)^{-1},\ M=(W+2B)^{-1},\ N_{\!*}=W^{-1},
$$
$$
Q = K - \tfrac{1}{2}(M+N_{\!*}),\quad P = N_{\!*}-M,
$$
$$
\mathrm{const}=\log\lvert W+B\rvert-\tfrac{1}{2}\log\lvert W+2B\rvert-\tfrac{1}{2}\log\lvert W\rvert.
$$

**LLR bruto de um par ($z=x-\mu$):**

$$
\log\mathrm{LR}=\tfrac{1}{2}\bigl(z_1^\top Q z_1 + z_2^\top Q z_2 + z_1^\top P z_2\bigr)+\mathrm{const}.
$$

**Shrinkage de $W$ (alvo scaled-identity):**

$$
W_{\text{shrunk}}=(1-\alpha)W_{\text{emp}}+\alpha\,\tfrac{\mathrm{tr}(W)}{D}I.
$$

**Ledoit-Wolf $\alpha$ (resíduos $R$, $S=R^\top R/N$):**

$$
d^2=\tfrac{\lVert S-\mu I\rVert_F^2}{D},\quad
\bar b^2=\tfrac{1}{N^2}\sum_n\tfrac{\lVert r_n r_n^\top-S\rVert_F^2}{D},\quad
\alpha=\mathrm{clip}\!\left(\tfrac{\min(d^2,\bar b^2)}{d^2},0,1\right).
$$

**Cross-condition (peso igual por condição):**

$$
W=\tfrac{1}{2}(W_A+W_B),\quad
\mu_c=\tfrac{1}{2}(\mu_c^A+\mu_c^B),\quad
B=\tfrac{1}{C}\sum_c(\mu_c-\mu)(\mu_c-\mu)^\top,\quad
\alpha=\tfrac{1}{2}(\alpha_A+\alpha_B).
$$

**Rank da variante SIMPLE:**

$$
r_{\max}=\max(1,\min(D,C-1)).
$$

FULL_RANK usa $r = r_{\max}$; LOW_RANK usa
$r=\max\!\bigl(1,\min(\ell,r_{\max})\bigr)$, onde $\ell$ é o `latent_dim`
configurado.

---

## Apêndice B — `PLDASimilarityModelConfig`

Arquivo: `faceaudit/models/similarity/plda_similarity_model_config.py`.

| Campo | Tipo | Default | Interpretação | Restrição / observação |
|---|---|---|---|---|
| `plda_variant` | `PLDAVariant` | `SIMPLE` | variante matemática | ver Apêndice C |
| `em_iterations` | `int` | `20` | iterações do EM | `> 0`; só em SIMPLE |
| `plda_mode` | `PLDAMode` | `LOW_RANK` | modo do rank latente | só em SIMPLE |
| `latent_dim` | `int` | `128` | rank em LOW_RANK | `> 0`; ignorado em FULL_RANK e em two-covariance |
| `regularization` | `float` | `0.0` | Tikhonov $\varepsilon$ | `>= 0`; só em two-covariance (em $B$ e $W$ na pura; só $W$ no shrinkage) |
| `shrinkage_mode` | `CovarianceShrinkageMode` | `MANUAL` | como obter $\alpha$ | só em SHRINKAGE_TWO_COVARIANCE |
| `shrinkage_intensity` | `float` | `0.1` | $\alpha$ manual | `[0,1]`; só lido em `MANUAL` |
| `cross_condition_within_mode` | `CrossConditionWithinMode` | `BALANCED_IDENTITY` | centragem dos resíduos | só tem efeito com dataset secundário |
| `random_seed` | `int` | `42` | reprodutibilidade | — |

Visibilidade dinâmica na UI (`get_hidden_fields`):

- **SIMPLE** oculta `regularization`, `shrinkage_mode`, `shrinkage_intensity`,
  `cross_condition_within_mode`.
- **TWO_COVARIANCE** oculta `em_iterations`, `plda_mode`, `latent_dim`,
  `shrinkage_mode`, `shrinkage_intensity`.
- **SHRINKAGE_TWO_COVARIANCE** oculta `em_iterations`, `plda_mode`,
  `latent_dim`; oculta `shrinkage_intensity` se `shrinkage_mode != MANUAL`.

Read-only (`get_readonly_fields`): `latent_dim` em SIMPLE + FULL_RANK (valor
ignorado em runtime). `REACTIVE_FIELDS = ("plda_variant", "plda_mode",
"shrinkage_mode")` — disparam rebuild do form aninhado.

Validações de `__post_init__`: `em_iterations > 0`, `latent_dim > 0`,
`regularization >= 0`, `shrinkage_intensity ∈ [0,1]`.

---

## Apêndice C — Enums

**`SimilarityModelType`** (`faceaudit/enums/similarity_model_type.py`):
`COSINE`, `PLDA`. Default `COSINE`.

**`PLDAVariant`** (`faceaudit/enums/plda_variant.py`):

| Membro | Significado |
|---|---|
| `SIMPLE` (default) | SpeechBrain, subspace $F$, EM; usa `em_iterations`/`plda_mode`/`latent_dim` |
| `TWO_COVARIANCE` | Brümmer 2010, $B$/$W$ cheias, forma fechada; usa `regularization` |
| `SHRINKAGE_TWO_COVARIANCE` | two-covariance + shrinkage de $W$; usa `shrinkage_*` |

**`PLDAMode`** (`faceaudit/enums/plda_mode.py`): `FULL_RANK`
(rank $=\min(D,C-1)$) e `LOW_RANK` (rank $\le$ `latent_dim`). `from_str`
defaulta `FULL_RANK`.

**`CovarianceShrinkageMode`** (`faceaudit/enums/covariance_shrinkage_mode.py`):
`NONE` (ablação), `LEDOIT_WOLF` (analítico; default do enum), `MANUAL`
($\alpha$ fixo). `from_str` defaulta `LEDOIT_WOLF`. Alvo sempre
`SCALED_IDENTITY`, aplicado só a $W$.

**`CrossConditionWithinMode`** (`faceaudit/enums/cross_condition_within_mode.py`):
`PER_CONDITION` e `BALANCED_IDENTITY` (default). Só tem efeito com
`secondary_dataset`.

> Todos os enums definem `_value_` em `__new__` para serem serializáveis via
> pickle e reconstruídos por valor-string. `from_str` é tolerante
> (`strip/upper/replace`) e cai num default seguro.

---

## Apêndice D — `TwoCovariancePLDAState`

Arquivo: `faceaudit/services/two_covariance_plda_service.py`. Frozen dataclass
que representa o estado treinado das variantes two-covariance.

| Campo | Tipo | Sempre presente? | Significado |
|---|---|---|---|
| `mu` | `np.ndarray (D,)` | sim | média global |
| `B` | `np.ndarray (D,D)` | sim | covariância entre-classes |
| `W` | `np.ndarray (D,D)` | sim | covariância intra-classe (pós-shrinkage se aplicável) |
| `n_classes` | `int` | sim | nº de classes usadas (single) ou união (cross) |
| `n_samples` | `int` | sim | nº de amostras de treino |
| `D` | `int` | sim | dimensionalidade |
| `shrinkage_mode` | `Optional[str]` | não | modo de shrinkage ou `None` |
| `shrinkage_target` | `Optional[str]` | não | `SCALED_IDENTITY` ou `None` |
| `shrinkage_alpha_w` | `Optional[float]` | não | $\alpha$ efetivo em $W$ |
| `shrinkage_alpha_b` | `Optional[float]` | não | reservado; sempre `None` na v1 (B empírica) |
| `cross_condition` | `bool` | sim (default `False`) | modo cross-condition |
| `n_ids_condition_a` | `Optional[int]` | não | identidades distintas em A |
| `n_ids_condition_b` | `Optional[int]` | não | identidades distintas em B |
| `n_common_identities` | `Optional[int]` | não | identidades em A ∩ B |
| `cross_condition_within_mode` | `Optional[str]` | não | `PER_CONDITION`/`BALANCED_IDENTITY` |

Os kernels $Q$, $P$, `const` **não** fazem parte do estado — são recomputados
em `fit()`/`load()` e cacheados em `PLDASimilarityModel._two_cov_kernels`.
Acesse os campos opcionais via `getattr` (compat com pickles antigos).

---

## Apêndice E — Regras de validação aplicáveis

| Regra | Tipo | Onde aplica | Critério |
|---|---|---|---|
| `FisherRule` | `FISHER` | treino PLDA (primário) | $n_{\text{ids}}>D$; CRÍTICA se $D\ge n_{\text{ids}}$ ou $n_{\text{ids}}<2$; warning se margem $\le 1{+}0{,}2$ |
| `MinIdentitiesWithMultipleSamplesRule` | `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES` | treino PLDA (primário; +secundário em cross) | $n_{\text{multi}}\ge 2$ (CRÍTICA); warning se $<10$ |
| `DatasetsEmbeddingCompatRule` | `DATASETS_EMBEDDING_COMPAT` | cross-condition (primário×secundário) | mesma coluna de embedding e dimensão |
| `DatasetOverlapRequiredRule` | `DATASET_OVERLAP_REQUIRED` | cross-condition (primário×secundário) | $\ge$ N identidades comuns; warning abaixo do ratio |
| `EmbeddingsCompleteRule` | `EMBEDDINGS_COMPLETE` (dinâmica) | scoring (probe; +referência em cross) | toda face tem embedding |
| `ModelFittedRule` | `MODEL_FITTED` (dinâmica) | scoring | `is_fitted=True` |
| `EmbeddingDimMatchesModelRule` | `EMBEDDING_DIM_MATCHES_MODEL` (dinâmica) | scoring (probe; +referência em cross) | `dataset.embedding_dim == model.input_dimension`; cosseno passa |

As duas primeiras são **defaults persistentes** (editáveis/removíveis na UI;
flag `validation_rules_user_managed`), assim como as duas seguintes no modo
cross-condition; as três últimas são **dinâmicas**
(construídas em runtime, não persistidas, não removíveis).

---

## Apêndice F — Checklist de revisão documental

- [ ] As fórmulas de $W$, $B$, $Q$, $P$, `const` conferem com
      `two_covariance_plda_service.py`?
- [ ] A regra de rank da SIMPLE confere com `_decide_rank_f`
      ($\max(1,\min(D,C-1))$)?
- [ ] A política de singletons (descarte single-dataset; preservação
      cross-condition) está descrita corretamente?
- [ ] A divergência de Tikhonov ($B$+$W$ na pura; só $W$ no shrinkage e no
      cross) está explícita?
- [ ] O default de shrinkage (config `MANUAL` α=0.10 vs enum `LEDOIT_WOLF`)
      está esclarecido com a justificativa empírica?
- [ ] Os campos opcionais de `TwoCovariancePLDAState` estão marcados como
      acessíveis via `getattr`?
- [ ] As regras de validação e seus thresholds (Fisher 0.2;
      MinIdentities 2/10) conferem com os arquivos das regras?
- [ ] O documento mantém a fronteira similaridade × calibração (não invade o
      escopo de `calibracao_faceaudit.md`)?

---

## Apêndice G — Checklist de validação operacional

- [ ] O dataset de treino tem $n_{\text{ids}} > D$ (regra de Fisher)?
- [ ] Há $\ge 10$ identidades com $\ge 2$ amostras (acima do warning)?
- [ ] A variante escolhida é compatível com o volume de dados (§13)?
- [ ] `regularization`/`shrinkage` foram ajustados por necessidade real, não
      para mascarar dados ruins?
- [ ] Em cross-condition: datasets compatíveis em dimensão e com overlap de
      identidades adequado?
- [ ] O modo de centragem (`BALANCED_IDENTITY` vs `PER_CONDITION`) foi
      comparado empiricamente?
- [ ] O modelo foi avaliado por métricas **pós-calibração** num holdout
      externo, não pelo LLR bruto isolado?
- [ ] Se há PLDA + interação de raridade, foi feita a comparação A/B
      (double-counting)?
- [ ] Hiperparâmetros, $\alpha$ efetivo e composição do treino foram
      registrados para reprodução?
