# Interação score × raridade no FaceAudit (`SCORE_INTERACTION_FEATURE`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `SCORE_INTERACTION_FEATURE` — a feature forense de
> interação entre um score de similaridade e a raridade derivada de
> tipicidade. Objetivo, fundamentação, configurações, fórmulas, exemplos e
> controles metodológicos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** código atual do repositório FaceAudit. Campos, fórmulas
> e enums conferidos em `faceaudit/executors/instructions/score_interaction_feature_instruction_executor.py`,
> `faceaudit/models/instructions/score_interaction_feature_instruction.py`,
> `faceaudit/models/calibration/logreg_s0_config.py`,
> `faceaudit/enums/score_interaction_mode.py`,
> `faceaudit/enums/s0_mode.py`.
>
> **⚠️ V2.14 — substituto declarativo para as arquiteturas LR canônicas.**
> A partir de V2.14, a forma recomendada de montar uma das arquiteturas LR
> forenses canônicas (15 no registry: as 9 two-face `COSINE_ONLY` →
> `BILATERAL_FULL_CENTERED` e as 6 variantes ONE_FACE da V2.14.1) é a instrução
> única **`LR_ARCHITECTURE_VECTORIZATION`**, que dispensa o encadeamento
> manual `SCORE_INTERACTION_FEATURE × N + SCORE_VECTORIZATION` e adiciona o
> modo `parameter_mode=FIT_PER_FOLD` — s₀ e r̄ ajustados POR FOLD dentro do
> `GridSearchCV` do calibrador via `Pipeline + ArchitectureVectorizerTransformer`,
> eliminando o leakage residual de s₀ (que neste fluxo é estimado em
> `SCORE_INTERACTION_FEATURE` ANTES da CV do calibrador a jusante — ver §6.3
> e §12 abaixo). **Esta documentação permanece canônica** para os fundamentos
> forenses (modos de interação, inversão typicality→rarity, centralização Tukey,
> protocolo de congelamento de s₀, double-counting com PLDA) — todos eles foram
> herdados pela nova instrução declarativa e continuam aplicáveis em sua
> semântica. `SCORE_INTERACTION_FEATURE` segue **funcional** e válido para:
> (i) compor features de raridade fora do registry; (ii) reproduzir estratégias
> antigas sem migração; (iii) construir uma feature isolada para empilhamento
> ad-hoc no `SCORE_VECTORIZATION`. Veja `CLAUDE.md` (seção "Arquitetura LR
> declarativa") e `docs/vetorizacao_scores_faceaudit.md` (§1.6) para a árvore
> de decisão entre os dois caminhos.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a feature se encaixa no FaceAudit](#1-onde-a-feature-se-encaixa-no-faceaudit)
- [2. Fundamentação forense](#2-fundamentação-forense)
- [3. Entradas e saída](#3-entradas-e-saída)
- [4. Pipeline de execução](#4-pipeline-de-execução)
- [5. Modos de interação](#5-modos-de-interação)
- [6. Resolução de s0 — o ponto de evidência neutra](#6-resolução-de-s0--o-ponto-de-evidência-neutra)
- [7. Conversão tipicidade → raridade](#7-conversão-tipicidade--raridade)
- [8. Centralização da raridade (r0) e o princípio de Tukey](#8-centralização-da-raridade-r0-e-o-princípio-de-tukey)
- [9. Alinhamento de pares](#9-alinhamento-de-pares)
- [10. O modelo multivariado a jusante](#10-o-modelo-multivariado-a-jusante)
- [11. Double-counting com PLDA](#11-double-counting-com-plda)
- [12. Casework e o protocolo de congelamento](#12-casework-e-o-protocolo-de-congelamento)
- [13. Validações e guard-rails](#13-validações-e-guard-rails)
- [14. Metadata de auditoria](#14-metadata-de-auditoria)
- [15. Exemplo numérico trabalhado](#15-exemplo-numérico-trabalhado)
- [16. Boas práticas resumidas](#16-boas-práticas-resumidas)
- [17. Perguntas frequentes](#17-perguntas-frequentes)
- [18. Glossário](#18-glossário)
- [19. Referências](#19-referências)
- [Apêndice A — Fórmulas de referência](#apêndice-a--fórmulas-de-referência)
- [Apêndice B — ScoreInteractionFeatureInstruction](#apêndice-b--scoreinteractionfeatureinstruction)
- [Apêndice C — LogRegS0Config](#apêndice-c--logregs0config)
- [Apêndice D — Enums](#apêndice-d--enums)
- [Apêndice E — Metadata do dataset de saída](#apêndice-e--metadata-do-dataset-de-saída)
- [Apêndice F — Checklist de revisão documental](#apêndice-f--checklist-de-revisão-documental)
- [Apêndice G — Checklist de validação operacional](#apêndice-g--checklist-de-validação-operacional)
- [Apêndice H — Protocolo de dupla interação (probe × referência)](#apêndice-h--protocolo-de-dupla-interação-probe--referência)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre a instrução `SCORE_INTERACTION_FEATURE`:
o que ela faz, todas as suas configurações, as fórmulas matemáticas que
implementa, a fundamentação forense por trás de cada decisão e exemplos de uso.

### 0.2 Público-alvo

Peritos, desenvolvedores e revisores técnicos. Didático na exposição inicial,
formal na matemática, exaustivo nas tabelas de configuração.

### 0.3 Escopo

**Incluído:** a construção da feature de interação `(base − s0)·rarity`, os cinco
modos de fórmula, a resolução e o congelamento de `s0`, a conversão
tipicidade→raridade, a centralização opcional da raridade, o alinhamento de
pares, o protocolo casework e a auditoria.

**Fora do escopo:**

- a produção do **score base** (cosseno/PLDA) — ver `docs/similaridade_faceaudit.md`;
- a produção da **tipicidade pareada** (`TYPICALITY_PAIR_SCORING`, modelos de
  tipicidade k-NN e MoVMF) — ver os módulos correspondentes em
  `faceaudit/services/`;
- a **calibração final** que consome a feature (`SCORE_VECTORIZATION` +
  `CALIBRATION_MODEL_TRAINING`) — ver `docs/calibracao_faceaudit.md`.

### 0.4 Documentos complementares

| Documento | Relação |
|---|---|
| `docs/similaridade_faceaudit.md` | produz o score base (cosseno/PLDA) consumido aqui |
| `docs/calibracao_faceaudit.md` | consome a feature de interação e converte em log-LR |
| `docs/interacao_score_raridade_faceaudit.md` (este) | a feature de interação entre os dois |

Este documento ocupa a **fronteira multivariada** explicitamente deixada de
fora dos outros dois.

---

## 1. Onde a feature se encaixa no FaceAudit

A `SCORE_INTERACTION_FEATURE` é uma etapa **intermediária** entre o scoring e a
calibração. Ela não produz log-LR — produz uma **feature 1D** que será
empilhada com o score base e entregue ao calibrador multivariado.

```text
embeddings probe/reference
  ↓                              ↓
score base (cosseno/PLDA)     tipicidade pareada (TYPICALITY_PAIR_SCORING)
  │                              │
  └──────────────┬───────────────┘
                 ↓
   SCORE_INTERACTION_FEATURE   ← ESTE DOCUMENTO
                 ↓
   feature de interação 1D  (raw_score)
                 ↓
   SCORE_VECTORIZATION  →  [ base , feature ]  (vetor multivariado)
                 ↓
   CALIBRATION_MODEL_TRAINING (LogReg multivariado)
                 ↓
   log-LR contínuo
```

A decisão de projeto central: a raridade entra **na mesma regressão** que o
score base, como termo de interação — **não** numa segunda regressão em
cascata. Uma cascata fixaria o log-LR do score como input rígido, perderia a
chance de aprender a interação no mesmo passo e duplicaria a variância de
estimação.

---

## 2. Fundamentação forense

### 2.1 A tese

A instrução materializa uma tese única, explícita nos docstrings dos enums:

> **Raridade, sozinha, não é evidência de mesma origem. Ela só deve modular a
> *força* de uma similaridade observada.**

Uma face rara não diz, por si, que duas imagens são da mesma pessoa. O que ela
faz é alterar o **valor probatório** de uma similaridade já observada: duas
faces raras que batem com determinada similaridade constituem evidência mais
forte do que duas faces comuns que batem com a *mesma* similaridade.

### 2.2 Por que interação (produto) e não termo aditivo

Se a raridade entrasse como termo aditivo independente
($\beta_r\cdot\mathrm{rarity}$), o modelo trataria "ser raro" como evidência
por si — o que é forensicamente indefensável. A forma de **produto** com o
score base garante que a raridade só age **onde já há similaridade**:

$$
\mathrm{feature} = f(\mathrm{base}, s_0)\cdot \mathrm{rarity}.
$$

Quando o fator do score base é nulo, a raridade não contribui — exatamente o
comportamento desejado.

### 2.3 O ponto de evidência neutra $s_0$

$s_0$ é o valor do score base no qual a evidência é **neutra** (log-LR = 0): a
fronteira entre suporte a $H_p$ (mesma origem) e a $H_d$ (origens distintas).
A interação é **centrada** em $s_0$ — é a diferença $(\mathrm{base}-s_0)$, não o
score absoluto, que a raridade modula. $s_0$ deve ser **aprendido no domínio de
calibração e congelado** para validação e casework (§6, §12).

---

## 3. Entradas e saída

### 3.1 Recursos de entrada e saída

| Papel | Campo | Tipo | Conteúdo |
|---|---|---|---|
| **base** | `base_score_resource_id` | `SimilarityScoreDataset` | score puro por par (cosseno recomendado; PLDA experimental). Lê `raw_score`. |
| **modifier** | `modifier_score_resource_id` | `SimilarityScoreDataset` | tipicidade pareada — output de `TYPICALITY_PAIR_SCORING`. Ignorado em `HINGE_BASE` e `HINGE_BASE_NEGATIVE`. |
| **herança** | `inherit_s0_from_resource_id` | `SimilarityScoreDataset` | de onde herdar `s0` (só em `s0_mode=INHERIT`). |
| **saída** | `output_resource_id` | `SimilarityScoreDataset` | `raw_score` 1D = a feature derivada. |

### 3.2 A saída não é log-LR

O `raw_score` de cada par no dataset de saída é o **valor da feature**, uma
quantidade crua sem interpretação probatória direta. Ele é consumido pelo
`SCORE_VECTORIZATION`, que monta o vetor `[base, feature]`, e por um
`CALIBRATION_MODEL_TRAINING` com LogReg multivariado que aprende os
coeficientes. Só após a calibração existe log-LR.

### 3.3 Anti-recursão

O `base` **não pode** ser ele mesmo uma interaction feature — o executor
rejeita qualquer base cujo metadata tenha `is_interaction_feature=True`. O base
deve ser score puro (cosseno/PLDA).

### 3.4 Validação de família do modifier

O modifier deve ser da **família tipicidade**: metadata com
`score_kind=typicality_pair` ou `score_type` em `{TYPICALITY_PERCENTILE,
MEAN_KNN_SIMILARITY, LOG_DENSITY_ESTIMATE}` (este último é o antigo
`LOG_DENSITY_KNN`, renomeado). Caso contrário, erro. Além disso, o modifier
precisa trazer `score_type` no metadata — sem ele a inversão
tipicidade→raridade fica ambígua e a execução falha.

---

## 4. Pipeline de execução

A ordem exata executada por `execute()`:

1. **Valida** a instrução (estrutural + plugável + dinâmica + específica).
2. **Busca o base** e rejeita se for recursivo.
3. **Busca o modifier** (exceto nos modos só-hinge, `HINGE_BASE` e
   `HINGE_BASE_NEGATIVE`) e valida a família tipicidade.
4. **Alinha os pares** entre base e modifier (§9): chave orientada vs
   frozenset, detecção de duplicatas, reordenação, consistência de
   `same_person` (fora de casework).
5. **Monta os vetores** `base_scores` e `modifier_raw` alinhados.
6. **Converte tipicidade → raridade** (§7).
7. **Resolve `s0`** conforme `s0_mode` (§6).
8. **Resolve `r0`** e, se `center_rarity=True`, faz `rarity ← rarity − r0` (§8).
9. **Constrói a feature** conforme `interaction_mode` (§5).
10. **Calcula diagnósticos**: `s0` dentro do intervalo `[min, max]` dos scores?
    fração de pares acima de `s0`? fração do hinge ativa? Emite WARNING se `s0`
    está fora do intervalo (feature quase toda zero ou quase toda ativa).
11. **Monta o dataset de saída** reusando probe/reference/identidades do base;
    em casework zera `same_person`.
12. Persiste o **metadata de auditoria** (§14) e registra o recurso.

Qualquer exceção no caminho vira um resultado de falha com a mensagem do erro.

---

## 5. Modos de interação

O campo `interaction_mode` (`ScoreInteractionMode`) define a fórmula. Seja a
**hinge** $h = \max(\mathrm{base}-s_0,\,0)$:

| Modo | Fórmula | Uso |
|---|---|---|
| `ONE_SIDED_POSITIVE` (default) | $\mathrm{feature} = \max(\mathrm{base}-s_0,\,0)\cdot \mathrm{rarity} = h\cdot\mathrm{rarity}$ | conservador/defensável |
| `LINEAR` | $\mathrm{feature} = (\mathrm{base}-s_0)\cdot \mathrm{rarity}$ | bilateral; experimental |
| `HINGE_BASE` | $\mathrm{feature} = \max(\mathrm{base}-s_0,\,0) = h$ | controle do A/B; sem raridade |
| `ONE_SIDED_NEGATIVE` (V2.13) | $\mathrm{feature} = \max(s_0-\mathrm{base},\,0)\cdot \mathrm{rarity} = h_-\cdot\mathrm{rarity}$ | espelho do positivo abaixo de $s_0$; experimental |
| `HINGE_BASE_NEGATIVE` (V2.13) | $\mathrm{feature} = \max(s_0-\mathrm{base},\,0) = h_-$ | companheira do modelo bilateral; sem raridade |

### 5.1 `ONE_SIDED_POSITIVE` (default)

A raridade só amplifica evidência **positiva** ($\mathrm{base}>s_0$). Abaixo de
$s_0$, a feature é zero — a raridade não modula evidência neutra ou negativa.
Forensicamente: uma face rara que **não** casa não vira "evidência forte
contra". É o default recomendado por ser o mais defensável.

### 5.2 `LINEAR` (bilateral)

A raridade modula em ambos os lados de $s_0$ — inclusive evidência negativa.
Matematicamente mais simples, mas pode produzir afirmações difíceis de
sustentar em corte quando o caso cai em região neutra ou negativa.
Experimental por construção.

### 5.3 `HINGE_BASE` (sem raridade)

Produz apenas a hinge, ignorando o modifier (por isso `modifier_score_resource_id`
é opcional neste modo). É a **feature-companheira** para montar o modelo
plenamente geral (§10) e o **controle do A/B**: se a hinge sozinha já bate o
baseline, o ganho veio da não-linearidade, não da raridade.

### 5.4 `ONE_SIDED_NEGATIVE` e `HINGE_BASE_NEGATIVE` (V2.13)

Espelhos exatos dos modos positivos para a região **abaixo** de $s_0$, com
$h_- = \max(s_0-\mathrm{base},\,0)$. `ONE_SIDED_NEGATIVE` faz a raridade modular
só a evidência **negativa**; `HINGE_BASE_NEGATIVE` produz só a hinge negativa e,
como `HINGE_BASE`, ignora o modifier. Pareados com os positivos, formam o modelo
bilateral plenamente geral $[\,\mathrm{base},\,h,\,h_-,\,h\cdot\mathrm{rarity},\,
h_-\cdot\mathrm{rarity}\,]$, e valem as identidades $h + h_- =
|\mathrm{base}-s_0|$ e $h\cdot\mathrm{rarity} + h_-\cdot\mathrm{rarity} =
|\mathrm{base}-s_0|\cdot\mathrm{rarity}$. `ONE_SIDED_NEGATIVE` é **apenas
experimental**: a afirmação de que a raridade reforça evidência contra a mesma
origem ainda exige validação. Nesses modos, `fraction_hinge_active` é a fração
de pares **abaixo** de $s_0$.

---

## 6. Resolução de s0 — o ponto de evidência neutra

O campo `s0_mode` (`S0Mode`) define como $s_0$ é obtido.

### 6.1 `AUTO_LOGREG` (default)

Ajusta um LogReg **univariado** `base → label` no próprio dataset e calcula o
ponto onde a reta logística cruza o logit zero:

$$
\mathrm{logit}\,p = \mathrm{intercept} + \mathrm{coef}\cdot\mathrm{base},
\qquad
s_0 = -\frac{\mathrm{intercept}}{\mathrm{coef}}.
$$

É um **calibrador auxiliar** (config `LogRegS0Config`, ver Apêndice C) — não
necessariamente igual ao calibrador final do projeto. Guard-rails:

- erro se $|\mathrm{coef}| < 10^{-6}$ (o score não separa as classes);
- erro se $\mathrm{coef} < 0$ ($s_0$ perde a interpretação "maior base = maior
  evidência $H_p$" — talvez o base seja distância, não similaridade);
- exige `same_person` binário e duas classes presentes;
- **bloqueado em casework** (sem ground truth);
- WARNING se há menos de 1000 pares (s0 instável).

Emite ainda um WARNING de **CV-leakage**: como $s_0$ foi estimado no conjunto
completo, métricas de CV interno do calibrador final neste mesmo dataset podem
ser otimistas — use holdout externo com `INHERIT` para avaliação honesta.

### 6.2 `MANUAL`

Usa `s0_manual_value` diretamente. Útil para fixar $s_0$ vindo de literatura,
de calibrador externo já analisado ou de regra de negócio explícita.

### 6.3 `INHERIT`

Lê `interaction_s0` do metadata de um `SimilarityScoreDataset` produzido por
uma execução anterior desta instrução (`inherit_s0_from_resource_id`). É
**obrigatório em validação e casework**: preserva a definição de "evidência
neutra" aprendida na calibração. O executor compara o base herdado vs o atual
(tipo, modelo, coluna de embedding) e emite WARNING em mismatch (promovível a
ERRO em versão futura).

### 6.4 O protocolo de congelamento

Refit silencioso de $s_0$ em validação ou casework **recalibra a definição de
evidência neutra e contamina o LR final**. O protocolo defensável:

| Fase | `s0_mode` |
|---|---|
| Calibração | `AUTO_LOGREG` (aprende e grava $s_0$ no metadata) |
| Validação A/B | `INHERIT` (aponta para a execução de calibração) |
| Casework | `INHERIT` (obrigatório; `AUTO_LOGREG` é bloqueado) |

---

## 7. Conversão tipicidade → raridade

O modifier carrega **tipicidade** (quão típica a face é); a feature precisa de
**raridade** (o oposto). A conversão (`invert_rarity=True`, default) depende do
`score_type` do modifier:

| `score_type` | escala de entrada | transform (invert=True) | label |
|---|---|---|---|
| `TYPICALITY_PERCENTILE` | já em $[0,1]$ | $1 - v$ | `1 - percentile_0_1` |
| `MEAN_KNN_SIMILARITY` | $[-1,1]$ | $-v$ | `−mean_knn_similarity` |
| `LOG_DENSITY_ESTIMATE` (antigo `LOG_DENSITY_KNN`) | $\mathbb{R}$ | $-v$ | `−log_density_knn` |

A inversão garante que **número maior = face mais rara**. Com
`invert_rarity=False`, é passthrough da tipicidade — útil quando se quer que o
sinal do coeficiente do LogReg reflita tipicidade diretamente (não-cosmético no
caso unilateral com offset, mas permitido para auditoria).

> **Nota histórica (bugfix).** O `TYPICALITY_PAIR_SCORING` já normaliza o
> percentil para $[0,1]$ (dividindo por 100). Uma versão anterior dividia por
> 100 de novo aqui, produzindo raridade saturada em ~0,99 para todos os pares e
> neutralizando a feature. Corrigido: o percentil é usado como recebido.

---

## 8. Centralização da raridade (r0) e o princípio de Tukey

Quando `center_rarity=True`, a feature usa $(\mathrm{rarity}-r_0)$ no lugar de
$\mathrm{rarity}$:

$$
\mathrm{feature} = h\cdot(\mathrm{rarity}-r_0).
$$

$r_0$ é resolvido em paralelo a $s_0$ (mesmo `s0_mode`): em `AUTO_LOGREG`,
$r_0 = \mathrm{mean}(\mathrm{rarity})$ no dataset atual; em `INHERIT`, lê
`interaction_r0` da mesma fonte. `MANUAL` não suporta centralização
($r_0$ ficaria indefinido) — a combinação `center_rarity=True` +
`s0_mode=MANUAL` dá erro na execução. Sem efeito nos modos só-hinge
(`HINGE_BASE`, `HINGE_BASE_NEGATIVE`).

### 8.1 Por que centrar — princípio hierárquico de Tukey

No modelo a jusante $[\,\mathrm{base},\,h,\,h\cdot(\mathrm{rarity}-r_0)\,]$, a
centralização torna a decomposição **interpretável**:

- o coeficiente de $h$ = **efeito-base** da similaridade numa face de raridade
  *típica* (raridade média);
- o coeficiente de $h\cdot(\mathrm{rarity}-r_0)$ = **modulação específica** pela
  raridade, em torno dessa face típica.

Sem centrar, os dois termos ficam correlacionados e a leitura "quanto é
similaridade × quanto é raridade" perde nitidez. $r_0$ é a **âncora de raridade
neutra** — não um corte rígido, mas a média da raridade do conjunto. (É o
refinamento correto da intuição de "raridade em torno de um ponto central":
contínuo, não dicotomizado.)

---

## 9. Alinhamento de pares

Base e modifier precisam descrever **o mesmo conjunto de pares**. O executor
reordena o modifier para casar par-a-par com o base.

### 9.1 Chave de alinhamento

- **orientada** — a chave é a tupla `(probe_hash, reference_hash)` — quando o
  combiner do modifier é **unilateral** (`REFERENCE_ONLY`/`PROBE_ONLY`) — a
  ordem do par importa;
- **`frozenset`** $\{\text{probe},\,\text{reference}\}$ caso contrário
  (combiner simétrico).

### 9.2 Defesas

- **Cardinalidade**: erro se base e modifier têm número de pares diferente
  (sinaliza `PAIR_MINING` em apenas um dos lados).
- **Duplicatas**: detecta chaves repetidas em qualquer lado e levanta erro
  explícito (evita silenciamento por sobrescrita em dict).
- **Pares ausentes**: erro se algum par do base não existe no modifier (com
  exemplos das chaves faltantes).
- **Consistência de `same_person`** (fora de casework): exige rótulo idêntico
  par-a-par — divergência indica splits cruzados ou corrupção (chaves alinharam
  por hash mas labels não batem).

---

## 10. O modelo multivariado a jusante

A feature é só uma coluna. O modelo forense completo nasce no
`SCORE_VECTORIZATION` + `CALIBRATION_MODEL_TRAINING`. O caso mínimo:

$$
\log\mathrm{LR} = \beta_0 + \beta_1\cdot\mathrm{base} + \beta_2\cdot\mathrm{feature}.
$$

### 10.1 O modelo plenamente geral

Combinando duas instâncias da instrução — uma `ONE_SIDED_POSITIVE` e uma
`HINGE_BASE` — no `SCORE_VECTORIZATION`, monta-se
$[\,\mathrm{base},\,h,\,h\cdot\mathrm{rarity}\,]$:

$$
\log\mathrm{LR} = \beta_0 + \beta_1\cdot\mathrm{base} + \beta_2\cdot h + \beta_3\cdot(h\cdot\mathrm{rarity}).
$$

Propriedades:

- **invariante à definição exata de raridade** (um offset constante na raridade
  é absorvido pelo termo $h$);
- o termo $h$ é o **controle do A/B**: se $[\mathrm{base},\,h]$ já bate o
  baseline, o ganho veio da não-linearidade (hinge), não da raridade. Só se
  $\beta_3$ for significativo a raridade está, de fato, contribuindo.

### 10.2 Por que multivariado e não cascata

Treinar uma segunda regressão sobre $[\log\mathrm{LR}_{\text{calibrado}},\,
\mathrm{percentil}]$ (cascata) fixa o log-LR como input rígido, não aprende a
interação raridade×similaridade no mesmo passo e empilha variância de
estimação. O modelo multivariado de uma só regressão é parcimonioso e
defensável: um calibrador, uma função de log-LR, interação explícita.

---

## 11. Double-counting com PLDA

Se o `base` é **PLDA**, há um risco específico. A covariância entre-classes
$B$ do PLDA **já encoda tipicidade** (mede quão distante da média da população
a face está). Combinar PLDA + interação de raridade pode **contar a tipicidade
duas vezes**.

Recomendação: com base PLDA, exija o **A/B explícito** (compare o modelo só-PLDA
contra PLDA + interação — M7 vs M8 da matriz A/B do Apêndice H.5) para verificar
se a raridade externa ainda agrega informação além do que $B$ já captura. Com
base **cosseno**, o risco não existe — o cosseno não modela tipicidade — e a
interação tende a agregar genuinamente.

---

## 12. Casework e o protocolo de congelamento

Com `is_casework=True`:

- o executor **zera `same_person`** em todos os samples e marca
  `metadata.is_casework=True` (não há ground truth no caso real);
- `s0_mode=AUTO_LOGREG` é **proibido** (sem rótulos para ajustar o calibrador
  auxiliar) — o validador rejeita a combinação;
- `s0_mode=INHERIT` é obrigatório, apontando para a execução de calibração.

Isso garante que a definição de "evidência neutra" usada no caso é exatamente a
aprendida na calibração — não uma redefinição silenciosa sobre os dados do
próprio caso.

---

## 13. Validações e guard-rails

| Validação | Quando | Severidade |
|---|---|---|
| `base_score_resource_id` definido e presente | sempre | erro |
| `modifier_score_resource_id` definido e presente | exceto `HINGE_BASE`/`HINGE_BASE_NEGATIVE` | erro |
| `output_resource_id` definido | sempre | erro |
| base é `SimilarityScoreDataset` e não-recursivo | sempre | erro |
| modifier é da família tipicidade | exceto `HINGE_BASE`/`HINGE_BASE_NEGATIVE` | erro |
| `MANUAL` exige `s0_manual_value` | `s0_mode=MANUAL` | erro |
| `INHERIT` exige recurso com `interaction_s0` no metadata | `s0_mode=INHERIT` | erro |
| base herdado ≠ base atual (tipo/modelo/coluna) | `s0_mode=INHERIT` | warning |
| `AUTO_LOGREG` + `is_casework` | combinação | erro |
| `AUTO_LOGREG` com < 1000 pares | `s0_mode=AUTO_LOGREG` | warning |
| `MIN_CALIBRATION_PAIRS` sobre o base | `s0_mode=AUTO_LOGREG` (default persistente) | configurável |
| `ScoreVectorPairCompatibilityRule` base × modifier | dinâmica (exceto `HINGE_BASE`/`HINGE_BASE_NEGATIVE`) | erro |
| `center_rarity=True` com `s0_mode=MANUAL` | runtime | erro |

Além disso, em runtime: `coef ≈ 0` ou `coef < 0` no `AUTO_LOGREG` → erro; `s0`
fora do intervalo dos scores → warning de extrapolação.

---

## 14. Metadata de auditoria

O dataset de saída persiste, entre outras chaves: `is_interaction_feature`,
`interaction_mode`, `interaction_s0`, `interaction_s0_mode`,
`interaction_s0_coef`/`interaction_s0_intercept`/`interaction_s0_C`/
`interaction_s0_coef_sign`, `interaction_s0_inside_score_range`,
`interaction_s0_source_resource_id`, `interaction_s0_training_method`,
`fraction_scores_above_s0`, `fraction_hinge_active`, `invert_rarity`,
`rarity_transform`, `modifier_score_type`, `modifier_combiner`,
`alignment_orientation`, `base_source_resource_id`,
`modifier_source_resource_id`, `base_score_type`/`base_score_name`/
`base_similarity_model_resource_id`/`base_embedding_column`,
`interaction_base_score_min`/`interaction_base_score_max`, `is_casework`,
`score_kind` (= `interaction_feature`), e — quando `center_rarity=True` —
`interaction_r0`/`interaction_r0_mode`/`interaction_r0_source_resource_id`.

Esse bloco é o que permite reproduzir e defender a feature: $s_0$ usado, como
foi obtido, a fração de pares onde o hinge está ativo e se há extrapolação.

---

## 15. Exemplo numérico trabalhado

Cosseno como base, `s0 = 0,30`, modo `ONE_SIDED_POSITIVE`. Dois pares com
**cosseno idêntico = 0,55**:

| Par | faces | $\mathrm{rarity}$ | $h=\max(0{,}55-0{,}30,\,0)$ | $\mathrm{feature}=h\cdot\mathrm{rarity}$ |
|---|---|---|---|---|
| A | comuns | 0,10 | 0,25 | 0,025 |
| B | raras | 0,90 | 0,25 | 0,225 |

Com o calibrador a jusante
$\log\mathrm{LR}=\beta_0+\beta_1\cdot 0{,}55+\beta_2\cdot\mathrm{feature}$ e
$\beta_2>0$, o par B (faces raras) recebe **log-LR maior** que o par A, apesar
do cosseno idêntico. É a tese funcionando: mesma similaridade, mas o match entre
faces raras é mais forte.

Um terceiro par, com cosseno 0,20 (abaixo de $s_0$): $h=0$, logo
$\mathrm{feature}=0$ — a raridade **não** infla nada, independentemente de quão
raras as faces sejam. Esse é o comportamento conservador do
`ONE_SIDED_POSITIVE`.

---

## 16. Boas práticas resumidas

- **Default conservador**: `interaction_mode=ONE_SIDED_POSITIVE`.
- **Congele $s_0$**: `AUTO_LOGREG` na calibração; `INHERIT` em validação e
  casework. Nunca refit silencioso fora da calibração.
- **Base cosseno** é o caso limpo; **base PLDA** exige A/B contra
  double-counting.
- **Modelo plenamente geral**: combine com `HINGE_BASE` para separar
  não-linearidade de raridade e ficar invariante à definição de raridade.
- **Centralize a raridade** (`center_rarity=True`) quando quiser interpretar
  separadamente o efeito-base e a modulação por raridade.
- **Avalie em holdout externo** — métricas no mesmo dataset do `AUTO_LOGREG`
  são otimistas.
- **Não dicotomize a raridade** (sem cortes rígidos em "percentil 50"); a
  raridade entra contínua e centrada em $r_0$.

---

## 17. Perguntas frequentes

**A feature é o log-LR final?**
Não. É uma feature 1D crua. O log-LR vem do calibrador multivariado que consome
`[base, feature]`.

**Por que centrar em $s_0$ e não usar o score absoluto?**
Porque é a posição **relativa à evidência neutra** que a raridade deve modular.
Multiplicar a raridade pelo score absoluto misturaria o offset arbitrário do
score com o efeito de amplificação.

**Devo cortar a raridade em 50% e substituir por valores fixos?**
Não. Isso reintroduz o cliff-edge que a calibração contínua existe para evitar.
A raridade entra **contínua**; se quiser uma âncora central, use
`center_rarity=True` (que ancora em $r_0=\mathrm{mean}(\mathrm{rarity})$).

**Interação na mesma regressão ou segunda regressão em cascata?**
Mesma regressão (multivariada). A cascata fixa o log-LR como input rígido e
duplica variância de estimação — ver §10.2.

**Posso usar PLDA como base?**
Pode, mas é experimental: risco de double-counting de tipicidade. Faça o A/B
explícito (§11).

**O que acontece com a raridade quando o score está abaixo de $s_0$?**
Em `ONE_SIDED_POSITIVE` e `HINGE_BASE`, a feature é zero — raridade não age. Em
`LINEAR`, a raridade modula também essa região (bilateral). Os modos
`ONE_SIDED_NEGATIVE`/`HINGE_BASE_NEGATIVE` agem **só** nessa região (§5.4).

**Por que `AUTO_LOGREG` é bloqueado em casework?**
Porque não há ground truth para ajustar o calibrador auxiliar. Em casework,
$s_0$ tem que vir congelado via `INHERIT`.

---

## 18. Glossário

- **base / score base** — score puro de similaridade (cosseno/PLDA) por par.
- **modifier** — dataset de tipicidade pareada (output de `TYPICALITY_PAIR_SCORING`).
- **tipicidade** — quão típica é uma face na população de referência.
- **raridade** — o complemento da tipicidade (`invert_rarity`).
- **$s_0$** — ponto de evidência neutra (log-LR = 0) no eixo do score base.
- **$r_0$** — âncora de raridade neutra (média da raridade), usada se
  `center_rarity=True`.
- **hinge** — $h=\max(\mathrm{base}-s_0,\,0)$.
- **feature de interação** — `raw_score` da saída; $f(\mathrm{base},s_0)\cdot
  \mathrm{rarity}$.
- **combiner** — como o `TYPICALITY_PAIR_SCORING` combinou as tipicidades das
  duas faces (simétrico ou unilateral).
- **double-counting** — contar tipicidade duas vezes (PLDA $B$ + raridade).

---

## 19. Referências

- Morrison, G. S. (2013/2018). *Tutorial on logistic-regression calibration and
  fusion of likelihood-ratio scores.*
- Tukey, J. W. (1977). *Exploratory Data Analysis* (princípio hierárquico de
  interações: incluir efeitos principais antes de interações; centralizar para
  interpretabilidade).
- Referências internas do FaceAudit:
  - `docs/similaridade_faceaudit.md` — score base.
  - `docs/calibracao_faceaudit.md` — calibração que consome a feature.
  - Docstrings de `score_interaction_feature_instruction_executor.py`,
    `score_interaction_mode.py`, `s0_mode.py`, `logreg_s0_config.py`.

---

## Apêndice A — Fórmulas de referência

**Hinge:**

$$
h = \max(\mathrm{base}-s_0,\,0).
$$

**Modos de interação:**

- `ONE_SIDED_POSITIVE`: $\mathrm{feature}=h\cdot \mathrm{rarity}$.
- `LINEAR`: $\mathrm{feature}=(\mathrm{base}-s_0)\cdot \mathrm{rarity}$.
- `HINGE_BASE`: $\mathrm{feature}=h$.
- `ONE_SIDED_NEGATIVE`: $\mathrm{feature}=h_-\cdot \mathrm{rarity}$, com $h_-=\max(s_0-\mathrm{base},\,0)$.
- `HINGE_BASE_NEGATIVE`: $\mathrm{feature}=h_-$.

**Centralização da raridade (`center_rarity=True`):**

$$
\mathrm{feature}=h\cdot(\mathrm{rarity}-r_0),\qquad r_0=\mathrm{mean}(\mathrm{rarity}).
$$

com $r_0$ calculado no dataset atual quando `s0_mode=AUTO_LOGREG` (ou herdado
via `INHERIT`).

**$s_0$ via `AUTO_LOGREG`:**

$$
\mathrm{logit}\,p=\mathrm{intercept}+\mathrm{coef}\cdot\mathrm{base},\qquad
s_0=-\frac{\mathrm{intercept}}{\mathrm{coef}}.
$$

**Conversão tipicidade → raridade (`invert_rarity=True`):**

- `TYPICALITY_PERCENTILE` (com $v\in[0,1]$): $\mathrm{rarity}=1-v$.
- `MEAN_KNN_SIMILARITY` / `LOG_DENSITY_ESTIMATE`: $\mathrm{rarity}=-v$.

**Modelo multivariado a jusante (caso geral):**

$$
\log\mathrm{LR}=\beta_0+\beta_1\cdot\mathrm{base}+\beta_2\cdot h+\beta_3\cdot(h\cdot \mathrm{rarity}).
$$

---

## Apêndice B — `ScoreInteractionFeatureInstruction`

Arquivo: `faceaudit/models/instructions/score_interaction_feature_instruction.py`.

| Campo | Tipo | Default | Interpretação | Restrição / observação |
|---|---|---|---|---|
| `base_score_resource_id` | `Optional[str]` | `None` | score base (cosseno/PLDA) | obrigatório; não pode ser interaction feature |
| `modifier_score_resource_id` | `Optional[str]` | `None` | tipicidade pareada | obrigatório exceto em `HINGE_BASE`/`HINGE_BASE_NEGATIVE` |
| `interaction_mode` | `ScoreInteractionMode` | `ONE_SIDED_POSITIVE` | fórmula da feature | ver Apêndice D |
| `s0_mode` | `S0Mode` | `AUTO_LOGREG` | resolução de `s0` | ver Apêndice D |
| `s0_manual_value` | `Optional[float]` | `None` | `s0` fixo | obrigatório em `MANUAL` |
| `inherit_s0_from_resource_id` | `Optional[str]` | `None` | fonte de herança de `s0` | obrigatório em `INHERIT` |
| `s0_logreg_config_override` | `Optional[LogRegS0Config]` | `None` | config do LogReg auxiliar | só em `AUTO_LOGREG`; Apêndice C |
| `invert_rarity` | `bool` | `True` | tipicidade → raridade | por `score_type` do modifier |
| `center_rarity` | `bool` | `False` | usa `(rarity − r0)` | sem efeito nos modos só-hinge; `MANUAL` não suporta |
| `output_resource_id` | `Optional[str]` | `None` | dataset de saída | obrigatório |
| `output_dataset_name` | `Optional[str]` | `None` | nome amigável | — |
| `is_casework` | `bool` | `False` | zera `same_person` | exige `s0_mode=INHERIT` |

Specs de formulário: `base_score_resource_id` dispara rebuild (a regra default
`MIN_CALIBRATION_PAIRS` depende dele). O dropdown do modifier só aparece fora dos
modos só-hinge (`HINGE_BASE`, `HINGE_BASE_NEGATIVE`); o de herança só em `INHERIT`; o campo `s0_manual_value` só em
`MANUAL`.

---

## Apêndice C — `LogRegS0Config`

Arquivo: `faceaudit/models/calibration/logreg_s0_config.py`. Config do
**calibrador auxiliar univariado** usado em `AUTO_LOGREG`.

| Campo | Tipo | Default | Restrição |
|---|---|---|---|
| `C` | `float` | `1.0` | `> 0` |
| `solver` | `str` | `"liblinear"` | solver do sklearn |
| `max_iter` | `int` | `1000` | `>= 1` |
| `class_weight` | `Optional[str]` | `"balanced"` | `"balanced"` ou `None` |

É **auxiliar** — não precisa coincidir com o calibrador final do projeto (que
pode usar GridSearch de `C`). Persistido no metadata (`interaction_s0_C`,
`interaction_s0_training_method`).

---

## Apêndice D — Enums

**`ScoreInteractionMode`** (`faceaudit/enums/score_interaction_mode.py`):

| Membro | Fórmula | `requires_modifier` |
|---|---|---|
| `ONE_SIDED_POSITIVE` (default) | $h\cdot\mathrm{rarity}$ | True |
| `LINEAR` | $(\mathrm{base}-s_0)\cdot\mathrm{rarity}$ | True |
| `HINGE_BASE` | $h$ | **False** |
| `ONE_SIDED_NEGATIVE` (V2.13) | $h_-\cdot\mathrm{rarity}$ | True |
| `HINGE_BASE_NEGATIVE` (V2.13) | $h_-$ | **False** |

`from_str` defaulta `ONE_SIDED_POSITIVE`. A property `requires_modifier`
controla se o validador exige o dropdown do modifier; `is_negative_side` marca
os dois modos espelhados e `is_hinge_only` agrupa `HINGE_BASE` e
`HINGE_BASE_NEGATIVE`.

**`S0Mode`** (`faceaudit/enums/s0_mode.py`):

| Membro | Como obtém `s0` | Restrição |
|---|---|---|
| `AUTO_LOGREG` (default) | $-\mathrm{intercept}/\mathrm{coef}$ de LogReg univariado | bloqueado em casework |
| `MANUAL` | `s0_manual_value` | exige o valor |
| `INHERIT` | `interaction_s0` de execução anterior | obrigatório em validação/casework |

`from_str` defaulta `AUTO_LOGREG`. Evolução futura mencionada no código:
`FROM_CALIBRATOR_RESOURCE` (não implementada na v1).

---

## Apêndice E — Metadata do dataset de saída

Chaves persistidas em `SimilarityScoreDataset.metadata` (auditoria):

| Chave | Conteúdo |
|---|---|
| `is_interaction_feature` | sempre `True` |
| `score_kind` | `"interaction_feature"` |
| `interaction_mode` | modo usado |
| `interaction_s0` | valor de `s0` |
| `interaction_s0_mode` | `auto_logreg` / `manual` / `inherit` |
| `interaction_s0_coef` / `_intercept` / `_C` / `_coef_sign` | diagnóstico do LogReg auxiliar (ou herdado) |
| `interaction_s0_training_method` | `auto_logreg` / `manual` / `inherit` |
| `interaction_s0_source_resource_id` | fonte (em `INHERIT`) |
| `interaction_s0_inside_score_range` | `s0` dentro de `[min, max]`? |
| `fraction_scores_above_s0` | fração de pares com `base > s0` |
| `fraction_hinge_active` | fração de pares com hinge ativa (acima de `s0` nos modos positivos, abaixo nos negativos, `1.0` em `LINEAR`) |
| `invert_rarity` / `rarity_transform` | tratamento da raridade |
| `modifier_score_type` / `modifier_combiner` | origem da tipicidade |
| `alignment_orientation` | `ordered` / `unordered` |
| `base_source_resource_id` / `modifier_source_resource_id` | ids dos recursos de entrada |
| `base_score_type` / `base_score_name` | identificação do base |
| `base_similarity_model_resource_id` / `base_embedding_column` | proveniência do base |
| `interaction_base_score_min` / `_max` | intervalo dos scores base |
| `is_casework` | modo casework |
| `interaction_r0` / `_mode` / `_source_resource_id` | só se `center_rarity=True` |

---

## Apêndice F — Checklist de revisão documental

- [ ] As cinco fórmulas de `interaction_mode` conferem com `_build_feature`?
- [ ] A fórmula `s0 = −intercept/coef` confere com `_resolve_s0` (`AUTO_LOGREG`)?
- [ ] A tabela de conversão tipicidade→raridade confere com `_to_rarity`
      (incluindo o bugfix do percentil já normalizado)?
- [ ] A centralização `h·(rarity − r0)` e a regra de $r_0$ por `s0_mode`
      conferem com `_resolve_r0`?
- [ ] As chaves de alinhamento (orientada vs frozenset) e as defesas (cardinalidade,
      duplicatas, `same_person`) conferem com `_align_pairs`?
- [ ] Os guard-rails do `AUTO_LOGREG` (coef≈0, coef<0, casework, <1000 pares)
      estão descritos?
- [ ] O bloco de metadata de auditoria está completo?
- [ ] O documento mantém a fronteira: não produz log-LR, só a feature?

---

## Apêndice G — Checklist de validação operacional

- [ ] O base é score puro (não interaction feature) e na direção certa
      (similaridade, não distância)?
- [ ] O modifier é output de `TYPICALITY_PAIR_SCORING` com `score_type` presente?
- [ ] Base e modifier descrevem o mesmo conjunto de pares (mesma orientação)?
- [ ] $s_0$ foi aprendido na calibração (`AUTO_LOGREG`) e **congelado**
      (`INHERIT`) em validação e casework?
- [ ] Em casework, `s0_mode=INHERIT` e `is_casework=True`?
- [ ] $s_0$ está dentro do intervalo dos scores (sem warning de extrapolação)?
- [ ] Há massa estatística suficiente (≥ 1000 pares) para o `AUTO_LOGREG`?
- [ ] A avaliação foi feita em holdout externo (não no dataset do `AUTO_LOGREG`)?
- [ ] Com base PLDA, o A/B contra double-counting foi feito?
- [ ] Se interpretabilidade importa, `center_rarity=True` e o modelo inclui o
      termo `HINGE_BASE`?

---

## Apêndice H — Protocolo de dupla interação (probe × referência)

A feature de interação da §5 usa **uma** raridade combinada por par. Quando o
caso é assimétrico (face questionada degradada vs documento limpo), pode-se
querer que o modelo aprenda **separadamente** quanto confiar na raridade de cada
papel. Isso se faz com **duas** features de interação — uma por papel — e a
estrutura a jusante:

$$
\log\mathrm{LR}=\beta_0+\beta_s\,s+\beta_q\,x_q+\beta_r\,x_r,
\qquad
x_q=(s-s_0)\,\tilde r_q,\quad x_r=(s-s_0)\,\tilde r_r,
$$

onde $\tilde r_q = r_q - r_{0q}$ e $\tilde r_r = r_r - r_{0r}$ são as raridades
de probe e referência (centradas, §H.2). O vetor de features é
$[\,s,\ x_q,\ x_r\,]$.

### H.1 Pipeline (só com instruções existentes)

A arquitetura é construível hoje, sem código novo, via os combinadores
unilaterais (`docs/tipicidade_pareada_faceaudit.md` §4.2):

```text
TYPICALITY_PAIR_SCORING (combiner=PROBE_ONLY,     score_type=TYPICALITY_PERCENTILE) → tipicidade_probe
TYPICALITY_PAIR_SCORING (combiner=REFERENCE_ONLY, score_type=TYPICALITY_PERCENTILE) → tipicidade_reference

SCORE_INTERACTION_FEATURE(base=cosseno, modifier=tipicidade_probe,     center_rarity=True, s0_mode=AUTO_LOGREG) → x_q
SCORE_INTERACTION_FEATURE(base=cosseno, modifier=tipicidade_reference, center_rarity=True, s0_mode=AUTO_LOGREG) → x_r

SCORE_VECTORIZATION(input_a=cosseno, input_b=x_q, extras=[x_r]) → [s, x_q, x_r]
CALIBRATION_MODEL_TRAINING(LogReg multivariado, standardize_features=True)
```

Em cross-população (lateral × frontal), treine **dois** modelos de tipicidade
e use o segundo no `reference_dataset` (`TYPICALITY_PAIR_SCORING`, §5). Com
`TYPICALITY_PERCENTILE`, cada percentil é relativo à sua própria população, o
que torna $r_q$ e $r_r$ comparáveis em $[0,1]$.

### H.2 Refinamento 1 — $s_0$ comum, mas $r_0$ por papel

- **$s_0$ é comum** às duas interações: ambas o estimam (`AUTO_LOGREG`) sobre o
  **mesmo** cosseno base, com a mesma config auxiliar — logo $s_0$ é idêntico
  por construção. (Um check defensivo $\lvert s_{0,q}-s_{0,r}\rvert \le
  \text{tol}$ é redundante mas inofensivo.)
- **$r_0$ é específico de cada papel.** As distribuições de raridade de probe e
  referência diferem (condições assimétricas), então
  $r_{0q}=\mathrm{mean}(r_q)$ e $r_{0r}=\mathrm{mean}(r_r)$ devem ser aprendidos
  **separadamente**. O executor já faz isso: em `AUTO_LOGREG`, `_resolve_r0`
  usa a média da raridade do **dataset atual**, então cada instância grava o seu.
- **Em validação e casework**, cada interação deve herdar de **sua própria**
  instância de calibração:
  `inherit_s0_from_resource_id = x_q_cal` para a probe e `= x_r_cal` para a
  referência. Como `r0` é lido do mesmo recurso que `s0` (`INHERIT`), apontar as
  duas validações para a **mesma** instância aplicaria a média de raridade de um
  papel ao outro — **erro a evitar**.

### H.3 Refinamento 2 — monotonicidade condicional (gate pericial)

A inclinação do log-LR em relação ao cosseno, com as raridades fixas, é:

$$
\frac{\partial \log\mathrm{LR}}{\partial s}
=\beta_s+\beta_q\,\tilde r_q+\beta_r\,\tilde r_r \quad(\text{modo LINEAR, todo o range}).
$$

No modo `ONE_SIDED_POSITIVE`, a feature é zero para $s<s_0$, então abaixo de
$s_0$ a inclinação é apenas $\beta_s$ (monótona se $\beta_s>0$) e a expressão
acima vale só para $s>s_0$. **Esse é um motivo teórico — não jurídico — a favor
de `ONE_SIDED_POSITIVE`**: ele garante monotonicidade na região de baixa
similaridade; o `LINEAR` fica em risco em todo o range.

**Problema:** com raridade centrada, $\tilde r_q$ e $\tilde r_r$ ficam
**negativos** para faces comuns ($r>\mathrm{mean}$). Logo, mesmo com
$\beta_s,\beta_q,\beta_r>0$, a inclinação pode ficar **negativa** para pares de
faces comuns — "mantendo a raridade fixa, aumentar a similaridade reduziu o
log-LR", indefensável.

**Check barato e exato:** a inclinação é **linear** em $(\tilde r_q,\tilde r_r)$,
então o mínimo está sempre num **canto** da caixa de raridade observada (ou da
caixa operacional/teórica). Para percentil centrado, a caixa é
$\tilde r \in [\,\mathrm{mean}(p)-1,\ \mathrm{mean}(p)\,]$. Avaliar os 4 cantos e
exigir inclinação $\ge 0$ em todos. **Em modo pericial, rejeitar o modelo se
houver inclinação condicional negativa na faixa operacional.**

> **Implementação operacional (V1.5).** O gate descrito acima é exatamente o
> que `ConditionalMonotonicityRule` (`faceaudit/validation_rules/conditional_monotonicity_rule.py`)
> executa para **arquiteturas declarativas de `LR_ARCHITECTURE_VECTORIZATION`**
> (15 arquiteturas, incluindo as ONE_FACE V2.14.1). A regra é invocada
> automaticamente pós-refit pelo `CalibrationModelTrainingInstructionExecutor`
> e avalia a inclinação `∂log_LR/∂s` (i) em **todos os pares observados em
> $D_{\text{cal}}$** (`observed_rows` — supera o "4 cantos" porque cobre a
> nuvem real) e (ii) nos **cantos do retângulo $[P_5, P_{95}]$** das raridades
> observadas (`central_box` — implementa diretamente a recomendação acima);
> desde a V2 (`RULE_VERSION = "2.0"`) avalia também os cantos de uma
> `reportable_box` opcional — o domínio operacional declarado, onde violação é
> CRITICAL. Status é persistido em `calibrator.metadata["monotonicity_audit"]`.
> Detalhes em `docs/conditional_monotonicity_rule_design.md`. O
> `PiecewiseLinearDerivativeProvider` já existe, mas cobre só o
> `LogisticRegressionCalibrationModel` 1D com `piecewise_linear_config`
> (expansão piecewise interna), e a regra só o audita quando adicionada como
> regra plugável — a auditoria automática pós-refit exige `architecture_state`.
> O encadeamento legado `SCORE_INTERACTION_FEATURE + SCORE_VECTORIZATION`
> (calibrador LogReg multivariado sem `architecture_state` nem
> `piecewise_enabled`) **não é coberto pela regra**: nele, o gate dos 4 cantos
> descrito acima continua sendo verificação manual do perito.

### H.4 Refinamento 3 — orientação na vetorização

O `SCORE_INTERACTION_FEATURE` usa alinhamento **orientado**
`(probe_hash, reference_hash)` para combinadores unilaterais, mas o
`SCORE_VECTORIZATION` alinha **sempre** por chave simétrica
`frozenset({probe_hash, reference_hash})`.

- **Hoje é seguro por construção**: $x_q$ e $x_r$ são gerados reusando os samples
  do **mesmo** cosseno âncora, herdando orientação idêntica; o `raw_score` já é o
  valor correto e a chave só decide a linha (o par não-ordenado é único). A troca
  $r_q\leftrightarrow r_r$ exigiria orientações **inconsistentes** entre as duas
  fontes — o que não ocorre no fluxo canônico.
- **Hardening recomendado** para uso operacional/auditável: a vetorização deve
  detectar `alignment_orientation`, usar chave orientada quando algum input for
  direcional, e persistir a orientação efetiva. Ainda não implementado: o
  `SCORE_VECTORIZATION` continua alinhando só por `frozenset`.

### H.5 Refinamento 4 — matriz A/B e diagnósticos

A escolha entre simétrico, bilateral e dupla interação é **empírica**, decidida
em holdout externo (não no dataset do `AUTO_LOGREG`, §6.1):

| ID | Features | Pergunta |
|---|---|---|
| M0 | $[s]$ | baseline |
| M1 | $[s,\ h]$ | não-linearidade pura (hinge) |
| M2 | $[s,\ h\cdot r_{\text{mean}}]$ | unilateral simétrica parcimoniosa |
| M3 | $[s,\ h,\ h\cdot r_{\text{mean}}]$ | separa hinge de raridade |
| M4 | $[s,\ (s-s_0)\,r_{\text{mean}}]$ | bilateral simétrica |
| M5 | $[s,\ x_q,\ x_r]$ | bilateral com papéis separados |
| M6 | $[s,\ h,\ h\,r_q,\ h\,r_r]$ | unilateral geral com papéis separados |
| M7 | $[\mathrm{PLDA}]$ | baseline PLDA |
| M8 | $[\mathrm{PLDA},\ x_q,\ x_r]$ | teste de redundância com PLDA |

Comparação central: **M5 vs M4 vs M0** (a tipicidade ajuda? o bilateral ajuda?
separar papéis agrega além da raridade média?). Sobre PLDA (M7 vs M8): com PLDA
as interações externas podem ser **parcialmente redundantes** com a covariância
between-class do backend — o efeito líquido se resolve por ablação externa; a
LogReg regularizada pode simplesmente atribuir peso pequeno (não há conclusão
automática de "dupla contagem").

**Diagnósticos de colinearidade** (porque $x_q$ e $x_r$ partilham o fator
$(s-s_0)$ e ficam correlacionados quando probe e referência têm raridade
parecida): `standardize_features=True` melhora estabilidade numérica e
regularização, mas **não** resolve identificabilidade. Persistir:
$\mathrm{corr}(x_q,x_r)$, VIF por feature, número de condição da matriz,
coeficientes por fold, desvio de $\beta_q,\beta_r$ entre folds, e bootstrap
**por identidade** (não por par — pares compartilham faces). Se M5 não superar
M4 externamente, o modelo simétrico é preferível por parcimônia.

### H.6 Refinamento 5 — `LINEAR` vs `ONE_SIDED_POSITIVE`

Ambos no A/B; decisão em holdout externo. Em empate, prefira
`ONE_SIDED_POSITIVE` por duas razões **teóricas** (não jurídicas): (1) a
justificativa probatória da raridade é mais forte para *matches* (probabilidade
de coincidência) que para não-matches; (2) garante monotonicidade abaixo de
$s_0$ (§H.3). Há ainda um risco prático: baixa similaridade pode vir de
qualidade/pose/falha do embedding (não de incompatibilidade real), e a raridade
nessa região pode estar contaminada pela qualidade — o que recomenda cautela com
o `LINEAR` na região negativa. Use `LINEAR` se vencer materialmente **e** passar
no gate de monotonicidade condicional.
