# Interação score × raridade no FaceAudit (`SCORE_INTERACTION_FEATURE`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `SCORE_INTERACTION_FEATURE` — a feature forense de
> interação entre um score de similaridade e a raridade derivada de
> tipicidade. Objetivo, fundamentação, configurações, fórmulas, exemplos e
> controles metodológicos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** repositório FaceAudit no commit `a60088e`
> (branch `claude/read-claude-md-SgzIV`). Campos, fórmulas e enums conferidos
> em `faceaudit/executors/instructions/score_interaction_feature_instruction_executor.py`,
> `faceaudit/models/instructions/score_interaction_feature_instruction.py`,
> `faceaudit/models/calibration/logreg_s0_config.py`,
> `faceaudit/enums/score_interaction_mode.py`,
> `faceaudit/enums/s0_mode.py`.

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

**Incluído:** a construção da feature de interação `(base − s0)·rarity`, os três
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
| **modifier** | `modifier_score_resource_id` | `SimilarityScoreDataset` | tipicidade pareada — output de `TYPICALITY_PAIR_SCORING`. Ignorado em `HINGE_BASE`. |
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
MEAN_KNN_SIMILARITY, LOG_DENSITY_KNN}`. Caso contrário, erro.

---

## 4. Pipeline de execução

A ordem exata executada por `execute()`:

1. **Valida** a instrução (estrutural + plugável + dinâmica + específica).
2. **Busca o base** e rejeita se for recursivo.
3. **Busca o modifier** (exceto em `HINGE_BASE`) e valida a família tipicidade.
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
| `LOG_DENSITY_KNN` | $\mathbb{R}$ | $-v$ | `−log_density_knn` |

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
($r_0$ ficaria indefinido). Sem efeito em `HINGE_BASE`.

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
contra PLDA + interação — modelos 8 e 9 da janela de Comparação) para verificar
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
| `modifier_score_resource_id` definido e presente | exceto `HINGE_BASE` | erro |
| `output_resource_id` definido | sempre | erro |
| base é `SimilarityScoreDataset` e não-recursivo | sempre | erro |
| modifier é da família tipicidade | exceto `HINGE_BASE` | erro |
| `MANUAL` exige `s0_manual_value` | `s0_mode=MANUAL` | erro |
| `INHERIT` exige recurso com `interaction_s0` no metadata | `s0_mode=INHERIT` | erro |
| base herdado ≠ base atual (tipo/modelo/coluna) | `s0_mode=INHERIT` | warning |
| `AUTO_LOGREG` + `is_casework` | combinação | erro |
| `AUTO_LOGREG` com < 1000 pares | `s0_mode=AUTO_LOGREG` | warning |
| `MIN_CALIBRATION_PAIRS` sobre o base | `s0_mode=AUTO_LOGREG` (default persistente) | configurável |
| `ScoreVectorPairCompatibilityRule` base × modifier | dinâmica (exceto `HINGE_BASE`) | erro |

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
`alignment_orientation`, `base_score_type`/`base_score_name`/
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
`LINEAR`, a raridade modula também essa região (bilateral).

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

**Centralização da raridade (`center_rarity=True`):**

$$
\mathrm{feature}=h\cdot(\mathrm{rarity}-r_0),\qquad r_0=\mathrm{mean}(\mathrm{rarity}).
$$

com $r_0$ obtido via `AUTO_LOGREG` (ou herdado via `INHERIT`).

**$s_0$ via `AUTO_LOGREG`:**

$$
\mathrm{logit}\,p=\mathrm{intercept}+\mathrm{coef}\cdot\mathrm{base},\qquad
s_0=-\frac{\mathrm{intercept}}{\mathrm{coef}}.
$$

**Conversão tipicidade → raridade (`invert_rarity=True`):**

- `TYPICALITY_PERCENTILE` (com $v\in[0,1]$): $\mathrm{rarity}=1-v$.
- `MEAN_KNN_SIMILARITY` / `LOG_DENSITY_KNN`: $\mathrm{rarity}=-v$.

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
| `modifier_score_resource_id` | `Optional[str]` | `None` | tipicidade pareada | obrigatório exceto em `HINGE_BASE` |
| `interaction_mode` | `ScoreInteractionMode` | `ONE_SIDED_POSITIVE` | fórmula da feature | ver Apêndice D |
| `s0_mode` | `S0Mode` | `AUTO_LOGREG` | resolução de `s0` | ver Apêndice D |
| `s0_manual_value` | `Optional[float]` | `None` | `s0` fixo | obrigatório em `MANUAL` |
| `inherit_s0_from_resource_id` | `Optional[str]` | `None` | fonte de herança de `s0` | obrigatório em `INHERIT` |
| `s0_logreg_config_override` | `Optional[LogRegS0Config]` | `None` | config do LogReg auxiliar | só em `AUTO_LOGREG`; Apêndice C |
| `invert_rarity` | `bool` | `True` | tipicidade → raridade | por `score_type` do modifier |
| `center_rarity` | `bool` | `False` | usa `(rarity − r0)` | sem efeito em `HINGE_BASE`; `MANUAL` não suporta |
| `output_resource_id` | `Optional[str]` | `None` | dataset de saída | obrigatório |
| `output_dataset_name` | `Optional[str]` | `None` | nome amigável | — |
| `is_casework` | `bool` | `False` | zera `same_person` | exige `s0_mode=INHERIT` |

Specs de formulário: `base_score_resource_id` dispara rebuild (a regra default
`MIN_CALIBRATION_PAIRS` depende dele). O dropdown do modifier só aparece fora de
`HINGE_BASE`; o de herança só em `INHERIT`; o campo `s0_manual_value` só em
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

`from_str` defaulta `ONE_SIDED_POSITIVE`. A property `requires_modifier`
controla se o validador exige o dropdown do modifier.

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
| `fraction_hinge_active` | fração de pares com hinge ativa |
| `invert_rarity` / `rarity_transform` | tratamento da raridade |
| `modifier_score_type` / `modifier_combiner` | origem da tipicidade |
| `alignment_orientation` | `ordered` / `unordered` |
| `base_score_type` / `base_score_name` | identificação do base |
| `base_similarity_model_resource_id` / `base_embedding_column` | proveniência do base |
| `interaction_base_score_min` / `_max` | intervalo dos scores base |
| `is_casework` | modo casework |
| `interaction_r0` / `_mode` / `_source_resource_id` | só se `center_rarity=True` |

---

## Apêndice F — Checklist de revisão documental

- [ ] As três fórmulas de `interaction_mode` conferem com `_build_feature`?
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
