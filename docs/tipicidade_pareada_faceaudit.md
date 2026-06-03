# Tipicidade pareada no FaceAudit (`TYPICALITY_PAIR_SCORING`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `TYPICALITY_PAIR_SCORING` — a aplicação de modelo(s)
> de tipicidade às faces de cada par e a combinação das duas tipicidades num
> score por par. Objetivo, fundamentação, métricas, combinadores, configurações
> e exemplos.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Fonte documental:** repositório FaceAudit no commit `55c3d8a`
> (branch `claude/read-claude-md-SgzIV`). Campos, fórmulas e enums conferidos
> em `faceaudit/executors/instructions/typicality_pair_scoring_instruction_executor.py`,
> `faceaudit/models/instructions/typicality_pair_scoring_instruction.py`,
> `faceaudit/enums/typicality_combiner.py`,
> `faceaudit/enums/typicality_score_type.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde a tipicidade pareada se encaixa](#1-onde-a-tipicidade-pareada-se-encaixa)
- [2. O que a instrução faz](#2-o-que-a-instrução-faz)
- [3. A métrica de tipicidade por face](#3-a-métrica-de-tipicidade-por-face)
- [4. Os combinadores](#4-os-combinadores)
- [5. Modelo único vs dois modelos (cross-população)](#5-modelo-único-vs-dois-modelos-cross-população)
- [6. Enumeração de pares — casamento com SIMILARITY_SCORING](#6-enumeração-de-pares--casamento-com-similarity_scoring)
- [7. Configurações](#7-configurações)
- [8. Casework](#8-casework)
- [9. Validações e guard-rails](#9-validações-e-guard-rails)
- [10. Metadata de saída](#10-metadata-de-saída)
- [11. Exemplo numérico](#11-exemplo-numérico)
- [12. Boas práticas resumidas](#12-boas-práticas-resumidas)
- [13. Perguntas frequentes](#13-perguntas-frequentes)
- [14. Glossário](#14-glossário)
- [15. Referências](#15-referências)
- [Apêndice A — Fórmulas dos combinadores](#apêndice-a--fórmulas-dos-combinadores)
- [Apêndice B — TypicalityPairScoringInstruction](#apêndice-b--typicalitypairscoringinstruction)
- [Apêndice C — Enums](#apêndice-c--enums)
- [Apêndice D — Checklist de validação operacional](#apêndice-d--checklist-de-validação-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência técnica sobre a instrução `TYPICALITY_PAIR_SCORING`: como
ela aplica um modelo de tipicidade às faces de cada par, como combina as duas
tipicidades num score por par, e como esse output se encaixa no pipeline.

### 0.2 Escopo

**Incluído:** a aplicação per-face do `KNNTypicalityModel`, a escolha da métrica
(`score_type`), os combinadores simétricos e unilaterais, o modo cross-população
de dois modelos, a enumeração de pares e o modo casework.

**Fora do escopo:**

- o **treino** do modelo de tipicidade (`KNN_TYPICALITY_MODEL_TRAINING`) — ver
  `docs/tipicidade_knn_faceaudit.md`;
- o **uso** da tipicidade pareada para modular a similaridade
  (`SCORE_INTERACTION_FEATURE`) — ver `docs/interacao_score_raridade_faceaudit.md`;
- o **empilhamento** com a similaridade (`SCORE_VECTORIZATION`) — ver
  `docs/vetorizacao_scores_faceaudit.md`.

### 0.3 Documentos complementares

| Documento | Relação |
|---|---|
| `docs/tipicidade_knn_faceaudit.md` | treina o modelo aplicado aqui |
| `docs/tipicidade_pareada_faceaudit.md` (este) | aplica o modelo a pares |
| `docs/interacao_score_raridade_faceaudit.md` | consome a tipicidade pareada (combinadores unilaterais) |
| `docs/vetorizacao_scores_faceaudit.md` | empilha a tipicidade pareada com a similaridade |

---

## 1. Onde a tipicidade pareada se encaixa

```text
TRAINED_TYPICALITY_MODEL (de KNN_TYPICALITY_MODEL_TRAINING)
  +  dataset(s) probe / reference
        ↓
TYPICALITY_PAIR_SCORING   ← ESTE DOCUMENTO
        ↓
SimilarityScoreDataset (score_kind = typicality_pair, um raw_score por par)
        ↓
   ┌──────────────────────────────┬─────────────────────────────┐
   ↓                              ↓
SCORE_INTERACTION_FEATURE     SCORE_VECTORIZATION
(modula a similaridade)       (empilha como feature)
        ↓                              ↓
              calibrador multivariado → log-LR
```

O output é um `SimilarityScoreDataset` na **mesma ordem de pares** que o
`SIMILARITY_SCORING` da mesma combinação de datasets — é isso que permite fundi-los
a jusante par-a-par.

---

## 2. O que a instrução faz

Para cada par $(probe, reference)$:

1. computa a **tipicidade individual** de cada face, aplicando o modelo treinado
   (uma chamada batch por dataset);
2. **combina** as duas tipicidades num único `raw_score` via o combinador
   escolhido;
3. registra o par num `SimilarityScoreDataset`, guardando as duas tipicidades
   individuais em `sample.metadata` (`probe_typicality`, `reference_typicality`).

A tipicidade individual é computada uma vez por face (vetorizado) e reusada em
todos os pares daquela face — não há recomputação por par.

---

## 3. A métrica de tipicidade por face

O campo `score_type` (`TypicalityScoreType`) escolhe qual das três métricas do
modelo é usada como tipicidade individual antes de combinar:

| `score_type` | Valor por face | Escala | Uso |
|---|---|---|---|
| `MEAN_KNN_SIMILARITY` (default) | $T_k$ bruto | $[-1, 1]$ | intuitivo; escala depende da população |
| `TYPICALITY_PERCENTILE` | percentil $/100$ | $[0, 1]$ | comparável entre populações; recomendado para vetorizar |
| `LOG_DENSITY_KNN` | log-densidade | $\mathbb{R}$ | LR generativo; padronizar antes do LogReg |

> **Normalização importante:** em `TYPICALITY_PERCENTILE`, o executor divide o
> percentil $[0,100]$ por 100, entregando a tipicidade já em $[0,1]$ — escala
> parecida com o cosseno, o que facilita o LogReg sem padronização. (É por isso
> que o `SCORE_INTERACTION_FEATURE` trata o percentil como já normalizado e
> aplica `1 − v` direto, sem dividir por 100 de novo.)

A escolha de `score_type` propaga para o metadata do output e é lida pelo
`SCORE_INTERACTION_FEATURE` para decidir a conversão tipicidade → raridade.

---

## 4. Os combinadores

O campo `combiner` (`TypicalityCombiner`) define como as tipicidades de probe
($t_p$) e reference ($t_r$) viram um único score por par. Há dois grupos.

### 4.1 Combinadores simétricos

Preservam $\mathrm{score}(A,B) = \mathrm{score}(B,A)$ — defensáveis quando a
hipótese forense não privilegia uma das faces.

| Combinador | Fórmula | Captura |
|---|---|---|
| `MEAN` (default) | $\tfrac{1}{2}(t_p + t_r)$ | raridade global do par |
| `MIN` | $\min(t_p, t_r)$ | a face **mais atípica** do par |
| `MAX` | $\max(t_p, t_r)$ | a face **mais típica** do par |
| `PRODUCT` | $t_p \cdot t_r$ | tipicidade conjunta (não-linear; exige inputs $\ge 0$, use percentil) |
| `ABS_DIFF` | $\lvert t_p - t_r\rvert$ | assimetria de raridade entre as faces |

### 4.2 Combinadores unilaterais

Quebram a simetria **propositalmente**, lendo a tipicidade de **um** dos papéis:

| Combinador | Resultado | Para que serve |
|---|---|---|
| `REFERENCE_ONLY` | $t_r$ | raridade só da referência (ex.: documento conhecido) |
| `PROBE_ONLY` | $t_p$ | raridade só do probe (face questionada) |

Existem para alimentar o `SCORE_INTERACTION_FEATURE` com `rarity_probe` e
`rarity_reference` separadamente — exatamente a "dupla interação" que separa
$\beta_q$ de $\beta_r$. Em **self-comparação** os papéis probe/reference são
convencionais (triangular superior $i<j$), então os modos unilaterais **emitem
WARNING** nesse caso — eles só fazem sentido em cross-comparação, onde os papéis
têm significado forense.

> **Detalhe de implementação:** os modos unilaterais **não** usam
> `combiner.combine()` — o executor lê a tipicidade do papel diretamente
> (`REFERENCE_ONLY → t_r`, `PROBE_ONLY → t_p`). Chamar `combine()` neles levanta
> erro.

### 4.3 A property `is_symmetric`

`REFERENCE_ONLY` e `PROBE_ONLY` têm `is_symmetric = False`; os demais, `True`.
Esse flag é o que o `SCORE_INTERACTION_FEATURE` usa para decidir o alinhamento
de pares: **orientado** `(probe_hash, reference_hash)` para combinadores
unilaterais, **frozenset** (simétrico) para os demais.

---

## 5. Modelo único vs dois modelos (cross-população)

A instrução aceita um ou dois modelos de tipicidade.

### 5.1 Modelo único (simétrico)

Só `typicality_model_probe_resource_id` é preenchido. Esse modelo é aplicado às
faces de **ambos** os datasets. É o caso quando probe e reference vêm da mesma
população de referência.

### 5.2 Dois modelos (cross-população)

`typicality_model_reference_resource_id` também é preenchido (só faz sentido em
cross-comparação, com `reference_dataset` definido). O modelo do probe é aplicado
às faces do probe; o do reference, às faces do reference. Caso canônico:
**cross-pose forense** — um modelo treinado em faces laterais aplicado ao probe
lateral, e um modelo treinado em faces frontais aplicado à referência frontal.
Quando o segundo modelo está vazio, o do probe é usado nos dois lados.

---

## 6. Enumeração de pares — casamento com SIMILARITY_SCORING

A ordem de enumeração é **idêntica** à do `SIMILARITY_SCORING`, por construção —
é isso que garante o casamento par-a-par a jusante:

- **self-comparison** (só `probe_dataset`): triangular superior,
  `for i: for j in range(i+1, N)` — $n(n-1)/2$ pares;
- **cross-comparison** (`probe` + `reference`): produto cartesiano,
  `for i in probe: for j in reference` — $n_p \times n_r$ pares.

Mesmo com a ordem garantida, o `SCORE_VECTORIZATION` e o
`SCORE_INTERACTION_FEATURE` revalidam o casamento por hash do par — então uma
divergência acidental é detectada, não silenciada.

---

## 7. Configurações

Campos do `TypicalityPairScoringInstruction`:

| Campo | Tipo | Default | O que faz |
|---|---|---|---|
| `typicality_model_probe_resource_id` | str | — | modelo aplicado às faces do probe (e do reference, se não houver segundo) |
| `typicality_model_reference_resource_id` | str? | None | modelo do reference (cross-população); só visível com `reference_dataset` |
| `probe_dataset_resource_id` | str | — | dataset 1 (probe) |
| `reference_dataset_resource_id` | str? | None | dataset 2 (reference); ausente → auto-comparação |
| `score_type` | `TypicalityScoreType` | `MEAN_KNN_SIMILARITY` | métrica de tipicidade por face |
| `combiner` | `TypicalityCombiner` | `MEAN` | como combinar as duas tipicidades |
| `output_scores_resource_id` | str | — | id do `SimilarityScoreDataset` de saída |
| `output_scores_name` | str? | None | nome amigável da saída |
| `is_casework` | bool | False | zera `same_person`; marca o dataset como caso |

A coluna de embedding vem do nível da estratégia (`context.embedding_column`).

---

## 8. Casework

Com `is_casework=True`, o executor **zera `same_person`** em todos os samples e
marca `metadata.is_casework=True` (e invalida os caches de contagem
gen/imp). É o mesmo padrão do `SIMILARITY_SCORING`, garantindo que o
`SCORE_VECTORIZATION` e a calibração tratem o caso de forma consistente com o
cosseno pareado.

---

## 9. Validações e guard-rails

| Validação | Quando | Severidade |
|---|---|---|
| modelo do probe definido e presente | sempre | erro |
| modelo do reference presente (se definido) | cross | erro |
| modelo do reference definido mas sem `reference_dataset` | — | warning (campo ignorado) |
| `probe_dataset_resource_id` definido e presente | sempre | erro |
| `reference_dataset_resource_id` presente (se definido) | — | erro |
| `output_scores_resource_id` definido | sempre | erro |
| combinador unilateral em self-comparação | runtime | warning |

Regras dinâmicas (intrínsecas, não removíveis): `MODEL_FITTED` nos modelos de
tipicidade (probe e, se houver, reference) e `EMBEDDING_DIM_MATCHES_MODEL` nos
datasets contra o modelo efetivo aplicado a cada um.

---

## 10. Metadata de saída

O `SimilarityScoreDataset` produzido carrega, no metadata do dataset:
`mode` (`self_comparison`/`cross_comparison`), `score_kind` (= `typicality_pair`),
`score_type`, `combiner`, `probe_dataset_name` (+ `reference_dataset_name` em
cross), `uses_dual_typicality_models` e `is_casework`. Cada `sample.metadata`
guarda `probe_typicality` e `reference_typicality` (as tipicidades individuais
antes da combinação).

O `score_kind = typicality_pair` é o que permite ao `SCORE_INTERACTION_FEATURE`
validar que o modifier é mesmo da família tipicidade.

---

## 11. Exemplo numérico

`score_type = TYPICALITY_PERCENTILE` (já em $[0,1]$), cross-comparação. Um par:

- probe (face questionada): percentil 5 → $t_p = 0{,}05$ (rara);
- reference (documento): percentil 20 → $t_r = 0{,}20$ (relativamente rara).

| Combinador | Resultado |
|---|---|
| `MEAN` | $0{,}5\cdot(0{,}05+0{,}20)=0{,}125$ |
| `MIN` | $0{,}05$ (a face mais atípica) |
| `MAX` | $0{,}20$ |
| `PRODUCT` | $0{,}05\cdot0{,}20=0{,}01$ |
| `ABS_DIFF` | $\lvert0{,}05-0{,}20\rvert=0{,}15$ |
| `PROBE_ONLY` | $0{,}05$ (só a questionada) |
| `REFERENCE_ONLY` | $0{,}20$ (só o documento) |

Para a "dupla interação" da §4.2: rode a instrução duas vezes, uma com
`PROBE_ONLY` (gera `rarity_probe`) e outra com `REFERENCE_ONLY` (gera
`rarity_reference`), e alimente as duas no `SCORE_INTERACTION_FEATURE`.

---

## 12. Boas práticas resumidas

- **Casamento com a similaridade**: gere a tipicidade pareada sobre os **mesmos
  datasets** (e na mesma orientação) do `SIMILARITY_SCORING` correspondente.
- **`TYPICALITY_PERCENTILE` para fundir com cosseno**: escala $[0,1]$
  consistente.
- **Combinadores unilaterais só em cross-comparação**: em self, os papéis são
  convencionais (warning).
- **Cross-população quando as condições diferem**: dois modelos (lateral × frontal).
- **Não minere pares só de um lado**: o conjunto de pares deve bater com o da
  similaridade, senão o casamento a jusante falha.

---

## 13. Perguntas frequentes

**Qual combinador usar?**
`MEAN` é o default razoável para raridade global. `MIN`/`MAX` caracterizam o
intervalo de tipicidade do par. Para a feature de interação que separa probe e
reference, use os unilaterais (`PROBE_ONLY`/`REFERENCE_ONLY`).

**Posso usar `PRODUCT` com `MEAN_KNN_SIMILARITY`?**
Cuidado: `PRODUCT` pressupõe inputs não-negativos para fazer sentido. Use
`TYPICALITY_PERCENTILE` (que é $[0,1]$) com `PRODUCT`.

**A tipicidade pareada é um log-LR?**
Não. É um score por par que será modulado pela similaridade ou empilhado como
feature. Sozinha, não é evidência de mesma origem.

**Por que a ordem dos pares importa?**
Porque o output precisa casar par-a-par com o `SIMILARITY_SCORING` para a fusão
a jusante. A enumeração é idêntica por construção.

**Preciso de dois modelos?**
Só em cross-população (probe e reference de condições distintas). No caso
simétrico, um modelo aplicado aos dois lados basta.

---

## 14. Glossário

- **tipicidade pareada** — score por par derivado das tipicidades das duas faces.
- $t_p$ / $t_r$ — tipicidade individual do probe / reference.
- **combinador** — função que funde $t_p$ e $t_r$ num escalar.
- **simétrico / unilateral** — combinador que preserva ou quebra
  $\mathrm{score}(A,B)=\mathrm{score}(B,A)$.
- **cross-população** — probe e reference avaliados por modelos distintos.
- **self / cross comparison** — pares dentro de um dataset (triangular) ou entre
  dois (cartesiano).

---

## 15. Referências

- Referências internas do FaceAudit:
  - `docs/tipicidade_knn_faceaudit.md` — treino do modelo de tipicidade.
  - `docs/interacao_score_raridade_faceaudit.md` — uso dos combinadores
    unilaterais na feature de interação.
  - `docs/vetorizacao_scores_faceaudit.md` — empilhamento com a similaridade.
  - Docstrings de `typicality_pair_scoring_instruction_executor.py`,
    `typicality_combiner.py`, `typicality_score_type.py`.

---

## Apêndice A — Fórmulas dos combinadores

Sendo $t_p$ e $t_r$ as tipicidades individuais de probe e reference:

- `MEAN`: $\tfrac{1}{2}(t_p + t_r)$
- `MIN`: $\min(t_p, t_r)$
- `MAX`: $\max(t_p, t_r)$
- `PRODUCT`: $t_p \cdot t_r$
- `ABS_DIFF`: $\lvert t_p - t_r\rvert$
- `REFERENCE_ONLY`: $t_r$
- `PROBE_ONLY`: $t_p$

---

## Apêndice B — `TypicalityPairScoringInstruction`

Arquivo: `faceaudit/models/instructions/typicality_pair_scoring_instruction.py`.

| Campo | Tipo | Default | Observação |
|---|---|---|---|
| `typicality_model_probe_resource_id` | `Optional[str]` | `None` | obrigatório |
| `typicality_model_reference_resource_id` | `Optional[str]` | `None` | só em cross-população |
| `probe_dataset_resource_id` | `Optional[str]` | `None` | obrigatório |
| `reference_dataset_resource_id` | `Optional[str]` | `None` | ausente → auto-comparação |
| `score_type` | `TypicalityScoreType` | `MEAN_KNN_SIMILARITY` | — |
| `combiner` | `TypicalityCombiner` | `MEAN` | — |
| `output_scores_resource_id` | `Optional[str]` | `None` | obrigatório; saída |
| `output_scores_name` | `Optional[str]` | `None` | — |
| `is_casework` | `bool` | `False` | zera `same_person` |

Compat: o campo legado `trained_typicality_model_resource_id` é migrado para
`typicality_model_probe_resource_id`.

---

## Apêndice C — Enums

**`TypicalityScoreType`** (`faceaudit/enums/typicality_score_type.py`):
`MEAN_KNN_SIMILARITY` (default), `TYPICALITY_PERCENTILE`, `LOG_DENSITY_KNN`.

**`TypicalityCombiner`** (`faceaudit/enums/typicality_combiner.py`):

| Membro | `is_symmetric` |
|---|---|
| `MEAN` (default) | True |
| `MIN` / `MAX` / `PRODUCT` / `ABS_DIFF` | True |
| `REFERENCE_ONLY` / `PROBE_ONLY` | **False** |

`from_str` defaulta `MEAN`. A property `is_symmetric` distingue os unilaterais.

---

## Apêndice D — Checklist de validação operacional

- [ ] Os datasets batem com os do `SIMILARITY_SCORING` correspondente (mesmos
      pares, mesma orientação)?
- [ ] O `score_type` é coerente com o uso a jusante (`TYPICALITY_PERCENTILE`
      para vetorizar/interagir)?
- [ ] O combinador é simétrico para fusão geral, ou unilateral para a feature de
      interação por papel?
- [ ] Combinadores unilaterais só em cross-comparação (sem warning de papéis
      convencionais)?
- [ ] Em cross-população, o segundo modelo está definido e treinado?
- [ ] O `embedding_column` da estratégia bate com a dimensão dos modelos?
- [ ] Em casework, `is_casework=True` (e o cosseno pareado também)?
