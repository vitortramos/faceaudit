# Scoring de tipicidade pareada (`TYPICALITY_PAIR_SCORING`)

> **Status:** documento técnico de referência interna.
> **Escopo:** a instrução `TYPICALITY_PAIR_SCORING`. Cobre o consumo
> polimórfico do `TypicalityScoringModelProtocol` (V2.15 / PR 3), a
> propagação de proveniência via `TypicalityScoreBatch` para o metadata
> do dataset de saída, as três métricas de tipicidade que servem como
> base, os combinadores simétricos e unilaterais, a integração com
> `LR_ARCHITECTURE_VECTORIZATION` downstream.
> **Público-alvo:** peritos, desenvolvedores e revisores técnicos com
> conhecimento intermediário.
> **Documentos relacionados:**
> `docs/typicality_model_training_faceaudit.md` (instrução upstream),
> `docs/typicality_models.md` (matemática dos backends),
> `docs/interacao_score_raridade_faceaudit.md`
> (`SCORE_INTERACTION_FEATURE` downstream).
> **Fontes-código:** `faceaudit/models/instructions/typicality_pair_scoring_instruction.py`,
> `faceaudit/executors/instructions/typicality_pair_scoring_instruction_executor.py`,
> `faceaudit/enums/typicality_combiner.py`,
> `faceaudit/enums/typicality_score_type.py`,
> `faceaudit/enums/typicality_percentile_basis.py`,
> `faceaudit/protocols/typicality_scoring_model_protocol.py`,
> `faceaudit/models/typicality/typicality_score_batch.py`.

---

## Sumário

- [0. Apresentação](#0-apresentação)
- [1. Onde se encaixa no pipeline](#1-onde-se-encaixa-no-pipeline)
- [2. O que a instrução faz](#2-o-que-a-instrução-faz)
- [3. Polimorfismo via `TypicalityScoringModelProtocol`](#3-polimorfismo-via-typicalityscoringmodelprotocol)
- [4. Score types](#4-score-types)
- [5. Combinadores](#5-combinadores)
- [6. Modelo único vs dois modelos (cross-população)](#6-modelo-único-vs-dois-modelos-cross-população)
- [7. `TypicalityScoreBatch` — DTO de retorno](#7-typicalityscorebatch--dto-de-retorno)
- [8. Propagação de proveniência no metadata do dataset](#8-propagação-de-proveniência-no-metadata-do-dataset)
- [9. Enumeração de pares — casamento com `SIMILARITY_SCORING`](#9-enumeração-de-pares--casamento-com-similarity_scoring)
- [10. Configurações](#10-configurações)
- [11. Casework](#11-casework)
- [12. Validações e guard-rails](#12-validações-e-guard-rails)
- [13. Migração V2.14 → V2.15](#13-migração-v214--v215)
- [14. Integração com `LR_ARCHITECTURE_VECTORIZATION`](#14-integração-com-lr_architecture_vectorization)
- [15. Exemplo numérico](#15-exemplo-numérico)
- [16. Boas práticas](#16-boas-práticas)
- [17. Perguntas frequentes](#17-perguntas-frequentes)
- [18. Glossário](#18-glossário)
- [19. Referências](#19-referências)
- [Apêndice A — `TypicalityPairScoringInstruction`](#apêndice-a--typicalitypairscoringinstruction)
- [Apêndice B — Combinadores](#apêndice-b--combinadores)
- [Apêndice C — Checklist operacional](#apêndice-c--checklist-operacional)

---

## 0. Apresentação

### 0.1 Finalidade

Servir como referência sobre `TYPICALITY_PAIR_SCORING`: como ela aplica
um modelo de tipicidade às faces de cada par e como combina as duas
tipicidades num score por par alinhado pair-a-pair com `SIMILARITY_SCORING`
para alimentar downstream `SCORE_INTERACTION_FEATURE` e
`LR_ARCHITECTURE_VECTORIZATION`.

### 0.2 Mudanças em relação à V2.14

A versão V2.14 da instrução fazia `isinstance(payload, KNNTypicalityModel)`
no executor e chamava métodos específicos do k-NN
(`get_typicality_infos`, `score_samples`, `get_percentiles`). A V2.15
(PR 3 do plano de refatoração da família de tipicidade) introduziu:

1. **Consumo polimórfico**. O executor consome qualquer modelo que
   satisfaça `TypicalityScoringModelProtocol` — runtime_checkable —
   via uma chamada única `model.compute_scores(embeddings, score_type)`.
2. **Batch tipado**. O retorno é `TypicalityScoreBatch`, dataclass
   frozen com array imutável (`scores.flags.writeable = False`) + 9
   campos de proveniência completos.
3. **Proveniência no metadata do dataset**. O `SimilarityScoreDataset`
   produzido carrega `source_typicality_model_resource_id`,
   `source_typicality_model_contract_fingerprint`,
   `source_typicality_model_state_fingerprint`,
   `typicality_score_type`, `typicality_score_semantics_id`,
   `typicality_percentile_basis_score_type`,
   `typicality_reference_score_distribution_fingerprint`. Estes
   campos vão por sua vez ao `LR_ARCHITECTURE_VECTORIZATION` downstream
   que os incorpora ao `vectorization_contract_fingerprint`.
4. **Rename `LOG_DENSITY_KNN` → `LOG_DENSITY_ESTIMATE`**. O enum
   `TypicalityScoreType` foi unificado para tratar k-NN e MoVMF de
   forma simétrica. O `score_semantics_id` no batch distingue a
   matemática efetiva: `LOG_DENSITY_ESTIMATE_LQ_SPHERICAL` (k-NN
   Loftsgaarden-Quesenberry) vs `LOG_LIKELIHOOD_MOVMF` (mistura
   paramétrica).
5. **Validações runtime explícitas**. O executor não confia apenas
   no `runtime_checkable` (que checa só presença de atributos), mas
   também verifica `score_type ∈ enabled_score_types`,
   `batch.score_type == requested`, `batch.n == N`,
   `np.all(isfinite(batch.scores))`.

---

## 1. Onde se encaixa no pipeline

```
TYPICALITY_MODEL_TRAINING  (k-NN ou MoVMF)
    │
    │ produz TRAINED_TYPICALITY_MODEL
    ▼
TYPICALITY_PAIR_SCORING
    │
    │ aplica model.compute_scores(embeddings, score_type) → TypicalityScoreBatch
    │ combina probe ⊕ reference via TypicalityCombiner
    │ produz SimilarityScoreDataset (raw_score por par + metadata enriquecido)
    ▼
[opcional] SCORE_INTERACTION_FEATURE   (interação cosseno × raridade)
    │
    ▼
LR_ARCHITECTURE_VECTORIZATION           (consome rarity_*_model_fingerprint)
    │
    ▼
CALIBRATION_MODEL_TRAINING → SCORE_CALIBRATION → STRATEGY_EVALUATION
```

A enumeração de pares é **idêntica** à de `SIMILARITY_SCORING` (self ou
cross com os mesmos índices). Isso garante que o `SCORE_VECTORIZATION`
ou o builder de arquitetura LR consiga combinar pares-a-pares com os
scores de cosseno.

---

## 2. O que a instrução faz

Dado:

- Um (ou dois) `TRAINED_TYPICALITY_MODEL`(s).
- Um `EmbeddingDataset` probe (e opcionalmente um reference).
- Um `score_type` (qual métrica usar como tipicidade individual).
- Um `combiner` (como reduzir duas tipicidades a um único valor por par).

Produz:

- Um `SimilarityScoreDataset` cujo `raw_score` em cada item é a
  combinação `combiner(t_probe, t_reference)`, com metadata enriquecido
  carregando a proveniência completa do modelo de tipicidade.

A ordem de enumeração de pares é:

- **Self-comparison** (`reference_dataset_resource_id` ausente):
  triangular superior `(i, j)` com `i < j`.
- **Cross-comparison** (ambos definidos): produto cartesiano `(i, j)`
  com `i ∈ probe`, `j ∈ reference`.

---

## 3. Polimorfismo via `TypicalityScoringModelProtocol`

`faceaudit/protocols/typicality_scoring_model_protocol.py`

```python
@runtime_checkable
class TypicalityScoringModelProtocol(Protocol):
    @property
    def model_type(self) -> TypicalityModelType: ...
    @property
    def enabled_score_types(self) -> frozenset[TypicalityScoreType]: ...
    @property
    def embedding_dim(self) -> int: ...
    @property
    def sample_count(self) -> int: ...
    @property
    def model_state_fingerprint(self) -> str: ...
    @property
    def model_contract_fingerprint(self) -> str: ...

    def compute_scores(
        self,
        embeddings: np.ndarray,
        score_type: TypicalityScoreType,
    ) -> TypicalityScoreBatch: ...

    def is_in_distribution(self, embeddings: np.ndarray) -> np.ndarray: ...

    def summary(self) -> TypicalityModelSummary: ...
```

Tanto `KNNTypicalityModel` quanto `MoVMFTypicalityModel` satisfazem o
contrato estruturalmente — não há herança explícita. O executor:

1. **Resolve o modelo** via `context.get_resource(resource_id)`.
2. **Valida** com `isinstance(payload, TypicalityScoringModelProtocol)`
   (runtime_checkable; sanity-check, não substitui validação).
3. **Valida fit**: `payload.is_fitted()` quando disponível,
   `enabled_score_types` não vazio.
4. **Valida capacidade**: `instruction.score_type ∈
   payload.enabled_score_types`. Senão, levanta
   `UnsupportedTypicalityScoreTypeError` com listagem dos suportados.

`UnsupportedTypicalityScoreTypeError` é o erro que captura o caso
"pediu `MEAN_KNN_SIMILARITY` para um modelo MoVMF" (MoVMF não tem
análogo da similaridade média aos k-NN).

---

## 4. Score types

### 4.1 O enum `TypicalityScoreType`

`faceaudit/enums/typicality_score_type.py`

| Valor | Disponibilidade | Significado |
|---|---|---|
| `MEAN_KNN_SIMILARITY` | k-NN apenas | T_k bruto em [-1, 1]. Cosseno médio aos k vizinhos mais próximos. |
| `TYPICALITY_PERCENTILE` | Universal | Rank empírico em [0, 100] no batch; pair scoring divide por 100 para [0, 1]. |
| `LOG_DENSITY_ESTIMATE` | Universal | Estimativa de log-densidade. k-NN: Loftsgaarden-Quesenberry esférico. MoVMF: log-likelihood da mistura. **Escalas absolutas diferem entre backends.** |

O `score_type` é apenas o **rótulo conceitual** do que o pair scoring
pede. A semântica matemática efetiva está no `score_semantics_id` que
acompanha cada `TypicalityScoreBatch`.

### 4.2 O `score_semantics_id` no batch

| Backend | `score_type` | `score_semantics_id` no batch |
|---|---|---|
| k-NN | `MEAN_KNN_SIMILARITY` | `"MEAN_KNN_SIMILARITY_SPHERICAL"` |
| k-NN | `TYPICALITY_PERCENTILE` | `"PERCENTILE_OF_MEAN_KNN_SIMILARITY"` |
| k-NN | `LOG_DENSITY_ESTIMATE` | `"LOG_DENSITY_ESTIMATE_LQ_SPHERICAL"` |
| MoVMF | `TYPICALITY_PERCENTILE` | `"PERCENTILE_OF_LOG_LIKELIHOOD_MOVMF"` |
| MoVMF | `LOG_DENSITY_ESTIMATE` | `"LOG_LIKELIHOOD_MOVMF"` |

Mesmo `score_type` pode produzir matemática diferente conforme o
backend. O `score_semantics_id` é o discriminador estável para hash,
auditoria e comparação científica honesta.

### 4.3 Base do percentil — `TypicalityPercentileBasis`

`TYPICALITY_PERCENTILE` é ambíguo sozinho: percentil sobre **qual**
grandeza subjacente? O config do modelo (de
`TYPICALITY_MODEL_TRAINING`) declara `percentile_basis_score_type`,
propagado pelo batch como `percentile_basis_score_type`.

| Backend | `percentile_basis_score_type` default | Significado |
|---|---|---|
| k-NN | `MEAN_KNN_SIMILARITY` | Rank de T_k na CDF empírica. Comportamento histórico. |
| MoVMF | `LOG_DENSITY_ESTIMATE` | Rank de log-likelihood na CDF empírica. MoVMF não tem T_k. |

Dois calibradores com mesma rotulação `score_type = TYPICALITY_PERCENTILE`
mas bases diferentes consumiriam features semanticamente distintas. O
`percentile_basis_score_type` no metadata elimina essa ambiguidade
silenciosa.

### 4.4 Escala 0..1 no pair scoring

O `TypicalityScoreBatch` carrega percentil na escala canônica 0..100.
O pair scoring **divide por 100** para entregar [0, 1] aos
combinadores, escala compatível com cosseno e amigável ao LogReg sem
padronização. Isso fica no helper privado
`TypicalityPairScoringInstructionExecutor._scores_for_pairing`:

```python
if batch.score_type is TypicalityScoreType.TYPICALITY_PERCENTILE:
    return batch.scores / 100.0
return batch.scores
```

Os demais score types passam direto.

---

## 5. Combinadores

`faceaudit/enums/typicality_combiner.py`

| Combinador | Simétrico? | Fórmula | Caso de uso |
|---|---|---|---|
| `MEAN` | sim | `(t_p + t_r) / 2` | Default; média das tipicidades. |
| `MIN` | sim | `min(t_p, t_r)` | "Mais raro dos dois". |
| `MAX` | sim | `max(t_p, t_r)` | "Mais típico dos dois". |
| `PRODUCT` | sim | `t_p · t_r` | "Both rare implies very rare". |
| `ABS_DIFF` | sim | `|t_p − t_r|` | Discordância entre lados. |
| `REFERENCE_ONLY` | **não** | `t_r` | Alimenta `SCORE_INTERACTION_FEATURE` cross-pose (raridade do lado reference só). |
| `PROBE_ONLY` | **não** | `t_p` | Equivalente para o lado probe. |

### 5.1 Por que combinadores unilaterais?

`SCORE_INTERACTION_FEATURE` consome um modificador unilateral (`REFERENCE_ONLY`
ou `PROBE_ONLY`) quando a tese forense é "a raridade de **um lado** modula
a similaridade base". O par mantém o cosseno como base; a raridade vem
de um lado só.

Em **self-comparação**, os papéis probe/reference são convencionais
(triangular superior) e usar `REFERENCE_ONLY`/`PROBE_ONLY` é
matematicamente questionável — o executor emite **warning** mas não
bloqueia.

### 5.2 Alinhamento de pares para `SCORE_INTERACTION_FEATURE`

Quando o downstream usa um combinador unilateral, o alinhamento entre
o dataset de tipicidade pareada e o de cosseno passa a ser
**orientado** `(probe_hash, reference_hash)` em vez de
`frozenset({probe, ref})`. Detalhes em
`docs/interacao_score_raridade_faceaudit.md`.

---

## 6. Modelo único vs dois modelos (cross-população)

A instrução suporta dois modos:

### 6.1 Modelo único

`typicality_model_probe_resource_id` definido, `typicality_model_reference_resource_id` **ausente**.

O mesmo modelo é aplicado em probe e reference. Caso canônico:
ambos os datasets vêm da mesma população (e.g., frontal-frontal).

### 6.2 Dois modelos

Ambos definidos. Cenário cross-população:

- Probe frontal → modelo treinado em frontais.
- Reference lateral → modelo treinado em laterais.

Cross-pose forense típico. O executor emite log explicitando:

```
🔀 Cross-população: aplicando modelo do reference 'MoVMF_lateral' nas
   faces do reference_dataset.
```

Quando há dois modelos, ambos os fingerprints (contract e state) são
propagados no metadata via `source_typicality_model_*` para o probe e
`source_typicality_model_reference_*` para o reference.

### 6.3 Restrição

Para preservar a comparação honesta, os dois modelos devem ter o
mesmo `score_type` em `enabled_score_types`. O executor valida — se
um modelo não suporta `LOG_DENSITY_ESTIMATE` e o outro sim, levanta
`UnsupportedTypicalityScoreTypeError`.

---

## 7. `TypicalityScoreBatch` — DTO de retorno

`faceaudit/models/typicality/typicality_score_batch.py`

```python
@dataclass(frozen=True)
class TypicalityScoreBatch:
    # Dados (imutáveis após __post_init__)
    scores: np.ndarray                # (N,) float64, write=False
    n: int
    embedding_dim: int

    # Identidade do score
    score_type: TypicalityScoreType
    score_semantics_id: str           # ID de máquina (estável para hash)

    # Quando score_type == TYPICALITY_PERCENTILE
    percentile_basis_score_type: Optional[TypicalityScoreType]
    reference_score_distribution_fingerprint: Optional[str]

    # Proveniência do modelo
    model_type: TypicalityModelType
    model_state_fingerprint: str
    model_contract_fingerprint: str

    # Hash do próprio batch (scores + metadata)
    batch_fingerprint: str
```

### 7.1 Imutabilidade real

`@dataclass(frozen=True)` impede atribuição em campos do DTO, mas
**não congela ndarray**. O `__post_init__` faz `.copy() +
setflags(write=False)` para garantir que o array é imutável de fato:

```python
arr = np.ascontiguousarray(np.asarray(scores, dtype=np.float64)).copy()
arr.setflags(write=False)
object.__setattr__(self, "scores", arr)
```

Tentar `batch.scores[0] = 999` levanta `ValueError`.

### 7.2 Validações no `__post_init__`

- `scores.ndim == 1`.
- `n == arr.shape[0]`.
- Em `TYPICALITY_PERCENTILE`: `percentile_basis_score_type` obrigatório.
- Em outros score types: `percentile_basis_score_type` e
  `reference_score_distribution_fingerprint` são silenciosamente
  zerados (limpeza defensiva).

### 7.3 `batch_fingerprint`

SHA-256 de `scores.tobytes() + score_type + score_semantics_id +
percentile_basis + reference_score_distribution_fingerprint +
model_type + model_state_fingerprint + model_contract_fingerprint`.

Detecta divergência exata em comparações cross-execução.

---

## 8. Propagação de proveniência no metadata do dataset

A V2.15 persiste a proveniência **no metadata do dataset** (não em
cada item), evitando redundância em milhares de pares.

### 8.1 Campos persistidos (qualquer modo)

```python
metadata = {
    # Existentes na V2.14
    "mode": "self_comparison" | "cross_comparison",
    "score_kind": "typicality_pair",
    "score_type": "MEAN_KNN_SIMILARITY" | "TYPICALITY_PERCENTILE" | "LOG_DENSITY_ESTIMATE",
    "combiner": "MEAN" | ...,
    "probe_dataset_name": ...,
    "is_casework": bool,
    "model_resource_id": ...,        # compat — contract fp duplicado abaixo

    # PR 3: proveniência completa
    "model_fingerprint": "<contract_fp>",       # alias compat
    "source_typicality_model_resource_id": ...,
    "source_typicality_model_type": "KNN" | "MOVMF",
    "source_typicality_model_contract_fingerprint": "...",
    "source_typicality_model_state_fingerprint": "...",
    "typicality_score_type": ...,
    "typicality_score_semantics_id": ...,                       # estável p/ hash
    "typicality_percentile_basis_score_type": ...,              # null se não-percentil
    "typicality_reference_score_distribution_fingerprint": ...,
}
```

### 8.2 Campos adicionais em cross com dual models

```python
metadata.update({
    "reference_dataset_name": ...,
    "uses_dual_typicality_models": True,
    "source_typicality_model_reference_resource_id": ...,
    "source_typicality_model_reference_type": "KNN" | "MOVMF",
    "source_typicality_model_reference_contract_fingerprint": "...",
    "source_typicality_model_reference_state_fingerprint": "...",
})
```

Quando o combiner é `REFERENCE_ONLY`, o `model_fingerprint` "principal"
do metadata aponta para o modelo do reference (lado consumido pela
combinação). Para `PROBE_ONLY` e simétricos, aponta para o probe.

### 8.3 Per-item metadata

Cada `SimilarityScoreSample` carrega apenas dois campos compactos:

```python
sample.metadata = {
    "probe_typicality": float,        # após escala (já / 100 em PERCENTILE)
    "reference_typicality": float,
}
```

Suficiente para reconstruir o cálculo do combiner via auditoria.

---

## 9. Enumeração de pares — casamento com `SIMILARITY_SCORING`

A enumeração é **deterministicamente idêntica** à de `SIMILARITY_SCORING`
para que o `SCORE_VECTORIZATION` (ou `LR_ARCHITECTURE_VECTORIZATION`)
consiga alinhar pares-a-pares por índice.

### 9.1 Self

```python
for i in range(n):
    for j in range(i + 1, n):
        emit(pair=(dataset[i], dataset[j]))
```

Total: `n*(n-1)/2` pares. Mesma ordem do `SimilarityScoringInstructionExecutor`.

### 9.2 Cross

```python
for i, probe in enumerate(probe_dataset):
    for j, reference in enumerate(reference_dataset):
        emit(pair=(probe, reference))
```

Total: `n_probe * n_ref` pares.

### 9.3 Por que mesmo ordem importa

O `LR_ARCHITECTURE_VECTORIZATION` alinha o dataset de tipicidade
pareada com o de cosseno via `pair_key_order_hash` (hash da sequência
ordenada `(probe_hash, reference_hash)`). Hashs iguais → pares
alinhados; hashs diferentes → executor levanta erro de alinhamento.

---

## 10. Configurações

| Campo | Tipo | Default | Descrição |
|---|---|---|---|
| `typicality_model_probe_resource_id` | `Optional[str]` | `None` | Recurso `TRAINED_TYPICALITY_MODEL` (obrigatório). |
| `typicality_model_reference_resource_id` | `Optional[str]` | `None` | Recurso `TRAINED_TYPICALITY_MODEL` opcional (cross-população). |
| `probe_dataset_resource_id` | `Optional[str]` | `None` | Dataset probe (obrigatório). |
| `reference_dataset_resource_id` | `Optional[str]` | `None` | Dataset reference (opcional; ausente → self). |
| `score_type` | `TypicalityScoreType` | `MEAN_KNN_SIMILARITY` | Métrica individual antes do combiner. |
| `combiner` | `TypicalityCombiner` | `MEAN` | Como combinar duas tipicidades. |
| `is_casework` | `bool` | `False` | Modo forense (zera `same_person` no output). |
| `output_scores_resource_id` | `Optional[str]` | `None` | Recurso de saída `SIMILARITY_SCORES`. |
| `output_scores_name` | `Optional[str]` | `None` | Nome amigável (default = composição auto). |

---

## 11. Casework

Quando `is_casework=True`, o executor:

1. Após o cálculo, zera `sample.same_person = None` em todos os itens.
2. Marca `metadata["is_casework"] = True` no `SimilarityScoreDataset`.
3. Limpa caches de contagem `n_gen`/`n_imp` (não fazem sentido sem
   ground truth).
4. Loga: `🔍 Modo casework — N pares sem ground truth.`

Downstream, instruções como `STRATEGY_EVALUATION` rejeitam datasets
casework (precisam de ground truth); `CASEWORK_VISUALIZATION` exige.

---

## 12. Validações e guard-rails

### 12.1 Validações estruturais (`_validate_instruction_specifics`)

- `typicality_model_probe_resource_id` definido e existe no contexto.
- `typicality_model_reference_resource_id` (se definido) existe.
- `probe_dataset_resource_id` definido e existe.
- `reference_dataset_resource_id` (se definido) existe.
- `output_scores_resource_id` definido.

### 12.2 Validações runtime no executor (PR 3)

A lista canônica do plano:

1. `model is_fitted()` — quando o backend expõe o método.
2. `payload.enabled_score_types` não vazio (modelo carregado deve
   estar pronto para produzir).
3. `instruction.score_type ∈ payload.enabled_score_types` —
   `UnsupportedTypicalityScoreTypeError` caso não.
4. `batch.score_type == instruction.score_type` — modelo retornou
   o tipo errado (defesa contra bug interno).
5. `batch.n == N_embeddings` — tamanho consistente.
6. `np.all(isfinite(batch.scores))` — sem NaN/Inf.

Cada uma levanta erro distinto com mensagem orientativa.

### 12.3 Validação cross-condition

Quando ambos os modelos definidos:

- Modelo do reference é um `TypicalityScoringModelProtocol` válido.
- O `score_type` solicitado está em `enabled_score_types` de **ambos**
  os modelos.

---

## 13. Migração V2.14 → V2.15

Migração automática via `LegacyStrategyMigrationService`:

| Antes (V2.14) | Depois (V2.15) | Step kind |
|---|---|---|
| `score_type = "LOG_DENSITY_KNN"` | `score_type = "LOG_DENSITY_ESTIMATE"` | `score_type_rename` |

Estrutura da instrução é preservada (campos, IDs, combiner). Apenas o
score type é renomeado. Detalhes em
`docs/typicality_model_training_faceaudit.md` §11.

---

## 14. Integração com `LR_ARCHITECTURE_VECTORIZATION`

O `LR_ARCHITECTURE_VECTORIZATION` consome o `SimilarityScoreDataset`
produzido pelo pair scoring como **dataset de raridade**
(`rarity_probe_resource_id` ou `rarity_reference_resource_id` para
two-face; `rarity_resource_id` para ONE_FACE).

### 14.1 Fingerprint downstream

O executor de `LR_ARCHITECTURE_VECTORIZATION` lê do metadata:

```python
probe_meta = ctx.get_resource(rarity_probe_resource_id).metadata
rarity_probe_model_fingerprint = probe_meta.get("model_fingerprint")
```

`model_fingerprint` aqui é o `model_contract_fingerprint` propagado
pelo pair scoring (PR 3 substituiu o `None` da V2.14 pelo
fingerprint real). Esse valor entra na construção do
`vectorization_contract_fingerprint` do calibrador upstream:

```python
contract_payload = {
    ...,
    "rarity_probe_model_fingerprint": ...,
    "rarity_reference_model_fingerprint": ...,
    "rarity_single_model_fingerprint": ...,
    ...
}
```

### 14.2 Consequência operacional

Mudar o modelo de tipicidade upstream (qualquer mudança em
`unit_norm_policy`, `population_weighting_mode`, `centering`, `k`,
`n_components`, etc.) → `model_contract_fingerprint` muda →
`vectorization_contract_fingerprint` muda → `INHERIT` em
`LR_ARCHITECTURE_VECTORIZATION` rejeita state com fingerprint
divergente.

Isso fecha o loop: re-treinos silenciosos do modelo de tipicidade
contaminam casework apenas se o calibrador for re-treinado também.

---

## 15. Exemplo numérico

### 15.1 Cenário

- Probe dataset: 50 faces frontais.
- Reference dataset: 100 faces laterais.
- Modelo do probe: k-NN treinado em frontais (50 identidades).
- Modelo do reference: MoVMF treinado em laterais (40 identidades).
- `score_type = TYPICALITY_PERCENTILE`.
- `combiner = REFERENCE_ONLY`.

### 15.2 Saída

- 5000 pares (50 × 100).
- Em cada par, `raw_score = t_reference / 100` (REFERENCE_ONLY +
  divisão por 100 do escalamento de pair scoring).
- Metadata do dataset:

```python
{
    "mode": "cross_comparison",
    "score_kind": "typicality_pair",
    "score_type": "TYPICALITY_PERCENTILE",
    "combiner": "REFERENCE_ONLY",
    "probe_dataset_name": "frontal_probe",
    "reference_dataset_name": "lateral_reference",
    "uses_dual_typicality_models": True,
    "is_casework": False,
    "model_resource_id": "MoVMF_lateral",       # = reference (REFERENCE_ONLY)
    "model_fingerprint": "<contract_fp_movmf_lateral>",
    "source_typicality_model_resource_id": "KNN_frontal",   # probe
    "source_typicality_model_type": "KNN",
    "source_typicality_model_contract_fingerprint": "<contract_fp_knn_frontal>",
    "source_typicality_model_state_fingerprint": "<state_fp_knn_frontal>",
    "typicality_score_type": "TYPICALITY_PERCENTILE",
    "typicality_score_semantics_id": "PERCENTILE_OF_MEAN_KNN_SIMILARITY",  # do probe
    "typicality_percentile_basis_score_type": "MEAN_KNN_SIMILARITY",       # k-NN default
    "typicality_reference_score_distribution_fingerprint": "<ref_dist_fp>",
    "source_typicality_model_reference_resource_id": "MoVMF_lateral",
    "source_typicality_model_reference_type": "MOVMF",
    "source_typicality_model_reference_contract_fingerprint": "<contract_fp_movmf_lateral>",
    "source_typicality_model_reference_state_fingerprint": "<state_fp_movmf_lateral>",
}
```

### 15.3 Downstream

`LR_ARCHITECTURE_VECTORIZATION` com `architecture = HIERARCHICAL_CENTERED_ONE_FACE`
consome este dataset como `rarity_resource_id` (ONE_FACE).
`rarity_single_model_fingerprint = "<contract_fp_movmf_lateral>"`
(porque o combiner é REFERENCE_ONLY → modelo do reference).

---

## 16. Boas práticas

1. **Use o mesmo backend em probe e reference** quando possível.
   Comparar k-NN vs MoVMF cross-pose é cientificamente questionável
   pelas escalas absolutas diferentes em `LOG_DENSITY_ESTIMATE`.
2. **Prefira `TYPICALITY_PERCENTILE` em multivariados**. A escala
   [0, 1] no pair scoring é amigável ao LogReg sem padronização e
   comparável entre populações com escalas absolutas distintas.
3. **`REFERENCE_ONLY`/`PROBE_ONLY` só em cross-comparison**. Em
   self, papéis probe/reference são convencionais.
4. **Em casework, sempre `is_casework=True`**. Mantém o pipeline
   forense intacto downstream (validações de `STRATEGY_EVALUATION`
   ↔ `CASEWORK_VISUALIZATION`).
5. **Registre o `model_contract_fingerprint` no laudo**. Permite
   re-validação independente.
6. **Quando trocar modelo upstream, re-treine o calibrador**.
   INHERIT rejeita silenciosamente; é melhor antecipar o
   re-treinamento.

---

## 17. Perguntas frequentes

### 17.1 Por que `MEAN_KNN_SIMILARITY` falha no MoVMF?

O MoVMF não tem análogo da similaridade média aos k vizinhos (T_k é
uma noção local não-paramétrica). Logo, MoVMF declara
`enabled_score_types = {LOG_DENSITY_ESTIMATE, TYPICALITY_PERCENTILE}`,
sem MEAN_KNN_SIMILARITY. `UnsupportedTypicalityScoreTypeError` é o
erro correto.

### 17.2 Posso comparar Cllr entre k-NN e MoVMF na mesma estratégia?

Não em uma única estratégia (não há como variar o tipo do modelo
dentro do mesmo pipeline). Para comparação científica, gere duas
estratégias paralelas com `typicality_model_type` diferente e
**TODOS** os outros parâmetros idênticos (split, calibrador, LR
architecture). Compare Cllr finais.

### 17.3 Como sei se o downstream LR vai aceitar meu modelo?

Verifique que o `score_type` que você pediu está em
`enabled_score_types` do modelo treinado. O metadata do dataset de
saída carrega `typicality_score_semantics_id` — se o LR
`vectorization_contract_fingerprint` já existe, o `INHERIT` vai
comparar via fingerprint. Mudou? Re-treine.

### 17.4 Preciso dos dois modelos em cross-comparison?

Não obrigatoriamente. Se você omitir `typicality_model_reference_resource_id`,
o modelo do probe é aplicado em ambos os datasets. Caso natural quando
probe e reference vêm da mesma população (e.g., frontal-frontal).

---

## 18. Glossário

- **`compute_scores`**. Método único do protocolo, aceita embeddings e
  score_type, retorna `TypicalityScoreBatch`.
- **`enabled_score_types`**. Capacidades efetivas da instância
  treinada. Subset do `potential_score_types` do backend (registry).
- **`UnsupportedTypicalityScoreTypeError`**. Erro de quando o
  `score_type` solicitado não está em `enabled_score_types`.
- **`score_semantics_id`**. ID de máquina estável para a semântica
  matemática do score. Distingue fórmulas que compartilham enum value.
- **`percentile_basis_score_type`**. Base sobre a qual o
  `TYPICALITY_PERCENTILE` é calculado. Persistido no batch e no
  metadata do dataset.
- **Combinador unilateral**. `REFERENCE_ONLY` ou `PROBE_ONLY`. Quebra
  simetria propositalmente para alimentar `SCORE_INTERACTION_FEATURE`.

---

## 19. Referências

- FaceAudit interno:
  - `docs/typicality_model_training_faceaudit.md` (instrução upstream).
  - `docs/typicality_models.md` (matemática dos backends).
  - `docs/interacao_score_raridade_faceaudit.md`
    (`SCORE_INTERACTION_FEATURE` downstream).
  - `docs/vetorizacao_scores_faceaudit.md` (`SCORE_VECTORIZATION`).
  - `docs/calibracao_faceaudit.md`.

---

## Apêndice A — `TypicalityPairScoringInstruction`

Definição em
`faceaudit/models/instructions/typicality_pair_scoring_instruction.py`.

### Resource specs (UI)

- `typicality_model_probe_resource_id`: dropdown filtrando
  `TRAINED_TYPICALITY_MODEL`. Obrigatório.
- `typicality_model_reference_resource_id`: dropdown filtrando
  `TRAINED_TYPICALITY_MODEL`. Visível apenas quando
  `reference_dataset_resource_id` está definido.
- `probe_dataset_resource_id`: dropdown filtrando `DATASET`. Obrigatório.
- `reference_dataset_resource_id`: dropdown filtrando `DATASET`. Opcional.
- `output_scores_resource_id`: campo de saída (`SIMILARITY_SCORES`,
  `is_output=True`).

### Choice specs

- `score_type`: dropdown com 3 valores do `TypicalityScoreType`.
- `combiner`: dropdown com 7 valores do `TypicalityCombiner`.

### Value specs

- `output_scores_name`: campo de texto opcional.
- `is_casework`: checkbox.

### Sync de I/O

`_sync_declared_io()` recalcula `input_resource_ids` (probe model +
probe dataset + ref model se cross + ref dataset se cross) e
`output_resource_ids` ([output_scores_resource_id]).

---

## Apêndice B — Combinadores

### Métodos do enum

| Método | Retorno | Notas |
|---|---|---|
| `combiner.combine(t_p, t_r)` | `float` | Para os simétricos. Levanta `NotImplementedError` em `REFERENCE_ONLY`/`PROBE_ONLY`. |
| `combiner.is_symmetric` | `bool` | True nos 5 simétricos. |
| `combiner.to_str()` | `str` | Valor string (estável em JSON). |

### Despacho no executor

```python
def _combine_typicalities(combiner, t_probe, t_reference):
    if combiner is REFERENCE_ONLY:
        return float(t_reference)
    if combiner is PROBE_ONLY:
        return float(t_probe)
    return float(combiner.combine(t_probe, t_reference))
```

---

## Apêndice C — Checklist operacional

Antes de declarar uma `TYPICALITY_PAIR_SCORING` válida:

- [ ] Modelo de tipicidade treinado e disponível em
      `TRAINED_TYPICALITY_MODEL` (verificar `enabled_score_types`).
- [ ] `score_type` ∈ `enabled_score_types` do modelo (e do modelo do
      reference se for cross-population com dual models).
- [ ] Probe e reference datasets têm a coluna de embedding correta.
- [ ] `embedding_column` consistente entre o modelo treinado e os
      datasets.
- [ ] Combiner adequado ao caso de uso:
  - Self → simétrico (`MEAN`, `MIN`, `MAX`, `PRODUCT`, `ABS_DIFF`).
  - Cross + `SCORE_INTERACTION_FEATURE` downstream → unilateral
    (`REFERENCE_ONLY` ou `PROBE_ONLY` conforme o lado modelado).
- [ ] `is_casework` setado conforme contexto.
- [ ] `output_scores_resource_id` único na estratégia.

Em casework forense, adicione:

- [ ] `model_contract_fingerprint` do upstream registrado no laudo.
- [ ] Downstream `LR_ARCHITECTURE_VECTORIZATION` usa `INHERIT` (não
      MANUAL nem FIT_FULL_CALIBRATION_LEGACY).

---

**Última revisão:** V2.15 (2026-06-08).
