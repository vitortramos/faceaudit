# 07 — Operações estruturais

Este capítulo cobre operações que **modificam a estrutura** do
dataset (esquema, linhas, conjuntos):

- ➕📋 **Adicionar Coluna** — colunas customizadas (incluindo
  *computed*)
- 🏷️ **Metadados** — editor de esquema completo (nome, descrição,
  todas as colunas)
- 🔗 **Juntar** — concatenar dois datasets
- ✂️ **Dividir** — separar em dois datasets
- ⚖ **Balancear** — uniformizar amostras por identidade
- ⚖️ **Comparar** — análise lado-a-lado de dois datasets

## Sumário do capítulo

1. [Adicionar Coluna](#adicionar-coluna)
2. [Editor de Metadados](#editor-de-metadados)
3. [Juntar (merge)](#juntar)
4. [Dividir (split)](#dividir)
5. [Balancear](#balancear)
6. [Comparar dois datasets](#comparar)

---

## Adicionar Coluna

![Diálogo Adicionar Coluna](assets/38_add_column.png)

Cria uma **coluna customizada** (`role=CUSTOM`, `source=CUSTOM`) no
dataset atual. Útil para anotações livres durante revisão visual,
pareceres, *flags* customizadas.

### Campos

| Campo | Tipo | Validação | Notas |
|---|---|---|---|
| **Name** | Texto | Snake_case obrigatório (letras/números/`_`); não pode colidir com coluna existente | Nome técnico — não exibido no cabeçalho |
| **Label** | Texto | Opcional (default = name) | Exibido no cabeçalho |
| **Type** | Combo: `str` / `float` / `int` / `bool` | — | Define o widget de edição inline |
| **Default value** | Conforme tipo | Opcional (vazio = NaN) | Aplicado a todas as linhas existentes |
| ☑️ **Coluna computada** | Boolean | — | Quando marcado, mostra textarea de expressão |
| **Expressão** | Texto Python-like | Sintaxe validada no OK | Veja abaixo |

### Coluna computada

Quando você marca **Coluna computada**, em vez de um valor
constante a coluna recebe uma **expressão** avaliada para cada
linha:

```python
fiqa_score > 0.8 and ofiq_score > 70
```

A expressão tem acesso aos valores de **outras colunas** do
dataset. Suporta operadores aritméticos, comparações, lógicos,
e funções matemáticas básicas (`abs`, `min`, `max`, `round`).

> ⚠️ **Validação**: a expressão é compilada e testada em uma face
> de exemplo no momento do OK. Erros de sintaxe ou referência a
> coluna inexistente bloqueiam a criação.

### Resultado

A nova coluna aparece imediatamente na tabela, **editável** (a
menos que `computed`, que é read-only). Não esqueça de **Salvar**
o dataset para persistir.

## Editor de Metadados

![Diálogo Metadados com aba Dataset e aba Colunas](assets/39_metadados.png)

O **hub central** para editar tudo sobre o esquema. Acessível pelo
botão 🏷️ **Metadados** da ribbon *Dataset*.

### Aba *Dataset*

Edição dos atributos globais:

| Campo | Editável? | Notas |
|---|---|---|
| **Nome** | 🔒 Read-only | Pra mudar, use *Salvar Como* com novo nome |
| **Descrição** | ✏️ Sim | Texto livre, salvo no metadado |
| ☑️ **is_aligned** | ✏️ Sim | Override manual *(também acessível via Enriquecer)* |
| **path_mode** / **path_root** | 🔒 Read-only | Pra mudar, use *Localizar* ([capítulo 03](03_carregando_e_localizando.md)) |

### Aba *Colunas*

Layout split-pane:

| Lado | Conteúdo |
|---|---|
| **Esquerda** | Lista de todas as colunas (tabela) — Name, Label, Role, Source, Type |
| **Direita** | Formulário scrollable com todas as propriedades da coluna selecionada |

Ao clicar numa coluna à esquerda, o formulário direito carrega
seus atributos. Edições propagam imediatamente para um
*snapshot* em memória; só são persistidas ao clicar **Salvar**.

### Campos da coluna no formulário direito

#### Universais (aparecem para qualquer role)

- **Label** — sempre editável
- **Descrição** — texto livre
- **Categoria** — agrupamento visual (ex: "Demografia", "Qualidade")
- **Editable** — toggle se a célula aceita edição inline
- **Visible default** — toggle de visibilidade-padrão
- **Aliases** — lista de nomes alternativos (compatibilidade com parquets de esquema antigo)

#### Específicos por role

| Role | Campos extras |
|---|---|
| **QUALITY** | `direction` (`HIGHER_IS_BETTER` / `LOWER_IS_BETTER`), `subtype`, `scale`, `metric` |
| **POSE** | `axis` (`YAW`/`PITCH`/`ROLL`), `unit` (`DEGREES`/`RADIANS`) |
| **EMBEDDING** | `dim` (dimensão do vetor), `model_name` |
| **DEMOGRAPHIC** | `is_estimate` (toggle entre valor anotado pelo humano vs estimado pelo modelo) |
| **STR** *(qualquer role)* | **Vocabulary** — lista de valores válidos (combo box na edição). Política tolerante: valores fora do vocabulary são aceitos mas flageados visualmente |
| **CUSTOM** *(apenas)* | `role`, `dtype`, `default_value` (todos editáveis) |

### Restrições importantes

- **Colunas canônicas** (qualquer não-CUSTOM) **não permitem**
  mudar `role`, `dtype`, `source` ou `default_value`. Edição
  limitada a label, descrição, vocabulary, aliases, visibilidade.
- **Singletons** (`IDENTITY`, `IMAGE_PATH`, `FILE_HASH`) **não
  permitem** trocar a role nem ter `dtype` editado — papéis
  fundamentais do esquema.

### Botão *Coherence check*

Faz uma varredura buscando inconsistências entre o esquema e os
dados reais (vocabulários com valores não declarados, etc.). Útil
após importar/mesclar datasets de origens diferentes.

> ⚠️ Em datasets grandes (> 10 mil faces), a coherence check pode
> demorar alguns segundos. O cursor vira *waiting*.

## Juntar

A operação **Juntar** (botão 🔗 da ribbon *Dataset*) **não tem
diálogo próprio** — ela:

1. Abre um `QFileDialog` pedindo para você selecionar o **outro
   parquet** a ser mesclado com o dataset atual
2. Faz a concatenação **em memória** — o resultado vira o novo
   dataset corrente
3. Marca como modificado (você precisa **Salvar Como** pra persistir)

### Como a concatenação trata diferenças de esquema

| Situação | Resultado |
|---|---|
| Mesmas colunas em A e B | Concatena normalmente |
| A tem coluna que B não tem | Coluna preservada; linhas de B viram NaN nessa coluna |
| B tem coluna que A não tem | Coluna adicionada; linhas de A viram NaN |
| Mesma coluna mas tipos diferentes | Tenta upcast (int→float, bool→int); falha se incompatível |

> 💡 Após Juntar, use o **Editor de Metadados → Coherence check**
> pra detectar valores fora de vocabulary, e o **Editor de
> Metadados** pra alinhar labels/roles se necessário.

### O que NÃO acontece

- **Linhas duplicadas** não são deduplicadas automaticamente. Se A
  e B têm a mesma face (mesmo `hash`), ela aparece duas vezes no
  resultado. Use a **Auditoria** ([capítulo 09](09_auditoria_forense.md))
  pra detectar e remover duplicatas exatas se necessário.
- **`column_specs`** do dataset A é preservado; o de B é descartado.
- Arquivos originais **não** são tocados; nenhum dos dois parquets
  de origem é alterado.

## Dividir

![Diálogo Dividir Dataset](assets/36_dividir.png)

Particiona o dataset atual em **dois novos parquets**.

### Modos

| Modo | Como divide |
|---|---|
| **Por porcentagem (ratio)** | Divisão aleatória estratificada por identidade. Você escolhe `ratio` ∈ [5%, 95%] para o lado A; B recebe o complemento |
| **Manual (lista de identidades)** | Você seleciona quais identidades vão pra B (drag-select entre duas listas); restante vai pra A |

A divisão **manual** garante separação por identidade — útil para
criar datasets de treino/calibração sem vazamento.

### Campos

| Campo | Notas |
|---|---|
| **Nome A** | Stem do parquet do lado A (sem `.parquet`) |
| **Nome B** | Stem do parquet do lado B |
| **Ratio** *(modo porcentagem)* | Spinner 5–95 (em %). Default 50 |
| **Listas A / B** *(modo manual)* | Drag-select ou botões `←` / `→` para mover identidades |
| **Após dividir:** | `Manter dataset atual aberto` / `Abrir lado A` / `Abrir lado B` |

### Saída

Os dois parquets são salvos em `paths.library`. O dataset **atual
não é modificado** — só lido. Toda a divisão acontece em memória
+ salva os dois resultados.

### Estratificação automática

No modo *por porcentagem*, a divisão é estratificada **por
identidade**: cada identidade individual é dividida na mesma
proporção entre A e B, garantindo que ambos os lados tenham
representação balanceada de cada pessoa.

> ⚠️ Quando uma identidade tem apenas **1 amostra**, ela vai
> integralmente pra A ou B — você não pode dividir "a metade"
> de uma única foto. Identidades singleton tendem a ir mais pra A
> (que recebe a "porção maior" em ratios > 50%).

## Balancear

![Diálogo Balancear Dataset](assets/37_balancear.png)

Uniformiza o **número de amostras por identidade**. Importante
porque algoritmos de score (PLDA, regressão logística) assumem
classes balanceadas — desbalanceamento severo distorce o LR.

### Modo de balanceamento

| Modo | O que faz |
|---|---|
| **CAP_ONLY** | Para cada identidade, mantém no máximo `target_n` amostras. Identidades com **menos** que `target_n` ficam intactas |
| **FLOOR_AND_CAP** | Descarta identidades com **menos** que `target_n` amostras (floor); aplica cap nas demais |
| **AUTO_MIN_BALANCE** | Encontra a identidade com **menos** amostras e usa esse valor como `target_n` automaticamente. Garante perfeito balanceamento ao custo de descartar muitos dados se houver identidade muito subrepresentada |

### Critério de seleção

Quando precisa **descartar** amostras de uma identidade (porque tem
mais que `target_n`), o critério decide quais ficam:

| Critério | Como escolhe |
|---|---|
| **RANDOM** | Sorteio uniforme (com `seed` reproduzível) |
| **BY_QUALITY** | Mantém as `target_n` amostras com **maior** score na métrica escolhida (FIQA / OFIQ / SER-FIQ / embedding norm) |

> 💡 `BY_QUALITY` é o padrão recomendado pra forense — descartar
> aleatoriamente pode tirar fotos boas e manter fotos ruins.

### Campos

| Campo | Notas |
|---|---|
| **Mode** | Os 3 modos acima |
| **target_n** | Spinner 1–100k. Visível em `CAP_ONLY` e `FLOOR_AND_CAP`; oculto em `AUTO_MIN_BALANCE` |
| **Selection** | `Random` ou `By Quality` |
| **Seed** *(Random)* | Spinner inteiro. Default 42 |
| **Quality metric** *(By Quality)* | Combo: FIQA / OFIQ / SERFIQ / embedding_norm |
| **Após balancear:** | `Manter original aberto + criar novo` ou `Substituir o atual` |

### Saída

Como o split, o balancear cria um **novo dataset** — o atual fica
intacto (a menos que você escolha "Substituir o atual").

## Comparar

A operação **Comparar** abre um diálogo lado-a-lado mostrando
diferenças entre o **dataset atual** e outro `.parquet` que você
selecione.

### Estrutura do diálogo

| Aba | Conteúdo |
|---|---|
| **Resumo** | Caixa "Dataset A" + "Interseção" + "Dataset B" com contagens de imagens, identidades, etc. |
| **Gênero** | Tabela: cada valor de gênero + contagem em A + contagem em B + diferença |
| **Etnia** | Idem |
| **Origem** | Idem |
| **Identidades** | Conjunto-A, intersecção, conjunto-B |
| **Colunas** | Comum (presente nos dois) / Apenas em A / Apenas em B |

As células de **diferença** ficam coloridas: 🟢 verde (A > B), 🔴
vermelho (A < B), ⚪ cinza (igual).

### Casos de uso

- **Antes e depois** de uma operação destrutiva (filtro, balance)
  — abra o original + o resultado e compare quantitativamente
- **Validar splits** — depois de Dividir, abra A e B e confira
  que a divisão foi como esperado
- **Auditar merge** — depois de Juntar, abra o resultado contra
  um dos originais pra ver o delta

> 💡 Comparar é **read-only** — nenhum dos dois datasets é tocado.
> Você pode fechá-lo sem efeitos colaterais.

---

[← Capítulo 06 — Filtros e estatísticas](06_filtros_e_estatisticas.md) · [Capítulo 08 — Pipeline de imagens →](08_pipeline_de_imagens.md)
