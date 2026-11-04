# 07 — Operações estruturais

Este capítulo cobre operações que **modificam a estrutura** do
dataset (esquema, linhas, conjuntos):

- ➕📋 **Coluna** (Adicionar Coluna) — colunas customizadas
  (incluindo *computed*)
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
dataset atual. Acessível pelo botão ➕📋 **Coluna** da ribbon
*Dataset*. Útil para anotações livres durante revisão visual,
pareceres, *flags* customizadas.

### Campos

| Campo (rótulo na tela) | Tipo | Validação | Notas |
|---|---|---|---|
| **Nome** | Texto | Obrigatório; apenas letras, números e `_` (sem espaços nem acentos); não pode colidir com coluna existente | Nome físico da coluna no parquet — não exibido no cabeçalho |
| **Rótulo** | Texto | Opcional (default = nome) | Exibido no cabeçalho |
| **Tipo** | Combo: *Texto (string)* / *Número decimal* / *Número inteiro* / *Verdadeiro/Falso* (`str`/`float`/`int`/`bool`) | — | Define o widget de edição inline |
| **Valor padrão** | Conforme tipo | Texto vazio = nulo; numéricos começam em 0; booleano é a caixa *Marcado por padrão* | Aplicado a todas as linhas existentes. Desabilitado quando a coluna é computada |
| ☑️ **Coluna computada (fórmula simbólica, read-only)** | Boolean | — | Quando marcado, mostra o quadro **Fórmula** |
| **Fórmula** | Texto Python-like | Sintaxe validada no OK | Veja abaixo |

### Coluna computada

Quando você marca **Coluna computada**, em vez de um valor
constante a coluna recebe uma **expressão** avaliada para cada
linha:

```python
fiqa_score > 0.8 and ofiq_score > 70
```

A expressão tem acesso aos valores de **outras colunas** do
dataset (use os **nomes técnicos** das colunas como variáveis).
Suporta operadores aritméticos, comparações, lógicos, o ternário
`a if b else c`, e um conjunto de funções seguras:

```
abs, min, max, round, len, str, int, float, bool,
sqrt, log, log2, log10, exp, floor, ceil
```

Não há acesso a atributos (`a.b`), índices (`a[0]`), *comprehensions*
nem *lambdas*. Uma computada **não** pode referenciar outra coluna
computada (a referência resolve como inexistente).

> ⚠️ **Validação no OK**: a fórmula é **analisada sintaticamente**
> (parse do AST) e só se verifica que os nós e as funções usados
> estão na lista permitida. A expressão **não** é testada numa face
> de exemplo, e **referência a coluna inexistente NÃO bloqueia a
> criação** — ela só falha em *runtime*, linha a linha, exibindo
> `#ERR` na célula daquela linha.

> ⚠️ **Pré-condição de esquema**: se o dataset foi carregado de um
> parquet **legado sem `column_specs`** na metadata, o botão é
> bloqueado com o aviso *"Schema legado"* pedindo para **Salvar**
> primeiro (isso reescreve o parquet no formato novo). Só depois de
> salvar você consegue adicionar colunas customizadas. O mesmo
> bloqueio vale para o **Editor de Metadados**.

### Resultado

A nova coluna aparece imediatamente na tabela, **editável** (a
menos que `computed`, que é read-only). Não esqueça de **Salvar**
o dataset para persistir. Numa coluna computada só a **fórmula** é
gravada (nos `column_specs` do parquet); o valor é recalculado por
linha a cada exibição.

> 💡 A criação de coluna é registrada no **Audit Log** forense
> (operação `ADD_COLUMN`), alimentando a cadeia de custódia — veja o
> [capítulo 09](09_auditoria_forense.md).

## Editor de Metadados

![Diálogo Metadados com aba Dataset e aba Colunas](assets/39_metadados.png)

O **hub central** para editar tudo sobre o esquema. Acessível pelo
botão 🏷️ **Metadados** da ribbon *Dataset*.

### Aba *Dataset*

Edição dos atributos globais, em grupos (*Identificação*, *Imagens*,
*Caminhos das imagens* e, quando houver, *Proveniência*):

| Campo | Editável? | Notas |
|---|---|---|
| **Nome** | ✏️ Sim | Nome do dataset gravado na metadata do parquet (`faceaudit:dataset_name`). Não renomeia o arquivo — para isso, use *Salvar Como* |
| **Descrição** | ✏️ Sim | Texto livre, salvo no metadado |
| ☑️ **Imagens já alinhadas (112×112 …)** (`is_aligned`) | ✏️ Sim | Override manual *(também acessível via Enriquecer)* |
| **Modo** (`path_mode`) / **Subpasta** (`path_root`) | 🔒 Read-only | Pra mudar/migrar, use *Modificar Caminhos* ([capítulo 03](03_carregando_e_localizando.md)) |

> ℹ️ **Proveniência é exibida (read-only).** Quando o dataset traz
> metadados de proveniência — caso típico dos datasets criados por
> extração de vídeo (veja o
> [capítulo 02b](02b_criando_dataset_de_video.md)): `dataset_origin_type`,
> `selection_warning` (aviso de viés de seleção),
> `selection_conditioned_on_embedding`, `selection_anchor_model`,
> `selection_threshold`, `source_run_id`, `source_run_manifest` —, o
> Editor de Metadados os mostra num grupo **Proveniência** somente-
> leitura na aba *Dataset* (com o aviso de seleção em destaque âmbar).
> Chaves fora dessa lista também aparecem, pelo nome técnico. Não são
> editáveis por aqui — viajam com o parquet.

### Aba *Colunas*

Layout split-pane:

| Lado | Conteúdo |
|---|---|
| **Esquerda** | Lista de todas as colunas (tabela) — Nome, Rótulo, Role, Source, Tipo |
| **Direita** | Formulário scrollable com todas as propriedades da coluna selecionada |

Ao clicar numa coluna à esquerda, o formulário direito carrega
seus atributos. Edições propagam imediatamente para um
*snapshot* em memória; só são aplicadas ao dataset ao clicar
**Salvar** no diálogo (o que marca o dataset como modificado e
registra `EDIT_SCHEMA` na cadeia de custódia). Para gravar no
parquet, salve o dataset depois.

### Campos da coluna no formulário direito

#### Universais (aparecem para qualquer role)

- **Nome** e **Source** — exibidos, nunca editáveis
- **Rótulo** — sempre editável
- **Descrição** — texto livre
- **Categoria** — agrupamento visual (ex: "Demografia", "Qualidade")
- **Editável célula a célula** — toggle se a célula aceita edição inline
- **Visível ao abrir** — toggle de visibilidade-padrão
- **Aliases (nomes legados)** — um por linha (compatibilidade com parquets de esquema antigo)

#### Específicos por role

| Role | Campos extras |
|---|---|
| **QUALITY** | `direction` (`HIGHER_IS_BETTER` / `LOWER_IS_BETTER`), `subtype`, `scale`, `metric` |
| **POSE** | `axis` — combo `Yaw`/`Pitch`/`Roll` (rótulo *Eixo (Pose)*); `unit` — campo de **texto livre** (*Unidade (Pose)*, placeholder `deg`), não um combo |
| **EMBEDDING** | `dim` (dimensão do vetor), `model_name` |
| **DEMOGRAPHIC** | `is_estimate` (toggle entre valor anotado pelo humano vs estimado pelo modelo) |
| **STR** *(qualquer role)* | **Vocabulary** — lista de valores válidos (combo box na edição). Política tolerante: valores fora do vocabulary são aceitos mas flageados visualmente |
| **CUSTOM** *(apenas — `source=CUSTOM`)* | `role`, `dtype`, `default_value` (todos editáveis) |

### Restrições importantes

- **Colunas canônicas** (qualquer não-CUSTOM) **não permitem**
  mudar `role`, `dtype`, `source` ou `default_value`. Edição
  limitada a rótulo, descrição, categoria, vocabulary, aliases,
  visibilidade e editabilidade.
- **Singletons** (`IDENTITY`, `IMAGE_PATH`, `FILE_HASH`) **não
  permitem** trocar a role, nem mesmo numa coluna CUSTOM — evita
  criar uma segunda coluna de identidade. Papéis fundamentais do
  esquema.
- O `source` **nunca** é editável, em coluna nenhuma.

### Botão 🔍 *Verificar coerência*

Lista quantos valores estão **fora do vocabulário** em cada coluna.
A verificação cobre **apenas colunas do tipo `str` com vocabulário
definido** — para cada uma, conta os valores que não constam do
vocabulário declarado. A política é **tolerante**: valores
fora-de-vocab continuam aceitos, só ficam flagados visualmente na
tabela. Útil após importar/mesclar datasets de origens diferentes.

> ⚠️ Em datasets grandes (> 10 mil faces), a verificação pode
> demorar alguns segundos. O cursor vira *waiting*.

## Juntar

A operação **Juntar** (botão 🔗 da ribbon *Dataset*) **não tem
diálogo próprio** — ela:

1. Abre um seletor de arquivo (*"Juntar com..."*) pedindo para você
   selecionar o **outro parquet** a ser mesclado com o dataset atual
2. Se só **um** dos dois é de faces já alinhadas (112×112), pergunta
   antes (*"Tipos de imagem diferentes"*, padrão **Não**): o
   resultado ficaria marcado como **não** alinhado
3. Faz a concatenação **em memória** — o resultado (nome
   `{A}_{B}_Merged`) vira o novo dataset corrente
4. Confere a existência das imagens numa **amostra de até 200
   faces** e avisa (*"Imagens não encontradas"*) se alguma não for
   achada no disco — sinal de que a pasta de imagens de um dos lados
   foi movida ou renomeada
5. Marca como modificado (você precisa **Salvar Como** pra persistir)

### Como a concatenação trata diferenças de esquema

A junção é uma **concatenação das listas de faces** (o resultado é
`faces de A + faces de B`). **Não** há reconciliação de esquema
tipada — nada de *upcast* `int→float`/`bool→int` nem verificação de
compatibilidade de tipos no momento da junção.

| Situação | Resultado |
|---|---|
| Mesmas colunas em A e B | Concatena normalmente |
| A tem coluna que B não tem | Ao montar a tabela, as linhas de B ficam **vazias/NaN** nessa coluna |
| B tem coluna que A não tem | Idem — as linhas de A ficam **vazias/NaN** |
| Mesma coluna com tipos diferentes | Sem tratamento explícito na junção; as diferenças aparecem só quando a tabela/parquet é montado |

> 💡 Após Juntar, use o **Editor de Metadados → 🔍 Verificar
> coerência** pra detectar valores fora de vocabulário, e o **Editor
> de Metadados** pra alinhar labels/roles se necessário.

### O que NÃO acontece

- **Linhas duplicadas** não são deduplicadas automaticamente. Se A
  e B têm a mesma face (mesmo `hash`), ela aparece duas vezes no
  resultado. Use a **Auditoria** ([capítulo 09](09_auditoria_forense.md))
  pra detectar e remover duplicatas exatas se necessário.
- Arquivos originais **não** são tocados; nenhum dos dois parquets
  de origem é alterado (quando os caminhos precisam ser reescritos, as
  faces entram no resultado como cópia).

### O que é preservado

- **Resolução das imagens.** Se A e B têm a mesma ancoragem (mesmo
  `path_mode` e, em modo relativo, a mesma pasta de imagens), ela é
  mantida e o resultado continua portátil. Se as âncoras diferem, os
  caminhos dos dois lados são resolvidos para **absolutos** e o
  resultado nasce em modo absoluto — a mensagem final avisa. Para
  voltar a ter um dataset portátil, use *Modificar Caminhos* ou salve
  como autocontido.
- **`is_aligned`** só sobrevive se **os dois** forem alinhados.
- **`column_specs`**: união das especificações de A e B (em conflito,
  vale a de A) — rótulo, role e categoria das colunas do segundo
  dataset não se perdem.
- **Proveniência**: união dos metadados de proveniência dos dois (em
  conflito, vale a de A).

## Dividir

![Diálogo Dividir Dataset](assets/36_dividir.png)

Particiona o dataset atual em **dois novos parquets**.

### Modos

Os dois modos aparecem no combo **Método** do diálogo:

| Modo (rótulo na tela) | Como divide |
|---|---|
| **Divisão Aleatória por Porcentagem** | Divisão aleatória **por identidade** (*identity-disjoint*): cada identidade vai **inteira** para A **ou** para B — nunca é fatiada entre os dois. A fração escolhida controla a proporção de **identidades** que vão para A. O sorteio usa semente fixa (42), então o mesmo dataset com a mesma fração dá sempre a mesma divisão |
| **Seleção Manual de Identidades** | Você seleciona quais identidades vão pra B (duas listas + botões); restante vai pra A |

> ⚠️ **Ambos os modos são identity-disjoint** — não só o manual. Em
> nenhum deles uma mesma pessoa aparece nos dois lados, o que
> **evita vazamento de identidade** (*data leakage*) entre treino e
> calibração/teste.

### Campos

| Campo | Notas |
|---|---|
| **Nome do Dataset A** (*Treino/Base*) | Stem do parquet do lado A (sem `.parquet`); sugerido `{nome}_A` |
| **Nome do Dataset B** (*Teste/Alvo*) | Stem do parquet do lado B; sugerido `{nome}_B` |
| **Proporção (A / Total)** *(modo porcentagem)* | `QDoubleSpinBox` de **fração** de 0,05 a 0,95 (passo 0,05), **default 0,80**, com sufixo *(Fração para o Dataset A)*. É fração, não porcentagem |
| **Listas A / B** *(modo manual)* | Selecione uma ou mais identidades e use os botões **`>>`** (mover para B) / **`<<`** (mover para A), ou **duplo-clique** num item para movê-lo. Não há *drag-and-drop* |
| **Após concluir** | `Manter Dataset Original Aberto` / `Carregar Dataset A no visualizador` / `Carregar Dataset B no visualizador` |

O botão de confirmação é rotulado **Dividir e Salvar**.

### Saída

Os dois parquets são gravados **em disco** na pasta de datasets
configurada (`paths.datasets` do `config.json`) — **não** na pasta do
dataset de origem —, com os nomes informados. O
dataset **atual não é modificado** — só lido. Conforme a opção
*Após concluir*, o lado A ou B pode ser aberto no visualizador; em
*Manter Dataset Original Aberto*, nada é recarregado na tela (os
parquets ficam salvos, mas a visualização continua no original).

## Balancear

![Diálogo Balancear Dataset](assets/37_balancear.png)

Uniformiza o **número de amostras por identidade**. Importante
porque algoritmos de score (PLDA, regressão logística) assumem
classes balanceadas — desbalanceamento severo distorce o LR.

### Modo de balanceamento

| Modo (rótulo na tela) | O que faz |
|---|---|
| **Cap apenas** (`CAP_ONLY`) | Para cada identidade, mantém no máximo `target_n` amostras. Identidades com **menos** que `target_n` ficam intactas (o resultado ainda pode ficar desbalanceado, mas preserva todas as identidades) |
| **Floor + Cap** (`FLOOR_AND_CAP`) | Descarta identidades com **menos** que `target_n` amostras (floor); aplica cap nas demais |
| **Auto-min (todas iguais ao mínimo)** (`AUTO_MIN_BALANCE`) | Ignora `target_n`; usa `N` = a menor contagem observada entre as identidades. Garante balanceamento perfeito ao custo de descartar muitos dados se houver identidade muito subrepresentada |

### Critério de seleção

Quando precisa **descartar** amostras de uma identidade (porque tem
mais que `target_n`), o critério decide quais ficam:

| Critério (rótulo na tela) | Como escolhe |
|---|---|
| **Aleatório (reprodutível)** (`RANDOM`) | Sorteio uniforme (com `seed` reproduzível). Não usa métrica de qualidade |
| **Por qualidade (top-N)** (`BY_QUALITY`) | Ordena por `quality_metric` (maior = melhor) e mantém as `target_n` melhores por identidade. Empates desfeitos por hash determinístico (do `face_hash`/nome do arquivo). Amostras **sem** a métrica vão para o fim da fila, e o resumo final informa quantas eram |

> 💡 `Por qualidade` é o padrão recomendado pra forense — descartar
> aleatoriamente pode tirar fotos boas e manter fotos ruins.

### Campos

| Campo | Notas |
|---|---|
| **Modo** | Os 3 modos acima |
| **N alvo por identidade** (`target_n`) | Spinner 1–100 000, **default 10**. Visível em *Cap apenas* e *Floor + Cap*; oculto em *Auto-min* |
| **Critério de seleção** | *Aleatório (reprodutível)* ou *Por qualidade (top-N)* |
| **Seed** *(Aleatório)* | Spinner inteiro. Default 42 |
| **Métrica** *(Por qualidade)* | Combo: `FIQA (ForensicFace)` / `OFIQ (ISO unified)` / `SER-FIQ (robustez)` / `Norma do embedding (AdaFace)` |
| **Após balancear** | Rádio: `Manter dataset atual aberto` (default) ou `Substituir o dataset corrente pelo balanceado` |

O botão de confirmação é rotulado **⚖ Balancear**.

### Saída

> ⚠️ **O balanceamento NUNCA grava um parquet automaticamente** — em
> qualquer modo é preciso usar **Salvar / Salvar Como** depois.

O comportamento depende da opção *Após balancear*:

- **Manter dataset atual aberto** — apenas **simula** o
  balanceamento e mostra um resumo do impacto num aviso
  (*"Pré-visualização de balanceamento"* / *"Balanceamento simulado
  (não aplicado ao dataset corrente)"*). O resultado é **descartado**:
  nada é criado, nada é aberto, o dataset na tela não muda. Serve
  para inspecionar o efeito antes de decidir.
- **Substituir o dataset corrente pelo balanceado** — troca o
  dataset **em memória** pelo resultado (nome `{nome}_Balanced`, com
  o mesmo modo de caminho, `is_aligned` e `column_specs` do original)
  e marca a tela como **modificada** (dirty). Nesse ponto, use
  **Salvar Como** para
  gravar como um novo parquet (o original em disco continua
  intacto).

## Comparar

A operação **Comparar** abre um diálogo lado-a-lado mostrando
diferenças entre o **dataset atual** e outro `.parquet` que você
selecione.

O título da janela vira **Comparação: {A} vs {B}**.

### Estrutura do diálogo

Há uma aba fixa mais **abas dinâmicas por coluna categórica**:

| Aba | Conteúdo |
|---|---|
| **📋 Resumo Geral** *(fixa)* | Três caixas — o dataset A, **Interseção (Comum)** e o dataset B — com contagem de imagens (🖼️) e identidades (👤): nas caixas de A e B, o total e a parte exclusiva de cada lado; mais uma seção **Detalhes Técnicos (Estrutura)** listando *Colunas em comum* e *Ausentes em {B}* |
| **Gênero / Etnia / Origem** *(dinâmicas)* | Uma aba por coluna categórica: tabela com `Categoria` + contagem em A + contagem em B + `Diferença` |
| *(colunas custom)* | Qualquer coluna categórica de baixa cardinalidade (até 50 valores distintos; identidade, caminho e hash ficam de fora) vira **automaticamente** uma aba de comparação, com a mesma tabela |

> ℹ️ Não existem abas separadas de "Identidades" nem de "Colunas" —
> a interseção de identidades já aparece nas caixas do **Resumo
> Geral**, e a comparação de colunas fica nos *Detalhes Técnicos*.
> Esses detalhes cobrem só as **colunas de embedding** (vetores):
> quais existem nos dois e quais de A faltam em B.

A coluna **Diferença** mostra o delta **B − A** com sinal (ex.:
`+5`, `-3`, `0`) e é colorida conforme esse sinal:

- 🟢 **verde** — valor **maior em B** (B > A)
- 🔴 **vermelho** — valor **menor em B** (B < A)
- ⚪ **cinza** — igual nos dois

> ⚠️ Atenção à convenção: a cor reflete a variação **em relação a
> B**, não "A maior que B". Verde significa que B tem mais itens
> naquela categoria do que A.

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
