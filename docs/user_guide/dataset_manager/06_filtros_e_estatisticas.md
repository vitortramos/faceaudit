# 06 — Filtros e estatísticas

Este capítulo cobre as três ferramentas de **visualização e
seleção** do Gerenciador de Datasets:

- 🔍 **Filtrar** — selecionar um subconjunto via expressões
- 📊 **Estatísticas** — distribuições por gênero / etnia / origem / identidade
- 📋 **Colunas** (Schema Report) — cobertura de cada coluna, exportação

Nenhuma das três muda o conteúdo do dataset — são operações
read-only (exceto a aplicação destrutiva do filtro, opcional).

## Sumário do capítulo

1. [O diálogo *Filtrar*](#o-diálogo-filtrar)
2. [Construindo expressões de filtro](#construindo-expressões-de-filtro)
3. [Salvando e carregando filtros](#salvando-e-carregando-filtros)
4. [Aplicação destrutiva vs visão temporária](#aplicação-vs-visão)
5. [O diálogo *Estatísticas*](#o-diálogo-estatísticas)
6. [O diálogo *Colunas* (Schema Report)](#o-diálogo-colunas)
7. [Exportação para CSV / Excel](#exportação)

---

## O diálogo *Filtrar*

![Diálogo Filtrar com construtor de condições](assets/33_filtrar.png)

O diálogo é um **construtor de condições**. Cada linha é uma
restrição independente; o conjunto é combinado por **AND** (todas
precisam valer pra a linha entrar no resultado).

| Elemento | Função |
|---|---|
| **Nome do filtro** | Identificador do filtro (vai pro nome do arquivo se salvar) |
| **Descrição** | Texto livre — útil pra documentar o que o filtro faz |
| **Linhas de condição** | Uma por restrição: `coluna + operador + valor` |
| **➕ Adicionar Condição** | Cria uma nova linha vazia |
| **🗑️ Remover** (por linha) | Remove a condição |
| **Salvar / Salvar Como / Abrir** | Persistência (ver seção abaixo) |
| **OK** | Aplica o filtro ao dataset atual |
| **Cancelar** | Descarta sem aplicar |

## Construindo expressões de filtro

Cada linha de condição tem três componentes:

| Componente | Comportamento |
|---|---|
| **Coluna** | Combo populado a partir dos `column_specs` do dataset. Colunas com structures complexas (embeddings) são omitidas |
| **Tipo** *(inferido)* | Numérico / String / Booleano — o widget de entrada muda conforme o tipo |
| **Valor(es)** | Conforme o tipo da coluna |

### Colunas numéricas

Aceitam **listas de valores** e **faixas**:

| Sintaxe | Significado |
|---|---|
| `0.85` | Exato |
| `0.85, 0.90, 0.95` | Lista de valores exatos (OR entre eles) |
| `0.5:1.0` | Faixa inclusiva |
| `0.5:1.0, 1.5:2.0` | Múltiplas faixas (OR entre elas) |
| `:0.5` ou `-inf:0.5` | "≤ 0.5" |
| `0.8:` ou `0.8:inf` | "≥ 0.8" |

### Colunas de string

Aceitam **valores exatos** separados por vírgula:

| Sintaxe | Significado |
|---|---|
| `Male` | Exato |
| `Male, Female` | OR entre valores |
| Botão **📋** | Abre seletor com todos os valores únicos no dataset (com checkboxes) |

> 💡 O seletor de valores únicos é **desabilitado** quando a
> coluna tem mais de 500 valores distintos — evita travar a UI ao
> renderizar uma lista enorme. Nesse caso, digite manualmente
> ou use uma coluna mais agregada.

### Colunas booleanas

Apenas uma checkbox: marcado = filtrar onde `true`; desmarcado =
filtrar onde `false`. Para *"valor ausente"* (NaN), use uma
condição extra sobre coluna numérica.

### Validação

Quando você clica **OK**:

| Situação | Resultado |
|---|---|
| Linha com coluna que não existe no dataset | Marcada em vermelho; ignorada na aplicação (com aviso) |
| Linha com valor inválido para o tipo | Marcada em vermelho; bloqueia o OK |
| Nenhuma linha preenchida | Equivalente a "sem filtro" — diálogo fecha sem mudar nada |

## Salvando e carregando filtros

Filtros podem ser persistidos em disco:

| Botão | Função |
|---|---|
| **💾 Salvar** | Salva no caminho atual (se carregado de arquivo) ou em `~/.faceaudit/filter_configs/<nome>.json` |
| **💾 Salvar Como…** | QFileDialog pra escolher caminho |
| **📂 Abrir** | Carrega de `~/.faceaudit/filter_configs/` ou arquivo arbitrário |

O formato é JSON contendo nome, descrição, e a lista de condições.
**Reutilize filtros** entre datasets — por exemplo, um filtro
"Boa qualidade" com `fiqa_score >= 0.7 AND ofiq_score >= 70` pode
ser carregado em qualquer dataset que tenha essas colunas.

## Aplicação destrutiva vs visão

> ⚠️ **Importante:** o filtro do FaceAudit é **destrutivo por
> padrão.** Ao aceitar o diálogo, o dataset em memória passa a
> ser a versão filtrada — as faces excluídas **desaparecem da
> tabela**. Para recuperar, clique 🗑️ **Limpar Filtro** na ribbon
> *Dataset*, ou re-carregue o parquet original.

> 💡 **Padrão recomendado para forenses:**
> 1. **Salvar Como** o dataset *antes* de filtrar (cria uma cópia
>    intacta)
> 2. Aplique o filtro na cópia
> 3. **Salvar Como** o subset filtrado com nome descritivo (ex:
>    `meu_dataset__alta_qualidade.parquet`)
>
> Isso preserva o original para auditoria.

A barra de status reflete o filtro ativo:

```
✅ Dataset 'X' carregado  •  450 faces (123 visíveis)  •  hash: a3f7e2c9…
```

O botão 🔍 **Filtrar** muda para **amarelo** quando há filtro
ativo, e o botão 🗑️ **Limpar Filtro** fica habilitado.

## O diálogo *Estatísticas*

![Diálogo Estatísticas — distribuições por categoria](assets/34_estatisticas.png)

Mostra **distribuições agregadas** em abas:

| Aba | Conteúdo |
|---|---|
| 👤 **Identidades** | Cada identidade + quantas amostras + % do total |
| ⚧️ **Gênero** | Male / Female / vazio + contagens |
| 🌍 **Etnia** | Cada etnia + contagens |
| 💾 **Fonte** *(origem)* | Cada `origin` + contagens |

Cada aba tem o **cabeçalho** com o número de valores únicos e uma
**tabela** ordenada por contagem decrescente:

| Categoria | Quantidade | % Total |
|---|---|---|
| `n000009` | 1 | 25.0% |
| `n000135` | 1 | 25.0% |
| `n000158` | 1 | 25.0% |
| `n000132` | 1 | 25.0% |

Valores com **menos de 1% do total** ficam destacados em **vermelho**
— indica subrepresentação que pode prejudicar análises.

> 💡 Datasets com poucos exemplos por identidade (singletons) são
> diagnosticados aqui *e* na aba de **Auditoria** ([capítulo 09](09_auditoria_forense.md))
> — o filtro de identidades singleton vira automaticamente um
> issue na auditoria.

O diálogo **respeita o filtro ativo** — se você está filtrando
faces com `fiqa_score >= 0.7`, as estatísticas se aplicam ao
subset filtrado.

### Tratamento de NaN

Valores ausentes (NaN) viram a categoria `(Não Informado)` na
tabela — visível como qualquer outra. Use isso pra detectar
"buracos" no preenchimento de metadados.

## O diálogo *Colunas*

Também chamado **Schema Report**. Acessível na ribbon *Exibir*
pelo botão 📋 **Colunas**.

![Diálogo Schema Report — relatório de cobertura por coluna](assets/35_schema_report.png)

Lista **todas** as colunas do dataset (visíveis e ocultas), com
estatísticas de cobertura:

| Coluna | O que mostra |
|---|---|
| ☑️ **Visível** | Toggle de visibilidade na tabela principal — mudanças aplicam imediatamente |
| **Nome** | Nome técnico (ex: `ofiq_sharpness_scalar`) |
| **Rótulo** | Nome amigável exibido no cabeçalho (ex: `Nitidez (OFIQ)`) |
| **Role** | `IDENTITY`, `QUALITY`, `DEMOGRAPHIC`, `POSE`, `EMBEDDING`, `CUSTOM` |
| **Source** | `INGEST`, `ENRICHMENT`, `CUSTOM` etc. |
| **Tipo** | `str`, `int`, `float`, `bool`, `array<float>` |
| **Não-nulo** | Contagem de faces com valor preenchido |
| **Total** | Contagem total de faces |
| **Cobertura** | Porcentagem (com cor: 🟢 100%, 🟡 50–99%, 🔴 <50%, ⚪ 0%) |
| **Min / Max / Média** | Para colunas numéricas |
| **Distintos** | Contagem de valores únicos |
| **Dim** | Dimensão (para embeddings) |

### Controles em lote

Botões no topo:

| Botão | Função |
|---|---|
| ☑️ **Mostrar todas** | Marca todos os toggles de visibilidade |
| ☐ **Ocultar todas** | Desmarca todos |
| 📤 **Exportar** | Abre diálogo de exportação CSV/Excel |

A visibilidade é **persistida** no `column_specs` do dataset
(atributo `visible_default`) e sobrevive ao save/reload.

## Exportação

O botão 📤 **Exportar** abre um diálogo secundário com opções:

| Opção | Comportamento |
|---|---|
| **Formato** | CSV ou Excel (`.xlsx`) |
| **Quais colunas** | Apenas visíveis / todas / seleção manual |
| **Excluir embeddings** | ☑️ por padrão — colunas `embedding_*` e arrays grandes ficam de fora (tornaria o CSV ilegível) |
| **Delimitador CSV** | `;` ou `,` (default `;` pra compatibilidade com Excel brasileiro) |
| **Codificação** | UTF-8 ou UTF-8-BOM (BOM ajuda o Excel a abrir caracteres acentuados corretamente) |

A exportação inclui **apenas faces visíveis no filtro atual** —
mesma lógica das estatísticas.

> 💡 Use a exportação CSV pra:
> - Levar dados pro Excel/Google Sheets pra cruzar com outras
>   planilhas (planilhas de caso, listas de evidência)
> - Gerar relatórios externos (laudos)
> - Backup do esquema + metadados (sem embeddings) num formato
>   universalmente legível

---

[← Capítulo 05 — Enriquecimento](05_enriquecimento.md) · [Capítulo 07 — Operações estruturais →](07_operacoes_estruturais.md)
