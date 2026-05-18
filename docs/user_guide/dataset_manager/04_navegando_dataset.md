# 04 — Navegando no dataset

Este capítulo cobre o **uso cotidiano** do Gerenciador de Datasets
com um dataset já carregado: como ler a tabela, editar células,
inspecionar uma imagem, alternar para a galeria e usar os menus
de contexto.

## Sumário do capítulo

1. [O leiaute em três blocos](#o-leiaute-em-três-blocos)
2. [A tabela — anatomia e operações](#a-tabela)
3. [A barra de status](#a-barra-de-status)
4. [O painel de detalhes](#o-painel-de-detalhes)
5. [A galeria de thumbnails](#a-galeria-de-thumbnails)
6. [Menu de contexto da tabela](#menu-de-contexto-da-tabela)
7. [Propriedades da coluna (cabeçalho)](#propriedades-da-coluna)
8. [Atalhos úteis](#atalhos-úteis)

---

## O leiaute em três blocos

Com um dataset carregado, a tela divide-se em três áreas:

![Dataset carregado: ribbon, tabela à esquerda e painel de detalhes à direita](assets/11_dataset_loaded_with_image.png)

1. **Ribbon** (topo): controles agrupados em 4 abas — `Arquivo`,
   `Dataset`, `Exibir`, `Imagens` (veja [capítulo 01](01_visao_geral.md)).
2. **Tabela** (esquerda, ~70% da largura): uma linha por imagem,
   com colunas configuráveis.
3. **Painel de detalhes** (direita, ~30%): preview da imagem
   selecionada + abas tabuladas com seus metadados.

O divisor entre tabela e detalhes é **arrastável** — clique e
arraste para redistribuir a largura. Esconder o painel direito
não é suportado diretamente, mas você pode arrastá-lo até
encostar na borda.

## A tabela

A tabela é onde você passa a maior parte do tempo. Algumas
características importantes:

### Esquema declarativo

Cada coluna tem um conjunto de atributos definidos no `column_specs`
do dataset:

| Atributo | O que controla |
|---|---|
| **Label** | Texto exibido no cabeçalho (pode diferir do nome técnico — ex: `Identidade` para a coluna `identity`) |
| **Role** | Papel da coluna: `IDENTITY`, `QUALITY`, `DEMOGRAPHIC`, `POSE`, `EMBEDDING`, `CUSTOM`, etc. — afeta como o app trata a coluna |
| **Editable** | Se a célula pode ser editada na UI (algumas colunas como `embedding_v2` são read-only) |
| **Visible default** | Se a coluna aparece por padrão (você pode ligar/desligar via *Schema Report*) |
| **Vocabulary** | Lista fechada de valores válidos (ex: `["Male", "Female"]` para `gender`) — a edição vira um combo box |
| **Aliases** | Nomes alternativos pra a coluna — usados pra carregar parquets antigos com esquemas diferentes |

### Colunas visíveis por padrão

Em um dataset criado pelo fluxo padrão (capítulo 02), as colunas
visíveis costumam ser:

| Coluna | Conteúdo |
|---|---|
| `file_name` | Nome do arquivo |
| `identity` | Identidade |
| `gender` | Gênero (combo: Male / Female / vazio) |
| `ethnicity` | Etnia |
| `origin` | Origem |
| `yaw`, `pitch`, `roll` | Pose facial em graus |
| `fiqa_score` | Score FIQA do ForensicFace |
| `ofiq_score` | Score OFIQ unificado |
| `det_score` | Confiança da detecção |
| `error` | Booleano — `True` indica falha em algum extractor |

Colunas como `embedding_v2`, `embedding_v4`, `ofiq_*_raw`, `bbox`,
`keypoints` ficam **ocultas por padrão** — você pode ligá-las pelo
diálogo *Schema Report* (ver [capítulo 06](06_filtros_e_estatisticas.md)).

### Tooltips de miniatura

**Ao passar o mouse** sobre uma linha, aparece um tooltip mostrando
**a miniatura da face**. Isso permite varredura visual rápida sem
precisar selecionar cada linha. O cache de miniaturas é mantido em
memória (lazy + LRU), então a primeira passagem por uma linha é
ligeiramente mais lenta.

### Ordenação

Clique no cabeçalho para ordenar por coluna. Clique de novo para
inverter. Funciona para colunas numéricas e ordenáveis (text,
booleans, números). Colunas com estruturas complexas (embeddings,
listas) não são ordenáveis e são silenciosamente ignoradas.

### Edição inline

Clique-duplo numa célula para editar:

- Colunas **com vocabulário** (`gender`, etc.) abrem um *combo box*
- Colunas **numéricas** mostram um spinner
- Colunas **booleanas** aceitam toggle direto
- Colunas **texto livre** abrem editor inline

Pressione `Enter` para confirmar ou `Esc` para cancelar.

> ⚠️ As edições ficam **em memória** até você clicar 💾 **Salvar**.
> A janela mostra um indicador *(modificado)* no título quando há
> mudanças não salvas.

## A barra de status

A barra fina logo abaixo do ribbon é a referência rápida do que
está carregado:

```
✅  Dataset 'Dataset_teste' carregado  •  4 faces (4 visíveis)  •  hash: a3f7e2c9…
```

Ao passar o mouse, um tooltip rich-text mostra:

- Caminho completo do parquet
- Descrição do dataset
- Cobertura por métrica (`Forensic Face: 100%`, `OFIQ: 100%`,
  `SER-FIQ: 75%`)
- Métricas de balanceamento (`Balanceamento: ✅ Balanceado`,
  `Amostras/ID: 1–2 (média 1.3, mediana 1.0)`)

Quando um filtro está ativo, o número de "faces visíveis" diverge
do total e o **botão Filtrar fica em amarelo** (status visual).

## O painel de detalhes

Selecione uma linha na tabela e o painel direito popula-se:

### Topo: thumbnail

A miniatura da face selecionada, dimensionada pra caber no painel.
Quando a imagem não existe no disco (caminho quebrado), aparece um
placeholder cinza com ícone de "imagem indisponível" — use
*Localizar* ([capítulo 03](03_carregando_e_localizando.md)) pra
corrigir.

### Aba *Geral / Bio*

![Aba Geral/Bio com identidade, gênero, etnia, origem, hash, dimensões](assets/12_detail_geral_bio.png)

| Campo | Origem |
|---|---|
| **Caminho** | `image_path` resolvido |
| **Hash** | `hash` (SHA-256 da imagem) |
| **Identidade** | `identity` |
| **Gênero** | `gender` |
| **Etnia** | `ethnicity` |
| **Origem** | `origin` |
| **Dimensões** | Calculadas em tempo real (largura × altura px) |
| **Status** | `error` + `error_stage` + `error_message` se houver |

### Aba *Qualidade ISO*

![Aba Qualidade ISO com barras OFIQ, SER-FIQ e FIQA](assets/12_detail_quality_iso.png)

Mostra todos os indicadores de qualidade como **barras visuais**:

| Grupo | Indicadores |
|---|---|
| **FIQA** | Score único do ForensicFace |
| **SER-FIQ** | Score único de robustez |
| **OFIQ — geral** | Unified Quality Score, Sharpness, Background Uniformity, Illumination Uniformity, Natural Colour, Compression Artifacts |
| **OFIQ — exposição** | Luminance Mean, Luminance Variance, Under/Over Exposure, Dynamic Range |
| **OFIQ — pose** | Head Pose Yaw / Pitch / Roll |
| **OFIQ — face** | Single Face Present, Eyes Open / Visible, Mouth Closed / Occlusion, Face Occlusion, Expression Neutrality, No Head Coverings |
| **OFIQ — enquadramento** | Inter Eye Distance, Head Size, Margins (Above / Below / Left / Right) |

Cada barra tem:
- Valor **scalar** OFIQ (0–100, comparável entre métricas)
- Valor **raw** OFIQ (escala nativa de cada métrica)
- Cor proporcional ao valor: ✅ verde alto → ⚠️ amarelo médio → ❌ vermelho baixo

> 💡 Use a aba *Qualidade ISO* para **investigar manualmente** por
> que uma imagem específica está com `error=True` ou foi rejeitada
> por um filtro de qualidade — os indicadores mostram exatamente
> qual atributo está fraco.

## A galeria de thumbnails

O botão 🖼 **Galeria** na ribbon *Exibir* alterna a vista da tabela
para uma **grade de miniaturas**. Útil para:

- Inspeção visual rápida de um dataset pequeno-a-médio
- Identificar pessoas (ou *outliers*) sem precisar de metadados
- Triagem visual antes de uma seleção/exclusão em massa

Limitações:
- Sem filtros nem ordenação — a ordem é a do parquet
- Performance cai com datasets grandes (> 10 mil imagens) — use
  a tabela com filtro nesses casos
- Edição inline não funciona na galeria

Para voltar à tabela, clique no mesmo botão (o ícone alterna).

## Menu de contexto da tabela

**Clique-direito numa célula** abre o menu de contexto:

| Item | Função |
|---|---|
| **Substituir valor…** | Abre o diálogo *Replace*: substitui o valor da célula (apenas esta linha, todas as linhas, ou regex global) |
| **Copiar valor** | Copia o conteúdo da célula para a área de transferência |
| **Abrir imagem externamente** | Abre o arquivo no visualizador padrão do SO |
| **Comparar par** *(em desenvolvimento)* | Marca a linha para comparação com outra |

O menu **adapta-se ao tipo da coluna**. Em colunas booleanas
você verá *Marcar todas* / *Desmarcar todas*; em colunas
numéricas, *Definir como NaN*.

## Propriedades da coluna

**Clique-direito no cabeçalho** abre o menu da coluna:

| Item | Função |
|---|---|
| **Propriedades da coluna…** | Diálogo de edição (label, role, vocabulary, aliases) |
| **Esconder coluna** | Tira a coluna da visão (sem afetar dados) |
| **Restaurar visibilidade padrão** | Volta a usar `visible_default` do spec |

O diálogo *Propriedades da Coluna* aceita:

- **Label**: rótulo exibido no cabeçalho
- **Role**: papel da coluna *(só editável em colunas custom)*
- **Vocabulary**: lista de valores válidos *(transforma a edição em combo)*
- **Aliases**: nomes alternativos pra leitura cruzada com outros parquets
- **Default value**: valor a usar quando uma linha não tem o campo

Veja também o diálogo *Metadados* (botão 🏷️ **Metadados** na
ribbon *Dataset*) — edita o mesmo `column_specs` mas em escala
de dataset, com vista de todas as colunas lado a lado
([capítulo 07](07_operacoes_estruturais.md)).

## Atalhos úteis

| Ação | Atalho |
|---|---|
| Abrir dataset | `Ctrl+O` (macOS: `⌘O`) |
| Salvar | `Ctrl+S` (`⌘S`) |
| Salvar Como | `Ctrl+Shift+S` (`⌘⇧S`) |
| Fechar dataset | `Ctrl+W` (`⌘W`) |
| Selecionar tudo na tabela | `Ctrl+A` (`⌘A`) |
| Copiar célula | `Ctrl+C` (`⌘C`) |
| Editar célula | `F2` ou clique-duplo |
| Cancelar edição | `Esc` |
| Confirmar edição | `Enter` ou `Tab` |
| Próxima linha | `↓` |
| Linha anterior | `↑` |

---

[← Capítulo 03 — Carregando e localizando](03_carregando_e_localizando.md) · [Capítulo 05 — Enriquecimento →](05_enriquecimento.md)
