# Dataset Manager — Roadmap de Funcionalidades

Brainstorm consolidado de candidatas a próximas features do Dataset Manager
(pós-migração schema-by-role, PRs 1–10).

## Requisito transversal — preview visual nas auditorias

Toda ação de sanitização (excluir, renomear identidade, mesclar
identidades, marcar com tag) precisa de **contexto visual** antes da
decisão. O perito não confirma "remover face X" só com path + descrição
— ele precisa **ver a face**.

Aplicar em todos os fluxos de auditoria/sanitização daqui pra frente:

- **Painel de preview** ao lado/abaixo da tabela de issues, mostrando a
  thumbnail da face selecionada (lazy-load do `image_path`).
- **Side-by-side** quando o issue tem 2+ paths (hash conflict, hard
  genuine/impostor pair, identidade duplicada): renderiza ambas as faces
  lado a lado.
- **Cache de QPixmap** por path pra evitar re-leitura ao re-selecionar.
- Tamanho default ~250×250 mantendo proporção; clicar abre fullsize.

Já existia no view legado (em `build_mining_tables` pre-PR 12) — vale
extrair pra widget reusável (`IssuePreviewWidget` ou similar) e plugar
em todos os pontos relevantes.

**Status:** PR 12 atual NÃO inclui isso ainda — tabela só mostra texto.
Próxima iteração imediata.

Estado atual (já entregue):

- Toolbar: Novo / Abrir / Fechar / Salvar / Salvar Como / Enriquecer / Juntar
  / Dividir / Comparar / Estatísticas / Analisar / Filtrar / Adicionar
  Coluna
- Tabela com edição célula-a-célula, vocabulário fechado renderizado como
  `QComboBox`, header context-menu (Propriedades, Remover, Mostrar/Ocultar)
- Detail panel da face selecionada
- Diálogos: Filter, Stats, Compare, Split, Enrich, Dataset Analysis (hard
  mining), New Dataset (template), Add Column, Column Properties
- Schema-by-role com `column_specs` na metadata do parquet, vocabulários
  fechados, alias resolution na leitura

---

## Tema 1 — Higiene de Identidade (alto valor forense)

O core do trabalho do perito é confiar nas identidades. Hoje o Dataset
Manager mostra IDs mas não ajuda a auditá-las.

| Funcionalidade | Valor | Custo | Notas |
|---|---|---|---|
| **Detectar identidades duplicadas** (mesma pessoa em 2 labels distintos via clustering por similaridade) | Alto | Médio | Reusa `ComparisonService` ou `HubnessAnalysisService` |
| **Detectar identidades misturadas** (mesmo label cobrindo 2+ pessoas — outlier por similaridade intra-classe) | Alto | Médio | Flag forense crítica |
| **Renomear identidade em lote** (regex / find-replace + preview de quantos rows mudam) | Médio | Baixo | Operação repetitiva no fluxo atual |
| **Mesclar duas identidades** (2 IDs → 1 com confirmação visual) | Médio | Baixo | Resolve falsos positivos do detector |

## Tema 2 — Visualização de Faces (gap visual)

A tabela mostra texto, mas o forense é visual. Hoje só dá pra ver imagem
no detail panel quando clica numa linha.

| Funcionalidade | Valor | Custo | Notas |
|---|---|---|---|
| **Modo galeria** — grid de thumbnails clicáveis, toggle ao lado da tabela | Alto | Médio | Lazy-load + cache |
| **Hover preview** — passar mouse no `image_path` mostra mini-thumb | Médio | Baixo | Tooltip personalizado |
| **Comparar pares de faces** lado-a-lado (selecionar 2 + side panel) | Médio | Baixo | Útil pra confirmar sósia |

## Tema 3 — Integração com V2 (fechamento do fluxo)

| Funcionalidade | Valor | Custo | Notas |
|---|---|---|---|
| **"Enviar pra Estratégia"** — botão na toolbar abre V2 com o dataset já vinculado | Alto | Baixo | Fluxo um-clique |
| **"Adicionar à Biblioteca Global"** — exporta o dataset atual pro `GlobalLibraryStorage` | Médio | Baixo | Fechamento do Pendrive Paradigm |
| **Re-enriquecer subconjunto** — Filtro + "enriquecer só os filtrados" (ex: só faces sem OFIQ) | Médio | Baixo | Economiza horas em datasets parcialmente enriquecidos |

## Tema 4 — Schema-by-Role: completar deferidos

Itens que ficaram fora dos PRs 1–10:

| Funcionalidade | Valor | Custo | Notas |
|---|---|---|---|
| **Drag-reorder colunas** (arrastar header pra reordenar) | Médio | Baixo | Item mais pedido em UX de planilha |
| **Filtro com colunas custom** — combos dinâmicos no filter_dialog | Médio | Médio | Reescreve `_populate_combos` + view |
| **Colunas computadas** — fórmula simbólica tipo `yaw_abs = abs(yaw)` (read-only, derivada em runtime) | Médio | Médio | Define DSL mínimo + executor |
| **Aliases editor** — editar `aliases` na Properties dialog (campo já existe no spec pós-`eddd203`) | Baixo | Baixo | Quase grátis |
| **`categorical_counts` genérico** no `DatasetStats` → custom columns aparecem no Compare | Baixo | Médio | Refactor de `DatasetStats` + `ComparisonService` + `ComparisonResult` |

## Tema 5 — Higiene de Dados (forense compliance)

| Funcionalidade | Valor | Custo | Notas |
|---|---|---|---|
| **Verificar paths órfãos** — flagar faces cujo `image_path` não existe em disco | Alto | Baixo | Detecta corrupção silenciosa |
| **Detectar duplicatas exatas** — mesmo `hash` em linhas diferentes | Médio | Baixo | Higiene básica |
| **Relatório de schema** — % NaN por coluna, colunas vazias, inconsistências | Baixo | Baixo | Lê do `column_coverage` já persistido |
| **Histórico/audit log** — quem mudou o que, quando (chain of custody) | Alto pra forense | Alto | Precisa storage; PR transversal |
| **Export CSV/Excel** com seleção de colunas | Baixo–médio | Baixo | Boa UX pra relatórios |

---

## Sugestão de priorização (top 3)

Por valor-por-LOC:

1. **Detectar identidades duplicadas/misturadas** (Tema 1) — única feature
   *forense* desta lista; reusa código existente do `ComparisonService` e
   `HubnessAnalysisService`.
2. **"Enviar pra Estratégia" + re-enriquecer subconjunto** (Tema 3) —
   fechamento do fluxo; usuário não precisa mais navegar entre abas.
3. **Modo galeria** (Tema 2) — gap visual gritante; once-and-done.

---

## Decisões abertas

- **Foco primário:** higiene de identidade (forense), UX (galeria), ou
  integração V2?
- **Drag-reorder de colunas é prioridade?** É o item de schema-by-role
  mais pedido em UX de planilha em geral.
- **Audit log / chain of custody** está no radar do projeto? Se sim, é
  uma feature transversal grande que merece PR próprio.
- **Existe alguma demanda real do uso atual** que não foi capturada no
  brainstorm? Tipo "no caso X eu sempre tenho que fazer Y manualmente"?

---

## Itens explicitamente excluídos do escopo deste roadmap

- **`EmbeddingDataset` parquet com mesma metadata role-aware** — datasets
  derivados, não precisam tratamento separado.
- **Refactor de `_face_matches_config`** em `face_dataset.py` para ser
  role-aware — hot-path, não estritamente necessário.

---

*Documento gerado a partir do brainstorm da sessão de planejamento.*
