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

**Status:** ✅ implementado — `IssuePreviewWidget`
(`interface/tabs/dataset_manager/dataset_analysis/issue_preview_widget.py`)
mostra as faces do issue selecionado em grade de miniaturas, com QPixmap
cacheado por path.

Estado atual (já entregue):

- Ribbons **Arquivo** (Novo Dataset / Dataset de Vídeo / Importar extração
  de vídeo / Abrir / Salvar / Salvar Como / Salvar Selecionados /
  Biblioteca / → Estratégia / Fechar), **Dataset** (Auditar / Filtrar /
  Limpar Filtro / Excluir Selecionados / Modificar Caminhos / Enriquecer /
  Coluna / Metadados / Juntar / Dividir / Balancear / Comparar /
  Estatísticas / Custódia), **Exibir** (Galeria / Colunas) e **Imagens**
  (Gerar Imagens / Editor / Processar)
- Tabela com edição célula-a-célula, vocabulário fechado renderizado como
  `QComboBox`, header context-menu (Propriedades, Remover, Mostrar/Ocultar)
- Detail panel da face selecionada
- Diálogos: Filter, Stats, Compare, Split, Balance, Enrich, Dataset
  Analysis (auditoria + hard mining), New Dataset (template), Add Column,
  Column Properties, Metadados, Modificar Caminhos, Custódia, Relatório de
  Esquema, revisão de imagens sem face detectada
- Schema-by-role com `column_specs` na metadata do parquet, vocabulários
  fechados, alias resolution na leitura

---

## Tema 1 — Higiene de Identidade (alto valor forense)

O core do trabalho do perito é confiar nas identidades. Hoje o Dataset
Manager mostra IDs mas não ajuda a auditá-las.

| Funcionalidade | Valor | Custo | Notas | Status |
|---|---|---|---|---|
| **Detectar identidades duplicadas** (mesma pessoa em 2 labels distintos via clustering por similaridade) | Alto | Médio | Reusa `ComparisonService` ou `HubnessAnalysisService` | ✅ auditoria `duplicate_identities` (`DatasetAuditService`), passe opt-in no Auditar |
| **Detectar identidades misturadas** (mesmo label cobrindo 2+ pessoas — outlier por similaridade intra-classe) | Alto | Médio | Flag forense crítica | ✅ auditoria `mixed_identities`, passe opt-in no Auditar |
| **Renomear identidade em lote** (regex / find-replace + preview de quantos rows mudam) | Médio | Baixo | Operação repetitiva no fluxo atual | 🟡 parcial — renomeia todas as faces de UMA identidade a partir do diálogo Auditar; regex/find-replace não existe |
| **Mesclar duas identidades** (2 IDs → 1 com confirmação visual) | Médio | Baixo | Resolve falsos positivos do detector | ✅ no diálogo Auditar, com desfazer |

## Tema 2 — Visualização de Faces (gap visual)

A tabela mostra texto, mas o forense é visual. Hoje só dá pra ver imagem
no detail panel quando clica numa linha.

| Funcionalidade | Valor | Custo | Notas | Status |
|---|---|---|---|---|
| **Modo galeria** — grid de thumbnails clicáveis, toggle ao lado da tabela | Alto | Médio | Lazy-load + cache | ✅ botão **Galeria** (ribbon Exibir), `GalleryWidget` |
| **Hover preview** — passar mouse no `image_path` mostra mini-thumb | Médio | Baixo | Tooltip personalizado | ✅ tooltip com miniatura em qualquer célula da linha (`DatasetTableModel`) |
| **Comparar pares de faces** lado-a-lado (selecionar 2 + side panel) | Médio | Baixo | Útil pra confirmar sósia | ✅ `compare_pair_dialog` (diálogo, não painel lateral) |

## Tema 3 — Integração com V2 (fechamento do fluxo)

| Funcionalidade | Valor | Custo | Notas | Status |
|---|---|---|---|---|
| **"Enviar pra Estratégia"** — botão na toolbar abre V2 com o dataset já vinculado | Alto | Baixo | Fluxo um-clique | ✅ botão **→ Estratégia** (ribbon Arquivo) |
| **"Adicionar à Biblioteca Global"** — exporta o dataset atual pro `GlobalLibraryStorage` | Médio | Baixo | Fechamento do Pendrive Paradigm | ✅ botão **Biblioteca** (ribbon Arquivo) |
| **Re-enriquecer subconjunto** — Filtro + "enriquecer só os filtrados" (ex: só faces sem OFIQ) | Médio | Baixo | Economiza horas em datasets parcialmente enriquecidos | 🟡 parcial — o **Enriquecer** processa só as faces em que falta o item pedido (salvo *Reprocessar*), o que cobre o exemplo; não há opção explícita "só os filtrados" |

## Tema 4 — Schema-by-Role: completar deferidos

Itens que ficaram fora dos PRs 1–10:

| Funcionalidade | Valor | Custo | Notas | Status |
|---|---|---|---|---|
| **Drag-reorder colunas** (arrastar header pra reordenar) | Médio | Baixo | Item mais pedido em UX de planilha | ✅ `sectionMoved` no `ManagerTableController` |
| **Filtro com colunas custom** — combos dinâmicos no filter_dialog | Médio | Médio | Reescreve `_populate_combos` + view | ✅ o filtro monta as opções a partir do `column_specs` do dataset |
| **Colunas computadas** — fórmula simbólica tipo `yaw_abs = abs(yaw)` (read-only, derivada em runtime) | Médio | Médio | Define DSL mínimo + executor | ✅ `ColumnSpec.computed_expr`; opção "Coluna computada" no diálogo **Coluna** |
| **Aliases editor** — editar `aliases` na Properties dialog (campo já existe no spec pós-`eddd203`) | Baixo | Baixo | Quase grátis | ✅ `column_properties_dialog` (sempre editável) |
| **`categorical_counts` genérico** no `DatasetStats` → custom columns aparecem no Compare | Baixo | Médio | Refactor de `DatasetStats` + `ComparisonService` + `ComparisonResult` | ✅ `DatasetStats.categorical_counts` + diffs por coluna em `ComparisonResult` |

## Tema 5 — Higiene de Dados (forense compliance)

| Funcionalidade | Valor | Custo | Notas | Status |
|---|---|---|---|---|
| **Verificar paths órfãos** — flagar faces cujo `image_path` não existe em disco | Alto | Baixo | Detecta corrupção silenciosa | ✅ auditoria `orphan_paths` |
| **Detectar duplicatas exatas** — mesmo `hash` em linhas diferentes | Médio | Baixo | Higiene básica | ✅ auditorias `exact_duplicates` e `hash_conflicts` |
| **Relatório de schema** — % NaN por coluna, colunas vazias, inconsistências | Baixo | Baixo | Lê do `column_coverage` já persistido | ✅ botão **Colunas** (ribbon Exibir), `SchemaReportDialog` |
| **Histórico/audit log** — quem mudou o que, quando (chain of custody) | Alto pra forense | Alto | Precisa storage; PR transversal | ✅ cadeia de custódia por dataset (`AuditLogService`), botão **Custódia** |
| **Export CSV/Excel** com seleção de colunas | Baixo–médio | Baixo | Boa UX pra relatórios | ✅ "Exportar para CSV/Excel…" no relatório de colunas (`ExportCsvDialog`) |

---

## Sugestão de priorização (top 3)

Por valor-por-LOC:

1. **Detectar identidades duplicadas/misturadas** (Tema 1) — única feature
   *forense* desta lista; reusa código existente do `ComparisonService` e
   `HubnessAnalysisService`.
2. **"Enviar pra Estratégia" + re-enriquecer subconjunto** (Tema 3) —
   fechamento do fluxo; usuário não precisa mais navegar entre abas.
3. **Modo galeria** (Tema 2) — gap visual gritante; once-and-done.

**Situação:** os três itens foram entregues (o re-enriquecer subconjunto,
parcialmente — ver Tema 3). Do roadmap inteiro, ficam em aberto apenas a
renomeação em lote por regex/find-replace (Tema 1) e a opção explícita de
enriquecer só os registros filtrados (Tema 3).

---

## Decisões abertas

- ~~**Foco primário:** higiene de identidade (forense), UX (galeria), ou
  integração V2?~~ Resolvida — os três temas foram implementados.
- ~~**Drag-reorder de colunas é prioridade?**~~ Resolvida — implementado.
- ~~**Audit log / chain of custody** está no radar do projeto?~~ Resolvida
  — a cadeia de custódia por dataset existe e registra criação,
  enriquecimento, exclusões (com justificativa), Salvar Como e o
  processamento de imagens.
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
