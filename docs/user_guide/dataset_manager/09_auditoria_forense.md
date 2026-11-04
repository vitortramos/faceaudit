# 09 — Auditoria forense

A **Auditoria** é a área mais sofisticada do Gerenciador de
Datasets — uma suíte de verificações de integridade biométrica
e análise topológica voltada para o **uso forense** dos
embeddings.

A motivação: antes de usar um dataset em cálculo de razão de
verossimilhança, é necessário garantir que ele está livre de
contaminações (mesma foto em duas identidades, identidades
mescladas, fotos órfãs, *singletons*, etc.) e entender a
**geometria** do espaço de embeddings (clusters, hubs, outliers).

Este capítulo cobre dois diálogos relacionados:

- 🔎 **Auditar** (ribbon *Dataset*) — análise completa
- 📜 **Custódia** (ribbon *Dataset*) — log forense de operações

## Sumário do capítulo

1. [O conceito de *AuditIssue*](#o-conceito-de-auditissue)
2. [Leiaute do diálogo Auditar](#leiaute-do-diálogo-auditar)
3. [Categorias de issues](#categorias-de-issues)
4. [Ações sobre issues selecionados](#ações-sobre-issues-selecionados)
5. [Visualizações analíticas](#visualizações-analíticas)
6. [Hubness — análise topológica](#hubness--análise-topológica)
7. [Auditoria de identidades (heavy)](#auditoria-de-identidades)
8. [UMAP — projeção 2D](#umap--projeção-2d)
9. [Undo/redo](#undoredo)
10. [O diálogo *Custódia* (Audit Log)](#o-diálogo-custódia)

---

## O conceito de *AuditIssue*

Um **AuditIssue** é um achado da análise — uma violação detectada
ou uma observação relevante. Os issues de um mesmo tipo são agrupados
numa **AuditFinding** (uma linha da árvore), que define categoria,
tipo e severidade:

| Atributo | Conteúdo |
|---|---|
| **Categoria** (da finding) | Grupo (Integridade Física, Integridade Lógica (rótulos), Qualidade das Amostras, Auditoria Biométrica, Hubness, etc.) |
| **Tipo** (da finding) | Sub-tipo específico (Paths órfãos, Hash compartilhado entre identidades distintas, etc.) |
| **Severidade** (da finding) | 🔴 CRÍTICO / 🟠 AVISO / ⚪ INFO |
| **Locators** | `image_path` das faces afetadas (1 por issue em auditorias por face; N em auditorias por grupo) |
| **Detalhes** | Descrição curta + metadados (identidades envolvidas, threshold violado) |

A auditoria é **dividida em duas passagens**:

- **Quick audit** (O(N)) — rodada automaticamente ao abrir o
  diálogo, segundos
- **Heavy audit** (O(N²)) — chamada explícita via botões **🔍
  Comparar Embeddings**, **🔬 Comparar Centroids** ou **🌐
  Hubness**, minutos em datasets grandes

## Leiaute do diálogo Auditar

![Diálogo Auditar com toolbar superior, árvore de achados, detalhes e painel de visualização](assets/40_auditar.png)

Duas colunas principais, abaixo da toolbar:

### Coluna esquerda — Árvore de auditorias + histórico

Árvore hierárquica (colunas **Auditoria** e **Total**), agrupada por
categoria. Categorias são colapsáveis.

Abaixo da árvore fica o painel **📜 Histórico de mudanças** (ver
[Undo/redo](#undoredo)).

Selecionar uma auditoria na árvore popula a coluna direita.

### Coluna direita — Detalhes e visualizações

No topo, o título e o resumo da auditoria selecionada (e, para
auditorias de semântica não óbvia, uma caixa de explicação). Abaixo,
um painel que **troca automaticamente** conforme a auditoria
selecionada na árvore (internamente um `QStackedWidget`, não um
`QTabWidget`). Você não alterna abas manualmente — a página muda com
o achado.

| Página | Quando aparece |
|---|---|
| **Detalhes** | Padrão — tabela de issues (colunas Locator / Descrição / Detalhes) + preview |
| **Histograma** | Histogramas de métricas de qualidade (FIQA, OFIQ, norma do embedding) e de amostras por identidade |
| **Scatter** | Correlação entre métricas (ex.: FIQA × Norma do embedding) |
| **Boxplot** | Norma por quartil de FIQA |
| **Balance Metrics** | Métricas de balanceamento por identidade |
| **Hubness Tables** | Achado de hubness (top hubs / anti-hubs) |
| **UMAP** | Após rodar 🌌 UMAP (Embeddings) |

Na página **Detalhes**, o preview abaixo da tabela mostra as
miniaturas das faces afetadas e suporta Ctrl+Click (multi-select) e
Shift+Click (range). Faces excluídas (ações já executadas) aparecem
em **vermelho** com a marca `✗ EXCLUÍDA`; faces renomeadas/mescladas
em **laranja** com `✎ ID: antigo → novo`.

### Toolbar superior — Botões de ação

Os botões ficam numa **toolbar no topo** do diálogo (não num rodapé):

| Botão | Função |
|---|---|
| **⚙ Configurações** | Ajusta os thresholds das auditorias (qualidade por métrica disponível, mín. amostras/ID, duplicatas, mixed) |
| **🔄 Reanalisar** | Reexecuta as auditorias rápidas O(N) (útil após edição manual no dataset) |
| **🔬 Comparar Centroids** | Detecta duplicate/mixed identities via centróides de embedding |
| **🔍 Comparar Embeddings** | Roda hard mining O(N²) — pares de mesma ID com baixa similaridade + pares de IDs distintas com alta |
| **🌐 Hubness** | KNN exaustivo O(N²·D) que computa o `Nk` de cada face |
| **🌌 UMAP (Embeddings)** | Projeção 2D/3D do espaço de embeddings |

## Categorias de issues

### 1. Integridade Física

| Issue | O que detecta | Por que importa | Severidade |
|---|---|---|---|
| **Paths órfãos** | `image_path` que não existe no disco (ou está vazio) | Quebra a inspeção visual; indica perda de arquivo (deleção acidental, ingest mal-sucedido). Bloqueia hash recomputation | 🔴 CRÍTICO |
| **Hash compartilhado entre identidades distintas** | Mesmo SHA-256 sob **duas identidades** diferentes | Indica contaminação — a mesma foto está rotulada como duas pessoas. **Fatal pra LR** — viola a premissa de mapeamento 1:1 foto→identidade | 🔴 CRÍTICO |
| **Duplicatas exatas (mesmo hash)** | Múltiplas linhas com hash idêntico | Inflaciona contagens artificialmente; testes estatísticos pressupõem independência. Pode ser intencional (augmentation) ou bug de ingest | 🟠 AVISO |
| **Faces sem embedding** (uma por coluna de embedding do schema) | Faces sem vetor de embedding | Sem embedding, mining/hubness/scores falham. Indica falha no extractor | 🔴 CRÍTICO |
| **Faces sem OFIQ** / **Faces sem SER-FIQ** | Cobertura incompleta dos enriquecimentos opcionais | Só sinaliza, para você decidir se quer rodar o enriquecimento | ⚪ INFO |

### 2. Integridade Lógica (rótulos)

| Issue | O que detecta | Por que importa | Severidade |
|---|---|---|---|
| **Identidades com < N amostra(s)** | Identidades com < `min_samples` amostras (default 2) | Não dá pra fazer LR within-identity (sem variância); quebra premissas de balanceamento de classes | 🟠 AVISO |

### 3. Qualidade das Amostras

Uma auditoria por métrica disponível no dataset, rotulada
`<métrica> < <threshold>` (ex.: `FIQA < 1.5`):

| Métrica | O que detecta | Threshold default |
|---|---|---|
| **FIQA** | Faces abaixo do threshold de CR-FIQA | 1.50 |
| **OFIQ** | Faces com OFIQ unificado baixo | 25 |
| **SER-FIQ** | Robustez baixa | 50 |
| **Norma do embedding** | Magnitude do embedding atipicamente baixa — embedding "fraco" | 17.0 |

Severidade sempre `🟠 AVISO`. Não são bloqueadores — servem como
sinal pra revisão manual. Os thresholds ficam em **⚙ Configurações**.

> ⚠️ **SER-FIQ só aparece em datasets legados.** A coleta de
> SER-FIQ foi removida da interface de criação/enriquecimento, então
> datasets novos não têm a coluna `serfiq_score` e a auditoria de
> SER-FIQ baixo não surge para eles. As métricas de qualidade
> auditáveis são descobertas pelas colunas presentes no dataset —
> tipicamente FIQA, OFIQ e norma de embedding.

### 4. Auditoria Biométrica (heavy, O(N²))

| Issue | O que detecta | Por que importa | Severidade |
|---|---|---|---|
| **Pessoas com mais de um rótulo** (duplicate identities) | Pares de **identidades diferentes** cujo centróide de embedding tem similaridade ≥ threshold (default cos ≥ 0.35, ajustável em ⚙ Configurações → duplicatas) | Sugere que duas labels são a mesma pessoa (data entry, alias) | 🟠 AVISO |
| **Faces com erro de identidade** (mixed identities) | **Faces** cujo cosseno contra o centróide *leave-one-out* das demais faces do mesmo ID cai abaixo do threshold (default cos < 0.20) | Cada face fora do padrão indica provável rótulo errado — face de outra pessoa marcada com esta identidade | 🟠 AVISO |

> 💡 **Mixed Identities é por face, não por identidade.** Para cada
> face X de um ID com ≥ N amostras (default 3), o auditor calcula o
> centróide das *outras* faces do mesmo ID (excluindo X) e mede o
> cosseno X vs esse centróide. Se X entrasse no cálculo, ela puxaria a
> média na própria direção e mascararia o outlier — por isso o
> *leave-one-out*. Cada face abaixo do threshold vira uma issue própria
> (uma linha na tabela), ordenada das mais discrepantes para as menos.

### 5. Hubness (informativa, O(N²·D))

A análise de **Hubness** **não** gera issues acionáveis — a categoria
é puramente **informativa** (a coluna "Total" da árvore fica em
branco para ela). Ela produz dois achados sem locators:

| Achado | O que mostra |
|---|---|
| **Histograma de `Nk`** | Distribuição do `Nk` (quantas vezes cada face aparece na vizinhança k-NN de outras) |
| **Tabela de hubs/anti-hubs** | Top 50 faces com `Nk` **alto** (hubs) e top 50 com `Nk` **baixo** (anti-hubs, isoladas) |

Veja [seção dedicada abaixo](#hubness--análise-topológica).

### 6. Histogramas, Correlações, Balanceamento e Visualização

Categorias informativas: não geram issues por si só — apenas
renderizam os gráficos (histogramas, scatter, boxplot, métricas de
balanceamento, UMAP) para inspeção manual. O painel direito troca automaticamente para a
visualização correspondente quando você seleciona o achado na árvore.

## Ações sobre issues selecionados

Abaixo do preview há uma **barra de ações em lote**. À esquerda ficam
os controles de seleção; à direita, as ações que **aplicam mutações**
ao dataset:

| Botão | Função |
|---|---|
| **☑ Sel. tudo** | Seleciona tudo (todas as imagens do preview em auditorias par-a-par, ou todas as linhas da tabela) |
| **☐ Limpar** | Limpa a seleção corrente |
| **✏️ Renomear ID** | Renomeia em lote todas as faces com a identidade selecionada (habilitado quando a seleção tem uma única identidade) |
| **🔗 Mesclar IDs** | Funde duas ou mais identidades numa só (habilitado quando há 2+ identidades distintas na seleção) |
| **🗑 Excluir face(s)** | Remove as faces selecionadas do dataset |

A barra some quando o painel direito troca para uma visualização sem
ações aplicáveis (histograma, scatter, etc.).

### Deletar faces

Remove as faces selecionadas do dataset (mutação em memória; só
persiste no save).

| Aspecto | Comportamento |
|---|---|
| Seleção | Ctrl+Click / Shift+Click no carrossel ou na tabela |
| Justificativa | Obrigatória — o diálogo *Excluir registros do dataset* só habilita **Excluir** com texto; cancelar não exclui nada |
| Estado visual | Faces deletadas marcadas em vermelho — clique fica desabilitado |
| Undo | Disponível via histórico (ver seção abaixo) |
| Persistência | Mudança fica em memória; **Salvar** para persistir |

### Renomear identidade

Renomeia em massa **todas as faces** com a identidade do issue
(não só as selecionadas). Use quando você descobriu que "Alice"
e "Alice S." são a mesma pessoa e quer unificar.

| Aspecto | Comportamento |
|---|---|
| Escopo | Identidade inteira, não apenas locators selecionados |
| UX | Diálogo pede novo nome; em seguida, a **justificativa** (obrigatória) |
| Visual | Faces marcadas em laranja com legenda `✎ ID: old → new` |

### Mesclar identidades

Combinação de N identidades em uma. Você seleciona faces de 2+
identidades, escolhe qual delas vira o alvo (no diálogo).

| Aspecto | Comportamento |
|---|---|
| UX | Lista as identidades candidatas, ordenadas por número de amostras (sugere a maior como alvo); em seguida, a **justificativa** (obrigatória) |
| Efeito | Todas as faces dos lados não-alvos ganham a identidade do alvo |
| Undo | Disponível |

## Visualizações analíticas

As páginas de visualização da coluna direita:

### Histograma

| Eixo / elemento | Conteúdo |
|---|---|
| **X** | Valor da métrica (FIQA, OFIQ, embedding norm, etc.) |
| **Y** | Contagem de faces |
| **Visual** | Barras + KDE leve por cima |
| **Bonus** | Duas faixas abaixo mostrando as 10 faces de menor valor (cauda esquerda) e as 10 de maior valor (cauda direita) — facilita inspeção forense direta |

Insight: forma da distribuição (bimodal → subpopulações?),
caudas (outliers a investigar).

### Scatter (Correlação)

| Eixo | Conteúdo |
|---|---|
| **X** | Primeira métrica |
| **Y** | Segunda métrica |
| **Visual** | Pontos pequenos (`α=0.45`) se N < 5k; hexbin density heatmap se N ≥ 5k |
| **Overlay** | Linha de regressão OLS + Pearson r na legenda |
| **Grid** | Linhas em thresholds dividindo 4 quadrantes (com contagem em cada) |

Insight: dependência (alto r) ou ortogonalidade entre as métricas.

### Boxplot estratificado

| Eixo | Conteúdo |
|---|---|
| **X** | Quartis de FIQA |
| **Y** | Norma do embedding |
| **Visual** | Box por grupo: mediana, Q1/Q3, whiskers, outliers |
| **Overlay** | Linha conectando medianas |

Insight: tendência monotônica (linha desce → correlação
confirmada) vs flat (correlação era artefato).

### UMAP Scatter

Projeção 2D do espaço de embeddings via UMAP. Cada ponto = uma
face.

| Elemento | Função |
|---|---|
| **Eixos** | `umap_x` / `umap_y` |
| **Cor** | Selecionável via dropdown, entre os atributos presentes: identidade (top 20 + "outros / sem valor" em cinza), gênero, etnia, origem, source, pose (yaw, pitch), FIQA, norma |
| **Interação** | No modo **por identidade (centroides)**, clique num ponto → painel lookalike abre à direita mostrando k-NN por cosine (o modo por face não habilita o clique) |

Insight: geometria dos embeddings — clusters por identidade,
espalhamento, outliers topologicamente isolados.

### Hubness Tables

Duas tabelas lado-a-lado:

| Tabela | Conteúdo |
|---|---|
| **Top Hubs** | Top 50 faces com Nk **alto** |
| **Top Anti-hubs** | Top 50 faces com Nk **baixo** (zero ou perto) |

Colunas: Identidade, Foto, FIQA, Norma, Nk, Fotos do ID.

Clique numa linha → preview da imagem no painel inferior.

## Hubness — análise topológica

**Hubness** (Nk) é a contagem de quantas vezes uma face aparece
na vizinhança k-NN das demais (k=20 por padrão). Um **hub** é
uma face que é "vizinho próximo" de muitas outras.

### Por que importa pra forense

| Tipo | Risco |
|---|---|
| **Hub** | Face genericamente parecida com muitos. Em LR, pode gerar **falsos positivos** porque match com hub é estatisticamente esperado mesmo entre não-mated. Hipotese de independência dos embeddings quebra |
| **Anti-hub** | Face topologicamente isolada. Indica feature atípica (pose extrema, oclusão, qualidade ruim, mislabel). Embedding pouco confiável |

### O que fazer

A análise é **informativa** (não gera issues acionáveis na árvore) —
produz o histograma de `Nk` e a tabela top-50 de hubs/anti-hubs.
Caminhos recomendados:

- **Hubs**: investigar manualmente; possivelmente excluir os mais
  extremos do conjunto de **calibração** (para não distorcer o
  score model). Manter no conjunto de **avaliação** apenas se
  representativos da população
- **Anti-hubs**: investigar visualmente — costumam ser fotos com
  problemas (oclusão, blur extremo). Deletar ou marcar `error=True`

## Auditoria de identidades

Botão **🔬 Comparar Centroids** roda uma análise específica:

| Tipo | O que faz |
|---|---|
| **Duplicate Identities** | Para cada par (identity_A, identity_B), compara os centróides de embedding. Pares com cosseno ≥ threshold (default 0.35) viram issues |
| **Mixed Identities** | Para cada face de um ID com ≥ N amostras (default 3), calcula o cosseno da face contra o centróide *leave-one-out* das demais faces do mesmo ID; faces com cosseno abaixo do threshold (default 0.20) viram issues — candidatas a rótulo errado |

Os thresholds ficam em **⚙ Configurações**. Estes audits são O(N²)
e podem demorar minutos em datasets grandes. O cancelamento é
suportado.

## UMAP — projeção 2D

Botão **🌌 UMAP (Embeddings)** abre um diálogo de configuração e
calcula a projeção UMAP do espaço de embeddings. Parâmetros
configuráveis:

| Parâmetro | Default | Notas |
|---|---|---|
| Granularidade | Por face | **Por face** (um ponto por foto) vs **Por identidade (centroides)** — um ponto por pessoa; só este modo habilita o clique-para-lookalikes |
| n_components | 2 | Dimensão da projeção: 2 (scatter) ou 3 (rotação 3D) |
| n_neighbors | 15 | Vizinhos locais — maior = preserva estrutura global |
| min_dist | 0.1 | Distância mínima entre pontos no plot |
| metric | cosine | Métrica: cosine / euclidean / manhattan / correlation |
| max_samples | 100 000 | Limite de amostras; datasets maiores são subsampled reproduzivelmente (seed 42) |
| random_state | 42 | Fixo — reprodutibilidade |

O resultado popula a página UMAP (descrita acima). No modo **por
identidade (centroides)**, clicar num ponto abre o painel lateral de
lookalikes (k-NN por cosseno). Pode demorar ~30s em 10k faces.

## Undo/redo

Toda ação destrutiva (delete / rename / merge) é registrada num
**stack de histórico**:

| Operação | O que armazena |
|---|---|
| Delete | Backup dos `FaceInfo` removidos |
| Rename | Mapeamento `image_path → identidade antiga` |
| Merge | Mapeamento `image_path → identidade antiga` (das faces dos lados não-alvo) |

**Ctrl+Z** reverte a última ação. No painel **📜 Histórico de
mudanças** (abaixo da árvore, mais recente em cima), o botão **↶** de
cada linha desfaz aquela ação específica (não só a última) — útil
quando você fez 3 ações e descobriu que a 2ª foi um erro.

> ⚠️ O histórico é **em memória** e pertence ao diálogo — fechar o
> diálogo Auditar o descarta (as mudanças continuam no dataset).
> Recomendação: **Salvar Como** entre operações irreversíveis se
> o caso for crítico.

### Registro na cadeia de custódia

Cada exclusão, renomeação ou mescla pede uma **justificativa
obrigatória** antes de ser aplicada. Ao **fechar** o diálogo Auditar
(por qualquer caminho, inclusive o X), as ações que **continuam
aplicadas** são gravadas na cadeia de custódia, na ordem em que foram
feitas, cada uma com a sua justificativa e o instante em que foi
aplicada (`applied_at`):

| Ação | Operação | Principais campos |
|---|---|---|
| Excluir | `REMOVE_FACES` | `removed_count`, `justification`, primeiros 50 `removed_identifiers`, `removed_set_sha256`, `pending_until_save` |
| Renomear | `EDIT_LABEL` (`action=rename_identity`) | `old_identities`, `new_identity`, `changed_count`, primeiros 50 `changed_identifiers`, `changed_set_sha256`, `justification` |
| Mesclar | `EDIT_LABEL` (`action=merge_identities`) | idem, com as identidades não-alvo em `old_identities` |

Todas levam `origin=audit_dialog` e a sessão do diálogo
(`audit_session_id`, `audit_session_actions_applied`,
`audit_session_actions_undone`). O que foi **desfeito** não vira
entrada — nunca chegou ao dataset —, mas a contagem de desfeitas fica
registrada. Registrar o saldo ao fechar, e não cada ato e cada
desfazer, faz a cadeia afirmar exatamente o que foi aplicado ao
dataset. Como na Tabela, a exclusão só vai para o arquivo no save
seguinte, que grava o hash e fecha o elo. Com filtro ativo, as faces
excluídas saem também do dataset de origem.

## O diálogo *Custódia*

![Diálogo Audit Log com cadeia de custódia SHA-256](assets/43_audit_log.png)

Acessível pelo botão 📜 **Custódia** na ribbon *Dataset*. É a
**cadeia de custódia forense** — um log append-only de todas as
operações que mutaram o dataset.

### Operações rastreadas

| Operação | Quando |
|---|---|
| ✨ CREATE | Dataset instanciado pela primeira vez |
| 📂 OPEN | Lido do disco |
| 💾 SAVE / SAVE_AS | Persistido |
| 🚪 CLOSE | Sessão encerrada |
| 🏷️ EDIT_SCHEMA | Metadados/esquema de coluna editados |
| ➕ ADD_COLUMN | Coluna adicionada |
| • REMOVE_FACES | Registros excluídos pela Tabela, Galeria ou pelo diálogo Auditar, com a justificativa obrigatória (contagem, primeiros 50 identificadores, `removed_set_sha256`; pendente até o save seguinte). A cadeia recusa a entrada sem justificativa |
| ✏️ EDIT_LABEL | Identidade renomeada ou mesclada no diálogo Auditar, com justificativa (ver [Registro na cadeia de custódia](#registro-na-cadeia-de-custódia)) |
| 🔗 MERGE | Dataset criado por **Juntar** — primeira entrada do arquivo unido (ver [cap. 07](07_operacoes_estruturais.md#juntar)) |
| ✂️ SPLIT | Metade criada por **Dividir** — primeira entrada de cada parquet (ver [cap. 07](07_operacoes_estruturais.md#dividir)) |
| 🔬 ENRICH | Forensic Face ou OFIQ executados (SER-FIQ em datasets legados) |
| 🎬 VIDEO_EXTRACT | Dataset criado a partir de vídeo (seleção condicionada à âncora — ver [cap. 02b](02b_criando_dataset_de_video.md)) |
| 📤 EXPORT | Exportação CSV/Excel |
| 🗂 COPY_IMAGES | Modificação de caminhos das imagens (path_root/REBASE/REWRITE/COPY) |
| ✅ INTEGRITY_VERIFIED / ❌ INTEGRITY_FAILED | Verificação da cadeia |

O enum também define REMOVE_COLUMN e RENAME_COLUMN, mas a versão
atual não grava entradas desses tipos.

**Dataset que ainda não tem arquivo** (o resultado de um Juntar, um
subconjunto criado pela galeria): a entrada não se perde — fica
pendente no dataset e é escrita na cadeia do arquivo quando ele é
gravado, antes da entrada SAVE/SAVE_AS (que traz o hash e fecha o
elo). Essas entradas levam `recorded_at` (instante do ato) e
`written_on_save=true`.

### Estrutura do log

Sidecar JSON: `<dataset_stem>.audit.json` (mesma pasta do parquet).
Não é JSON-Lines — é um **array JSON único, indentado** (uma lista
de entradas, `indent=2`). No **Salvar Como**, o sidecar é **copiado**
(não movido) para o arquivo novo — a origem mantém a própria cadeia e
recebe uma entrada SAVE_AS com `this_file_unchanged=True`.

Cada entrada tem:

| Campo | Conteúdo |
|---|---|
| `entry_id` | UUID4 da entrada |
| `timestamp` | UTC ISO 8601 |
| `operator` | `usuário@máquina` do sistema operacional |
| `operation` | Um dos valores acima |
| `description` | Texto livre |
| `dataset_hash` | SHA-256 do parquet no momento da operação (vazio quando ainda não foi salvo) |
| `metadata` | Detalhes estruturados específicos da operação (contagens, thresholds, proveniência) |
| `parent_entry_hash` | `entry_hash` da entrada anterior (vazio só na primeira) |
| `entry_hash` | SHA-256 de todos os campos acima |

Toda entrada traz ainda `metadata["software"]`, preenchido
automaticamente ao gravar a entrada (quando a operação não o fornece):
`faceaudit_version` (versão do FaceAudit), `git_commit` (commit do
código em execução; vazio fora de um checkout git ou sem git no PATH)
e `git_dirty` (`true` quando havia alterações não commitadas em
`faceaudit/` ou `interface/` — aí o commit sozinho não reproduz o
código). É lido uma vez por sessão do programa e, por estar dentro de
`metadata`, é coberto pelo `entry_hash`.

> 💡 **Duplo-clique numa entrada** abre um diálogo com o JSON
> completo do registro — `metadata`, hashes e todos os campos. Útil
> para inspecionar a proveniência sem abrir o sidecar no disco.

### Proveniência de extração de vídeo (VIDEO_EXTRACT)

Quando um dataset é criado a partir de vídeo (ver
[cap. 02b](02b_criando_dataset_de_video.md)), o evento 🎬
**VIDEO_EXTRACT** é gravado com uma `metadata` de cadeia de custódia
que o perito precisa auditar:

| Campo | Conteúdo |
|---|---|
| `run_id` | Identificador da pasta de execução (run) que produziu o dataset |
| `video_sha256` | SHA-256 do primeiro vídeo-fonte |
| `video_sha256_all` | Lista de SHA-256 de **todos** os vídeos-fonte (multivídeo) |
| `n_videos` / `n_frames_sampled` | Quantos vídeos e quantos frames foram amostrados |
| `n_accepted_auto` | Faces aceitas automaticamente (acima do threshold da âncora) |
| `n_accepted_after_review` / `n_rejected_human` | Contagens da revisão humana (aceitas / rejeitadas) |
| `review_skipped` | Se a etapa de revisão humana foi pulada |

Além do evento no log, o próprio dataset ganha **flags de
proveniência** persistidas no parquet:

- `dataset_origin_type = VIDEO_TARGET_EXTRACTION` — marca a origem.
- `selection_warning` — o aviso *"Dataset criado por seleção
  condicionada a similaridade biométrica com uma âncora. Não
  representa amostra populacional neutra."*
- `selection_anchor_model`, `selection_threshold`, `source_run_id` —
  qual modelo/limiar de âncora e qual run geraram a seleção.

> ⚠️ **Viés de seleção.** Um dataset de origem `VIDEO_TARGET_EXTRACTION`
> foi montado escolhendo faces por similaridade com uma âncora — ele
> **não** é uma amostra populacional neutra. Não o use como população
> de referência/calibração de LR sem tratar esse viés. Veja a
> ressalva forense no [cap. 02b](02b_criando_dataset_de_video.md).

### Encadeamento (chain-of-custody)

O `entry_hash` de cada entrada **inclui** o `parent_entry_hash`
(o `entry_hash` do registro anterior). Isso forma uma **cadeia hash** (igual à
estrutura de blockchain): qualquer modificação retroativa quebra
todos os hashes a partir dela.

### Verificação

Botão **🔒 Verificar integridade** no diálogo:

1. Lê todo o log
2. Recomputa cada `entry_hash` na sequência
3. Compara com o gravado
4. Linhas íntegras ficam verdes; linhas que não batem ficam
   destacadas em vermelho

A própria verificação é **registrada no log** (autoreferente): um
evento ✅ `INTEGRITY_VERIFIED` quando a cadeia está íntegra, ou ❌
`INTEGRITY_FAILED` quando há problemas — com a contagem de entradas
verificadas e os códigos de problema na `metadata`.

Em datasets críticos pra perícia, rode a verificação **sempre que
abrir** o parquet — detecta tampering ou corrupção.

### Exportação

Botão **📤 Exportar log** salva uma cópia do sidecar JSON em local
à escolha. Use pra **preservação de evidência** — entregue ao
laudo, juntamente com o parquet, como prova de procedência.

---

[← Capítulo 08 — Pipeline de imagens](08_pipeline_de_imagens.md) · [Capítulo 10 — Integração com Estratégias →](10_integracao_estrategias.md)
