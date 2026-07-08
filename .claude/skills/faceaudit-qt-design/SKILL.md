---
name: faceaudit-qt-design
description: >-
  Design, implementação e auditoria da interface PyQt6 (QWidgets) do FaceAudit —
  uma ferramenta forense pericial. Usar ao criar/redesenhar telas, diálogos,
  painéis, widgets, temas ou estilos; ao mexer em QSS/QPalette/stylesheets; ou ao
  auditar uma UI existente. Encoda a arquitetura não-negociável (regra de negócio
  no backend, UI↔backend via DTOs, trabalho pesado em workers), o sistema de tema
  claro/escuro configurável (tokens semânticos centralizados, zero hex cravado) e
  os idiomas Qt/QWidgets do projeto. Complementa a skill `qt-ui-design` (leis de
  UX, WCAG, escala tipográfica) — não a duplica.
---

# FaceAudit — Design de UI (PyQt6 / QWidgets)

Aja como o líder de design de uma **ferramenta forense pericial** usada por peritos
e forças de segurança. As prioridades, nesta ordem: **legibilidade, densidade de
informação, consistência e defensibilidade** — acima de ousadia visual. Um laudo
se sustenta na previsibilidade e clareza da ferramenta, não em efeitos. A UI é
inteiramente em **português (pt-BR)**.

Esta skill cobre o que é **específico do FaceAudit**. Para leis de UX nomeadas
(Hick, Miller, Jakob…), acessibilidade WCAG, escala tipográfica e modo auditoria
genérico, **use a skill `qt-ui-design` em conjunto** — não repita o conteúdo dela.

---

## 1. Arquitetura — regra dura (NÃO-NEGOCIÁVEL)

Toda mudança de UI respeita esta separação. Ela existe para que **trocar o frontend
signifique apenas chamar os mesmos métodos do backend**.

- **Regra de negócio mora no backend** (`faceaudit/services/`, executors, models).
  A UI **orquestra**; não implementa lógica de domínio dentro de widgets/controllers.
- **UI ↔ backend só troca DTOs.** O frontend monta um DTO, chama o serviço, recebe
  um DTO de volta. Nada de cálculo/regra pericial vazando para a camada visual.
- **Trabalho pesado SEMPRE em worker** (`QObject` + `moveToThread`), nunca na thread
  da GUI. Progresso e resultado voltam por **signals/slots**. Ver os workers
  existentes (ex.: `interface/tabs/strategy_manager_v2/workers/`,
  `interface/tabs/dataset_manager/workers/`) como referência de padrão.
- **Separação MVC-ish**: View (widgets) declara a UI; Controller conecta
  signals/slots e despacha para serviços/workers; backend faz o trabalho.
- Se uma nova tela precisa de dados derivados, **crie/estenda um serviço no backend
  que devolve um DTO** — não faça o cálculo no widget.

> Ao revisar/gerar código de UI, se aparecer lógica de domínio, acesso a
> parquet/modelo, ou laço pesado **dentro de um widget/controller**, isso é um
> erro de camada: mova para um serviço + worker.

---

## 2. Tema claro/escuro — o sistema EXISTE em `interface/theme/`

Tema claro e escuro configurável, com fonte única de verdade em tokens semânticos.
**Nunca cravar hex inline em `setStyleSheet` — sempre rotear por tokens.** Detalhes
completos (tokens, integração) na seção "Sistema de tema claro/escuro" do CLAUDE.md.

- `interface/theme/tokens.py` — dicts `LIGHT`/`DARK` (mesmas chaves; `assert` garante).
  Tokens: `SURFACE`/`SURFACE_ALT`/`SURFACE_RAISED`, `BORDER`, `TEXT`/`TEXT_MUTED`,
  `ACCENT`/`ACCENT_BG`/`ACCENT_TEXT` (âmbar forense), `PRIMARY`/`PRIMARY_BG`/`PRIMARY_STRONG`,
  `HEADER_BG`/`HEADER_TEXT` (cabeçalho navy = fundo escuro + texto claro),
  `SUCCESS`/`DANGER`/`WARNING` (+`_BG`), `HIGHLIGHT`/`HIGHLIGHT_TEXT`, papéis de `QPalette`.
  **Modo claro é byte-idêntico** à paleta histórica.
- `interface/theme/theme_manager.py` — `ThemeManager` singleton (`instance()`,
  `initialize(app, mode)`, `apply(mode)`, `set_mode(mode)`, sinal `theme_changed`,
  `color(token)`/`qcolor(token)`). Constrói `QPalette` + QSS global dos tokens.
- `interface/theme/themed_widget.py` — **`apply_themed_stylesheet(widget, builder)`**
  (aplica e **re-aplica ao trocar de tema**) e **`current_token_fn()`** (lê o modo atual;
  para QSS dinâmico/por-item/por-estado, ex.: `setForeground` de célula).
- `interface/theme/native_titlebar.py` — `apply_titlebar_theme(widget, dark)` (barra de
  título nativa no Windows via DWM). Já aplicado na `MainGuiView`.

**Padrão obrigatório para QUALQUER QSS novo com cor:**
```python
from interface.theme import apply_themed_stylesheet, current_token_fn
# estático (self/filho fixo) — re-aplica ao vivo:
apply_themed_stylesheet(self, lambda t: "QFrame { background: " + t("SURFACE_RAISED")
    + "; color: " + t("TEXT") + "; border: 1px solid " + t("BORDER") + "; }")
# dinâmico/por-estado — lê o token na criação:
t = current_token_fn(); item.setForeground(QtGui.QColor(t("DANGER")))
```

**Persistência**: `SystemConfig.theme.mode` em `config.json`; toggle no `ConfigDialog`
(grupo 🎨 Aparência); aplicado no bootstrap (`run.py`) e ao vivo pelo `MainGuiController`.

**Status**: infra + ~65 arquivos migrados. **Contraste** deve passar WCAG AA nos dois
modos (delegar à `qt-ui-design` §2). Verifique com `widget.grab()` offscreen.

### Armadilhas / lições aprendidas (regras duras da migração)
1. **Fundo colorido = SÓ título/callout, NUNCA conteúdo/campo.** Ex.: a linha de saída
   (`OutputRowWidget`) deve ter fundo neutro; o verde vai no título da seção e no label.
2. **Nunca `color: inherit` em `::item:selected`** — no escuro resolve para PRETO sobre
   o fundo de seleção. Use `t("TEXT")` explícito (lista, árvore, comparação).
3. **Callouts de ajuda/nota = estilo NEUTRO** (surface + barra de acento cinza `TEXT_MUTED`
   + texto suave). O âmbar `ACCENT` fica reservado a **títulos/validação**, não a dicas.
4. **Ao migrar cor quase-branca/quase-preta** (ex.: `#fafdf6`, `#fff8e8` sutil) →
   **surface neutro** (`SURFACE_RAISED`), NUNCA um token de status saturado
   (`SUCCESS_BG` etc.) — senão vira destaque onde antes era só um sussurro.
5. **Barra de accent precisa ser VISÍVEL no escuro** — não use `BORDER`/`HEADER_BG`
   como barra de acento (somem no fundo escuro); use `TEXT_MUTED`/`PRIMARY`/cor semântica.
6. **`ACCENT`/`PRIMARY_STRONG`/etc. como TEXTO vs FUNDO**: navy `#2c3e50` como *fundo*
   de cabeçalho → `HEADER_BG`; como *cor de texto* de título → `PRIMARY_STRONG`/`TEXT`.
7. **Word-wrap + `QFormLayout` + `QScrollArea`**: reserve a barra de rolagem
   (`setVerticalScrollBarPolicy(ScrollBarAlwaysOn)`) — senão a última linha corta quando
   a barra aparece e estreita o viewport (QFormLayout não recomputa heightForWidth).
8. **`QColor` de pintura de canvas** (máscara, scrubber, ícone via `QPainter`) fica como
   está; **`QColor` de ESTADO de célula** (ok/erro) pode virar token via `current_token_fn()`.
9. **Verificação offscreen**: `QT_QPA_PLATFORM=offscreen` + `widget.grab().save(png)` para
   conferir contraste/legibilidade nos dois modos sem subir o app inteiro.

---

## 3. Idiomas Qt / QWidgets (fazer certo)

- **Manter o estilo `Fusion`** (`app.setStyle("Fusion")`): é cross-SO, consistente,
  e **respeita `QPalette` + QSS**. Não confiar em estilos nativos (ignoram QSS e
  seguem o dark/light do SO — origem da bagunça).
- **Sempre passar `parent=` ao construir widgets** (`QLabel(..., parent=self)`).
  Widget sem parent vira janela top-level por um instante → **flash de janelinha**
  no Windows. Para `FigureCanvas` (não aceita `parent=`): `canvas.setParent(self)`
  logo após criar. (Ver seção "Qt/PyQt patterns" no CLAUDE.md.)
- **Layout managers são o grid**: `QVBoxLayout`/`QHBoxLayout`/`QGridLayout`/
  `QSplitter`, com `spacing`/`margins` deliberados. Nada de posição fixa em pixel.
- **Model/View para listas/tabelas que CRESCEM.** A partir de alguns milhares de
  linhas (tabela de dataset, score datasets, grades de imagem), use
  `QAbstractItemModel`/`QAbstractTableModel` + `QTableView`/`QTreeView`/`QListView` —
  **não** `QTableWidget`/`QListWidget`/`QTreeWidget` populados linha a linha. A view
  virtualiza (só renderiza o viewport) e não trava a GUI; ordenação/filtro vão em
  `QSortFilterProxyModel`, não em recomputar a lista. Item-widgets continuam ok para
  listas **curtas e estáveis** (árvore de recursos, lista de instruções). Combina com
  os summarizers por amostragem já usados — o modelo puxa dado sob demanda, não
  materializa tudo. Regra de camada continua valendo: o modelo lê **DTOs** do backend
  (via serviço/worker), sem tocar em parquet/negócio dentro da view.
- **QSS via `apply_themed_stylesheet`** (nível do widget mas theme-aware) ou global;
  evitar `setStyleSheet(...)` com **hex cravado** — foi o anti-padrão que gerou a antiga
  dívida de tema (já migrada). Toda cor nova passa por token.
- **Movimento com parcimônia**: `QPropertyAnimation` / `QGraphicsOpacityEffect`.
  Nada de animação gratuita — reforça o "cara de gerado por IA" e distrai o perito.
- **Tipografia** via `QFont` com escala clara; fontes do sistema/empacotadas (não
  web fonts). Ciente de High-DPI.
- **Persistência de config/estado é `config.json`, NUNCA `QSettings`.** Decisão
  deliberada do paradigma pendrive (projeto autocontido/portátil): estado vai em
  `SystemConfig` + `JSONSerializableMixin` (por-máquina) ou nos artefatos do projeto.
  Ignore recomendações genéricas de PyQt que sugiram `QSettings` (INI/Registry/Plist) —
  elas quebram a portabilidade e espalham estado fora do projeto.

---

## 4. Filosofia visual (anti-genérico, adaptada ao forense)

- **Distinto porém disciplinado.** Uma ferramenta pericial ganha confiança pela
  consistência e legibilidade, não por risco estético. Gaste ousadia em **um
  elemento-assinatura** (o âmbar `#E8A020` dos cabeçalhos de resultado forense já é
  esse acento — reuse-o com intenção, não decore com ele).
- **Densidade de informação é feature.** Peritos leem tabelas e plots densos.
  Otimize para varredura: alinhamento, agrupamento (proximidade/similaridade),
  hierarquia clara. Não "arejar" ao ponto de esconder dados.
- **Reuse os padrões que já existem**: editor de estratégia em 3 colunas, cabeçalhos
  âmbar para resultados forenses, `STRATEGY_COLOR_PALETTE` (12 cores estáveis).
  Consistência entre diálogos vale mais que originalidade por tela.
- **Microcopy**: pt-BR, voz ativa, termos do usuário (não nomes internos do sistema);
  o mesmo nome de ação atravessa o fluxo inteiro (botão "Salvar" → toast "Salvo").

---

## 5. Processo

1. **Brainstorm** do sistema de tokens (light + dark) e wireframe ASCII do layout de
   widgets **antes** de codar.
2. **Autocrítica**: "isto parece um Fusion default genérico? tem hex cravado?
   quebra no modo escuro? algum cálculo vazou para o widget?"
3. **Implemente** via tokens + layout managers + QSS no nível do app.
4. **Rode o app** e tire **screenshot nos DOIS modos**; cheque contraste, ordem de
   foco/tab, tamanho de alvo. (Rodar com `PYTHONIOENCODING=utf-8` no Windows para o
   log de startup não quebrar em emoji.)
5. **Critique de novo** e anote o que já foi tentado.

---

## 6. Auditoria de tela existente

Ao auditar, procure e reporte (com prioridade):
- **hex cravado** e `setStyleSheet` inline espalhado → candidatos a token/central;
- widgets **sem `parent=`** (risco de flash);
- **chamada bloqueante na thread da GUI** (deveria ser worker);
- **lógica de domínio/acesso a dado pesado dentro de widget/controller** (erro de
  camada — mover para serviço + worker + DTO);
- **quebra no modo escuro** (cor que não deriva de token).

Para o checklist de UX/acessibilidade genérico, delegue à `qt-ui-design` §5.

---

## Pontos de integração no repositório

- **Tema**: `interface/theme/` (tokens, `ThemeManager`, `apply_themed_stylesheet`,
  `native_titlebar`). Bootstrap em `interface/run.py` (`ThemeManager.initialize` após
  ler `config.json`); toggle no `ConfigDialog`; aplicação ao vivo no `MainGuiController`.
- **Workers**: `interface/tabs/*/workers/` (padrão `QObject` + `moveToThread`).
- **Padrões Qt do projeto**: seção "Qt/PyQt patterns e convenções" do `CLAUDE.md`
  (sistema de tema, anti-flash `parent=`, tuning anti-freeze no Windows).
- **Cores/assinatura**: âmbar `#E8A020` (resultados forenses),
  `STRATEGY_COLOR_PALETTE` (janela de Comparação).
