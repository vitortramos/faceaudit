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

## 2. Tema — sistema claro/escuro configurável (ARQUITETURA-ALVO)

### Estado atual (dívida técnica, a corrigir)
- **Não existe módulo central de tema.** ~76 arquivos da UI chamam `setStyleSheet`
  direto, com **centenas de cores hex cravadas** assumindo fundo claro (ex.:
  `#1976d2`, `#c62828`, `#2e7d32`, o âmbar `#E8A020`…).
- `interface/run.py::_apply_light_palette` (linha ~98) **força a paleta clara** como
  curativo. O próprio docstring diz que é temporário *"até o team auditar todos os
  stylesheets pra serem theme-aware"*. Não é um princípio de design — é um band-aid.
- Consequência histórica: no macOS em dark mode, os widgets que usavam `QPalette`
  ficavam escuros e os de hex fixo ficavam claros → contraste quebrado.

### Alvo: tema claro E escuro, alternável pelo usuário
1. **UM módulo de tema** (ex.: `interface/theme/theme.py`) com **tokens semânticos**
   — não cores literais. Dois dicionários `LIGHT` e `DARK` mapeando token → hex:
   `SURFACE`, `SURFACE_ALT`, `TEXT`, `TEXT_MUTED`, `BORDER`, `ACCENT` (o âmbar
   forense), `SUCCESS`, `WARNING`, `DANGER`, `INFO`, `HIGHLIGHT`, etc.
2. **`ThemeManager`** (singleton/`QObject`): guarda o modo atual, persiste a escolha
   (`config.json` ou `QSettings`), expõe `color(token) -> QColor`, e emite
   `theme_changed` ao trocar.
3. **Duas camadas de aplicação, MESMA fonte de verdade (os tokens):**
   - **QPalette** construída dos tokens — `build_palette(mode)` substitui o
     `_apply_light_palette` e é aplicada no `QApplication`.
   - **QSS global** gerado dos tokens — `app.setStyleSheet(build_qss(mode))`,
     aplicado **no nível do `QApplication`**, não por widget.
4. **Nunca cravar hex inline.** Código novo/alterado referencia tokens via o módulo
   de tema; prefira **papéis de `QPalette`** + **QSS escopado por `objectName`** a
   `setStyleSheet` inline espalhado.
5. **Troca em runtime**: uma opção em Preferências alterna o modo →
   `ThemeManager` reconstrói paleta + QSS e emite `theme_changed`; widgets que
   pintam algo custom (ex.: plots matplotlib, swatches) escutam e repintam.
6. **Contraste**: cada par token (texto sobre superfície) passa **WCAG AA nos DOIS
   modos** (delegar verificação à `qt-ui-design` §2). Plots matplotlib também
   precisam de par de temas (facecolor/edgecolor/grid dos tokens).

### Migração (incremental, NÃO big-bang)
- A cada vez que tocar num widget com hex cravado, **substitua por token**. Mantenha
  `_apply_light_palette` funcionando até a migração avançar; quando os pontos
  críticos estiverem theme-aware, o toggle habilita o dark de fato.
- Fluxo pesado é só reescrita de cor — sem tocar em lógica.

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
- **QSS no nível do app ou escopado por `objectName`**; evitar `setStyleSheet` inline
  espalhado por widget (o anti-padrão atual que gerou a dívida de tema).
- **Movimento com parcimônia**: `QPropertyAnimation` / `QGraphicsOpacityEffect`.
  Nada de animação gratuita — reforça o "cara de gerado por IA" e distrai o perito.
- **Tipografia** via `QFont` com escala clara; fontes do sistema/empacotadas (não
  web fonts). Ciente de High-DPI.

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

- **Paleta/estilo**: `interface/run.py` → `_apply_light_palette` (~L98),
  `setStyle("Fusion")` + aplicação (~L147). É aqui que o `ThemeManager` se conecta.
- **Workers**: `interface/tabs/*/workers/` (padrão `QObject` + `moveToThread`).
- **Padrões Qt do projeto**: seção "Qt/PyQt patterns e convenções" do `CLAUDE.md`
  (anti-flash `parent=`, paleta clara forçada, tuning anti-freeze no Windows).
- **Cores/assinatura**: âmbar `#E8A020` (resultados forenses),
  `STRATEGY_COLOR_PALETTE` (janela de Comparação).
