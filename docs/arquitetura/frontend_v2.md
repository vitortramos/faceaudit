# Frontend: Strategy Manager V2, visualizações e padrões Qt

> Conteúdo migrado do `CLAUDE.md` da raiz na reestruturação de 2026-11. As regras permanentes continuam resumidas lá; aqui fica o detalhe.

## Visualizações e filtros por fase

**Filtro de instruções por fase (UI).** O menu "➕ Adicionar" da `InstructionListWidget` filtra os tipos disponíveis pela fase da aba: `_TRAINING_ONLY_TYPES = {PIPELINE_TRAINING, SIMILARITY_MODEL_TRAINING, CALIBRATION_MODEL_TRAINING, TYPICALITY_MODEL_TRAINING}` só na Preparação; `_VALIDATION_ONLY_TYPES = {STRATEGY_EVALUATION}` só na Validação; `_CASEWORK_ONLY_TYPES = {CASEWORK_VISUALIZATION}` só na Análise de Caso.

**Visualização de resultados forenses (`ValidationResultViewerDialog`).** A instrução `STRATEGY_EVALUATION` produz um `StrategyValidationResult`. O `InstructionConfigPanel` mostra um botão `📊 Visualizar Resultado de Validação`; clique abre um `QDialog` 1100×780 com header âmbar + `QTabWidget` de **7 abas de plot + 1 aba de Métricas**: **Distribuição de LRs**, **Tippett Plot**, **Curva DET** (escala normit com marca no EER), **Calibração ECE** (curva real + curva mínima PAVA in-sample com área de gap diagnóstico), **Curva de Calibração** (2-panel: scatter raw×logLR + pós-afim Brümmer + Δ resíduo), **Diagnóstico Afim em Escala log-LR** (3 camadas: y=x sem recalibração + pós-afim + PAVA oracle in-sample), **Quantis por Classe** (2-panel: percentis P1..P99 per-classe com halos críticos + bottom panel de Contribuição ao Cllr por bin). Cada aba tem `FigureCanvas` + `NavigationToolbar`. Lazy hydrate via `_resource_loader.hydrate` quando o payload está no disco.

**Terminologia PAVA (Morrison 2021).** Toda a UI/help/tooltips foi reframed: `Cllr_min` é "Cllr após recalibração monotônica PAVA ajustada no PRÓPRIO conjunto avaliado" — referência diagnóstica **in-sample**, NÃO medida operacional de calibração nem teto teórico real. `cal_loss = Cllr − Cllr_min` é "gap em relação à PAVA in-sample" — útil em desenvolvimento e comparação inter-estratégias, não deve ser apresentado isoladamente como grau de calibração em laudo. PAVA é chamada de "oracle in-sample; não operacional" nos labels de plot (Diagnóstico Afim e ECE). O subtítulo do Diagnóstico Afim mostra `pava_residual_gap = Cllr_affine − Cllr_min` (renomeação só de string visível; campo Python `nonlinear_cal_loss` em `ValidationMetrics.affine_cal_loss` preservado). A property `is_well_calibrated` (heurística `cal_loss < 0.1`) está marcada DEPRECATED e a linha equivalente foi REMOVIDA da tabela de métricas — Morrison critica explicitamente esse julgamento binário in-sample. A defensibilidade forense da calibração depende sobretudo do modelo de calibração escolhido (LogReg como default parcimonioso) e da representatividade dos dados de calibração, não das métricas PAVA in-sample.

**Visualização forense de caso (`CaseworkVisualizationViewerDialog`).** A instrução `CASEWORK_VISUALIZATION` produz um `CaseworkVisualizationResult` com `score_type` (RAW/CALIBRATED), arrays `genuine_scores`/`impostor_scores`/`case_scores`, `eer_threshold` opcional (computado on-the-fly em modo RAW via FNR=FPR crossover), e `case_summary` pré-computado.

**A figura vai para o laudo, e o desenho segue disso.** `draw_casework_visualization(figure, result, *, show_title, show_kde, legend_below, show_eer_threshold)`. Histograma puro por padrão (`show_kde=False`) — a curva KDE sugere suavidade que a amostra não tem. Legenda embaixo (`legend_below=True`), em português e na linguagem do leitor do laudo: "Mesma pessoa" em verde e "Pessoas diferentes" em vermelho, nunca "genuínos" e "impostores"; o caso em azul, e com `n=1` a legenda diz "Caso (n=1, marca)" porque uma linha vertical não é distribuição. Título "Contextualização do(s) escore(s) do caso", eixo x "Escore de similaridade", eixo y "Densidade de frequência (%)". Limiar de EER fora por padrão (`show_eer_threshold=False`). Número em convenção brasileira nos DOIS eixos e na legenda, via `faceaudit/models/plot_number_format.py` — tratar só a legenda e deixar as marcas com ponto exibiria as duas convenções na mesma figura, o que chama mais atenção que o ponto uniforme. Faixas do histograma por `max(10, min(50, round(sqrt(n))))`.

## Painel de configuração e painel de recurso

**Edição inline de configs editáveis (botão ✎ no `InstructionConfigPanel`).** Dropdowns que selecionam recursos do tipo `PIPELINE_CONFIG` ou `MINING_CONFIG` ganham um botão `✎` à direita do combo. O controller V2 (`_on_resource_edit_requested`) despacha por `resource.resource_type`: PIPELINE_CONFIG → `PipelineCreatorController(existing_config=..., lock_name=True)`; MINING_CONFIG → `NewMiningConfigDialogController(parent=..., lock_name=True) + ctrl.load_config(...)`. Helpers compartilhados: `_hydrate_resource_if_needed` (lazy-load do payload) e `_persist_edited_config_resource` (atualiza payload em memória + contexto vivo, re-persiste sobre o mesmo path preservando o nome físico do arquivo, propaga novo `artifact_ref`, **e dispara cascade transitivo** via `_invalidate_dependents_of`).

**`lock_name=True` em editores de config inline.** Tanto `PipelineCreatorController` quanto `NewMiningConfigDialogController` aceitam `lock_name: bool = False`. Quando `True`: `input_name` recebe `setReadOnly(True)` + tooltip + stylesheet cinza. Razão: o `resource_id` exibido do recurso é o stem do arquivo no disco (`Path(ref.path).stem`); editar o `config.name` interno **não** muda o nome do arquivo nem o `resource_id` mostrado nos dropdowns/árvore. **Sempre passar `lock_name=True`** quando o dialog é aberto via botão ✎.

**`ValueFieldSpec.triggers_rebuild` para campos que alteram outros specs.** O dataclass `ValueFieldSpec` tem `triggers_rebuild: bool = False`. Quando `True`, o `InstructionConfigPanel` conecta o widget ao slot `_on_value_triggers_rebuild`, que aplica o valor ao DTO, chama `_sync_declared_io()` e dispara rebuild. Caso canônico: `PipelineTrainingInstruction.apply_to_training_datasets`.

**`ResourceFieldSpec.triggers_rebuild` análogo para dropdowns de recurso.** Útil quando outros specs dependem do recurso escolhido. Caso canônico: `DatasetBalancingInstruction.dataset_resource_id` é marcado `triggers_rebuild=True` porque o `quality_metric` usa `dynamic_choices` que filtra as métricas pelo dataset escolhido.

**`ChoiceFieldSpec.dynamic_choices` para opções filtradas em runtime.** Hook genérico: callable `(available_resources) → Optional[Tuple[str, ...]]`. **Caso canônico**: `DatasetBalancingInstruction._available_quality_metric_choices` consulta `QualityMetricAvailabilityService` e retorna apenas as métricas com cobertura ≥ 50% no dataset selecionado.

**`get_hidden_fields()` + `REACTIVE_FIELDS` + `get_readonly_fields()` em DTOs aninhados (PLDA).** O `NestedDTOInlineWidget` suporta visibilidade dinâmica de fields via hooks no DTO inline: método `get_hidden_fields() → Set[str]`, atributo `REACTIVE_FIELDS: Tuple[str, ...]` e método `get_readonly_fields() → Set[str]` (campos visíveis mas desabilitados). **Caso canônico**: `PLDASimilarityModelConfig` define `REACTIVE_FIELDS = ("plda_variant", "plda_mode", "shrinkage_mode")` e `get_hidden_fields()` retorna, por variante: SIMPLE → `{"regularization", "shrinkage_mode", "shrinkage_intensity", "cross_condition_within_mode"}`; TWO_COVARIANCE → `{"em_iterations", "plda_mode", "latent_dim", "shrinkage_mode", "shrinkage_intensity"}`; SHRINKAGE_TWO_COVARIANCE → `{"em_iterations", "plda_mode", "latent_dim"}` + `shrinkage_intensity` quando `shrinkage_mode != MANUAL`. `get_readonly_fields()` retorna `{"latent_dim"}` em SIMPLE + FULL_RANK (valor ignorado em runtime).

**Recursos do projeto: PIPELINE_CONFIG e MINING_CONFIG.** Ambos são recursos de PROJETO (vivem no disco em `pipelines/*.json` e `mining_configs/*.json`) e **não estão necessariamente declarados em `LRStrategyDefinition.resources`** — só aparecem nos dropdowns via `_collect_project_resources()` (combina recursos de definitions + scan de disco). Para edição/lookup, sempre usar `_collect_project_resources()`.

**Detalhes de modelos no `ResourceInfoWidget`.** O painel de propriedades do recurso tem seções dedicadas, visíveis apenas para os tipos correspondentes: **Detalhes do Modelo de Similaridade** (Cosine/PLDA — tipo, dim. entrada/latente, EM iters, variante PLDA), **Detalhes do Modelo de Calibração** (LogReg: `C (efetivo)` lido de `best_C_` pós-GridSearch; KDE: kernel, bandwidths, clipping log-LR), **Detalhes do Score Dataset** (`Total de pares`, `Genuínos (Hp)`, `Impostores (Hd)`, `Identidades`, `Calibrados (log-LR)`). Lazy-load via `_load_pickle(abs_path)`.

**Métricas de balanceamento de identidade no `ResourceInfoWidget` (DATASET stats).** Linhas adicionais alimentadas por `DatasetStats.balance_metrics`: **`Balanceamento`** mostra verdict (`✅ Balanceado`, `🟢 Aceitável`, `🟡 Moderadamente desbalanceado`, `🔴 Severamente desbalanceado`); **`Amostras/ID`** mostra `min–max (média, mediana)`. Verdict por **worst-of-two** entre CV (std/mean) e imbalance ratio (max/min). Alimentado por `DatasetStats.samples_per_identity: Dict[str, int]`.

**Stale-cache check via mtime nos sidecars de stats.** Tanto `_load_scores_details` quanto `_load_stats` comparam `parquet.mtime` com `cache.mtime` antes de usar o sidecar. Se `parquet.mtime > cache.mtime + 1.0`, o cache é invalidado e recomputado. Necessário porque re-execução de instruções reescreve o parquet sobre o mesmo path.

**Tuning anti-freeze no Windows (`interface/run.py` topo).** Antes de qualquer import pesado: `sys.setswitchinterval(0.001)` reduz o intervalo de troca da GIL de 5ms para 1ms; e em `sys.platform=="win32"`, `ctypes.windll.user32.DisableProcessWindowsGhosting()` desabilita o "ghost window" do Windows. `HashService.calculate_file_hash` usa `hashlib.file_digest` em Python ≥ 3.11.

**Tema claro/escuro via `ThemeManager` (substituiu a paleta clara forçada).** O sistema de tema já existe — ver **"Sistema de tema claro/escuro (`interface/theme/`)"** na seção Qt/PyQt patterns. No bootstrap (`interface/run.py`), `_apply_light_palette(app)` roda como default seguro imediatamente após `setStyle("Fusion")`, mas é **sobrescrito** logo em seguida por `ThemeManager.initialize(app, mode)` (após ler `config.json`), que constrói a `QPalette` + QSS global a partir dos tokens do modo escolhido. O modo claro dos tokens é byte-idêntico ao `_apply_light_palette`. Regra permanente: **nunca cravar hex em `setStyleSheet` — usar `apply_themed_stylesheet`/`current_token_fn`** (tokens em `interface/theme/tokens.py`).

## Strategy Manager V2 — Frontend

A aba é registrada em `interface/main_gui_controller._init_strategy_manager_v2()`. Recebe `global_storage` (`GlobalLibraryStorage`, opcional) e `app_config` (para resolver os diretórios iniciais dos FileDialogs).

### Estrutura de pastas

```
interface/tabs/strategy_manager_v2/
├── strategy_manager_v2_controller.py   ← controller central (QObject); dono do projeto e do estado de execução
├── strategy_manager_v2_view.py         ← QStackedWidget: boas-vindas (0) + área de trabalho (1), splitter [LeftPanel | StrategyEditor]
├── seal_project_dialog.py              ← confirmação de "Selar projeto" (com simulação); serviço em faceaudit/services/project_seal_service.py
├── toolbar/
│   ├── global_resources_toolbar.py     ← Pipeline / Mining / Split / Gerenciar Biblioteca
│   ├── project_toolbar.py              ← Novo / Abrir / Salvar / Fechar / Executar / Salvar como Template
│   └── ribbons/                        ← biblioteca_ribbon.py, projeto_ribbon.py
├── left_panel/
│   ├── left_panel_widget.py            ← QSplitter vertical com 3 seções; repassa sinais
│   ├── strategy_list_widget.py         ← lista de LRStrategyDefinition; Novo/Duplicar/Remover; Executar/Salvar como Template/Comparar
│   ├── resource_tree_widget.py         ← árvore por StrategyResourceType; "Adicionar Recurso"
│   └── resource_info_widget.py         ← metadados, uso, detalhes de pipeline e estatísticas do recurso
├── strategy_editor/
│   ├── strategy_editor_widget.py       ← QTabWidget com 3 abas (Preparação/Validação/Caso), cada uma com lista + config + resultados (3 colunas)
│   ├── instruction_list_widget.py      ← lista arrastável com ▶ Executar e estados (🔒/✅)
│   ├── instruction_config_panel.py     ← formulário dinâmico (ResourceFieldSpec) + Validar + seção "Regras de Validação"
│   ├── instruction_results_panel.py    ← 3ª coluna: saídas + gráfico + 🔍 Expandir; cache por (strategy_id, instruction_id)
│   ├── add_validation_rule_dialog.py   ← "➕ Adicionar Regra" (tipos configuráveis + templates)
│   ├── save_validation_rule_template_dialog.py ← "📤" salva regra como template
│   ├── face_comparison_dialog.py       ← duas faces lado a lado, com os pontos do dataset e o escore
│   ├── section_header.py / widget_helpers.py ← cabeçalho de seção; WrappedLabel e enable_height_for_width
│   ├── result_widgets/
│   │   ├── score_distribution_widget.py     — histograma + KDE (gen × imp), use_log_lr para SCORE_CALIBRATION
│   │   ├── validation_result_widget.py      — QTabWidget com 7 abas de gráfico + aba Métricas
│   │   ├── casework_visualization_widget.py — figura do caso (ver draw_casework_visualization)
│   │   ├── hubness_report_widget.py         — 5 abas (captura do HubnessPlotService via monkey-patch de plt.show + PNG)
│   │   ├── hubness_ablation_widget.py       — sobreposição KDE dos cenários de ablação
│   │   ├── typicality_distribution_widget.py — distribuição de tipicidade
│   │   ├── png_snapshot_widget.py           — exibição de PNG pré-renderizado
│   │   └── score_comparison_table_widget.py — tabela das comparações com miniaturas
│   ├── validation_result_viewer_dialog.py / casework_visualization_viewer_dialog.py
│   ├── hubness_report_viewer_dialog.py / hubness_ablation_viewer_dialog.py / typicality_distribution_viewer_dialog.py
│   ├── graph_viewer_dialog.py          ← diálogo genérico de Figure (trajetória 3D, distribuição)
│   ├── output_row_widget.py            ← linha por saída com estado (⬜/🟡/🟢) + resumo preguiçoso
│   └── nested_dto_widget.py            ← editor inline de DTOs aninhados (PLDA / LogReg / KDE / políticas)
├── comparison/                         ← janela de comparação de estratégias
│   ├── comparison_window.py / comparison_kind.py / availability_resolver.py / comparison_left_panel.py
│   ├── flow_layout.py                  ← FlowLayout canônico do Qt + STRATEGY_COLOR_PALETTE
│   └── views/                          ← calibration_distribution / validation_results / casework_results
├── dialogs/                            ← new_project_v2 / new_strategy / manage_library / add_resources_from_library
└── workers/
    ├── strategy_execution_worker.py    ← executa StrategyExecutor.execute() (todas as fases)
    ├── instruction_execution_worker.py ← executa uma instrução
    ├── seal_project_worker.py          ← selagem do projeto
    └── project_io_worker.py            ← sem uso hoje; abrir é síncrono
```

### Decisões de desenho do frontend

**Project open is synchronous with WaitCursor, not threaded.** On Windows, `ApplicationModal` dialogs prevent queued signal delivery from worker threads, causing `LogProgressDialog` to hang forever.

**"Adicionar Recurso" lives in the left panel, not the toolbar.** `ResourceTreeWidget` tem `QToolButton` com `MenuButtonPopup` mostrando 4 opções (Dataset / Split Config / Mining Config / Pipeline Config).

**Resource discovery = definitions scan + disk scan.** `_collect_project_resources()` faz duas passagens: (1) resources declarados nas definitions, (2) arquivos em disco via `storage.list_artifacts()`.

**`app_config.paths.*` drives FileDialog initial directories.** O controller lê `app_config.paths.<attr>` (`datasets`, `split_configs`, `mining_configs`, `pipelines`).

**Per-instruction execution com dependency tracking e auto-persistence.** O controller mantém: `_execution_context`, `_executed_instruction_ids`, `_executor`. Run de uma instrução: `_ensure_executor_and_context()` (lazy), `can_execute_instruction()` (gate), `InstructionExecutionWorker` (thread). Após sucesso: `_persist_produced_resources()` + `_refresh_instruction_states()` + `_refresh_produced_resources()`.

**Three-tab strategy editor com 3 colunas.** `QTabWidget` com 3 abas (Preparação, Validação, Análise de Caso). Cada aba: splitter horizontal de 3 colunas — esquerda `InstructionListWidget` (~200px); meio `InstructionConfigPanel` (~400px); direita `InstructionResultsPanel` (~600px). `InstructionConfigPanel` recebe `outputs_in_results_panel=True` quando instanciado pelo `StrategyEditorWidget`.

**Painel de resultados (3ª coluna) — `InstructionResultsPanel`.** Renderiza, para a instrução selecionada: outputs (`OutputRowWidget`) + gráfico inline + botão `🔍 Expandir`. Mapeamento de plot por tipo via `_instruction_plot_kind`. Datasets grandes em disco mostram botão `📊 Carregar e visualizar` em vez de hidratar automaticamente.

**O cache dos cards é por `(strategy_id, instruction_id)`, nunca só pelo id.** `instruction_id` é único DENTRO da estratégia e só dentro dela — o próprio DTO declara isso — e **duplicar uma estratégia copia os ids das instruções**. Com a chave só no id, abrir a cópia servia o card construído para a original; como o card carrega a instrução que o construiu, o painel procurava o gráfico do recurso de saída da OUTRA estratégia dentro da pasta da estratégia aberta, não encontrava, e caía em "Execute para gerar visualização" numa instrução já executada. Silencioso de ponta a ponta. O editor chama `results_panel.set_strategy_id(definition.strategy_id)` ANTES de qualquer instrução chegar. E `invalidate_all` zera `_current_instruction` ANTES de descartar: ele passava por `invalidate_instruction`, que RECONSTRÓI o card da instrução selecionada (para que a re-execução apareça de imediato), e na troca de estratégia a selecionada é a da estratégia abandonada — o descarte que devia impedir a colisão era o que a instalava.

**Instruções de escore têm três abas: Gráfico, Comparações e Estatísticas.** Vale para `SIMILARITY_SCORING`, `PAIR_MINING`, `TYPICALITY_PAIR_SCORING` e `SCORE_CALIBRATION`. O gráfico mostrava a FORMA do conjunto e não deixava saber qual escore saiu de qual par de faces, que num confronto é o resultado do exame. As contagens saíram da linha de saída e foram para a aba: a linha diz o que o artefato É (caminho, hash, tamanho) e a aba diz o que ele CONTÉM — daí `ResourceInfoWidget(compact=True, show_basic_info=False)` na aba e `show_scores_details=False` na linha. A tabela é montada só quando o usuário abre a aba; a varredura a cada `update_state` seria trabalho jogado fora, já que a aba que abre por padrão é a do gráfico.

**A tabela de comparações mostra as duas faces, e o duplo clique amplia.** Quando as imagens são localizáveis, cada linha traz a miniatura de cada face e o escore, com identidade, nome do arquivo e início do hash na dica do mouse; a coluna de classe some em conjunto de caso, onde seria constante. Sem imagens alcançáveis a tabela volta inteira ao formato em texto — caso de `PAIR_MINING` e `SCORE_CALIBRATION`, cujas instruções não nomeiam os datasets de origem, e de dataset movido. As miniaturas são decodificadas em thread própria (`ThumbnailLoadWorker`, reusado do Dataset Manager) e o pareamento miniatura↔célula é guardado pelo **item** da tabela, não pela coordenada, para sobreviver à ordenação durante a carga. O duplo clique abre `FaceComparisonDialog` e encontra a linha pelo índice de backend gravado no item (`_PAPEL_INDICE_DA_LINHA`) — a tabela é ordenável, e abrir a comparação errada seria pior que não abrir.

**`FaceComparisonDialog` desenha os pontos DO DATASET, nunca redetecta.** São os pontos que definiram o alinhamento de onde saiu o embedding, e portanto descrevem a face que o sistema de fato comparou; uma detecção feita na hora poderia marcar outra face do quadro e ninguém notaria. Sem geometria a janela abre do mesmo jeito, sem marcação, e diz por quê. O enquadramento pela face vem ligado quando há caixa dos dois lados: uma foto de documento e um quadro de vídeo têm resoluções e distâncias completamente diferentes, e sem isso os rostos aparecem lado a lado em tamanhos incomparáveis, justamente na janela que existe para compará-los. A marcação usa a MESMA transformação da imagem (`_transformacao` + `_recorte_de_origem`), o que a mantém sobre a face em qualquer tamanho de janela e em qualquer enquadramento. Cores da marcação são fixas de propósito — têm de ser legíveis sobre pele clara e escura nos dois temas —, o que o projeto já trata como desenho e não chrome.

**Automatic placeholder sync e resource persistence.** `_sync_all_output_placeholders()` cria/remove placeholders. `_persist_produced_resources()` chama `StrategyExecutor.persist_produced_resource()` após cada execução.

**Type-filtered resource dropdowns.** `InstructionConfigPanel` usa `get_resource_field_specs()` para mostrar só recursos do `StrategyResourceType` correto. Output fields são text inputs.

**Editor de nome de output na coluna 2 (config), espelhado read-only na coluna 3 (resultados).** Decisão de configuração permanece no config panel. Col 3 usa `OutputRowWidget(read_only=True)` com `QLabel` puro (`TextSelectableByMouse`). Sync via signal `output_name_changed`.

**Painel lateral colapsável (`Cmd+B` / `Ctrl+B`).** `StrategyManagerV2View.toggle_left_panel()` esconde/mostra via `widget.hide()/show()`. Tamanhos memorizados antes do collapse, restaurados na expansão.

**Janela de Comparação de Estratégias (`ComparisonWindow`).** `QDialog` não-modal aberto pelo botão "Comparar". `showMaximized()`. Mantida em `self._comparison_window` para preservar referência forte. Re-clique faz `raise_() + activateWindow()`.

**Três tipos de comparação** (`ComparisonKind`):
- **`CALIBRATION_DATASET_DISTRIBUTION`** — distribuição gen × imp que alimentou cada calibrador. Usa sidecar `<stem>_dist.json` quando disponível.
- **`VALIDATION_RESULTS`** — tabela 11 colunas + 3 plots overlay (DET, Tippett, ECE) + grid de cells de distribuição log-LR.
- **`CASEWORK_RESULTS`** — tabela 9 colunas + grid de plots forenses, cada um com borda esquerda na cor da estratégia.

**`FlowLayout` canônico Qt para reflow responsivo.** Em `comparison/flow_layout.py` — porta direta de `flowlayout.cpp` do qtbase. Usa `hasHeightForWidth() + heightForWidth(width)`. Cells `setFixedSize(640, 480)`. **Não reimplementar manualmente.**

**Paleta `STRATEGY_COLOR_PALETTE` compartilhada.** 12 cores estáveis baseadas em matplotlib tab colors. Usada em swatch da tabela, legendas dos plots overlay, e borda esquerda das cells.

**Plots usam `figure.set_layout_engine("constrained")`.** Constrained layout reserva espaço para ylabel/xlabel/legenda antes de dimensionar o axes. Aplicado em `ScoreDistributionWidget._render` e `CaseworkVisualizationWidget._render`. Widgets compartilhados ganham flag `show_title: bool = True` (False para views de comparação).

**`InstructionConfigPanel._get_field_tooltip()` lê `metadata.help` do dataclass field e aplica como tooltip.** Strings começando com `<html>` são renderizadas como rich text pelo Qt. **Genérico**: qualquer DTO com `metadata={"help": "<html>..."}` ganha tooltip automático.

**Info label sob dropdowns de DATASET.** Cada combo que aceita `StrategyResourceType.DATASET` recebe um QLabel abaixo mostrando `N° Amostras • N° Identidades • Dim. Embedding`. Fonte preferida: sidecar `<stem>_stats.json` ao lado do parquet (gerado on-demand via `DatasetStatsService._read_parquet_stats` se faltar). Fallback: payload em memória. Lazy init via `showEvent` do container (evita flash — ver "Qt patterns" abaixo).

**`OutputRowWidget(read_only=True)` embeda `ResourceInfoWidget(compact=True)` lazy via toggle "▾ Detalhes".** Botão `▸ Detalhes` em cada output row da 3ª coluna. `compact=True` oculta o form básico e a barra de abas — só sobram as seções ricas. `project_root` extraído de `resource_loader._storage.project_root` em `_ResultsCard.update_state`. **Crítico**: o controller V2 chama `editor.set_resource_loader(loader)` logo após construir o executor.

**Handler `_on_hydrate_resource_requested` no controller V2.** Botão "📊 Carregar e visualizar" emite `hydrate_requested` que sobe panel → editor → controller. O handler: garante executor + loader, busca o recurso, hidrata, registra no contexto vivo, invalida o card cacheado e força reconstrução.

**Trajectory button só em PIPELINE_APPLICATION (não PIPELINE_TRAINING).** O `PipelineTrainingInstructionExecutor._apply_to_training_datasets` usa `PipelineApplicationPolicy.default()` (`save_trajectory_data=False`).

**Cascade transitivo após edição inline de PIPELINE_CONFIG / MINING_CONFIG (botão ✎).** Após editar e salvar nova versão do config, `_invalidate_dependents_of(resource_id)` faz BFS sobre o grafo de dependência:
- Itera `self._definitions` (todas as estratégias do projeto).
- Para cada instrução com `input_resource_ids` interseccionando `dirty`: chama `_delete_instruction_output_files` (apaga artefato + sidecars + PNGs), seta `existing.artifact_ref = None`/`payload = None` em `defn.resources`, descarta `instruction_id` de `_executed_instruction_ids`, adiciona outputs em `dirty` e em `touched_strategy_ids`.
- Termina quando dirty estabiliza.
- Salva `save_definition(defn)` para cada strategy em `touched_strategy_ids` (erros surfaceados via QMessageBox — não swallow silencioso).
- Refresh visual: `_refresh_instruction_states` + `_refresh_produced_resources` + `_refresh_left_panel`.

**Gate do cascade**: `if not touched_strategy_ids: return` (NÃO `if not any_invalidated`). A versão anterior com `any_invalidated` retornava cedo se nenhum output tinha `existing` em `defn.resources` (caso raro de placeholder fora de sync), perdendo o refresh visual. Touched-based gate garante que se o BFS atingiu alguma instrução, o save+refresh sempre acontece.

**`_refresh_left_panel` re-aponta `self._current_definition`** para a instância correspondente em `self._definitions` após recarregar do disco. Crítico: antes `self._current_definition` ficava como instância órfã (não estava mais em `self._definitions`); cascade mutava as instâncias frescas, mas `_compute_executed_ids_from_disk` lia da instância antiga não-mutada → ✅ falso persistente na lista de instruções.

**`_refresh_instruction_states` usa REPLACEMENT (`=`), não UNION (`|=`).** `self._executed_instruction_ids = self._compute_executed_ids_from_disk()`. Union podia re-adicionar IDs que o cascade acabara de descartar. Estado de disco é sempre autoritativo após save_definition.

**Defesa adicional em `_persist_edited_config_resource`**: após `_invalidate_dependents_of`, força `self._executed_instruction_ids = self._compute_executed_ids_from_disk()` + atualiza UI diretamente via `update_instruction_states`. Garante que a UI sempre se sincroniza com o estado autoritativo de disco, mesmo se algum caminho intermediário corrompeu o set in-memory.

**Erros de `save_definition` agora são surfaceados** via QMessageBox (antes eram silenciosamente engolidos via `except Exception: pass`). Aplicado em `_save_current_definition`, `_persist_edited_config_resource`, e no save loop de `_invalidate_dependents_of`. Bug histórico: silent failure mascarava casos onde mutações in-memory (novo hash após edição, cascade de invalidação) nunca chegavam ao disco — usuário só percebia no próximo open com integrity warning.

**Reused from outros tabs do app** (imported directly, no copy):
- `LogProgressDialog` — used for strategy execution
- `PipelineCreatorController` — "Pipeline" button in global resources toolbar
- `NewMiningConfigDialogController` — "Mining" button
- `NewSplitConfigDialogController` — "Split" button

## Padrões Qt/PyQt

### Design de UI — SEMPRE usar as skills de design

Qualquer trabalho de interface — criar/redesenhar telas, diálogos, painéis, widgets, ou mexer em QSS/`QPalette`/stylesheets/tema — **deve seguir as skills de design do projeto** (em `.claude/skills/`, versionadas no git):

- **`faceaudit-qt-design`** — específica do FaceAudit: arquitetura, tema e idiomas Qt/QWidgets.
- **`qt-ui-design`** (oficial da Qt Company) — leis de UX nomeadas, WCAG, escala tipográfica, modo auditoria.

**Regras duras (não-negociáveis)** encodadas na skill:
- **Regra de negócio mora no backend** (services/executors/models); a UI apenas orquestra.
- **UI ↔ backend troca apenas DTOs** — sem lógica de domínio dentro de widgets/controllers. Meta: trocar o frontend = chamar os mesmos métodos do backend.
- **Trabalho pesado sempre em worker** (`QObject` + `moveToThread`), nunca na thread da GUI.
- **Tema**: migrar para **tokens semânticos centralizados** com modo **claro/escuro configurável** — **não** cravar hex inline nem assumir light-only .

Essas skills valem em qualquer superfície (CLI, IDE, web/nuvem).

### Sistema de tema claro/escuro (`interface/theme/`)

A UI tem tema **claro e escuro configurável**, com fonte única de verdade em tokens semânticos. **Nunca cravar hex inline em `setStyleSheet` — sempre rotear por tokens.**

**Estrutura (`interface/theme/`):**
- `tokens.py` — dicts `LIGHT` e `DARK` (mesmo conjunto de chaves) mapeando token → hex. Tokens: `SURFACE`, `SURFACE_ALT`, `SURFACE_RAISED`, `BORDER`, `TEXT`, `TEXT_MUTED`, `ACCENT`/`ACCENT_BG`/`ACCENT_TEXT` (âmbar forense), `PRIMARY`/`PRIMARY_BG`/`PRIMARY_STRONG`, `HEADER_BG`/`HEADER_TEXT` (cabeçalho navy = fundo escuro + texto claro), `SUCCESS`/`DANGER`/`WARNING` (+ `_BG`), `HIGHLIGHT`/`HIGHLIGHT_TEXT`, e os papéis de `QPalette` (`WINDOW`, `BASE`, etc.). **O modo claro é byte-idêntico à paleta histórica** (`_apply_light_palette`) — ativar o tema não regride o visual claro. Módulo puro (sem PyQt), com `assert set(LIGHT)==set(DARK)`.
- `theme_manager.py` — `ThemeManager(QObject)` **singleton**. `ThemeManager.initialize(app, mode)` no bootstrap; `ThemeManager.instance()` em qualquer lugar. `apply(mode)` seta Fusion + `build_palette(mode)` (QPalette) + `build_global_qss(mode)` (QSS global — vazio no claro p/ não regredir; baseline no escuro) e emite `theme_changed`. `set_mode(mode)` troca em runtime. `color(token)`/`qcolor(token)` leem o modo atual.
- `themed_widget.py` — **`apply_themed_stylesheet(widget, builder)`**: o padrão de migração. `builder(t)` recebe `t(token)->hex` e retorna o QSS; aplica imediatamente E **re-aplica ao trocar de tema** (um `_ThemedStyler` filho do widget escuta `theme_changed`; morre junto com o widget). Para QSS montado dinamicamente (loop/por-item/por-estado, ex.: `setForeground` de célula), use `current_token_fn()` no momento da criação (nasce com a cor certa; não re-aplica ao vivo).

**Padrão de migração (obrigatório para QUALQUER QSS novo com cor):**
```python
from interface.theme import apply_themed_stylesheet, current_token_fn
# estático (self ou filho fixo) — re-aplica ao vivo:
apply_themed_stylesheet(self, lambda t: "QFrame { background: " + t("SURFACE_RAISED")
    + "; color: " + t("TEXT") + "; border: 1px solid " + t("BORDER") + "; }")
# dinâmico/por-estado — lê o token na criação:
t = current_token_fn(); item.setForeground(QtGui.QColor(t("DANGER")))
```

**Persistência e toggle:** `SystemConfig.theme.mode` (`"light"`/`"dark"`) em `config.json` (por-máquina, tolerante a chave ausente). Lido em `interface/run.py` no bootstrap (após `bootstrap_configuration`, antes de criar a view) e aplicado via `ThemeManager.initialize`. O `ConfigDialog` tem o grupo **🎨 Aparência** com seletor Claro/Escuro; `MainGuiController.abrir_dialogo_configuracoes` chama `ThemeManager.instance().set_mode(...)` para aplicar **ao vivo** ao salvar.

**Exceções intencionais (NÃO migrar para tema):**
- **Plots matplotlib ficam BRANCOS sempre** (decisão forense — gráfico de laudo é branco/print-consistente). Nunca tocar em `color=`/`facecolor=`/`edgecolor=`/`cmap`/`ax.*`/`plt.*`/`Figure`/`sns.*` nem em `STRATEGY_COLOR_PALETTE`. Os *cards* que hospedam PNGs de plot também ficam brancos (ex.: `hubness_report_viewer_dialog`, `plot_card`) — plot branco em card branco é coerente.
- **Pintura de canvas** via `QColor`/`QPainter` (máscara de oclusão, scrubber de vídeo, placeholders de ícone): é desenho, não chrome — deixar como está.
- **Console de log** (`log_progress_dialog`, `#1e1e1e`): estética de terminal, escuro nos dois modos.

**Status:** infra + todo o chrome sempre-visível + a maior parte dos diálogos migrada. Ainda há hex cravado em parte dos arquivos de `interface/` — uma parcela é exceção legítima (plots, canvas, console); o resto é dívida a migrar. Os widgets de plot permanecem claros por design (acima).

**Ciladas comuns (aprendidas na migração — lista completa na skill `faceaudit-qt-design`):**
- **Fundo colorido = SÓ título/callout, nunca conteúdo/campo.** Conteúdo usa surface neutro; a cor semântica vai no título da seção e no label.
- **Nunca `color: inherit` em `::item:selected`** — no escuro resolve para PRETO sobre o fundo de seleção; use `t("TEXT")`.
- **Cor quase-branca/quase-preta → surface neutro** (`SURFACE_RAISED`), NUNCA um token de status saturado (`SUCCESS_BG` etc.).
- **Barra de accent precisa ser visível no escuro** — não usar `BORDER`/`HEADER_BG` como acento (somem); usar `TEXT_MUTED`/`PRIMARY`/cor semântica.
- **Word-wrap + `QFormLayout` + `QScrollArea`** → reservar a barra de rolagem (`ScrollBarAlwaysOn`), senão a última linha corta quando a barra aparece.
- **Rótulo com word-wrap nunca é `QtWidgets.QLabel` cru** — usar `WrappedLabel` (`strategy_editor/widget_helpers.py`), e `enable_height_for_width()` no container que o embrulha. `QLabel.sizeHint()` com wrap é heurística (estima a altura numa largura arbitrária) e `QWidgetItem.hasHeightForWidth()` lê a *size policy*, não o layout interno — então o layout do pai reserva a altura estimada. Numa coluna mais estreita que a estimada o texto quebra em mais linhas do que cabe e o `QFormLayout` comprime TODAS as linhas da seção: o sintoma é texto sobreposto/cortado em campos vizinhos que nada têm a ver com o rótulo. `WrappedLabel` fixa `minimumHeight = heightForWidth(largura)` (piso duro, recalculado a cada resize e `setText`); `enable_height_for_width` religa a cadeia para que o pai pergunte a altura PARA a largura real. Regressão coberta por `tests/test_config_panel_wrapped_label_height.py` (exercita a coluna estreita — em largura folgada o bug não aparece).

### Sempre passar `parent=` ao criar QWidgets — anti-flash de janela top-level

**Bug**: ao criar um `QWidget`/`QLabel`/`QFrame`/`FigureCanvas`/etc. **sem parent** (`parent=None` é o default), Qt o trata como **top-level window**. Mesmo que você adicione o widget a um layout logo depois via `addWidget` (que faz reparent), há um intervalo (no event loop seguinte) em que Qt dispara um `Show` event no widget enquanto ele ainda é top-level. Em algumas plataformas (Windows especialmente) isso causa um **flash visível de janelinha** antes do reparent. O sintoma típico: janela pequena (~20×19 px ou tamanho default 640×480) aparece e some rapidamente quando o usuário seleciona algo na UI (lazy widget creation).

**Solução**: sempre passar `parent=` no construtor — mesmo que o widget seja imediatamente adicionado a um layout. O parent provisório pode ser qualquer ancestral legítimo (geralmente `self` do widget que está construindo a UI), Qt reparente normalmente via `addWidget` depois sem efeito colateral.

**Padrão correto**:
```python
class MyWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        # CERTO: parent=self desde o construtor
        self._lbl = QtWidgets.QLabel("texto", parent=self)
        layout.addWidget(self._lbl)
```

**Padrão errado** (causa flash):
```python
self._lbl = QtWidgets.QLabel("texto")  # ← parentless, vira top-level
layout.addWidget(self._lbl)  # reparenteia, mas tarde demais
```

**Para `FigureCanvas` do matplotlib** (que não aceita `parent=` no construtor): chamar `self.canvas.setParent(self)` logo após criar.

**Como diagnosticar**: instalar event filter de debug em `interface/run.py` (já removido do repo, mas pode ser reinstalado se precisar). Pattern:
```python
class _TopLevelShowDebugFilter(QtCore.QObject):
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Show:
            if isinstance(obj, QtWidgets.QWidget) and obj.parent() is None:
                # Log classe, texto, geometria, stack trace de criação
                ...
        return False
app.installEventFilter(_TopLevelShowDebugFilter())
```

Combinar com monkey-patch em `QWidget.__init__`/`QLabel.__init__` que anexa stack trace de criação ao widget — quando o filtro detectar o show, mostra exatamente onde no Python o widget foi instanciado.

**Histórico**: o bug afetou múltiplos widgets (`_LazyInitContainer` do dataset info label, `_InstructionItemWidget` da lista de instruções, `OutputRowWidget._status_lbl` ⬜, `ResourceInfoWidget` compact mode, vários `QFrame`/`QWidget` em `_ResultsCard` e plot widgets). Solução foi aplicar `parent=self` em massa via replace_all nos arquivos relevantes.
