# CLAUDE.md

Orientação para assistentes de IA (Claude Code e similares) que trabalham neste repositório. Este arquivo é carregado inteiro em toda sessão: aqui ficam só as regras permanentes, os invariantes e as armadilhas, cada uma com o PORQUÊ. O detalhe de cada área está em `docs/` (ver "Índice de docs" no fim).

## Visão geral e licença

O FaceAudit é uma plataforma de análise e comparação forense de imagens faciais. O objetivo é permitir que o perito **crie e configure estratégias de cálculo de razão de verossimilhança (LR)** sobre embeddings faciais inteiramente pela interface — nada fixo no código. Software livre (GPL v3, ver `LICENSE`), mantido como iniciativa pessoal.

**Versão.** Fonte única em `faceaudit/_version.py` (hoje 0.2.0; o `pyproject.toml` lê de lá). **Suba a versão e crie a tag `vX.Y.Z` sempre que uma mudança alterar o que o programa PRODUZ** (imagens, embeddings, escores, LRs). Porquê: versão e commit vão para a cadeia de custódia de todo artefato (`software_provenance_service` → `AuditLogService.append` põe `metadata["software"] = {faceaudit_version, git_commit, git_dirty}` em TODA entrada, coberto pelo `entry_hash`); uma versão que não sobe faz artefatos diferentes declararem a mesma origem.

## Comandos

```bash
pip install -e .                 # instalar em modo de desenvolvimento
python interface/run.py          # rodar a aplicação

python -m pytest tests/          # todos os testes
python -m pytest tests/test_full_pipeline.py
python -m pytest tests/test_storage_and_repositories.py
python -m pytest tests/test_identity_disjoint_pair_kfold.py      # splitter STRICT_IDENTITY
python -m pytest tests/test_face_geometry_persistence.py         # bbox/keypoints no parquet + invariante
python -m pytest tests/test_relative_path_root_on_save_as.py     # nenhum dataset RELATIVE fora da pasta das imagens
python -m pytest tests/test_software_provenance.py               # versão + commit em toda entrada da custódia

python tests/check_parquet.py    # utilitário de conferência de parquet
```

A lista temática completa (detecção, geometria, metadados, OFIQ, pipeline de imagens, vídeo, interface…) está em `docs/dev/testes.md`. Ao mexer numa área, rode a suíte dela.

## Configuração

- `config.json` — caminhos de modelos, biblioteca, datasets e saídas; tema; opções do FF. **Não é versionado** (está no `.gitignore`): é criado com padrões pelo bootstrap na primeira execução (`bootstrap_configuration` em `interface/run.py`). O modelo versionado é `config.example.json`.

## Fluxo de trabalho git

- Trabalhe num branch próprio → commits → `git merge --no-ff` no `master` → apague o branch.
- `git push` só quando o usuário pedir.
- `config.json` nunca entra em commit (é por máquina; o modelo é `config.example.json`).
- Mensagens de commit terminam com `Co-Authored-By: <modelo que fez o trabalho> <noreply@anthropic.com>`, nomeando o modelo REAL da sessão (ex.: `Claude Fable 5.1`, `Claude Opus 5.5`). Piso: Opus 5.5 — modelos abaixo disso não devem assinar commits neste repositório. Porquê: a atribuição vai para o histórico e precisa ser verdadeira.

---

## Arquitetura do motor

Motor de estratégias **declarativo**: a estratégia é um `LRStrategyDefinition` (recursos + instruções ordenadas em três fases); o motor a interpreta. O usuário define o que acontece, não o código.

```
StrategyExecutorFactory.build()
  └─ StrategyExecutor.execute(LRStrategyDefinition, additional_resources=...)
       ├─ cria StrategyExecutionContext  ← barramento de estado compartilhado
       ├─ registra recursos iniciais (sem hidratar)
       ├─ valida a estratégia (regras de nível de estratégia)
       └─ para cada fase (Preparação → Validação → Análise de Caso), para cada instrução:
            InstructionExecutorRegistry.resolve(instruction_type)
              ├─ executor.validate(instruction, context)   ← ver "Validação"
              └─ executor.execute(instruction, context)    ← lê recursos, escreve StrategyResource novos
       └─ empacota o contexto → LRStrategyTrainedModel

Execução passo a passo (UI):
  prepare_context(definition) → can_execute_instruction(instr, ctx) → execute_single_instruction(instr, ctx)
```

- **`StrategyResource` é a unidade de troca entre instruções.** Cada DTO de instrução declara `input_resource_ids`/`output_resource_ids` (sincronizados por `_sync_declared_io()`). Recursos de PROJETO (DATASET, SPLIT_CONFIG, MINING_CONFIG, PIPELINE_CONFIG) têm sempre `artifact_ref` e não estão necessariamente em `definition.resources` — por isso o controller passa `_collect_project_resources()` como `additional_resources`. Recursos PRODUZIDOS nascem como placeholders (payload e `artifact_ref` vazios) e ganham `artifact_ref` ao serem persistidos.
- **`StrategyInstructionDefinition.from_dict()` é um despachante polimórfico** por `instruction_type` — é o que faz a estratégia sobreviver ao JSON com o tipo certo.
- **Similaridade e calibração são separadas**: `SimilarityModelProtocol` produz escore bruto; `CalibrationModelProtocol` converte em LR; o elo é o `SimilarityScoreDataset`.
- **Cada instrução roda com um `InstructionExecutionFrame` (fase)** injetado no contexto; executores que dependem da fase falham fechados (`MissingExecutionFrameError`) em vez de adivinhar pela posição.
- **Hidratação é preguiçosa.** `prepare_context` só registra; o payload é carregado sob demanda (`context.ensure_payload`, `StrategyResourceLoaderService`). Porquê: datasets e escores têm milhões de linhas; hidratar tudo ao abrir travaria a interface.
- **`set_resource` invalida o cache `"{rid}_stats"`.** Porquê: reexecutar uma instrução reescreve o dataset sob o mesmo id, e regras como Fisher leriam estatísticas velhas.
- **Projeto autocontido.** Um projeto é uma pasta ancorada num `.fa`; toda `StrategyArtifactReference.path` é **relativa** à raiz (e `\` vira `/` na leitura). A biblioteca global (`GlobalLibraryStorage`) guarda peças reutilizáveis, copiadas ao projeto por `copy_artifact_to_project()`. A abertura verifica integridade (hash SHA-256) automaticamente.

Detalhe: `docs/arquitetura/motor_estrategias.md` (inclui a tabela completa de instruções) e `docs/arquitetura/mapa_de_pastas.md`.

### Mapa de pastas (resumo)

| Pasta | Papel |
|---|---|
| `faceaudit/contexts/` | `StrategyExecutionContext` — barramento de estado |
| `faceaudit/executors/` (+ `instructions/`) | `StrategyExecutor`, `InstructionExecutorRegistry`, um executor por instrução (base `StrategyInstructionExecutorBase`) |
| `faceaudit/factories/` | `StrategyExecutorFactory` (ponto único de montagem) + `ValidationRuleFactory` |
| `faceaudit/models/` | DTOs: `instructions/`, `strategy/`, `similarity/`, `calibration/`, `lr/`, `validation/`, `storage/`, `pipeline/`, além dos modelos de dataset/face |
| `faceaudit/enums/` | Enums de domínio (tipos de instrução, regra, recurso, arquiteturas LR, modos) |
| `faceaudit/protocols/` | Contratos estruturais (`typing.Protocol`) |
| `faceaudit/services/` | Regra de negócio: datasets, detecção/geometria, metadados, OFIQ, PLDA, tipicidade, calibração, custódia |
| `faceaudit/validation_rules/` | Regras de validação, todas estendendo `ValidationRuleBase` |
| `faceaudit/transformators/`, `faceaudit/pipelines/` | Transformações de embedding e `EmbeddingTransformationPipeline` (LINEAR/DUAL) |
| `faceaudit/storage/`, `faceaudit/repositories/` | `LocalProjectArtifactStorage`, `GlobalLibraryStorage`; repositórios de definição (JSON) e modelo treinado (pickle) |
| `interface/` | PyQt6: `run.py` (bootstrap), `theme/`, `tabs/strategy_manager_v2/`, `tabs/dataset_manager/` |
| `docs/` | Documentação técnica (ver índice) |

## Validação

Dois níveis. **Estratégia**: regras globais antes de qualquer fase (`LRStrategyDefinition.validation_rule_ids` + as sempre ligadas da factory, como `StrategyEmbeddingConsistencyRule`). **Instrução**, em `StrategyInstructionExecutorBase.validate()`, nesta ordem: (1) estrutural (nulos); (2) plugáveis — `instruction.validation_rule_configs`, instanciadas por `ValidationRuleFactory`; (3) dinâmicas — `instruction.build_validation_rules()`; (4) específicas do executor — `_validate_instruction_specifics()`. O gate de dependência (`can_execute_instruction`) é separado: confere que os insumos existem no contexto.

- **Persistente × dinâmica.** Regra com limiar que o usuário pode querer ajustar/remover vai em `_ensure_default_validation_rules` (persistida no JSON, removível na UI). Regra que é pré-condição matemática (modelo ajustado, dimensão compatível, estado RAW/CALIBRATED, caso × não caso) vai em `build_validation_rules` (recriada a cada validação, não removível). Porquê: desligar uma pré-condição só troca um erro claro por um erro tardio.
- **`validation_rules_user_managed`.** Ao remover uma regra, a flag vira `True` e os defaults não voltam ao reabrir. Porquê: sem ela, lista vazia seria lida como "ainda não populei" e a escolha do usuário seria desfeita em silêncio.
- **Regra nova estende `ValidationRuleBase`** (que dá `_get_or_compute_stats`), nunca implementa o protocolo diretamente. Os tipos estão em `ValidationRuleType`; não crave contagens em documentação.
- `ValidationResult` agrega tudo (`merge()`; fábricas `ok`/`neutral`/`warning`/`critical`).

Detalhe (defaults por DTO, regras dinâmicas, templates, seção da UI): `docs/arquitetura/validacao.md`.

## Instruções

Fonte de verdade: `StrategyInstructionType`. Tabela completa com campos e regras em `docs/arquitetura/motor_estrategias.md`.

| Tipo | O que faz |
|---|---|
| `PIPELINE_TRAINING` / `PIPELINE_APPLICATION` | Treina / aplica `EmbeddingTransformationPipeline` (LINEAR ou DUAL) |
| `DATASET_SPLIT` / `PAIRED_DATASET_SPLIT` | Divide um dataset (por identidade ou índice) / dois datasets pareados com o mesmo sorteio de identidades |
| `DATASET_MERGE` / `DATASET_BALANCING` | Junta datasets / equilibra amostras por identidade (aleatório ou por qualidade) |
| `EMBEDDING_AGGREGATION` | Funde embeddings da mesma identidade em templates (centroide ou combinatório); a saída não é dataset de fotos |
| `SIMILARITY_MODEL_TRAINING` / `SIMILARITY_SCORING` | Cosseno ou PLDA (SIMPLE, TWO_COVARIANCE, SHRINKAGE_TWO_COVARIANCE, com modo cross-condition) / escores de todos os pares |
| `PAIR_MINING` | Cura pares genuínos/impostores por regra (HARD/EASY/TYPICAL/RANDOM/MIXED, ciente da classe) |
| `TYPICALITY_MODEL_TRAINING` / `TYPICALITY_PAIR_SCORING` | Modelo de tipicidade (k-NN; MoVMF experimental) / tipicidade por par |
| `SCORE_INTERACTION_FEATURE` / `SCORE_VECTORIZATION` | Feature escore × raridade centrada em s0 / vetor de escores para calibrador multivariado |
| `LR_ARCHITECTURE_VECTORIZATION` | Substituto declarativo do encadeamento acima: uma arquitetura de `LRCalibrationArchitecture` numa instrução |
| `CALIBRATION_MODEL_TRAINING` / `SCORE_CALIBRATION` | Treina LogReg (com piecewise-linear opcional), KDE ou BiGauss / aplica e grava `log_lr` |
| `STRATEGY_EVALUATION` | Só na Validação: EER, Cllr, Cllr_min (PAVA in-sample), AUC, Cllr afim, ECE |
| `CASEWORK_VISUALIZATION` | Só na Análise de Caso: escore(s) do caso contra a distribuição de referência |
| `HUBNESS_ANALYSIS` / `HUBNESS_ABLATION` | Diagnóstico de hubness / ablação por quantidade ou pose (custoso) |

**Regras forenses que não podem regredir:**
- **s0 é aprendido na calibração e congelado.** Em Validação e Análise de Caso, `SCORE_INTERACTION_FEATURE` usa `S0Mode.INHERIT` (AUTO_LOGREG é bloqueado em caso) e `LR_ARCHITECTURE_VECTORIZATION` exige `parameter_mode=INHERIT`. Porquê: reajustar s0 fora da calibração redefine o que é "evidência neutra" e contamina o LR.
- **Compatibilidade dataset × calibrador por `vectorization_contract_fingerprint`.** `SCORE_CALIBRATION` valida em STRICT; `LEGACY_PERMISSIVE` existe só no `CalibrationModelMetadataService`. `LRArchitectureState` recalcula os fingerprints em `from_dict` e acusa adulteração.
- **CV do calibrador por identidade.** `STRICT_IDENTITY` usa `IdentityDisjointPairKFold`; com `FIT_PER_FOLD`, s0/médias de raridade são ajustados por fold dentro do `Pipeline`. Porquê: estimar sobre todo o conjunto antes da CV vaza informação e torna as métricas otimistas.
- **Monotonicidade condicional** (`ConditionalMonotonicityRule`) é auditada pelo executor de `CALIBRATION_MODEL_TRAINING` após o refit, só quando o calibrador tem `architecture_state`.
- **Linguagem PAVA (Morrison 2021).** `Cllr_min`/`cal_loss` são referência diagnóstica in-sample, não medida operacional de calibração; a UI não emite veredito binário "bem calibrado".
- **Caso nunca afirma a conclusão.** Em dataset de caso (`is_casework`) a tabela de comparações mostra a classe sempre como caso (`CLASSE_CASO`) e `TYPICALITY_PAIR_SCORING` zera `same_person`. Porquê: o modelo infere "mesma pessoa" comparando strings de identidade, que em caso são convencionais — exibir isso afirmaria a conclusão do exame.

## Invariantes de dados e imagem

- **A geometria da face está sempre no sistema da imagem que o dataset guarda.** `FFInfo.bbox`/`keypoints` referem-se a `FaceInfo.image_path`; quem troca a imagem converte a geometria na criação (recorte: translada; alinhada: descarta). Pontos 112×112 (KPRPE) moram só em `FaceInfo.extra_data[ALIGNED_KEYPOINTS_EXTRA_KEY]`. Porquê: com dois sistemas sob o mesmo nome, o KPRPE recebia a régua errada e produzia embedding errado sem erro aparente.
- **A caixa persistida está em coordenadas do arquivo original, mesmo quando a face só foi detectada girada.** A rotação vai em `ff_info.detection_rotation`; `caixa_girada_para_original` converte. Porquê: guardar a caixa girada faz recortes e desenhos caírem fora da face sem nada no dado denunciar.
- **A detecção tenta as quatro orientações** (`ROTACOES_PADRAO`), e no enriquecimento a orientação já registrada é tentada primeiro (`rotacao_registrada`). Porquê: sem isso, faces que só existem giradas falham em silêncio no enriquecimento.
- **A orientação EXIF faz parte da imagem.** O OpenCV a aplica; `QPixmap(caminho)` não. Toda imagem exibida passa por `interface/oriented_image.py`; no backend, quem lê só o tamanho com Pillow troca largura/altura na orientação 5–8.
- **`path_root` é relativo à pasta do parquet: nenhum fluxo grava dataset RELATIVE numa pasta que não contém as imagens.** Quem salva declara a intenção (`prepare_dataset_for_destination(..., portable=)`); `to_parquet` tem a rede de proteção. Porquê: o dataset abre, lista as faces e só falha ao ler a imagem, muito depois.
- **Dataset autocontido é uma pasta própria** (`<pasta>/<nome>/` com o parquet e `images/`), montada só por `self_contained_dataset_dir()` + `SELF_CONTAINED_IMAGES_SUBDIR`. Porquê: assim a pasta pode ser movida ou levada a outra máquina sem quebrar.
- **Juntar datasets preserva a resolução das imagens**: âncoras iguais mantêm RELATIVE; diferentes viram ABSOLUTE; `is_aligned` só sobrevive se os dois forem alinhados. Porquê: misturar foto inteira num dataset marcado como alinhado faz o encoder receber lixo como face.
- **Dividir e 📚 Biblioteca referenciam as imagens onde estão.** As metades herdam a ancoragem da origem e são reapontadas ou absolutizadas para a pasta de destino (`dataset_split_save_service.save_split_halves`, que recusa metade vazia); a biblioteca absolutiza como o "→ Estratégia" (`GlobalLibraryStorage.import_dataset`). O aviso de sobrescrita da criação pergunta o caminho final ao backend (`FeatureExtractionPipeline.destination_parquet`) e Modificar Caminhos recusa resultado sem faces e pede confirmação em falha parcial (`LocalizationResult.refusal_message`). Porquê: as quatro coisas falhavam em silêncio.
- **Nunca gravar dataset vazio nem perder face em silêncio.** `finalize_storage_and_save` recusa zero faces; a localização que pula imagem ausente reporta a falha e quem chama exige decisão. Porquê: parquet vazio é aceito pela UI e some dos seletores sem explicação.
- **Imagem sem face detectada só entra no dataset por decisão humana.** Com `DatasetIngestionConfig.review_undetected=True` as pendentes são separadas antes dos estágios e a gravação é adiada até a revisão; cancelar a revisão nunca descarta. Em dataset de recortes/alinhadas a inclusão fica bloqueada enquanto a face não for detectada (`resolve_inclusion_policy`). Face recuperada passa por `complete_stages_for_recovered` antes de gravar. A escalada de `det_size` é manual.
- **Falha no estágio FF do enriquecimento é "faltou o embedding pedido"**, não `ff_info is None` — a face pode chegar com `FFInfo` de passada anterior.
- **Dataset de vídeo em modo recorte usa a face alinhada da EXTRAÇÃO, nunca redetecta**; o realinhamento posterior pelos pontos é decidido pela proveniência da geometria, não pela presença de pontos. Porquê: redetectar no recorte perde faces pequenas e pode embutir outra região.
- **A rotação declarada no contêiner de vídeo é aplicada a todo quadro**, por um ponto único (`VideoFrameFaceExtractionService._aplicar_rotacao`). Porquê: se scrubber, âncora e extração discordassem, o usuário escolheria o recorte numa imagem e o sistema extrairia de outra.
- **Metadados declarados ≠ medidos.** `ColumnSpec.is_declared=True` marca o que o arquivo afirma (EXIF/XMP, editável ou forjável); data normalizada só com fuso EXPLÍCITO; a qualidade JPEG vem da tabela de quantização, não dos bits por pixel (que dependem do conteúdo).
- **OFIQ: nomes de medida por versão numa tabela só** (`faceaudit/models/ofiq_measure_names.py`, `field_for_column`); toda métrica guarda `ofiq_version`; a cópia temporária leva extensão que o executável aceita (`extensao_para_ofiq`). Porquê: nome desconhecido ou extensão rejeitada perdiam valores sem erro.
- **Coluna inteira com vazio continua inteira no parquet** (decide pelos valores originais, não pelo DataFrame, onde 43 e 43.0 já não se distinguem).
- **Caminho de executável externo: o backend resolve, a tela pergunta** (`OFIQService.resolve_exe_path`, `ExifExtractionService.exiftool_version`). Campo vazio = "procurar automaticamente".
- **Pipeline de imagens:** a geometria acompanha a imagem pelos passos e os passos não detectam por conta própria (sem geometria, `FaceGeometryRequiredError`, nunca oclusão no lugar errado); o sorteio usa `stable_rng`, não `hash()`; a concordância com a questionada é medida imagem a imagem, não só pela mediana; o ajuste automático otimiza o excesso além da tolerância e é validado fora da amostra; nada é aplicado sem confirmação do perito.

Detalhe: `docs/arquitetura/deteccao_e_geometria.md`, `datasets_e_caminhos.md`, `metadados_captura.md`, `ofiq.md`, `pipeline_imagens.md`.

## Custódia e proveniência

- Toda entrada da cadeia leva versão + commit (ver "Versão"). O mesmo vai no manifesto do Processar, no `execution_metadata` da estratégia e no manifesto do vídeo.
- **Salvar Como COPIA a cadeia** (`AuditLogService.copy_sidecar`), e a origem recebe entrada `save_as`. Porquê: mover deixava o dataset de origem sem cadeia.
- **Excluir faces passa pela custódia** (`AuditOperation.REMOVE_FACES`) com justificativa obrigatória, contagem e hash do conjunto removido. Com filtro ativo, a remoção atinge o dataset de origem (os objetos `FaceInfo` são compartilhados), senão "Limpar filtro" desfaria a exclusão.
- **Toda alteração estrutural entra na cadeia** (`dataset_custody_service`). Juntar grava `MERGE` e Dividir grava `SPLIT` em cada metade, com a origem pelo SHA-256 do parquet e o `audit_chain_head` da cadeia dela, critério/semente e contagens. O Auditar pede justificativa e, ao fechar, registra só o que ficou aplicado (`REMOVE_FACES` / `EDIT_LABEL`, com `applied_at` e a contagem de desfeitas). Entrada de dataset ainda sem arquivo fica pendente no objeto e é escrita na gravação, antes do SAVE/SAVE_AS. `AuditLogService.append` recusa `REMOVE_FACES` sem justificativa. Porquê: um dataset derivado sem registro da origem não se sustenta num laudo.
- Criação e enriquecimento registram quantas faces só apareceram giradas, incluídas sem biometria, descartadas e recuperadas em resolução maior — "recuperada" (tem embedding) nunca se mistura com "incluída sem biometria".
- Chave nova de proveniência do dataset precisa entrar em `_PROVENANCE_METADATA_KEYS` (allowlist deliberada).
- Derivado do pipeline de imagens nasce com entrada CREATE própria (origem + SHA-256, pipeline, manifesto) e só leva as colunas que descrevem a pessoa/registro (`colunas_que_acompanham`), mais a pose.

Detalhe: `docs/arquitetura/custodia.md`.

## Desempenho (não regredir)

- **`StrategyResource.to_dict()` NÃO chama `super().to_dict()`.** Porquê: o mixin usaria `dataclasses.asdict`, que recursa no `payload` — milhões de objetos por chamada num `SimilarityScoreDataset` grande.
- **Nenhum `@dataclass` concreto herda de `Protocol`** (`LRStrategyDefinition`, `LRStrategyTrainedModel`…). Porquê: as `@property` do protocolo quebram o `__init__` gerado; a tipagem estrutural já basta.
- Contagens de `SimilarityScoreDataset` são cacheadas (invalidadas em `add()`); summarizers da UI amostram e leem `items[0]` em vez de montar a matriz.
- Executores que varrem escores iteram em chunks com `time.sleep(0)`; `to_parquet` grava em chunks com o GC cíclico desligado; `execution_logs` têm teto (`_MAX_EXECUTION_LOGS`). Porquê: liberar a GIL e não congelar a interface.
- Sidecars de estatística são invalidados por mtime (o parquet é reescrito no mesmo caminho).

## Frontend: regras

Use **sempre** as skills de design ao criar/alterar telas, diálogos, widgets, QSS ou tema: `.claude/skills/faceaudit-qt-design` (arquitetura, tema e idiomas Qt do projeto) e `.claude/skills/qt-ui-design` (UX, WCAG, tipografia).

- **Regra de negócio mora no backend** (services/executors/models); a UI orquestra e troca só DTOs com ele. Porquê: trocar o frontend tem de ser chamar os mesmos métodos. (Ainda há resíduo na UI, ex.: contagens de proveniência da detecção no `dataset_manager_controller` — não aumente.)
- **Trabalho pesado sempre em worker** (`QObject` + `moveToThread`), nunca na thread da GUI.
- **Tema por tokens, nunca hex cravado.** Cor nova passa por `apply_themed_stylesheet(widget, builder)` ou `current_token_fn()` (`interface/theme/`, tokens em `tokens.py`, `ThemeManager` singleton). Exceções intencionais: plots matplotlib (sempre brancos — figura de laudo), pintura de canvas, console de log. Porquê: hex cravado quebra o modo escuro.
- **Sempre `parent=` no construtor de QWidget** (para `FigureCanvas`, `setParent(self)` logo após criar). Porquê: widget sem pai é janela top-level e pisca na tela antes do reparent, sobretudo no Windows.
- **Rótulo com quebra de linha é `WrappedLabel`**, com `enable_height_for_width()` no contêiner (`interface/tabs/strategy_manager_v2/strategy_editor/widget_helpers.py`). Porquê: `QLabel` com wrap reserva altura estimada e, em coluna estreita, comprime e sobrepõe as linhas vizinhas.
- **No Windows, diálogo `ApplicationModal` engole sinais enfileirados de worker**: abrir projeto é síncrono (WaitCursor), e diálogos modais só abrem depois que o worker terminou (`WorkerProgressRunner` desmonta o progresso antes de `on_finished`).
- **Cache dos cards de resultado é por `(strategy_id, instruction_id)`**, e o editor chama `results_panel.set_strategy_id` antes de qualquer instrução. Porquê: duplicar estratégia copia os ids das instruções, e a cópia mostrava os resultados da original.
- **Estado de execução vem do disco.** Após editar config inline (✎, sempre com `lock_name=True`), `_invalidate_dependents_of` invalida em cascata, salva as estratégias tocadas e `_executed_instruction_ids` é SUBSTITUÍDO pelo cálculo do disco (nunca união). Erros de `save_definition` aparecem ao usuário, nunca são engolidos.
- **A janela de comparação de faces desenha os pontos DO DATASET, nunca redetecta** — são os que definiram o embedding comparado.
- **Figura de laudo**: histograma sem KDE por padrão, legenda em português do leitor ("Mesma pessoa"/"Pessoas diferentes"), número em convenção brasileira em eixos e legenda (`faceaudit/models/plot_number_format.py`).

Detalhe (árvore do Strategy Manager V2, painéis, comparação, tema): `docs/arquitetura/frontend_v2.md`.

## FFService e GPU

- `FFService` carrega um `ForensicFace` por modelo de reconhecimento declarado. `process_dataset` escolhe entre quatro caminhos (alinhadas em disco, cache em memória, alinhar+lote, extração crua por face).
- **Windows/CUDA:** `ensure_cuda_dlls_loaded()` pré-carrega as DLLs do CUDA/cuDNN instaladas por pip (uma fonte só). Porquê: sem isso o ONNX Runtime cai para CPU em silêncio.
- **macOS/CoreML:** um patch configura as sessões (formas estáticas, MLProgram, batch fixo nos encoders). Com GPU o encoder só aceita lote de tamanho N (o último lote é preenchido); a extração crua por face em GPU não funciona — rode com `use_gpu=False`.
- Divergência conhecida: `ff_det_size` padrão é 320 na criação/enriquecimento e 640 no vídeo.

Detalhe: `docs/arquitetura/ffservice_gpu.md`.

## Como estender

Use os comandos do projeto (`.claude/commands/`) em vez de montar à mão:

| Comando | Cria |
|---|---|
| `/new-instruction <Nome>` | tipo de instrução: enum + DTO + despacho + executor + registro na factory |
| `/new-similarity-model <Nome>` | modelo de similaridade: enum + DTO de config + implementação + integração no executor |
| `/new-calibration-model <Nome>` | modelo de calibração, idem |
| `/new-validation-rule <Nome>` | regra estendendo `ValidationRuleBase` |
| `/new-transformation <Nome>` | transformação de embedding: `StepType` + DTO + `TransformationType` + transformador + registro no pipeline |

Pontos de registro que costumam ser esquecidos: `StrategyExecutorFactory` (executores e regras sempre ligadas); `_PROVENANCE_METADATA_KEYS` (chave de proveniência); `MEASURES` em `image_degradation_profile_service` + campo com default `None` em `DegradationMeasures` (medida nova); `ImageProcessorFactory` + config com `param_specs()` (passo de imagem — o menu lê `available_step_types()`); `faceaudit/models/ofiq_measure_names.py` (medida do OFIQ); `tests/test_ofiq_staging_extensions.py` (extensão de imagem nova).

## Índice de docs

- `docs/arquitetura/README.md` — índice da arquitetura.
- `docs/arquitetura/motor_estrategias.md` — decisões do motor, desempenho, projeto/biblioteca, tabela completa das instruções, ferramentas externas.
- `docs/arquitetura/mapa_de_pastas.md` — mapa detalhado de pastas e protocolos.
- `docs/arquitetura/validacao.md` — sistema de validação completo.
- `docs/arquitetura/arquiteturas_lr.md` — `LR_ARCHITECTURE_VECTORIZATION`, `SCORE_INTERACTION_FEATURE`, auditoria de monotonicidade.
- `docs/arquitetura/deteccao_e_geometria.md` · `datasets_e_caminhos.md` · `metadados_captura.md` · `ofiq.md` · `pipeline_imagens.md` · `custodia.md` · `ffservice_gpu.md` · `frontend_v2.md`.
- `docs/dev/testes.md` — lista temática dos testes.
- Teoria: `docs/similaridade_faceaudit.md` (PLDA), `docs/typicality_models.md` (tipicidade), `docs/piecewise_linear_calibration.md` (calibração e CV por identidade), `docs/conditional_monotonicity_rule_design.md`, `docs/lr_architecture_vectorization_faceaudit.md`, `docs/interacao_score_raridade_faceaudit.md`.
