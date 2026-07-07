# FaceAudit

**Plataforma forense para perícia de comparação facial.** Permite que o perito
desenhe, treine e aplique estratégias declarativas de cálculo de **likelihood
ratio (LR)** sobre embeddings faciais extraídos por modelos Sepael (v2–v6), com
fundamentação estatística defensável em corte.

O FaceAudit é composto de **duas partes integradas**, acessadas como abas do
aplicativo:

1. **Gerenciador de Datasets** — onde imagens viram datasets auditáveis,
   enriquecidos com métricas de qualidade, transformados e preparados para uso
   forense.
2. **Gerenciador de Estratégias** (rótulo da aba na UI) — onde o perito monta
   a estratégia de cálculo de LR como uma sequência de instruções declarativas,
   treina os modelos, valida com métricas forenses e aplica em casos reais.

Não há nada hard-coded sobre como o LR é calculado. A estratégia é descrita
pelo perito, persistida em JSON, executada por uma engine genérica e auditada
ponta-a-ponta com hashes SHA-256 e cadeia de custódia.

---

## Sumário

- [Instalação](#instalação)
- [Gerenciador de Datasets](#gerenciador-de-datasets)
- [Gerenciador de Estratégias de Comparação](#gerenciador-de-estratégias-de-comparação)
- [Documentação técnica](#documentação-técnica)
- [Testes](#testes)
- [Referências forenses](#referências-forenses)

---

## Instalação

### Pré-requisitos

- **Python 3.11+**
- **Sistema operacional**: Windows (otimizado), macOS ou Linux.
- **Modelos Sepael (v2 a v6)**: arquivos ONNX baixados manualmente em
  `~/.forensicface/models/<sepaelv2|v3|v4|v5|v6>/`. O FaceAudit detecta
  automaticamente quais estão disponíveis (`SepaelModel.available_on_disk()`).
- **OFIQ** (opcional, mas recomendado): binário externo para auditoria de
  qualidade ICAO. Paths em `config.json` → `ofiq.*`.
- **GPU CUDA** (opcional, mas recomendado): aceleração significativa na
  extração de embeddings via ForensicFace.

### Dependências core (instaladas via `pip install -e .`)

| Dependência | Função |
|---|---|
| `numpy`, `pandas`, `scipy`, `scikit-learn`, `pyarrow` | Núcleo numérico, parquet I/O, LogReg, KDE, GridSearch |
| `opencv-python-headless` | Processamento de imagens (sem deps de GUI nativa) |
| `matplotlib`, `seaborn`, `statsmodels` | Plots forenses (Tippett, DET, ECE) e LOWESS de auditoria |
| `umap-learn` | t-SNE/UMAP scatter na auditoria de embeddings |
| `speechbrain` | Backend do PLDA variante SIMPLE |
| `PyQt6 >= 6.5` | Interface gráfica |
| `dependency-injector` | Wiring da arquitetura |
| `loguru` | Logging estruturado |

### Componentes externos

| Ferramenta | Função |
|---|---|
| **ForensicFace** | Lib que orquestra detecção (`det_10g`) + alinhamento + encoder. Instalada via git URL (PyPI tem apenas 0.4.1; precisamos ≥ 0.7.1 do upstream `rafribeiro/forensicface@v0.7.1`) |
| **Sepael V2** | Encoder AdaFace ResNet — embeddings 512-D (`embedding_v2` no parquet, `FFInfo.embedding_sepaelv2`) |
| **Sepael V3** | Encoder AdaFace ResNet — embeddings 512-D (`embedding_v3`, `FFInfo.embedding_sepaelv3`) |
| **Sepael V4** | Encoder AdaFace ViT WebFace4M — embeddings 512-D (`embedding_v4`, `FFInfo.embedding_sepaelv4`) |
| **Sepael V5** | Encoder TopoFR R200 Glint360K — embeddings 512-D (`embedding_v5`, `FFInfo.embedding_sepaelv5`) |
| **Sepael V6** | Encoder KPRPE — embeddings 512-D, requer `aligned_keypoints` como segundo input ONNX (`embedding_v6`, `FFInfo.embedding_sepaelv6`) |
| **OFIQ** | Avaliação de qualidade NIST/BSI (binário externo) |
| **SER-FIQ** | Score de qualidade alternativo — serviço de backend legado; **não mais exposto na interface de dataset** (o enriquecimento de qualidade user-facing é OFIQ + norma do embedding) |
| **SpeechBrain** | Implementação do PLDA SIMPLE |

### Passos

```bash
git clone <repo-url>
cd faceaudit
pip install -e .

# Documentação (opcional):
pip install -e ".[docs]"

# Desenvolvimento:
pip install -e ".[dev]"
```

### Configuração mínima

Um único arquivo `config.json` na raiz do projeto:

```json
{
  "paths": {
    "library": "C:/caminho/para/biblioteca",
    "dir_templates": "data/templates"
  },
  "ofiq": {
    "ofiq_exe_path": "",
    "ofiq_data_path": ""
  },
  "ff": {
    "use_gpu": true,
    "batch_size": 32
  }
}
```

**Chaves:**

| Chave | Função |
|---|---|
| `paths.library` | Raiz da biblioteca global. Subpastas `datasets/`, `pipelines/`, `strategies/`, `mining_configs/`, `split_configs/`, `validation_rules/` são derivadas automaticamente — você não precisa criar/listar cada uma |
| `paths.dir_templates` | Pasta onde ficam os templates de estrutura de pasta usados pelo botão "Criar Template" na criação de dataset |
| `ofiq.ofiq_exe_path` | Caminho para `OFIQSampleApp.exe`. Vazio = OFIQ desligado (operações de qualidade que dependem dele ficam indisponíveis) |
| `ofiq.ofiq_data_path` | Caminho para a pasta `OFIQ-MODELS` |
| `ff.use_gpu` | Se `true`, ForensicFace usa CUDA quando disponível |
| `ff.batch_size` | Tamanho do batch na extração de embeddings |

O FaceAudit cria o `config.json` automaticamente na primeira execução com
defaults relativos à pasta do projeto, e migra schemas legados (paths separados
por subpasta, `ff.model`, `serfiq.*`, `plda.*`) automaticamente para o formato
acima.

A escolha do modelo Sepael (v2/v3/v4/v5/v6) é feita em runtime — qualquer
modelo presente em `~/.forensicface/models/` aparece nas operações de
extração.

### Executar

```bash
python interface/run.py
```

A primeira inicialização demora ~1 min (JIT compile do UMAP). Subsequentes são
rápidas.

---

## Gerenciador de Datasets

A primeira aba do aplicativo. Aqui o perito transforma um conjunto de imagens
faciais — **ou faces extraídas de arquivos de vídeo** (busca por âncora em
material de vigilância) — num **dataset auditável** com schema tipado,
metadados de qualidade, identidades verificadas e cadeia de custódia.

### O que é um dataset no FaceAudit

Um dataset é um arquivo **parquet** (`.parquet`) com:

- **Uma linha por face detectada** (não por imagem original — uma imagem com
  várias faces gera várias linhas).
- **Colunas tipadas** descritas pelo `column_specs` na metadata do parquet
  (schema-by-role): identidade, file_name, image_path, face_hash, embeddings
  (`embedding_v2`/`v3`/`v4`/`v5`/`v6` — um por modelo Sepael, todos
  `float32` 512-D empacotados, presentes apenas para os modelos que foram
  rodados sobre o dataset), métricas de qualidade (FIQA, OFIQ, SER-FIQ),
  pose (yaw, pitch, roll), demografia (gênero, idade), origem, e colunas
  customizadas que o perito adicionar.
- **Metadata estruturada**: nome, descrição, paths absolutos vs relativos
  (`path_mode`), pasta-âncora portátil (`path_root`), flag `is_aligned` (faces
  já alinhadas em 112×112), `column_specs` com tipos e vocabulários fechados
  para colunas categóricas, hash do conjunto.
- **PNGs das faces** (opcional): subpasta autocontida no projeto, paths
  relativos. Move o pendrive entre Windows/macOS e continua funcionando.

### Ribbon "Arquivo" — I/O do dataset

| Botão | Função |
|---|---|
| **Novo Dataset** | Cria dataset a partir de pasta de imagens com identidades como subpastas; opção de alinhar durante a criação |
| **Dataset de Vídeo** | Extrai faces de um ou mais vídeos por âncora (procura um investigado no material), com revisão humana e manifesto de reprodutibilidade → gera um dataset |
| **Importar extração de vídeo** | Reaproveita uma pasta de extração de vídeo já produzida (revisar + gerar dataset) |
| **Abrir** | Carrega `.parquet` existente |
| **Salvar / Salvar Como** | Persiste com SHA-256 do conteúdo |
| **Biblioteca** | Adiciona dataset à biblioteca global (reusável entre projetos) |
| **→ Estratégia** | Envia o dataset corrente para o Strategy Manager — switch de aba + import automático |
| **Fechar** | Encerra dataset corrente |

### Ribbon "Dataset" — operações sobre o conteúdo

| Botão | Função |
|---|---|
| **Auditar** | Auditoria forense completa — 14+ verificações automáticas (ver abaixo) |
| **Filtrar / Limpar Filtro** | Filtros multi-coluna com critérios tipados (numérico, string/categórico, booleano) |
| **Modificar Caminhos** | Corrige a subpasta `path_root` de datasets relativos sem mover arquivos (`FIX_PATH_ROOT`), reescreve paths in-place como relativos (`REWRITE_RELATIVE`), copia imagens para pasta autocontida (`COPY_ORIGINALS`) ou substitui um prefixo de path cross-machine (`REBASE_PREFIX`). Para alinhar faces use o Editor de Pipeline de Imagens (`Origem = Face alinhada (112×112)`, zero steps) |
| **Enriquecer** | Adiciona embeddings de outros modelos Sepael, FIQA, OFIQ (qualidade ISO), pose e demografia — cada item com selo de cobertura (presente/parcial/ausente); só processa o que falta, salvo *Reprocessar* |
| **Coluna** | Adiciona coluna custom (anotação livre) |
| **Metadados** | Edita nome, descrição, rótulos, roles e vocabulários de colunas |
| **Juntar** | Mescla dois datasets — modos `CONCAT`, `CONCAT_PRESERVE_ORDER`, `REPLACE_BY_INDEX_LIST`, `UPDATE_BY_INDEX_LIST` |
| **Dividir** | Split em treino/validação/teste — modos `BY_IDENTITY_RATIO`, `BY_IDENTITY_LIST`, `BY_IDENTITY_ABSOLUTE`, `BY_INDEX_RATIO`, `BY_INDEX_LIST`, `BY_SUPERVISED_ADAPTATION` |
| **Balancear** | Balanceia amostras por identidade — modos `CAP_ONLY`, `FLOOR_AND_CAP`, `AUTO_MIN_BALANCE`, com critério `RANDOM` ou `BY_QUALITY` (FIQA, OFIQ, SER-FIQ, norma do embedding) |
| **Comparar** | Compara estatísticas de dois datasets lado a lado |
| **Estatísticas** | Estatísticas detalhadas — distribuição por identidade, balanceamento, cobertura de colunas |
| **Custódia** | Cadeia de custódia forense — histórico de operações com verificação SHA-256 |

### Auditoria forense (botão "Auditar")

Roda uma bateria de verificações que o perito pode revisar e usar pra "limpar"
o dataset. Cada finding tem severidade (`info`, `warning`, `critical`) e
sugestão acionável:

| Categoria | Verificações |
|---|---|
| **Integridade física** | Paths órfãos (imagem não existe em disco), conflitos de hash (mesmo `face_hash` em linhas diferentes), duplicatas exatas, embeddings faltantes |
| **Estrutura de identidades** | Identidades singleton (1 sample), distribuição de samples por ID, balanceamento (CV e razão max/min) |
| **Identidades duplicadas/misturadas** | Clustering por similaridade — detecta mesma pessoa em labels diferentes (`duplicate_identities`) ou label cobrindo várias pessoas (`mixed_identities`) |
| **Qualidade** | Histogramas de FIQA/OFIQ/SER-FIQ/norma, low-quality outliers, correlação FIQA × norma do embedding, boxplot por quartil de qualidade |
| **Topologia (hubness)** | Faces que atraem vizinhos errados em alta dimensionalidade — diagnóstico de viés do modelo |

As verificações biométricas (identidades duplicadas/misturadas) e a topologia
(hubness) são **passes pesados e opt-in** — cada uma é disparada em execução
separada, com confirmação, e não faz parte da bateria rápida O(N) do botão
"Auditar" (que roda apenas as checagens de integridade física, estrutura de
identidades e qualidade).

Cada finding pode ser explorado num preview que mostra as faces afetadas
lado a lado. O perito decide o que fazer (deletar linhas, renomear identidade,
mesclar, etc.).

### Ribbon "Imagens" — pipeline de processamento

| Botão | Função |
|---|---|
| **Editor** | Janela interativa para desenhar pipeline de processamento de imagens com preview lado-a-lado. Steps: `BLUR`, `JPEG`, `WebP`, `CROP`, `OCCLUSION` (livre via canvas de máscara + semântica via landmarks faciais com raio configurável) |
| **Processar** | Aplica pipeline em batch ao dataset todo, produz dataset enxuto com PNGs processados + manifest forense |

Pipelines de imagens são persistidos na biblioteca global como JSON
reusáveis. Útil para estudos de degradação (qual o efeito do blur 5×5 no LR?)
e para mascaramento privacy-preserving (ocluir região dos olhos antes de
compartilhar evidência).

### Schema-by-role

Cada coluna do parquet tem um **role** declarado no `column_specs`:

- `IDENTITY`, `IMAGE_PATH`, `FILE_HASH`, `EMBEDDING`, `QUALITY`, `POSE`,
  `GEOMETRY`, `DEMOGRAPHIC`, `DISPLAY`, `METADATA`, `CUSTOM`.
- Vocabulários fechados para colunas categóricas (renderizados como
  QComboBox na edição).
- Aliases para resolver nomes de coluna legados na leitura.
- Tipo de dado tipado (`str`, `int`, `float`, `bool`, `vector` (embeddings), `list`).

O role determina como a UI renderiza, valida e usa a coluna em operações
downstream (ex.: o filtro mostra slider em `QUALITY`, combo em colunas com
vocabulário fechado como `DEMOGRAPHIC`).

### Pendrive paradigm

Datasets com `path_mode=RELATIVE` + `path_root="images"` são totalmente
portáteis: a pasta do projeto contém o parquet + a pasta `images/` com PNGs,
e move entre máquinas Windows/macOS/Linux sem reconfigurar paths. O
FaceAudit normaliza separadores `\` ↔ `/` na serialização.

---

## Gerenciador de Estratégias de Comparação

A segunda aba do aplicativo. Aqui o perito monta a **estratégia de cálculo
de LR** como uma sequência de instruções declarativas, executa em datasets
auditados pelo Dataset Manager, valida com métricas forenses e aplica em
casos reais.

### Estrutura de uma estratégia

Uma estratégia (`LRStrategyDefinition`) é organizada em **três fases**:

1. **Preparação** — splits, pipeline training, similarity model training,
   calibration model training. Produz os modelos treinados.
2. **Validação** — aplicação dos modelos no holdout, scoring, calibração,
   métricas forenses. Confirma que a estratégia funciona em dados não vistos.
3. **Análise de Caso** — aplicação ao par questionada × padrão de um caso
   forense real; produz log-LR + visualização explicativa.

Cada fase contém uma sequência ordenada de **instruções**. Cada instrução
consome recursos do contexto (datasets, configs, modelos treinados) e produz
novos recursos.

### Tipos de instrução (20)

#### Preparação de dados

| Tipo | Função |
|---|---|
| `DATASET_SPLIT` | Divide um dataset em dois — modos por identidade (ratio/list/absolute) ou por índice. Inclui `BY_SUPERVISED_ADAPTATION` que separa source/target preservando overlap controlado |
| `PAIRED_DATASET_SPLIT` | Divide DOIS datasets simultaneamente por identidade — útil para preparar pares cross-pose (frontal × lateral) |
| `DATASET_MERGE` | Combina dois datasets — concat ou update by index list |
| `DATASET_BALANCING` | Balanceia amostras por identidade — `CAP_ONLY`, `FLOOR_AND_CAP`, `AUTO_MIN_BALANCE`, critério aleatório ou por qualidade |
| `PAIR_MINING` | Curadoria estatística de pares para treino de calibração (HARD/EASY/TYPICAL/RANDOM/MIXED — class-aware) |

#### Pipeline de transformação de embeddings

| Tipo | Função |
|---|---|
| `PIPELINE_TRAINING` | Treina um pipeline (sequência de transformators). Modos LINEAR (1 fluxo) ou DUAL (source/target com policy de compartilhamento de estado) |
| `PIPELINE_APPLICATION` | Aplica pipeline treinado a um ou mais datasets |

**Transformators disponíveis:** `CENTERING`, `WHITENING_TRANSFORMATION` (PCA),
`WCCN_TRANSFORMATION` (within-class covariance normalization — Hatch et al. 2006),
`LDA_TRANSFORMATION`, `L2_NORMALIZATION`, `CORAL_TRANSFORMATION` (domain
alignment), `RADIAL_GAUSSIANIZATION` (Lyu & Simoncelli 2009),
`IDENTITY_AGGREGATION` (template por identidade).

#### Modelagem de similaridade

| Tipo | Função |
|---|---|
| `SIMILARITY_MODEL_TRAINING` | Prepara/treina modelo de similaridade. Cosseno (stateless) ou PLDA em três variantes (SIMPLE via SpeechBrain, TWO_COVARIANCE em forma fechada Brümmer-de Villiers 2010, ou SHRINKAGE_TWO_COVARIANCE — two-covariance com shrinkage de W via Ledoit-Wolf) |
| `SIMILARITY_SCORING` | Aplica modelo treinado a um dataset (self-comparison) ou par (cross-comparison); produz `SimilarityScoreDataset` |
| `SCORE_VECTORIZATION` | **Score-level combination** (Brümmer & du Preez 2006): combina dois ou mais `SimilarityScoreDataset` (mesmos pares) em features multivariadas para calibrador 2D+ |

#### Calibração

| Tipo | Função |
|---|---|
| `CALIBRATION_MODEL_TRAINING` | Treina calibrador a partir de scores + labels. Famílias: **Logistic Regression** (com GridSearch e suporte a features multivariadas pós-vectorization), **KDE** (com Fisher transform e GridSearch de bandwidth), **Bi-Gaussianized** (Morrison 2024, com variante `KDE_SMOOTHED` própria do projeto para eliminar stairsteps) |
| `SCORE_CALIBRATION` | Aplica calibrador treinado a um `SimilarityScoreDataset`, produz log-LR por par |

#### Avaliação forense

| Tipo | Função |
|---|---|
| `STRATEGY_EVALUATION` | Calcula Cllr, Cllr_min via PAVA, EER, AUC, ECE empírico; produz dados de Tippett, DET e ECE plots |
| `CASEWORK_VISUALIZATION` | Plota distribuições de referência (gen/imp) com escores do caso forense sobrepostos — KDE quando há ≥30 pares, linhas verticais anotadas quando há menos |

#### Diagnóstico

| Tipo | Função |
|---|---|
| `HUBNESS_ANALYSIS` | Diagnóstico de viés topológico em alta dimensionalidade — Nk score, similaridade impostora, correlações com pose/norma. 5 estudos compósitos |
| `HUBNESS_ABLATION` | Estudo de ablação progressiva (por quantidade de fotos ou pose) — custoso, refaz O(N²) em N subsets |

#### Tipicidade forense

| Tipo | Função |
|---|---|
| `TYPICALITY_MODEL_TRAINING` | Treina um modelo de tipicidade a partir de um dataset de referência da subpopulação relevante (já estratificada). Backend escolhido via `typicality_model_type`: **k-NN esférico** (não-paramétrico, default) ou **MoVMF** (mistura von Mises-Fisher paramétrica, EXPERIMENTAL). No backend k-NN, três métricas por face: `mean_knn_similarity` (T_k), `typicality_percentile` (rank empírico LOO), `log_density_knn` (Loftsgaarden-Quesenberry esférico). Defensível em corte como estatística não-paramétrica de ranking — hubness deixa de violar a premissa do modelo e vira o sinal medido. |
| `TYPICALITY_PAIR_SCORING` | Aplica modelo(s) treinado(s) a cada face de cada par, combina simetricamente (MEAN/MIN/MAX/PRODUCT/ABS_DIFF) num score por par. **Suporta cross-população** (ex.: cross-pose forense — modelo lateral aplicado em probe lateral, modelo frontal aplicado em reference frontal). Produz `SimilarityScoreDataset` pareável diretamente pelo `SCORE_VECTORIZATION` com scores de similaridade pra treinar calibrador 2D+ que conhece raridade local. Reduz Cllr quando há heterogeneidade de tipicidade na população — validado num cenário de hubness sintético com ganho de 6–9%. |

#### Calibração LR forense (raridade)

| Tipo | Função |
|---|---|
| `SCORE_INTERACTION_FEATURE` | Constrói feature forense de **interação cosseno × raridade** `(base − s0)·rarity` a partir de dois `SimilarityScoreDataset` alinháveis por par (base cosseno/PLDA + modifier de tipicidade). Materializa a tese de que raridade **não é evidência independente** — só modula a *força* de uma similaridade observada. Modos `ONE_SIDED_POSITIVE`/`LINEAR`/`HINGE_BASE` (+ espelhos `_NEGATIVE`); `s0` (evidência neutra, logLR=0) resolvido via `AUTO_LOGREG`/`MANUAL`/`INHERIT` (INHERIT obrigatório em validação/casework). Output `raw_score` 1D consumível por `SCORE_VECTORIZATION`. |
| `LR_ARCHITECTURE_VECTORIZATION` | **Substituto declarativo** do encadeamento manual `SCORE_INTERACTION_FEATURE × N + SCORE_VECTORIZATION`: materializa em UMA instrução uma das arquiteturas de calibração LR (de `COSINE_ONLY` a `BILATERAL_FULL_CENTERED`), cuidando internamente de hinges, raridades centradas e vetorização. Quatro modos de parâmetro (`FIT_FULL_CALIBRATION_LEGACY`/`FIT_PER_FOLD`/`MANUAL`/`INHERIT` — INHERIT obrigatório em VALIDATION/CASEWORK). Persiste `architecture_state` + fingerprints para binding com o calibrador. |

### Recursos (StrategyResource)

Unidade de troca entre instruções. Cada instrução declara `input_resource_ids`
(consumidos) e `output_resource_ids` (produzidos), com type checking via
`StrategyResourceType`:

`DATASET`, `PIPELINE_CONFIG`, `SPLIT_CONFIG`, `MINING_CONFIG`,
`SIMILARITY_MODEL_CONFIG`, `CALIBRATION_MODEL_CONFIG`, `TRAINED_PIPELINE`,
`TRAINED_SIMILARITY_MODEL`, `TRAINED_CALIBRATION_MODEL`,
`TRAINED_TYPICALITY_MODEL`, `SIMILARITY_SCORES`,
`CALIBRATED_SCORES`, `LR_ARCHITECTURE_RAW_INPUT`, `VALIDATION_RESULT`,
`CASEWORK_VISUALIZATION`, `HUBNESS_REPORT`, `HUBNESS_ABLATION_RESULT`,
`TRAJECTORY_DATA`, ...

A UI usa esse type checking para filtrar dropdowns: o campo "Modelo
treinado" de uma instrução `SCORE_CALIBRATION` só lista recursos
`TRAINED_CALIBRATION_MODEL` no projeto corrente.

### Sistema de validação (22 regras)

Antes de executar, cada instrução passa por validação em 4 estágios:

1. **Estrutural** — null checks, tipos corretos.
2. **Plugável** — regras configuráveis pelo usuário (ex: `FISHER`,
   `MIN_CALIBRATION_PAIRS`, `MIN_IDENTITIES_WITH_MULTIPLE_SAMPLES`,
   `DATASET_OVERLAP_REQUIRED`, `MAX_SAMPLES_WARNING`).
3. **Dinâmica** — regras intrínsecas que não fazem sentido desabilitar
   (`MODEL_FITTED`, `EMBEDDING_DIM_MATCHES_MODEL`, `SCORE_CALIBRATION_STATE`,
   `DUAL_PIPELINE_SUBSEQUENCE`, `SCORE_VECTOR_PAIR_COMPATIBILITY`).
4. **Executor-specific** — checagens hard-coded por executor.

### Comparação de Estratégias

Janela top-level que abre via botão **Comparar** no painel esquerdo. Permite
comparar:

- **Distribuição do dataset de calibração** — gen × imp que alimentaram cada
  calibrador.
- **Resultados de Validação** — tabela de 11 colunas + 3 plots overlay (DET,
  Tippett, ECE) + grid de distribuições log-LR por estratégia.
- **Resultados de Casework** — tabela de 9 colunas + grid de plots forenses.

Permite ver lado a lado se o LR de uma estratégia A é mais conservador que o
de B, mesmo com Cllr comparável.

### Inferência forense (Análise de Caso)

Quando a estratégia está validada, o perito carrega na fase **Análise de
Caso** o par (probe questionado, reference padrão) e roda `SIMILARITY_SCORING`
+ `SCORE_CALIBRATION` + `CASEWORK_VISUALIZATION`. Resultado:

- **log-LR calibrado** por par.
- **Plot da distribuição de referência** (gen/imp da validação) com os
  escores do caso sobrepostos — KDE se ≥30 pares, linhas verticais
  anotadas se menos.

### Integridade e cadeia de custódia

- **SHA-256 de cada artefato persistido** (modelos, scores, datasets).
- **Verificação automática na abertura do projeto** (`FaceAuditProject.check_integrity()`).
- **Audit log** — toda operação sobre dataset ou estratégia é registrada
  no histórico de custódia.
- **Pendrive paradigm** — projeto autocontido em pasta, paths relativos,
  cross-platform.

---

## Documentação técnica

PDFs gerados sob demanda em `docs/`:

| Documento | Conteúdo |
|---|---|
| `dataset_manager_roadmap.pdf` | Roadmap de features do Dataset Manager |
| `biGauss_KDE_smoothed.pdf` | Variante KDE_SMOOTHED do calibrador Bi-Gaussianized — motivação, matemática, ajuste de hyperparams, A/B real |

Regenerar (o stem do script nem sempre coincide com o do PDF):

```bash
python docs/_build_roadmap_pdf.py            # → dataset_manager_roadmap.pdf
python docs/_build_bigauss_kde_smoothed_pdf.py  # → biGauss_KDE_smoothed.pdf
```

---

## Testes

```bash
python -m pytest tests/
python -m pytest tests/test_full_pipeline.py
python -m pytest tests/transformation_pipeline_test.py
python -m pytest tests/test_storage_and_repositories.py

# Validação de parquet:
python tests/check_parquet.py
```

---

## Referências forenses

A implementação dos calibradores e do framework de avaliação segue literatura
forense estabelecida:

- **Brümmer & du Preez (2006)** — *Application Independent Evaluation of
  Speaker Detection* (Cllr, score-level combination).
- **Brümmer & de Villiers (2010)** — PLDA two-covariance, forma fechada.
- **Lyu & Simoncelli (2009)** — *Nonlinear Extraction of Independent
  Components of Natural Images Using Radial Gaussianization*.
- **Morrison & Enzinger (2018)** — *Score-based Procedures for the
  Calculation of Forensic Likelihood Ratios*.
- **Morrison et al. (2021)** — *Consensus on Validation of Forensic Voice
  Comparison*, Science & Justice.
- **Morrison (2024)** — *Bi-Gaussianized calibration of likelihood ratios*,
  Law, Probability and Risk 23.
- **Neumann & Ausdemore (2020)** — *Defence Against the Modern Arts*.
- **Vergeer et al. (2016)** — ELUB bounds para LRs.

---

## Status

Versão **0.1.0** — em desenvolvimento ativo. Os componentes core (engine
declarativa, calibração tripla LogReg/KDE/biGauss, PLDA tri-variante
(SIMPLE/TWO_COVARIANCE/SHRINKAGE_TWO_COVARIANCE), score vectorization
multivariado, hubness diagnostics, tipicidade kNN não-paramétrica — e MoVMF
paramétrica (experimental) — integrada ao calibrador via score vectorization,
dataset management com schema-by-role e auditoria forense) estão maduros. A
documentação técnica está em construção.

---

## Licença

A definir.

## Autoria

A definir.
