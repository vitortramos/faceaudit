# FaceAudit

**Plataforma forense de análise e comparação de imagens faciais** — para perícia
oficial em casos onde o resultado da comparação precisa ser apresentado em corte
como **likelihood ratio (LR) defensável**, não como score de similaridade bruto.

O usuário descreve a estratégia de cálculo do LR através da interface (pipeline
de transformação de embeddings → modelo de similaridade → calibrador). O
mecanismo de execução interpreta essa estratégia. Nada é hard-coded no código —
o que muda entre casos forenses são as estratégias declaradas pelo perito.

## O que o FaceAudit faz

- **Extrai embeddings faciais** a partir de imagens via AdaFace (com pipeline
  de detecção + alinhamento da ForensicFace).
- **Treina pipelines de transformação** desses embeddings: centering, whitening
  PCA, LDA, L2-norm, CORAL alignment, Radial Gaussianization, identity
  aggregation.
- **Modela similaridade** via cosseno ou PLDA (variantes SIMPLE/SpeechBrain
  e TWO_COVARIANCE/Brümmer-de Villiers 2010 em forma fechada).
- **Calibra scores brutos** em log-likelihood ratios via três métodos:
  Regressão Logística (com GridSearch), KDE (estimação de densidade com
  transformação de Fisher) e Bi-Gaussianized (Morrison 2024, com variante
  KDE_SMOOTHED própria do projeto para eliminar stairsteps).
- **Funde múltiplos sistemas** via score-level fusion (Brümmer & du Preez 2006):
  combina, por exemplo, cosseno em embedding bruto com PLDA em embedding
  whitened, herdando o melhor de cada.
- **Avalia performance forense** com Cllr, Cllr_min via PAVA, EER, AUC, ECE
  empírico, Tippett plots e DET curves.
- **Aplica a estratégia treinada em casos forenses reais** comparando uma
  imagem questionada contra uma de referência conhecida, produzindo log-LR
  calibrado com explicação visual da posição do caso entre as distribuições
  de same-source e different-source.

## Quick start

### Pré-requisitos

- Python 3.11+
- Windows (otimizado), macOS ou Linux
- Modelos AdaFace + ForensicFace + OFIQ + SERFIQ (paths configurados em
  `config.json`)

### Instalação

```bash
git clone <repo-url>
cd faceaudit
pip install -e .
```

Para gerar PDFs de documentação (opcional):

```bash
pip install -e ".[docs]"
```

Para desenvolvimento:

```bash
pip install -e ".[dev]"
```

### Executar

```bash
python interface/run.py
```

A primeira inicialização demora ~1 min (JIT compile do UMAP). Subsequentes
são rápidas.

### Configuração mínima

Edite `config.json` com os paths locais para a biblioteca global de recursos
(datasets, configs, modelos):

```json
{
  "paths": {
    "library": "C:/caminho/para/library",
    "dir_templates": "data/templates"
  },
  "ofiq": {
    "ofiq_exe_path": "caminho/para/OFIQSampleApp.exe",
    "ofiq_data_path": "caminho/para/OFIQ-MODELS"
  },
  "ff": {
    "use_gpu": true,
    "batch_size": 32
  }
}
```

E `config.yaml` para opções de sistema (GPU, seeds aleatórios, paths de saída).

## Casos de uso forense

| Cenário | Fluxo |
|---|---|
| **Comparação 1:1 questionada × padrão** | Extrair embeddings → aplicar pipeline → score → calibrar → reportar log-LR + Tippett plot do caso |
| **Validação de uma estratégia nova** | Carregar dataset com ground truth → split em treino/calibração/teste → rodar estratégia → avaliar Cllr, EER, ECE |
| **Auditoria de qualidade de dataset** | FIQA, OFIQ, SERFIQ, norma do embedding por face → balanceamento por quality → detectar outliers topológicos (hubness) |
| **Comparação de duas estratégias** | Calibration distribution overlap + tabela de métricas + plots overlay (DET, Tippett, ECE) lado a lado |
| **Score fusion** | Combinar dois sistemas (ex: cosseno + PLDA) via LogReg multivariado → calibrador único com tipicidade incorporada |

## Arquitetura

O FaceAudit usa uma **engine declarativa de estratégias**: uma estratégia é
descrita como um `LRStrategyDefinition` (lista de recursos + sequência ordenada
de instruções organizadas em três fases). O `StrategyExecutor` interpreta e
executa.

```
StrategyExecutorFactory.build()
  └─ StrategyExecutor.execute(LRStrategyDefinition)
       ├─ Cria StrategyExecutionContext (state bus compartilhado)
       ├─ Carrega recursos iniciais no contexto
       ├─ Valida estratégia (regras plugáveis + always-on)
       └─ Para cada fase (Preparação → Validação → Análise de Caso):
            Para cada instrução:
              InstructionExecutorRegistry.resolve(instruction_type)
                ├─ Valida (4 estágios: structural, plugável, dinâmica, executor)
                └─ Executa (lê recursos → roda lógica → grava resultados)
       └─ Empacota contexto final em LRStrategyTrainedModel
```

### Tipos de instrução

| Categoria | Instruções |
|---|---|
| **Preparação de dados** | `DATASET_SPLIT`, `PAIRED_DATASET_SPLIT`, `DATASET_MERGE`, `DATASET_BALANCING`, `PAIR_MINING` |
| **Pipeline** | `PIPELINE_TRAINING`, `PIPELINE_APPLICATION` |
| **Modelagem** | `SIMILARITY_MODEL_TRAINING`, `SIMILARITY_SCORING`, `CALIBRATION_MODEL_TRAINING`, `SCORE_CALIBRATION`, `SCORE_FUSION` |
| **Avaliação** | `STRATEGY_EVALUATION`, `CASEWORK_VISUALIZATION` |
| **Diagnóstico** | `HUBNESS_ANALYSIS`, `HUBNESS_ABLATION` |

Cada instrução é um DTO (`StrategyInstructionDefinition`) com seus parâmetros e
um `Executor` correspondente. Adicionar uma instrução nova é tipicamente 4-5
commits: enum + DTO + dispatch + executor + factory registration.

### Princípios de design

- **Pendrive paradigm**: cada projeto FaceAudit é uma pasta autocontida com
  arquivo `.fa` na raiz. Move o pendrive entre máquinas (Windows/macOS) e tudo
  continua funcionando — paths são relativos, sistema cross-platform por
  construção.
- **Biblioteca global**: datasets, configs e templates reutilizáveis entre
  projetos ficam numa pasta global gerenciada por `GlobalLibraryStorage`.
- **Verificação automática de integridade**: SHA-256 de cada artefato
  persistido; ao abrir o projeto, integridade é checada e qualquer
  modificação/perda é sinalizada.
- **Validação em múltiplos níveis**: regras plugáveis configuráveis pelo
  usuário (MIN_CALIBRATION_PAIRS, FISHER, DATASET_OVERLAP, ...) + regras
  intrínsecas dinâmicas (MODEL_FITTED, EMBEDDING_DIM_MATCHES, ...) + hard-coded
  por executor. Total: 19 tipos de regra.

## Estrutura de pastas

```
faceaudit/
├── enums/                 Enums tipados (instruction types, transformation types, ...)
├── contexts/              StrategyExecutionContext — state bus da execução
├── executors/
│   ├── strategy_executor.py        Orquestrador principal
│   └── instructions/               Um executor por tipo de instrução
├── factories/             Factories de assembly (StrategyExecutor, ValidationRule)
├── models/
│   ├── calibration/                LogReg, KDE, Bi-Gaussianized
│   ├── similarity/                 Cosine, PLDA, SimilarityScoreDataset
│   ├── instructions/               DTOs por tipo de instrução
│   ├── pipeline/                   Pipeline training/application policies
│   ├── strategy/                   StrategyResource, LRStrategyDefinition
│   └── validation/                 ValidationResult, ValidationRuleConfig
├── pipelines/             EmbeddingTransformationPipeline (modos LINEAR e DUAL)
├── protocols/             Structural typing — contratos sem herança
├── repositories/          Persistência (JSON para definitions, pickle para modelos)
├── services/              FFService, OFIQService, SERFIQService, plotting,
│                          inference, hubness, integrity, dataset balancing
├── storage/               LocalProjectArtifactStorage + GlobalLibraryStorage
├── transformators/        Implementações concretas: Centering, Whitening, LDA,
│                          L2Norm, CORAL, RadialGaussianization, IdentityAggregation
├── validation_rules/      19 regras concretas estendendo ValidationRuleBase
└── scripts/               Utilities (validate_bigauss_against_paper.py, ...)

interface/                 PyQt6 GUI
├── tabs/
│   ├── dataset_manager/            Gestão de datasets + qualidade + filtros
│   └── strategy_manager_v2/        Editor de estratégias + execução + comparação
└── run.py                          Entry point

docs/                      Documentação técnica (PDFs gerados)
config.json                Paths runtime (modelos, datasets, output)
config.yaml                Settings de sistema (GPU, seeds)
```

## Componentes externos

| Ferramenta | Função |
|---|---|
| **AdaFace** | Modelo primário de embedding facial |
| **ForensicFace / sepaelv2/v4** | Detecção + alinhamento + extração (via FFService) |
| **OFIQ** | Avaliação de qualidade de imagem (NIST/BSI standard) |
| **SERFIQ** | Score de qualidade alternativo |
| **SpeechBrain** | Backend PLDA variante SIMPLE |
| **scikit-learn** | LogReg, KDE, GridSearch |
| **PyQt6** | GUI |

## Documentação técnica

Documentos PDF gerados sob demanda em `docs/`:

| Documento | Conteúdo |
|---|---|
| `dataset_manager_roadmap.pdf` | Roadmap de features do Dataset Manager |
| `biGauss_KDE_smoothed.pdf` | Variante KDE_SMOOTHED de G(x) no calibrador Bi-Gaussianized (motivação, matemática, ajuste de hyperparams, comparação com paper Morrison 2024) |

Para regenerar:

```bash
python docs/_build_<nome>_pdf.py
```

## Skills do projeto (para contribuidores que usam Claude Code)

`.claude/commands/` define skills que automatizam a adição de novos componentes:

| Comando | Cria |
|---|---|
| `/new-instruction <Nome>` | Tipo de instrução completo: enum + DTO + dispatch + executor + factory |
| `/new-similarity-model <Nome>` | Modelo de similaridade: enum + config + impl + executor |
| `/new-calibration-model <Nome>` | Modelo de calibração análogo |
| `/new-validation-rule <Nome>` | Regra estendendo `ValidationRuleBase` com cache helper |
| `/new-transformation <Nome>` | Step de pipeline: StepType + config + TransformationType + transformator |

## Testes

```bash
python -m pytest tests/
python -m pytest tests/test_full_pipeline.py
python -m pytest tests/transformation_pipeline_test.py
python -m pytest tests/test_storage_and_repositories.py

# Validação de parquet:
python tests/check_parquet.py
```

## Status

Versão **0.1.0** — em desenvolvimento ativo. Os componentes core (engine,
calibração, PLDA, score fusion, biGauss com variantes, hubness, dataset
management com schema-by-role) estão maduros. A documentação técnica está em
construção (este README marca o início).

## Referências forenses

A implementação dos calibradores e do framework de avaliação segue
literatura forense estabelecida:

- Brümmer & du Preez (2006) — *Application Independent Evaluation of Speaker
  Detection* (Cllr, score fusion).
- Brümmer & de Villiers (2010) — PLDA two-covariance.
- Morrison (2024) — *Bi-Gaussianized calibration of likelihood ratios*,
  Law, Probability and Risk 23.
- Morrison et al. (2021) — *Consensus on Validation of Forensic Voice
  Comparison*, Science & Justice.
- Vergeer et al. (2016) — ELUB bounds para LRs.
- Aitken, Roberts, Jackson — recomendações da ENFSI guideline.

## Como contribuir

Em construção. Antes de PRs grandes, abrir issue para discussão. Convenções:

- Docstrings em PT-BR (consistência com o resto do projeto).
- Não criar arquivos `.md` ou `README.md` adicionais sem solicitação explícita.
- Seguir o padrão arquitetural descrito em `CLAUDE.md` (Strategy Engine
  declarativa, separation of concerns entre DTO/Executor/Service).
- Smoke tests obrigatórios para qualquer novo Calibrador, Modelo de
  Similaridade, Transformator ou Instrução.

## Licença

A definir.

## Autoria

Em construção.
