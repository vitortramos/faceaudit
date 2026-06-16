# QUICKSTART — Simulador de embeddings AdaFace-like

Ponto de entrada operacional do `adaface_embedding_simulator`. Não substitui o manual: para a referência completa de API, fórmulas e caveats forenses, veja [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Os dois regimes

O simulador gera embeddings sintéticos no estilo AdaFace em **dois regimes** que você precisa distinguir desde já. No regime **sintético puro**, a população de identidades é *sorteada* de uma mistura de von Mises-Fisher (MoVMF) genérica — ideal para estudar a matemática de um backend (PLDA, calibração, tipicidade) sob *ground-truth* totalmente conhecido. No regime **ancorado em dataset real do FaceAudit**, a população é *derivada de embeddings reais* — seja semeando os centros nas médias por identidade reais (Modo A), seja ajustando uma distribuição MoVMF a esses centros e sorteando identidades-fantasma inéditas (Modo B). A ancoragem torna a geometria sintética fiel aos seus dados, ao custo de o *ground-truth* deixar de ser totalmente conhecido e de exigir guardas anti-vazamento.

## ⚠️ Layout plano — antes de qualquer import

A pasta `simulator/` é um **conjunto de módulos planos (top-level), sem `__init__.py`** — não é um pacote Python. Os imports são sem prefixo de pacote: `import adaface_embedding_simulator`, `import faceaudit_seed as fs`, etc. Eles **só resolvem se `simulator/` estiver no `sys.path`**. Três formas suportadas (Seção 2):

- **(a)** rodar de dentro de `simulator/` (`cd simulator` e então `python ...`);
- **(b)** inserir a pasta no `sys.path` em scripts/notebooks: `sys.path.insert(0, "/caminho/para/o/repo/simulator")`;
- **(c)** sob `pytest`, o `simulator/conftest.py` insere o diretório no `sys.path` automaticamente.

Todos os exemplos abaixo assumem que uma dessas formas está em vigor.

## Quero X → use Y

| Quero… | Use | Seção |
|---|---|---|
| Gerar dados sintéticos puros (split isolado em uma chamada) | `simulator.sample_dataset(...)` | 4, 12 |
| Ancorar a população em centros de identidade reais (**Modo A**) | `faceaudit_seed.population_from_dataset(simulator, dataset, min_samples=2)` + `simulator.sample_observations(pop, ...)` | 14 |
| Casar a distribuição angular com um dataset real (**Modo B / MoVMF**) | `faceaudit_seed.simulator_config_from_dataset(dataset, ...)` (ou só a população via `population_config_from_dataset`) | 14 |
| Usar dispersão por calota esférica (testar `IDPERTURB_CAP`) | `make_idperturb_cap_config(dimension=..., lower_bound=...)` | 6, 10 |
| Gerar cross-condition pareado (mesmas identidades entre domínios) | `simulator.sample_cross_condition_datasets(...)` | 13 |
| Exportar para o FaceAudit como `EmbeddingDataset` real | `faceaudit_concrete_adapter.to_embedding_dataset(...)` (ou `cross_condition_to_embedding_datasets(...)`) | 19 |
| Serializar / reproduzir uma configuração bit-a-bit | `config_to_dict` / `config_from_dict` / `config_from_manifest` | 11, 18 |

## Quando usar cada perfil / construtor

| Objetivo | Construtor recomendado |
|---|---|
| Controle de referência isotrópico (isolar o efeito de qualquer estrutura adicional) | `make_spherical_baseline_config` |
| Baseline PLDA realista mas benigno (dois domínios, nuisance moderado) — perfil padrão | `make_cooperative_plda_config` |
| Estresse cross-condition forte (mismatch sistemático, nuisance hierárquico + identidade) | `make_cross_condition_stress_config` |
| Afastamento angular máximo interpretável por identidade (piso de cosseno) | `make_idperturb_cap_config` |
| Geometria casada com um dataset real (Modo B, MoVMF + ajuste radial) | `faceaudit_seed.simulator_config_from_dataset` |
| População ancorada em identidades reais (Modo A) | `faceaudit_seed.population_from_dataset` |

## Snippets mínimos

### 1. Sintético puro

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

config = make_cooperative_plda_config(dimension=512)
simulator = AdaFaceEmbeddingSimulator(config, random_seed=42)

dataset = simulator.sample_dataset(
    identity_count=200,
    samples_per_identity=(4, 12),   # intervalo [min, max] por identidade
    split_name="train",
    identity_offset=0,
)

print(dataset.raw_embeddings.shape)         # (n_amostras, 512)
print(dataset.normalized_embeddings.shape)  # (n_amostras, 512), normas == 1

# Bundle auditável (NPZ + metadata CSV + diagnósticos + manifest JSON):
dataset.save_bundle("saida/synthetic", "train")
```

### 2. Ancorado em dataset real (Modo A)

```python
import faceaudit_seed as fs
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_cooperative_plda_config(dimension=512),
    random_seed=7,
)

# Centros = média esférica por identidade dos embeddings reais.
# min_samples=2 descarta identidades com uma única foto (âncora ruidosa).
population = fs.population_from_dataset(simulator, real_embedding_dataset, min_samples=2)

# Guarda anti-vazamento: a guarda compara os source_identity_labels dos dois lados, então
# eval_population deve ser OUTRA população ancorada (de um split de avaliação disjunto):
eval_population = fs.population_from_dataset(simulator, eval_embedding_dataset, min_samples=2)
fs.assert_disjoint_source_identities(population, eval_population)  # ValueError se houver interseção

# Gera variação sintética em torno das âncoras reais.
anchored = simulator.sample_observations(
    population,
    samples_per_identity=(4, 10),
    split_name="anchored",
)

# Proveniência por amostra na metadata:
print(anchored.metadata[["source_identity", "source_sample_count"]].head())
```

### 3. Exportar para o FaceAudit

```python
import numpy as np
from faceaudit_concrete_adapter import to_embedding_dataset

fa_dataset = to_embedding_dataset(
    anchored,                    # saída de sample_observations / sample_dataset
    name="synthetic_anchored",
    embedding_view="raw",        # preserva a norma
    quality_key="generic",
    origin="adaface_simulator",
    dtype=np.float32,            # casa com datasets reais extraídos
)
# fa_dataset é um EmbeddingDataset real, pronto para o pipeline do FaceAudit.
# Para datasets cross-condition pareados, use cross_condition_to_embedding_datasets(...).
```

## Caveats

- **Modo A NÃO é anonimização.** Ele produz variação sintética *em torno de âncoras biométricas reais*; as identidades resultantes são reais. Use apenas para augmentação, validação metodológica ou experimentos internos sobre base autorizada.
- **Mantenha a base-âncora disjunta da avaliação.** Invoque sempre `assert_disjoint_source_identities` ao ancorar populações distintas a partir de bases reais — é a guarda canônica contra vazamento âncora↔avaliação.
- **Ground-truth totalmente conhecido só no regime sintético puro.** Ao ancorar (Modo A/B), parte da geometria vem dos dados reais, e o Modo B reproduz apenas uma aproximação angular de **1ª ordem** da distribuição empírica, não a variedade fina dos embeddings.
- **Nenhum parâmetro é um valor "correto" por si só.** Perfis prontos e valores como `lb = 0.6` são ilustrativos/inspirados na literatura — calibre contra embeddings reais representativos e reporte o `profile_name` em laudo.
- Para a referência completa (assinaturas, fórmulas, modos de dispersão, geração de pares, diagnósticos e integração), consulte [`DOCUMENTATION.md`](DOCUMENTATION.md).
