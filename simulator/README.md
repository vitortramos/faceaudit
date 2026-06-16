# AdaFace Embedding Simulator

Gerador sintético configurável de embeddings **AdaFace-like** para experimentos controlados com similaridade de cosseno, LDA, *whitening*, Gaussianização radial, PLDA, PSDA, calibração e tipicidade.

O pacote não implementa a rede neural AdaFace. Ele cria um mundo sintético auditável que opera em **dois regimes**: **sintético puro** (população sorteada de uma mistura vMF genérica) e — o recurso mais distintivo — **ancorado em datasets reais do FaceAudit** (população derivada de embeddings reais, para que a geometria sintética seja fiel aos seus dados; ver [População ancorada em dataset real](#população-ancorada-em-dataset-real)).

Em qualquer regime, é possível variar separadamente:

- geometria populacional esférica por mistura de vMF;
- dispersão intraidentidade (vMF, calota esférica IDperturb ou híbrido);
- norma do embedding e caudas radiais baixa e alta;
- qualidade latente;
- estrutura anisotrópica global, por domínio e por identidade;
- viés médio de domínio, viés domínio × identidade e viés de sessão;
- acoplamentos entre qualidade, norma e direção;
- condições de aquisição alinhadas por identidade.

## Recursos

- **⭐ População ancorada em dataset real** via `faceaudit_seed` — o recurso central. Modo A (`population_from_dataset`): centros de identidade reais (média esférica por identidade); Modo B (`simulator_config_from_dataset`): centros-fantasma de um MoVMF ajustado aos centros reais, com seleção de K por BIC/AIC e casamento opcional da norma. Guardas anti-vazamento (`assert_disjoint_source_identities`).
- **Dispersão intra-classe selecionável** via `IntraClassSamplingMode`: `VMF` (concentração `κ`, padrão), `IDPERTURB_CAP` (calota esférica com cosseno uniforme, Boutros et al. CVPR 2026) e `HYBRID_VMF_CAP` (corpo vMF com piso rígido de calota).
- **Perfis prontos** `make_spherical_baseline_config`, `make_cooperative_plda_config`, `make_cross_condition_stress_config` e `make_idperturb_cap_config`.
- **Serialização de configuração round-trippável**: `config_to_dict` / `config_from_dict` / `config_from_manifest`.
- **Adapter concreto FaceAudit** (`faceaudit_concrete_adapter`) para `EmbeddingInfo`/`EmbeddingDataset` reais (float32, parquet, cross-condition).

## Instalação

```bash
pip install numpy pandas
```

Para executar os testes:

```bash
pip install pytest
python -m pytest -q
```

## Exemplo rápido

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_cooperative_plda_config(dimension=512),
    random_seed=2026,
)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 10),
    split_name="test",
    identity_offset=200_000,
    random_seed=103,
    population_random_seed=1003,
)

test.save_bundle("synthetic_output", "test")
```

## População ancorada em dataset real

O recurso mais distintivo do pacote: em vez de sortear a população de uma mistura vMF
genérica, derive-a de embeddings **reais** de um `EmbeddingDataset` do FaceAudit, para que a
geometria populacional sintética seja fiel aos seus dados. Centros sorteados de uma MoVMF
genérica podem não respeitar a variedade fina/anisotrópica/agrupada dos embeddings reais — a
ancoragem fecha essa lacuna.

- **Modo A — ancorado** (`population_from_dataset`): os centros de identidade são a média
  esférica por identidade dos embeddings reais. As identidades são reais; só a dispersão
  intra-classe (e, opcionalmente, a norma) é sintética. **Não é anonimização** — o
  conjunto-âncora precisa permanecer **disjunto** da base de avaliação
  (`assert_disjoint_source_identities`).
- **Modo B — distribuição-casada** (`simulator_config_from_dataset`): ajusta-se um MoVMF aos
  centros reais (via `MoVMFTypicalityModel`, seleção de K por BIC/AIC) e geram-se
  centros-fantasma inéditos — uma aproximação angular de primeira ordem da distribuição
  empírica, **não** o manifold real. Casa opcionalmente a norma radial real.

```python
import faceaudit_seed as fs
from adaface_embedding_simulator import AdaFaceEmbeddingSimulator, make_cooperative_plda_config

simulator = AdaFaceEmbeddingSimulator(make_cooperative_plda_config(dimension=512), random_seed=7)

# Modo A: centros = média esférica por identidade dos embeddings reais.
population = fs.population_from_dataset(simulator, real_embedding_dataset, min_samples=2)
anchored = simulator.sample_observations(population, samples_per_identity=(4, 10), split_name="anchored")
# Proveniência por amostra na metadata: 'source_identity', 'source_sample_count'.
```

Para levar o resultado de volta ao FaceAudit como um `EmbeddingDataset` real, use
`faceaudit_concrete_adapter.to_embedding_dataset`. Referência completa em
[`DOCUMENTATION.md`](DOCUMENTATION.md) (Seção 14).

## Dispersão por calota esférica (IDperturb)

O modo `IDPERTURB_CAP` substitui a dispersão vMF por uma calota esférica de cosseno
uniforme: `s ~ U[lb, 1]` e `v = s·μ + sqrt(1−s²)·u`. O piso `lb` controla a separação
intra-classe (`E[cos] = (1+lb)/2`; `lb = 1` ⇒ sem perturbação) e é modulado por qualidade,
penalidades de norma e a guarda do bissetor anti-colisão.

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_idperturb_cap_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_idperturb_cap_config(dimension=512, lower_bound=0.6),
    random_seed=7,
)

dataset = simulator.sample_dataset(
    identity_count=500,
    samples_per_identity=(4, 10),
    split_name="cap_demo",
    random_seed=11,
    population_random_seed=111,
)

# Sempre reporte o lb EFETIVO (a guarda do bissetor tem prioridade e pode elevá-lo
# acima de cap_maximum_lower_bound):
print(dataset.metadata["cap_lower_bound_effective"].describe())
```

## Stress test cross-condition alinhado

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cross_condition_stress_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_cross_condition_stress_config(dimension=512),
    random_seed=3030,
)

aligned = simulator.sample_cross_condition_datasets(
    identity_count=800,
    identity_offset=300_000,
    split_name="cross_condition_test",
    samples_per_identity_by_domain={
        "controlled": (3, 6),
        "surveillance": (3, 6),
    },
    sessions_per_identity=(1, 3),
    population_random_seed=4001,
    observation_random_seeds={
        "controlled": 4002,
        "surveillance": 4003,
    },
)
```

## Serialização e round-trip de configuração

A configuração persistida no manifesto pode ser relida sem perda, fechando a cadeia de
auditoria entre o que foi gerado e o que foi documentado:

```python
from adaface_embedding_simulator import (
    config_from_dict,
    config_from_manifest,
    config_to_dict,
)

# 'manifest.json' é escrito por dataset.save_bundle(...) / simulator.save_manifest(...)
config = config_from_manifest("synthetic_output/test/manifest.json")

# Idempotência verificável: serializar -> desserializar -> serializar é estável.
payload = config_to_dict(config)
assert config_to_dict(config_from_dict(payload)) == payload
```

`config_from_dict` aceita o dicionário produzido por `config_to_dict` (enums como string,
arrays como listas); chaves ausentes caem nos *defaults* do dataclass (compatível com
configs antigas) e chaves desconhecidas levantam erro, expondo *version drift* cedo.

## Documentação

Novo por aqui? Comece pelo [`QUICKSTART.md`](QUICKSTART.md) — porta de entrada operacional (árvore de decisão "quero X → use Y", tabela de perfis e snippets mínimos).

Consulte [`DOCUMENTATION.md`](DOCUMENTATION.md) para a referência completa da API, fórmulas,
parâmetros configuráveis, perfis prontos, modos de dispersão intra-classe, geração de pares,
diagnósticos radiais e por calota, integração com o FaceAudit e roteiro de calibração contra
embeddings reais.
