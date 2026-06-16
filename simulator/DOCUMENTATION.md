# `adaface_embedding_simulator`

## Manual completo do gerador sintético de embeddings AdaFace-like

## 1. Visão geral

O pacote `adaface_embedding_simulator` é um **gerador sintético de embeddings no estilo
AdaFace**, projetado para experimentos controlados de **PLDA, calibração de razões de
verossimilhança (LR), tipicidade e backends de similaridade** (cosseno, LDA, whitening,
alternativas esféricas). Ele produz diretamente os **vetores de embedding** e seus
**metadados** — junto com tabelas de pares, escores e diagnósticos — sob um modelo
generativo totalmente parametrizado, determinístico e auditável.

### O que o pacote NÃO é

A delimitação de escopo é essencial para a defensibilidade forense de qualquer estudo que
use o simulador:

- **Não implementa a rede neural AdaFace.** Não há extração de features de imagens, nem
  pesos de modelo, nem inferência de uma CNN/ViT. O pacote não "roda o AdaFace".
- **Não gera imagens faciais.** Não há pixels, alinhamento, detecção, nem qualidade óptica
  real. O que o simulador chama de "qualidade" (`latent_quality`) e de "pose"/"nuisance" são
  variáveis **latentes** do modelo generativo, e não medidas físicas extraídas de uma face.
- **Não é uma fonte de verdade biométrica.** Os embeddings são amostras de uma distribuição
  paramétrica escolhida pelo usuário. Eles imitam *propriedades geométricas* plausíveis de
  embeddings AdaFace reais (norma variável, dispersão intra-classe controlada, anisotropia,
  domínios de aquisição), mas **não correspondem a indivíduos reais** a menos que a população
  seja explicitamente ancorada em um dataset real (ver Seção 14) — e mesmo aí gera-se
  *variação sintética em torno* de âncoras reais, nunca os próprios dados originais.

### A decomposição central `x = r · u`

Todo embedding sintético é construído pela fatoração geométrica fundamental

\[
x = r \cdot u, \qquad r = \lVert x \rVert_2 > 0, \qquad u = \frac{x}{\lVert x \rVert_2} \in S^{D-1},
\]

em que:

- **`r`** é a **norma radial** (a magnitude do vetor bruto), modelada em escala log via
  `RadialConfig`. Em embeddings AdaFace reais a norma carrega sinal de qualidade/dificuldade
  da amostra, e o simulador reproduz esse acoplamento.
- **`u`** é a **direção** (o vetor unitário na hiperesfera \( S^{D-1} \)), modelado pela
  geometria de população + dispersão intra-classe + perturbações tangenciais.

Os dois fatores variam **independentemente**: o `raw_embeddings` carrega `r·u`, enquanto o
`normalized_embeddings` carrega apenas `u`. Isso permite estudar separadamente backends que
usam a norma (ex.: detecção via norma) e backends puramente angulares (cosseno, PLDA esférico).

### As cinco camadas conceituais

O modelo generativo é organizado em **cinco camadas que podem ser variadas de forma
independente**, cada uma com sua própria seção de configuração. Essa separação é o que torna
o simulador útil para **ablações causais**: pode-se ligar/desligar uma camada de cada vez e
medir o efeito isolado sobre o backend.

| # | Camada conceitual | Onde se configura | O que controla |
|---|---|---|---|
| 1 | **População de identidades** | `PopulationConfig` | Onde vivem os centros de identidade na hiperesfera — mistura de componentes (`component_count`), concentração dos centros (`identity_center_kappas`), espalhamento global das componentes (`component_mean_global_kappa`). |
| 2 | **Norma radial** | `RadialConfig` | A distribuição de `r` (log-norma) por regime (central / cauda baixa / cauda alta) e seu acoplamento com a qualidade latente. |
| 3 | **Dispersão intra-classe** | `DirectionalConfig` (campos de concentração e de calota) | Quão espalhadas ficam as amostras de uma mesma identidade em torno do seu centro — via vMF (`kappa`), calota esférica cosseno-uniforme (`IDPERTURB_CAP`) ou híbrido (ver Seção 6). |
| 4 | **Perturbações tangenciais / anisotropia** | `DirectionalConfig` (campos `anisotropic_*`, `nuisance_structure_mode`, `tangent_update_mode`) | A estrutura direcional do ruído (global, por domínio, hierárquica) e a forma de mapeá-lo de volta à esfera (retração ou mapa exponencial). |
| 5 | **Domínio de aquisição** | `AcquisitionDomainConfig` (uma ou mais) | Condições heterogêneas (ex.: `controlled` vs `surveillance`): probabilidade de regime, deslocamentos de norma e de κ, viés de domínio/sessão, distribuição de qualidade. |

Acima dessas cinco camadas, o `CouplingConfig` funciona como um **painel de chaves causais**:
liga/desliga acoplamentos entre as camadas (ex.: "a qualidade afeta a norma?", "a qualidade
afeta a concentração?") para permitir experimentos contrafactuais limpos (ver Seção 9).

### Propósito honesto

O simulador existe para **experimentos controlados**: comparar backends, estressar
calibradores, validar pipelines de tipicidade e estudar a sensibilidade de métricas forenses
(Cllr, EER, ECE, Tippett) a propriedades geométricas conhecidas — porque aqui o
*ground-truth* gerador é totalmente conhecido.

> **Aviso de defensibilidade.** Nenhum parâmetro do simulador é, por si só, um valor
> "correto" para AdaFace/PLDA/FaceAudit. Os perfis prontos e os valores como `lb = 0.6` são
> **ilustrativos** e/ou **inspirados** na literatura, nunca ótimos comprovados. Para qualquer
> alegação científica ou pericial: **calibre os parâmetros contra embeddings reais
> representativos** (Seção 21) e **reporte explicitamente qual perfil sintético foi usado**
> (o nome vive em `config.profile_name` e é persistido no manifesto). O simulador é uma
> bancada de teste, não evidência.

---

## 2. Instalação e dependências

### Dependências de runtime

O núcleo do simulador depende de exatamente duas bibliotecas científicas:

- **`numpy`** — toda a álgebra de embeddings, amostragem RNG e geometria esférica.
- **`pandas`** — a tabela de metadados (`SyntheticEmbeddingDataset.metadata`), a tabela de
  pares (`sample_pair_table`) e os diagnósticos de norma. O módulo importa `pandas` no topo e
  falha com uma `ImportError` explicativa se ele não estiver disponível:

```python
try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "This module requires pandas. Install it with: pip install pandas"
    ) from exc
```

O núcleo gerador (`adaface_embedding_simulator.py`), o adapter de referência
(`faceaudit_adapter.py`) e os testes só exigem `numpy` + `pandas`. As funcionalidades de
**ancoragem em dataset real** (`faceaudit_seed.py`, modo MoVMF) e o **adapter concreto**
(`faceaudit_concrete_adapter.py`) importam, sob demanda, classes do projeto hospedeiro
FaceAudit (`MoVMFTypicalityModel`, `EmbeddingDataset`, `EmbeddingInfo`) — esses imports só são
acionados quando você usa essas funções específicas, então o núcleo permanece autossuficiente.

### Instalação

```bash
pip install numpy pandas
```

Para usar a ancoragem em dataset real e os adapters concretos, é necessário ter o pacote
`faceaudit` instalado e importável no mesmo ambiente (no repositório, via
`pip install -e .` na raiz).

### `simulator/` não tem `__init__.py` — como importar

A pasta `simulator/` é um **conjunto de módulos planos** (top-level), sem `__init__.py` e
**não é um pacote Python**. Os módulos são importados sem prefixo de pacote — `import
adaface_embedding_simulator`, `import faceaudit_seed`, etc. — espelhando o layout standalone.
Há três formas suportadas de torná-los importáveis:

**(a) Rodar a partir de dentro de `simulator/`.** Se o diretório de trabalho for a própria
pasta `simulator/`, os imports planos resolvem naturalmente:

```bash
cd simulator
python -c "from adaface_embedding_simulator import make_cooperative_plda_config; print('ok')"
```

**(b) Inserir a pasta no `sys.path` programaticamente.** Útil em scripts e notebooks fora da
pasta:

```python
import sys
sys.path.insert(0, "/caminho/para/o/repo/simulator")

from adaface_embedding_simulator import AdaFaceEmbeddingSimulator, make_cooperative_plda_config
```

**(c) Via `conftest.py` (automático sob pytest).** O `simulator/conftest.py` insere o próprio
diretório no `sys.path` no momento da coleta, de modo que os imports planos resolvem tanto se
os testes forem rodados da raiz do repositório quanto de dentro de `simulator/`:

```python
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
```

### Rodando os testes

A suíte vive em `simulator/tests/`. Com o `conftest.py` na pasta acima, qualquer um dos
comandos abaixo funciona:

```bash
# A partir da raiz do repositório
python -m pytest simulator/tests/

# Ou de dentro da pasta do simulador
cd simulator
python -m pytest tests/

# Um arquivo específico
python -m pytest simulator/tests/test_idperturb_cap.py -v
```

A cobertura dos testes está detalhada na Seção 23.

---

## 3. Estrutura do pacote

A árvore abaixo reflete o conteúdo real da pasta `simulator/`. O coração do pacote é um único
módulo monolítico (`adaface_embedding_simulator.py`); os demais módulos são pontes opcionais
para o FaceAudit, documentação, exemplo executável e testes.

```
simulator/
├── adaface_embedding_simulator.py     ← NÚCLEO: enums, configs, funções geométricas,
│                                          AdaFaceEmbeddingSimulator, dataset, pares, perfis
├── faceaudit_seed.py                  ← Ancoragem em dataset real (Modo A e Modo B / MoVMF)
├── faceaudit_adapter.py               ← Adapter de REFERÊNCIA (scaffold com injeção genérica)
├── faceaudit_concrete_adapter.py      ← Adapter CONCRETO de produção (DTOs reais do FaceAudit)
├── example_generate_adaface_embeddings.py  ← Script executável de ponta a ponta
├── conftest.py                        ← Insere a pasta no sys.path para os imports planos
├── DOCUMENTATION.md                   ← Documentação técnica de referência
├── README.md                          ← Visão de entrada do pacote
└── tests/
    ├── test_adaface_embedding_simulator.py   ← Núcleo: geometria, dataset, escores
    ├── test_config_roundtrip.py              ← Serialização config → dict → config / manifesto
    ├── test_core_population_injection.py     ← Injeção de centros/componentes na população
    ├── test_idperturb_cap.py                 ← Calota cosseno-uniforme + guarda de overlap
    ├── test_faceaudit_seed.py                ← Ancoragem em dataset real (A e B), disjunção
    ├── test_faceaudit_adapter.py             ← Adapter de referência (injeção genérica)
    └── test_faceaudit_concrete_adapter.py    ← Adapter concreto contra DTOs reais
```

### Papel de cada módulo

- **`adaface_embedding_simulator.py` (núcleo).** Contém tudo que é necessário para gerar
  embeddings sem nenhuma dependência do FaceAudit. Define: os enums
  (`NuisanceStructureMode`, `TangentUpdateMode`, `PairSamplingStrategy`,
  `IntraClassSamplingMode`); os dataclasses de configuração (`SimulatorConfig`,
  `PopulationConfig`, `RadialConfig`, `RadialReference`, `DirectionalConfig`,
  `CouplingConfig`, `AcquisitionDomainConfig`); as funções geométricas (`l2_normalize_rows`,
  `sample_uniform_directions`, `sample_vmf`, `sample_spherical_cap`, `sample_hybrid_vmf_cap`,
  `cosine_scores`); a classe geradora `AdaFaceEmbeddingSimulator`; os contêineres de resultado
  (`SyntheticIdentityPopulation`, `SyntheticEmbeddingDataset`); a geração de pares
  (`sample_pair_table`); a serialização (`config_to_dict`/`config_from_dict`/
  `config_from_manifest`); e os **perfis prontos** (`make_spherical_baseline_config`,
  `make_cooperative_plda_config`, `make_cross_condition_stress_config`,
  `make_idperturb_cap_config`, `make_example_config`).

- **`faceaudit_seed.py` (ancoragem em dataset real).** Funções que constroem populações e
  configurações a partir de embeddings reais. **Modo A** (`identity_centers_from_dataset`,
  `population_from_dataset`) usa os centros empíricos das identidades reais como centros
  sintéticos. **Modo B** (`population_config_from_dataset`, `simulator_config_from_dataset`)
  ajusta uma mistura de von Mises-Fisher (MoVMF, via `MoVMFTypicalityModel` do FaceAudit) à
  distribuição dos centros e fabrica centros "fantasma". Inclui salvaguardas de disjunção
  (`source_identity_set`, `assert_disjoint_source_identities`). Detalhado na Seção 14.

- **`faceaudit_adapter.py` (adapter de referência / scaffold).** Helper de **referência** com
  **injeção genérica de construtores**. Expõe `dataset_to_embedding_records` e
  `to_faceaudit_embedding_dataset`, recebendo `embedding_info_factory` e
  `embedding_dataset_factory` como parâmetros. **Importante:** assume a assinatura
  `EmbeddingInfo(embedding=, identity=, metadata=)`, que **não corresponde** ao DTO real do
  FaceAudit. Serve como andaime didático e não deve ser usado em produção — use o adapter
  concreto. Detalhado na Seção 19.

- **`faceaudit_concrete_adapter.py` (adapter concreto / produção).** A ponte real. Importa os
  DTOs do hospedeiro (`EmbeddingDataset`, `EmbeddingInfo`) e mapeia as colunas de metadata do
  simulador sobre os campos reais (`face_hash`, `image_path`, `quality_scores`, `source`,
  etc.). Expõe `to_embedding_dataset` e `cross_condition_to_embedding_datasets`. É o caminho
  recomendado para alimentar uma estratégia FaceAudit com dados sintéticos. Detalhado na
  Seção 19.

- **`example_generate_adaface_embeddings.py` (exemplo executável).** Script de ponta a ponta
  que demonstra os fluxos principais — split train/dev/test via `sample_dataset` +
  `save_bundle`, demo de calota, e ancoragem em dataset real. Executável diretamente
  (`python example_generate_adaface_embeddings.py`).

- **`conftest.py`.** Apenas insere a pasta no `sys.path` para que os módulos planos resolvam
  sob pytest (ver Seção 2).

- **`tests/`.** Suíte de validação numérica e de contrato, descrita na Seção 23.

---

## 4. Início rápido

O fluxo canônico de geração de um split é: **escolher um perfil → construir o simulador →
amostrar um dataset → persistir um bundle auditável**. O exemplo abaixo usa o perfil
cooperativo de PLDA (`make_cooperative_plda_config`), gera um split com identidades disjuntas
via `sample_dataset`, e grava o pacote completo (NPZ + metadata CSV + diagnósticos de norma +
manifesto JSON) com `save_bundle`.

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

# 1) Perfil pronto: baseline cooperativo de PLDA (dimensão 512).
config = make_cooperative_plda_config(dimension=512)

# 2) Construa o simulador com uma semente estrutural explícita.
#    A semente fixa população, anisotropia e vieses de domínio.
simulator = AdaFaceEmbeddingSimulator(config, random_seed=42)

# 3) Amostre um split com identidades disjuntas. A geração é
#    independente da ordem de chamada (RNG nomeado por split).
dataset = simulator.sample_dataset(
    identity_count=200,
    samples_per_identity=(4, 12),   # intervalo [min, max] por identidade
    split_name="train",
    identity_offset=0,
)

# 4) Inspecione os artefatos em memória.
print(dataset.raw_embeddings.shape)         # (n_amostras, 512)
print(dataset.normalized_embeddings.shape)  # (n_amostras, 512), normas == 1
print(dataset.metadata.columns.tolist())    # colunas de metadata (ver Seção 15)

# 5) Persista um bundle auditável em disco.
#    Gera: train_embeddings.npz, train_metadata.csv,
#          train_norm_diagnostics.csv, train_manifest.json
paths = dataset.save_bundle("saida/synthetic", "train")
print(paths)
```

### Anatomia do que acabou de ser gerado

- **`dataset.raw_embeddings`** — a matriz \( N \times D \) dos vetores brutos \( x = r\,u \),
  com a norma preservada (carrega o sinal radial).
- **`dataset.normalized_embeddings`** — a mesma matriz após L2-normalização, portanto apenas
  as direções \( u \in S^{D-1} \) (todas as linhas com norma 1).
- **`dataset.metadata`** — um `pandas.DataFrame` com uma linha por amostra e colunas
  totalmente auditáveis (identidade, índice de amostra, sessão, domínio de aquisição,
  qualidade latente, regime radial, norma, concentração intra-classe, modo de dispersão e
  diagnósticos de calota). O dicionário completo está na Seção 15.
- **`save_bundle`** — grava NPZ (arrays), metadata CSV, diagnósticos de norma (o padrão
  "U invertido" esperado entre quantis de norma) e um **manifesto JSON** com `profile_name`,
  as três sementes (`structural_seed`, `population_sampling_seed`,
  `observation_sampling_seed`), o hash SHA-256 do código-fonte do simulador e a configuração
  completa serializada. O manifesto é a peça central da **rastreabilidade**: ele permite
  reconstruir bit-a-bit a geração (ver Seções 11 e 18).

### Gerando um split de avaliação disjunto

Para um segundo split que **não compartilhe identidades** com o de treino, use o
`identity_offset` para deslocar os rótulos de identidade:

```python
eval_dataset = simulator.sample_dataset(
    identity_count=100,
    samples_per_identity=(4, 12),
    split_name="test",
    identity_offset=200,   # identidades 200..299, disjuntas de train (0..199)
)
```

### Ponteiros para os modos de uso principais

O início rápido cobre apenas o caminho cooperativo padrão. Os modos especializados — cada um
com sua própria seção — são:

- **Dispersão por calota esférica (`IDPERTURB_CAP` / `HYBRID_VMF_CAP`).** Em vez de vMF, use
  `make_idperturb_cap_config(...)` para um teto angular interpretável (cosseno-uniforme).
  Atenção à guarda de overlap, que pode elevar o `lb` efetivo. Detalhado na **Seção 6**.

- **Geração cross-condition (condições assimétricas pareadas).** Use
  `simulator.sample_cross_condition_datasets(...)` para produzir datasets que **compartilham
  as mesmas identidades** entre domínios (ex.: frontal vs lateral), garantindo pares genuínos
  cross-condition. Detalhado na **Seção 13**.

- **Ancoragem em dataset real (Modo A e Modo B / MoVMF).** Use `faceaudit_seed` para construir
  a população a partir de embeddings reais, mantendo as identidades-âncora **disjuntas** do
  conjunto de avaliação via `assert_disjoint_source_identities`. Detalhado na **Seção 14**.

- **Round-trip de configuração e reprodutibilidade.** Serialize qualquer configuração com
  `config_to_dict` / `config_from_dict` / `config_from_manifest` e reproduza uma geração a
  partir do manifesto. Detalhado nas **Seções 11 e 18**.

- **Integração no FaceAudit.** Converta o `SyntheticEmbeddingDataset` em um
  `EmbeddingDataset` real com `to_embedding_dataset` (adapter concreto). Detalhado na
  **Seção 19**.
## 5. Modelo generativo

O `adaface_embedding_simulator` não implementa a rede neural AdaFace. Ele implementa um **mundo sintético configurável** cuja saída imita as propriedades geométricas de embeddings faciais reais (direção na hiperesfera unitária × norma euclidiana), de modo que backends como cosseno, LDA, whitening, PLDA e alternativas esféricas possam ser estudados sob causas conhecidas e auditáveis. Cada embedding bruto é o produto de uma **norma** por uma **direção**:

\[
x \;=\; r \cdot u, \qquad r \in \mathbb{R}_{>0}, \quad u \in \mathbb{S}^{d-1},
\]

onde \( d \) é `SimulatorConfig.dimension` (padrão 512), \( r \) é a norma do embedding e \( u \) é a direção L2-normalizada. O simulador entrega **dois** arrays paralelos: `raw_embeddings` (\( x = r\,u \)) e `normalized_embeddings` (\( u \)). A coluna de metadata `embedding_norm` é exatamente \( r \); a validação do dataset confere \( \lVert x \rVert = r \) e \( \lVert u \rVert = 1 \).

O ponto central de auditabilidade é que **cinco camadas conceituais variam independentemente**, cada uma controlada por um bloco de configuração distinto. A norma \( r \) e a direção \( u \) são geradas por mecanismos separados, mas **acoplados de forma explícita e desligável** (Seção 9). Isto permite atribuir qualquer efeito observado num backend a uma camada específica do gerador.

| Camada | O que produz | Bloco de configuração |
|---|---|---|
| Mistura populacional | centros de identidade na hiperesfera | `PopulationConfig` |
| Centro angular da identidade | direção média \( \mu_i \) de cada identidade | `PopulationConfig.identity_center_kappas` |
| Domínio de aquisição | regime de condição (controlado/vigilância/etc.) | `AcquisitionDomainConfig` |
| Qualidade latente + regimes radiais | norma \( r \) por amostra | `RadialConfig` + Beta por domínio |
| Confiabilidade angular (dispersão intra-classe) | desvio de \( u \) em torno de \( \mu_i \) | `DirectionalConfig` (Seção 6) |

Esta seção descreve as quatro primeiras camadas (a quinta — dispersão intra-classe — é o capítulo central, Seção 6).

### 5.1 Mistura populacional de von Mises-Fisher

Os centros de identidade não são uniformes na hiperesfera: pertencem a uma **mistura de von Mises-Fisher (vMF)** com \( K \) componentes, espelhando a estrutura de subpopulações (etnia, gênero, faixa etária) que existe em galerias reais. Os parâmetros vivem em `PopulationConfig`:

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `component_count` | `int` | `8` | número \( K \) de componentes da mistura |
| `component_weights` | `ArrayLike \| None` | `None` | pesos \( w_k \) (probabilidade de uma identidade vir da componente \( k \)); `None` ⇒ uniforme \( 1/K \) |
| `identity_center_kappas` | `ArrayLike \| float` | `70.0` | concentração \( \kappa_k^{\mathrm{id}} \) dos centros de identidade dentro de cada componente (escalar ou um valor por componente) |
| `component_means` | `ArrayLike \| None` | `None` | médias \( m_k \) das componentes \( (K, d) \); se dadas, são L2-normalizadas e usadas diretamente |
| `component_mean_global_kappa` | `float` | `20.0` | concentração \( \kappa^{\mathrm{glob}} \) que dispersa as médias das componentes em torno de uma direção global |
| `global_mean_direction` | `ArrayLike \| None` | `None` | direção global \( g \in \mathbb{S}^{d-1} \); `None` ⇒ amostrada uniformemente |

A geração das médias das componentes \( m_k \) (em `_create_population_component_means`) segue:

- se `component_means` é fornecido, usa-se \( m_k \) L2-normalizado;
- senão, define-se a direção global \( g \) (de `global_mean_direction` ou amostrada uniformemente). Se \( \kappa^{\mathrm{glob}} \le 0 \), as \( K \) médias são uniformes na esfera; caso contrário, são amostradas de uma vMF concentrada em \( g \):

\[
m_k \;\sim\; \mathrm{vMF}\!\left(g,\; \kappa^{\mathrm{glob}}\right), \qquad k = 1,\dots,K.
\]

Um \( \kappa^{\mathrm{glob}} \) alto agrupa as subpopulações numa calota comum (galeria homogênea); um \( \kappa^{\mathrm{glob}} \) baixo as espalha (galeria heterogênea).

### 5.2 Centro angular da identidade

Cada identidade sintética \( i \) recebe primeiro uma **componente** \( c_i \in \{1,\dots,K\} \) amostrada segundo os pesos \( w_k \), e depois um **centro angular** \( \mu_i \in \mathbb{S}^{d-1} \) amostrado de uma vMF em torno da média da sua componente, com concentração \( \kappa_{c_i}^{\mathrm{id}} \) (em `_sample_identity_centers`):

\[
c_i \;\sim\; \mathrm{Categorical}(w_1,\dots,w_K), \qquad
\mu_i \;\sim\; \mathrm{vMF}\!\left(m_{c_i},\; \kappa_{c_i}^{\mathrm{id}}\right).
\]

O \( \mu_i \) é a **assinatura biométrica** da identidade — o alvo invariante em torno do qual todas as suas observações se dispersam. `identity_center_kappas` controla quão **separadas** as identidades de uma mesma subpopulação são entre si: \( \kappa^{\mathrm{id}} \) alto produz identidades vizinhas e difíceis (impostores próximos), \( \kappa^{\mathrm{id}} \) baixo produz identidades bem separadas. A amostragem vMF usada em todos esses passos é a função pública `sample_vmf` (amostrador por rejeição de Wood, 1994, com rotação de Householder vetorizada; \( \kappa = 0 \) recai em direção uniforme).

### 5.3 Domínio de aquisição

Cada observação é atribuída a um **domínio de aquisição** — uma condição de captura nomeada (ex.: `controlled`, `surveillance`). Os domínios são escolhidos por amostragem categórica segundo seus pesos `weight` (normalizados internamente), ou podem ser forçados via `forced_domain` em `sample_observations`. O domínio é o **interruptor mestre da heterocedasticidade**: ele modula simultaneamente a distribuição da qualidade latente, as probabilidades dos regimes radiais, a localização/escala da norma, a concentração angular, o ruído anisotrópico e os vieses. Os campos de `AcquisitionDomainConfig` relevantes ao modelo generativo da norma e da qualidade são:

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `name` | `str` | — | nome único do domínio |
| `weight` | `float` | `1.0` | peso de seleção (probabilidade relativa do domínio) |
| `quality_beta_alpha` | `float` | `4.0` | parâmetro \( \alpha \) da Beta de qualidade latente |
| `quality_beta_beta` | `float` | `2.0` | parâmetro \( \beta \) da Beta de qualidade latente |
| `central_regime_probability` | `float` | `0.86` | probabilidade base do regime radial central |
| `low_tail_probability` | `float` | `0.08` | probabilidade base do regime radial de cauda baixa |
| `high_tail_probability` | `float` | `0.06` | probabilidade base do regime radial de cauda alta |
| `low_tail_quality_logit_slope` | `float` | `2.0` | inclinação que aumenta a chance de cauda baixa em qualidade baixa |
| `high_tail_quality_logit_slope` | `float` | `0.0` | inclinação que aumenta a chance de cauda alta em qualidade baixa |
| `log_norm_shift` | `float` | `0.0` | deslocamento aditivo da log-norma média do domínio |
| `log_norm_sigma_multiplier` | `float` | `1.0` | multiplicador da escala da log-norma do domínio |
| `log_kappa_shift` | `float` | `0.0` | deslocamento aditivo de \( \log\kappa \) intra-classe do domínio (Seção 6) |

(Os demais campos — `anisotropic_noise_multiplier`, vieses e `cap_lower_bound_offset` — pertencem às Seções 6, 7 e 9.)

### 5.4 Qualidade latente (Beta por domínio)

Cada observação recebe uma **qualidade latente** \( q \in [0,1] \), amostrada de uma distribuição Beta cujos parâmetros dependem do domínio (em `_sample_latent_quality`):

\[
q \;\sim\; \mathrm{Beta}\!\left(\alpha_{\mathrm{dom}},\; \beta_{\mathrm{dom}}\right).
\]

A qualidade é a **variável causal oculta** que liga as três camadas seguintes: ela enviesa os regimes radiais, desloca a norma e modula a concentração angular. Um domínio controlado (ex.: \( \alpha = 5, \beta = 2 \)) gera qualidade tipicamente alta; um domínio de vigilância (ex.: \( \alpha = 2, \beta = 4{,}5 \)) gera qualidade tipicamente baixa. A qualidade é exportada como coluna de metadata `latent_quality` e é a porta de entrada de várias ablações causais (Seção 9).

### 5.5 Regimes radiais e logits modulados por qualidade

A norma de embeddings faciais reais exibe um padrão **não-monotônico** em relação à confiabilidade: faces nítidas e bem-pose tendem a normas centrais altas, mas tanto faces degradadas (baixa qualidade) quanto faces "saturadas" atípicas podem cair em caudas. O simulador modela isso com **três regimes radiais** — `central`, `low_tail`, `high_tail` — escolhidos por amostra (em `_sample_radial_regimes`).

Partindo das probabilidades base \( (p_{\mathrm{cen}}, p_{\mathrm{low}}, p_{\mathrm{high}}) \) do domínio, formam-se logits \( \ell = \log p \). Quando o acoplamento `quality_affects_regime_probabilities` está ligado, a qualidade enviesa as caudas (centrado em \( q = 0{,}5 \)):

\[
\ell_{\mathrm{low}} \mathrel{+}= s^{\mathrm{low}}_{\mathrm{dom}}\,(0{,}5 - q), \qquad
\ell_{\mathrm{high}} \mathrel{+}= s^{\mathrm{high}}_{\mathrm{dom}}\,(0{,}5 - q),
\]

onde \( s^{\mathrm{low}}_{\mathrm{dom}} \) e \( s^{\mathrm{high}}_{\mathrm{dom}} \) são `low_tail_quality_logit_slope` e `high_tail_quality_logit_slope`. Como \( (0{,}5 - q) > 0 \) para qualidade baixa, inclinações positivas **empurram amostras de baixa qualidade para as caudas**. Os logits são então normalizados via softmax estável:

\[
P(\mathrm{regime}=g) \;=\; \frac{\exp(\ell_g - \max_{g'} \ell_{g'})}{\sum_{g'} \exp(\ell_{g'} - \max_{g''} \ell_{g''})}.
\]

O regime escolhido é exportado em `radial_regime`.

### 5.6 Norma do embedding (log-normal por regime + domínio + qualidade)

Dado o regime \( g \), a **log-norma** é amostrada de uma normal cuja média e escala combinam quatro contribuições (em `_sample_log_norms`). A média:

\[
\mu_{\log r} \;=\; \underbrace{m_{\mathrm{ref}}}_{\text{referência}}
\;+\; \underbrace{\beta_q\,(q - 0{,}5)}_{\text{efeito de qualidade}}
\;+\; \underbrace{o_g}_{\text{offset de regime}}
\;+\; \underbrace{\delta^{\log r}_{\mathrm{dom}}}_{\text{shift de domínio}},
\]

onde \( m_{\mathrm{ref}} = \) `reference_log_norm_mean`, \( \beta_q = \) `quality_log_norm_slope` (zerado se `quality_affects_log_norm` estiver desligado), \( o_g \in \{ \) `central_offset`, `low_tail_offset`, `high_tail_offset` \( \} \) e \( \delta^{\log r}_{\mathrm{dom}} = \) `log_norm_shift`. A escala:

\[
\sigma_{\log r} \;=\; \sigma_g \cdot \rho_{\mathrm{dom}},
\]

com \( \sigma_g \in \{ \) `central_sigma`, `low_tail_sigma`, `high_tail_sigma` \( \} \) e \( \rho_{\mathrm{dom}} = \) `log_norm_sigma_multiplier`. A log-norma é \( \log r \sim \mathcal{N}(\mu_{\log r}, \sigma_{\log r}^2) \); a norma é \( r = e^{\log r} \), recortada em \( [\,\)`minimum_norm`\(,\,\)`maximum_norm`\(\,] \) (e a log-norma é recomputada após o clip para consistência). Os campos de `RadialConfig`:

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `reference_log_norm_mean` | `float` | `log(25.0)` | localização de referência \( m_{\mathrm{ref}} \) da log-norma |
| `reference_log_norm_scale` | `float` | `0.24` | escala de referência \( s_{\mathrm{ref}} \) da log-norma (usada na padronização, §5.7) |
| `reference_method` | `str` | `"configured_reference"` | rótulo da procedência da referência (exportado em `radial_reference_method`) |
| `quality_log_norm_slope` | `float` | `0.12` | inclinação \( \beta_q \) do efeito de qualidade na log-norma |
| `central_offset` | `float` | `0.0` | offset \( o_g \) do regime central |
| `central_sigma` | `float` | `0.10` | escala \( \sigma_g \) do regime central |
| `low_tail_offset` | `float` | `-0.70` | offset do regime de cauda baixa |
| `low_tail_sigma` | `float` | `0.16` | escala do regime de cauda baixa |
| `high_tail_offset` | `float` | `0.70` | offset do regime de cauda alta |
| `high_tail_sigma` | `float` | `0.16` | escala do regime de cauda alta |
| `minimum_norm` | `float` | `1e-6` | piso de recorte da norma |
| `maximum_norm` | `float` | `1e6` | teto de recorte da norma |

### 5.7 Padronização radial: `reference_scaled_log_norm_z`

Para que a posição radial de uma amostra seja **comparável entre domínios e datasets** (e para alimentar as penalidades não-monotônicas da concentração angular, §6), a log-norma é padronizada em torno de uma referência robusta:

\[
z \;=\; \frac{\log r - m_{\mathrm{ref}}}{s_{\mathrm{ref}}},
\]

com \( m_{\mathrm{ref}} = \) `reference_log_norm_mean` e \( s_{\mathrm{ref}} = \) `reference_log_norm_scale`. Esse \( z \) é exportado na coluna `reference_scaled_log_norm_z`, **também disponível como** `robust_norm_z` (alias com valor idêntico, presente para conveniência de consumidores que esperam esse nome). O valor \( z = 0 \) corresponde à norma de referência; \( z > 0 \) indica norma atipicamente alta e \( z < 0 \) atipicamente baixa.

#### A classe `RadialReference` (mediana + MAD escalado)

Quando a referência deve ser **estimada a partir de embeddings reais** (em vez de configurada), a classe `RadialReference` fornece um ajuste robusto. Ela guarda uma localização e uma escala da log-norma:

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `log_norm_location` | `float` | — | localização de referência \( m_{\mathrm{ref}} \) (mediana das log-normas) |
| `log_norm_scale` | `float` | — | escala de referência \( s_{\mathrm{ref}} \) (MAD escalado) |
| `method` | `str` | `"median_scaled_mad"` | rótulo do método de estimação |

O método de classe `RadialReference.fit(raw_embeddings_or_norms)` aceita uma matriz de embeddings \( (N,d) \) (das quais extrai as normas por linha) ou um vetor de normas, e estima:

\[
m_{\mathrm{ref}} = \mathrm{median}\big(\log r_n\big), \qquad
s_{\mathrm{ref}} = 1{,}4826 \cdot \mathrm{median}\big(\lvert \log r_n - m_{\mathrm{ref}} \rvert\big).
\]

O fator \( 1{,}4826 \) torna o MAD um estimador consistente do desvio-padrão sob normalidade. O ajuste falha (levanta exceção) se algum \( r_n \le 0 \) ou se o MAD escalado for nulo. A `RadialConfig` pode ser construída diretamente a partir de uma referência ajustada via `RadialConfig.from_reference(reference, **kwargs)`, que injeta `reference_log_norm_mean`, `reference_log_norm_scale` e `reference_method`. Este é o caminho recomendado para **calibrar a camada radial contra uma galeria real** (Seção 21), preservando a auditabilidade: o `method` registrado documenta a procedência da referência.

---

## 6. Dispersão intra-classe (VMF, IDPERTURB_CAP, HYBRID)

A camada de **dispersão intra-classe** decide como cada observação de uma identidade se afasta angularmente do centro \( \mu_i \). É o capítulo mais consequente do simulador: ela governa diretamente a **distância genuína** (cosseno entre duas amostras da mesma pessoa) e, portanto, o grau de sobreposição entre as distribuições genuína e impostora — o substrato exato sobre o qual PLDA, cosseno e calibração de razão de verossimilhança operam.

O simulador oferece **três modos** de dispersão, selecionados por `DirectionalConfig.intra_class_sampling_mode` (enum `IntraClassSamplingMode`): `VMF`, `IDPERTURB_CAP` e `HYBRID_VMF_CAP`. Os três produzem direções unitárias \( v \in \mathbb{S}^{d-1} \) em torno de \( \mu_i \), mas com **filosofias de controle diferentes**: um controla a *concentração* (cauda angular macia, sem piso), outro controla a *excursão máxima* (piso angular duro), e o terceiro combina os dois.

A direção \( v \) produzida aqui é a direção **pré-ruído**: as perturbações tangenciais de domínio/sessão/identidade (Seção 7) são aplicadas em seguida, sobre \( v \). A coluna de metadata `pre_nuisance_center_cosine` registra \( \langle v, \mu_i \rangle \) antes do ruído (invariante \( \ge \) `cap_lower_bound_effective` nos modos com calota); a coluna `own_center_cosine` registra o cosseno final, pós-ruído (que pode cair ligeiramente abaixo do piso por efeito da perturbação tangencial).

### 6.1 Modo VMF — concentração \( \kappa \)

No modo `VMF`, a direção é amostrada de uma von Mises-Fisher centrada em \( \mu_i \) com concentração \( \kappa \):

\[
v \;\sim\; \mathrm{vMF}(\mu_i,\; \kappa), \qquad
p(v) \;\propto\; \exp\!\big(\kappa\,\langle v, \mu_i\rangle\big).
\]

A amostragem é feita pela função pública `sample_vmf`. Não há **piso angular**: o cosseno \( \langle v, \mu_i \rangle \) tem uma cauda contínua que pode, em princípio, descer arbitrariamente perto de \( -1 \) (com probabilidade decrescente em \( \kappa \)). O \( \kappa \) por amostra é **modulado por covariáveis** (em `_calculate_intra_kappas`): qualidade, posição radial e domínio. Partindo do pico `peak_intra_kappa`:

\[
\log\kappa \;=\; \log(\kappa_{\mathrm{peak}})
\;+\; \underbrace{s_\kappa\,(q - 0{,}5)}_{\text{qualidade}}
\;+\; \underbrace{\delta^{\log\kappa}_{\mathrm{dom}}}_{\text{domínio}}
\;\underbrace{-\; \lambda_{\mathrm{low}}\,d_{\mathrm{low}}^2 - \lambda_{\mathrm{high}}\,d_{\mathrm{high}}^2}_{\text{penalidades radiais não-monotônicas}},
\]

onde \( s_\kappa = \) `quality_log_kappa_slope`, \( \delta^{\log\kappa}_{\mathrm{dom}} = \) `log_kappa_shift`, \( \lambda_{\mathrm{low}/\mathrm{high}} = \) `low_tail_quadratic_penalty`/`high_tail_quadratic_penalty`, e as distâncias radiais ao ponto ótimo são:

\[
d_{\mathrm{low}} = \max(0,\; z^\star - z), \qquad d_{\mathrm{high}} = \max(0,\; z - z^\star),
\]

com \( z = \) `reference_scaled_log_norm_z` (§5.7) e \( z^\star = \) `optimal_reference_scaled_log_norm_z`. O termo de qualidade é zerado se `quality_affects_intra_kappa` estiver desligado; o termo radial é zerado se `norm_z_affects_intra_kappa` estiver desligado. Por fim, \( \kappa = e^{\log\kappa} \) é recortado em \( [\,\)`minimum_intra_kappa`\(,\,\)`maximum_intra_kappa`\(\,] \). O \( \kappa \) efetivo é exportado em `intra_kappa`.

### 6.2 Modo IDPERTURB_CAP — calota esférica cosseno-uniforme

No modo `IDPERTURB_CAP`, em vez de uma concentração, controla-se diretamente a **excursão angular máxima** via um piso de cosseno \( lb \) (lower bound). Para cada amostra, sorteia-se um cosseno-alvo uniforme em \( [lb, 1] \) e constrói-se a direção (em `sample_spherical_cap`):

\[
s \;\sim\; \mathcal{U}[\,lb,\; 1\,], \qquad
v \;=\; s\,\mu_i \;+\; \sqrt{1 - s^2}\;u,
\]

onde \( u \in \mathbb{S}^{d-1} \) é uma direção tangente uniforme ortogonal a \( \mu_i \). Por construção \( \lVert v \rVert = 1 \) e \( \langle v, \mu_i \rangle = s \ge lb \) — ou seja, o piso é **garantido** na direção pré-ruído. O cosseno esperado é

\[
\mathbb{E}[\,\langle v, \mu_i\rangle\,] \;=\; \frac{1 + lb}{2},
\]

e \( lb = 1 \) colapsa \( v = \mu_i \) (sem perturbação). **Caveat essencial**: \( s \) é uniforme *em cosseno*, o que **não** é a distribuição uniforme sobre a área da calota em alta dimensão. Em \( \mathbb{S}^{d-1} \) com \( d \) grande, a maior parte da área de uma calota concentra-se perto da borda; uniformidade-em-cosseno coloca massa apreciável em cossenos altos que a uniformidade-em-área quase nunca visitaria. Esta é uma escolha deliberada de interpretabilidade (o parâmetro \( lb \) é a excursão máxima legível), não um modelo de difusão isotrópica na calota.

### 6.3 Modo HYBRID_VMF_CAP — corpo vMF com piso duro

O modo `HYBRID_VMF_CAP` combina o **corpo macio** da vMF com o **piso duro** da calota. Amostra-se \( v \sim \mathrm{vMF}(\mu_i, \kappa) \); qualquer linha cujo cosseno caia abaixo de \( lb \) é **re-sorteada** da vMF, até `hybrid_max_resample_rounds` vezes; as linhas que ainda violam o piso após todas as rodadas caem num **fallback de calota** via `sample_spherical_cap` no piso \( lb \), o que garante o piso. A função pública é `sample_hybrid_vmf_cap`, que retorna, além das amostras, dois diagnósticos por linha:

- `hybrid_resample_count` — número de re-sorteios vMF que cada amostra precisou;
- `hybrid_used_cap_fallback` — se a amostra acabou no fallback de calota.

A interpretação destes diagnósticos é central para a auditoria: uma **alta taxa de fallback** significa que o piso \( lb \) está cortando uma fração grande do corpo vMF, de modo que o resultado é **dominado pela calota** e não mais pela vMF — sinal de que \( \kappa \) e \( lb \) estão em desacordo (vMF larga demais sob piso apertado). Neste regime, a forma da cauda angular efetiva é a da calota, não a da vMF, e isso deve ser reportado.

### 6.4 Tabela comparativa dos três modos

| Aspecto | `VMF` | `IDPERTURB_CAP` | `HYBRID_VMF_CAP` |
|---|---|---|---|
| Parâmetro de controle | concentração \( \kappa \) | piso de cosseno \( lb \) | \( \kappa \) **+** \( lb \) |
| Função amostradora | `sample_vmf` | `sample_spherical_cap` | `sample_hybrid_vmf_cap` |
| Distribuição do cosseno | \( p(\cos)\propto e^{\kappa\cos}\,(1-\cos^2)^{(d-3)/2} \) | \( \cos = s \sim \mathcal{U}[lb,1] \) | corpo vMF truncado em \( lb \); calota no fallback |
| Cauda angular | macia, **sem piso** (cossenos baixos possíveis) | **piso duro** em \( lb \); cossenos altos sobre-representados | macia acima de \( lb \), **piso duro** em \( lb \) |
| Cosseno esperado | \( A_d(\kappa) \) (razão de Bessel) | \( (1+lb)/2 \) | entre os dois, conforme taxa de fallback |
| Interpretação do parâmetro | "quão concentrada" | "excursão máxima legível" | concentração com limite de segurança |
| `intra_kappa` usado? | sim | não (campo presente, mas não dirige a calota) | sim (dirige o corpo vMF) |
| `cap_lower_bound_effective` | `NaN` | preenchido | preenchido |
| Diagnósticos híbridos | `0` / `False` | `0` / `False` | `hybrid_resample_count` / `hybrid_used_cap_fallback` |
| Uso típico | baseline esférico; comparação cosseno/PLDA/PSDA | dispersão com garantia de separabilidade por design (estilo IDperturb) | vMF realista com salvaguarda anti-colisão dura |

### 6.5 Quando escolher cada modo

- **`VMF`** — quando se quer o modelo direcional canônico, sem piso artificial, e a métrica de interesse é a concentração. É o modo dos perfis de baseline e o que melhor casa com a teoria de PLDA esférico/PSDA. Use quando a pergunta for "como o backend se comporta sob uma cauda angular contínua".
- **`IDPERTURB_CAP`** — quando se quer **garantir por construção** uma excursão angular máxima e estudar o efeito de uma separabilidade imposta (inspirado no regime experimental IDperturb de Boutros et al., 2026). O parâmetro \( lb \) é interpretável e auditável como "cosseno mínimo intra-classe". Lembre-se do caveat de uniformidade-em-cosseno (§6.2): não é difusão isotrópica na calota.
- **`HYBRID_VMF_CAP`** — quando se quer o realismo da vMF **mais** uma salvaguarda dura contra amostras que colidem com outras identidades, monitorando via `hybrid_used_cap_fallback` se o piso está distorcendo o corpo. É a escolha mais conservadora forensicamente quando se teme que a cauda vMF gere genuínos artificialmente difíceis.

### 6.6 O piso \( lb \) e sua modulação

Nos modos com calota (`IDPERTURB_CAP` e `HYBRID_VMF_CAP`), o piso \( lb \) **não é uma constante**: ele é modulado por amostra pelas mesmas covariáveis que modulam \( \kappa \), espelhando `_calculate_intra_kappas`. O cálculo (em `_calculate_cap_lower_bounds`) opera **em escala de cosseno** (não de \( \log\kappa \)):

\[
lb \;=\; lb_{\mathrm{base}}
\;+\; \underbrace{\Delta^{lb}_{\mathrm{dom}}}_{\text{offset de domínio}}
\;+\; \underbrace{s_{lb}\,(q - 0{,}5)}_{\text{qualidade}}
\;\underbrace{-\; \pi_{\mathrm{low}}\,d_{\mathrm{low}}^2 - \pi_{\mathrm{high}}\,d_{\mathrm{high}}^2}_{\text{penalidades radiais não-monotônicas}},
\]

onde \( lb_{\mathrm{base}} = \) `cap_base_lower_bound`, \( \Delta^{lb}_{\mathrm{dom}} = \) `cap_lower_bound_offset` (campo de `AcquisitionDomainConfig`, de qualquer sinal: positivo aperta a identidade — condição controlada; negativo afrouxa — vigilância), \( s_{lb} = \) `cap_quality_lower_bound_slope`, e \( \pi_{\mathrm{low}/\mathrm{high}} = \) `cap_low_tail_penalty`/`cap_high_tail_penalty`. As distâncias radiais \( d_{\mathrm{low}}, d_{\mathrm{high}} \) e o ponto ótimo \( z^\star \) são os mesmos da §6.1.

Crucialmente, **os mesmos switches de `CouplingConfig`** que governam \( \kappa \) governam \( lb \), apesar do nome historicamente centrado em kappa: o termo de qualidade é desligado por `quality_affects_intra_kappa` e o termo radial por `norm_z_affects_intra_kappa`. Nos modos de calota, esses switches portam-se como "a qualidade/posição radial afeta a dispersão intra-classe". As penalidades agem **na escala de cosseno** — uma penalidade radial subtrai diretamente do cosseno-piso, não de \( \log\kappa \). Após a soma, \( lb \) é recortado em \( [\,\)`cap_minimum_lower_bound`\(,\,\)`cap_maximum_lower_bound`\(\,] \).

Os campos de `DirectionalConfig` que controlam a calota:

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `intra_class_sampling_mode` | `IntraClassSamplingMode \| str` | `VMF` | modo de dispersão intra-classe |
| `cap_base_lower_bound` | `float` | `0.6` | piso de cosseno base \( lb_{\mathrm{base}} \) |
| `cap_quality_lower_bound_slope` | `float` | `0.10` | inclinação \( s_{lb} \) do efeito de qualidade no piso (qualquer sinal) |
| `cap_low_tail_penalty` | `float` | `0.0` | penalidade quadrática \( \pi_{\mathrm{low}} \) de cauda baixa, em escala de cosseno |
| `cap_high_tail_penalty` | `float` | `0.0` | penalidade quadrática \( \pi_{\mathrm{high}} \) de cauda alta, em escala de cosseno |
| `cap_minimum_lower_bound` | `float` | `0.0` | piso de recorte do \( lb \) nominal |
| `cap_maximum_lower_bound` | `float` | `0.999` | teto de recorte do \( lb \) **nominal** (não do efetivo, §6.7) |
| `cap_overlap_guard` | `bool` | `True` | ativa a regra do bissetor (§6.7) |
| `hybrid_max_resample_rounds` | `int` | `8` | rodadas máximas de re-sorteio vMF no modo híbrido |

### 6.7 A regra do bissetor (`cap_overlap_guard`) e o caveat do piso efetivo

Mesmo com um \( lb \) nominal modulado, identidades em **regiões densas da população** (centros mutuamente próximos) podem gerar perturbações que ultrapassam o ponto médio angular até a identidade vizinha mais próxima — criando, por construção, genuínos que estão mais perto de outra pessoa do que da própria. Para impedir isso, a **regra do bissetor** (`cap_overlap_guard=True`, inspirada em Boutros et al., 2026, Eq. 6) impõe um **piso por identidade**: o cosseno do semi-ângulo até o centro mais próximo. Para a identidade \( i \), com vizinho mais próximo dado por \( \max_{j\neq i} \langle \mu_i, \mu_j \rangle \):

\[
\mathrm{floor}_i \;=\; \sqrt{\frac{1 + \max_{j \neq i}\, \langle \mu_i, \mu_j \rangle}{2}} \;=\; \cos\!\left(\frac{\text{ângulo ao vizinho mais próximo}}{2}\right).
\]

Este piso é calculado em blocos de linhas (em `_overlap_lower_bound_floor`), sem materializar a matriz \( C \times C \) inteira (custo \( O(C^2 d) \) em tempo, \( O(\text{bloco}\times C) \) em memória); identidades singleton recebem piso \( 0 \). O piso é **cacheado** na população para reuso entre observações cross-condition.

**Caveat crítico de auditoria**: a regra do bissetor é aplicada **após** o recorte do \( lb \) nominal e **tem prioridade** sobre ele:

\[
lb_{\mathrm{efetivo},\,n} \;=\; \min\!\Big(1,\; \max\big(lb_{\mathrm{nominal},\,n},\; \mathrm{floor}_{i(n)}\big)\Big).
\]

Como a anti-colisão tem prioridade, o **\( lb \) efetivo pode exceder `cap_maximum_lower_bound`**. Em outras palavras, o `cap_base_lower_bound`/`cap_maximum_lower_bound` nominais são um **piso de pedido, não o valor realizado**. Por isso, ao reportar uma configuração, **nunca** se deve apresentar apenas o \( lb \) nominal: deve-se reportar percentis empíricos da coluna `cap_lower_bound_effective`, que é a única descrição fiel da excursão angular efetivamente imposta a cada amostra. Em populações densas, o piso efetivo pode estar consistentemente acima do nominal para muitas identidades, e isso muda materialmente a dificuldade dos pares genuínos.

Quando `cap_overlap_guard=False`, o piso do bissetor não é aplicado e \( lb_{\mathrm{efetivo}} = \min(1, lb_{\mathrm{nominal}}) \) — útil para ablações que querem isolar o efeito da modulação por covariáveis sem a salvaguarda anti-colisão, mas com o risco de gerar genuínos que colidem com identidades vizinhas.
## 7. Perturbações tangenciais e atualização

Depois que a dispersão intra-classe produz uma direção \(\mathbf{u}_{ij}^{(0)}\) em torno do centro de identidade \(\mathbf{c}_i\) (Seção 6), o simulador adiciona uma **perturbação estruturada** que modela ruído de aquisição, viés de domínio e efeitos de sessão. Essa perturbação é montada em coordenadas ambientais, projetada no plano tangente da hiperesfera e mapeada de volta para a superfície unitária. Todo o cálculo vive em `AdaFaceEmbeddingSimulator._apply_anisotropic_nuisance`.

O passo bruto somado é

\[
\boldsymbol{\delta}_{ij}
=
\boldsymbol{\delta}_{ij}^{\text{aniso}}
+
\mathbf{b}_d
+
\mathbf{b}_{d,i}
+
\mathbf{b}_{d,s},
\]

com quatro termos de natureza distinta:

| Termo | Coluna de diagnóstico | Interpretação forense |
|---|---|---|
| \(\boldsymbol{\delta}_{ij}^{\text{aniso}}\) | `anisotropic_noise_norm` | nuisance anisotrópico de baixo posto (ruído estruturado, dependente de qualidade) |
| \(\mathbf{b}_d\) | `fixed_domain_bias_norm` | deslocamento médio **fixo** do domínio \(d\) (mesmo para todas as identidades) |
| \(\mathbf{b}_{d,i}\) | `domain_identity_bias_norm` | interação **domínio × identidade** (deslocamento próprio de cada identidade naquele domínio) |
| \(\mathbf{b}_{d,s}\) | `session_bias_norm` | deslocamento de **sessão** (compartilhado por todas as observações da mesma tripla identidade × domínio × sessão) |

Cada termo é auditável individualmente pela sua coluna de norma na metadata, e o passo tangencial efetivo é reportado em `combined_tangent_step_norm` (Seção 7.4). Essa decomposição é deliberada: em laudo é possível afirmar qual parcela do deslocamento angular vem de ruído puro, de viés sistemático de domínio, de idiossincrasia identidade-em-domínio ou de efeito de sessão.

### 7.1 Nuisance anisotrópico — base de baixo posto

O nuisance anisotrópico \(\boldsymbol{\delta}_{ij}^{\text{aniso}}\) não é ruído isotrópico: ele vive num **subespaço de baixo posto** \(r\), modelando o fato empírico de que erros de aquisição concentram-se em poucas direções estruturadas (iluminação, pose residual, compressão). Uma base ortonormal \(\mathbf{B} \in \mathbb{R}^{D \times r}\) é amostrada uma única vez no construtor (via decomposição QR de uma matriz Gaussiana, em `_orthonormal_basis`) e fixada na seed estrutural. A cada observação, coeficientes Gaussianos \(\mathbf{z} \sim \mathcal{N}(0, I_r)\) são projetados pela base:

\[
\boldsymbol{\delta}^{\text{base}} = \mathbf{z}\,\mathbf{B}^{\top}.
\]

O posto \(r\) é controlado por `DirectionalConfig.anisotropic_rank` (deve satisfazer \(0 \le r < D\)); a magnitude base por `anisotropic_base_scale`. Se qualquer um dos dois for zero, **nenhuma** base é construída (`anisotropic_basis is None`) e o nuisance anisotrópico é integralmente desligado.

#### Escala dependente de qualidade e domínio

A magnitude de cada amostra é modulada por qualidade e domínio antes da projeção tangencial:

\[
s_{ij}
=
\underbrace{\texttt{anisotropic\_base\_scale}}_{\text{base}}
\cdot
\underbrace{m_{d}}_{\substack{\texttt{anisotropic\_}\\\texttt{noise\_multiplier}}}
\cdot
\Bigl(
1 + \texttt{anisotropic\_quality\_slope}\cdot(1 - q_{ij})
\Bigr),
\]

onde \(q_{ij} \in [0,1]\) é a qualidade latente e \(m_d\) é o multiplicador por domínio. **Quanto menor a qualidade, maior o nuisance** — o fator \((1 - q_{ij})\) faz a perturbação crescer em imagens ruins, reproduzindo a degradação de confiabilidade angular sob baixa qualidade. Domínios de vigilância recebem `anisotropic_noise_multiplier > 1` para ampliar esse efeito.

### 7.2 Os três modos de estrutura do nuisance (`NuisanceStructureMode`)

O atributo `DirectionalConfig.nuisance_structure_mode` (enum `NuisanceStructureMode`) decide **como o subespaço de nuisance é compartilhado** entre amostras. Há três modos:

| Modo | Valor | Subespaços ativos | Quando usar |
|---|---|---|---|
| `GLOBAL_SHARED` | `"global_shared"` | uma única base global, comum a todos os domínios | um cenário homogêneo; viés de nuisance correlacionado entre condições |
| `PER_DOMAIN` | `"per_domain"` | uma base **independente por domínio** (sem base global) | cada condição de aquisição tem suas próprias direções de ruído, descorrelacionadas |
| `HIERARCHICAL` | `"hierarchical"` | base global **mais** base por domínio, com pesos | parte do nuisance é compartilhada entre condições, parte é específica do domínio |

A combinação concreta segue a lógica de `_apply_anisotropic_nuisance`:

- **Componente global** (ativa em `GLOBAL_SHARED` e `HIERARCHICAL`):
  \[
  \boldsymbol{\delta}^{\text{global}}_{ij}
  =
  w_{\text{global}}\,\bigl(\mathbf{z}_{ij}\,\mathbf{B}_{\text{global}}^{\top}\bigr),
  \qquad
  w_{\text{global}} = \texttt{global\_nuisance\_weight}.
  \]
- **Componente por domínio** (ativa em `PER_DOMAIN` e `HIERARCHICAL`): para cada domínio \(d\) com base própria \(\mathbf{B}_d\),
  \[
  \boldsymbol{\delta}^{\text{domínio}}_{ij}
  =
  w_d\,\bigl(\mathbf{z}_{ij}\,\mathbf{B}_d^{\top}\bigr),
  \qquad
  w_d =
  \begin{cases}
  \texttt{domain\_nuisance\_weight}, & \text{modo } \mathtt{HIERARCHICAL},\\
  1, & \text{modo } \mathtt{PER\_DOMAIN}.
  \end{cases}
  \]

Em `HIERARCHICAL` os dois pesos coexistem: `global_nuisance_weight` pondera a parcela comum e `domain_nuisance_weight` a parcela específica, permitindo controlar exatamente quanto do ruído é correlacionado entre condições (importante para estressar backends PLDA cross-condition).

### 7.3 Nuisance específico de identidade (`identity_specific_*`)

Independentemente do modo acima, é possível adicionar um subespaço de nuisance **próprio de cada identidade**. Para cada identidade da população, uma base ortonormal de posto `identity_specific_nuisance_rank` é amostrada (em `_sample_identity_specific_bases`) e armazenada na `SyntheticIdentityPopulation`. A contribuição é

\[
\boldsymbol{\delta}^{\text{id}}_{ij}
=
\texttt{identity\_specific\_nuisance\_scale}\cdot
\bigl(\mathbf{B}_{i}\,\mathbf{z}_{ij}\bigr),
\]

aplicada via `np.einsum("ndr,nr->nd", ...)` linha a linha, selecionando a base \(\mathbf{B}_i\) da identidade correta. Esse termo modela variação angular intrínseca e idiossincrática de cada sujeito (por exemplo, um traço facial que sistematicamente desloca os embeddings daquela pessoa numa direção própria). Como as bases pertencem à população, elas são **compartilhadas entre condições cross-condition** — a mesma identidade carrega a mesma idiossincrasia em todos os domínios. Se o posto ou a escala forem zero, o termo é integralmente desligado (`identity_specific_nuisance_bases is None`).

A soma \(\boldsymbol{\delta}^{\text{global}} + \boldsymbol{\delta}^{\text{domínio}} + \boldsymbol{\delta}^{\text{id}}\) é então multiplicada, linha a linha, pela escala \(s_{ij}\) da Seção 7.1, produzindo o nuisance anisotrópico final \(\boldsymbol{\delta}_{ij}^{\text{aniso}}\) reportado em `anisotropic_noise_norm`.

### 7.4 Viés de domínio, domínio × identidade e de sessão

Aos termos estocásticos somam-se três **vieses determinísticos** (na seed) que modelam deslocamentos sistemáticos:

| Viés | Como é gerado | Escala | Escopo de compartilhamento |
|---|---|---|---|
| \(\mathbf{b}_d\) — domínio fixo | direção uniforme amostrada uma vez no construtor (`_create_domain_bias_vectors`) | `AcquisitionDomainConfig.domain_bias_scale` | **idêntico** para todas as observações daquele domínio |
| \(\mathbf{b}_{d,i}\) — domínio × identidade | direção uniforme por identidade dentro do domínio (em `sample_identity_population`) | `AcquisitionDomainConfig.domain_identity_bias_scale` | próprio de cada par (domínio, identidade), constante entre observações dessa identidade no domínio |
| \(\mathbf{b}_{d,s}\) — sessão | direção uniforme por tripla (identidade, domínio, sessão), em `_sample_session_bias_matrix` | `AcquisitionDomainConfig.session_bias_scale` | compartilhado por todas as observações da mesma sessão |

O viés fixo de domínio \(\mathbf{b}_d\) é a fonte canônica de **mismatch sistemático entre condições** num estudo cross-condition: ele desloca *toda* a nuvem de embeddings de um domínio numa direção comum, exatamente o tipo de viés que um PLDA two-covariance cross-condition deve absorver. O viés de sessão modela correlação dentro de uma mesma captura (várias imagens de uma sessão compartilham um deslocamento), o que infla a similaridade genuína intra-sessão de forma realista. Todos os três têm escala padrão `0.0` (desligados) e qualquer escala negativa é rejeitada na validação.

### 7.5 Projeção tangente e atualização: retração vs. mapa exponencial

O passo bruto \(\boldsymbol{\delta}_{ij}\) é primeiro **projetado no plano tangente** de um vetor de referência \(\mathbf{t}\), removendo sua componente radial:

\[
\boldsymbol{\delta}_{ij}^{\top}
=
\boldsymbol{\delta}_{ij}
-
\bigl(\boldsymbol{\delta}_{ij}\cdot \mathbf{t}\bigr)\,\mathbf{t}.
\]

A escolha do vetor de referência \(\mathbf{t}\) e da fórmula de retorno à hiperesfera é governada por `DirectionalConfig.tangent_update_mode` (enum `TangentUpdateMode`), e essa escolha tem uma **diferença geométrica concreta** — ela muda inclusive qual ponto define o plano tangente:

| Modo | Valor | Plano tangente em | Retorno à esfera |
|---|---|---|---|
| `RETRACTION` | `"retraction"` | **centro de identidade** \(\mathbf{c}_i\) | soma e renormalização |
| `EXPONENTIAL_MAP` | `"exponential_map"` | **direção atual** \(\mathbf{u}_{ij}^{(0)}\) | deslocamento geodésico exato |

#### Retração

No modo `RETRACTION`, o passo é projetado no tangente do **centro de identidade** \(\mathbf{c}_i\), somado à direção corrente e renormalizado:

\[
\widetilde{\mathbf{u}}_{ij}
=
\frac{
\mathbf{u}_{ij}^{(0)} + \boldsymbol{\delta}_{ij}^{\top}
}{
\bigl\lVert \mathbf{u}_{ij}^{(0)} + \boldsymbol{\delta}_{ij}^{\top}\bigr\rVert_2
}.
\]

É uma aproximação de primeira ordem, rápida e adequada quando os passos são pequenos. Como o tangente é definido em \(\mathbf{c}_i\), o deslocamento é coerente com a geometria da identidade, mas a renormalização introduz uma leve compressão não-geodésica para passos grandes.

#### Mapa exponencial

No modo `EXPONENTIAL_MAP`, o passo é projetado no tangente da **direção corrente** \(\mathbf{u}_{ij}^{(0)}\) e mapeado de volta pelo mapa exponencial Riemanniano exato da esfera:

\[
\widetilde{\mathbf{u}}_{ij}
=
\operatorname{Exp}_{\mathbf{u}_{ij}^{(0)}}\!\bigl(\boldsymbol{\delta}_{ij}^{\top}\bigr)
=
\cos\!\bigl(\lVert\boldsymbol{\delta}_{ij}^{\top}\rVert\bigr)\,\mathbf{u}_{ij}^{(0)}
+
\sin\!\bigl(\lVert\boldsymbol{\delta}_{ij}^{\top}\rVert\bigr)\,
\frac{\boldsymbol{\delta}_{ij}^{\top}}{\lVert\boldsymbol{\delta}_{ij}^{\top}\rVert}.
\]

Esse é o deslocamento geodésico exato: a norma do passo tangente vira, literalmente, o **ângulo geodésico** percorrido na esfera, preservando comprimento de arco. Passos de norma \(\le 10^{-12}\) são tratados de forma degenerada (a direção permanece inalterada). Recomenda-se o mapa exponencial quando os passos deixam de ser desprezíveis — em perfis com nuisance forte ou vieses grandes (por exemplo, o perfil de estresse cross-condition), o mapa exponencial preserva melhor a métrica angular do que a retração.

> **Diferença geométrica resumida.** A retração mede o deslocamento no plano tangente do *centro de identidade* e volta por soma+renormalização (aproximação de 1ª ordem); o mapa exponencial mede no plano tangente da *direção atual* e volta pelo arco geodésico exato (a norma do passo é o ângulo percorrido). Para passos pequenos os dois coincidem em 1ª ordem; para passos grandes divergem, e só o mapa exponencial preserva comprimento de arco.

### 7.6 Coluna `combined_tangent_step_norm`

Em ambos os modos, o simulador registra a norma do passo tangencial efetivo, \(\lVert\boldsymbol{\delta}_{ij}^{\top}\rVert\), na coluna de metadata **`combined_tangent_step_norm`**. Essa é a quantidade auditável central desta etapa:

- Em `EXPONENTIAL_MAP`, é **exatamente** o ângulo geodésico (em radianos) que a perturbação afastou a observação da sua direção pré-nuisance.
- Em `RETRACTION`, é o comprimento do passo tangente antes da renormalização (aproxima o ângulo para passos pequenos).

Ela agrega os quatro termos (anisotrópico + domínio + domínio×identidade + sessão) *após* a projeção tangente, então funciona como medida única do quanto o nuisance/viés total moveu cada amostra. O diagnóstico de bins de norma (`norm_bin_diagnostics`) reporta a média dessa coluna por quantil de norma sob o nome `tangent_step_norm_mean`, permitindo verificar se o passo tangencial cresce nas caudas de norma (como esperado quando a qualidade cai).

> **Interação com a calota (modos de cap).** O piso de cosseno `lb` dos modos `IDPERTURB_CAP` e `HYBRID_VMF_CAP` é garantido sobre a direção **pré-nuisance** \(\mathbf{u}_{ij}^{(0)}\). O passo tangencial desta seção pode empurrar o cosseno realizado (pós-nuisance) ligeiramente abaixo de `lb`. Para auditar a calota, use sempre a coluna invariante `pre_nuisance_center_cosine` (que satisfaz \(\ge\) `cap_lower_bound_effective`), nunca `own_center_cosine`.

---

## 8. Referência completa da configuração

Esta seção tabula **todos** os campos de **todas** as dataclasses de configuração. Toda configuração é uma `@dataclass(frozen=True)` (imutável; clone com `dataclasses.replace`). A árvore de configuração é:

```
SimulatorConfig
├── population: PopulationConfig
├── radial: RadialConfig            (referência via RadialReference)
├── directional: DirectionalConfig
├── coupling: CouplingConfig
└── acquisition_domains: tuple[AcquisitionDomainConfig, ...]
```

Todo objeto expõe `validate()`, chamado automaticamente por `SimulatorConfig.validate()` (que o construtor do simulador invoca). Enums podem ser passados como membro do enum **ou** como seu valor `str` — ambos são aceitos pelas DTOs e pelo simulador.

### 8.1 `SimulatorConfig`

Configuração de topo do gerador.

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dimension` | `int` | `512` | dimensionalidade \(D\) do embedding; deve ser \(\ge 3\) |
| `profile_name` | `str` | `"custom"` | rótulo do perfil, gravado no manifesto; deve ser não-vazio |
| `population` | `PopulationConfig` | `PopulationConfig()` | distribuição dos centros de identidade |
| `radial` | `RadialConfig` | `RadialConfig()` | modelo de norma não-monotônico |
| `directional` | `DirectionalConfig` | `DirectionalConfig()` | confiabilidade angular + nuisance + modos |
| `coupling` | `CouplingConfig` | `CouplingConfig()` | switches de acoplamento causal |
| `acquisition_domains` | `tuple[AcquisitionDomainConfig, ...]` | `(AcquisitionDomainConfig(name="default"),)` | domínios de aquisição; nomes devem ser únicos e não-vazios; pelo menos um |

### 8.2 `PopulationConfig`

Distribuição dos centros de identidade sintéticos na hiperesfera unitária (uma mistura vMF de componentes).

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `component_count` | `int` | `8` | número de componentes (sub-populações) da mistura; deve ser positivo |
| `component_weights` | `ArrayLike \| None` | `None` | pesos da mistura (normalizados internamente); `None` ⇒ uniforme; se dado, um valor por componente |
| `identity_center_kappas` | `ArrayLike \| float` | `70.0` | concentração vMF dos centros em torno da média do componente; escalar ou um valor por componente; finito e \(\ge 0\) |
| `component_means` | `ArrayLike \| None` | `None` | médias dos componentes \((C, D)\) fornecidas explicitamente; `None` ⇒ amostradas |
| `component_mean_global_kappa` | `float` | `20.0` | concentração das médias dos componentes em torno de uma média global; \(\ge 0\); `0` ⇒ médias uniformes na esfera |
| `global_mean_direction` | `ArrayLike \| None` | `None` | direção global em torno da qual os componentes se concentram; `None` ⇒ amostrada uniformemente |

### 8.3 `RadialReference`

Localização e escala de referência para normas logarítmicas padronizadas. Usada para padronizar \(\log\lVert\mathbf{x}\rVert\) em z-scores robustos. Frozen.

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `log_norm_location` | `float` | — (obrigatório) | locação de referência da log-norma (tipicamente a mediana) |
| `log_norm_scale` | `float` | — (obrigatório) | escala de referência da log-norma (tipicamente MAD escalado por 1.4826) |
| `method` | `str` | `"median_scaled_mad"` | rótulo do método de estimativa, propagado para a metadata |

O classmethod `RadialReference.fit(raw_embeddings_or_norms)` estima uma referência **robusta** a partir de embeddings reais (ou de um vetor de normas): calcula a mediana e o MAD escalado das log-normas. O z-score padronizado resultante é gravado na coluna `reference_scaled_log_norm_z` (também disponível como `robust_norm_z`, alias da mesma coluna).

### 8.4 `RadialConfig`

Modelo de norma não-monotônico com três regimes (central, cauda baixa, cauda alta), reproduzindo a relação em U invertido entre norma e confiabilidade dos embeddings AdaFace.

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `reference_log_norm_mean` | `float` | `log(25.0)` | média de referência da log-norma central |
| `reference_log_norm_scale` | `float` | `0.24` | escala de referência da log-norma (denominador do z); deve ser positiva |
| `reference_method` | `str` | `"configured_reference"` | rótulo do método de referência, gravado em `radial_reference_method` |
| `quality_log_norm_slope` | `float` | `0.12` | inclinação do efeito de qualidade sobre a log-norma |
| `central_offset` | `float` | `0.0` | deslocamento da log-norma no regime central |
| `central_sigma` | `float` | `0.10` | desvio-padrão da log-norma no regime central; positivo |
| `low_tail_offset` | `float` | `-0.70` | deslocamento da log-norma na cauda baixa (normas pequenas) |
| `low_tail_sigma` | `float` | `0.16` | desvio-padrão na cauda baixa; positivo |
| `high_tail_offset` | `float` | `0.70` | deslocamento da log-norma na cauda alta (normas grandes) |
| `high_tail_sigma` | `float` | `0.16` | desvio-padrão na cauda alta; positivo |
| `minimum_norm` | `float` | `1e-6` | piso físico da norma (clipping) |
| `maximum_norm` | `float` | `1e6` | teto físico da norma; deve satisfazer \(0 < \texttt{minimum} < \texttt{maximum}\) |

O classmethod `RadialConfig.from_reference(reference, **kwargs)` constrói um `RadialConfig` a partir de uma `RadialReference` ajustada em dados reais, copiando locação, escala e método e aceitando overrides dos demais campos via `kwargs`.

### 8.5 `DirectionalConfig`

Modelo de confiabilidade angular (κ intra-classe), modelo opcional de nuisance tangencial e seleção dos modos de dispersão/atualização. É a dataclass mais rica. Os campos agrupam-se em quatro blocos.

#### Bloco A — confiabilidade angular (κ intra-classe)

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `peak_intra_kappa` | `float` | `1400.0` | κ de pico (máxima concentração angular, na qualidade/norma ótimas); positivo |
| `quality_log_kappa_slope` | `float` | `0.90` | inclinação do efeito de qualidade sobre \(\log\kappa\) |
| `optimal_reference_scaled_log_norm_z` | `float` | `0.0` | z de norma padronizada onde κ é máximo (ápice do U invertido) |
| `low_tail_quadratic_penalty` | `float` | `0.55` | penalidade quadrática de \(\log\kappa\) para z **abaixo** do ótimo; \(\ge 0\) |
| `high_tail_quadratic_penalty` | `float` | `0.75` | penalidade quadrática de \(\log\kappa\) para z **acima** do ótimo; \(\ge 0\) |
| `minimum_intra_kappa` | `float` | `20.0` | piso de κ (clipping); positivo |
| `maximum_intra_kappa` | `float` | `10000.0` | teto de κ (clipping); \(\ge\) `minimum_intra_kappa` |

A property `optimal_robust_norm_z` é alias de `optimal_reference_scaled_log_norm_z` (mesmo valor, leitura apenas).

#### Bloco B — nuisance anisotrópico e estrutura

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `anisotropic_rank` | `int` | `0` | posto \(r\) da base de nuisance global/por-domínio; \(0 \le r < D\); `0` desliga |
| `anisotropic_base_scale` | `float` | `0.0` | magnitude base do nuisance anisotrópico; \(\ge 0\); `0` desliga |
| `anisotropic_quality_slope` | `float` | `1.0` | inclinação do crescimento do nuisance com a baixa qualidade; \(\ge 0\) |
| `nuisance_structure_mode` | `NuisanceStructureMode \| str` | `GLOBAL_SHARED` | compartilhamento da base: `global_shared` / `per_domain` / `hierarchical` |
| `global_nuisance_weight` | `float` | `1.0` | peso da base global (modos `global_shared`/`hierarchical`); \(\ge 0\) |
| `domain_nuisance_weight` | `float` | `0.0` | peso da base por domínio em `hierarchical`; \(\ge 0\) |
| `identity_specific_nuisance_rank` | `int` | `0` | posto da base de nuisance **própria de cada identidade**; \(0 \le r < D\); `0` desliga |
| `identity_specific_nuisance_scale` | `float` | `0.0` | magnitude do nuisance específico de identidade; \(\ge 0\); `0` desliga |
| `tangent_update_mode` | `TangentUpdateMode \| str` | `RETRACTION` | retorno à hiperesfera: `retraction` / `exponential_map` |

#### Bloco C — modo de dispersão intra-classe e parâmetros da calota

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `intra_class_sampling_mode` | `IntraClassSamplingMode \| str` | `VMF` | dispersão intra-classe: `vmf` / `idperturb_cap` / `hybrid_vmf_cap` |
| `cap_base_lower_bound` | `float` | `0.6` | piso de cosseno base `lb` da calota (modos de cap); \([0,1]\) |
| `cap_quality_lower_bound_slope` | `float` | `0.10` | inclinação do efeito de qualidade sobre `lb` (qualquer sinal; maior qualidade ⇒ maior `lb`) |
| `cap_low_tail_penalty` | `float` | `0.0` | penalidade não-monotônica sobre `lb` na cauda baixa de norma; \(\ge 0\) |
| `cap_high_tail_penalty` | `float` | `0.0` | penalidade não-monotônica sobre `lb` na cauda alta de norma; \(\ge 0\) |
| `cap_minimum_lower_bound` | `float` | `0.0` | piso de clipping de `lb`; \([0,1]\); \(\le\) `cap_maximum_lower_bound` |
| `cap_maximum_lower_bound` | `float` | `0.999` | teto de clipping de `lb`; \([0,1]\) (o overlap guard PODE excedê-lo) |
| `cap_overlap_guard` | `bool` | `True` | aplica o piso da bissetriz por identidade (anti-colisão), com **prioridade** sobre o clipping |
| `hybrid_max_resample_rounds` | `int` | `8` | número de redraws vMF no modo `hybrid_vmf_cap` antes do fallback de calota; \(\ge 1\) |

> As penalidades de calota (`cap_low_tail_penalty`, `cap_high_tail_penalty`) e a inclinação de qualidade atuam na **escala de cosseno**, não em \(\log\kappa\). O overlap guard tem prioridade sobre o teto: o `lb` *efetivo* pode exceder `cap_maximum_lower_bound`, por isso sempre se reporta percentis de `cap_lower_bound_effective` (Seção 6/16) — o `lb` nominal é um piso, não o valor realizado.

A modulação de κ (Bloco A) e de `lb` (Bloco C) é controlada pelos switches de qualidade/norma do `CouplingConfig` (Seção 9). Nos modos de cap, os switches `quality_affects_intra_kappa` e `norm_z_affects_intra_kappa` continuam sendo os que liberam, respectivamente, o efeito de qualidade e o efeito de norma sobre `lb` (apesar do nome centrado em κ, eles regem a dispersão intra-classe em qualquer modo).

### 8.6 `CouplingConfig`

Switches booleanos para ablações causais dos caminhos qualidade → regime/norma/κ e norma → κ. Todos `True` por padrão (mundo totalmente acoplado).

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `quality_affects_regime_probabilities` | `bool` | `True` | qualidade desloca as probabilidades dos regimes radiais (central/cauda) |
| `quality_affects_log_norm` | `bool` | `True` | qualidade desloca a média da log-norma |
| `quality_affects_intra_kappa` | `bool` | `True` | qualidade modula κ intra-classe (e `lb` nos modos de cap) |
| `norm_z_affects_intra_kappa` | `bool` | `True` | a posição radial (z padronizado) modula κ via penalidades de cauda (e `lb` nos modos de cap) |

O uso desses switches para isolar caminhos causais está detalhado na Seção 9.

### 8.7 `AcquisitionDomainConfig`

Parâmetros de um domínio de aquisição — a unidade que cria mismatch controlado entre condições (ex.: `controlled` vs. `surveillance`). Cada domínio define sua distribuição de qualidade, seus regimes radiais, seus deslocamentos de norma/κ e suas escalas de nuisance/viés.

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `name` | `str` | — (obrigatório) | nome único do domínio; não-vazio |
| `weight` | `float` | `1.0` | peso de amostragem do domínio (normalizado entre domínios); positivo |
| `quality_beta_alpha` | `float` | `4.0` | α da Beta da qualidade latente; positivo |
| `quality_beta_beta` | `float` | `2.0` | β da Beta da qualidade latente; positivo |
| `central_regime_probability` | `float` | `0.86` | probabilidade base do regime radial central |
| `low_tail_probability` | `float` | `0.08` | probabilidade base do regime de cauda baixa |
| `high_tail_probability` | `float` | `0.06` | probabilidade base do regime de cauda alta (os três somam 1 após normalização) |
| `low_tail_quality_logit_slope` | `float` | `2.0` | quanto a baixa qualidade aumenta o logit da cauda baixa |
| `high_tail_quality_logit_slope` | `float` | `0.0` | quanto a baixa qualidade aumenta o logit da cauda alta |
| `log_norm_shift` | `float` | `0.0` | deslocamento aditivo da média da log-norma neste domínio |
| `log_norm_sigma_multiplier` | `float` | `1.0` | multiplicador do desvio da log-norma; positivo |
| `log_kappa_shift` | `float` | `0.0` | deslocamento aditivo de \(\log\kappa\) neste domínio |
| `anisotropic_noise_multiplier` | `float` | `1.0` | multiplicador da magnitude do nuisance anisotrópico; \(\ge 0\) |
| `domain_bias_scale` | `float` | `0.0` | escala do viés **fixo** de domínio \(\mathbf{b}_d\); \(\ge 0\) |
| `domain_identity_bias_scale` | `float` | `0.0` | escala do viés **domínio × identidade** \(\mathbf{b}_{d,i}\); \(\ge 0\) |
| `session_bias_scale` | `float` | `0.0` | escala do viés de **sessão** \(\mathbf{b}_{d,s}\); \(\ge 0\) |
| `cap_lower_bound_offset` | `float` | `0.0` | deslocamento aditivo de `lb` da calota neste domínio (qualquer sinal; positivo aperta a identidade — controlado; negativo afrouxa — vigilância); espelha `log_kappa_shift` para a parametrização de calota |

---

## 9. Acoplamentos e ablações causais

O `CouplingConfig` é o que permite usar o simulador como **bancada de ablação causal**: cada switch liga ou desliga uma aresta do grafo gerador. Desligando arestas seletivamente, isola-se o efeito de cada caminho sobre os scores, algo impossível com dados reais, onde os fatores são confundidos.

### 9.1 O grafo causal e os quatro switches

O mundo padrão (todos os switches `True`) tem o grafo:

```
qualidade ──(quality_affects_regime_probabilities)──▶ regime radial
qualidade ──(quality_affects_log_norm)─────────────▶ log-norma  ─▶ norma ─▶ z padronizado
qualidade ──(quality_affects_intra_kappa)──────────▶ κ intra-classe (e lb nos modos de cap)
z padronizado ─(norm_z_affects_intra_kappa)────────▶ κ intra-classe (e lb nos modos de cap)
```

| Switch | Aresta controlada | Onde atua no código |
|---|---|---|
| `quality_affects_regime_probabilities` | qualidade → probabilidades de regime radial | `_sample_radial_regimes` (ajuste de logits das caudas pela qualidade) |
| `quality_affects_log_norm` | qualidade → média da log-norma | `_sample_log_norms` (termo `quality_log_norm_slope·(q−0.5)`) |
| `quality_affects_intra_kappa` | qualidade → κ (e `lb`) | `_calculate_intra_kappas` / `_calculate_cap_lower_bounds` (termo de qualidade) |
| `norm_z_affects_intra_kappa` | posição radial → κ (e `lb`) | `_calculate_intra_kappas` / `_calculate_cap_lower_bounds` (penalidades de cauda) |

Quando um switch é `False`, o termo correspondente é substituído por zero, então o restante da geração segue idêntico — a ablação é **cirúrgica**, sem alterar dimensão, população ou nuisance.

### 9.2 Padrão de ablação

Toda ablação parte de um perfil e clona o `CouplingConfig` com `dataclasses.replace`, mexendo só nos switches de interesse:

```python
from dataclasses import replace
from adaface_embedding_simulator import (
    make_cooperative_plda_config,
    AdaFaceEmbeddingSimulator,
    CouplingConfig,
)

base = make_cooperative_plda_config(dimension=128)

# Mundo totalmente acoplado (referência)
full = base

# Ablação: corta TODO efeito da norma sobre a confiabilidade angular
no_norm_kappa = replace(
    base,
    coupling=replace(base.coupling, norm_z_affects_intra_kappa=False),
)

# Gera os dois e compara as distribuições de score
for tag, cfg in [("full", full), ("no_norm->kappa", no_norm_kappa)]:
    sim = AdaFaceEmbeddingSimulator(cfg, random_seed=7)
    ds = sim.sample_dataset(identity_count=200, samples_per_identity=(4, 8), split_name=tag)
    print(tag, ds.metadata["intra_kappa"].mean(), ds.metadata["own_center_cosine"].mean())
```

A seed estrutural fixa garante que população, bases de nuisance e vieses sejam os mesmos entre as duas execuções — a única diferença observável vem da aresta cortada.

### 9.3 Exemplos de ablação isolando caminhos

**Isolar o efeito da qualidade sobre a norma (relação norma↔confiabilidade).** Desligando `quality_affects_log_norm`, a norma deixa de carregar informação de qualidade; comparando com o mundo cheio, mede-se quanto da estrutura em U invertido vem desse caminho específico:

```python
no_q_norm = replace(base, coupling=replace(base.coupling, quality_affects_log_norm=False))
```

**Isolar o caminho qualidade → κ do caminho norma → κ.** Os dois alimentam κ. Para atribuir a confiabilidade angular a um ou a outro, gere quatro mundos no grid \(2\times 2\):

```python
grid = {}
for q_kappa in (True, False):
    for z_kappa in (True, False):
        cfg = replace(base, coupling=replace(
            base.coupling,
            quality_affects_intra_kappa=q_kappa,
            norm_z_affects_intra_kappa=z_kappa,
        ))
        grid[(q_kappa, z_kappa)] = cfg
```

O mundo `(False, False)` produz κ **constante** (sem dependência de covariáveis), servindo de baseline homocedástico angular; comparado com `(True, False)` isola o caminho qualidade→κ, e com `(False, True)` isola o caminho norma→κ.

**Mundo totalmente desacoplado (controle negativo).** Com os quatro switches `False`, qualidade vira um fator puramente nominal: não afeta regime, norma nem κ. Útil como controle negativo — qualquer correlação score↔qualidade observada nesse mundo é artefato de amostragem, não causal:

```python
fully_decoupled = replace(base, coupling=CouplingConfig(
    quality_affects_regime_probabilities=False,
    quality_affects_log_norm=False,
    quality_affects_intra_kappa=False,
    norm_z_affects_intra_kappa=False,
))
```

**Ablação nos modos de calota.** Nos modos `IDPERTURB_CAP` e `HYBRID_VMF_CAP`, os mesmos dois switches de κ regem a modulação de `lb`. Desligar `norm_z_affects_intra_kappa` num perfil de calota remove as penalidades `cap_low_tail_penalty`/`cap_high_tail_penalty` (o piso de cosseno deixa de depender da posição radial); desligar `quality_affects_intra_kappa` remove o termo `cap_quality_lower_bound_slope`. Assim, a mesma maquinaria de ablação vale para a parametrização angular por κ e por calota.

---

## 10. Perfis prontos

O pacote oferece quatro construtores de configuração prontos, cada um retornando um `SimulatorConfig` completo e validável. São pontos de partida documentados e auditáveis — o nome do perfil é gravado no manifesto (`profile_name`), tornando explícito em laudo qual mundo sintético gerou os dados. Todos aceitam `dimension` por keyword.

### 10.1 `make_spherical_baseline_config`

```python
make_spherical_baseline_config(*, dimension: int = 512) -> SimulatorConfig
```

**Perfil `SPHERICAL_BASELINE`.** Baseline esférico isotrópico para comparar backends cosine / PLDA / PSDA num mundo limpo. Características:

- População de 8 componentes (`identity_center_kappas=70.0`, `component_mean_global_kappa=20.0`).
- **Sem nuisance anisotrópico** (`anisotropic_rank=0`, `anisotropic_base_scale=0.0`) — toda a dispersão intra-classe é vMF isotrópica em torno do centro.
- Um único domínio (`default`), sem viés de domínio, sessão ou identidade.
- `DirectionalConfig` no padrão: modo vMF, atualização por retração.

**Quando usar:** como controle de referência para isolar o efeito de qualquer estrutura adicional. É o mundo mais simples; qualquer dificuldade de um backend aqui é intrínseca, não causada por nuisance ou mismatch de domínio.

### 10.2 `make_cooperative_plda_config`

```python
make_cooperative_plda_config(*, dimension: int = 512) -> SimulatorConfig
```

**Perfil `COOPERATIVE_PLDA_BASELINE`.** Baseline cooperativo de PLDA — um mundo realista mas benigno, com dois domínios e nuisance moderado. Características:

- Mesma população de 8 componentes do baseline esférico.
- `RadialConfig` com U invertido explícito: normas central, cauda baixa (`offset=-0.70`) e cauda alta (`offset=0.70`), inclinação de qualidade `0.12`.
- `DirectionalConfig` com κ de pico `1400`, penalidades de cauda `0.55`/`0.75`, **nuisance anisotrópico global** de posto `min(24, D−1)` e escala `0.010`, modo `GLOBAL_SHARED`, atualização por retração.
- **Dois domínios:** `controlled` (peso `0.65`, qualidade alta via Beta(5,2), regime majoritariamente central, multiplicador de nuisance `0.7`) e `surveillance` (peso `0.35`, qualidade baixa via Beta(2,4.5), caudas mais prováveis, `log_kappa_shift=-0.30`, multiplicador de nuisance `1.7`, `domain_bias_scale=0.008`).

**Quando usar:** experimento PLDA representativo de operação cotidiana, com mismatch suave entre uma condição controlada e uma de vigilância. É o perfil padrão ilustrativo do pacote.

#### `make_example_config` — alias do baseline cooperativo

```python
make_example_config(*, dimension: int = 512) -> SimulatorConfig
```

`make_example_config` é **alias de** `make_cooperative_plda_config`: chama-o diretamente e retorna a mesma configuração (perfil `COOPERATIVE_PLDA_BASELINE`). Existe apenas como nome amigável de entrada; prefira `make_cooperative_plda_config` em código novo para deixar o perfil explícito.

### 10.3 `make_cross_condition_stress_config`

```python
make_cross_condition_stress_config(*, dimension: int = 512) -> SimulatorConfig
```

**Perfil `CROSS_CONDITION_HETEROSCEDASTIC`.** Perfil heterogêneo de estresse cross-condition, projetado para forçar os limites de backends sob mismatch forte. Características:

- `RadialConfig` igual ao baseline cooperativo.
- `DirectionalConfig` agressivo: nuisance **hierárquico** (`nuisance_structure_mode=HIERARCHICAL`, `global_nuisance_weight=0.55`, `domain_nuisance_weight=0.90`), posto `min(20, D−1)`; **nuisance específico de identidade** (`identity_specific_nuisance_rank=min(2, D−1)`, escala `0.40`); atualização pelo **mapa exponencial** (passos grandes, métrica preservada).
- **Dois domínios de peso igual** (`0.5`/`0.5`), com `controlled` carregando viés domínio×identidade (`0.002`) e sessão (`0.001`), e `surveillance` carregando viés de domínio (`0.006`), domínio×identidade (`0.012`), sessão (`0.006`), `log_kappa_shift=-0.35` e multiplicador de nuisance `1.85`.

**Quando usar:** estudos cross-condition exigentes — avaliar PLDA two-covariance cross-condition, robustez de calibração sob mismatch sistemático, ou estresse de modelos de tipicidade. O mapa exponencial e o nuisance hierárquico tornam este o mundo mais difícil dos quatro.

### 10.4 `make_idperturb_cap_config`

```python
make_idperturb_cap_config(*, dimension: int = 512, lower_bound: float = 0.6) -> SimulatorConfig
```

**Perfil `IDPERTURB_CAP`.** Baseline cooperativo de PLDA usando a **calota esférica uniforme em cosseno** (IDperturb) como dispersão intra-classe, em vez de vMF. Parte de `make_cooperative_plda_config` e troca apenas o bloco de dispersão:

- `intra_class_sampling_mode=IDPERTURB_CAP`.
- `cap_base_lower_bound=lower_bound` (piso de cosseno base `lb`).
- `cap_quality_lower_bound_slope=0.10`, `cap_low_tail_penalty=0.15`, `cap_high_tail_penalty=0.20` — `lb` cresce com a qualidade e cai nas caudas de norma.
- `cap_overlap_guard=True` — o piso da bissetriz por identidade está ativo, de modo que o `lb` **efetivo** pode exceder `lower_bound` para identidades em regiões densas da população.

O `lower_bound` padrão `0.6` é **inspirado** no regime experimental do IDperturb (Boutros et al., 2026), **não** um ótimo para AdaFace/PLDA/FaceAudit — calibre empiricamente antes de tirar conclusões.

**Quando usar:** quando o experimento exige um **afastamento angular máximo interpretável** por identidade (piso de cosseno garantido), em vez de uma concentração média vMF sem piso rígido. Útil para estudar o impacto de um piso de dispersão sobre PLDA/calibração e para reproduzir o paradigma de perturbação por calota. Reporte sempre os percentis de `cap_lower_bound_effective`, pois o `lb` nominal é um piso, não o valor realizado.

### 10.5 Resumo comparativo dos perfis

| Construtor | `profile_name` | Domínios | Dispersão intra-classe | Nuisance | Atualização |
|---|---|---|---|---|---|
| `make_spherical_baseline_config` | `SPHERICAL_BASELINE` | 1 (`default`) | vMF isotrópico | nenhum | retração |
| `make_cooperative_plda_config` (= `make_example_config`) | `COOPERATIVE_PLDA_BASELINE` | 2 (`controlled`/`surveillance`) | vMF | global, moderado | retração |
| `make_cross_condition_stress_config` | `CROSS_CONDITION_HETEROSCEDASTIC` | 2 (peso igual) | vMF | hierárquico + identidade, forte | mapa exponencial |
| `make_idperturb_cap_config` | `IDPERTURB_CAP` | 2 (`controlled`/`surveillance`) | calota (IDPERTURB_CAP) | global, moderado | retração |
## 11. Reprodutibilidade

A reprodutibilidade é tratada como requisito de auditabilidade forense, não como conveniência. Toda a aleatoriedade do `AdaFaceEmbeddingSimulator` deriva de uma **única semente raiz** (`structural_seed`, passada como `random_seed` ao construtor, padrão `42`) através de uma função de derivação determinística e independente da ordem das chamadas. Isso garante que dois experimentos com a mesma configuração e a mesma semente produzam embeddings bit-a-bit idênticos — uma propriedade indispensável para reproduzir um laudo ou uma ablação meses depois.

### 11.1 Semente raiz e derivação estável

O construtor armazena `self.structural_seed = int(random_seed)`. A partir daí, cada fluxo aleatório recebe sua própria semente derivada via `_stable_seed`:

\[
\text{seed}(\text{root}, p_1, \dots, p_n) = \text{blake2b}_{64}\bigl(\text{root} \,|\, p_1 \,|\, \dots \,|\, p_n\bigr)
\]

```python
def _stable_seed(root_seed: int, *parts: object) -> int:
    """Deriva uma semente de 64 bits determinística e independente da ordem das chamadas."""
    payload = "|".join([str(int(root_seed)), *(str(part) for part in parts)])
    digest = blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)
```

A propriedade central é a **independência da ordem das chamadas**: a semente de um fluxo depende apenas do conjunto de `parts` que o nomeiam, nunca de quantos outros `Generator` já foram criados antes. Não há um único `np.random.Generator` global cujo estado seja consumido sequencialmente; cada operação cria seu próprio gerador via `np.random.default_rng(seed)`. Consequência prática: você pode reordenar a geração de splits, pular uma condição, gerar apenas o split de calibração hoje e o de avaliação amanhã — e cada um sai idêntico ao que sairia se todos tivessem sido gerados de uma vez na ordem canônica.

### 11.2 Os três fluxos de semente de um dataset

Cada `SyntheticEmbeddingDataset` carrega três sementes explícitas, persistidas no manifesto (Seção 18) e expostas como atributos:

| Atributo | Origem | Papel |
|---|---|---|
| `structural_seed` | `random_seed` do construtor | Fixa a **geometria estrutural**: médias dos componentes da população (`population_component_means`), bases anisotrópicas globais/por-domínio, vetores de viés fixo de domínio. Computada uma vez no `__init__` a partir do stream `"structure"`. |
| `population_sampling_seed` | `sampling_seed` da `SyntheticIdentityPopulation` | Fixa **quais identidades latentes existem**: centros por identidade, bases de ruído específicas por identidade, vieses domínio×identidade. |
| `observation_sampling_seed` | semente do stream de observação | Fixa **como cada identidade é observada**: domínio, qualidade, regime radial, norma, dispersão intra-classe, perturbações tangenciais. |

Essa tripartição permite ablações causais limpas: fixar `structural_seed` + `population_sampling_seed` e variar apenas a observação isola o efeito do ruído de aquisição mantendo idênticos a geometria e o elenco de identidades.

### 11.3 Streams nomeados

Cada API de geração recebe um `stream_name` que entra na derivação da semente, isolando fluxos logicamente distintos mesmo sob a mesma `structural_seed`:

- **Estrutura** — stream fixo `"structure"`, derivado no construtor.
- **População** — `sample_identity_population(stream_name=...)`, padrão `"population"`. A semente combina `("population", stream_name, identity_offset, count)`.
- **Observação** — `sample_observations(stream_name=...)`. A semente combina `("observations", stream, forced_domain or "mixed")`.

Os atalhos de alto nível compõem nomes determinísticos. `sample_dataset(split_name="train")` usa `"train:population"` e `"train:observations"`; `sample_cross_condition_datasets(split_name="case")` usa `"case:population"` e `"case:observations:<domínio>"` por condição. Dois splits com `split_name` distinto recebem, portanto, fluxos disjuntos automaticamente, sem o experimentador precisar gerenciar sementes manualmente.

### 11.4 Override explícito de semente

Qualquer fluxo aceita um `random_seed`/`population_random_seed`/`observation_random_seeds` explícito que **substitui** a derivação `_stable_seed` para aquele fluxo:

```python
def _resolved_seed(self, *parts, explicit_seed: int | None = None) -> int:
    return int(explicit_seed) if explicit_seed is not None else _stable_seed(self.structural_seed, *parts)
```

Override explícito é útil para forçar duas execuções a divergirem ou convergirem deliberadamente (por exemplo, repetir uma observação com semente fixa enquanto se varia outro parâmetro). Para máxima auditabilidade, prefira deixar as sementes derivadas e registrar apenas a `structural_seed` — o manifesto reconstrói todo o resto.

### 11.5 Garantia: injetar centros NÃO consome o RNG de amostragem de centros

Quando uma população é **injetada** (forma `centers=` de `sample_identity_population`, Seção 12), o método pula `_sample_identity_centers` por completo — o gerador de números aleatórios **não é consumido** para sortear centros:

```python
if injected:
    # centros validados/normalizados; _sample_identity_centers NÃO é chamado
    ...
else:
    components, centers = self._sample_identity_centers(count, rng)
```

Isso tem duas consequências de reprodutibilidade que devem ser entendidas:

1. **O caminho amostrado permanece bit-a-bit idêntico.** Como a injeção desvia antes de tocar o `rng` para centros, a sequência de consumo do gerador no caminho sorteado (`identity_count=`) é exatamente a mesma independentemente de a feature de injeção existir.
2. **O `rng` da população ainda avança após os centros.** Mesmo no caminho injetado, o gerador é usado em seguida para `identity_specific_nuisance_bases` e para os vieses domínio×identidade. Portanto, populações injetadas continuam determinísticas e auditáveis sob a mesma `structural_seed` + `stream_name`, mas seu estado de RNG difere do caminho sorteado a partir do ponto em que os centros teriam sido consumidos.

Recomendação de laudo: registre sempre a `structural_seed`, os `stream_name` usados e (para Modo A/B) a proveniência da base-âncora, junto com o `source_sha256` do manifesto. Esses cinco itens reconstroem o experimento integralmente.

---

## 12. População e observações

O simulador separa rigidamente **população** (quais identidades latentes existem, sua geometria na hiperesfera) de **observação** (como cada identidade é capturada sob condições de aquisição). Essa separação é o que torna possível observar a mesma população sob múltiplas condições alinhadas por identidade (Seção 13) e ancorar a população em dados reais (Seção 14).

- `sample_identity_population(...)` → produz uma `SyntheticIdentityPopulation` (latente, sem observações).
- `sample_observations(population, ...)` → observa uma população existente, produzindo um `SyntheticEmbeddingDataset`.
- `sample_dataset(...)` → atalho que encadeia os dois para um split isolado de uma só vez.

### 12.1 `sample_identity_population` — assinatura completa

```python
def sample_identity_population(
    self,
    *,
    identity_count: int | None = None,
    centers: ArrayLike | None = None,
    components: ArrayLike | None = None,
    identity_labels: Sequence[str] | None = None,
    source_sample_counts: ArrayLike | None = None,
    identity_offset: int = 0,
    random_seed: int | None = None,
    stream_name: str = "population",
) -> SyntheticIdentityPopulation:
```

O método tem **duas formas mutuamente exclusivas**, selecionadas pela presença de `identity_count` (forma sorteada) ou de `centers` (forma injetada).

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `identity_count` | `int \| None` | `None` | **Forma sorteada.** Número de identidades a sortear da mistura vMF configurada (`PopulationConfig`). Exatamente um entre `identity_count` e `centers` deve ser informado. |
| `centers` | `ArrayLike \| None` | `None` | **Forma injetada.** Matriz `(M, dimension)` de direções de centro fornecidas externamente (ex.: médias esféricas reais por identidade). Re-normalizada por linha; a dimensão deve bater com `config.dimension`. |
| `components` | `ArrayLike \| None` | `None` | (Apenas injetada) Rótulo de componente populacional por centro `(M,)`, inteiros em `[0, component_count)`. Se omitido, todos recebem componente `0`. |
| `identity_labels` | `Sequence[str] \| None` | `None` | (Apenas injetada) Rótulo de identidade de origem por centro `(M,)`. Armazenado como `source_identity_labels`; alimenta as colunas de proveniência e os helpers anti-vazamento (Seção 14). |
| `source_sample_counts` | `ArrayLike \| None` | `None` | (Apenas injetada) Número de imagens reais que formaram cada centro `(M,)`, inteiros positivos. Armazenado como `source_sample_counts`. |
| `identity_offset` | `int` | `0` | Deslocamento somado aos `identity_ids` gerados (`np.arange(count) + identity_offset`). Garante IDs disjuntos entre populações geradas separadamente. |
| `random_seed` | `int \| None` | `None` | Override explícito da semente da população. Se `None`, a semente é derivada de `("population", stream_name, identity_offset, count)`. |
| `stream_name` | `str` | `"population"` | Nome do fluxo, entra na derivação da semente e é registrado em `SyntheticIdentityPopulation.stream_name`. |

**Regra de exclusividade e validações.** O método exige exatamente um entre `identity_count` e `centers`:

```python
injected = centers is not None
if injected == (identity_count is not None):
    raise ValueError("Provide exactly one of identity_count or centers.")
```

Validações por forma:

- **Sorteada (`identity_count=`):** `identity_count` deve ser positivo; os centros são sorteados via `_sample_identity_centers`, que escolhe um componente por identidade segundo `component_weights` e amostra uma direção vMF em torno da média do componente com concentração `identity_center_kappas`.
- **Injetada (`centers=`):** `centers` é re-normalizada por linha; `centers.shape[1]` deve igualar `config.dimension`. Se `components` for dado, deve ter forma `(M,)`, ser de tipo inteiro e estar em `[0, component_count)`. Se `identity_labels` for dado, deve ter comprimento `M`. Se `source_sample_counts` for dado, deve ter forma `(M,)` e conter apenas inteiros positivos.

Em ambas as formas, após determinar os centros, o `rng` da população continua sendo usado para `identity_specific_nuisance_bases` (se `identity_specific_nuisance_rank > 0` e escala positiva) e para os vetores de viés domínio×identidade por domínio. O resultado passa por `population.validate()` antes de retornar (verifica formas e que os centros são unitários).

**A `SyntheticIdentityPopulation` resultante** contém: `identity_ids`, `population_components`, `identity_centers` (unitários), `identity_specific_nuisance_bases`, `domain_identity_bias_vectors`, `sampling_seed`, `stream_name`, mais o `overlap_lower_bound_floor` (piso de bissetriz, computado preguiçosamente e cacheado para reuso entre condições — Seção 6) e, no caso injetado, `source_identity_labels`/`source_sample_counts`.

### 12.2 `sample_observations` — observar uma população existente

```python
def sample_observations(
    self,
    population: SyntheticIdentityPopulation,
    *,
    samples_per_identity: int | tuple[int, int],
    split_name: str = "synthetic",
    forced_domain: str | None = None,
    sessions_per_identity: int | tuple[int, int] = 1,
    random_seed: int | None = None,
    stream_name: str | None = None,
) -> SyntheticEmbeddingDataset:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `population` | `SyntheticIdentityPopulation` | — | População latente a observar (de `sample_identity_population`). Revalidada na entrada. |
| `samples_per_identity` | `int \| tuple[int, int]` | — | Número de observações por identidade. Inteiro fixo, ou faixa `(mínimo, máximo)` sorteada uniformemente por identidade (`0 < mínimo ≤ máximo`). |
| `split_name` | `str` | `"synthetic"` | Rótulo gravado na coluna `split` do metadata. Não pode ser vazio. |
| `forced_domain` | `str \| None` | `None` | Se informado, todas as observações vêm do domínio de aquisição nomeado (deve existir em `acquisition_domains`). Se `None`, os domínios são sorteados segundo seus pesos. |
| `sessions_per_identity` | `int \| tuple[int, int]` | `1` | Número de sessões por identidade (fixo ou faixa). Cada observação recebe um índice de sessão; o viés de sessão (`session_bias_scale`) é compartilhado dentro de uma mesma `(identidade, domínio, sessão)`. |
| `random_seed` | `int \| None` | `None` | Override explícito da semente de observação. Se `None`, deriva de `("observations", stream, forced_domain or "mixed")`. |
| `stream_name` | `str \| None` | `None` | Nome do fluxo de observação; se `None`, usa `split_name`. Registrado em `dataset.stream_name`. |

O método executa o pipeline gerativo completo de cinco camadas (Seção 5): expande contagens por identidade, sorteia domínios, qualidade latente, regime radial, log-normas e o `reference_scaled_log_norm_z`; calcula `intra_kappa` (e, em modos de cap, `cap_lower_bound_effective`); aplica a dispersão intra-classe (VMF, IDPERTURB_CAP ou HYBRID_VMF_CAP — Seção 6); aplica as perturbações tangenciais de ruído (Seção 7); e monta `raw_embeddings = embedding_norm · direção_normalizada`. Produz um `SyntheticEmbeddingDataset` validado, com o `DataFrame` de metadata completo (Seção 15) e as três sementes registradas.

### 12.3 `sample_dataset` — split isolado em uma chamada

```python
def sample_dataset(
    self,
    *,
    identity_count: int,
    samples_per_identity: int | tuple[int, int],
    split_name: str = "synthetic",
    identity_offset: int = 0,
    sessions_per_identity: int | tuple[int, int] = 1,
    random_seed: int | None = None,
    population_random_seed: int | None = None,
) -> SyntheticEmbeddingDataset:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `identity_count` | `int` | — | Número de identidades a sortear para o split. |
| `samples_per_identity` | `int \| tuple[int, int]` | — | Observações por identidade (fixo ou faixa). |
| `split_name` | `str` | `"synthetic"` | Rótulo do split; também compõe os `stream_name` (`"{split}:population"`, `"{split}:observations"`). |
| `identity_offset` | `int` | `0` | Deslocamento de IDs — use valores distintos por split para garantir identidades disjuntas. |
| `sessions_per_identity` | `int \| tuple[int, int]` | `1` | Sessões por identidade. |
| `random_seed` | `int \| None` | `None` | Override da semente de observação. |
| `population_random_seed` | `int \| None` | `None` | Override da semente da população. |

`sample_dataset` encadeia `sample_identity_population(identity_count=...)` e `sample_observations(...)` com nomes de stream derivados do `split_name`. Como as sementes são independentes da ordem de chamada (Seção 11), gerar `train`, `cal` e `eval` com `split_name` distinto e `identity_offset` distinto produz splits com identidades disjuntas e fluxos aleatórios disjuntos — sem necessidade de gerenciar sementes manualmente.

---

## 13. Geração cross-condition

A geração cross-condition observa **uma única população** sob **múltiplas condições de aquisição forçadas**, com as observações **alinhadas por identidade**. É o modo canônico para estudar comparações forenses entre condições assimétricas (por exemplo, padrão cooperativo frontal versus questionada de vigilância), pois garante que cada identidade aparece em todas as condições, gerando pares genuínos cross-condition legítimos.

### 13.1 `sample_cross_condition_datasets`

```python
def sample_cross_condition_datasets(
    self,
    *,
    identity_count: int,
    samples_per_identity_by_domain: Mapping[str, int | tuple[int, int]],
    split_name: str = "cross_condition",
    identity_offset: int = 0,
    sessions_per_identity: int | tuple[int, int] = 1,
    population_random_seed: int | None = None,
    observation_random_seeds: Mapping[str, int] | None = None,
) -> dict[str, SyntheticEmbeddingDataset]:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `identity_count` | `int` | — | Número de identidades latentes — compartilhadas por **todas** as condições. |
| `samples_per_identity_by_domain` | `Mapping[str, int \| tuple[int, int]]` | — | Mapa `nome_do_domínio → observações por identidade` (fixo ou faixa). As chaves devem nomear domínios existentes em `acquisition_domains`; cada uma vira uma condição forçada. |
| `split_name` | `str` | `"cross_condition"` | Prefixo dos rótulos e dos `stream_name`. |
| `identity_offset` | `int` | `0` | Deslocamento de IDs para a população compartilhada. |
| `sessions_per_identity` | `int \| tuple[int, int]` | `1` | Sessões por identidade, aplicado em cada condição. |
| `population_random_seed` | `int \| None` | `None` | Override da semente da população compartilhada. |
| `observation_random_seeds` | `Mapping[str, int] \| None` | `None` | Override por domínio da semente de observação. Domínios ausentes do mapa usam a semente derivada. |

O método sorteia **uma** `SyntheticIdentityPopulation` (stream `"{split_name}:population"`) e, para cada domínio do mapa, chama `sample_observations` com `forced_domain=<domínio>`, `split_name=f"{split_name}:{domínio}"` e `stream_name=f"{split_name}:observations:{domínio}"`. Retorna um `dict[str, SyntheticEmbeddingDataset]` indexado pelo nome do domínio.

**Garantia de alinhamento por identidade.** Como todas as condições observam a mesma população, os `identity_id` são idênticos entre os datasets retornados. Cada identidade produz, portanto, pares genuínos *cross-condition* legítimos (mesma identidade, condições diferentes) — exatamente o material necessário para treinar/avaliar PLDA cross-condition e calibradores que enxergam a assimetria entre condições. O `overlap_lower_bound_floor` da população é computado uma vez e reutilizado entre as condições (cacheado na própria `SyntheticIdentityPopulation`), garantindo que o piso de guarda anti-colisão seja consistente entre os domínios.

### 13.2 Exportação para o FaceAudit

Para transformar a saída cross-condition em datasets reais do FaceAudit, use `cross_condition_to_embedding_datasets` do **adapter concreto** (`simulator/faceaudit_concrete_adapter.py`, detalhado na Seção 19):

```python
from faceaudit_concrete_adapter import cross_condition_to_embedding_datasets

datasets_by_domain = simulator.sample_cross_condition_datasets(
    identity_count=300,
    samples_per_identity_by_domain={"controlled": 4, "surveillance": (2, 5)},
)
faceaudit_datasets = cross_condition_to_embedding_datasets(datasets_by_domain)
# -> {"controlled": EmbeddingDataset, "surveillance": EmbeddingDataset}
```

`cross_condition_to_embedding_datasets` mapeia cada `SyntheticEmbeddingDataset` via `to_embedding_dataset`, preservando os mesmos valores de `identity` entre as condições. Os datasets resultantes formam um conjunto cross-condition pareado pronto para `PAIRED_DATASET_SPLIT` ou para treino de PLDA two-covariance cross-condition no FaceAudit. (O scaffold genérico `faceaudit_adapter.py` **não** deve ser usado em produção — ver Seção 19.)

---

## 14. População ancorada em dataset real

Os centros sorteados por uma mistura vMF genérica não respeitam a geometria empírica fina dos embeddings faciais reais. O módulo `simulator/faceaudit_seed.py` é a ponte que alimenta a população a partir de um `EmbeddingDataset` real do FaceAudit, em dois modos. Para que `import simulator.faceaudit_seed` funcione sem o FaceAudit instalado, `faceaudit` é importado preguiçosamente apenas dentro das funções que dele precisam.

> **Caveat de vazamento — proeminente.** Em ambos os modos, a base usada para ancorar/ajustar **deve ser disjunta em identidades** de qualquer conjunto de avaliação. Use `assert_disjoint_source_identities` (Seção 14.5). O Modo A **não é anonimização**.

### 14.1 `identity_centers_from_dataset` — centros esféricos por identidade

Computa um centro de média esférica por identidade:

\[
\hat{c}_i = \text{normalize}\!\left(\sum_{j \in i} w_j\, u_{ij}\right), \qquad u_{ij} = \frac{x_{ij}}{\lVert x_{ij}\rVert}
\]

```python
def identity_centers_from_dataset(
    dataset: Any,
    *,
    min_samples: int = 1,
    weight_by_quality: bool = False,
    eps: float = 1e-12,
) -> IdentityCenterExtractionResult:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dataset` | `Any` | — | `EmbeddingDataset` real (expõe `matrix`, `identities` e, se `weight_by_quality`, `items` com `quality_score`). |
| `min_samples` | `int` | `1` | Mínimo de imagens por identidade para formar um centro. Identidades com menos são descartadas. Padrão `1` é flexível; prefira `2–3` (`1` = âncora ruidosa, `2+` mais estável, `5+` ideal). |
| `weight_by_quality` | `bool` | `False` | Pondera cada imagem pelo seu `quality_score` na média esférica. |
| `eps` | `float` | `1e-12` | Limiar numérico de degeneração da norma. |

Comportamento de agrupamento e robustez:

- **Agrupa pelo valor ORIGINAL de `identity`**, não por `str(identity)` — evita fundir `1` (int) e `"1"` (str). Preserva a ordem de primeira ocorrência. Identidades `None` são ignoradas.
- **Ponderação por qualidade** (`weight_by_quality=True`): exige `dataset.items` com `quality_score`. Tratamento de casos: `None` ou `NaN` → peso `1.0`; valor **negativo** → erro (`ValueError`); se todos os pesos de uma identidade forem zero → cai para a média **não ponderada** do bloco.
- **Fallback degenerado**: se a média ponderada tiver norma `≤ eps` (vetores quase se cancelando), emite um `UserWarning` e cai para a imagem de **melhor qualidade** (ou a primeira, se não houver pesos positivos) daquela identidade.
- Se nenhuma identidade atingir `min_samples`, levanta `ValueError`.

**`IdentityCenterExtractionResult`** (dataclass) carrega o produto da extração:

| Campo | Tipo | Significado |
|---|---|---|
| `centers` | `FloatArray` `(M, D)` | Direções unitárias dos centros por identidade. |
| `labels` | `list[str]` | Rótulo original de identidade do FaceAudit por centro. |
| `counts` | `IntArray` `(M,)` | Número de imagens reais que formaram cada centro. |
| `quality_weighted` | `bool` | Se a extração usou ponderação por qualidade. |

### 14.2 Modo A — `population_from_dataset` (ancorado)

No Modo A, **os centros da população SÃO as médias esféricas reais por identidade**. As identidades são reais; apenas a dispersão intra-identidade (e a norma) é sintética.

```python
def population_from_dataset(
    simulator: Any,
    dataset: Any,
    *,
    min_samples: int = 1,
    identity_offset: int = 0,
    random_seed: int | None = None,
    weight_by_quality: bool = False,
    stream_name: str = "dataset_seeded",
) -> SyntheticIdentityPopulation:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `simulator` | `AdaFaceEmbeddingSimulator` | — | Simulador cuja `config.dimension` deve bater com a dimensão dos embeddings. |
| `dataset` | `Any` | — | `EmbeddingDataset` real a ancorar. |
| `min_samples` | `int` | `1` | Repassado a `identity_centers_from_dataset`. |
| `identity_offset` | `int` | `0` | Deslocamento de IDs sintéticos. |
| `random_seed` | `int \| None` | `None` | Override da semente da população injetada. |
| `weight_by_quality` | `bool` | `False` | Repassado a `identity_centers_from_dataset`. |
| `stream_name` | `str` | `"dataset_seeded"` | Nome do fluxo da população. |

A função extrai os centros e os **injeta** via `simulator.sample_identity_population(centers=..., identity_labels=..., source_sample_counts=...)` (forma injetada da Seção 12.1). As observações subsequentes (`sample_observations`) ganham as colunas de proveniência `source_identity` e `source_sample_count` no metadata (Seção 15).

> **Caveat proeminente — Modo A NÃO é anonimização.** Ele produz variação sintética em torno de **âncoras biométricas reais**. As identidades resultantes são reais. Use apenas para augmentação, validação metodológica ou experimentos internos sobre uma base autorizada, e **mantenha o conjunto-âncora disjunto de qualquer conjunto de avaliação** (`assert_disjoint_source_identities`).

### 14.3 Modo B — `population_config_from_dataset` (distribution-matched)

No Modo B, ajusta-se uma MoVMF (mistura de von Mises-Fisher) aos centros reais por identidade e usam-se os `μ_k/κ_k/π_k` ajustados como `PopulationConfig`. O simulador então sorteia **novos centros fantasma** de uma aproximação MoVMF da distribuição empírica de centros.

```python
def population_config_from_dataset(
    dataset: Any,
    *,
    n_components: int | None = None,
    component_candidates: Iterable[int] = (1, 2, 4, 8, 16),
    max_k: int | None = None,
    criterion: str = "bic",
    min_samples: int = 1,
    weight_by_quality: bool = False,
    random_state: int = 42,
    n_init: int = 5,
) -> tuple[PopulationConfig, dict]:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dataset` | `Any` | — | `EmbeddingDataset` real. |
| `n_components` | `int \| None` | `None` | Se informado, ajusta esse `K` fixo. Se `None`, seleciona `K` por critério. |
| `component_candidates` | `Iterable[int]` | `(1, 2, 4, 8, 16)` | Candidatos de `K` na seleção, restringidos a `[1, min(M, max_k)]`. |
| `max_k` | `int \| None` | `None` | Teto de `K` na seleção. Se `None`, usa `max(component_candidates)`. |
| `criterion` | `str` | `"bic"` | Critério de seleção de `K`: `"bic"` ou `"aic"`. |
| `min_samples` | `int` | `1` | Repassado a `identity_centers_from_dataset`. |
| `weight_by_quality` | `bool` | `False` | Repassado a `identity_centers_from_dataset`. |
| `random_state` | `int` | `42` | Semente do ajuste da MoVMF. |
| `n_init` | `int` | `5` | Número de reinícios (restarts) do EM. |

Pontos-chave:

- **Ajusta nos CENTROS por identidade**, não em todas as observações (que misturariam variância intra e inter-classe).
- Usa `MoVMFTypicalityModel` do FaceAudit com **`centering="none"`** (importado preguiçosamente). Exige `M ≥ 2` centros.
- **Seleção de `K`**: itera os candidatos válidos, ajusta uma MoVMF por `K`, lê `bic_`/`aic_` e escolhe o menor valor finito. Falhas numéricas/EM por `K` são capturadas (o valor do critério vira `None`) e o `K` é pulado. Se nenhuma MoVMF convergir, levanta `RuntimeError` com o dicionário de critérios por `K`.
- Constrói a `PopulationConfig` com `component_count=K`, `component_means=μ_k`, `component_weights=π_k`, `identity_center_kappas=κ_k`.

**Dicionário `fit_summary`** retornado (segundo elemento da tupla) — registro de auditoria do ajuste:

| Chave | Significado |
|---|---|
| `population_source` | Sempre `"faceaudit_movmf"`. |
| `selected_k` | `K` efetivamente ajustado. |
| `n_components_fixed` | `K` fixo, ou `None` se houve seleção. |
| `candidate_k` | Lista de `K` candidatos avaliados (ou `None` se `K` fixo). |
| `criterion` | `"bic"` ou `"aic"`. |
| `criterion_values` | Valor do critério por `K` candidato (`None` para `K` que falhou). |
| `fit_seed` | `random_state` usado. |
| `n_identity_centers` | `M`, número de centros ajustados. |
| `dimension` | Dimensão dos embeddings. |
| `n_degenerate` | Número de componentes degenerados da MoVMF. |
| `min_component_weight` | Menor `π_k`. |
| `max_component_kappa` | Maior `κ_k`. |
| `component_weight_entropy` | Entropia dos pesos dos componentes. |
| `weight_by_quality` | Se a extração de centros usou qualidade. |
| `min_samples` | `min_samples` usado na extração. |

### 14.4 Modo B completo — `simulator_config_from_dataset` (com ajuste radial)

Monta uma `SimulatorConfig` completa de Modo B: população MoVMF ajustada aos centros reais **mais** um ajuste radial (de norma) opcional.

```python
def simulator_config_from_dataset(
    dataset: Any,
    *,
    base_config: SimulatorConfig | None = None,
    fit_norm: bool = True,
    embedding_view_for_norm: str = "raw",
    warn_if_unit_norms: bool = True,
    norm_tol: float = 1e-3,
    **movmf_kwargs: Any,
) -> tuple[SimulatorConfig, dict]:
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dataset` | `Any` | — | `EmbeddingDataset` real. |
| `base_config` | `SimulatorConfig \| None` | `None` | Config base. Se `None`, usa `make_cooperative_plda_config(dimension=<dim do dataset>)`. Sua `dimension` deve bater com a do dataset. |
| `fit_norm` | `bool` | `True` | Se `True`, substitui a referência radial por uma ajustada às normas reais. |
| `embedding_view_for_norm` | `str` | `"raw"` | Visão usada para a norma. Apenas `"raw"` (`dataset.matrix`) é suportado; outro valor levanta `ValueError`. |
| `warn_if_unit_norms` | `bool` | `True` | Emite aviso se o dataset parecer L2-normalizado. |
| `norm_tol` | `float` | `1e-3` | Tolerância para detectar normas ~unitárias. |
| `**movmf_kwargs` | `Any` | — | Repassados a `population_config_from_dataset` (ex.: `n_components`, `criterion`, `max_k`, `weight_by_quality`). |

Comportamento:

- Parte do `base_config` (padrão `make_cooperative_plda_config` na dimensão do dataset), substitui a população pela MoVMF ajustada e define `profile_name="FACEAUDIT_MOVMF"`.
- **Ajuste radial** (`fit_norm=True`): computa as normas reais e a escala robusta (MAD escalado) dos log-normas. Se a escala robusta for desprezível (`≤ 1e-9`) — caso de normas constantes ou L2-normalizadas — **mantém a referência radial base e pula o ajuste**, emitindo um `UserWarning` quando `warn_if_unit_norms`. Caso contrário, substitui o radial por `RadialConfig.from_reference(RadialReference.fit(matrix))`.

  > **Aviso de normas ~unitárias.** Uma referência radial robusta precisa que os log-normas variem. Um dataset L2-normalizado (`|média(‖x‖) − 1| < tol` e `desvio(‖x‖) < tol`) é degenerado para esse ajuste — daí o aviso e a manutenção do radial base em vez de uma falha.

- O `fit_summary` retornado é o de `population_config_from_dataset` acrescido da chave `norm_summary` (média/desvio/mín/máx das normas, `unit_norm_warning`, `radial_fitted`), ou `norm_summary=None` se `fit_norm=False`.

> **Caveat de aproximação de 1ª ordem.** Os centros fantasma do Modo B são amostrados de uma **mistura isotrópica por componente** — uma densidade angular de **primeira ordem** da distribuição empírica de centros, **não** a variedade real dos embeddings faciais. O Modo B reproduz a estrutura angular grosseira da população real, não sua geometria fina.

### 14.5 Helpers anti-vazamento — `source_identity_set` / `assert_disjoint_source_identities`

Independentes do FaceAudit (puro numpy/conjuntos):

```python
def source_identity_set(population: SyntheticIdentityPopulation) -> set[str]:
    """Conjunto de rótulos de identidade de origem em que uma população Modo-A foi ancorada
    (vazio se a população foi sorteada)."""

def assert_disjoint_source_identities(
    population_a: SyntheticIdentityPopulation,
    population_b: SyntheticIdentityPopulation,
) -> None:
    """Levanta ValueError se duas populações ancoradas compartilharem qualquer identidade de
    origem (vazamento âncora↔avaliação)."""
```

- `source_identity_set` lê `population.source_identity_labels`; retorna conjunto vazio para populações sorteadas (sem proveniência).
- `assert_disjoint_source_identities` intersecta os dois conjuntos de origem e, se houver qualquer identidade em comum, levanta `ValueError` informando a contagem e exemplos (até cinco). É a guarda canônica para garantir que a base-âncora de treino/calibração não vaze para a avaliação — invoque-a sempre que ancorar populações distintas a partir de bases reais.
## 15. O dataset e o dicionário de colunas de metadata

Toda chamada de geração (`sample_observations`, `sample_dataset`,
`sample_cross_condition_datasets`) devolve um objeto `SyntheticEmbeddingDataset`. Ele empacota,
em um único bundle auditável, as matrizes de embedding (bruta e normalizada), os centros
latentes, a tabela de metadados por observação, os artefatos estruturais usados na geração
(bases anisotrópicas, vetores de bias de domínio) e a procedência reproducível (config + três
seeds + nome do stream). É a fronteira entre o mundo gerativo e qualquer experimento a jusante:
PLDA, calibração, tipicidade, ou os adapters do FaceAudit (Seção 19).

### 15.1 Campos do `SyntheticEmbeddingDataset`

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `raw_embeddings` | `FloatArray` `(N, D)` | obrigatório | embeddings brutos \(x_{ij}=r_{ij}\,u_{ij}\) (direção × norma); a norma carrega o sinal radial usado em calibração/qualidade |
| `normalized_embeddings` | `FloatArray` `(N, D)` | obrigatório | direções unitárias \(u_{ij}\) (cada linha com \(\lVert u\rVert=1\)); base dos escores de cosseno |
| `identity_centers` | `FloatArray` `(M, D)` | obrigatório | centros angulares latentes \(\mathbf{c}_i\) das \(M\) identidades observadas |
| `metadata` | `pandas.DataFrame` | obrigatório | uma linha por observação; dicionário completo de colunas na Seção 15.2 |
| `population_component_means` | `FloatArray` | obrigatório | médias \(\mu_k\) dos componentes populacionais (artefato estrutural) |
| `anisotropic_basis` | `FloatArray` ou `None` | obrigatório | base ortonormal do nuisance anisotrópico global (`None` quando desativado) |
| `domain_anisotropic_bases` | `Mapping[str, FloatArray\|None]` | obrigatório | base anisotrópica por domínio (`None` por domínio quando desativada) |
| `domain_bias_vectors` | `Mapping[str, FloatArray]` | obrigatório | vetor de bias fixo por domínio (zeros quando `domain_bias_scale=0`) |
| `config` | `SimulatorConfig` | obrigatório | a configuração integral que gerou o dataset (serializável, Seção 18) |
| `structural_seed` | `int` | obrigatório | seed estrutural do simulador (artefatos estáveis: componentes, bases, bias de domínio) |
| `population_sampling_seed` | `int` | obrigatório | seed efetiva da amostragem da população latente |
| `observation_sampling_seed` | `int` | obrigatório | seed efetiva da amostragem das observações |
| `stream_name` | `str` | obrigatório | nome do stream nomeado de RNG usado nas observações |

O método `validate()` reverifica os invariantes físicos do bundle antes de qualquer uso:
`normalized_embeddings` tem o mesmo shape de `raw_embeddings`, o `metadata` tem exatamente \(N\)
linhas, todas as direções normalizadas são unitárias (`atol=1e-8`), e as normas observadas de
`raw_embeddings` batem com a coluna `embedding_norm` do metadata (`rtol=1e-8`). `validate()` é
chamado internamente ao final de cada geração — uma falha aqui sinaliza corrupção do estado, não
configuração imprópria.

### 15.2 Dicionário completo das colunas de metadata por observação

A tabela abaixo é a referência completa do `DataFrame` `dataset.metadata`. As colunas estão na
ordem em que são construídas dentro de `sample_observations`. Toda observação (uma linha) carrega
sua proveniência gerativa completa — é o que torna cada amostra sintética rastreável e o dataset
inteiro reauditável.

| Coluna | Tipo | Significado |
|---|---|---|
| `split` | `str` | nome do split passado em `split_name` |
| `identity_id` | `int` | identidade sintética (com `identity_offset` aplicado) |
| `local_identity_index` | `int` | índice local na população (0-based, antes do offset) |
| `sample_index` | `int` | índice da observação dentro da identidade |
| `session_index` | `int` | sessão sintética da observação |
| `population_component` | `int` | componente populacional latente \(k\) da identidade |
| `acquisition_domain` | `str` | domínio de aquisição amostrado (ou forçado via `forced_domain`) |
| `radial_regime` | `str` | regime radial: `central`, `low_tail` ou `high_tail` |
| `latent_quality` | `float` | qualidade sintética em \([0,1]\) |
| `embedding_norm` | `float` | norma do embedding bruto \(r_{ij}\) |
| `log_embedding_norm` | `float` | log-norma \(\log r_{ij}\) (após clipping de `minimum_norm`/`maximum_norm`) |
| `reference_scaled_log_norm_z` | `float` | posição radial padronizada \(z=(\log r-\mu_{\log})/\sigma_{\log}\) |
| `robust_norm_z` | `float` | **alias de** `reference_scaled_log_norm_z` (mesma coluna, também disponível sob este nome) |
| `radial_reference_method` | `str` | método da referência radial (`RadialConfig.reference_method`) |
| `intra_kappa` | `float` | concentração vMF intra-identidade \(\kappa_{ij}\) |
| `own_center_cosine` | `float` | cosseno ao centro latente **após** o passo de nuisance/viés |
| `pre_nuisance_center_cosine` | `float` | cosseno ao centro latente **antes** do passo de nuisance/viés |
| `intra_class_sampling_mode` | `str` | modo de dispersão usado: `vmf`, `idperturb_cap` ou `hybrid_vmf_cap` |
| `cap_lower_bound_effective` | `float` | piso de cosseno `lb` efetivo por amostra (`NaN` no modo `vmf`) |
| `hybrid_resample_count` | `int` | nº de reamostragens vMF por rejeição (`0` fora de `hybrid_vmf_cap`) |
| `hybrid_used_cap_fallback` | `bool` | indicador de uso do fallback de calota (`False` fora de `hybrid_vmf_cap`) |
| `tangent_update_mode` | `str` | `retraction` ou `exponential_map` |
| `anisotropic_noise_norm` | `float` | magnitude do nuisance anisotrópico aplicado |
| `fixed_domain_bias_norm` | `float` | magnitude do bias médio (fixo) de domínio |
| `domain_identity_bias_norm` | `float` | magnitude da interação domínio × identidade |
| `session_bias_norm` | `float` | magnitude do bias de sessão |
| `combined_tangent_step_norm` | `float` | tamanho do passo tangencial total que moveu a direção |
| `source_identity` | `str` | rótulo de identidade real de origem (**somente** em populações ancoradas, Seção 14) |
| `source_sample_count` | `int` | nº de imagens reais que formaram o centro (**somente** em populações ancoradas) |

#### Colunas de dispersão intra-classe e da calota

As cinco colunas `intra_class_sampling_mode`, `cap_lower_bound_effective`,
`pre_nuisance_center_cosine`, `hybrid_resample_count` e `hybrid_used_cap_fallback` estão
**sempre presentes**, inclusive no modo `vmf`. No modo `vmf`, `cap_lower_bound_effective` é
`NaN`, `hybrid_resample_count` é `0` e `hybrid_used_cap_fallback` é `False` — o que torna os três
campos auditáveis como "calota não utilizada" sem ambiguidade. Já `pre_nuisance_center_cosine` é
populada nos três modos: registra o cosseno ao centro **imediatamente após** a amostragem
vMF/calota/híbrido e **antes** da aplicação do passo tangencial de nuisance/viés.

`hybrid_resample_count` e `hybrid_used_cap_fallback` são os auditores específicos do modo
`hybrid_vmf_cap`. Uma taxa alta de `hybrid_used_cap_fallback=True` significa que o piso `lb` está
ativo a ponto de o corpo vMF raramente respeitá-lo por conta própria — o resultado é então
dominado pela calota, não pela concentração vMF configurada. Reportar a fração de fallback e a
distribuição de `hybrid_resample_count` é o diagnóstico que separa "vMF com piso de segurança" de
"calota disfarçada".

#### Invariante crítico para auditar a calota

\[
\texttt{pre\_nuisance\_center\_cosine} \;\ge\; \texttt{cap\_lower\_bound\_effective}.
\]

Esse é o invariante que prova que o piso geométrico `lb` foi respeitado na geração. O piso é
garantido **na direção pré-nuisance**. O passo tangencial de nuisance/viés aplicado em seguida
pode empurrar o `own_center_cosine` (pós-nuisance) **abaixo** de `lb` — isso é esperado e não
viola nada. Por isso, **para auditar a calota use sempre `pre_nuisance_center_cosine`**, nunca
`own_center_cosine`. A frase defensável em laudo é: *"nenhuma observação sintética desta
identidade foi gerada com cosseno menor que `cap_lower_bound_effective` ao centro latente, antes
dos ruídos de domínio/sessão."*

#### Colunas de proveniência (somente populações ancoradas em dataset real)

As colunas `source_identity` e `source_sample_count` aparecem **apenas** quando o dataset foi
observado a partir de uma população ancorada em dataset real (Modo A, Seção 14), ou seja, quando
`SyntheticIdentityPopulation.source_identity_labels` / `source_sample_counts` estão preenchidos.
Em populações puramente amostradas da mistura vMF essas colunas estão **ausentes** — pela simples
presença/ausência da coluna o auditor sabe se há âncoras biométricas reais por trás do dataset.

`source_identity` carrega o rótulo de identidade real do FaceAudit que originou cada centro;
`source_sample_count` registra quantas imagens reais formaram a média esférica daquele centro.
Como populações ancoradas geram variação sintética em torno de âncoras biométricas **reais** — e
não anonimizam nada —, essas duas colunas são o gancho de cadeia de custódia para verificar que o
conjunto de âncoras permanece disjunto de qualquer conjunto de avaliação (ver
`assert_disjoint_source_identities`, Seção 14).

### 15.3 Campos do `SyntheticIdentityPopulation`

O objeto latente `SyntheticIdentityPopulation` (Seção 12) é a população de identidades que pode
ser observada sob uma ou mais condições. Além dos campos estruturais obrigatórios, carrega campos
auxiliares opcionais que sustentam a guarda do bissetor (cache de overlap) e a proveniência. Esses
campos opcionais **não** são serializados — ficam fora de `asdict`/`save_manifest`/`save_npz` por
serem vetores grandes ou caches derivados (`repr=False`, `compare=False`).

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `identity_ids` | `IntArray` | obrigatório | ids das identidades (com `identity_offset` aplicado) |
| `population_components` | `IntArray` | obrigatório | componente populacional \(k\) por identidade |
| `identity_centers` | `FloatArray` | obrigatório | centro angular latente \(\mathbf{c}_i\) por identidade (unitário) |
| `identity_specific_nuisance_bases` | `FloatArray` ou `None` | obrigatório | base de nuisance específica por identidade (quando habilitada) |
| `domain_identity_bias_vectors` | `Mapping[str, FloatArray]` | obrigatório | bias domínio × identidade |
| `sampling_seed` | `int` | obrigatório | seed efetiva da amostragem da população |
| `stream_name` | `str` | obrigatório | nome do stream nomeado de RNG |
| `overlap_lower_bound_floor` | `FloatArray` ou `None` | `None` | piso de cosseno do bissetor por identidade (cache preguiçoso, não serializado) |
| `source_identity_labels` | `list[str]` ou `None` | `None` | rótulo de identidade real por centro (Modo A; não serializado) |
| `source_sample_counts` | `IntArray` ou `None` | `None` | nº de imagens reais por centro (Modo A; não serializado) |

O campo `overlap_lower_bound_floor` é calculado preguiçosamente na primeira observação em modo de
calota (via `_overlap_lower_bound_floor`, em blocos de linhas, sem materializar a matriz
\(C\times C\) de cossenos) e **cacheado** para reuso entre observações cross-condition da mesma
população — isso garante que o piso anti-colisão seja idêntico em todos os domínios alinhados. Os
campos `source_identity_labels` e `source_sample_counts` são `None` em populações amostradas e
preenchidos apenas no caminho de injeção de centros (Modo A); é deles que `sample_observations`
deriva as colunas `source_identity` e `source_sample_count` da Seção 15.2.

`validate()` reverifica que `identity_ids` e `population_components` têm uma entrada por
identidade, que `identity_centers` são unitários (`atol=1e-8`), e que as formas das bases de
nuisance e dos vetores de bias domínio × identidade são compatíveis com `(identity_count,
dimension)`.

---

## 16. Diagnósticos

O `SyntheticEmbeddingDataset` carrega diagnósticos pensados para auditoria. Dois são centrais: o
resumo radial em U-invertido (`norm_bin_diagnostics`) e a regra de ouro dos percentis de
`cap_lower_bound_effective`.

### 16.1 `norm_bin_diagnostics` — o padrão de U-invertido

O método `norm_bin_diagnostics` estratifica as observações por quantis de norma e resume o padrão
esperado de **U-invertido** entre norma, qualidade e estabilidade angular: as observações de norma
intermediária tendem a ter maior qualidade latente e maior \(\kappa\) intra-classe (direções mais
estáveis, mais próximas do centro), enquanto as caudas radiais (`low_tail`/`high_tail`) tendem a
ser de menor qualidade e direção mais dispersa. É o diagnóstico que confirma que o acoplamento
não-monotônico norma↔confiabilidade está ativo conforme configurado.

```python
diagnostics = test.norm_bin_diagnostics(bin_count=10)
print(diagnostics)
```

Colunas retornadas:

| Coluna | Significado |
|---|---|
| `requested_bin_count` | quantidade de bins solicitada |
| `actual_bin_count` | quantidade efetivamente produzida |
| `norm_quantile_bin` | faixa de norma (intervalo) |
| `sample_count` | tamanho do grupo |
| `norm_mean` | norma média |
| `reference_scaled_log_norm_z_mean` | posição radial média |
| `latent_quality_mean` | qualidade média |
| `intra_kappa_mean` | estabilidade angular média |
| `own_center_cosine_mean` | proximidade média ao centro |
| `own_center_cosine_median` | proximidade mediana ao centro |
| `tangent_step_norm_mean` | passo tangencial médio |

O método usa `pd.qcut(..., duplicates="drop")`. Se existirem bordas de quantil repetidas, o
pandas produzirá **menos** bins do que o solicitado; nesse caso o método **emite um `UserWarning`**
e informa tanto `requested_bin_count` quanto `actual_bin_count`, evitando que um diagnóstico
silenciosamente degradado seja interpretado como completo. O parâmetro `bin_count` deve ser
\(\ge 2\). Esse diagnóstico também é gravado automaticamente por `save_bundle` no CSV
`<stem>_norm_diagnostics.csv` (Seção 18), usando `bin_count=10`.

### 16.2 Sempre reporte os percentis de `cap_lower_bound_effective`

Nos modos de calota (`IDPERTURB_CAP` e `HYBRID_VMF_CAP`), o piso `lb` **nominal** configurado em
`DirectionalConfig.cap_base_lower_bound` **não** é, em geral, o piso realmente aplicado. Com
`cap_overlap_guard=True` (padrão), a regra do bissetor impõe um piso por identidade a partir da
densidade local dos centros, e **essa guarda tem prioridade**: aplicada **após** o clip, ela pode
empurrar o `lb` efetivo **acima** de `cap_maximum_lower_bound` em regiões densas da população. Em
512-D, mesmo centros aleatórios têm vizinhos relativamente próximos, então um `lower_bound`
nominal de, por exemplo, `0.4` pode corresponder a um piso efetivo bem maior para identidades
típicas.

Por isso, a regra de ouro do diagnóstico por cap é: **sempre reporte os percentis empíricos de
`cap_lower_bound_effective` (mean / p10 / p50 / p90) lado a lado com o `lb` nominal.** O `lb`
nominal documenta a *intenção*; os percentis do efetivo documentam a *realização*. A separação
genuína entre identidades reflete o efetivo, não o nominal. O `lb` nominal é um piso, não o valor
realizado.

```python
import numpy as np

eff = test.metadata["cap_lower_bound_effective"].to_numpy()
eff = eff[~np.isnan(eff)]  # ignora linhas do modo vmf
print({
    "nominal_lb": test.config.directional.cap_base_lower_bound,
    "effective_mean": float(eff.mean()),
    "p10": float(np.percentile(eff, 10)),
    "p50": float(np.percentile(eff, 50)),
    "p90": float(np.percentile(eff, 90)),
})
```

No modo `hybrid_vmf_cap`, acrescente ao relatório a fração de
`hybrid_used_cap_fallback == True` e a distribuição de `hybrid_resample_count` (Seção 15.2): juntos,
eles revelam se o resultado foi de fato dominado pelo corpo vMF ou pela calota de segurança.

---

## 17. Geração de pares

A função `sample_pair_table` produz uma tabela de pares (genuínos e impostores aleatórios) já
escorada por cosseno, pronta para alimentar avaliação de backend (PLDA, calibração, EER, Cllr).
Ela consome um `SyntheticEmbeddingDataset` e devolve um `pandas.DataFrame`.

```python
from adaface_embedding_simulator import sample_pair_table, PairSamplingStrategy

pairs = sample_pair_table(
    test,
    genuine_pair_count=20_000,
    random_impostor_pair_count=20_000,
    random_seed=123,
    strategy=PairSamplingStrategy.BALANCED_IDENTITY,
    unique_pairs=False,
)
```

### 17.1 Parâmetros

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dataset` | `SyntheticEmbeddingDataset` | obrigatório | dataset cujas observações serão pareadas |
| `genuine_pair_count` | `int` | obrigatório | nº de pares genuínos (mesma identidade) a amostrar |
| `random_impostor_pair_count` | `int` | obrigatório | nº de pares impostores (identidades distintas) a amostrar |
| `random_seed` | `int` | `123` | seed do RNG local da amostragem de pares |
| `strategy` | `PairSamplingStrategy` ou `str` | `BALANCED_IDENTITY` | estratégia de ponderação dos impostores |
| `unique_pairs` | `bool` | `False` | se `True`, deduplica pares não-ordenados após a amostragem |

Contagens negativas levantam erro. Os genuínos exigem ao menos uma identidade com \(\ge 2\)
amostras; os impostores exigem ao menos duas identidades — caso contrário a função levanta erro
explícito.

### 17.2 Estratégias de amostragem (`PairSamplingStrategy`)

Em ambas as estratégias os **pares genuínos são identity-balanced**: cada identidade elegível
(com \(\ge 2\) amostras) recebe peso aproximadamente igual, e os dois índices internos são
sorteados sem repetição dentro da identidade. As estratégias diferem apenas na ponderação dos
**impostores**:

- **`BALANCED_IDENTITY`** (padrão) — atribui peso de trial aproximadamente igual a cada
  identidade. Os impostores são formados sorteando primeiro duas identidades distintas (com peso
  uniforme por identidade) e, em seguida, uma observação dentro de cada uma. Resultado: identidades
  com poucas amostras não ficam sub-representadas nos impostores.
- **`BALANCED_OBSERVATION`** — amostra impostores aproximadamente em proporção ao número de
  embeddings observados, sorteando duas linhas quaisquer do dataset e rejeitando colisões de
  identidade. Resultado: identidades com muitas observações dominam os impostores, refletindo a
  distribuição empírica de observações.

### 17.3 `unique_pairs`

Quando `unique_pairs=True`, a tabela é deduplicada por par não-ordenado \(\{i,j\}\) (ordena
`min`/`max` das linhas e remove duplicatas, preservando o rótulo). Como a deduplicação remove
trials repetidos, o `DataFrame` retornado pode conter **menos** linhas do que
`genuine_pair_count + random_impostor_pair_count`; nesse caso a função **emite um `UserWarning`**.
Use `unique_pairs=True` quando a independência entre trials importar (por exemplo, para não
inflar artificialmente a contagem efetiva de pares distintos numa métrica).

### 17.4 Colunas de saída

| Coluna | Significado |
|---|---|
| `left_row` | índice da linha esquerda no `dataset.metadata` |
| `right_row` | índice da linha direita |
| `is_genuine` | `1` para par genuíno (mesma identidade), `0` para impostor |
| `cosine_score` | escore de cosseno entre as duas direções normalizadas |
| `left_identity_id` / `right_identity_id` | ids de identidade dos dois lados |
| `left_norm` / `right_norm` | normas dos embeddings brutos dos dois lados |
| `left_reference_scaled_log_norm_z` / `right_reference_scaled_log_norm_z` | posição radial padronizada de cada lado |
| `left_robust_norm_z` / `right_robust_norm_z` | **alias** das colunas `*_reference_scaled_log_norm_z` (mesmo valor sob nome alternativo) |
| `left_quality` / `right_quality` | qualidade latente de cada lado |

Além das colunas, o `DataFrame` registra dois metadados em `result.attrs`: `sampling_strategy`
(o valor da estratégia efetivamente usada) e `unique_pairs` (booleano). Os pares de colunas de
covariáveis (`norm`, `reference_scaled_log_norm_z`, `quality`) por lado existem justamente para
permitir calibração e diagnóstico condicionados em norma e qualidade.

---

## 18. Serialização e persistência

A persistência é desenhada para ser **round-trippável e auditável**: a configuração inteira
sobrevive a um ciclo JSON sem perda semântica, e o bundle de saída combina arrays, metadados,
diagnóstico radial e um manifesto JSON com seeds e hash de fonte.

### 18.1 Round-trip da configuração

```python
from adaface_embedding_simulator import config_to_dict, config_from_dict, config_from_manifest

blob = config_to_dict(config)          # dicts JSON-ready (enums como string, arrays como listas)
config_again = config_from_dict(blob)  # reconstrói e VALIDA
config_from_disk = config_from_manifest("run_manifest.json")  # lê o bloco "config" do manifesto
```

- **`config_to_dict(config)`** — converte um `SimulatorConfig` em dicionários aninhados
  JSON-ready: enums viram seus valores string, arrays NumPy viram listas Python, escalares NumPy
  viram escalares Python nativos. O resultado é diretamente serializável com `json.dumps`.
- **`config_from_dict(data)`** — fecha o round-trip: reconstrói o `SimulatorConfig` a partir do
  dicionário e **chama `validate()` antes de retornar**. Campos de array permanecem como listas
  (todo consumidor chama `np.asarray` internamente, então a geração reproduz); campos enum
  permanecem como string (os DTOs e o simulador aceitam ambos).
  - **Chaves faltantes caem em default**: a reconstrução usa os defaults do dataclass para chaves
    ausentes (por exemplo o bloco `coupling` inteiro é opcional). Isso torna a leitura tolerante a
    configurações escritas com menos campos.
  - **Chaves desconhecidas erram**: passar uma chave que nenhum dataclass reconhece levanta
    `TypeError`, surfaceando incompatibilidades cedo em vez de ignorá-las silenciosamente.
- **`config_from_manifest(path)`** — abre um manifesto JSON escrito por `save_manifest`, extrai o
  bloco `"config"` e delega a `config_from_dict`. Levanta erro se o manifesto não tiver o bloco
  `config`. É o caminho canônico para reconstruir a configuração exata de uma execução passada a
  partir do seu manifesto.

### 18.2 Persistência do dataset

Quatro métodos do `SyntheticEmbeddingDataset` cobrem desde a gravação granular até o bundle
completo:

```python
test.save_npz("out/test_embeddings.npz")
test.save_metadata_csv("out/test_metadata.csv")
test.save_manifest("out/test_manifest.json", extra={"experiment": "exp_42"})
paths = test.save_bundle("out", stem="test")   # npz + csv + manifesto + diagnóstico radial
```

| Método | O que grava |
|---|---|
| `save_npz(path)` | arquivo NumPy comprimido com `raw_embeddings`, `normalized_embeddings`, `identity_centers`, `population_component_means`, `anisotropic_basis` (matriz vazia `(D, 0)` quando `None`), os nomes de domínio ordenados, a matriz empilhada de `domain_bias_vectors` e os registros do metadata |
| `save_metadata_csv(path)` | apenas o `DataFrame` `metadata` como CSV (sem índice) |
| `save_manifest(path, extra=None)` | manifesto JSON com proveniência completa (ver Seção 18.3) |
| `save_bundle(output_dir, stem)` | conveniência: grava `<stem>_embeddings.npz`, `<stem>_metadata.csv`, `<stem>_manifest.json` e o diagnóstico radial `<stem>_norm_diagnostics.csv` (`norm_bin_diagnostics(bin_count=10)`); retorna um `dict` com os quatro `Path` sob as chaves `npz`, `metadata`, `manifest`, `norm_diagnostics` |

Todos os métodos criam os diretórios pais que faltarem e retornam o `Path` (ou `dict` de
`Path`) escrito.

### 18.3 O manifesto como artefato de proveniência

`save_manifest` escreve um JSON indentado com o seguinte conteúdo, projetado como o **artefato de
cadeia de custódia** de uma execução:

| Chave | Significado |
|---|---|
| `simulator_version` | versão do gerador |
| `profile_name` | nome do perfil da configuração |
| `structural_seed` | seed estrutural |
| `population_sampling_seed` | seed da amostragem da população |
| `observation_sampling_seed` | seed da amostragem das observações |
| `stream_name` | nome do stream de RNG das observações |
| `row_count` / `dimension` | nº de observações e dimensão dos embeddings |
| `identity_count` | nº de identidades distintas no metadata |
| `source_sha256` | SHA-256 do próprio arquivo-fonte do simulador (quando acessível) |
| `config` | a configuração integral via `config_to_dict` (re-lível por `config_from_manifest`) |
| `extra` | dicionário arbitrário do usuário, normalizado para JSON (`_json_ready`) |

Como o manifesto carrega o bloco `config` completo e as três seeds, ele é **round-trippável**:
`config_from_manifest` reconstrói a configuração exata, e reinstanciar o simulador com a mesma
`structural_seed` reproduz os artefatos estruturais; com as mesmas seeds de população e observação
o dataset inteiro é reproduzido bit-a-bit. O `source_sha256` permite verificar que o código-fonte
usado na geração não mudou entre a execução e a auditoria.

---

## 19. Adapters para o FaceAudit

O pacote oferece dois caminhos de integração com o FaceAudit, com papéis distintos: um **scaffold
genérico de referência** (`faceaudit_adapter`), que mostra a forma da conversão sem acoplar ao
projeto hospedeiro, e o **adapter concreto** (`faceaudit_concrete_adapter`), que é a ponte real de
produção, importando os DTOs do FaceAudit e mapeando os campos corretamente.

### 19.1 O scaffold genérico `faceaudit_adapter` (referência, NÃO usar em produção)

`faceaudit_adapter` é um scaffold de **referência por injeção de dependência**: os construtores do
FaceAudit (`EmbeddingInfo`, `EmbeddingDataset`) não são importados; são **injetados** como
callables. Ele expõe:

- **`dataset_to_embedding_records(dataset, *, embedding_view="normalized", domain_filter=None)`**
  — converte o dataset sintético em uma lista de `dict`s simples (`embedding`, `identity`,
  `metadata`), opcionalmente filtrando por domínio de aquisição. `embedding_view` escolhe a matriz
  bruta (`"raw"`) ou normalizada (`"normalized"`).
- **`to_faceaudit_embedding_dataset(dataset, *, embedding_info_factory, embedding_dataset_factory,
  ...)`** — monta DTOs chamando os construtores injetados.

> **Caveat decisivo — não casa com o DTO real.** O scaffold assume
> `EmbeddingInfo(embedding=, identity=, metadata=)`, isto é, um `EmbeddingInfo` que aceita um
> kwarg `metadata`. **O `EmbeddingInfo` real do FaceAudit NÃO tem esse kwarg** (sua assinatura é
> `EmbeddingInfo(embedding, face_hash, image_path, identity, file_name, ...)`). Portanto
> `faceaudit_adapter` serve **apenas como referência didática** da forma da conversão — não deve
> ser usado para produzir datasets reais do FaceAudit. Para produção, use o adapter concreto
> (Seção 19.2).

### 19.2 O adapter concreto `faceaudit_concrete_adapter` (produção)

`faceaudit_concrete_adapter` é a ponte real: importa `EmbeddingDataset` e `EmbeddingInfo` do
FaceAudit e mapeia as colunas de metadata do simulador sobre os campos reais do `EmbeddingInfo`.
Expõe duas funções.

#### `to_embedding_dataset`

```python
from simulator.faceaudit_concrete_adapter import to_embedding_dataset

fa_dataset = to_embedding_dataset(
    test,
    name="synthetic_test",
    embedding_view="raw",      # padrão; norma preservada
    quality_key="generic",
    origin="adaface_simulator",
    domain_filter=None,
    dtype=np.float32,           # cast padrão para casar com datasets extraídos reais
)
```

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---|---|
| `dataset` | `SyntheticEmbeddingDataset` | obrigatório | saída de `sample_observations` / `sample_dataset` |
| `name` | `str` ou `None` | `None` | nome do dataset (default `f"synthetic_{split}"`) |
| `embedding_view` | `"raw"` ou `"normalized"` | `"raw"` | matriz fonte; `"raw"` preserva a norma |
| `quality_key` | `str` | `"generic"` | chave sob a qual `latent_quality` é gravada em `quality_scores` |
| `origin` | `str` | `"adaface_simulator"` | valor do campo `origin` de cada item |
| `domain_filter` | `str` ou `None` | `None` | se definido, mantém só as linhas cujo `acquisition_domain` casa |
| `dtype` | `np.dtype` ou `type` | `np.float32` | dtype dos embeddings gravados |

Mapeamento de campos (covariáveis forensicamente acionáveis preservadas):

- **`embedding`** ← `raw_embeddings` (padrão, norma preservada) ou `normalized_embeddings`,
  **cast para `float32`** por padrão para casar com datasets reais extraídos.
- **`identity`** ← `str(identity_id)`.
- **`quality_scores[quality_key]`** ← `latent_quality` (bucket `"generic"` por padrão, que é o que
  a property `quality_score` e o balanceamento BY_QUALITY do FaceAudit leem).
- **`source`** ← `acquisition_domain` (handle por amostra usado em splits cross-condition).
- **`origin`** ← tag de proveniência (`"adaface_simulator"` por padrão).
- **`face_hash`** ← **SHA-256 determinístico** de `(split, identity_id, sample_index,
  session_index)` — um hash estável e reproduzível por observação, não um valor aleatório.
- **`image_path`** / **`file_name`** ← caminhos sintéticos determinísticos
  (`synthetic/{split}/{identity_id}/{sample_index}.png` e `{identity_id}_{sample_index}.png`).
- **`face_image`** ← `None`; **`pose`** deixado vazio (o simulador modela nuisance, não pose
  literal de cabeça).

O `EmbeddingDataset` resultante recebe um `loaded_metadata` de proveniência com `generator`,
`simulator_version`, `profile_name`, `split`, `embedding_view`, `dimension`, `n_samples`,
`n_identities` (e `domain_filter`, quando aplicado). As covariáveis contínuas que não entram no
`EmbeddingInfo` (`reference_scaled_log_norm_z`, `intra_kappa`, `radial_regime`, diagnósticos de
passo tangencial, etc.) permanecem nos metadados do simulador e são persistidas pelo `save_bundle`
como sidecars CSV (Seção 18) — nada essencial para estratificação se perde, pois a norma L2 é
recuperável a partir do embedding bruto.

#### `cross_condition_to_embedding_datasets`

```python
from simulator.faceaudit_concrete_adapter import cross_condition_to_embedding_datasets

fa_by_domain = cross_condition_to_embedding_datasets(
    datasets_by_domain,        # saída de sample_cross_condition_datasets
    embedding_view="raw",
    quality_key="generic",
    origin="adaface_simulator",
    dtype=np.float32,
)
```

Mapeia o `dict` de saída de `sample_cross_condition_datasets` (Seção 13) para um `dict` de
`EmbeddingDataset` do FaceAudit, aplicando `to_embedding_dataset` a cada domínio com nome
`f"synthetic_{domain_name}"`. Os datasets resultantes mantêm valores de **`identity` idênticos
entre condições**, formando um conjunto cross-condition pareado: cada identidade produz pares
genuínos cross-condition, exatamente o insumo de que o treino e a calibração PLDA cross-condition
do FaceAudit precisam.
## 20. Experimentos recomendados

Esta seção descreve um conjunto de experimentos canônicos. O objetivo nunca é
obter um único número sintético para "provar" que um backend funciona, e sim
**isolar regimes**: em quais condições o PLDA ganha do cosseno, quando as
hipóteses homocedásticas começam a falhar, e quanto a dispersão intra-classe
afeta a separabilidade. Sempre use identidades disjuntas entre treino,
desenvolvimento e teste, e sempre registre o perfil + a configuração completa
(ver Seção 25).

Os perfis prontos citados aqui — `make_spherical_baseline_config`,
`make_cooperative_plda_config`, `make_cross_condition_stress_config` e
`make_idperturb_cap_config` — estão descritos em detalhe na Seção 10.

### 20.1 Baseline esférico (PLDA / calibração)

Objetivo:

> comparar cosseno, PLDA e PSDA em geometria esférica simples.

```python
from adaface_embedding_simulator import make_spherical_baseline_config

config = make_spherical_baseline_config(dimension=512)
```

Comparar:

| Sistema | Entrada |
|---|---|
| Cosseno | `normalized_embeddings` |
| PLDA bruto | `raw_embeddings` |
| PLDA L2 | `normalized_embeddings` |
| PSDA | `normalized_embeddings` |

Interpretação: no `SPHERICAL_BASELINE` não existe estrutura anisotrópica
relevante nem heterocedasticidade de domínio. Se um backend introduz ganho
expressivo aqui, suspeite de vazamento ou de superajuste ao próprio mundo
sintético. Para a calibração, esse é o cenário onde a curva de calibração
(raw × log-LR) deve ficar próxima de afim e o `Cllr` próximo do `Cllr_min`
in-sample — qualquer desvio grande sugere problema de pipeline, não do mundo.

### 20.2 Baseline cooperativo (PLDA / calibração)

Objetivo:

> verificar se LDA, whitening e PLDA recuperam uma estrutura compartilhada
> aprendível.

```python
from adaface_embedding_simulator import make_cooperative_plda_config

config = make_cooperative_plda_config(dimension=512)
```

Interpretação correta:

- um resultado positivo mostra que o **pipeline** funciona quando a estrutura é
  aprendível;
- **não** demonstra desempenho provável em qualquer base AdaFace real.

O perfil `COOPERATIVE_PLDA_BASELINE` é um cenário potencialmente otimista e
cooperativo para PLDA — ele **não** é um limite superior matemático (ver Seção
22.1).

### 20.3 Stress cross-condition (PLDA / calibração)

Objetivo:

> avaliar os limites do PLDA homocedástico e comparar estratégias de estimação
> cross-condition.

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
    identity_count=1_000,
    identity_offset=300_000,
    split_name="cross_test",
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

Comparar estratégias de estimação cross-condition do PLDA:

```text
merge ingênuo
PER_CONDITION
BALANCED_IDENTITY
```

Métricas:

- EER;
- `Cllr`;
- `Cllr_min` (referência diagnóstica in-sample, não operacional);
- Tippett;
- estabilidade de LR;
- espectro de \(W\);
- traço de \(W\);
- desempenho por domínio;
- desempenho por quantis de norma;
- desempenho por tipicidade.

O perfil `CROSS_CONDITION_HETEROSCEDASTIC` usa nuisance hierárquico, bases por
domínio, bases específicas por identidade, viés de domínio, viés domínio ×
identidade, viés de sessão e mapa exponencial. É o cenário projetado para
**quebrar** o PLDA homocedástico de forma controlada — exatamente onde a
calibração afim sozinha tende a falhar e a estratificação por domínio passa a
importar.

### 20.4 Tipicidade

Compare estimadores de tipicidade sobre uma mesma população:

1. tipicidade-oráculo sobre os centros latentes (`dataset.identity_centers`);
2. MoVMF ajustada sobre centroides observados;
3. k-NN `PER_IDENTITY`;
4. k-NN `PER_FACE`;
5. estratificação por qualidade e por norma.

Pergunta importante:

> quanto a estimação da tipicidade muda quando os centroides observados são
> contaminados por amostras extremas (caudas radiais, baixa qualidade)?

A tipicidade-oráculo é a única que conhece os centros verdadeiros; toda a
distância entre ela e os estimadores empíricos é viés de estimação. Como a coluna
`pre_nuisance_center_cosine` registra o cosseno ao centro **antes** do passo de
nuisance, ela serve de referência interna para medir o quanto o nuisance degrada
a tipicidade observada.

### 20.5 Experimento de sweep de `lb` (modos de calota)

Este experimento usa o eixo `intra_class_sampling_mode` para varrer a **dispersão
intra-classe** de forma interpretável. Em vez de varrer `κ` (difícil de
interpretar em 512-D), varre-se o piso de cosseno `lb` do modo
`IDPERTURB_CAP`.

Gere a varredura em duas versões, exatamente como na Seção 16:

**(A) guarda desligada + sem nuisance** — a tendência é limpa:

\[
lb\downarrow \;\Rightarrow\; \text{dispersão genuína}\uparrow \;\Rightarrow\; \text{EER}\uparrow.
\]

```python
from dataclasses import replace

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_idperturb_cap_config,
    sample_pair_table,
)

records = []
for lb in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
    config = make_idperturb_cap_config(dimension=512, lower_bound=lb)
    # (A) guarda off + sem nuisance anisotrópico
    directional = replace(
        config.directional,
        cap_overlap_guard=False,
        anisotropic_base_scale=0.0,
        cap_quality_lower_bound_slope=0.0,
        cap_low_tail_penalty=0.0,
        cap_high_tail_penalty=0.0,
    )
    config = replace(config, profile_name=f"LB_SWEEP_A_{lb}", directional=directional)

    simulator = AdaFaceEmbeddingSimulator(config, random_seed=7000)
    test = simulator.sample_dataset(
        identity_count=1_000,
        samples_per_identity=(4, 8),
        split_name="lb_sweep",
        identity_offset=400_000,
        population_random_seed=7001,
        random_seed=7002,
    )
    pairs = sample_pair_table(
        test,
        genuine_pair_count=50_000,
        random_impostor_pair_count=50_000,
        random_seed=7003,
    )
    eff = test.metadata["cap_lower_bound_effective"]
    records.append(
        {
            "lb_nominal": lb,
            "lb_eff_p10": float(eff.quantile(0.10)),
            "lb_eff_p50": float(eff.quantile(0.50)),
            "lb_eff_p90": float(eff.quantile(0.90)),
            "genuine_cos_mean": float(
                pairs.loc[pairs["is_genuine"] == 1, "cosine_score"].mean()
            ),
        }
    )
```

**(B) guarda ligada** — repita o laço com `cap_overlap_guard=True`. Aqui a curva
pode **achatar** em regiões densas, porque o piso efetivo deixa de seguir `lb`
nominal: em 512-D, mesmo centros aleatórios têm vizinhos relativamente próximos,
então a guarda do bissetor eleva o `lb` efetivo. **Sempre reporte os percentis de
`cap_lower_bound_effective` (p10/p50/p90)** ao lado do `lb` nominal — é a única
forma de mostrar a separação realizada. Com a guarda ligada, um `lb` nominal de
`0.4` **não** significa que as amostras foram geradas com piso `0.4`.

Comparação esperada entre as duas versões da varredura:

| Variante | Guarda | Curva `EER × lb` | `lb` efetivo |
|---|---|---|---|
| A | off | monotônica e limpa | ≈ `lb` nominal |
| B | on | achatada em regiões densas | ≥ `lb` nominal (reportar percentis) |

O piso por densidade é exatamente o que liga a calota à tipicidade: identidades
em regiões densas (típicas) recebem piso maior e toleram menos variação intra;
identidades raras toleram mais. O sweep com a guarda ligada mede esse efeito de
forma realista; o sweep com a guarda desligada mede o efeito geométrico puro de
`lb`.

### 20.6 Experimento Modo A vs Modo B

Compare as duas formas de ancoragem em dados reais (ver Seção 14) no MESMO
backend, mantendo as identidades de ancoragem disjuntas das de avaliação:

- **Modo A** (`population_from_dataset`): os centros das identidades **são** as
  médias esféricas reais por identidade. As identidades são reais; só a dispersão
  intra-classe e a norma são sintéticas.
- **Modo B** (`simulator_config_from_dataset`): ajusta uma MoVMF aos centros
  reais e amostra **novos** ghost centres dessa aproximação angular de primeira
  ordem.

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)
from faceaudit_seed import (
    population_from_dataset,
    simulator_config_from_dataset,
    assert_disjoint_source_identities,
)

# Modo A — âncoras reais (anchor_dataset DISJUNTO do conjunto de avaliação)
sim_a = AdaFaceEmbeddingSimulator(
    make_cooperative_plda_config(dimension=anchor_dataset.matrix.shape[1]),
    random_seed=11,
)
pop_a = population_from_dataset(sim_a, anchor_dataset, min_samples=3)

# Modo B — população distribuição-matched
config_b, fit_summary_b = simulator_config_from_dataset(
    anchor_dataset, criterion="bic"
)
sim_b = AdaFaceEmbeddingSimulator(config_b, random_seed=11)
pop_b = sim_b.sample_identity_population(
    identity_count=len(pop_a.identity_ids),
    identity_offset=900_000,
    stream_name="mode_b_ghosts",
)
```

Pergunta importante:

> o quanto as conclusões sobre PLDA (ganho vs cosseno, estabilidade de `W`) mudam
> quando a geometria populacional vem dos centros reais (Modo A) versus de uma
> aproximação MoVMF de primeira ordem (Modo B)?

Divergências grandes entre A e B sinalizam que a **estrutura populacional** —
não apenas a dispersão intra-classe — está dirigindo o resultado, e que a MoVMF
de primeira ordem não captura a geometria empírica relevante (ver Seção 22.2).

### 20.7 Matriz de ablações causais

Para atribuir causalmente cada mudança de desempenho, cruze este eixo
experimental com a matriz de acoplamentos e modos de dispersão descrita na
Seção 9 (`CouplingConfig` + `intra_class_sampling_mode`). O cruzamento
recomendado para um relatório completo é `{VMF, IDPERTURB_CAP, HYBRID_VMF_CAP} ×
{A0, A4, A6}`, reportando por célula as métricas da Seção 20.3 mais os percentis
de `cap_lower_bound_effective` nos modos de calota.

---

## 21. Calibração contra embeddings reais

> Os valores padrão dos perfis são ilustrativos. Antes de usar resultados
> sintéticos para afirmações quantitativas, ancore o simulador em uma base real
> representativa.

A calibração do gerador contra estatísticas reais é feita por ferramentas
dedicadas do módulo `faceaudit_seed` e por `RadialReference.fit`. Use-as em vez
de reproduzir as fórmulas manualmente — elas tratam pesos por qualidade, médias
degeneradas, seleção de K e o caso L2.

### 21.1 Estimar a referência radial real — `RadialReference.fit`

`RadialReference.fit` aceita diretamente a matriz `(N, D)` de embeddings brutos
reais (ou um vetor de normas) e estima localização e escala robustas:

\[
\text{localização} = \operatorname{mediana}(\log r),
\qquad
\text{escala} = 1{,}4826\cdot\operatorname{MAD}(\log r).
\]

```python
from adaface_embedding_simulator import RadialReference, RadialConfig

reference = RadialReference.fit(real_raw_embeddings)  # (N, D) ou vetor de normas
radial = RadialConfig.from_reference(
    reference,
    quality_log_norm_slope=0.12,
    low_tail_offset=-0.70,
    high_tail_offset=0.70,
)
```

`RadialConfig.from_reference` propaga `log_norm_location`, `log_norm_scale` e
`method` da referência para os campos `reference_log_norm_mean`,
`reference_log_norm_scale` e `reference_method` do `RadialConfig`, aceitando
overrides dos demais campos por keyword.

**Aviso L2:** se a base já estiver L2-normalizada (normas ≈ 1, dispersão ≈ 0), o
MAD escalado é zero e o ajuste é indefinido. Passe embeddings **brutos**, não
normalizados. O caminho automático (Seção 21.3) detecta esse caso e mantém a
referência radial-base com um warning, em vez de quebrar.

### 21.2 Estimar centros por identidade (Modo A) — `identity_centers_from_dataset`

Em vez de calcular a média esférica manualmente, use
`identity_centers_from_dataset`, que computa
\(\widehat{\mathbf{c}}_i = \operatorname{normalize}\big(\sum_j w_j \mathbf{u}_{ij}\big)\)
por identidade, com agrupamento pelo valor **original** de `identity` (evita
fundir `1` e `"1"`), filtro `min_samples`, ponderação opcional por qualidade e
fallback robusto para médias degeneradas:

```python
from faceaudit_seed import identity_centers_from_dataset

extraction = identity_centers_from_dataset(
    real_dataset,
    min_samples=3,            # 1 = âncora ruidosa; 2+ estável; 5+ melhor
    weight_by_quality=True,   # pondera cada imagem por quality_score
)
# extraction.centers (M, D), extraction.labels, extraction.counts
```

O retorno é um `IdentityCenterExtractionResult` (campos `centers`, `labels`,
`counts`, `quality_weighted`). Esses centros alimentam diretamente uma população
ancorada via `population_from_dataset` (Modo A — Seção 14).

### 21.3 Ajuste automático completo (Modo B) — `simulator_config_from_dataset`

Para um config completo distribuição-matched — população MoVMF ajustada aos
centros reais **mais** referência radial ajustada às normas reais — use o
caminho único `simulator_config_from_dataset`:

```python
from faceaudit_seed import simulator_config_from_dataset

config, fit_summary = simulator_config_from_dataset(
    real_dataset,
    fit_norm=True,          # também ajusta a referência radial
    criterion="bic",        # ou "aic" para selecionar K
)
print(fit_summary["selected_k"], fit_summary["norm_summary"])
```

Esse caminho parte do `make_cooperative_plda_config` na dimensão do dataset,
substitui a `PopulationConfig` pela MoVMF ajustada (perfil `FACEAUDIT_MOVMF`) e,
quando `fit_norm=True`, substitui a referência radial via
`RadialConfig.from_reference(RadialReference.fit(matrix))`. O `fit_summary` traz
K selecionado, valores de critério por K, número de componentes degenerados,
entropia dos pesos e o `norm_summary` (mean/std/min/max das normas,
`unit_norm_warning`, `radial_fitted`) — tudo persistível no manifesto para
auditoria. A subfunção `population_config_from_dataset` retorna o par
`(PopulationConfig, fit_summary)` quando só a população importa.

### 21.4 Estatísticas mínimas a reproduzir

Mesmo com as ferramentas automáticas, valide visualmente que o mundo sintético
reproduz as estatísticas reais de interesse:

| Fenômeno | Diagnóstico |
|---|---|
| norma | histogramas, quantis, mediana, MAD |
| norma por domínio | curvas separadas |
| genuínos | distribuição de cosseno e score PLDA |
| impostores | distribuição de cosseno e score PLDA |
| caudas | quantis extremos |
| intraidentidade | \(\widehat{\mathbf{c}}_i^T\mathbf{u}_{ij}\) (compare com `pre_nuisance_center_cosine`) |
| relação radial | estabilidade por quantil de norma |
| domínio | diferença controlled vs surveillance |
| estrutura anisotrópica | espectro de covariância intraidentidade |
| população | espectro de covariância entre identidades |
| tipicidade | densidade e vizinhanças |

### 21.5 Ajuste incremental e separação ajuste/avaliação

Não ajuste tudo simultaneamente. Sequência recomendada:

1. distribuição radial (Seção 21.1);
2. distribuição populacional impostora (Seção 21.3);
3. dispersão genuína central (`κ` ou `lb`);
4. penalizações de cauda;
5. diferenças de domínio;
6. nuisance anisotrópico global;
7. bases por domínio;
8. bias domínio × identidade;
9. sessões;
10. validação externa.

Use identidades **disjuntas** para cada papel:

| Partição | Uso |
|---|---|
| fit do simulador | estimar parâmetros sintéticos (centros, radial, MoVMF) |
| desenvolvimento | escolher hiperparâmetros do pipeline |
| teste | avaliação final |

No Modo A, garanta explicitamente a disjunção entre a base de ancoragem e a de
avaliação com `assert_disjoint_source_identities` (Seção 14) — é uma asserção
obrigatória do checklist da Seção 25.

---

## 22. Cuidados interpretativos forenses

Esta seção consolida todos os cuidados interpretativos. Nenhum resultado deste
simulador deve aparecer em laudo sem que estes limites estejam explícitos.

### 22.1 O baseline cooperativo não é um teto

O perfil `COOPERATIVE_PLDA_BASELINE` tende a favorecer modelos capazes de
aprender covariâncias compartilhadas. Isso **não** significa que exista uma
garantia matemática de desempenho superior ao observado em dados reais.

Use a formulação:

> cenário potencialmente otimista e cooperativo para PLDA.

Evite:

> limite superior garantido do PLDA.

### 22.2 O gerador não reproduz a rede AdaFace nem a variedade real

O gerador modela propriedades **selecionadas**: direção, concentração angular,
norma, caudas, qualidade latente, domínios e nuisance. Ele **não** reproduz
automaticamente:

- a arquitetura da rede AdaFace;
- a distribuição completa dos pixels;
- efeitos específicos de pré-processamento;
- vieses demográficos reais;
- artefatos de câmera reais;
- relações causais verdadeiras entre qualidade e reconhecibilidade.

Em particular, os **ghost centres do Modo B** (`simulator_config_from_dataset`)
são uma **aproximação angular de primeira ordem** da distribuição empírica dos
centros — uma mistura isotrópica por componente (MoVMF), **não** a variedade real
dos embeddings. Use a formulação recomendada: *MoVMF modela
população/tipicidade; vMF modela dispersão estatística; IDPERTURB-cap impõe um
limite geométrico interpretável e citável para a variação intra-identidade.*

### 22.3 Cosseno-uniforme não é área-uniforme

A calota (`IDPERTURB_CAP` e o fallback do `HYBRID_VMF_CAP`) sorteia o cosseno-alvo
`s ~ U[lb, 1]`: é **uniforme em cosseno**, **não** uniforme na área da calota. Em
alta dimensão essas distribuições são bem diferentes. Consequência direta:

\[
\mathbb{E}[\langle \tilde v, \mu\rangle] = \frac{1 + lb}{2},
\]

e `lb = 1` colapsa a amostra ao próprio centro (sem perturbação). Não descreva a
calota como "amostragem uniforme sobre a calota esférica".

### 22.4 A guarda pode elevar o `lb` efetivo acima do máximo nominal

A guarda do bissetor (`cap_overlap_guard`) impõe um piso por identidade a partir
da densidade local dos centros e **tem prioridade** sobre o clipping. Por isso o
`lb` **efetivo** pode exceder `cap_maximum_lower_bound` em regiões densas — isso
é intencional, não bug. Em 512-D, um `lb` nominal de `0.4` com guarda ligada
**não** significa que as amostras foram geradas com piso `0.4`. **Sempre reporte
os percentis de `cap_lower_bound_effective` (mean/p10/p50/p90)** junto do `lb`
nominal — o `lb` nominal é apenas um piso, não o valor realizado. Ligação com
tipicidade: regiões densas (identidades típicas) toleram menos variação; regiões
raras toleram mais.

Note ainda que `cap_lower_bound_effective` é o piso garantido na direção
**pré-nuisance**. O passo tangencial de nuisance/viés pode empurrar o
`own_center_cosine` (pós-nuisance) ligeiramente **abaixo** de `lb`. Para auditar
a calota, use `pre_nuisance_center_cosine`, cujo invariante é
`pre_nuisance_center_cosine ≥ cap_lower_bound_effective`.

### 22.5 O Modo A não é anonimização e as âncoras devem ser disjuntas

O Modo A (`population_from_dataset`) **não** é anonimização: ele gera **variação
sintética em torno de âncoras biométricas reais**. As identidades produzidas são
reais. Use-o apenas para augmentation, validação metodológica ou experimentos
internos sobre uma base autorizada, e **mantenha o conjunto de ancoragem disjunto
de qualquer conjunto de avaliação**. Garanta a disjunção explicitamente com
`assert_disjoint_source_identities` — vazamento âncora↔avaliação invalida toda
métrica subsequente. As colunas de proveniência `source_identity` e
`source_sample_count` existem precisamente para tornar esse vínculo auditável.

### 22.6 Qualidade latente, bases por identidade e viés de sessão

- **Qualidade latente é uma variável experimental.** Não trate `latent_quality`
  como ground truth universal de qualidade biométrica ou de reconhecibilidade
  real.
- **Bases específicas por identidade são um stress test.** Aumentar
  excessivamente `identity_specific_nuisance_scale` cria um mundo artificialmente
  hostil ao PLDA. Use gradações e **reporte o valor**.
- **Viés de sessão não substitui um modelo temporal completo.** O viés de sessão
  é uma aproximação simples compartilhada pela combinação
  `identidade × domínio × session_index`. Ele não representa correlações
  temporais contínuas.

---

## 23. Testes automatizados

Executar a partir da raiz do repositório:

```bash
.venv/Scripts/python.exe -m pytest simulator/tests/ -q
```

A suíte está organizada em sete módulos. Os módulos de Modo B e do adapter
concreto fazem `pytest.importorskip("faceaudit")` e são automaticamente pulados
quando o FaceAudit não está instalado; os demais rodam sem o host.

### 23.1 `test_adaface_embedding_simulator.py` — núcleo

Valida a geometria e a reprodutibilidade básicas:

- normas unitárias e alinhamento da média empírica da vMF (`sample_vmf`);
- crescimento do comprimento resultante com \(\kappa\);
- consistência entre embeddings brutos e metadados (norma bate com
  `embedding_norm`, presença de `combined_tangent_step_norm`);
- independência da ordem das chamadas e seeds explícitas reproduzíveis;
- compartilhamento de população em cross-condition (centros e conjunto de
  `identity_id` idênticos entre domínios);
- bases anisotrópicas distintas por domínio;
- probabilidades dos regimes radiais batem com a configuração quando o
  acoplamento qualidade→regime está desligado;
- rótulos corretos dos pares (genuínos com mesma identidade, impostores com
  identidades distintas) e estratégia `BALANCED_OBSERVATION`;
- diagnóstico de bins do `qcut` reporta `requested_bin_count`/`actual_bin_count`.

### 23.2 `test_idperturb_cap.py` — calota, guarda e golden VMF

O módulo mais extenso, cobrindo:

- norma unitária da calota e preservação do cosseno sorteado, com
  \(\mathbb{E}[s] = (1+lb)/2\) (`sample_spherical_cap`);
- `lb = 1` retorna a direção média;
- pisos vetoriais respeitados por linha;
- ortogonalidade e norma unitária dos tangentes (`_deterministic_orthogonal`,
  `_sample_tangent_directions`);
- modulação de `lb` por qualidade/norma quando acoplado (correlação positiva com
  qualidade, negativa com `|z|`), e `lb` plano quando desacoplado;
- offset de `lb` por domínio (`cap_lower_bound_offset`);
- piso do bissetor (`_overlap_lower_bound_floor`) exclui o próprio centro, vale
  `sqrt(0.5)` para centros ortogonais, cresce com a densidade, dá 0 para
  singletons, e a implementação em blocos bate com a completa;
- a guarda mantém a perturbação mais próxima do próprio centro que do vizinho;
- **`lb` efetivo pode exceder o máximo configurado sob a guarda**, com percentis
  reportáveis;
- piso exato sem nuisance (`own_center_cosine == pre_nuisance_center_cosine`) e
  invariante pré/pós-nuisance: o nuisance empurra parte das amostras abaixo do
  piso geométrico, mas `pre_nuisance_center_cosine ≥ cap_lower_bound_effective`
  sempre vale;
- piso do overlap **não** serializado no manifesto;
- modo HYBRID respeita o piso em todas as amostras;
- taxa de fallback do HYBRID (`hybrid_used_cap_fallback`) sobe com `lb` alto e
  `κ` baixo, e fica baixa com `lb` baixo e `κ` alto;
- determinismo de **cada** modo de `IntraClassSamplingMode` (parametrizado);
- **golden VMF** (`test_golden_vmf_core_arrays_unchanged`): trava a saída do modo
  `VMF` por hash SHA-256 dos arrays `raw`/`normalized`/`centers`;
- round-trip do config do perfil `IDPERTURB_CAP`.

### 23.3 `test_config_roundtrip.py` — serialização

- `config_to_dict`/`config_from_dict` idempotente e JSON-safe nos quatro perfis;
- round-trip funcional bate bit-a-bit na geração (arrays e metadata);
- enums sobrevivem como strings (`tangent_update_mode`,
  `nuisance_structure_mode`);
- arrays NumPy serializam e reproduzem;
- `config_from_manifest` lê o bloco `config` do manifesto;
- chaves ausentes caem em defaults; chave desconhecida é rejeitada (`TypeError`).

### 23.4 `test_core_population_injection.py` — injeção e extração de centros

Roda sem FaceAudit (a extração, o Modo A e os helpers de leakage nunca importam
o host em runtime):

- injeção de `centers` usa e normaliza os vetores e atribui `identity_ids`;
- rejeição de "ambos" e "nenhum" entre `identity_count` e `centers`;
- validação de dimensão, componentes inteiros dentro de
  `[0, component_count)`, e comprimentos de `identity_labels` /
  `source_sample_counts`;
- **a injeção não altera o golden da população amostrada** (caminho amostrado
  permanece bit-a-bit idêntico);
- extração: norma unitária, contagens, média esférica manual, agrupamento pelo
  valor original de identidade (`1` ≠ `"1"`), filtro `min_samples`, identidades
  `None` ignoradas;
- qualidade NaN/None vira peso 1 e rejeição de pesos negativos;
- `weight_by_quality=True` exige `items`;
- fallback de média degenerada (centros antipodais) com warning;
- população Modo A produz colunas de proveniência `source_identity` /
  `source_sample_count`;
- helpers de leakage (`source_identity_set`, `assert_disjoint_source_identities`);
- import do módulo sem disparar o import do `MoVMFTypicalityModel`.

### 23.5 `test_faceaudit_seed.py` — Modo B (MoVMF)

Requer FaceAudit (`MoVMFTypicalityModel` + `EmbeddingDataset`):

- K fixo constrói config válido (`FACEAUDIT_MOVMF`, médias unitárias, pesos
  somando 1, `κ > 0`);
- K automático seleciona dentro dos candidatos (`criterion_values` ⊆ candidatos);
- aviso quando as normas são unitárias (caso L2), com
  `unit_norm_warning is True`;
- config do Modo B faz round-trip;
- **ghost centres do Modo B se parecem mais com os reais do que centros
  genéricos** (cosseno do vizinho mais próximo fica mais perto do real);
- exige ao menos dois centros;
- round-trip completo: gerar sintético → adaptar para `EmbeddingDataset` real →
  re-semear pela ponte.

### 23.6 `test_faceaudit_adapter.py` — adapter genérico

- registros simples (`dataset_to_embedding_records`) e adapter por factory
  injetada (`to_faceaudit_embedding_dataset`) usando DTOs dummy.

### 23.7 `test_faceaudit_concrete_adapter.py` — adapter concreto

Requer FaceAudit:

- mapeamento básico para `EmbeddingInfo`/`EmbeddingDataset` (identity, quality,
  source, origin, pose vazio, dtype `float32`);
- `raw` preserva norma e `normalized` é unitário;
- face hashes únicos e proveniência em `loaded_metadata`;
- filtro de domínio;
- round-trip por parquet;
- cross-condition compartilha identidades entre condições.

Para produção, acrescente testes específicos do seu pipeline FaceAudit.

---

## 24. Solução de problemas

### `ValueError: anisotropic_rank must satisfy 0 <= rank < dimension`

Reduza `anisotropic_rank` (ou `identity_specific_nuisance_rank`). As factories
prontas já limitam o posto para dimensões reduzidas usadas em testes. O posto tem
de ser estritamente menor que a dimensão.

### `ValueError: centers have dimension X, expected Y` (dimensão dos centros)

A matriz de centros injetada em `sample_identity_population(centers=...)` — ou os
centros extraídos por `identity_centers_from_dataset` — tem dimensão diferente da
`SimulatorConfig.dimension`. Construa o simulador com
`dimension=anchor_dataset.matrix.shape[1]` antes de ancorar. O mesmo vale para
`simulator_config_from_dataset`, que rejeita `base_config` com dimensão
divergente do dataset.

### O PLDA está performando bem demais

Verifique se está usando `make_cooperative_plda_config()`. Troque para
`make_cross_condition_stress_config()` e varie `domain_nuisance_weight`,
`identity_specific_nuisance_scale`, `domain_identity_bias_scale`,
`session_bias_scale`.

### O PLDA está performando mal demais

Reduza gradualmente os mesmos parâmetros. Não conclua que o backend é inadequado
apenas com base em um cenário extremo artificial.

### Calota com `lb` degenerado (`lb = 1` ou faixa vazia)

Se `cap_base_lower_bound = 1.0` (ou a modulação levar o `lb` efetivo a 1), a
calota colapsa ao próprio centro e não há dispersão intra-classe — todas as
amostras viram cópias do centro. Inversamente, se `cap_minimum_lower_bound >
cap_maximum_lower_bound` a validação rejeita o config. Confira a faixa
`[cap_minimum_lower_bound, cap_maximum_lower_bound]` e o sinal das penalidades.

### `qcut()` produziu menos bins

O método `norm_bin_diagnostics` emite warning e informa os dois contadores
(`requested_bin_count`/`actual_bin_count`). Examine clipping radial, quantidade de
valores idênticos, tamanho da base e parâmetros degenerados.

### `RadialReference.fit()` falhou com "scaled MAD is zero" (normas ~unitárias)

As normas fornecidas não têm dispersão suficiente. A causa mais comum é passar
embeddings **já normalizados em L2** (normas ≈ 1). Passe embeddings **brutos**.
No caminho automático (`simulator_config_from_dataset` com `fit_norm=True`) esse
caso é detectado: a função **mantém a referência radial-base** e emite warning,
em vez de quebrar — confira `fit_summary["norm_summary"]["radial_fitted"]` e
`unit_norm_warning`.

### Taxa de fallback alta no modo HYBRID

Sintoma: `hybrid_used_cap_fallback` True em fração grande das amostras (ou
`hybrid_resample_count` saturado em `hybrid_max_resample_rounds`). Causa: `lb`
alto combinado com `κ` baixo — o corpo vMF raramente cai dentro da calota, então
quase tudo vem do fallback de calota. O modo deixa de ser "vMF truncado" na
prática. Soluções: aumente `κ`, reduza `lb`, ou aumente
`hybrid_max_resample_rounds` (custa tempo). Se o objetivo é a calota pura, use
`IDPERTURB_CAP` diretamente.

### `lb` efetivo acima de `cap_maximum_lower_bound` via guarda

**Não é bug.** A guarda do bissetor (`cap_overlap_guard=True`) tem prioridade
sobre o clipping e impõe um piso por densidade local. Em regiões densas o `lb`
efetivo sobe acima do máximo nominal. Para varreduras limpas de `lb`, desligue a
guarda (`cap_overlap_guard=False`). Para auditar, reporte os percentis de
`cap_lower_bound_effective`.

### Penalidades de calota não têm o efeito esperado

As penalidades `cap_low_tail_penalty` / `cap_high_tail_penalty` agem em **escala
de cosseno**, **não** em `logκ`. Uma penalidade de `0.1` com `z = 2` derruba o
`lb` em `0.4`. Calibre-as separadamente das penalidades de `κ`
(`low_tail_quadratic_penalty` / `high_tail_quadratic_penalty`), que entram em
`logκ`.

### Média esférica degenerada / pesos de qualidade

Em `identity_centers_from_dataset`, uma identidade cuja média esférica colapsa
para ≈0 (ex.: imagens antipodais) gera warning e fallback para a melhor imagem
por qualidade (ou a primeira). Pesos de qualidade negativos são rejeitados;
None/NaN viram peso 1; todos-zero caem para média não-ponderada. Se o aviso de
degeneração aparecer com frequência, aumente `min_samples` (1 é ruidoso; 2+ é
mais estável) ou revise a coluna `quality_score`.

### Devo usar o adapter genérico ou o concreto?

Use **sempre o concreto** (`faceaudit_concrete_adapter.to_embedding_dataset`) em
produção: ele importa os DTOs reais e mapeia os campos corretos. O genérico
(`faceaudit_adapter.to_faceaudit_embedding_dataset`) assume
`EmbeddingInfo(metadata=...)`, que **não** corresponde ao DTO real do FaceAudit
(`EmbeddingInfo(embedding, face_hash, image_path, identity, file_name, ...)`, sem
`metadata`). O genérico é apenas referência/scaffold para projetos com outra
assinatura de DTO.

---

## 25. API pública resumida e checklist de experimento auditável

### 25.1 API pública resumida

**Enums** (`adaface_embedding_simulator`):

| Nome | Membros |
|---|---|
| `NuisanceStructureMode` | `GLOBAL_SHARED`, `PER_DOMAIN`, `HIERARCHICAL` |
| `TangentUpdateMode` | `RETRACTION`, `EXPONENTIAL_MAP` |
| `PairSamplingStrategy` | `BALANCED_IDENTITY`, `BALANCED_OBSERVATION` |
| `IntraClassSamplingMode` | `VMF`, `IDPERTURB_CAP`, `HYBRID_VMF_CAP` |

**Configs (dataclasses frozen):** `SimulatorConfig`, `PopulationConfig`,
`RadialConfig`, `RadialReference` (com `fit`), `DirectionalConfig`,
`CouplingConfig`, `AcquisitionDomainConfig`.

**Funções numéricas / amostragem:** `l2_normalize_rows`,
`sample_uniform_directions`, `sample_vmf`, `sample_spherical_cap`,
`sample_hybrid_vmf_cap`, `cosine_scores`, `sample_pair_table`.

**Serialização:** `config_to_dict`, `config_from_dict`, `config_from_manifest`.

**Perfis prontos:** `make_spherical_baseline_config` (`SPHERICAL_BASELINE`),
`make_cooperative_plda_config` (`COOPERATIVE_PLDA_BASELINE`; também disponível
como `make_example_config`), `make_cross_condition_stress_config`
(`CROSS_CONDITION_HETEROSCEDASTIC`), `make_idperturb_cap_config`
(`IDPERTURB_CAP`).

**Objetos de população/dataset:** `SyntheticIdentityPopulation`,
`SyntheticEmbeddingDataset` (métodos `save_npz`, `save_metadata_csv`,
`save_manifest`, `save_bundle`, `norm_bin_diagnostics`).

**Classe principal:** `AdaFaceEmbeddingSimulator` (métodos
`sample_identity_population` — forma amostrada e forma de injeção via
`centers=`/`components=`/`identity_labels=`/`source_sample_counts=`, mutuamente
exclusiva com `identity_count`; `sample_observations`; `sample_dataset`;
`sample_cross_condition_datasets`).

**Ponte com dados reais** (`faceaudit_seed`): `identity_centers_from_dataset`,
`IdentityCenterExtractionResult`, `population_from_dataset` (Modo A),
`population_config_from_dataset`, `simulator_config_from_dataset` (Modo B),
`source_identity_set`, `assert_disjoint_source_identities`.

**Adapters:** `faceaudit_concrete_adapter.to_embedding_dataset` /
`cross_condition_to_embedding_datasets` (produção, DTOs reais);
`faceaudit_adapter.dataset_to_embedding_records` /
`to_faceaudit_embedding_dataset` (scaffold de referência por injeção).

### 25.2 Checklist de experimento auditável

Anexe estes itens a qualquer relatório que use saída sintética. Sem todos eles, o
resultado não é reproduzível nem defensável.

- [ ] **Perfil** nomeado (`profile_name`) e justificativa de escolha.
- [ ] **Configuração completa** serializada via `config_to_dict` / manifesto
      (`save_manifest` ou `save_bundle`), incluindo todos os blocos
      (`population`, `radial`, `directional`, `coupling`, `acquisition_domains`).
- [ ] **Todas as seeds** registradas: `random_seed` estrutural,
      `population_random_seed` e `random_seed`/`observation_random_seeds` de
      observação. As seeds são derivadas de forma independente da ordem das
      chamadas; registre as explícitas usadas.
- [ ] **Política de trials**: contagens de pares genuínos/impostores, `strategy`
      (`BALANCED_IDENTITY` vs `BALANCED_OBSERVATION`), `unique_pairs` e a seed de
      `sample_pair_table`.
- [ ] **Hash do código**: `source_sha256` do manifesto (SHA-256 do módulo
      gerador) registrado, para fixar o código-fonte exato do gerador.
- [ ] **Métricas estratificadas**, não só agregadas: por domínio, por quantil de
      norma e por tipicidade (ver Seção 20.3).
- [ ] **Modo de dispersão intra-classe** (`intra_class_sampling_mode`)
      explicitado E, nos modos de calota, a **distribuição realizada de
      `cap_lower_bound_effective`** (mean/p10/p50/p90) — o `lb` nominal é apenas
      um piso; a guarda pode elevá-lo (Seção 22.4).
- [ ] No `HYBRID_VMF_CAP`, reportar `hybrid_used_cap_fallback` e
      `hybrid_resample_count` para mostrar quanto da amostra veio da calota vs do
      corpo vMF.
- [ ] **Asserção de disjunção** quando ancorado em dados reais (Modo A):
      `assert_disjoint_source_identities(pop_anchor, pop_eval)` executada e as
      colunas `source_identity` / `source_sample_count` preservadas.
- [ ] Quando ancorado via Modo B, registrar o `fit_summary`
      (`selected_k`, `criterion`, `criterion_values`, `n_degenerate`,
      `norm_summary`).
- [ ] Limites interpretativos da Seção 22 reproduzidos explicitamente no laudo.