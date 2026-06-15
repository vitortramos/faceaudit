# `adaface_embedding_simulator`

## Documentação completa — versão 0.2.0

## 1. Finalidade do pacote

O arquivo `adaface_embedding_simulator.py` implementa um **gerador sintético configurável de embeddings AdaFace-like** para experimentos controlados com backends biométricos, especialmente:

- similaridade de cosseno;
- normalização L2;
- LDA;
- *whitening*;
- Gaussianização radial;
- PLDA de duas covariâncias;
- PSDA e outros modelos esféricos;
- regressão logística para calibração;
- mineração de pares;
- modelos de tipicidade;
- comparação entre estratégias cross-condition.

O simulador **não implementa a rede neural AdaFace**. Ele também não afirma reproduzir automaticamente a distribuição de qualquer modelo AdaFace real. Seu objetivo é gerar dados artificiais com mecanismos geométricos conhecidos, configuráveis e auditáveis.

A decomposição fundamental é:

\[
\mathbf{x}_{ij}=r_{ij}\mathbf{u}_{ij},
\]

em que:

- \(\mathbf{x}_{ij}\in\mathbb{R}^{d}\) é o embedding bruto da amostra \(j\) da identidade \(i\);
- \(r_{ij}=\lVert\mathbf{x}_{ij}\rVert_2\) é a norma;
- \(\mathbf{u}_{ij}\in S^{d-1}\) é a direção normalizada;
- \(d\) é a dimensão, normalmente 512.

A versão 0.2.0 foi criada após uma revisão crítica da primeira versão. Ela separa claramente:

1. o **mundo sintético estrutural**;
2. a **população latente de identidades**;
3. as **observações** produzidas em um ou mais domínios;
4. as **políticas de geração de trials**;
5. os **artefatos de rastreabilidade**.

Essa separação evita um erro comum em experimentos sintéticos: interpretar como generalização real um resultado obtido em um cenário excessivamente cooperativo para o modelo avaliado.

---

## 2. O que mudou na versão 0.2.0

A versão inicial era útil para criar um baseline, mas apresentava limitações relevantes:

- base anisotrópica única para todo o dataset;
- vetor de viés único por domínio;
- consumo de um RNG estatal compartilhado entre splits;
- inexistência de uma API específica para datasets cross-condition alinhados por identidade;
- ausência de ablações independentes dos caminhos qualidade → norma → direção;
- amostragem de pares em loops Python;
- nome ambíguo `robust_norm_z`;
- ausência de adapter para DTOs do FaceAudit;
- ausência de testes automatizados.

A versão 0.2.0 acrescenta:

| Área | Melhoria |
|---|---|
| Reprodutibilidade | streams nomeados e seeds explícitas independentes da ordem das chamadas |
| Cross-condition | geração de datasets alinhados por identidade |
| Estrutura anisotrópica | modos global, por domínio e hierárquico |
| Heterogeneidade | bases específicas por identidade, bias domínio × identidade e bias de sessão |
| Geometria | retração e mapa exponencial |
| Ablation studies | switches independentes dos acoplamentos qualidade, norma e direção |
| Norma | `reference_scaled_log_norm_z` e `RadialReference.fit()` |
| Trials | sampler vetorizado e estratégias explícitas de ponderação |
| Diagnóstico | aviso quando `qcut()` retorna menos bins do que solicitado |
| Persistência | bundle completo com manifesto JSON |
| Integração | adapter genérico para `EmbeddingInfo` e `EmbeddingDataset` |
| Qualidade de software | testes automatizados |

A versão atual preserva o comportamento conceitual anterior por meio do perfil:

```text
COOPERATIVE_PLDA_BASELINE
```

Para *stress testing* cross-condition, utilize:

```text
CROSS_CONDITION_HETEROSCEDASTIC
```

---

## 3. Estrutura do pacote

```text
adaface_embedding_simulator/
├── adaface_embedding_simulator.py
├── faceaudit_adapter.py
├── example_generate_adaface_embeddings.py
├── README.md
├── DOCUMENTATION.md
├── CHANGELOG.md
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── tests/
    ├── test_adaface_embedding_simulator.py
    └── test_faceaudit_adapter.py
```

### `adaface_embedding_simulator.py`

Módulo principal. Contém:

- funções numéricas;
- sampler vetorizado von Mises–Fisher;
- enums;
- classes de configuração;
- classe `RadialReference`;
- classe `SyntheticIdentityPopulation`;
- classe `SyntheticEmbeddingDataset`;
- classe `AdaFaceEmbeddingSimulator`;
- funções de score e amostragem de pares;
- perfis prontos.

### `faceaudit_adapter.py`

Adapter de referência desacoplado das classes concretas do projeto FaceAudit. Como o simulador é um pacote independente, as factories de `EmbeddingInfo` e `EmbeddingDataset` são injetadas pelo usuário.

### `example_generate_adaface_embeddings.py`

Exemplo executável que cria:

- treino, desenvolvimento e teste disjuntos;
- pares genuínos e impostores;
- bundles persistidos;
- datasets cross-condition alinhados por identidade.

### `tests/`

Testes automatizados para:

- geometria L2;
- amostragem vMF;
- regimes radiais;
- reprodutibilidade;
- alinhamento cross-condition;
- bases por domínio;
- geração de pares;
- adapter.

---

## 4. Instalação

Requisitos mínimos:

```bash
pip install numpy pandas
```

Para desenvolvimento e testes:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

O sampler vMF foi implementado diretamente com NumPy. O SciPy não é uma dependência obrigatória.

---

## 5. Início rápido

### 5.1 Gerar um split independente

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cooperative_plda_config,
)

config = make_cooperative_plda_config(dimension=512)

simulator = AdaFaceEmbeddingSimulator(
    config,
    random_seed=2026,  # seed estrutural do mundo sintético
)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 10),
    split_name="test",
    identity_offset=200_000,
    random_seed=103,             # observações
    population_random_seed=1003, # identidades latentes
)

print(test.raw_embeddings.shape)
print(test.normalized_embeddings.shape)
print(test.metadata.head())
```

### 5.2 Salvar o bundle completo

```python
test.save_bundle(
    output_dir="synthetic_output",
    stem="test",
)
```

Arquivos produzidos:

```text
synthetic_output/
├── test_embeddings.npz
├── test_metadata.csv
├── test_norm_diagnostics.csv
└── test_manifest.json
```

### 5.3 Gerar pares

```python
from adaface_embedding_simulator import sample_pair_table

pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    random_seed=202,
)

pairs.to_csv("synthetic_output/test_pairs.csv", index=False)
```

---

## 6. Modelo generativo

O gerador possui seis níveis:

1. mundo estrutural;
2. componentes populacionais;
3. identidade latente;
4. domínio, qualidade e regime radial;
5. direção vMF intraidentidade;
6. nuisance tangencial e embedding bruto.

### 6.1 Mundo estrutural

Ao criar:

```python
simulator = AdaFaceEmbeddingSimulator(config, random_seed=2026)
```

são geradas estruturas compartilhadas pelo mundo sintético:

- médias dos componentes populacionais;
- base anisotrópica global;
- bases anisotrópicas por domínio;
- vetores médios de bias por domínio.

Essas estruturas dependem da seed estrutural, mas não do número nem da ordem de chamadas posteriores.

### 6.2 Mistura populacional de vMF

Cada identidade pertence a um componente populacional latente:

\[
z_i\sim\operatorname{Categorical}(\pi_1,\ldots,\pi_K).
\]

Cada componente possui uma direção média:

\[
\boldsymbol{\mu}_k\in S^{d-1}.
\]

Quando as médias não são fornecidas, o simulador sorteia uma direção global \(\boldsymbol{\mu}_0\) e cria os componentes por:

\[
\boldsymbol{\mu}_k
\sim
\operatorname{vMF}
\left(
\boldsymbol{\mu}_0,
\kappa_{\text{componentes}}
\right).
\]

Se:

```python
component_mean_global_kappa=0.0
```

as médias dos componentes são amostradas uniformemente sobre a hiperesfera.

### 6.3 Centro angular da identidade

Cada centro de identidade é sorteado por:

\[
\mathbf{c}_i
\mid z_i=k
\sim
\operatorname{vMF}
\left(
\boldsymbol{\mu}_k,
\kappa_{\text{identidade},k}
\right).
\]

Quanto maior `identity_center_kappas`, mais próximas as identidades ficam da média do componente. Quanto menor o valor, mais espalhados ficam os centros.

### 6.4 Qualidade latente

Para cada amostra:

\[
q_{ij}
\sim
\operatorname{Beta}
\left(
\alpha_d,
\beta_d
\right),
\]

em que \(d\) é o domínio de aquisição.

A qualidade é um índice sintético em \([0,1]\). Ela não deve ser interpretada automaticamente como uma medida objetiva de qualidade visual ou reconhecibilidade real.

### 6.5 Regimes radiais

Cada amostra pertence a um dos estados:

```text
central
low_tail
high_tail
```

A probabilidade pode variar com a qualidade:

\[
P(s_{ij}=\text{low\_tail})
\propto
\exp
\left[
\ell_{\text{low},d}
+
\gamma_{\text{low},d}(0{,}5-q_{ij})
\right].
\]

A mesma lógica é aplicada à cauda alta.

### 6.6 Norma do embedding

A log-norma é gerada por:

\[
\log r_{ij}
=
\mu_r
+
\beta_r(q_{ij}-0{,}5)
+
\Delta_{s_{ij}}
+
\Delta_d
+
\varepsilon_{ij},
\]

com:

\[
\varepsilon_{ij}
\sim
\mathcal{N}
\left(
0,
\sigma_{s_{ij}}^2 m_d^2
\right).
\]

O modelo não impõe que norma alta seja sempre melhor. Existem uma faixa central, uma cauda inferior e uma cauda superior.

### 6.7 Padronização radial

O código calcula:

\[
z_{r,ij}
=
\frac{
\log r_{ij}-\mu_r
}{
\sigma_r
}.
\]

A coluna principal se chama:

```text
reference_scaled_log_norm_z
```

O nome é deliberadamente explícito. O valor só é robusto quando a referência foi estimada por uma regra robusta.

A coluna legada:

```text
robust_norm_z
```

continua disponível como alias para preservar compatibilidade.

### 6.8 Concentração angular intraidentidade

A direção observada é amostrada por:

\[
\mathbf{u}_{ij}^{(0)}
\mid
\mathbf{c}_i,r_{ij},q_{ij},d
\sim
\operatorname{vMF}
\left(
\mathbf{c}_i,
\kappa_{ij}
\right).
\]

A concentração segue:

\[
\log\kappa_{ij}
=
\log\kappa_{\text{pico}}
+
\beta_q(q_{ij}-0{,}5)
+
\Delta_{\kappa,d}
-
\lambda_{\text{low}}
\left[
\max(0,z_0-z_{r,ij})
\right]^2
-
\lambda_{\text{high}}
\left[
\max(0,z_{r,ij}-z_0)
\right]^2.
\]

Assim:

- a faixa radial central pode gerar maior estabilidade;
- normas muito baixas reduzem \(\kappa\);
- normas muito altas também podem reduzir \(\kappa\);
- as duas caudas podem possuir penalizações assimétricas.

### 6.9 Perturbações tangenciais

Após a geração vMF, o simulador adiciona uma perturbação estruturada:

\[
\boldsymbol{\delta}_{ij}
=
\boldsymbol{\delta}_{ij}^{\text{aniso}}
+
\mathbf{b}_d
+
\mathbf{b}_{d,i}
+
\mathbf{b}_{d,s}.
\]

Componentes:

| Termo | Interpretação |
|---|---|
| \(\boldsymbol{\delta}_{ij}^{\text{aniso}}\) | nuisance anisotrópico |
| \(\mathbf{b}_d\) | deslocamento médio do domínio |
| \(\mathbf{b}_{d,i}\) | interação domínio × identidade |
| \(\mathbf{b}_{d,s}\) | deslocamento de sessão |

O passo é projetado no plano tangente antes da atualização esférica.

### 6.10 Atualização por retração

Modo:

```text
retraction
```

Fórmula:

\[
\widetilde{\mathbf{u}}_{ij}
=
\frac{
\mathbf{u}_{ij}^{(0)}
+
\boldsymbol{\delta}_{ij}^{\top}
}{
\left\|
\mathbf{u}_{ij}^{(0)}
+
\boldsymbol{\delta}_{ij}^{\top}
\right\|_2
}.
\]

Esse modo preserva a aproximação utilizada na versão inicial. É apropriado para passos pequenos.

### 6.11 Atualização por mapa exponencial

Modo:

```text
exponential_map
```

Fórmula:

\[
\operatorname{Exp}_{\mathbf{u}}
(\boldsymbol{\delta})
=
\cos
\left(
\lVert\boldsymbol{\delta}\rVert
\right)
\mathbf{u}
+
\sin
\left(
\lVert\boldsymbol{\delta}\rVert
\right)
\frac{
\boldsymbol{\delta}
}{
\lVert\boldsymbol{\delta}\rVert
}.
\]

O mapa exponencial é recomendado quando os passos deixam de ser desprezíveis.

### 6.12 Embedding bruto

Por fim:

\[
\mathbf{x}_{ij}
=
r_{ij}\widetilde{\mathbf{u}}_{ij}.
\]

O objeto retornado expõe simultaneamente:

```python
dataset.raw_embeddings
dataset.normalized_embeddings
```

---

## 7. Perfis prontos

A versão 0.2 fornece três perfis.

### 7.1 `SPHERICAL_BASELINE`

Factory:

```python
make_spherical_baseline_config(dimension=512)
```

Características:

- mistura populacional vMF;
- variação intraidentidade vMF;
- sem nuisance anisotrópico;
- sem bias cross-condition obrigatório.

Uso recomendado:

- comparar cosseno, PLDA e PSDA em cenário esférico simples;
- verificar se um backend introduz ganho quando não existe estrutura anisotrópica relevante.

### 7.2 `COOPERATIVE_PLDA_BASELINE`

Factory:

```python
make_cooperative_plda_config(dimension=512)
```

Alias legado:

```python
make_example_config(dimension=512)
```

Características:

- base anisotrópica global compartilhada;
- relação não monotônica norma ↔ estabilidade;
- domínios `controlled` e `surveillance`;
- bias médio pequeno para vigilância;
- atualização por retração.

Interpretação correta:

> cenário cooperativo no qual PLDA, LDA e whitening conseguem aprender uma estrutura compartilhada.

Não use esse perfil isoladamente para inferir desempenho quantitativo em embeddings AdaFace reais.

### 7.3 `CROSS_CONDITION_HETEROSCEDASTIC`

Factory:

```python
make_cross_condition_stress_config(dimension=512)
```

Características:

- nuisance hierárquico;
- base global;
- bases diferentes por domínio;
- bases específicas por identidade;
- bias médio de domínio;
- bias domínio × identidade;
- bias de sessão;
- mapa exponencial;
- domínio `surveillance` com maior dispersão.

Uso recomendado:

- comparar merge ingênuo, `PER_CONDITION` e `BALANCED_IDENTITY`;
- estudar robustez fora de domínio;
- medir inflação ou subestimação de \(W\);
- testar calibração sob heterocedasticidade.

---

## 8. Referência completa da configuração

## 8.1 `SimulatorConfig`

```python
@dataclass(frozen=True)
class SimulatorConfig:
    dimension: int = 512
    profile_name: str = "custom"
    population: PopulationConfig = ...
    radial: RadialConfig = ...
    directional: DirectionalConfig = ...
    coupling: CouplingConfig = ...
    acquisition_domains: tuple[AcquisitionDomainConfig, ...] = ...
```

| Parâmetro | Tipo | Padrão | Função |
|---|---|---:|---|
| `dimension` | `int` | `512` | dimensão dos embeddings |
| `profile_name` | `str` | `custom` | nome persistido no manifesto |
| `population` | `PopulationConfig` | default | estrutura populacional |
| `radial` | `RadialConfig` | default | distribuição das normas |
| `directional` | `DirectionalConfig` | default | dispersão angular e nuisance |
| `coupling` | `CouplingConfig` | default | switches de ablação |
| `acquisition_domains` | tupla | `default` | parâmetros por domínio |

---

## 8.2 `PopulationConfig`

```python
@dataclass(frozen=True)
class PopulationConfig:
    component_count: int = 8
    component_weights: ArrayLike | None = None
    identity_center_kappas: ArrayLike | float = 70.0
    component_means: ArrayLike | None = None
    component_mean_global_kappa: float = 20.0
    global_mean_direction: ArrayLike | None = None
```

| Parâmetro | Tipo | Padrão | Efeito |
|---|---|---:|---|
| `component_count` | `int` | `8` | quantidade de regiões populacionais |
| `component_weights` | vetor ou `None` | `None` | pesos da mistura; `None` usa pesos uniformes |
| `identity_center_kappas` | escalar ou vetor | `70.0` | concentração dos centros de identidade dentro de cada componente |
| `component_means` | matriz ou `None` | `None` | médias explícitas dos componentes, shape `(K, d)` |
| `component_mean_global_kappa` | `float` | `20.0` | proximidade dos componentes em torno da média global |
| `global_mean_direction` | vetor ou `None` | `None` | média global explícita |

### Como interpretar `identity_center_kappas`

Valor maior:

- identidades mais agrupadas dentro do componente;
- maior densidade local;
- impostores potencialmente mais semelhantes.

Valor menor:

- centros mais dispersos;
- população mais espalhada;
- menor densidade local.

---

## 8.3 `RadialConfig`

```python
@dataclass(frozen=True)
class RadialConfig:
    reference_log_norm_mean: float = log(25.0)
    reference_log_norm_scale: float = 0.24
    reference_method: str = "configured_reference"
    quality_log_norm_slope: float = 0.12
    central_offset: float = 0.0
    central_sigma: float = 0.10
    low_tail_offset: float = -0.70
    low_tail_sigma: float = 0.16
    high_tail_offset: float = 0.70
    high_tail_sigma: float = 0.16
    minimum_norm: float = 1e-6
    maximum_norm: float = 1e6
```

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `reference_log_norm_mean` | `log(25)` | centro da padronização radial |
| `reference_log_norm_scale` | `0.24` | escala da padronização |
| `reference_method` | `configured_reference` | descrição persistida nos metadados |
| `quality_log_norm_slope` | `0.12` | contribuição direta da qualidade à log-norma |
| `central_offset` | `0.0` | deslocamento do regime central |
| `central_sigma` | `0.10` | dispersão do regime central |
| `low_tail_offset` | `-0.70` | deslocamento da cauda inferior |
| `low_tail_sigma` | `0.16` | dispersão da cauda inferior |
| `high_tail_offset` | `0.70` | deslocamento da cauda superior |
| `high_tail_sigma` | `0.16` | dispersão da cauda superior |
| `minimum_norm` | `1e-6` | clipping inferior |
| `maximum_norm` | `1e6` | clipping superior |

### Ajustar referência radial contra dados reais

```python
from adaface_embedding_simulator import RadialReference, RadialConfig

reference = RadialReference.fit(real_raw_embeddings)

radial = RadialConfig.from_reference(
    reference,
    quality_log_norm_slope=0.12,
    low_tail_offset=-0.70,
    high_tail_offset=0.70,
)
```

`RadialReference.fit()` usa:

\[
\operatorname{mediana}(\log r)
\]

como localização e:

\[
1{,}4826\cdot\operatorname{MAD}(\log r)
\]

como escala.

---

## 8.4 `DirectionalConfig`

```python
@dataclass(frozen=True)
class DirectionalConfig:
    peak_intra_kappa: float = 1400.0
    quality_log_kappa_slope: float = 0.90
    optimal_reference_scaled_log_norm_z: float = 0.0
    low_tail_quadratic_penalty: float = 0.55
    high_tail_quadratic_penalty: float = 0.75
    minimum_intra_kappa: float = 20.0
    maximum_intra_kappa: float = 10000.0

    anisotropic_rank: int = 0
    anisotropic_base_scale: float = 0.0
    anisotropic_quality_slope: float = 1.0
    nuisance_structure_mode: str = "global_shared"
    global_nuisance_weight: float = 1.0
    domain_nuisance_weight: float = 0.0
    identity_specific_nuisance_rank: int = 0
    identity_specific_nuisance_scale: float = 0.0
    tangent_update_mode: str = "retraction"
```

### Parâmetros da dispersão vMF

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `peak_intra_kappa` | `1400` | estabilidade máxima em condições favoráveis |
| `quality_log_kappa_slope` | `0.90` | efeito direto da qualidade sobre \(\log\kappa\) |
| `optimal_reference_scaled_log_norm_z` | `0.0` | faixa radial ótima |
| `low_tail_quadratic_penalty` | `0.55` | penalização da cauda baixa |
| `high_tail_quadratic_penalty` | `0.75` | penalização da cauda alta |
| `minimum_intra_kappa` | `20` | clipping inferior |
| `maximum_intra_kappa` | `10000` | clipping superior |

### Parâmetros de nuisance anisotrópico

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `anisotropic_rank` | `0` | posto das bases global e por domínio |
| `anisotropic_base_scale` | `0.0` | amplitude base do nuisance |
| `anisotropic_quality_slope` | `1.0` | aumenta nuisance quando qualidade cai |
| `nuisance_structure_mode` | `global_shared` | compartilhamento da base |
| `global_nuisance_weight` | `1.0` | peso da base global |
| `domain_nuisance_weight` | `0.0` | peso das bases por domínio no modo hierárquico |
| `identity_specific_nuisance_rank` | `0` | posto da base específica por identidade |
| `identity_specific_nuisance_scale` | `0.0` | amplitude da parcela específica por identidade |
| `tangent_update_mode` | `retraction` | retorno à hiperesfera |

### Modos de nuisance

```python
from adaface_embedding_simulator import NuisanceStructureMode
```

| Valor | Descrição |
|---|---|
| `global_shared` | uma base compartilhada pelo dataset inteiro |
| `per_domain` | uma base diferente para cada domínio |
| `hierarchical` | combinação de base global, base por domínio e parcela específica da identidade |

### Modos de atualização tangencial

```python
from adaface_embedding_simulator import TangentUpdateMode
```

| Valor | Descrição |
|---|---|
| `retraction` | soma e renormalização; rápida e compatível com v0.1 |
| `exponential_map` | deslocamento geodésico exato no plano tangente da direção atual |

---

## 8.5 `CouplingConfig`

```python
@dataclass(frozen=True)
class CouplingConfig:
    quality_affects_regime_probabilities: bool = True
    quality_affects_log_norm: bool = True
    quality_affects_intra_kappa: bool = True
    norm_z_affects_intra_kappa: bool = True
```

| Parâmetro | Padrão | Caminho causal |
|---|---:|---|
| `quality_affects_regime_probabilities` | `True` | qualidade → regime radial |
| `quality_affects_log_norm` | `True` | qualidade → norma |
| `quality_affects_intra_kappa` | `True` | qualidade → direção |
| `norm_z_affects_intra_kappa` | `True` | norma → estabilidade angular |

### Exemplo de ablação puramente radial

```python
from dataclasses import replace
from adaface_embedding_simulator import CouplingConfig, make_cooperative_plda_config

base = make_cooperative_plda_config()
config = replace(
    base,
    profile_name="ABLATION_RADIAL_ONLY",
    coupling=CouplingConfig(
        quality_affects_regime_probabilities=True,
        quality_affects_log_norm=True,
        quality_affects_intra_kappa=False,
        norm_z_affects_intra_kappa=False,
    ),
)
```

Nesse cenário, normalização L2 deve remover praticamente todo o efeito radial direto.

### Exemplo de ablação norma → direção

```python
config = replace(
    base,
    profile_name="ABLATION_NORM_TO_DIRECTION",
    coupling=CouplingConfig(
        quality_affects_regime_probabilities=True,
        quality_affects_log_norm=True,
        quality_affects_intra_kappa=False,
        norm_z_affects_intra_kappa=True,
    ),
)
```

---

## 8.6 `AcquisitionDomainConfig`

```python
@dataclass(frozen=True)
class AcquisitionDomainConfig:
    name: str
    weight: float = 1.0
    quality_beta_alpha: float = 4.0
    quality_beta_beta: float = 2.0

    central_regime_probability: float = 0.86
    low_tail_probability: float = 0.08
    high_tail_probability: float = 0.06
    low_tail_quality_logit_slope: float = 2.0
    high_tail_quality_logit_slope: float = 0.0

    log_norm_shift: float = 0.0
    log_norm_sigma_multiplier: float = 1.0
    log_kappa_shift: float = 0.0

    anisotropic_noise_multiplier: float = 1.0
    domain_bias_scale: float = 0.0
    domain_identity_bias_scale: float = 0.0
    session_bias_scale: float = 0.0
```

### Domínio e qualidade

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `name` | obrigatório | nome do domínio |
| `weight` | `1.0` | peso quando domínios são sorteados em conjunto |
| `quality_beta_alpha` | `4.0` | parâmetro \(\alpha\) da Beta |
| `quality_beta_beta` | `2.0` | parâmetro \(\beta\) da Beta |

### Probabilidades radiais

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `central_regime_probability` | `0.86` | probabilidade-base central |
| `low_tail_probability` | `0.08` | probabilidade-base cauda baixa |
| `high_tail_probability` | `0.06` | probabilidade-base cauda alta |
| `low_tail_quality_logit_slope` | `2.0` | aumenta cauda baixa quando qualidade cai |
| `high_tail_quality_logit_slope` | `0.0` | efeito análogo na cauda alta |

### Efeitos de domínio

| Parâmetro | Padrão | Efeito |
|---|---:|---|
| `log_norm_shift` | `0.0` | deslocamento radial médio |
| `log_norm_sigma_multiplier` | `1.0` | multiplicador da dispersão radial |
| `log_kappa_shift` | `0.0` | deslocamento da estabilidade angular |
| `anisotropic_noise_multiplier` | `1.0` | multiplicador do nuisance anisotrópico |
| `domain_bias_scale` | `0.0` | bias médio do domínio |
| `domain_identity_bias_scale` | `0.0` | heterogeneidade domínio × identidade |
| `session_bias_scale` | `0.0` | variação entre sessões |

---

## 9. Reprodutibilidade

## 9.1 Seed estrutural

A seed fornecida ao construtor controla o mundo:

```python
simulator = AdaFaceEmbeddingSimulator(
    config,
    random_seed=2026,
)
```

Ela determina:

- médias dos componentes populacionais;
- base anisotrópica global;
- bases anisotrópicas por domínio;
- biases médios dos domínios.

## 9.2 Seeds de população e observações

Cada split pode receber seeds explícitas:

```python
train = simulator.sample_dataset(
    identity_count=2_000,
    samples_per_identity=(5, 12),
    split_name="train",
    identity_offset=0,
    population_random_seed=1001,
    random_seed=101,
)
```

- `population_random_seed`: centros latentes e estruturas específicas das identidades;
- `random_seed`: observações, qualidade, normas, sessões e ruído.

## 9.3 Streams nomeados independentes da ordem

Se as seeds explícitas forem omitidas, o simulador deriva seeds estáveis a partir de:

- seed estrutural;
- namespace;
- nome do split;
- domínio;
- `identity_offset`;
- quantidade de identidades.

Isso significa que inserir uma geração intermediária não altera silenciosamente o split posterior.

Exemplo:

```python
first_dev = simulator.sample_dataset(
    identity_count=500,
    samples_per_identity=4,
    split_name="dev",
    identity_offset=100_000,
)

_ = simulator.sample_dataset(
    identity_count=10,
    samples_per_identity=2,
    split_name="diagnostic",
    identity_offset=999_000,
)

second_dev = simulator.sample_dataset(
    identity_count=500,
    samples_per_identity=4,
    split_name="dev",
    identity_offset=100_000,
)

assert (first_dev.raw_embeddings == second_dev.raw_embeddings).all()
```

---

## 10. Separação entre população e observações

A mudança arquitetural mais importante da versão 0.2 é a separação entre:

```python
sample_identity_population()
```

e:

```python
sample_observations()
```

## 10.1 Gerar população latente

```python
population = simulator.sample_identity_population(
    identity_count=800,
    identity_offset=300_000,
    random_seed=4001,
    stream_name="cross_condition_population",
)
```

Objeto retornado:

```python
SyntheticIdentityPopulation
```

Campos:

| Campo | Conteúdo |
|---|---|
| `identity_ids` | ids das identidades |
| `population_components` | componente populacional por identidade |
| `identity_centers` | centro angular latente |
| `identity_specific_nuisance_bases` | base específica por identidade, quando habilitada |
| `domain_identity_bias_vectors` | bias domínio × identidade |
| `sampling_seed` | seed efetiva |
| `stream_name` | nome do stream |

## 10.2 Observar a população em domínio controlado

```python
controlled = simulator.sample_observations(
    population,
    samples_per_identity=(3, 6),
    split_name="test:controlled",
    forced_domain="controlled",
    sessions_per_identity=(1, 3),
    random_seed=4002,
)
```

## 10.3 Observar a mesma população em vigilância

```python
surveillance = simulator.sample_observations(
    population,
    samples_per_identity=(3, 6),
    split_name="test:surveillance",
    forced_domain="surveillance",
    sessions_per_identity=(1, 3),
    random_seed=4003,
)
```

As duas bases compartilham:

\[
\mathbf{c}_i^{\text{controlled}}
=
\mathbf{c}_i^{\text{surveillance}}.
\]

Mas suas observações diferem:

\[
\mathbf{x}_{ij}^{\text{controlled}}
\neq
\mathbf{x}_{ik}^{\text{surveillance}}.
\]

---

## 11. Helper cross-condition

A API oferece um helper equivalente:

```python
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

controlled = aligned["controlled"]
surveillance = aligned["surveillance"]
```

Esse desenho foi criado para testar estratégias como:

```text
PER_CONDITION
BALANCED_IDENTITY
```

---

## 12. Objeto `SyntheticEmbeddingDataset`

Campos principais:

| Campo | Shape ou tipo | Conteúdo |
|---|---|---|
| `raw_embeddings` | `(N, d)` | embeddings brutos |
| `normalized_embeddings` | `(N, d)` | direções unitárias |
| `identity_centers` | `(I, d)` | centros latentes |
| `metadata` | `DataFrame` | metadados por amostra |
| `population_component_means` | `(K, d)` | médias populacionais |
| `anisotropic_basis` | `(d, r)` ou `None` | base global |
| `domain_anisotropic_bases` | mapping | bases por domínio |
| `domain_bias_vectors` | mapping | bias médio por domínio |
| `config` | `SimulatorConfig` | configuração |
| `structural_seed` | `int` | seed do mundo |
| `population_sampling_seed` | `int` | seed da população |
| `observation_sampling_seed` | `int` | seed das observações |
| `stream_name` | `str` | stream efetivo |

### 12.1 Metadados por observação

| Coluna | Significado |
|---|---|
| `split` | nome do split |
| `identity_id` | identidade sintética |
| `local_identity_index` | índice local na população |
| `sample_index` | índice da observação na identidade |
| `session_index` | sessão sintética |
| `population_component` | componente populacional |
| `acquisition_domain` | domínio |
| `radial_regime` | `central`, `low_tail` ou `high_tail` |
| `latent_quality` | qualidade sintética |
| `embedding_norm` | norma do embedding bruto |
| `log_embedding_norm` | log-norma |
| `reference_scaled_log_norm_z` | posição radial padronizada |
| `robust_norm_z` | alias legado |
| `radial_reference_method` | método da referência radial |
| `intra_kappa` | concentração vMF intraidentidade |
| `own_center_cosine` | similaridade com o centro latente |
| `tangent_update_mode` | retração ou mapa exponencial |
| `anisotropic_noise_norm` | magnitude do nuisance anisotrópico |
| `fixed_domain_bias_norm` | magnitude do bias médio de domínio |
| `domain_identity_bias_norm` | magnitude da interação domínio × identidade |
| `session_bias_norm` | magnitude do bias de sessão |
| `combined_tangent_step_norm` | tamanho do passo tangencial total |

---

## 13. Diagnóstico por faixa radial

```python
diagnostics = test.norm_bin_diagnostics(bin_count=10)
print(diagnostics)
```

Colunas retornadas:

| Coluna | Significado |
|---|---|
| `requested_bin_count` | quantidade solicitada |
| `actual_bin_count` | quantidade efetivamente produzida |
| `norm_quantile_bin` | faixa de norma |
| `sample_count` | tamanho do grupo |
| `norm_mean` | norma média |
| `reference_scaled_log_norm_z_mean` | posição radial média |
| `latent_quality_mean` | qualidade média |
| `intra_kappa_mean` | estabilidade média |
| `own_center_cosine_mean` | proximidade média ao centro |
| `own_center_cosine_median` | proximidade mediana |
| `tangent_step_norm_mean` | passo tangencial médio |

O método usa:

```python
pd.qcut(..., duplicates="drop")
```

Se existirem bordas repetidas, o pandas poderá produzir menos bins do que solicitado. A versão 0.2 emite um warning e informa os dois contadores.

---

## 14. Geração de pares

### 14.1 Uso básico

```python
pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    random_seed=202,
)
```

### 14.2 Estratégia `balanced_identity`

```python
from adaface_embedding_simulator import PairSamplingStrategy

pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    strategy=PairSamplingStrategy.BALANCED_IDENTITY,
)
```

Interpretação:

- genuínos: identidade escolhida aproximadamente uniformemente;
- impostores: duas identidades escolhidas aproximadamente uniformemente;
- observações: sorteadas dentro das identidades selecionadas.

Esse modo evita que identidades com muitas amostras dominem os trials.

### 14.3 Estratégia `balanced_observation`

```python
pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    strategy=PairSamplingStrategy.BALANCED_OBSERVATION,
)
```

Interpretação:

- genuínos continuam balanceados por identidade;
- impostores são amostrados aproximadamente em proporção ao número de observações.

### 14.4 Trials únicos

```python
pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    unique_pairs=True,
)
```

A deduplicação trata pares como simétricos:

\[
(a,b)\equiv(b,a).
\]

Se duplicatas forem removidas, o método emite warning e pode retornar menos trials do que solicitado.

### 14.5 Colunas dos trials

| Coluna | Significado |
|---|---|
| `left_row`, `right_row` | índices das observações |
| `is_genuine` | `1` genuíno, `0` impostor |
| `cosine_score` | cosseno |
| `left_identity_id`, `right_identity_id` | identidades |
| `left_norm`, `right_norm` | normas |
| `left_reference_scaled_log_norm_z`, `right_reference_scaled_log_norm_z` | posições radiais |
| `left_quality`, `right_quality` | qualidades sintéticas |

---

## 15. Persistência e manifesto

### 15.1 `save_npz()`

```python
dataset.save_npz("test_embeddings.npz")
```

Salva:

- embeddings brutos;
- embeddings normalizados;
- centros de identidade;
- médias populacionais;
- base anisotrópica global;
- nomes dos domínios;
- vetores médios de bias;
- metadados estruturados.

### 15.2 `save_metadata_csv()`

```python
dataset.save_metadata_csv("test_metadata.csv")
```

### 15.3 `save_manifest()`

```python
dataset.save_manifest("test_manifest.json")
```

O manifesto contém:

- versão do simulador;
- perfil;
- seed estrutural;
- seed da população;
- seed das observações;
- stream;
- número de amostras;
- dimensão;
- número de identidades;
- SHA-256 do código-fonte;
- configuração completa.

### 15.4 `save_bundle()`

```python
dataset.save_bundle("synthetic_output", "test")
```

É o método recomendado para experimentos auditáveis.

---

## 16. Adapter para o FaceAudit

O arquivo:

```text
faceaudit_adapter.py
```

oferece duas funções.

### 16.1 Registros simples

```python
from faceaudit_adapter import dataset_to_embedding_records

records = dataset_to_embedding_records(
    test,
    embedding_view="normalized",
    domain_filter="surveillance",
)
```

Cada item possui:

```python
{
    "embedding": ...,
    "identity": ...,
    "metadata": ...,
}
```

### 16.2 Criar DTOs do FaceAudit

```python
from faceaudit_adapter import to_faceaudit_embedding_dataset
from seu_projeto import EmbeddingDataset, EmbeddingInfo

faceaudit_dataset = to_faceaudit_embedding_dataset(
    test,
    embedding_info_factory=EmbeddingInfo,
    embedding_dataset_factory=EmbeddingDataset,
    embedding_view="normalized",
    dataset_factory_keyword="embeddings",
    include_metadata=True,
)
```

Como o construtor real do FaceAudit pode possuir assinatura diferente, os parâmetros de factory foram mantidos explícitos.

Se `EmbeddingInfo` não aceitar metadados:

```python
faceaudit_dataset = to_faceaudit_embedding_dataset(
    test,
    embedding_info_factory=EmbeddingInfo,
    embedding_dataset_factory=EmbeddingDataset,
    embedding_view="normalized",
    dataset_factory_keyword="embeddings",
    include_metadata=False,
)
```

---

## 17. Experimentos recomendados com PLDA

## 17.1 Baseline esférico

Objetivo:

> comparar cosseno, PLDA e PSDA em geometria simples.

```python
config = make_spherical_baseline_config()
```

Comparar:

| Sistema | Entrada |
|---|---|
| Cosseno | `normalized_embeddings` |
| PLDA bruto | `raw_embeddings` |
| PLDA L2 | `normalized_embeddings` |
| PSDA | `normalized_embeddings` |

## 17.2 Baseline cooperativo

Objetivo:

> verificar se LDA, whitening e PLDA recuperam uma estrutura compartilhada.

```python
config = make_cooperative_plda_config()
```

Interpretação:

- resultado positivo mostra que o pipeline funciona quando a estrutura é aprendível;
- não demonstra desempenho provável em qualquer base AdaFace real.

## 17.3 Stress cross-condition

Objetivo:

> avaliar limites do PLDA homocedástico e comparar estratégias de estimação cross-condition.

```python
config = make_cross_condition_stress_config()
```

Gerar bases alinhadas:

```python
aligned = simulator.sample_cross_condition_datasets(
    identity_count=1_000,
    identity_offset=300_000,
    split_name="cross_test",
    samples_per_identity_by_domain={
        "controlled": (3, 6),
        "surveillance": (3, 6),
    },
)
```

Comparar:

```text
merge ingênuo
PER_CONDITION
BALANCED_IDENTITY
```

Métricas:

- EER;
- `Cllr`;
- `Cllr_min`;
- Tippett;
- estabilidade de LR;
- espectro de \(W\);
- traço de \(W\);
- desempenho por domínio;
- desempenho por quantis de norma;
- desempenho por tipicidade.

## 17.4 Tipicidade

Compare:

1. tipicidade-oráculo sobre centros latentes;
2. MoVMF ajustada sobre centroides observados;
3. k-NN `PER_IDENTITY`;
4. k-NN `PER_FACE`;
5. estratificação por qualidade e norma.

Pergunta importante:

> quanto a estimação da tipicidade muda quando os centroides observados são contaminados por amostras extremas?

---

## 18. Calibração contra embeddings AdaFace reais

Os valores padrão são ilustrativos. Antes de usar resultados sintéticos para afirmações quantitativas, ajuste o simulador contra uma base real representativa.

## 18.1 Preparar dados reais

Para embeddings brutos:

\[
r_{ij}=\lVert\mathbf{x}_{ij}\rVert_2,
\]

\[
\mathbf{u}_{ij}=\frac{\mathbf{x}_{ij}}{r_{ij}}.
\]

## 18.2 Estimar centros por identidade

\[
\widehat{\mathbf{c}}_i
=
\frac{
\sum_j\mathbf{u}_{ij}
}{
\left\|
\sum_j\mathbf{u}_{ij}
\right\|_2
}.
\]

## 18.3 Estatísticas mínimas a reproduzir

| Fenômeno | Diagnóstico |
|---|---|
| norma | histogramas, quantis, mediana, MAD |
| norma por domínio | curvas separadas |
| genuínos | distribuição de cosseno e score PLDA |
| impostores | distribuição de cosseno e score PLDA |
| caudas | quantis extremos |
| intraidentidade | \(\widehat{\mathbf{c}}_i^T\mathbf{u}_{ij}\) |
| relação radial | estabilidade por quantil de norma |
| domínio | diferença controlled vs surveillance |
| estrutura anisotrópica | espectro de covariância intraidentidade |
| população | espectro de covariância entre identidades |
| tipicidade | densidade e vizinhanças |

## 18.4 Ajuste incremental

Não ajuste tudo simultaneamente. Sequência recomendada:

1. distribuição radial;
2. distribuição populacional impostora;
3. dispersão genuína central;
4. penalizações de cauda;
5. diferenças de domínio;
6. nuisance anisotrópico global;
7. bases por domínio;
8. bias domínio × identidade;
9. sessões;
10. validação externa.

## 18.5 Separar ajuste e avaliação

Use identidades disjuntas para:

| Partição | Uso |
|---|---|
| fit do simulador | estimar parâmetros sintéticos |
| desenvolvimento | escolher hiperparâmetros do pipeline |
| teste | avaliação final |

---

## 19. Matriz de ablações recomendada

| Cenário | Qualidade → regime | Qualidade → norma | Qualidade → direção | Norma → direção | Nuisance |
|---|---:|---:|---:|---:|---|
| A0 | não | não | não | não | nenhum |
| A1 | sim | sim | não | não | nenhum |
| A2 | não | não | sim | não | nenhum |
| A3 | sim | sim | não | sim | nenhum |
| A4 | sim | sim | sim | sim | global |
| A5 | sim | sim | sim | sim | por domínio |
| A6 | sim | sim | sim | sim | hierárquico |

Objetivo:

- atribuir causalmente alterações de desempenho;
- evitar interpretar um resultado agregado como efeito isolado de um único mecanismo.

---

## 20. Cuidados interpretativos

### 20.1 O baseline cooperativo não é um upper bound matemático

O perfil `COOPERATIVE_PLDA_BASELINE` tende a favorecer modelos capazes de aprender covariâncias compartilhadas. Isso não significa que exista uma garantia matemática de desempenho superior ao observado em dados reais.

Use a formulação:

> cenário potencialmente otimista e cooperativo para PLDA.

Evite:

> limite superior garantido do PLDA.

### 20.2 mVMF não reproduz toda a rede AdaFace

O gerador modela propriedades selecionadas:

- direção;
- concentração angular;
- norma;
- caudas;
- qualidade latente;
- domínios;
- nuisance.

Ele não reproduz automaticamente:

- arquitetura da rede;
- distribuição completa dos pixels;
- efeitos específicos de pré-processamento;
- vieses demográficos reais;
- artefatos de câmera reais;
- relações causais verdadeiras entre qualidade e reconhecibilidade.

### 20.3 A qualidade latente é uma variável experimental

Não trate `latent_quality` como ground truth universal de qualidade biométrica.

### 20.4 Bases específicas por identidade são um stress test

Aumentar excessivamente:

```python
identity_specific_nuisance_scale
```

pode criar um mundo artificialmente hostil ao PLDA. Use gradações e reporte o valor.

### 20.5 Bias de sessão não substitui um modelo temporal completo

O bias de sessão atual é uma aproximação simples compartilhada por observações da mesma combinação:

```text
identidade × domínio × session_index
```

Ele não representa automaticamente correlações temporais contínuas.

---

## 21. Migração da versão 0.1

## 21.1 Código antigo continua funcionando

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_example_config,
)

simulator = AdaFaceEmbeddingSimulator(
    make_example_config(dimension=512),
    random_seed=2026,
)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 10),
    split_name="test",
    identity_offset=200_000,
)
```

## 21.2 Alteração de reprodutibilidade

Na versão 0.1, o RNG era consumido sequencialmente. Na versão 0.2, streams nomeados tornam a geração independente da ordem das chamadas.

Isso pode alterar os números gerados por scripts antigos mesmo quando a seed do construtor é a mesma. A alteração é intencional.

## 21.3 Coluna radial

Preferir:

```text
reference_scaled_log_norm_z
```

em código novo.

Alias preservado:

```text
robust_norm_z
```

## 21.4 Configuração direcional

Nome novo:

```text
optimal_reference_scaled_log_norm_z
```

Alias de leitura preservado:

```text
optimal_robust_norm_z
```

---

## 22. Testes automatizados

Executar:

```bash
python -m pytest -q
```

Os testes atuais validam:

- normas unitárias da vMF;
- alinhamento da média empírica da vMF;
- crescimento do comprimento resultante com \(\kappa\);
- consistência entre embeddings brutos e metadados;
- independência da ordem das chamadas;
- seeds explícitas;
- compartilhamento de população em cross-condition;
- bases distintas por domínio;
- probabilidades dos regimes radiais;
- classes dos pares;
- estratégia `balanced_observation`;
- diagnóstico de bins;
- adapter.

Para produção, acrescente testes específicos do seu pipeline FaceAudit.

---

## 23. Solução de problemas

### `ValueError: anisotropic_rank must satisfy 0 <= rank < dimension`

Reduza:

```python
anisotropic_rank
```

A factory pronta limita automaticamente o posto para dimensões reduzidas usadas em testes.

### O PLDA está performando bem demais

Verifique se está usando:

```python
make_cooperative_plda_config()
```

Troque para:

```python
make_cross_condition_stress_config()
```

Depois varie:

- `domain_nuisance_weight`;
- `identity_specific_nuisance_scale`;
- `domain_identity_bias_scale`;
- `session_bias_scale`.

### O PLDA está performando mal demais

Reduza gradualmente os mesmos parâmetros. Não conclua que o backend é inadequado apenas com base em um cenário extremo artificial.

### `qcut()` produziu menos bins

O método emite warning. Examine:

- clipping radial;
- quantidade de valores idênticos;
- tamanho da base;
- parâmetros degenerados.

### `RadialReference.fit()` falhou com MAD zero

As normas fornecidas não possuem dispersão suficiente. Verifique se você passou embeddings brutos reais ou vetores já normalizados em L2.

### Adapter não encaixa no construtor real

Ajuste:

```python
dataset_factory_keyword
include_metadata
```

ou use:

```python
dataset_to_embedding_records()
```

para escrever um adapter específico de poucas linhas.

---

## 24. Exemplo completo recomendado

```python
from pathlib import Path

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_cross_condition_stress_config,
    sample_pair_table,
)

output_dir = Path("synthetic_faceaudit_experiment")
output_dir.mkdir(exist_ok=True)

simulator = AdaFaceEmbeddingSimulator(
    make_cross_condition_stress_config(dimension=512),
    random_seed=3030,
)

aligned = simulator.sample_cross_condition_datasets(
    identity_count=1_000,
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

for domain_name, dataset in aligned.items():
    dataset.save_bundle(
        output_dir=output_dir,
        stem=f"test_{domain_name}",
    )

surveillance_pairs = sample_pair_table(
    aligned["surveillance"],
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    random_seed=5001,
    unique_pairs=False,
)

surveillance_pairs.to_csv(
    output_dir / "surveillance_pairs.csv",
    index=False,
)
```

---

## 25. Roadmap sugerido

Melhorias futuras úteis:

1. serialização explícita das bases por domínio no `.npz`;
2. carregamento completo de bundles;
3. enumeração eficiente de todos os trials;
4. mineração de impostores difíceis por centro latente ou backend;
5. modelo temporal de sessão;
6. ajuste automático dos parâmetros contra estatísticas reais;
7. adapter específico do FaceAudit após confirmação da assinatura concreta dos DTOs;
8. métricas prontas de divergência entre dados reais e sintéticos;
9. geração de relatórios diagnósticos;
10. testes de propriedade adicionais para regimes extremos de \(\kappa\).

---

## 26. Conclusão

A versão 0.2 transforma o simulador em uma família de cenários explícitos:

\[
\boxed{
\text{baseline esférico}
\rightarrow
\text{baseline cooperativo}
\rightarrow
\text{stress cross-condition heterocedástico}
}
\]

O uso correto não é buscar um único número sintético para “provar” que um backend funciona. O objetivo é identificar:

- em quais regimes o PLDA ganha do cosseno;
- quando hipóteses homocedásticas começam a falhar;
- em quais intensidades `BALANCED_IDENTITY` supera alternativas ingênuas;
- quanto tipicidade e qualidade ficam acopladas;
- quais propriedades dos embeddings reais precisam ser medidas antes de uma conclusão quantitativa.

Para experimentos auditáveis, sempre registre:

- perfil;
- configuração completa;
- seed estrutural;
- seeds de população e observações;
- política de trials;
- versão do simulador;
- hash do código;
- métricas estratificadas.
