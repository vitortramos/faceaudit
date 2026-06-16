# `adaface_embedding_simulator`

## Documentação completa — gerador sintético de embeddings AdaFace-like (v0.3.0)

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

A arquitetura herdada da versão 0.2.0 separa claramente:

1. o **mundo sintético estrutural**;
2. a **população latente de identidades**;
3. as **observações** produzidas em um ou mais domínios;
4. as **políticas de geração de trials**;
5. os **artefatos de rastreabilidade**.

Essa separação evita um erro comum em experimentos sintéticos: interpretar como generalização real um resultado obtido em um cenário excessivamente cooperativo para o modelo avaliado.

A versão 0.3.0 acrescenta três eixos sobre essa base, sem alterar o comportamento padrão: um mecanismo alternativo de **dispersão intra-classe por calota esférica** (com piso angular interpretável e citável em laudo), o **round-trip completo de configuração** em JSON e o **ancoramento da população em embeddings reais** do FaceAudit. Todo o foco continua sendo a **auditabilidade forense**: cada decisão geométrica é nomeada, configurável, registrada no manifesto e reproduzível por seed.

---

## 2. Visão geral das capacidades 0.3

A versão 0.3.0 organiza-se em **quatro áreas** que podem ser usadas isoladamente ou combinadas. Esta seção é o mapa de uma tela; cada área tem uma seção dedicada adiante.

| Área | O que faz | Entrada principal | Seções |
|---|---|---|---|
| **Amostragem intra-classe por calota** | Substitui (ou complementa) a dispersão vMF por uma calota esférica com **piso angular `lb`** — *"nenhuma observação foi gerada com cosseno menor que `lb` ao centro da identidade"*. Inclui guarda de bissetor anti-colisão. | `DirectionalConfig.intra_class_sampling_mode` | 6.x, 8.4, 15, 19, 21 |
| **Round-trip JSON de config** | Serializa **e** reconstrói uma `SimulatorConfig` completa, fechando o ciclo que na 0.2 só tinha o lado de ida (`config_to_dict`). Permite reabrir um experimento a partir do manifesto. | `config_to_dict` / `config_from_dict` / `config_from_manifest` | 9, 17 |
| **População ancorada em dataset real** | Gera variação sintética em torno de **âncoras biométricas reais** (Modo A: centros médios esféricos) ou de **centros-fantasma MoVMF** ajustados ao dataset (Modo B), com guardas explícitas de vazamento. | `faceaudit_seed` | 10, 12, 20 |
| **Adapter concreto FaceAudit** | Converte o `SyntheticEmbeddingDataset` no DTO real `EmbeddingDataset` (float32, parquet, cross-condition), em vez do *scaffold* genérico de referência. | `faceaudit_concrete_adapter` | 18 |

**O que NÃO mudou (garantia forense de regressão).** O modo padrão de dispersão intra-classe é `IntraClassSamplingMode.VMF` e reproduz a saída da 0.2.0 **bit a bit** — há um teste golden que trava essa invariância. Os modos `IDPERTURB_CAP` e `HYBRID_VMF_CAP` são estritamente *opt-in*. Quem atualiza de 0.2 para 0.3 sem tocar na configuração obtém exatamente os mesmos números (mesma seed estrutural, mesmas seeds de população e observação).

Os três perfis da 0.2 (`SPHERICAL_BASELINE`, `COOPERATIVE_PLDA_BASELINE`, `CROSS_CONDITION_HETEROSCEDASTIC`) continuam disponíveis e inalterados. A 0.3 adiciona um **quarto perfil** pronto, `IDPERTURB_CAP` (`make_idperturb_cap_config`), descrito na Seção 7.

---

## 3. Instalação

O simulador depende apenas de **NumPy** e **pandas**:

```bash
pip install numpy pandas
```

O sampler von Mises–Fisher e o sampler de calota esférica foram implementados diretamente com NumPy. O SciPy **não** é uma dependência obrigatória.

Para rodar os testes basta acrescentar o `pytest`:

```bash
pip install pytest
python -m pytest -q
```

### Modos B e adapter concreto exigem o FaceAudit

As funcionalidades que tocam o projeto hospedeiro têm dependências adicionais opcionais:

- a **população ancorada em dataset real no Modo B** (`population_config_from_dataset` / `simulator_config_from_dataset`) usa o `MoVMFTypicalityModel` (MoVMF) do FaceAudit;
- o **adapter concreto** (`faceaudit_concrete_adapter`) importa os DTOs reais `EmbeddingInfo` / `EmbeddingDataset` e usa parquet (via `pandas`/`pyarrow`, conforme o host).

O **Modo A** (centros médios esféricos) e os helpers de extração/vazamento de `faceaudit_seed` funcionam **sem** o FaceAudit instalado — eles nunca importam o host no topo do módulo, apenas operam sobre arrays NumPy. Use o `.venv` do projeto FaceAudit quando precisar do Modo B ou do adapter concreto.

### Nota sobre `simulator/` sem `__init__.py`

A pasta `simulator/` **não é um pacote Python instalável**: ela ship como um conjunto de **módulos flat de nível superior** (`adaface_embedding_simulator`, `faceaudit_adapter`, `faceaudit_concrete_adapter`, `faceaudit_seed`), importados **sem prefixo de pacote** — espelhando o layout standalone original. Não há `__init__.py`, `setup.py`, `requirements*.txt` nem `pytest.ini` na pasta.

Em consequência, os imports só resolvem se o diretório `simulator/` estiver no `sys.path`. Há duas formas de garantir isso:

1. **Sob `pytest`**: o arquivo `conftest.py` insere o próprio diretório no `sys.path` no início da coleta, de modo que os testes resolvem `import adaface_embedding_simulator` tanto a partir da raiz do repositório quanto de dentro de `simulator/`:

   ```python
   import os, sys
   sys.path.insert(0, os.path.dirname(__file__))
   ```

2. **Em scripts/notebooks próprios**: ou ajuste o **working directory** para `simulator/` antes de importar, ou insira o caminho explicitamente no início do script:

   ```python
   import sys
   sys.path.insert(0, "/caminho/para/faceaudit/simulator")

   from adaface_embedding_simulator import (
       AdaFaceEmbeddingSimulator,
       make_cooperative_plda_config,
   )
   ```

Sem um desses passos, `import adaface_embedding_simulator` falha com `ModuleNotFoundError` mesmo com NumPy e pandas instalados — o problema é de caminho, não de dependência.

---

## 4. Estrutura do pacote

A árvore abaixo foi regenerada a partir da pasta `simulator/` real da versão 0.3.0.

```text
simulator/
├── adaface_embedding_simulator.py      ← módulo principal (núcleo gerativo + perfis)
├── faceaudit_adapter.py                ← adapter genérico de referência (factories injetadas)
├── faceaudit_concrete_adapter.py       ← adapter concreto: DTOs reais do FaceAudit
├── faceaudit_seed.py                   ← população ancorada em dataset real (Modos A e B)
├── example_generate_adaface_embeddings.py  ← script de exemplo executável
├── conftest.py                         ← insere simulator/ no sys.path para os testes
├── README.md
├── DOCUMENTATION.md
├── CHANGELOG.md
└── tests/
    ├── test_adaface_embedding_simulator.py     ← geometria L2, vMF, regimes, reprodutibilidade, pares
    ├── test_config_roundtrip.py                ← config_to_dict ↔ config_from_dict / manifesto
    ├── test_idperturb_cap.py                   ← calota, modulação de lb, guarda do bissetor, híbrido, golden VMF
    ├── test_core_population_injection.py        ← injeção de centros + bridge Modo A (sem faceaudit)
    ├── test_faceaudit_seed.py                  ← Modo B (MoVMF) — requer faceaudit
    ├── test_faceaudit_adapter.py               ← adapter genérico de referência
    └── test_faceaudit_concrete_adapter.py      ← adapter concreto (importa DTOs reais)
```

### `adaface_embedding_simulator.py`

Módulo principal (`__version__ = "0.3.0"`). Contém:

- funções numéricas e geométricas (`l2_normalize_rows`, bases ortonormais, rotação de Householder);
- sampler vetorizado von Mises–Fisher (`sample_vmf`);
- samplers de calota da 0.3 (`sample_spherical_cap`, `sample_hybrid_vmf_cap`);
- enums (`NuisanceStructureMode`, `TangentUpdateMode`, `PairSamplingStrategy`, `IntraClassSamplingMode`);
- classes de configuração (`PopulationConfig`, `RadialConfig`, `DirectionalConfig`, `CouplingConfig`, `AcquisitionDomainConfig`, `SimulatorConfig`);
- serialização de config (`config_to_dict`, `config_from_dict`, `config_from_manifest`);
- classes `RadialReference`, `SyntheticIdentityPopulation`, `SyntheticEmbeddingDataset`;
- classe `AdaFaceEmbeddingSimulator`;
- funções de score e amostragem de pares (`cosine_scores`, `sample_pair_table`);
- os quatro perfis prontos (`make_spherical_baseline_config`, `make_cooperative_plda_config`, `make_cross_condition_stress_config`, `make_idperturb_cap_config`) e o alias legado `make_example_config`.

### `faceaudit_adapter.py`

Adapter **genérico de referência**, desacoplado das classes concretas do FaceAudit. As factories de `EmbeddingInfo` e `EmbeddingDataset` são injetadas pelo usuário. **Atenção:** ele assume `EmbeddingInfo(metadata=...)`, que **não** corresponde ao DTO real do FaceAudit — serve apenas como scaffold ilustrativo. Para produção, use o adapter concreto (Seção 18).

### `faceaudit_concrete_adapter.py`

Adapter **concreto**: importa os DTOs reais do host e mapeia as colunas de metadata do simulador sobre os campos efetivos de `EmbeddingInfo`. Expõe `to_embedding_dataset` e `cross_condition_to_embedding_datasets` (float32, parquet, alinhamento cross-condition).

### `faceaudit_seed.py`

Ancora a população do simulador em um `EmbeddingDataset` real, em dois modos (Modo A: centros médios esféricos; Modo B: centros-fantasma MoVMF via `MoVMFTypicalityModel`), com helpers de extração de centros, ajuste radial e verificação de vazamento (`source_identity_set`, `assert_disjoint_source_identities`).

### `example_generate_adaface_embeddings.py`

Exemplo executável que cria splits disjuntos, pares genuínos e impostores, bundles persistidos e datasets cross-condition alinhados por identidade.

### `conftest.py`

Insere o diretório `simulator/` no `sys.path` para que os testes resolvam os módulos flat independentemente do diretório de execução (ver Seção 3).

### `tests/`

Sete módulos de teste cobrindo geometria L2, vMF, regimes radiais, reprodutibilidade, alinhamento cross-condition, geração de pares, round-trip de configuração, amostragem por calota (incluindo o golden VMF que trava a regressão de saída), injeção de centros, Modo A/B de ancoramento e ambos os adapters.

---

## 5. Início rápido

### 5.1 Gerar um split e salvar o bundle completo

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

test.save_bundle(
    output_dir="synthetic_output",
    stem="test",
)
```

Arquivos produzidos pelo `save_bundle`:

```text
synthetic_output/
├── test_embeddings.npz
├── test_metadata.csv
├── test_norm_diagnostics.csv
└── test_manifest.json
```

O `save_bundle` é o método recomendado para experimentos auditáveis: além dos embeddings e metadados, ele grava o diagnóstico por faixa radial e um **manifesto JSON** com perfil, seeds, dimensão, contagem de amostras/identidades, SHA-256 do código-fonte e a **configuração completa**.

### 5.2 Calota esférica com piso angular (`make_idperturb_cap_config`)

Para impor um **limite geométrico interpretável** à variação intra-identidade — *"nenhuma observação foi gerada com cosseno menor que `lb` ao centro da identidade"* —, use o perfil `IDPERTURB_CAP` em vez da dispersão vMF padrão:

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_idperturb_cap_config,
)

# lb = 0.6 é INSPIRADO no regime do IDperturb (Boutros et al., CVPR 2026),
# NÃO um ótimo para AdaFace/PLDA — calibre empiricamente antes de concluir.
config = make_idperturb_cap_config(dimension=512, lower_bound=0.6)

simulator = AdaFaceEmbeddingSimulator(config, random_seed=2026)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 10),
    split_name="test_cap",
    identity_offset=200_000,
    random_seed=103,
    population_random_seed=1003,
)

# A guarda do bissetor (cap_overlap_guard) impõe um piso por identidade e TEM
# PRIORIDADE: o lb efetivo pode exceder o nominal em regiões densas.
# SEMPRE reporte os percentis do lb efetivo:
eff = test.metadata["cap_lower_bound_effective"]
print(eff.describe(percentiles=[0.1, 0.5, 0.9]))
```

O perfil `IDPERTURB_CAP` parte do baseline cooperativo, troca o modo de dispersão para `IDPERTURB_CAP` e liga a guarda anti-colisão. Detalhes da geometria da calota, da modulação de `lb` e das colunas de metadata novas estão nas Seções 6.x e 15.

### 5.3 Round-trip de configuração: `save_manifest` → `config_from_manifest`

Todo experimento pode ser **reaberto a partir do seu manifesto**, fechando o ciclo de serialização que na 0.2 só tinha o lado de ida:

```python
from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    make_idperturb_cap_config,
    config_to_dict,
    config_from_dict,
    config_from_manifest,
)

# 1) Gera e persiste o manifesto (save_bundle também o escreve)
config = make_idperturb_cap_config(dimension=512)
simulator = AdaFaceEmbeddingSimulator(config, random_seed=2026)
dataset = simulator.sample_dataset(
    identity_count=200,
    samples_per_identity=(3, 6),
    split_name="roundtrip",
    identity_offset=0,
)
dataset.save_manifest("synthetic_output/roundtrip_manifest.json")

# 2) Reconstrói a SimulatorConfig a partir do manifesto persistido
restored = config_from_manifest("synthetic_output/roundtrip_manifest.json")

# 3) Round-trip puro em memória (dict <-> config)
assert config_from_dict(config_to_dict(config)) == config

# 4) A config reconstruída reproduz a geração com a mesma seed estrutural
replay = AdaFaceEmbeddingSimulator(restored, random_seed=2026)
```

`config_from_dict` valida a configuração antes de devolvê-la; chaves ausentes caem para os defaults da dataclass (compatível com manifestos escritos por versões anteriores), enquanto chaves desconhecidas levantam erro — expondo *version drift* cedo, em vez de silenciá-lo. Mais detalhes na Seção 9.

## 6. Modelo generativo

O gerador possui seis níveis hierárquicos, do mundo compartilhado até o embedding bruto de cada observação:

1. mundo estrutural;
2. componentes populacionais;
3. identidade latente;
4. domínio, qualidade e regime radial;
5. **dispersão intra-classe** (vMF, calota cosseno-uniforme ou híbrido);
6. nuisance tangencial e embedding bruto.

A decomposição fundamental permanece:

\[
\mathbf{x}_{ij}=r_{ij}\widetilde{\mathbf{u}}_{ij},
\]

em que \(r_{ij}=\lVert\mathbf{x}_{ij}\rVert_2\) é a norma e \(\widetilde{\mathbf{u}}_{ij}\in S^{d-1}\) é a direção final (já submetida à dispersão intra-classe e às perturbações tangenciais).

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

> Nota (0.3): este caminho de amostragem de centros pode ser **substituído por injeção direta**. Ao passar `centers=` para `sample_identity_population`, os centros vMF não são sorteados (o RNG não é consumido) e a população é ancorada em direções fornecidas externamente — por exemplo, médias esféricas reais por identidade extraídas de um dataset FaceAudit (Modo A, Seção 12). O caminho amostrado padrão permanece bit-a-bit idêntico ao v0.2.

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

A mesma lógica é aplicada à cauda alta. O acoplamento qualidade → regime é controlado pelo switch `coupling.quality_affects_regime_probabilities`.

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

O modelo não impõe que norma alta seja sempre melhor. Existem uma faixa central, uma cauda inferior e uma cauda superior. A contribuição direta da qualidade à norma é governada por `coupling.quality_affects_log_norm`.

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

O nome é deliberadamente explícito. O valor só é robusto quando a referência foi estimada por uma regra robusta. A coluna legada `robust_norm_z` continua disponível como alias para preservar compatibilidade.

### 6.8 Dispersão intra-classe

A direção observada é dispersa em torno do centro latente \(\mathbf{c}_i\). A versão 0.3 introduz **três modos** de dispersão intra-classe — a subseção 6.13 descreve os três em detalhe. O modo padrão é `VMF`, idêntico ao v0.2:

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

A concentração \(\kappa_{ij}\) segue:

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

As contribuições de qualidade e de norma sobre \(\kappa\) são respectivamente gatilhadas pelos switches `coupling.quality_affects_intra_kappa` e `coupling.norm_z_affects_intra_kappa`.

### 6.9 Perturbações tangenciais

Após a dispersão intra-classe, o simulador adiciona uma perturbação estruturada:

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

> Interação com a calota (0.3): o piso de cosseno `lb` é garantido sobre a direção **pré-nuisance** \(\mathbf{u}_{ij}^{(0)}\). O passo tangencial de nuisance/viés pode empurrar o cosseno realizado (pós-nuisance) ligeiramente abaixo de `lb`. Para auditar a calota, use sempre a coluna `pre_nuisance_center_cosine` (Seção 6.13), não `own_center_cosine`.

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

### 6.13 Dispersão intra-classe: VMF, IDPERTURB_CAP e HYBRID_VMF_CAP

A dispersão das observações em torno do centro de identidade \(\mathbf{c}_i\) (nível 5 do modelo generativo) tem três modos, escolhidos por `DirectionalConfig.intra_class_sampling_mode` (enum `IntraClassSamplingMode`):

| Modo | Função geradora | Controle | Interpretação |
|---|---|---|---|
| `VMF` (padrão) | `sample_vmf` | \(\kappa\) | concentração angular média (distribuição direcional, sem piso angular rígido) |
| `IDPERTURB_CAP` | `sample_spherical_cap` | `lb` | **afastamento angular máximo** permitido (garante cosseno ≥ `lb`) |
| `HYBRID_VMF_CAP` | `sample_hybrid_vmf_cap` | \(\kappa\) + `lb` | corpo vMF com **piso rígido** de identidade em `lb` |

O modo padrão é `VMF` — a saída do v0.2 permanece **bit-a-bit idêntica** (um teste golden trava isso). Os modos `IDPERTURB_CAP` e `HYBRID_VMF_CAP` são **opt-in**.

#### A construção da calota (IDperturb, Boutros et al., CVPR 2026)

Para cada amostra, a função `sample_spherical_cap` sorteia um cosseno-alvo \(s\sim U[lb,1]\), gera um ruído \(\mathbf{n}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\) projetado no hiperplano tangente ortogonal a \(\mathbf{c}_i\) (mesma projeção tangencial do nuisance), produzindo \(\mathbf{u}\perp\mathbf{c}_i\) com \(\lVert\mathbf{u}\rVert=1\), e constrói:

\[
\widetilde{\mathbf{v}} = s\,\mathbf{c}_i + \sqrt{1 - s^2}\;\mathbf{u}, \qquad \mathbf{u}\perp\mathbf{c}_i,\ \lVert\mathbf{u}\rVert=1.
\]

Isso garante \(\lVert\widetilde{\mathbf{v}}\rVert=1\) e \(\langle\widetilde{\mathbf{v}},\mathbf{c}_i\rangle = s \ge lb\). O caso `lb = 1` produz \(\widetilde{\mathbf{v}}=\mathbf{c}_i\) (sem perturbação).

**Cuidado de nomenclatura crítico:** isto é **uniforme em cosseno** (\(s\) uniforme), **não** uniforme na área da calota — em alta dimensão essas duas distribuições são radicalmente diferentes. Como consequência direta:

\[
\mathbb{E}\!\left[\langle\widetilde{\mathbf{v}},\mathbf{c}_i\rangle\right] = \frac{1 + lb}{2}.
\]

Interprete `lb` como um limite geométrico de excursão angular, não como uma medida de densidade da calota.

#### Interpretação de `lb` para laudo

`lb` é muito mais transparente que \(\kappa\) em 512-D: *"nenhuma observação sintética desta identidade foi gerada com cosseno menor que `lb` em relação ao centro latente, antes dos ruídos de domínio/sessão."* `lb` baixo ⇒ mais diversidade intra-classe (e EER maior); `lb` perto de 1 ⇒ identidade mais rígida.

#### O caminho generativo de `lb` e a modulação por coupling

`lb` é modulado pelas mesmas covariáveis do \(\kappa\), espelhando `_calculate_intra_kappas`. A construção, em escala de cosseno, é:

\[
lb_{ij} = \underbrace{\text{base}}_{\texttt{cap\_base\_lower\_bound}}
        + \underbrace{\Delta_{\text{domínio}}}_{\texttt{cap\_lower\_bound\_offset}}
        + \beta_q\,(q_{ij} - 0{,}5)
        - \lambda_{\text{low}}\,d_{\text{low}}^2
        - \lambda_{\text{high}}\,d_{\text{high}}^2,
\]

com \(d_{\text{low}}=\max(0, z_0 - z_{r,ij})\) e \(d_{\text{high}}=\max(0, z_{r,ij} - z_0)\). Em seguida o valor é cortado para `[cap_minimum_lower_bound, cap_maximum_lower_bound]`.

Os termos de qualidade e de norma são governados pelos **mesmos switches** de `CouplingConfig` usados para \(\kappa\):

- `coupling.quality_affects_intra_kappa` gatilha o termo \(\beta_q\,(q_{ij}-0{,}5)\) (slope `cap_quality_lower_bound_slope`);
- `coupling.norm_z_affects_intra_kappa` gatilha as penalidades de cauda \(\lambda_{\text{low}}\), \(\lambda_{\text{high}}\) (`cap_low_tail_penalty`, `cap_high_tail_penalty`).

Nos modos de calota, esses switches controlam a **dispersão intra-classe**, apesar de o nome histórico mencionar `kappa`. Quando ambos estão desligados, `lb` depende apenas da base, do offset de domínio e da guarda de bissetor.

**Escala — não reaproveite as penalidades de \(\kappa\):** as penalidades de calota agem em **escala de cosseno**, não em \(\log\kappa\). Uma penalidade de `0.15` com \(z=2\) derruba `lb` em `0.6` diretamente no cosseno. Calibre `cap_low_tail_penalty`/`cap_high_tail_penalty` de forma independente das penalidades quadráticas direcionais (`low_tail_quadratic_penalty`/`high_tail_quadratic_penalty`), que entram em \(\log\kappa\).

**Offset por domínio** (`AcquisitionDomainConfig.cap_lower_bound_offset`) — aperta ou afrouxa a identidade por condição de aquisição, espelhando `log_kappa_shift` para a parametrização de calota:

```text
controlled:   cap_lower_bound_offset = +0.10   # identidade mais rígida
surveillance: cap_lower_bound_offset = -0.15   # mais dispersão intra-classe
```

#### Regra do bissetor (`cap_overlap_guard`)

Para evitar que a perturbação de uma identidade fique angularmente mais próxima de outra identidade do que do próprio centro, define-se um **piso** por identidade a partir da densidade local dos centros (Boutros et al. 2026, Eq. 6):

\[
\text{floor}_i = \cos\!\Big(\tfrac{1}{2}\min_{j \ne i} \angle(\mathbf{c}_i, \mathbf{c}_j)\Big)
              = \sqrt{\tfrac{1 + \max_{j \ne i} \mathbf{c}_i^\top \mathbf{c}_j}{2}}.
\]

O `lb` efetivo é \(\max(lb_{\text{modulado}}, \text{floor}_i)\). **A guarda tem prioridade e é aplicada DEPOIS do clip**, podendo produzir um `lb` efetivo `> cap_maximum_lower_bound` em regiões densas — isso é intencional, não bug (anticolisão tem prioridade sobre o teto configurado). Singletons recebem piso `0`. O cálculo é feito em blocos de linhas (`chunk_size=2048`), sem materializar a matriz C×C de cossenos (tempo \(O(C^2 d)\), memória \(O(\text{chunk}\cdot C)\)); o vetor é cacheado em `SyntheticIdentityPopulation.overlap_lower_bound_floor` para reuso entre observações cross-condition.

Em 512-D, mesmo centros aleatórios têm vizinhos relativamente próximos. Logo, com `cap_overlap_guard=True`, um `lb` nominal de `0.4` **não** significa que as amostras foram geradas com piso `0.4`: o piso efetivo pode ser bem maior para identidades densas. **Sempre reporte `cap_lower_bound_effective` (mean/p10/p50/p90)** junto do `lb` nominal — a separação intra-classe realizada difere da `lb` nominal configurada. Ligação com tipicidade: regiões densas (identidades típicas) toleram menos variação; regiões raras toleram mais.

#### Modo HYBRID_VMF_CAP

A função `sample_hybrid_vmf_cap` amostra vMF e, por rejeição, reamostra as observações cuja \(\cos < lb\) (até `hybrid_max_resample_rounds` rodadas). As observações que ainda violam o piso após as rodadas **caem para a calota** (`sample_spherical_cap`) em `lb`, garantindo o piso rígido. A função retorna, por amostra, o número de reamostragens vMF e se o fallback de calota foi usado.

As colunas `hybrid_resample_count` e `hybrid_used_cap_fallback` permitem **auditar a taxa de fallback**: com `lb` alto e \(\kappa\) baixo, grande parte das amostras pode acabar vindo da calota — nesse caso o modo deixa de ser, na prática, "vMF truncado" e passa a ser dominado pela geometria da calota. Reporte a taxa de fallback ao usar este modo.

#### Colunas de metadata da dispersão intra-classe

| Coluna | Significado |
|---|---|
| `intra_class_sampling_mode` | modo usado (`vmf` / `idperturb_cap` / `hybrid_vmf_cap`) |
| `cap_lower_bound_effective` | `lb` efetivo por amostra (NaN no modo `VMF`) |
| `pre_nuisance_center_cosine` | cosseno ao centro **logo após** vMF/cap/hybrid, antes do passo tangencial |
| `hybrid_resample_count` | reamostragens vMF (0 fora de `HYBRID_VMF_CAP`) |
| `hybrid_used_cap_fallback` | usou fallback de calota (False fora de `HYBRID_VMF_CAP`) |

**Invariante a verificar:** `pre_nuisance_center_cosine ≥ cap_lower_bound_effective` (a calota é garantida na direção pré-nuisance). O `own_center_cosine` (pós-nuisance) **pode cair abaixo** de `lb` por causa do passo tangencial de nuisance/viés — para auditar a calota use `pre_nuisance_center_cosine`.

#### Caveat metodológico

O IDperturb original gera **imagens** via difusão e treina um reconhecedor facial; o nosso simulador gera **embeddings diretamente**. O elo é **geométrico, não quantitativo**. O `lb = 0.6` do perfil `IDPERTURB_CAP` é **inspirado** no regime do artigo, **não** um ótimo para AdaFace/PLDA/FaceAudit sem calibração empírica. Formulação recomendada para laudo: *MoVMF modela população/tipicidade; vMF modela a dispersão estatística; IDPERTURB-cap impõe um limite geométrico interpretável e citável para a variação intra-identidade.*

---

## 7. Perfis prontos

A versão 0.3 fornece **quatro** perfis prontos (a v0.2 tinha três; o quarto, `IDPERTURB_CAP`, é novo). Todos retornam um `SimulatorConfig` validado.

### 7.1 `SPHERICAL_BASELINE`

Factory:

```python
make_spherical_baseline_config(dimension=512)
```

Características:

- mistura populacional vMF (`component_count=8`, `identity_center_kappas=70`);
- variação intraidentidade vMF;
- sem nuisance anisotrópico (`anisotropic_rank=0`, `anisotropic_base_scale=0.0`);
- domínio único `default`, sem bias cross-condition.

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

- base anisotrópica global compartilhada (`anisotropic_rank=min(24, d-1)`, `anisotropic_base_scale=0.010`);
- relação não monotônica norma ↔ estabilidade;
- domínios `controlled` e `surveillance`;
- bias médio pequeno para vigilância (`domain_bias_scale=0.008`);
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

- nuisance hierárquico (`nuisance_structure_mode=HIERARCHICAL`, `global_nuisance_weight=0.55`, `domain_nuisance_weight=0.90`);
- bases diferentes por domínio;
- bases específicas por identidade (`identity_specific_nuisance_rank=min(2, d-1)`, `identity_specific_nuisance_scale=0.40`);
- bias médio de domínio, bias domínio × identidade e bias de sessão;
- mapa exponencial;
- domínio `surveillance` com maior dispersão (`anisotropic_noise_multiplier=1.85`, `log_kappa_shift=-0.35`).

Uso recomendado:

- comparar merge ingênuo, `PER_CONDITION` e `BALANCED_IDENTITY`;
- estudar robustez fora de domínio;
- medir inflação ou subestimação de \(W\);
- testar calibração sob heterocedasticidade.

### 7.4 `IDPERTURB_CAP`

Factory:

```python
make_idperturb_cap_config(dimension=512, lower_bound=0.6)
```

Este perfil parte do `COOPERATIVE_PLDA_BASELINE` e **troca o modo de dispersão intra-classe** para a calota cosseno-uniforme (`IntraClassSamplingMode.IDPERTURB_CAP`), via `replace` no bloco `directional`. Diferenças em relação ao cooperativo:

| Campo | Valor no perfil |
|---|---:|
| `intra_class_sampling_mode` | `IDPERTURB_CAP` |
| `cap_base_lower_bound` | `lower_bound` (default `0.6`) |
| `cap_quality_lower_bound_slope` | `0.10` |
| `cap_low_tail_penalty` | `0.15` |
| `cap_high_tail_penalty` | `0.20` |
| `cap_overlap_guard` | `True` |

O parâmetro `lower_bound` é o **piso de cosseno base** para a dispersão intra-classe. Como `cap_overlap_guard=True`, o `lb` efetivo pode **exceder** `lower_bound` para identidades em regiões densas da população (ver Seção 6.13).

Uso recomendado:

- impor um limite geométrico interpretável e citável para a variação intra-identidade;
- executar um *sweep* de `lb` para mapear a relação dispersão genuína ↔ EER (ver Seção 19);
- contrastar calota cosseno-uniforme contra vMF como eixo de ablação (ver Seção 21).

> Lembrete: o `lb = 0.6` é inspirado no regime do IDperturb (Boutros et al. 2026), **não** um ótimo calibrado para AdaFace/PLDA/FaceAudit. Calibre empiricamente antes de qualquer afirmação quantitativa.

---

## 8. Referência completa da configuração

### 8.1 `SimulatorConfig`

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

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `dimension` | `int` | `512` | dimensão dos embeddings (mínimo 3) |
| `profile_name` | `str` | `custom` | nome persistido no manifesto |
| `population` | `PopulationConfig` | default | estrutura populacional |
| `radial` | `RadialConfig` | default | distribuição das normas |
| `directional` | `DirectionalConfig` | default | dispersão angular, calota e nuisance |
| `coupling` | `CouplingConfig` | default | switches de ablação |
| `acquisition_domains` | `tuple[AcquisitionDomainConfig, ...]` | `(default,)` | parâmetros por domínio (nomes únicos) |

---

### 8.2 `PopulationConfig`

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

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `component_count` | `int` | `8` | quantidade de regiões populacionais |
| `component_weights` | vetor ou `None` | `None` | pesos da mistura; `None` usa pesos uniformes |
| `identity_center_kappas` | escalar ou vetor | `70.0` | concentração dos centros de identidade dentro de cada componente |
| `component_means` | matriz `(K, d)` ou `None` | `None` | médias explícitas dos componentes |
| `component_mean_global_kappa` | `float` | `20.0` | proximidade dos componentes em torno da média global |
| `global_mean_direction` | vetor ou `None` | `None` | média global explícita |

#### Como interpretar `identity_center_kappas`

Valor maior: identidades mais agrupadas dentro do componente; maior densidade local; impostores potencialmente mais semelhantes. Valor menor: centros mais dispersos; população mais espalhada; menor densidade local.

---

### 8.3 `RadialConfig`

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

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `reference_log_norm_mean` | `float` | `log(25)` | centro da padronização radial |
| `reference_log_norm_scale` | `float` | `0.24` | escala da padronização |
| `reference_method` | `str` | `configured_reference` | descrição persistida nos metadados |
| `quality_log_norm_slope` | `float` | `0.12` | contribuição direta da qualidade à log-norma |
| `central_offset` | `float` | `0.0` | deslocamento do regime central |
| `central_sigma` | `float` | `0.10` | dispersão do regime central |
| `low_tail_offset` | `float` | `-0.70` | deslocamento da cauda inferior |
| `low_tail_sigma` | `float` | `0.16` | dispersão da cauda inferior |
| `high_tail_offset` | `float` | `0.70` | deslocamento da cauda superior |
| `high_tail_sigma` | `float` | `0.16` | dispersão da cauda superior |
| `minimum_norm` | `float` | `1e-6` | clipping inferior |
| `maximum_norm` | `float` | `1e6` | clipping superior |

#### Ajustar referência radial contra dados reais

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

`RadialReference.fit()` usa \(\operatorname{mediana}(\log r)\) como localização e \(1{,}4826\cdot\operatorname{MAD}(\log r)\) como escala.

---

### 8.4 `DirectionalConfig`

Esta dataclass concentra o modelo de confiabilidade angular (\(\kappa\)), o **modo de dispersão intra-classe e os parâmetros da calota cosseno-uniforme (0.3)**, e o modelo opcional de nuisance tangencial.

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

    # Dispersão intra-classe e calota cosseno-uniforme (0.3)
    intra_class_sampling_mode: str = "vmf"
    cap_base_lower_bound: float = 0.6
    cap_quality_lower_bound_slope: float = 0.10
    cap_low_tail_penalty: float = 0.0
    cap_high_tail_penalty: float = 0.0
    cap_minimum_lower_bound: float = 0.0
    cap_maximum_lower_bound: float = 0.999
    cap_overlap_guard: bool = True
    hybrid_max_resample_rounds: int = 8
```

#### Parâmetros da dispersão vMF (\(\kappa\))

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `peak_intra_kappa` | `float` | `1400` | estabilidade máxima em condições favoráveis |
| `quality_log_kappa_slope` | `float` | `0.90` | efeito direto da qualidade sobre \(\log\kappa\) |
| `optimal_reference_scaled_log_norm_z` | `float` | `0.0` | faixa radial ótima \(z_0\) (compartilhada com a calota) |
| `low_tail_quadratic_penalty` | `float` | `0.55` | penalização de \(\log\kappa\) na cauda baixa |
| `high_tail_quadratic_penalty` | `float` | `0.75` | penalização de \(\log\kappa\) na cauda alta |
| `minimum_intra_kappa` | `float` | `20` | clipping inferior de \(\kappa\) |
| `maximum_intra_kappa` | `float` | `10000` | clipping superior de \(\kappa\) |

> Alias de leitura: a property `optimal_robust_norm_z` retorna `optimal_reference_scaled_log_norm_z` (compat com a documentação 0.1).

#### Parâmetros da dispersão intra-classe e da calota cosseno-uniforme (0.3)

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `intra_class_sampling_mode` | `IntraClassSamplingMode` ou `str` | `vmf` | modo de dispersão: `vmf` / `idperturb_cap` / `hybrid_vmf_cap` |
| `cap_base_lower_bound` | `float` | `0.6` | piso de cosseno base `lb` antes da modulação (em `[0, 1]`) |
| `cap_quality_lower_bound_slope` | `float` | `0.10` | inclinação do termo de qualidade sobre `lb` (qualquer sinal); gatilhado por `quality_affects_intra_kappa` |
| `cap_low_tail_penalty` | `float` | `0.0` | penalidade de `lb` na cauda baixa, **em escala de cosseno**; gatilhada por `norm_z_affects_intra_kappa` |
| `cap_high_tail_penalty` | `float` | `0.0` | penalidade de `lb` na cauda alta, **em escala de cosseno**; gatilhada por `norm_z_affects_intra_kappa` |
| `cap_minimum_lower_bound` | `float` | `0.0` | clip inferior de `lb` (antes da guarda) |
| `cap_maximum_lower_bound` | `float` | `0.999` | clip superior de `lb` (antes da guarda; a guarda pode ultrapassá-lo) |
| `cap_overlap_guard` | `bool` | `True` | aplica o piso de bissetor por identidade (anticolisão) |
| `hybrid_max_resample_rounds` | `int` | `8` | rodadas máximas de reamostragem vMF no modo `HYBRID_VMF_CAP` (mínimo 1) |

**Semântica da guarda de bissetor.** Quando `cap_overlap_guard=True`, o `lb` modulado e cortado é elevado para o piso por identidade \(\text{floor}_i=\sqrt{(1+\max_{j\ne i}\mathbf{c}_i^\top\mathbf{c}_j)/2}\). A guarda é aplicada **após** o clip e tem **prioridade**: o `lb` efetivo realizado pode **exceder** `cap_maximum_lower_bound` em regiões densas (comportamento intencional — anticolisão prevalece sobre o teto). Por isso, a separação intra-classe efetiva difere da `lb` nominal configurada; reporte sempre os percentis de `cap_lower_bound_effective`. Detalhamento completo, fórmulas e cuidados em Seção 6.13.

#### Parâmetros de nuisance anisotrópico

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `anisotropic_rank` | `int` | `0` | posto das bases global e por domínio (`0 <= rank < dimension`) |
| `anisotropic_base_scale` | `float` | `0.0` | amplitude base do nuisance |
| `anisotropic_quality_slope` | `float` | `1.0` | aumenta nuisance quando a qualidade cai |
| `nuisance_structure_mode` | `NuisanceStructureMode` ou `str` | `global_shared` | compartilhamento da base |
| `global_nuisance_weight` | `float` | `1.0` | peso da base global |
| `domain_nuisance_weight` | `float` | `0.0` | peso das bases por domínio no modo hierárquico |
| `identity_specific_nuisance_rank` | `int` | `0` | posto da base específica por identidade |
| `identity_specific_nuisance_scale` | `float` | `0.0` | amplitude da parcela específica por identidade |
| `tangent_update_mode` | `TangentUpdateMode` ou `str` | `retraction` | retorno à hiperesfera |

#### Modos de nuisance

```python
from adaface_embedding_simulator import NuisanceStructureMode
```

| Valor | Descrição |
|---|---|
| `global_shared` | uma base compartilhada pelo dataset inteiro |
| `per_domain` | uma base diferente para cada domínio |
| `hierarchical` | combinação de base global, base por domínio e parcela específica da identidade |

#### Modos de atualização tangencial

```python
from adaface_embedding_simulator import TangentUpdateMode
```

| Valor | Descrição |
|---|---|
| `retraction` | soma e renormalização; rápida e compatível com v0.1 |
| `exponential_map` | deslocamento geodésico exato no plano tangente da direção atual |

#### Modos de dispersão intra-classe

```python
from adaface_embedding_simulator import IntraClassSamplingMode
```

| Valor | Descrição |
|---|---|
| `vmf` | concentração angular von Mises-Fisher (padrão; comportamento v0.2 bit-a-bit) |
| `idperturb_cap` | calota cosseno-uniforme com piso `lb` (afastamento angular máximo) |
| `hybrid_vmf_cap` | corpo vMF com piso rígido em `lb` e fallback de calota |

---

### 8.5 `CouplingConfig`

```python
@dataclass(frozen=True)
class CouplingConfig:
    quality_affects_regime_probabilities: bool = True
    quality_affects_log_norm: bool = True
    quality_affects_intra_kappa: bool = True
    norm_z_affects_intra_kappa: bool = True
```

| Parâmetro | Tipo | Padrão | Caminho causal |
|---|---|---:|---|
| `quality_affects_regime_probabilities` | `bool` | `True` | qualidade → regime radial |
| `quality_affects_log_norm` | `bool` | `True` | qualidade → norma |
| `quality_affects_intra_kappa` | `bool` | `True` | qualidade → dispersão intra-classe (\(\kappa\) **e** `lb`) |
| `norm_z_affects_intra_kappa` | `bool` | `True` | norma → dispersão intra-classe (\(\kappa\) **e** `lb`) |

> Nota (0.3): nos modos de calota, os dois últimos switches gatilham a modulação de `lb` exatamente como gatilham a de \(\kappa\) no modo vMF — apesar do nome histórico `intra_kappa`. Desligá-los nas configurações de calota torna `lb` insensível a qualidade/norma.

#### Exemplo de ablação puramente radial

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

#### Exemplo de ablação norma → direção

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

### 8.6 `AcquisitionDomainConfig`

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

    # Offset de calota por domínio (0.3)
    cap_lower_bound_offset: float = 0.0
```

#### Domínio e qualidade

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `name` | `str` | obrigatório | nome do domínio (não vazio, único) |
| `weight` | `float` | `1.0` | peso quando domínios são sorteados em conjunto |
| `quality_beta_alpha` | `float` | `4.0` | parâmetro \(\alpha\) da Beta da qualidade |
| `quality_beta_beta` | `float` | `2.0` | parâmetro \(\beta\) da Beta da qualidade |

#### Probabilidades radiais

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `central_regime_probability` | `float` | `0.86` | probabilidade-base do regime central |
| `low_tail_probability` | `float` | `0.08` | probabilidade-base da cauda baixa |
| `high_tail_probability` | `float` | `0.06` | probabilidade-base da cauda alta |
| `low_tail_quality_logit_slope` | `float` | `2.0` | aumenta a cauda baixa quando a qualidade cai |
| `high_tail_quality_logit_slope` | `float` | `0.0` | efeito análogo na cauda alta |

#### Efeitos de domínio

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `log_norm_shift` | `float` | `0.0` | deslocamento radial médio |
| `log_norm_sigma_multiplier` | `float` | `1.0` | multiplicador da dispersão radial |
| `log_kappa_shift` | `float` | `0.0` | deslocamento da estabilidade angular \(\log\kappa\) |
| `anisotropic_noise_multiplier` | `float` | `1.0` | multiplicador do nuisance anisotrópico |
| `domain_bias_scale` | `float` | `0.0` | bias médio do domínio |
| `domain_identity_bias_scale` | `float` | `0.0` | heterogeneidade domínio × identidade |
| `session_bias_scale` | `float` | `0.0` | variação entre sessões |
| `cap_lower_bound_offset` | `float` | `0.0` | **(0.3)** deslocamento aditivo do piso de calota `lb` por domínio |

**Sobre `cap_lower_bound_offset` (0.3).** Espelha `log_kappa_shift` para a parametrização de calota: nos modos `IDPERTURB_CAP`/`HYBRID_VMF_CAP`, esse offset é somado ao `lb` base de cada amostra do domínio antes do clip e da guarda. Aceita **qualquer sinal**: positivo aperta a identidade (condição controlada, menos dispersão intra-classe), negativo a afrouxa (vigilância, mais dispersão). No modo `VMF` o campo não tem efeito.

```text
controlled:   cap_lower_bound_offset = +0.10
surveillance: cap_lower_bound_offset = -0.15
```

## 9. Serialização de configuração e round-trip

Um experimento sintético só é auditável se a **configuração exata** que o gerou puder ser
recuperada bit-a-bit a partir de um artefato em disco. A versão 0.3 fecha esse ciclo com três
funções de módulo — `config_to_dict`, `config_from_dict` e `config_from_manifest` — todas
operando sobre `SimulatorConfig` e suas DTOs aninhadas (`PopulationConfig`, `RadialConfig`,
`DirectionalConfig`, `CouplingConfig`, `AcquisitionDomainConfig`).

### 9.1 `config_to_dict`

```python
from adaface_embedding_simulator import config_to_dict, make_cooperative_plda_config

config = make_cooperative_plda_config(dimension=512)
payload = config_to_dict(config)
```

`config_to_dict` chama `dataclasses.asdict(config)` e passa o resultado por um normalizador
JSON-ready (`_json_ready`) que:

- converte qualquer `Enum` em seu **valor de string** (ex.: `IntraClassSamplingMode.VMF → "vmf"`,
  `TangentUpdateMode.RETRACTION → "retraction"`, `NuisanceStructureMode.GLOBAL_SHARED →
  "global_shared"`);
- converte `np.ndarray` em listas Python e `np.generic` em escalares nativos;
- recursa em `Mapping`, `tuple` e `list`.

A saída é um dicionário aninhado puramente serializável (sem objetos NumPy, sem enums), pronto
para `json.dumps`. Os campos opcionais que aceitam arrays — `component_weights`,
`component_means`, `global_mean_direction`, `identity_center_kappas` — são emitidos como listas
(ou `None`), nunca como `ndarray`.

### 9.2 `config_from_dict`

```python
from adaface_embedding_simulator import config_from_dict

config = config_from_dict(payload)
```

`config_from_dict` reconstrói o `SimulatorConfig` a partir da saída de `config_to_dict`,
reinstanciando cada DTO aninhada e a tupla de `AcquisitionDomainConfig`. Antes de retornar, ele
chama `config.validate()`, de forma que **uma configuração inválida nunca atravessa o
round-trip silenciosamente**.

Pontos importantes do contrato:

- **Arrays permanecem listas.** Os campos de array não são reconvertidos para `np.ndarray` na
  desserialização — todo consumidor interno chama `np.asarray` no ponto de uso, então a geração
  reproduz idêntica.
- **Enums permanecem strings.** Os campos `intra_class_sampling_mode`, `tangent_update_mode` e
  `nuisance_structure_mode` voltam como seu valor de string; tanto as DTOs quanto o simulador
  aceitam string ou membro de enum (a normalização interna usa `_enum_value`).

### 9.3 Semântica de chaves: faltantes-default vs desconhecidas-erro

A política de chaves do `config_from_dict` é deliberada e assimétrica:

| Situação | Comportamento | Razão de design |
|---|---|---|
| Chave **faltante** | recai no **default** da dataclass | compatibilidade futura: um config escrito por uma versão antiga (sem os campos `cap_*`, por exemplo) ainda carrega; os campos novos assumem o default 0.3 |
| Chave **desconhecida** | **levanta `TypeError`** | os campos chegam às DTOs frozen via `**data["..."]`; uma chave inexistente vira argumento inválido do construtor e aborta — expondo *version drift* cedo, em vez de descartar parâmetros silenciosamente |

Concretamente, `population`, `radial`, `directional` e cada `acquisition_domains[i]` são
instanciados via `*Config(**bloco)`, então qualquer chave extra no bloco quebra a chamada. Os
blocos `coupling` e `profile_name` toleram ausência via `data.get(...)`; `dimension`,
`population`, `radial`, `directional` e `acquisition_domains` são acessados diretamente
(`data["..."]`) e são, portanto, obrigatórios.

> **Implicação forense.** Se um manifesto antigo for lido por uma versão mais nova do simulador,
> os parâmetros de calota (`cap_base_lower_bound`, `cap_overlap_guard`, etc.) e o modo
> `intra_class_sampling_mode` recebem seus defaults 0.3 — e como o default é `VMF`, o
> comportamento herdado permanece o do v0.2. Já um manifesto novo lido por um binário antigo que
> desconhecesse esses campos falharia ruidosamente, em vez de gerar dados com uma configuração
> diferente da declarada.

### 9.4 `config_from_manifest`

```python
from adaface_embedding_simulator import config_from_manifest

config = config_from_manifest("synthetic_output/test_manifest.json")
```

`config_from_manifest` lê o arquivo de manifesto produzido por `save_manifest`/`save_bundle`,
extrai o bloco `"config"` e delega a `config_from_dict`. Se o manifesto não tiver um bloco
`"config"`, levanta `ValueError`. Esse é o caminho recomendado para **reabrir um experimento
inteiro** a partir do bundle persistido: o manifesto guarda, além do bloco de configuração, o
perfil, as seeds estrutural/população/observação, o stream, a dimensão, a contagem de
identidades e o SHA-256 do código-fonte (ver Seção 17).

```python
# Reprodução completa a partir do disco
config = config_from_manifest("synthetic_output/test_manifest.json")
simulator = AdaFaceEmbeddingSimulator(config, random_seed=2026)
```

### 9.5 `RadialReference.fit` + `RadialConfig.from_reference` — a ponte de dados reais

A serialização garante reprodutibilidade *dentro* do mundo sintético. Para ancorar a escala
radial em **dados reais**, o par `RadialReference.fit` + `RadialConfig.from_reference` permite
substituir a referência configurada por uma estimada empiricamente.

```python
from adaface_embedding_simulator import RadialReference, RadialConfig

reference = RadialReference.fit(real_raw_embeddings)  # aceita (N, d) ou vetor de normas

radial = RadialConfig.from_reference(
    reference,
    quality_log_norm_slope=0.12,
    low_tail_offset=-0.70,
    high_tail_offset=0.70,
)
```

`RadialReference.fit` aceita tanto a matriz bruta `(N, d)` (calcula `‖x‖₂` por linha) quanto um
vetor 1-D de normas. A estimativa é **robusta**:

\[
\text{location} = \operatorname{mediana}(\log r),
\qquad
\text{scale} = 1{,}4826 \cdot \operatorname{MAD}(\log r).
\]

O `method` resultante é gravado como `"median_scaled_mad"` e propaga para o metadado
`radial_reference_method` de cada amostra. `RadialConfig.from_reference` copia `location`,
`scale` e `method` para os campos `reference_log_norm_mean`, `reference_log_norm_scale` e
`reference_method`, e repassa por `**kwargs` quaisquer offsets/sigmas de regime que você queira
ajustar separadamente.

> **Cuidado com embeddings já L2-normalizados.** `RadialReference.fit` levanta
> `ValueError("Cannot fit a robust radial reference: scaled MAD is zero.")` quando as normas não
> têm dispersão suficiente — exatamente o caso de embeddings já normalizados em L2 (todas as
> normas ≈ 1). O helper de mais alto nível `simulator_config_from_dataset` (ver Seção 12) trata
> esse caso de forma graciosa: detecta normas quase constantes
> (`|mean(‖x‖) − 1| < tol ∧ std(‖x‖) < tol`), **emite um `UserWarning`**, mantém a referência
> radial base e pula o ajuste, registrando `radial_fitted=False` no `fit_summary`. Sempre
> alimente a ponte radial com embeddings **brutos** (pré-normalização).

---

## 10. Reprodutibilidade

A reprodutibilidade é uma propriedade de primeira classe do simulador, não um efeito colateral.
Existem **três seeds independentes**, cada uma controlando um eixo distinto da geração, e um
sistema de **streams nomeados** que torna a saída independente da ordem das chamadas.

### 10.1 Seed estrutural

A seed fornecida ao construtor controla o **mundo sintético** — as estruturas compartilhadas por
todos os splits:

```python
simulator = AdaFaceEmbeddingSimulator(config, random_seed=2026)
```

Ela determina, no `__init__`, via o stream estrutural `_rng("structure")`:

- as médias dos componentes populacionais (`population_component_means`);
- a base anisotrópica global (`anisotropic_basis`);
- as bases anisotrópicas por domínio (`domain_anisotropic_bases`);
- os vetores médios de bias por domínio (`domain_bias_vectors`).

Essas estruturas dependem da seed estrutural, mas **não** do número nem da ordem das chamadas
posteriores de amostragem.

### 10.2 Seeds de população e observação

Cada split pode receber duas seeds explícitas, controlando eixos distintos:

```python
train = simulator.sample_dataset(
    identity_count=2_000,
    samples_per_identity=(5, 12),
    split_name="train",
    identity_offset=0,
    population_random_seed=1001,  # identidades latentes
    random_seed=101,              # observações
)
```

- **`population_random_seed`** governa os centros angulares latentes, os componentes
  populacionais, as bases específicas por identidade e os vetores de bias domínio × identidade
  (tudo o que vive em `SyntheticIdentityPopulation`);
- **`random_seed`** governa as observações: domínio, qualidade, regime radial, normas, sessões,
  a dispersão intra-classe (vMF/cap/hybrid) e o ruído tangencial de nuisance.

### 10.3 Streams nomeados independentes da ordem

Quando as seeds explícitas são omitidas, o simulador **deriva uma seed estável** a partir da seed
estrutural e de um conjunto de partes nomeadas, via `_stable_seed`:

\[
\text{seed} = \operatorname{blake2b}\big(\text{structural\_seed} \,\|\, \text{parte}_1
\,\|\, \cdots \,\|\, \text{parte}_n\big)_{64\text{ bits}}.
\]

As partes incluem o **namespace** (`"population"` ou `"observations"`), o **nome do stream**, o
domínio forçado, o `identity_offset` e a contagem de identidades. Como a seed depende apenas
desses identificadores semânticos — e **não** de um RNG estatal consumido sequencialmente —
inserir uma geração intermediária **não altera** silenciosamente um split posterior.

```python
first_dev = simulator.sample_dataset(
    identity_count=500, samples_per_identity=4,
    split_name="dev", identity_offset=100_000,
)

# Geração intercalada — não deve afetar o split "dev"
_ = simulator.sample_dataset(
    identity_count=10, samples_per_identity=2,
    split_name="diagnostic", identity_offset=999_000,
)

second_dev = simulator.sample_dataset(
    identity_count=500, samples_per_identity=4,
    split_name="dev", identity_offset=100_000,
)

assert (first_dev.raw_embeddings == second_dev.raw_embeddings).all()
```

Tanto `_rng(...)` quanto `_resolved_seed(...)` retornam imediatamente a seed explícita quando ela
é fornecida, ou derivam a seed estável caso contrário — de modo que seeds explícitas e streams
nomeados coexistem sem ambiguidade.

### 10.4 Garantia de paridade bit-a-bit em 0.3 (e a injeção de centros)

A versão 0.3 introduz os modos de dispersão intra-classe `IDPERTURB_CAP` e `HYBRID_VMF_CAP`, além
do caminho de **injeção de centros** em `sample_identity_population`. Ambas as adições foram
projetadas para **não perturbar** o caminho sorteado herdado:

- O modo padrão `IntraClassSamplingMode.VMF` reproduz a saída do v0.2 **bit-a-bit** — um *golden
  test* trava essa invariante. Os modos de calota são estritamente *opt-in*.
- Quando se **injeta** uma população via `centers=` (Seção 11), o método
  `_sample_identity_centers` **não é chamado**, e portanto **o RNG da população não é consumido**
  por ele. As demais consultas ao RNG (bases específicas por identidade, biases domínio ×
  identidade) ocorrem na mesma ordem de antes. O resultado é que o caminho **sorteado** (com
  `identity_count=`) permanece bit-a-bit idêntico ao comportamento anterior à introdução da
  injeção — a nova capacidade não desloca o stream de números aleatórios do caminho legado.

> **Auditabilidade.** Para reproduzir um experimento exatamente, registre as três seeds (a
> estrutural no construtor; população e observação por split), o perfil, a configuração completa
> e a versão/hash do simulador. O manifesto de `save_bundle` (Seção 17) já captura todos esses
> campos. Os campos `structural_seed`, `population_sampling_seed` e `observation_sampling_seed`
> também ficam expostos no próprio `SyntheticEmbeddingDataset`.

---

## 11. Separação entre população e observações

A separação entre **identidades latentes** e **observações** é a peça arquitetural central que
permite os experimentos cross-condition. Uma população é amostrada (ou injetada) uma vez e pode
então ser observada em múltiplos domínios, mantendo os **mesmos centros** `c_i` em todas as
condições. Os dois métodos envolvidos são `sample_identity_population` e `sample_observations`.

### 11.1 `sample_identity_population` — assinatura completa

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
    ...
```

O método tem **duas formas mutuamente exclusivas**, selecionadas por qual dos dois parâmetros
ancorantes é fornecido:

**Forma A — `identity_count` (centros sorteados, padrão).** Amostra os centros da mistura vMF
configurada (`PopulationConfig`), exatamente como no v0.2. `components` e `centers` são
calculados internamente por `_sample_identity_centers`.

```python
population = simulator.sample_identity_population(
    identity_count=800,
    identity_offset=300_000,
    random_seed=4001,
    stream_name="cross_condition_population",
)
```

**Forma B — `centers=` (centros injetados).** Ancora a população em direções fornecidas
externamente — por exemplo, médias esféricas por identidade extraídas de embeddings reais (ver
Modo A, Seção 12). Os centros são L2-normalizados internamente.

```python
population = simulator.sample_identity_population(
    centers=real_per_identity_centers,        # (M, dimension)
    components=cluster_labels,                 # opcional
    identity_labels=real_identity_labels,      # opcional (proveniência)
    source_sample_counts=images_per_identity,  # opcional (proveniência)
    identity_offset=900_000,
    stream_name="anchored_population",
)
```

#### Regra "exatamente um de `identity_count` | `centers`"

O método valida logo de início:

```python
injected = centers is not None
if injected == (identity_count is not None):
    raise ValueError("Provide exactly one of identity_count or centers.")
```

Ou seja: fornecer **ambos** ou **nenhum** levanta `ValueError`. A contagem efetiva de
identidades (`count`) vem de `centers.shape[0]` na forma injetada ou de `identity_count` na forma
sorteada.

#### Validações por parâmetro

| Parâmetro | Tipo | Padrão | Significado e validação |
|---|---|---:|---|
| `identity_count` | `int \| None` | `None` | número de identidades a sortear (forma A). Deve ser positivo. Exatamente um entre este e `centers`. |
| `centers` | `ArrayLike \| None` | `None` | direções injetadas `(M, dimension)` (forma B). L2-normalizadas internamente; `centers.shape[1]` deve igualar `config.dimension`, senão `ValueError`. |
| `components` | `ArrayLike \| None` | `None` | rótulo de componente populacional por identidade injetada. Só usado na forma B. Se omitido, todas recebem componente `0`. Se fornecido: shape `(count,)`, dtype inteiro, valores em `[0, population.component_count)`. |
| `identity_labels` | `Sequence[str] \| None` | `None` | rótulo de identidade de origem (proveniência, Modo A). Comprimento deve ser `count`. Armazenado em `source_identity_labels`. |
| `source_sample_counts` | `ArrayLike \| None` | `None` | número de imagens reais que formaram cada centro (proveniência, Modo A). Shape `(count,)`, valores estritamente positivos. |
| `identity_offset` | `int` | `0` | deslocamento aplicado aos `identity_ids` (`arange(count) + offset`); garante disjunção entre splits. |
| `random_seed` | `int \| None` | `None` | seed explícita da população; se omitida, deriva-se via stream nomeado (Seção 10.3). |
| `stream_name` | `str` | `"population"` | nome do stream para derivação estável de seed. |

Na forma sorteada (A), `components` e `centers` são produzidos por
`_sample_identity_centers(count, rng)`. Na forma injetada (B), esse método **não é chamado** —
preservando a paridade bit-a-bit do caminho sorteado (Seção 10.4). Independentemente da forma,
as bases específicas por identidade e os biases domínio × identidade são amostrados do mesmo RNG
de população, e o objeto retornado passa por `population.validate()` antes de retornar
(verifica unitariedade dos centros, shapes consistentes etc.).

#### O objeto `SyntheticIdentityPopulation`

| Campo | Conteúdo |
|---|---|
| `identity_ids` | ids globais (`arange(count) + identity_offset`) |
| `population_components` | componente populacional por identidade |
| `identity_centers` | centros angulares latentes `(count, dimension)`, unitários |
| `identity_specific_nuisance_bases` | base específica por identidade, quando habilitada (senão `None`) |
| `domain_identity_bias_vectors` | mapping domínio → vetores de bias `(count, dimension)` |
| `sampling_seed` | seed efetiva usada |
| `stream_name` | nome do stream |
| `overlap_lower_bound_floor` | piso de calota por identidade (regra do bissetor), preenchido sob demanda no primeiro uso de um modo de calota; **não serializado** |
| `source_identity_labels` | rótulos de origem (Modo A); `None` em populações sorteadas; não serializado |
| `source_sample_counts` | contagem de imagens reais por centro (Modo A); `None` em populações sorteadas; não serializado |

### 11.2 `sample_observations`

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
    ...
```

`sample_observations` recebe uma população já materializada e produz um
`SyntheticEmbeddingDataset` observando-a em um domínio (forçado ou misto).

| Parâmetro | Tipo | Padrão | Significado |
|---|---|---:|---|
| `population` | `SyntheticIdentityPopulation` | obrigatório | população latente a observar; revalidada na entrada |
| `samples_per_identity` | `int \| tuple[int, int]` | obrigatório | número fixo, ou faixa `(min, max)` sorteada por identidade |
| `split_name` | `str` | `"synthetic"` | nome do split (gravado em metadata); deve ser não-vazio |
| `forced_domain` | `str \| None` | `None` | força todas as observações num domínio; `None` sorteia por `weight` |
| `sessions_per_identity` | `int \| tuple[int, int]` | `1` | número de sessões sintéticas por identidade |
| `random_seed` | `int \| None` | `None` | seed explícita das observações; se omitida, deriva-se via stream nomeado |
| `stream_name` | `str \| None` | `None` | nome do stream; se omitido, usa `split_name` |

A seed efetiva é resolvida via `_resolved_seed("observations", stream, forced_domain or "mixed",
explicit_seed=random_seed)`. O pipeline interno percorre, em ordem: expansão de
`samples_per_identity` → sessões → domínios → qualidade latente → regimes radiais → log-normas →
`intra_kappa` → dispersão intra-classe (vMF, cap ou hybrid, conforme `intra_class_sampling_mode`)
→ nuisance tangencial → embedding bruto. O dataset retornado carrega os centros da população
(compartilhados) e expõe `population_sampling_seed` e `observation_sampling_seed` para
rastreabilidade.

### 11.3 Observar a mesma população em dois domínios

```python
population = simulator.sample_identity_population(
    identity_count=800, identity_offset=300_000, random_seed=4001,
)

controlled = simulator.sample_observations(
    population,
    samples_per_identity=(3, 6),
    split_name="test:controlled",
    forced_domain="controlled",
    sessions_per_identity=(1, 3),
    random_seed=4002,
)

surveillance = simulator.sample_observations(
    population,
    samples_per_identity=(3, 6),
    split_name="test:surveillance",
    forced_domain="surveillance",
    sessions_per_identity=(1, 3),
    random_seed=4003,
)
```

As duas bases compartilham os centros latentes:

\[
\mathbf{c}_i^{\text{controlled}} = \mathbf{c}_i^{\text{surveillance}},
\]

mas suas observações diferem:

\[
\mathbf{x}_{ij}^{\text{controlled}} \neq \mathbf{x}_{ik}^{\text{surveillance}}.
\]

Esse desenho é a base dos experimentos que comparam merge ingênuo, `PER_CONDITION` e
`BALANCED_IDENTITY` (Seção 19). Para o caso comum de gerar todos os domínios alinhados numa única
chamada, prefira o helper `sample_cross_condition_datasets` (Seção 13), que internamente amostra a
população uma vez e itera `sample_observations` por domínio.

---

## 12. População ancorada em dataset real + vazamento

A população de identidades amostrada nos capítulos anteriores nasce de uma mistura genérica de
vMF (seção 6.2): os centros angulares `μ_i` são *sintéticos* e não respeitam a geometria empírica
fina de embeddings faciais reais. Para experimentos que exigem maior fidelidade — calibração
contra dados reais, validação metodológica de PLDA, *augmentation* — o módulo
`simulator/faceaudit_seed.py` ancora a população em um `EmbeddingDataset` real do FaceAudit, em
**dois modos complementares**:

- **Modo A (ancorado)** — `population_from_dataset`: cada centro de identidade É a média esférica
  por identidade dos embeddings reais. As identidades são reais; apenas a dispersão intra-classe
  (e a norma) é sintética.
- **Modo B (distribuição-pareada)** — `population_config_from_dataset` /
  `simulator_config_from_dataset`: ajusta-se uma mistura de von Mises–Fisher (MoVMF) aos centros
  reais por identidade (via `MoVMFTypicalityModel` do próprio FaceAudit) e usam-se `μ_k`/`κ_k`/`π_k`
  como `PopulationConfig`. O simulador então amostra centros-fantasma NOVOS dessa aproximação
  angular da distribuição empírica.

> **Caveat de import.** `faceaudit` é importado **preguiçosamente** apenas dentro das funções que
> precisam dele (Modo B). Portanto `import faceaudit_seed` funciona sem o FaceAudit instalado, e
> toda a cadeia de extração de centros, Modo A e helpers anti-vazamento roda *faceaudit-free*.

### 12.1 Extração de centros por identidade — `identity_centers_from_dataset`

O bloco fundamental, compartilhado pelos dois modos, computa **um centro de média esférica por
identidade**:

\[
\hat{c}_i \;=\; \operatorname{normalize}\!\left( \sum_{j \in i} w_j \, u_{ij} \right),
\qquad u_{ij} \;=\; \frac{x_{ij}}{\lVert x_{ij} \rVert}.
\]

```python
import faceaudit_seed as fs

extraction = fs.identity_centers_from_dataset(
    dataset,
    min_samples=2,           # 1 = âncora ruidosa, 2+ mais estável, 5+ ideal
    weight_by_quality=False,
)
print(extraction.centers.shape)   # (M, D) — direções unitárias
print(extraction.labels)          # rótulo FaceAudit original por centro
print(extraction.counts)          # nº de imagens reais que formaram cada centro
```

Detalhes forensicamente relevantes do agrupamento e da ponderação:

- **Agrupa pelo valor ORIGINAL de `identity`**, não por `str(identity)`. Isso evita fundir uma
  identidade `1` (int) com `"1"` (str) — elas permanecem dois grupos distintos, ainda que ambos os
  rótulos finais sejam serializados como `"1"`. A ordem de primeira ocorrência é preservada.
- **Identidades `None` são ignoradas**; identidades com menos de `min_samples` amostras são
  descartadas. Se *nenhuma* identidade atinge o piso, levanta-se `ValueError`.
- **Ponderação por qualidade (`weight_by_quality=True`)** pondera cada imagem pelo seu
  `quality_score` (exige `dataset.items`), com tratamento explícito de borda:
  - `None` ou `NaN` → peso `1.0`;
  - valor **negativo** → `ValueError` (pesos não podem ser negativos);
  - todos os pesos do grupo nulos (`0`) → cai para a **média não-ponderada** do grupo.
- **Fallback de média degenerada.** Se a soma ponderada das direções tem norma ≈ 0 (ex.: imagens
  antípodas que se cancelam), a média esférica não está definida; emite-se um `UserWarning` e o
  centro cai para a **imagem de melhor qualidade** (ou a primeira, se não houver pesos positivos).
  O centro resultante permanece unitário.

#### `IdentityCenterExtractionResult`

Dataclass de retorno, com a procedência completa da extração:

| Campo | Tipo | Significado |
|---|---|---|
| `centers` | `FloatArray` `(M, D)` | direções unitárias — uma média esférica por identidade |
| `labels` | `list[str]` | rótulo de identidade FaceAudit original de cada centro |
| `counts` | `IntArray` `(M,)` | número de imagens reais que formaram cada centro |
| `quality_weighted` | `bool` | se a média foi ponderada por `quality_score` |

### 12.2 Modo A — `population_from_dataset`

O Modo A injeta os centros extraídos diretamente como centros da população, via o caminho de
injeção `sample_identity_population(centers=..., identity_labels=..., source_sample_counts=...)`
(seção 11). As identidades resultantes são **reais**; o simulador adiciona apenas a variação
intra-classe sintética e a norma.

```python
population = fs.population_from_dataset(
    simulator,
    dataset,
    min_samples=2,
    weight_by_quality=False,
    identity_offset=0,
    random_seed=None,
    stream_name="dataset_seeded",
)
# population.source_identity_labels e population.source_sample_counts ficam preenchidos
```

As observações geradas a partir dessa população carregam duas colunas de procedência adicionais —
`source_identity` e `source_sample_count` (seção 14) — que ligam cada amostra sintética de volta à
identidade real e ao número de imagens que originaram o centro.

> ### ⚠️ CAVEAT FORENSE PROEMINENTE — Modo A **NÃO é anonimização**
>
> O Modo A **gera variação sintética em torno de âncoras biométricas REAIS**. As identidades não
> são fabricadas: os centros são médias esféricas de embeddings de pessoas reais, e cada amostra
> sintética rastreia, via `source_identity`, exatamente de qual identidade real ela deriva.
> Tratar a saída do Modo A como dado anonimizado é um **erro de procedência**.
>
> **Use o Modo A apenas para** *augmentation*, validação metodológica ou experimentos internos
> sobre uma base autorizada — **nunca** como substituto de anonimização. E, sob qualquer uso, o
> **conjunto de âncoras DEVE permanecer disjunto de qualquer conjunto de avaliação** (seção 12.5).

### 12.3 Modo B — `population_config_from_dataset`

O Modo B ajusta uma MoVMF aos centros reais por identidade e devolve um `PopulationConfig` pronto
para amostrar centros-fantasma novos, mais um dicionário `fit_summary` com a procedência completa
do ajuste.

```python
population_config, fit_summary = fs.population_config_from_dataset(
    dataset,
    n_components=None,                  # None → seleção automática por critério
    component_candidates=(1, 2, 4, 8, 16),
    max_k=None,                         # teto opcional do K
    criterion="bic",                    # "bic" ou "aic"
    min_samples=2,
    weight_by_quality=False,
    random_state=42,
    n_init=5,
)
```

Pontos essenciais:

- **Ajusta sobre os CENTROS por identidade**, não sobre todas as observações. Ajustar sobre as
  observações brutas misturaria variância intra e inter-identidade — o alvo aqui é a distribuição
  *dos centros*.
- **MoVMF via `MoVMFTypicalityModel` com `centering="none"`.** O `MoVMFTypicalityModel` do FaceAudit é
  reutilizado como estimador MoVMF; a centralização é desligada deliberadamente, pois os centros
  já estão na esfera unitária e qualquer recentragem distorceria a geometria angular.
- **Seleção de K por BIC/AIC.** Com `n_components` fixo, ajusta-se aquele K. Caso contrário,
  avaliam-se os `component_candidates` *clampados* ao intervalo `[1, min(M, max_k)]` (onde `M` é o
  número de centros e `max_k` o teto opcional), escolhendo-se o K que minimiza o critério
  (`bic_`/`aic_` do modelo). Ajustes que falham numericamente para um dado K são registrados como
  `None` em `criterion_values` e pulados; se **nenhum** K converge, levanta-se `RuntimeError`
  reportando o critério por K.
- **Resultado.** Os `μ_k`/`κ_k`/`π_k` ajustados viram `component_means`/`identity_center_kappas`/
  `component_weights` de um `PopulationConfig` com `component_count = K`.

#### Dicionário `fit_summary`

| Chave | Significado |
|---|---|
| `population_source` | sempre `"faceaudit_movmf"` |
| `selected_k` | K efetivamente ajustado |
| `n_components_fixed` | K fixo solicitado, ou `None` se houve seleção automática |
| `candidate_k` | lista de candidatos avaliados (`None` se K fixo) |
| `criterion` | `"bic"` ou `"aic"` |
| `criterion_values` | mapa `{K: valor}` por candidato (`None` para K que falharam) |
| `fit_seed` | `random_state` usado no ajuste |
| `n_identity_centers` | número de centros reais usados no ajuste |
| `dimension` | dimensão do embedding |
| `n_degenerate` | número de componentes degeneradas detectadas |
| `min_component_weight` | menor peso de componente `π_k` |
| `max_component_kappa` | maior concentração `κ_k` |
| `component_weight_entropy` | entropia dos pesos das componentes |
| `weight_by_quality` | se a extração de centros foi ponderada por qualidade |
| `min_samples` | piso de amostras por identidade na extração |

### 12.4 Modo B completo — `simulator_config_from_dataset`

Envolve `population_config_from_dataset` em uma configuração de simulador completa: parte de um
`base_config` (por padrão `make_cooperative_plda_config` na dimensão do dataset), substitui a
população pela MoVMF ajustada, marca `profile_name = "FACEAUDIT_MOVMF"` e — quando `fit_norm=True`
— substitui também a referência radial por uma ajustada às normas reais.

```python
config, fit_summary = fs.simulator_config_from_dataset(
    dataset,
    base_config=None,                  # default: make_cooperative_plda_config(dimension)
    fit_norm=True,
    embedding_view_for_norm="raw",     # único suportado: dataset.matrix
    warn_if_unit_norms=True,
    norm_tol=1e-3,
    # **movmf_kwargs repassados a population_config_from_dataset
    component_candidates=(1, 2, 4, 8, 16),
    criterion="bic",
)
config.validate()
assert config.profile_name == "FACEAUDIT_MOVMF"
```

**Ajuste radial com aviso de L2-normalizado.** O ajuste radial via `RadialReference.fit` (seção 6.7)
precisa que as **log-normas variem** (MAD escalado > 0). Se o dataset parece L2-normalizado —
`|mean(‖x‖) − 1| < norm_tol` **e** `std(‖x‖) < norm_tol`, ou, mais geralmente, MAD escalado das
log-normas ≈ 0 — o ajuste radial seria não-informativo. Nesse caso o módulo emite um `UserWarning`
e **mantém a referência radial do `base_config`** em vez de falhar. O bloco `norm_summary` no
`fit_summary` registra o diagnóstico:

| Chave de `norm_summary` | Significado |
|---|---|
| `norm_mean` | média das normas reais |
| `norm_std` | desvio-padrão das normas |
| `norm_min` / `norm_max` | extremos das normas |
| `unit_norm_warning` | `True` se o dataset parece L2-normalizado |
| `radial_fitted` | `True` se a referência radial foi efetivamente reajustada |

Quando `fit_norm=False`, `norm_summary` é `None`.

> ### ⚠️ CAVEAT FORENSE — Modo B é aproximação angular de 1ª ordem
>
> Os centros-fantasma do Modo B são uma **aproximação angular de primeira ordem da distribuição
> empírica dos centros** (uma mistura isotrópica por componente), **NÃO o manifold real dos
> embeddings**. A MoVMF captura apenas a densidade angular agregada dos centros por identidade;
> ela ignora estrutura anisotrópica local, correlações de alta ordem e a geometria fina do
> manifold facial. Em laudo, a saída do Modo B deve ser descrita como população *sintética inspirada
> em* dados reais, jamais como reprodução fiel da população real.

### 12.5 Helpers anti-vazamento — `source_identity_set` / `assert_disjoint_source_identities`

Ambos os modos rastreiam de qual base real cada população foi ancorada. Esses dois helpers
(*faceaudit-free*) fecham o risco de vazamento âncora↔avaliação:

```python
ids = fs.source_identity_set(population)          # set[str] dos rótulos de origem; vazio se amostrada

fs.assert_disjoint_source_identities(pop_train, pop_eval)  # ValueError se compartilham origem
```

- `source_identity_set(population)` devolve o conjunto de rótulos de identidade originais sobre os
  quais a população foi ancorada (Modo A), ou um conjunto **vazio** se a população foi puramente
  amostrada (sem ancoragem).
- `assert_disjoint_source_identities(a, b)` levanta `ValueError` — reportando quantas identidades
  são compartilhadas e uma amostra dos rótulos — se as duas populações ancoradas têm **qualquer**
  identidade de origem em comum.

> **Nota forense de uso obrigatório.** A base usada para ancorar (Modo A) ou ajustar (Modo B)
> **DEVE ser disjunta em identidades de qualquer conjunto de avaliação**. Ancorar treino e avaliação
> na mesma identidade real é vazamento direto e invalida toda métrica de calibração/discriminação
> a jusante. Em qualquer pipeline forense que use `faceaudit_seed`, a chamada a
> `assert_disjoint_source_identities` entre a população de desenvolvimento e a de avaliação é
> **etapa obrigatória**, não opcional.

---

## 13. Helper cross-condition

Para *stress testing* de estratégias cross-condition (ex.: PLDA frontal × lateral), o simulador
oferece `sample_cross_condition_datasets`: ele amostra **uma única população latente** e a observa
em **múltiplos domínios de aquisição alinhados por identidade**. Como todos os domínios partilham as
mesmas identidades, cada identidade rende pares genuínos cross-condition — exatamente o material que
um estimador PLDA two-covariance precisa.

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

Comportamento:

- A população é amostrada **uma vez** (via `sample_identity_population`, stream
  `f"{split_name}:population"`), com `identity_offset` e `population_random_seed` aplicados a ela.
- Para cada domínio em `samples_per_identity_by_domain`, observa-se a *mesma* população com
  `forced_domain=domain_name`, número de amostras por identidade próprio do domínio, e uma seed de
  observação opcional via `observation_random_seeds[domain_name]`.
- O retorno é um `dict[str, SyntheticEmbeddingDataset]` cujas chaves são os nomes de domínio e cujos
  valores compartilham os mesmos `identity_id` — garantindo pares genuínos cross-condition.

Esse desenho foi criado para comparar estratégias como:

```text
PER_CONDITION
BALANCED_IDENTITY
```

**Ponte para o FaceAudit.** Para converter o dicionário cross-condition diretamente em
`EmbeddingDataset`s reais — preservando `identity` idêntico entre as condições (e portanto o
caráter pareado) — use o export dedicado do adapter concreto,
`cross_condition_to_embedding_datasets` (seção 18.2). Ele mapeia cada `SyntheticEmbeddingDataset`
do dicionário para um `EmbeddingDataset` nomeado `synthetic_{domain}`, com embeddings em `float32`
e os covariáveis forensicamente acionáveis preservados.

---

## 14. `SyntheticEmbeddingDataset` e dicionário de colunas de metadata

O objeto `SyntheticEmbeddingDataset` é o produto de `AdaFaceEmbeddingSimulator.sample_observations`
(e dos atalhos `sample_dataset` e `sample_cross_condition_datasets`). Ele carrega
simultaneamente os embeddings brutos, as direções normalizadas, a estrutura latente do mundo
sintético e o `DataFrame` de metadados por observação. Toda a rastreabilidade forense do
experimento — perfil, seeds e configuração — viaja acoplada a esse objeto e é persistida pelos
métodos da Seção 17.

### 14.1 Campos do `SyntheticEmbeddingDataset`

| Campo | Shape ou tipo | Conteúdo |
|---|---|---|
| `raw_embeddings` | `(N, d)` | embeddings brutos \(\mathbf{x}_{ij}=r_{ij}\widetilde{\mathbf{u}}_{ij}\) |
| `normalized_embeddings` | `(N, d)` | direções unitárias |
| `identity_centers` | `(I, d)` | centros latentes das identidades |
| `metadata` | `pd.DataFrame` | metadados por observação (Seção 14.2) |
| `population_component_means` | `(K, d)` | médias dos componentes populacionais |
| `anisotropic_basis` | `(d, r)` ou `None` | base anisotrópica global |
| `domain_anisotropic_bases` | mapping | bases anisotrópicas por domínio |
| `domain_bias_vectors` | mapping | bias médio por domínio |
| `config` | `SimulatorConfig` | configuração completa |
| `structural_seed` | `int` | seed estrutural do mundo sintético |
| `population_sampling_seed` | `int` | seed efetiva da população |
| `observation_sampling_seed` | `int` | seed efetiva das observações |
| `stream_name` | `str` | nome do stream efetivo |

O método `validate()` checa as invariantes estruturais: `normalized_embeddings` contém vetores
unitários, o número de linhas dos metadados casa com `raw_embeddings`, e as normas de
`raw_embeddings` coincidem com a coluna `embedding_norm` dos metadados. A validação roda
automaticamente ao final de `sample_observations`.

### 14.2 Metadados por observação

A tabela abaixo é a referência completa e atualizada das colunas do `DataFrame`
`dataset.metadata`. As colunas estão na ordem em que são construídas em `sample_observations`.

| Coluna | Significado |
|---|---|
| `split` | nome do split |
| `identity_id` | identidade sintética (com `identity_offset` aplicado) |
| `local_identity_index` | índice local na população |
| `sample_index` | índice da observação dentro da identidade |
| `session_index` | sessão sintética |
| `population_component` | componente populacional latente |
| `acquisition_domain` | domínio de aquisição |
| `radial_regime` | `central`, `low_tail` ou `high_tail` |
| `latent_quality` | qualidade sintética em \([0,1]\) |
| `embedding_norm` | norma do embedding bruto \(r_{ij}\) |
| `log_embedding_norm` | log-norma |
| `reference_scaled_log_norm_z` | posição radial padronizada |
| `robust_norm_z` | alias legado de `reference_scaled_log_norm_z` |
| `radial_reference_method` | método da referência radial |
| `intra_kappa` | concentração vMF intra-identidade \(\kappa_{ij}\) |
| `own_center_cosine` | cosseno ao centro latente **após** o passo de nuisance |
| `pre_nuisance_center_cosine` | cosseno ao centro latente **antes** do passo de nuisance |
| `intra_class_sampling_mode` | modo de dispersão intra-classe usado (`vmf`, `idperturb_cap`, `hybrid_vmf_cap`) |
| `cap_lower_bound_effective` | piso de cosseno `lb` efetivo por amostra (`NaN` em `VMF`) |
| `hybrid_resample_count` | número de reamostragens vMF por rejeição (`0` fora de `HYBRID_VMF_CAP`) |
| `hybrid_used_cap_fallback` | indicador booleano de fallback de calota (`False` fora de `HYBRID_VMF_CAP`) |
| `tangent_update_mode` | `retraction` ou `exponential_map` |
| `anisotropic_noise_norm` | magnitude do nuisance anisotrópico |
| `fixed_domain_bias_norm` | magnitude do bias médio de domínio |
| `domain_identity_bias_norm` | magnitude da interação domínio × identidade |
| `session_bias_norm` | magnitude do bias de sessão |
| `combined_tangent_step_norm` | tamanho do passo tangencial total |
| `source_identity` | rótulo de identidade real de origem (**somente** em populações ancoradas em dataset) |
| `source_sample_count` | nº de imagens reais que formaram o centro (**somente** em populações ancoradas) |

#### Colunas novas de dispersão intra-classe (v0.3)

As cinco colunas `intra_class_sampling_mode`, `cap_lower_bound_effective`,
`pre_nuisance_center_cosine`, `hybrid_resample_count` e `hybrid_used_cap_fallback` foram
introduzidas na versão 0.3 e estão **sempre presentes**, inclusive no modo `VMF` padrão. No
modo `VMF`, `cap_lower_bound_effective` é `NaN`, `hybrid_resample_count` é `0` e
`hybrid_used_cap_fallback` é `False` — o que torna os três campos auditáveis como
"calota não utilizada" sem ambiguidade. Já `pre_nuisance_center_cosine` é populada nos três
modos: ela registra o cosseno ao centro **imediatamente após** a amostragem vMF/calota/híbrido
e **antes** da aplicação do passo tangencial de nuisance/viés.

#### Invariante crítico para auditar a calota

\[
\texttt{pre\_nuisance\_center\_cosine} \;\ge\; \texttt{cap\_lower\_bound\_effective}.
\]

Esse é o invariante que prova que o piso geométrico `lb` foi respeitado na geração. O piso é
garantido **na direção pré-nuisance**. O passo tangencial de nuisance/viés aplicado em seguida
pode empurrar o `own_center_cosine` (pós-nuisance) **abaixo** de `lb` — isso é esperado e não
viola nada. Por isso, **para auditar a calota use sempre `pre_nuisance_center_cosine`**, nunca
`own_center_cosine`. Em laudo, a frase defensável é: *"nenhuma observação sintética desta
identidade foi gerada com cosseno menor que `cap_lower_bound_effective` ao centro latente,
antes dos ruídos de domínio/sessão."*

#### Colunas de proveniência (somente populações ancoradas)

As colunas `source_identity` e `source_sample_count` aparecem **apenas** quando o dataset foi
observado a partir de uma população ancorada em dataset real (Modo A, Seção 12), ou seja,
quando `SyntheticIdentityPopulation.source_identity_labels` / `source_sample_counts` estão
preenchidos. Em populações puramente amostradas da mistura vMF essas colunas estão **ausentes**
— o que mantém os metadados de populações sintéticas inalterados em relação à v0.2 e sinaliza,
pela simples presença/ausência da coluna, se há âncoras biométricas reais por trás do dataset.

`source_identity` carrega o rótulo de identidade real do FaceAudit que originou cada centro, e
`source_sample_count` registra quantas imagens reais formaram a média esférica daquele centro.
Como populações ancoradas geram variação sintética em torno de âncoras biométricas **reais** —
e não anonimizam nada —, essas duas colunas são o gancho de cadeia de custódia para verificar
que o conjunto de âncoras permanece disjunto de qualquer conjunto de avaliação (ver
`assert_disjoint_source_identities`, Seção 12).

### 14.3 Campos de proveniência e overlap do `SyntheticIdentityPopulation`

O objeto latente `SyntheticIdentityPopulation` (Seção 11) ganhou, na v0.3, campos auxiliares
opcionais que sustentam tanto a guarda do bissetor quanto a proveniência. Eles **não** são
serializados (ficam fora de `asdict`/`save_manifest`/`save_npz`), por serem vetores grandes ou
caches derivados.

| Campo | Tipo | Padrão | Significado |
|---|---|---:|---|
| `identity_ids` | `IntArray` | obrigatório | ids das identidades (com `identity_offset`) |
| `population_components` | `IntArray` | obrigatório | componente populacional por identidade |
| `identity_centers` | `FloatArray` | obrigatório | centro angular latente \(\mathbf{c}_i\) |
| `identity_specific_nuisance_bases` | `FloatArray` ou `None` | obrigatório | base específica por identidade (quando habilitada) |
| `domain_identity_bias_vectors` | `Mapping[str, FloatArray]` | obrigatório | bias domínio × identidade |
| `sampling_seed` | `int` | obrigatório | seed efetiva da população |
| `stream_name` | `str` | obrigatório | nome do stream |
| `overlap_lower_bound_floor` | `FloatArray` ou `None` | `None` | piso de cosseno do bissetor por identidade (cache preguiçoso) |
| `source_identity_labels` | `list[str]` ou `None` | `None` | rótulo de identidade real por centro (Modo A) |
| `source_sample_counts` | `IntArray` ou `None` | `None` | nº de imagens reais por centro (Modo A) |

O campo `overlap_lower_bound_floor` é calculado preguiçosamente na primeira observação em modo
de calota (via `_overlap_lower_bound_floor`, em blocos, sem materializar a matriz \(C\times C\))
e **cacheado** para reuso entre observações cross-condition da mesma população — garantindo que
o piso anti-colisão seja idêntico em todos os domínios alinhados. Os campos
`source_identity_labels` e `source_sample_counts` são `None` em populações amostradas e
preenchidos apenas no caminho de injeção de centros (Modo A); é deles que `sample_observations`
deriva as colunas `source_identity` e `source_sample_count` da Seção 14.2.

---

## 15. Diagnóstico radial e por cap

### 15.1 `norm_bin_diagnostics`

O método `norm_bin_diagnostics` estratifica as observações por quantis de norma e resume o
padrão esperado de U-invertido entre norma, qualidade e estabilidade angular.

```python
diagnostics = test.norm_bin_diagnostics(bin_count=10)
print(diagnostics)
```

Colunas retornadas:

| Coluna | Significado |
|---|---|
| `requested_bin_count` | quantidade de bins solicitada |
| `actual_bin_count` | quantidade efetivamente produzida |
| `norm_quantile_bin` | faixa de norma |
| `sample_count` | tamanho do grupo |
| `norm_mean` | norma média |
| `reference_scaled_log_norm_z_mean` | posição radial média |
| `latent_quality_mean` | qualidade média |
| `intra_kappa_mean` | estabilidade angular média |
| `own_center_cosine_mean` | proximidade média ao centro |
| `own_center_cosine_median` | proximidade mediana ao centro |
| `tangent_step_norm_mean` | passo tangencial médio |

O método usa `pd.qcut(..., duplicates="drop")`. Se existirem bordas de quantil repetidas, o
pandas produzirá menos bins do que solicitado; nesse caso o método **emite um warning** e
informa tanto `requested_bin_count` quanto `actual_bin_count`, evitando que um diagnóstico
silenciosamente degradado seja interpretado como completo. Esse diagnóstico também é gravado
automaticamente por `save_bundle` no CSV `<stem>_norm_diagnostics.csv` (Seção 17).

### 15.2 Diagnóstico por cap: sempre reporte os percentis de `cap_lower_bound_effective`

Nos modos de calota (`IDPERTURB_CAP` e `HYBRID_VMF_CAP`), o piso `lb` **nominal** configurado
em `DirectionalConfig.cap_base_lower_bound` **não** é, em geral, o piso realmente aplicado. Com
`cap_overlap_guard=True` (padrão), a regra do bissetor impõe um piso por identidade a partir da
densidade local dos centros, e **essa guarda tem prioridade**: o `lb` efetivo pode exceder
`cap_maximum_lower_bound` em regiões densas da população. Em 512-D, mesmo centros aleatórios têm
vizinhos relativamente próximos, então um `lower_bound` nominal de, por exemplo, `0.4` pode
corresponder a um piso efetivo bem maior para identidades típicas.

Por isso, a regra de ouro do diagnóstico por cap é: **sempre reporte os percentis empíricos de
`cap_lower_bound_effective` (mean / p10 / p50 / p90) lado a lado com o `lb` nominal.** O `lb`
nominal documenta a *intenção*; os percentis do efetivo documentam a *realização*. A separação
genuína entre identidades reflete o efetivo, não o nominal.

```python
import numpy as np

lb_eff = test.metadata["cap_lower_bound_effective"].to_numpy()
print({
    "lb_nominal": test.config.directional.cap_base_lower_bound,
    "lb_eff_mean": float(np.nanmean(lb_eff)),
    "lb_eff_p10": float(np.nanpercentile(lb_eff, 10)),
    "lb_eff_p50": float(np.nanpercentile(lb_eff, 50)),
    "lb_eff_p90": float(np.nanpercentile(lb_eff, 90)),
})
```

No modo `HYBRID_VMF_CAP`, complemente o relatório com a **taxa de fallback de calota**: a
fração de `hybrid_used_cap_fallback == True` e a distribuição de `hybrid_resample_count`. Com
`lb` alto e `κ` baixo, grande parte das amostras pode acabar vindo da calota por fallback — e
nesse caso o modo deixa de ser, na prática, "vMF truncado". Auditar essa taxa evita descrever o
dataset com uma geometria que ele não tem.

---

## 16. Geração de pares

### 16.1 Uso básico

A função `sample_pair_table` amostra pares genuínos e impostores aleatórios, calcula o cosseno
de cada par e devolve um `DataFrame` de trials anotado.

```python
from adaface_embedding_simulator import sample_pair_table

pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    random_seed=202,
)
```

Os pares genuínos são sempre balanceados por identidade. A amostragem de impostores depende da
estratégia escolhida.

### 16.2 Estratégia `BALANCED_IDENTITY`

```python
from adaface_embedding_simulator import PairSamplingStrategy

pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    strategy=PairSamplingStrategy.BALANCED_IDENTITY,
)
```

É a estratégia padrão. Interpretação:

- genuínos: a identidade é escolhida aproximadamente de forma uniforme, e duas observações
  distintas dela são sorteadas;
- impostores: duas identidades distintas são escolhidas aproximadamente de forma uniforme, e
  uma observação de cada uma é sorteada.

Esse modo evita que identidades com muitas amostras dominem os trials.

### 16.3 Estratégia `BALANCED_OBSERVATION`

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
- impostores são amostrados aproximadamente em proporção ao número de observações (linhas) — o
  par é formado por duas linhas aleatórias de identidades distintas.

### 16.4 Trials únicos (`unique_pairs`)

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

Se duplicatas forem removidas, o método emite um warning e a tabela retornada pode conter
menos trials do que solicitado. As escolhas de estratégia e de unicidade ficam registradas em
`pairs.attrs["sampling_strategy"]` e `pairs.attrs["unique_pairs"]` para rastreabilidade.

### 16.5 Colunas dos trials

| Coluna | Significado |
|---|---|
| `left_row`, `right_row` | índices das observações no dataset |
| `is_genuine` | `1` genuíno (Hp), `0` impostor (Hd) |
| `cosine_score` | cosseno entre as direções normalizadas |
| `left_identity_id`, `right_identity_id` | identidades dos dois lados |
| `left_norm`, `right_norm` | normas dos embeddings brutos |
| `left_reference_scaled_log_norm_z`, `right_reference_scaled_log_norm_z` | posições radiais padronizadas |
| `left_robust_norm_z`, `right_robust_norm_z` | aliases legados das posições radiais |
| `left_quality`, `right_quality` | qualidades sintéticas |

O cosseno é computado por `cosine_scores`, que re-normaliza as linhas selecionadas de
`dataset.normalized_embeddings` antes do produto interno. As colunas radiais e de qualidade dos
dois lados permitem estratificar diretamente o desempenho dos trials por norma e por qualidade.

---

## 17. Persistência e manifesto

Todos os métodos de persistência do `SyntheticEmbeddingDataset` foram desenhados para serem
**round-trippáveis**: o manifesto carrega a configuração completa, e
`config_from_manifest` (Seção 9) reconstrói o `SimulatorConfig` exato a partir dele, de modo que
um experimento pode ser re-gerado bit-a-bit a partir dos artefatos persistidos somados às seeds
registradas.

### 17.1 `save_npz()`

```python
dataset.save_npz("test_embeddings.npz")
```

Salva, em arquivo NumPy comprimido:

- embeddings brutos e normalizados;
- centros de identidade;
- médias dos componentes populacionais;
- base anisotrópica global (matriz `(d, 0)` quando ausente);
- nomes dos domínios;
- matriz de vetores médios de bias por domínio;
- os metadados estruturados (`metadata.to_records`).

### 17.2 `save_metadata_csv()`

```python
dataset.save_metadata_csv("test_metadata.csv")
```

Grava o `DataFrame` de metadados (Seção 14.2) como CSV, incluindo as colunas de cap e de
proveniência quando presentes.

### 17.3 `save_manifest()`

```python
dataset.save_manifest("test_manifest.json")
```

O manifesto JSON contém:

- `simulator_version`;
- `profile_name`;
- `structural_seed`;
- `population_sampling_seed`;
- `observation_sampling_seed`;
- `stream_name`;
- `row_count`, `dimension`, `identity_count`;
- `source_sha256` (SHA-256 do código-fonte do simulador);
- o bloco `config` completo (via `config_to_dict`);
- um bloco `extra` opcional para metadados arbitrários do experimento.

O bloco `config` é exatamente o que `config_from_manifest` lê para fechar o round-trip:

```python
from adaface_embedding_simulator import config_from_manifest

restored_config = config_from_manifest("test_manifest.json")
# restored_config reproduz o SimulatorConfig original (validado).
```

### 17.4 `save_bundle()`

```python
dataset.save_bundle("synthetic_output", "test")
```

É o método recomendado para experimentos auditáveis. Produz, de uma vez, os quatro artefatos:

```text
synthetic_output/
├── test_embeddings.npz
├── test_metadata.csv
├── test_norm_diagnostics.csv
└── test_manifest.json
```

O CSV de diagnóstico é gerado por `norm_bin_diagnostics(bin_count=10)` (Seção 15). O dicionário
retornado por `save_bundle` mapeia cada chave (`npz`, `metadata`, `manifest`, `norm_diagnostics`)
para o `Path` correspondente, facilitando o registro dos caminhos no log do experimento.

---

## 18. Adapters FaceAudit

O simulador é um pacote independente e oferece dois caminhos de integração com o FaceAudit. Eles
têm propósitos distintos: um é um **scaffold genérico de referência** desacoplado dos DTOs reais;
o outro é o **adapter concreto de produção**, que importa os DTOs reais do FaceAudit. **Para
qualquer uso real, utilize o adapter concreto da Seção 18.2.**

### 18.1 Scaffold genérico de referência — `faceaudit_adapter.py`

> **Caveat (obrigatório ler antes de usar):** o `faceaudit_adapter.py` é um scaffold de
> **referência apenas**. Ele assume uma assinatura `EmbeddingInfo(embedding=, identity=,
> metadata=)` que **NÃO corresponde ao DTO real do FaceAudit** — o `EmbeddingInfo` de produção
> tem campos como `embedding`, `face_hash`, `image_path`, `identity`, `file_name`, `source`,
> `origin`, `quality_scores`, e **não** aceita um kwarg `metadata`. Não use este módulo em
> produção; ele existe para documentar o padrão de injeção de factories e para casos em que o
> usuário injeta os próprios construtores compatíveis com a assinatura assumida.

O scaffold expõe duas funções, ambas com as factories de DTO **injetadas** pelo chamador.

`dataset_to_embedding_records` converte o dataset sintético em registros simples (sem qualquer
DTO):

```python
from faceaudit_adapter import dataset_to_embedding_records

records = dataset_to_embedding_records(
    test,
    embedding_view="normalized",   # "raw" ou "normalized"
    domain_filter="surveillance",  # opcional
)
# cada item: {"embedding": ..., "identity": ..., "metadata": {...}}
```

`to_faceaudit_embedding_dataset` monta DTOs a partir de factories injetadas. Como o construtor
real difere da assinatura assumida, os parâmetros de factory são explícitos:

```python
from faceaudit_adapter import to_faceaudit_embedding_dataset

faceaudit_dataset = to_faceaudit_embedding_dataset(
    test,
    embedding_info_factory=EmbeddingInfo,        # injetado pelo usuário
    embedding_dataset_factory=EmbeddingDataset,  # injetado pelo usuário
    embedding_view="normalized",
    dataset_factory_keyword="embeddings",        # se o construtor espera um kwarg
    include_metadata=True,                       # exige metadata= no EmbeddingInfo
)
```

Se a factory injetada não aceitar metadados, passe `include_metadata=False`. Reforçando: por
não casar com o DTO real, este caminho serve só como referência — a ponte real é o adapter
concreto.

### 18.2 Adapter concreto de produção — `faceaudit_concrete_adapter.py`

Este módulo importa os DTOs reais (`from faceaudit.models.embedding_dataset import
EmbeddingDataset` e `from faceaudit.models.embedding_info import EmbeddingInfo`) e mapeia as
colunas de metadados do simulador sobre os campos reais do `EmbeddingInfo`. É o adapter a usar
em qualquer pipeline FaceAudit.

```python
from faceaudit_concrete_adapter import to_embedding_dataset

ds = to_embedding_dataset(
    test,
    name="synthetic_test",        # default: f"synthetic_{split}"
    embedding_view="raw",         # default "raw" (norma preservada)
    quality_key="generic",        # bucket lido por quality_score / BY_QUALITY
    origin="adaface_simulator",
    domain_filter=None,           # opcional, filtra por acquisition_domain
    dtype=np.float32,             # cast para casar com datasets reais extraídos
)
```

#### Mapeamento de campos

| Campo do `EmbeddingInfo` | Fonte no dataset sintético |
|---|---|
| `embedding` | `raw_embeddings` (default, norma preservada) ou `normalized_embeddings`, **cast para `float32`** |
| `identity` | `str(identity_id)` |
| `quality_scores[quality_key]` | `latent_quality` (bucket `"generic"` por padrão) |
| `source` | `acquisition_domain` (handle por amostra para splits cross-condition) |
| `origin` | tag de proveniência (default `"adaface_simulator"`) |
| `face_hash` | SHA-256 **determinístico** de `(split, identity_id, sample_index, session_index)` |
| `image_path` | caminho sintético determinístico `synthetic/{split}/{identity}/{sample}.png` |
| `file_name` | nome sintético determinístico `{identity}_{sample}.png` |
| `face_image` | `None` |
| `pose` | vazio (o simulador modela nuisance, não pose literal de cabeça) |

Pontos de projeto importantes:

- **`embedding_view="raw"` é o default.** Diferentemente do scaffold genérico (que usa
  `"normalized"`), o adapter concreto entrega por padrão o embedding bruto com a norma
  preservada, pois é o formato que casa com datasets reais extraídos e permite ao FaceAudit
  recuperar a norma L2 quando necessário.
- **Cast determinístico para `float32`** alinha o tipo com os datasets de embeddings reais
  extraídos pelo FaceAudit.
- **`face_hash` é um SHA-256 determinístico** dos identificadores da amostra, garantindo hashes
  estáveis e reproduzíveis entre execuções (mesmo `(split, identity_id, sample_index,
  session_index)` ⇒ mesmo hash).
- Covariáveis contínuas não mapeadas (`reference_scaled_log_norm_z`, `intra_kappa`,
  `radial_regime`, diagnósticos de passo tangencial, e as colunas de cap da v0.3) **permanecem**
  nos metadados do simulador e são persistidas pelos sidecars CSV de `save_bundle` (Seção 17) —
  nada essencial para estratificação é perdido, já que a norma é recuperável do embedding bruto.

#### `loaded_metadata` de proveniência

Cada `EmbeddingDataset` produzido recebe um `loaded_metadata` com a proveniência do experimento:
`generator`, `simulator_version`, `profile_name`, `split`, `embedding_view`, `dimension`,
`n_samples`, `n_identities` e, quando aplicável, `domain_filter`. Esse bloco é o registro de
cadeia de custódia do lado FaceAudit.

#### Cross-condition

```python
from faceaudit_concrete_adapter import cross_condition_to_embedding_datasets

aligned = simulator.sample_cross_condition_datasets(...)  # Seção 13
ds_by_domain = cross_condition_to_embedding_datasets(
    aligned,
    embedding_view="raw",
    quality_key="generic",
    origin="adaface_simulator",
    dtype=np.float32,
)
```

`cross_condition_to_embedding_datasets` mapeia a saída de `sample_cross_condition_datasets`
(Seção 13) para um dicionário de `EmbeddingDataset` por domínio. Como os valores de `identity`
são **idênticos** entre as condições, os datasets resultantes formam um conjunto cross-condition
pareado: cada identidade rende pares genuínos cross-condition — exatamente o que estratégias como
`PER_CONDITION` e `BALANCED_IDENTITY` precisam para comparações forenses entre condições
assimétricas.

## 19. Experimentos recomendados com PLDA

Esta seção descreve um conjunto de experimentos canônicos. O objetivo nunca é
obter um único número sintético para "provar" que um backend funciona, e sim
**isolar regimes**: em quais condições o PLDA ganha do cosseno, quando as
hipóteses homocedásticas começam a falhar, e quanto a dispersão intra-classe
afeta a separabilidade. Sempre use identidades disjuntas entre treino,
desenvolvimento e teste, e sempre registre o perfil + a configuração completa
(ver Seção 28).

### 19.1 Baseline esférico

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
sintético.

### 19.2 Baseline cooperativo

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

### 19.3 Stress cross-condition

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

Comparar:

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
**quebrar** o PLDA homocedástico de forma controlada.

### 19.4 Tipicidade

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
distância entre ela e os estimadores empíricos é viés de estimação.

### 19.5 Experimento de sweep de `lb` (modos de calota)

Este experimento usa o eixo `intra_class_sampling_mode` para varrer a **dispersão
intra-classe** de forma interpretável. Em vez de varrer `κ` (difícil de
interpretar em 512-D), varre-se o piso de cosseno `lb` do modo
`IDPERTURB_CAP`.

Gere a varredura em duas versões, exatamente como na Seção 15:

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

Comparação esperada entre as duas versões:

| Versão | Guarda | Curva `EER × lb` | `lb` efetivo |
|---|---|---|---|
| A | off | monotônica e limpa | ≈ `lb` nominal |
| B | on | achatada em regiões densas | ≥ `lb` nominal (reportar percentis) |

### 19.6 Experimento Modo B vs Modo A

Compare as duas formas de ancoragem em dados reais (ver Seção 12) no MESMO
backend, mantendo as identidades de ancoragem disjuntas das de avaliação:

- **Modo A** (`population_from_dataset`): os centros das identidades **são** as
  médias esféricas reais por identidade. As identidades são reais; só a dispersão
  intra-classe e a norma são sintéticas.
- **Modo B** (`simulator_config_from_dataset`): ajusta uma MoVMF aos centros
  reais e amostra **novos** ghost centres dessa aproximação angular de primeira
  ordem.

```python
from adaface_embedding_simulator import AdaFaceEmbeddingSimulator
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

---

## 20. Calibração contra embeddings AdaFace reais

> Os valores padrão dos perfis são ilustrativos. Antes de usar resultados
> sintéticos para afirmações quantitativas, ancore o simulador em uma base real
> representativa.

A partir da versão 0.3, o procedimento manual da v0.2 (estimar centros e
referência radial "na mão") foi **substituído por ferramentas dedicadas** no
módulo `faceaudit_seed`. Use-as em vez de reproduzir as fórmulas manualmente —
elas tratam pesos por qualidade, médias degeneradas, seleção de K e o caso L2.

### 20.1 Estimar a referência radial real

`RadialReference.fit` aceita diretamente a matriz `(N, D)` de embeddings brutos
reais e estima localização e escala robustas:

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

**Aviso L2:** se a base já estiver L2-normalizada (normas ≈ 1, dispersão ≈ 0), o
MAD escalado é zero e o ajuste é indefinido. Passe embeddings **brutos**, não
normalizados. O caminho automático (Seção 20.3) detecta esse caso e mantém a
referência radial-base com um warning, em vez de quebrar.

### 20.2 Estimar centros por identidade (Modo A)

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

Esses centros alimentam diretamente uma população ancorada via
`population_from_dataset` (ver Seção 12, Modo A).

### 20.3 Ajuste automático completo (Modo B)

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
substitui a `PopulationConfig` pela MoVMF ajustada (perfil renomeado para
`FACEAUDIT_MOVMF`) e, quando `fit_norm=True`, substitui a referência radial. O
`fit_summary` traz K selecionado, valores de critério por K, número de componentes
degenerados, entropia dos pesos e o resumo das normas — tudo persistível no
manifesto para auditoria.

### 20.4 Estatísticas mínimas a reproduzir

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

### 20.5 Ajuste incremental e separação ajuste/avaliação

Não ajuste tudo simultaneamente. Sequência recomendada:

1. distribuição radial (Seção 20.1);
2. distribuição populacional impostora (Seção 20.3);
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
avaliação com `assert_disjoint_source_identities` (Seção 12) — é uma asserção
obrigatória do checklist da Seção 28.

---

## 21. Matriz de ablações

A matriz de ablações tem dois eixos. O **eixo de acoplamento/nuisance** (A0–A6)
controla os caminhos causais qualidade → regime → norma → direção e o tipo de
nuisance. O **eixo `intra_class_sampling_mode`** controla a mecânica de
dispersão intra-classe. Os dois eixos são ortogonais e podem ser cruzados.

### 21.1 Eixo de acoplamento e nuisance (A0–A6)

| Cenário | Qualidade → regime | Qualidade → norma | Qualidade → direção | Norma → direção | Nuisance |
|---|---:|---:|---:|---:|---|
| A0 | não | não | não | não | nenhum |
| A1 | sim | sim | não | não | nenhum |
| A2 | não | não | sim | não | nenhum |
| A3 | sim | sim | não | sim | nenhum |
| A4 | sim | sim | sim | sim | global |
| A5 | sim | sim | sim | sim | por domínio |
| A6 | sim | sim | sim | sim | hierárquico |

As três primeiras colunas mapeiam diretamente para `CouplingConfig`:

| Coluna | Campo |
|---|---|
| Qualidade → regime | `quality_affects_regime_probabilities` |
| Qualidade → norma | `quality_affects_log_norm` |
| Qualidade → direção | `quality_affects_intra_kappa` |
| Norma → direção | `norm_z_affects_intra_kappa` |

A coluna Nuisance mapeia para `DirectionalConfig.nuisance_structure_mode`
(`global_shared` / `per_domain` / `hierarchical`).

Objetivo:

- atribuir causalmente alterações de desempenho;
- evitar interpretar um resultado agregado como efeito isolado de um único
  mecanismo.

### 21.2 Eixo `intra_class_sampling_mode`

Este eixo varia **apenas** o modo de dispersão intra-classe, mantendo o resto da
configuração fixo. Ver Seção 6.x para a mecânica.

| Modo | Controle | Pergunta experimental |
|---|---|---|
| `VMF` | `κ` | baseline direcional sem piso angular rígido |
| `IDPERTURB_CAP` | `lb` | o piso geométrico interpretável muda a separabilidade? |
| `HYBRID_VMF_CAP` | `κ` + `lb` | quanto o piso rígido distorce o corpo vMF? |

Observações ao cruzar este eixo com A0–A6:

- nos modos de calota, os switches `quality_affects_intra_kappa` e
  `norm_z_affects_intra_kappa` continuam ativos, mas gateiam a modulação do `lb`
  (dispersão intra-classe), apesar do nome histórico mencionar `kappa`;
- as penalidades de cauda da calota (`cap_low_tail_penalty`,
  `cap_high_tail_penalty`) agem em **escala de cosseno**, não em `logκ` — não
  reaproveite os mesmos valores das penalidades de `κ`;
- para uma ablação limpa do eixo, desligue `cap_overlap_guard` (a guarda eleva o
  `lb` efetivo de forma dependente da densidade populacional, contaminando a
  varredura — ver Seção 19.5);
- no `HYBRID_VMF_CAP`, audite `hybrid_used_cap_fallback`: com `lb` alto e `κ`
  baixo, grande parte das amostras vem da calota e o modo deixa de ser "vMF
  truncado" na prática.

Cruzamento recomendado para um relatório completo: rodar `{VMF, IDPERTURB_CAP,
HYBRID_VMF_CAP} × {A0, A4, A6}`, reportando para cada célula as métricas da
Seção 19.3 mais os percentis de `cap_lower_bound_effective` (nos modos de
calota).

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

### 22.2 MoVMF não reproduz a rede AdaFace nem a variedade real

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
nominal. Ligação com tipicidade: regiões densas (identidades típicas) toleram
menos variação; regiões raras toleram mais.

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

A suíte está organizada em sete módulos.

### 23.1 `test_adaface_embedding_simulator.py` — núcleo v0.2

Valida a geometria e a reprodutibilidade básicas:

- normas unitárias e alinhamento da média empírica da vMF;
- crescimento do comprimento resultante com \(\kappa\);
- consistência entre embeddings brutos e metadados;
- independência da ordem das chamadas e seeds explícitas;
- compartilhamento de população em cross-condition;
- bases distintas por domínio;
- probabilidades dos regimes radiais quando desacoplados;
- rótulos e estratégia `balanced_observation` dos pares;
- diagnóstico de bins do `qcut`.

### 23.2 `test_idperturb_cap.py` — calota, guarda e golden VMF

O módulo mais extenso da v0.3, cobrindo:

- norma unitária da calota e preservação do cosseno sorteado;
- `lb = 1` retorna a direção média;
- pisos vetoriais respeitados por linha;
- ortogonalidade e norma unitária dos tangentes determinístico e aleatório;
- modulação de `lb` por qualidade/norma quando acoplado, e `lb` plano quando
  desacoplado;
- offset de `lb` por domínio (`cap_lower_bound_offset`);
- piso do bissetor exclui o próprio centro e cresce com a densidade;
- versão em blocos do piso bate com a versão completa;
- a guarda mantém a perturbação mais próxima do próprio centro;
- **`lb` efetivo pode exceder o máximo configurado sob a guarda**;
- piso exato sem nuisance e invariante pré/pós-nuisance
  (`test_nuisance_can_push_post_cosine_below_lb_but_pre_holds`);
- piso do overlap **não** serializado no manifesto;
- modo HYBRID respeita o piso em todas as amostras;
- taxa de fallback do HYBRID sobe com `lb` alto e `κ` baixo;
- determinismo de cada modo;
- **golden VMF** (`test_golden_vmf_core_arrays_unchanged`): trava a saída do modo
  `VMF` bit-a-bit em relação à v0.2.0;
- round-trip do config do perfil `IDPERTURB_CAP`.

### 23.3 `test_config_roundtrip.py` — serialização

- `config_to_dict`/`config_from_dict` idempotente e JSON-safe;
- round-trip funcional bate bit-a-bit na geração;
- enums sobrevivem como strings;
- arrays NumPy serializam e reproduzem;
- `config_from_manifest` lê o bloco `config` do manifesto;
- chaves ausentes caem em defaults; chave desconhecida é rejeitada.

### 23.4 `test_core_population_injection.py` — injeção e extração de centros

- injeção de `centers` usa e normaliza os vetores;
- rejeição de "ambos" e "nenhum" entre `identity_count` e `centers`;
- validação de dimensão, componentes e comprimentos;
- **a injeção não altera o golden da população amostrada**;
- extração: norma unitária, contagens, média esférica manual, agrupamento pelo
  valor original de identidade, filtro `min_samples`, identidades `None`
  ignoradas;
- qualidade NaN/None e rejeição de pesos negativos;
- `weight_by_quality` exige `items`;
- fallback de média degenerada;
- população Modo A e colunas de proveniência;
- helpers de leakage;
- import do módulo sem disparar o import do `MoVMFTypicalityModel`.

### 23.5 `test_faceaudit_seed.py` — Modo B (MoVMF)

- K fixo constrói config válido;
- K automático seleciona dentro dos candidatos;
- aviso quando as normas são unitárias (caso L2);
- config do Modo B faz round-trip;
- **ghost centres do Modo B se parecem mais com os reais do que centros
  genéricos**;
- exige ao menos dois centros;
- round-trip com um `EmbeddingDataset` real.

### 23.6 `test_faceaudit_adapter.py` — adapter genérico

- registros simples e adapter por factory injetada.

### 23.7 `test_faceaudit_concrete_adapter.py` — adapter concreto

- mapeamento básico para `EmbeddingInfo`/`EmbeddingDataset`;
- `raw` preserva norma e `normalized` é unitário;
- face hashes únicos e proveniência;
- filtro de domínio;
- round-trip por parquet;
- cross-condition compartilha identidades.

Para produção, acrescente testes específicos do seu pipeline FaceAudit.

---

## 24. Solução de problemas

### `ValueError: anisotropic_rank must satisfy 0 <= rank < dimension`

Reduza `anisotropic_rank`. As factories prontas já limitam o posto para
dimensões reduzidas usadas em testes.

### O PLDA está performando bem demais

Verifique se está usando `make_cooperative_plda_config()`. Troque para
`make_cross_condition_stress_config()` e varie `domain_nuisance_weight`,
`identity_specific_nuisance_scale`, `domain_identity_bias_scale`,
`session_bias_scale`.

### O PLDA está performando mal demais

Reduza gradualmente os mesmos parâmetros. Não conclua que o backend é inadequado
apenas com base em um cenário extremo artificial.

### `qcut()` produziu menos bins

O método emite warning e informa os dois contadores. Examine clipping radial,
quantidade de valores idênticos, tamanho da base e parâmetros degenerados.

### `RadialReference.fit()` falhou com "scaled MAD is zero"

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

### `lb` efetivo excedendo `cap_maximum_lower_bound`

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

### Devo usar o adapter genérico ou o concreto?

Use **sempre o concreto** (`faceaudit_concrete_adapter.to_embedding_dataset`) em
produção: ele importa os DTOs reais e mapeia os campos corretos. O genérico
(`faceaudit_adapter.to_faceaudit_embedding_dataset`) assume
`EmbeddingInfo(metadata=...)`, que **não** corresponde ao DTO real do FaceAudit
(`EmbeddingInfo(embedding, face_hash, image_path, identity, file_name, ...)`, sem
`metadata`). O genérico é apenas referência/scaffold para projetos com outra
assinatura de DTO.

### `ValueError: Degenerate spherical mean` / pesos de qualidade

Em `identity_centers_from_dataset`, uma identidade cuja média esférica colapsa
para ≈0 gera warning e fallback para a melhor imagem por qualidade (ou a
primeira). Pesos de qualidade negativos são rejeitados; None/NaN viram peso 1;
todos-zero caem para média não-ponderada. Se o aviso de degeneração aparecer com
frequência, aumente `min_samples` (1 é ruidoso; 2+ é mais estável) ou revise a
coluna `quality_score`.

---

## 25. Exemplo completo recomendado

Este exemplo end-to-end exercita as capacidades centrais da v0.3: modo de
calota, round-trip de configuração, ancoragem Modo A/B, export pelo adapter
concreto e asserção de vazamento.

```python
from pathlib import Path

import numpy as np

from adaface_embedding_simulator import (
    AdaFaceEmbeddingSimulator,
    config_from_manifest,
    make_idperturb_cap_config,
    sample_pair_table,
)
from faceaudit_seed import (
    population_from_dataset,
    simulator_config_from_dataset,
    assert_disjoint_source_identities,
)
from faceaudit_concrete_adapter import to_embedding_dataset

output_dir = Path("synthetic_faceaudit_experiment")
output_dir.mkdir(exist_ok=True)

# 1) Mundo sintético com calota (IDPERTURB_CAP, lb = 0.6, guarda ligada)
config = make_idperturb_cap_config(dimension=512, lower_bound=0.6)
simulator = AdaFaceEmbeddingSimulator(config, random_seed=3030)

test = simulator.sample_dataset(
    identity_count=1_000,
    samples_per_identity=(4, 8),
    split_name="cap_test",
    identity_offset=300_000,
    population_random_seed=4001,
    random_seed=4002,
)

# Reporte a separação realizada: lb efetivo (a guarda pode elevá-lo)
eff = test.metadata["cap_lower_bound_effective"]
print("lb efetivo p10/p50/p90:", eff.quantile([0.10, 0.50, 0.90]).round(3).tolist())

# Bundle auditável (npz + metadata CSV + diagnostics CSV + manifesto JSON)
paths = test.save_bundle(output_dir=output_dir, stem="cap_test")

# 2) Round-trip de configuração: reconstruir o config a partir do manifesto
reloaded_config = config_from_manifest(paths["manifest"])
assert reloaded_config.profile_name == "IDPERTURB_CAP"

# 3) Pares + scores de cosseno
pairs = sample_pair_table(
    test,
    genuine_pair_count=50_000,
    random_impostor_pair_count=50_000,
    random_seed=5001,
)
pairs.to_csv(output_dir / "cap_test_pairs.csv", index=False)

# 4) Ancoragem em dados reais (requer faceaudit + datasets reais DISJUNTOS)
#    anchor_dataset  -> base autorizada para ancorar
#    eval_dataset    -> base de avaliação (NUNCA usada para ancorar)
sim_anchor = AdaFaceEmbeddingSimulator(
    make_idperturb_cap_config(dimension=anchor_dataset.matrix.shape[1], lower_bound=0.6),
    random_seed=11,
)

# Modo A — centros = médias esféricas reais por identidade
pop_anchor = population_from_dataset(sim_anchor, anchor_dataset, min_samples=3)
pop_eval = population_from_dataset(sim_anchor, eval_dataset, min_samples=3)

# 4a) ASSERÇÃO DE VAZAMENTO: âncora e avaliação devem ser disjuntas
assert_disjoint_source_identities(pop_anchor, pop_eval)

mode_a = sim_anchor.sample_observations(
    pop_anchor,
    samples_per_identity=(4, 8),
    split_name="mode_a",
    random_seed=4100,
)

# Modo B — população distribuição-matched (MoVMF ajustada aos centros reais)
config_b, fit_summary = simulator_config_from_dataset(anchor_dataset, criterion="bic")
print("Modo B — K selecionado:", fit_summary["selected_k"])
sim_b = AdaFaceEmbeddingSimulator(config_b, random_seed=11)
mode_b = sim_b.sample_dataset(
    identity_count=len(pop_anchor.identity_ids),
    samples_per_identity=(4, 8),
    split_name="mode_b",
    identity_offset=900_000,
    population_random_seed=4200,
    random_seed=4201,
)

# 5) Export via adapter CONCRETO (DTOs reais, float32)
faceaudit_mode_a = to_embedding_dataset(mode_a, embedding_view="raw")
faceaudit_mode_b = to_embedding_dataset(mode_b, embedding_view="raw")
```

---

## 26. Migração da versão 0.2 para 0.3

### 26.1 O modo `VMF` mantém a saída bit-a-bit

O padrão de `DirectionalConfig.intra_class_sampling_mode` é `IntraClassSamplingMode.VMF`,
e a saída do modo `VMF` é **idêntica bit-a-bit** à v0.2.0. Um teste golden
(`test_golden_vmf_core_arrays_unchanged`) trava esse contrato. Scripts da v0.2
continuam produzindo exatamente os mesmos números, sem qualquer mudança.

### 26.2 Os novos modos, campos e colunas são opt-in

Tudo que é novo na 0.3 só entra em ação quando explicitamente solicitado:

- **Modos de calota**: `IDPERTURB_CAP` e `HYBRID_VMF_CAP` exigem alterar
  `intra_class_sampling_mode`. Os campos `cap_*` do `DirectionalConfig` e o
  `cap_lower_bound_offset` do `AcquisitionDomainConfig` só têm efeito nesses
  modos.
- **Colunas novas de metadata** aparecem sempre (`intra_class_sampling_mode`,
  `pre_nuisance_center_cosine`, `cap_lower_bound_effective`,
  `hybrid_resample_count`, `hybrid_used_cap_fallback`), mas no modo `VMF` a
  coluna `cap_lower_bound_effective` é `NaN`, `hybrid_resample_count` é `0` e
  `hybrid_used_cap_fallback` é `False`. As colunas de proveniência
  `source_identity` / `source_sample_count` aparecem **apenas** em populações
  ancoradas em dataset (Modo A).

### 26.3 Helpers de serialização

A v0.3 fecha o round-trip de configuração que a v0.2 só tinha pela metade:

- `config_from_dict` reconstrói um `SimulatorConfig` a partir da saída de
  `config_to_dict` (valida, aceita defaults para chaves ausentes, rejeita chaves
  desconhecidas);
- `config_from_manifest` lê o bloco `config` de um manifesto escrito por
  `save_manifest`.

Isso permite reproduzir um experimento inteiro a partir do manifesto persistido.

### 26.4 Ferramentas de ancoragem em dados reais

O módulo `faceaudit_seed` é novo e substitui o procedimento manual de calibração
da v0.2:

- `identity_centers_from_dataset` — médias esféricas por identidade;
- `population_from_dataset` — população ancorada (Modo A);
- `population_config_from_dataset` / `simulator_config_from_dataset` — população
  e config distribuição-matched (Modo B, MoVMF via `MoVMFTypicalityModel`);
- `assert_disjoint_source_identities` — guarda de vazamento âncora↔avaliação.

O adapter concreto `faceaudit_concrete_adapter` (DTOs reais, float32, parquet,
cross-condition) também é novo e deve substituir o adapter genérico em produção.

### 26.5 Itens herdados da migração 0.1 → 0.2 (ainda válidos)

- prefira `reference_scaled_log_norm_z`; `robust_norm_z` segue como alias;
- prefira `optimal_reference_scaled_log_norm_z`; `optimal_robust_norm_z` segue
  como alias de leitura;
- streams nomeados tornam a geração independente da ordem das chamadas.

---

## 27. Roadmap

Melhorias futuras úteis (os itens de ajuste automático e adapter concreto da v0.2
foram **concluídos** na 0.3 e removidos desta lista):

1. serialização explícita das bases por domínio no `.npz`;
2. carregamento completo de bundles — incluindo reconstrução de
   `SimulatorConfig` direto do manifesto via `config_from_manifest` (o lado de
   config já existe; falta reidratar arrays e bases do `.npz`);
3. enumeração eficiente de todos os trials;
4. mineração de impostores difíceis por centro latente ou backend;
5. modelo temporal de sessão (substituir o viés de sessão simples por
   correlação temporal contínua);
6. métricas prontas de divergência entre dados reais e sintéticos;
7. geração de relatórios diagnósticos;
8. testes de propriedade adicionais para regimes extremos de \(\kappa\) e de
   `lb`.

---

## 28. Conclusão e checklist

A versão 0.3 mantém a família de cenários explícitos da 0.2 e adiciona um eixo
ortogonal de **dispersão intra-classe geometricamente interpretável**:

\[
\boxed{
\underbrace{\text{esférico} \rightarrow \text{cooperativo} \rightarrow \text{stress cross-condition}}_{\text{eixo de cenário (perfis)}}
\;\times\;
\underbrace{\text{VMF} \;|\; \text{IDPERTURB\_CAP} \;|\; \text{HYBRID\_VMF\_CAP}}_{\text{eixo de dispersão intra-classe}}
}
\]

O uso correto não é buscar um único número sintético para "provar" que um backend
funciona. O objetivo é identificar em quais regimes o PLDA ganha do cosseno,
quando as hipóteses homocedásticas falham, em quais intensidades
`BALANCED_IDENTITY` supera alternativas ingênuas, quanto tipicidade e qualidade
ficam acopladas, e quais propriedades dos embeddings reais precisam ser medidas
antes de qualquer conclusão quantitativa.

### Checklist de experimento auditável

Para todo experimento destinado a embasar uma afirmação, registre:

- [ ] **perfil** usado (`profile_name`);
- [ ] **configuração completa** (`config_to_dict` ou o bloco `config` do
  manifesto);
- [ ] **seed estrutural** + **seeds de população e observações** (por split e por
  condição);
- [ ] **política de trials** (estratégia de pares, `unique_pairs`, contagens);
- [ ] **versão do simulador** (`__version__`, atualmente `0.3.0`);
- [ ] **hash do código** (`source_sha256` no manifesto);
- [ ] **métricas estratificadas** (por domínio, por quantil de norma, por
  tipicidade);
- [ ] **`intra_class_sampling_mode`** usado;
- [ ] **distribuição realizada de `cap_lower_bound_effective`**
  (mean/p10/p50/p90) sempre que o modo for de calota — especialmente com
  `cap_overlap_guard=True`, em que o `lb` efetivo pode exceder o máximo nominal;
- [ ] **asserção de disjunção no Modo A** (`assert_disjoint_source_identities`
  passou) sempre que a população for ancorada em dados reais, registrando que as
  âncoras são disjuntas de qualquer conjunto de avaliação.

O manifesto JSON produzido por `save_bundle` / `save_manifest` cobre
automaticamente perfil, configuração completa, seeds, versão e hash do código. As
métricas estratificadas, o modo de amostragem, os percentis de `lb` efetivo e a
asserção de disjunção são responsabilidade do script de experimento — inclua-os
no relatório.
