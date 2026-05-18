# 02 — Criando um dataset

Este capítulo descreve o fluxo completo de **criar um dataset do zero**,
a partir de uma pasta de imagens faciais. O ponto de entrada é o botão
➕ **Novo** da aba *Arquivo* da toolbar.

## Sumário do capítulo

1. [Antes de começar — o que você precisa](#antes-de-começar)
2. [O diálogo *Novo Dataset* — visão geral](#o-diálogo-novo-dataset)
3. [Pasta de imagens e nome do dataset](#pasta-de-imagens-e-nome)
4. [Modo `flat` ou `recursivo`](#modo-flat-ou-recursivo)
5. [O Criador de Template — mapeando metadados a partir da estrutura de pastas](#o-criador-de-template)
6. [Metadados gerais (origem e descrição)](#metadados-gerais)
7. [O bloco *Processamento*](#o-bloco-processamento)
8. [Opções avançadas do *Forensic Face*](#opções-avançadas-do-forensic-face)
9. [Opções avançadas do *SER-FIQ*](#opções-avançadas-do-ser-fiq)
10. [Alinhamento facial — três cenários](#alinhamento-facial)
11. [O que acontece quando você clica **OK**](#o-que-acontece-quando-você-clica-ok)
12. [Erros comuns e como evitá-los](#erros-comuns)
13. [Workflows recomendados](#workflows-recomendados)

---

## Antes de começar

Para criar um dataset você precisa de:

| Item | Por que |
|---|---|
| **Uma pasta de imagens** | A entrada de tudo. Formatos suportados: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`. |
| **Forensic Face** (lib `forensicface`) instalada e funcional | Detecção facial, alinhamento e extração de embeddings. Sem ela, o botão *Novo* fica indisponível e a aba inteira é desabilitada. |
| **Modelos SEPAEL** (V2 e/ou V4) acessíveis | Pesos do encoder de embedding. Configurados na variável `paths.library` do `config.json`. |
| **Binário OFIQ** (`OFIQSampleApp`) + pasta `data/` do OFIQ | Apenas se você quiser scores OFIQ. Caminho configurado em `config.json → ofiq`. |
| **GPU NVIDIA com CUDA** *(opcional, mas fortemente recomendado)* | Inferência em CPU é viável para datasets pequenos mas fica impraticável acima de algumas centenas de imagens. |

> 💡 Se você só vai abrir um `.parquet` que outra pessoa preparou,
> nada disso é necessário — vá direto para [capítulo 03](03_carregando_e_localizando.md).

## O diálogo *Novo Dataset*

Ao clicar em ➕ **Novo** abre-se o diálogo abaixo, dividido em quatro
seções principais:

![Diálogo Novo Dataset, visão completa](assets/30_novo_dataset.png)

| Seção | O que controla |
|---|---|
| **1 — Entrada** | Pasta de origem, nome do dataset, recursividade, status de alinhamento |
| **2 — Estratégia de metadados** | Como derivar `identity`/`gender`/`ethnicity` a partir da estrutura de pastas + nomes de arquivo |
| **3 — Metadados gerais** | `origin` e `description` aplicados ao dataset inteiro |
| **4 — Processamento** | Quais extractors rodar: Forensic Face, OFIQ, SER-FIQ — cada um com seu painel avançado retrátil |

## Pasta de imagens e nome

| Campo | Descrição |
|---|---|
| **Pasta Imagens:** | Caminho da pasta a ser ingerida. O botão `…` à direita abre um seletor nativo do SO. Ao selecionar, o campo *Nome* é **auto-preenchido** com o nome da pasta (sem extensão), mas pode ser editado livremente. |
| **Nome:** | Nome lógico do dataset — vira o nome do arquivo `.parquet` final e a string que identifica o dataset em todos os diálogos posteriores. Caracteres válidos: letras, números, `_`, `-`. |
| ☑️ **Buscar em subpastas (Recursivo)** | Quando marcado, o ingestor desce em todas as subpastas. Quando desmarcado, considera apenas os arquivos diretos da pasta selecionada (modo *flat*). Marcado por padrão. |

## Modo *flat* ou *recursivo*

A escolha entre **flat** e **recursivo** muda completamente a seção 2
do diálogo:

### Modo `flat` (sem subpastas)

> Quando você desmarca **Buscar em subpastas**, todas as imagens da
> pasta selecionada são tratadas como **da mesma pessoa**. Os campos
> abaixo aplicam-se a *todas* as faces ingeridas:

| Campo | Tipo | Obrigatório | Observação |
|---|---|---|---|
| **Identidade:** | Texto livre | ✅ Sim | Vira a coluna `identity` para todas as imagens |
| **Gênero:** | `""` / `Male` / `Female` | Não | Vira a coluna `gender` |
| **Etnia:** | Texto livre | Não | Vira a coluna `ethnicity` |

> 💡 Use o modo *flat* quando estiver criando um dataset de **uma
> única pessoa** (por exemplo, para um caso pericial onde você só tem
> as fotos do investigado).

### Modo `recursivo` (com subpastas)

> No modo recursivo, o sistema percorre toda a árvore de subpastas e
> precisa de uma **estratégia para descobrir a identidade de cada
> imagem**. Há duas alternativas:

| Estratégia | Como funciona |
|---|---|
| **Nenhum (usar Identidade Base)** *(padrão)* | A identidade de cada imagem é o **nome da pasta-pai imediata**. Pasta `John_Doe/img_001.png` → `identity="John_Doe"`. Suficiente para os datasets clássicos (CASIA-WebFace, LFW, etc.). |
| **Template salvo** | Um arquivo JSON descrevendo regras de extração para `identity`, `gender`, `ethnicity` e `origin` tanto a partir de **níveis de pasta** quanto a partir de **partes do nome do arquivo**. Veja a próxima seção. |

O botão **Criar Template** abre o diálogo de criação descrito a
seguir.

## O Criador de Template

Um *template* é um arquivo `.json` salvo em `data/templates/` que
descreve como derivar metadados a partir da estrutura de pastas e
nomes de arquivo. Uma vez salvo, o template aparece automaticamente
no dropdown **Template** do diálogo *Novo Dataset*.

### Anatomia do template

```json
{
  "name": "BFW_Oficial",
  "description": "Balanced Faces in the Wild",
  "separator": "_",
  "use_separator": true,
  "folder_map": {
    "0": ["identity"],
    "1": ["gender"]
  },
  "filename_map": {
    "0": "identity",
    "1": "ethnicity"
  },
  "identity_from_parent": false
}
```

| Campo | Significado |
|---|---|
| `name` | Nome exibido no dropdown |
| `description` | Texto livre (não usado pela engine) |
| `separator` | Caractere separador para dividir nomes de pasta e arquivo (`_`, `-`, `.`, etc.) |
| `use_separator` | Quando `false`, a divisão não é aplicada — o nome inteiro vira o valor único |
| `folder_map` | Para cada **nível de profundidade** (`0` = primeira subpasta), lista de atributos a extrair, na ordem |
| `filename_map` | Para cada **índice de parte do nome do arquivo** (após split pelo separador), qual atributo recebe o valor |
| `identity_from_parent` | Quando `true`, **força** `identity` a ser o nome da pasta-pai imediata, ignorando `folder_map` para esse atributo |

### Aplicando o template

Para a estrutura abaixo, com o template do exemplo acima:

```
dataset_root/
├── John_Male/
│   └── person_001.jpg     → identity="John", gender="Male" (folder_map)
│                            identity sobrescrita="person", ethnicity="001" (filename_map)
└── Jane_Female/
    └── person_002.jpg
```

O sistema:
1. Percorre os níveis de pasta aplicando `folder_map` em ordem
2. Aplica `filename_map` sobrepondo eventuais valores anteriores
3. Se `identity_from_parent=true`, sobrescreve `identity` no fim

> ⚠️ Quando o mesmo atributo (ex: `identity`) aparece tanto em
> `folder_map` quanto em `filename_map`, **a última definição
> prevalece**. Se você quer ignorar uma parte específica, use o
> atributo especial `Ignorar` (não diferencia maiúsculas).

### O diálogo do criador

O criador tem duas abas:

| Aba | Função |
|---|---|
| 📁 **Estrutura de Pastas** | Permite adicionar regras `[Nível X] → atributo`. Use o `SpinBox` (0–10) para o nível e o combo para o atributo. |
| 📄 **Nome do Arquivo** | Permite adicionar regras `[Parte X] → atributo`. Mesmas faixas e atributos. |

No topo: nome do template + ativação do separador + checkbox
*"Usar nome da pasta pai imediata como Identidade"*.

Templates são salvos automaticamente em `data/templates/{name}.json`.

## Metadados gerais

| Campo | Padrão | Observação |
|---|---|---|
| **Origem:** | `Local` | Rótulo livre — útil pra distinguir várias fontes (`Local`, `CCTV_Aeroporto`, `Documento_ID`, `Webcam_Forense`). Vira a coluna `origin`. |
| **Descrição:** | (vazio) | Texto livre, salvo como metadado do dataset. Visível no painel de detalhes e no diálogo *Metadados*. Sem limite rígido, mas mantenha curto (≤ 500 caracteres). |

## O bloco *Processamento*

Aqui você decide **quais extractors** vão rodar sobre cada imagem.
Os três são opcionais e podem ser ativados de forma independente:

| Checkbox | Padrão | O que produz |
|---|---|---|
| ☑️ **Forensic Face** | Marcado | `embedding_v2` e/ou `embedding_v4`, FIQA, pose (yaw/pitch/roll), gênero/idade estimados, IPD (interpupillary distance), bbox e keypoints faciais |
| ☑️ **OFIQ** | Marcado | ~40 indicadores ISO/IEC 29794-5 (sharpness, exposição, oclusões, pose, single-face, etc.) + score unificado |
| ☑️ **SER-FIQ** | Marcado | Score de robustez por Monte-Carlo dropout (mais alto = embedding estável a perturbações) |

> 💡 **Desmarque** algum se você não precisa dele — economiza tempo
> significativo. Por exemplo, em datasets *puramente para auditoria
> visual* (sem cálculo de LR), pode desativar Forensic Face e manter
> só OFIQ + SER-FIQ.

Cada extractor tem um **botão de avançadas retrátil** logo abaixo.

## Opções avançadas do *Forensic Face*

Clique em *▶ Opções avançadas* sob o checkbox **Forensic Face** para
expandir:

| Controle | Faixa | Padrão | Significado |
|---|---|---|---|
| ☑️ **`sepaelv2`** / ☑️ **`sepaelv4`** | — | só V2 | Quais backbones rodar. Marcar os dois gera as colunas `embedding_v2` *e* `embedding_v4` no parquet — dobra a memória de GPU em uso simultâneo, mas o custo por imagem permanece igual (uma engine independente por modelo). |
| **det_size** | 160 – 1280 (passo 32) | 320 | Tamanho da imagem de entrada do detector facial. Maior = melhor detecção em fotos de baixa resolução, mas **~2× mais custo** de inferência ao dobrar. Use 640 quando a face na imagem original mede menos de 80 px. |
| ☑️ **extended** | — | Marcado | Quando ligado, calcula FIQA, gênero, idade e pose junto com o embedding. Desligar economiza tempo mas remove essas colunas. |

> 💡 Recomendação geral: **sepaelv2 + det_size=320 + extended=True**.
> Use sepaelv4 adicional quando estiver fazendo experimentos
> comparativos de pipeline (veja [capítulo 05](05_enriquecimento.md)
> para adicionar embeddings adicionais a um dataset já criado).

## Opções avançadas do *SER-FIQ*

| Controle | Faixa | Padrão | Significado |
|---|---|---|---|
| **T (iterações)** | 1 – 100 | 10 | Número de passes estocásticos (com dropout ativo) usados para estimar a estabilidade do embedding. Mais iterações = score mais estável, mas tempo linear. |
| **alpha** | 0.1 – 10.0 (passo 0.1) | 1.5 | Fator de escala do score final (parâmetro da fórmula original de Terhörst et al.). Mantenha em 1.5 a menos que esteja comparando com a literatura sob outra escala. |

## Alinhamento facial

O FaceAudit suporta **três cenários** de alinhamento, mutuamente
exclusivos:

| Cenário | Configuração | Quando usar |
|---|---|---|
| **A — Imagens originais, alinhar na hora da ingestão** | Ambos checkboxes desmarcados (padrão) | Suas imagens estão "como saíram da câmera" — fotos casuais, frames de vídeo, scans. O ForensicFace detecta e alinha durante a extração de embedding. **O resultado é só usado em RAM** — as imagens no disco continuam as originais. |
| **B — Imagens originais, alinhar e materializar** | ☑️ *Alinhar faces durante a criação* + **Destino:** preenchido | Igual ao A, mas **salva** as faces alinhadas como PNGs 112×112 num diretório à sua escolha; o parquet aponta para esses novos arquivos. Idêntico ao resultado de rodar o cenário A seguido de exportação manual. |
| **C — Imagens já alinhadas (112×112)** | ☑️ *Estas imagens são faces já alinhadas* | Seu input já está em crops 112×112 pós-ForensicFace (por exemplo, recebeu o dataset de outra pessoa). Pula a etapa de detect+align — **~50 % mais rápido**. |

> ⚠️ Os dois checkboxes são **mutuamente exclusivos** — marcar um
> desabilita automaticamente o outro. Se marcar *alinhar durante a
> criação* sem informar a pasta destino, ou sem que o **Forensic
> Face** esteja ligado, o sistema bloqueia a criação com erro.

## O que acontece quando você clica **OK**

O fluxo é assíncrono — um *worker* em thread separada conduz toda a
extração, enquanto a UI mostra um diálogo de progresso. Em alto
nível, o pipeline executa:

```
0 % — Inicializando
 ↓
EXTRACT (≈10 %)
 ├── Varre a pasta (recursivo ou flat)
 ├── Aplica o template (ou identifica pela pasta-pai)
 └── Calcula SHA-256 de cada imagem
 ↓
TRANSFORM
 ├── Aplica gênero/etnia/origem padrão se informados
 ├── (15 → 50 %) Forensic Face — embeddings + FIQA + pose
 │    [se alinhar+materializar: salva PNGs no destino]
 ├── (50 → 75 %) OFIQ — métricas ISO
 └── (75 → 95 %) SER-FIQ — robustez
 ↓
LOAD (≈100 %)
 ├── Marca faces que falharam (coluna `error`, `error_stage`, `error_message`)
 └── Salva `{paths.library}/{nome}.parquet`
 ↓
Dataset carregado na UI
```

O progresso é exibido em tempo real (mensagens + porcentagem).
Você pode cancelar a qualquer momento — o sistema **salva o parquet
parcial** antes de abortar.

## Erros comuns

| Sintoma | Causa | Solução |
|---|---|---|
| Botão **OK** travado em cinza | Faltou preencher *Pasta Imagens*, *Nome*, ou (em modo flat) *Identidade* | Preencha o campo destacado em vermelho |
| Diálogo erro `align_during_ingestion=True mas FF desligado` | Você marcou *alinhar* sem marcar Forensic Face | Marque o checkbox **Forensic Face** ou desmarque *alinhar* |
| `Diretório destino não informado` | `align_during_ingestion=True` mas o campo *Destino PNGs alinhados* ficou vazio | Informe uma pasta existente onde os PNGs serão escritos |
| Dataset criado mas **vazio** | A pasta não continha imagens de formato suportado | Confira se há `.jpg`/`.png`/`.bmp`/`.tif`. O log do worker lista exatamente o que foi varrido |
| Muitas linhas com `error=True` | ForensicFace não detectou face nas imagens (faces muito pequenas, pose extrema, fora de foco) | Aumente `det_size` para 640 ou 1024; verifique se as imagens **não estão já alinhadas** sendo tratadas como originais |
| Erro `ForensicFace não encontrado` na inicialização | A lib `forensicface` ou os modelos SEPAEL não estão instalados/configurados | Veja a seção [Antes de começar](#antes-de-começar) |
| GPU OOM (Out of Memory) | Imagens grandes + `det_size` alto + `sepaelv2 + sepaelv4` simultâneos | Reduza para apenas um modelo, baixe `det_size`, ou diminua `batch_size` em `config.json → ff` |

## Workflows recomendados

### Workflow 1 — Caso pericial simples (uma pessoa)

> 📷 *"Tenho 200 fotos do investigado em pasta única, formato JPG variado."*

1. ➕ **Novo** → selecionar pasta
2. Desmarcar **Buscar em subpastas**
3. *Identidade:* nome ou código do investigado (`SUSPEITO_001`)
4. *Gênero* e *Etnia*: opcional
5. *Origem:* `Caso_2024_001`
6. Manter Forensic Face + OFIQ + SER-FIQ marcados
7. **OK**

### Workflow 2 — Dataset de pesquisa (várias identidades)

> 🎓 *"CASIA-WebFace baixado, estrutura `casia/n000001/0001.jpg`."*

1. ➕ **Novo** → selecionar pasta `casia/`
2. **Buscar em subpastas** marcado
3. Estratégia: **Nenhum (usar Identidade Base)** — o nome da subpasta vira a identidade
4. *Origem:* `CASIA-WebFace`
5. Forensic Face com `sepaelv2 + sepaelv4` (se quiser comparar pipelines depois)
6. **OK**

### Workflow 3 — Dataset complexo (template customizado)

> 🧪 *"Tenho `bfw/asian_male/identity_42/photo_03.jpg` e quero
> `identity`, `gender`, `ethnicity` separados."*

1. ➕ **Novo** → selecionar pasta `bfw/`
2. **Buscar em subpastas** marcado
3. Clicar **Criar Template**
4. Na aba *Estrutura de Pastas*:
   - Nível 0 → `ethnicity` *(separator=`_` ativo: `asian_male` vira `["asian","male"]`)*
   - Nível 0 → `gender` *(captura o segundo item)*
   - Nível 1 → `identity`
5. Sem split de filename (ou Nível 0 → `Ignorar`)
6. Salvar template como `BFW`
7. Voltar ao diálogo *Novo Dataset*, selecionar `BFW` no dropdown
8. **OK**

### Workflow 4 — Já alinhado, máxima velocidade

> ⚡ *"500 PNGs 112×112 prontos, só preciso dos embeddings."*

1. ➕ **Novo** → selecionar pasta
2. Modo *flat* (uma identidade) ou *recursivo* (uma pasta por identidade)
3. ☑️ **Estas imagens são faces já alinhadas**
4. Desmarcar **OFIQ** *(opcional — OFIQ exige imagens não cropadas para a maioria das métricas funcionarem direito)*
5. **OK**

---

[← Capítulo 01 — Visão geral](01_visao_geral.md) · [Capítulo 03 — Carregando e localizando →](03_carregando_e_localizando.md)
