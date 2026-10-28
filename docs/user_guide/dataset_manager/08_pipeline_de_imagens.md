# 08 — Pipeline de imagens

A aba **Imagens** do ribbon expõe dois botões que trabalham juntos:

- 🎨 **Editor** — monta um *pipeline* de transformações de imagem
  interativamente, com pré-visualização e métricas de qualidade
- 🌀 **Processar** — aplica um pipeline salvo a todo o dataset,
  gerando um **novo dataset** com as imagens transformadas

A motivação forense: simular **degradações realistas** (compressão
de mensageiro, oclusão, blur de zoom digital, etc.) sobre um
dataset de referência e verificar como o sistema de comparação
biométrica se comporta.

## Sumário do capítulo

1. [Conceito de *pipeline de imagens*](#conceito)
2. [O Editor — leiaute em três painéis](#o-editor)
3. [Os tipos de step](#os-tipos-de-step)
4. [Configuração detalhada de cada step](#configuração-detalhada)
5. [Métricas de qualidade ao vivo](#métricas-ao-vivo)
6. [Salvando e carregando pipelines](#salvando-e-carregando)
7. [O diálogo *Processar*](#o-diálogo-processar)
8. [O dataset de saída](#o-dataset-de-saída)
9. [Casos de uso forense](#casos-de-uso-forense)

---

## Conceito

Um **pipeline de imagens** é uma sequência ordenada de
**transformações** aplicadas a cada imagem do dataset. Exemplos:

- `Blur(σ=2) → JPEG(qual=60)` — simula uma foto desfocada
  comprimida pelo WhatsApp
- `Crop(face_detected, padding=0.2) → JPEG(qual=85)` — recorta
  só a face e comprime
- `Occlusion(rectangle, 30%) → JPEG(qual=70)` — simula uma
  evidência parcialmente bloqueada

> 💡 Pipelines são **reutilizáveis** — você salva como `.json` na
> biblioteca global, e qualquer dataset pode ser processado por
> qualquer pipeline. Esse é o objeto declarativo equivalente ao
> *StrategyDefinition* do Gerenciador de Estratégias, mas para
> processamento de imagens.

## O Editor

![Editor de Pipeline de Imagens com painéis de steps, configuração e preview](assets/41_image_pipeline_editor.png)

O diálogo tem **três painéis principais**:

### Painel esquerdo — Lista de steps (split vertical 35%)

| Sub-área | Função |
|---|---|
| **Topo — Lista** | Steps na ordem de aplicação. Cada linha mostra `Nº.  Rótulo` (ex.: `1.  Desfoque`). Drag-drop pra reordenar. Botão ✕ vermelho remove |
| **Base — Configuração** | Formulário dinâmico que reage ao step selecionado — campos variam conforme o tipo |

Botão verde **➕ Adicionar** no topo abre um menu com os tipos
disponíveis, agrupados (Enquadramento, Tom, Impressão e cópia,
Desfoque, Compressão, Oclusão). Cada linha da lista mostra um resumo
dos parâmetros do passo — dois *Redimensionar* na mesma cadeia ficam
distinguíveis.

### Painel direito superior — Image Compare (65%)

| Elemento | Função |
|---|---|
| **Imagem original** (480×480 esquerda) | Imagem de entrada, sem transformação |
| **Imagem processada** (480×480 direita) | Resultado após aplicar o pipeline atual |
| ◀ ▶ 🎲 | Navegação entre faces do dataset (anterior / próxima / aleatória) |
| Status label | `17 / 234  · foto_05.png` (índice atual / total · nome do arquivo) |
| **▶ Aplicar** | Reaplica o pipeline na face atual (após alterar steps/parâmetros) |

### Painel direito inferior — Scores de qualidade

![Painel de scores de qualidade (FIQA, norma do embedding, cosseno)](assets/41_image_pipeline_editor.png)

Tabela 3×4 (**Scores de qualidade**) que se preenche após cada
*Aplicar*:

| Métrica | Original | Processada | Δ |
|---|---|---|---|
| FIQA | 0.8 | 0.6 | -0.2 ↓ *(vermelho)* |
| ‖embedding‖ | 22.400 | 19.800 | -2.600 |
| Cosine sim. | 1.0000 | 0.9100 | -0.0900 |

Apenas o **delta do FIQA** ganha cor: 🔴 vermelho + seta ↓ quando o
FIQA cai mais de 0.1 (degradação), 🟢 verde + seta ↑ quando sobe mais
de 0.1; variações pequenas ficam sem cor. Os deltas de ‖embedding‖ e
Cosine sim. são **informativos** — mostrados sem cor. O cálculo roda
num *worker* separado pra não travar a UI.

### Topo — controles globais

| Controle | Função |
|---|---|
| **Nome** | Identificador do pipeline (vira o stem do JSON salvo) |
| **Origem** | Fonte que alimenta o pipeline: **Imagem original** (foto inteira, com fundo e pose variada) ou **Face alinhada (112×112)** (o ForensicFace detecta e alinha a face **antes** de qualquer step rodar) |
| **Seed** | Inteiro pra steps aleatórios (`Oclusão` com posição aleatória, etc.) — torna o pipeline reproduzível |
| **📂 Carregar** | Carrega um pipeline salvo da biblioteca global (substitui o pipeline em memória; mudanças não salvas são descartadas após confirmação) |
| **💾 Salvar** | Pergunta o ID do pipeline (default = nome atual) e grava em `<biblioteca>/image_pipelines/<id>.json`; se já existir um pipeline com esse ID, pergunta se quer sobrescrever |

> 💡 **Truque**: selecione **Origem = Face alinhada (112×112)** e deixe o
> pipeline **sem nenhum step**. O *Processar* então produz um dataset
> contendo apenas as faces alinhadas 112×112 como PNG — é o substituto do
> antigo *Copiar Faces Alinhadas* do diálogo *Modificar Caminhos*. O dataset
> derivado é marcado como `is_aligned=True`, de modo que um **Enriquecer**
> posterior ([capítulo 05](05_enriquecimento.md)) não re-detecta a face
> dentro do crop já alinhado.

> ⚠️ **Salvar/Carregar exigem biblioteca global.** Se `paths.library` não
> estiver configurado, os dois botões ficam desabilitados com um tooltip
> explicativo. O título da janela mostra um `*` enquanto houver mudanças
> não salvas, e fechar o editor com pendências pede confirmação.

## Os tipos de step

Os steps aparecem na lista e no menu **➕ Adicionar** com os rótulos
em português abaixo. Os nomes em maiúsculas (`BLUR`, `JPEG`, …) são os
identificadores internos usados no JSON — a UI não exibe ícones por
tipo de step.

| Rótulo na UI | Tipo interno | O que faz |
|---|---|---|
| **Desfoque** | `BLUR` | Aplica filtro Gaussiano. Simula desfoque por movimento, falha de foco, ou degradação de zoom |
| **Compressão JPEG** | `JPEG` | Codifica + decodifica em JPEG. Simula compressão de mensageiros, e-mail, redes sociais |
| **Compressão WebP** | `WEBP` | Como JPEG mas no formato WebP (compressão moderna, melhor qualidade pelo mesmo bitrate) |
| **Recorte** | `CROP` | Recorta a imagem. Modos centralizado (fixo) ou detectado por face (com padding) |
| **Oclusão** | `OCCLUSION` | Adiciona oclusão (máscara desenhada à mão, retângulo preto ou círculo preto). Simula obstruções parciais |
| **Redimensionar** | `RESIZE` | Muda a resolução — por **distância entre os olhos alvo** (iguala o tamanho da face entre fotos diferentes) ou por fator |
| **Tons de cinza** | `GRAYSCALE` | Converte para tons de cinza |
| **Curva de tons** | `TONE_CURVE` | Ponto de preto, ponto de branco e gama — o contraste estourado da copiadora |
| **Ganho de toner** | `TONER_GAIN` | Expande as áreas escuras (o toner espalha) |
| **Meio-tom** | `HALFTONE` | Reduz a preto e branco puros: difusão de erro, pontilhado ordenado ou retícula de pontos |
| **Ruído de toner** | `TONER_NOISE` | Pontos escuros soltos (toner fora do lugar, poeira no vidro) |

> 💡 A intenção é o caso pericial *"reproduza no dataset de
> referência o que aconteceu na evidência"* — se a foto do caso veio
> do WhatsApp, comprima a referência com JPEG; se a identificação só
> mostra parte do rosto, oclua a referência; se veio de **xerox de
> documento**, use a cadeia de impressão e cópia (veja o Caso E).

## Configuração detalhada

> 💡 A coluna **Padrão** lista o valor de esquema (usado quando o
> config é carregado de JSON). Ao **adicionar** um step novo no editor,
> alguns campos vêm pré-preenchidos com valores mais visíveis: Desfoque
> abre com kernel **7** e sigma **2.0**; Compressão JPEG e WebP abrem
> com qualidade **50**. Isso é só o valor inicial do formulário — você
> ajusta livremente.

### Desfoque (BLUR) — Borrão Gaussiano

| Campo (rótulo na UI) | Faixa | Padrão | Notas |
|---|---|---|---|
| **Tipo** | Combo (só Gaussiano no MVP) | Gaussiano | Outros tipos (motion blur, defocus) podem vir em versões futuras |
| **Kernel** | Ímpar entre 1 e 99 | 5 | Tamanho do kernel; quanto maior, mais alcance |
| **Sigma** | 0.1 – 50.0 | 1.0 | Desvio padrão da gaussiana; quanto maior, mais borrão |

### Compressão JPEG (JPEG)

| Campo (rótulo na UI) | Faixa | Padrão | Notas |
|---|---|---|---|
| **Qualidade** | 1 – 100 | 85 | Qualidade JPEG. Valores baixos mostram artefatos pesados; > 95 são quase imperceptíveis |

### Compressão WebP (WEBP)

| Campo (rótulo na UI) | Faixa | Padrão | Notas |
|---|---|---|---|
| **Qualidade** | 1 – 100 | 80 | WebP geralmente preserva melhor que JPEG no mesmo valor |

### Recorte (CROP)

| Campo (rótulo na UI) | Faixa | Padrão | Notas |
|---|---|---|---|
| **Modo** | `CENTERED` / `FACE_DETECTED` | CENTERED | |
| *(CENTERED)* **Largura** | 8 – 4096 | 224 | Largura do recorte em px |
| *(CENTERED)* **Altura** | 8 – 4096 | 224 | Altura do recorte em px |
| *(FACE_DETECTED)* **Margem** | 1,0 – 5,0 | 1,0 | Fator sobre a caixa da face: 1,0 = caixa exata, 2,0 = o dobro em cada dimensão (mesma unidade da extração de vídeo) |

> 💡 No modo `FACE_DETECTED`, o recorte é feito em volta da caixa da
> face **do dataset** (ver *A posição da face* abaixo). Útil pra
> **extrair só a face** preservando a pose original.

### Oclusão (OCCLUSION)

O tipo **default é Máscara desenhada** (`MASK_FACE`), que esconde os
campos de posição/dimensão e usa apenas a máscara que você desenha. Os
tipos geométricos (retângulo e círculo) usam posição + dimensões.

| Campo (rótulo na UI) | Faixa | Padrão | Notas |
|---|---|---|---|
| **Tipo** | Máscara desenhada / Retângulo preto / Círculo preto | Máscara desenhada | |
| **Posição** | Centralizada / Aleatória / Sobre os olhos / Sobre a boca | Centralizada | Oferecida por **Retângulo preto** e **Círculo preto**; oculta em Máscara desenhada. Sobre os olhos / Sobre a boca usam os pontos da face |
| *(Retângulo)* **Largura** | 0.01 – 1.0 | 0.5 | Fração da largura da imagem ocluída |
| *(Retângulo)* **Altura** | 0.01 – 1.0 | 0.5 | Fração da altura da imagem ocluída |
| *(Círculo)* **Raio** | 0.01 – 0.50 | 0.10 | Raio como fração do menor lado |
| *(Máscara)* **Máscara** | Formas desenhadas | (vazio) | Botão **🎨 Desenhar máscara…** abre o canvas |

> 💡 **Máscara desenhada** é o modo mais flexível — o botão
> **🎨 Desenhar máscara…** abre um canvas onde você desenha
> **polígonos, retângulos e elipses** sobre a imagem do preview. As
> coordenadas são salvas em `[0..1]` (normalizadas), então a mesma
> máscara se aplica a imagens de tamanhos diferentes. Útil pra simular
> obstruções específicas (faixa de evidência, perda de resolução em
> região específica). Com uma ou mais formas, o botão passa a mostrar
> **🎨 Editar (N shapes)…**.

### Passos de impressão e cópia

| Passo | Campo | Faixa | Padrão | Notas |
|---|---|---|---|---|
| **Redimensionar** | Escala por | IPD / Fator fixo | IPD | |
| | IPD alvo | 2 – 2000 px | 40 | Distância entre os centros dos olhos depois do passo. Meça na questionada |
| | Fator | 0,01 – 20 | 0,5 | 0,5 = metade |
| | Interpolação | Automática / área / bilinear / bicúbica / vizinho | Automática | Automática = área ao reduzir, bicúbica ao ampliar |
| | Voltar ao tamanho original | sim / não | não | Perde resolução sem mudar as dimensões em pixels |
| **Curva de tons** | Ponto de preto | 0 – 254 | 40 | Abaixo dele, tudo vira preto |
| | Ponto de branco | 1 – 255 | 200 | Acima dele, tudo vira branco |
| | Gama | 0,1 – 5 | 1,0 | > 1 escurece os meios-tons; < 1 clareia |
| **Ganho de toner** | Raio | 0 – 20 px | 1 | Quanto o escuro avança, na resolução em que o passo roda |
| | Intensidade | 0 – 1 | 1 | |
| **Meio-tom** | Técnica | Difusão de erro / ordenado / retícula | Difusão de erro | |
| | Tamanho da célula | 2 – 64 px | 6 | Só retícula |
| | Ângulo | 0 – 180° | 45 | Só retícula |
| **Ruído de toner** | Densidade | 0 – 0,05 | 0,002 | Fração dos pixels que recebe um ponto |
| | Raio do ponto | 0 – 5 px | 0 | 0 = um pixel |

> ⚠️ **O meio-tom produz preto e branco puros na resolução em que
> roda.** Uma cópia digitalizada tem cinzas porque o scanner (ou a
> redução) junta os pontos. Por isso o meio-tom vem **antes** do
> *Redimensionar* que leva à resolução da questionada — e a
> resolução em que o meio-tom roda controla o grão: quanto mais perto
> da final, mais grosso.

### A posição da face

Recorte pela face, oclusão sobre os olhos ou a boca (e, nos passos de
cópia, o redimensionamento pela distância entre os olhos) precisam saber
onde está a face. Essa posição **acompanha a imagem pelo pipeline**:

- vem do **dataset** — a caixa e os pontos gravados na criação, os
  mesmos que geraram o embedding;
- se o dataset não os tem, a face é detectada **uma vez**, na imagem de
  entrada (exige o FFService);
- cada passo que muda o enquadramento (recorte, redimensionamento)
  converte a posição para a imagem que produziu.

Os passos **não detectam de novo** na imagem já degradada pelos passos
anteriores, onde a detecção pode falhar ou achar a face em outro lugar.
Se a posição não estiver disponível, o passo **falha** (no Processar, a
imagem vai para o parquet de erros) — ele não cai no centro da imagem,
o que produziria uma oclusão no lugar errado sem aviso.

## Métricas ao vivo

O painel inferior direito (**Scores de qualidade**) está **sempre
visível** sob o preview — não há como ligá-lo ou desligá-lo. Ele fica
vazio (`—`) para um pipeline sem steps e se preenche assim que você
clica **▶ Aplicar**:

| Métrica | O que indica |
|---|---|
| **FIQA** | Score de qualidade da face. Cair mais de 0.1 indica degradação que afeta reconhecimento (único delta colorido) |
| **‖embedding‖** | Magnitude do vetor de embedding. Valores muito baixos indicam *embedding collapse* (a transformação destruiu informação demais) |
| **Cosine sim.** | Cosseno entre embeddings de original e processada. 1.0 = idêntico; valores baixos significam que o pipeline mudou muito a "assinatura" biométrica |

Quando a face não é detectada na imagem processada, a linha de status
do painel avisa (`⚠ face destruída pelo pipeline (ou não detectada)`).

### Recomputação

As métricas são recalculadas **somente quando você clica ▶ Aplicar**.
Mudar steps/parâmetros não dispara recálculo automático — você
controla o momento (evita cálculos desnecessários durante
edição).

O cálculo roda num *worker* assíncrono. A UI continua responsiva
durante o cálculo; enquanto ele roda, as células mostram `…` e a linha
de status exibe *"Calculando scores…"*.

## A imagem questionada como alvo

O botão **📄 Questionada…** (no topo do Editor) carrega a imagem
questionada que o pipeline deve reproduzir. O FaceAudit localiza a face
nela, mede, e a partir daí:

- ela aparece num **terceiro painel**, ao lado da processada;
- o painel **Medidas de degradação** mostra, lado a lado, as medidas da
  **original**, da **processada** e da **questionada**. A célula da
  processada fica **verde** quando a medida está dentro da tolerância da
  questionada e **âmbar** quando não está (a tolerância de cada uma está
  na dica do mouse);
- o pipeline guarda o **perfil da questionada** — nome, **SHA-256** e as
  medidas. Ao reabrir o pipeline, a imagem é procurada no caminho
  gravado e o SHA-256 é conferido; se o arquivo sumiu ou mudou, as
  medidas gravadas continuam valendo e a linha de informação avisa.

| Medida | O que diz | Tolerância |
|---|---|---|
| Distância entre os olhos | Resolução da face, em px | ± 10% |
| Pretos (≤ 20) | Fração da região da face em preto | ± 5 pontos percentuais |
| Brancos (≥ 235) | Fração em branco | ± 5 p.p. |
| Meios-tons (60–195) | Fração em tons intermediários | ± 5 p.p. |
| Grão (variância do laplaciano) | Energia de alta frequência: grão, ruído, nitidez | ± 30% |
| Níveis de cinza distintos | Quantos dos 256 níveis aparecem | ± 25% |
| Nível médio | Média dos cinzas (0–255) | ± 15 |
| Tons de cinza | Se os três canais são iguais | igual |

> 💡 As medidas são feitas na **região da face** — o fundo (papel
> branco de uma cópia, fundo de documento) varia entre imagens e
> dominaria as frações sem dizer nada sobre a face. Sem a posição da
> face, a imagem inteira é medida e o painel avisa, porque as duas não
> são comparáveis.

> 💡 O **grão** depende da resolução: só compare com a mesma distância
> entre os olhos (por isso o *Redimensionar* final vem antes da medida).

### No Processar

Toda imagem adaptada é medida. O `manifest.json` e a entrada **CREATE**
da cadeia de custódia do dataset gerado registram:

- a mediana, o P5 e o P95 de cada medida no conjunto;
- quando o pipeline tem questionada: o nome e o SHA-256 dela e a
  comparação, medida a medida, da questionada com a **mediana da base
  adaptada** (dentro ou fora da tolerância).

A janela de conclusão mostra essa comparação com ✓ e ✗. É o que permite
escrever no laudo que a base de referência foi degradada até apresentar
as mesmas características mensuráveis da questionada — e com quais
números.

## Salvando e carregando

Pipelines são salvos como JSON em
`<paths.library>/image_pipelines/`. Cada step é serializado **plano**
(sem objeto `config` aninhado): a chave `step_type` guarda o tipo em
minúsculas e os parâmetros ficam no mesmo nível. O pipeline também
grava `description` e `input_source`:

```json
{
  "name": "whatsapp_simulation",
  "description": "",
  "input_source": "original",
  "seed": 42,
  "steps": [
    {"step_type": "blur", "kind": "gaussian", "kernel_size": 5, "sigma": 1.5},
    {"step_type": "jpeg", "quality": 65}
  ]
}
```

Use o botão **📂 Carregar** para carregar; **💾 Salvar** para
persistir. Pipelines aparecem automaticamente no dropdown do
diálogo *Processar*.

## O diálogo *Processar*

![Diálogo Processar (Aplicar Pipeline de Imagens)](assets/42_aplicar_image_pipeline.png)

Aplica um pipeline **a todas as imagens do dataset** num batch
único. Não tem preview — é "rode e espere".

### Campos

| Campo | Notas |
|---|---|
| **Pipeline** | Combo listando os nomes (stems dos `*.json` de `image_pipelines/`). Um rótulo abaixo mostra **Origem · Seed · Steps · Descrição** do pipeline selecionado |
| **Pasta** (**📁 Escolher…**) | Qualquer pasta. O dataset ganha uma pasta própria **dentro** dela. Default sugerido: a pasta que contém o dataset aberto |
| **Nome do dataset** | Nome do dataset derivado — também nome da pasta e do parquet. Default: `<dataset>__<pipeline>`. Espaços e acentos são mantidos |
| *Será criado:* | Mostra o caminho exato que será criado (`<pasta>/<nome>/`); avisa quando ele já existe e não está vazio |
| ☑️ **Pular imagens cujo PNG destino já existe** | Idempotente: pula quando o PNG de destino **já existe** (por existência do arquivo, não por comparação de hash). Default ON |
| **Após processar** | Radio: `Manter dataset atual aberto` (default) ou `Abrir o novo dataset (fecha o atual)` |

O botão de confirmação exibe **🌀 Processar**.

### Execução

Worker em thread separada. Diálogo de progresso mostra:
- Mensagem por face no formato `[47/234] …/foto_05.png` (throttle a cada 25 faces)
- Botão Cancelar

Cada face do dataset original é:
1. Lida do disco (`cv2.imdecode` com suporte a paths Unicode no Windows)
2. Passada pelo pipeline declarado
3. Escrita como **PNG lossless** em `images/`, na pasta do dataset

> ⚠️ **Cancelar não interrompe o lote no meio.** Não há ponto de
> cancelamento granular dentro do laço por face: o lote atual roda até
> o fim e o resultado é simplesmente **descartado**. Cancelar evita
> abrir/registrar o dataset derivado, mas os PNGs já gravados
> permanecem no disco.

## O dataset de saída

A operação cria um **novo dataset autocontido** (não modifica o atual),
no mesmo formato dos outros fluxos que criam datasets com imagens
próprias. Os PNGs **preservam o nome do arquivo de origem** (com
extensão trocada para `.png`), desambiguados por sufixos `_1`, `_2` em
caso de colisão:

```
<pasta escolhida>/<nome>/
├── <nome>.parquet              ← dataset derivado (caminhos relativos)
├── <nome>.audit.json           ← cadeia de custódia — nasce com CREATE
├── <nome>__errors.parquet      ← (só se houver falhas)
├── manifest.json               ← âncora forense (config do pipeline + contagens)
└── images/
    ├── foto_05.png             ← imagem processada (sempre PNG lossless)
    └── ...
```

> 💡 A pasta é uma **unidade**: pode ser movida, compactada ou levada
> para outra máquina inteira, e o dataset continua achando as imagens.

### Estrutura do parquet de saída

| Coluna | Conteúdo |
|---|---|
| `image_path` | Caminho do PNG, relativo à pasta `images/` |
| `hash` | SHA-256 do PNG novo |
| `file_name` | Nome do PNG |
| `identity` | Preservado do dataset original |
| `origin` / `ethnicity` / `gender` | Preservados do dataset original |
| Colunas de template e manuais (ex.: `idade`) | Preservadas, com o tipo declarado |

> ⚠️ **O que foi medido na imagem de origem é descartado** —
> embeddings, `ff_info`/`ofiq_info`/`serfiq_info`, pose, qualidade,
> geometria e metadados de captura (EXIF, qualidade JPEG estimada…). A
> imagem mudou, e essas medidas deixaram de descrevê-la. O que descreve
> a **pessoa** ou o **registro** (identidade, demografia, colunas de
> template) é preservado. O `manifest.json` e a cadeia de custódia
> listam as colunas levadas e as descartadas. Para ter embeddings e
> métricas de qualidade no dataset processado, rode **Enriquecer**
> ([capítulo 05](05_enriquecimento.md)) depois.

> ℹ️ **Face alinhada continua alinhada.** Se a origem já é um dataset
> de faces alinhadas 112×112 (ou o pipeline usa *Origem = Face
> alinhada*), o derivado é marcado como alinhado — **a não ser que o
> pipeline tenha um Recorte**, que tira a imagem do gabarito.

### Cadeia de custódia do derivado

O derivado nasce com uma entrada **CREATE** que registra o dataset de
origem e o SHA-256 do parquet dele, o pipeline completo, o SHA-256 do
`manifest.json`, as contagens e as colunas levadas/descartadas — e o
texto diz que as imagens foram transformadas de propósito. Quem abrir
o derivado sabe, pela própria cadeia, de onde ele veio. Se a origem
tinha alterações não salvas, isso também fica registrado (o hash
gravado descreve o arquivo, não o que estava em memória). O dataset de
origem continua recebendo a sua entrada `COPY_IMAGES`.

### Manifesto e parquet de erros

O `manifest.json` (na pasta do dataset) é a âncora forense da execução.
Registra: nome/seed/origem do pipeline, tipos e número de steps,
contagens (processadas / concluídas / puladas / falhas), timestamp de
início, duração, dataset-fonte, operador e o **config completo do
pipeline**. Ele **não** contém hashes por imagem — o SHA-256 de cada
PNG vive na coluna `hash` do parquet derivado.

Quando alguma face falha, é gravado `<nome>__errors.parquet` (na
pasta do dataset), com uma linha por falha e as colunas
`image_path` / `identity` / `file_name` / `hash` + `error_message` (a
razão da falha). Os paths apontam para a **imagem-fonte**, em caminho
absoluto (o PNG não chegou a ser gerado). Você pode abrir esse arquivo como um dataset
normal para inspecionar, filtrar e gerar uma lista de reanálise.

### Por que sempre PNG?

PNG é **lossless** — preserva exatamente o que o pipeline produziu.
Se o último step do pipeline foi JPEG(85), o PNG final contém os
artefatos do JPEG (que são reais e visíveis), mas o arquivo em si
não adiciona nenhuma perda extra. Garante reprodutibilidade exata.

### Determinismo

O *seed* do pipeline + o `image_path` da face de entrada formam
o seed efetivo para steps aleatórios (como **Oclusão** com posição
**Aleatória**), derivado por SHA-256. O mesmo pipeline produz o
**mesmo resultado bit a bit** em qualquer execução e em qualquer
máquina — inclusive nos passos aleatórios.

## Casos de uso forense

### Caso A — Simular degradação WhatsApp

Você tem fotos de boa qualidade do investigado e a evidência
veio do WhatsApp (resolução reduzida + recomprimida). Para
comparar embeddings em pé de igualdade:

```
Pipeline "whatsapp_typical":
  1. Recorte  modo=CENTERED (640 × 640)  ← simula crop pra caber no app
  2. Compressão JPEG  qualidade=60       ← compressão padrão do WhatsApp
```

Rode esse pipeline no dataset de referência. Agora ambos os
lados estão na mesma "escala de degradação" para a comparação.

> ⚠️ O modo `CENTERED` **não** faz padding automático: se a imagem for
> menor que o alvo (640 × 640 aqui), o step falha para aquela face (a
> falha vai para `<nome>__errors.parquet`). Ajuste o alvo à resolução
> das suas imagens.

### Caso B — Foco em região da face

A evidência mostra o rosto com a parte inferior coberta (ex.: máscara
facial). Para ocluir a região da boca, use posição **Sobre a boca**
(`MOUTH`) — que exige o FFService para localizar os landmarks:

```
Pipeline "mask_lower_face":
  1. Oclusão  tipo=Retângulo preto, posição=Sobre a boca,
              largura=0.6, altura=0.3
```

> 💡 Para um recorte exato da região a esconder, use o tipo **Máscara
> desenhada** e trace o polígono da faixa inferior do rosto no canvas —
> mais preciso que uma barra retangular e não depende de detecção de
> landmarks. Não existe posição `BOTTOM`/`TOP`: as posições geométricas
> disponíveis são Centralizada, Aleatória, Sobre os olhos e Sobre a boca.

### Caso E — Xerox de documento

A questionada é a cópia (xerox) de um documento com foto: contraste
estourado, sombras fechadas, grão de meio-tom, resolução baixa. A base
mais fiel para adaptar é a de **fotos de documento** (o original da
cópia era uma foto impressa no documento). A cadeia segue a ordem
física — tom da copiadora, toner, meio-tom, digitalização:

```
Pipeline "xerox_documento":
  1. Tons de cinza
  2. Curva de tons      preto=90, branco=160, gama=1,0   ← contraste da copiadora
  3. Redimensionar      IPD 80 px                        ← resolução em que ela "pontilha"
  4. Meio-tom           difusão de erro
  5. Redimensionar      IPD = o da questionada           ← digitalização
  6. Curva de tons      preto=30, branco=225, gama=1,0   ← contraste do scanner
```

Carregue a questionada no Editor (**📄 Questionada…**) e ajuste os
valores até as medidas da processada ficarem verdes. Os valores acima
vieram de um caso real (questionada com distância entre os olhos de
48,8 px): aplicados à base de fotos de documento, a mediana da base
adaptada ficou dentro da tolerância nas **8 medidas**. São ponto de
partida, não receita — cada cópia tem a sua copiadora e o seu scanner.
O *Ganho de toner* entra quando as sombras da questionada são mais
fechadas que as da base processada.

### Caso C — Investigar robustez do pipeline biométrico

Cenário de pesquisa: quero quantificar a queda de FIQA quando a
qualidade JPEG cai de 90 para 30. Crio 7 pipelines
(`jpeg_q90`, `jpeg_q80`, …, `jpeg_q30`), processo o mesmo
dataset com cada um, e gero 7 novos parquets. Junto via **Juntar**
([capítulo 07](07_operacoes_estruturais.md)) e analiso o efeito
em **Estatísticas** + **Auditoria**.

### Caso D — Anonimização para divulgação científica

Aplicar oclusões nos olhos pra disseminar o dataset sem expor
identidades:

```
Pipeline "anonymize_eyes":
  1. Oclusão  tipo=Círculo preto, posição=Sobre os olhos, raio=0.15
```

(Posição **Sobre os olhos** exige o FFService para localizar os
landmarks.)

> ⚠️ Anonimização por oclusão é uma técnica básica — não é
> garantia contra re-identificação automatizada. Use com critério.

---

[← Capítulo 07 — Operações estruturais](07_operacoes_estruturais.md) · [Capítulo 09 — Auditoria forense →](09_auditoria_forense.md)
