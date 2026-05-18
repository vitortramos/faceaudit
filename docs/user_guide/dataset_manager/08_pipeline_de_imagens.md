# 08 — Pipeline de imagens

A aba **Imagens** do ribbon expõe dois botões que trabalham juntos:

- 🎨 **Editor** — monta um *pipeline* de transformações de imagem
  interativamente, com pré-visualização e métricas de qualidade
- ▶ **Processar** — aplica um pipeline salvo a todo o dataset,
  gerando um **novo dataset** com as imagens transformadas

A motivação forense: simular **degradações realistas** (compressão
de mensageiro, oclusão, blur de zoom digital, etc.) sobre um
dataset de referência e verificar como o sistema de comparação
biométrica se comporta.

## Sumário do capítulo

1. [Conceito de *pipeline de imagens*](#conceito)
2. [O Editor — leiaute em três painéis](#o-editor)
3. [Os 5 tipos de step](#os-5-tipos-de-step)
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
| **Top — Lista** | Steps na ordem de aplicação. Cada linha mostra `Nº  Tipo`. Drag-drop pra reordenar. Botão ✕ vermelho remove |
| **Bottom — Configuração** | Formulário dinâmico que reage ao step selecionado — campos variam conforme o tipo |

Botão verde **➕ Adicionar** no topo abre um menu com os 5 tipos
disponíveis.

### Painel direito superior — Image Compare (65%)

| Elemento | Função |
|---|---|
| **Imagem original** (480×480 esquerda) | Imagem de entrada, sem transformação |
| **Imagem processada** (480×480 direita) | Resultado após aplicar o pipeline atual |
| ◀ ▶ 🎲 | Navegação entre faces do dataset (anterior / próxima / aleatória) |
| Status label | `face 17 / 234 — n000009/foto_05.png` |
| **▶ Apply** | Reaplica o pipeline na face atual (após alterar steps/parâmetros) |

### Painel direito inferior — Quality Scores

![Quality scores panel (FIQA, embedding norm, cosine similarity)](assets/41_image_pipeline_editor.png)

Tabela 3×4 que **atualiza em tempo real** após cada *Apply*:

| Métrica | Original | Processada | Δ |
|---|---|---|---|
| FIQA | 0.84 | 0.62 | ↓ -0.22 *(vermelho)* |
| ‖embedding‖ | 22.4 | 19.8 | ↓ -2.6 |
| Cosine Similarity | — | — | 0.91 |

O **delta** ganha cor: 🔴 vermelho (degradação > 0.1), 🟢 verde
(melhora), ⚪ cinza (próximo de zero). O cálculo roda num *worker*
separado pra não travar a UI.

### Topo — controles globais

| Controle | Função |
|---|---|
| **Nome** | Identificador do pipeline (vira o stem do JSON salvo) |
| **Input source** | Fonte da imagem original: `Foto original` ou `Face alinhada 112×112` |
| **Seed** | Inteiro pra steps aleatórios (`OCCLUSION`, etc.) — torna o pipeline reproduzível |
| **📂 Abrir** | Carregar pipeline salvo da biblioteca |
| **💾 Salvar** | Salva no caminho atual ou abre Salvar Como |

## Os 5 tipos de step

| Tipo | Ícone | O que faz |
|---|---|---|
| **BLUR** | 🌫 | Aplica filtro Gaussiano. Simula desfoque por movimento, falha de foco, ou degradação de zoom |
| **JPEG** | 🗜 | Codifica + decodifica em JPEG. Simula compressão de mensageiros, e-mail, redes sociais |
| **WEBP** | 🗜 | Como JPEG mas no formato WebP (compressão moderna, melhor qualidade pelo mesmo bitrate) |
| **CROP** | ✂️ | Recorta a imagem. Modos centralizado (fixo) ou detectado por face (com padding) |
| **OCCLUSION** | ⬛ | Adiciona oclusão (retângulo preto, círculo preto, ou máscara desenhada manualmente). Simula obstruções parciais |

> 💡 Os 5 cobrem os cenários mais comuns de degradação realista.
> A intenção do MVP é cobrir o caso pericial *"reproduza no
> dataset de referência o que aconteceu na evidência"* — se a foto
> do caso veio do WhatsApp, comprima a referência com JPEG; se a
> identificação só mostra parte do rosto, oclua a referência.

## Configuração detalhada

### BLUR — Borrão Gaussiano

| Campo | Faixa | Padrão | Notas |
|---|---|---|---|
| **Kind** | Combo (só GAUSSIAN no MVP) | GAUSSIAN | Outros tipos (motion blur, defocus) podem vir em versões futuras |
| **Kernel size** | Ímpar entre 1 e 99 | 5 | Tamanho do kernel; quanto maior, mais alcance |
| **Sigma** | 0.1 – 50.0 | 1.0 | Desvio padrão da gaussiana; quanto maior, mais borrão |

### JPEG — Compressão JPEG

| Campo | Faixa | Padrão | Notas |
|---|---|---|---|
| **Quality** | 1 – 100 | 85 | Qualidade JPEG. Valores < 20 mostram artefatos pesados; > 95 são quase imperceptíveis |

### WEBP — Compressão WebP

| Campo | Faixa | Padrão | Notas |
|---|---|---|---|
| **Quality** | 1 – 100 | 80 | WebP geralmente preserva melhor que JPEG no mesmo valor |

### CROP — Recorte

| Campo | Faixa | Padrão | Notas |
|---|---|---|---|
| **Mode** | `CENTERED` / `FACE_DETECTED` | CENTERED | |
| *(CENTERED)* **Target width** | ≥ 1 | 224 | Largura do recorte em px |
| *(CENTERED)* **Target height** | ≥ 1 | 224 | Altura do recorte em px |
| *(FACE_DETECTED)* **Face padding** | ≥ 0.0 | 0.0 | Fração do bbox detectado a adicionar como margem (0.2 = 20% além) |

> 💡 No modo `FACE_DETECTED`, o ForensicFace é chamado pra detectar
> a maior face, e o recorte é feito em volta do bbox dela (com
> padding opcional). Útil pra **extrair só a face** preservando a
> pose original.

### OCCLUSION — Oclusão

| Campo | Faixa | Padrão | Notas |
|---|---|---|---|
| **Kind** | `MASK_FACE` / `RECTANGLE_BLACK` / `CIRCLE_BLACK` | MASK_FACE | |
| **Position** | `CENTER` / `RANDOM` / `EYES` / `MOUTH` | CENTER | Varia conforme `kind`. `EYES` e `MOUTH` só para `CIRCLE_BLACK` |
| *(RECTANGLE)* **Width %** | 0.0 – 1.0 | 0.5 | Fração da largura ocluída |
| *(RECTANGLE)* **Height %** | 0.0 – 1.0 | 0.5 | Fração da altura ocluída |
| *(CIRCLE)* **Radius %** | 0.0 – 1.0 | 0.10 | Raio como fração do menor lado |
| *(MASK_FACE)* **mask** | Polígonos desenhados | (vazio) | Editor de máscara interativo |

> 💡 `MASK_FACE` é o mais flexível — abre um canvas onde você
> desenha **polígonos arbitrários** sobre a imagem. Útil pra
> simular obstruções específicas (faixa de evidência, perda de
> resolução em região específica).

## Métricas ao vivo

O painel inferior direito (*Quality Scores*) é **opcional** —
você pode usar o editor só pelo preview visual. Mas habilitar as
métricas dá insights forenses precisos:

| Métrica | O que indica |
|---|---|
| **FIQA** | Score 0–1 de qualidade da face. Cair > 0.1 indica degradação que afeta reconhecimento |
| **‖embedding‖** | Magnitude do vetor de embedding. Valores muito baixos indicam *embedding collapse* (a transformação destruiu informação demais) |
| **Cosine Similarity** | Cosseno entre embeddings de original e processada. 1.0 = idêntico; valores baixos significam que o pipeline mudou muito a "assinatura" biométrica |

### Recomputação

As métricas são recalculadas **somente quando você clica ▶ Apply**.
Mudar steps/parâmetros não dispara recálculo automático — você
controla o momento (evita cálculos desnecessários durante
edição).

O cálculo roda num *worker* assíncrono (~80–300 ms em CPU para
faces alinhadas 112×112; ~240–600 ms para originais com detect).
A UI continua responsiva durante o cálculo; um indicador
*"Computing..."* aparece nas células.

## Salvando e carregando

Pipelines são salvos como JSON em
`<paths.library>/image_pipelines/`:

```json
{
  "name": "whatsapp_simulation",
  "seed": 42,
  "steps": [
    {"type": "BLUR", "config": {"kind": "GAUSSIAN", "kernel_size": 5, "sigma": 1.5}},
    {"type": "JPEG", "config": {"quality": 65}}
  ]
}
```

Use o botão **📂 Abrir** para carregar; **💾 Salvar** para
persistir. Pipelines aparecem automaticamente no dropdown do
diálogo *Processar*.

## O diálogo *Processar*

![Diálogo Processar (Aplicar Pipeline de Imagens)](assets/42_aplicar_image_pipeline.png)

Aplica um pipeline **a todas as imagens do dataset** num batch
único. Não tem preview — é "rode e espere".

### Campos

| Campo | Notas |
|---|---|
| **Pipeline** | Combo populado com todos os `*.json` de `image_pipelines/`. Mostra nome + origem + seed + número de steps |
| **Pasta destino** | Onde os PNGs de saída serão escritos. Default sugerido: `<parquet_dir>/<dataset>__<pipeline>/` |
| **Nome de saída** | Stem do parquet de saída. Default: `<dataset>__<pipeline>` |
| ☑️ **Skip existing** | Se o PNG já existe com hash idêntico, pula. Default ON (idempotente) |
| **Após o processamento** | Radio: `Manter dataset atual aberto` (default) ou `Abrir o novo dataset` |

### Execução

Worker em thread separada. Diálogo de progresso mostra:
- "Processando 47/234 (20%)"
- Botão Cancelar (cancelamento cooperativo)

Cada face do dataset original é:
1. Lida do disco (`cv2.imdecode` com suporte a paths Unicode no Windows)
2. Passada pelo pipeline declarado
3. Escrita como **PNG lossless** na pasta destino

## O dataset de saída

A operação cria um **novo dataset** (não modifica o atual):

```
<pasta_destino>/
├── face_001.png            ← imagem processada (sempre PNG lossless)
├── face_002.png
├── ...
└── manifest.json           ← âncora forense (timestamps, hashes, versão do pipeline)

<dataset_output_dir>/
├── <dataset>__<pipeline>.parquet
└── <dataset>__<pipeline>__errors.parquet   ← (só se houver falhas)
```

### Estrutura do parquet de saída

| Coluna | Conteúdo |
|---|---|
| `image_path` | Caminho absoluto pra o PNG processado |
| `hash` | SHA-256 do PNG novo |
| `file_name` | Nome do PNG |
| `identity` | Preservado do dataset original |

> ⚠️ **O dataset de saída é minimalista** — só essas 4 colunas.
> Sem embeddings, sem qualidade, sem demografia. Se você quiser
> métricas no dataset processado, rode **Enriquecer**
> ([capítulo 05](05_enriquecimento.md)) depois.

### Por que sempre PNG?

PNG é **lossless** — preserva exatamente o que o pipeline produziu.
Se o último step do pipeline foi JPEG(85), o PNG final contém os
artefatos do JPEG (que são reais e visíveis), mas o arquivo em si
não adiciona nenhuma perda extra. Garante reprodutibilidade exata.

### Determinismo

O *seed* do pipeline + o `image_path` da face de entrada formam
o seed efetivo para steps aleatórios (como `OCCLUSION` com
`position=RANDOM`). Resultado: **rodar o mesmo pipeline duas
vezes no mesmo dataset produz outputs idênticos** — bit a bit.

## Casos de uso forense

### Caso A — Simular degradação WhatsApp

Você tem fotos de boa qualidade do investigado e a evidência
veio do WhatsApp (resolução reduzida + recomprimida). Para
comparar embeddings em pé de igualdade:

```
Pipeline "whatsapp_typical":
  1. CROP centered (640 × 640)      ← simula crop pra caber no app
  2. JPEG quality=60                ← compressão padrão do WhatsApp
```

Rode esse pipeline no dataset de referência. Agora ambos os
lados estão na mesma "escala de degradação" para a comparação.

### Caso B — Foco em região da face

A evidência mostra só a parte superior do rosto (oclusão por
máscara). Replica isso:

```
Pipeline "mask_lower_face":
  1. OCCLUSION kind=RECTANGLE_BLACK, position=BOTTOM, height=0.5
```

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
  1. OCCLUSION kind=CIRCLE_BLACK, position=EYES, radius=0.15
```

> ⚠️ Anonimização por oclusão é uma técnica básica — não é
> garantia contra re-identificação automatizada. Use com critério.

---

[← Capítulo 07 — Operações estruturais](07_operacoes_estruturais.md) · [Capítulo 09 — Auditoria forense →](09_auditoria_forense.md)
