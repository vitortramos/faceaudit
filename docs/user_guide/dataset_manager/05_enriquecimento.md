# 05 — Enriquecimento

**Enriquecer** = adicionar (ou recalcular) atributos biométricos de
um dataset *já existente*. Diferente do *Novo Dataset* (que parte de
imagens cruas), o Enriquecer trabalha sobre o parquet em memória —
ideal para:

- Completar um dataset parcialmente enriquecido (faltou rodar OFIQ,
  FIQA ou extrair os metadados EXIF/XMP, por exemplo)
- Adicionar um **segundo embedding** (`embedding_v4` numa base que
  já tinha só `embedding_v2`)
- Retentar faces que falharam (`error=True`)
- Corrigir metadados errados (`is_aligned` marcado incorretamente)

## Sumário do capítulo

1. [Quando usar Enriquecer vs Novo Dataset](#quando-usar-enriquecer)
2. [O diálogo *Enriquecer Dataset* — visão geral](#o-diálogo-enriquecer-dataset)
3. [Selos de status e auto-check](#selos-de-status-e-auto-check)
4. [Configurações ForensicFace e conteúdo a adicionar](#grupo-1--configurações-forensicface)
5. [`is_aligned` — correção de metadados](#is_aligned--correção-de-metadados)
6. [Idempotência — `Reprocessar mesmo as faces já enriquecidas`](#idempotência)
7. [Cenários práticos](#cenários-práticos)
8. [O que muda no parquet](#o-que-muda-no-parquet)
9. [Cancelamento e progresso](#cancelamento-e-progresso)
10. [Erros e diagnóstico](#erros-e-diagnóstico)

---

## Quando usar Enriquecer

| Caso | Use… |
|---|---|
| Tenho uma pasta de imagens, quero criar um dataset | [Novo Dataset](02_criando_dataset.md) |
| Tenho um dataset, quero **adicionar** uma métrica que não está nele | **Enriquecer** |
| Tenho um dataset, quero **substituir** todas as métricas (porque mudei parâmetros) | **Enriquecer** com *Reprocessar* marcado |
| Algumas faces falharam, quero retentar só elas | **Enriquecer** sem *Reprocessar* |
| O parquet veio com `is_aligned` errado | **Enriquecer** (qualquer extractor; a flag é ajustada) |
| Quero corrigir caminhos de imagem | [Modificar Caminhos](03_carregando_e_localizando.md), **não** Enriquecer |

## O diálogo *Enriquecer Dataset*

Ao clicar em 💎 **Enriquecer** abre-se o diálogo **"Enriquecer Dataset"**:

![Diálogo Enriquecer Dataset, com selos de status por item](assets/32_enriquecer.png)

Diferente do *Novo Dataset*, não há a parte de ingestão (pasta, template).
O dataset já está em memória, então o pipeline só executa *Transform* e
*Load*. O layout tem:

| Bloco | Função |
|---|---|
| **Texto de informação** (topo) | Resumo: *"N face(s) · fotos originais. Os selos mostram o que já existe; marque o que deseja adicionar."* |
| **1 · Configurações ForensicFace** | GPU, *batch size*, *det size* + o override *Imagens são faces já alinhadas 112×112* |
| **2 · Conteúdo a adicionar** | Embeddings / Qualidade / Pose / Outros — cada item com um **selo de status** — e Metadados (EXIF/XMP) |
| **Reprocessar mesmo as faces já enriquecidas** | Toggle global de idempotência |
| **Iniciar processamento** | Botão que dispara o enriquecimento |

## Selos de status e auto-check

Quando o diálogo abre, o controller varre o dataset e calcula a
**cobertura atual de cada item** (por modelo de embedding, por atributo).
Ao lado de cada checkbox aparece um **selo colorido**:

| Selo | Cor | Cobertura |
|---|---|---|
| **presente** | verde | ≥ 99,9% das faces já têm |
| **parcial N%** | âmbar | entre 0,1% e 99,9% (N = % que já tem) |
| **ausente** | vermelho | ≤ 0,1% |

**Auto-check conservador** — o sistema pré-marca **só os itens PARCIAIS**
(para completar um dado inconsistente). Itens **presentes** e itens
**totalmente ausentes** ficam **desmarcados**: o selo *ausente* avisa,
mas o sistema **não presume** que você quer adicionar tudo que falta —
marcar é opt-in. Desmarcar um item o **remove** do processamento.

> 💡 Em datasets recém-criados com todo o conteúdo ligado, todos os selos
> ficam verdes (**presente**) e nada vem pré-marcado — o enriquecimento é
> supérfluo. Use o diálogo para *adicionar coisas novas* (um segundo
> modelo, um atributo que faltou).

## Grupo 1 — Configurações ForensicFace

| Controle | Faixa | Padrão | Notas |
|---|---|---|---|
| ☑️ **Usar GPU** | — | do Sistema | Override só deste enriquecimento (não grava no global). |
| **Batch size:** | 1–512 | do Sistema | Faces por chamada do modelo. |
| **Det size:** | 160–1280 (passo 32) | 320 | Resolução do detector. Só afeta fotos originais — some em faces alinhadas. |
| ☑️ **Imagens são faces já alinhadas 112×112** | — | herdado do dataset | Interruptor mestre do *esconde-se-alinhado* + correção de metadados (veja abaixo). |

## Grupo 2 — Conteúdo a adicionar

Mesma grade por tipo de coluna do *Novo Dataset*, mas **cada item traz o
seu selo de status**:

| Categoria | Itens (cada um com selo) |
|---|---|
| **Embeddings:** | Um checkbox **por modelo instalado** (`Sepael V2`..`V6`). |
| **Qualidade:** | **FIQA** · **OFIQ (qualidade ISO)** (exige a foto original; some em faces alinhadas) |
| **Pose:** | **Yaw**, **Pitch**, **Roll** (cobertura independente por item; somem em faces alinhadas) |
| **Outros:** | **Det Conf**, **IPD**, **Gênero**, **Idade** (somem em faces alinhadas) |
| **Metadados:** | **Extrair EXIF/XMP da imagem** (sem selo) — lê os metadados embutidos no arquivo; só faz sentido em foto original (em faces alinhadas a opção é ignorada) |

> 💡 Adicionar **pose/gênero/idade** a faces que **já têm embedding** roda
> só o **detector** — não recalcula o embedding (rápido). **FIQA** exige o
> passo de reconhecimento.

> 💡 **Adicionar um modelo novo é o caso mais comum.** Se o parquet só tem
> `embedding_v2` e você quer `embedding_v4`: marque `Sepael V4` (selo
> *ausente*), deixe `Sepael V2` desmarcado (selo *presente*) e
> *Reprocessar* desmarcado → só as faces sem `v4` são processadas.

> 💡 A antiga opção **SER-FIQ** foi **removida da interface** (backend
> mantido). Para qualidade de imagem use **FIQA** ou **OFIQ**.

## `is_aligned` — correção de metadados

A checkbox **"Imagens são faces já alinhadas 112×112"** no grupo
*1 · Configurações ForensicFace* é especial:

- **Pré-preenchida** com o valor atual de `dataset.is_aligned`
- Quando marcada, o ForensicFace manda a imagem direto ao encoder —
  pula a etapa de detect+align (~50% mais rápido)
- O valor que você deixar **é persistido** na metadata do parquet
  ao final do enriquecimento

> 💡 Use a flag para **corrigir** datasets antigos que foram criados
> antes do FaceAudit ter esse atributo, ou que herdaram um valor
> errado de uma migração. Marque/desmarque conforme a realidade das
> imagens e rode qualquer enriquecimento — o flag é gravado.

> ⚠️ **Não marque** essa flag se as imagens **não** são crops
> alinhados 112×112 — o detect+align será pulado e o embedding sai
> ruim sem aviso (face descentralizada). Em caso de dúvida, abra
> uma imagem da pasta no visualizador do SO; se ela tem fundo,
> ombros, etc., **não** está alinhada.

## Idempotência

A checkbox **"Reprocessar mesmo as faces já enriquecidas"** (abaixo
do grupo *2 · Conteúdo a adicionar*, logo acima dos botões) tem o
seguinte comportamento:

| Configuração | Comportamento |
|---|---|
| ☐ **Desmarcado** *(padrão)* | Cada extractor filtra as faces que **já têm** o resultado e processa só as que estão faltando. Faces com `error=True` que já têm o resultado pedido **não são retentadas** |
| ☑️ **Marcado** | Reprocessa **todas as faces**, sobrescrevendo resultados anteriores. Use quando mudou parâmetros (det_size diferente, T diferente, etc.) |

### Idempotência por modelo (Forensic Face)

A filtragem do FF é mais granular: ele verifica não só se `ff_info`
existe, mas também se o **modelo específico** e cada **atributo**
pedido estão populados:

```
Face tem ff_info com embedding_sepaelv2 mas não sepaelv4
+ usuário marcou sepaelv4 no diálogo
→ Face é incluída no processamento (FF preencherá só sepaelv4)

Face já tem todos os embeddings pedidos, mas falta pose/gênero/idade/
Det Conf/IPD pedido
→ roda só o detector, sem recalcular o embedding
```

Isso significa: você pode rodar enriquecimento progressivamente
adicionando modelos, sem desperdiçar GPU re-extraindo o que já
tem.

### Retentar faces falhadas

A coluna `error=True` é **independente** da checagem de
idempotência: o teste é sobre o dado pedido (o embedding do modelo
marcado, o FIQA, o atributo, o `*_info` do OFIQ), não sobre `error`.
Uma face que falhou e continua sem o dado pedido **é reprocessada**
mesmo com a flag desmarcada; uma que já tem o dado pedido só é
reprocessada com `Reprocessar` marcado.

### Faces que continuam sem biometria

Se, ao fim do estágio ForensicFace, alguma face continuar sem
embedding, abre-se a janela *Faces sem biometria neste
enriquecimento* (não acontece em datasets de faces já alinhadas). Ela
**não** decide inclusão — a face já está no dataset e continua nele;
para removê-la use **Excluir Selecionados**, que exige justificativa.
O que a janela oferece é inspecionar as imagens e repetir a detecção
numa resolução maior (640, 960 ou 1280 px). As recuperadas ganham a
biometria e passam pelos estágios que já tinham rodado antes da
gravação.

> 💡 Em datasets de **recortes de vídeo** que gravaram a geometria da
> extração, o enriquecimento realinha a face pelos pontos gravados em
> vez de detectar de novo no recorte (onde a detecção falha em face
> pequena), e avisa quantas faces têm parte da face fora do recorte.
> Veja o [capítulo 02b](02b_criando_dataset_de_video.md).

## Cenários práticos

### Cenário A — Adicionar `embedding_v4` a um dataset com `embedding_v2`

```
Selos ao abrir:
  Embeddings: Sepael V2  → presente
              Sepael V4  → ausente
  FIQA:                  → presente
```

**Ações:**
1. Embeddings → ☑️ `Sepael V4` (o `Sepael V2` fica desmarcado — já tem)
2. ☐ Reprocessar (padrão)
3. **Iniciar processamento**

**Resultado:** Todas as faces ganham `embedding_v4`, sem reprocessar
`embedding_v2`/FIQA. Custo de GPU: ~1× o tempo de extrair `v4`.

### Cenário B — Adicionar OFIQ a um dataset que só tem embeddings

```
Selos ao abrir:
  Embeddings: Sepael V2  → presente
  OFIQ:                  → ausente
```

**Ações:**
1. Qualidade → ☑️ **OFIQ (qualidade ISO)** (deixe os embeddings desmarcados)
2. **Iniciar processamento**

**Resultado:** Cada face roda OFIQ; as colunas `ofiq_*` aparecem.
~2s/face em CPU (single-threaded — o binário externo é o gargalo).

### Cenário C — Retentar faces que falharam

```
Dataset com 1000 faces, das quais 47 estão com error=True
porque a detecção falhou em Det size = 320.
```

**Ações:**
1. Embeddings → ☑️ o modelo que falhou
2. **Configurações ForensicFace** → **Det size = 640** *(aumenta resolução)*
3. ☑️ **Reprocessar mesmo as faces já enriquecidas**
4. **Iniciar processamento**

**Resultado:** As 1000 faces reprocessadas com *Det size* maior.
Algumas das 47 antes falhadas agora detectam. A coluna `error` é
recalculada — `error=True` permanece só nas que ainda falharem.

> 💡 Sem *Reprocessar*, só as 47 faces sem embedding vão ao detector, e
> as que continuarem sem face aparecem na janela *Faces sem biometria
> neste enriquecimento*, onde dá para tentar 640/960/1280 só nelas.

> 💡 Outra alternativa: filtre o dataset por `error=True`
> ([capítulo 06](06_filtros_e_estatisticas.md)), salve como
> dataset separado, rode Enriquecer nele, depois mescle de volta
> com *Juntar* ([capítulo 07](07_operacoes_estruturais.md)).

### Cenário D — Corrigir `is_aligned`

```
Dataset antigo marcou is_aligned=False, mas as imagens são
crops 112×112 — o ForensicFace roda detect+align sem necessidade.
```

**Ações:**
1. Embeddings → ☑️ o modelo desejado
2. **Configurações ForensicFace** → ☑️ **Imagens são faces já alinhadas 112×112**
3. ☑️ Reprocessar
4. **Iniciar processamento**

**Resultado:** Os embeddings são recalculados pulando detect+align
(mais rápido e mais preciso pra crops). O atributo
`dataset.is_aligned=True` é persistido. Próxima abertura do parquet
já vem com a flag correta.

## O que muda no parquet

Após o enriquecimento, o parquet é **regravado na pasta de onde foi
aberto**, como `{nome do dataset}.parquet` (sobrescreve o arquivo
original — não cria backup automático).

| Tipo de mudança | Detalhes |
|---|---|
| Colunas FF preenchidas/atualizadas | `embedding_v2`/`embedding_v4`, `fiqa_score`, `det_score`, `gender_est`, `age_est`, `yaw`/`pitch`/`roll`, `ipd`, `bbox`, `keypoints` |
| Colunas OFIQ preenchidas/atualizadas | ~40 colunas `ofiq_*` (scalar + raw), mais `ofiq_version` e `ofiq_backend` (de qual OFIQ veio cada medida) |
| Colunas de captura (se *Extrair EXIF/XMP* marcado) | data/hora (`capture_datetime_raw`, `capture_datetime`…), câmera, exposição, compressão e o dump completo em `exif_json`, exibido na aba **Metadados** |
| Colunas SER-FIQ preenchidas/atualizadas | `serfiq_score`, `serfiq_t`, `serfiq_alpha` |
| Colunas de erro recalculadas | `error`, `error_stage`, `error_message` |
| Metadados do dataset | `is_aligned` persistido se houve override |

Identidades, gênero, etnia, origem, caminho, hash e qualquer
coluna *custom* **não** são tocadas pelo enriquecimento — ele só
atua nos slots biométricos.

> ⚠️ **Sem backup automático.** Se você está com receio (mudou
> parâmetros, dataset crítico, etc.), use **Salvar Como** *antes*
> de enriquecer pra ter cópia do estado anterior.

## Cancelamento e progresso

O enriquecimento roda num *worker* em thread separada. Um diálogo
de progresso mostra:

- **Etapa atual** (ex: "Forensic Face — face 234/1000")
- **Porcentagem** *(escala global: 0–100%)*
- **Botão Cancelar**

Ao **cancelar**, o sistema:
1. Sinaliza interrupção ao pipeline
2. Espera a face atual terminar (não corta no meio)
3. **Salva o parquet parcial** antes de abortar (sem abrir a revisão
   das faces sem biometria)
4. Mostra mensagem informando que o resultado é parcial

Você pode retomar onde parou apenas rodando *Enriquecer* novamente
com a mesma configuração — a idempotência cobre o que já foi
processado.

## Erros e diagnóstico

| Sintoma | Causa provável | Solução |
|---|---|---|
| **Iniciar processamento** sem efeito | Nenhum item marcado | Marque ao menos um embedding/atributo |
| `Erro ao salvar pós-enriquecimento` | Parquet aberto em outro programa, ou disco read-only | Feche o arquivo em outro lugar; verifique permissões |
| `OFIQ não produziu resultado` | Binário OFIQ não acessível ou pasta `data/` ausente | Confira `config.json → ofiq.ofiq_exe_path` e `ofiq_data_path` |
| GPU OOM (Out of Memory) | **Det size** alto + dois modelos simultâneos | Marque só um modelo, baixe **Det size** ou o **Batch size** |
| `nenhum embedding gerado para …` em várias faces | Detecção falhou (rosto pequeno demais ou marcado como já alinhado erroneamente) | Tente de novo em resolução maior na janela *Faces sem biometria*, aumente **Det size**, ou verifique o override *Imagens são faces já alinhadas* |
| `0/N faces — FALHA TOTAL` | Pipeline configurado errado (modelo wrong path, GPU off, etc.) | Cheque logs do worker no terminal — costuma ter detalhes |
| Dataset com `is_aligned` virou `True` sozinho | Você marcou a checkbox sem perceber | Re-rode o Enriquecer com a flag desmarcada e Reprocessar marcado |

---

[← Capítulo 04 — Navegando no dataset](04_navegando_dataset.md) · [Capítulo 06 — Filtros e estatísticas →](06_filtros_e_estatisticas.md)
