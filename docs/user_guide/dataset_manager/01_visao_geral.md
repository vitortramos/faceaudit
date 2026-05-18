# 01 — Visão geral

## O que é o Gerenciador de Datasets

O **Gerenciador de Datasets** é o primeiro dos dois módulos principais
do FaceAudit (o outro é o **Gerenciador de Estratégias**). Sua função
é transformar uma pasta de imagens faciais em um **dataset
estruturado** — um arquivo `.parquet` contendo, para cada imagem,
todas as informações necessárias aos cálculos forenses:

- **Embeddings biométricos** (vetores numéricos) extraídos por modelos
  SEPAEL V2 e/ou SEPAEL V4 via biblioteca *ForensicFace*.
- **Métricas de qualidade** — FIQA (do próprio ForensicFace), OFIQ
  (ISO/IEC 29794-5) e SER-FIQ (Stochastic Embedding Robustness).
- **Pose facial** estimada (yaw, pitch, roll).
- **Metadados administrativos** — identidade, gênero, etnia, origem,
  hash SHA-256 de cada imagem, caminho relativo no disco, dimensões.

Quando o dataset está pronto, ele é **enviado** ao Gerenciador de
Estratégias, onde participa de uma ou mais estratégias de cálculo de
razão de verossimilhança (LR).

> 💡 **Regra prática:** se a sua dúvida é *“como produzir embeddings a
> partir das minhas imagens?”* ou *“como conferir a qualidade do que
> tenho em mãos?”*, este é o módulo certo. Se for *“como comparar
> imagens A e B e obter um LR?”*, vá para o Gerenciador de Estratégias.

## Quando aparece a aba

A aba é exibida automaticamente na inicialização do FaceAudit, desde
que a biblioteca `forensicface` esteja instalada e os modelos SEPAEL
estejam acessíveis. Em ambientes onde estas dependências não estejam
disponíveis (apenas leitura de datasets já prontos, por exemplo), a
aba é silenciosamente desativada — o app sobe direto no Gerenciador
de Estratégias.

## Leiaute geral

Quando vazia (sem dataset carregado), a aba mostra a toolbar superior e
duas áreas em branco abaixo:

![Aba do Gerenciador de Datasets, sem dataset carregado](assets/10_dataset_manager_empty.png)

Os botões da toolbar que dependem de um dataset estão **desabilitados
em cinza**, indicando que é necessário primeiro **Abrir** ou criar
(**Novo**) um dataset.

Com um dataset carregado, o leiaute fica:

![Dataset carregado, primeira linha selecionada, com thumbnail no painel direito](assets/11_dataset_loaded_with_image.png)

A janela divide-se em três blocos:

| Bloco | Posição | Conteúdo |
|---|---|---|
| **Toolbar (Ribbon)** | Faixa superior, ~130 px de altura | 4 abas (Arquivo / Dataset / Exibir / Imagens) + barra de status |
| **Tabela** | Painel esquerdo, ~70 % da largura | Uma linha por imagem; colunas configuráveis |
| **Painel de detalhes** | Painel direito, ~30 % da largura | Thumbnail + abas *Geral / Bio* e *Qualidade ISO* |

O divisor entre tabela e detalhes é arrastável; a tabela tem
*stretch-factor* maior para que ela ocupe a maior parte do espaço quando
a janela é maximizada.

## A toolbar em formato ribbon

O FaceAudit adota o estilo **ribbon** do Microsoft Office para a barra
de ferramentas do Gerenciador de Datasets — os comandos são
agrupados em **4 abas**:

### Aba *Arquivo*

![Ribbon Arquivo: Novo, Abrir, Salvar, Salvar Como, Biblioteca, → Estratégia, Fechar](assets/20_ribbon_0_arquivo.png)

Operações de **ciclo de vida** do dataset:

| Botão | Função |
|---|---|
| ➕ **Novo** | Cria um dataset a partir de uma pasta de imagens (ver [capítulo 02](02_criando_dataset.md)) |
| 📂 **Abrir** | Lê um `.parquet` existente do disco |
| 💾 **Salvar** | Sobrescreve o arquivo carregado |
| 💾 **Salvar Como…** | Salva em um caminho à escolha |
| 📚 **Biblioteca** | Copia o dataset para a biblioteca global (acessível a todos os projetos) |
| ➡ **→ Estratégia** | Envia o dataset ao Gerenciador de Estratégias |
| ❌ **Fechar** | Descarta o dataset atual |

### Aba *Dataset*

![Ribbon Dataset: Auditar, Filtrar, Localizar, Enriquecer, Coluna, Metadados, Juntar, Dividir, Balancear, Comparar, Estatísticas, Custódia](assets/20_ribbon_1_dataset.png)

Operações que **agem sobre o conteúdo**:

| Grupo | Botões | O que fazem |
|---|---|---|
| **Auditoria** | 🔎 *Auditar* | Análise de integridade biométrica + visualizações forenses |
| **Filtragem** | 🔍 *Filtrar* / 🗑️ *Limpar Filtro* | Subconjuntos por expressão; o filtro vira a visão ativa da tabela |
| **Localização** | 🗂 *Localizar* | Corrige `image_path` quebrado e/ou empacota o dataset em pasta portátil ([capítulo 03](03_carregando_e_localizando.md)) |
| **Enriquecimento** | 💎 *Enriquecer* / ➕📋 *Coluna* / 🏷️ *Metadados* | Adicionar/recalcular FIQA/OFIQ/SER-FIQ; coluna custom; editar rótulos e vocabulários |
| **Estrutural** | 🔗 *Juntar* / ✂️ *Dividir* / ⚖ *Balancear* | Concatenar; particionar por identidade ou índice; balancear amostras |
| **Análise** | ⚖️ *Comparar* / 📊 *Estatísticas* | Comparação par-a-par com outro dataset; estatísticas agregadas |
| **Custódia** | 📜 *Custódia* | Cadeia de custódia forense — log de operações com integridade SHA-256 |

### Aba *Exibir*

![Ribbon Exibir: Galeria, Colunas](assets/20_ribbon_2_exibir.png)

Controla **como o dataset é apresentado** — sem alterar o conteúdo:

| Botão | Função |
|---|---|
| 🖼 **Galeria** | Alterna entre tabela e galeria de thumbnails |
| 📋 **Colunas** | Abre o *Relatório de Esquema*: cobertura, NaNs, mín/máx por coluna; exporta para CSV/Excel |

### Aba *Imagens*

![Ribbon Imagens: Editor, Processar](assets/20_ribbon_3_imagens.png)

Pipelines de **processamento de imagem** (rotação, recorte, watermark,
alinhamento, etc.) que podem ser aplicados em lote a todas as imagens
do dataset:

| Botão | Função |
|---|---|
| 🎨 **Editor** | Monta o pipeline interativamente com pré-visualização |
| ▶ **Processar** | Aplica o pipeline a todas as imagens, gerando novos arquivos e atualizando os caminhos |

## A barra de status

Logo abaixo do ribbon, uma barra fina exibe o **status** do dataset
carregado: número de faces, número visível após o filtro atual, hash
SHA-256 do parquet, presença de erros. Passe o mouse para ver um
tooltip rich-text com metadados extras (caminho completo, descrição
do dataset, cobertura por métrica de qualidade).

Quando um filtro está ativo, o botão **Filtrar** muda para a cor
amarela e **Limpar Filtro** é habilitado.

## A tabela

A tabela do lado esquerdo é uma `QTableView` ligada a um modelo
declarativo (`DatasetTableModel`) que se adapta à estrutura do
dataset: cada coluna tem **rótulo** legível, **role** (identidade,
metadado ou derivada), **vocabulário** (para colunas tipo enum) e
**visibilidade-padrão**. Por isso:

- A primeira coluna **não** é o `image_path` cru; ela mostra a
  miniatura da imagem ao **passar o mouse** sobre as linhas.
- Colunas com vocabulário definido (gênero, etnia, origem) abrem um
  *combo box* na edição em vez de campo de texto livre.
- A ordenação por clique em cabeçalho funciona para colunas
  numéricas e ordenáveis; embeddings e estruturas complexas são
  ocultados por padrão.
- Clique-direito numa célula abre o menu de contexto, que inclui
  *Substituir valor* (todas as linhas ou por regex).
- Clique-direito no cabeçalho abre as **Propriedades da Coluna**
  (rótulo, role, vocabulário).

## O painel de detalhes

Quando uma linha é selecionada, o painel direito mostra:

- **Topo:** miniatura grande da imagem (somente leitura).
- **Aba *Geral / Bio*:** identidade, gênero, etnia, origem, hash,
  dimensões, status de erro:

![Aba Geral/Bio do painel de detalhes](assets/12_detail_geral_bio.png)

- **Aba *Qualidade ISO*:** barras visuais com todos os indicadores
  OFIQ (sharpness, exposição, oclusão, pose, etc.), SER-FIQ e FIQA:

![Aba Qualidade ISO do painel de detalhes, com todas as métricas OFIQ](assets/12_detail_quality_iso.png)

> ⚠️ Se a miniatura aparece em branco (placeholder) mesmo com o
> dataset carregado, é provável que o caminho `image_path` esteja
> quebrado — o parquet foi gerado em outra máquina (ou foi
> *commitado* sem as imagens). Use a função **Localizar** ([capítulo
> 03](03_carregando_e_localizando.md)) para corrigir.

## Próximos passos

- **Quer criar um dataset do zero?** Vá para [capítulo 02 — Criando
  um dataset](02_criando_dataset.md).
- **Já tem um parquet pronto?** Pule para [capítulo 03 — Carregando
  e localizando](03_carregando_e_localizando.md).
- **Quer adicionar métricas em um dataset existente?** Veja
  [capítulo 05 — Enriquecimento](05_enriquecimento.md).
