# 10 — Integração com o Gerenciador de Estratégias

Este capítulo fecha o guia descrevendo **como o dataset preparado
chega no Gerenciador de Estratégias** (e como ele pode voltar
para edição). Os dois módulos do FaceAudit não vivem isolados —
o Gerenciador de Datasets é o **produtor** dos artefatos (seja o
dataset criado de uma pasta de imagens ou extraído de vídeo, ver
[capítulo 02b](02b_criando_dataset_de_video.md)), e o Gerenciador
de Estratégias é o **consumidor**.

> ⚠️ **Datasets extraídos de vídeo carregam viés de seleção.**
> Um dataset criado por **🎬 Dataset de Vídeo** ou **📥 Importar
> extração de vídeo** recebe metadados de proveniência
> (`dataset_origin_type=VIDEO_TARGET_EXTRACTION` +
> `selection_warning`) porque suas faces foram selecionadas
> *condicionadas à similaridade com uma âncora* — **não** é uma
> amostra populacional neutra. O Gerenciador de Estratégias **não
> exibe esse aviso** ao importar. Portanto: **não** use um dataset
> de origem vídeo como população de referência/*background* de
> calibração de LR ou de modelos de tipicidade sem tratar o viés.
> Detalhes em [capítulo 02b — Ressalva forense](02b_criando_dataset_de_video.md).

## Sumário do capítulo

1. [Os dois caminhos de saída](#os-dois-caminhos-de-saída)
2. [🎯 → Estratégia (enviar para o projeto atual)](#--estratégia)
3. [Biblioteca (salvar para uso futuro)](#biblioteca)
4. [Voltando do Gerenciador de Estratégias](#voltando-do-gerenciador-de-estratégias)
5. [Boas práticas de organização](#boas-práticas-de-organização)

---

## Os dois caminhos de saída

Depois que seu dataset está pronto (ingerido, enriquecido,
auditado), você pode movê-lo para uso em estratégias por dois
caminhos:

| Botão | Destino | Quando usar |
|---|---|---|
| 🎯 **→ Estratégia** (ribbon Arquivo) | Diretamente para o **projeto FaceAudit atualmente aberto** no Gerenciador de Estratégias | Você sabe o projeto e quer importar agora |
| 📚 **Biblioteca** (ribbon Arquivo) | **Biblioteca global** (compartilhada entre projetos) | Você quer usar o dataset em **vários projetos** futuros, ou ainda não criou o projeto |

Os dois NÃO são exclusivos — em geral o fluxo recomendado é:

1. Criar o dataset
2. **Salvar na Biblioteca** (cópia canônica)
3. Em cada projeto FaceAudit, importar o dataset *via* biblioteca

## → Estratégia

O botão 🎯 **→ Estratégia** envia o **dataset atualmente aberto**
para o projeto aberto no Gerenciador de Estratégias:

| Ação | O que acontece |
|---|---|
| 1. Salvar antes | A pré-condição no lado do Dataset Manager é o dataset estar **salvo em disco**. Se houver edições pendentes, aparece um prompt **Salvar antes** — cancelar aborta o envio |
| 2. Switch | A aba muda para o Gerenciador de Estratégias, onde acontecem os passos seguintes |
| 3. Projeto aberto | Se **nenhum projeto** está aberto no Gerenciador de Estratégias, um diálogo oferece **Abrir existente…** / **Criar novo…** / Cancelar (não é um erro) |
| 4. ID do recurso | Um diálogo pede para **confirmar ou editar o ID do recurso** (default = nome do arquivo, sem extensão; deve ser único no projeto) |
| 5. Cópia | O parquet é **copiado** para `<projeto>/datasets/<id_do_recurso>.parquet` (estrutura do *Pendrive Paradigm*). Datasets com paths de imagem relativos têm essas referências **absolutizadas** na cópia, para não quebrar ao mudar de pasta. As imagens **não** são copiadas para o projeto |
| 6. Registro | O dataset passa a aparecer como recurso `DATASET` do projeto (o sidecar de estatísticas é calculado na hora), e um diálogo de sucesso mostra o path final no projeto |

> 💡 O envio em si **não** gera entrada no Audit Log. Uma entrada
> `SAVE` só aparece se o dataset estava com edições pendentes e
> você optou por salvá-lo antes do envio (passo 1).

O dataset chega no painel esquerdo do Gerenciador de Estratégias
(árvore de Recursos → DATASETS), pronto pra ser referenciado em
instruções.

> 💡 O dataset é **copiado**, não vinculado. Depois do envio, o
> arquivo no projeto vira uma cópia independente — modificar o
> original no Gerenciador de Datasets **não** afeta a cópia no
> projeto.

### Caminhos relativos no projeto

Quando o dataset é importado, o `StrategyArtifactReference.path`
fica **relativo** à raiz do projeto (ex: `datasets/meu_dataset.parquet`).
Isso é parte do *Pendrive Paradigm* — o projeto inteiro é portável.

## Biblioteca

O botão 📚 **Biblioteca** copia o dataset para a **biblioteca
global** — a pasta configurada na máquina (default
`<paths.library>/datasets/`).

| Ação | O que acontece |
|---|---|
| 1. Salvar antes | O dataset precisa estar **salvo em disco** — se houver edições pendentes, aparece o prompt **Salvar antes** |
| 2. Configuração | Verifica que `paths.library` está **configurado** no `config.json` (não testa gravabilidade) |
| 3. Identificador | Um diálogo pede o **Identificador (sem extensão)** — será o nome do arquivo na biblioteca (default = nome atual do dataset) |
| 4. Overwrite | Se já existir um `.parquet` com esse identificador na biblioteca, pede confirmação para sobrescrever |
| 5. Cópia | Parquet copiado para `<global_library>/datasets/<identificador>.parquet`, e um diálogo exibe o path final e o início (16 caracteres) do **hash SHA-256** do arquivo |

> 💡 O salvamento na biblioteca **não** grava entrada no Audit Log
> do dataset (só o `SAVE` do passo 1, se houve edições pendentes).

Depois disso, o dataset aparece no diálogo *Adicionar Recursos ao
Projeto* (botão 📥 **Adicionar Recursos** do ribbon *Projeto* do
Gerenciador de Estratégias), de onde pode ser puxado para qualquer
projeto.

### Diferença chave entre Biblioteca e Projeto

| Característica | Biblioteca Global | Projeto |
|---|---|---|
| Pasta | Fixa na máquina (`paths.library`) | Pasta do `.fa` (portátil) |
| Escopo | Compartilhada entre todos os projetos | Apenas o projeto que o contém |
| Portabilidade | Específica da máquina | Inteiro vai pra pendrive |
| Sobrevive ao close do projeto | ✅ Sim | ✅ Sim (o `.fa` é arquivo independente) |

## Voltando do Gerenciador de Estratégias

O caminho inverso também existe. No Gerenciador de Estratégias,
ao selecionar um recurso `DATASET` **primário** do projeto (com
arquivo presente), o painel de propriedades (*resource info
widget*) exibe o botão **📂 Abrir no Gerenciador de Datasets**.

| Ação | O que acontece |
|---|---|
| 1. Resolução | O `artifact_ref.path` do recurso é resolvido para caminho absoluto |
| 2. Switch | Aba muda para Gerenciador de Datasets |
| 3. Open | `open_dataset_from_path()` é chamado direto (sem FileDialog) |
| 4. Audit | Uma entrada `OPEN` padrão é registrada no Audit Log do dataset (idêntica à de qualquer abertura) |

O dataset vai pra UI do Dataset Manager pra você editar livremente.

> ⚠️ **Cuidado**: o dataset aberto **é o arquivo no projeto**
> (`<projeto>/datasets/X.parquet`) — não uma cópia. Salvar
> sobrescreve direto o que o Gerenciador de Estratégias está
> usando. Se você fizer mudanças destrutivas (filter, delete),
> elas afetam o projeto.
>
> Além disso, **salvar dispara reprocessamento automático**: o
> Gerenciador de Estratégias detecta que o parquet primário mudou,
> atualiza o hash e **invalida em cascata** as instruções das
> estratégias que dependem desse dataset (elas precisarão ser
> re-executadas).
>
> Para edição segura: faça **Salvar Como** num caminho temporário,
> edite, depois importe de volta como recurso novo se aprovar.

## Boas práticas de organização

Aqui ficam alguns padrões que ajudam a manter os datasets
gerenciáveis em projetos forenses reais:

### 1. Use a Biblioteca como repositório canônico

```
~/faceaudit_library/
├── datasets/                             ← datasets .parquet reutilizáveis
│   ├── celeb_train.parquet               ← dataset de treino (LFW, CASIA-WebFace)
│   ├── celeb_train_aligned.parquet
│   ├── ofiq_baseline_2024.parquet
│   └── caso_002_ref_2024.parquet         ← referência de um caso ativo
├── strategy_templates/                   ← templates de estratégia (LRStrategyDefinition)
│   ├── BFW.json
│   └── LFW.json
├── image_pipelines/                      ← pipelines de processamento de imagens
│   ├── whatsapp_simulation.json
│   └── id_card_simulation.json
├── split_configs/                        ← configurações de split
├── mining_configs/                       ← configurações de mineração de pares
├── pipeline_configs/                     ← pipelines de transformação de embeddings
├── filter_configs/                       ← filtros salvos (reusáveis entre projetos)
└── validation_rules/                     ← templates de regra de validação
```

Cada projeto importa o que precisa via biblioteca — evita
duplicação descontrolada de parquets.

### 2. Use `origin` consistentemente

A coluna `origin` é o **único** atributo que rastreia procedência
de cada face. Adote uma convenção:

| Valor | Significado |
|---|---|
| `Local` | Dados do operador local |
| `CCTV` | Câmeras de monitoramento |
| `Documento_ID` | Foto do documento |
| `Webcam_Forense` | Captação ativa pra perícia |
| `Caso_XX_YYYY` | Específico de um caso |
| `LFW` / `CASIA` / `BFW` | Datasets públicos |

Isso permite filtrar/agrupar por origem ao montar estratégias.

### 3. Nomeie datasets com convenção

| Padrão | Exemplo |
|---|---|
| `<conjunto>_<estado>` | `casia_aligned`, `lfw_filtered_q70` |
| `<caso>_<lado>` | `caso_002_ref`, `caso_002_probe` |
| `<conjunto>__<pipeline>` *(saída de Apply Pipeline)* | `casia__whatsapp_simulation` |

Stems consistentes facilitam identificar o conteúdo sem abrir.

### 4. Mantenha o Audit Log limpo

Cada `Salvar` registra uma entrada no log. Em datasets ativos
isso pode ficar grande. Periódico:

- Verifique a integridade da cadeia (botão **🔒 Verificar integridade** no diálogo **📜 Custódia** / *Audit Log*)
- Exporte o log pra arquivo morto se for entregar como evidência
- O log **não** é truncado automaticamente — design forense

### 5. Salve a estratégia + dataset juntos

Quando um caso fecha:

1. Guarde o projeto FaceAudit completo (compacte a pasta que
   contém o `.fa`; o **Selar projeto** do Gerenciador de Estratégias
   copia antes as imagens referenciadas para dentro dela)
2. Exporte o Audit Log do dataset (botão **📤 Exportar log** do diálogo **📜 Custódia**)
3. Exporte o dataset como CSV/Excel pelo botão **📤 Exportar para CSV/Excel…** do painel de **Colunas**
4. Inclua um README com a sequência de operações executadas
5. **Para datasets extraídos de vídeo**: preserve também a
   **pasta do run** (com o `manifesto_extracao.json` + subpasta
   `_meta/`) — ela contém os SHA-256 dos vídeos, âncoras e o
   commit do código, essenciais para reproduzir a extração. Ver
   [capítulo 02b](02b_criando_dataset_de_video.md).

Isso garante reprodutibilidade forense — outro perito consegue
abrir e refazer.

## Próximos passos

Você terminou o guia do Gerenciador de Datasets. Próximas
leituras:

- **Manual do Gerenciador de Estratégias** *(em construção)* —
  como construir e executar estratégias LR usando os datasets
  preparados aqui
- **CLAUDE.md** *(na raiz do repositório)* — referência técnica
  para desenvolvedores estendendo o FaceAudit
- **Os documentos em `docs/`** — textos sobre tópicos específicos
  (calibração, similaridade, tipicidade, vetorização de escores) e
  os PDFs da calibração bi-Gaussianizada e do roadmap do Gerenciador
  de Datasets

---

[← Capítulo 09 — Auditoria forense](09_auditoria_forense.md) · [Início do manual](README.md)
