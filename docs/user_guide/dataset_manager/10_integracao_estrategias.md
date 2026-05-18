# 10 — Integração com o Gerenciador de Estratégias

Este capítulo fecha o guia descrevendo **como o dataset preparado
chega no Gerenciador de Estratégias** (e como ele pode voltar
para edição). Os dois módulos do FaceAudit não vivem isolados —
o Gerenciador de Datasets é o **produtor** dos artefatos, e o
Gerenciador de Estratégias é o **consumidor**.

## Sumário do capítulo

1. [Os dois caminhos de saída](#os-dois-caminhos-de-saída)
2. [→ Estratégia (enviar para o projeto atual)](#--estratégia)
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
| ➡ **→ Estratégia** (ribbon Arquivo) | Diretamente para o **projeto FaceAudit atualmente aberto** no Gerenciador de Estratégias | Você sabe o projeto e quer importar agora |
| 📚 **Biblioteca** (ribbon Arquivo) | **Biblioteca global** (compartilhada entre projetos) | Você quer usar o dataset em **vários projetos** futuros, ou ainda não criou o projeto |

Os dois NÃO são exclusivos — em geral o fluxo recomendado é:

1. Criar o dataset
2. **Salvar na Biblioteca** (cópia canônica)
3. Em cada projeto FaceAudit, importar o dataset *via* biblioteca

## → Estratégia

O botão ➡ **→ Estratégia** envia o **dataset atualmente aberto**
para o projeto aberto no Gerenciador de Estratégias:

| Ação | O que acontece |
|---|---|
| 1. Validação | Se nenhum projeto está aberto no Gerenciador de Estratégias, mostra erro |
| 2. Confirmação | Diálogo perguntando se quer importar (mostrando nome + tamanho) |
| 3. Cópia | O parquet é **copiado** para `<projeto>/datasets/<nome>.parquet` (estrutura do *Pendrive Paradigm*) |
| 4. Registro | O dataset vira um `StrategyResource` do tipo `DATASET` no projeto |
| 5. Cadeia | A operação é registrada no Audit Log do dataset original |
| 6. Switch | A aba muda automaticamente para Gerenciador de Estratégias |

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
| 1. Validação | Verifica que a pasta da biblioteca é gravável |
| 2. Confirmação | Diálogo perguntando se quer copiar |
| 3. Cópia | Parquet copiado para `<global_library>/datasets/<nome>.parquet` |
| 4. Audit | Registrado no log |

Depois disso, o dataset aparece no diálogo *Adicionar Recurso da
Biblioteca* (toolbar do Gerenciador de Estratégias), de onde
pode ser puxado para qualquer projeto.

### Diferença chave entre Biblioteca e Projeto

| Característica | Biblioteca Global | Projeto |
|---|---|---|
| Pasta | Fixa na máquina (`paths.library`) | Pasta do `.fa` (portátil) |
| Escopo | Compartilhada entre todos os projetos | Apenas o projeto que o contém |
| Portabilidade | Específica da máquina | Inteiro vai pra pendrive |
| Sobrevive ao close do projeto | ✅ Sim | ✅ Sim (o `.fa` é arquivo independente) |

## Voltando do Gerenciador de Estratégias

O caminho inverso também existe. No Gerenciador de Estratégias,
ao clicar com botão direito num recurso `DATASET` na árvore (ou
seu equivalente no resource info widget), aparece **"Editar no
Gerenciador de Datasets"**.

| Ação | O que acontece |
|---|---|
| 1. Resolução | O `artifact_ref.path` do recurso é resolvido para caminho absoluto |
| 2. Switch | Aba muda para Gerenciador de Datasets |
| 3. Open | `open_dataset_from_path()` é chamado direto (sem FileDialog) |
| 4. Audit | Operação `OPEN` registrada com origem `cross-tab` |

O dataset vai pra UI do Dataset Manager pra você editar livremente.

> ⚠️ **Cuidado**: o dataset aberto **é o arquivo no projeto**
> (`<projeto>/datasets/X.parquet`) — não uma cópia. Salvar (`Ctrl+S`)
> sobrescreve direto o que o Gerenciador de Estratégias está
> usando. Se você fizer mudanças destrutivas (filter, delete),
> elas afetam o projeto.
>
> Para edição segura: faça **Salvar Como** num caminho temporário,
> edite, depois importe de volta como recurso novo se aprovar.

## Boas práticas de organização

Aqui ficam alguns padrões que ajudam a manter os datasets
gerenciáveis em projetos forenses reais:

### 1. Use a Biblioteca como repositório canônico

```
~/faceaudit_library/
├── datasets/
│   ├── celeb_train.parquet               ← dataset de treino (LFW, CASIA-WebFace)
│   ├── celeb_train_aligned.parquet
│   ├── ofiq_baseline_2024.parquet
│   └── caso_002_ref_2024.parquet         ← referência de um caso ativo
├── templates/
│   ├── BFW.json
│   └── LFW.json
└── image_pipelines/
    ├── whatsapp_simulation.json
    └── id_card_simulation.json
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

- Verifique a integridade (botão 🔒 do diálogo Custódia)
- Exporte o log pra arquivo morto se for entregar como evidência
- O log **não** é truncado automaticamente — design forense

### 5. Salve a estratégia + dataset juntos

Quando um caso fecha:

1. Exporte o projeto FaceAudit completo (zip do `.fa`)
2. Exporte o Audit Log do dataset
3. Exporte a Schema Report como CSV
4. Inclua um README com a sequência de operações executadas

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
- **Os PDFs em `docs/`** — documentos sobre tópicos específicos:
  bi-Gaussianized calibration, arquitetura, roadmap

---

[← Capítulo 09 — Auditoria forense](09_auditoria_forense.md) · [Início do manual](README.md)
