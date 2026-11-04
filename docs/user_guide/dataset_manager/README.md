# Manual do Usuário — Gerenciador de Datasets

Este guia documenta o **Gerenciador de Datasets** do FaceAudit, módulo
responsável por **extrair** (de pastas de imagens **ou de vídeos** de
vigilância), **organizar** e **auditar** conjuntos de imagens faciais
antes que eles sejam consumidos pelo Gerenciador de Estratégias nos
cálculos forenses de razão de verossimilhança (LR).

A documentação é **híbrida**: começa com um tutorial guiado em torno de
um caso concreto e, em seguida, descreve cada tela em detalhe.

## Sumário

| Capítulo | Conteúdo |
|---|---|
| [01 — Visão geral](01_visao_geral.md) | O que é o Gerenciador de Datasets, quando usar, leiaute da janela principal |
| [02 — Criando um dataset](02_criando_dataset.md) | Diálogo *Novo dataset* (quatro grupos), *Criador de Template*, modelos SEPAEL, forma da imagem (original/recorte/alinhada), revisão das imagens sem face detectada |
| [02b — Criando a partir de vídeo](02b_criando_dataset_de_video.md) | Extrair faces de vídeo(s) por âncora: janela de extração, revisão humana, importação e manifesto de reprodutibilidade |
| [03 — Carregando e modificando caminhos](03_carregando_e_localizando.md) | Abrir um `.parquet`; corrigir a subpasta `path_root` (sem mover arquivos), reescrever/copiar caminhos (`REWRITE_RELATIVE`, `COPY_ORIGINALS`, `REBASE_PREFIX`) |
| [04 — Navegando no dataset](04_navegando_dataset.md) | Tabela, painel de detalhes (Geral/Bio, Qualidade ISO e Metadados), galeria e seu menu de contexto, ribbons |
| [05 — Enriquecimento](05_enriquecimento.md) | Adicionar/recalcular embeddings, FIQA, OFIQ (qualidade ISO), pose, demografia e metadados EXIF/XMP em um dataset existente |
| [06 — Filtros e estatísticas](06_filtros_e_estatisticas.md) | Diálogo *Filtrar*, *Estatísticas*, *Relatório de Esquema* |
| [07 — Operações estruturais](07_operacoes_estruturais.md) | Adicionar coluna, editar metadados, dividir, balancear, mesclar e comparar datasets |
| [08 — Pipeline de imagens](08_pipeline_de_imagens.md) | *Editor* e *Processar* de pipelines de transformação por imagem (desfoque, compressão, recorte, oclusão, passos de impressão/cópia) e *Gerar Imagens* |
| [09 — Auditoria forense](09_auditoria_forense.md) | Diálogo *Auditar* (problemas biométricos, hubness, UMAP) e *Log de Custódia* |
| [10 — Integração com o Gerenciador de Estratégias](10_integracao_estrategias.md) | *Enviar para Estratégia*, *Salvar na Biblioteca*, fluxos cross-tab |

## Como gerar o PDF

```bash
python docs/user_guide/dataset_manager/build_pdf.py
```

Requer a biblioteca `reportlab`.

Saída: `docs/user_guide/dataset_manager/Manual_Gerenciador_de_Datasets.pdf`.
O PDF é gerado localmente, a partir destes arquivos Markdown, e **não é
versionado** no repositório — gere-o de novo sempre que quiser uma
cópia atualizada.

## Como atualizar as imagens

Os PNGs em `assets/` são gerados programaticamente. Veja
[`scripts/README.md`](scripts/README.md) para o procedimento de
regeneração após mudanças na interface.

## Convenções deste manual

- **Negrito** marca elementos da interface (botões, abas, campos).
- `Código` marca nomes de arquivos, colunas e termos técnicos.
- ⚠️ Sinaliza ações irreversíveis ou pegadinhas comuns.
- 💡 Sinaliza dicas e padrões recomendados.
