# Manual do Usuário — Gerenciador de Datasets

Este guia documenta o **Gerenciador de Datasets** do FaceAudit, módulo
responsável por extrair, organizar e auditar conjuntos de imagens
faciais antes que eles sejam consumidos pelo Gerenciador de Estratégias
nos cálculos forenses de razão de verossimilhança (LR).

A documentação é **híbrida**: começa com um tutorial guiado em torno de
um caso concreto e, em seguida, descreve cada tela em detalhe.

## Sumário

| Capítulo | Conteúdo |
|---|---|
| [01 — Visão geral](01_visao_geral.md) | O que é o Gerenciador de Datasets, quando usar, leiaute da janela principal |
| [02 — Criando um dataset](02_criando_dataset.md) | Diálogo *Novo dataset* (quatro grupos), *Criador de Template*, modelos SEPAEL, opções de alinhamento |
| [02b — Criando a partir de vídeo](02b_criando_dataset_de_video.md) | Extrair faces de vídeo(s) por âncora: janela de extração, revisão humana, importação e manifesto de reprodutibilidade |
| [03 — Carregando e localizando](03_carregando_e_localizando.md) | Abrir um `.parquet`, corrigir caminhos quebrados (modos `REWRITE_RELATIVE`, `COPY_ORIGINALS`, `REBASE_PREFIX`) |
| [04 — Navegando no dataset](04_navegando_dataset.md) | Tabela, painel de detalhes (Geral/Bio e Qualidade ISO), galeria, ribbons |
| [05 — Enriquecimento](05_enriquecimento.md) | Adicionar/recalcular embeddings, FIQA, OFIQ e SER-FIQ em um dataset existente |
| [06 — Filtros e estatísticas](06_filtros_e_estatisticas.md) | Diálogo *Filtrar*, *Estatísticas*, *Relatório de Esquema* |
| [07 — Operações estruturais](07_operacoes_estruturais.md) | Adicionar coluna, editar metadados, dividir, balancear, mesclar e comparar datasets |
| [08 — Pipeline de imagens](08_pipeline_de_imagens.md) | *Editor* e *Aplicação* de pipelines de transformação por imagem (recorte, alinhamento, etc.) |
| [09 — Auditoria forense](09_auditoria_forense.md) | Diálogo *Auditar* (problemas biométricos, hubness, UMAP) e *Log de Custódia* |
| [10 — Integração com o Gerenciador de Estratégias](10_integracao_estrategias.md) | *Enviar para Estratégia*, *Salvar na Biblioteca*, fluxos cross-tab |

## Como gerar o PDF

```bash
python docs/user_guide/dataset_manager/build_pdf.py
```

Saída: `docs/user_guide/dataset_manager/Manual_Gerenciador_de_Datasets.pdf`.

## Como atualizar as imagens

Os PNGs em `assets/` são gerados programaticamente. Veja
[`scripts/README.md`](scripts/README.md) para o procedimento de
regeneração após mudanças na interface.

## Convenções deste manual

- **Negrito** marca elementos da interface (botões, abas, campos).
- `Código` marca nomes de arquivos, colunas e termos técnicos.
- ⚠️ Sinaliza ações irreversíveis ou pegadinhas comuns.
- 💡 Sinaliza dicas e padrões recomendados.
