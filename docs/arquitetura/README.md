# Arquitetura do FaceAudit — índice

Documentação técnica de implementação. O `CLAUDE.md` da raiz resume as regras permanentes e aponta para cá; estes arquivos guardam o detalhe (decisões, campos, histórico de por que cada regra existe).

| Arquivo | Conteúdo |
|---|---|
| [`motor_estrategias.md`](motor_estrategias.md) | Decisões de desenho do motor, desempenho, contexto de execução, projeto autocontido e biblioteca global, tabela completa dos tipos de instrução, mapeamento de importação, ferramentas externas |
| [`mapa_de_pastas.md`](mapa_de_pastas.md) | Mapa detalhado de pastas/arquivos do backend e lista dos protocolos |
| [`validacao.md`](validacao.md) | Sistema de validação: níveis, regras plugáveis × dinâmicas, defaults por instrução, templates, seção da UI, regras concretas |
| [`arquiteturas_lr.md`](arquiteturas_lr.md) | `LR_ARCHITECTURE_VECTORIZATION`, `SCORE_INTERACTION_FEATURE` (s0 congelado, dupla contagem), auditoria de monotonicidade, arquivos do núcleo LR |
| [`deteccao_e_geometria.md`](deteccao_e_geometria.md) | Invariante da geometria, orientação/EXIF, retentativa girada, revisão de imagens sem face detectada, dataset de vídeo em modo recorte |
| [`datasets_e_caminhos.md`](datasets_e_caminhos.md) | `path_root`, dataset autocontido, juntar/localizar datasets, colunas inteiras com vazio, seleção e exclusão, schema legado |
| [`pipeline_imagens.md`](pipeline_imagens.md) | Pipeline de degradação de imagens: geometria pelos passos, formulário declarativo, passos de cópia, questionada como alvo, ajuste automático |
| [`metadados_captura.md`](metadados_captura.md) | Metadados de captura de foto e vídeo, fuso, rotação do contêiner, qualidade JPEG e taxa de compressão |
| [`ofiq.md`](ofiq.md) | OFIQ: nomes de medida por versão, proveniência, extensões aceitas; resolução de executáveis externos |
| [`custodia.md`](custodia.md) | Versão e commit em toda entrada da custódia, Salvar Como, proveniência da detecção |
| [`ffservice_gpu.md`](ffservice_gpu.md) | FFService: CUDA no Windows, CoreML/ANE no Apple Silicon, os quatro caminhos de `process_dataset` |
| [`frontend_v2.md`](frontend_v2.md) | Strategy Manager V2 (árvore, painéis, comparação), visualizações, padrões Qt, sistema de tema |

Documentos de teoria e desenho que continuam em `docs/`: `similaridade_faceaudit.md`, `typicality_models.md`, `piecewise_linear_calibration.md`, `conditional_monotonicity_rule_design.md`, `lr_architecture_vectorization_faceaudit.md`, `interacao_score_raridade_faceaudit.md`, entre outros. Lista de testes por tema: [`../dev/testes.md`](../dev/testes.md).
