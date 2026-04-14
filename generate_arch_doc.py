"""
Gerador do documento PDF: Arquitetura FaceAudit — Nova Estratégia Declarativa
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether,
)
from reportlab.platypus.flowables import HRFlowable


# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm
INNER_W = PAGE_W - 2 * MARGIN

BRAND_DARK  = colors.HexColor("#1A2942")   # azul escuro
BRAND_MED   = colors.HexColor("#2E5090")   # azul médio
BRAND_LIGHT = colors.HexColor("#D6E4F7")   # azul claro (fundo de tabela)
BRAND_ACCENT= colors.HexColor("#E8A020")   # âmbar (destaque)
GRAY_LIGHT  = colors.HexColor("#F4F6FA")
GRAY_MED    = colors.HexColor("#C8D0DC")
GRAY_DARK   = colors.HexColor("#555E6D")
WHITE       = colors.white
BLACK       = colors.black

# ---------------------------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------------------------

def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title", parent=base["Normal"],
        fontSize=28, leading=36, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_CENTER,
        spaceAfter=12,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle", parent=base["Normal"],
        fontSize=14, leading=20, textColor=colors.HexColor("#B0C8E8"),
        fontName="Helvetica", alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["cover_meta"] = ParagraphStyle(
        "cover_meta", parent=base["Normal"],
        fontSize=10, leading=14, textColor=colors.HexColor("#8AA8C8"),
        fontName="Helvetica", alignment=TA_CENTER,
    )

    styles["h1"] = ParagraphStyle(
        "h1", parent=base["Normal"],
        fontSize=18, leading=24, textColor=BRAND_DARK,
        fontName="Helvetica-Bold", spaceBefore=18, spaceAfter=6,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=base["Normal"],
        fontSize=13, leading=18, textColor=BRAND_MED,
        fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=4,
    )
    styles["h3"] = ParagraphStyle(
        "h3", parent=base["Normal"],
        fontSize=11, leading=15, textColor=BRAND_DARK,
        fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3,
    )

    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"],
        fontSize=10, leading=15, textColor=colors.HexColor("#2A2A2A"),
        fontName="Helvetica", spaceAfter=6, alignment=TA_JUSTIFY,
    )
    styles["body_small"] = ParagraphStyle(
        "body_small", parent=base["Normal"],
        fontSize=9, leading=13, textColor=GRAY_DARK,
        fontName="Helvetica", spaceAfter=4, alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=base["Normal"],
        fontSize=10, leading=14, textColor=colors.HexColor("#2A2A2A"),
        fontName="Helvetica", leftIndent=16, spaceAfter=3,
        bulletIndent=4,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Normal"],
        fontSize=8.5, leading=13, textColor=colors.HexColor("#1A1A2E"),
        fontName="Courier", leftIndent=10, spaceAfter=2,
        backColor=colors.HexColor("#F0F2F8"),
    )
    styles["caption"] = ParagraphStyle(
        "caption", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=GRAY_DARK,
        fontName="Helvetica-Oblique", alignment=TA_CENTER,
        spaceAfter=8,
    )
    styles["callout"] = ParagraphStyle(
        "callout", parent=base["Normal"],
        fontSize=9.5, leading=14, textColor=BRAND_DARK,
        fontName="Helvetica", leftIndent=10, rightIndent=10,
        spaceAfter=6, spaceBefore=4,
    )

    # Células de tabela
    styles["th"] = ParagraphStyle(
        "th", parent=base["Normal"],
        fontSize=9, leading=12, textColor=WHITE,
        fontName="Helvetica-Bold", alignment=TA_LEFT,
    )
    styles["td"] = ParagraphStyle(
        "td", parent=base["Normal"],
        fontSize=9, leading=12, textColor=colors.HexColor("#1A1A1A"),
        fontName="Helvetica", alignment=TA_LEFT,
    )
    styles["td_code"] = ParagraphStyle(
        "td_code", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=colors.HexColor("#1A1A2E"),
        fontName="Courier", alignment=TA_LEFT,
    )
    styles["td_small"] = ParagraphStyle(
        "td_small", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=GRAY_DARK,
        fontName="Helvetica", alignment=TA_LEFT,
    )

    return styles


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def rule(color=BRAND_LIGHT, thickness=1, width="100%"):
    return HRFlowable(width=width, thickness=thickness, color=color,
                      spaceAfter=4, spaceBefore=4)


def callout_box(story, S, text, bg=BRAND_LIGHT, border=BRAND_MED):
    data = [[Paragraph(text, S["callout"])]]
    t = Table(data, colWidths=[INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("BOX",        (0, 0), (-1, -1), 1, border),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))


def two_col_table(story, S, header, rows, col_widths=None):
    """Tabela genérica de 2+ colunas com cabeçalho escuro."""
    if col_widths is None:
        col_widths = [INNER_W * 0.35, INNER_W * 0.65]

    header_cells = [Paragraph(h, S["th"]) for h in header]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(cell, S["td"]) for cell in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND",    (0, 0), (-1, 0),  BRAND_DARK),
        ("BACKGROUND",    (0, 1), (-1, -1), GRAY_LIGHT),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, GRAY_MED),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    t.setStyle(TableStyle(style))
    story.append(t)
    story.append(Spacer(1, 8))


def three_col_table(story, S, header, rows, col_widths=None):
    if col_widths is None:
        col_widths = [INNER_W * 0.22, INNER_W * 0.28, INNER_W * 0.50]

    header_cells = [Paragraph(h, S["th"]) for h in header]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(cell, S["td"]) for cell in row])

    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  BRAND_DARK),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, GRAY_LIGHT]),
        ("GRID",          (0, 0), (-1, -1), 0.5, GRAY_MED),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 8),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def code_block(story, S, lines):
    """Bloco de código monoespaçado com fundo cinza."""
    data = [[Paragraph(line.replace(" ", "&nbsp;").replace("<", "&lt;"), S["code"])]
            for line in lines]
    t = Table(data, colWidths=[INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), colors.HexColor("#F0F2F8")),
        ("BOX",          (0, 0), (-1, -1), 0.5, GRAY_MED),
        ("LEFTPADDING",  (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING",   (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 2),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


def section_banner(story, S, text):
    """Banner de seção numerada com fundo azul escuro."""
    data = [[Paragraph(text, S["h1"])]]
    t = Table(data, colWidths=[INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), BRAND_DARK),
        ("TEXTCOLOR",    (0, 0), (-1, -1), WHITE),
        ("LEFTPADDING",  (0, 0), (-1, -1), 16),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
    ]))
    story.append(t)
    story.append(Spacer(1, 8))


# ---------------------------------------------------------------------------
# CAPA
# ---------------------------------------------------------------------------

def build_cover(story, S):
    # Fundo simulado: tabela de largura total com cor de fundo
    cover_data = [[
        Paragraph("FaceAudit", S["cover_title"]),
    ]]
    # Usamos múltiplas linhas para simular um banner
    lines = [
        Paragraph("FaceAudit", S["cover_title"]),
        Spacer(1, 4),
        Paragraph("Nova Arquitetura Declarativa de Estratégias", S["cover_subtitle"]),
        Spacer(1, 8),
        Paragraph("Documento Técnico de Arquitetura", S["cover_subtitle"]),
        Spacer(1, 24),
        Paragraph("FaceAudit — documentação de arquitetura", S["cover_meta"]),
        Spacer(1, 4),
        Paragraph("Versão 1.0 · Abril 2026", S["cover_meta"]),
    ]

    banner_data = [[item] for item in lines]

    # Altura estimada
    t = Table([[item] for item in [
        Paragraph("FaceAudit", S["cover_title"]),
        Spacer(1, 4),
        Paragraph("Nova Arquitetura Declarativa de Estratégias", S["cover_subtitle"]),
        Spacer(1, 8),
        Paragraph("Documento Técnico", S["cover_subtitle"]),
        Spacer(1, 24),
        Paragraph("FaceAudit — documentação de arquitetura", S["cover_meta"]),
        Spacer(1, 4),
        Paragraph("Versão 1.0 · Abril 2026", S["cover_meta"]),
    ]], colWidths=[INNER_W])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), BRAND_DARK),
        ("LEFTPADDING",  (0, 0), (-1, -1), 24),
        ("RIGHTPADDING", (0, 0), (-1, -1), 24),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(Spacer(1, 80))
    story.append(t)
    story.append(Spacer(1, 30))

    # Linha de resumo
    callout_box(story, S,
        "Este documento descreve a arquitetura declarativa do FaceAudit para criação e "
        "execução de estratégias de cálculo de Likelihood Ratio (LR) em perícias forenses "
        "de reconhecimento facial. Destina-se a desenvolvedores e peritos que precisam "
        "entender, estender ou manter o sistema.",
        bg=BRAND_LIGHT, border=BRAND_MED,
    )
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# ÍNDICE (manual)
# ---------------------------------------------------------------------------

def build_toc(story, S):
    story.append(Paragraph("Índice", S["h1"]))
    rule(BRAND_MED, 2)
    story.append(rule(BRAND_MED, 2))
    story.append(Spacer(1, 8))

    toc_items = [
        ("1.", "Visão Geral e Motivação"),
        ("2.", "Arquitetura Antiga — Problemas Resolvidos"),
        ("3.", "Visão Geral da Nova Arquitetura"),
        ("4.", "Fluxo de Execução Detalhado"),
        ("5.", "Componentes Principais"),
        ("  5.1", "LRStrategyDefinition — O Plano da Estratégia"),
        ("  5.2", "StrategyExecutionContext — O Barramento de Estado"),
        ("  5.3", "StrategyResource — A Unidade de Troca"),
        ("  5.4", "StrategyExecutor — O Orquestrador"),
        ("  5.5", "Executores de Instrução"),
        ("6.", "Os 8 Tipos de Instrução"),
        ("7.", "Sistema de Dois Níveis: Biblioteca Global + Projeto"),
        ("  7.1", "Biblioteca Global"),
        ("  7.2", "Projeto Autocontido (Pendrive Paradigm)"),
        ("  7.3", "Fluxo de Montagem de um Projeto"),
        ("8.", "Modelos de Similaridade e Calibração"),
        ("9.", "Pipeline de Transformação de Embeddings"),
        ("  9.1", "PipelineTrajectoryService — Visualização 3D"),
        ("10.", "Protocolo de Persistência (ArtifactStorageProtocol)"),
        ("  10.1", "Verificação de Integridade (LocalProjectArtifactStorage)"),
        ("11.", "Sistema de Validação"),
        ("  11.1", "Dois Mecanismos Complementares"),
        ("  11.2", "ValidationRuleBase e Cache de Stats"),
        ("  11.3", "Regras Concretas Implementadas"),
        ("12.", "Ciclo de Vida de um Projeto FaceAudit"),
        ("  12.1", "Fluxo de Treinamento (end-to-end)"),
        ("  12.2", "Fluxo de Inferência (StrategyInferenceService)"),
        ("13.", "Decisões de Design e Princípios Arquiteturais"),
        ("14.", "Extensibilidade — Como Adicionar Novos Componentes"),
        ("15.", "Estrutura de Pastas"),
    ]

    for num, title in toc_items:
        indent = 16 if num.startswith("  ") else 0
        style = ParagraphStyle("toc_item", parent=S["body"],
                               leftIndent=indent, spaceAfter=3)
        story.append(Paragraph(f"{num.strip()}  {title}", style))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 1 — VISÃO GERAL E MOTIVAÇÃO
# ---------------------------------------------------------------------------

def build_s1(story, S):
    section_banner(story, S, "1. Visão Geral e Motivação")

    story.append(Paragraph(
        "O FaceAudit é uma plataforma forense para análise e comparação de imagens faciais. "
        "Seu objetivo central é permitir que peritos criem e configurem <b>estratégias de cálculo "
        "de Likelihood Ratio (LR)</b> para comparações forenses de embeddings faciais AdaFace — "
        "inteiramente pela interface da aplicação, sem alterar código-fonte.",
        S["body"]
    ))

    story.append(Paragraph(
        "O LR é a medida estatística que quantifica o quanto a evidência (semelhança entre "
        "dois embeddings faciais) suporta a hipótese de que as imagens pertencem à mesma "
        "pessoa, em relação à hipótese de que pertencem a pessoas diferentes. Um LR alto "
        "indica forte suporte para mesma identidade; um LR próximo de zero indica forte "
        "suporte para identidades distintas.",
        S["body"]
    ))

    story.append(Paragraph("Desafio endereçado", S["h2"]))
    story.append(Paragraph(
        "Na arquitetura antiga, a ordem de execução, os tipos de dados utilizados e as "
        "etapas de transformação eram fixos no código Python. Qualquer mudança de abordagem "
        "— por exemplo, treinar primeiro o pipeline de transformação e depois o modelo de "
        "similaridade, ou inserir uma etapa de curadoria de pares — exigia modificar arquivos "
        "de código, testar e reimplantar o sistema. Isso tornava a experimentação perita "
        "completamente dependente do desenvolvedor.",
        S["body"]
    ))

    callout_box(story, S,
        "<b>Solução adotada:</b> A nova arquitetura é <b>declarativa</b>. Uma estratégia é "
        "descrita como uma lista ordenada de instruções configuráveis. O motor de execução "
        "interpreta e roda essa lista. O perito define o que acontece; o código executa.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 2 — ARQUITETURA ANTIGA
# ---------------------------------------------------------------------------

def build_s2(story, S):
    section_banner(story, S, "2. Arquitetura Antiga — Problemas Resolvidos")

    story.append(Paragraph(
        "A arquitetura anterior foi construída em três camadas que colaboravam via "
        "callbacks de dirty-flag:",
        S["body"]
    ))

    three_col_table(story, S,
        ["Camada", "Classes principais", "Papel"],
        [
            ["Managers", "KDEManager, PLDAManager, StrategiesManager, ProjectManager",
             "Classes com estado que possuíam o ciclo de vida completo de cada estratégia: "
             "config, carregamento de dados, preparação, treino, persistência e inferência."],
            ["Controllers", "KDEPreparationController, PLDATrainingController, PLDATransformationPipelinesController",
             "Orquestravam as etapas dentro de cada fase. Ordem de execução implícita e hard-coded nos métodos."],
            ["Strategies", "KDEStrategy, PLDAStrategy, PLDASnormStrategy",
             "Implementavam os modelos matemáticos: fit, predict, infer. Implementavam LRModelProtocol."],
        ],
        col_widths=[INNER_W*0.18, INNER_W*0.30, INNER_W*0.52],
    )

    story.append(Paragraph("Limitações identificadas", S["h2"]))

    bullet_items = [
        "A ordem de execução estava hard-coded nos métodos dos controllers.",
        "Era impossível inserir, remover ou reordenar etapas sem alterar Python.",
        "O tipo de dados trafegado entre etapas não era declarado — dependia de convenção implícita.",
        "Não havia separação entre similaridade (score bruto) e calibração (conversão para LR).",
        "O modelo KDE era monolítico: recebia embeddings e produzia LRs diretamente.",
        "Projetos não eram autocontidos: dependiam de caminhos absolutos da máquina.",
        "Recursos reutilizáveis (datasets, configs) não tinham um local centralizado gerenciado.",
    ]
    for item in bullet_items:
        story.append(Paragraph(f"• {item}", S["bullet"]))

    story.append(Spacer(1, 8))
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 3 — VISÃO GERAL DA NOVA ARQUITETURA
# ---------------------------------------------------------------------------

def build_s3(story, S):
    section_banner(story, S, "3. Visão Geral da Nova Arquitetura")

    story.append(Paragraph(
        "A nova arquitetura é baseada no padrão <b>Command + Context + Registry</b>. "
        "Uma estratégia de cálculo de LR é representada por um objeto "
        "<b>LRStrategyDefinition</b> — um plano declarativo que lista os recursos "
        "iniciais e a sequência de instruções a executar. O motor de execução "
        "(<b>StrategyExecutor</b>) interpreta esse plano instrução por instrução, "
        "mantendo todo o estado compartilhado em um <b>StrategyExecutionContext</b>.",
        S["body"]
    ))

    story.append(Paragraph("Diagrama de alto nível", S["h2"]))

    code_block(story, S, [
        "┌─────────────────────────────────────────────────────────────┐",
        "│                    LRStrategyDefinition                      │",
        "│  resources: [dataset_A, split_cfg, mining_cfg, ...]          │",
        "│  instructions: [PIPELINE_TRAINING, DATASET_SPLIT,            │",
        "│                  PAIR_MINING, SIMILARITY_MODEL_TRAINING,     │",
        "│                  SIMILARITY_SCORING, CALIBRATION_TRAINING]   │",
        "└───────────────────────────┬─────────────────────────────────┘",
        "                            │ StrategyExecutorFactory.build()",
        "                            ▼",
        "┌─────────────────────────────────────────────────────────────┐",
        "│                      StrategyExecutor                        │",
        "│  ┌────────────────────────────────────────────────────────┐  │",
        "│  │               StrategyExecutionContext                  │  │",
        "│  │  get_resource() / set_resource() / has_resource()      │  │",
        "│  │  log / artifacts / validation_rules / metadata         │  │",
        "│  └────────────────────────────────────────────────────────┘  │",
        "│  Para cada instrução:                                         │",
        "│    InstructionExecutorRegistry.resolve(instruction_type)      │",
        "│      → executor.execute(instruction, context)                 │",
        "│            lê recursos do contexto                            │",
        "│            executa lógica                                     │",
        "│            escreve novos recursos no contexto                 │",
        "└─────────────────────────────────────────────────────────────┘",
        "                            │",
        "                            ▼",
        "                  LRStrategyTrainedModel",
        "             (recursos treinados + artefatos + log)",
    ])

    story.append(Paragraph(
        "A principal característica desta arquitetura é que <b>cada instrução é "
        "completamente autocontida</b>: ela declara quais recursos consome "
        "(<code>input_resource_ids</code>) e quais produz (<code>output_resource_ids</code>). "
        "O contexto funciona como um barramento de objetos nomeados — as instruções se "
        "comunicam exclusivamente por meio dele.",
        S["body"]
    ))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 4 — FLUXO DE EXECUÇÃO DETALHADO
# ---------------------------------------------------------------------------

def build_s4(story, S):
    section_banner(story, S, "4. Fluxo de Execução Detalhado")

    story.append(Paragraph(
        "O método <code>StrategyExecutor.execute(strategy_definition)</code> segue "
        "os seguintes passos:",
        S["body"]
    ))

    steps = [
        ("1. Criação do contexto",
         "StrategyExecutionContext é instanciado via context_factory. "
         "Este objeto será o barramento de estado durante toda a execução."),
        ("2. Hidratação e carregamento de recursos iniciais",
         "Os StrategyResources declarados em strategy_definition.resources são "
         "processados antes de entrar no contexto. Quando o executor foi montado "
         "com artifact_storage (caso típico ao executar uma estratégia carregada "
         "de JSON), o StrategyResourceLoaderService hidrata cada recurso cujo "
         "payload é None mas possui artifact_ref: "
         "DATASET → EmbeddingDataset.from_parquet(); "
         "PIPELINE_CONFIG → TransformationPipelineConfig.from_dict(). "
         "Após a hidratação, os recursos são inseridos no contexto via set_resource(). "
         "Sem artifact_storage (fluxo programático puro), os recursos são inseridos "
         "diretamente — o chamador é responsável por preencher os payloads."),
        ("3. Registro de regras de validação",
         "As ValidationRules passadas à factory são registradas no contexto, "
         "tornando-as disponíveis para qualquer instrução que as referencie "
         "pelo ID."),
        ("4. Validação da estratégia",
         "As validation_rule_ids declaradas na LRStrategyDefinition são executadas "
         "contra o contexto inicial. Se qualquer regra crítica falhar, a execução "
         "é abortada."),
        ("5. Loop de execução das instruções",
         "Para cada StrategyInstructionDefinition na lista instructions (em ordem): "
         "(a) resolve o executor concreto no InstructionExecutorRegistry; "
         "(b) chama executor.validate(instruction, context); "
         "(c) se válido, chama executor.execute(instruction, context); "
         "(d) o resultado é registrado; se crítico, a execução pode ser abortada."),
        ("6. Empacotamento do resultado",
         "Após todas as instruções, o contexto final é empacotado em um "
         "LRStrategyTrainedModel contendo: recursos treinados, referências de "
         "artefatos persistidos, log de execução e metadados."),
    ]

    for title, desc in steps:
        story.append(KeepTogether([
            Paragraph(title, S["h3"]),
            Paragraph(desc, S["body"]),
        ]))

    story.append(Paragraph("Exemplo de estratégia típica", S["h2"]))
    story.append(Paragraph(
        "Uma estratégia completa de PLDA + Regressão Logística tipicamente contém "
        "as seguintes instruções em ordem:",
        S["body"]
    ))

    code_block(story, S, [
        "Instrução 1: PIPELINE_TRAINING",
        "  → Treina pipeline LINEAR [Centering → L2Norm → Whitening → LDA]",
        "  → Produz: trained_pipeline",
        "",
        "Instrução 2: PIPELINE_APPLICATION",
        "  → Aplica trained_pipeline ao dataset de treino",
        "  → Produz: transformed_dataset",
        "",
        "Instrução 3: DATASET_SPLIT",
        "  → Divide transformed_dataset em train_A (70%) e train_B (30%)",
        "  → Produz: split_A, split_B",
        "",
        "Instrução 4: SIMILARITY_MODEL_TRAINING",
        "  → Treina PLDASimilarityModel em split_A",
        "  → Produz: trained_plda",
        "",
        "Instrução 5: SIMILARITY_SCORING",
        "  → Gera matriz de scores em split_B usando trained_plda",
        "  → Produz: scores_dataset",
        "",
        "Instrução 6: PAIR_MINING",
        "  → Cura impostores/genuínos de scores_dataset via ComparisonMiningConfig",
        "  → Produz: mined_impostors, mined_genuines",
        "",
        "Instrução 7: CALIBRATION_MODEL_TRAINING",
        "  → Treina LogisticRegressionCalibrationModel com mined_impostors + mined_genuines",
        "  → Produz: trained_calibrator",
    ])

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 5 — COMPONENTES PRINCIPAIS
# ---------------------------------------------------------------------------

def build_s5(story, S):
    section_banner(story, S, "5. Componentes Principais")

    # 5.1
    story.append(Paragraph("5.1 LRStrategyDefinition — O Plano da Estratégia", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/models/strategy/lr_strategy_definition.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "É o DTO (Data Transfer Object) central da nova arquitetura. Representa um plano "
        "completamente declarativo de como calcular LRs para um tipo de comparação forense. "
        "É um <code>@dataclass</code> que herda apenas de <code>JSONSerializableMixin</code> "
        "(não herda de Protocol — ver Seção 12).",
        S["body"]
    ))

    two_col_table(story, S,
        ["Campo", "Descrição"],
        [
            ["strategy_id", "UUID único da estratégia"],
            ["name", "Nome legível pelo perito"],
            ["description", "Descrição da abordagem"],
            ["resources", "Lista de StrategyResource com os dados e configs de entrada"],
            ["instructions", "Lista ordenada de StrategyInstructionDefinition"],
            ["validation_rule_ids", "IDs de regras de validação globais da estratégia"],
            ["created_at / updated_at", "Timestamps de criação e atualização"],
        ],
        col_widths=[INNER_W*0.32, INNER_W*0.68],
    )

    story.append(Paragraph(
        "O método <code>from_dict()</code> é um dispatcher polimórfico: ao ler "
        "<code>instruction_type</code>, hidrata o DTO especializado correto "
        "(DatasetSplitInstruction, PairMiningInstruction, etc.).",
        S["body"]
    ))

    # 5.2
    story.append(Paragraph("5.2 StrategyExecutionContext — O Barramento de Estado", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/contexts/strategy_execution_context.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "O contexto é o único canal de comunicação entre instruções. "
        "Nenhuma instrução conhece as outras — cada uma só enxerga o contexto. "
        "Essa separação elimina acoplamento direto entre executores.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Responsabilidade", "Métodos principais"],
        [
            ["Gerenciar recursos",
             "set_resource(r), get_resource(id), has_resource(id), list_resources()"],
            ["Registrar artefatos persistidos",
             "add_artifact(ref), list_artifacts()"],
            ["Gerenciar regras de validação",
             "register_rule(id, rule), get_validation_rule(id), has_validation_rule(id)"],
            ["Log de execução",
             "log(message) — registro cronológico das operações"],
            ["Metadados de execução",
             "strategy_id, started_at, status"],
        ],
    )

    # 5.3
    story.append(Paragraph("5.3 StrategyResource — A Unidade de Troca", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/models/strategy/strategy_resource.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Todo objeto que transita entre instruções é um <code>StrategyResource</code>. "
        "Ele encapsula um payload tipado junto com metadados de identificação.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Campo", "Papel"],
        [
            ["resource_id", "Identificador único dentro do contexto (string)"],
            ["name", "Nome legível"],
            ["resource_type", "StrategyResourceType (DATASET, TRAINED_PIPELINE, etc.)"],
            ["payload", "O objeto real: EmbeddingDataset, PLDASimilarityModel, etc."],
            ["artifact_ref", "StrategyArtifactReference quando o recurso foi persistido em disco"],
        ],
    )

    # 5.4
    story.append(Paragraph("5.4 StrategyExecutor — O Orquestrador", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/executors/strategy_executor.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "O executor é o motor de iteração. Ele não contém lógica de negócio — "
        "apenas coordena o fluxo: cria o contexto, hidrata e carrega recursos iniciais, "
        "resolve executores pelo registry, delega e coleta resultados. "
        "Isso permite testar cada executor de instrução de forma completamente isolada.",
        S["body"]
    ))
    story.append(Paragraph(
        "Parâmetro chave: <code>artifact_storage: Optional[ArtifactStorageProtocol]</code>. "
        "Quando fornecido, o executor instancia internamente um "
        "<code>StrategyResourceLoaderService</code> que hidrata automaticamente os "
        "payloads dos recursos declarados antes da execução. Obrigatório quando a "
        "estratégia é carregada de JSON (payload=None, artifact_ref preenchido). "
        "Passe <code>project.storage</code> como valor.",
        S["body"]
    ))

    # 5.5
    story.append(Paragraph("5.5 Executores de Instrução", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/executors/instructions/</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Cada tipo de instrução tem um executor concreto que estende "
        "<code>StrategyInstructionExecutorBase</code>. A base implementa a lógica "
        "de validação genérica (regras declaradas na instrução + chamada a "
        "<code>_validate_instruction_specifics()</code>). O executor concreto "
        "sobrescreve <code>execute()</code> e <code>_validate_instruction_specifics()</code>.",
        S["body"]
    ))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 6 — OS 8 TIPOS DE INSTRUÇÃO
# ---------------------------------------------------------------------------

def build_s6(story, S):
    section_banner(story, S, "6. Os 8 Tipos de Instrução")

    story.append(Paragraph(
        "Cada instrução é um <code>@dataclass</code> que herda de "
        "<code>StrategyInstructionDefinition</code> e declara campos específicos. "
        "O método <code>__post_init__</code> define <code>instruction_type</code> e "
        "sincroniza <code>input_resource_ids</code> / <code>output_resource_ids</code> "
        "com os campos específicos via <code>_sync_declared_io()</code>.",
        S["body"]
    ))

    instructions = [
        ("PIPELINE_TRAINING", "PipelineTrainingInstructionExecutor",
         "Treina um EmbeddingTransformationPipeline (modo LINEAR ou DUAL). "
         "O modo LINEAR usa apenas o dataset de referência. O modo DUAL "
         "permite que etapas como CORAL e Whitening usem também o dataset alvo "
         "durante o treino (controlado por PipelineTrainingPolicy). "
         "Produz: recurso TRAINED_PIPELINE."),
        ("PIPELINE_APPLICATION", "PipelineApplicationInstructionExecutor",
         "Aplica um pipeline treinado a um ou mais datasets. "
         "Cada dataset de entrada gera um dataset transformado de saída. "
         "Produz: recursos INTERMEDIATE_DATASET."),
        ("DATASET_SPLIT", "DatasetSplitInstructionExecutor",
         "Divide um dataset em dois subconjuntos. "
         "Modos: BY_IDENTITY_RATIO, BY_IDENTITY_LIST, BY_INDEX_RATIO, BY_INDEX_LIST. "
         "Funciona com FaceDataset e EmbeddingDataset. "
         "Produz: dois recursos INTERMEDIATE_DATASET."),
        ("DATASET_MERGE", "DatasetMergeInstructionExecutor",
         "Combina ou atualiza datasets. "
         "Modo CONCATENATE une dois datasets; modo UPDATE_BY_INDEX substitui "
         "entradas por índice. "
         "Produz: recurso INTERMEDIATE_DATASET."),
        ("PAIR_MINING", "PairMiningInstructionExecutor",
         "Curadoria estatística de pares de treino. Recebe um "
         "EmbeddingComparisonDataset (pares com scores pré-calculados) e aplica "
         "ComparisonMiningConfig. Estratégias: HARD, EASY, TYPICAL, RANDOM, MIXED. "
         "Modos de seleção: ROUND_ROBIN_BY_PIVOT (balanceado por identidade) e "
         "GLOBAL_ORDER (fatiamento direto). "
         "Produz: recursos impostores minerados + genuínos minerados."),
        ("SIMILARITY_MODEL_TRAINING", "SimilarityModelTrainingInstructionExecutor",
         "Treina ou prepara um modelo de similaridade: CosineSimilarityModel (sem treino, "
         "baseado em distância cosseno) ou PLDASimilarityModel (PLDA via SpeechBrain). "
         "O modelo treinado produz scores brutos — não LRs. "
         "Produz: recurso TRAINED_SIMILARITY_MODEL."),
        ("SIMILARITY_SCORING", "SimilarityScoringInstructionExecutor",
         "Gera scores de similaridade usando um modelo treinado. "
         "Pode operar em modo self-comparison (um dataset contra si mesmo) ou "
         "cross-comparison (dois datasets). "
         "Produz: recurso SimilarityScoreDataset."),
        ("CALIBRATION_MODEL_TRAINING", "CalibrationModelTrainingInstructionExecutor",
         "Treina um modelo de calibração: LogisticRegressionCalibrationModel ou "
         "KDECalibrationModel. Recebe scores brutos + labels e aprende a converter "
         "para LRs. KDE usa dois kernels (genuínos/impostores) com Fisher transform "
         "e GridSearch de bandwidth. "
         "Produz: recurso TRAINED_CALIBRATION_MODEL."),
    ]

    for i, (itype, executor, desc) in enumerate(instructions):
        data = [
            [Paragraph(f"<b>{itype}</b>", S["td_code"]),
             Paragraph(executor, S["td_small"])],
            [Paragraph(desc, S["td"]), ""],
        ]
        t = Table(data, colWidths=[INNER_W * 0.45, INNER_W * 0.55])
        bg = BRAND_LIGHT if i % 2 == 0 else colors.HexColor("#EDF2FB")
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  bg),
            ("BACKGROUND",   (0, 1), (-1, 1),  WHITE),
            ("SPAN",         (0, 1), (-1, 1)),
            ("BOX",          (0, 0), (-1, -1), 0.5, GRAY_MED),
            ("INNERGRID",    (0, 0), (-1, 0),  0.5, GRAY_MED),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 7 — SISTEMA DE DOIS NÍVEIS
# ---------------------------------------------------------------------------

def build_s7(story, S):
    section_banner(story, S, "7. Sistema de Dois Níveis: Biblioteca Global + Projeto")

    story.append(Paragraph(
        "A nova arquitetura separa claramente dois espaços de armazenamento com "
        "finalidades distintas:",
        S["body"]
    ))

    # 7.1
    story.append(Paragraph("7.1 Biblioteca Global", S["h2"]))
    story.append(Paragraph(
        "Classe: <code>GlobalLibraryStorage</code> · "
        "Pasta: configurada em <code>config.json</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "A biblioteca global é uma pasta na máquina do perito que armazena "
        "recursos <b>reutilizáveis entre projetos</b>. Sua estrutura de subpastas "
        "é definida pelo enum <code>GlobalLibraryFolder</code>:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Subpasta", "Conteúdo"],
        [
            ["datasets/",           "Datasets .parquet reutilizáveis entre projetos"],
            ["split_configs/",      "Configurações de split (SplitConfig JSON)"],
            ["mining_configs/",     "Configurações de mineração (ComparisonMiningConfig JSON)"],
            ["strategy_templates/", "Templates de estratégia (LRStrategyDefinition JSON "
                                    "com instruções configuradas mas artifact_ref=None nos recursos)"],
        ],
        col_widths=[INNER_W*0.30, INNER_W*0.70],
    )

    story.append(Paragraph(
        "Um <b>template de estratégia</b> é uma LRStrategyDefinition cujas instruções "
        "já estão configuradas (tipos, parâmetros, nomes de recursos), mas cujos "
        "recursos ainda não têm artefatos vinculados (<code>artifact_ref=None</code>). "
        "Ele serve de ponto de partida para novos projetos — ao instanciar um "
        "template, os recursos são vinculados e copiados para o projeto.",
        S["body"]
    ))

    # 7.2
    story.append(Paragraph("7.2 Projeto Autocontido — Pendrive Paradigm", S["h2"]))
    story.append(Paragraph(
        "Classe: <code>LocalProjectArtifactStorage</code> · "
        "Âncora: arquivo <code>project.fa</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Um projeto FaceAudit na nova arquitetura é uma <b>pasta autocontida</b>. "
        "Tudo que o projeto precisa para ser executado e reproduzido está dentro "
        "dessa pasta. A pasta pode ser copiada, comprimida ou transferida para outra "
        "máquina — nada se quebra.",
        S["body"]
    ))

    callout_box(story, S,
        "<b>Pendrive Paradigm:</b> Toda referência a artefatos em "
        "StrategyArtifactReference.path é sempre <b>relativa à raiz do projeto</b>. "
        "Mover a pasta não invalida nenhuma referência.",
        bg=colors.HexColor("#E8F5E9"), border=colors.HexColor("#4CAF50"),
    )

    story.append(Paragraph(
        "O arquivo <code>project.fa</code> é o identificador do projeto. "
        "Ele contém apenas os metadados (<code>FaceAuditProjectMetadata</code>): "
        "nome, id, descrição, timestamps e versão.",
        S["body"]
    ))

    story.append(Paragraph("Estrutura de subpastas do projeto", S["h3"]))
    story.append(Paragraph(
        "Criada por <code>LocalProjectArtifactStorage.create_project_structure()</code> "
        "a partir do enum <code>ProjectFolder</code>:",
        S["body"]
    ))

    code_block(story, S, [
        "meu_projeto/",
        "  project.fa              ← metadados do projeto (JSON)",
        "  definitions/            ← LRStrategyDefinition (JSON)",
        "  models/                 ← LRStrategyTrainedModel (pickle)",
        "  datasets/               ← EmbeddingDataset (parquet)",
        "  pipelines/              ← pipelines treinados (pickle)",
        "  configs/                ← configs de split, mining, pipeline",
        "  results/                ← scores, métricas, logs",
        "  graphics/               ← visualizações geradas",
        "  checkpoints/            ← checkpoints de execução",
    ])

    # 7.3
    story.append(Paragraph("7.3 Fluxo de Montagem de um Projeto", S["h2"]))

    steps = [
        ("1.", "O perito abre a interface e clica em 'Novo Projeto'."),
        ("2.", "FaceAuditProjectService.create(folder, name) cria a pasta, "
               "o arquivo project.fa e a estrutura de subpastas."),
        ("3.", "O perito navega na Biblioteca Global e seleciona: "
               "um dataset (parquet), uma split config e opcionalmente um template de estratégia."),
        ("4.", "O sistema chama copy_artifact_to_project() para cada seleção, "
               "copiando os arquivos para as subpastas do projeto."),
        ("5.", "Se um template foi selecionado, os recursos da LRStrategyDefinition "
               "são vinculados aos artefatos copiados (artifact_ref preenchido)."),
        ("6.", "O perito ajusta as instruções pela interface (parâmetros, ordem, "
               "adição ou remoção de etapas)."),
        ("7.", "StrategyDefinitionRepository.save_definition() persiste o plano final em definitions/."),
        ("8.", "O perito inicia a execução: StrategyExecutorFactory.build() + "
               "executor.execute(definition) → LRStrategyTrainedModel."),
        ("9.", "O modelo treinado é salvo via StrategyTrainedModelRepository.save_model() em models/."),
    ]

    for num, desc in steps:
        story.append(Paragraph(f"{num} {desc}", S["bullet"]))

    story.append(Spacer(1, 4))
    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 8 — MODELOS DE SIMILARIDADE E CALIBRAÇÃO
# ---------------------------------------------------------------------------

def build_s8(story, S):
    section_banner(story, S, "8. Modelos de Similaridade e Calibração")

    story.append(Paragraph(
        "Uma das decisões arquiteturais mais importantes foi a <b>separação entre "
        "similaridade e calibração</b>. Na arquitetura antiga, o modelo KDE era "
        "monolítico: recebia embeddings e produzia LRs diretamente. Na nova "
        "arquitetura, o processo é dividido em dois estágios independentes:",
        S["body"]
    ))

    code_block(story, S, [
        "EmbeddingDataset",
        "      │",
        "      ▼",
        "SimilarityModelProtocol.fit(dataset)",
        "SimilarityModelProtocol.score_matrix(dataset_A, dataset_B)",
        "      │",
        "      ▼  (scores brutos — distâncias coseno ou log-verossimilhanças PLDA)",
        "SimilarityScoreDataset  { scores, labels }",
        "      │",
        "      ▼",
        "CalibrationModelProtocol.fit(scores, labels)",
        "CalibrationModelProtocol.calibrate_scores(scores)",
        "      │",
        "      ▼  (Likelihood Ratios)",
        "LR values",
    ])

    story.append(Paragraph("Modelos de Similaridade disponíveis", S["h2"]))

    two_col_table(story, S,
        ["Modelo", "Comportamento"],
        [
            ["CosineSimilarityModel",
             "Computa similaridade cosseno entre embeddings. Não requer treino. "
             "Adequado para cenários sem dados de treino disponíveis."],
            ["PLDASimilarityModel",
             "PLDA (Probabilistic Linear Discriminant Analysis) via SpeechBrain. "
             "Treina um modelo generativo nos embeddings de treino. Produz "
             "log-verossimilhanças que são scores brutos mais calibrados estatisticamente."],
        ],
    )

    story.append(Paragraph("Modelos de Calibração disponíveis", S["h2"]))

    two_col_table(story, S,
        ["Modelo", "Comportamento"],
        [
            ["LogisticRegressionCalibrationModel",
             "Regressão logística sobre os scores brutos. Treino rápido e robusto. "
             "Adequado para grandes volumes de pares de treino."],
            ["KDECalibrationModel",
             "Dois kernels KDE (genuínos e impostores) com Fisher transform + "
             "GridSearch adaptativo de bandwidth. Usa cross-validation para "
             "escolher o melhor bandwidth. Mais flexível que LogReg para "
             "distribuições não-gaussianas."],
        ],
    )

    story.append(Paragraph(
        "A conexão entre os dois estágios é feita por <code>SimilarityScoreDataset</code>, "
        "que encapsula os scores brutos e os labels correspondentes "
        "(1 = mesmo indivíduo, 0 = indivíduos diferentes).",
        S["body"]
    ))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 9 — PIPELINE DE TRANSFORMAÇÃO
# ---------------------------------------------------------------------------

def build_s9(story, S):
    section_banner(story, S, "9. Pipeline de Transformação de Embeddings")

    story.append(Paragraph(
        "Antes de calcular scores de similaridade, os embeddings frequentemente "
        "precisam ser normalizados e projetados. O <code>EmbeddingTransformationPipeline</code> "
        "encapsula uma sequência ordenada de transformadores.",
        S["body"]
    ))

    story.append(Paragraph("Transformadores disponíveis", S["h2"]))

    two_col_table(story, S,
        ["Transformador", "Função"],
        [
            ["CoralTransformation",
             "Alinhamento de covariância entre domínios (CORAL). "
             "Reduz shift de domínio entre dataset de referência e alvo."],
            ["WhiteningTransformation",
             "Branqueamento dos embeddings (PCA whitening). "
             "Decorrelaciona as dimensões e normaliza a variância."],
            ["LdaTransformation",
             "Análise Discriminante Linear. "
             "Maximiza a separação entre identidades no espaço projetado."],
            ["CenteringTransformation",
             "Subtrai a média do conjunto de treino. "
             "Pré-requisito para Whitening e LDA."],
            ["L2Normalization",
             "Normaliza cada vetor para norma unitária. "
             "Melhora a estabilidade numérica do cosseno."],
            ["IdentityAggregation",
             "Agrega múltiplos embeddings de uma mesma identidade "
             "(média ou outro método). Útil para datasets com múltiplas imagens por pessoa."],
        ],
    )

    story.append(Paragraph("Modos de pipeline", S["h2"]))

    two_col_table(story, S,
        ["Modo", "Comportamento"],
        [
            ["LINEAR",
             "Treina o pipeline usando apenas o dataset de referência. "
             "Aplica a todos os datasets. O caso mais simples."],
            ["DUAL",
             "Permite que etapas selecionadas (CORAL, Whitening, LDA) usem "
             "também o dataset alvo durante o treino. "
             "PipelineTrainingPolicy controla quais etapas podem usar o alvo "
             "e se o estado da branch de referência é copiado para a branch alvo."],
        ],
    )

    story.append(Paragraph(
        "Os transformadores são a <b>camada matemática compartilhada</b> entre as "
        "duas arquiteturas. O <code>TransformationPipeline</code> antigo e o "
        "<code>EmbeddingTransformationPipeline</code> novo instanciam as mesmas "
        "6 classes. A diferença arquitetural está no pipeline orquestrador, "
        "não nos transformadores.",
        S["body"]
    ))

    # 9.1
    story.append(Paragraph("9.1 PipelineTrajectoryService — Visualização 3D", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/services/pipeline_trajectory_service.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Conecta um <code>EmbeddingTransformationPipeline</code> já treinado à "
        "infraestrutura de visualização existente (<code>EmbeddingMathService</code> + "
        "<code>TrajectoryPlotterService</code>), sem modificar nenhum desses componentes. "
        "Executa o pipeline passo a passo, constrói frames de alta dimensão e delega "
        "a projeção 3D via Joint PCA + estabilização de Procrustes.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Método", "Descrição"],
        [
            ["compute_trajectory(pipeline, source_dataset, target_dataset=None)",
             "Executa o pipeline passo a passo e retorna TransformationTrajectoryData "
             "com um TransformationStep3D por passo (incluindo frame 0 'Raw'). "
             "Suporta modos LINEAR (target_dataset ignorado) e DUAL (sincroniza "
             "source_steps x target_steps por transformation_type)."],
            ["export_plots(trajectory_data, base_path_template, logger=print)",
             "Exporta cada passo como PNG usando backend Matplotlib Agg (headless). "
             "Template de path aceita {step} como índice inteiro. "
             "Restaura o backend original no finally. Retorna List[Path]."],
        ],
    )

    callout_box(story, S,
        "Convenção DUAL: probe → ramo source_steps, reference → ramo target_steps. "
        "A sincronização é por subsequência: para cada src_step, avança o ponteiro "
        "target apenas quando transformation_type.value coincide.",
        bg=BRAND_LIGHT, border=BRAND_MED,
    )

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 10 — PERSISTÊNCIA
# ---------------------------------------------------------------------------

def build_s10(story, S):
    section_banner(story, S, "10. Protocolo de Persistência (ArtifactStorageProtocol)")

    story.append(Paragraph(
        "Toda operação de leitura e escrita de artefatos em disco é feita através "
        "do protocolo <code>ArtifactStorageProtocol</code>. Há duas implementações concretas:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Implementação", "Uso"],
        [
            ["LocalProjectArtifactStorage",
             "Gerencia os artefatos dentro de um projeto específico. "
             "Todos os paths são relativos à raiz do projeto (Pendrive Paradigm). "
             "Usa StrategyFileType para mapear tipos de artefato em subpastas."],
            ["GlobalLibraryStorage",
             "Gerencia a biblioteca global de recursos reutilizáveis. "
             "Usa GlobalLibraryFileType. "
             "Aceita apenas os tipos globais: GLOBAL_DATASET, SPLIT_CONFIG, "
             "MINING_CONFIG, STRATEGY_TEMPLATE. "
             "Lança ValueError para tipos de projeto."],
        ],
    )

    story.append(Paragraph("Interface do protocolo", S["h2"]))

    code_block(story, S, [
        "class ArtifactStorageProtocol(Protocol):",
        "",
        "    def save_artifact(",
        "        self, artifact_type: StrategyArtifactType, data: Any, **kwargs",
        "    ) -> StrategyArtifactReference: ...",
        "",
        "    def copy_artifact_to_project(",
        "        self, source_path: Path, artifact_type: StrategyArtifactType, **kwargs",
        "    ) -> StrategyArtifactReference: ...",
        "",
        "    def resolve_artifact(",
        "        self, artifact_ref: StrategyArtifactReference",
        "    ) -> Optional[Path]: ...",
        "",
        "    def list_artifacts(",
        "        self, artifact_type: Optional[StrategyArtifactType] = None",
        "    ) -> List[StrategyArtifactReference]: ...",
    ])

    story.append(Paragraph(
        "<code>StrategyArtifactReference</code> é o objeto de referência lógica a "
        "um artefato persistido. Contém <code>path</code> (sempre relativo), "
        "<code>artifact_type</code> e <code>hash</code> (SHA-256 calculado por HashService "
        "no momento da gravação). O hash é a base da verificação de integridade.",
        S["body"]
    ))

    # 10.1
    story.append(Paragraph("10.1 Verificação de Integridade (LocalProjectArtifactStorage)", S["h2"]))
    story.append(Paragraph(
        "A <code>LocalProjectArtifactStorage</code> expõe o método "
        "<code>verify_integrity(refs)</code> que não faz parte do protocolo — é uma "
        "capacidade exclusiva da implementação local, chamada na abertura de projetos.",
        S["body"]
    ))

    two_col_table(story, S,
        ["ArtifactIntegrityStatus", "Significado"],
        [
            ["OK",
             "Arquivo presente em disco e hash SHA-256 atual confere com o registrado "
             "em StrategyArtifactReference no momento da gravação."],
            ["MODIFIED",
             "Arquivo presente, mas hash atual difere do registrado. "
             "Indica possível adulteração, substituição acidental ou recriação do arquivo."],
            ["MISSING",
             "Arquivo não encontrado no path esperado (project_root / ref.path). "
             "Indica deleção ou movimentação indevida."],
            ["HASH_UNAVAILABLE",
             "Arquivo presente, mas a referência não tem hash registrado. "
             "Ocorre em artefatos salvos antes da implementação de integridade, "
             "ou em referências criadas sem cálculo de hash. Arquivo existe, "
             "conteúdo não pode ser auditado."],
        ],
    )

    story.append(Paragraph(
        "O resultado de cada artefato é um <code>ArtifactIntegrityResult</code> "
        "(frozen dataclass em <code>faceaudit/models/storage/artifact_integrity_result.py</code>) "
        "com campos: <code>artifact_ref</code>, <code>status</code>, "
        "<code>expected_hash</code>, <code>actual_hash</code>, <code>abs_path</code>. "
        "O método <code>describe()</code> gera uma mensagem legível, incluindo os dois "
        "hashes lado a lado no caso MODIFIED.",
        S["body"]
    ))

    callout_box(story, S,
        "HASH_UNAVAILABLE não é tratado como problema (is_problem = False) — o arquivo "
        "existe e pode ser usado. Apenas MODIFIED e MISSING ativam has_integrity_problems "
        "no FaceAuditProject. Isso evita bloqueios em projetos antigos ou criados sem hash.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 11 — SISTEMA DE VALIDAÇÃO
# ---------------------------------------------------------------------------

def build_s11(story, S):
    section_banner(story, S, "11. Sistema de Validação")

    # 11.1
    story.append(Paragraph("11.1 Dois Mecanismos Complementares", S["h2"]))
    story.append(Paragraph(
        "A nova arquitetura possui dois mecanismos de validação que atuam em camadas distintas:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Mecanismo", "Características"],
        [
            ["Validações internas dos executores",
             "Hard-coded em _validate_instruction_specifics() de cada executor concreto. "
             "Verificam: tipo correto da instrução, campos obrigatórios preenchidos, "
             "recursos existentes no contexto, tipos corretos dos payloads. "
             "Automáticas e invisíveis para o perito — sempre executam."],
            ["ValidationRuleProtocol (plugável)",
             "Regras reutilizáveis registradas no StrategyExecutor por rule_id. "
             "Referenciadas na LRStrategyDefinition (antes de qualquer instrução) ou "
             "em instruções específicas (antes dessa instrução). "
             "Permitem políticas do laboratório como limiares mínimos de dados e "
             "restrições de isolamento entre datasets."],
        ],
    )

    callout_box(story, S,
        "As regras plugáveis não precisam ser copiadas para cada projeto — elas vivem no "
        "código da aplicação e são referenciadas pelo rule_id. Templates de estratégia na "
        "biblioteca global já carregam os rule_ids embutidos. Ao instanciar o template, "
        "as políticas do laboratório vêm junto automaticamente.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    # 11.2
    story.append(Paragraph("11.2 ValidationRuleBase e Cache de Stats", S["h2"]))
    story.append(Paragraph(
        "Todas as regras concretas estendem <code>ValidationRuleBase</code> "
        "(<code>faceaudit/validation_rules/validation_rule_base.py</code>). "
        "A classe base fornece dois grupos de facilidades:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Facilidade", "Comportamento"],
        [
            ["Campos do protocolo via construtor",
             "rule_id, name, description e default_severity são passados ao super().__init__() "
             "em vez de serem declarados como @property em cada subclasse."],
            ["_get_or_compute_stats(resource_id, context)",
             "Busca \"{resource_id}_stats\" no contexto (cache hit). "
             "Se não existe, carrega o payload do recurso, computa DatasetStats via "
             "DatasetStatsService e armazena no contexto como DATASET_STATS resource. "
             "Custo pago uma vez por dataset por execução — todas as regras e executores "
             "subsequentes reutilizam o mesmo objeto."],
        ],
    )

    story.append(Paragraph(
        "<code>DatasetStats</code> foi estendido com o campo <code>embedding_dim: int</code> "
        "para suportar a Regra de Fisher. <code>DatasetStatsService</code> ganhou o método "
        "<code>compute_from_embedding_dataset()</code> — a versão anterior só operava sobre "
        "<code>FaceDataset</code>.",
        S["body"]
    ))

    # 11.3
    story.append(Paragraph("11.3 Regras Concretas Implementadas", S["h2"]))

    rules = [
        ("DatasetIsolationRule", "dataset_isolation_rule.py",
         "CRITICAL",
         "Verifica que dois datasets são completamente disjuntos em identidades. "
         "Uso: garantir que dataset de calibração não vaze identidades do treino. "
         "Parâmetros: dataset_a_resource_id, dataset_b_resource_id."),
        ("DatasetOverlapRequiredRule", "dataset_overlap_required_rule.py",
         "CRITICAL / WARNING",
         "Verifica que dois datasets compartilham overlap mínimo de identidades. "
         "CRITICAL se abaixo de min_overlap; WARNING se proporção < warn_below_ratio. "
         "Uso: Probe e Reference devem ter identidades em comum para formar pares genuínos. "
         "Parâmetros: dataset_a_resource_id, dataset_b_resource_id, min_overlap (default 1), "
         "warn_below_ratio (default 10%)."),
        ("FisherRule", "fisher_rule.py",
         "CRITICAL / WARNING",
         "Regra de Fisher para PLDA: n_identidades > embedding_dim. "
         "CRITICAL se singular (emb_dim >= n_ids); WARNING se margem estreita "
         "(n_ids <= emb_dim * (1 + safety_margin_ratio)). "
         "Parâmetros: dataset_resource_id, safety_margin_ratio (default 20%)."),
        ("MinCalibrationPairsRule", "min_calibration_pairs_rule.py",
         "CRITICAL / WARNING",
         "Verifica mínimo de pares genuínos e impostores para treino do calibrador. "
         "CRITICAL se abaixo dos mínimos absolutos; WARNING se proporção genuínos/total "
         "muito baixa (dataset fortemente desbalanceado). "
         "Parâmetros: scores_resource_id, min_genuines (100), min_impostors (500), "
         "warn_ratio_threshold (2%)."),
        ("HoldoutLeakageRule", "holdout_leakage_rule.py",
         "CRITICAL",
         "Verifica que todos os datasets de hold-out são disjuntos de todos os datasets "
         "de treino. Aceita listas de resource_ids para cruzamento múltiplo. "
         "Parâmetros: holdout_resource_ids: List[str], train_resource_ids: List[str]."),
    ]

    for rname, rfile, sev, desc in rules:
        sev_color = colors.HexColor("#FDECEA") if "CRITICAL" in sev else BRAND_LIGHT
        data = [
            [Paragraph(f"<b>{rname}</b>", S["td_code"]),
             Paragraph(rfile, S["td_small"]),
             Paragraph(sev, S["td_small"])],
            [Paragraph(desc, S["td"]), "", ""],
        ]
        t = Table(data, colWidths=[INNER_W * 0.38, INNER_W * 0.32, INNER_W * 0.30])
        t.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0),  sev_color),
            ("BACKGROUND",   (0, 1), (-1, 1),  WHITE),
            ("SPAN",         (0, 1), (-1, 1)),
            ("BOX",          (0, 0), (-1, -1), 0.5, GRAY_MED),
            ("INNERGRID",    (0, 0), (-1, 0),  0.5, GRAY_MED),
            ("LEFTPADDING",  (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 4))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 12 — CICLO DE VIDA
# ---------------------------------------------------------------------------

def build_s12(story, S):
    section_banner(story, S, "12. Ciclo de Vida de um Projeto FaceAudit")

    story.append(Paragraph("12.1 Fluxo de Treinamento (end-to-end)", S["h2"]))
    story.append(Paragraph(
        "O <code>FaceAuditProjectService</code> é o ponto de entrada para todas as "
        "operações relacionadas ao ciclo de vida de um projeto.",
        S["body"]
    ))

    code_block(story, S, [
        "# Criar novo projeto",
        "project = FaceAuditProjectService.create(",
        "    folder=Path('/perícias/caso_42'),",
        "    name='Caso 42 — Reconhecimento Passaporte',",
        ")",
        "",
        "# Abrir projeto existente",
        "project = FaceAuditProjectService.open(",
        "    fa_file=Path('/perícias/caso_42/project.fa')",
        ")",
        "",
        "# Verificar se uma pasta é um projeto FaceAudit",
        "is_proj = FaceAuditProjectService.is_project_folder(folder)",
        "",
        "# Encontrar o arquivo .fa dentro de uma pasta",
        "fa_path = FaceAuditProjectService.find_project(folder)",
    ])

    story.append(Paragraph(
        "O objeto <code>FaceAuditProject</code> retornado agrega as três "
        "peças principais do ciclo de vida do projeto:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Propriedade / Método", "Papel"],
        [
            ["project.metadata",
             "FaceAuditProjectMetadata — nome, id, descrição, timestamps"],
            ["project.storage",
             "LocalProjectArtifactStorage — leitura, escrita e verificação de artefatos"],
            ["project.definition_repository",
             "StrategyDefinitionRepository — save/load de LRStrategyDefinition"],
            ["project.model_repository",
             "StrategyTrainedModelRepository — save/load de LRStrategyTrainedModel"],
            ["project.fa_file",
             "Path absoluto do arquivo project.fa"],
            ["project.root",
             "Path absoluto da raiz do projeto"],
            ["project.integrity_results",
             "List[ArtifactIntegrityResult] da última verificação; None se nunca executada"],
            ["project.has_integrity_problems",
             "True se algum artefato está MISSING ou MODIFIED na última verificação"],
            ["project.check_integrity()",
             "Carrega todas as definições, coleta artifact_refs e delega a "
             "storage.verify_integrity(). Atualiza integrity_results. "
             "Pode ser chamado a qualquer momento, não apenas na abertura."],
        ],
    )

    story.append(Paragraph("Abertura com verificação de integridade", S["h2"]))
    story.append(Paragraph(
        "Por padrão, <code>open()</code> executa <code>check_integrity()</code> "
        "automaticamente antes de retornar o projeto. O parâmetro "
        "<code>verify_integrity=False</code> desativa a verificação quando não necessária.",
        S["body"]
    ))

    code_block(story, S, [
        "# Abertura padrão — verificação automática",
        "project = FaceAuditProjectService.open('/perícias/caso_42/project.fa')",
        "",
        "if project.has_integrity_problems:",
        "    for r in project.integrity_results:",
        "        if r.is_problem:",
        "            print(r.describe())",
        "    # ex: [MODIFICADO] datasets/treino.parquet",
        "    #       Esperado : a3f1...",
        "    #       Atual    : 9c2b...",
        "",
        "# Re-verificar a qualquer momento (ex: antes de treinar)",
        "results = project.check_integrity()",
        "",
        "# Abertura sem verificação (ex: ao criar projeto logo antes)",
        "project = FaceAuditProjectService.open(fa_file, verify_integrity=False)",
    ])

    story.append(Paragraph("Fluxo end-to-end completo", S["h2"]))
    story.append(Paragraph(
        "O trecho abaixo mostra o fluxo real desde a criação do projeto até a "
        "obtenção do modelo treinado. O passo crítico é passar <code>project.storage</code> "
        "para a factory — isso ativa a hidratação automática dos recursos antes da execução.",
        S["body"]
    ))

    code_block(story, S, [
        "# 1. Criar projeto",
        "project = FaceAuditProjectService.create('/perícias/caso_42', 'Caso 42')",
        "",
        "# 2. Copiar recursos da biblioteca global para o projeto",
        "ref_ds = project.storage.copy_artifact_to_project(",
        "    source_path='/biblioteca/embeddings_treino.parquet',",
        "    artifact_type=StrategyArtifactType.EMBEDDING_DATASET,",
        "    dataset_id='ds_treino',",
        ")",
        "ref_cfg = project.storage.copy_artifact_to_project(",
        "    source_path='/biblioteca/pipeline_lda.json',",
        "    artifact_type=StrategyArtifactType.PIPELINE_CONFIG,",
        "    config_id='cfg_lda',",
        ")",
        "",
        "# 3. Montar a definição da estratégia",
        "definition = LRStrategyDefinition(strategy_id='ST_01', name='LDA+PLDA')",
        "definition.add_resource(StrategyResource(",
        "    'ds_treino', 'Dataset Treino',",
        "    StrategyResourceType.DATASET, artifact_ref=ref_ds",
        "))",
        "definition.add_resource(StrategyResource(",
        "    'cfg_lda', 'Config Pipeline LDA',",
        "    StrategyResourceType.PIPELINE_CONFIG, artifact_ref=ref_cfg",
        "))",
        "# ... adicionar instruções ...",
        "",
        "# 4. Salvar a definição",
        "project.definition_repository.save(definition)",
        "",
        "# 5. Executar — artifact_storage ativa hidratação automática dos payloads",
        "executor = StrategyExecutorFactory.build(artifact_storage=project.storage)",
        "trained_model = executor.execute(definition)",
        "",
        "# 6. Salvar modelo treinado",
        "project.model_repository.save(trained_model)",
    ])

    story.append(Paragraph(
        "Nota: quando a estratégia é carregada do repositório "
        "(<code>project.definition_repository.load(id)</code>), os recursos têm "
        "<code>payload=None</code> e <code>artifact_ref</code> preenchido. "
        "O <code>StrategyResourceLoaderService</code> — instanciado internamente "
        "pelo executor quando <code>artifact_storage</code> é fornecido — "
        "carrega automaticamente os payloads antes da validação e execução das instruções.",
        S["body"]
    ))

    # 12.2
    story.append(Paragraph("12.2 Fluxo de Inferência (StrategyInferenceService)", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/services/strategy_inference_service.py</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Executa o fluxo completo de comparação forense a partir de um "
        "<code>LRStrategyTrainedModel</code> já carregado (pickle — todos os "
        "modelos treinados prontos em memória). Recebe <code>StrategyResource</code> "
        "para probe e reference: se o payload estiver None (recurso no disco), "
        "hidrata via <code>StrategyResourceLoaderService</code> usando o storage do projeto.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Etapa", "Detalhe"],
        [
            ["1. Hidratação",
             "probe_resource e reference_resource são hidratados se payload=None. "
             "Exige artifact_storage no construtor do serviço."],
            ["2. Extração dos modelos",
             "Pipeline (TRAINED_PIPELINE), modelo de similaridade "
             "(TRAINED_SIMILARITY_MODEL) e modelo de calibração "
             "(TRAINED_CALIBRATION_MODEL) são localizados em "
             "trained_model.trained_resources por resource_type. "
             "O modelo de calibração é opcional."],
            ["3. Transformação",
             "LINEAR: pipeline.transform() nos dois datasets. "
             "DUAL: pipeline.transform_source(probe) + pipeline.transform_target(reference). "
             "Se não há pipeline, datasets são usados diretamente."],
            ["4. Score matrix",
             "similarity_model.score_matrix(probe_matrix, ref_matrix) → "
             "matriz (n_probe × n_reference) de scores brutos."],
            ["5. Calibração",
             "calibration_model.calibrate_scores(scores) → log-LRs. "
             "Armazenados em SimilarityScoreSample.metadata['log_lr']. "
             "raw_score sempre preserva o score bruto."],
            ["6. Resultado",
             "SimilarityScoreDataset com um SimilarityScoreSample por par. "
             "same_person inferido automaticamente das identidades (auto-ground-truth)."],
        ],
    )

    code_block(story, S, [
        "# Carregar modelo treinado da estratégia (pickle — totalmente hidratado)",
        "trained_model = project.model_repository.load(strategy_id='ST_01')",
        "",
        "# Construir recursos de probe e reference",
        "probe_res = StrategyResource(",
        "    resource_id='probe', name='Probe Dataset',",
        "    resource_type=StrategyResourceType.DATASET,",
        "    artifact_ref=ref_probe,  # StrategyArtifactReference do projeto",
        ")",
        "ref_res = StrategyResource(",
        "    resource_id='reference', name='Reference Dataset',",
        "    resource_type=StrategyResourceType.DATASET,",
        "    artifact_ref=ref_standard,",
        ")",
        "",
        "# Executar inferência",
        "svc = StrategyInferenceService(artifact_storage=project.storage)",
        "result = svc.run(trained_model, probe_res, ref_res)",
        "",
        "# Para validação forense — extrair log-LRs por classe",
        "genuine_log_lrs = [s.metadata['log_lr'] for s in result if s.same_person]",
        "impostor_log_lrs = [s.metadata['log_lr'] for s in result if not s.same_person]",
        "",
        "# Para perícia — LR consolidado de todos os pares",
        "all_log_lrs = [s.metadata.get('log_lr', s.raw_score) for s in result]",
    ])

    callout_box(story, S,
        "Isolamento por estratégia garantido: model_repository.load(strategy_id) "
        "carrega sempre o modelo correto daquela estratégia — pipeline, modelo de "
        "similaridade e calibrador são os treinados especificamente para ela. "
        "Um projeto com múltiplas estratégias executa cada uma com seus próprios modelos.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 13 — DECISÕES DE DESIGN
# ---------------------------------------------------------------------------

def build_s13(story, S):
    section_banner(story, S, "13. Decisões de Design e Princípios Arquiteturais")

    decisions = [
        (
            "Herança de Protocol proibida em @dataclass",
            "Em Python, herança explícita de Protocol é desnecessária — a tipagem "
            "estrutural garante compatibilidade sem declaração. Além disso, as @property "
            "declaradas no Protocol bloqueavam o __init__ gerado pelo @dataclass (sem setter "
            "→ AttributeError). LRStrategyDefinition e LRStrategyTrainedModel herdam apenas "
            "de JSONSerializableMixin. O isinstance() continua funcionando via @runtime_checkable "
            "(verifica presença de atributos, não hierarquia). Regra: nunca adicionar herança "
            "de Protocol em @dataclass concreto da nova arquitetura.",
        ),
        (
            "StrategyResource como unidade de troca",
            "Toda informação entre instruções transita como StrategyResource no contexto. "
            "Os nomes são declarados nos DTOs (_sync_declared_io). Isso torna o grafo de "
            "dependências explícito e inspecionável sem executar o código.",
        ),
        (
            "from_dict() como dispatcher polimórfico",
            "StrategyInstructionDefinition.from_dict() lê instruction_type e hidrata o "
            "DTO concreto correto. Isso garante que uma estratégia survive a serialização "
            "JSON com toda a informação de tipo preservada.",
        ),
        (
            "ArtifactStorageService é adaptador transitório",
            "O ArtifactStorageService existente (faceaudit/storage/artifact_storage_service.py) "
            "é um wrapper para a arquitetura antiga. Os repositórios da nova arquitetura aceitam "
            "ArtifactStorageProtocol diretamente. Não usar ArtifactStorageService em código novo.",
        ),
        (
            "Repositórios usam StrategyFileType para paths",
            "Os repositórios derivam o path esperado via StrategyFileType.build_path() — "
            "não guardam strings hardcoded. Isso garante que mudanças na convenção de nomes "
            "se propagam automaticamente.",
        ),
        (
            "Hash de artefatos para verificação de integridade",
            "Todo artefato salvo tem seu SHA-256 calculado por HashService e armazenado em "
            "StrategyArtifactReference.hash. Isso permite detectar corrupção ou substituição "
            "acidental de arquivos.",
        ),
        (
            "Separação similaridade / calibração",
            "SimilarityModelProtocol produz scores brutos; CalibrationModelProtocol converte "
            "para LRs. A separação permite combinar qualquer modelo de similaridade com qualquer "
            "modelo de calibração sem alterar o fluxo de execução.",
        ),
    ]

    for title, body in decisions:
        story.append(KeepTogether([
            Paragraph(title, S["h3"]),
            Paragraph(body, S["body"]),
            Spacer(1, 4),
        ]))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 14 — EXTENSIBILIDADE
# ---------------------------------------------------------------------------

def build_s14(story, S):
    section_banner(story, S, "14. Extensibilidade — Como Adicionar Novos Componentes")

    story.append(Paragraph(
        "O projeto possui skills (comandos) em <code>.claude/commands/</code> que "
        "automatizam a criação de novos componentes seguindo os padrões estabelecidos.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Comando", "O que cria"],
        [
            ["/new-instruction <Nome>",
             "Enum + DTO + dispatch + executor + registro na factory"],
            ["/new-similarity-model <Nome>",
             "Enum + config DTO + implementação + integração no executor"],
            ["/new-calibration-model <Nome>",
             "Enum + config DTO + implementação + integração no executor"],
            ["/new-validation-rule <Nome>",
             "Nova regra em faceaudit/validation_rules/ estendendo ValidationRuleBase; "
             "dois templates: regra de dataset (usa _get_or_compute_stats) e regra de outro recurso"],
            ["/new-transformation <Nome>",
             "StepType + config DTO + TransformationType + classe transformadora + registro no pipeline"],
        ],
    )

    story.append(Paragraph("Checklist manual para novo tipo de instrução", S["h2"]))

    steps = [
        "Adicionar valor ao enum StrategyInstructionType",
        "Criar DTO em faceaudit/models/instructions/ herdando StrategyInstructionDefinition",
        "Implementar __post_init__ com instruction_type + _sync_declared_io()",
        "Implementar from_dict() hidratando enums aninhados",
        "Adicionar bloco de dispatch em StrategyInstructionDefinition.from_dict()",
        "Criar executor em faceaudit/executors/instructions/ herdando StrategyInstructionExecutorBase",
        "Sobrescrever execute() e _validate_instruction_specifics()",
        "Registrar executor em StrategyExecutorFactory.build_instruction_executor_registry()",
    ]
    for i, s in enumerate(steps):
        story.append(Paragraph(f"{i+1}. {s}", S["bullet"]))

    story.append(Paragraph("Checklist manual para nova regra de validação", S["h2"]))

    rule_steps = [
        "Criar arquivo em faceaudit/validation_rules/ herdando ValidationRuleBase",
        "Passar rule_id, name, description, default_severity para super().__init__()",
        "Implementar validate(target, context) → ValidationResult",
        "Para regras de dataset: usar _get_or_compute_stats(resource_id, context)",
        "Prefixar todas as mensagens com [{self.rule_id}]",
        "Chamar result.mark_ok() quando não há erros nem avisos",
        "Registrar no StrategyExecutorFactory.build(validation_rules={...})",
        "Referenciar o rule_id na LRStrategyDefinition ou instrução correspondente",
    ]
    for i, s in enumerate(rule_steps):
        story.append(Paragraph(f"{i+1}. {s}", S["bullet"]))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 15 — ESTRUTURA DE PASTAS
# ---------------------------------------------------------------------------

def build_s15(story, S):
    section_banner(story, S, "15. Estrutura de Pastas")

    three_col_table(story, S,
        ["Pasta", "Status", "Conteúdo"],
        [
            ["faceaudit/contexts/", "Nova arch",
             "StrategyExecutionContext — barramento de estado"],
            ["faceaudit/executors/", "Nova arch",
             "StrategyExecutor + InstructionExecutorRegistry"],
            ["faceaudit/executors/instructions/", "Nova arch",
             "Um executor concreto por tipo de instrução"],
            ["faceaudit/factories/", "Nova arch",
             "StrategyExecutorFactory — ponto de montagem da engine"],
            ["faceaudit/repositories/", "Nova arch",
             "StrategyDefinitionRepository + StrategyTrainedModelRepository"],
            ["faceaudit/storage/", "Nova arch",
             "LocalProjectArtifactStorage (save/copy/resolve/list + verify_integrity) "
             "+ GlobalLibraryStorage"],
            ["faceaudit/services/project_service.py", "Nova arch",
             "FaceAuditProjectService — ciclo de vida de projetos"],
            ["faceaudit/services/strategy_resource_loader_service.py", "Nova arch",
             "StrategyResourceLoaderService — hidrata payloads de recursos a partir de artifact_ref"],
            ["faceaudit/services/strategy_inference_service.py", "Nova arch",
             "StrategyInferenceService — inferência forense: pipeline + score_matrix + calibração"],
            ["faceaudit/services/pipeline_trajectory_service.py", "Nova arch",
             "PipelineTrajectoryService — visualização 3D da trajetória de transformação do pipeline"],
            ["faceaudit/models/calibration/", "Nova arch",
             "Configs e implementações de modelos de calibração"],
            ["faceaudit/models/instructions/", "Nova arch",
             "DTOs de instrução"],
            ["faceaudit/models/pipeline/", "Nova arch",
             "PipelineTrainingPolicy, PipelineApplicationPolicy"],
            ["faceaudit/models/similarity/", "Nova arch",
             "Configs e implementações de modelos de similaridade"],
            ["faceaudit/models/storage/", "Nova arch",
             "StrategyArtifactReference — referência lógica com path relativo + hash SHA-256; "
             "ArtifactIntegrityResult — resultado por artefato da verificação de integridade"],
            ["faceaudit/enums/artifact_integrity_status.py", "Nova arch",
             "ArtifactIntegrityStatus — OK / MODIFIED / MISSING / HASH_UNAVAILABLE"],
            ["faceaudit/models/strategy/", "Nova arch",
             "LRStrategyDefinition, LRStrategyTrainedModel, StrategyResource"],
            ["faceaudit/models/validation/", "Nova arch",
             "ValidationResult"],
            ["faceaudit/models/project/", "Nova arch",
             "FaceAuditProjectMetadata"],
            ["faceaudit/validation_rules/", "Nova arch",
             "ValidationRuleBase + 5 regras concretas: DatasetIsolationRule, "
             "DatasetOverlapRequiredRule, FisherRule, MinCalibrationPairsRule, HoldoutLeakageRule"],
            ["faceaudit/transformators/", "Compartilhada",
             "6 transformadores de embeddings — usados por ambas as archs"],
            ["faceaudit/pipelines/EmbeddingTransformationPipeline", "Nova arch",
             "Pipeline LINEAR + DUAL com PipelineTrainingPolicy"],
            ["faceaudit/protocols/", "Mista",
             "Contratos de tipagem estrutural — ver seção 5"],
            ["faceaudit/manager/", "Arch antiga",
             "Managers stateful — não usar em código novo"],
            ["faceaudit/controllers/", "Arch antiga",
             "Controllers hard-coded — não usar em código novo"],
            ["faceaudit/strategies/", "Arch antiga",
             "KDEStrategy, PLDAStrategy, PLDASnormStrategy"],
            ["interface/", "Arch antiga",
             "Frontend — ainda não migrado para nova arch"],
        ],
        col_widths=[INNER_W*0.38, INNER_W*0.16, INNER_W*0.46],
    )

    story.append(Spacer(1, 20))
    rule(BRAND_MED, 2)
    story.append(rule(BRAND_MED, 2))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Documento gerado automaticamente a partir do código-fonte do FaceAudit · "
        "Versão 1.2 · Abril 2026",
        S["caption"]
    ))


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN + 0.5 * cm,
        title="FaceAudit — Nova Arquitetura Declarativa",
        author="FaceAudit",
        subject="Documento Técnico de Arquitetura",
    )

    S = build_styles()
    story = []

    build_cover(story, S)
    build_toc(story, S)
    build_s1(story, S)
    build_s2(story, S)
    build_s3(story, S)
    build_s4(story, S)
    build_s5(story, S)
    build_s6(story, S)
    build_s7(story, S)
    build_s8(story, S)
    build_s9(story, S)
    build_s10(story, S)
    build_s11(story, S)
    build_s12(story, S)
    build_s13(story, S)
    build_s14(story, S)
    build_s15(story, S)

    doc.build(story)
    print(f"PDF gerado: {output_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(
        os.path.dirname(__file__),
        "FaceAudit_Nova_Arquitetura.pdf"
    )
    build_pdf(out)
