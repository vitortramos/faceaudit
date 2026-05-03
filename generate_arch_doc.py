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
        ("  1.1", "Modelo Mental — A Arquitetura em 30 Segundos"),
        ("2.", "Arquitetura Antiga — Problemas Resolvidos"),
        ("3.", "Visão Geral da Nova Arquitetura"),
        ("  3.1", "Padrão Command + Context + Registry"),
        ("  3.2", "Fluxo de Dados entre Instruções (DAG de Recursos)"),
        ("4.", "Fluxo de Execução Detalhado"),
        ("  4.1", "Execução Individual (step-by-step)"),
        ("  4.2", "Hello World — Exemplo Mínimo Completo"),
        ("5.", "Componentes Principais"),
        ("  5.1", "LRStrategyDefinition — O Plano da Estratégia"),
        ("  5.2", "StrategyExecutionContext — O Barramento de Estado"),
        ("  5.3", "StrategyResource — A Unidade de Troca (categorias e placeholders)"),
        ("  5.4", "StrategyExecutor — O Orquestrador"),
        ("  5.5", "Executores de Instrução"),
        ("6.", "Os 12 Tipos de Instrução"),
        ("7.", "Sistema de Dois Níveis: Biblioteca Global + Projeto"),
        ("  7.1", "Biblioteca Global"),
        ("  7.2", "Projeto Autocontido (Pendrive Paradigm)"),
        ("  7.3", "Fluxo de Montagem de um Projeto"),
        ("8.", "Modelos de Similaridade e Calibração"),
        ("9.", "Pipeline de Transformação de Embeddings"),
        ("  9.1", "PipelineTrajectoryService — Visualização 3D"),
        ("10.", "Protocolo de Persistência (ArtifactStorageProtocol)"),
        ("  10.1", "Verificação de Integridade (LocalProjectArtifactStorage)"),
        ("11.", "Pipeline de Validação"),
        ("  11.1", "Visão Geral — 2 Níveis, 5 Estágios"),
        ("  11.2", "Nível 1: Validação da Estratégia"),
        ("  11.3", "Nível 2: Validação de Instrução (4 estágios)"),
        ("  11.4", "Gate de Dependência (can_execute_instruction)"),
        ("  11.5", "ValidationRuleBase e Cache de Stats"),
        ("  11.6", "Regras Concretas Implementadas"),
        ("  11.7", "Validação Dinâmica por Instrução (build_validation_rules)"),
        ("12.", "Ciclo de Vida de um Projeto FaceAudit"),
        ("  12.1", "Fluxo de Treinamento (end-to-end)"),
        ("  12.2", "Fluxo de Inferência (StrategyInferenceService)"),
        ("13.", "Decisões de Design e Princípios Arquiteturais"),
        ("14.", "Extensibilidade — Como Adicionar Novos Componentes"),
        ("15.", "Interface Gráfica: Estratégias (Nova Arch)"),
        ("  15.1", "Visão Geral do Layout"),
        ("  15.2", "Toolbars"),
        ("  15.3", "Painel Esquerdo"),
        ("  15.4", "Editor de Estratégia com Três Abas"),
        ("  15.5", "Execução Individual de Instruções"),
        ("  15.6", "Workers (Execução em Thread)"),
        ("  15.7", "Decisões de Design da Interface"),
        ("16.", "Estrutura de Pastas"),
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

    # 1.1 — Modelo Mental
    story.append(Paragraph("1.1 Modelo Mental — A Arquitetura em 30 Segundos", S["h2"]))
    story.append(Paragraph(
        "Antes de entrar nos detalhes, esta é a forma mais rápida de entender "
        "como a nova arquitetura funciona:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Conceito", "Analogia"],
        [
            ["Estratégia (LRStrategyDefinition)",
             "Uma <b>receita</b> — lista ordenada de passos (instruções) "
             "e os ingredientes necessários (recursos)."],
            ["Instrução (StrategyInstructionDefinition)",
             "Um <b>passo da receita</b> — transforma ingredientes de entrada "
             "em produtos de saída. Ex: \"treinar pipeline\", \"dividir dataset\"."],
            ["Recurso (StrategyResource)",
             "Um <b>ingrediente ou produto</b> — qualquer objeto nomeado que flui entre instruções: "
             "datasets, pipelines treinados, scores, modelos de calibração."],
            ["Contexto (StrategyExecutionContext)",
             "A <b>bancada de trabalho</b> — memória compartilhada onde os recursos ficam "
             "acessíveis. Uma instrução lê o que precisa e deixa o que produziu."],
            ["Executor (StrategyExecutor)",
             "O <b>cozinheiro</b> — segue a receita passo a passo, resolvendo qual "
             "especialista (executor de instrução) cuida de cada passo."],
            ["Registry (InstructionExecutorRegistry)",
             "O <b>catálogo de especialistas</b> — mapeia cada tipo de instrução "
             "ao executor concreto que sabe executá-la."],
        ],
    )

    code_block(story, S, [
        "Estratégia    = lista de instruções + recursos iniciais",
        "Instrução     = transformação: recursos de entrada → recursos de saída",
        "Recurso       = objeto nomeado que flui entre instruções",
        "Contexto      = memória compartilhada (bancada de trabalho)",
        "Executor      = motor que segue a receita",
        "Registry      = mapa tipo_instrução → executor_concreto",
    ])

    story.append(Paragraph(
        "O perito monta a receita na interface. O motor executa. "
        "Nenhum passo está hard-coded — a ordem e os ingredientes são declarados pelo perito.",
        S["body"]
    ))

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
        "│  preparation_instructions:  [PIPELINE_TRAINING, ...]         │",
        "│  validation_instructions:   [SIMILARITY_SCORING, ...]        │",
        "│  casework_instructions:     [CALIBRATION_TRAINING, ...]      │",
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
        "│  Execução por fase (Preparação → Validação → Caso):          │",
        "│    Para cada instrução da fase:                               │",
        "│      InstructionExecutorRegistry.resolve(instruction_type)    │",
        "│        → executor.execute(instruction, context)               │",
        "│              lê recursos do contexto                          │",
        "│              executa lógica                                   │",
        "│              escreve novos recursos no contexto               │",
        "│                                                               │",
        "│  Execução individual (step-by-step):                          │",
        "│    prepare_context(definition) → contexto persistente         │",
        "│    can_execute_instruction(instr, ctx) → (bool, missing)      │",
        "│    execute_single_instruction(instr, ctx) → result            │",
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

    # 3.1 — Command + Context + Registry
    story.append(Paragraph("3.1 Padrão Command + Context + Registry", S["h2"]))
    story.append(Paragraph(
        "A nova arquitetura implementa uma variação do padrão "
        "<b>Command + Context + Registry</b>. Cada peça tem uma responsabilidade única:",
        S["body"]
    ))

    three_col_table(story, S,
        ["Peça", "Implementação", "Responsabilidade"],
        [
            ["Command\n(Instrução)",
             "StrategyInstructionDefinition\n(8 subclasses concretas)",
             "Descreve UMA ação a ser executada: qual tipo de operação, quais recursos "
             "consome (input_resource_ids), quais produz (output_resource_ids), e todos os "
             "parâmetros configuráveis. É um DTO puro — não contém lógica de execução."],
            ["Context\n(Estado)",
             "StrategyExecutionContext\n(implementa StrategyExecution\nContextProtocol)",
             "Barramento de estado compartilhado. Armazena recursos nomeados, regras de "
             "validação, artefatos, logs e metadados. Uma instrução lê o que precisa com "
             "get_resource() e publica o que produziu com set_resource(). É o ÚNICO canal "
             "de comunicação entre instruções — não há chamadas diretas."],
            ["Registry\n(Roteador)",
             "InstructionExecutorRegistry\n(dict tipo → executor)",
             "Mapeia cada StrategyInstructionType ao executor concreto que sabe executá-la. "
             "O StrategyExecutor chama registry.resolve(instruction_type) para obter o executor "
             "correto. Novos tipos de instrução são adicionados registrando um novo par no mapa."],
        ],
        col_widths=[INNER_W*0.15, INNER_W*0.28, INNER_W*0.57],
    )

    story.append(Paragraph(
        "O <b>StrategyExecutor</b> é o orquestrador que conecta as três peças: "
        "recebe a lista de Commands (instruções), cria o Context, e usa o Registry "
        "para resolver o executor correto de cada instrução. A separação entre o DTO "
        "(o que fazer) e o executor (como fazer) permite testar, substituir e estender "
        "cada parte independentemente.",
        S["body"]
    ))

    code_block(story, S, [
        "Command (DTO puro)          Registry (mapa)             Executor (lógica)",
        "┌──────────────────┐    ┌───────────────────────┐    ┌──────────────────┐",
        "│ PipelineTraining │───▶│ PIPELINE_TRAINING     │───▶│ PipelineTraining │",
        "│ Instruction      │    │  → PipelineTraining   │    │ InstructionExec  │",
        "│                  │    │    InstructionExecutor │    │                  │",
        "│ input_ids: [...]  │    ├───────────────────────┤    │ validate()       │",
        "│ output_ids: [...] │    │ DATASET_SPLIT         │    │ execute()        │",
        "│ pipeline_mode:DUAL│    │  → DatasetSplit       │    │                  │",
        "└──────────────────┘    │    InstructionExecutor │    └────────┬─────────┘",
        "                        └───────────────────────┘             │",
        "                                                              ▼",
        "                        ┌─────────────────────────────────────────────┐",
        "                        │           StrategyExecutionContext           │",
        "                        │  get_resource(id) → recurso                 │",
        "                        │  set_resource(recurso) → publica resultado  │",
        "                        └─────────────────────────────────────────────┘",
    ])

    # 3.2 — DAG de Recursos
    story.append(Paragraph("3.2 Fluxo de Dados entre Instruções (DAG de Recursos)", S["h2"]))
    story.append(Paragraph(
        "Cada instrução declara explicitamente seus <code>input_resource_ids</code> "
        "(o que consome) e <code>output_resource_ids</code> (o que produz). Isso forma "
        "um <b>grafo acíclico dirigido (DAG)</b> implícito entre as instruções — o output "
        "de uma é o input da seguinte. O diagrama abaixo mostra o fluxo de dados "
        "de uma estratégia típica PLDA + Regressão Logística:",
        S["body"]
    ))

    code_block(story, S, [
        "                    ┌─────────────────┐",
        "                    │   dataset_raw    │  (recurso do projeto)",
        "                    │   pipeline_cfg   │",
        "                    └────────┬─────────┘",
        "                             │",
        "                             ▼",
        "                 ┌───────────────────────┐",
        "                 │   PIPELINE_TRAINING    │",
        "                 │   in:  dataset_raw     │",
        "                 │        pipeline_cfg    │",
        "                 │   out: trained_pipe    │",
        "                 └───────────┬───────────┘",
        "                             │",
        "               ┌─────────────┤",
        "               ▼             ▼",
        " ┌─────────────────┐  ┌──────────────────────┐",
        " │ trained_pipe     │  │ PIPELINE_APPLICATION  │",
        " │ (persistido)     │  │ in: trained_pipe      │",
        " └─────────────────┘  │     dataset_raw       │",
        "                      │ out: transformed_ds   │",
        "                      └──────────┬────────────┘",
        "                                 │",
        "                                 ▼",
        "                     ┌───────────────────────┐",
        "                     │    DATASET_SPLIT       │",
        "                     │    in: transformed_ds  │",
        "                     │    out: split_A         │",
        "                     │         split_B         │",
        "                     └─────┬─────────┬────────┘",
        "                           │         │",
        "              ┌────────────┘         └────────────┐",
        "              ▼                                    ▼",
        "  ┌────────────────────────┐         ┌──────────────────────┐",
        "  │ SIMILARITY_MODEL_TRAIN │         │  SIMILARITY_SCORING  │",
        "  │ in:  split_A           │         │  in: split_B         │",
        "  │ out: trained_plda      │────────▶│      trained_plda    │",
        "  └────────────────────────┘         │  out: scores_ds      │",
        "                                     └──────────┬───────────┘",
        "                                                │",
        "                                                ▼",
        "                                    ┌───────────────────────┐",
        "                                    │    PAIR_MINING         │",
        "                                    │    in: scores_ds      │",
        "                                    │    out: impostors      │",
        "                                    │         genuines       │",
        "                                    └─────┬─────────┬───────┘",
        "                                          │         │",
        "                                          ▼         ▼",
        "                                 ┌────────────────────────────┐",
        "                                 │ CALIBRATION_MODEL_TRAINING │",
        "                                 │ in:  impostors, genuines   │",
        "                                 │ out: trained_calibrator    │",
        "                                 └────────────────────────────┘",
    ])

    story.append(Paragraph(
        "A sequência de execução respeita a ordem das instruções na definição da estratégia. "
        "O <code>can_execute_instruction()</code> verifica se todos os "
        "<code>input_resource_ids</code> existem no contexto antes de permitir a execução — "
        "garantindo que o DAG é respeitado mesmo na execução instrução-a-instrução.",
        S["body"]
    ))

    callout_box(story, S,
        "O DAG é <b>implícito</b> — não existe uma estrutura de grafo explícita no código. "
        "A dependência é declarada nos campos <code>input_resource_ids</code> e "
        "<code>output_resource_ids</code> de cada instrução, sincronizados automaticamente "
        "pelo <code>_sync_declared_io()</code> no <code>__post_init__</code> do DTO.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

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
        ("2. Registro preguiçoso de recursos iniciais",
         "Os StrategyResources declarados em strategy_definition.resources são "
         "registrados no contexto via set_resource() — SEM hidratar payloads. "
         "Recursos vindos do disco entram com payload=None e artifact_ref preenchido. "
         "O StrategyResourceLoaderService é injetado no contexto (atributo runtime "
         "resource_loader, junto com embedding_column) para hidratação sob demanda. "
         "A hidratação real acontece em três pontos: (a) execute_single_instruction "
         "chama _rehydrate_inputs_if_needed antes do executor concreto, carregando "
         "apenas os inputs declarados daquela instrução; (b) validation rules chamam "
         "context.ensure_payload(rid) antes de ler payload (configs, datasets para "
         "stats); (c) UI controllers carregam sob clique (ex: visualizar trajetória). "
         "Esse design evita pagar o custo de carregar parquets gigantes / pickles de "
         "modelos a cada troca de estratégia ou refresh — paga-se apenas o que "
         "realmente é usado."),
        ("3. Registro de regras de validação",
         "As ValidationRules passadas à factory são registradas no contexto, "
         "tornando-as disponíveis para qualquer instrução que as referencie "
         "pelo ID."),
        ("4. Validação da estratégia",
         "As validation_rule_ids declaradas na LRStrategyDefinition são executadas "
         "contra o contexto inicial. Se qualquer regra crítica falhar, a execução "
         "é abortada."),
        ("5. Execução por fases (três fases em sequência)",
         "As instruções são organizadas em três fases: Preparação "
         "(preparation_instructions), Validação (validation_instructions) e "
         "Análise de Caso (casework_instructions). O executor itera cada fase "
         "em ordem. Para cada instrução da fase: "
         "(a) resolve o executor concreto no InstructionExecutorRegistry; "
         "(b) chama executor.validate(instruction, context); "
         "(c) se válido, chama executor.execute(instruction, context); "
         "(d) o resultado é registrado; se crítico, a execução é abortada."),
        ("6. Empacotamento do resultado",
         "Após todas as fases, o contexto final é empacotado em um "
         "LRStrategyTrainedModel contendo: recursos treinados, referências de "
         "artefatos persistidos, log de execução e metadados."),
    ]

    for title, desc in steps:
        story.append(KeepTogether([
            Paragraph(title, S["h3"]),
            Paragraph(desc, S["body"]),
        ]))

    story.append(Paragraph("Exemplo de estratégia típica (com fases)", S["h2"]))
    story.append(Paragraph(
        "Uma estratégia completa de PLDA + Regressão Logística tipicamente contém "
        "as seguintes instruções distribuídas em três fases:",
        S["body"]
    ))

    code_block(story, S, [
        "═══ Fase: Preparação (preparation_instructions) ═══",
        "",
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
        "═══ Fase: Validação (validation_instructions) ═══",
        "",
        "Instrução 5: SIMILARITY_SCORING",
        "  → Gera matriz de scores em split_B usando trained_plda",
        "  → Produz: scores_dataset",
        "",
        "Instrução 6: PAIR_MINING",
        "  → Cura impostores/genuínos de scores_dataset via ComparisonMiningConfig",
        "  → Produz: mined_impostors, mined_genuines",
        "",
        "═══ Fase: Análise de Caso (casework_instructions) ═══",
        "",
        "Instrução 7: CALIBRATION_MODEL_TRAINING",
        "  → Treina LogisticRegressionCalibrationModel com mined_impostors + mined_genuines",
        "  → Produz: trained_calibrator",
    ])

    story.append(Paragraph("4.1 Execução individual (step-by-step)", S["h2"]))
    story.append(Paragraph(
        "Além do fluxo completo (<code>execute()</code>), o <code>StrategyExecutor</code> "
        "expõe três métodos para execução instrução por instrução, usados pela interface "
        "gráfica para permitir ao perito executar uma instrução de cada vez:",
        S["body"]
    ))

    step_by_step = [
        ("prepare_context(strategy_definition) → context",
         "Cria o StrategyExecutionContext, hidrata e carrega os recursos iniciais, "
         "registra as regras de validação. O contexto retornado é persistente e deve "
         "ser reutilizado entre chamadas a execute_single_instruction()."),
        ("can_execute_instruction(instruction, context) → (bool, missing_ids)",
         "Verifica se todos os input_resource_ids da instrução estão disponíveis no "
         "contexto. Retorna (True, []) se executável, ou (False, [lista de IDs faltantes]) "
         "se bloqueada. Usado pela UI para mostrar estados visuais: 🔒 bloqueada, "
         "ícone normal executável, ✅ já executada."),
        ("execute_single_instruction(instruction, context) → result",
         "Executa uma única instrução dentro do contexto fornecido. Os recursos produzidos "
         "ficam disponíveis no contexto para instruções futuras. O resultado é um "
         "InstructionExecutionResult com success, errors, produced_resource_ids e "
         "produced_artifacts."),
        ("persist_produced_resource(resource, strategy_id) → artifact_ref",
         "Persiste um recurso produzido em disco via artifact_storage. Mapeia "
         "StrategyResourceType → StrategyArtifactType via to_artifact_type(), gera o "
         "path relativo via StrategyFileType.build_path() e chama save_artifact(). "
         "Retorna StrategyArtifactReference (com path relativo e hash SHA-256) ou None "
         "se o recurso não for persistível (sem payload ou sem artifact_storage). "
         "Chamado pela UI após cada execução individual de instrução."),
    ]

    for title, desc in step_by_step:
        story.append(KeepTogether([
            Paragraph(title, S["h3"]),
            Paragraph(desc, S["body"]),
        ]))

    callout_box(story, S,
        "O contexto é <b>descartado</b> quando o perito edita, reordena ou remove "
        "qualquer instrução da estratégia. Isso garante consistência: os recursos no "
        "contexto sempre refletem exatamente as instruções já executadas. "
        "Porém, os recursos já <b>persistidos em disco</b> (artifact_ref preenchido) "
        "sobrevivem ao reset — estão salvos na definição da estratégia.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    # 4.2 — Hello World
    story.append(Paragraph("4.2 Hello World — Exemplo Mínimo Completo", S["h2"]))
    story.append(Paragraph(
        "O exemplo mais simples possível de uma estratégia funcional: um dataset de "
        "embeddings faciais → modelo de similaridade cosseno → scores de todos os pares. "
        "Apenas 3 instruções, sem pipeline de transformação e sem calibração.",
        S["body"]
    ))

    story.append(Paragraph("Definição da estratégia", S["h3"]))
    code_block(story, S, [
        "from faceaudit.models.strategy.lr_strategy_definition import LRStrategyDefinition",
        "from faceaudit.models.strategy.strategy_resource import StrategyResource",
        "from faceaudit.enums.strategy_resource_type import StrategyResourceType",
        "from faceaudit.models.instructions.similarity_model_training_instruction import (",
        "    SimilarityModelTrainingInstruction,",
        ")",
        "from faceaudit.models.instructions.similarity_scoring_instruction import (",
        "    SimilarityScoringInstruction,",
        ")",
        "from faceaudit.enums.similarity_model_type import SimilarityModelType",
        "",
        "# 1. Criar recurso de dataset (payload será hidratado do disco)",
        "dataset = StrategyResource(",
        "    resource_id='meu_dataset',",
        "    name='Dataset de Embeddings',",
        "    resource_type=StrategyResourceType.DATASET,",
        "    artifact_ref=StrategyArtifactReference(",
        "        path='datasets/meu_dataset.parquet', hash=None",
        "    ),",
        ")",
        "",
        "# 2. Montar instruções",
        "train_sim = SimilarityModelTrainingInstruction(",
        "    name='Treinar Cosine',",
        "    similarity_model_type=SimilarityModelType.COSINE,",
        "    dataset_resource_id='meu_dataset',",
        "    trained_similarity_model_resource_id='cosine_model',",
        ")",
        "",
        "score = SimilarityScoringInstruction(",
        "    name='Gerar Scores',",
        "    trained_similarity_model_resource_id='cosine_model',",
        "    probe_dataset_resource_id='meu_dataset',",
        "    # sem reference → auto-comparação (upper triangle)",
        "    output_similarity_scores_resource_id='scores',",
        ")",
        "",
        "# 3. Montar definição",
        "definition = LRStrategyDefinition(",
        "    name='Hello World Cosine',",
        "    resources=[dataset],",
        "    preparation_instructions=[train_sim],",
        "    validation_instructions=[score],",
        ")",
    ])

    story.append(Paragraph("Execução", S["h3"]))
    code_block(story, S, [
        "from faceaudit.factories.strategy_executor_factory import StrategyExecutorFactory",
        "",
        "# Montar executor com artifact_storage (para hidratar dataset do disco)",
        "executor = StrategyExecutorFactory.build(artifact_storage=storage)",
        "",
        "# Executar tudo",
        "trained_model = executor.execute(definition)",
        "",
        "# Acessar resultado",
        "scores_resource = trained_model.get_trained_resource('scores')",
        "score_dataset = scores_resource.payload  # SimilarityScoreDataset",
        "print(f'{len(score_dataset.samples)} pares comparados')",
    ])

    story.append(Paragraph("Fluxo de dados", S["h3"]))
    code_block(story, S, [
        "meu_dataset ──▶ SIMILARITY_MODEL_TRAINING ──▶ cosine_model",
        "                                                   │",
        "meu_dataset ─────────────────────────────────┐     │",
        "                                             ▼     ▼",
        "                                      SIMILARITY_SCORING ──▶ scores",
    ])

    story.append(Paragraph(
        "Este exemplo demonstra o fluxo essencial: recursos iniciais são carregados no "
        "contexto, instruções consomem e produzem recursos nomeados, e o resultado final "
        "contém todos os recursos produzidos. Estratégias reais adicionam pipeline de "
        "transformação, split de dataset, mineração de pares e calibração — mas o padrão "
        "fundamental é o mesmo.",
        S["body"]
    ))

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
            ["preparation_instructions", "Instruções da fase de Preparação (treino de pipelines, modelos)"],
            ["validation_instructions", "Instruções da fase de Validação (métricas, scores, gráficos)"],
            ["casework_instructions", "Instruções da fase de Análise de Caso (scores do caso real)"],
            ["instructions (property)", "Read-only: concatenação das três fases (retrocompat)"],
            ["validation_rule_ids", "IDs de regras de validação globais da estratégia"],
            ["created_at / updated_at", "Timestamps de criação e atualização"],
        ],
        col_widths=[INNER_W*0.32, INNER_W*0.68],
    )

    story.append(Paragraph(
        "As instruções são organizadas em <b>três fases</b>: Preparação, Validação e "
        "Análise de Caso. A property <code>instructions</code> (read-only) concatena as "
        "três listas para retrocompatibilidade com validadores e outros consumidores. "
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
            ["has_payload (property)", "True se payload não é None"],
            ["has_artifact (property)", "True se artifact_ref não é None"],
        ],
    )

    story.append(Paragraph("Duas categorias de recursos", S["h3"]))
    story.append(Paragraph(
        "Na prática, os recursos de uma estratégia dividem-se em duas categorias:",
        S["body"]
    ))
    two_col_table(story, S,
        ["Categoria", "Descrição"],
        [
            ["Recursos do projeto",
             "Importados pelo perito e compartilhados entre estratégias: DATASET, "
             "PIPELINE_CONFIG, SPLIT_CONFIG, MERGE_CONFIG, SIMILARITY_MODEL_CONFIG, "
             "CALIBRATION_MODEL_CONFIG. Sempre possuem artifact_ref (arquivo em disco)."],
            ["Recursos produzidos pela estratégia",
             "Gerados pela execução de instruções: TRAINED_PIPELINE, "
             "INTERMEDIATE_DATASET, TRAINED_SIMILARITY_MODEL, TRAINED_CALIBRATION_MODEL. "
             "Criados inicialmente como placeholders (payload=None, artifact_ref=None) "
             "quando o perito configura os campos de output de uma instrução. "
             "Após execução, o payload é preenchido e o recurso é persistido em disco "
             "(artifact_ref ganha path + hash)."],
        ],
    )

    story.append(Paragraph("Placeholders de output", S["h3"]))
    story.append(Paragraph(
        "O método <code>StrategyInstructionDefinition.get_output_resource_placeholders()</code> "
        "itera os <code>ResourceFieldSpec</code> com <code>is_output=True</code> e cria "
        "<code>StrategyResource</code> com <code>payload=None</code> e "
        "<code>artifact_ref=None</code>. O controller da UI chama este método em cada "
        "edição da estratégia (<code>_sync_all_output_placeholders()</code>) para manter "
        "a lista <code>LRStrategyDefinition.resources</code> sincronizada com as "
        "instruções declaradas. Placeholders órfãos (de instruções removidas) são "
        "limpos automaticamente — exceto os que já possuem artifact_ref (persistidos).",
        S["body"]
    ))

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

    two_col_table(story, S,
        ["Método", "Descrição"],
        [
            ["execute(definition)",
             "Execução completa: cria contexto, valida, executa todas as fases "
             "(Preparação → Validação → Análise de Caso) e retorna LRStrategyTrainedModel."],
            ["prepare_context(definition)",
             "Cria e inicializa o contexto: REGISTRA recursos da definição (sem hidratar "
             "payloads) e injeta resource_loader/embedding_column como atributos runtime "
             "do contexto. Hidratação é preguiçosa (sob demanda). Retorna contexto "
             "persistente entre execuções de instruções."],
            ["can_execute_instruction(instr, ctx)",
             "Verifica se todos os input_resource_ids da instrução estão disponíveis. "
             "Retorna (bool, list[missing_ids])."],
            ["execute_single_instruction(instr, ctx)",
             "Executa uma única instrução no contexto fornecido. Os recursos produzidos "
             "ficam disponíveis para instruções futuras."],
            ["persist_produced_resource(resource, strategy_id)",
             "Persiste um recurso produzido em disco via artifact_storage. "
             "Mapeia resource_type → artifact_type via to_artifact_type(). "
             "Retorna StrategyArtifactReference ou None se não persistível."],
        ],
    )

    # 5.5
    story.append(Paragraph("5.5 Executores de Instrução", S["h2"]))
    story.append(Paragraph(
        "Localização: <code>faceaudit/executors/instructions/</code>",
        S["body_small"]
    ))
    story.append(Paragraph(
        "Cada tipo de instrução tem um executor concreto que estende "
        "<code>StrategyInstructionExecutorBase</code>. A base implementa a lógica "
        "de validação genérica: (1) regras declaradas via <code>validation_rule_ids</code>, "
        "(2) regras dinâmicas via <code>instruction.build_validation_rules()</code>, "
        "(3) chamada a <code>_validate_instruction_specifics()</code>. "
        "O executor concreto sobrescreve <code>execute()</code> e "
        "<code>_validate_instruction_specifics()</code>.",
        S["body"]
    ))

    story.append(PageBreak())


# ---------------------------------------------------------------------------
# SEÇÃO 6 — OS 8 TIPOS DE INSTRUÇÃO
# ---------------------------------------------------------------------------

def build_s6(story, S):
    section_banner(story, S, "6. Os 14 Tipos de Instrução")

    story.append(Paragraph(
        "Cada instrução é um <code>@dataclass</code> que herda de "
        "<code>StrategyInstructionDefinition</code> e declara campos específicos. "
        "O método <code>__post_init__</code> define <code>instruction_type</code> e "
        "sincroniza <code>input_resource_ids</code> / <code>output_resource_ids</code> "
        "com os campos específicos via <code>_sync_declared_io()</code>.",
        S["body"]
    ))

    story.append(Paragraph(
        "Cada DTO concreto também implementa <code>get_resource_field_specs()</code>, "
        "que retorna uma lista de <code>ResourceFieldSpec</code> — dataclass frozen com "
        "<code>field_name</code>, <code>label</code>, <code>accepted_types</code> "
        "(tupla de StrategyResourceType) e <code>is_output</code> (bool). "
        "A UI usa esses specs para renderizar dropdowns filtrados por tipo de recurso "
        "no painel de configuração de cada instrução. É um método de instância "
        "(não classmethod) porque algumas instruções variam os campos conforme o modo "
        "(ex: Pipeline Training LINEAR vs DUAL).",
        S["body"]
    ))

    instructions = [
        ("PIPELINE_TRAINING", "PipelineTrainingInstructionExecutor",
         "Treina um EmbeddingTransformationPipeline (modo LINEAR ou DUAL). "
         "O modo LINEAR usa apenas o dataset de referência. O modo DUAL "
         "permite que etapas como CORAL e Whitening usem também o dataset alvo "
         "durante o treino (controlado por PipelineTrainingPolicy). "
         "Flag opcional <code>apply_to_training_datasets</code> (default False): "
         "quando ativa, o pipeline recém-treinado é aplicado IMEDIATAMENTE ao "
         "(s) dataset(s) de treino e os datasets transformados são registrados "
         "como outputs adicionais — evita encadear um PIPELINE_APPLICATION "
         "redundante. Os specs de output condicionais aparecem só quando a "
         "flag está ATIVA (declarada com <code>triggers_rebuild=True</code> para "
         "re-renderizar a UI no toggle). "
         "Produz: TRAINED_PIPELINE + (opcional) INTERMEDIATE_DATASET(s)."),
        ("PIPELINE_APPLICATION", "PipelineApplicationInstructionExecutor",
         "Aplica um pipeline treinado a um ou mais datasets. "
         "Cada dataset de entrada gera um dataset transformado de saída. "
         "Produz: recursos INTERMEDIATE_DATASET."),
        ("DATASET_SPLIT", "DatasetSplitInstructionExecutor",
         "Divide um dataset em dois subconjuntos. Modos: BY_IDENTITY_RATIO, "
         "BY_IDENTITY_LIST, BY_IDENTITY_ABSOLUTE (pisos rígidos de pares/IDs), "
         "BY_INDEX_RATIO, BY_INDEX_LIST e BY_SUPERVISED_ADAPTATION (recebe "
         "source+target, prioriza identidades da interseção source∩target para "
         "calibração e completa com identidades exclusivas do target). "
         "Modos por identidade aceitam grouping_strategy=PRESERVE_LOOKALIKES "
         "para preservar pacotes de sósias no mesmo subset. "
         "Modos avançados (ABSOLUTE, SDA, RATIO+LOOKALIKES) delegam ao "
         "PLDASplitService da arquitetura antiga via injeção. "
         "Funciona com FaceDataset e EmbeddingDataset (modos avançados exigem "
         "EmbeddingDataset). Produz: dois recursos INTERMEDIATE_DATASET."),
        ("PAIRED_DATASET_SPLIT", "PairedDatasetSplitInstructionExecutor",
         "Divide DOIS datasets pareados por identidade de forma sincronizada — "
         "um único sorteio de IDs é aplicado em ambos. Inputs: primary "
         "(ex: frontais transformados) e secondary (ex: laterais transformados) "
         "com identidades compartilhadas. Outputs: 4 datasets "
         "(primary_a/b, secondary_a/b), onde os splits B carregam EXATAMENTE "
         "as mesmas IDs em primary e secondary — garantindo pares genuínos "
         "cross-dataset na calibração. Apenas a interseção de identidades é "
         "candidata ao sorteio; IDs exclusivas de um lado caem em A do seu "
         "próprio dataset (não pareadas). Suporta PRESERVE_LOOKALIKES via "
         "cluster_size + min_calibration_pairs/ids/samples_per_id, reusando "
         "PLDASplitService._select_ids no pool combinado restrito à "
         "interseção. Caso de uso típico: estratégias forenses cross-pose "
         "(probe lateral × reference frontal) onde o calibrador deve treinar "
         "na mesma distribuição de pares do casework."),
        ("DATASET_MERGE", "DatasetMergeInstructionExecutor",
         "Combina ou atualiza datasets. "
         "Modo CONCATENATE une dois datasets; modo UPDATE_BY_INDEX substitui "
         "entradas por índice. "
         "Produz: recurso INTERMEDIATE_DATASET."),
        ("PAIR_MINING", "PairMiningInstructionExecutor",
         "Curadoria estatística de pares para treino de calibração. Recebe um "
         "SimilarityScoreDataset (output de SIMILARITY_SCORING) + um recurso "
         "MINING_CONFIG (referência a ComparisonMiningConfig — editável via "
         "botão ✎ no painel). Regras independentes para genuínos e impostores: "
         "HARD/EASY/TYPICAL/RANDOM/MIXED com proporções; foco em PARES ou "
         "IDENTIDADES; round-robin por pivô ou fatiamento por ordem global. "
         "Semântica de HARD/EASY é class-aware: HARD sempre significa 'difícil "
         "de classificar' (top-score para impostores, bottom-score para "
         "genuínos). Output é um único SimilarityScoreDataset concatenado "
         "(impostores primeiro, genuínos depois) — separação por classe via "
         "<code>same_person</code>/<code>label</code>. Produz: SIMILARITY_SCORES "
         "curado."),
        ("SIMILARITY_MODEL_TRAINING", "SimilarityModelTrainingInstructionExecutor",
         "Treina ou prepara um modelo de similaridade: CosineSimilarityModel "
         "(stateless, sem dataset de treino) ou PLDASimilarityModel. "
         "Hyperparâmetros do PLDA são embutidos diretamente na instrução via "
         "campo <code>similarity_model_config</code> (PLDASimilarityModelConfig). "
         "<b>PLDA tem DUAS variantes</b>, controladas por "
         "<code>plda_variant: PLDAVariant</code>: SIMPLE (default, retrocompat) "
         "usa o backend SpeechBrain (formulação Ioffe simplificada com subspace "
         "F entre-classes — usa em_iterations, plda_mode, latent_dim); "
         "TWO_COVARIANCE usa implementação numpy pura "
         "(faceaudit/services/two_covariance_plda_service.py) — formulação "
         "Brümmer &amp; de Villiers 2010, padrão na literatura forense via FoCal "
         "toolkit. Covariâncias B (entre-classes) e W (intra-classe) completas "
         "D×D, treino em forma fechada (sem EM iterativo), scoring por log-LR "
         "vetorizado com kernels Q/P/const cacheados. <code>regularization: "
         "float ≥ 0</code> soma ε·I antes da inversão para estabilidade em alta "
         "dimensionalidade; em_iterations/plda_mode/latent_dim são IGNORADOS em "
         "TWO_COVARIANCE. <code>MODEL_VERSION=2</code> no pickle; payloads v1 "
         "(apenas <code>plda</code> SpeechBrain) carregam como SIMPLE "
         "automaticamente. O modelo treinado produz scores brutos — não LRs. "
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
         "e GridSearch de bandwidth. LogReg suporta GridSearch para o C (DEFAULT "
         "use_gridsearch=True) via c_grid + cv_folds + scoring "
         "(LogRegGridScoringMetric enum: NEG_LOG_LOSS/ROC_AUC); usa StratifiedKFold "
         "com folds auto-reduzidos para a classe minoritária; o melhor C é exposto "
         "como model.best_C_ e emitido em metadata[\"best_C\"]. "
         "Hyperparâmetros são embutidos diretamente na instrução em dois campos "
         "aninhados (logreg_config e kde_config) — apenas o config do tipo "
         "selecionado é usado pelo executor; o outro pode permanecer preenchido "
         "para alternância sem perda. Produz: recurso TRAINED_CALIBRATION_MODEL."),
        ("SCORE_CALIBRATION", "ScoreCalibrationInstructionExecutor",
         "Aplica um modelo de calibração TREINADO a um SimilarityScoreDataset "
         "(raw_score) e produz um novo dataset com log_lr por par populado em "
         "sample.metadata[\"log_lr\"]. O output é registrado com tipo "
         "StrategyResourceType.CALIBRATED_SCORES (distinto de SIMILARITY_SCORES). "
         "Reusa a classe SimilarityScoreDataset — to_parquet detecta log_lr no "
         "metadata e adiciona uma coluna log_lr (float64) nativa; o loader popula "
         "metadata[\"log_lr\"] quando a coluna existe (parquets legados sem a "
         "coluna seguem carregando inalterados). Vetorizado via "
         "calibrator.calibrate_scores(scores); preserva raw_score, same_person, "
         "probe/reference do input. Sem hyperparâmetros próprios — toda a "
         "calibração já está embutida no calibrador treinado. "
         "Produz: recurso CALIBRATED_SCORES."),
        ("STRATEGY_EVALUATION", "StrategyEvaluationInstructionExecutor",
         "Instrução exclusiva da fase Validação. Recebe um CALIBRATED_SCORES "
         "(SimilarityScoreDataset com metadata[\"log_lr\"] preenchido) e "
         "calcula métricas forenses delegando ao ValidationMathService: EER + "
         "threshold, Cllr (estável via logaddexp), Cllr_min via PAVA "
         "(IsotonicRegression do sklearn), AUC e dados de plot ECE por "
         "broadcasting numpy. O output é um StrategyValidationResult contendo "
         "ValidationMetrics (escalares) + ValidationPlotData (arrays para "
         "Tippett, DET, ECE), registrado com tipo VALIDATION_RESULT e "
         "persistido em pickle (preserva ndarray nativamente). Sem hiperparâmetros "
         "próprios — cálculo determinístico. A UI oferece um botão "
         "'📊 Visualizar Resultado de Validação' que abre um dialog com 4 "
         "abas: Distribuição de LRs, Tippett, Curva DET (escala normit) e "
         "Calibração ECE. Produz: recurso VALIDATION_RESULT."),
        ("CASEWORK_VISUALIZATION", "CaseworkVisualizationInstructionExecutor",
         "Instrução exclusiva da fase Análise de Caso "
         "(em <code>_CASEWORK_ONLY_TYPES</code>, nunca aparece fora dessa fase). "
         "Compara a distribuição de scores do caso forense contra uma população de "
         "referência (gen + imp da validação). Choice <code>score_type</code> "
         "(CALIBRATED/RAW) filtra os dropdowns: CALIBRATED aceita CALIBRATED_SCORES "
         "em ambos (referência e caso); RAW aceita SIMILARITY_SCORES. Validador exige: "
         "referência NÃO pode ser <code>is_casework=True</code>; caso DEVE ser "
         "<code>is_casework=True</code>; CALIBRATED requer <code>log_lr</code> "
         "preenchido. Em RAW, computa o threshold de EER on-the-fly (FNR=FPR via "
         "varredura linear de 1000 thresholds — &lt;10ms para 100k pares). Output é "
         "um <code>CaseworkVisualizationResult</code> (ndarrays gen/imp/case + "
         "metadata + case_summary) registrado como CASEWORK_VISUALIZATION e "
         "persistido em .pkl. UI: botão '📊 Visualizar Caso' abre dialog que "
         "sobrepõe à distribuição de referência (KDE+histograma) o caso: KDE azul "
         "quando n_case ≥ 30, ou linhas verticais tracejadas anotadas com valor "
         "numérico quando n_case &lt; 30. Linha de referência em log-LR=0 "
         "(CALIBRATED) ou no EER threshold (RAW). Produz: recurso "
         "CASEWORK_VISUALIZATION."),
        ("DATASET_BALANCING", "DatasetBalancingInstructionExecutor",
         "Balanceia o número de amostras por identidade — produz um novo "
         "INTERMEDIATE_DATASET preservando o input. <b>Duas dimensões "
         "ortogonais.</b> "
         "<code>balancing_mode: DatasetBalancingMode</code>: <b>CAP_ONLY</b> "
         "(limita a target_n por ID; IDs com menos amostras permanecem inalteradas), "
         "<b>FLOOR_AND_CAP</b> (cap + descarta IDs inteiras com &lt; target_n "
         "amostras), <b>AUTO_MIN_BALANCE</b> (N = min observado em runtime — "
         "todas as IDs ficam com a mesma contagem; potencialmente lossy). "
         "<code>selection_criterion: DatasetBalancingSelection</code>: "
         "<b>RANDOM</b> (sorteio reprodutível via random_seed) ou "
         "<b>BY_QUALITY</b> (top-N por métrica, tie-breaker determinístico via "
         "SHA1 de face_hash/file_name). <code>quality_metric: QualityScoreType</code> "
         "aparece só quando BY_QUALITY — opções FIQA, OFIQ, SERFIQ ou "
         "EMBEDDING_NORM (nova; calculada on-the-fly via np.linalg.norm do "
         "embedding sepaelv2/AdaFace, não requer coluna no parquet). Direção "
         "consistente: maior = melhor para todas as métricas. Executor agnóstico "
         "ao tipo do payload (FaceDataset ou EmbeddingDataset). target_n é "
         "ocultado da UI quando AUTO_MIN_BALANCE está selecionado. Logs "
         "registram N efetivo, IDs descartadas no floor e contagem de fallbacks "
         "(samples sem métrica). Produz: recurso INTERMEDIATE_DATASET."),
        ("HUBNESS_ANALYSIS", "HubnessAnalysisInstructionExecutor",
         "Diagnóstico de viés topológico (hubness) em alta dimensionalidade — "
         "quantifica quais faces atraem vizinhos errados via Nk score, similaridade "
         "impostora e correlações com pose/norma. Recebe um <code>EmbeddingDataset</code>, "
         "instancia <code>HubnessAnalysisService</code> (com <code>ComparisonService</code> "
         "injetado pela <code>StrategyExecutorFactory</code>), executa "
         "<code>compute_analysis</code> (custoso: O(N²) com batching B×C, soma por "
         "exclusão e top-K contínuo) + os 4 estudos derivados (stratification, "
         "paired_study, quality_diagnostics, optimization_study) e empacota tudo "
         "em um <code>HubnessReport</code> compósito persistido como pickle "
         "(preserva ndarrays). Hyperparâmetros: <code>k_neighbors</code>, "
         "<code>batch_size</code>, <code>chunk_size</code>, "
         "<code>paired_study_threshold_nf</code>. O service aceita "
         "<code>logger_callback</code> opcional que substitui prints+tqdm por "
         "progresso periódico (a cada 10%) — o executor passa "
         "<code>lambda msg: context.log(msg)</code>. Validador exige "
         "EmbeddingDataset, k_neighbors &lt; n_samples e avisa quando há &lt; 5 "
         "identidades. UI: 3ª coluna do editor (InstructionResultsPanel) instancia "
         "<code>HubnessReportWidget</code> com 5 abas inline (Diagnóstico Topológico, "
         "Estratificação por Pose, Estudo Pareado, Qualidade × Norma, Densidade); "
         "botão 🔍 Expandir abre <code>HubnessReportViewerDialog</code> com 5 abas "
         "curadas + galeria scrollável de 16 plots. Produz: recurso HUBNESS_REPORT."),
        ("HUBNESS_ABLATION", "HubnessAblationInstructionExecutor",
         "Estudo de ablação topológica — recorta progressivamente o dataset (1, 2, 3 "
         "fotos por ID + total) seguindo critério escolhido via "
         "<code>ablation_mode: HubnessAblationMode</code>: <b>BY_QUANTITY</b> "
         "prioriza melhores fotos por FIQA; <b>BY_POSE</b> prioriza mais frontais "
         "(menor abs(yaw)+abs(pitch)). Mede o hubness em cada cenário. "
         "<b>Custoso:</b> refaz O(N²) em N subsets — use HUBNESS_ANALYSIS para "
         "diagnóstico geral barato e use ABLATION apenas para o estudo formal de "
         "evolução do Nk. Validador avisa quando BY_QUANTITY é usado em dataset "
         "sem quality_score. Produz: AblationStudyInfo registrado como recurso "
         "HUBNESS_ABLATION_RESULT (pickle). UI: 3ª coluna do editor instancia "
         "<code>HubnessAblationWidget</code> (single-plot inline com KDEs "
         "sobrepostas dos cenários); botão 🔍 Expandir abre "
         "<code>HubnessAblationViewerDialog</code>."),
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

    story.append(Paragraph("Mapeamento StrategyResourceType → StrategyArtifactType", S["h3"]))
    story.append(Paragraph(
        "O método <code>StrategyResourceType.to_artifact_type()</code> converte o tipo "
        "lógico do recurso no tipo de artefato físico para persistência. Usado pelo "
        "<code>StrategyExecutor.persist_produced_resource()</code> para determinar em "
        "qual subpasta e formato salvar o recurso produzido:",
        S["body"]
    ))
    two_col_table(story, S,
        ["StrategyResourceType", "StrategyArtifactType"],
        [
            ["TRAINED_PIPELINE", "TRAINED_PIPELINE (.pkl em artifacts/)"],
            ["TRAINED_SIMILARITY_MODEL", "TRAINED_SIMILARITY_MODEL (.pkl em artifacts/)"],
            ["TRAINED_CALIBRATION_MODEL", "TRAINED_CALIBRATION_MODEL (.pkl em artifacts/)"],
            ["INTERMEDIATE_DATASET", "INTERMEDIATE_DATASET (.parquet em datasets/)"],
        ],
    )

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
    section_banner(story, S, "11. Pipeline de Validação")

    # 11.1
    story.append(Paragraph("11.1 Visão Geral — 2 Níveis, 5 Estágios", S["h2"]))
    story.append(Paragraph(
        "A validação na nova arquitetura opera em <b>dois níveis</b> com <b>cinco estágios</b> "
        "no total. Cada estágio agrega seus resultados no mesmo <code>ValidationResult</code> "
        "via <code>merge()</code>. Além disso, um <b>gate de dependência</b> separado impede "
        "a execução de instruções cujas dependências não estão satisfeitas.",
        S["body"]
    ))

    # Diagrama ASCII do pipeline
    pipeline_lines = [
        "┌─────────────────────────────────────────────────────────────────┐",
        "│              PIPELINE DE VALIDAÇÃO — VISÃO GERAL               │",
        "├─────────────────────────────────────────────────────────────────┤",
        "│                                                                 │",
        "│  ┌─── Nível 1: Estratégia (antes de qualquer instrução) ────┐  │",
        "│  │  Strategy Rules (LRStrategyDefinition.validation_rule_ids)│  │",
        "│  │  → Regras globais: isolamento de datasets, etc.          │  │",
        "│  └──────────────────────────────────────────────────────────┘  │",
        "│                          │                                      │",
        "│                          ▼                                      │",
        "│  ┌─── Gate: can_execute_instruction() ──────────────────────┐  │",
        "│  │  Todos os input_resource_ids presentes no contexto?      │  │",
        "│  │  Não → instrução bloqueada (🔒 na UI)                    │  │",
        "│  └──────────────────────────────────────────────────────────┘  │",
        "│                          │                                      │",
        "│                          ▼                                      │",
        "│  ┌─── Nível 2: Instrução (validate() do executor base) ─────┐  │",
        "│  │                                                          │  │",
        "│  │  Estágio 1: Structural                                   │  │",
        "│  │  └─ Null checks (instruction, context)                   │  │",
        "│  │                                                          │  │",
        "│  │  Estágio 2: Pluggable                                    │  │",
        "│  │  └─ instruction.validation_rule_ids → regras do contexto │  │",
        "│  │                                                          │  │",
        "│  │  Estágio 3: Dynamic                                      │  │",
        "│  │  └─ instruction.build_validation_rules()                  │  │",
        "│  │                                                          │  │",
        "│  │  Estágio 4: Executor                                     │  │",
        "│  │  └─ _validate_instruction_specifics()                    │  │",
        "│  │                                                          │  │",
        "│  └──────────────────────────────────────────────────────────┘  │",
        "│                                                                 │",
        "└─────────────────────────────────────────────────────────────────┘",
    ]
    for line in pipeline_lines:
        story.append(Paragraph(line, S["code"]))
    story.append(Spacer(1, 8))

    callout_box(story, S,
        "O pipeline de validação executa em dois momentos: (1) manualmente pelo botão "
        "<b>Validar</b> na interface (sem executar a instrução) e (2) automaticamente "
        "dentro de <code>execute()</code> antes da lógica de negócio. Ambos os caminhos "
        "executam exatamente os mesmos 4 estágios do Nível 2.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    # 11.2
    story.append(Paragraph("11.2 Nível 1: Validação da Estratégia", S["h2"]))
    story.append(Paragraph(
        "Executada em <code>StrategyExecutor._validate_strategy()</code> durante a execução "
        "completa (<b>Run All</b>). Itera os <code>validation_rule_ids</code> declarados na "
        "<code>LRStrategyDefinition</code> e executa cada regra contra a definição inteira. "
        "Se qualquer regra falha, a execução é abortada com <code>ValueError</code> antes de "
        "qualquer instrução ser processada.",
        S["body"]
    ))
    story.append(Paragraph(
        "Na execução instrução-a-instrução (step-by-step via UI), este nível <b>não é executado</b> "
        "— a validação fica a cargo do Nível 2 em cada instrução individual. As regras de estratégia "
        "referenciadas por <code>validation_rule_ids</code> da estratégia são políticas globais que "
        "só fazem sentido quando a definição completa está disponível.",
        S["body"]
    ))

    # 11.3
    story.append(Paragraph("11.3 Nível 2: Validação de Instrução (4 estágios)", S["h2"]))
    story.append(Paragraph(
        "Implementada em <code>StrategyInstructionExecutorBase.validate()</code>. "
        "Os 4 estágios executam sequencialmente — apenas o Estágio 1 pode abortar os demais "
        "(null check). Os estágios 2–4 sempre executam e agregam todos os erros/warnings.",
        S["body"]
    ))

    two_col_table(story, S,
        ["Estágio", "Comportamento"],
        [
            ["1. Structural",
             "Null checks em instruction e context. Se qualquer um for None, retorna "
             "ValidationResult.critical() imediatamente — os demais estágios não executam."],
            ["2. Pluggable (validation_rule_ids)",
             "Itera instruction.validation_rule_ids e executa cada regra registrada no contexto. "
             "Regras não encontradas geram erro mas não abortam os estágios seguintes. "
             "Permitem políticas do laboratório como limiares mínimos e restrições de isolamento."],
            ["3. Dynamic (build_validation_rules)",
             "Chama instruction.build_validation_rules() e executa cada regra retornada. "
             "As regras são instanciadas com os resource_ids atuais do DTO — refletem "
             "as escolhas mais recentes do perito na interface. Default: retorna []."],
            ["4. Executor (_validate_instruction_specifics)",
             "Validações hard-coded do executor concreto. Verificam: tipo correto da instrução, "
             "campos obrigatórios preenchidos, recursos existentes no contexto com payloads corretos. "
             "Automáticas e invisíveis para o perito — sempre executam."],
        ],
    )

    # 11.4
    story.append(Paragraph("11.4 Gate de Dependência (can_execute_instruction)", S["h2"]))
    story.append(Paragraph(
        "Separado do pipeline de validação, o <code>StrategyExecutor.can_execute_instruction()</code> "
        "verifica se todos os <code>input_resource_ids</code> declarados pela instrução estão "
        "presentes no contexto de execução. Na UI, instruções bloqueadas aparecem com ícone 🔒 "
        "e o botão ▶ Run desabilitado.",
        S["body"]
    ))
    story.append(Paragraph(
        "Este gate roda <b>antes</b> do pipeline de validação e não é um estágio dele — é uma "
        "verificação estrutural de pré-condições. Retorna <code>(bool, List[str])</code>: "
        "<code>(True, [])</code> se executável ou <code>(False, [missing_ids])</code> se bloqueada. "
        "Os <code>input_resource_ids</code> são mantidos em sincronia pelo <code>_sync_declared_io()</code> "
        "no <code>__post_init__</code> de cada DTO de instrução.",
        S["body"]
    ))

    # 11.5
    story.append(Paragraph("11.5 ValidationRuleBase e Cache de Stats", S["h2"]))
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

    callout_box(story, S,
        "As regras plugáveis não precisam ser copiadas para cada projeto — elas vivem no "
        "código da aplicação e são referenciadas pelo rule_id. Templates de estratégia na "
        "biblioteca global já carregam os rule_ids embutidos. Ao instanciar o template, "
        "as políticas do laboratório vêm junto automaticamente.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

    # 11.6
    story.append(Paragraph("11.6 Regras Concretas Implementadas", S["h2"]))

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
        ("DualPipelineDimensionCompatibilityRule",
         "dual_pipeline_dimension_compatibility_rule.py",
         "CRITICAL / WARNING",
         "Verifica que source e target pipeline configs produzem a mesma dimensão de saída "
         "(output_dimension). CRITICAL se dimensões explícitas divergem; WARNING se ambas "
         "são None (verificação adiada para runtime). "
         "Parâmetros: source_config_resource_id, target_config_resource_id. "
         "Uso: validação dinâmica via build_validation_rules() de PipelineTrainingInstruction DUAL."),
        ("DualPipelineSubsequenceRule",
         "dual_pipeline_subsequence_rule.py",
         "CRITICAL",
         "Verifica que os tipos do pipeline target são subsequência ordenada dos tipos do "
         "source (excluindo CORAL). CRITICAL se target contém CORAL_TRANSFORMATION (exclusivo "
         "do source) ou se a sequência de tipos não é válida. "
         "Parâmetros: source_config_resource_id, target_config_resource_id. "
         "Uso: validação dinâmica via build_validation_rules() de PipelineTrainingInstruction DUAL."),
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

    # 11.7
    story.append(Paragraph("11.7 Validação Dinâmica por Instrução (build_validation_rules)", S["h2"]))
    story.append(Paragraph(
        "O Estágio 3 (Dynamic) permite que cada DTO de instrução construa "
        "regras de validação <b>dinamicamente</b>, com base no seu estado atual. "
        "A classe base <code>StrategyInstructionDefinition</code> declara o método "
        "<code>build_validation_rules() → list</code> (retorna <code>[]</code> por padrão). "
        "DTOs concretos fazem override para instanciar regras parametrizadas com os "
        "<code>resource_ids</code> configurados pelo perito na interface.",
        S["body"]
    ))

    story.append(Paragraph("Exemplo: PipelineTrainingInstruction (modo DUAL)", S["h3"]))
    story.append(Paragraph(
        "Quando <code>pipeline_mode == DUAL</code> e os campos de recurso estão preenchidos, "
        "<code>build_validation_rules()</code> retorna três regras:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Regra", "O que valida"],
        [
            ["DatasetOverlapRequiredRule",
             "Source e target datasets compartilham identidades em comum "
             "(necessário para formar pares genuínos no treino DUAL)."],
            ["DualPipelineDimensionCompatibilityRule",
             "Source e target pipeline configs produzem a mesma dimensão de saída "
             "(necessário para que o scoring cruzado funcione)."],
            ["DualPipelineSubsequenceRule",
             "Target pipeline é subsequência ordenada do source sem CORAL "
             "(CORAL é adaptação de domínio, exclusiva do ramo source)."],
        ],
    )

    callout_box(story, S,
        "As regras são instanciadas a cada chamada de <code>build_validation_rules()</code> "
        "com os resource_ids <b>atuais</b> dos campos da instrução. Se o perito altera um "
        "dropdown de dataset e clica Validar, a nova regra já usa o novo resource_id — "
        "sem necessidade de re-registrar regras no contexto.",
        bg=colors.HexColor("#FFF8E8"), border=BRAND_ACCENT,
    )

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
        "Implementar get_resource_field_specs() retornando ResourceFieldSpec para cada campo de recurso",
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
# SEÇÃO 15 — INTERFACE GRÁFICA: STRATEGY MANAGER V2
# ---------------------------------------------------------------------------

def build_s15(story, S):
    section_banner(story, S, "15. Interface Gráfica: Estratégias (Nova Arch)")

    story.append(Paragraph(
        "A aba <b>Estratégias (Nova Arch)</b> é o primeiro componente de interface "
        "construído sobre a nova arquitetura declarativa. Ela permite ao perito criar "
        "projetos, importar recursos, compor estratégias e executar instruções "
        "individualmente ou em sequência — tudo pela interface gráfica.",
        S["body"]
    ))

    story.append(Paragraph("15.1 Visão geral do layout", S["h2"]))

    code_block(story, S, [
        "┌─────────────────────────────────────────────────────────────────────┐",
        "│  Biblioteca Global:  [Pipeline] [Mining] [Split] [Gerenciar]        │",
        "│  Projeto:  [Novo] [Abrir] [Salvar] [Fechar]  │  [▶ Exec]            │",
        "├──────────────┬──────────────────────────────────────────────────────┤",
        "│  Left Panel  │  Main Area (QTabWidget)                              │",
        "│  ┌──────────┐│  ┌────────────────────────────────────────────────┐  │",
        "│  │Estratégias││  │ Estratégia: LDA+PLDA   │   Embedding: ▼      │  │",
        "│  │ • LDA+PLDA││  ├──────────┬─────────────┬─────────────────────┤  │",
        "│  │ • Cosine  ││  │Preparação│ Validação   │ Análise de Caso     │  │",
        "│  ├──────────┤│  ├──────────┴─────────────┴─────────────────────┤  │",
        "│  │ Recursos  ││  │ Lista       │ Configuração  │ Resultados    │  │",
        "│  │ 📁Dataset ││  │ instruções  │ (dropdowns    │ (3ª coluna)   │  │",
        "│  │ 📁Pipeline││  │ [▶ Run]    │  tipados)     │ ─────────────  │  │",
        "│  │ 📁Config  ││  │ 🔒 / ✅    │ Pipeline ▼    │ Saídas:        │  │",
        "│  │ 📁Outputs ││  │ ⚙ Score    │ Dataset:  ▼   │ 🟡 raw_scores  │  │",
        "│  ├──────────┤│  │             │ Output: [___] │ ────────────── │  │",
        "│  │ Info      ││  │             │ [Aplicar]    │ Visualização:  │  │",
        "│  │ recurso   ││  │             │ [Validar]    │ [📊 inline]    │  │",
        "│  │ (pipeline)││  │             │               │ [🔍 Expandir]  │  │",
        "│  └──────────┘│  │ ~200px      │ ~400px        │ ~600px         │  │",
        "│              │  └─────────────┴───────────────┴────────────────┘  │",
        "└──────────────┴──────────────────────────────────────────────────────┘",
    ])

    story.append(Paragraph(
        "A interface é registrada em <code>main_gui_controller._init_strategy_manager_v2()</code>. "
        "Recebe <code>global_storage</code> (biblioteca global, opcional) e "
        "<code>app_config</code> (paths padrão para FileDialogs). "
        "O controlador central (<code>StrategyManagerV2Controller</code>, QObject) coordena "
        "todos os widgets e mantém o estado do projeto.",
        S["body"]
    ))

    # 15.2
    story.append(Paragraph("15.2 Toolbars", S["h2"]))
    story.append(Paragraph(
        "A aba possui duas barras de ferramentas horizontais no topo:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Toolbar", "Ações"],
        [
            ["Biblioteca Global (GlobalResourcesToolbar)",
             "Pipeline — abre PipelineCreatorController (criação de configs de pipeline). "
             "Mining — abre NewMiningConfigDialogController. "
             "Split — abre NewSplitConfigDialogController. "
             "Gerenciar Biblioteca — lista/remove recursos da GlobalLibraryStorage."],
            ["Projeto (ProjectToolbar)",
             "Novo — cria projeto FaceAudit (.fa). "
             "Abrir — abre projeto existente. "
             "Salvar — persiste definições. "
             "Fechar — fecha projeto. "
             "▶ Executar — executa todas as fases da estratégia selecionada (Run All). "
             "Salvar como Template — exporta estratégia para a biblioteca global."],
        ],
    )

    # 15.3
    story.append(Paragraph("15.3 Painel Esquerdo (LeftPanelWidget)", S["h2"]))
    story.append(Paragraph(
        "QSplitter vertical com três seções, dividindo aproximadamente 35%/40%/25%:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Seção", "Funcionalidade"],
        [
            ["StrategyListWidget",
             "Lista de LRStrategyDefinitions do projeto. "
             "Botões: Nova / Duplicar / Remover. "
             "Clique seleciona; duplo-clique abre no editor."],
            ["ResourceTreeWidget",
             "Árvore de recursos agrupados por StrategyResourceType. "
             "Botão 'Adicionar Recurso' com menu: Dataset / Split Config / Mining Config / Pipeline Config. "
             "Importa arquivos para o projeto via copy_artifact_to_project()."],
            ["ResourceInfoWidget",
             "Exibe metadados do recurso selecionado: ID, tipo, arquivo em disco (artifact_ref), "
             "hash SHA-256, tamanho em disco e lista de instruções que o utilizam. "
             "Para DATASET: mostra estatísticas (amostras, identidades, colunas de embedding). "
             "Para PIPELINE_CONFIG: mostra etapas do pipeline, dimensão de saída "
             "(com fallback para EmbeddingType.dimensions) e flags (PCA/LDA)."],
        ],
    )

    # 15.4
    story.append(Paragraph("15.4 Editor de Estratégia com Três Abas", S["h2"]))
    story.append(Paragraph(
        "O <code>StrategyEditorWidget</code> é o coração da aba. Quando uma estratégia "
        "é aberta, ele exibe um QTabWidget com três abas — uma por fase de instrução:",
        S["body"]
    ))

    two_col_table(story, S,
        ["Aba", "Fase"],
        [
            ["Preparação",
             "preparation_instructions — treino de pipelines, divisão de datasets, "
             "treino de modelos de similaridade."],
            ["Validação",
             "validation_instructions — cálculo de scores, mineração de pares, "
             "métricas de desempenho."],
            ["Análise de Caso",
             "casework_instructions — treino de calibrador, scores do caso real."],
        ],
    )

    story.append(Paragraph(
        "Cada aba contém um <b>splitter horizontal com TRÊS colunas</b> — "
        "lista de instruções (esquerda, ~200px), painel de configuração (meio, "
        "~400px) e painel de resultados (direita, ~600px, mais largo porque "
        "acomoda gráficos inline). Tamanhos via QSplitter, redimensionáveis pelo "
        "usuário. O antigo <code>ProducedResourcesWidget</code> por aba foi "
        "removido — recursos produzidos pela estratégia aparecem globalmente no "
        "<code>LeftPanelWidget</code> (junto com os recursos de projeto), e o "
        "estado runtime de cada output aparece dentro do "
        "<code>InstructionResultsPanel</code> via <code>OutputRowWidget</code>.",
        S["body"]
    ))

    code_block(story, S, [
        "QSplitter (horizontal)",
        "├── InstructionListWidget       (~200px)",
        "├── InstructionConfigPanel      (~400px) — outputs_in_results_panel=True",
        "└── InstructionResultsPanel     (~600px) — saídas + gráficos inline + 🔍",
    ])

    two_col_table(story, S,
        ["Painel", "Funcionalidade"],
        [
            ["InstructionListWidget (esquerda)",
             "Lista reordenável (drag-and-drop) de instruções da fase. "
             "Botões: ➕ Adicionar (menu filtrado pelos tipos válidos da fase), "
             "🗑 Remover, ▶ Run (instrução individual). Estados visuais por "
             "instrução: 🔒 bloqueada (cinza, dependências faltantes), ícone "
             "normal (executável), ✅ já executada (verde)."],
            ["InstructionConfigPanel (meio)",
             "Formulário dinâmico por tipo de instrução. Os campos de recurso "
             "são renderizados a partir de <code>get_resource_field_specs()</code>: "
             "inputs como dropdowns filtrados por StrategyResourceType. Botão "
             "'Aplicar' grava alterações no DTO e chama <code>_sync_declared_io()</code>. "
             "Botão 'Validar' aplica o formulário e executa a validação completa "
             "(regras dinâmicas via <code>build_validation_rules()</code> + "
             "validações internas do executor). Quando instanciado pelo "
             "StrategyEditorWidget com flag <code>outputs_in_results_panel=True</code>, "
             "a seção 'Saídas' e os 4 botões de visualização (📊) ficam ocultos "
             "— migram para o painel de resultados (3ª coluna)."],
            ["InstructionResultsPanel (direita)",
             "Para a instrução SELECIONADA: lista de outputs (OutputRowWidget) "
             "com estado runtime (⬜ placeholder / 🟡 em memória / 🟢 salvo) + "
             "gráfico inline quando aplicável + botão 🔍 Expandir que abre o "
             "gráfico em dialog tela cheia. Mapeamento de plot por tipo: "
             "SIMILARITY_SCORING / PAIR_MINING / SCORE_CALIBRATION → "
             "ScoreDistributionWidget; STRATEGY_EVALUATION → "
             "ValidationResultWidget (4 abas lazy: distribuição de LRs, Tippett, "
             "DET, ECE); CASEWORK_VISUALIZATION → CaseworkVisualizationWidget; "
             "PIPELINE_TRAINING / PIPELINE_APPLICATION → BOTÃO de trajetória 3D "
             "(não inline — abre <code>GraphViewerDialog</code> por exigir espaço "
             "para rotação/zoom). Demais instruções (DATASET_SPLIT, _MERGE, "
             "_BALANCING, PAIRED_DATASET_SPLIT, SIMILARITY_MODEL_TRAINING, "
             "CALIBRATION_MODEL_TRAINING) exibem só os outputs. Mantém um cache "
             "<code>_cards: Dict[instruction_id, _ResultsCard]</code> via "
             "QStackedWidget interno — trocar de instrução é swap do card "
             "ativo (gráficos cacheados). Re-execução de uma instrução invalida "
             "o card daquela instrução (drop + redraw na próxima exibição). Para "
             "datasets grandes (SIMILARITY_SCORES / CALIBRATED_SCORES) ainda no "
             "disco após reabrir um projeto, o card mostra um botão "
             "'📊 Carregar e visualizar' em vez de hidratar automaticamente — "
             "evita travar a main thread com leituras de parquets de 5M+ pares."],
        ],
    )

    callout_box(story, S,
        "Os dropdowns de recurso são <b>filtrados por tipo</b>. Por exemplo, o campo "
        "'Pipeline Config' de um PIPELINE_TRAINING mostra apenas recursos do tipo "
        "PIPELINE_CONFIG, enquanto o campo 'Dataset' mostra apenas DATASET e "
        "INTERMEDIATE_DATASET. Isso evita erros de configuração.",
        bg=BRAND_LIGHT, border=BRAND_MED,
    )

    callout_box(story, S,
        "<b>Widgets de plot reutilizáveis</b> (em <code>strategy_editor/result_widgets/</code>): "
        "ScoreDistributionWidget, ValidationResultWidget e CaseworkVisualizationWidget "
        "foram extraídos dos dialogs originais. Cada um aceita flags "
        "<code>show_header</code> / <code>show_toolbar</code> / <code>show_help</code> "
        "para alternar entre modo dialog (full, com toolbar matplotlib + texto de "
        "ajuda) e modo inline (compacto, sem toolbar). Os dialogs antigos "
        "(ValidationResultViewerDialog, CaseworkVisualizationViewerDialog) viraram "
        "wrappers finos que embutem o widget — usados pelo botão 🔍 Expandir.",
        bg=BRAND_LIGHT, border=BRAND_MED,
    )

    # 15.5
    story.append(Paragraph("15.5 Execução Individual de Instruções", S["h2"]))
    story.append(Paragraph(
        "O controlador mantém estado de execução persistente entre cliques do botão "
        "▶ Run: <code>_execution_context</code> (StrategyExecutionContext), "
        "<code>_executed_instruction_ids</code> (set de IDs já executados) e "
        "<code>_executor</code> (StrategyExecutor). O fluxo é:",
        S["body"]
    ))

    steps = [
        ("1.", "Perito abre uma estratégia → estado de execução vazio; todas as instruções mostram 🔒. "
               "Recursos já persistidos em execuções anteriores aparecem como 🟢 nos OutputRowWidget "
               "do InstructionResultsPanel (3ª coluna) ao selecionar a instrução correspondente."),
        ("2.", "Perito clica ▶ Run na primeira instrução → _ensure_executor_and_context() "
               "cria o executor e prepara o contexto (hidrata recursos iniciais)."),
        ("3.", "_refresh_instruction_states() calcula quais instruções são executáveis "
               "(todos os input_resource_ids disponíveis no contexto) → instruções sem "
               "dependências passam de 🔒 para executáveis."),
        ("4.", "Instrução é executada em thread separada (InstructionExecutionWorker). "
               "Após sucesso, outputs ficam disponíveis no contexto → instruções que dependiam "
               "desses outputs tornam-se executáveis."),
        ("5.", "Recursos produzidos são persistidos em disco automaticamente via "
               "_persist_produced_resources(): para cada produced_resource_id do resultado, "
               "chama StrategyExecutor.persist_produced_resource(), obtém StrategyArtifactReference "
               "e atualiza a LRStrategyDefinition em disco. Cada OutputRowWidget no "
               "InstructionResultsPanel muda de ⬜ placeholder para 🟢 salvo em disco; "
               "se a instrução tem visualização inline (scoring/calibration/evaluation/casework), "
               "o gráfico é renderizado automaticamente na 3ª coluna após a execução."),
        ("6.", "Perito edita, reordena ou remove qualquer instrução → todo o estado de "
               "execução é descartado (_reset_execution_state()). Instruções voltam a 🔒. "
               "Recursos já persistidos (artifact_ref) permanecem salvos."),
    ]
    for num, desc in steps:
        story.append(Paragraph(f"{num} {desc}", S["bullet"]))

    story.append(Spacer(1, 4))

    story.append(Paragraph("15.6 Workers (Execução em Thread)", S["h2"]))
    story.append(Paragraph(
        "A execução de instruções e estratégias completas ocorre em threads separadas "
        "para não bloquear a interface. Ambos os workers seguem o padrão "
        "QObject + moveToThread (não herdam de QThread).",
        S["body"]
    ))

    two_col_table(story, S,
        ["Worker", "Uso"],
        [
            ["StrategyExecutionWorker",
             "Executa StrategyExecutor.execute() — todas as fases em sequência. "
             "Usado pelo botão '▶ Executar' da toolbar. "
             "Sinais: log_message(str), finished(LRStrategyTrainedModel), error(str)."],
            ["InstructionExecutionWorker",
             "Executa StrategyExecutor.execute_single_instruction() — uma instrução. "
             "Usado pelo botão '▶ Run' de cada instrução na lista. "
             "Sinais: log_message(str), finished(InstructionExecutionResult), error(str)."],
        ],
    )

    story.append(Paragraph("15.7 Decisões de Design da Interface", S["h2"]))

    decisions = [
        (
            "Abertura de projeto é síncrona com WaitCursor",
            "No Windows, diálogos ApplicationModal impedem entrega de sinais "
            "enfileirados de threads worker, fazendo LogProgressDialog travar. "
            "Solução: FaceAuditProjectService.open() roda na main thread com "
            "QApplication.setOverrideCursor(WaitCursor). O ProjectIOWorker existe mas "
            "não é usado para abertura.",
        ),
        (
            "'Adicionar Recurso' vive no painel esquerdo, não na toolbar",
            "O ResourceTreeWidget tem um QToolButton com MenuButtonPopup mostrando "
            "4 opções (Dataset / Split Config / Mining Config / Pipeline Config). "
            "O LeftPanelWidget repassa os sinais ao controller. "
            "O botão é desabilitado até um projeto estar aberto.",
        ),
        (
            "Descoberta de recursos = scan de definições + scan de disco",
            "_collect_project_resources() no controller faz dois passes: "
            "(1) recursos declarados nas LRStrategyDefinitions, "
            "(2) arquivos em disco via storage.list_artifacts() por tipo importável. "
            "Isso garante que arquivos importados apareçam na árvore mesmo antes de "
            "serem referenciados por uma estratégia.",
        ),
        (
            "Contexto descartado a cada edição",
            "Qualquer modificação na estratégia (adicionar/remover/reordenar instrução, "
            "alterar parâmetros) chama _reset_execution_state(), descartando o contexto e "
            "os IDs executados. Isso garante que os recursos no contexto sempre refletem "
            "exatamente as instruções já executadas — sem ambiguidade.",
        ),
        (
            "Placeholders de output sincronizados automaticamente",
            "_sync_all_output_placeholders() é chamado a cada _on_strategy_modified(). "
            "Itera todas as instruções, coleta os outputs declarados via "
            "get_output_resource_placeholders(), adiciona placeholders faltantes em "
            "LRStrategyDefinition.resources e remove placeholders órfãos (sem payload "
            "e sem artifact_ref) de instruções removidas ou renomeadas. Recursos já "
            "persistidos nunca são removidos automaticamente.",
        ),
        (
            "Persistência automática após execução individual",
            "Após cada execução bem-sucedida de instrução, _persist_produced_resources() "
            "itera os produced_resource_ids do InstructionExecutionResult e chama "
            "StrategyExecutor.persist_produced_resource() para cada um. O artefato é salvo "
            "em disco e a definição da estratégia é atualizada com o artifact_ref. "
            "Isso garante que os recursos produzidos sobrevivam ao fechamento do app.",
        ),
        (
            "Layout de 3 colunas com cache de gráficos por instrução",
            "Cada aba do editor é um QSplitter horizontal com 3 colunas "
            "(lista ~200px / config ~400px / resultados ~600px — a direita é "
            "mais larga porque acomoda gráficos). O InstructionResultsPanel mantém "
            "um cache <code>_cards: Dict[instruction_id, _ResultsCard]</code> via "
            "QStackedWidget interno: trocar de instrução é só um swap de página, sem "
            "redraw do gráfico. Para STRATEGY_EVALUATION (4 abas matplotlib), as "
            "abas são lazy-rendered — só desenham na primeira ativação, e o "
            "ValidationResultWidget mantém um cache interno por aba. Re-execução "
            "de uma instrução invalida o card via "
            "<code>StrategyEditorWidget.invalidate_results_for_instruction()</code> "
            "(drop + redraw). Trajetória 3D de pipeline NÃO renderiza inline — "
            "abre apenas via botão em GraphViewerDialog (3D precisa de espaço "
            "para rotação/zoom).",
        ),
        (
            "Lazy hydrate de score datasets grandes em disco",
            "Quando o usuário seleciona uma instrução cujo output é "
            "SIMILARITY_SCORES ou CALIBRATED_SCORES e o payload está apenas em "
            "disco (pós-reabertura de projeto), o InstructionResultsPanel mostra "
            "um botão <b>'📊 Carregar e visualizar'</b> em vez de hidratar "
            "automaticamente — evita travar a main thread com leituras de "
            "parquets de 5M+ pares. O clique emite "
            "<code>hydrate_requested(instruction, resource_id)</code>; o controller "
            "deve tratar (via worker se quiser não bloquear). Validation e casework "
            "results são pequenos (.pkl) — hidratam direto.",
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
# SEÇÃO 16 — ESTRUTURA DE PASTAS
# ---------------------------------------------------------------------------

def build_s16(story, S):
    section_banner(story, S, "16. Estrutura de Pastas")

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
            ["faceaudit/services/two_covariance_plda_service.py", "Nova arch",
             "Implementação numpy pura do PLDA de duas covariâncias "
             "(Brümmer &amp; de Villiers 2010): fit_two_covariance em forma fechada, "
             "score_two_covariance_pair/_matrix por log-LR vetorizado. "
             "Usado pelo PLDASimilarityModel quando plda_variant=TWO_COVARIANCE."],
            ["faceaudit/services/hubness_analysis_service.py", "Nova arch",
             "HubnessAnalysisService — diagnóstico de viés topológico em alta "
             "dimensionalidade. compute_analysis (KNN exaustivo O(N²) com "
             "batching B×C, soma por exclusão, label encoding e top-K contínuo) "
             "+ 4 estudos derivados + 2 ablações. Aceita logger_callback "
             "opcional pra integração com o LogProgressDialog da UI."],
            ["faceaudit/services/hubness_plot_service.py", "Nova arch",
             "HubnessPlotService — 20 métodos matplotlib/seaborn para visualizar "
             "os DTOs do HubnessAnalysisService. Reutilizados pelo "
             "HubnessReportWidget via monkey-patch local de plt.show + captura de "
             "plt.gcf() + transplante via PNG buffer (evita refatorar 20 métodos)."],
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
             "ValidationRuleBase + 7 regras concretas: DatasetIsolationRule, "
             "DatasetOverlapRequiredRule, FisherRule, MinCalibrationPairsRule, HoldoutLeakageRule, "
             "DualPipelineDimensionCompatibilityRule, DualPipelineSubsequenceRule"],
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
            ["interface/tabs/strategy_manager_v2/", "Nova arch",
             "Frontend V2 — editor com 3 abas (Preparação/Validação/Caso) e 3 "
             "colunas (lista/config/resultados), execução individual + workers "
             "em thread separada"],
            ["interface/tabs/strategy_manager_v2/strategy_editor/result_widgets/",
             "Nova arch",
             "Widgets de plot reutilizáveis (inline na coluna de resultados ou "
             "em dialog full-screen): ScoreDistributionWidget, "
             "ValidationResultWidget (4 tabs lazy), CaseworkVisualizationWidget, "
             "HubnessReportWidget (5 tabs lazy: topology/stratification/paired/"
             "quality/density), HubnessAblationWidget (single-plot KDE overlay). "
             "Cada um aceita flags show_header/show_toolbar/show_help."],
            ["interface/tabs/strategy_manager_v2/strategy_editor/instruction_results_panel.py",
             "Nova arch",
             "InstructionResultsPanel — 3ª coluna do editor: outputs com estado "
             "runtime + gráficos inline + botão 🔍 Expandir. Cache de _ResultsCard "
             "por instrução, lazy hydrate de payloads grandes em disco."],
            ["interface/ (demais)", "Arch antiga",
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
        "Versão 1.8 · Maio 2026",
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
    build_s16(story, S)

    doc.build(story)
    print(f"PDF gerado: {output_path}")


if __name__ == "__main__":
    import os
    out = os.path.join(
        os.path.dirname(__file__),
        "FaceAudit_Nova_Arquitetura.pdf"
    )
    build_pdf(out)
