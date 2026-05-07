"""Gera dataset_manager_roadmap.pdf a partir do conteúdo do brainstorm.

Uso: `python docs/_build_roadmap_pdf.py`
Saída: `docs/dataset_manager_roadmap.pdf`
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ── Estilos ───────────────────────────────────────────────────────────────

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"],
    fontSize=18, textColor=colors.HexColor("#1a3d6e"),
    spaceAfter=10, spaceBefore=4,
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"],
    fontSize=14, textColor=colors.HexColor("#1a3d6e"),
    spaceAfter=6, spaceBefore=14,
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"],
    fontSize=12, textColor=colors.HexColor("#444"),
    spaceAfter=4, spaceBefore=8,
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontSize=10, leading=14, spaceAfter=6,
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY,
    leftIndent=14, bulletIndent=4, spaceAfter=2,
)
SMALL = ParagraphStyle(
    "Small", parent=BODY, fontSize=8.5, textColor=colors.HexColor("#666"),
)
TABLE_CELL = ParagraphStyle(
    "TableCell", parent=BODY, fontSize=9, leading=12, spaceAfter=0,
)


def heading(text, level=2):
    style = {1: H1, 2: H2, 3: H3}[level]
    return Paragraph(text, style)


def body(text):
    return Paragraph(text, BODY)


def small(text):
    return Paragraph(text, SMALL)


def bullet(text):
    return Paragraph(f"• {text}", BULLET)


def feature_table(rows):
    """Tabela com 4 colunas: Funcionalidade | Valor | Custo | Notas."""
    header = [
        Paragraph("<b>Funcionalidade</b>", TABLE_CELL),
        Paragraph("<b>Valor</b>", TABLE_CELL),
        Paragraph("<b>Custo</b>", TABLE_CELL),
        Paragraph("<b>Notas</b>", TABLE_CELL),
    ]
    body_rows = [
        [Paragraph(c, TABLE_CELL) for c in row] for row in rows
    ]
    data = [header] + body_rows
    tbl = Table(
        data,
        colWidths=[5.2 * cm, 1.8 * cm, 1.8 * cm, 7.0 * cm],
        repeatRows=1,
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3d6e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#1a3d6e")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f3f6fa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


# ── Conteúdo ──────────────────────────────────────────────────────────────

THEMES = [
    {
        "title": "Tema 1 — Higiene de Identidade (alto valor forense)",
        "intro": (
            "O core do trabalho do perito é confiar nas identidades. Hoje o "
            "Dataset Manager mostra IDs mas não ajuda a auditá-las."
        ),
        "rows": [
            (
                "<b>Detectar identidades duplicadas</b><br/>"
                "(mesma pessoa em 2 labels distintos via clustering por similaridade)",
                "Alto", "Médio",
                "Reusa <i>ComparisonService</i> ou <i>HubnessAnalysisService</i>",
            ),
            (
                "<b>Detectar identidades misturadas</b><br/>"
                "(mesmo label cobrindo 2+ pessoas — outlier por similaridade intra-classe)",
                "Alto", "Médio",
                "Flag forense crítica",
            ),
            (
                "<b>Renomear identidade em lote</b><br/>"
                "(regex / find-replace + preview de quantos rows mudam)",
                "Médio", "Baixo",
                "Operação repetitiva no fluxo atual",
            ),
            (
                "<b>Mesclar duas identidades</b><br/>"
                "(2 IDs → 1 com confirmação visual)",
                "Médio", "Baixo",
                "Resolve falsos positivos do detector",
            ),
        ],
    },
    {
        "title": "Tema 2 — Visualização de Faces (gap visual)",
        "intro": (
            "A tabela mostra texto, mas o forense é visual. Hoje só dá pra ver "
            "imagem no detail panel quando clica numa linha."
        ),
        "rows": [
            (
                "<b>Modo galeria</b><br/>"
                "(grid de thumbnails clicáveis, toggle ao lado da tabela)",
                "Alto", "Médio",
                "Lazy-load + cache",
            ),
            (
                "<b>Hover preview</b><br/>"
                "(passar mouse no <i>image_path</i> mostra mini-thumb)",
                "Médio", "Baixo",
                "Tooltip personalizado",
            ),
            (
                "<b>Comparar pares de faces</b><br/>"
                "(selecionar 2 + side panel lado-a-lado)",
                "Médio", "Baixo",
                "Útil pra confirmar sósia",
            ),
        ],
    },
    {
        "title": "Tema 3 — Integração com V2 (fechamento do fluxo)",
        "intro": "",
        "rows": [
            (
                "<b>'Enviar pra Estratégia'</b><br/>"
                "(botão na toolbar abre V2 com o dataset já vinculado)",
                "Alto", "Baixo",
                "Fluxo um-clique",
            ),
            (
                "<b>'Adicionar à Biblioteca Global'</b><br/>"
                "(exporta o dataset atual pro <i>GlobalLibraryStorage</i>)",
                "Médio", "Baixo",
                "Fechamento do Pendrive Paradigm",
            ),
            (
                "<b>Re-enriquecer subconjunto</b><br/>"
                "(Filtro + 'enriquecer só os filtrados')",
                "Médio", "Baixo",
                "Economiza horas em datasets parcialmente enriquecidos",
            ),
        ],
    },
    {
        "title": "Tema 4 — Schema-by-Role: completar deferidos",
        "intro": "Itens que ficaram fora dos PRs 1–10.",
        "rows": [
            (
                "<b>Drag-reorder colunas</b><br/>"
                "(arrastar header pra reordenar)",
                "Médio", "Baixo",
                "Item mais pedido em UX de planilha",
            ),
            (
                "<b>Filtro com colunas custom</b><br/>"
                "(combos dinâmicos no filter_dialog)",
                "Médio", "Médio",
                "Reescreve _populate_combos + view",
            ),
            (
                "<b>Colunas computadas</b><br/>"
                "(fórmula simbólica tipo <i>yaw_abs = abs(yaw)</i>, read-only)",
                "Médio", "Médio",
                "Define DSL mínimo + executor",
            ),
            (
                "<b>Aliases editor</b><br/>"
                "(editar aliases na Properties dialog)",
                "Baixo", "Baixo",
                "Quase grátis (campo já existe)",
            ),
            (
                "<b>categorical_counts genérico</b><br/>"
                "(no DatasetStats → custom columns no Compare)",
                "Baixo", "Médio",
                "Refactor de DatasetStats + Service + Result",
            ),
        ],
    },
    {
        "title": "Tema 5 — Higiene de Dados (forense compliance)",
        "intro": "",
        "rows": [
            (
                "<b>Verificar paths órfãos</b><br/>"
                "(flagar faces cujo image_path não existe em disco)",
                "Alto", "Baixo",
                "Detecta corrupção silenciosa",
            ),
            (
                "<b>Detectar duplicatas exatas</b><br/>"
                "(mesmo hash em linhas diferentes)",
                "Médio", "Baixo",
                "Higiene básica",
            ),
            (
                "<b>Relatório de schema</b><br/>"
                "(% NaN por coluna, colunas vazias, inconsistências)",
                "Baixo", "Baixo",
                "Lê do column_coverage já persistido",
            ),
            (
                "<b>Histórico / audit log</b><br/>"
                "(quem mudou o que, quando — chain of custody)",
                "Alto<br/>(forense)", "Alto",
                "Precisa storage; PR transversal",
            ),
            (
                "<b>Export CSV/Excel</b><br/>"
                "(com seleção de colunas)",
                "Baixo–<br/>Médio", "Baixo",
                "Boa UX pra relatórios",
            ),
        ],
    },
]


def build_story():
    story = []

    # === Capa ===
    story.append(heading(
        "Dataset Manager — Roadmap de Funcionalidades", level=1
    ))
    story.append(small(
        "Brainstorm consolidado de candidatas a próximas features do "
        "Dataset Manager (pós-migração schema-by-role, PRs 1–10)."
    ))
    story.append(Spacer(1, 12))

    # === Estado atual ===
    story.append(heading("Estado atual (já entregue)", level=2))
    story.append(bullet(
        "<b>Toolbar:</b> Novo / Abrir / Fechar / Salvar / Salvar Como / "
        "Enriquecer / Juntar / Dividir / Comparar / Estatísticas / "
        "Analisar / Filtrar / Adicionar Coluna"
    ))
    story.append(bullet(
        "<b>Tabela</b> com edição célula-a-célula, vocabulário fechado "
        "renderizado como QComboBox, header context-menu (Propriedades, "
        "Remover, Mostrar/Ocultar)"
    ))
    story.append(bullet("<b>Detail panel</b> da face selecionada"))
    story.append(bullet(
        "<b>Diálogos:</b> Filter, Stats, Compare, Split, Enrich, Dataset "
        "Analysis (hard mining), New Dataset (template), Add Column, "
        "Column Properties"
    ))
    story.append(bullet(
        "<b>Schema-by-role</b> com column_specs na metadata do parquet, "
        "vocabulários fechados, alias resolution na leitura"
    ))

    # === Temas 1-5 ===
    for theme in THEMES:
        story.append(Spacer(1, 12))
        story.append(heading(theme["title"], level=2))
        if theme["intro"]:
            story.append(body(theme["intro"]))
        story.append(feature_table(theme["rows"]))

    story.append(PageBreak())

    # === Sugestão de priorização ===
    story.append(heading("Sugestão de priorização (top 3)", level=2))
    story.append(body("Por valor-por-LOC:"))
    story.append(bullet(
        "<b>Detectar identidades duplicadas/misturadas</b> (Tema 1) — única "
        "feature <i>forense</i> desta lista; reusa código existente do "
        "ComparisonService e HubnessAnalysisService."
    ))
    story.append(bullet(
        "<b>'Enviar pra Estratégia' + re-enriquecer subconjunto</b> (Tema 3) "
        "— fechamento do fluxo; usuário não precisa mais navegar entre abas."
    ))
    story.append(bullet(
        "<b>Modo galeria</b> (Tema 2) — gap visual gritante; once-and-done."
    ))

    # === Decisões abertas ===
    story.append(Spacer(1, 12))
    story.append(heading("Decisões abertas", level=2))
    story.append(bullet(
        "<b>Foco primário:</b> higiene de identidade (forense), UX (galeria), "
        "ou integração V2?"
    ))
    story.append(bullet(
        "<b>Drag-reorder de colunas é prioridade?</b> É o item de "
        "schema-by-role mais pedido em UX de planilha em geral."
    ))
    story.append(bullet(
        "<b>Audit log / chain of custody</b> está no radar do projeto? Se "
        "sim, é uma feature transversal grande que merece PR próprio."
    ))
    story.append(bullet(
        "<b>Existe alguma demanda real do uso atual</b> que não foi "
        "capturada no brainstorm? Tipo 'no caso X eu sempre tenho que "
        "fazer Y manualmente'?"
    ))

    # === Excluídos ===
    story.append(Spacer(1, 12))
    story.append(heading(
        "Itens explicitamente excluídos do escopo deste roadmap", level=2,
    ))
    story.append(bullet(
        "<b>EmbeddingDataset parquet com mesma metadata role-aware</b> — "
        "datasets derivados, não precisam tratamento separado."
    ))
    story.append(bullet(
        "<b>Refactor de _face_matches_config</b> em face_dataset.py para ser "
        "role-aware — hot-path, não estritamente necessário."
    ))

    story.append(Spacer(1, 18))
    story.append(small(
        "Documento gerado a partir do brainstorm da sessão de planejamento. "
        "Versão Markdown editável em <i>docs/dataset_manager_roadmap.md</i>."
    ))

    return story


def main():
    out_path = Path(__file__).parent / "dataset_manager_roadmap.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Dataset Manager — Roadmap",
        author="FaceAudit",
    )
    doc.build(build_story())
    print(f"PDF gerado: {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
