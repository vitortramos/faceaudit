"""Gera Manual_Gerenciador_de_Datasets.pdf a partir dos capítulos .md.

Uso:
    python docs/user_guide/dataset_manager/build_pdf.py

Saída:
    docs/user_guide/dataset_manager/Manual_Gerenciador_de_Datasets.pdf

Converte os arquivos 0X_*.md (em ordem) para um PDF único com capa,
sumário, e cada capítulo iniciando em nova página. Implementa um parser
de markdown minimalista cobrindo apenas o subconjunto usado nos
capítulos: headings, parágrafos, **bold**, *italic*, `code`, blockquotes,
listas (ordenadas e não), tabelas, imagens, links. NÃO suporta nested
lists nem code blocks multi-linha — esses não são usados nas docs.

Não tem dependência externa além de reportlab (já instalada no repo).
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_THIS_DIR = Path(__file__).resolve().parent
_ASSETS = _THIS_DIR / "assets"
_OUT_PDF = _THIS_DIR / "Manual_Gerenciador_de_Datasets.pdf"

PAGE_W, PAGE_H = A4
USABLE_W = PAGE_W - 2 * 2 * cm  # = page minus left/right margins (2cm each)


# ── Estilos ────────────────────────────────────────────────────────────────

_styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1", parent=_styles["Heading1"],
    fontSize=20, leading=24,
    textColor=colors.HexColor("#1a3d6e"),
    spaceAfter=12, spaceBefore=6,
    fontName="Helvetica-Bold",
)
H2 = ParagraphStyle(
    "H2", parent=_styles["Heading2"],
    fontSize=15, leading=19,
    textColor=colors.HexColor("#1a3d6e"),
    spaceAfter=8, spaceBefore=14,
    fontName="Helvetica-Bold",
)
H3 = ParagraphStyle(
    "H3", parent=_styles["Heading3"],
    fontSize=12.5, leading=15,
    textColor=colors.HexColor("#324a6e"),
    spaceAfter=5, spaceBefore=10,
    fontName="Helvetica-Bold",
)
H4 = ParagraphStyle(
    "H4", parent=_styles["Heading4"],
    fontSize=11, leading=14,
    textColor=colors.HexColor("#444"),
    spaceAfter=4, spaceBefore=8,
    fontName="Helvetica-Bold",
)
BODY = ParagraphStyle(
    "Body", parent=_styles["BodyText"],
    fontSize=10, leading=14, spaceAfter=6,
    fontName="Helvetica",
)
BODY_GREY = ParagraphStyle(
    "BodyGrey", parent=BODY,
    textColor=colors.HexColor("#666"),
)
QUOTE = ParagraphStyle(
    "Quote", parent=BODY,
    leftIndent=14, rightIndent=8,
    textColor=colors.HexColor("#444"),
    borderColor=colors.HexColor("#a0b4cf"),
    borderPadding=6, borderWidth=0,  # left bar drawn via Table below
    backColor=colors.HexColor("#f0f4fa"),
)
BULLET = ParagraphStyle(
    "Bullet", parent=BODY,
    leftIndent=18, bulletIndent=4, spaceAfter=2,
)
NUMBERED = ParagraphStyle(
    "Numbered", parent=BODY,
    leftIndent=20, bulletIndent=4, spaceAfter=2,
)
TABLE_CELL = ParagraphStyle(
    "TableCell", parent=BODY,
    fontSize=8.5, leading=11, spaceAfter=0,
)
TABLE_HEADER = ParagraphStyle(
    "TableHeader", parent=TABLE_CELL,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a3d6e"),
)
CAPTION = ParagraphStyle(
    "Caption", parent=BODY,
    fontSize=8.5, leading=11, alignment=1,  # center
    textColor=colors.HexColor("#666"),
    spaceAfter=10, spaceBefore=0,
)


# ── Inline conversion (markdown → reportlab HTML-ish) ───────────────────────

_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")
_STRIKE_RE = re.compile(r"~~([^~]+)~~")


def inline_md(text: str) -> str:
    """Convert markdown inline syntax to reportlab Paragraph markup.

    Uses placeholder tokens to protect code spans and links from being
    re-processed by the bold/italic regexes.
    """
    text = html.escape(text, quote=False)

    # 1) Extract code spans → placeholders
    code_spans: list[str] = []
    def _store_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans) - 1}\x00"
    text = _CODE_RE.sub(_store_code, text)

    # 2) Extract link spans → placeholders
    link_spans: list[tuple[str, str]] = []
    def _store_link(m):
        link_spans.append((m.group(1), m.group(2)))
        return f"\x00LINK{len(link_spans) - 1}\x00"
    text = _LINK_RE.sub(_store_link, text)

    # 3) Bold (**) BEFORE italic (*) so that **text** doesn't get
    # mis-tokenized as italic.
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)
    text = _STRIKE_RE.sub(r"<strike>\1</strike>", text)

    # 4) Restore links (rendered as underlined blue text)
    def _restore_link(m):
        idx = int(m.group(1))
        label, _url = link_spans[idx]
        return f'<font color="#1a73e8"><u>{label}</u></font>'
    text = re.sub(r"\x00LINK(\d+)\x00", _restore_link, text)

    # 5) Restore code spans (escape any HTML chars inside)
    def _restore_code(m):
        idx = int(m.group(1))
        content = code_spans[idx]
        return (
            f'<font face="Courier" backColor="#f0f0f0" '
            f'color="#a5305b">{content}</font>'
        )
    text = re.sub(r"\x00CODE(\d+)\x00", _restore_code, text)

    return text


# ── Block parser ───────────────────────────────────────────────────────────

def _strip_comments(md_text: str) -> str:
    """Remove HTML-style comments (rare in our docs)."""
    return re.sub(r"<!--.*?-->", "", md_text, flags=re.DOTALL)


def _split_blocks(md_text: str) -> List[str]:
    """Split markdown into top-level blocks separated by blank lines."""
    lines = md_text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []
    in_table = False
    in_code = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            cur.append(line)
            continue
        if in_code:
            cur.append(line)
            continue
        if "|" in line and not in_code:
            if not in_table and re.match(r"^\s*\|", line):
                # Starting a table block
                if cur and cur[-1].strip():
                    # flush previous block
                    blocks.append("\n".join(cur).strip())
                    cur = []
                in_table = True
        else:
            if in_table:
                in_table = False
                blocks.append("\n".join(cur).strip())
                cur = []
        if not stripped and not in_table:
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def _is_heading(block: str) -> Optional[Tuple[int, str]]:
    m = re.match(r"^(#{1,4})\s+(.+)$", block.split("\n")[0])
    if m:
        return len(m.group(1)), m.group(2).strip()
    return None


def _is_image(block: str) -> Optional[Tuple[str, str]]:
    m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", block.strip())
    if m:
        return m.group(1), m.group(2)
    return None


def _is_blockquote(block: str) -> bool:
    return all(line.strip().startswith(">") for line in block.split("\n"))


def _is_bullet_list(block: str) -> bool:
    return all(
        re.match(r"^\s*[-*]\s+", line) for line in block.split("\n") if line.strip()
    )


def _is_numbered_list(block: str) -> bool:
    return all(
        re.match(r"^\s*\d+\.\s+", line) for line in block.split("\n") if line.strip()
    )


def _is_table(block: str) -> bool:
    lines = [l for l in block.split("\n") if l.strip()]
    if len(lines) < 2:
        return False
    return all("|" in l for l in lines) and re.match(
        r"^\s*\|?[\s:|-]+\|?\s*$", lines[1]
    )


def _is_code_block(block: str) -> bool:
    return block.startswith("```") and block.rstrip().endswith("```")


# ── Flowables ──────────────────────────────────────────────────────────────

def make_heading(text: str, level: int):
    style = {1: H1, 2: H2, 3: H3, 4: H4}.get(level, H4)
    return Paragraph(inline_md(text), style)


def make_paragraph(text: str):
    return Paragraph(inline_md(text.replace("\n", " ")), BODY)


def make_bullet_list(block: str):
    items = []
    for line in block.split("\n"):
        m = re.match(r"^\s*[-*]\s+(.+)$", line)
        if m:
            items.append(Paragraph("• " + inline_md(m.group(1)), BULLET))
    return items


def make_numbered_list(block: str):
    items = []
    for line in block.split("\n"):
        m = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if m:
            items.append(Paragraph(
                f"{m.group(1)}. {inline_md(m.group(2))}", NUMBERED,
            ))
    return items


def make_blockquote(block: str):
    text = "\n".join(
        re.sub(r"^\s*>\s?", "", line) for line in block.split("\n")
    ).strip()
    para = Paragraph(inline_md(text.replace("\n", " ")), QUOTE)
    # Wrap in a table with a colored left border
    tbl = Table([[para]], colWidths=[USABLE_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4fa")),
        ("LINEBEFORE", (0, 0), (-1, -1), 3, colors.HexColor("#1a73e8")),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


def make_table(block: str):
    """Parse a markdown table to a reportlab Table."""
    lines = [l for l in block.split("\n") if l.strip()]
    rows: List[List[str]] = []
    for i, line in enumerate(lines):
        if i == 1:
            continue  # skip alignment row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    if not rows:
        return Paragraph(inline_md(block), BODY)

    # Convert to Paragraphs
    converted = []
    for i, row in enumerate(rows):
        style = TABLE_HEADER if i == 0 else TABLE_CELL
        converted.append([Paragraph(inline_md(cell), style) for cell in row])

    n_cols = max(len(r) for r in converted)
    # Pad rows to uniform length
    for r in converted:
        while len(r) < n_cols:
            r.append(Paragraph("", TABLE_CELL))

    col_w = USABLE_W / n_cols
    tbl = Table(converted, colWidths=[col_w] * n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")),
        ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#1a3d6e")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, colors.HexColor("#d0d6dc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def make_image(alt: str, path: str, base_dir: Path):
    """Insert an image; resolve `path` relative to base_dir."""
    img_path = (base_dir / path).resolve()
    if not img_path.exists():
        return Paragraph(
            f"<i>[Imagem não encontrada: {html.escape(path)}]</i>", BODY_GREY,
        )

    # Compute size: fit width to 90% of usable width
    from PIL import Image as PILImage
    with PILImage.open(img_path) as pil:
        w, h = pil.size

    max_w = USABLE_W * 0.92
    max_h = PAGE_H * 0.6
    scale = min(max_w / w, max_h / h, 1.0)
    img_w = w * scale
    img_h = h * scale

    img = Image(str(img_path), width=img_w, height=img_h)
    img.hAlign = "CENTER"
    cap = Paragraph(inline_md(alt), CAPTION) if alt else None
    return KeepTogether([img, cap] if cap else [img])


def make_code_block(block: str):
    """Render a fenced code block as monospace box."""
    inner = block.strip().strip("`")
    # Drop language hint line if present
    lines = inner.split("\n")
    if lines and not lines[0].strip().startswith(" ") and len(lines[0].split()) == 1:
        lines = lines[1:]
    text = "\n".join(lines).rstrip()
    para = Paragraph(
        f'<font face="Courier" size="9">{html.escape(text).replace(chr(10), "<br/>")}</font>',
        ParagraphStyle(
            "Code", parent=BODY,
            backColor=colors.HexColor("#f5f5f5"),
            borderColor=colors.HexColor("#d0d0d0"),
            borderWidth=0.5, borderPadding=8,
            leading=12, spaceAfter=8, spaceBefore=4,
            fontName="Courier",
        ),
    )
    return para


# ── Markdown → flowables ───────────────────────────────────────────────────

def parse_markdown_to_flowables(md_text: str, base_dir: Path) -> List:
    """Top-level: parse the markdown text and return a list of flowables."""
    md_text = _strip_comments(md_text)
    flowables: List = []

    for block in _split_blocks(md_text):
        if _is_code_block(block):
            flowables.append(make_code_block(block))
            continue
        h = _is_heading(block)
        if h:
            level, text = h
            # Render any remaining lines below the heading as a paragraph
            lines = block.split("\n")
            flowables.append(make_heading(text, level))
            if len(lines) > 1:
                rest = "\n".join(lines[1:]).strip()
                if rest:
                    flowables.append(make_paragraph(rest))
            continue
        img = _is_image(block)
        if img:
            alt, path = img
            flowables.append(make_image(alt, path, base_dir))
            continue
        if _is_blockquote(block):
            flowables.append(make_blockquote(block))
            continue
        if _is_table(block):
            flowables.append(make_table(block))
            continue
        if _is_bullet_list(block):
            flowables.extend(make_bullet_list(block))
            continue
        if _is_numbered_list(block):
            flowables.extend(make_numbered_list(block))
            continue
        # Default: paragraph
        flowables.append(make_paragraph(block))

    return flowables


# ── Construção do PDF ─────────────────────────────────────────────────────

CHAPTERS = [
    ("01_visao_geral.md", "Visão geral"),
    ("02_criando_dataset.md", "Criando um dataset"),
    ("03_carregando_e_localizando.md", "Carregando e localizando"),
    ("04_navegando_dataset.md", "Navegando no dataset"),
    ("05_enriquecimento.md", "Enriquecimento"),
    ("06_filtros_e_estatisticas.md", "Filtros e estatísticas"),
    ("07_operacoes_estruturais.md", "Operações estruturais"),
    ("08_pipeline_de_imagens.md", "Pipeline de imagens"),
    ("09_auditoria_forense.md", "Auditoria forense"),
    ("10_integracao_estrategias.md", "Integração com Estratégias"),
]


def _drop_chapter_nav_footer(flowables: List) -> List:
    """Remove the last "[← anterior] · [próximo →]" paragraph from a chapter,
    which is irrelevant in a single-PDF context."""
    if not flowables:
        return flowables
    # Walk backwards looking for the footer paragraph
    for i in range(len(flowables) - 1, -1, -1):
        f = flowables[i]
        if isinstance(f, Paragraph):
            text = getattr(f, "text", "")
            if "Capítulo" in text and "→" in text:
                return flowables[:i]
            if "Início do manual" in text:
                return flowables[:i]
            # Heuristic: navigation lines are usually right at the end
            break
    return flowables


def build_cover():
    return [
        Spacer(1, 4 * cm),
        Paragraph(
            "<b>Manual do Usuário</b>",
            ParagraphStyle(
                "CoverTitle", parent=H1,
                fontSize=28, leading=34, alignment=1,
                textColor=colors.HexColor("#1a3d6e"),
            ),
        ),
        Spacer(1, 1 * cm),
        Paragraph(
            "Gerenciador de Datasets",
            ParagraphStyle(
                "CoverSub", parent=H1,
                fontSize=22, leading=28, alignment=1,
                textColor=colors.HexColor("#324a6e"),
            ),
        ),
        Spacer(1, 0.5 * cm),
        Paragraph(
            "FaceAudit — Plataforma forense de análise de imagem facial",
            ParagraphStyle(
                "CoverTag", parent=BODY,
                fontSize=11, leading=14, alignment=1,
                textColor=colors.HexColor("#666"),
            ),
        ),
        Spacer(1, 6 * cm),
        Paragraph(
            "Versão gerada automaticamente a partir dos arquivos<br/>"
            "<font face='Courier'>docs/user_guide/dataset_manager/*.md</font>",
            ParagraphStyle(
                "CoverFoot", parent=BODY,
                fontSize=9, leading=12, alignment=1,
                textColor=colors.HexColor("#888"),
            ),
        ),
        PageBreak(),
    ]


def build_toc():
    items = [Paragraph("Sumário", H1), Spacer(1, 0.4 * cm)]
    for i, (_, title) in enumerate(CHAPTERS, start=1):
        items.append(Paragraph(
            f"<b>{i:02d}</b> &nbsp;&nbsp; {title}",
            ParagraphStyle(
                "TOCItem", parent=BODY,
                fontSize=11.5, leading=20, leftIndent=12,
            ),
        ))
    items.append(PageBreak())
    return items


def add_page_number(canvas, doc):
    page_num = canvas.getPageNumber()
    if page_num <= 2:  # cover + TOC: no number
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, str(page_num - 2))
    canvas.drawRightString(
        PAGE_W - 2 * cm, 1.2 * cm,
        "Manual do Gerenciador de Datasets",
    )
    canvas.restoreState()


def main():
    flowables: List = []
    flowables.extend(build_cover())
    flowables.extend(build_toc())

    for filename, title in CHAPTERS:
        md_path = _THIS_DIR / filename
        if not md_path.exists():
            print(f"⚠ Capítulo não encontrado: {md_path}")
            continue

        md_text = md_path.read_text(encoding="utf-8")
        ch_flowables = parse_markdown_to_flowables(md_text, _THIS_DIR)
        ch_flowables = _drop_chapter_nav_footer(ch_flowables)
        # PageBreak before each chapter (cover/TOC already broke)
        flowables.extend(ch_flowables)
        flowables.append(PageBreak())

    # Remove trailing PageBreak
    if flowables and isinstance(flowables[-1], PageBreak):
        flowables.pop()

    doc = SimpleDocTemplate(
        str(_OUT_PDF),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="Manual do Gerenciador de Datasets — FaceAudit",
        author="FaceAudit",
    )
    doc.build(
        flowables,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    print(f"✅ PDF gerado: {_OUT_PDF}")
    print(f"   Tamanho: {_OUT_PDF.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
