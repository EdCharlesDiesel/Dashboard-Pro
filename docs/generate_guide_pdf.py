"""Render docs/System_Guide.md to assets/Dashboard_Pro_System_Guide.pdf.

A small, purpose-built Markdown -> reportlab converter — not a general
Markdown engine. It only needs to understand the exact subset of Markdown
this repo's guide is written in: #/##/### headers, pipe tables, "- " bullet
lists, **bold** spans, and "---" horizontal rules. Re-run this after editing
docs/System_Guide.md so the in-app PDF stays in sync.

Usage:
    python docs/generate_guide_pdf.py
"""
from __future__ import annotations

import os
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, ListFlowable, ListItem,
                                PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MD_PATH = os.path.join(_HERE, "System_Guide.md")
_PDF_PATH = os.path.join(_HERE, "..", "assets", "Dashboard_Pro_System_Guide.pdf")

_ACCENT = colors.HexColor("#00873c")   # print-friendly stand-in for the terminal green
_DARK = colors.HexColor("#1a1a1a")
_GREY = colors.HexColor("#666666")
_BORDER = colors.HexColor("#cccccc")


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"  # symbols & pictographs, emoticons, transport, supplemental
    "\U00002600-\U000027BF"  # misc symbols & dingbats (includes many emoji)
    "\U0001F1E6-\U0001F1FF"  # regional indicators (flag emoji)
    "\U0000FE0F"             # variation selector-16
    "]+"
)
_ASCII_MAP = {
    "→": "->", "←": "<-", "⇒": "=>", "↔": "<->",
    "–": "-", "—": "-",             # en/em dash
    "‘": "'", "’": "'",   # curly single quotes
    "“": '"', "”": '"',   # curly double quotes
    "…": "...",
}


def _sanitize_for_pdf(text: str) -> str:
    """Strip emoji and force everything else to plain ASCII.

    The in-app Markdown view (rendered by a browser via Streamlit) keeps the
    original emoji/typography unchanged — this sanitization only applies to
    the PDF export, which uses reportlab's built-in Helvetica. Rather than
    rely on exactly which above-Latin-1 characters WinAnsiEncoding does or
    doesn't cover (unverifiable here without a PDF page-rasterizer), plain
    ASCII is simply guaranteed to render correctly.
    """
    text = _EMOJI_RE.sub("", text)
    for char, replacement in _ASCII_MAP.items():
        text = text.replace(char, replacement)
    text = "".join(c if ord(c) <= 0x7E else "" for c in text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def _inline(text: str) -> str:
    """**bold** / *italic* -> <b>/<i> tags; sanitize first, escape XML chars."""
    text = _sanitize_for_pdf(text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"\0BOLD\1\0/BOLD", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = text.replace("\0BOLD", "<b>").replace("\0/BOLD", "</b>")
    return text


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1", parent=ss["Heading1"], textColor=_ACCENT,
                          fontSize=20, spaceBefore=18, spaceAfter=10))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=_DARK,
                          fontSize=15, spaceBefore=14, spaceAfter=8))
    ss.add(ParagraphStyle("H3", parent=ss["Heading3"], textColor=_DARK,
                          fontSize=12.5, spaceBefore=10, spaceAfter=6))
    ss.add(ParagraphStyle("Body", parent=ss["BodyText"], fontSize=9.5,
                          leading=13.5, spaceAfter=6))
    ss.add(ParagraphStyle("Cell", parent=ss["BodyText"], fontSize=8.3,
                          leading=11))
    ss.add(ParagraphStyle("CellHead", parent=ss["Cell"], textColor=colors.white,
                          fontName="Helvetica-Bold"))
    ss.add(ParagraphStyle("GuideTitle", parent=ss["Title"], textColor=_ACCENT,
                          fontSize=26, spaceAfter=6))
    ss.add(ParagraphStyle("GuideSubTitle", parent=ss["BodyText"], textColor=_GREY,
                          fontSize=12, spaceAfter=24, alignment=1))
    return ss


def _parse_table(lines: list, i: int, styles) -> tuple:
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2  # skip the |---|---| separator row
    rows = [header]
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1

    n_cols = len(header)
    page_width = LETTER[0] - 1.4 * inch
    col_width = page_width / n_cols

    data = []
    for r, row in enumerate(rows):
        style = styles["CellHead"] if r == 0 else styles["Cell"]
        data.append([Paragraph(_inline(c), style) for c in row])

    tbl = Table(data, colWidths=[col_width] * n_cols, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.5, _BORDER),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl, i


def _reflow(raw_lines: list) -> list:
    """Join hard-wrapped paragraph continuation lines into one logical line
    each, so e.g. a *...* italic byline or a long sentence spanning two
    source lines is classified (and bolded/italicized) as a single unit."""
    special_prefixes = ("#", "|", "- ", "---")
    out: list = []
    for line in raw_lines:
        stripped = line.strip()
        is_special = (not stripped) or stripped.startswith(special_prefixes)
        if (not is_special and out
                and out[-1]
                and not out[-1].strip().startswith(special_prefixes)):
            out[-1] = out[-1].rstrip() + " " + stripped
        else:
            out.append(stripped)
    return out


def build_story(md_text: str, styles) -> list:
    story: list = []
    lines = _reflow(md_text.splitlines())
    bullets: list = []
    i = 0
    seen_title = False

    def flush_bullets():
        if bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(_inline(b), styles["Body"]), leftIndent=6)
                 for b in bullets],
                bulletType="bullet", start="circle", leftIndent=14,
            ))
            bullets.clear()

    while i < len(lines):
        stripped = lines[i]

        if not stripped:
            flush_bullets()
            i += 1
            continue
        if stripped == "---":
            flush_bullets()
            story.append(HRFlowable(width="100%", color=_BORDER, thickness=0.75,
                                    spaceBefore=6, spaceAfter=10))
            i += 1
            continue
        if stripped.startswith("# "):
            flush_bullets()
            if not seen_title:
                # The guide's own H1 doubles as the PDF's cover title.
                story.append(Paragraph(_inline(stripped[2:]), styles["GuideTitle"]))
                seen_title = True
            else:
                story.append(PageBreak())
                story.append(Paragraph(_inline(stripped[2:]), styles["H1"]))
            i += 1
            continue
        if stripped.startswith("## "):
            flush_bullets()
            story.append(Paragraph(_inline(stripped[3:]), styles["H2"]))
            i += 1
            continue
        if stripped.startswith("### "):
            flush_bullets()
            story.append(Paragraph(_inline(stripped[4:]), styles["H3"]))
            i += 1
            continue
        if stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            flush_bullets()
            style = styles["GuideSubTitle"] if len(story) == 1 else styles["Body"]
            story.append(Paragraph(f"<i>{_inline(stripped[1:-1])}</i>", style))
            i += 1
            continue
        if stripped.startswith("|"):
            flush_bullets()
            tbl, i = _parse_table(lines, i, styles)
            story.append(tbl)
            story.append(Spacer(1, 10))
            continue
        if stripped.startswith("- "):
            bullets.append(stripped[2:])
            i += 1
            continue

        flush_bullets()
        story.append(Paragraph(_inline(stripped), styles["Body"]))
        i += 1

    flush_bullets()
    return story


def main() -> None:
    with open(_MD_PATH, "r", encoding="utf-8") as fh:
        md_text = fh.read()

    styles = _styles()
    story = build_story(md_text, styles)

    os.makedirs(os.path.dirname(_PDF_PATH), exist_ok=True)
    doc = SimpleDocTemplate(
        _PDF_PATH, pagesize=LETTER,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        title="Dashboard Pro — System Guide & Trading Playbook",
    )
    doc.build(story)
    print(f"Wrote {_PDF_PATH}")


if __name__ == "__main__":
    main()
