"""PDF 원본 형식 저장: PyMuPDF redaction으로 글자를 실제로 지우고(절대 규칙 5) 그 자리에 가명코드를 쓴다.

- 쪽 글자는 글자 단위 좌표(rawdict)로 찾는다. 줄이 바뀌면 줄마다 지운다.
- 그림과 표의 선은 지우지 않는다(images=NONE, graphics=LINE_ART_NONE). 지운 자리는 흰색으로 채운다.
- 문서 속성, XMP, 주석, 책갈피, 입력 칸 값도 바꾼다.
- 그림 속 글자는 바꾸지 못한다(사각지대).
"""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf

from privacy_operator.reader.blocks import OCR, Block
from privacy_operator.reader.ocr import rects_for_span
from privacy_operator.writer.plan import ReplacementPlan

MALGUN = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "malgun.ttf"
MIN_FONT = 5.0


def _font() -> tuple[dict, pymupdf.Font]:
    # 글꼴 자원 이름은 문서에 이미 있는 이름과 겹치면 안 된다. 겹치면 원본의 부분 글꼴(쓴 글자만 든 것)을
    # 다시 써서 가명코드 글자가 네모로 나온다.
    if MALGUN.is_file():
        return {"fontname": "POAliasCode", "fontfile": str(MALGUN)}, pymupdf.Font(fontfile=str(MALGUN))
    return {"fontname": "korea"}, pymupdf.Font("korea")


def _page_chars(page) -> tuple[str, list]:
    """쪽 글을 한 줄 문자열로. chars[i] = (bbox, origin, size, line_no) 또는 줄바꿈이면 None."""
    text, chars, line_no = [], [], 0
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                for ch in span["chars"]:
                    text.append(ch["c"])
                    chars.append((pymupdf.Rect(ch["bbox"]), ch["origin"], span["size"], line_no))
            text.append("\n")
            chars.append(None)
            line_no += 1
    return "".join(text), chars


def _ocr_redact_page(
    page,
    blocks: list[Block],
    plan: ReplacementPlan,
    font_args: dict,
    font: pymupdf.Font,
    redact_style: str,
) -> None:
    """Physically erase image pixels under OCR-matched PII and optionally draw alias text."""
    fill = (1, 1, 1) if redact_style == "white_alias" else (0, 0, 0)
    txt_color = (0, 0, 0) if redact_style == "white_alias" else (1, 1, 1)
    writes: list[tuple[pymupdf.Rect, str]] = []
    for blk in blocks:
        if not blk.location.ocr_boxes:
            continue
        for s, e, i in plan.pick(blk.text, wrapped=True):
            replacement = plan.rules[i].replacement
            rects = rects_for_span(blk, s, e, pad_x=2.5, pad_y=1.5)
            first = True
            for rx0, ry0, rx1, ry1 in rects:
                rect = pymupdf.Rect(rx0, ry0, rx1, ry1) & page.rect
                if rect.is_empty:
                    continue
                page.add_redact_annot(rect, fill=fill)
                if first:
                    writes.append((rect, replacement))
                    first = False
    if not writes:
        return
    page.apply_redactions(
        images=pymupdf.PDF_REDACT_IMAGE_PIXELS,
        graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
        text=pymupdf.PDF_REDACT_TEXT_REMOVE,
    )
    if redact_style != "black_only":
        for rect, replacement in writes:
            width = font.text_length(replacement, fontsize=1) or 1.0
            fs = max(MIN_FONT, min(rect.height * 0.72, rect.width / width))
            baseline_y = min(rect.y1 - 1.5, rect.y0 + (rect.height + fs * 0.75) / 2.0)
            page.insert_text((rect.x0 + 1.0, baseline_y), replacement, fontsize=fs, color=txt_color, **font_args)


def write_pdf(
    src: Path,
    dst: Path,
    plan: ReplacementPlan,
    ocr_blocks: list[Block] | None = None,
    redact_style: str = "black_alias",
) -> None:
    font_args, font = _font()
    pymupdf.TOOLS.mupdf_display_errors(False)
    ocr_by_page: dict[int, list[Block]] = {}
    for blk in ocr_blocks or ():
        if blk.kind == OCR and blk.location.page is not None:
            ocr_by_page.setdefault(blk.location.page, []).append(blk)

    doc = pymupdf.open(src)
    try:
        for pno, page in enumerate(doc, start=1):
            text, chars = _page_chars(page)
            writes = []
            # 쪽 글에는 줄마다 줄바꿈이 들어가 있어 줄에 걸친 값도 찾는다
            for s, e, i in plan.pick(text, wrapped=True):
                replacement = plan.rules[i].replacement
                lines: dict[int, list] = {}
                for c in chars[s:e]:
                    if c is not None:
                        lines.setdefault(c[3], []).append(c)
                first = True
                for line in lines.values():
                    rect = pymupdf.Rect(line[0][0])
                    for c in line[1:]:
                        rect |= c[0]
                    page.add_redact_annot(rect, fill=(1, 1, 1))
                    if first:
                        writes.append((rect, line[0][1], line[0][2], replacement))
                        first = False
            if writes:
                page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                                      graphics=pymupdf.PDF_REDACT_LINE_ART_NONE,
                                      text=pymupdf.PDF_REDACT_TEXT_REMOVE)
                for rect, origin, size, replacement in writes:
                    width = font.text_length(replacement, fontsize=1) or 1
                    fs = max(MIN_FONT, min(size, rect.width / width))
                    page.insert_text((rect.x0, origin[1]), replacement, fontsize=fs, **font_args)

            if pno in ocr_by_page:
                _ocr_redact_page(page, ocr_by_page[pno], plan, font_args, font, redact_style)

            for annot in page.annots() or []:
                info = annot.info
                new = {k: plan.replace(info.get(k) or "") for k in ("content", "title", "subject")}
                if any(new[k] != (info.get(k) or "") for k in new):
                    annot.set_info(**new)
                    annot.update()
            for widget in page.widgets() or []:
                if isinstance(widget.field_value, str) and plan.replace(widget.field_value) != widget.field_value:
                    widget.field_value = plan.replace(widget.field_value)
                    widget.update()

        meta = {k: plan.replace(v) if isinstance(v, str) else v for k, v in (doc.metadata or {}).items()
                if k not in ("format", "encryption")}
        doc.set_metadata(meta)
        doc.del_xml_metadata()  # XMP에는 작성자 등이 따로 들어 있다
        toc = doc.get_toc(simple=True)
        if toc:
            doc.set_toc([[lvl, plan.replace(title), pno] for lvl, title, pno in toc])
        doc.subset_fonts()
        doc.save(dst, garbage=4, deflate=True, clean=True)
    finally:
        doc.close()
