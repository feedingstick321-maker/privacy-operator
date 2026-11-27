"""PPTX 문서 읽기: 슬라이드 텍스트, 그룹 도형, 표, 발표자 노트, 문서 속성, 내장 그림 OCR."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from privacy_operator.reader.blocks import (
    CELL,
    COMMENT,
    HEADING,
    METADATA,
    PARAGRAPH,
    BlindSpot,
    Block,
    Location,
    ReadResult,
)
from privacy_operator.reader.convert import metadata_blocks
from privacy_operator.reader.extras import IMAGE
from privacy_operator.reader.ocr import OcrLine, PaddleOcrEngine, lines_to_blocks

_TITLE_PLACEHOLDERS = {
    PP_PLACEHOLDER.TITLE,
    PP_PLACEHOLDER.CENTER_TITLE,
    PP_PLACEHOLDER.VERTICAL_TITLE,
}

_RASTER_EXTS = {"png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp"}


def _is_title(shape) -> bool:
    try:
        return bool(shape.is_placeholder and shape.placeholder_format.type in _TITLE_PLACEHOLDERS)
    except Exception:  # noqa: BLE001
        return False


def _cell_text(cell) -> str:
    return "\n".join(p.text for p in cell.text_frame.paragraphs).strip()


def _table_blocks(table, slide_no: int, table_index: int) -> list[Block]:
    out: list[Block] = []
    grid = [[_cell_text(cell) for cell in row.cells] for row in table.rows]
    if not grid:
        return out
    first_row = grid[0]
    has_header = bool(table.first_row) and any(first_row)
    for r, row in enumerate(grid):
        for c, text in enumerate(row):
            if not text:
                continue
            header = first_row[c] if has_header and c < len(first_row) and first_row[c] else None
            row_label = next((left for left in reversed(row[:c]) if left), None)
            loc = Location(
                page=slide_no,
                table_index=table_index,
                row=r,
                col=c,
                header=header,
                header_path=(header,) if header else (),
                row_label=row_label,
            )
            is_hdr = has_header and r == 0
            out.append(Block(f"t{table_index:03d}-r{r:03d}-c{c:03d}", CELL, text, loc, is_header=is_hdr))
    return out


def _shape_type(shape):
    try:
        return shape.shape_type
    except NotImplementedError:
        return None


def _iter_shapes(shapes):
    for shape in shapes:
        if _shape_type(shape) == MSO_SHAPE_TYPE.GROUP:
            yield from _iter_shapes(shape.shapes)
        else:
            yield shape


def _can_raster_ocr(ext: str, blob: bytes) -> bool:
    if ext.lower().lstrip(".") not in _RASTER_EXTS:
        return False
    try:
        from PIL import Image

        with Image.open(io.BytesIO(blob)) as img:
            w, h = img.size
            # Skip tiny bullet icons/spacers while keeping banners and text screenshots
            return min(w, h) >= 20 and max(w, h) >= 60 and (w * h) >= 2500
    except Exception:  # noqa: BLE001
        return False


def _picture_part(shape):
    blip = getattr(getattr(shape._element, "blipFill", None), "blip", None)
    rid = getattr(blip, "rEmbed", None) if blip is not None else None
    if not rid:
        return None
    return shape.part.related_part(rid)


def read_pptx(path: Path, ocr_engine: PaddleOcrEngine | None = None) -> ReadResult:
    """Read PPTX slides, tables, speaker notes, core metadata, and embedded images."""
    prs = Presentation(str(path))
    blocks: list[Block] = []
    blind: list[BlindSpot] = []
    b_idx = 0
    table_index = 0
    ocr_cache: dict[bytes, list[OcrLine]] = {}

    for slide_no, slide in enumerate(prs.slides, start=1):
        pic_count = 0
        seen_media_on_slide: set[str] = set()
        for shape in _iter_shapes(slide.shapes):
            if shape.has_text_frame:
                text = "\n".join(p.text for p in shape.text_frame.paragraphs).strip()
                if text:
                    kind = HEADING if _is_title(shape) else PARAGRAPH
                    blocks.append(
                        Block(
                            f"s{slide_no:03d}-b{b_idx:04d}",
                            kind,
                            text,
                            Location(page=slide_no, block_index=b_idx),
                        )
                    )
                    b_idx += 1
            elif shape.has_table:
                blocks.extend(_table_blocks(shape.table, slide_no, table_index))
                table_index += 1
            elif _shape_type(shape) == MSO_SHAPE_TYPE.PICTURE or _picture_part(shape) is not None:
                try:
                    part = _picture_part(shape)
                    if part is None:
                        continue
                    partname = str(part.partname).lstrip("/")
                    ext = str(getattr(part, "ext", "") or getattr(shape.image, "ext", ""))
                    blob = part.blob
                except Exception:  # noqa: BLE001
                    continue
                if ocr_engine is None:
                    pic_count += 1
                    continue
                if partname in seen_media_on_slide:
                    continue
                seen_media_on_slide.add(partname)
                if not _can_raster_ocr(ext, blob):
                    continue
                digest = hashlib.sha256(blob).digest()
                if digest not in ocr_cache:
                    ocr_cache[digest] = ocr_engine.ocr_image_bytes(blob)
                lines = ocr_cache[digest]
                if lines:
                    prefix = f"s{slide_no:03d}-ocr{len(blocks):03d}"
                    blocks.extend(
                        lines_to_blocks(
                            lines,
                            page_no=slide_no,
                            ocr_source="pptx_media",
                            prefix=prefix,
                            media_path=partname,
                        )
                    )

        if pic_count > 0:
            blind.append(BlindSpot(IMAGE, page=slide_no, detail=str(pic_count)))

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            note_text = slide.notes_slide.notes_text_frame.text or ""
            if note_text.strip():
                blocks.append(
                    Block(
                        f"s{slide_no:03d}-note",
                        COMMENT,
                        note_text.strip(),
                        Location(page=slide_no, field="notes"),
                    )
                )

    cp = prs.core_properties
    props = {
        "author": cp.author,
        "lastModifiedBy": cp.last_modified_by,
        "title": cp.title,
        "subject": cp.subject,
        "comments": cp.comments,
        "keywords": cp.keywords,
    }
    blocks.extend(metadata_blocks(props, prefix="pptxprop"))
    return ReadResult(
        source_name=path.name,
        format="pptx",
        blocks=blocks,
        blind_spots=blind,
        page_count=len(prs.slides),
    )
