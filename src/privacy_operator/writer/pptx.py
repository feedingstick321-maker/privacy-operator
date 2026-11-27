"""PPTX 원본 형식 저장: 슬라이드 텍스트/표/노트/문서속성 가명처리 및 내장 이미지 픽셀 마스킹."""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from privacy_operator.reader.blocks import OCR, Block
from privacy_operator.reader.ocr import rects_for_span
from privacy_operator.writer.plan import ReplacementPlan

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_NOTO_FONT = _PROJECT_ROOT / "assets" / "fonts" / "NotoSansKR-VF.ttf"
_MALGUN_FONT = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "malgun.ttf"


def _picture_part(shape):
    blip = getattr(getattr(shape._element, "blipFill", None), "blip", None)
    rid = getattr(blip, "rEmbed", None) if blip is not None else None
    if not rid:
        return None
    return shape.part.related_part(rid)


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


def _replace_text_frame(tf, plan: ReplacementPlan) -> None:
    if tf is None:
        return
    for para in tf.paragraphs:
        runs = list(para.runs)
        if not runs:
            if para.text:
                new_t, n = plan.apply(para.text)
                if n:
                    para.text = new_t
            continue
        full = "".join(r.text for r in runs)
        if not full.strip():
            continue
        new_full, full_hits = plan.apply(full)
        if not full_hits:
            continue
        run_edits = [plan.apply(r.text) for r in runs]
        if sum(n for _, n in run_edits) == full_hits:
            for r, (new_r, n) in zip(runs, run_edits):
                if n:
                    r.text = new_r
        else:
            runs[0].text = new_full
            for r in runs[1:]:
                r.text = ""


def _load_pil_font(size_px: int):
    px = max(10, size_px)
    for candidate in (_MALGUN_FONT, _NOTO_FONT):
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), px)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def _redact_image_blob(
    blob: bytes,
    boxes: list[tuple[tuple[float, float, float, float], str | None]],
    redact_style: str,
) -> bytes:
    if not boxes:
        return blob
    with Image.open(io.BytesIO(blob)) as src:
        orig_fmt = src.format or "PNG"
        has_alpha = src.mode in ("RGBA", "LA") or (src.mode == "P" and "transparency" in src.info)
        canvas = src.convert("RGBA" if has_alpha else "RGB")
        draw = ImageDraw.Draw(canvas)
        w, h = canvas.size
        fill = (255, 255, 255, 255) if has_alpha and redact_style == "white_alias" else (
            (255, 255, 255) if redact_style == "white_alias" else ((0, 0, 0, 255) if has_alpha else (0, 0, 0))
        )
        txt_fill = (0, 0, 0, 255) if has_alpha and redact_style == "white_alias" else (
            (0, 0, 0) if redact_style == "white_alias" else ((255, 255, 255, 255) if has_alpha else (255, 255, 255))
        )
        for (x0, y0, x1, y1), replacement in boxes:
            ix0 = max(0, min(w - 1, int(x0)))
            iy0 = max(0, min(h - 1, int(y0)))
            ix1 = max(ix0 + 1, min(w, int(x1)))
            iy1 = max(iy0 + 1, min(h, int(y1)))
            draw.rectangle([ix0, iy0, ix1, iy1], fill=fill)
            if replacement and redact_style != "black_only":
                bh = max(10, iy1 - iy0)
                bw = max(12, ix1 - ix0)
                fsize = max(10, int(bh * 0.68))
                font = _load_pil_font(fsize)
                while fsize > 10:
                    bbox = draw.textbbox((0, 0), replacement, font=font)
                    if (bbox[2] - bbox[0]) <= bw - 4:
                        break
                    fsize = max(10, fsize - 2)
                    font = _load_pil_font(fsize)
                draw.text((ix0 + 2, iy0 + max(0, (bh - fsize) // 4)), replacement, fill=txt_fill, font=font)

        buf = io.BytesIO()
        if orig_fmt.upper() in ("JPG", "JPEG"):
            canvas.convert("RGB").save(buf, format="JPEG", quality=92)
        else:
            canvas.save(buf, format=orig_fmt)
        return buf.getvalue()


def write_pptx(
    src: Path,
    dst: Path,
    plan: ReplacementPlan,
    ocr_blocks: list[Block] | None = None,
    redact_style: str = "black_alias",
) -> None:
    """Write a redacted/pseudonymized copy of ``src`` PPTX to ``dst``."""
    prs = Presentation(str(src))

    # Map media_path -> [(rect, replacement_or_none), ...]
    media_boxes: dict[str, list[tuple[tuple[float, float, float, float], str | None]]] = {}
    for blk in ocr_blocks or ():
        if blk.kind != OCR or not blk.location.media_path or not blk.location.ocr_boxes:
            continue
        mpath = blk.location.media_path.lstrip("/")
        for s, e, i in plan.pick(blk.text, wrapped=True):
            replacement = plan.rules[i].replacement
            rects = rects_for_span(blk, s, e, pad_x=4.0, pad_y=3.0)
            for idx, rect in enumerate(rects):
                media_boxes.setdefault(mpath, []).append((rect, replacement if idx == 0 else None))

    patched_parts: set[str] = set()
    for slide in prs.slides:
        for shape in _iter_shapes(slide.shapes):
            if shape.has_text_frame:
                _replace_text_frame(shape.text_frame, plan)
            elif shape.has_table:
                for row in shape.table.rows:
                    for cell in row.cells:
                        _replace_text_frame(cell.text_frame, plan)
            elif (_shape_type(shape) == MSO_SHAPE_TYPE.PICTURE or _picture_part(shape) is not None) and media_boxes:
                try:
                    part = _picture_part(shape)
                    if part is None:
                        continue
                    pname = str(part.partname).lstrip("/")
                except Exception:  # noqa: BLE001
                    continue
                if pname in media_boxes and pname not in patched_parts:
                    part._blob = _redact_image_blob(part._blob, media_boxes[pname], redact_style)
                    patched_parts.add(pname)
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            _replace_text_frame(slide.notes_slide.notes_text_frame, plan)

    cp = prs.core_properties
    for attr in ("author", "last_modified_by", "title", "subject", "comments", "keywords"):
        val = getattr(cp, attr, None)
        if isinstance(val, str) and val:
            updated = plan.replace(val)
            if updated != val:
                setattr(cp, attr, updated)

    prs.save(str(dst))
