"""PaddleOCR 기반 이미지/스캔 PDF·PPTX 내장 그림 OCR 및 좌표 매핑.

- PaddleOCR 3.x (`predict()`)와 레거시 2.x (`ocr()`)를 모두 지원한다.
- Windows + PaddlePaddle 3.3.x PIR/oneDNN 충돌 방지를 위해 import 전 MKLDNN/모델 소스 체크를 끈다.
- 인식된 줄마다 좌표(PDF points 또는 이미지 pixels)를 `Location.ocr_boxes`로 보존해,
  추후 물리적 픽셀 마스킹/가명코드 덮어쓰기 시 정확한 위치만 덮는다.
"""

from __future__ import annotations

import io
import logging
import os
import threading
from dataclasses import dataclass

from privacy_operator.reader.blocks import OCR, Block, Location

log = logging.getLogger(__name__)

# (start, end, x0, y0, x1, y1)
OcrBoxEntry = tuple[int, int, float, float, float, float]


@dataclass(frozen=True)
class OcrLine:
    text: str
    box: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    score: float = 1.0


def _char_weight(ch: str) -> float:
    if not ch or ch.isspace() or ch.isascii():
        return 0.55
    return 1.0


def sub_bbox_for_span(
    line_text: str,
    box: tuple[float, float, float, float],
    start: int,
    end: int,
    pad_x: float = 2.0,
    pad_y: float = 1.5,
) -> tuple[float, float, float, float]:
    """Compute proportional horizontal sub-box for ``line_text[start:end]`` within ``box``."""
    x0, y0, x1, y1 = box
    width = max(1.0, x1 - x0)
    if not line_text or start >= end:
        return (x0, max(0.0, y0 - pad_y), x1, y1 + pad_y)
    weights = [_char_weight(ch) for ch in line_text]
    total = sum(weights) or float(len(line_text))
    s_clamped = max(0, min(start, len(line_text)))
    e_clamped = max(s_clamped, min(end, len(line_text)))
    prefix = sum(weights[:s_clamped]) / total
    suffix = sum(weights[:e_clamped]) / total
    sub_x0 = max(x0 - pad_x, x0 + width * prefix - pad_x)
    sub_x1 = min(x1 + pad_x, x0 + width * suffix + pad_x)
    if sub_x1 <= sub_x0:
        sub_x1 = sub_x0 + 4.0
    return (sub_x0, max(0.0, y0 - pad_y), sub_x1, y1 + pad_y)


def rects_for_span(
    block: Block,
    start: int,
    end: int,
    pad_x: float = 2.5,
    pad_y: float = 1.5,
) -> list[tuple[float, float, float, float]]:
    """Map character span ``[start, end)`` of an OCR Block to one rect per covered OCR line."""
    boxes = block.location.ocr_boxes
    if not boxes or start >= end:
        return []
    rects: list[tuple[float, float, float, float]] = []
    for ls, le, x0, y0, x1, y1 in boxes:
        ov_s = max(start, ls)
        ov_e = min(end, le)
        if ov_s >= ov_e:
            continue
        line_text = block.text[ls:le]
        rects.append(sub_bbox_for_span(line_text, (x0, y0, x1, y1), ov_s - ls, ov_e - ls, pad_x=pad_x, pad_y=pad_y))
    return rects


def _sort_lines(lines: list[OcrLine]) -> list[OcrLine]:
    if not lines:
        return []
    heights = sorted(max(6.0, ln.box[3] - ln.box[1]) for ln in lines)
    median_h = heights[len(heights) // 2]
    row_thresh = max(6.0, median_h * 0.55)

    ordered: list[list[OcrLine]] = []
    for ln in sorted(lines, key=lambda l: ((l.box[1] + l.box[3]) / 2.0, l.box[0])):
        cy = (ln.box[1] + ln.box[3]) / 2.0
        if ordered:
            prev_cy = sum((p.box[1] + p.box[3]) / 2.0 for p in ordered[-1]) / len(ordered[-1])
            if abs(cy - prev_cy) <= row_thresh:
                ordered[-1].append(ln)
                continue
        ordered.append([ln])

    out: list[OcrLine] = []
    for row in ordered:
        row.sort(key=lambda l: l.box[0])
        out.extend(row)
    return out


def lines_to_blocks(
    lines: list[OcrLine],
    page_no: int | None,
    ocr_source: str,
    prefix: str,
    media_path: str | None = None,
) -> list[Block]:
    """Group OCR lines into readable Blocks while preserving per-line coordinate entries."""
    sorted_lines = [ln for ln in _sort_lines(lines) if ln.text.strip()]
    if not sorted_lines:
        return []

    # Group lines with modest vertical gap into blocks so adjacent label/value pairs stay together.
    groups: list[list[OcrLine]] = [[sorted_lines[0]]]
    for ln in sorted_lines[1:]:
        prev = groups[-1][-1]
        prev_h = max(6.0, prev.box[3] - prev.box[1])
        cur_h = max(6.0, ln.box[3] - ln.box[1])
        same_row = abs(((ln.box[1] + ln.box[3]) / 2.0) - ((prev.box[1] + prev.box[3]) / 2.0)) <= min(prev_h, cur_h) * 0.55
        vert_gap = ln.box[1] - prev.box[3]
        if not same_row and vert_gap > max(prev_h, cur_h) * 1.8:
            groups.append([ln])
        else:
            groups[-1].append(ln)

    blocks: list[Block] = []
    for bi, grp in enumerate(groups):
        parts: list[str] = []
        boxes: list[OcrBoxEntry] = []
        pos = 0
        for idx, ln in enumerate(grp):
            txt = ln.text.strip()
            if idx > 0:
                prev = grp[idx - 1]
                prev_cy = (prev.box[1] + prev.box[3]) / 2.0
                cur_cy = (ln.box[1] + ln.box[3]) / 2.0
                avg_h = max(6.0, ((prev.box[3] - prev.box[1]) + (ln.box[3] - ln.box[1])) / 2.0)
                sep = " " if (abs(cur_cy - prev_cy) <= avg_h * 0.55 and ln.box[0] >= prev.box[0]) else "\n"
                parts.append(sep)
                pos += len(sep)
            start = pos
            parts.append(txt)
            pos += len(txt)
            end = pos
            boxes.append((start, end, float(ln.box[0]), float(ln.box[1]), float(ln.box[2]), float(ln.box[3])))
        full_text = "".join(parts)
        if not full_text.strip():
            continue
        x0 = min(b[2] for b in boxes)
        y0 = min(b[3] for b in boxes)
        x1 = max(b[4] for b in boxes)
        y1 = max(b[5] for b in boxes)
        loc = Location(
            page=page_no,
            block_index=bi,
            bbox=(x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0)),
            ocr_source=ocr_source,
            media_path=media_path,
            ocr_boxes=tuple(boxes),
        )
        blocks.append(Block(f"{prefix}-{bi:03d}", OCR, full_text, loc))
    return blocks


def _poly_to_rect(poly) -> tuple[float, float, float, float] | None:
    try:
        pts = list(poly)
        if len(pts) == 4 and isinstance(pts[0], (int, float)):
            return (float(pts[0]), float(pts[1]), float(pts[2]), float(pts[3]))
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        if not xs or not ys:
            return None
        return (min(xs), min(ys), max(xs), max(ys))
    except Exception:  # noqa: BLE001
        return None


def _extract_v3_lines(item, min_score: float = 0.2) -> list[OcrLine]:
    if not hasattr(item, "get"):
        return []
    texts = item.get("rec_texts") or []
    scores = item.get("rec_scores") or [1.0] * len(texts)
    polys = item.get("rec_polys")
    if polys is None or len(polys) == 0:
        polys = item.get("rec_boxes")
    if polys is None or len(polys) == 0:
        polys = item.get("dt_polys") or []

    lines: list[OcrLine] = []
    for txt, sc, poly in zip(texts, scores, polys):
        if not isinstance(txt, str) or not txt.strip():
            continue
        score = float(sc) if sc is not None else 1.0
        if score < min_score:
            continue
        rect = _poly_to_rect(poly)
        if rect is not None:
            lines.append(OcrLine(txt.strip(), rect, score))
    return lines


def _extract_v2_lines(raw, min_score: float = 0.2) -> list[OcrLine]:
    if not raw:
        return []
    page_items = raw[0] if isinstance(raw[0], list) and raw[0] and isinstance(raw[0][0], (list, tuple)) else raw
    lines: list[OcrLine] = []
    for entry in page_items or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        rect = _poly_to_rect(entry[0])
        val = entry[1]
        if rect is None or not isinstance(val, (list, tuple)) or not val:
            continue
        txt = str(val[0]).strip()
        score = float(val[1]) if len(val) > 1 and val[1] is not None else 1.0
        if txt and score >= min_score:
            lines.append(OcrLine(txt, rect, score))
    return lines


class PaddleOcrEngine:
    """Lazy-initialized PaddleOCR engine with GPU acceleration (and safe CPU fallback)."""

    def __init__(self, lang: str = "korean", min_score: float = 0.25, device: str | None = None):
        self.lang = lang
        self.min_score = min_score
        self.device = device
        self._ocr = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        env_dev = os.environ.get("PRIVACY_OPERATOR_OCR_DEVICE")
        if env_dev:
            return env_dev
        try:
            import paddle

            if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
                return "gpu:0"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        with self._lock:
            if self._ocr is not None:
                return self._ocr
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            dev = self._resolve_device()
            if dev == "cpu":
                os.environ.setdefault("FLAGS_use_mkldnn", "0")
                os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
            from paddleocr import PaddleOCR

            kwargs = dict(
                lang=self.lang,
                device=dev,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            if dev == "cpu":
                kwargs["enable_mkldnn"] = False
            try:
                self._ocr = PaddleOCR(**kwargs)
            except TypeError:
                self._ocr = PaddleOCR(lang=self.lang)
            return self._ocr

    def ocr_array(self, rgb_array) -> list[OcrLine]:
        engine = self._ensure_ocr()
        if hasattr(engine, "predict"):
            res = list(engine.predict(rgb_array))
            return _extract_v3_lines(res[0], min_score=self.min_score) if res else []
        res = engine.ocr(rgb_array, cls=False)
        return _extract_v2_lines(res, min_score=self.min_score)

    def ocr_image_bytes(self, image_bytes: bytes) -> list[OcrLine]:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            rgb = img.convert("RGB")
            w, h = rgb.size
            if w < 16 or h < 16:
                return []
            arr = np.asarray(rgb)
        return self.ocr_array(arr)

    @staticmethod
    def _capped_matrix(width: float, height: float, dpi: int, max_side_px: int):
        import pymupdf

        zoom = dpi / 72.0
        longest = max(width, height) * zoom
        if longest > max_side_px and longest > 0:
            zoom *= max_side_px / longest
        return pymupdf.Matrix(zoom, zoom)

    def ocr_pdf_page(self, page, dpi: int = 150, max_side_px: int = 1800) -> list[OcrLine]:
        """Render a PDF page to RGB and map recognized boxes back into PDF page points."""
        import numpy as np

        mat = self._capped_matrix(page.rect.width, page.rect.height, dpi, max_side_px)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if pix.width < 16 or pix.height < 16:
            return []
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            arr = arr[:, :, :3]
        raw_lines = self.ocr_array(arr)
        sx = page.rect.width / float(pix.width)
        sy = page.rect.height / float(pix.height)
        rx0, ry0 = float(page.rect.x0), float(page.rect.y0)
        return [
            OcrLine(
                text=ln.text,
                box=(rx0 + ln.box[0] * sx, ry0 + ln.box[1] * sy, rx0 + ln.box[2] * sx, ry0 + ln.box[3] * sy),
                score=ln.score,
            )
            for ln in raw_lines
        ]

    def ocr_pdf_embedded_images(self, page, dpi: int = 150, min_px: int = 40, max_side_px: int = 1600) -> list[OcrLine]:
        """OCR non-tiny embedded images placed on a PDF page, returning PDF page point coords."""
        import numpy as np
        import pymupdf

        lines: list[OcrLine] = []
        seen_rects: list[tuple[float, float, float, float]] = []
        for info in page.get_image_info(xrefs=True) or ():
            bbox = info.get("bbox")
            if not bbox:
                continue
            rect = pymupdf.Rect(bbox) & page.rect
            if rect.is_empty or rect.width < 20 or rect.height < 20:
                continue
            key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
            if key in seen_rects:
                continue
            seen_rects.append(key)
            mat = self._capped_matrix(rect.width, rect.height, dpi, max_side_px)
            pix = page.get_pixmap(clip=rect, matrix=mat, alpha=False)
            if pix.width < min_px or pix.height < min_px:
                continue
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                arr = arr[:, :, :3]
            raw = self.ocr_array(arr)
            sx = rect.width / float(pix.width)
            sy = rect.height / float(pix.height)
            for ln in raw:
                lines.append(
                    OcrLine(
                        text=ln.text,
                        box=(
                            rect.x0 + ln.box[0] * sx,
                            rect.y0 + ln.box[1] * sy,
                            rect.x0 + ln.box[2] * sx,
                            rect.y0 + ln.box[3] * sy,
                        ),
                        score=ln.score,
                    )
                )
        return lines
