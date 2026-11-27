"""PaddleOCR 기반 이미지/스캔 PDF·PPTX 내장 그림 OCR 및 좌표 매핑.

- PaddleOCR 3.x (`predict()`)와 레거시 2.x (`ocr()`)를 모두 지원한다.
- Windows + PaddlePaddle 3.3.x PIR/oneDNN 충돌 방지를 위해 import 전 MKLDNN/모델 소스 체크를 끈다.
- 인식된 줄마다 좌표(PDF points 또는 이미지 pixels)를 `Location.ocr_boxes`로 보존해,
  추후 물리적 픽셀 마스킹/가명코드 덮어쓰기 시 정확한 위치만 덮는다.
"""

from __future__ import annotations

import io
import logging
import math
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


def _estimate_skew_angle(gray) -> float | None:
    """Estimate small document skew angle in [-8, +8] degrees for Hough deskewing."""
    import cv2
    import numpy as np

    h, w = gray.shape[:2]
    max_side = max(h, w)
    if max_side < 64:
        return None
    if max_side > 1200:
        scale = 1200.0 / float(max_side)
        small = cv2.resize(gray, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    else:
        small = gray
    sh, sw = small.shape[:2]
    _, bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    min_len = max(30, int(round(0.04 * sw)))
    raw_lines = cv2.HoughLinesP(closed, 1.0, math.pi / 180.0, threshold=80, minLineLength=min_len, maxLineGap=10)
    if raw_lines is None:
        return None
    angles: list[float] = []
    for x1, y1, x2, y2 in np.asarray(raw_lines).reshape(-1, 4).tolist():
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        ang = math.degrees(math.atan2(float(dy), float(dx)))
        if abs(ang) <= 8.0:
            angles.append(ang)
    if len(angles) < 5:
        return None
    median_ang = float(np.median(np.array(angles, dtype=np.float64)))
    if 0.35 <= abs(median_ang) <= 8.0:
        return median_ang
    return None


def preprocess_scan_rgb(rgb_array, deskew: bool = True):
    """Apply small-angle deskew, LAB CLAHE luminance boost, and mild unsharp mask.

    Returns ``(processed_rgb, deskew_meta)`` where ``deskew_meta`` is either ``None``
    or ``(angle_deg, cx, cy)`` for mapping OCR coordinates back to the source image.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return rgb_array, None

    if rgb_array is None or getattr(rgb_array, "ndim", 0) != 3 or rgb_array.shape[0] < 16 or rgb_array.shape[1] < 16:
        return rgb_array, None

    h, w = rgb_array.shape[:2]
    work = np.ascontiguousarray(rgb_array, dtype=np.uint8)
    deskew_meta: tuple[float, float, float] | None = None

    if deskew:
        gray = cv2.cvtColor(work, cv2.COLOR_RGB2GRAY)
        angle = _estimate_skew_angle(gray)
        if angle is not None:
            cx, cy = w / 2.0, h / 2.0
            mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
            work = cv2.warpAffine(work, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            deskew_meta = (angle, cx, cy)

    lab = cv2.cvtColor(work, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge((l_eq, a_ch, b_ch))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    sharpened = cv2.addWeighted(enhanced, 1.25, blur, -0.25, 0)
    return np.ascontiguousarray(sharpened, dtype=np.uint8), deskew_meta


def _invert_deskew_box(
    box: tuple[float, float, float, float],
    angle_deg: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> tuple[float, float, float, float]:
    """Map an axis-aligned box in deskewed image coordinates back onto the original image."""
    import cv2
    import numpy as np

    x0, y0, x1, y1 = box
    mat = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    corners = np.array(
        [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]],
        dtype=np.float64,
    )
    mapped = corners @ mat.T
    nx0 = max(0.0, min(float(w), float(np.min(mapped[:, 0]))))
    ny0 = max(0.0, min(float(h), float(np.min(mapped[:, 1]))))
    nx1 = max(nx0 + 1.0, min(float(w), float(np.max(mapped[:, 0]))))
    ny1 = max(ny0 + 1.0, min(float(h), float(np.max(mapped[:, 1]))))
    return (nx0, ny0, nx1, ny1)


def _rotate_180_box(box: tuple[float, float, float, float], w: int, h: int) -> tuple[float, float, float, float]:
    """Map a box detected on a 180-degree rotated image back into original image coordinates."""
    x0, y0, x1, y1 = box
    return (
        max(0.0, float(w) - x1),
        max(0.0, float(h) - y1),
        min(float(w), float(w) - x0),
        min(float(h), float(h) - y0),
    )


def _box_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float]:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0, 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(1.0, (bx1 - bx0) * (by1 - by0))
    union = area_a + area_b - inter
    return inter / max(1.0, union), inter / min(area_a, area_b)


def _merge_180_lines(
    lines_0: list[OcrLine],
    lines_180: list[OcrLine],
    min_new_score: float = 0.65,
    replace_margin: float = 0.12,
) -> list[OcrLine]:
    """Combine 0-degree OCR lines with 180-degree pass lines mapped back to 0-degree coords."""
    merged = list(lines_0)
    for ln180 in lines_180:
        txt = ln180.text.strip()
        if not txt:
            continue
        best_idx = -1
        best_metric = 0.0
        for idx, cur in enumerate(merged):
            iou, min_cov = _box_overlap_ratio(cur.box, ln180.box)
            if iou >= 0.25 or min_cov >= 0.45:
                metric = max(iou, min_cov)
                if metric > best_metric:
                    best_metric = metric
                    best_idx = idx
        if best_idx >= 0:
            cur = merged[best_idx]
            if ln180.score >= 0.72 and (ln180.score > cur.score + replace_margin or cur.score < 0.55):
                merged[best_idx] = ln180
        else:
            req_score = 0.80 if len(txt) == 2 else min_new_score
            if len(txt) >= 2 and ln180.score >= req_score:
                merged.append(ln180)
    return merged


class _BaseOcrEngine:
    """Shared scan preprocessing, deskew inverse mapping, 180-degree pass, and PDF raster OCR."""

    default_dpi: int = 150
    default_max_side_px: int = 1800
    default_img_max_side_px: int = 1600

    def __init__(self, lang: str = "korean", min_score: float = 0.25, preprocess: bool = True, rotate_180: bool = False):
        self.lang = lang
        self.min_score = min_score
        self.preprocess = preprocess
        self.rotate_180 = rotate_180
        self._ocr = None
        self._lock = threading.Lock()

    def _ensure_ocr(self):
        raise NotImplementedError

    def _run_raw_ocr(self, engine, rgb_array) -> list[OcrLine]:
        raise NotImplementedError

    def ocr_array(self, rgb_array) -> list[OcrLine]:
        import numpy as np

        engine = self._ensure_ocr()
        h, w = rgb_array.shape[:2]
        deskew_meta = None
        proc = rgb_array
        if self.preprocess:
            proc, deskew_meta = preprocess_scan_rgb(rgb_array, deskew=True)

        lines_0 = self._run_raw_ocr(engine, proc)
        if self.rotate_180 and proc.shape[0] >= 40 and proc.shape[1] >= 40:
            rot180 = np.ascontiguousarray(proc[::-1, ::-1])
            ph, pw = proc.shape[:2]
            raw_180 = self._run_raw_ocr(engine, rot180)
            mapped_180 = [
                OcrLine(text=ln.text, box=_rotate_180_box(ln.box, pw, ph), score=ln.score)
                for ln in raw_180
            ]
            lines = _merge_180_lines(lines_0, mapped_180)
        else:
            lines = lines_0

        if deskew_meta is not None:
            angle, cx, cy = deskew_meta
            lines = [
                OcrLine(text=ln.text, box=_invert_deskew_box(ln.box, angle, cx, cy, w, h), score=ln.score)
                for ln in lines
            ]
        return lines

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

    def ocr_pdf_page(self, page, dpi: int | None = None, max_side_px: int | None = None) -> list[OcrLine]:
        """Render a PDF page to RGB and map recognized boxes back into PDF page points."""
        import numpy as np

        use_dpi = self.default_dpi if dpi is None else dpi
        use_max = self.default_max_side_px if max_side_px is None else max_side_px
        mat = self._capped_matrix(page.rect.width, page.rect.height, use_dpi, use_max)
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

    def ocr_pdf_embedded_images(
        self,
        page,
        dpi: int | None = None,
        min_px: int = 40,
        max_side_px: int | None = None,
    ) -> list[OcrLine]:
        """OCR non-tiny embedded images placed on a PDF page, returning PDF page point coords."""
        import numpy as np
        import pymupdf

        use_dpi = self.default_dpi if dpi is None else dpi
        use_max = self.default_img_max_side_px if max_side_px is None else max_side_px
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
            mat = self._capped_matrix(rect.width, rect.height, use_dpi, use_max)
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


class RapidOcrEngine(_BaseOcrEngine):
    """Lightweight ONNX Runtime CPU OCR engine using PP-OCRv5 mobile Korean models."""

    def __init__(
        self,
        lang: str = "korean",
        min_score: float = 0.25,
        preprocess: bool = True,
        rotate_180: bool = False,
        use_cls: bool = True,
        default_dpi: int = 150,
        default_max_side_px: int = 1800,
    ):
        super().__init__(lang=lang, min_score=min_score, preprocess=preprocess, rotate_180=rotate_180)
        self.use_cls = use_cls
        self.default_dpi = default_dpi
        self.default_max_side_px = default_max_side_px
        self.default_img_max_side_px = min(1600, default_max_side_px)

    def _ensure_ocr(self):
        if self._ocr is not None:
            return self._ocr
        with self._lock:
            if self._ocr is not None:
                return self._ocr
            from rapidocr import EngineType, LangRec, ModelType, OCRVersion, RapidOCR

            rec_lang = LangRec.KOREAN if self.lang == "korean" else LangRec.CH
            params = {
                "Global.use_cls": self.use_cls,
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.ocr_version": OCRVersion.PPOCRV5,
                "Det.model_type": ModelType.MOBILE,
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.lang_type": rec_lang,
                "Rec.model_type": ModelType.MOBILE,
            }
            self._ocr = RapidOCR(params=params)
            return self._ocr

    def _run_raw_ocr(self, engine, rgb_array) -> list[OcrLine]:
        import numpy as np

        bgr = np.ascontiguousarray(rgb_array[:, :, ::-1])
        out = engine(bgr, use_cls=self.use_cls)
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)
        if boxes is None or txts is None:
            return []
        if scores is None:
            scores = [1.0] * len(txts)
        lines: list[OcrLine] = []
        for poly, txt, sc in zip(boxes, txts, scores):
            if not isinstance(txt, str) or not txt.strip():
                continue
            score = float(sc) if sc is not None else 1.0
            if score < self.min_score:
                continue
            rect = _poly_to_rect(poly)
            if rect is not None:
                lines.append(OcrLine(txt.strip(), rect, score))
        return lines


class PaddleOcrEngine(_BaseOcrEngine):
    """Lazy-initialized PaddleOCR engine with GPU acceleration, scan preprocessing, and 180-degree pass."""

    default_dpi: int = 200
    default_max_side_px: int = 2400
    default_img_max_side_px: int = 2000

    def __init__(
        self,
        lang: str = "korean",
        min_score: float = 0.25,
        device: str | None = None,
        preprocess: bool = True,
        rotate_180: bool = True,
        use_textline_orientation: bool = True,
    ):
        super().__init__(lang=lang, min_score=min_score, preprocess=preprocess, rotate_180=rotate_180)
        self.device = device
        self.use_textline_orientation = use_textline_orientation

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
                use_textline_orientation=self.use_textline_orientation,
            )
            if dev == "cpu":
                kwargs["enable_mkldnn"] = False
            try:
                self._ocr = PaddleOCR(**kwargs)
            except TypeError:
                self._ocr = PaddleOCR(lang=self.lang)
            except Exception as exc:  # noqa: BLE001
                if self.use_textline_orientation:
                    log.warning("PaddleOCR textline orientation init failed (%s); falling back", type(exc).__name__)
                    kwargs["use_textline_orientation"] = False
                    self._ocr = PaddleOCR(**kwargs)
                else:
                    raise
            return self._ocr

    def _run_raw_ocr(self, engine, rgb_array) -> list[OcrLine]:
        if hasattr(engine, "predict"):
            res = list(engine.predict(rgb_array))
            return _extract_v3_lines(res[0], min_score=self.min_score) if res else []
        res = engine.ocr(rgb_array, cls=False)
        return _extract_v2_lines(res, min_score=self.min_score)
