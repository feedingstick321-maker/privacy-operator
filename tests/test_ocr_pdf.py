"""통짜 이미지 PDF OCR(PaddleOCR) + 물리 픽셀 마스킹 + 재-OCR 검증 테스트. 모든 값은 합성 데이터다."""

from __future__ import annotations

import os
from pathlib import Path

import pymupdf

from privacy_operator.alias.codes import AliasBook
from privacy_operator.alias.pseudonymize import pseudonymize
from privacy_operator.detect import detect
from privacy_operator.detect.finding import STATUS_CONFIRMED, TIER_CERTAIN, TIER_REVIEW
from privacy_operator.reader import DocumentReader
from privacy_operator.reader.blocks import OCR
from privacy_operator.writer.original import write_original
from privacy_operator.writer.plan import ReplacementPlan

MALGUN = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "malgun.ttf"


def _make_pure_image_pdf(path: Path) -> None:
    """글자층이 전혀 없는 순수 래스터 이미지 PDF 생성."""
    src = pymupdf.open()
    p = src.new_page(width=720, height=540)
    font_kw = {"fontname": "malgun", "fontfile": str(MALGUN)} if MALGUN.is_file() else {"fontname": "korea"}
    p.insert_text((60, 110), "강의안 샘플 화면", fontsize=24, **font_kw)
    p.insert_text((60, 180), "강사: 홍길동", fontsize=22, **font_kw)
    p.insert_text((60, 240), "연락처: 010-2345-6789", fontsize=22, **font_kw)
    p.insert_text((60, 300), "이메일: teacher99@example.com", fontsize=22, **font_kw)
    png = p.get_pixmap(dpi=150).tobytes("png")
    src.close()

    doc = pymupdf.open()
    page = doc.new_page(width=720, height=540)
    page.insert_image(pymupdf.Rect(0, 0, 720, 540), stream=png)
    doc.save(path)
    doc.close()


def test_pure_image_pdf_ocr_and_pixel_redaction(tmp_path):
    pdf_path = tmp_path / "pure_image_slide.pdf"
    _make_pure_image_pdf(pdf_path)

    with pymupdf.open(pdf_path) as d:
        assert d[0].get_text().strip() == ""  # 글자층 없음 확인

    with DocumentReader(ocr_enabled=True) as reader:
        res = reader.read(pdf_path)
        assert not res.blind_spots
        ocr_blocks = [b for b in res.blocks if b.kind == OCR]
        assert ocr_blocks
        combined = "\n".join(b.text for b in ocr_blocks)
        assert "010-2345-6789" in combined
        assert "teacher99@example.com" in combined

        findings = detect([res])
        for f in findings:
            if f.tier in (TIER_CERTAIN, TIER_REVIEW):
                f.review_status = STATUS_CONFIRMED
        book = AliasBook("TEST-OCR")
        docs = pseudonymize([res], findings, book)
        plan = ReplacementPlan(findings, docs)

        out = write_original(
            pdf_path,
            "pdf",
            plan,
            tmp_path / "out",
            "pure_image_redacted",
            reader,
            ocr_blocks=ocr_blocks,
            redact_style="black_alias",
        )
        assert out.ok and out.path is not None

        # 물리적 픽셀 마스킹 뒤 다시 OCR로 읽어도 원본 개인정보가 남지 않아야 한다
        re_res = reader.read(out.path)
        re_text = "\n".join(b.text for b in re_res.blocks)
        assert "010-2345-6789" not in re_text
        assert "teacher99@example.com" not in re_text
        assert "홍길동" not in re_text


def test_preprocess_deskew_round_trip():
    import cv2
    import numpy as np

    from privacy_operator.reader.ocr import _invert_deskew_box, preprocess_scan_rgb

    h, w = 800, 1000
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in range(140, 680, 70):
        cv2.line(canvas, (120, y), (880, y), (20, 20, 20), 6)
    cx, cy = w / 2.0, h / 2.0
    tilted = cv2.warpAffine(canvas, cv2.getRotationMatrix2D((cx, cy), -3.2, 1.0), (w, h), borderValue=(255, 255, 255))

    proc, deskew_meta = preprocess_scan_rgb(tilted, deskew=True)
    assert proc.shape == tilted.shape
    assert deskew_meta is not None
    angle, rx, ry = deskew_meta
    assert 2.0 <= angle <= 4.5

    # Forward rotation of point (250, 210) on tilted image lands near proc coords; inverse must restore it
    fwd = cv2.getRotationMatrix2D((rx, ry), angle, 1.0)
    px, py = (fwd @ np.array([250.0, 210.0, 1.0])).tolist()
    inv_box = _invert_deskew_box((px - 5.0, py - 5.0, px + 5.0, py + 5.0), angle, rx, ry, w, h)
    assert abs((inv_box[0] + inv_box[2]) / 2.0 - 250.0) < 3.0
    assert abs((inv_box[1] + inv_box[3]) / 2.0 - 210.0) < 3.0


def test_upside_down_fax_image_pdf_detected_by_180_pass(tmp_path):
    from PIL import Image

    src = pymupdf.open()
    p = src.new_page(width=720, height=540)
    font_kw = {"fontname": "malgun", "fontfile": str(MALGUN)} if MALGUN.is_file() else {"fontname": "korea"}
    p.insert_text((80, 120), "테스트 문서 제목", fontsize=24, **font_kw)
    normal_png = p.get_pixmap(dpi=150).tobytes("png")
    src.close()

    # Render upright fax line separately and rotate 180 degrees onto the bottom of the page
    fax_doc = pymupdf.open()
    fp = fax_doc.new_page(width=420, height=80)
    fp.insert_text((20, 52), "FAX 031 4654298", fontsize=24, **font_kw)
    fax_pix = fp.get_pixmap(dpi=150)
    fax_doc.close()

    import io

    with Image.open(io.BytesIO(normal_png)) as base_img, Image.open(io.BytesIO(fax_pix.tobytes("png"))) as fax_img:
        rot_fax = fax_img.rotate(180)
        base_img.paste(rot_fax, (420, 820))
        buf = io.BytesIO()
        base_img.save(buf, format="PNG")
        merged_png = buf.getvalue()

    pdf_path = tmp_path / "upside_down_fax.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=720, height=540)
    page.insert_image(pymupdf.Rect(0, 0, 720, 540), stream=merged_png)
    doc.save(pdf_path)
    doc.close()

    with DocumentReader(ocr_enabled=True) as reader:
        res = reader.read(pdf_path)
        ocr_blocks = [b for b in res.blocks if b.kind == OCR]
        combined = "\n".join(b.text for b in ocr_blocks)
        assert "4654298" in combined.replace(" ", "")

        findings = detect([res])
        assert any(f.category == "전화" and "4654298" in f.raw_value.replace(" ", "") for f in findings)
        for f in findings:
            f.review_status = STATUS_CONFIRMED
        book = AliasBook("TEST-180")
        docs = pseudonymize([res], findings, book)
        plan = ReplacementPlan(findings, docs)
        out = write_original(
            pdf_path, "pdf", plan, tmp_path / "out", "upside_redacted", reader, ocr_blocks=ocr_blocks
        )
        assert out.ok and out.path is not None
        re_res = reader.read(out.path)
        re_text = "\n".join(b.text for b in re_res.blocks)
        assert "4654298" not in re_text.replace(" ", "")
