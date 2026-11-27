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
