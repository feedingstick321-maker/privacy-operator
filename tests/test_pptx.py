"""PPTX 슬라이드/표/노트/문서속성/내장 이미지 OCR 및 가림 테스트. 모든 값은 합성 데이터다."""

from __future__ import annotations

import io
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

from privacy_operator.alias.codes import AliasBook
from privacy_operator.alias.pseudonymize import pseudonymize
from privacy_operator.detect import detect
from privacy_operator.detect.finding import STATUS_CONFIRMED, TIER_CERTAIN, TIER_REVIEW
from privacy_operator.reader import DocumentReader
from privacy_operator.reader.blocks import OCR
from privacy_operator.writer.original import _zip_leftovers, write_original
from privacy_operator.writer.plan import ReplacementPlan

MALGUN = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "malgun.ttf"


def _make_screenshot_png() -> bytes:
    img = Image.new("RGB", (680, 240), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(MALGUN), 28) if MALGUN.is_file() else ImageFont.load_default()
    draw.text((30, 35), "시스템 화면 캡처", fill="black", font=font)
    draw.text((30, 95), "담당자: 김영희", fill="black", font=font)
    draw.text((30, 155), "전화: 010-9876-5432", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_sample_pptx(path: Path) -> None:
    prs = Presentation()
    prs.core_properties.author = "김영희"
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "2026 강의안 안내"
    slide.placeholders[1].text = "문의: test_lecturer@example.com"

    tbl_shape = slide.shapes.add_table(2, 2, Inches(1), Inches(3.2), Inches(5), Inches(1.2))
    tbl = tbl_shape.table
    tbl.cell(0, 0).text = "강사명"
    tbl.cell(0, 1).text = "비상연락처"
    tbl.cell(1, 0).text = "김영희"
    tbl.cell(1, 1).text = "010-9876-5432"

    slide.notes_slide.notes_text_frame.text = "발표자 메모: 김영희 선생님께 확인"
    slide.shapes.add_picture(io.BytesIO(_make_screenshot_png()), Inches(6.2), Inches(2.5), width=Inches(3.0))
    prs.save(str(path))


def test_pptx_blind_spot_when_ocr_disabled(tmp_path):
    pptx_path = tmp_path / "lecture_no_ocr.pptx"
    _make_sample_pptx(pptx_path)
    with DocumentReader(ocr_enabled=False) as reader:
        res = reader.read(pptx_path)
        assert res.format == "pptx"
        assert any(b.kind == "image" and b.page == 1 for b in res.blind_spots)
        assert any("김영희" in b.text for b in res.blocks)


def test_pptx_ocr_and_redaction(tmp_path):
    pptx_path = tmp_path / "lecture.pptx"
    _make_sample_pptx(pptx_path)

    with DocumentReader(ocr_enabled=True) as reader:
        res = reader.read(pptx_path)
        assert not res.blind_spots
        ocr_blocks = [b for b in res.blocks if b.kind == OCR]
        assert ocr_blocks
        ocr_text = "\n".join(b.text for b in ocr_blocks)
        assert "010-9876-5432" in ocr_text

        findings = detect([res])
        for f in findings:
            if f.tier in (TIER_CERTAIN, TIER_REVIEW):
                f.review_status = STATUS_CONFIRMED

        book = AliasBook("TEST-PPTX")
        docs = pseudonymize([res], findings, book)
        plan = ReplacementPlan(findings, docs)

        out = write_original(
            pptx_path,
            "pptx",
            plan,
            tmp_path / "out",
            "lecture_redacted",
            reader,
            ocr_blocks=ocr_blocks,
            redact_style="black_alias",
        )
        assert out.ok and out.path is not None

        places, _ = _zip_leftovers(out.path, plan)
        assert not places

        re_res = reader.read(out.path)
        re_text = "\n".join(b.text for b in re_res.blocks)
        assert "010-9876-5432" not in re_text
        assert "test_lecturer@example.com" not in re_text
        assert "김영희" not in re_text
