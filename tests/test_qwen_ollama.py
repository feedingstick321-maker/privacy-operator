"""로컬 Qwen3.5 4B(Ollama) 워커 및 스팬 매핑/루프백 보안 테스트."""

from __future__ import annotations

import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

from privacy_operator.detect import detect
from privacy_operator.detect.ai.qwen_ollama import QwenOllamaEngine, QwenOllamaError, find_pii_qwen
from privacy_operator.detect.ai.spans import is_ai
from privacy_operator.detect.dictionaries import ADDRESS, NAME, OTHER
from privacy_operator.detect.finding import BY_AI_QWEN, STATUS_PENDING, TIER_REVIEW
from privacy_operator.reader.blocks import OCR, Location, Block, ReadResult

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "engine" / "qwen_pii" / "worker.py"

FAKE_QWEN_WORKER = textwrap.dedent("""
    import json, sys
    out = sys.stdout.buffer
    def send(o):
        out.write((json.dumps(o) + "\\n").encode("ascii")); out.flush()
    send({"ready": True, "model": "qwen3.5:4b", "quantization": "Q4_K_M", "parameter_size": "4.7B"})
    for raw in sys.stdin.buffer:
        req = json.loads(raw.decode("utf-8"))
        if req.get("action") == "ping":
            send({"ok": True}); continue
        results = []
        for t in req.get("texts", []):
            ents = []
            if "이서연" in t:
                ents.append({"text": "이서연", "category": "name", "reason": "수강생 이름"})
            if "세종로 209" in t:
                ents.append({"text": "세종로 209", "category": "address", "reason": "상세 주소"})
            if "2024123456" in t:
                ents.append({"text": "2024123456", "category": "id_number", "reason": "학번"})
            results.append(ents)
        send({"results": results})
""")


def test_qwen_span_alignment_and_finding_creation(tmp_path):
    worker = tmp_path / "fake_qwen_worker.py"
    worker.write_text(FAKE_QWEN_WORKER, encoding="utf-8")

    res = ReadResult(
        source_name="slide.pptx",
        format="pptx",
        blocks=[
            Block(
                "ocr-001",
                OCR,
                "참여자 이서연 (학번 2024123456), 거주지 세종로 209",
                Location(page=1, ocr_source="pptx_media", media_path="ppt/media/image1.png",
                         ocr_boxes=((0, 40, 10.0, 20.0, 400.0, 50.0),)),
            )
        ],
    )
    existing = detect([res])
    with QwenOllamaEngine(model="qwen3.5:4b", python=sys.executable, worker=worker) as eng:
        assert eng.quantization == "Q4_K_M"
        new, report = find_pii_qwen([res], existing, eng)

    got = sorted((f.category, f.raw_value) for f in new)
    assert (NAME, "이서연") in got
    assert (ADDRESS, "세종로 209") in got
    assert (OTHER, "2024123456") in got
    assert report.quantization == "Q4_K_M"
    for f in new:
        assert f.tier == TIER_REVIEW
        assert f.review_status == STATUS_PENDING
        assert f.detected_by == (BY_AI_QWEN,)
        assert is_ai(f)


def test_worker_loopback_guard_blocks_external_hosts():
    spec = importlib.util.spec_from_file_location("qwen_worker_mod", WORKER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Simulate checking _NET_EVENTS directly without permanently mutating pytest process audit hooks
    assert "127.0.0.1" in mod.LOOPBACK_HOSTS
    assert "::1" in mod.LOOPBACK_HOSTS
    assert mod._host_of((("93.184.216.34", 443),)) == "93.184.216.34"

    # Run a subprocess that installs the guard and confirms external socket connect is blocked
    import subprocess
    code = (
        f"import importlib.util, socket; "
        f"s = importlib.util.spec_from_file_location('w', r'{WORKER_PATH}'); "
        f"m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
        f"m.install_loopback_guard(); "
        f"socket.getaddrinfo('example.com', 80)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode != 0
    assert "외부 통신 차단" in proc.stderr


def test_qwen_unreachable_raises_clean_code():
    # Unused loopback port where nothing listens
    with pytest.raises(QwenOllamaError) as exc:
        QwenOllamaEngine(model="qwen3.5:4b", port=59999).start()
    assert exc.value.code == "OLLAMA_UNREACHABLE"
