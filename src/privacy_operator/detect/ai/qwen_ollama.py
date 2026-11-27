"""로컬 Ollama (Qwen3.5 4B Q4_K_M) 개인정보 정찰 클라이언트.

- `src/` 안에서는 통신 모듈(`http.client`, `socket`, `requests` 등)을 절대 import하지 않는다(`tests/test_no_network.py`).
- 실제 Ollama 로컬 HTTP 호출은 루프백 감사 훅을 건 별도 작업 프로세스(`engine/qwen_pii/worker.py`)가 수행한다.
- 탐지 결과는 항상 '검토' 등급, '대기' 상태로 올라간다.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from privacy_operator.detect.ai.spans import AiSpan, to_findings
from privacy_operator.detect.dictionaries import ACCOUNT, ADDRESS, EMAIL, NAME, OTHER, PHONE, looks_like_name
from privacy_operator.detect.finding import BY_AI_QWEN, TIER_REVIEW, Finding
from privacy_operator.detect.patterns import fold
from privacy_operator.detect.rules import is_role_word, trim_name_span
from privacy_operator.reader.blocks import METADATA, OCR, ReadResult

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKER = _ROOT / "engine" / "qwen_pii" / "worker.py"
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

CATEGORY_MAP = {
    "name": NAME,
    "phone": PHONE,
    "email": EMAIL,
    "address": ADDRESS,
    "id_number": OTHER,
    "account": ACCOUNT,
    "other_pii": OTHER,
}

_SKIP_META_FIELDS = frozenset({"creationdate", "moddate", "producer", "creator", "format", "encryption", "trapped"})

_QWEN_LABEL_STOPWORDS = frozenset({
    "성명", "한글", "한자", "생년월일", "주소", "전화번호", "연락처", "팩스", "담당자",
    "교부번호", "발급번호", "문서번호", "등록번호", "입사일자", "퇴사일자", "근무기간",
    "근무처", "소속", "직위", "직종", "이하여백", "해당없음", "공란", "기초학력",
    "기간제교사", "원본대조필", "직인생략", "경력증명서", "재직증명서", "확인자",
    "작성자", "발급자", "대표자",
})
_ORG_CORP_SUFFIXES = (
    "초등학교", "중학교", "고등학교", "초등학", "중학", "고등학", "대학교",
    "아카데미", "교육청", "교육지원청", "지원청", "주식회사",
)
_ORG_SHORT_RE = re.compile(r"^[가-힣]{2,}(?:초|중|고)$")
_JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")
_CERT_ID_RE = re.compile(r"^(?:[A-Z]\d{8,}-\d{4}-\d+|(?:19|20)\d{2}-\d{3,6}|\d{1,4})$")
_STANDALONE_DATE_RE = re.compile(r"^(?:19|20)?\d{2}\s*[./년-]\s*\d{1,2}\s*[./월-]\s*\d{1,2}\s*[.일]?$")
_BIRTH_BEFORE_RE = re.compile(r"(?:생년월일|출생일|출생|생일)\s*[:：]?\s*$")
_BIRTH_AFTER_RE = re.compile(r"^\s*(?:생|출생|년생)(?![가-힣])")
_DIGIT_RE = re.compile(r"\d")
_HANGUL_OR_ALPHA_RE = re.compile(r"[가-힣A-Za-z]")


def _is_qwen_false_positive(category: str, value: str, text: str, start: int, end: int) -> bool:
    """Filter common form labels, agency/school names, cert numbers, and non-birth dates from Qwen hits."""
    compact = re.sub(r"\s+", "", value or "")
    if len(compact) < 2 or compact in _QWEN_LABEL_STOPWORDS:
        return True
    if _JAMO_RE.search(compact):
        return True
    if category in (NAME, OTHER):
        if compact.startswith(("(주)", "주식회사")) or compact.endswith(_ORG_CORP_SUFFIXES) or is_role_word(compact):
            return True
    if category == OTHER and _ORG_SHORT_RE.match(compact):
        return True
    if category == NAME:
        if _DIGIT_RE.search(compact) or not _HANGUL_OR_ALPHA_RE.search(compact):
            return True
        if len(compact) == 2 and not (
            (re.fullmatch(r"[가-힣]{2}", compact) and looks_like_name(compact))
            or re.fullmatch(r"[A-Za-z]{2}", compact)
        ):
            return True
    if category == OTHER:
        if _CERT_ID_RE.match(compact):
            return True
        if _STANDALONE_DATE_RE.match(compact):
            before_ctx = text[max(0, start - 20): start]
            after_ctx = text[end: min(len(text), end + 8)]
            if not (_BIRTH_BEFORE_RE.search(before_ctx) or _BIRTH_AFTER_RE.match(after_ctx)):
                return True
    return False

BATCH = 8
MAX_CHUNK_CHARS = 1500
CHUNK_OVERLAP = 150


class QwenOllamaError(RuntimeError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class QwenReport:
    blocks: int = 0
    entities: int = 0
    added: int = 0
    model: str = ""
    quantization: str = ""


class QwenOllamaEngine:
    """stdin/stdout JSON-lines wrapper around ``engine/qwen_pii/worker.py``."""

    def __init__(
        self,
        model: str = "qwen3.5:4b",
        port: int = 11434,
        host: str = "127.0.0.1",
        python: str | Path | None = None,
        worker: Path | None = None,
    ):
        self.model = model
        self.port = port
        self.host = host
        self.python = str(python or sys.executable)
        self.worker = Path(worker or DEFAULT_WORKER)
        self.proc: subprocess.Popen | None = None
        self.quantization: str = ""
        self.parameter_size: str = ""

    def __enter__(self) -> QwenOllamaEngine:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> QwenOllamaEngine:
        if not self.worker.is_file():
            raise QwenOllamaError("QWEN_WORKER_MISSING")
        cmd = [
            self.python,
            str(self.worker),
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONUTF8", "1")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            creationflags=NO_WINDOW,
        )
        raw = self.proc.stdout.readline()
        if not raw:
            self.close()
            raise QwenOllamaError("QWEN_WORKER_DIED")
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            self.close()
            raise QwenOllamaError("QWEN_WORKER_DIED") from None
        if not msg.get("ready"):
            code = str(msg.get("error") or "OLLAMA_UNREACHABLE")
            self.close()
            raise QwenOllamaError(code)
        self.quantization = str(msg.get("quantization") or "")
        self.parameter_size = str(msg.get("parameter_size") or "")
        return self

    def detect(self, texts: list[str]) -> list[list[dict]]:
        if self.proc is None or self.proc.stdin is None or self.proc.stdout is None:
            raise QwenOllamaError("QWEN_WORKER_DIED")
        payload = (json.dumps({"action": "detect", "texts": texts}, ensure_ascii=True) + "\n").encode("ascii")
        try:
            self.proc.stdin.write(payload)
            self.proc.stdin.flush()
            raw = self.proc.stdout.readline()
        except OSError:
            raise QwenOllamaError("DETECT_FAILED") from None
        if not raw:
            raise QwenOllamaError("QWEN_WORKER_DIED")
        try:
            msg = json.loads(raw.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            raise QwenOllamaError("DETECT_FAILED") from None
        if "error" in msg:
            code = str(msg["error"])
            kind = str(msg.get("kind") or "")
            raise QwenOllamaError(f"{code}:{kind}" if kind else code)
        return list(msg.get("results") or [[] for _ in texts])

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()
        self.proc = None


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[tuple[int, str]]:
    if len(text) <= max_chars:
        return [(0, text)]
    chunks: list[tuple[int, str]] = []
    pos = 0
    step = max(200, max_chars - overlap)
    while pos < len(text):
        end = min(len(text), pos + max_chars)
        chunks.append((pos, text[pos:end]))
        if end >= len(text):
            break
        pos += step
    return chunks


def _find_entity_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Locate ``needle`` in ``text`` via exact match or whitespace-tolerant fallback."""
    needle = (needle or "").strip()
    if len(needle) < 2:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + max(1, len(needle))
    if spans:
        return spans

    # Fallback: OCR text often inserts or omits spaces around digits/Korean syllables
    chars = [ch for ch in needle if not ch.isspace()]
    if len(chars) < 2:
        return []
    pattern = re.compile(r"\s*".join(re.escape(ch) for ch in chars))
    for m in pattern.finditer(text):
        spans.append((m.start(), m.end()))
    return spans


def find_pii_qwen(
    results: list[ReadResult],
    existing: list[Finding],
    engine: QwenOllamaEngine,
    doc_indexes: list[int] | None = None,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> tuple[list[Finding], QwenReport]:
    """Run local Qwen3.5 4B across blocks (OCR blocks prioritized first) to find PII spans."""
    report = QwenReport(model=getattr(engine, "model", "qwen3.5:4b"), quantization=getattr(engine, "quantization", ""))
    target_docs = doc_indexes if doc_indexes is not None else list(range(len(results)))
    todo_items: list[tuple[int, object, int, str]] = []
    for di in target_docs:
        blocks = sorted(results[di].blocks, key=lambda b: (0 if b.kind == OCR else 1))
        for b in blocks:
            if not b.text.strip():
                continue
            if b.kind == METADATA and (b.location.field or "").lower() in _SKIP_META_FIELDS:
                continue
            for offset, chunk in _chunk_text(b.text):
                if chunk.strip():
                    todo_items.append((di, b, offset, chunk))

    batches = [todo_items[i : i + BATCH] for i in range(0, len(todo_items), BATCH)]
    spans: list[AiSpan] = []
    seen_spans: set[tuple[int, str, int, int, str]] = set()

    for idx, batch in enumerate(batches, start=1):
        if cancelled and cancelled():
            break
        rows = engine.detect([item[3] for item in batch])
        report.blocks += len(batch)
        for (di, block, base_offset, chunk), ents in zip(batch, rows):
            folded = fold(block.text)
            for ent in ents or ():
                raw_cat = str(ent.get("category") or "other_pii").lower()
                category = CATEGORY_MAP.get(raw_cat, OTHER)
                text_val = str(ent.get("text") or "").strip()
                reason = str(ent.get("reason") or "").strip() or None
                for ls, le in _find_entity_spans(chunk, text_val):
                    gs = base_offset + ls
                    ge = base_offset + le
                    if category == NAME:
                        trimmed = trim_name_span(folded, gs, ge)
                        if trimmed is None:
                            continue
                        gs, ge = trimmed
                    if _is_qwen_false_positive(category, block.text[gs:ge], block.text, gs, ge):
                        continue
                    key = (di, block.block_id, gs, ge, category)
                    if key in seen_spans:
                        continue
                    seen_spans.add(key)
                    spans.append(AiSpan(di, block.block_id, gs, ge, category, note=reason))
        if progress:
            progress(idx, max(1, len(batches)))

    report.entities = len(spans)
    new = to_findings(results, existing, spans, BY_AI_QWEN, TIER_REVIEW)
    report.added = len(new)
    log.info(
        "Qwen(%s %s): 청크 %d, 탐지 %d, 새 탐지 %d",
        report.model,
        report.quantization or "unknown",
        report.blocks,
        report.entities,
        report.added,
    )
    return new, report
