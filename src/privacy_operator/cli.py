"""강의안(PDF, PPTX 등) 사전검열 및 이미지 OCR + 로컬 Qwen3.5 4B 마스킹 CLI."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from privacy_operator.alias.codes import AliasBook
from privacy_operator.alias.pseudonymize import pseudonymize
from privacy_operator.detect import detect, escalate
from privacy_operator.detect.ai.qwen_ollama import QwenOllamaEngine, QwenOllamaError, find_pii_qwen
from privacy_operator.detect.dictionaries import GRAY_CATEGORIES
from privacy_operator.detect.finding import (
    ACTION_ALIAS,
    ACTION_DELETE,
    STATUS_CONFIRMED,
    TIER_CERTAIN,
    TIER_REVIEW,
    Finding,
)
from privacy_operator.reader import DocumentReader
from privacy_operator.reader.blocks import OCR
from privacy_operator.reader.read import SUPPORTED
from privacy_operator.writer.original import OUTPUT_SUFFIX, verify, write_original
from privacy_operator.writer.plan import ReplacementPlan


def _mask_preview(val: str) -> str:
    val = (val or "").strip()
    if len(val) <= 2:
        return "*" * len(val)
    return val[0] + "*" * max(1, len(val) - 2) + val[-1]


def _collect_files(paths: list[str], exts: set[str] | None, max_files: int | None) -> list[Path]:
    found: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if not p.name.startswith("~$") and p.suffix.lower() in SUPPORTED:
                if exts is None or p.suffix.lower() in exts:
                    found.append(p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if not child.is_file() or child.name.startswith("~$"):
                    continue
                suf = child.suffix.lower()
                if suf in SUPPORTED and (exts is None or suf in exts):
                    found.append(child)
    dedup = list(dict.fromkeys(found))
    return dedup[:max_files] if max_files and max_files > 0 else dedup


def _confirm_for_cli(findings: list[Finding], include_review: bool = True, include_gray: bool = False) -> list[Finding]:
    for f in findings:
        if f.exclude_hint:
            continue
        if f.category in GRAY_CATEGORIES and not include_gray:
            continue
        if f.tier == TIER_CERTAIN or (include_review and f.tier == TIER_REVIEW):
            f.review_status = STATUS_CONFIRMED
            if f.action not in (ACTION_ALIAS, ACTION_DELETE):
                f.action = ACTION_ALIAS
    return [f for f in findings if f.review_status == STATUS_CONFIRMED]


def _cmd_scan(args: argparse.Namespace) -> int:
    exts = {("." + e.lstrip(".")).lower() for e in args.ext} if args.ext else None
    files = _collect_files(args.paths, exts, args.max_files)
    if not files:
        print("검사할 파일이 없습니다.", file=sys.stderr)
        return 1

    engine = None
    model_info = "disabled"
    if not args.no_llm:
        try:
            engine = QwenOllamaEngine(model=args.model, port=args.port).start()
            model_info = f"{engine.model} ({engine.quantization or 'unknown'})"
        except QwenOllamaError as e:
            print(f"[경고] Ollama 연결 실패({e.code}), 규칙 탐지만 실행합니다.", file=sys.stderr)

    reports = []
    try:
        with DocumentReader(ocr_enabled=not args.no_ocr) as reader:
            for path in files:
                try:
                    res = reader.read(path)
                except Exception as e:  # noqa: BLE001
                    reports.append({"file": str(path), "error": type(e).__name__})
                    continue
                findings = detect([res])
                qwen_added = 0
                if engine is not None:
                    new_q, _rep = find_pii_qwen([res], findings, engine)
                    findings.extend(new_q)
                    escalate(findings)
                    qwen_added = len(new_q)

                ocr_blocks = [b for b in res.blocks if b.kind == OCR]
                cats = dict(Counter(f.category for f in findings))
                items = [
                    {
                        "id": f.finding_id,
                        "category": f.category,
                        "tier": f.tier,
                        "by": list(f.detected_by),
                        "page": f.location.page,
                        "kind": next((b.kind for b in res.blocks if b.block_id == f.block_id), ""),
                        "value": f.raw_value if args.show_values else _mask_preview(f.raw_value),
                    }
                    for f in findings
                ]
                reports.append(
                    {
                        "file": str(path),
                        "format": res.format,
                        "pages": res.page_count,
                        "blocks": len(res.blocks),
                        "ocr_blocks": len(ocr_blocks),
                        "blind_spots": len(res.blind_spots),
                        "findings_total": len(findings),
                        "qwen_added": qwen_added,
                        "categories": cats,
                        "items": items,
                    }
                )
    finally:
        if engine is not None:
            engine.close()

    if args.json:
        print(json.dumps({"llm": model_info, "files": reports}, ensure_ascii=False, indent=2))
    else:
        print(f"[PrivacyOperator CLI Scan] files={len(files)} ocr={not args.no_ocr} llm={model_info}")
        for r in reports:
            if "error" in r:
                print(f"  ERROR {r['file']}: {r['error']}")
                continue
            print(
                f"  {r['file']} [{r['format']}] pages={r['pages']} ocr_blocks={r['ocr_blocks']} "
                f"blind={r['blind_spots']} findings={r['findings_total']} (qwen+{r['qwen_added']}) cats={r['categories']}"
            )
    return 0


def _cmd_redact(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        print(f"입력 파일을 찾을 수 없습니다: {src}", file=sys.stderr)
        return 1
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    engine = None
    model_info = "disabled"
    if not args.no_llm:
        try:
            engine = QwenOllamaEngine(model=args.model, port=args.port).start()
            model_info = f"{engine.model} ({engine.quantization or 'unknown'})"
        except QwenOllamaError as e:
            print(f"[경고] Ollama 연결 실패({e.code}), 규칙 탐지만 실행합니다.", file=sys.stderr)

    try:
        with DocumentReader(ocr_enabled=not args.no_ocr) as reader:
            res = reader.read(src)
            findings = detect([res])
            qwen_added = 0
            if engine is not None:
                new_q, _rep = find_pii_qwen([res], findings, engine)
                findings.extend(new_q)
                escalate(findings)
                qwen_added = len(new_q)

            confirmed = _confirm_for_cli(findings, include_review=not args.certain_only, include_gray=args.include_gray)
            for f in findings:
                if f.review_status != STATUS_CONFIRMED:
                    f.review_status = "제외"

            book = AliasBook("CLI-REDACT")
            docs = pseudonymize([res], findings, book)
            plan = ReplacementPlan(findings, docs)
            ocr_blocks = [b for b in res.blocks if b.kind == OCR]

            tmp_stem = out_path.stem
            # write_original generates <stem><suffix>; ensure exact destination path when requested
            result = write_original(
                src,
                res.format,
                plan,
                out_path.parent,
                tmp_stem,
                reader,
                ocr_blocks=ocr_blocks,
                redact_style=args.style,
            )
            if not result.ok or result.path is None:
                print(f"[실패] 출력 생성 실패 status={result.status} detail={result.detail} places={result.places}", file=sys.stderr)
                return 2
            final_path = result.path
            if final_path != out_path:
                final_path.replace(out_path)
                final_path = out_path

            verified_leftovers: list[str] = []
            if args.verify_ocr:
                fmt_kind = "pdf" if res.format == "pdf" else "zip"
                places, _ = verify(final_path, fmt_kind, plan, reader)
                verified_leftovers = places

            summary = {
                "input": str(src),
                "output": str(final_path),
                "format": res.format,
                "ocr_enabled": not args.no_ocr,
                "ocr_blocks": len(ocr_blocks),
                "llm": model_info,
                "qwen_added": qwen_added,
                "confirmed_rules": len(plan.rules),
                "confirmed_findings": len(confirmed),
                "categories": dict(Counter(f.category for f in confirmed)),
                "style": args.style,
                "verify_ocr": args.verify_ocr,
                "leftovers": verified_leftovers,
            }
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 3 if verified_leftovers else 0
    finally:
        if engine is not None:
            engine.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="privacy-operator-cli", description="강의안 PDF/PPTX OCR + 로컬 Qwen3.5 4B 개인정보 검열/마스킹")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_scan = sub.add_parser("scan", help="PDF/PPTX 등 문서의 개인정보 사전 스캔")
    p_scan.add_argument("paths", nargs="+", help="파일 또는 폴더 경로 (하위 폴더 포함)")
    p_scan.add_argument("--ext", nargs="+", help="검사할 확장자 (예: pdf pptx)")
    p_scan.add_argument("--max-files", type=int, default=0, help="최대 검사 파일 수 (0=전체)")
    p_scan.add_argument("--no-ocr", action="store_true", help="PaddleOCR 비활성화")
    p_scan.add_argument("--no-llm", action="store_true", help="로컬 Ollama Qwen3.5 비활성화")
    p_scan.add_argument("--model", default="qwen3.5:4b", help="Ollama 모델 이름")
    p_scan.add_argument("--port", type=int, default=11434, help="Ollama 로컬 포트")
    p_scan.add_argument("--show-values", action="store_true", help="마스킹 없이 원문 탐지값 표시")
    p_scan.add_argument("--json", action="store_true", help="JSON 출력")
    p_scan.set_defaults(func=_cmd_scan)

    p_red = sub.add_parser("redact", help="PDF/PPTX 개인정보 마스킹 및 물리 픽셀 삭제")
    p_red.add_argument("input", help="입력 파일 경로")
    p_red.add_argument("-o", "--output", required=True, help="출력 파일 경로")
    p_red.add_argument("--no-ocr", action="store_true", help="PaddleOCR 비활성화")
    p_red.add_argument("--no-llm", action="store_true", help="로컬 Ollama Qwen3.5 비활성화")
    p_red.add_argument("--model", default="qwen3.5:4b", help="Ollama 모델 이름")
    p_red.add_argument("--port", type=int, default=11434, help="Ollama 로컬 포트")
    p_red.add_argument("--style", choices=["black_alias", "black_only", "white_alias"], default="black_alias")
    p_red.add_argument("--certain-only", action="store_true", help="확실 등급만 마스킹")
    p_red.add_argument("--include-gray", action="store_true", help="소속/직위 등 참고 항목까지 포함")
    p_red.add_argument("--verify-ocr", action="store_true", help="저장 후 재-OCR 검증 수행")
    p_red.set_defaults(func=_cmd_redact)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
