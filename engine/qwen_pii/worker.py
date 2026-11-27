"""Ollama local Qwen3.5 4B PII detection worker process.

Lives under engine/ (outside src/) so tests/test_no_network.py keeps strict
AST isolation inside src/. Installs a loopback-only sys.addaudithook allowing
only 127.0.0.1 / ::1 and communicates with the parent via stdin/stdout JSON lines.
"""

from __future__ import annotations

import http.client
import json
import sys

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_NET_EVENTS = (
    "socket.connect",
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.gethostbyname_ex",
    "socket.gethostbyaddr",
    "socket.sendto",
    "socket.sendmsg",
)

PII_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["name", "phone", "email", "address", "id_number", "account", "other_pii"],
                    },
                    "reason": {"type": "string"},
                },
                "required": ["text", "category", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["entities"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = (
    "너는 강의안·증명서(PDF, PPTX)와 OCR 텍스트에서 개인정보(PII)만 정밀하게 추출하는 보안 감사관이다.\n"
    "입력된 텍스트(OCR 오탈자가 섞일 수 있음)에서 실존 개인의 개인정보에 해당하는 정확한 원문 부분문자열(verbatim substring)을 추출하라.\n"
    "추출 대상 카테고리:\n"
    "- name: 실존 인명(성명/담당자/확인자 칸의 인명, 수기 성명, 직인/원본대조필 인명)\n"
    "- phone: 휴대전화·유선전화·팩스 번호(010-..., 031-..., 031)..., FAX 031 ... 등)\n"
    "- email: 이메일 주소\n"
    "- address: 개인 거주지, 상세 도로명/지번 주소\n"
    "- id_number: 주민등록번호, 외국인등록번호, 학번, 교번, 여권번호, 운전면허번호, 차량번호\n"
    "- account: 은행 계좌번호, 카드번호\n"
    "- other_pii: 개인 SNS 계정(@id), 명시적 생년월일('생년월일' 인접 날짜)\n"
    "제외 대상(절대 추출하지 말 것):\n"
    "1. 양식/표 머리글·라벨: 성명, 한글, 한자, 생년월일, 주소, 연락처, 전화번호, 담당자, 교부번호, 발급번호, 입사일자, 퇴사일자, 근무기간, 소속, 직위, 이하 여백, 해당없음, 기초학력, 기간제교사, 원본대조필\n"
    "2. 기관·학교·법인명: ○○초등학교, ○○중학교, ○○고등학교, ○○교육청, (주)○○, ○○아카데미\n"
    "3. 공문서·증명서 발급/교부 번호(예: J1000..., 2025-1859, 제 2025-000236 호, 162 같은 단순 숫자)\n"
    "4. 출생일이 아닌 근무기간·계약기간·발급일·시행일 날짜(YYYY.MM.DD. 등)\n"
    "5. 의미 불명의 짧은 OCR 노이즈 조각(1~2자 자모/기호 섞임)\n"
    "규칙:\n"
    "1. text 필드에는 반드시 입력 문자열 안에 실제로 등장하는 글자를 그대로 적어라.\n"
    "2. 직함('교수', '팀장', '선생님', '강사', '주무관')은 name에 포함하지 말고 이름만 추출하라.\n"
    "3. 개인정보가 없으면 {\"entities\": []}만 반환하라."
)


def _host_of(args) -> str | None:
    if not args:
        return None
    val = args[1] if len(args) > 1 and isinstance(args[1], tuple) else args[0]
    if isinstance(val, tuple) and val:
        host = val[0]
        return host.decode() if isinstance(host, bytes) else str(host)
    if isinstance(val, bytes):
        return val.decode(errors="ignore")
    return str(val) if val is not None else None


def install_loopback_guard() -> None:
    def _hook(event: str, args: tuple) -> None:
        if event not in _NET_EVENTS:
            return
        host = _host_of(args)
        if host is None:
            return
        norm = host.strip().lower()
        if norm.startswith("::ffff:"):
            norm = norm[len("::ffff:") :]
        if norm in LOOPBACK_HOSTS:
            return
        raise RuntimeError(f"외부 통신 차단: {event} -> {host}")

    sys.addaudithook(_hook)


def _send(out, obj: dict) -> None:
    out.write((json.dumps(obj, ensure_ascii=True) + "\n").encode("ascii"))
    out.flush()


def _http_json(host: str, port: int, method: str, path: str, body: dict | None = None, timeout: int = 180) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8", errors="replace")
        if resp.status >= 400:
            raise RuntimeError(f"HTTP_{resp.status}")
        return json.loads(raw)
    finally:
        conn.close()


def _check_model(host: str, port: int, model: str) -> tuple[dict | None, str | None]:
    try:
        tags = _http_json(host, port, "GET", "/api/tags", timeout=10)
    except Exception:  # noqa: BLE001
        return None, "OLLAMA_UNREACHABLE"
    models = tags.get("models", [])
    base = model.split(":")[0]
    matched = next((m for m in models if m.get("name") == model), None)
    if matched is None and ":" not in model:
        matched = next((m for m in models if (m.get("name") or "").split(":")[0] == base), None)
    if matched is not None:
        details = matched.get("details") or {}
        return {
            "model": matched.get("name") or model,
            "quantization": str(details.get("quantization_level") or ""),
            "parameter_size": str(details.get("parameter_size") or ""),
        }, None
    return None, "MODEL_NOT_FOUND"


def _detect_one(host: str, port: int, model: str, text: str, num_gpu: int | None = None) -> list[dict]:
    if not text or not text.strip():
        return []
    opts: dict[str, object] = {"temperature": 0, "num_ctx": 4096, "num_predict": 512}
    if num_gpu is not None:
        opts["num_gpu"] = int(num_gpu)
    req: dict[str, object] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "stream": False,
        "keep_alive": "10m",
        "options": opts,
        "format": PII_SCHEMA,
    }
    if "qwen" in model.lower():
        req["think"] = False
    resp = _http_json(host, port, "POST", "/api/chat", body=req, timeout=180)
    content = (resp.get("message") or {}).get("content") or "{}"
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return []
    entities = parsed.get("entities") or []
    out = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        t = str(ent.get("text") or "").strip()
        cat = str(ent.get("category") or "other_pii").strip()
        reason = str(ent.get("reason") or "").strip()
        if t:
            out.append({"text": t, "category": cat, "reason": reason})
    return out


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    model = "qwen3.5:4b"
    host = "127.0.0.1"
    port = 11434
    num_gpu: int | None = None
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--num-gpu" and i + 1 < len(args):
            num_gpu = int(args[i + 1])
            i += 2
        else:
            i += 1

    install_loopback_guard()
    out = sys.stdout.buffer
    sys.stdout = sys.stderr

    info, err = _check_model(host, port, model)
    if info is None:
        _send(out, {"error": err or "OLLAMA_UNREACHABLE"})
        return 2
    _send(out, {"ready": True, **info})

    for raw in sys.stdin.buffer:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("action") == "ping":
                _send(out, {"ok": True, "model": model})
                continue
            texts = req.get("texts") or []
            results = [_detect_one(host, port, model, t, num_gpu=num_gpu) for t in texts]
            _send(out, {"results": results})
        except Exception as e:  # noqa: BLE001
            _send(out, {"error": "DETECT_FAILED", "kind": type(e).__name__})
    return 0


if __name__ == "__main__":
    sys.exit(main())
