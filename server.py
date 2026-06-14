#!/usr/bin/env python3
"""
LocalLLMArena — a zero-dependency local web app for benchmarking local LLM models
served through an LM Studio (OpenAI-compatible) endpoint.

Run:  python3 server.py
Then open the URL it prints (default http://localhost:8080).

Only the Python standard library is used, so there is nothing to pip install.
"""

import json
import os
import re
import time
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "benchmarks")
RESULTS_DIR = os.path.join(ROOT, "results")
WEB_DIR = os.path.join(ROOT, "web")
CONFIG_PATH = os.path.join(ROOT, "config.json")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config()


# --------------------------------------------------------------------------- #
#  LM Studio client (OpenAI-compatible Chat Completions API)
# --------------------------------------------------------------------------- #
def lmstudio_url(path):
    base = CONFIG["lmstudio_base_url"].rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base + path


def http_json(method, url, payload=None, timeout=60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models():
    try:
        data = http_json("GET", lmstudio_url("/models"), timeout=15)
        models = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return {"ok": True, "models": models}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e), "models": []}


def chat_completion(model, messages, params):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature", 0),
        "max_tokens": params.get("max_tokens", 1024),
        "stream": False,
    }
    if "top_p" in params:
        payload["top_p"] = params["top_p"]
    t0 = time.time()
    data = http_json(
        "POST",
        lmstudio_url("/chat/completions"),
        payload,
        timeout=CONFIG.get("request_timeout_seconds", 600),
    )
    elapsed = time.time() - t0
    content = ""
    try:
        content = data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        content = ""
    usage = data.get("usage", {}) or {}
    return {
        "content": content,
        "latency_s": round(elapsed, 3),
        "tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


# --------------------------------------------------------------------------- #
#  Scoring engine
# --------------------------------------------------------------------------- #
LETTER_RE = re.compile(r"\b([A-J])\b")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def normalize(text):
    return (text or "").strip().lower()


def parse_option_letters(prompt):
    """Find the option letters a multiple-choice prompt actually offers,
    e.g. lines like 'A) Walk' or '(B) Drive' -> {'A', 'B'}."""
    letters = set(re.findall(r"(?m)^\s*\(?([A-J])[\).\:]", prompt or ""))
    return {l.upper() for l in letters}


def extract_letter(text, allowed=None):
    """Pull a multiple-choice letter (A-J) out of a model answer.
    If `allowed` (a set of valid letters) is given, only those count, which
    avoids grabbing stray capitals from the model's reasoning prose."""
    if not text:
        return None
    t = text.strip()
    valid = {l.upper() for l in allowed} if allowed else None

    def ok(letter):
        return valid is None or letter in valid

    # Explicit phrasing, e.g. "answer is C", "the answer: (B)". Use the LAST
    # such match, since models often restate their final answer at the end.
    matches = [m.group(1).upper() for m in re.finditer(
        r"(?:answer|correct|option|choice)\D{0,15}?([A-J])\b", t, re.IGNORECASE)
        if ok(m.group(1).upper())]
    if matches:
        return matches[-1]
    # A bare leading letter, e.g. "B) because ..." or "B."
    m = re.match(r"\s*\(?([A-J])[\).:\s]", t)
    if m and ok(m.group(1).upper()):
        return m.group(1).upper()
    # Fallback: last standalone capital letter among the allowed/A-J set.
    found = [l for l in LETTER_RE.findall(t.upper()) if ok(l)]
    return found[-1] if found else None


def extract_number(text):
    """Pull the final numeric value out of a model answer (GSM8K style)."""
    if not text:
        return None
    cleaned = text.replace("$", "").replace("%", "")
    # Prefer a number that follows an explicit answer marker.
    m = re.search(
        r"(?:answer|=|is|total|result)\D{0,10}?(-?\d[\d,]*\.?\d*)",
        cleaned,
        re.IGNORECASE,
    )
    candidates = NUMBER_RE.findall(cleaned)
    raw = None
    if m:
        raw = m.group(1)
    elif candidates:
        raw = candidates[-1]
    if raw is None:
        return None
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def score_case(scoring, expected, output, prompt=None):
    """Return (passed: bool, detail: str)."""
    if scoring == "multiple_choice":
        allowed = parse_option_letters(prompt) if prompt else None
        got = extract_letter(output, allowed)
        exp = (expected or "").strip().upper()
        return got == exp, f"extracted={got!r} expected={exp!r}"

    if scoring == "numeric":
        got = extract_number(output)
        try:
            exp = float(str(expected).replace(",", ""))
        except ValueError:
            return False, f"bad expected value {expected!r}"
        if got is None:
            return False, f"no number found; expected={exp}"
        ok = abs(got - exp) <= 1e-6 * max(1.0, abs(exp))
        return ok, f"extracted={got} expected={exp}"

    if scoring == "exact_match":
        return normalize(output) == normalize(expected), "normalized exact match"

    if scoring == "contains":
        # expected may be a single string or list of acceptable strings (any).
        exps = expected if isinstance(expected, list) else [expected]
        out = normalize(output)
        hit = any(normalize(e) in out for e in exps)
        return hit, f"looked for any of {exps!r}"

    if scoring == "contains_all":
        exps = expected if isinstance(expected, list) else [expected]
        out = normalize(output)
        ok = all(normalize(e) in out for e in exps)
        return ok, f"required all of {exps!r}"

    if scoring == "regex":
        try:
            return bool(re.search(expected, output or "", re.IGNORECASE | re.DOTALL)), f"regex {expected!r}"
        except re.error as e:
            return False, f"bad regex: {e}"

    return False, f"unknown scoring type {scoring!r}"


# --------------------------------------------------------------------------- #
#  Benchmark loading
# --------------------------------------------------------------------------- #
def load_benchmark_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("id", os.path.splitext(os.path.basename(path))[0])
    data.setdefault("name", data["id"])
    data.setdefault("category", "Custom")
    data.setdefault("description", "")
    data.setdefault("scoring", "exact_match")
    data.setdefault("default_params", {})
    data.setdefault("cases", [])
    data["_file"] = os.path.basename(path)
    return data


def all_benchmarks():
    out = []
    if not os.path.isdir(BENCH_DIR):
        return out
    for fn in sorted(os.listdir(BENCH_DIR)):
        if fn.endswith(".json"):
            try:
                out.append(load_benchmark_file(os.path.join(BENCH_DIR, fn)))
            except Exception as e:  # noqa: BLE001
                out.append({"id": fn, "name": fn, "error": str(e), "cases": []})
    return out


def get_benchmark(bid):
    for b in all_benchmarks():
        if b.get("id") == bid:
            return b
    return None


# --------------------------------------------------------------------------- #
#  Run a benchmark (streams NDJSON events back to the client)
# --------------------------------------------------------------------------- #
def build_messages(bench, case):
    messages = []
    system = case.get("system_prompt") or bench.get("system_prompt")
    if not system:
        # A light default that nudges models toward gradeable answers.
        if bench.get("scoring") == "multiple_choice":
            system = "Answer the multiple choice question. End with 'Answer: <letter>'."
        elif bench.get("scoring") == "numeric":
            system = "Solve the problem. End your reply with 'Answer: <number>'."
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


def run_benchmark_stream(model, bench, params, emit):
    """emit(dict) pushes one NDJSON event to the client."""
    cases = bench.get("cases", [])
    total = len(cases)
    passed = 0
    latencies = []
    case_results = []
    emit({"type": "start", "benchmark_id": bench["id"],
          "benchmark_name": bench["name"], "total": total})

    for i, case in enumerate(cases):
        merged = dict(CONFIG.get("default_params", {}))
        merged.update(bench.get("default_params", {}))
        merged.update(params or {})
        try:
            messages = build_messages(bench, case)
            res = chat_completion(model, messages, merged)
            ok, detail = score_case(bench["scoring"], case.get("expected"),
                                    res["content"], case.get("prompt"))
            if res["latency_s"] is not None:
                latencies.append(res["latency_s"])
            if ok:
                passed += 1
            cr = {
                "id": case.get("id", f"case_{i+1}"),
                "prompt": case["prompt"],
                "expected": case.get("expected"),
                "output": res["content"],
                "passed": ok,
                "detail": detail,
                "latency_s": res["latency_s"],
                "tokens": res["tokens"],
            }
        except Exception as e:  # noqa: BLE001
            cr = {
                "id": case.get("id", f"case_{i+1}"),
                "prompt": case["prompt"],
                "expected": case.get("expected"),
                "output": "",
                "passed": False,
                "detail": f"ERROR: {e}",
                "latency_s": None,
                "tokens": None,
                "error": True,
            }
        case_results.append(cr)
        emit({"type": "case", "index": i, "total": total, "result": cr})

    summary = {
        "type": "done",
        "benchmark_id": bench["id"],
        "benchmark_name": bench["name"],
        "category": bench.get("category"),
        "model": model,
        "total": total,
        "passed": passed,
        "score": round(100.0 * passed / total, 1) if total else 0.0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
    }
    emit(summary)
    return summary, case_results


def save_result(model, summaries, all_cases, params):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rid = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    total = sum(s["total"] for s in summaries)
    passed = sum(s["passed"] for s in summaries)
    record = {
        "id": rid,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "params": params,
        "overall_score": round(100.0 * passed / total, 1) if total else 0.0,
        "total": total,
        "passed": passed,
        "benchmarks": summaries,
        "cases": all_cases,
    }
    with open(os.path.join(RESULTS_DIR, rid + ".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return record


def list_results():
    out = []
    if not os.path.isdir(RESULTS_DIR):
        return out
    for fn in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(RESULTS_DIR, fn), "r", encoding="utf-8") as f:
                r = json.load(f)
            out.append({
                "id": r["id"],
                "timestamp": r["timestamp"],
                "model": r["model"],
                "overall_score": r["overall_score"],
                "total": r["total"],
                "passed": r["passed"],
                "benchmarks": [
                    {"benchmark_name": b["benchmark_name"], "score": b["score"],
                     "passed": b["passed"], "total": b["total"]}
                    for b in r.get("benchmarks", [])
                ],
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def get_result(rid):
    path = os.path.join(RESULTS_DIR, rid + ".json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
#  HTTP handler
# --------------------------------------------------------------------------- #
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalLLMArena/1.0"

    def log_message(self, fmt, *args):  # quieter logs
        pass

    # -- helpers --------------------------------------------------------- #
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel):
        rel = rel.lstrip("/")
        if rel == "" or rel == "index.html":
            rel = "index.html"
        path = os.path.normpath(os.path.join(WEB_DIR, rel))
        if not path.startswith(WEB_DIR) or not os.path.isfile(path):
            self.send_error(404, "Not found")
            return
        ext = os.path.splitext(path)[1]
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}

    # -- routing --------------------------------------------------------- #
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/config":
            self._send_json({
                "lmstudio_base_url": CONFIG["lmstudio_base_url"],
                "default_params": CONFIG.get("default_params", {}),
            })
            return
        if path == "/api/models":
            self._send_json(list_models())
            return
        if path == "/api/benchmarks":
            benches = all_benchmarks()
            meta = [{
                "id": b["id"], "name": b["name"], "category": b.get("category"),
                "description": b.get("description"), "scoring": b.get("scoring"),
                "count": len(b.get("cases", [])), "file": b.get("_file"),
                "error": b.get("error"),
            } for b in benches]
            self._send_json({"benchmarks": meta})
            return
        if path.startswith("/api/benchmarks/"):
            bid = path.rsplit("/", 1)[-1]
            b = get_benchmark(bid)
            if b is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(b)
            return
        if path == "/api/results":
            self._send_json({"results": list_results()})
            return
        if path.startswith("/api/results/"):
            rid = path.rsplit("/", 1)[-1]
            r = get_result(rid)
            if r is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(r)
            return

        # static files
        self._send_static(path)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            self._handle_run()
            return
        if path == "/api/reload-config":
            global CONFIG
            CONFIG = load_config()
            self._send_json({"ok": True, "lmstudio_base_url": CONFIG["lmstudio_base_url"]})
            return

        self.send_error(404, "Not found")

    # -- streaming run --------------------------------------------------- #
    def _handle_run(self):
        body = self._read_body()
        model = body.get("model")
        bench_ids = body.get("benchmark_ids", [])
        params = body.get("params", {})
        if not model or not bench_ids:
            self._send_json({"error": "model and benchmark_ids are required"}, 400)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def emit(obj):
            try:
                self.wfile.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                raise

        summaries = []
        all_cases = []
        try:
            for bid in bench_ids:
                bench = get_benchmark(bid)
                if bench is None:
                    emit({"type": "error", "message": f"benchmark {bid} not found"})
                    continue
                summary, cases = run_benchmark_stream(model, bench, params, emit)
                summaries.append(summary)
                for c in cases:
                    c["benchmark_id"] = bid
                    all_cases.append(c)
            record = save_result(model, summaries, all_cases, params)
            emit({"type": "saved", "result_id": record["id"],
                  "overall_score": record["overall_score"],
                  "passed": record["passed"], "total": record["total"]})
        except (BrokenPipeError, ConnectionResetError):
            return  # client navigated away
        except Exception as e:  # noqa: BLE001
            try:
                emit({"type": "fatal", "message": str(e)})
            except Exception:  # noqa: BLE001
                pass


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    port = int(CONFIG.get("server_port", 8080))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    url = f"http://localhost:{port}"
    print("=" * 64)
    print("  LocalLLMArena is running")
    print(f"  Open:        {url}")
    print(f"  LM Studio:   {CONFIG['lmstudio_base_url']}")
    print(f"  Benchmarks:  {BENCH_DIR}")
    print("  Stop:        Ctrl+C")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
