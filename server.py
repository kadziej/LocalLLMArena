#!/usr/bin/env python3
"""
LocalLLMArena — a zero-dependency local web app for benchmarking local LLM models
served through an LM Studio (OpenAI-compatible) endpoint.

Run:  python3 server.py
Then open the URL it prints (default http://localhost:8080).

Only the Python standard library is used, so there is nothing to pip install.
"""

import bisect
import json
import math
import os
import re
import threading
import time
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import storage

ROOT = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.join(ROOT, "benchmarks")
WEB_DIR = os.path.join(ROOT, "web")
CONFIG_PATH = os.path.join(ROOT, "config.json")


LOCAL_SETTINGS_PATH = os.path.join(ROOT, "settings.local.json")


def load_config():
    """Load config.json, then overlay the gitignored settings.local.json so
    machine-specific values (LM Studio URL, OpenRouter key) stay off GitHub."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if os.path.isfile(LOCAL_SETTINGS_PATH):
        try:
            with open(LOCAL_SETTINGS_PATH, "r", encoding="utf-8") as f:
                local = json.load(f)
            for k, v in (local or {}).items():
                cfg[k] = v
        except (OSError, json.JSONDecodeError):
            pass
    return cfg


def save_local_settings(updates):
    current = {}
    if os.path.isfile(LOCAL_SETTINGS_PATH):
        try:
            with open(LOCAL_SETTINGS_PATH, "r", encoding="utf-8") as f:
                current = json.load(f) or {}
        except (OSError, json.JSONDecodeError):
            current = {}
    for k, v in updates.items():
        current[k] = v
    storage.atomic_write_json(LOCAL_SETTINGS_PATH, current)
    return current


CONFIG = load_config()


def resolve_results_dir(cfg=None):
    """Where run records live. Override with the ARENA_RESULTS_DIR env var or
    "results_dir" in settings.local.json - e.g. a folder outside the checkout,
    so `git clean` or a fresh clone can never take the history with it."""
    cfg = CONFIG if cfg is None else cfg
    d = os.environ.get("ARENA_RESULTS_DIR") or cfg.get("results_dir") or "results"
    d = os.path.expanduser(d)
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


RESULTS_DIR = resolve_results_dir()

# Registry of in-flight runs so a separate request can cancel them.
RUNS = {}            # run_id -> {"cancel": bool}
RUNS_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
#  Provider clients (OpenAI-compatible Chat Completions API)
#  "local"      -> LM Studio (or any local OpenAI-compatible server)
#  "openrouter" -> https://openrouter.ai  (cloud, for comparison)
# --------------------------------------------------------------------------- #
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def provider_base_url(provider):
    if provider == "openrouter":
        return OPENROUTER_BASE
    base = (CONFIG.get("lmstudio_base_url") or "http://localhost:1234").rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


def provider_headers(provider):
    h = {"Content-Type": "application/json"}
    if provider == "openrouter":
        key = CONFIG.get("openrouter_api_key")
        if key:
            h["Authorization"] = "Bearer " + key
        # OpenRouter asks apps to identify themselves.
        h["HTTP-Referer"] = "https://github.com/kadziej/LocalLLMArena"
        h["X-Title"] = "LocalLLMArena"
    return h


def http_json(method, url, headers=None, payload=None, timeout=60):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {"Content-Type": "application/json"}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(provider="local"):
    """Return models for a provider. For OpenRouter, include pricing (USD per
    token) and context length so the UI can show and compare cost."""
    try:
        url = provider_base_url(provider) + "/models"
        data = http_json("GET", url, headers=provider_headers(provider), timeout=25)
        models = []
        for m in data.get("data", []):
            mid = m.get("id")
            if not mid:
                continue
            if provider != "openrouter" and "embed" in mid.lower():
                continue  # embedding models can't run chat benchmarks
            entry = {"id": mid, "name": m.get("name") or mid}
            pricing = m.get("pricing") or {}
            try:
                entry["prompt_price"] = float(pricing.get("prompt") or 0) or 0.0
                entry["completion_price"] = float(pricing.get("completion") or 0) or 0.0
            except (TypeError, ValueError):
                pass
            if m.get("context_length"):
                entry["context_length"] = m["context_length"]
            models.append(entry)
        models.sort(key=lambda x: x["id"].lower())
        return {"ok": True, "provider": provider, "models": models}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "provider": provider, "error": str(e), "models": []}


def chat_completion(provider, model, messages, params, should_cancel=None, on_token=None):
    """Call the streaming chat-completions API for `provider` and collect rich
    timing/token/cost metrics. `should_cancel()` is polled between chunks to
    allow instant stop; `on_token(piece, count)` is called for each token."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature", 0),
        "max_tokens": params.get("max_tokens", 1024),
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if "top_p" in params:
        payload["top_p"] = params["top_p"]
    if provider == "openrouter":
        # Ask OpenRouter to report the exact USD cost in the usage object.
        payload["usage"] = {"include": True}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        provider_base_url(provider) + "/chat/completions", data=data, method="POST")
    headers = provider_headers(provider)
    headers["Accept"] = "text/event-stream"
    for k, v in headers.items():
        req.add_header(k, v)

    t0 = time.time()
    ttft = None
    parts = []
    reasoning_parts = []
    reasoning_chunks = 0
    chunk_tokens = 0
    usage = {}
    finish_reason = None
    cancelled = False

    timeout = CONFIG.get("request_timeout_seconds", 600)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if should_cancel and should_cancel():
                cancelled = True
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue  # ignore SSE comments / keep-alives
            chunk = line[len("data:"):].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                delta = ch.get("delta") or {}
                # Reasoning models stream their thinking in a side channel:
                # LM Studio calls it `reasoning_content`, OpenRouter `reasoning`.
                # It arrives before any content, so it - not the first content
                # token - is the true time to first token.
                rpiece = delta.get("reasoning_content") or delta.get("reasoning")
                if rpiece:
                    if ttft is None:
                        ttft = time.time() - t0
                    reasoning_parts.append(rpiece)
                    chunk_tokens += 1
                    reasoning_chunks += 1
                    if on_token:
                        on_token(rpiece, chunk_tokens, "reasoning")
                piece = delta.get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    parts.append(piece)
                    chunk_tokens += 1
                    if on_token:
                        on_token(piece, chunk_tokens, "content")
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]

    end = time.time()
    content = "".join(parts)
    # Thinking reaches us two ways - the side channel above and inline tags in
    # the content - and a model may use either. Merge both into one transcript
    # and keep `answer` (what gets scored) free of all of it.
    inline_reasoning, answer = split_reasoning(content)
    reasoning = "\n".join(
        p for p in ("".join(reasoning_parts).strip(), inline_reasoning) if p).strip()
    latency = end - t0
    completion_tokens = usage.get("completion_tokens") or (chunk_tokens or None)
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and completion_tokens is not None and prompt_tokens is not None:
        total_tokens = completion_tokens + prompt_tokens

    # Cost: OpenRouter reports exact USD in usage.cost; local models are free.
    cost = usage.get("cost")
    if cost is None and provider == "local":
        cost = 0.0

    # tokens/sec = generation speed over the decode window (first token ->
    # last token). Time before the first token (model load, prompt processing)
    # is TTFT and is reported separately. A 1-3 token answer has no measurable
    # decode window, so it gets no speed at all rather than a misleading one.
    decode_s = (end - (t0 + ttft)) if ttft is not None else None
    tps = storage.decode_tps(completion_tokens, decode_s)

    # How much of the completion budget went to thinking. Exact when the API
    # reports it or when thinking came over the side channel; for inline tags we
    # can only apportion by character share, so flag that as an estimate.
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = details.get("reasoning_tokens")
    reasoning_exact = reasoning_tokens is not None
    if reasoning_tokens is None and reasoning_chunks:
        reasoning_tokens, reasoning_exact = reasoning_chunks, True
    if reasoning_tokens is None and inline_reasoning and completion_tokens and content:
        reasoning_tokens = int(round(
            completion_tokens * len(inline_reasoning) / len(content)))
        reasoning_exact = False

    return {
        "content": content,
        "answer": answer,
        "reasoning": reasoning,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_tokens_exact": reasoning_exact,
        "latency_s": round(latency, 3),
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(tps, 1) if tps is not None else None,
        "decode_s": round(decode_s, 3) if decode_s is not None else None,
        "cost_usd": cost,
        "finish_reason": finish_reason,
        "cancelled": cancelled,
    }


# --------------------------------------------------------------------------- #
#  Scoring engine
# --------------------------------------------------------------------------- #
LETTER_RE = re.compile(r"\b([A-J])\b")
NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


REASONING_TAGS = ("think", "thinking", "thought", "reason", "reasoning",
                  "reflection", "scratchpad")
_OPEN_TAG_RE = re.compile(
    r"<\s*(" + "|".join(REASONING_TAGS) + r")\s*>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(
    r"<\s*/\s*(?:" + "|".join(REASONING_TAGS) + r")\s*>", re.IGNORECASE)


def split_reasoning(text):
    """Separate a model's chain-of-thought from its actual answer.

    Reasoning models emit thinking either in a dedicated `reasoning_content`
    stream field (handled in chat_completion) or inline in the content, wrapped
    in tags like <think>...</think>. Scoring must only ever see the answer:
    thinking is full of restated options, discarded candidate numbers and the
    words "the answer is ...", every one of which corrupts letter/number
    extraction and `contains` matching.

    Returns (reasoning, answer). Two ragged real-world shapes are handled:
    a chat template that pre-fills the opening tag, so the reply starts mid-
    thought and only the closing tag appears; and an opening tag that is never
    closed, which is what a max_tokens truncation mid-thought looks like -
    there, everything after the tag is reasoning and the answer is empty.
    """
    if not text:
        return "", text or ""

    if not _OPEN_TAG_RE.search(text):
        # Template pre-filled the opening tag: reasoning runs until the close.
        cm = _CLOSE_TAG_RE.search(text)
        if cm:
            return text[:cm.start()].strip(), text[cm.end():].strip()
        return "", text.strip()

    reasoning, answer, pos = [], [], 0
    while True:
        m = _OPEN_TAG_RE.search(text, pos)
        if not m:
            answer.append(text[pos:])
            break
        answer.append(text[pos:m.start()])
        close = re.compile(r"<\s*/\s*" + re.escape(m.group(1)) + r"\s*>",
                           re.IGNORECASE)
        cm = close.search(text, m.end())
        if not cm:  # truncated mid-thought - no answer was ever produced
            reasoning.append(text[m.end():])
            break
        reasoning.append(text[m.end():cm.start()])
        pos = cm.end()

    return ("\n".join(p.strip() for p in reasoning if p.strip()).strip(),
            "".join(answer).strip())


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
    # Prefer the number after the LAST explicit "Answer" marker (models reason
    # first, then state the final answer). Taking the last match avoids grabbing
    # an intermediate number from phrases like "...half of 48 is 24...".
    ans = list(re.finditer(
        r"answer\D{0,15}?(-?\d[\d,]*\.?\d*)", cleaned, re.IGNORECASE))
    if ans:
        raw = ans[-1].group(1)
    else:
        # Otherwise fall back to the last number in the text, which is almost
        # always the final computed result.
        candidates = NUMBER_RE.findall(cleaned)
        raw = candidates[-1] if candidates else None
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

    if scoring in ("regex", "regex_cs"):
        # "regex" is case-insensitive; "regex_cs" is case-sensitive (needed to
        # grade format/case constraints like "reply in UPPERCASE").
        flags = re.DOTALL | (re.IGNORECASE if scoring == "regex" else 0)
        try:
            ok = bool(re.search(expected, output or "", flags))
            return ok, f"{scoring} {expected!r}"
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
        scoring = case.get("scoring") or bench.get("scoring")
        if scoring == "multiple_choice":
            system = "Answer the multiple choice question. End with 'Answer: <letter>'."
        elif scoring == "numeric":
            system = "Solve the problem. End your reply with 'Answer: <number>'."
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": case["prompt"]})
    return messages


DEFAULT_REASONING_BUDGET = 4096
_REASONING_MODELS = None          # set of (provider, model) seen thinking
_REASONING_LOCK = threading.Lock()


def effective_params(bench, params):
    """Generation params for one benchmark: config defaults < benchmark
    defaults < what the user set for this run. A blank/None value from the UI
    means "use the default", not zero.

    max_tokens is the *answer* budget each benchmark is designed around.
    reasoning_budget is extra room granted on top of it to models that think
    before answering (their thinking counts against max_tokens), so they are
    not graded on an answer cut off mid-thought."""
    merged = {"max_tokens": 1024,
              "reasoning_budget": CONFIG.get("reasoning_budget", DEFAULT_REASONING_BUDGET)}
    for layer in (CONFIG.get("default_params", {}), bench.get("default_params", {}),
                  params or {}):
        merged.update({k: v for k, v in layer.items() if v is not None and v != ""})
    merged["max_tokens"] = max(1, int(merged["max_tokens"]))
    merged["reasoning_budget"] = max(0, int(merged["reasoning_budget"]))
    return merged


def _load_reasoning_models():
    """Models that produced reasoning in any saved run - so the thinking
    allowance applies from the very first case, not after a retry."""
    found = set()
    try:
        records = storage.scan(RESULTS_DIR)[0]
    except Exception:  # noqa: BLE001 - a lookup must never fail a run
        records = []
    for r in records:
        if any(c.get("reasoning") or (c.get("reasoning_tokens") or 0) > 0
               for c in r["cases"]):
            found.add((r["provider"], r["model"]))
    return found


def model_reasons(provider, model):
    global _REASONING_MODELS
    with _REASONING_LOCK:
        if _REASONING_MODELS is None:
            _REASONING_MODELS = _load_reasoning_models()
        return (provider, model) in _REASONING_MODELS


def remember_reasoning_model(provider, model):
    model_reasons(provider, model)  # ensure loaded
    with _REASONING_LOCK:
        _REASONING_MODELS.add((provider, model))


def run_benchmark_stream(provider, model, bench, params, emit, should_cancel=None,
                         on_progress=None):
    """emit(dict) pushes one NDJSON event to the client. `on_progress(summary,
    cases)` is called after every case so the caller can checkpoint to disk.
    Returns (summary, case_results, cancelled)."""
    cases = bench.get("cases", [])
    total = len(cases)
    case_results = []
    cancelled = False
    merged = effective_params(bench, params)
    answer_budget = merged["max_tokens"]
    thinking_budget = merged["reasoning_budget"]
    # Everything needed to interpret (or re-score) this result later, even if
    # the benchmark file has since been edited or removed.
    header = {
        "type": "done",
        "benchmark_id": bench["id"],
        "benchmark_name": bench["name"],
        "category": bench.get("category"),
        "benchmark_hash": storage.benchmark_fingerprint(bench),
        "benchmark_file": bench.get("_file"),
        "scoring": bench.get("scoring"),
        "system_prompt": bench.get("system_prompt"),
        "params": merged,
        "provider": provider,
        "model": model,
        "total": total,
    }

    def summary_now():
        s = dict(header)
        s.update(storage.summarize_cases(case_results))
        s["cancelled"] = cancelled
        return s

    emit({"type": "start", "benchmark_id": bench["id"],
          "benchmark_name": bench["name"], "total": total})

    for i, case in enumerate(cases):
        if should_cancel and should_cancel():
            cancelled = True
            break
        case_id = case.get("id", f"case_{i+1}")
        emit({"type": "case_start", "benchmark_id": bench["id"], "index": i,
              "total": total, "case_id": case_id, "prompt": case["prompt"]})
        # a case may override the benchmark's scoring type
        scoring = case.get("scoring") or bench["scoring"]
        cr = {
            "id": case_id,
            "benchmark_id": bench["id"],
            "prompt": case["prompt"],
            "expected": case.get("expected"),
            "scoring": scoring,
        }
        if case.get("system_prompt"):
            cr["system_prompt"] = case["system_prompt"]

        def on_token(piece, count, kind="content"):
            emit({"type": "token", "benchmark_id": bench["id"],
                  "index": i, "piece": piece, "n": count, "kind": kind})

        try:
            messages = build_messages(bench, case)
            thinks = model_reasons(provider, model)
            budget = answer_budget + (thinking_budget if thinks else 0)
            res = chat_completion(provider, model, messages,
                                  dict(merged, max_tokens=budget),
                                  should_cancel=should_cancel, on_token=on_token)
            retried = False
            if res.get("reasoning"):
                remember_reasoning_model(provider, model)
                if (not thinks and thinking_budget > 0 and not res.get("cancelled")
                        and res.get("finish_reason") == "length"):
                    # First sign that this model thinks, and it ran out of room:
                    # it was judged on the answer-only budget. Re-ask once with
                    # the thinking allowance so the case is graded fairly.
                    budget = answer_budget + thinking_budget
                    emit({"type": "case_retry", "benchmark_id": bench["id"], "index": i,
                          "reason": "reasoning model hit the answer budget",
                          "max_tokens": budget})
                    res = chat_completion(provider, model, messages,
                                          dict(merged, max_tokens=budget),
                                          should_cancel=should_cancel, on_token=on_token)
                    retried = True
            if res.get("cancelled"):
                cancelled = True
                break
            answer = res.get("answer") or ""
            if not answer.strip() and res.get("reasoning"):
                # The model was still thinking when generation stopped, so there
                # is no answer to grade. Say so instead of reporting a mystery
                # "no number found" - the fix is almost always a bigger budget.
                ok = False
                detail = ("no answer: still reasoning when generation stopped "
                          f"(finish_reason={res.get('finish_reason')}) - "
                          "raise max_tokens")
            else:
                ok, detail = score_case(scoring, case.get("expected"),
                                        answer, case.get("prompt"))
            cr.update({
                "output": answer,
                "reasoning": res.get("reasoning") or "",
                "reasoning_tokens": res.get("reasoning_tokens"),
                "reasoning_tokens_exact": res.get("reasoning_tokens_exact"),
                "passed": ok,
                "detail": detail,
                "latency_s": res["latency_s"],
                "ttft_s": res.get("ttft_s"),
                "tokens": res["tokens"],
                "prompt_tokens": res.get("prompt_tokens"),
                "total_tokens": res.get("total_tokens"),
                "tokens_per_second": res.get("tokens_per_second"),
                "decode_s": res.get("decode_s"),
                "cost_usd": res.get("cost_usd"),
                "finish_reason": res.get("finish_reason"),
                "max_tokens": budget,
            })
            if retried:
                cr["retried_with_thinking_budget"] = True
            # keep the untouched model text when inline <think> tags were
            # stripped, so the split can be redone if the parser improves
            raw = res.get("content") or ""
            if raw.strip() != answer.strip():
                cr["raw_output"] = raw
        except Exception as e:  # noqa: BLE001
            cr.update({
                "output": "",
                "passed": False,
                "detail": f"ERROR: {e}",
                "latency_s": None,
                "tokens": None,
                "error": True,
            })
        case_results.append(cr)
        emit({"type": "case", "index": i, "total": total, "result": cr})
        if on_progress:
            try:
                on_progress(summary_now(), case_results)
            except OSError as e:
                emit({"type": "error", "message": f"checkpoint failed: {e}"})

    summary = summary_now()
    emit(summary)
    return summary, case_results, cancelled


# --------------------------------------------------------------------------- #
#  Saved runs (see storage.py for the durability rules)
# --------------------------------------------------------------------------- #
def list_results():
    records, unreadable = storage.scan(RESULTS_DIR)
    out = [{
        "id": r["id"],
        "timestamp": r["timestamp"],
        "status": r["status"],
        "schema_version": r["schema_version"],
        "provider": r["provider"],
        "model": r["model"],
        "overall_score": r["overall_score"],
        "total": r["total"],
        "answered": r["answered"],
        "passed": r["passed"],
        "cancelled": r["cancelled"],
        "total_tokens": r["total_tokens"],
        "total_time_s": r["total_time_s"],
        "avg_tokens_per_second": r["avg_tokens_per_second"],
        "total_cost_usd": r["total_cost_usd"],
        "benchmarks": [
            {"benchmark_id": b["benchmark_id"], "benchmark_name": b["benchmark_name"],
             "benchmark_hash": b["benchmark_hash"], "score": b["score"],
             "passed": b["passed"], "total": b["total"]}
            for b in r["benchmarks"]
        ],
    } for r in reversed(records)]
    return {"results": out, "unreadable": unreadable}


def _wilson(p, n, z=1.96):
    """Half-width of the 95% Wilson interval for a pass rate p over n
    questions - how much a score could move by luck of the question sample."""
    if not n:
        return None
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return round(100 * max(centre + half - p, p - (centre - half)), 1)


def build_leaderboard():
    """Consolidate every saved run into one row per (provider, model), with a
    cell per benchmark.

    For each cell the newest *clean* run wins (finished, every case answered,
    no request errors); a stopped/interrupted/errored run only fills a cell
    that has nothing better, so a failed re-run never hides a good earlier
    result. When the winning run was one of N repeats launched together, the
    cell pools all clean repeats in that group: mean score, spread (std, min,
    max) and flaky questions (passed in some repeats, failed in others).
    Cells whose benchmark has changed since the run are flagged `stale`."""
    current = [b for b in all_benchmarks() if not b.get("error")]
    bench_order = {b["id"]: i for i, b in enumerate(current)}
    bench_name = {b["id"]: b["name"] for b in current}
    bench_hash = {b["id"]: storage.benchmark_fingerprint(b) for b in current}

    results, unreadable = storage.scan(RESULTS_DIR)
    models = {}      # (provider, model) -> aggregate
    seen = {}        # bench_id -> display name

    for r in results:
        if r["status"] == "running":
            continue  # still in flight; it lands here once it has finished
        prov, model, ts, rid = r["provider"], r["model"], r["timestamp"], r["id"]
        m = models.setdefault((prov, model), {"provider": prov, "model": model, "entries": {},
                                              "last_run": ts, "run_ids": set()})
        m["run_ids"].add(rid)
        if ts > m["last_run"]:
            m["last_run"] = ts
        for b in r["benchmarks"]:
            if not b["answered"]:
                continue
            bid = b["benchmark_id"]
            seen[bid] = b["benchmark_name"]
            clean = (not b.get("cancelled") and b["answered"] >= b["total"]
                     and not b["errors"])
            h = b["benchmark_hash"]
            m["entries"].setdefault(bid, []).append({
                "score": b["score"], "passed": b["passed"], "answered": b["answered"],
                "total": b["total"], "errors": b["errors"],
                "avg_tokens_per_second": b.get("avg_tokens_per_second"),
                "total_cost_usd": b.get("total_cost_usd"),
                "total_time_s": b.get("total_time_s"),
                "total_tokens": b.get("total_tokens"),
                "timestamp": ts, "result_id": rid, "clean": clean,
                "group": r.get("repeat_group") or rid,
                "rescored": bool(r.get("rescored_at")),
                "case_results": {c["id"]: bool(c.get("passed")) for c in r["cases"]
                                 if c["benchmark_id"] == bid and not c.get("error")},
                # stale: graded against a different version of the benchmark
                # than the one on disk now (None = unknown, pre-v2 record)
                "stale": (h != bench_hash[bid]) if (h and bid in bench_hash) else None,
                "removed": bid not in bench_hash,
            })

    col_ids = sorted(seen.keys(), key=lambda b: (bench_order.get(b, 999), seen[b]))
    columns = [{"id": b, "name": bench_name.get(b, seen[b]),
                "removed": b not in bench_hash} for b in col_ids]

    def cell(entries):
        head = max(entries, key=lambda e: (e["clean"], e["timestamp"]))
        pool = ([e for e in entries if e["clean"] and e["group"] == head["group"]]
                if head["clean"] else [head])
        n = len(pool)
        scores = [e["score"] for e in pool]
        mean = round(sum(scores) / n, 1)
        std = (round(math.sqrt(sum((s - mean) ** 2 for s in scores) / (n - 1)), 1)
               if n > 1 else None)
        outcomes = {}
        for e in pool:
            for cid, ok in e["case_results"].items():
                outcomes.setdefault(cid, set()).add(ok)
        flaky = sorted(cid for cid, oks in outcomes.items() if len(oks) > 1)
        tps = [e["avg_tokens_per_second"] for e in pool if e["avg_tokens_per_second"] is not None]
        return {
            "score": mean, "runs": n, "std": std,
            "min": min(scores), "max": max(scores),
            "flaky": flaky,
            "ci95": _wilson(mean / 100.0, head["total"]),
            "passed": head["passed"], "answered": head["answered"], "total": head["total"],
            "errors": head["errors"],
            "avg_tokens_per_second": round(sum(tps) / len(tps), 1) if tps else None,
            "total_cost_usd": head["total_cost_usd"], "total_time_s": head["total_time_s"],
            "total_tokens": head["total_tokens"],
            "result_id": head["result_id"], "result_ids": [e["result_id"] for e in pool],
            "partial": not head["clean"], "stale": head["stale"], "removed": head["removed"],
            "rescored": head["rescored"],
        }

    rows = []
    for m in models.values():
        benches = {bid: cell(es) for bid, es in m["entries"].items()}
        # overall: question-weighted mean of the cells (repeats don't add weight)
        weight = sum(e["total"] or 0 for e in benches.values())
        overall = (round(sum(e["score"] * (e["total"] or 0) for e in benches.values()) / weight, 1)
                   if weight else 0.0)
        tps = [e["avg_tokens_per_second"] for e in benches.values() if e["avg_tokens_per_second"] is not None]
        costs = [e["total_cost_usd"] for e in benches.values() if e["total_cost_usd"] is not None]
        times = [e["total_time_s"] for e in benches.values() if e["total_time_s"] is not None]
        toks = [e["total_tokens"] for e in benches.values() if e["total_tokens"] is not None]
        rows.append({
            "provider": m["provider"], "model": m["model"],
            "overall_score": overall,
            "passed": sum(e["passed"] or 0 for e in benches.values()),
            "answered": sum(e["answered"] or 0 for e in benches.values()),
            "benches": benches,
            "avg_tokens_per_second": round(sum(tps) / len(tps), 1) if tps else None,
            "total_cost_usd": round(sum(costs), 6) if costs else None,
            "total_time_s": round(sum(times), 1) if times else None,
            "total_tokens": sum(toks) if toks else None,
            "benchmarks_tested": len(benches),
            "run_count": len(m["run_ids"]),
            "last_run": m["last_run"],
        })
    return {"columns": columns, "rows": rows,
            "storage": {"dir": RESULTS_DIR, "runs": len(results),
                        "schema_version": storage.SCHEMA_VERSION,
                        "unreadable": unreadable}}


def get_result(rid):
    return storage.load_record(RESULTS_DIR, rid)


# --------------------------------------------------------------------------- #
#  Run manager — runs execute in a background thread and buffer their events,
#  so a run survives the browser disconnecting (e.g. a page refresh). Clients
#  attach/re-attach to the live NDJSON event stream by run_id.
# --------------------------------------------------------------------------- #
FINISHED_RUN_TTL = 300  # keep finished runs around this long for late re-attach


def _prune_runs_locked():
    now = time.time()
    for rid in list(RUNS.keys()):
        r = RUNS[rid]
        if r.get("done") and r.get("finished_at") and now - r["finished_at"] > FINISHED_RUN_TTL:
            del RUNS[rid]


def new_run(targets, benchmark_ids, params):
    run_id = uuid.uuid4().hex[:12]
    lock = threading.Lock()
    selected = [get_benchmark(b) for b in benchmark_ids]
    total_cases = sum(len(b.get("cases", [])) for b in selected if b)
    run = {
        "id": run_id,
        "cancel": False,
        "events": [],        # buffered events, oldest first
        "seqs": [],          # parallel list: each event's sequence number
        "next_seq": 0,
        "done": False,
        "lock": lock,
        "cond": threading.Condition(lock),
        "targets": targets,
        "benchmark_ids": benchmark_ids,
        "params": params,
        "total_cases_per_model": total_cases,
        "started": time.time(),
        "finished_at": None,
    }
    with RUNS_LOCK:
        _prune_runs_locked()
        RUNS[run_id] = run
    return run


# Token events are only useful while their question is being answered: once
# the question's result arrives (or it is retried) they are dropped, so a long
# queue buffers one question's tokens instead of every token of every run.
# Events keep their sequence numbers, so streams resume correctly across drops.
_TOKEN_FLUSH_TYPES = ("case", "case_retry")


def run_emit(run, ev):
    with run["cond"]:
        if ev.get("type") in _TOKEN_FLUSH_TYPES and run["events"]:
            keep = [k for k, e in enumerate(run["events"]) if e.get("type") != "token"]
            if len(keep) != len(run["events"]):
                run["events"] = [run["events"][k] for k in keep]
                run["seqs"] = [run["seqs"][k] for k in keep]
        run["events"].append(ev)
        run["seqs"].append(run["next_seq"])
        run["next_seq"] += 1
        run["cond"].notify_all()


def iter_run_events(run, since=0, wait=1.0):
    """Yield batches of a run's buffered events with sequence number >= since,
    following the run until it is done. A reader that falls behind simply
    skips token events that were dropped meanwhile."""
    nxt = since
    while True:
        with run["cond"]:
            while run["next_seq"] <= nxt and not run["done"]:
                run["cond"].wait(timeout=wait)
            start = bisect.bisect_left(run["seqs"], nxt)
            pending = run["events"][start:]
            nxt = run["next_seq"]
            done = run["done"]
        if pending:
            yield pending
        if done:
            return


def run_worker(run):
    """Execute every (provider, model) target in the run sequentially, saving
    each separately and buffering all events. Runs in its own thread."""
    run_id = run["id"]
    benchmark_ids = run["benchmark_ids"]
    params = run["params"]
    targets = run["targets"]
    total_cases = run["total_cases_per_model"]

    def should_cancel():
        return run["cancel"]

    try:
        run_emit(run, {
            "type": "run_started", "run_id": run_id,
            "targets": [{"provider": t.get("provider", "local"), "model": t.get("model")} for t in targets],
            "num_targets": len(targets), "total_cases_per_model": total_cases,
        })
        cancelled = False
        for idx, target in enumerate(targets):
            if should_cancel():
                cancelled = True
                break
            provider = target.get("provider", "local")
            model = target.get("model")
            reps = int(target.get("repeats") or 1)
            group = {"repeat_group": f"{run_id}-{target.get('group', idx)}",
                     "repeat_index": int(target.get("repeat") or 1),
                     "repeat_count": reps} if reps > 1 else {}
            run_emit(run, {"type": "target_start", "index": idx, "total": len(targets),
                           "provider": provider, "model": model, "total_cases": total_cases,
                           "repeat": group.get("repeat_index"), "repeats": reps})
            summaries, all_cases, tcancel = [], [], False
            rid, started_at = storage.new_result_id(), storage.now_iso()

            def write(status, partial_summary=None, partial_cases=()):
                extra = [partial_summary] if partial_summary else []
                rec = storage.build_record(
                    rid, started_at, provider, model, params,
                    summaries + extra, all_cases + list(partial_cases), status,
                    extra=group)
                storage.write_record(RESULTS_DIR, rec)
                return rec

            storage.mark_active(rid)
            try:
                for bid in benchmark_ids:
                    if should_cancel():
                        tcancel = True
                        break
                    bench = get_benchmark(bid)
                    if bench is None or bench.get("error"):
                        run_emit(run, {"type": "error", "message": f"benchmark {bid} not found"})
                        continue
                    # checkpoint after every case: a crash keeps what finished
                    summary, cases, was_cancelled = run_benchmark_stream(
                        provider, model, bench, params, lambda ev: run_emit(run, ev),
                        should_cancel,
                        on_progress=lambda ps, pc: write("running", ps, pc))
                    summaries.append(summary)
                    all_cases.extend(cases)
                    if was_cancelled:
                        tcancel = True
                        break
                record = write("cancelled" if tcancel else "complete")
            finally:
                storage.mark_active(rid, False)
            run_emit(run, {"type": "saved", "index": idx, "result_id": record["id"],
                           "provider": provider, "model": model,
                           "overall_score": record["overall_score"], "passed": record["passed"],
                           "answered": record["answered"], "total": record["total"],
                           "cancelled": tcancel, "total_tokens": record["total_tokens"],
                           "total_time_s": record["total_time_s"],
                           "avg_tokens_per_second": record["avg_tokens_per_second"],
                           "total_cost_usd": record["total_cost_usd"]})
            if tcancel:
                cancelled = True
                break
        run_emit(run, {"type": "queue_done", "run_id": run_id, "cancelled": cancelled})
    except Exception as e:  # noqa: BLE001
        run_emit(run, {"type": "fatal", "message": str(e)})
    finally:
        with run["cond"]:
            run["done"] = True
            run["finished_at"] = time.time()
            run["cond"].notify_all()


def start_run(targets, benchmark_ids, params):
    run = new_run(targets, benchmark_ids, params)
    threading.Thread(target=run_worker, args=(run,), daemon=True).start()
    return run


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

        if path in ("/api/config", "/api/settings"):
            self._send_json({
                "lmstudio_base_url": CONFIG.get("lmstudio_base_url", ""),
                "openrouter_configured": bool(CONFIG.get("openrouter_api_key")),
                "default_params": CONFIG.get("default_params", {}),
                "reasoning_budget": CONFIG.get("reasoning_budget", DEFAULT_REASONING_BUDGET),
            })
            return
        if path == "/api/models":
            provider = (parse_qs(parsed.query).get("provider") or ["local"])[0]
            self._send_json(list_models(provider))
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
        if path == "/api/runs":
            # active (not yet finished) runs, so a refreshed page can re-attach
            with RUNS_LOCK:
                _prune_runs_locked()
                runs = [{
                    "run_id": r["id"], "done": r["done"],
                    "targets": [{"provider": t.get("provider", "local"), "model": t.get("model")}
                                for t in r["targets"]],
                    "num_targets": len(r["targets"]),
                    "events_count": r["next_seq"],
                    "started": r["started"],
                } for r in RUNS.values() if not r["done"]]
            runs.sort(key=lambda x: x["started"], reverse=True)
            self._send_json({"runs": runs})
            return
        if path == "/api/run/stream":
            qs = parse_qs(parsed.query)
            rid = (qs.get("run_id") or [""])[0]
            since = (qs.get("since") or ["0"])[0]
            try:
                since = max(0, int(since))
            except ValueError:
                since = 0
            with RUNS_LOCK:
                run = RUNS.get(rid)
            if run is None:
                self._send_json({"error": "run not found or expired"}, 404)
            else:
                self._stream_run(run, since=since)
            return
        if path == "/api/leaderboard":
            self._send_json(build_leaderboard())
            return
        if path == "/api/results":
            self._send_json(list_results())
            return
        if path == "/api/export":
            body = storage.export_zip(RESULTS_DIR)
            name = "LocalLLMArena-results-" + time.strftime("%Y%m%d-%H%M%S") + ".zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path.startswith("/api/results/"):
            rid = path.rsplit("/", 1)[-1]
            try:
                r = get_result(rid)
            except Exception as e:  # noqa: BLE001
                self._send_json({"error": f"unreadable result file: {e}"}, 500)
                return
            if r is None:
                self._send_json({"error": "not found"}, 404)
            else:
                self._send_json(r)
            return

        # static files
        self._send_static(path)

    def do_POST(self):
        global CONFIG, RESULTS_DIR, _REASONING_MODELS
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/run":
            self._handle_run()
            return
        if path == "/api/rescore":
            import rescore
            body = self._read_body()
            self._send_json(rescore.rescore_all(RESULTS_DIR, apply=bool(body.get("apply"))))
            return
        if path == "/api/run/stop":
            body = self._read_body()
            rid = body.get("run_id")
            with RUNS_LOCK:
                run = RUNS.get(rid)
                if run is not None:
                    run["cancel"] = True
            self._send_json({"ok": run is not None, "run_id": rid})
            return
        if path in ("/api/settings", "/api/reload-config"):
            body = self._read_body()
            updates = {}
            if "lmstudio_base_url" in body:
                updates["lmstudio_base_url"] = (body.get("lmstudio_base_url") or "").strip()
            # Only overwrite the API key when a non-empty value is supplied, so
            # saving other settings doesn't wipe a previously stored key.
            if body.get("openrouter_api_key"):
                updates["openrouter_api_key"] = body["openrouter_api_key"].strip()
            if body.get("clear_openrouter_api_key"):
                updates["openrouter_api_key"] = ""
            if updates:
                save_local_settings(updates)
            CONFIG = load_config()
            RESULTS_DIR = resolve_results_dir()
            _REASONING_MODELS = None
            self._send_json({
                "ok": True,
                "lmstudio_base_url": CONFIG.get("lmstudio_base_url", ""),
                "openrouter_configured": bool(CONFIG.get("openrouter_api_key")),
            })
            return

        self.send_error(404, "Not found")

    # -- streaming run --------------------------------------------------- #
    def _handle_run(self):
        body = self._read_body()
        targets = body.get("targets")
        if not targets:  # legacy single-model form
            model = body.get("model")
            if model:
                targets = [{"provider": body.get("provider", "local"), "model": model}]
        targets = [t for t in (targets or []) if t.get("model")]
        bench_ids = body.get("benchmark_ids", [])
        params = dict(body.get("params") or {})
        try:
            repeats = max(1, min(20, int(params.pop("repeats", None) or body.get("repeats") or 1)))
        except (TypeError, ValueError):
            repeats = 1
        if repeats > 1:
            # each repeat is its own saved run; the leaderboard pools the group
            targets = [dict(t, group=g, repeat=k + 1, repeats=repeats)
                       for g, t in enumerate(targets) for k in range(repeats)]
        if not targets or not bench_ids:
            self._send_json({"error": "targets (or model) and benchmark_ids are required"}, 400)
            return
        run = start_run(targets, bench_ids, params)
        self._stream_run(run, since=0)

    def _stream_run(self, run, since=0):
        """Stream a run's buffered NDJSON events from sequence `since`, following
        until the run is done. Safe to call repeatedly (re-attach after refresh).
        If the client disconnects, the run keeps going in its worker thread."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for pending in iter_run_events(run, since):
                for ev in pending:
                    self.wfile.write((json.dumps(ev, ensure_ascii=False) + "\n").encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return  # client navigated away; worker thread continues


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    port = int(os.environ.get("PORT") or CONFIG.get("server_port", 8080))
    # Loopback only by default: the app has no auth and can spend OpenRouter
    # credit. Set "server_host": "0.0.0.0" (or HOST) to expose it on the LAN.
    host = os.environ.get("HOST") or CONFIG.get("server_host") or "127.0.0.1"
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://localhost:{port}"
    print("=" * 64)
    print("  LocalLLMArena is running")
    print(f"  Open:        {url}")
    print(f"  LM Studio:   {CONFIG['lmstudio_base_url']}")
    print(f"  Benchmarks:  {BENCH_DIR}")
    print(f"  Results:     {RESULTS_DIR}")
    print("  Stop:        Ctrl+C")
    print("=" * 64)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
