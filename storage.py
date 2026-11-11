"""Durable storage for benchmark runs.

One JSON file per run, in RESULTS_DIR. The rules that keep old data safe while
the app and the benchmark sets keep changing:

  1. Records are append-only. A finished run file is never rewritten, migrated
     in place, or deleted by the app.
  2. Every write is atomic (temp file + fsync + rename), so a crash can never
     leave a half-written file where a good one used to be.
  3. Records are self-describing: each one carries a schema version, the app
     version, the effective parameters, and for every benchmark a fingerprint
     plus the prompt/expected/scoring of every case. A run stays readable and
     re-scorable even after its benchmark file is edited or deleted.
  4. Reading is tolerant. `upgrade_record` lifts any older shape to the current
     one *in memory*; missing fields get defaults instead of raising. Files
     that cannot be parsed are reported, never silently dropped.
  5. Runs are checkpointed after every case, so killing the server mid-run
     keeps everything completed so far (shown as "interrupted").
  6. Re-scoring never edits a run. New verdicts go to an overlay file in
     RESULTS_DIR/rescores/<id>.json that is applied on read; the original
     verdict stays in the run file (and is shown as `passed_recorded`).
     Deleting the overlay restores the original scores exactly.

When the record shape changes: bump SCHEMA_VERSION, teach `upgrade_record` the
old shape, and add a fixture of the old shape under tests/fixtures/ - the test
suite loads every fixture through the full read path.
"""
import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone

SCHEMA_VERSION = 2

ROOT = os.path.dirname(os.path.abspath(__file__))
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Result ids being written by a live run in this process. A record that says
# "running" but is not in here was cut short by a crash/kill -> "interrupted".
ACTIVE_IDS = set()
_ACTIVE_LOCK = threading.Lock()


def mark_active(rid, active=True):
    with _ACTIVE_LOCK:
        (ACTIVE_IDS.add if active else ACTIVE_IDS.discard)(rid)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def new_result_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


_APP_VERSION = None


def app_version():
    """Short git commit of the code that produced a record (best effort)."""
    global _APP_VERSION
    if _APP_VERSION is None:
        try:
            out = subprocess.run(
                ["git", "-C", ROOT, "describe", "--always", "--dirty"],
                capture_output=True, text=True, timeout=5)
            _APP_VERSION = out.stdout.strip() or "unknown"
        except Exception:  # noqa: BLE001
            _APP_VERSION = "unknown"
    return _APP_VERSION


# --------------------------------------------------------------------------- #
#  Benchmark fingerprint
# --------------------------------------------------------------------------- #
def benchmark_fingerprint(bench):
    """Hash of everything that affects what a score *means*: the prompts, the
    expected answers, the scoring rules and the system prompt. Cosmetic fields
    (name, description, category) are excluded, so renaming a benchmark does
    not make its old results look stale - editing a case does."""
    core = {
        "scoring": bench.get("scoring"),
        "system_prompt": bench.get("system_prompt"),
        "cases": [{
            "id": c.get("id", f"case_{i + 1}"),
            "prompt": c.get("prompt"),
            "expected": c.get("expected"),
            "scoring": c.get("scoring"),
            "system_prompt": c.get("system_prompt"),
        } for i, c in enumerate(bench.get("cases", []))],
    }
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


# --------------------------------------------------------------------------- #
#  Writing
# --------------------------------------------------------------------------- #
def atomic_write_json(path, obj):
    """Write JSON so that `path` always holds either the old or the new
    complete content - never a truncated mix."""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-", suffix=".part")
    try:
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; match a normal file
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def result_path(results_dir, rid):
    if not rid or not _ID_RE.match(rid):
        return None
    return os.path.join(results_dir, rid + ".json")


def write_record(results_dir, record):
    path = result_path(results_dir, record.get("id"))
    if path is None:
        raise ValueError(f"bad result id {record.get('id')!r}")
    atomic_write_json(path, record)
    return path


# --------------------------------------------------------------------------- #
#  Building records from case results
# --------------------------------------------------------------------------- #
def _avg(xs, nd=3):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), nd) if xs else None


def decode_tps(tokens, decode_s):
    """Generation speed, or None when the decode window is too short to
    measure (a few tokens, or output that arrived in one buffered burst)."""
    if not tokens or tokens < 4 or decode_s is None or decode_s <= 0.15:
        return None
    return tokens / decode_s


def _case_decode(c):
    d = c.get("decode_s")
    if d is None and c.get("latency_s") is not None and c.get("ttft_s") is not None:
        d = c["latency_s"] - c["ttft_s"]      # older records: derive it
    return d


def speed_of(cases):
    """Token-weighted generation speed over the cases where it is measurable:
    total decode tokens / total decode time. Unlike a mean of per-case rates,
    one tiny answer can't drag it around."""
    toks = secs = 0.0
    for c in cases:
        d = _case_decode(c)
        if decode_tps(c.get("tokens"), d) is not None:
            toks += c["tokens"]
            secs += d
    return round(toks / secs, 1) if secs else None


def summarize_cases(cases):
    """Aggregate metrics for a list of case results. Pure function of the
    cases, so a summary can always be recomputed from a stored record."""
    answered = len(cases)
    passed = sum(1 for c in cases if c.get("passed"))
    errors = sum(1 for c in cases if c.get("error"))
    latencies = [c.get("latency_s") for c in cases if c.get("latency_s") is not None]
    comp = sum(c.get("tokens") or 0 for c in cases)
    prompt = sum(c.get("prompt_tokens") or 0 for c in cases)
    costs = [c["cost_usd"] for c in cases if c.get("cost_usd") is not None]
    return {
        "answered": answered,
        "passed": passed,
        "errors": errors,
        "score": round(100.0 * passed / answered, 1) if answered else 0.0,
        "avg_latency_s": _avg(latencies),
        "avg_ttft_s": _avg([c.get("ttft_s") for c in cases]),
        "avg_tokens_per_second": speed_of(cases),
        "total_completion_tokens": comp,
        "total_prompt_tokens": prompt,
        "total_reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in cases),
        "total_tokens": comp + prompt,
        "total_time_s": round(sum(latencies), 3) if latencies else 0.0,
        "total_cost_usd": round(sum(costs), 6) if costs else None,
    }


def build_record(rid, started_at, provider, model, params, summaries, cases, status,
                 extra=None):
    total = sum(s.get("total") or 0 for s in summaries)
    answered = sum(s.get("answered", s.get("total") or 0) for s in summaries)
    passed = sum(s.get("passed") or 0 for s in summaries)
    comp = sum(s.get("total_completion_tokens") or 0 for s in summaries)
    prompt = sum(s.get("total_prompt_tokens") or 0 for s in summaries)
    costs = [s.get("total_cost_usd") for s in summaries if s.get("total_cost_usd") is not None]
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": app_version(),
        "id": rid,
        "timestamp": started_at,
        "updated_at": now_iso(),
        "status": status,                     # running | complete | cancelled
        "provider": provider,
        "model": model,
        "params": params,
        "cancelled": status == "cancelled",
        "overall_score": round(100.0 * passed / answered, 1) if answered else 0.0,
        "total": total,
        "answered": answered,
        "passed": passed,
        "errors": sum(s.get("errors") or 0 for s in summaries),
        "total_completion_tokens": comp,
        "total_prompt_tokens": prompt,
        "total_reasoning_tokens": sum(s.get("total_reasoning_tokens") or 0 for s in summaries),
        "total_tokens": comp + prompt,
        "total_time_s": round(sum(s.get("total_time_s") or 0 for s in summaries), 3),
        "avg_tokens_per_second": speed_of(cases),
        "total_cost_usd": round(sum(costs), 6) if costs else None,
        **(extra or {}),
        "benchmarks": summaries,
        "cases": cases,
    }


# --------------------------------------------------------------------------- #
#  Reading (tolerant, upgrades old shapes in memory only)
# --------------------------------------------------------------------------- #
def upgrade_record(r, fallback_id=None):
    """Return `r` lifted to the current schema. Never raises on missing fields
    and never touches the file on disk."""
    if not isinstance(r, dict):
        raise ValueError("record is not a JSON object")
    r = dict(r)
    r.setdefault("schema_version", 1)
    r.setdefault("id", fallback_id)
    r.setdefault("timestamp", "")
    r.setdefault("provider", "local")
    r.setdefault("model", "?")
    r.setdefault("params", {})
    r.setdefault("app_version", None)
    cases = [dict(c) for c in (r.get("cases") or []) if isinstance(c, dict)]
    benches = [dict(b) for b in (r.get("benchmarks") or []) if isinstance(b, dict)]
    r["cases"], r["benchmarks"] = cases, benches

    for b in benches:
        b.setdefault("benchmark_id", b.get("benchmark_name") or "?")
        b.setdefault("benchmark_name", b["benchmark_id"])
        b.setdefault("benchmark_hash", None)      # unknown for v1 records
        b.setdefault("total", b.get("answered") or 0)
        b.setdefault("answered", b["total"])
        b.setdefault("passed", 0)
        b.setdefault("errors", 0)
        if b.get("score") is None:
            b["score"] = round(100.0 * b["passed"] / b["answered"], 1) if b["answered"] else 0.0
    for c in cases:
        c.setdefault("id", "?")
        c.setdefault("benchmark_id", benches[0]["benchmark_id"] if len(benches) == 1 else "?")
        c.setdefault("passed", False)
        c.setdefault("output", "")
        c.setdefault("reasoning", "")           # pre-reasoning-support runs
        c.setdefault("raw_output", c["output"])
        c.setdefault("detail", "")

    # Speed is a derived display figure, not a measurement, so recompute it
    # with the current definition (older versions mixed load time into it).
    for b in benches:
        bc = [c for c in cases if c["benchmark_id"] == b["benchmark_id"]]
        if bc:
            b["avg_tokens_per_second"] = speed_of(bc)
    if cases:
        r["avg_tokens_per_second"] = speed_of(cases)

    r.setdefault("total", sum(b["total"] for b in benches))
    r.setdefault("answered", sum(b["answered"] for b in benches))
    r.setdefault("passed", sum(b["passed"] for b in benches))
    r.setdefault("errors", sum(b["errors"] for b in benches))
    if r.get("overall_score") is None:
        r["overall_score"] = round(100.0 * r["passed"] / r["answered"], 1) if r["answered"] else 0.0
    for k in ("total_tokens", "total_time_s", "avg_tokens_per_second", "total_cost_usd"):
        r.setdefault(k, None)

    status = r.get("status") or ("cancelled" if r.get("cancelled") else "complete")
    if status == "running" and r["id"] not in ACTIVE_IDS:
        status = "interrupted"
    r["status"] = status
    r["cancelled"] = status in ("cancelled", "interrupted")
    return r


# --------------------------------------------------------------------------- #
#  Re-score overlays
# --------------------------------------------------------------------------- #
OVERLAY_SUBDIR = "rescores"


def overlay_path(results_dir, rid):
    p = result_path(results_dir, rid)
    return p and os.path.join(results_dir, OVERLAY_SUBDIR, rid + ".json")


def load_overlay(results_dir, rid):
    p = overlay_path(results_dir, rid)
    if not p or not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None   # a broken overlay just means "show the original scores"


def write_overlay(results_dir, overlay):
    atomic_write_json(overlay_path(results_dir, overlay["run_id"]), overlay)


def apply_overlay(r, ov):
    """Apply re-scored verdicts to an upgraded record (in memory) and
    recompute the pass counts and scores that depend on them."""
    if not ov or not isinstance(ov.get("cases"), list):
        return r
    cases = r["cases"]
    for e in ov["cases"]:
        i = e.get("i")
        if not isinstance(i, int) or not 0 <= i < len(cases):
            continue
        c = cases[i]
        if c.get("id") != e.get("id"):
            continue          # overlay does not match this record; ignore it
        c["passed_recorded"] = c.get("passed")
        c["detail_recorded"] = c.get("detail")
        for k in ("passed", "detail", "expected", "scoring"):
            if k in e:
                c[k] = e[k]
        c["rescored"] = True
    for b in r["benchmarks"]:
        bc = [c for c in cases if c["benchmark_id"] == b["benchmark_id"]]
        if bc:
            b["passed_recorded"] = b["passed"]
            b["passed"] = sum(1 for c in bc if c.get("passed"))
            b["score"] = round(100.0 * b["passed"] / b["answered"], 1) if b["answered"] else 0.0
        new_hash = (ov.get("benchmark_hashes") or {}).get(b["benchmark_id"])
        if new_hash:
            b["benchmark_hash_recorded"] = b["benchmark_hash"]
            b["benchmark_hash"] = new_hash
    r["passed_recorded"] = r["passed"]
    r["passed"] = sum(b["passed"] for b in r["benchmarks"])
    r["overall_score"] = round(100.0 * r["passed"] / r["answered"], 1) if r["answered"] else 0.0
    r["rescored_at"] = ov.get("rescored_at")
    return r


def load_record(results_dir, rid, overlay=True):
    path = result_path(results_dir, rid)
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        r = upgrade_record(json.load(f), fallback_id=rid)
    return apply_overlay(r, load_overlay(results_dir, rid)) if overlay else r


def scan(results_dir, overlay=True):
    """Read every run. Returns (records sorted oldest first, unreadable) where
    unreadable is a list of {"file", "error"} - surfaced in the UI so a damaged
    file is noticed instead of quietly vanishing from the leaderboard."""
    records, unreadable = [], []
    if not os.path.isdir(results_dir):
        return records, unreadable
    try:
        names = sorted(os.listdir(results_dir))
    except OSError as e:
        # e.g. macOS privacy protection on ~/Documents or ~/Desktop
        unreadable.append({"file": results_dir, "error": f"cannot list folder: {e}"})
        return records, unreadable
    for fn in names:
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        try:
            with open(os.path.join(results_dir, fn), "r", encoding="utf-8") as f:
                r = upgrade_record(json.load(f), fallback_id=fn[:-5])
            if overlay:
                r = apply_overlay(r, load_overlay(results_dir, r["id"]))
            records.append(r)
        except Exception as e:  # noqa: BLE001
            unreadable.append({"file": fn, "error": f"{type(e).__name__}: {e}"})
    records.sort(key=lambda r: (r.get("timestamp") or "", r.get("id") or ""))
    return records, unreadable


def export_zip(results_dir):
    """All run files as one zip (bytes) - a one-click backup."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if os.path.isdir(results_dir):
            for fn in sorted(os.listdir(results_dir)):
                if fn.endswith(".json") and not fn.startswith("."):
                    z.write(os.path.join(results_dir, fn), arcname=fn)
            od = os.path.join(results_dir, OVERLAY_SUBDIR)
            if os.path.isdir(od):
                for fn in sorted(os.listdir(od)):
                    if fn.endswith(".json") and not fn.startswith("."):
                        z.write(os.path.join(od, fn), arcname=OVERLAY_SUBDIR + "/" + fn)
    return buf.getvalue()
