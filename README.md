# 🏟️ LocalLLMArena

A small, **zero-dependency** local web app for benchmarking and comparing LLMs.
Run models **locally** via [LM Studio](https://lmstudio.ai/) and/or in the cloud
via [OpenRouter](https://openrouter.ai/), then compare them side by side on a
leaderboard by **score, speed, and price**. It ships with a handful of
public-style benchmarks (math, knowledge, reasoning, coding,
instruction-following) and lets you drop in your own tests as plain JSON files.

No `pip install`, no Node, no build step — just Python 3 standard library.

---

## Quick start

1. In **LM Studio**, load a model and start the local server (the "Developer" →
   "Start Server" tab). It exposes an OpenAI-compatible API.

2. Run the app:

   ```bash
   python3 server.py
   ```

3. Open the printed URL (default <http://localhost:8080>) in your browser.

4. Open the **Settings** tab and set your **LM Studio base URL** (e.g.
   `http://localhost:1234`). Optionally paste an **OpenRouter API key** to also
   test cloud models. These are saved to `settings.local.json`, which is
   **gitignored** — your URL and key never end up in the repository.

5. Pick a **provider** (LM Studio or OpenRouter) and a **model** (the OpenRouter
   list is searchable and shows per-token pricing), tick the benchmarks you
   want, and click **Run selected**. Results stream in live and are saved for
   the **Leaderboard** tab. Hit **■ Stop** any time to cancel mid-run (the
   partial result is saved).

### Live view & metrics

Each question is generated via the streaming API. A focal **"now generating"**
panel shows the prompt going **in** and the answer streaming **out token by
token**, while a live stats strip shows elapsed time, cases done, pass count,
running token total, average speed, and (for OpenRouter) running cost.

Detailed metrics are captured per case and aggregated per benchmark and per run:

- **Tokens** — prompt, completion, and total token counts (from the model's usage report)
- **Speed** — tokens/second (measured over the decode window, with a fall-back
  for very short outputs)
- **Time to first token (TTFT)** and **total latency** per question
- **Cost (USD)** for OpenRouter runs (exact, reported by OpenRouter); local runs are free
- **Total wall-clock time** and **average speed** for the whole run
- **Finish reason** (e.g. `stop`, `length`)

Click any question row to inspect its full prompt, expected answer, model
output, and all metrics.

### Queue several models

Use **➕ Add to queue** to line up multiple models — any mix of local (LM Studio)
and OpenRouter — and benchmark them one after another in a single click. The
selected benchmarks and parameters apply to every model in the queue, each model
runs in its own live section, and each is saved as its own leaderboard row.
**■ Stop** halts the whole queue (the current model finishes its current question,
remaining models are skipped). If the queue is empty, the Run button just runs
the single model currently selected.

### Runs survive a page refresh

Runs (and queues) execute in a **background thread on the server** and buffer
their events, so they are **not** tied to the browser tab. If you refresh the
page — or close it and come back — the app detects the run still in progress via
`GET /api/runs` and **re-attaches** to its live event stream, rebuilding the UI
(all sections, rows, and the live token view) and continuing from where it is.
Closing the tab no longer kills the run.

### Comparing models (leaderboard)

The **Leaderboard** tab is a **consolidated matrix**: one row per model with a
column for every benchmark, plus overall score, speed (tok/s), cost and total
time. Runs are merged per model — if you run a model more than once (or run
different benchmark subsets at different times), the **most recent run wins per
benchmark**, so re-running just refreshes that cell. **Click any column header to
sort** (by overall, by a single benchmark, by speed, cost, …); **click a
benchmark cell** to open that run's full case-by-case breakdown. Queue a local
model and an OpenRouter model together to compare quality vs. speed vs. price
directly.

### Configuration

The committed [`config.json`](config.json) holds only generic defaults (it points
at `http://localhost:1234`). Your machine-specific **LM Studio URL** and
**OpenRouter API key** live in `settings.local.json` (gitignored), edited from
the **Settings** tab or via `POST /api/settings`. This keeps secrets and private
addresses out of the public repo.

---

## What's included

**Core set** (good for ranging weaker→mid models; strong 27B models largely saturate these):

| Benchmark | Category | Scoring | What it tests |
|-----------|----------|---------|---------------|
| Grade-School Math (GSM8K style) | Math & Reasoning | `numeric` | Multi-step arithmetic word problems |
| MMLU (Mixed Subjects) | Knowledge | `multiple_choice` | Academic knowledge across subjects |
| ARC (Science Reasoning) | Knowledge | `multiple_choice` | Grade-school science reasoning |
| Commonsense Reasoning | Reasoning | `multiple_choice` | Everyday commonsense |
| TruthfulQA (Misconceptions) | Reasoning | `multiple_choice` | Resisting common myths |
| Code Output Prediction | Coding | `contains` | Tracing short Python snippets |
| Instruction Following | Instruction Following | `regex` | Obeying precise format constraints |
| Hidden-Premise Reasoning | Reasoning | `multiple_choice` | Purpose-over-heuristic traps (e.g. car-wash) |
| Custom Example — World Capitals | Custom | `contains` | Template you can copy |

**Hard set** (added because strong models saturate the core set — these stay discriminating):

| Benchmark | Category | Scoring | What it tests |
|-----------|----------|---------|---------------|
| Hard Math (Competition / Number Theory) | Math & Reasoning | `numeric` | Modular arithmetic, combinatorics, number theory |
| Hard Reasoning (Traps & Counterintuition) | Reasoning | `multiple_choice` | CRT, Monty Hall, knights-and-knaves, conjunction fallacy |
| Graduate Science (GPQA-style) | Knowledge | `multiple_choice` | Upper-undergrad / grad physics, chem, bio, stats |
| Character & Token Traps | Reasoning | `numeric` + `multiple_choice` | Letter counting / spelling (the classic tokenizer blind spot) |
| Hard Instruction Following | Instruction Following | `regex_cs` | Strict compound output-format constraints |

The hard set was designed after observing three 26–27B models score **88–99%** on
the core set, with discrimination surviving only in math, trap-reasoning, and
instruction-following — so the hard set leans into exactly those, plus the
character-level tasks LLMs notoriously fail. Answer keys for the objective
benchmarks (math, character traps) are verified programmatically.

These are **representative samples** written in the style of the well-known public
benchmarks, sized to run quickly against a local model. To run the *full* official
datasets, export them into the same JSON format (see below) — the runner doesn't
care how many cases a file has.

---

## Adding your own benchmark

Drop a new `.json` file into the [`benchmarks/`](benchmarks/) folder and click
**Refresh** (or reload the page). The format:

```json
{
  "id": "my_benchmark",
  "name": "My Benchmark",
  "category": "Custom",
  "description": "What this measures.",
  "scoring": "multiple_choice",
  "system_prompt": "Optional system prompt sent before every question.",
  "default_params": { "temperature": 0, "max_tokens": 512 },
  "cases": [
    { "id": "q1", "prompt": "Question text...", "expected": "B" }
  ]
}
```

- `id` is optional (defaults to the filename).
- `system_prompt` and `default_params` are optional.
- Each case may also carry its own `system_prompt` to override the benchmark's.
- Each case may also carry its own `scoring` to override the benchmark's, so one
  file can mix scoring types (e.g. `numeric` counting questions plus
  `multiple_choice` ones, as in **Character & Token Traps**).

### Scoring types

| `scoring` | `expected` is… | Passes when… |
|-----------|----------------|--------------|
| `multiple_choice` | a letter `"A"`–`"J"` | the letter extracted from the answer matches (constrained to the options the prompt offers) |
| `numeric` | a number (string or number) | the final number in the answer equals it (tiny tolerance) |
| `exact_match` | a string | the whole answer matches (case/space-insensitive) |
| `contains` | a string **or list of strings** | the answer contains it (any of, if a list) |
| `contains_all` | a list of strings | the answer contains every string |
| `regex` | a regex pattern | the pattern matches the raw answer (**case-insensitive**, dotall) |
| `regex_cs` | a regex pattern | the pattern matches the raw answer (**case-sensitive**, dotall) — for grading case/format constraints |

For `multiple_choice` and `numeric`, the runner adds a light default system
prompt nudging the model to end with `Answer: <…>`, which makes extraction
reliable. You can override it with your own `system_prompt`.

---

## How it works

```
server.py            Pure-stdlib HTTP server. Serves the web UI and a small JSON
                     API, calls the provider's /v1/chat/completions (streaming),
                     scores each answer, computes metrics/cost, and streams
                     results back as newline-delimited JSON (NDJSON).
benchmarks/*.json    One file per benchmark (see format above).
results/*.json       One file per run, powering the Leaderboard tab.
web/                 index.html + app.js + styles.css — no framework, no build.
config.json          Generic defaults (committed). Points at localhost.
settings.local.json  Your LM Studio URL + OpenRouter key (gitignored, never committed).
```

Two providers are supported, both OpenAI-compatible: **local** (LM Studio or any
local server) and **openrouter** (cloud). The token stream is parsed for live
tokens, usage, and — for OpenRouter — exact cost (`usage.include`).

API endpoints (handy if you want to script runs):

- `GET  /api/models?provider=local|openrouter` — available models (OpenRouter incl. pricing)
- `GET  /api/settings` / `POST /api/settings` — read / save the LM Studio URL & OpenRouter key
- `GET  /api/benchmarks` — list benchmark metadata
- `GET  /api/benchmarks/{id}` — full benchmark incl. cases
- `POST /api/run` — `{ "targets": [{provider, model}, ...], "benchmark_ids": [...], "params": {...} }`
  starts a background run (queue) and streams its NDJSON events. Events: `run_started`,
  `target_start` (per model), `start`/`case_start`/`token`/`case`/`done` (per benchmark),
  `saved` (per model), `queue_done`. Still accepts the legacy `{ provider, model }` form.
- `GET  /api/runs` — runs currently in progress (so a refreshed page can re-attach)
- `GET  /api/run/stream?run_id=…&since=N` — (re)attach to a run's event stream from index N
- `POST /api/run/stop` — `{ "run_id" }` cancels an in-flight run/queue
- `GET  /api/leaderboard` — runs consolidated into one row per model, with a per-benchmark column matrix
- `GET  /api/results` / `GET /api/results/{id}` — saved runs

---

## Notes & limitations

- **Sample sizes are small** so a run finishes in a minute or two on a local
  model. Scores are indicative, not leaderboard-grade — swap in full datasets for
  rigorous numbers.
- Scoring for open-ended tasks (`contains`/`regex`) is heuristic. The multiple-choice
  and numeric benchmarks are the most robust.
- `temperature: 0` is recommended for reproducible benchmarking.
- **OpenRouter costs real money.** Cost shown is what OpenRouter reports for the
  run. Watch `max_tokens` and benchmark sizes when testing expensive models.
- The server binds to `0.0.0.0` so you can reach it from other devices on your
  LAN; it has no authentication, so only run it on a trusted network. Your
  OpenRouter key is stored in plain text in `settings.local.json` (gitignored).
