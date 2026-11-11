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
- **Speed** — generation speed in tokens/second, measured from the first token
  to the last. Model loading and prompt processing are excluded (they show up
  in TTFT). Answers of only a few tokens are too short to time and show "—";
  benchmark/model speeds are token-weighted over the measurable answers.
- **Time to first token (TTFT)** and **total latency** per question
- **Cost (USD)** for OpenRouter runs (exact, reported by OpenRouter); local runs are free
- **Total wall-clock time** and **average speed** for the whole run
- **Finish reason** (e.g. `stop`, `length`)

Click any question row to inspect its full prompt, expected answer, model
output, and all metrics.

### Reasoning models

Reasoning models (Qwen3, DeepSeek-R1, gpt-oss, Gemma "thinking" builds, the
OpenRouter `:thinking` variants …) emit a chain-of-thought before their actual
answer. The app separates the two, and **only the answer is ever scored**:

- **Both transports are handled.** Thinking arrives either on a dedicated stream
  field (`reasoning_content` in LM Studio, `reasoning` on OpenRouter) or inline
  in the content wrapped in `<think>…</think>` tags. Tag variants (`<thinking>`,
  `<reasoning>`, `<reflection>`, `<scratchpad>`, any case) are recognised, as is
  the common chat-template quirk of pre-filling the opening tag so only the
  closing tag appears in the reply.
- **Scoring sees the answer only.** This matters: thinking is full of restated
  options, rejected candidate numbers and the phrase "the answer is …". Grading
  the raw text lets a guess the model *talked itself out of* win, and can fail a
  case whose format constraint (e.g. "reply in UPPERCASE") the answer satisfies
  perfectly.
- **Live view shows the thinking.** A 🧠 *Thinking* pane streams the chain-of-
  thought above the answer pane, so a long thinking phase looks like progress
  rather than a frozen panel.
- **TTFT is honest.** Time-to-first-token is measured from the first token of
  *any* kind. Timing it from the first answer token would silently fold the whole
  thinking phase into "latency to first token".
- **Reasoning tokens are counted**, per case and totalled per benchmark and run
  (`total_reasoning_tokens`) — that is what a thinking model actually costs you.
  The count is exact when it comes from the API or the stream side-channel, and
  marked *(est.)* when apportioned from inline tags.
- **Thinking gets its own budget.** A benchmark's `max_tokens` is the budget
  for the *answer*; a thinking model's reasoning also counts against it, so a
  1024-token budget can be spent before it answers. Models known to think get
  an extra **Thinking budget** (default 4096, Run tab or `"reasoning_budget"`
  in settings) on top. A model counts as "known" once any saved run shows it
  reasoning; the first time an unknown model is caught thinking and runs out
  of room, that case is retried once with the allowance (marked *retried* in
  the case details). Leave **Max tokens** blank to use each benchmark's own
  answer budget.
- **Truncated thoughts are diagnosed.** If the budget still runs out
  mid-thought there is no answer to grade, and the case says exactly that
  instead of a confusing "no number found".

Each case stores its `reasoning` alongside its `output`; open a question row and
expand **🧠 Show reasoning** to read what the model was thinking.

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

The listen port comes from `server_port` in `config.json`; set the `PORT`
environment variable to override it for a single launch (handy when 8080 is
already taken, or when a launcher assigns the port).

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
storage.py           Durable run storage: atomic writes, schema versioning,
                     tolerant reads, benchmark fingerprints, backup export.
results/*.json       One file per run, powering the Leaderboard tab.
web/                 index.html + app.js + styles.css — no framework, no build.
tests/               stdlib unittest suite (scoring, reasoning split, storage
                     durability). tests/fixtures/ holds a run file in every
                     format the app has ever written.
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
  `target_start` (per model), `start`/`case_start`/`token`/`case`/`done` (per benchmark; each `token`
  event carries `kind`: `"content"` or `"reasoning"`),
  `saved` (per model), `queue_done`. Still accepts the legacy `{ provider, model }` form.
- `GET  /api/runs` — runs currently in progress (so a refreshed page can re-attach)
- `GET  /api/run/stream?run_id=…&since=N` — (re)attach to a run's event stream from index N
- `POST /api/run/stop` — `{ "run_id" }` cancels an in-flight run/queue
- `GET  /api/leaderboard` — runs consolidated into one row per model, with a per-benchmark column matrix
- `GET  /api/results` / `GET /api/results/{id}` — saved runs

---

## Your run history is safe to build on

Results are the valuable part of this app, so storage follows a few hard rules
(implemented in `storage.py`, enforced by `tests/test_storage.py`):

- **Append-only.** One JSON file per run. The app never rewrites, migrates or
  deletes a finished run file - old formats are upgraded *in memory* when read.
- **Atomic writes.** Temp file + fsync + rename, so a crash can't corrupt a file.
- **Checkpointed.** The run file is updated after every case. Kill the server
  mid-run and everything finished so far is kept, shown as *interrupted*.
- **Self-describing.** Each record stores a `schema_version`, the app's git
  commit, the effective parameters, and - per benchmark - a content fingerprint,
  system prompt and scoring type, plus every case's prompt, expected answer and
  the model's output. A run stays readable and re-scorable after its benchmark
  file is edited or deleted.
- **Honest leaderboard.** Editing a benchmark's cases changes its fingerprint;
  older results in that column get a `*` (not comparable with newer runs). The
  newest *complete* run wins a cell - a stopped, interrupted or errored re-run
  (`◔`) never hides a good earlier result.
- **Nothing vanishes silently.** A run file that can't be parsed is listed with
  a warning on the Leaderboard tab and left untouched on disk.
- **Backups.** *Leaderboard → Download backup (.zip)* (or `GET /api/export`)
  bundles every run file.

By default runs live in `results/` inside the checkout (gitignored - so
`git clean -x` or a fresh clone would not have them). To keep them somewhere
safer, set `"results_dir"` in `settings.local.json` (or the `ARENA_RESULTS_DIR`
env var) to e.g. `~/LocalLLMArena-results` and move the files there. On macOS
avoid `~/Documents`, `~/Desktop` and `~/Downloads`: privacy protection can stop
the server process from listing them (the Leaderboard tab then shows a warning).

**When you change the record format:** bump `SCHEMA_VERSION` in `storage.py`,
teach `upgrade_record` the old shape, and add a sample file of the *old* shape
to `tests/fixtures/` (never edit existing fixtures) - the suite loads every
fixture through the full read path.

## Tests

Still zero-dependency — the suite is plain `unittest` from the standard library:

```bash
python3 -m unittest discover -s tests -v
```

It covers answer extraction, every scoring type, and the reasoning/answer split,
including regression cases that show what goes wrong when chain-of-thought is
graded as if it were the answer.

---

## Notes & limitations

- **Sample sizes are small** so a run finishes in a minute or two on a local
  model. Scores are indicative, not leaderboard-grade — swap in full datasets for
  rigorous numbers.
- Scoring for open-ended tasks (`contains`/`regex`) is heuristic. The multiple-choice
  and numeric benchmarks are the most robust.
- A model that reasons **without** any tag or side-channel marker (plain prose
  thinking, then an answer) can't be split automatically — that is what the
  default "End with 'Answer: <…>'" system prompt is for.
- `temperature: 0` is recommended for reproducible benchmarking.
- **OpenRouter costs real money.** Cost shown is what OpenRouter reports for the
  run. Watch `max_tokens` and benchmark sizes when testing expensive models.
- The server listens on `127.0.0.1` only. It has no authentication, so to
  reach it from other devices on a trusted LAN you must opt in with
  `"server_host": "0.0.0.0"` in `settings.local.json` (or `HOST=0.0.0.0`). Your
  OpenRouter key is stored in plain text in `settings.local.json` (gitignored).
