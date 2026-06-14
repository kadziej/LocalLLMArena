# 🏟️ LocalLLMArena

A small, **zero-dependency** local web app for benchmarking the LLMs you run in
[LM Studio](https://lmstudio.ai/). It ships with a handful of public-style
benchmarks (math, knowledge, reasoning, coding, instruction-following) and lets
you drop in your own tests as plain JSON files.

No `pip install`, no Node, no build step — just Python 3 standard library.

---

## Quick start

1. In **LM Studio**, load a model and start the local server (the "Developer" →
   "Start Server" tab). It exposes an OpenAI-compatible API. This app is
   pre-pointed at `http://192.168.9.125:1234`.

2. Run the app:

   ```bash
   python3 server.py
   ```

3. Open the printed URL (default <http://localhost:8080>) in your browser.

4. Pick a model, tick the benchmarks you want, and click **Run selected**.
   Results stream in live and are saved to `results/` for the History tab.

If your LM Studio endpoint differs, edit `lmstudio_base_url` in
[`config.json`](config.json) and restart (or POST `/api/reload-config`).

---

## What's included

| Benchmark | Category | Scoring | What it tests |
|-----------|----------|---------|---------------|
| Grade-School Math (GSM8K style) | Math & Reasoning | `numeric` | Multi-step arithmetic word problems |
| MMLU (Mixed Subjects) | Knowledge | `multiple_choice` | Academic knowledge across subjects |
| ARC (Science Reasoning) | Knowledge | `multiple_choice` | Grade-school science reasoning |
| Commonsense Reasoning | Reasoning | `multiple_choice` | Everyday commonsense |
| TruthfulQA (Misconceptions) | Reasoning | `multiple_choice` | Resisting common myths |
| Code Output Prediction | Coding | `contains` | Tracing short Python snippets |
| Instruction Following | Instruction Following | `regex` | Obeying precise format constraints |
| Custom Example — World Capitals | Custom | `contains` | Template you can copy |

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

### Scoring types

| `scoring` | `expected` is… | Passes when… |
|-----------|----------------|--------------|
| `multiple_choice` | a letter `"A"`–`"J"` | the letter extracted from the answer matches |
| `numeric` | a number (string or number) | the final number in the answer equals it (tiny tolerance) |
| `exact_match` | a string | the whole answer matches (case/space-insensitive) |
| `contains` | a string **or list of strings** | the answer contains it (any of, if a list) |
| `contains_all` | a list of strings | the answer contains every string |
| `regex` | a regex pattern | the pattern matches the raw answer (case-insensitive, dotall) |

For `multiple_choice` and `numeric`, the runner adds a light default system
prompt nudging the model to end with `Answer: <…>`, which makes extraction
reliable. You can override it with your own `system_prompt`.

---

## How it works

```
server.py        Pure-stdlib HTTP server. Serves the web UI and a small JSON API,
                 calls LM Studio's /v1/chat/completions, scores each answer, and
                 streams results back as newline-delimited JSON (NDJSON).
benchmarks/*.json  One file per benchmark (see format above).
results/*.json     One file per run, powering the History tab.
web/             index.html + app.js + styles.css — no framework, no build.
config.json      LM Studio URL, default params, server port.
```

API endpoints (handy if you want to script runs):

- `GET  /api/models` — models currently loaded in LM Studio
- `GET  /api/benchmarks` — list benchmark metadata
- `GET  /api/benchmarks/{id}` — full benchmark incl. cases
- `POST /api/run` — `{ "model", "benchmark_ids": [...], "params": {...} }` → NDJSON stream
- `GET  /api/results` / `GET /api/results/{id}` — saved runs

---

## Notes & limitations

- **Sample sizes are small** so a run finishes in a minute or two on a local
  model. Scores are indicative, not leaderboard-grade — swap in full datasets for
  rigorous numbers.
- Scoring for open-ended tasks (`contains`/`regex`) is heuristic. The multiple-choice
  and numeric benchmarks are the most robust.
- `temperature: 0` is recommended for reproducible benchmarking.
- The server binds to `0.0.0.0` so you can reach it from other devices on your
  LAN; it has no authentication, so only run it on a trusted network.
