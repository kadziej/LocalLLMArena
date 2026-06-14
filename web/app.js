// ---------------------------------------------------------------------------
// LocalLLMArena frontend
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  benchmarks: [],
  models: [],
};

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function scoreClass(score) {
  if (score >= 80) return "good";
  if (score >= 50) return "mid";
  return "bad";
}

// ---- tabs -----------------------------------------------------------------
$$(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.remove("active"));
    $$(".tab-panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "history") loadHistory();
    if (t.dataset.tab === "benchmarks") loadBenchmarkDetails();
  });
});

// ---- config / endpoint status ---------------------------------------------
async function loadConfig() {
  try {
    const cfg = await fetch("/api/config").then((r) => r.json());
    $("#endpoint-label").textContent = cfg.lmstudio_base_url;
    if (cfg.default_params) {
      if (cfg.default_params.temperature != null) $("#param-temp").value = cfg.default_params.temperature;
      if (cfg.default_params.max_tokens != null) $("#param-maxtok").value = cfg.default_params.max_tokens;
    }
  } catch (e) {
    $("#endpoint-label").textContent = "config error";
  }
}

// ---- models ---------------------------------------------------------------
async function loadModels() {
  const sel = $("#model-select");
  sel.innerHTML = `<option>loading…</option>`;
  $("#status-dot").className = "dot";
  try {
    const data = await fetch("/api/models").then((r) => r.json());
    if (!data.ok) throw new Error(data.error || "unreachable");
    state.models = data.models;
    if (!data.models.length) {
      sel.innerHTML = `<option value="">(no models loaded in LM Studio)</option>`;
      $("#status-dot").className = "dot bad";
      return;
    }
    sel.innerHTML = data.models.map((m) => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
    $("#status-dot").className = "dot ok";
  } catch (e) {
    sel.innerHTML = `<option value="">LM Studio unreachable</option>`;
    $("#status-dot").className = "dot bad";
    $("#endpoint-label").title = String(e);
  }
}

// ---- benchmarks list ------------------------------------------------------
async function loadBenchmarks() {
  const data = await fetch("/api/benchmarks").then((r) => r.json());
  state.benchmarks = data.benchmarks;
  const list = $("#benchmark-list");
  list.innerHTML = state.benchmarks.map((b) => {
    const err = b.error ? ` <span class="warn">(load error)</span>` : "";
    return `<label class="bench-item">
      <input type="checkbox" class="bench-cb" value="${escapeHtml(b.id)}" ${b.error ? "disabled" : "checked"} />
      <div>
        <div class="bi-name">${escapeHtml(b.name)}<span class="cat-tag">${escapeHtml(b.category || "")}</span>${err}</div>
        <div class="bi-meta">${b.count} cases · ${escapeHtml(b.scoring)}</div>
      </div>
    </label>`;
  }).join("");
}

$("#select-all").addEventListener("click", () => $$(".bench-cb").forEach((c) => { if (!c.disabled) c.checked = true; }));
$("#select-none").addEventListener("click", () => $$(".bench-cb").forEach((c) => c.checked = false));
$("#refresh-models").addEventListener("click", loadModels);

// ---- run ------------------------------------------------------------------
let running = false;

$("#run-btn").addEventListener("click", runSelected);
$("#stop-btn").addEventListener("click", stopRun);

// live-run state
let currentRunId = null;
let runAbort = null;
let statsTimer = null;

function fmtDur(s) {
  if (s == null) return "—";
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return `${m}m ${r}s`;
}

async function stopRun() {
  $("#stop-btn").disabled = true;
  $("#stop-btn").textContent = "■ Stopping…";
  if (currentRunId) {
    try {
      await fetch("/api/run/stop", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: currentRunId }),
      });
    } catch (e) { /* ignore */ }
  }
}

async function runSelected() {
  if (running) return;
  const model = $("#model-select").value;
  const benchmark_ids = $$(".bench-cb").filter((c) => c.checked).map((c) => c.value);
  if (!model) { alert("No model selected. Load a model in LM Studio first."); return; }
  if (!benchmark_ids.length) { alert("Select at least one benchmark."); return; }

  const params = {
    temperature: parseFloat($("#param-temp").value),
    max_tokens: parseInt($("#param-maxtok").value, 10),
  };

  running = true;
  currentRunId = null;
  $("#run-btn").classList.add("hidden");
  $("#stop-btn").classList.remove("hidden");
  $("#stop-btn").disabled = false;
  $("#stop-btn").textContent = "■ Stop";
  $("#live-results").innerHTML = "";
  $("#progress-bar").style.width = "0%";
  $("#live-stats").classList.remove("hidden");

  // running tally across the whole run
  const agg = {
    startedAt: Date.now(),
    totalCases: state.benchmarks.filter((b) => benchmark_ids.includes(b.id))
      .reduce((acc, b) => acc + b.count, 0),
    doneCases: 0, passed: 0, completionTokens: 0, promptTokens: 0,
    tpsSum: 0, tpsN: 0, finished: false,
  };
  renderLiveStats(agg);
  statsTimer = setInterval(() => renderLiveStats(agg), 250); // keep the clock ticking

  const blocks = {}; // benchmark_id -> {passed, seen, total}

  try {
    runAbort = new AbortController();
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, benchmark_ids, params }),
      signal: runAbort.signal,
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        handleEvent(ev, blocks, agg);
      }
    }
  } catch (e) {
    if (!(e && e.name === "AbortError")) {
      $("#run-status").innerHTML = `<span class="warn">Run failed: ${escapeHtml(String(e))}</span>`;
    }
  } finally {
    agg.finished = true;
    clearInterval(statsTimer);
    renderLiveStats(agg);
    running = false;
    currentRunId = null;
    runAbort = null;
    $("#run-btn").classList.remove("hidden");
    $("#stop-btn").classList.add("hidden");
  }
}

function renderLiveStats(agg) {
  const elapsed = (Date.now() - agg.startedAt) / 1000;
  const avgTps = agg.tpsN ? (agg.tpsSum / agg.tpsN) : null;
  const totalTok = agg.completionTokens + agg.promptTokens;
  const chips = [
    ["⏱ Elapsed", fmtDur(elapsed)],
    ["✔ Cases", `${agg.doneCases} / ${agg.totalCases}`],
    ["✓ Passed", `${agg.passed}`],
    ["🔢 Tokens", `${totalTok.toLocaleString()} (${agg.completionTokens.toLocaleString()} out)`],
    ["⚡ Avg speed", avgTps != null ? `${avgTps.toFixed(1)} tok/s` : "—"],
  ];
  $("#live-stats").innerHTML = chips.map(([k, v]) =>
    `<div class="stat-chip"><span class="sk">${k}</span><span class="sv">${v}</span></div>`).join("");
}

function handleEvent(ev, blocks, agg) {
  if (ev.type === "run_started") {
    currentRunId = ev.run_id;
    if (ev.total_cases) agg.totalCases = ev.total_cases;
    $("#run-status").textContent = "running…";
  }
  if (ev.type === "start") {
    const id = ev.benchmark_id;
    const wrap = document.createElement("div");
    wrap.className = "result-block";
    wrap.innerHTML = `
      <div class="result-head">
        <div class="score-badge" id="badge-${id}">…</div>
        <div class="rh-name">${escapeHtml(ev.benchmark_name)}</div>
        <div class="rh-meta" id="meta-${id}">0 / ${ev.total}</div>
      </div>
      <div class="case-table" id="table-${id}"></div>`;
    $("#live-results").appendChild(wrap);
    blocks[id] = { passed: 0, seen: 0, total: ev.total };
  }
  if (ev.type === "case_start") {
    // show a "running" placeholder row that the matching "case" event replaces
    const bid = ev.benchmark_id || currentBenchId(blocks);
    const table = $("#table-" + bid);
    if (table) {
      const row = document.createElement("div");
      row.className = "case-row running";
      row.id = `running-${bid}`;
      row.innerHTML = `<span class="mark spin">◴</span>
        <span class="cprompt">${escapeHtml(firstLine(ev.prompt))}</span>
        <span class="clat">generating…</span>`;
      // remove any stale running row first
      const old = $("#running-" + bid);
      if (old) old.remove();
      table.appendChild(row);
      table.parentElement.scrollIntoView({ block: "nearest" });
    }
    $("#run-status").textContent = `running case ${ev.index + 1} / ${ev.total}…`;
  }
  if (ev.type === "case") {
    const bid = currentBenchId(blocks);
    const blk = blocks[bid];
    const running = $("#running-" + bid);
    if (running) running.remove();
    if (blk) {
      blk.seen++;
      if (ev.result.passed) blk.passed++;
      const table = $("#table-" + bid);
      if (table) table.appendChild(renderCaseRow(ev.result));
      const meta = $("#meta-" + bid);
      if (meta) meta.textContent = `${blk.passed} / ${blk.total}`;
    }
    // global aggregate
    agg.doneCases++;
    if (ev.result.passed) agg.passed++;
    agg.completionTokens += ev.result.tokens || 0;
    agg.promptTokens += ev.result.prompt_tokens || 0;
    if (ev.result.tokens_per_second != null) { agg.tpsSum += ev.result.tokens_per_second; agg.tpsN++; }
    $("#progress-bar").style.width = (100 * agg.doneCases / Math.max(1, agg.totalCases)) + "%";
    renderLiveStats(agg);
  }
  if (ev.type === "done") {
    const id = ev.benchmark_id;
    const badge = $("#badge-" + id);
    if (badge) {
      badge.textContent = ev.score + "%";
      badge.className = "score-badge " + scoreClass(ev.score);
    }
    const meta = $("#meta-" + id);
    if (meta) meta.innerHTML = benchMetaLine(ev);
  }
  if (ev.type === "stopped") {
    $("#run-status").innerHTML = `<span class="warn">■ Stopped by user</span>`;
  }
  if (ev.type === "saved") {
    const tail = ev.cancelled ? "stopped · partial saved" : "done · saved";
    $("#run-status").innerHTML =
      `${tail} — ${ev.passed}/${ev.answered ?? ev.total} passed · `
      + `${(ev.total_tokens || 0).toLocaleString()} tokens · ${fmtDur(ev.total_time_s)}`
      + (ev.avg_tokens_per_second ? ` · ${ev.avg_tokens_per_second} tok/s` : "");
    if (!ev.cancelled) $("#progress-bar").style.width = "100%";
  }
  if (ev.type === "fatal") {
    $("#run-status").innerHTML = `<span class="warn">Error: ${escapeHtml(ev.message)}</span>`;
  }
}

function benchMetaLine(s) {
  const bits = [`${s.passed} / ${s.answered ?? s.total}`];
  if (s.avg_tokens_per_second != null) bits.push(`${s.avg_tokens_per_second} tok/s`);
  if (s.avg_latency_s != null) bits.push(`${s.avg_latency_s}s/case`);
  if (s.total_tokens) bits.push(`${s.total_tokens.toLocaleString()} tok`);
  if (s.total_time_s != null) bits.push(fmtDur(s.total_time_s));
  return bits.map(escapeHtml).join(" · ");
}

function currentBenchId(blocks) {
  const ids = Object.keys(blocks);
  // the most recently started, not-yet-complete block
  for (let i = ids.length - 1; i >= 0; i--) {
    if (blocks[ids[i]].seen < blocks[ids[i]].total) return ids[i];
  }
  return ids[ids.length - 1];
}

function renderCaseRow(r) {
  const div = document.createElement("div");
  div.className = "case-row";
  const mark = r.passed ? `<span class="mark pass">✓</span>` : `<span class="mark fail">✗</span>`;
  const metaBits = [];
  if (r.latency_s != null) metaBits.push(`${r.latency_s}s`);
  if (r.tokens != null) metaBits.push(`${r.tokens} tok`);
  if (r.tokens_per_second != null) metaBits.push(`${r.tokens_per_second} tok/s`);
  div.innerHTML = `${mark}<span class="cprompt">${escapeHtml(firstLine(r.prompt))}</span>`
    + `<span class="clat">${escapeHtml(metaBits.join(" · "))}</span>`;
  div.addEventListener("click", () => showCaseModal(r));
  return div;
}

function firstLine(s) {
  const line = String(s || "").split("\n")[0];
  return line.length > 90 ? line.slice(0, 90) + "…" : line;
}

function showCaseModal(r) {
  const exp = Array.isArray(r.expected) ? r.expected.join(" | ") : r.expected;
  const metricRows = [
    ["Latency (total)", r.latency_s != null ? r.latency_s + " s" : "—"],
    ["Time to first token", r.ttft_s != null ? r.ttft_s + " s" : "—"],
    ["Speed", r.tokens_per_second != null ? r.tokens_per_second + " tok/s" : "—"],
    ["Completion tokens", r.tokens ?? "—"],
    ["Prompt tokens", r.prompt_tokens ?? "—"],
    ["Total tokens", r.total_tokens ?? "—"],
    ["Finish reason", r.finish_reason ?? "—"],
  ];
  $("#modal-content").innerHTML = `
    <h2>${r.passed ? "✓ Passed" : "✗ Failed"} <span class="muted">· ${escapeHtml(r.id)}</span></h2>
    <div class="metric-grid">
      ${metricRows.map(([k, v]) => `<div class="metric"><span class="mk">${k}</span><span class="mv">${escapeHtml(String(v))}</span></div>`).join("")}
    </div>
    <div class="kv"><div class="k">Prompt</div><pre>${escapeHtml(r.prompt)}</pre></div>
    <div class="kv"><div class="k">Expected</div><pre>${escapeHtml(exp)}</pre></div>
    <div class="kv"><div class="k">Model output</div><pre>${escapeHtml(r.output)}</pre></div>
    <div class="kv"><div class="k">Scoring detail</div><pre>${escapeHtml(r.detail)}</pre></div>`;
  $("#modal").classList.remove("hidden");
}

$("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });

// ---- history --------------------------------------------------------------
async function loadHistory() {
  const data = await fetch("/api/results").then((r) => r.json());
  const list = $("#history-list");
  if (!data.results.length) {
    list.innerHTML = `<div class="empty">No runs yet. Run some benchmarks first.</div>`;
    return;
  }
  list.innerHTML = data.results.map((r) => {
    const when = new Date(r.timestamp).toLocaleString();
    const benches = r.benchmarks.map((b) => `${escapeHtml(b.benchmark_name)} ${b.score}%`).join(" · ");
    const stopped = r.cancelled ? ` <span class="warn">(stopped)</span>` : "";
    const perf = [];
    if (r.total_tokens) perf.push(`${r.total_tokens.toLocaleString()} tok`);
    if (r.avg_tokens_per_second != null) perf.push(`${r.avg_tokens_per_second} tok/s`);
    if (r.total_time_s != null) perf.push(fmtDur(r.total_time_s));
    return `<div class="hist-row" data-id="${r.id}">
      <div class="hr-score ${scoreClass(r.overall_score)}" style="color:var(--${scoreColorVar(r.overall_score)})">${r.overall_score}%</div>
      <div>
        <div class="hr-model">${escapeHtml(r.model)}${stopped}</div>
        <div class="hr-bench">${benches}</div>
      </div>
      <div class="hr-time">${when}<br>${escapeHtml(perf.join(" · "))}</div>
      <div class="hr-time">${r.passed}/${r.answered ?? r.total}</div>
    </div>`;
  }).join("");
  $$(".hist-row").forEach((row) => row.addEventListener("click", () => showRunDetail(row.dataset.id)));
}

function scoreColorVar(s) { return s >= 80 ? "green" : s >= 50 ? "amber" : "red"; }

async function showRunDetail(id) {
  const r = await fetch("/api/results/" + id).then((x) => x.json());
  const byBench = {};
  (r.cases || []).forEach((c) => {
    (byBench[c.benchmark_id] = byBench[c.benchmark_id] || []).push(c);
  });
  const overall = [
    ["Overall", `${r.overall_score}%  (${r.passed}/${r.answered ?? r.total})`],
    ["Total tokens", (r.total_tokens || 0).toLocaleString() + (r.total_completion_tokens != null ? ` (${r.total_completion_tokens.toLocaleString()} out)` : "")],
    ["Avg speed", r.avg_tokens_per_second != null ? r.avg_tokens_per_second + " tok/s" : "—"],
    ["Total time", fmtDur(r.total_time_s)],
  ];
  let html = `<div class="card"><div class="row between"><h2>${escapeHtml(r.model)}`
    + `${r.cancelled ? ' <span class="warn">(stopped)</span>' : ""}</h2>`
    + `<div class="muted">${new Date(r.timestamp).toLocaleString()}</div></div>`
    + `<div class="metric-grid">`
    + overall.map(([k, v]) => `<div class="metric"><span class="mk">${k}</span><span class="mv">${escapeHtml(String(v))}</span></div>`).join("")
    + `</div>`;
  r.benchmarks.forEach((b) => {
    html += `<div class="result-block">
      <div class="result-head">
        <div class="score-badge ${scoreClass(b.score)}">${b.score}%</div>
        <div class="rh-name">${escapeHtml(b.benchmark_name)}</div>
        <div class="rh-meta">${benchMetaLine(b)}</div>
      </div>
      <div class="case-table" id="dtable-${escapeHtml(b.benchmark_id)}"></div></div>`;
  });
  html += `</div>`;
  $("#history-detail").innerHTML = html;
  r.benchmarks.forEach((b) => {
    const table = $("#dtable-" + b.benchmark_id);
    (byBench[b.benchmark_id] || []).forEach((c) => table.appendChild(renderCaseRow(c)));
  });
  $("#history-detail").scrollIntoView({ behavior: "smooth" });
}

$("#refresh-history").addEventListener("click", loadHistory);

// ---- benchmark details tab ------------------------------------------------
async function loadBenchmarkDetails() {
  const data = await fetch("/api/benchmarks").then((r) => r.json());
  const el = $("#benchmarks-detail");
  el.innerHTML = data.benchmarks.map((b) => `
    <div class="bench-detail">
      <h3>${escapeHtml(b.name)} <span class="cat-tag">${escapeHtml(b.category || "")}</span></h3>
      <div class="muted">${escapeHtml(b.description || "")}</div>
      <div class="muted" style="margin-top:6px">
        <code>${escapeHtml(b.file || b.id + ".json")}</code> · ${b.count} cases · scoring: <code>${escapeHtml(b.scoring)}</code>
        ${b.error ? `<span class="warn"> · error: ${escapeHtml(b.error)}</span>` : ""}
      </div>
    </div>`).join("");
}
$("#refresh-benchmarks").addEventListener("click", loadBenchmarkDetails);

// ---- init -----------------------------------------------------------------
loadConfig();
loadModels();
loadBenchmarks();
