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
  $("#run-btn").disabled = true;
  $("#live-results").innerHTML = "";
  $("#progress-bar").style.width = "0%";

  // pre-compute total cases for the progress bar
  const totalCases = state.benchmarks
    .filter((b) => benchmark_ids.includes(b.id))
    .reduce((acc, b) => acc + b.count, 0);
  let doneCases = 0;

  const blocks = {}; // benchmark_id -> {el, tableEl, passed, seen}

  try {
    const resp = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, benchmark_ids, params }),
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
        handleEvent(ev, blocks);
        if (ev.type === "case") {
          doneCases++;
          $("#progress-bar").style.width = (100 * doneCases / Math.max(1, totalCases)) + "%";
          $("#run-status").textContent = `${doneCases} / ${totalCases} cases`;
        }
        if (ev.type === "saved") {
          $("#run-status").textContent = `Done — overall ${ev.passed}/${ev.total} · saved`;
          $("#progress-bar").style.width = "100%";
        }
        if (ev.type === "fatal") {
          $("#run-status").innerHTML = `<span class="warn">Error: ${escapeHtml(ev.message)}</span>`;
        }
      }
    }
  } catch (e) {
    $("#run-status").innerHTML = `<span class="warn">Run failed: ${escapeHtml(String(e))}</span>`;
  } finally {
    running = false;
    $("#run-btn").disabled = false;
  }
}

function handleEvent(ev, blocks) {
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
  if (ev.type === "case") {
    // streamed case events arrive between "start" and "done"; attach to the
    // most recently started, not-yet-complete benchmark block.
    const bid = currentBenchId(blocks);
    const blk = blocks[bid];
    if (blk) {
      blk.seen++;
      if (ev.result.passed) blk.passed++;
      const table = $("#table-" + bid);
      if (table) table.appendChild(renderCaseRow(ev.result));
      const meta = $("#meta-" + bid);
      if (meta) meta.textContent = `${blk.passed} / ${blk.total}`;
    }
  }
  if (ev.type === "done") {
    const id = ev.benchmark_id;
    const badge = $("#badge-" + id);
    if (badge) {
      badge.textContent = ev.score + "%";
      badge.className = "score-badge " + scoreClass(ev.score);
    }
    const meta = $("#meta-" + id);
    if (meta) {
      const lat = ev.avg_latency_s != null ? ` · ${ev.avg_latency_s}s avg` : "";
      meta.textContent = `${ev.passed} / ${ev.total}${lat}`;
    }
  }
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
  const lat = r.latency_s != null ? r.latency_s + "s" : "";
  div.innerHTML = `${mark}<span class="cprompt">${escapeHtml(firstLine(r.prompt))}</span><span class="clat">${lat}</span>`;
  div.addEventListener("click", () => showCaseModal(r));
  return div;
}

function firstLine(s) {
  const line = String(s || "").split("\n")[0];
  return line.length > 110 ? line.slice(0, 110) + "…" : line;
}

function showCaseModal(r) {
  const exp = Array.isArray(r.expected) ? r.expected.join(" | ") : r.expected;
  $("#modal-content").innerHTML = `
    <h2>${r.passed ? "✓ Passed" : "✗ Failed"} <span class="muted">· ${escapeHtml(r.id)}</span></h2>
    <div class="kv"><div class="k">Prompt</div><pre>${escapeHtml(r.prompt)}</pre></div>
    <div class="kv"><div class="k">Expected</div><pre>${escapeHtml(exp)}</pre></div>
    <div class="kv"><div class="k">Model output</div><pre>${escapeHtml(r.output)}</pre></div>
    <div class="kv"><div class="k">Scoring detail</div><pre>${escapeHtml(r.detail)}</pre></div>
    <div class="kv muted">latency: ${r.latency_s ?? "—"}s · completion tokens: ${r.tokens ?? "—"}</div>`;
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
    return `<div class="hist-row" data-id="${r.id}">
      <div class="hr-score ${scoreClass(r.overall_score)}" style="color:var(--${scoreColorVar(r.overall_score)})">${r.overall_score}%</div>
      <div>
        <div class="hr-model">${escapeHtml(r.model)}</div>
        <div class="hr-bench">${benches}</div>
      </div>
      <div class="hr-time">${when}</div>
      <div class="hr-time">${r.passed}/${r.total}</div>
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
  let html = `<div class="card"><div class="row between"><h2>${escapeHtml(r.model)}</h2>`
    + `<div class="muted">${new Date(r.timestamp).toLocaleString()}</div></div>`;
  r.benchmarks.forEach((b) => {
    html += `<div class="result-block">
      <div class="result-head">
        <div class="score-badge ${scoreClass(b.score)}">${b.score}%</div>
        <div class="rh-name">${escapeHtml(b.benchmark_name)}</div>
        <div class="rh-meta">${b.passed}/${b.total}${b.avg_latency_s != null ? " · " + b.avg_latency_s + "s avg" : ""}</div>
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
