// ---------------------------------------------------------------------------
// LocalLLMArena frontend
// ---------------------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  benchmarks: [],
  provider: "local",
  models: [],            // [{id, name, prompt_price, completion_price, context_length}]
  modelMap: {},          // id -> model info
};

function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function scoreClass(score) { return score >= 80 ? "good" : score >= 50 ? "mid" : "bad"; }
function scoreColorVar(s) { return s >= 80 ? "green" : s >= 50 ? "amber" : "red"; }

function fmtDur(s) {
  if (s == null) return "—";
  if (s < 60) return s.toFixed(1) + "s";
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return `${m}m ${r}s`;
}
function perMillion(p) { return p ? "$" + (p * 1e6).toFixed(2) : "$0"; }
function fmtCost(c) {
  if (c == null) return "—";
  if (c === 0) return "free";
  if (c < 0.01) return "$" + c.toFixed(5);
  return "$" + c.toFixed(4);
}
function fmtCtx(n) {
  if (!n) return "";
  if (n >= 1000) return Math.round(n / 1000) + "K ctx";
  return n + " ctx";
}

// ---- tabs -----------------------------------------------------------------
$$(".tab").forEach((t) => {
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.remove("active"));
    $$(".tab-panel").forEach((x) => x.classList.remove("active"));
    t.classList.add("active");
    $("#tab-" + t.dataset.tab).classList.add("active");
    if (t.dataset.tab === "leaderboard") loadHistory();
    if (t.dataset.tab === "benchmarks") loadBenchmarkDetails();
    if (t.dataset.tab === "settings") loadSettingsForm();
  });
});
$("#endpoint-pill").addEventListener("click", () => {
  $$(".tab").forEach((x) => x.classList.remove("active"));
  $$(".tab-panel").forEach((x) => x.classList.remove("active"));
  $('.tab[data-tab="settings"]').classList.add("active");
  $("#tab-settings").classList.add("active");
  loadSettingsForm();
});

// ---- settings -------------------------------------------------------------
async function loadSettings() {
  try {
    const s = await fetch("/api/settings").then((r) => r.json());
    state.settings = s;
    updateEndpointLabel();
    if (s.default_params) {
      if (s.default_params.temperature != null) $("#param-temp").value = s.default_params.temperature;
      if (s.default_params.max_tokens != null) $("#param-maxtok").value = s.default_params.max_tokens;
    }
  } catch (e) {
    $("#endpoint-label").textContent = "settings error";
  }
}

function updateEndpointLabel() {
  const s = state.settings || {};
  if (state.provider === "openrouter") {
    $("#endpoint-label").textContent = "OpenRouter" + (s.openrouter_configured ? "" : " (no key)");
  } else {
    $("#endpoint-label").textContent = s.lmstudio_base_url || "not set";
  }
}

function loadSettingsForm() {
  const s = state.settings || {};
  $("#set-lmurl").value = s.lmstudio_base_url || "";
  $("#or-status").innerHTML = s.openrouter_configured
    ? `<span class="ok-tag">key saved</span>` : `<span class="warn">no key</span>`;
}

$("#save-settings").addEventListener("click", async () => {
  const body = { lmstudio_base_url: $("#set-lmurl").value.trim() };
  const key = $("#set-orkey").value.trim();
  if (key) body.openrouter_api_key = key;
  $("#settings-status").textContent = "saving…";
  try {
    const r = await fetch("/api/settings", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((x) => x.json());
    state.settings = Object.assign(state.settings || {}, r);
    $("#set-orkey").value = "";
    $("#settings-status").innerHTML = `<span class="ok-tag">saved ✓</span>`;
    loadSettingsForm();
    updateEndpointLabel();
    loadModels(state.provider); // pick up new URL / key
  } catch (e) {
    $("#settings-status").innerHTML = `<span class="warn">save failed</span>`;
  }
});
$("#clear-orkey").addEventListener("click", async () => {
  const r = await fetch("/api/settings", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clear_openrouter_api_key: true }),
  }).then((x) => x.json());
  state.settings = Object.assign(state.settings || {}, r);
  loadSettingsForm();
  updateEndpointLabel();
});
$("#or-link").addEventListener("click", () => window.open("https://openrouter.ai/keys", "_blank"));

// ---- provider + models ----------------------------------------------------
$("#provider-select").addEventListener("change", (e) => {
  state.provider = e.target.value;
  updateEndpointLabel();
  loadModels(state.provider);
});
$("#refresh-models").addEventListener("click", () => loadModels(state.provider));
$("#model-search").addEventListener("input", renderModelList);
$("#model-select").addEventListener("change", renderModelInfo);

async function loadModels(provider) {
  const sel = $("#model-select");
  sel.innerHTML = `<option>loading…</option>`;
  $("#status-dot").className = "dot";
  $("#model-info").textContent = "";
  try {
    const data = await fetch("/api/models?provider=" + encodeURIComponent(provider)).then((r) => r.json());
    if (!data.ok) throw new Error(data.error || "unreachable");
    state.models = data.models;
    state.modelMap = {};
    data.models.forEach((m) => { state.modelMap[m.id] = m; });
    $("#status-dot").className = "dot ok";
    renderModelList();
  } catch (e) {
    sel.innerHTML = `<option value="">${provider === "openrouter" ? "OpenRouter" : "LM Studio"} unreachable</option>`;
    $("#status-dot").className = "dot bad";
    $("#model-info").innerHTML = `<span class="warn">${escapeHtml(String(e))}</span>`;
  }
}

function renderModelList() {
  const sel = $("#model-select");
  const q = $("#model-search").value.trim().toLowerCase();
  const prev = sel.value;
  const list = state.models.filter((m) =>
    !q || m.id.toLowerCase().includes(q) || (m.name || "").toLowerCase().includes(q));
  if (!list.length) {
    sel.innerHTML = `<option value="">no models match "${escapeHtml(q)}"</option>`;
    return;
  }
  sel.innerHTML = list.map((m) => {
    let label = m.id;
    if (state.provider === "openrouter" && (m.prompt_price || m.completion_price)) {
      label += `  —  in ${perMillion(m.prompt_price)} / out ${perMillion(m.completion_price)} per 1M`;
    }
    return `<option value="${escapeHtml(m.id)}">${escapeHtml(label)}</option>`;
  }).join("");
  // keep previous selection if still present, else select first
  if (list.some((m) => m.id === prev)) sel.value = prev;
  else sel.selectedIndex = 0;
  renderModelInfo();
}

function renderModelInfo() {
  const m = state.modelMap[$("#model-select").value];
  if (!m) { $("#model-info").textContent = ""; return; }
  const bits = [];
  if (state.provider === "openrouter") {
    bits.push(`input <b>${perMillion(m.prompt_price)}</b> / output <b>${perMillion(m.completion_price)}</b> per 1M tokens`);
    if (m.context_length) bits.push(fmtCtx(m.context_length));
  } else {
    bits.push("local model · no per-token cost");
  }
  $("#model-info").innerHTML = bits.join(" · ");
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

// ---- run ------------------------------------------------------------------
let running = false;
let currentRunId = null;
let runAbort = null;
let statsTimer = null;
let activeBuffer = "";   // streaming output text for the focal panel

$("#run-btn").addEventListener("click", runAll);
$("#stop-btn").addEventListener("click", stopRun);
$("#add-to-queue").addEventListener("click", addToQueue);

// ---- model queue ----------------------------------------------------------
let queueStopped = false;
let runSeq = 0;

function providerIcon(p) { return p === "openrouter" ? "☁️" : "🖥️"; }

function addToQueue() {
  const provider = state.provider;
  const model = $("#model-select").value;
  if (!model) { alert("No model selected."); return; }
  if (state.queue.some((t) => t.provider === provider && t.model === model)) return; // dedupe
  const info = state.modelMap[model] || {};
  state.queue.push({ provider, model, name: info.name || model });
  renderQueue();
}

function removeFromQueue(i) { state.queue.splice(i, 1); renderQueue(); }

function renderQueue() {
  state.queue = state.queue || [];
  const el = $("#queue-list");
  if (!state.queue.length) { el.innerHTML = ""; updateRunButton(); return; }
  el.innerHTML = `<div class="queue-head">Queue (${state.queue.length})
      <button id="queue-clear" class="link-btn">clear</button></div>`
    + state.queue.map((t, i) => `
      <div class="queue-item">
        <span class="qi-prov">${providerIcon(t.provider)}</span>
        <span class="qi-model" title="${escapeHtml(t.model)}">${escapeHtml(t.model)}</span>
        <button class="qi-remove" data-i="${i}" title="remove">×</button>
      </div>`).join("");
  $("#queue-clear").addEventListener("click", () => { state.queue = []; renderQueue(); });
  el.querySelectorAll(".qi-remove").forEach((b) =>
    b.addEventListener("click", () => removeFromQueue(parseInt(b.dataset.i, 10))));
  updateRunButton();
}

function updateRunButton() {
  const n = (state.queue || []).length;
  $("#run-btn").textContent = n > 1 ? `▶ Run queue (${n} models)`
    : n === 1 ? "▶ Run 1 model" : "▶ Run selected model";
}

async function stopRun() {
  queueStopped = true;
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
  if (runAbort) { try { runAbort.abort(); } catch (e) { /* ignore */ } }
}

async function runAll() {
  if (running) return;
  const benchmark_ids = $$(".bench-cb").filter((c) => c.checked).map((c) => c.value);
  if (!benchmark_ids.length) { alert("Select at least one benchmark."); return; }

  // Targets: the queue, or — if empty — the currently selected model.
  let targets = (state.queue && state.queue.length)
    ? state.queue.slice()
    : (() => {
        const m = $("#model-select").value;
        return m ? [{ provider: state.provider, model: m }] : [];
      })();
  if (!targets.length) { alert("Add a model to the queue, or select one."); return; }

  if (targets.some((t) => t.provider === "openrouter") && state.settings && !state.settings.openrouter_configured) {
    if (!confirm("The queue includes OpenRouter models but no API key is set (Settings tab). Those runs will fail. Continue anyway?")) return;
  }

  const params = {
    temperature: parseFloat($("#param-temp").value),
    max_tokens: parseInt($("#param-maxtok").value, 10),
  };

  running = true;
  queueStopped = false;
  $("#run-btn").classList.add("hidden");
  $("#stop-btn").classList.remove("hidden");
  $("#stop-btn").disabled = false;
  $("#stop-btn").textContent = "■ Stop";
  $("#live-results").innerHTML = "";
  $("#live-stats").classList.add("hidden");
  $("#live-active").classList.add("hidden");

  try {
    for (let i = 0; i < targets.length; i++) {
      if (queueStopped) break;
      await streamOneRun(targets[i], benchmark_ids, params, i, targets.length);
    }
    if (queueStopped) {
      $("#queue-status").innerHTML = `<span class="warn">■ Queue stopped</span>`;
    } else if (targets.length > 1) {
      $("#queue-status").innerHTML = `✓ Queue complete — ${targets.length} models. See the <b>Leaderboard</b> to compare.`;
    }
  } finally {
    running = false; currentRunId = null; runAbort = null;
    $("#run-btn").classList.remove("hidden");
    $("#stop-btn").classList.add("hidden");
    $("#live-active").classList.add("hidden");
  }
}

// Run a single (provider, model) target; resolves when its run finishes or is
// cancelled. Each target is saved separately, so each becomes a leaderboard row.
async function streamOneRun(target, benchmark_ids, params, idx, total) {
  const { provider, model } = target;
  const seq = ++runSeq;
  currentRunId = null;
  $("#progress-bar").style.width = "0%";
  $("#live-stats").classList.remove("hidden");

  // per-model section in the results area
  const section = document.createElement("div");
  section.className = "model-section";
  const queueTag = total > 1 ? `<span class="ms-count">model ${idx + 1}/${total}</span>` : "";
  section.innerHTML = `<div class="model-banner">
      <span class="ms-prov">${providerIcon(provider)}</span>
      <span class="ms-model">${escapeHtml(model)}</span>${queueTag}
      <span class="ms-status" id="msstatus-${seq}">starting…</span>
    </div><div class="ms-body" id="msbody-${seq}"></div>`;
  $("#live-results").appendChild(section);

  const agg = {
    startedAt: Date.now(), provider, seq,
    totalCases: state.benchmarks.filter((b) => benchmark_ids.includes(b.id)).reduce((a, b) => a + b.count, 0),
    doneCases: 0, passed: 0, completionTokens: 0, promptTokens: 0,
    tpsSum: 0, tpsN: 0, cost: 0, costKnown: false,
  };
  if (total > 1) $("#queue-status").innerHTML = `Running model <b>${idx + 1}/${total}</b>: ${escapeHtml(model)}`;
  renderLiveStats(agg);
  clearInterval(statsTimer);
  statsTimer = setInterval(() => renderLiveStats(agg), 250);

  const ctx = { seq, container: $("#msbody-" + seq), blocks: {}, agg, model, idx, total };

  try {
    runAbort = new AbortController();
    const resp = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model, benchmark_ids, params }),
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
        handleEvent(JSON.parse(line), ctx);
      }
    }
  } catch (e) {
    if (!(e && e.name === "AbortError")) {
      const st = $("#msstatus-" + seq);
      if (st) st.innerHTML = `<span class="warn">failed: ${escapeHtml(String(e))}</span>`;
    }
  } finally {
    clearInterval(statsTimer);
    renderLiveStats(agg);
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
  if (agg.provider === "openrouter") chips.push(["💲 Cost", agg.costKnown ? fmtCost(agg.cost) : "—"]);
  $("#live-stats").innerHTML = chips.map(([k, v]) =>
    `<div class="stat-chip"><span class="sk">${k}</span><span class="sv">${v}</span></div>`).join("");
}

function handleEvent(ev, ctx) {
  const { seq, agg, blocks } = ctx;
  switch (ev.type) {
    case "run_started":
      currentRunId = ev.run_id;
      if (ev.total_cases) agg.totalCases = ev.total_cases;
      break;
    case "start": {
      const wrap = document.createElement("div");
      wrap.className = "result-block";
      wrap.innerHTML = `
        <div class="result-head">
          <div class="score-badge" id="badge-${seq}-${ev.benchmark_id}">…</div>
          <div class="rh-name">${escapeHtml(ev.benchmark_name)}</div>
          <div class="rh-meta" id="meta-${seq}-${ev.benchmark_id}">0 / ${ev.total}</div>
        </div>
        <div class="case-table" id="table-${seq}-${ev.benchmark_id}"></div>`;
      ctx.container.appendChild(wrap);
      blocks[ev.benchmark_id] = { passed: 0, seen: 0, total: ev.total };
      break;
    }
    case "case_start": {
      // focal panel: show the prompt going in, reset the streaming output area
      activeBuffer = "";
      $("#live-active").classList.remove("hidden");
      const who = ctx.total > 1 ? `${ctx.model} · ` : "";
      $("#la-head").textContent = `${who}${ev.benchmark_id} · case ${ev.index + 1} / ${ev.total} · generating…`;
      $("#la-prompt").textContent = ev.prompt;
      $("#la-output").innerHTML = `<span id="la-text"></span><span class="cursor">▋</span>`;
      $("#la-toks").textContent = "";
      const st = $("#msstatus-" + seq);
      if (st) st.textContent = `running case ${ev.index + 1} / ${ev.total}…`;
      break;
    }
    case "token": {
      // every token streamed from the model, appended live
      activeBuffer += ev.piece;
      const t = $("#la-text");
      if (t) t.textContent = activeBuffer;
      $("#la-toks").textContent = `${ev.n} tok`;
      const out = $("#la-output");
      if (out) out.scrollTop = out.scrollHeight;
      break;
    }
    case "case": {
      const bid = currentBenchId(blocks);
      const blk = blocks[bid];
      if (blk) {
        blk.seen++;
        if (ev.result.passed) blk.passed++;
        const table = $(`#table-${seq}-${bid}`);
        if (table) table.appendChild(renderCaseRow(ev.result));
        const meta = $(`#meta-${seq}-${bid}`);
        if (meta) meta.textContent = `${blk.passed} / ${blk.total}`;
      }
      agg.doneCases++;
      if (ev.result.passed) agg.passed++;
      agg.completionTokens += ev.result.tokens || 0;
      agg.promptTokens += ev.result.prompt_tokens || 0;
      if (ev.result.tokens_per_second != null) { agg.tpsSum += ev.result.tokens_per_second; agg.tpsN++; }
      if (ev.result.cost_usd != null) { agg.cost += ev.result.cost_usd; agg.costKnown = true; }
      $("#progress-bar").style.width = (100 * agg.doneCases / Math.max(1, agg.totalCases)) + "%";
      renderLiveStats(agg);
      break;
    }
    case "done": {
      const badge = $(`#badge-${seq}-${ev.benchmark_id}`);
      if (badge) { badge.textContent = ev.score + "%"; badge.className = "score-badge " + scoreClass(ev.score); }
      const meta = $(`#meta-${seq}-${ev.benchmark_id}`);
      if (meta) meta.innerHTML = benchMetaLine(ev);
      break;
    }
    case "stopped": {
      const st = $("#msstatus-" + seq);
      if (st) st.innerHTML = `<span class="warn">■ stopped</span>`;
      break;
    }
    case "saved": {
      const tail = ev.cancelled ? "stopped · partial saved" : "done · saved";
      let s = `${tail} — ${ev.passed}/${ev.answered ?? ev.total} · `
        + `${(ev.total_tokens || 0).toLocaleString()} tok · ${fmtDur(ev.total_time_s)}`;
      if (ev.avg_tokens_per_second) s += ` · ${ev.avg_tokens_per_second} tok/s`;
      if (ev.total_cost_usd != null) s += ` · ${fmtCost(ev.total_cost_usd)}`;
      const st = $("#msstatus-" + seq);
      if (st) st.innerHTML = s;
      if (!ev.cancelled) $("#progress-bar").style.width = "100%";
      break;
    }
    case "fatal": {
      const st = $("#msstatus-" + seq);
      if (st) st.innerHTML = `<span class="warn">Error: ${escapeHtml(ev.message)}</span>`;
      break;
    }
  }
}

function benchMetaLine(s) {
  const bits = [`${s.passed} / ${s.answered ?? s.total}`];
  if (s.avg_tokens_per_second != null) bits.push(`${s.avg_tokens_per_second} tok/s`);
  if (s.avg_latency_s != null) bits.push(`${s.avg_latency_s}s/case`);
  if (s.total_tokens) bits.push(`${s.total_tokens.toLocaleString()} tok`);
  if (s.total_cost_usd != null) bits.push(fmtCost(s.total_cost_usd));
  if (s.total_time_s != null) bits.push(fmtDur(s.total_time_s));
  return bits.map(escapeHtml).join(" · ");
}

function currentBenchId(blocks) {
  const ids = Object.keys(blocks);
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
  if (r.cost_usd != null && r.cost_usd > 0) metaBits.push(fmtCost(r.cost_usd));
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
    ["Cost", r.cost_usd != null ? fmtCost(r.cost_usd) : "—"],
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

// ---- leaderboard ----------------------------------------------------------
async function loadHistory() {
  const data = await fetch("/api/results").then((r) => r.json());
  const list = $("#history-list");
  if (!data.results.length) {
    list.innerHTML = `<div class="empty">No runs yet. Run some benchmarks first.</div>`;
    return;
  }
  const rows = data.results.map((r) => {
    const when = new Date(r.timestamp).toLocaleString();
    const prov = r.provider === "openrouter"
      ? `<span class="prov-tag or">☁️ OpenRouter</span>` : `<span class="prov-tag local">🖥️ local</span>`;
    const stopped = r.cancelled ? ` <span class="warn">(stopped)</span>` : "";
    return `<tr class="lb-row" data-id="${r.id}">
      <td class="lb-score" style="color:var(--${scoreColorVar(r.overall_score)})">${r.overall_score}%</td>
      <td><div class="lb-model">${escapeHtml(r.model)}${stopped}</div><div class="muted lb-when">${when}</div></td>
      <td>${prov}</td>
      <td>${r.avg_tokens_per_second != null ? r.avg_tokens_per_second + " tok/s" : "—"}</td>
      <td>${r.total_cost_usd != null ? fmtCost(r.total_cost_usd) : (r.provider === "openrouter" ? "—" : "free")}</td>
      <td>${fmtDur(r.total_time_s)}</td>
      <td>${r.passed}/${r.answered ?? r.total}</td>
    </tr>`;
  }).join("");
  list.innerHTML = `<table class="leaderboard">
    <thead><tr><th>Score</th><th>Model</th><th>Provider</th><th>Speed</th><th>Cost</th><th>Time</th><th>Passed</th></tr></thead>
    <tbody>${rows}</tbody></table>`;
  $$(".lb-row").forEach((row) => row.addEventListener("click", () => showRunDetail(row.dataset.id)));
}
$("#refresh-history").addEventListener("click", loadHistory);

async function showRunDetail(id) {
  const r = await fetch("/api/results/" + id).then((x) => x.json());
  const byBench = {};
  (r.cases || []).forEach((c) => { (byBench[c.benchmark_id] = byBench[c.benchmark_id] || []).push(c); });
  const overall = [
    ["Overall", `${r.overall_score}%  (${r.passed}/${r.answered ?? r.total})`],
    ["Provider", r.provider === "openrouter" ? "OpenRouter" : "local"],
    ["Total tokens", (r.total_tokens || 0).toLocaleString()],
    ["Avg speed", r.avg_tokens_per_second != null ? r.avg_tokens_per_second + " tok/s" : "—"],
    ["Total cost", r.total_cost_usd != null ? fmtCost(r.total_cost_usd) : (r.provider === "openrouter" ? "—" : "free")],
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

// ---- benchmark details tab ------------------------------------------------
async function loadBenchmarkDetails() {
  const data = await fetch("/api/benchmarks").then((r) => r.json());
  $("#benchmarks-detail").innerHTML = data.benchmarks.map((b) => `
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
(async function init() {
  state.queue = [];
  renderQueue();
  await loadSettings();
  loadModels(state.provider);
  loadBenchmarks();
})();
