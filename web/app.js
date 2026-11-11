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
    }
    if (s.reasoning_budget != null) $("#param-thinkbudget").value = s.reasoning_budget;
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
let runAbort = null;
let statsTimer = null;
let activeBuffer = "";   // streaming answer text for the focal panel
let thinkBuffer = "";    // streaming chain-of-thought, when the model emits one
let thinkTokens = 0;

$("#run-btn").addEventListener("click", runAll);
$("#stop-btn").addEventListener("click", stopRun);
$("#add-to-queue").addEventListener("click", addToQueue);

// ---- model queue ----------------------------------------------------------
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

// ---- run execution (runs server-side; survives a page refresh) ------------
// `session` tracks the live run; the actual work happens in a server worker
// thread, so refreshing the page just re-attaches to the same run.
let session = null;   // { runId, curCtx, numTargets }

function newAgg(provider, totalCases) {
  return {
    startedAt: Date.now(), provider, totalCases: totalCases || 0,
    doneCases: 0, passed: 0, completionTokens: 0, promptTokens: 0,
    tpsSum: 0, tpsN: 0, cost: 0, costKnown: false,
  };
}

function startRunUI() {
  running = true;
  session = { runId: null, curCtx: null, numTargets: 1 };
  $("#run-btn").classList.add("hidden");
  $("#stop-btn").classList.remove("hidden");
  $("#stop-btn").disabled = false;
  $("#stop-btn").textContent = "■ Stop";
  $("#live-results").innerHTML = "";
  $("#live-stats").classList.add("hidden");
  $("#live-active").classList.add("hidden");
  clearInterval(statsTimer);
  statsTimer = setInterval(() => { if (session && session.curCtx) renderLiveStats(session.curCtx.agg); }, 250);
}

function endRunUI() {
  clearInterval(statsTimer);
  if (session && session.curCtx) renderLiveStats(session.curCtx.agg);
  running = false; runAbort = null;
  $("#run-btn").classList.remove("hidden");
  $("#stop-btn").classList.add("hidden");
  $("#live-active").classList.add("hidden");
}

async function stopRun() {
  $("#stop-btn").disabled = true;
  $("#stop-btn").textContent = "■ Stopping…";
  const rid = session && session.runId;
  if (rid) {
    try {
      await fetch("/api/run/stop", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: rid }),
      });
    } catch (e) { /* ignore */ }
  }
  // Don't abort the stream — we want the saved / queue_done events to arrive.
}

// Read an NDJSON run stream (from POST /api/run or GET /api/run/stream).
async function consumeStream(resp) {
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
      dispatch(JSON.parse(line));
    }
  }
}

async function runAll() {
  if (running) return;
  const benchmark_ids = $$(".bench-cb").filter((c) => c.checked).map((c) => c.value);
  if (!benchmark_ids.length) { alert("Select at least one benchmark."); return; }

  // Targets: the queue, or — if empty — the currently selected model.
  let targets = (state.queue && state.queue.length)
    ? state.queue.map((t) => ({ provider: t.provider, model: t.model }))
    : (() => {
        const m = $("#model-select").value;
        return m ? [{ provider: state.provider, model: m }] : [];
      })();
  if (!targets.length) { alert("Add a model to the queue, or select one."); return; }

  if (targets.some((t) => t.provider === "openrouter") && state.settings && !state.settings.openrouter_configured) {
    if (!confirm("The queue includes OpenRouter models but no API key is set (Settings tab). Those runs will fail. Continue anyway?")) return;
  }

  const params = { temperature: parseFloat($("#param-temp").value) };
  // blank = let each benchmark use its own answer budget
  const maxTok = parseInt($("#param-maxtok").value, 10);
  if (Number.isFinite(maxTok) && maxTok > 0) params.max_tokens = maxTok;
  const reps = parseInt($("#param-repeats").value, 10);
  if (Number.isFinite(reps) && reps > 1) params.repeats = Math.min(20, reps);
  const think = parseInt($("#param-thinkbudget").value, 10);
  if (Number.isFinite(think) && think >= 0) params.reasoning_budget = think;

  startRunUI();
  try {
    runAbort = new AbortController();
    const resp = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ targets, benchmark_ids, params }),
      signal: runAbort.signal,
    });
    await consumeStream(resp);
  } catch (e) {
    if (!(e && e.name === "AbortError"))
      $("#queue-status").innerHTML = `<span class="warn">Run failed: ${escapeHtml(String(e))}</span>`;
  } finally {
    endRunUI();
  }
}

// On page load, re-attach to a run that's still going (e.g. after a refresh).
async function reattachActiveRun() {
  let active;
  try {
    const data = await fetch("/api/runs").then((r) => r.json());
    active = (data.runs || []).find((r) => !r.done) || (data.runs || [])[0];
  } catch (e) { return; }
  if (!active) return;

  // bring the Run tab forward
  $$(".tab").forEach((x) => x.classList.remove("active"));
  $$(".tab-panel").forEach((x) => x.classList.remove("active"));
  $('.tab[data-tab="run"]').classList.add("active");
  $("#tab-run").classList.add("active");

  startRunUI();
  session.runId = active.run_id;
  $("#queue-status").innerHTML = `↻ Re-attached to a run already in progress…`;
  try {
    runAbort = new AbortController();
    const resp = await fetch(`/api/run/stream?run_id=${encodeURIComponent(active.run_id)}&since=0`,
      { signal: runAbort.signal });
    await consumeStream(resp);
  } catch (e) { /* ignore */ } finally {
    endRunUI();
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

// Top-level dispatcher for the unified run stream (covers the whole queue).
function dispatch(ev) {
  switch (ev.type) {
    case "run_started":
      session.runId = ev.run_id;
      session.numTargets = ev.num_targets || 1;
      $("#queue-status").textContent = session.numTargets > 1
        ? `Queue: ${session.numTargets} runs` : "running…";
      return;
    case "target_start": {
      const seq = ++runSeq;
      const section = document.createElement("div");
      section.className = "model-section";
      const queueTag = (ev.total > 1 ? `<span class="ms-count">run ${ev.index + 1}/${ev.total}</span>` : "")
        + (ev.repeats > 1 ? `<span class="ms-count">repeat ${ev.repeat}/${ev.repeats}</span>` : "");
      section.innerHTML = `<div class="model-banner">
          <span class="ms-prov">${providerIcon(ev.provider)}</span>
          <span class="ms-model">${escapeHtml(ev.model)}</span>${queueTag}
          <span class="ms-status" id="msstatus-${seq}">starting…</span>
        </div><div class="ms-body" id="msbody-${seq}"></div>`;
      $("#live-results").appendChild(section);
      const agg = newAgg(ev.provider, ev.total_cases);
      session.curCtx = { seq, container: $("#msbody-" + seq), blocks: {}, agg, model: ev.model, idx: ev.index, total: ev.total };
      if (ev.total > 1) $("#queue-status").innerHTML = `Running <b>${ev.index + 1}/${ev.total}</b>: ${escapeHtml(ev.model)}`;
      $("#progress-bar").style.width = "0%";
      $("#live-stats").classList.remove("hidden");
      renderLiveStats(agg);
      return;
    }
    case "queue_done":
      if (ev.cancelled) $("#queue-status").innerHTML = `<span class="warn">■ Queue stopped</span>`;
      else if ((session.numTargets || 1) > 1) $("#queue-status").innerHTML = `✓ Queue complete — ${session.numTargets} runs. See the <b>Leaderboard</b> to compare.`;
      else $("#queue-status").innerHTML = `✓ Done — saved. See the <b>Leaderboard</b>.`;
      return;
    case "error":
      if (session.curCtx) {
        const st = $("#msstatus-" + session.curCtx.seq);
        if (st) st.innerHTML = `<span class="warn">${escapeHtml(ev.message)}</span>`;
      }
      return;
    default:
      if (session.curCtx) handleTargetEvent(ev, session.curCtx);
  }
}

// Per-target events (one model's benchmarks) → rendered into that model's section.
function handleTargetEvent(ev, ctx) {
  const { seq, agg, blocks } = ctx;
  switch (ev.type) {
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
    case "case_retry": {
      // model turned out to think and ran out of room: same case, bigger budget
      activeBuffer = ""; thinkBuffer = ""; thinkTokens = 0;
      $("#la-think").textContent = ""; $("#la-rtoks").textContent = "";
      $("#la-text").textContent = ""; $("#la-toks").textContent = "";
      $("#la-head").textContent += ` · retrying with thinking budget (${ev.max_tokens} tok)`;
      break;
    }
    case "case_start": {
      // focal panel: show the prompt going in, reset the streaming output area
      activeBuffer = "";
      thinkBuffer = "";
      thinkTokens = 0;
      $("#la-think-wrap").classList.add("hidden");
      $("#la-think").textContent = "";
      $("#la-rtoks").textContent = "";
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
      // every token streamed from the model, appended live. Reasoning models
      // emit thinking first on its own channel - route it to the think pane so
      // the panel shows progress instead of sitting empty until the answer.
      if (ev.kind === "reasoning") {
        thinkBuffer += ev.piece;
        thinkTokens++;
        $("#la-think-wrap").classList.remove("hidden");
        const th = $("#la-think");
        if (th) { th.textContent = thinkBuffer; th.scrollTop = th.scrollHeight; }
        $("#la-rtoks").textContent = `${thinkTokens} tok`;
        break;
      }
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
  if (r.reasoning_tokens) metaBits.push(`🧠 ${r.reasoning_tokens}`);
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

// Thinking is shown collapsed: it is the first thing you want when a case fails
// unexpectedly, and the last thing you want in the way when it passes.
function reasoningBlock(r) {
  if (!r.reasoning) return "";
  const n = r.reasoning_tokens != null ? ` · ${r.reasoning_tokens} tok` : "";
  return `<details class="reasoning"><summary>🧠 Show reasoning${n} `
    + `(excluded from scoring)</summary><pre>${escapeHtml(r.reasoning)}</pre></details>`;
}

function showCaseModal(r) {
  const exp = Array.isArray(r.expected) ? r.expected.join(" | ") : r.expected;
  const metricRows = [
    ["Latency (total)", r.latency_s != null ? r.latency_s + " s" : "—"],
    ["Time to first token", r.ttft_s != null ? r.ttft_s + " s" : "—"],
    ["Speed", r.tokens_per_second != null ? r.tokens_per_second + " tok/s" : "— (too short to measure)"],
    ["Completion tokens", r.tokens ?? "—"],
    ["Reasoning tokens", r.reasoning_tokens != null
      ? r.reasoning_tokens + (r.reasoning_tokens_exact === false ? " (est.)" : "") : "—"],
    ["Prompt tokens", r.prompt_tokens ?? "—"],
    ["Total tokens", r.total_tokens ?? "—"],
    ["Cost", r.cost_usd != null ? fmtCost(r.cost_usd) : "—"],
    ["Finish reason", r.finish_reason ?? "—"],
    ["Token budget", r.max_tokens != null
      ? r.max_tokens + (r.retried_with_thinking_budget ? " (retried)" : "") : "—"],
  ];
  $("#modal-content").innerHTML = `
    <h2>${r.passed ? "✓ Passed" : "✗ Failed"} <span class="muted">· ${escapeHtml(r.id)}</span></h2>
    <div class="metric-grid">
      ${metricRows.map(([k, v]) => `<div class="metric"><span class="mk">${k}</span><span class="mv">${escapeHtml(String(v))}</span></div>`).join("")}
    </div>
    <div class="kv"><div class="k">Prompt</div><pre>${escapeHtml(r.prompt)}</pre></div>
    <div class="kv"><div class="k">Expected</div><pre>${escapeHtml(exp)}</pre></div>
    ${reasoningBlock(r)}
    <div class="kv"><div class="k">Answer (this is what was scored)</div><pre>${escapeHtml(r.output)}</pre></div>
    <div class="kv"><div class="k">Scoring detail</div><pre>${escapeHtml(r.detail)}</pre></div>
    ${r.rescored ? `<div class="kv"><div class="k">Re-scored</div><pre>Originally ${r.passed_recorded ? "passed" : "failed"}: ${escapeHtml(r.detail_recorded || "")}</pre></div>` : ""}`;
  $("#modal").classList.remove("hidden");
}
$("#modal-close").addEventListener("click", () => $("#modal").classList.add("hidden"));
$("#modal").addEventListener("click", (e) => { if (e.target.id === "modal") $("#modal").classList.add("hidden"); });

// ---- leaderboard (consolidated, one row per model, sortable) ---------------
const BENCH_SHORT = {
  gsm8k_math: "GSM8K", mmlu_mixed: "MMLU", arc_science: "ARC", commonsense: "Common",
  truthfulqa: "Truthful", code_reasoning: "Code", instruction_following: "Instr",
  hidden_premise: "Hidden", custom_example_trivia: "Capitals",
  hard_math: "H-Math", hard_reasoning: "H-Reason", gpqa_science: "GPQA",
  tokenizer_traps: "Tokens", hard_instruction: "H-Instr",
};
function shortBench(col) { return BENCH_SHORT[col.id] || (col.name || col.id).slice(0, 8); }

let lbData = null;
let lbSort = { key: "overall", dir: "desc" };

async function loadHistory() {
  lbData = await fetch("/api/leaderboard").then((r) => r.json());
  renderLeaderboard();
}

function lbColumns() {
  // fixed columns + one per benchmark, returning sort accessors
  const cols = [
    { key: "model", label: "Model", type: "str", get: (r) => r.model, cls: "lb-sticky" },
    { key: "provider", label: "Prov", type: "str", get: (r) => r.provider },
    { key: "overall", label: "Overall", type: "num", get: (r) => r.overall_score },
  ];
  (lbData.columns || []).forEach((c) => cols.push({
    key: "b:" + c.id, label: shortBench(c), title: c.name, type: "num", bench: c.id,
    get: (r) => (r.benches[c.id] ? r.benches[c.id].score : null),
  }));
  cols.push({ key: "speed", label: "Speed", type: "num", get: (r) => r.avg_tokens_per_second });
  cols.push({ key: "cost", label: "Cost", type: "num", get: (r) => r.total_cost_usd });
  cols.push({ key: "time", label: "Time", type: "num", get: (r) => r.total_time_s });
  cols.push({ key: "tested", label: "Done", title: "benchmarks tested · runs", type: "num", get: (r) => r.benchmarks_tested });
  return cols;
}

function renderLeaderboard() {
  const host = $("#history-list");
  if (!lbData || !lbData.rows.length) {
    host.innerHTML = `<div class="empty">No runs yet. Run some benchmarks first.</div>${lbData ? storageFooter() : ""}`;
    return;
  }
  const cols = lbColumns();
  const col = cols.find((c) => c.key === lbSort.key) || cols[2];
  const dir = lbSort.dir === "asc" ? 1 : -1;
  const rows = lbData.rows.slice().sort((a, b) => {
    let va = col.get(a), vb = col.get(b);
    if (col.type === "str") return (va || "").localeCompare(vb || "") * dir;
    // numeric: nulls always sink to the bottom regardless of direction
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    return (va - vb) * dir;
  });

  const arrow = (c) => c.key === lbSort.key ? (lbSort.dir === "asc" ? " ▲" : " ▼") : "";
  const head = cols.map((c) =>
    `<th class="${c.cls || ""} ${c.bench ? "lb-bench" : ""} ${c.key === lbSort.key ? "sorted" : ""}" data-key="${c.key}"`
    + `${c.title ? ` title="${escapeHtml(c.title)}"` : ""}>${escapeHtml(c.label)}${arrow(c)}</th>`).join("");

  const body = rows.map((r) => {
    const prov = r.provider === "openrouter" ? `<span class="prov-tag or">☁️</span>` : `<span class="prov-tag local">🖥️</span>`;
    let tds = `<td class="lb-sticky lb-modelcell">
        <div class="lb-model">${escapeHtml(r.model)}</div>
        <div class="muted lb-when">${r.run_count} run${r.run_count > 1 ? "s" : ""} · ${new Date(r.last_run).toLocaleDateString()}</div>
      </td>`;
    tds += `<td>${prov}</td>`;
    tds += `<td class="lb-score" style="color:var(--${scoreColorVar(r.overall_score)})">${r.overall_score}%</td>`;
    (lbData.columns || []).forEach((c) => {
      const e = r.benches[c.id];
      if (!e) { tds += `<td class="lb-cell empty-cell">·</td>`; return; }
      const notes = e.runs > 1
        ? [`mean of ${e.runs} repeats · ±${e.std} (std) · range ${e.min}–${e.max}%`,
           e.flaky.length ? `${e.flaky.length} flaky question(s): ${e.flaky.join(", ")}` : "no flaky questions"]
        : [`${e.passed}/${e.answered}`];
      if (e.ci95 != null) notes.push(`luck of the question sample: up to ±${e.ci95} pts (95% CI, ${e.total} questions)`);
      if (e.rescored) notes.push("re-scored");
      let mark = "";
      if (e.partial) {
        mark += "◔";
        notes.push(e.errors ? `${e.errors} request error(s)` : `incomplete run: ${e.answered} of ${e.total} cases`);
      }
      if (e.stale) { mark += "*"; notes.push("benchmark was edited after this run — not comparable with newer runs"); }
      if (e.removed) notes.push("benchmark file no longer exists");
      tds += `<td class="lb-cell ${scoreClass(e.score)} ${e.stale || e.partial ? "lb-flagged" : ""}" data-rid="${e.result_id}" title="${escapeHtml(notes.join(" · "))}">${e.score}%${e.runs > 1 ? `<span class="lb-spread">±${e.std}</span>` : ""}${mark ? `<sup class="lb-mark">${mark}</sup>` : ""}</td>`;
    });
    tds += `<td>${r.avg_tokens_per_second != null ? r.avg_tokens_per_second : "—"}</td>`;
    tds += `<td>${r.total_cost_usd != null ? fmtCost(r.total_cost_usd) : (r.provider === "openrouter" ? "—" : "free")}</td>`;
    tds += `<td>${fmtDur(r.total_time_s)}</td>`;
    tds += `<td class="muted">${r.benchmarks_tested}</td>`;
    return `<tr>${tds}</tr>`;
  }).join("");

  host.innerHTML = `<div class="lb-scroll"><table class="leaderboard consolidated">
    <thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>
    <div class="muted lb-hint">Click a column header to sort · click a benchmark cell to open that run's details
      · <sup>*</sup> benchmark edited since the run · <sup>◔</sup> incomplete run · ±N = spread over repeats · hover a cell for details</div>${storageFooter()}`;

  host.querySelectorAll("th[data-key]").forEach((th) => th.addEventListener("click", () => {
    const key = th.dataset.key;
    if (lbSort.key === key) lbSort.dir = lbSort.dir === "asc" ? "desc" : "asc";
    else lbSort = { key, dir: key === "model" || key === "provider" ? "asc" : "desc" };
    renderLeaderboard();
  }));
  host.querySelectorAll(".lb-cell[data-rid]").forEach((td) =>
    td.addEventListener("click", () => showRunDetail(td.dataset.rid)));
}
$("#refresh-history").addEventListener("click", loadHistory);

async function rescoreRuns() {
  const btn = $("#rescore-btn");
  btn.disabled = true;
  try {
    const post = (apply) => fetch("/api/rescore", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ apply }),
    }).then((r) => r.json());
    const rep = await post(false);
    const lines = rep.changes.slice(0, 25).map((c) =>
      `${c.after ? "fail → PASS" : "pass → FAIL"}  ${c.model} · ${c.benchmark_id}/${c.case_id}`);
    if (rep.changes.length > 25) lines.push(`…and ${rep.changes.length - 25} more`);
    if (rep.refreshed_benchmarks) lines.push(`${rep.refreshed_benchmarks} result(s) now match the current benchmark files`);
    if (!lines.length) {
      alert(`Re-graded ${rep.cases_graded} saved answers in ${rep.runs_checked} runs — nothing changes.`);
      return;
    }
    if (!confirm(`Re-grading ${rep.cases_graded} saved answers would change:\n\n${lines.join("\n")}\n\n`
      + "Apply? Run files are not modified; new verdicts are stored alongside them and can be undone by deleting the rescores folder.")) return;
    const done = await post(true);
    alert(`Applied to ${done.runs_changed} run(s).`);
    loadHistory();
  } catch (e) {
    alert("Re-score failed: " + e);
  } finally {
    btn.disabled = false;
  }
}
$("#rescore-btn").addEventListener("click", rescoreRuns);

function storageFooter() {
  const st = lbData.storage;
  if (!st) return "";
  const bad = (st.unreadable || []).map((u) =>
    `<div class="lb-unreadable">⚠ Could not read <code>${escapeHtml(u.file)}</code> — ${escapeHtml(u.error)}. The file was left untouched.</div>`).join("");
  return `${bad}<div class="muted lb-hint">${st.runs} saved run${st.runs === 1 ? "" : "s"} in <code>${escapeHtml(st.dir)}</code>
    · <a href="/api/export">⬇ Download backup (.zip)</a></div>`;
}

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
    + `${r.status === "interrupted" ? ' <span class="warn">(interrupted — partial)</span>' : r.cancelled ? ' <span class="warn">(stopped)</span>' : ""}</h2>`
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
  await loadBenchmarks();
  reattachActiveRun();   // resume a run that's still going (e.g. after a refresh)
})();
