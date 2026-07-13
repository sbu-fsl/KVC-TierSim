"use strict";

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);

// Policies are loaded dynamically from /api/catalog so a new policy module on
// the backend appears here with no frontend changes. Each entry:
//   { key, label, sub, color }
let POLICIES = [];

// Heatmap state colors (match sweep.py).
const HEAT = {
  fail: "#e01c1c",     // (false,false)
  rpc: "#fab333",      // (true,false)  latency fails
  latency: "#679ff2",  // (false,true)  rpc fails
  pass: "#33bf59",     // (true,true)
};

let CATALOG = null;
const charts = {};

function num(v) {
  if (v === "Infinity") return Infinity;
  if (v === "-Infinity") return -Infinity;
  if (v === null || v === undefined) return NaN;
  return Number(v);
}

function fmtTime(s) {
  s = num(s);
  if (!isFinite(s)) return "∞";
  if (s <= 0) return "0";
  if (s < 1e-3) return (s * 1e6).toFixed(0) + " µs";
  if (s < 1) return (s * 1e3).toFixed(1) + " ms";
  if (s < 60) return s.toFixed(2) + " s";
  if (s < 3600) return (s / 60).toFixed(1) + " min";
  return (s / 3600).toFixed(1) + " hr";
}

function fmtInt(v) {
  v = num(v);
  if (!isFinite(v)) return "∞";
  return Math.round(v).toLocaleString();
}

function fmtRate(v) {
  v = num(v);
  if (!isFinite(v)) return "∞";
  if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
  if (v >= 1e3) return (v / 1e3).toFixed(1) + "k";
  return v.toFixed(0);
}

function fmtBlocks(b) {
  b = num(b);
  if (!isFinite(b)) return "∞";
  const bytes = b * (parseFloat($("bytesPerBlock").value) * 1e6);
  const gb = bytes / 1e9;
  return `${Math.round(b).toLocaleString()} blk · ${gb.toFixed(1)} GB`;
}

// ------------------------------------------------------------------ catalog
async function loadCatalog() {
  const res = await fetch("/api/catalog");
  CATALOG = await res.json();

  POLICIES = (CATALOG.policies || []).map((p) => ({
    key: p.name, label: p.label, sub: p.description, color: p.color,
  }));

  fillSelect("gpu", CATALOG.gpus, (k, v) => `${v.name} · ${(v.hbm_capacity / 1e9).toFixed(0)}GB · $${v.cost.toLocaleString()}`);
  fillSelect("dram", CATALOG.drams, (k, v) => `${v.name} · ${(v.capacity / 1e9).toFixed(0)}GB`);
  fillSelect("disk", CATALOG.disks, (k, v) => `${v.name} · ${fmtCap(v.capacity)}`);
  fillSelect("link", CATALOG.links, (k, v) => `${v.name} · ${(v.bandwidth / 1e9).toFixed(0)}GB/s`);

  const stackSel = $("stackPreset");
  Object.keys(CATALOG.stacks).forEach((k) => stackSel.add(new Option(k, k)));
  const modelSel = $("modelPreset");
  Object.entries(CATALOG.models).forEach(([k, v]) => modelSel.add(new Option(v.name, k)));

  // sensible defaults
  $("gpu").value = "H200";
  $("dram").value = "DDR5";
  $("disk").value = "NVMe";
  $("link").value = "NVLink";
  updateHwSummary();
}

function fmtCap(bytes) {
  return bytes >= 1e12 ? (bytes / 1e12).toFixed(1) + "TB" : (bytes / 1e9).toFixed(0) + "GB";
}

function fillSelect(id, mapping, labeler) {
  const sel = $(id);
  sel.innerHTML = "";
  Object.entries(mapping).forEach(([k, v]) => sel.add(new Option(labeler(k, v), k)));
}

function updateHwSummary() {
  if (!CATALOG) return;
  const g = CATALOG.gpus[$("gpu").value];
  const d = CATALOG.drams[$("dram").value];
  const k = CATALOG.disks[$("disk").value];
  const l = CATALOG.links[$("link").value];
  if (!g || !d || !k || !l) return;
  const cost = g.cost * (+$("gpuCount").value) + d.cost * (+$("dramCount").value) + k.cost + l.cost;
  $("hwSummary").textContent =
    `HBM ${(g.hbm_bandwidth / 1e12).toFixed(2)} TB/s · DRAM ${(d.bandwidth / 1e9).toFixed(0)} GB/s · ` +
    `Disk ${(k.bandwidth / 1e9).toFixed(1)} GB/s · Link ${(l.bandwidth / 1e9).toFixed(0)} GB/s · ~$${cost.toLocaleString()}`;
}

// ------------------------------------------------------------------ inputs
function applyStackPreset() {
  const key = $("stackPreset").value;
  if (!key || !CATALOG.stacks[key]) return;
  const s = CATALOG.stacks[key];
  $("gpu").value = s.gpu_key;
  $("dram").value = s.dram_key;
  $("disk").value = s.disk_key;
  $("link").value = s.link_key;
  $("gpuCount").value = s.gpu_count;
  $("dramCount").value = s.dram_count;
  updateHwSummary();
}

function applyModelPreset() {
  const key = $("modelPreset").value;
  if (!key || !CATALOG.models[key]) return;
  const m = CATALOG.models[key];
  $("modelParams").value = m.params / 1e9;
  $("tokensPerBlock").value = m.tokens_per_block;
  $("bytesPerBlock").value = m.bytes_per_block / 1e6;
  $("gpuEta").value = m.gpu_eta;
}

function updateHitRatio() {
  const total = +$("totalBlocks").value;
  const hr = +$("hitRatio").value;
  $("hitRatioVal").textContent = hr + "%";
  const miss = Math.round(total * (100 - hr) / 100);
  $("missHint").textContent = `${(total - miss).toLocaleString()} hit · ${miss.toLocaleString()} miss blocks`;
}

function normalizeManual() {
  const v = +$("vram").value, d = +$("dramFrac").value, k = +$("diskFrac").value;
  const t = v + d + k || 1;
  $("vramVal").textContent = Math.round(100 * v / t) + "%";
  $("dramVal").textContent = Math.round(100 * d / t) + "%";
  $("diskVal").textContent = Math.round(100 * k / t) + "%";
}

function placementSpec() {
  const mode = document.querySelector(".seg.active").dataset.mode;
  return {
    mode,
    vram: +$("vram").value,
    dram: +$("dramFrac").value,
    disk: +$("diskFrac").value,
  };
}

function hardwareSpec() {
  return {
    gpu_key: $("gpu").value,
    dram_key: $("dram").value,
    disk_key: $("disk").value,
    link_key: $("link").value,
    gpu_count: +$("gpuCount").value,
    dram_count: +$("dramCount").value,
  };
}

function workloadSpec() {
  const total = +$("totalBlocks").value;
  const miss = Math.round(total * (100 - +$("hitRatio").value) / 100);
  return {
    total_blocks: total,
    miss_blocks: miss,
    request_rate: +$("requestRate").value,
    p95_seconds: +$("p95").value,
    model_params: (+$("modelParams").value) * 1e9,
    tokens_per_block: +$("tokensPerBlock").value,
    bytes_per_block: (+$("bytesPerBlock").value) * 1e6,
    gpu_eta: +$("gpuEta").value,
  };
}

// ------------------------------------------------------------------ simulate
async function runSimulation() {
  const status = $("status");
  status.className = "status";
  status.textContent = "Running…";
  try {
    const res = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hardware: hardwareSpec(),
        workload: workloadSpec(),
        placement: placementSpec(),
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderSingle(await res.json());
    status.textContent = "";
  } catch (e) {
    status.className = "status error";
    status.textContent = "Error: " + e.message;
  }
}

function renderSingle(data) {
  $("singlePlaceholder").classList.add("hidden");
  $("singleResults").classList.remove("hidden");

  $("mX").textContent = fmtRate(data.storage_throughput);
  $("mY").textContent = fmtRate(data.recompute_throughput);
  const ref = data.results[POLICIES[0].key];
  $("mHM").textContent = `${fmtInt(ref.hit_blocks)} / ${fmtInt(ref.miss_blocks)}`;
  $("mFits").textContent = data.capacity_fits ? "✓ fits" : "✗ overflow";
  $("mFits").style.color = data.capacity_fits ? "var(--green)" : "var(--red)";

  renderTiers(data.tiers);
  renderPolicyCards(data);
  renderLatencyChart(data);
  renderBreakdownChart(data);
}

function renderTiers(tiers) {
  const host = $("tierBars");
  host.innerHTML = "";
  tiers.forEach((t) => {
    const cap = num(t.capacity_blocks);
    const resident = num(t.resident_blocks);
    const overflow = num(t.overflow_blocks);
    const pct = cap > 0 ? Math.min(100, 100 * resident / cap) : (resident > 0 ? 100 : 0);
    const overPct = cap > 0 ? Math.min(100, 100 * overflow / cap) : 0;
    const el = document.createElement("div");
    el.className = "tier";
    el.innerHTML = `
      <div class="tier-head">
        <span class="tname">${t.label}</span>
        <span class="tmeta">${fmtBlocks(resident)} / ${cap > 0 ? fmtBlocks(cap) : "—"} · X=${fmtRate(t.restore_throughput_blocks_per_s)} blk/s</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${overflow > 0 ? "over" : t.tier}" style="width:${overflow > 0 ? overPct : pct}%"></div>
      </div>`;
    host.appendChild(el);
  });
}

function renderPolicyCards(data) {
  const host = $("policyCards");
  host.innerHTML = "";
  host.style.gridTemplateColumns = `repeat(${Math.min(POLICIES.length, 4)}, 1fr)`;
  POLICIES.forEach((p) => {
    const r = data.results[p.key];
    const meets = r.meets_slo;
    const card = document.createElement("div");
    card.className = "pcard";
    card.style.borderTopColor = p.color;
    card.innerHTML = `
      <h4>${p.label}</h4>
      <div class="policy-sub">${p.sub}</div>
      <div class="big">${fmtTime(r.total_time)}<small> latency</small></div>
      <span class="badge ${meets ? "pass" : "fail"}">${meets ? "MEETS SLO" : "VIOLATES SLO"}</span>
      <div class="kv"><span>Restore blocks</span><b>${fmtInt(r.storage_blocks)}</b></div>
      <div class="kv"><span>Recompute blocks</span><b>${fmtInt(r.recompute_blocks)}</b></div>
      <div class="kv"><span>Reassigned k</span><b>${fmtInt(r.reassigned_hit_blocks)}</b></div>
      <div class="kv"><span>GPU util</span><b>${(num(r.gpu_utilization) * 100).toFixed(0)}%</b></div>
      <div class="kv"><span>Storage util</span><b>${(num(r.storage_utilization) * 100).toFixed(0)}%</b></div>
      <div class="kv"><span>SLO margin</span><b>${fmtTime(Math.abs(num(r.slo_margin)))} ${num(r.slo_margin) >= 0 ? "spare" : "over"}</b></div>`;
    host.appendChild(card);
  });
}

function destroyChart(name) {
  if (charts[name]) { charts[name].destroy(); delete charts[name]; }
}

const CHART_GRID = "#2a3348";
const CHART_TEXT = "#8a95ad";

function renderLatencyChart(data) {
  destroyChart("latency");
  const labels = POLICIES.map((p) => p.label);
  const values = POLICIES.map((p) => {
    const t = num(data.results[p.key].total_time);
    return isFinite(t) ? t : 0;
  });
  const slo = num(data.results[POLICIES[0].key].p95_seconds);
  charts.latency = new Chart($("latencyChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Latency (s)",
        data: values,
        backgroundColor: POLICIES.map((p) => p.color),
        borderRadius: 5,
      }],
    },
    options: {
      plugins: {
        legend: { display: false },
        annotation: false,
        tooltip: { callbacks: { label: (c) => fmtTime(c.raw) } },
      },
      scales: {
        y: {
          beginAtZero: true,
          grid: { color: CHART_GRID },
          ticks: { color: CHART_TEXT, callback: (v) => fmtTime(v) },
          title: { display: true, text: `SLO = ${fmtTime(slo)}`, color: CHART_TEXT },
        },
        x: { grid: { display: false }, ticks: { color: CHART_TEXT, font: { size: 10 } } },
      },
      // Draw SLO reference line.
      animation: false,
    },
    plugins: [{
      id: "sloLine",
      afterDraw(chart) {
        const y = chart.scales.y.getPixelForValue(slo);
        if (!isFinite(y)) return;
        const { ctx, chartArea } = chart;
        ctx.save();
        ctx.strokeStyle = "#f0c000";
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(chartArea.left, y);
        ctx.lineTo(chartArea.right, y);
        ctx.stroke();
        ctx.restore();
      },
    }],
  });
}

function renderBreakdownChart(data) {
  destroyChart("breakdown");
  const labels = POLICIES.map((p) => p.label);
  charts.breakdown = new Chart($("breakdownChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Restore time",
          data: POLICIES.map((p) => { const v = num(data.results[p.key].storage_time); return isFinite(v) ? v : 0; }),
          backgroundColor: "#fab333",
          borderRadius: 4,
        },
        {
          label: "Recompute time",
          data: POLICIES.map((p) => { const v = num(data.results[p.key].recompute_time); return isFinite(v) ? v : 0; }),
          backgroundColor: "#7c5cff",
          borderRadius: 4,
        },
      ],
    },
    options: {
      animation: false,
      plugins: {
        legend: { labels: { color: CHART_TEXT, font: { size: 11 } } },
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${fmtTime(c.raw)}` } },
      },
      scales: {
        y: { beginAtZero: true, grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT, callback: (v) => fmtTime(v) } },
        x: { grid: { display: false }, ticks: { color: CHART_TEXT, font: { size: 10 } } },
      },
    },
  });
}

// ------------------------------------------------------------------ sweep
async function runSweep() {
  const status = $("sweepStatus");
  status.className = "status";
  status.textContent = "Running sweep…";
  try {
    const total = +$("totalBlocks").value;
    const res = await fetch("/api/sweep", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        hardware: hardwareSpec(),
        placement: placementSpec(),
        total_blocks: total,
        hit_ratio: +$("hitRatio").value,
        model_params: (+$("modelParams").value) * 1e9,
        tokens_per_block: +$("tokensPerBlock").value,
        bytes_per_block: (+$("bytesPerBlock").value) * 1e6,
        gpu_eta: +$("gpuEta").value,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    renderSweep(await res.json());
    status.textContent = "";
  } catch (e) {
    status.className = "status error";
    status.textContent = "Error: " + e.message;
  }
}

function renderSweep(data) {
  $("sweepResults").classList.remove("hidden");

  $("sweepLegend").innerHTML = [
    ["pass", "Pass (SLO met)"],
    ["latency", "RPC-bound only"],
    ["rpc", "P95-bound only"],
    ["fail", "Fail (both)"],
  ].map(([k, t]) => `<span class="item"><span class="swatch" style="background:${HEAT[k]}"></span>${t}</span>`).join("");

  const host = $("heatmaps");
  host.innerHTML = "";
  host.style.gridTemplateColumns = `repeat(${Math.min(POLICIES.length, 3)}, 1fr)`;
  POLICIES.forEach((p) => {
    const box = document.createElement("div");
    box.className = "heatmap-box";
    box.innerHTML = `<h4 style="color:${p.color}">${p.label}</h4>
      <div class="hsub">P95 (y) vs request rate (x)</div>
      <canvas class="grid-canvas" id="heat_${p.key}"></canvas>
      <div class="axis-x">request rate → ${data.request_rates[0]} … ${data.request_rates[data.request_rates.length - 1]} req/s</div>`;
    host.appendChild(box);
    drawHeatmap(`heat_${p.key}`, data.grids[p.key]);
  });

  renderSuccessChart(data);
}

function drawHeatmap(canvasId, grid) {
  const rows = grid.length;
  const cols = grid[0].length;
  const cell = 14;
  const canvas = $(canvasId);
  canvas.width = cols * cell;
  canvas.height = rows * cell;
  const ctx = canvas.getContext("2d");
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const st = grid[r][c];
      let color;
      if (st.rpc_pass && st.latency_pass) color = HEAT.pass;
      else if (st.rpc_pass && !st.latency_pass) color = HEAT.rpc;
      else if (!st.rpc_pass && st.latency_pass) color = HEAT.latency;
      else color = HEAT.fail;
      ctx.fillStyle = color;
      // origin lower: smallest p95 (row 0) at the bottom.
      ctx.fillRect(c * cell, (rows - 1 - r) * cell, cell - 0.5, cell - 0.5);
    }
  }
}

function renderSuccessChart(data) {
  destroyChart("success");
  charts.success = new Chart($("successChart"), {
    type: "line",
    data: {
      labels: data.cache_ratio_sweep,
      datasets: POLICIES.map((p) => ({
        label: p.label,
        data: data.success_rates[p.key],
        borderColor: p.color,
        backgroundColor: p.color,
        tension: 0.25,
        pointRadius: 3,
      })),
    },
    options: {
      animation: false,
      plugins: { legend: { labels: { color: CHART_TEXT, font: { size: 11 } } } },
      scales: {
        y: { min: 0, max: 100, grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT, callback: (v) => v + "%" }, title: { display: true, text: "Success rate", color: CHART_TEXT } },
        x: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT }, title: { display: true, text: "Cache hit ratio (%)", color: CHART_TEXT } },
      },
    },
  });
}

// ------------------------------------------------------------------ wiring
function initEvents() {
  // Tabs
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const single = tab.dataset.tab === "single";
      $("singleView").classList.toggle("hidden", !single);
      $("sweepView").classList.toggle("hidden", single);
    });
  });

  // Placement mode
  document.querySelectorAll(".seg").forEach((seg) => {
    seg.addEventListener("click", () => {
      document.querySelectorAll(".seg").forEach((s) => s.classList.remove("active"));
      seg.classList.add("active");
      $("manualPlacement").classList.toggle("hidden", seg.dataset.mode !== "manual");
    });
  });

  $("stackPreset").addEventListener("change", applyStackPreset);
  $("modelPreset").addEventListener("change", applyModelPreset);
  ["gpu", "dram", "disk", "link", "gpuCount", "dramCount"].forEach((id) =>
    $(id).addEventListener("change", updateHwSummary));
  $("hitRatio").addEventListener("input", updateHitRatio);
  $("totalBlocks").addEventListener("input", updateHitRatio);
  ["vram", "dramFrac", "diskFrac"].forEach((id) => $(id).addEventListener("input", normalizeManual));
  $("runBtn").addEventListener("click", runSimulation);
  $("sweepBtn").addEventListener("click", runSweep);
}

async function main() {
  initEvents();
  await loadCatalog();
  updateHitRatio();
  normalizeManual();
  runSimulation();
}

main();
