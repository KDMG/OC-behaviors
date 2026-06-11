"use strict";

if (typeof cytoscape !== "undefined" &&
    typeof cytoscapeDagre !== "undefined") {
  cytoscape.use(cytoscapeDagre);
}

const STATE = {
  summary: null,
  lattice: null,
  cfg_idx: null,
  behavior_idx: null,
  kpi: null,
  picker: {
    options: null,
    choice:  {},
  },
};


function $(id) { return document.getElementById(id); }
function escapeHtml(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
async function api(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return await res.json();
}
function fmtNum(v, dp) {
  if (v == null || isNaN(v)) return "—";
  const d = dp === undefined ? 3 : dp;
  return Number(v).toLocaleString(undefined, {
    maximumFractionDigits: d, minimumFractionDigits: 0,
  });
}
function fmtPct(v, dp) {
  if (v == null || isNaN(v)) return "—";
  return (v * 100).toFixed(dp === undefined ? 1 : dp) + "%";
}


function switchTab(name) {
  document.querySelectorAll(".topbar .tabs .tab").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll("section.panel").forEach(p =>
    p.classList.toggle("hidden", p.id !== `tab-${name}`));
  if (name === "behaviors") {
    const hasConfig = STATE.cfg_idx != null;
    $("behaviors-no-cfg").classList.toggle("hidden", hasConfig);
    $("behaviors-main").classList.toggle("hidden", !hasConfig);
  }
}

async function boot() {
  document.querySelectorAll(".topbar .tabs .tab").forEach(b =>
    b.addEventListener("click", () => switchTab(b.dataset.tab)));
  $("modal-close").addEventListener("click", () => closeModal());
  document.querySelector(".modal-backdrop")
    .addEventListener("click", () => closeModal());
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") closeModal();
  });

  STATE.summary = await api("/api/summary");
  STATE.kpi     = STATE.summary.primary_kpi;

  const fname = (STATE.summary.ocel_path || "").split("/").pop()
              || STATE.summary.ocel_path || "";
  $("meta").innerHTML =
      `<span class="pill"><code>${escapeHtml(fname)}</code></span>`
    + `<span class="pill">${STATE.summary.n_executions} executions</span>`
    ;



  STATE.lattice = await api("/api/lattice");
  renderLattice(STATE.lattice);

  STATE.picker.options = await api("/api/lattice/options");
  STATE.picker.choice = {};
  renderPicker();
}

function fdChain(tau, L, lat) {
  const names = (lat.level_names || {})[tau] || [];
  const covers = (lat.level_covers || {})[tau] || {};
  if (L === 0) return ["id"];
  function findPath(from, target, visited) {
    if (from === target) return [from];
    const nexts = (covers[from] || covers[String(from)] || []);
    for (const n of nexts) {
      if (visited.has(n)) continue;
      visited.add(n);
      const sub = findPath(n, target, visited);
      if (sub) return [from, ...sub];
    }
    return null;
  }
  const path = findPath(0, L, new Set([0]));
  if (!path) return [names[L] || `lv${L}`];
  return path.slice(1).map(lv => names[lv] || `lv${lv}`);
}



const TYPE_COLORS = [
  "#6366f1", "#10b981", "#f59e0b", "#ef4444",
  "#06b6d4", "#a855f7", "#84cc16", "#ec4899",
];

function colorFor(typeIndex) {
  return TYPE_COLORS[typeIndex % TYPE_COLORS.length];
}

/* Vertical layer for a cfg = sum over types of the BFS depth of
   cfg[i] from level 0 in the per-type FD-DAG.  This produces a Hasse
   diagram with id-everywhere at the bottom (drill-down) and
   type-everywhere at the top (max roll-up). */
function layerOf(cfg, types, level_covers) {
  let total = 0;
  types.forEach((tau, ti) => {
    const adj = level_covers[tau] || {};
    const target = cfg[ti];
    const depth = new Map([[0, 0]]);
    const queue = [0];
    while (queue.length) {
      const u = queue.shift();
      const next = adj[u] || [];
      for (const v of next) {
        const d = depth.get(u) + 1;
        if (!depth.has(v) || d > depth.get(v)) {
          depth.set(v, d);
          queue.push(v);
        }
      }
    }
    total += (depth.get(target) || 0);
  });
  return total;
}

/* Cover edges in the *useful* sub-lattice: for every cfg, for every
   type τ, BFS down level_covered_by[τ] until we hit another cfg
   present in the lattice — that's a cover edge.  This handles
   skipped (uninteresting) cfgs by allowing edges that span multiple
   levels along τ. */
function computeCoverEdges(rows, types, lcb) {
  const byKey = new Map();
  rows.forEach(r => byKey.set(r.cfg.join(","), r));
  const edges = [];
  rows.forEach(r => {
    const cfg = r.cfg;
    types.forEach((tau, ti) => {
      const adj = lcb[tau] || {};
      const start = cfg[ti];
      const seen = new Set([start]);
      const queue = [...(adj[start] || [])];
      queue.forEach(v => seen.add(v));
      const found = new Set();
      while (queue.length) {
        const L = queue.shift();
        const next = cfg.slice(); next[ti] = L;
        const k2 = next.join(",");
        if (byKey.has(k2)) {
          if (!found.has(k2)) {
            found.add(k2);
            edges.push({
              src: cfg.join(","),
              dst: k2,
              tau: tau, ti: ti,
            });
          }
        } else {
          (adj[L] || []).forEach(v => {
            if (!seen.has(v)) { seen.add(v); queue.push(v); }
          });
        }
      }
    });
  });
  return edges;
}

function renderLattice(lat) {
  const types = lat.types || [];
  const allRows = (lat.rows || []);
  const rows = allRows.filter(r => r.is_interesting && r.n_abstractions > 0);
  const nDropped = allRows.length - rows.length;

  if (rows.length === 0) {
    $("lattice-cy").innerHTML = "";
    $("lattice-empty").classList.remove("hidden");
    return;
  }
  $("lattice-empty").classList.add("hidden");

  $("lattice-legend").innerHTML = types.map((t, i) =>
    `<span><span class="swatch" style="background:${
      colorFor(i)};"></span>${escapeHtml(t)}</span>`
  ).join("") +
  `<span class="lattice-h-legend"><span class="lattice-h-swatch"></span>` +
  `<span class="muted">darker border = higher unique abstractions; h = normalised abstraction height (0%=granular, 100%=coarser)</span></span>`;

  const lvc = lat.level_covers || {};
  rows.forEach(r => { r._layer = layerOf(r.cfg, types, lvc); });
  const byLayer = new Map();
  rows.forEach(r => {
    if (!byLayer.has(r._layer)) byLayer.set(r._layer, []);
    byLayer.get(r._layer).push(r);
  });
  const layers = [...byLayer.keys()].sort((a, b) => a - b);
  const maxLayer = layers[layers.length - 1] || 0;
  const widestRow = Math.max(...layers.map(L => byLayer.get(L).length));


  const NODE_W = 72;
  const NODE_H = 72;
  const X_GAP  = 170;
  const Y_GAP  = 190;
  const ROW_W  = Math.max(700, widestRow * (NODE_W + X_GAP));
  const MARGIN_X = 70;
  const MARGIN_Y = 90;
  const NATURAL_H = (maxLayer + 1) * Y_GAP + 2 * MARGIN_Y;

  // Sort each layer for stable, deterministic placement (lexicographic
  // by cfg vector → mirrors ui_dash, where "neighbouring" cfgs sit close).
  layers.forEach(L => {
    byLayer.get(L).sort((a, b) => {
      for (let i = 0; i < a.cfg.length; i++) {
        if (a.cfg[i] !== b.cfg[i]) return a.cfg[i] - b.cfg[i];
      }
      return 0;
    });
  });

  rows.forEach(r => {
    const layer = byLayer.get(r._layer);
    const i = layer.indexOf(r);
    const n = layer.length;
    r._x = (i + 0.5) / n * ROW_W + MARGIN_X;
    r._y = (maxLayer - r._layer) * Y_GAP + MARGIN_Y;
  });

  // node payloads
  // border darkness = n_abstractions_kept / n_executions, normalised across nodes
  // lower ratio → darker border (fewer distinct behaviors per execution)
  function borderColor(t) {
    // t=0 (low ratio) → dark slate #1e293b ; t=1 (high ratio) → light gray #cbd5e1
    const lo = [30, 41, 59], hi = [203, 213, 225];
    const rv = Math.round(lo[0] + t * (hi[0] - lo[0]));
    const gv = Math.round(lo[1] + t * (hi[1] - lo[1]));
    const bv = Math.round(lo[2] + t * (hi[2] - lo[2]));
    return `rgb(${rv},${gv},${bv})`;
  }
  const ratios = rows.map(r => r.n_executions > 0 ? r.K / r.n_executions : 0);
  const minRatio = Math.min(...ratios);
  const maxRatio = Math.max(...ratios);
  const ratioSpan = maxRatio - minRatio || 1;
  const nodes = rows.map((r, i) => {
    const h = maxLayer > 0 ? r._layer / maxLayer : 0;
    const hPct = Math.round(h * 100);
    const tRatio = (ratios[i] - minRatio) / ratioSpan;
    const bw = 2 + Math.round((1 - tRatio) * 4);
    return {
      data: {
        id: r.cfg.join(","),
        cfg: r.cfg,
        cfg_idx: r.cfg_idx,
        cfg_str: r.cfg_str,
        K: r.K,
        n_abstractions: r.n_abstractions,
        label: `${r.K} abs · ${r.n_abstractions_kept} beh\nh=${hPct}%`,
        border_color: borderColor(tRatio),
        border_width: bw,
      },
      position: { x: r._x, y: r._y },
    };
  });

  const lcb = lat.level_covered_by || {};
  const covers = computeCoverEdges(rows, types, lcb);
  const edges = covers.map((e, i) => ({
    data: {
      id: "e" + i,
      source: e.dst,
      target: e.src,
      tau: e.tau,
      color: colorFor(types.indexOf(e.tau)),
    }
  }));

  const styles = [
    { selector: "node", style: {
        "shape": "ellipse",
        "background-color": "#eef2ff",
        "border-color": "data(border_color)", "border-width": "data(border_width)",
        "width": NODE_W, "height": NODE_H,
        "label": "data(label)",
        "color": "#374151",
        "text-valign": "center", "text-halign": "center",
        "font-size": "10px", "font-weight": 600,
        "font-family": "ui-monospace, monospace",
        "text-wrap": "wrap", "text-max-width": 65,
        "line-height": 1.35,
    }},
    { selector: "edge", style: {
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "target-arrow-color": "data(color)",
        "line-color": "data(color)",
        "width": 1.6,
        "opacity": 0.7,
    }},
    { selector: "node.selected", style: {
        "border-color": "#1d4ed8",
        "border-width": 4,
        "background-color": "#dbeafe",
    }},
    { selector: "node.picker-match", style: {
        "border-color": "#f97316",
        "border-width": 5,
        "background-color": "#ffedd5",
        "color": "#9a3412",
    }},
    { selector: "node.picker-dim", style: {
        "opacity": 0.35,
    }},
    { selector: "edge.picker-dim", style: {
        "opacity": 0.2,
    }},
  ];

  const cy = cytoscape({
    container: $("lattice-cy"),
    elements: [...nodes, ...edges],
    style: styles,
    layout: {
      name: "preset",
      fit: true,
      padding: 28,
    },
    wheelSensitivity: 0.2,
    minZoom: 0.2, maxZoom: 3,
    boxSelectionEnabled: false,
    autoungrabify: true,
  });
  cy.ready(() => cy.fit(undefined, 28));

  let _ttEl = null;
  function _getTooltip() {
    if (_ttEl) return _ttEl;
    _ttEl = document.createElement("div");
    _ttEl.className = "lattice-tooltip hidden";
    $("lattice-cy").parentElement.appendChild(_ttEl);
    return _ttEl;
  }
  cy.on("mouseover", "node", (ev) => {
    const cfg = ev.target.data("cfg") || [];
    const tt = _getTooltip();
    tt.innerHTML = types.map((tau, i) => {
      const L = cfg[i];
      const nm = (lat.level_names[tau] || [])[L] || `lv${L}`;
      return `<div><span class="tt-swatch" style="background:${colorFor(i)}"></span><b>${escapeHtml(tau)}</b>: ${escapeHtml(nm)}</div>`;
    }).join("");
    const rbb = ev.target.renderedBoundingBox({includeLabels: false});
    const cont = $("lattice-cy");
    tt.style.left = (rbb.x2 + 6) + "px";
    tt.style.top  = (rbb.y1 + cont.scrollTop) + "px";
    tt.classList.remove("hidden");
  });
  cy.on("mouseout tap", "node", () => { const tt = _getTooltip(); tt.classList.add("hidden"); });

  cy.on("tap", "node", (ev) => {
    const ci = ev.target.data("cfg_idx");
    const cfg = ev.target.data("cfg") || [];
    cy.elements("node").removeClass("selected");
    ev.target.addClass("selected");
    if (STATE.picker.options && STATE.picker.options.types) {
      const types = STATE.picker.options.types;
      const choice = {};
      types.forEach((t, i) => { if (cfg[i] != null) choice[t] = cfg[i]; });
      STATE.picker.choice = choice;
      renderPicker();
    }
    if (ci != null) selectCfg(ci);
  });

  STATE.cy_lattice = cy;
}

function toDays(v, unit) {
  if (v == null) return null;
  if (unit === "s" || unit === "second" || unit === "seconds") return v / 86400;
  return v;
}
function fmtDays(v) {
  if (v == null) return "—";
  return v.toFixed(2);
}

const ST = {
  data: null,
  sortCol: null,
  sortAsc: true,
  filterMin: {},
  filterMax: {},
  filterSupportMin: 0,
  filterSupportMax: 1,
};

async function selectCfg(ci) {
  STATE.cfg_idx = ci;
  STATE.behavior_idx = null;
  await loadSituation(ci);
  switchTab("behaviors");
}

async function loadSituation(ci) {
  const r = await api(`/api/cfg/${ci}/situation`);
  ST.data = r;
  ST.sortCol = null;
  ST.sortAsc = true;
  ST.filterMin = {};
  ST.filterMax = {};
  ST.filterSupportMin = 0;
  ST.filterSupportMax = 1;
  (function() {
    const el = $("behaviors-cfg");
    if (!el) return;
    const lat = STATE.lattice;
    const types = (lat && lat.types) ? lat.types : [];
    if (!types.length || !r.cfg_str) { el.textContent = r.cfg_str || ""; return; }
    const parts = r.cfg_str.split(/\s*\|\s*/);
    const map = {};
    parts.forEach(p => { const [t, l] = p.split(":"); if (t && l) map[t.trim()] = l.trim(); });
    el.innerHTML = types.map((t, ti) => {
      const nm = map[t] || "?";
      return `<span class="cfg-chip-colored" style="background:${colorFor(ti)};color:#fff">${escapeHtml(t)}: <strong>${escapeHtml(nm)}</strong></span>`;
    }).join("");
  })();
  $("behaviors-subsume-note").textContent = "";
  renderSituationTable();
}

function renderSituationTable() {
  const container = $("behaviors-list");
  const d = ST.data;
  if (!d || !d.rows.length) {
    container.innerHTML = "";
    $("behaviors-empty").classList.remove("hidden");
    return;
  }
  $("behaviors-empty").classList.add("hidden");

  const timeKpi = d.kpi_meta.find(k => ["s","second","seconds"].includes(k.unit));
  const timeKey = timeKpi ? timeKpi.name : null;
  const timeUnit = timeKpi ? timeKpi.unit : null;

  let rows = d.rows.filter(row => {
    for (const [col, min] of Object.entries(ST.filterMin)) {
      const v = col === "_time" ? toDays(row[timeKey], timeUnit) : row[col];
      if (v != null && v < min) return false;
    }
    for (const [col, max] of Object.entries(ST.filterMax)) {
      const v = col === "_time" ? toDays(row[timeKey], timeUnit) : row[col];
      if (v != null && v > max) return false;
    }
    // support filter: hide rows that have NO behavior with support >= threshold
    if (ST.filterSupportMin > 0) {
      const hasSup = d.behavior_meta.some(b =>
        b.support >= ST.filterSupportMin && b.support <= ST.filterSupportMax && row[`b${b.behavior_idx}`]
      );
      if (!hasSup) return false;
    }
    return true;
  });

  // Sort
  if (ST.sortCol) {
    rows = [...rows].sort((a, b) => {
      let va = ST.sortCol === "_time" ? toDays(a[timeKey], timeUnit) : a[ST.sortCol];
      let vb = ST.sortCol === "_time" ? toDays(b[timeKey], timeUnit) : b[ST.sortCol];
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
      return ST.sortAsc ? va - vb : vb - va;
    });
  }

  // Compute ranges for sliders
  const allTimes  = timeKey ? d.rows.map(r => toDays(r[timeKey], timeUnit)).filter(v => v != null) : [];
  const allEvents = d.rows.map(r => r.n_events).filter(v => v != null);
  const allSupports = d.behavior_meta.map(b => b.support);
  const timeMin   = allTimes.length  ? Math.floor(Math.min(...allTimes))   : 0;
  const timeMax   = allTimes.length  ? Math.ceil(Math.max(...allTimes))    : 1;
  const evMin     = allEvents.length ? Math.min(...allEvents)              : 0;
  const evMax     = allEvents.length ? Math.max(...allEvents)              : 1;
  const supMin    = allSupports.length ? Math.min(...allSupports)          : 0;
  const supMax    = allSupports.length ? Math.max(...allSupports)          : 1;
  const curTimeMin  = ST.filterMin["_time"]    ?? timeMin;
  const curTimeMax  = ST.filterMax["_time"]    ?? timeMax;
  const curEvMin    = ST.filterMin["n_events"] ?? evMin;
  const curEvMax    = ST.filterMax["n_events"] ?? evMax;
  const curSupMin   = ST.filterSupportMin;
  const curSupMax   = ST.filterSupportMax ?? supMax;

  // Build toolbar
  const sortArrow = (col) => ST.sortCol === col ? (ST.sortAsc ? " ▲" : " ▼") : "";
  const dualSlider = (label, id, min, max, valLo, valHi, step="1") => `
    <div class="sit-slider-group">
      <div class="sit-slider-header">
        <span class="sit-slider-label">${label}</span>
        <span class="sit-slider-vals" id="${id}-vals">${valLo} – ${valHi}</span>
      </div>
      <div class="ds-track" id="${id}-track" data-min="${min}" data-max="${max}" data-step="${step}" data-mode="dual">
        <div class="ds-bar"></div>
        <div class="ds-fill" id="${id}-fill"></div>
        <div class="ds-thumb" id="${id}-tlo" data-role="lo" tabindex="0"></div>
        <div class="ds-thumb" id="${id}-thi" data-role="hi" tabindex="0"></div>
      </div>
    </div>`;

  const minSlider = (label, id, min, max, valLo, step="1") => `
    <div class="sit-slider-group">
      <div class="sit-slider-header">
        <span class="sit-slider-label">${label}</span>
        <span class="sit-slider-vals" id="${id}-vals">≥ ${valLo}</span>
      </div>
      <div class="ds-track" id="${id}-track" data-min="${min}" data-max="${max}" data-step="${step}" data-mode="min">
        <div class="ds-bar"></div>
        <div class="ds-fill" id="${id}-fill" style="left:0"></div>
        <div class="ds-thumb" id="${id}-tlo" data-role="lo" tabindex="0"></div>
      </div>
    </div>`;

  const toolbarHtml = `
    <div class="sit-toolbar">
      <div class="export-btns">
        <button class="btn-export" id="sit-export">Export CSV</button>
        <button class="btn-export btn-export-zip" id="sit-export-zip">Export all (ZIP)</button>
      </div>
    </div>`;

  // Build table
  const behaviorCols = d.behavior_meta;
  $("behaviors-count").textContent = `${d.n_executions} executions · ${behaviorCols.length} behaviors`;
  let thead = `<tr>
    <th class="sit-th-num">PE</th>
    ${timeKey ? `<th class="sit-th-num sortable" data-sort="_time">Completion time (days)${sortArrow("_time")}</th>` : ""}
    <th class="sit-th-num sortable" data-sort="n_events"># Events${sortArrow("n_events")}</th>
    ${behaviorCols.map(b => {
      const sign = b.signed_delta >= 0 ? "+" : "-";
      const cls  = b.signed_delta >= 0 ? "beh-th-up" : "beh-th-down";
      const daysDelta = timeKey ? toDays(b.delta, timeUnit) : b.delta;
      return `<th class="sit-th-beh ${cls}" data-bi="${b.behavior_idx}" title="support ${b.support} · ${sign}${fmtDays(daysDelta)} d">B<sub>${b.behavior_idx}</sub></th>`;
    }).join("")}
  </tr>`;

  let tbody = rows.map(row => {
    const timeDays = timeKey ? toDays(row[timeKey], timeUnit) : null;
    return `<tr>
      <td class="sit-td-num">P<sub>${row.exec_idx}</sub></td>
      ${timeKey ? `<td class="sit-td-num">${fmtDays(timeDays)}</td>` : ""}
      <td class="sit-td-num">${row.n_events}</td>
      ${behaviorCols.map(b => {
        const has = row[`b${b.behavior_idx}`];
        return `<td class="sit-td-beh ${has ? "sit-yes" : "sit-no"}">${has ? "Y" : "N"}</td>`;
      }).join("")}
    </tr>`;
  }).join("");

  // Legend describing relevance functions — shown once above the table
  const legendHtml = `
    <div class="sit-legend">
      <div class="sit-legend-title">Relevance functions <span class="sit-legend-subtitle">of a behavior B w.r.t. a set of process executions PEs</span></div>
      <div class="sit-legend-defs">
        <div class="sit-legend-def">
          <span class="sit-legend-lhs">S(B, PEs)</span>
          <span class="sit-legend-deftext">percentage of PEs where B occurs</span>
        </div>
        <div class="sit-legend-def">
          <span class="sit-legend-lhs">CT(B, PEs)</span>
          <span class="sit-legend-deftext">mean completion time of the PEs where B occurs</span>
        </div>
        <div class="sit-legend-def">
          <span class="sit-legend-lhs"><span class="sit-overline">CT</span>(B, PEs)</span>
          <span class="sit-legend-deftext">mean completion time of the PEs where B does not occur</span>
        </div>
      </div>
      <div class="sit-legend-hint">Click a column header B<sub>i</sub> to inspect behavior relevance.</div>
    </div>`;

  container.innerHTML = legendHtml + toolbarHtml + `
    <div class="sit-table-wrap">
      <table class="sit-table">
        <thead>${thead}</thead>
        <tbody>${tbody}</tbody>
      </table>
    </div>`;

  // Sort click on headers
  container.querySelectorAll("th.sortable").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (ST.sortCol === col) ST.sortAsc = !ST.sortAsc;
      else { ST.sortCol = col; ST.sortAsc = true; }
      renderSituationTable();
    });
  });

  // Behavior header click → open graph modal
  container.querySelectorAll("th.sit-th-beh").forEach(th => {
    th.addEventListener("click", () => openBehaviorModal_byIdx(STATE.cfg_idx, +th.dataset.bi));
  });

  // Export
  $("sit-export").addEventListener("click", () => exportSituationCSV(rows, timeKey, timeUnit, behaviorCols));
  $("sit-export-zip").addEventListener("click", () => exportAllCSVsZip());
}

function buildCSVContent(d) {
  if (!d) return null;
  const timeKpi  = d.kpi_meta.find(k => ["s","second","seconds"].includes(k.unit));
  const timeKey  = timeKpi ? timeKpi.name : null;
  const timeUnit = timeKpi ? timeKpi.unit : null;
  const bCols    = d.behavior_meta;

  const headers = ["exec_idx"];
  if (timeKey) headers.push("time_days");
  headers.push("n_events");
  bCols.forEach(b => headers.push(`B${b.behavior_idx}`));

  const lines = [headers.join(",")];
  d.rows.forEach(row => {
    const cells = [row.exec_idx];
    if (timeKey) cells.push(fmtDays(toDays(row[timeKey], timeUnit)));
    cells.push(row.n_events);
    bCols.forEach(b => cells.push(row[`b${b.behavior_idx}`] ? "Y" : "N"));
    lines.push(cells.join(","));
  });
  return lines.join("\n");
}

async function exportSituationCSV(rows, timeKey, timeUnit, behaviorCols) {
  const d    = ST.data;
  const csv  = buildCSVContent(d);
  const name = `behavior_cfg${d.cfg_idx}.csv`;
  const blob = new Blob([csv], { type: "text/csv" });

  if (window.showSaveFilePicker) {
    try {
      const fh = await window.showSaveFilePicker({
        suggestedName: name,
        types: [{ description: "CSV file", accept: { "text/csv": [".csv"] } }],
      });
      const w = await fh.createWritable();
      await w.write(blob);
      await w.close();
      return;
    } catch (e) {
      if (e.name === "AbortError") return; // user cancelled — do not export
    }
  }
  // Fallback: auto-download (API unavailable)
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

async function exportAllCSVsZip() {
  const btn = $("sit-export-zip");
  btn.disabled = true; btn.textContent = "Building ZIP…";

  if (!window.JSZip) {
    await new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = "https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js";
      s.onload = res; s.onerror = rej;
      document.head.appendChild(s);
    });
  }

  const zip  = new JSZip();
  const cfgs = (STATE.lattice && STATE.lattice.rows) ? STATE.lattice.rows : [];

  // Root-level index: one column per object type, semicolon-separated
  const types = (STATE.lattice && STATE.lattice.types) ? STATE.lattice.types : [];
  const idxLines = [["cfg_idx", ...types].join(";")];
  for (const row of cfgs) {
    const parts = (row.cfg_str || "").split(/\s*\|\s*/);
    const map = {};
    parts.forEach(p => { const i = p.indexOf(":"); if (i > 0) map[p.slice(0,i).trim()] = p.slice(i+1).trim(); });
    idxLines.push([row.cfg_idx, ...types.map(t => map[t] || "")].join(";"));
  }
  zip.file("configs.csv", idxLines.join("\n"));

  // One folder per cfg, with behavior table inside
  for (const row of cfgs) {
    try {
      const d   = await api(`/api/cfg/${row.cfg_idx}/situation`);
      const csv = buildCSVContent(d);
      const folder = `behavior_cfg${row.cfg_idx}/`;
      zip.file(folder + `behavior_cfg${row.cfg_idx}.csv`, csv);
    } catch(e) { /* skip */ }
  }

  const blob = await zip.generateAsync({ type: "blob" });
  const name = "behaviors_all.zip";

  if (window.showSaveFilePicker) {
    try {
      const fh = await window.showSaveFilePicker({
        suggestedName: name,
        types: [{ description: "ZIP archive", accept: { "application/zip": [".zip"] } }],
      });
      const w = await fh.createWritable();
      await w.write(blob);
      await w.close();
      btn.disabled = false; btn.textContent = "Export all (ZIP)";
      return;
    } catch(e) {
      if (e.name === "AbortError") { btn.disabled = false; btn.textContent = "Export all (ZIP)"; return; }
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
  btn.disabled = false; btn.textContent = "Export all (ZIP)";
}

async function openBehaviorModal_byIdx(ci, pi) {
  const p = await api(`/api/behavior/${ci}/${pi}?kpi=${encodeURIComponent(STATE.kpi)}`);
  openBehaviorModal(p);
}

function openBehaviorModal(p) {
  const unit = p.kpi_unit || "";
  const isDays = ["s","second","seconds"].includes(unit);
  const fmt = (v) => v == null ? "—" : isDays ? fmtDays(toDays(v, unit)) + " d" : fmtNum(v, 3) + (unit ? " " + unit : "");
  const lat  = STATE.lattice;
  const types = (lat && lat.types) ? lat.types : [];
  // cfg chips
  const parts = (p.cfg_str || "").split(/\s*\|\s*/);
  const cfgMap = {};
  parts.forEach(pt => { const [t, l] = pt.split(":"); if (t && l) cfgMap[t.trim()] = l.trim(); });
  // Render colored chips: prefer types order, fall back to cfg_str parse order
  const chipEntries = types.length
    ? types.map((t, ti) => [t, cfgMap[t] || "?", ti])
    : parts.map((pt, ti) => { const [t, l] = pt.split(":"); return [t ? t.trim() : pt, l ? l.trim() : "?", ti]; });
  const cfgChips = chipEntries.map(([t, nm, ti]) =>
    `<span class="cfg-chip-colored" style="background:${colorFor(ti)};color:#fff">${escapeHtml(t)}: <strong>${escapeHtml(nm)}</strong></span>`
  ).join("");

  // KPI comparison
  const inMean  = p.kpi_in  ? fmt(p.kpi_in.mean)  : "—";
  const outMean = p.kpi_out ? fmt(p.kpi_out.mean) : "—";
  const inN     = p.kpi_in  ? p.kpi_in.n  : 0;
  const outN    = p.kpi_out ? p.kpi_out.n : 0;
  const delta   = (p.kpi_in && p.kpi_out && p.kpi_in.mean != null && p.kpi_out.mean != null)
    ? (isDays ? fmtDays(toDays(p.kpi_in.mean - p.kpi_out.mean, unit)) + " d"
              : fmtNum(p.kpi_in.mean - p.kpi_out.mean, 3) + (unit ? " " + unit : ""))
    : null;
  const deltaSign = (p.kpi_in && p.kpi_out && p.kpi_in.mean != null && p.kpi_out.mean != null)
    ? (p.kpi_in.mean >= p.kpi_out.mean ? "delta-pos" : "delta-neg") : "";

  // Support as percentage (support is a count, divide by total executions)
  const totalExec = (p.n_in || 0) + (p.n_out || 0);
  const suppFrac = totalExec > 0 ? p.n_in / totalExec : (p.support || 0);
  const suppPct  = (suppFrac * 100).toFixed(1) + "%";
  const suppLabel = suppFrac >= 0.5 ? "High support" : suppFrac >= 0.2 ? "Medium support" : "Low support";
  const suppLabelCls = suppFrac >= 0.5 ? "supp-high" : suppFrac >= 0.2 ? "supp-mid" : "supp-low";

  // Human-readable relevance sentence
  const absDelta = (p.kpi_in && p.kpi_out && p.kpi_in.mean != null && p.kpi_out.mean != null)
    ? fmt(Math.abs(p.kpi_in.mean - p.kpi_out.mean)) : null;
  const isFaster = deltaSign === "delta-neg"; // lower CT with behavior = faster
  const sentenceHtml = absDelta != null
    ? '<div class="bmod-sentence ' + (isFaster ? "bmod-sentence-fast" : "bmod-sentence-slow") + '">Process executions containing this behavior are on average <strong>' + absDelta + (isFaster ? " faster" : " slower") + '</strong> than those without it.</div>'
    : "";

  const html = `
    <div class="bmod-header">
      <div class="bmod-title">Behavior B<sub>${p.behavior_idx}</sub></div>
      <div class="bmod-chips">${cfgChips}</div>

    </div>
    <div class="bmod-kpi-row">
      <div class="bmod-kpi-card bmod-kpi-supp">
        <div class="bmod-kpi-label">S</div>
        <div class="bmod-kpi-val">${suppPct}</div>
        <div class="bmod-kpi-n">#PEs = ${p.n_in}</div>
      </div>
      <div class="bmod-kpi-card bmod-kpi-in">
        <div class="bmod-kpi-label">CT</div>
        <div class="bmod-kpi-val">${inMean}</div>
        <div class="bmod-kpi-n">#PEs = ${inN}</div>
      </div>
      <div class="bmod-kpi-card bmod-kpi-out">
        <div class="bmod-kpi-label"><span style="text-decoration:overline">CT</span></div>
        <div class="bmod-kpi-val">${outMean}</div>
        <div class="bmod-kpi-n">#PEs = ${outN}</div>
      </div>
    </div>
    ${sentenceHtml}
    <div class="bmod-graph-section">
      <div id="behavior-cy"></div>
    </div>
  `;
  $("modal-content").innerHTML = html;
  $("modal").classList.remove("hidden");
  setTimeout(() => renderBehaviorGraph(p.graph), 30);
}

function renderBehaviorGraph(graph) {
  const el = $("behavior-cy");
  if (!el) return;
  const pairs = new Map();
  (graph.edges || []).forEach(e => {
    const k = e.data.source + ">" + e.data.target;
    pairs.set(k, (pairs.get(k) || 0) + 1);
  });
  const maxPair = Math.max(1, ...pairs.values());
  const cy = cytoscape({
    container: el,
    elements: [...(graph.nodes || []), ...(graph.edges || [])],
    style: [
      { selector: "node", style: {
          "shape": "round-rectangle", "background-color": "#1f2937",
          "color": "#fff", "label": "data(label)", "font-size": "12px",
          "text-valign": "center", "text-halign": "center",
          "text-wrap": "wrap", "text-max-width": 200,
          "width": 220, "height": 70, "padding": 10,
          "font-family": "ui-monospace, monospace", "line-height": 1.25,
      }},
      { selector: "edge", style: {
          "curve-style": "bezier", "control-point-step-size": 90,
          "control-point-distance": 50, "target-arrow-shape": "triangle",
          "target-arrow-color": "#475569", "line-color": "#475569",
          "width": 1.8, "label": "data(label)", "font-size": "11px",
          "color": "#0f172a", "text-background-color": "#fff",
          "text-background-opacity": 0.96, "text-background-padding": 4,
          "text-background-shape": "round-rectangle",
          "text-border-color": "#e4e7ee", "text-border-width": 1,
          "text-border-opacity": 1, "text-rotation": "autorotate",
          "text-margin-y": -2, "font-family": "ui-monospace, monospace",
      }},
      { selector: "edge[parallel_n > 1]", style: {
          "line-color": "#8b5cf6", "target-arrow-color": "#8b5cf6", "width": 2.4,
      }},
    ],
    layout: {
      name: "dagre", rankDir: "LR", nodeSep: 60, edgeSep: 30,
      rankSep: 130 + 30 * Math.min(4, maxPair - 1), ranker: "longest-path",
    },
    minZoom: 0.3, maxZoom: 3, wheelSensitivity: 0.2,
  });
  cy.ready(() => cy.fit(undefined, 24));
}

function closeModal() { $("modal").classList.add("hidden"); }


/* Compute which (tau, level) choices still lead to at least one useful cfg. */
function pickerAllowed() {
  const opts = STATE.picker.options;
  if (!opts) return {};
  const types = opts.types || [];
  const choice = STATE.picker.choice || {};
  const cfgs = (opts.useful_cfgs || []).map(c => c.cfg);

  const allowed = {};
  types.forEach(t => allowed[t] = new Set());
  for (let i = 0; i < types.length; i++) {
    const tau = types[i];
    for (const c of cfgs) {
      let ok = true;
      for (let j = 0; j < types.length; j++) {
        if (j === i) continue;
        const v = choice[types[j]];
        if (v != null && c[j] !== v) { ok = false; break; }
      }
      if (ok) allowed[tau].add(c[i]);
    }
  }
  return allowed;
}

/* Find the useful cfg matching the current selection. */
function pickerMatch() {
  const opts = STATE.picker.options;
  if (!opts) return null;
  const types = opts.types || [];
  const choice = STATE.picker.choice || {};
  const fixed = types.every(t => choice[t] != null);
  if (!fixed) return null;
  const target = types.map(t => choice[t]);
  return (opts.useful_cfgs || []).find(c =>
    c.cfg.length === target.length &&
    c.cfg.every((v, i) => v === target[i])
  ) || null;
}

function _prettyLevelShort(name) {
  if (name === "id")   return "id";
  if (name === "type") return "type";
  return name;
}

// Build Hasse diagram SVG for one object type using dagre for layout
function buildTypeHasseSvg(tau, ti, lat, choice, allowed) {
  const names   = (lat.level_names  || {})[tau] || [];
  const coversR = (lat.level_covers || {})[tau] || {};
  const k       = names.length - 1;
  const col     = colorFor(ti);
  const adjFn   = L => coversR[L] || coversR[String(L)] || [];
  const NH = 24, PAD = 12;
  const nw = L => Math.max(50, (names[L] || `lv${L}`).length * 7.5 + 22);

  // Build dagre graph
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 34, ranksep: 36, marginx: 0, marginy: 0 });
  g.setDefaultEdgeLabel(() => ({}));
  for (let L = 0; L <= k; L++)
    g.setNode(String(L), { width: nw(L), height: NH });
  for (let L = 0; L <= k; L++)
    adjFn(L).forEach(nL => g.setEdge(String(L), String(nL)));
  dagre.layout(g);

  {
    const inMap = {};
    g.edges().forEach(e => { if (!inMap[e.w]) inMap[e.w] = []; inMap[e.w].push(e.v); });
    const sources = new Set(g.nodes().filter(n => !(inMap[n] || []).length));
    const byX = g.nodes().slice().sort((a, b) => g.node(a).x - g.node(b).x);
    byX.forEach(n => {
      const preds = inMap[n] || [];
      if (!preds.length) return;
      if (preds.every(p => sources.has(p))) return;
      g.node(n).y = Math.min(...preds.map(p => g.node(p).y));
    });
    // Backward pass: snap source nodes to topmost successor (already snapped)
    sources.forEach(n => {
      const succs = g.successors(n) || [];
      if (succs.length) g.node(n).y = Math.min(...succs.map(s => g.node(s).y));
    });
  }

  // Bounding box → SVG size
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  g.nodes().forEach(n => {
    const nd = g.node(n);
    minX = Math.min(minX, nd.x - nd.width/2);
    maxX = Math.max(maxX, nd.x + nd.width/2);
    minY = Math.min(minY, nd.y - NH/2);
    maxY = Math.max(maxY, nd.y + NH/2);
  });
  const ox = PAD - minX, oy = PAD - minY;
  const SW = Math.ceil(maxX - minX + PAD*2);
  const SH = Math.ceil(maxY - minY + PAD*2);
  const mid = "arrh_" + tau.replace(/[^a-zA-Z0-9]/g,"_") + "_" + ti;

  // Edges: straight lines between snapped node centres
  // (dagre's pts are stale after y-snapping, so we use node.y directly)
  let edgeSvg = "";
  g.edges().forEach(e => {
    const sn = g.node(e.v), tn = g.node(e.w);
    const x1 = (sn.x + sn.width/2 + ox - 1).toFixed(1);
    const y1 = (sn.y              + oy).toFixed(1);
    const x2 = (tn.x - tn.width/2 + ox - 7).toFixed(1);
    const y2 = (tn.y              + oy).toFixed(1);
    edgeSvg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}"
      stroke="${col}" stroke-width="1.4" marker-end="url(#${mid})"/>`;
  });

  // Nodes
  let nodeSvg = "";
  for (let L = 0; L <= k; L++) {
    const nd = g.node(String(L)); if (!nd) continue;
    const nm    = names[L] || `lv${L}`;
    const isSel = choice[tau] === L;
    const isOk  = allowed.has(L);
    const fill   = isSel ? col    : (isOk ? "#fff"    : "#f9fafb");
    const stroke = isSel ? col    : (isOk ? col       : "#e5e7eb");
    const tc     = isSel ? "#fff" : (isOk ? "#1f2937" : "#9ca3af");
    const fw     = isSel ? "700"  : "500";
    const op     = isOk ? "1" : "0.4";
    const nx = nd.x+ox, ny = nd.y+oy;
    nodeSvg += `<g class="${isOk?"hasse-node":"hasse-node-off"}" data-tau="${tau}" data-lv="${L}"
                   style="cursor:${isOk?"pointer":"default"};opacity:${op}">
      <rect x="${(nx-nd.width/2).toFixed(1)}" y="${(ny-NH/2).toFixed(1)}"
            width="${nd.width.toFixed(1)}" height="${NH}" rx="5"
            fill="${fill}" stroke="${stroke}" stroke-width="1.5"/>
      <text x="${nx.toFixed(1)}" y="${ny.toFixed(1)}"
            text-anchor="middle" dominant-baseline="middle"
            fill="${tc}" font-size="11" font-weight="${fw}"
            font-family="ui-monospace,monospace">${escapeHtml(nm)}</text>
    </g>`;
  }

  return `<div class="hasse-wrap">
    <div class="hasse-type-label" style="color:${col}">${escapeHtml(tau)}</div>
    <svg width="${SW}" height="${SH}" class="hasse-svg" data-tau="${tau}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="${mid}" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="${col}"/>
        </marker>
      </defs>
      ${edgeSvg}${nodeSvg}
    </svg>
  </div>`;
}

function renderPicker() {
  const root = $("picker-rows");
  if (!root) return;
  const opts = STATE.picker.options;
  if (!opts) { root.innerHTML = `<div class="muted">loading…</div>`; return; }

  const types   = opts.types || [];
  const lat     = STATE.lattice || {};
  const choice  = STATE.picker.choice || {};
  const allowed = pickerAllowed();

  root.innerHTML = `<div class="hasse-col">${
    types.map((tau, ti) =>
      buildTypeHasseSvg(tau, ti, lat, choice, allowed[tau] || new Set())
    ).join("")
  }</div>`;

  // Click handlers on SVG nodes
  root.querySelectorAll(".hasse-node").forEach(el => {
    el.addEventListener("click", () => {
      const tau = el.dataset.tau;
      const lv  = parseInt(el.dataset.lv, 10);
      if ((STATE.picker.choice || {})[tau] === lv)
        delete STATE.picker.choice[tau];
      else
        STATE.picker.choice[tau] = lv;
      renderPicker();
    });
  });

  renderPickerPreview();
  syncLatticeHighlight();
}

/* Bidirectional binding picker ↔ Hasse:
   - When some types are fixed in the picker, dim the lattice nodes
     that don't match the fixed coordinates and highlight the (single)
     matching node in orange.
   - When ALL types are fixed (a unique cfg picked), the matching
     node also receives the existing .selected halo.
   - When no types are fixed, restore the default look. */
function syncLatticeHighlight() {
  const cy = STATE.cy_lattice;
  if (!cy) return;
  const opts = STATE.picker.options;
  const types = opts ? (opts.types || []) : [];
  const choice = STATE.picker.choice || {};
  const fixedCount = types.filter(t => choice[t] != null).length;

  cy.batch(() => {
    cy.elements().removeClass("picker-match picker-dim");
    if (fixedCount === 0) return;
    cy.nodes().forEach(n => {
      const cfg = n.data("cfg") || [];
      let matches = true;
      for (let i = 0; i < types.length; i++) {
        const v = choice[types[i]];
        if (v != null && cfg[i] !== v) { matches = false; break; }
      }
      if (matches) {
        n.addClass("picker-match");
      } else {
        n.addClass("picker-dim");
      }
    });
    // Dim edges whose endpoints are both dim.
    cy.edges().forEach(e => {
      const sa = e.source().hasClass("picker-dim");
      const sb = e.target().hasClass("picker-dim");
      if (sa || sb) e.addClass("picker-dim");
    });
  });

  // If a unique cfg is picked, also center it.
  if (fixedCount === types.length) {
    const m = pickerMatch();
    if (m) {
      const node = cy.getElementById(m.cfg.join(","));
      if (node && node.length) {
        cy.animate({
          center: { eles: node },
          duration: 280,
          easing: "ease",
        });
      }
    }
  }
}

function renderPickerPreview() {
  const root = $("picker-preview");
  if (!root) return;
  const opts = STATE.picker.options;
  if (!opts) { root.innerHTML = ""; return; }
  const types = opts.types || [];
  const choice = STATE.picker.choice || {};
  const fixedCount = types.filter(t => choice[t] != null).length;

  // Progress strip — one segment per type, lit when its level is set.
  const progress = `
    <div class="pp-progress">
      ${types.map(t => `<div class="step ${
        choice[t] != null ? "done" : ""}"></div>`).join("")}
    </div>`;

  // Helper: colored chip for a selected type+level
  const chip = (t, ti) => {
    const lv = choice[t];
    if (lv == null) return `<span class="pp-chip pp-chip-empty">${escapeHtml(t)}</span>`;
    const nm = (opts.level_names[t] || [])[lv] || `lv${lv}`;
    return `<span class="pp-chip" style="background:${colorFor(ti)};color:#fff">${escapeHtml(nm)}</span>`;
  };

  if (fixedCount === 0) {
    root.innerHTML = progress + `<div class="pp-empty">Select a configuration.</div>`;
    return;
  }
  const match = pickerMatch();

  const chipsHtml = `<div class="pp-chips">${types.map((t, ti) => chip(t, ti)).join("")}</div>`;

  if (fixedCount < types.length) {
    root.innerHTML = progress + chipsHtml +
      `<div class="muted" style="margin-top:6px;">${fixedCount} / ${types.length} types selected</div>`;
    return;
  }
  if (!match) {
    root.innerHTML = progress + chipsHtml +
      `<div class="muted" style="margin-top:6px;">no valid configuration</div>`;
    return;
  }

  root.innerHTML = progress + chipsHtml + `
    <div class="pp-badges" style="margin-top:8px;">
      <span class="badge">${match.K} abs</span>
      <span class="badge">${match.n_abstractions} beh</span>
    </div>
    <div class="pp-actions">
      <button class="btn-primary" id="pp-go-patterns">Open behaviors</button>
    </div>`;
  $("pp-go-patterns").addEventListener("click", () => selectCfg(match.cfg_idx));
}

async function openCrossLeafModal(srcCi, srcLeafId) {
  // Open the existing modal shell with a placeholder while we fetch.
  $("modal-content").innerHTML = `
    <div class="cross-modal-head">
      <h3 style="margin:0;">Cross-leaf distribution</h3>
      <div class="muted" style="margin-top:6px;">
        Loading executions of leaf <code>${srcLeafId}</code>
        of cfg <code>#${srcCi}</code>…
      </div>
    </div>`;
  $("modal").classList.remove("hidden");

  let payload;
  try {
    payload = await api(
      `/api/tree/cfg/${srcCi}/leaf/${srcLeafId}/cross`
      + `?kpi=${encodeURIComponent(STATE.kpi)}`);
  } catch (err) {
    $("modal-content").innerHTML = `
      <div style="color:#b91c1c;">cross-distribution failed:
        ${escapeHtml(err.message)}</div>`;
    return;
  }

  if (payload.status !== "ok") {
    $("modal-content").innerHTML = `
      <div class="muted">cannot compute cross-distribution
        (status: ${escapeHtml(payload.status)}).</div>`;
    return;
  }

  $("modal-content").innerHTML = renderCrossLeafContent(payload);

  $("modal-content").querySelectorAll(".cross-target-card").forEach(card => {
    card.addEventListener("click", (ev) => {
      const seg = ev.target.closest(".cross-bar-seg");
      const ci = parseInt(card.dataset.ci, 10);
      if (seg) {
        ev.stopPropagation();
        return;
      }
    });
  });

  $("modal-content").querySelectorAll(".cross-cond.clickable")
    .forEach(chip => {
      chip.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        const ci = parseInt(chip.dataset.ci, 10);
        const pi = parseInt(chip.dataset.pi, 10);
        try {
          const p = await api(`/api/behavior/${ci}/${pi}?kpi=${
            encodeURIComponent(STATE.kpi)}`);
          openBehaviorModal(p);
        } catch (err) { console.error(err); }
      });
    });
}


function renderCrossLeafContent(payload) {
  const kpi = escapeHtml(payload.kpi || STATE.kpi || "KPI");
  const nSrc = payload.source_n_exec || 0;
  const srcPath = payload.source_path || [];
  const srcCi = payload.source_cfg_idx;

  const srcCondHtml = srcPath.length === 0
    ? `<span class="muted">no conditions (root leaf)</span>`
    : srcPath.map(s => {
        const m = String(s.feature || "").match(/^P(\d+)$/);
        const pi = m ? parseInt(m[1], 10) : null;
        const present = s.value === 1;
        const cls = ["cross-cond",
                     pi != null ? "clickable" : "",
                     present ? "present" : "absent"].filter(Boolean).join(" ");
        const data = (pi != null) ? `data-ci="${srcCi}" data-bi="${pi}"` : "";
        return `<span class="${cls}" ${data}>
          <strong>${escapeHtml(s.feature)}</strong>
          <em>${present ? "present" : "absent"}</em>
        </span>`;
      }).join(`<span class="cross-cond-conj">AND</span>`);

  const PAL = ["#6366f1", "#10b981", "#f59e0b", "#ef4444",
               "#06b6d4", "#a855f7", "#84cc16", "#ec4899"];
  const palette = i => PAL[i % PAL.length];

  const targets = payload.targets || [];
  const targetsHtml = targets.length === 0
    ? `<div class="empty">no other useful configurations to compare with.</div>`
    : targets.map(t => {
        // Build the stacked bar.  Each segment width = frac_of_source × 100%.
        const segs = t.leaves.map((lf, i) => {
          const w = Math.max(0.5, lf.frac_of_source * 100);  // min visual width
          const color = palette(i);
          const condShort = (lf.path || []).slice(0, 3)
            .map(s => `${s.feature}${s.value === 1 ? "✓" : "✗"}`)
            .join(" · ") || "root";
          const tip = `${condShort}\n`
                    + `${lf.n_overlap} of ${nSrc} source executions `
                    + `(${(lf.frac_of_source * 100).toFixed(1)}%) | `
                    + `avg ${kpi} = ${escapeHtml(lf.mean_kpi_fmt)}`;
          return `<div class="cross-bar-seg"
                       style="width:${w}%; background:${color};"
                       title="${escapeHtml(tip)}"></div>`;
        }).join("");
        // Per-leaf legend rows (compact).
        const legendRows = t.leaves.map((lf, i) => {
          const color = palette(i);
          const condChips = (lf.path || []).map(s => {
            const m = String(s.feature || "").match(/^P(\d+)$/);
            const pi = m ? parseInt(m[1], 10) : null;
            const present = s.value === 1;
            const cls = ["cross-cond", "small",
                         pi != null ? "clickable" : "",
                         present ? "present" : "absent"]
                       .filter(Boolean).join(" ");
            const data = (pi != null)
              ? `data-ci="${t.cfg_idx}" data-bi="${pi}"` : "";
            return `<span class="${cls}" ${data}>
              <strong>${escapeHtml(s.feature)}</strong>
              <em>${present ? "P" : "A"}</em>
            </span>`;
          }).join("");
          const pct = (lf.frac_of_source * 100).toFixed(1);
          return `
            <div class="cross-leaf-row">
              <span class="cross-leaf-swatch" style="background:${color};"></span>
              <div class="cross-leaf-conds">${condChips
                || `<span class="muted small">root</span>`}</div>
              <div class="cross-leaf-meta">
                <strong>${lf.n_overlap}</strong>/${nSrc}
                <span class="muted">(${pct}%)</span>
                · avg ${escapeHtml(lf.mean_kpi_fmt)}
              </div>
            </div>`;
        }).join("");

        return `
          <div class="cross-target-card" data-ci="${t.cfg_idx}">
            <div class="cross-target-head">
              <h4>cfg #${t.cfg_idx}</h4>
              <span class="cross-target-cfg muted small">
                ${escapeHtml(t.cfg_str || "")}</span>
              <span class="grow"></span>
              <span class="badge">entropy ${fmtNum(t.entropy, 2)}</span>
              <span class="badge">${t.n_leaves} leaf${t.n_leaves === 1 ? "" : "es"} hit</span>
            </div>
            <div class="cross-bar">${segs}</div>
            <div class="cross-leaves-legend">${legendRows}</div>
          </div>`;
      }).join("");

  return `
    <div class="cross-modal-head">
      <h3 style="margin:0;">Where do these executions go in other cfgs?</h3>
      <div class="cross-source">
        <div class="muted small">SOURCE LEAF</div>
        <div>cfg <code>#${srcCi}</code> · leaf <code>${
          payload.source_leaf_id}</code>
          · <strong>${nSrc}</strong> executions
          · avg ${kpi} = <strong>${escapeHtml(payload.source_value_fmt)}</strong>
        </div>
        <div class="cross-source-cond">${srcCondHtml}</div>
      </div>
      <div class="muted small" style="margin-top:6px;">
        Each card below is one other cfg's tree, ranked by how much it
        spreads the source cohort across its own leaves (high entropy
        = different segmentation; low entropy = agrees with the source).
        Click a bar segment to open that cfg's detail.
      </div>
    </div>
    <div class="cross-targets-list">${targetsHtml}</div>
  `;
}


boot().catch(err => {
  console.error("boot failed:", err);
  $("meta").innerHTML =
    `<span class="pill" style="color:#b91c1c;">error: ${escapeHtml(err.message)}</span>`;
});
