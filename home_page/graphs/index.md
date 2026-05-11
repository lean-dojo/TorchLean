---
title: Dependency Maps
---

<p class="lede">
  Explore how TorchLean is organized: which modules import each other, where the main hubs are, and
  how the API, runtime, proof, and verification layers connect.
</p>

<div class="callout">
  Looking for the interactive graph?
  Open the <a href="{{ '/importgraph/' | relative_url }}">module import graph</a>.
</div>

<div class="callout">
  This page starts from Lean imports. A module edge means one Lean file imports another; it does not
  mean every theorem in the source file depends on every theorem in the target file.
</div>

<div class="dep-dashboard" id="dep-dashboard">
  <section class="dep-panel dep-summary">
    <h2>Repository Shape</h2>
    <p class="dep-panel-intro">
      The longest chain counts the longest path through direct imports in this build. It is a rough
      measure of layering depth, not a correctness result.
    </p>
    <div class="dep-stat-grid" id="dep-summary-cards">
      <div class="dep-loading">Loading dependency audit…</div>
    </div>
  </section>

  <section class="dep-panel">
    <h2>Codebase Snapshot</h2>
    <p class="dep-panel-intro">
      These counts are generated from the Lean source tree during the site build. They are useful
      scale indicators, not semantic proof-dependency measurements.
    </p>
    <div class="dep-stat-grid" id="dep-code-cards">
      <div class="dep-loading">Loading codebase statistics…</div>
    </div>
    <h3>Largest Layers By Source Lines</h3>
    <div id="dep-layer-size" class="dep-layer-flow"></div>
  </section>

  <section class="dep-panel">
    <h2>Module Explorer</h2>
    <div class="dep-controls">
      <label>
        Search module
        <input id="dep-search" type="search" placeholder="GraphSpec, Runtime, Fno1d…" />
      </label>
      <label>
        Source layer
        <select id="dep-src-layer"></select>
      </label>
      <label>
        Destination layer
        <select id="dep-dst-layer"></select>
      </label>
    </div>
    <div class="dep-mini-note">
      Click a module to see its direct imports and direct importers. Edges are module imports, not
      theorem-premise dependencies.
    </div>
    <div class="dep-two-col">
      <div>
        <h3>Matching Modules</h3>
        <div id="dep-module-list" class="dep-list"></div>
      </div>
      <div>
        <h3 id="dep-detail-title">Selected Module</h3>
        <div id="dep-module-detail" class="dep-detail muted">Select a module on the left.</div>
      </div>
    </div>
  </section>

  <section class="dep-panel">
    <h2>Layer Flow</h2>
    <div id="dep-layer-flow" class="dep-layer-flow"></div>
  </section>

  <section class="dep-panel">
    <h2>Import Hubs</h2>
    <div class="dep-two-col">
      <div>
        <h3>Most Imported</h3>
        <div id="dep-fan-in" class="dep-list compact"></div>
      </div>
      <div>
        <h3>Most Importing</h3>
        <div id="dep-fan-out" class="dep-list compact"></div>
      </div>
    </div>
  </section>
</div>

<script>
(() => {
  const dataUrl = "{{ '/graphs/dependency-audit.json' | relative_url }}";
  const state = { report: null, selected: null };

  const $ = (id) => document.getElementById(id);
  const fmt = (n) => Number(n).toLocaleString();
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
  const layerOf = (mod) => {
    const parts = mod.split(".");
    if (parts[0] !== "NN" || parts.length < 2) return parts[0] || "unknown";
    const named = new Set([
      "API", "CI", "Entrypoint", "Examples", "Floats", "GraphSpec", "IR", "MLTheory",
      "Proofs", "Runtime", "Spec", "Tensor", "Tests", "Verification"
    ]);
    return named.has(parts[1]) ? `NN.${parts[1]}` : parts[1];
  };

  function setOptions(select, values, label) {
    select.innerHTML = [`<option value="">${label}</option>`]
      .concat(values.map(v => `<option value="${esc(v)}">${esc(v)}</option>`))
      .join("");
  }

  function renderSummary(report) {
    const s = report.summary;
    $("dep-summary-cards").innerHTML = [
      ["Modules", fmt(s.modules)],
      ["Import edges", fmt(s.import_edges)],
      ["Internal edges", fmt(s.internal_import_edges)],
      ["Public imports", fmt(s.public_import_edges)],
      ["Longest chain", fmt(s.critical_path_import_edges)],
    ].map(([k, v]) => `
      <div class="dep-stat-card">
        <div class="dep-stat-value">${esc(v)}</div>
        <div class="dep-stat-label">${esc(k)}</div>
      </div>
    `).join("");
  }

  function renderCodeStats(report) {
    const stats = report.code_stats;
    if (!stats) {
      $("dep-code-cards").innerHTML = `<div class="dep-error">No code statistics found in audit JSON.</div>`;
      $("dep-layer-size").innerHTML = "";
      return;
    }
    $("dep-code-cards").innerHTML = [
      ["Lean files", fmt(stats.lean_files)],
      ["Lean source lines", fmt(stats.total_lines)],
      ["Code lines", fmt(stats.code_lines)],
      ["Declaration headers", fmt(stats.declarations)],
      ["Theorems and lemmas", fmt(stats.theorem_like_declarations)],
      ["Comment/blank lines", fmt(stats.blank_or_comment_lines)],
    ].map(([k, v]) => `
      <div class="dep-stat-card">
        <div class="dep-stat-value">${esc(v)}</div>
        <div class="dep-stat-label">${esc(k)}</div>
      </div>
    `).join("");

    const layers = (stats.layer_sizes || []).slice(0, 16);
    const max = Math.max(1, ...layers.map(x => x.lines));
    $("dep-layer-size").innerHTML = layers.map(item => `
      <div class="dep-layer-row">
        <span class="dep-layer-label">${esc(item.layer)}</span>
        <span class="dep-layer-bar"><span style="width:${(100 * item.lines / max).toFixed(1)}%"></span></span>
        <span class="dep-count">${esc(fmt(item.lines))} lines · ${esc(fmt(item.files))} files</span>
      </div>
    `).join("");
  }

  function renderBars(container, items, labelKey, valueKey, onClick) {
    const max = Math.max(1, ...items.map(x => x[valueKey]));
    container.innerHTML = items.map(item => {
      const label = item[labelKey];
      const value = item[valueKey];
      return `
        <button class="dep-row" type="button" data-module="${esc(label)}">
          <span class="dep-row-main">
            <span class="dep-row-label">${esc(label)}</span>
            <span class="dep-bar"><span style="width:${(100 * value / max).toFixed(1)}%"></span></span>
          </span>
          <span class="dep-count">${esc(value)}</span>
        </button>
      `;
    }).join("");
    if (onClick) {
      container.querySelectorAll("button[data-module]").forEach(btn => {
        btn.addEventListener("click", () => onClick(btn.dataset.module));
      });
    }
  }

  function buildIndex(report) {
    const modules = new Set();
    for (const edge of report.edges) {
      modules.add(edge.src);
      modules.add(edge.dst);
    }
    const imports = new Map();
    const importers = new Map();
    for (const mod of modules) {
      imports.set(mod, []);
      importers.set(mod, []);
    }
    for (const edge of report.edges) {
      imports.get(edge.src)?.push(edge);
      importers.get(edge.dst)?.push(edge);
    }
    return { modules: [...modules].sort(), imports, importers };
  }

  function renderModuleList() {
    const report = state.report;
    const idx = report._index;
    const query = $("dep-search").value.trim().toLowerCase();
    const srcLayer = $("dep-src-layer").value;
    const dstLayer = $("dep-dst-layer").value;
    const imports = idx.imports;

    let mods = idx.modules.filter(mod => {
      if (query && !mod.toLowerCase().includes(query)) return false;
      if (srcLayer && layerOf(mod) !== srcLayer) return false;
      if (dstLayer) {
        const out = imports.get(mod) || [];
        if (!out.some(edge => layerOf(edge.dst) === dstLayer)) return false;
      }
      return true;
    });
    mods = mods.slice(0, 180);

    $("dep-module-list").innerHTML = mods.map(mod => {
      const out = idx.imports.get(mod)?.length || 0;
      const inc = idx.importers.get(mod)?.length || 0;
      const active = mod === state.selected ? " active" : "";
      return `
        <button class="dep-module${active}" type="button" data-module="${esc(mod)}">
          <span>${esc(mod)}</span>
          <small>${esc(layerOf(mod))} · imports ${out} · imported by ${inc}</small>
        </button>
      `;
    }).join("") || `<div class="muted">No modules match the current filter.</div>`;

    $("dep-module-list").querySelectorAll("button[data-module]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.selected = btn.dataset.module;
        renderModuleList();
        renderDetail();
      });
    });
  }

  function edgeList(title, edges, side) {
    if (!edges.length) return `<h4>${title}</h4><div class="muted">None.</div>`;
    return `
      <h4>${title}</h4>
      <ul class="dep-edge-list">
        ${edges.slice(0, 80).map(e => {
          const mod = side === "dst" ? e.dst : e.src;
          return `<li><button type="button" data-module="${esc(mod)}">${esc(mod)}</button>
            <small>${e.public ? "public" : "private"} · ${esc(e.path)}:${e.line}</small></li>`;
        }).join("")}
      </ul>
    `;
  }

  function renderDetail() {
    const mod = state.selected;
    if (!mod) return;
    const idx = state.report._index;
    const out = (idx.imports.get(mod) || []).sort((a, b) => a.dst.localeCompare(b.dst));
    const inc = (idx.importers.get(mod) || []).sort((a, b) => a.src.localeCompare(b.src));
    $("dep-detail-title").textContent = mod;
    $("dep-module-detail").innerHTML = `
      <div class="dep-module-meta">
        <span>${esc(layerOf(mod))}</span>
        <span>${out.length} imports</span>
        <span>${inc.length} importers</span>
      </div>
      ${edgeList("Imports", out, "dst")}
      ${edgeList("Imported By", inc, "src")}
    `;
    $("dep-module-detail").querySelectorAll("button[data-module]").forEach(btn => {
      btn.addEventListener("click", () => {
        state.selected = btn.dataset.module;
        $("dep-search").value = state.selected;
        renderModuleList();
        renderDetail();
      });
    });
  }

  function renderLayerFlow(report) {
    const items = report.layer_edges.slice(0, 32);
    const max = Math.max(1, ...items.map(x => x.count));
    $("dep-layer-flow").innerHTML = items.map(item => `
      <div class="dep-layer-row">
        <span class="dep-layer-label">${esc(item.src_layer)} → ${esc(item.dst_layer)}</span>
        <span class="dep-layer-bar"><span style="width:${(100 * item.count / max).toFixed(1)}%"></span></span>
        <span class="dep-count">${esc(item.count)}</span>
      </div>
    `).join("");
  }

  fetch(dataUrl)
    .then(resp => {
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      return resp.json();
    })
    .then(report => {
      state.report = report;
      report._index = buildIndex(report);
      const layers = [...new Set(report._index.modules.map(layerOf))].sort();
      setOptions($("dep-src-layer"), layers, "All source layers");
      setOptions($("dep-dst-layer"), layers, "Any imported layer");
      renderSummary(report);
      renderCodeStats(report);
      renderLayerFlow(report);
      renderBars($("dep-fan-in"), report.top_fan_in, "module", "count", mod => {
        state.selected = mod; $("dep-search").value = mod; renderModuleList(); renderDetail();
      });
      renderBars($("dep-fan-out"), report.top_fan_out, "module", "count", mod => {
        state.selected = mod; $("dep-search").value = mod; renderModuleList(); renderDetail();
      });
      $("dep-search").addEventListener("input", renderModuleList);
      $("dep-src-layer").addEventListener("change", renderModuleList);
      $("dep-dst-layer").addEventListener("change", renderModuleList);
      renderModuleList();
    })
    .catch(err => {
      $("dep-summary-cards").innerHTML =
        `<div class="dep-error">Could not load ${esc(dataUrl)}: ${esc(err.message)}</div>`;
      $("dep-code-cards").innerHTML =
        `<div class="dep-error">Could not load codebase statistics.</div>`;
    });
})();
</script>
