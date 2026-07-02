---
title: Graphs
---

<div class="graph-actions">
  <a class="primary-link" href="{{ '/importgraph/' | relative_url }}">Open interactive graph</a>
  <a class="secondary-link" href="{{ '/graphs/dependency-audit.md' | relative_url }}">Read audit summary</a>
  <a class="secondary-link" href="{{ '/graphs/dependency-audit.json' | relative_url }}">Download JSON</a>
</div>

<div class="dep-dashboard" id="dep-dashboard">
  <section class="dep-panel dep-summary graph-overview">
    <div>
      <h2>Repository Shape</h2>
      <p class="dep-panel-intro">
        Import edges show file-level Lean dependencies. They are useful for architecture review;
        theorem-level proof dependencies are a different measurement.
      </p>
    </div>
    <div class="dep-stat-grid" id="dep-summary-cards">
      <div class="dep-loading">Loading dependency audit…</div>
    </div>
  </section>

  <section class="dep-panel graph-map-panel">
    <div class="graph-map-head">
      <div>
        <h2>Layer Map</h2>
        <p class="dep-panel-intro">
          Use a layer below to filter the module explorer. The map is organized by the names that
          appear in the Lean import tree.
        </p>
      </div>
      <button class="secondary-link graph-layer-button" type="button" data-layer-filter="">Show all</button>
    </div>
    <div class="graph-layer-map" aria-label="TorchLean layer filters">
      <button type="button" data-layer-filter="NN.API">
        <strong>API</strong>
        <span>Public facade and user-facing names</span>
      </button>
      <button type="button" data-layer-filter="NN.Spec">
        <strong>Spec</strong>
        <span>Tensor, layer, model, and shape contracts</span>
      </button>
      <button type="button" data-layer-filter="NN.Runtime">
        <strong>Runtime</strong>
        <span>Autograd, training, CUDA, and interop paths</span>
      </button>
      <button type="button" data-layer-filter="NN.Proofs">
        <strong>Proofs</strong>
        <span>Autograd, approximation, and semantic lemmas</span>
      </button>
      <button type="button" data-layer-filter="NN.Verification">
        <strong>Verification</strong>
        <span>Bounds, certificates, geometry, and checkers</span>
      </button>
      <button type="button" data-layer-filter="NN.Floats">
        <strong>Floats</strong>
        <span>IEEE-style executable and interval semantics</span>
      </button>
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
      Click a module to see its direct imports and direct importers.
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

  function setLayerFilter(layer) {
    $("dep-src-layer").value = layer;
    state.selected = null;
    renderModuleList();
    renderDetail();
    document.querySelectorAll("[data-layer-filter]").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.layerFilter === layer);
    });
    document.getElementById("dep-dashboard")?.scrollIntoView({ behavior: "smooth", block: "start" });
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
    if (!mod) {
      $("dep-detail-title").textContent = "Selected Module";
      $("dep-module-detail").innerHTML = "Select a module on the left.";
      $("dep-module-detail").classList.add("muted");
      return;
    }
    $("dep-module-detail").classList.remove("muted");
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
      $("dep-search").addEventListener("input", renderModuleList);
      $("dep-src-layer").addEventListener("change", renderModuleList);
      $("dep-dst-layer").addEventListener("change", renderModuleList);
      document.querySelectorAll("[data-layer-filter]").forEach(btn => {
        btn.addEventListener("click", () => setLayerFilter(btn.dataset.layerFilter));
      });
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
