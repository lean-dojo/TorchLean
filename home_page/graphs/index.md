---
title: Graphs
---

<div class="graph-actions">
  <a class="primary-link" href="{{ '/importgraph/' | relative_url }}">Open import graph</a>
  <a class="secondary-link" href="{{ '/performance/' | relative_url }}">View build performance</a>
</div>

Search the Lean import graph by module name to inspect its dependencies, reverse dependencies, and
the part of the source tree affected by a change.

FloatLib supplies the generic scalar formats and numerical theory; `NN.Floats` contains
TorchLean's adapters. External imports appear in the explorer, but their source trees are not
expanded here.

<div class="dep-dashboard" id="dep-dashboard">
  <section class="dep-panel dep-summary graph-overview">
    <div>
      <h2>Import Graph</h2>
      <p class="dep-panel-intro">
        Each edge means that one Lean module directly imports another.
      </p>
    </div>
    <div class="dep-stat-grid" id="dep-summary-cards">
      <div class="dep-loading">Loading import graph…</div>
    </div>
  </section>

  <section class="dep-panel">
    <h2>Source Snapshot</h2>
    <p class="dep-panel-intro">
      The site build reads these counts from the TorchLean checkout, including tests and guide
      sources. Dependency sources, including FloatLib, are excluded. Declaration headers measure
      source size; they do not measure proof coverage.
    </p>
    <div class="dep-stat-grid" id="dep-code-cards">
      <div class="dep-loading">Loading source statistics…</div>
    </div>
  </section>

  <section class="dep-panel">
    <h2>Module Explorer</h2>
    <div class="dep-controls">
      <label>
        Search module
        <input id="dep-search" type="search" placeholder="GraphSpec, Runtime, FNO…" />
      </label>
      <label>
        Importing area
        <select id="dep-src-layer"></select>
      </label>
      <label>
        Imported area
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
      "API", "CI", "Examples", "Floats", "GraphSpec", "IR", "MLTheory",
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
      ["Re-export edges", fmt(s.public_import_edges)],
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
      $("dep-code-cards").innerHTML = `<div class="dep-error">No source statistics found.</div>`;
      return;
    }
    $("dep-code-cards").innerHTML = [
      ["Lean files", fmt(stats.lean_files)],
      ["Lean source lines", fmt(stats.total_lines)],
      ["Code lines", fmt(stats.code_lines)],
      ["Declaration headers", fmt(stats.declarations)],
      ["Theorem/lemma headers", fmt(stats.theorem_like_declarations)],
      ["Comment/blank lines", fmt(stats.blank_or_comment_lines)],
    ].map(([k, v]) => `
      <div class="dep-stat-card">
        <div class="dep-stat-value">${esc(v)}</div>
        <div class="dep-stat-label">${esc(k)}</div>
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

  function edgeList(title, edges, side) {
    if (!edges.length) return `<h4>${title}</h4><div class="muted">None.</div>`;
    return `
      <h4>${title}</h4>
      <ul class="dep-edge-list">
        ${edges.slice(0, 80).map(e => {
          const mod = side === "dst" ? e.dst : e.src;
          return `<li><button type="button" data-module="${esc(mod)}">${esc(mod)}</button>
            <small>${e.public ? "re-exported" : "local"} · ${esc(e.path)}:${e.line}</small></li>`;
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
      setOptions($("dep-src-layer"), layers, "All importing areas");
      setOptions($("dep-dst-layer"), layers, "Any imported area");
      renderSummary(report);
      renderCodeStats(report);
      $("dep-search").addEventListener("input", renderModuleList);
      $("dep-src-layer").addEventListener("change", renderModuleList);
      $("dep-dst-layer").addEventListener("change", renderModuleList);
      renderModuleList();
    })
    .catch(err => {
      $("dep-summary-cards").innerHTML =
        `<div class="dep-error">Could not load the import graph: ${esc(err.message)}</div>`;
      $("dep-code-cards").innerHTML =
        `<div class="dep-error">Could not load codebase statistics.</div>`;
    });
})();
</script>
