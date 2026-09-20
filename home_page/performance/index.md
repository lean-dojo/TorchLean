---
title: Performance
---

# Performance

<p class="lede performance-lede">
  How long does TorchLean take to build and test on its ordinary continuous-integration runner?
</p>

The charts use timing records from successful `main`-branch CI runs; opening them does not start
another workflow or contact a benchmark server. Because GitHub-hosted machines vary,
these numbers are useful for spotting changes worth investigating, not for comparing hardware or
making fine-grained runtime claims.

<aside class="performance-provenance">
  The commit-by-commit view is inspired by
  <a href="https://radar.lean-lang.org/about">Lean Radar</a>. Radar runs controlled benchmark
  suites on dedicated machines; this smaller page reports TorchLean's existing GitHub Actions
  timings and labels them accordingly.
</aside>

<div class="performance-dashboard" id="performance-dashboard">
  <div class="performance-notice" id="performance-notice" role="status" aria-live="polite">
    Loading recent CI runs…
  </div>

  <section class="performance-panel" aria-labelledby="performance-latest-title">
    <div class="performance-section-head">
      <div>
        <h2 id="performance-latest-title">Latest run in selected workflow</h2>
        <p class="performance-panel-intro">
          Wall-clock time for the named steps in the workflow selected below.
        </p>
      </div>
      <div class="performance-latest" id="performance-latest"></div>
    </div>
    <div class="performance-card-grid" id="performance-cards">
      <div class="performance-loading">Loading timings…</div>
    </div>
  </section>

  <section class="performance-panel" aria-labelledby="performance-history-title">
    <div class="performance-chart-head">
      <div>
        <h2 id="performance-history-title">Recent history</h2>
        <p class="performance-panel-intro">
          Successful pushes to <code>main</code>, ordered by run start time.
        </p>
      </div>
      <label for="performance-scope">
        Workflow
        <select id="performance-scope">
          <option value="native">Current workflow</option>
          <option value="current">Earlier maintained modules build</option>
          <option value="maintained">Earlier combined build</option>
          <option value="legacy">Earlier workflow</option>
        </select>
      </label>
      <label for="performance-metric">
        Metric
        <select id="performance-metric">
          <option value="native">Modules and native commands build</option>
          <option value="current">Earlier maintained modules build</option>
          <option value="maintained">Earlier combined build</option>
          <option value="build">Legacy library build</option>
          <option value="tests">Test suite</option>
          <option value="broad">Legacy broad CI import</option>
          <option value="lint">Repository lint</option>
          <option value="total">Complete CI job</option>
        </select>
      </label>
    </div>
    <p class="performance-panel-intro" id="performance-scope-description"></p>
    <div class="performance-chart" id="performance-chart">
      <div class="performance-loading">Loading chart…</div>
    </div>
    <p class="performance-chart-caption" id="performance-chart-caption"></p>
  </section>

  <section class="performance-panel" aria-labelledby="performance-runs-title">
    <h2 id="performance-runs-title">Runs</h2>
    <div class="performance-table-wrap">
      <table class="performance-table">
        <thead>
          <tr>
            <th scope="col">Commit</th>
            <th scope="col">Date</th>
            <th scope="col" id="performance-build-heading">Build step</th>
            <th scope="col">Lint</th>
            <th scope="col">Tests</th>
            <th scope="col">Complete job</th>
          </tr>
        </thead>
        <tbody id="performance-runs">
          <tr><td colspan="6">Loading recent runs…</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</div>

<script>
(() => {
  "use strict";

  const repository = "lean-dojo/TorchLean";
  const workflow = "ci.yml";
  const maximumRuns = 8;
  const cacheKey = "torchlean-ci-performance-v3";
  const cacheLifetimeMs = 30 * 60 * 1000;
  const apiRoot = `https://api.github.com/repos/${repository}`;
  // A cold page load makes one run-list request and at most eight job requests.
  const runsUrl = `${apiRoot}/actions/workflows/${workflow}/runs` +
    `?branch=main&event=push&status=success&per_page=${maximumRuns}`;

  const metrics = {
    native: {
      label: "Modules and native commands build",
      step: "Build library, CI, examples, tests, and native commands",
    },
    current: {
      label: "Earlier maintained modules build",
      step: "Build library, CI, examples, and tests",
    },
    maintained: {
      label: "Earlier combined build",
      step: "Build maintained library, CI, example, and test modules",
    },
    build: { label: "Legacy library build", step: "Build curated library surface" },
    tests: { label: "Test suite", step: "Run curated test suite" },
    broad: { label: "Legacy broad CI import", step: "Build broad CI import surface" },
    lint: { label: "Repository lint", step: "Repo lint (TorchLean policies)" },
    total: { label: "Complete CI job", step: null },
  };
  const scopes = {
    native: {
      buildMetric: "native",
      description: "Current workflow: one build step covers the maintained library, CI, " +
        "examples, tests, and native commands. Earlier workloads are shown separately.",
    },
    current: {
      buildMetric: "current",
      description: "Earlier maintained modules build: one step covered the library, CI, " +
        "examples, and tests, before native commands were added to that build step.",
    },
    maintained: {
      buildMetric: "maintained",
      description: "Earlier combined build: this workload also included the now-removed " +
        "timing programs. Its timings are kept separate from the current workload.",
    },
    legacy: {
      buildMetric: "build",
      description: "Earlier workflow: the library build and broad CI import were separate steps. " +
        "Select Legacy broad CI import to view that step. Trends stay within this workflow.",
    },
  };
  const recordScope = record => record.scope ||
    (Number.isFinite(record.durations.native) ? "native" :
      Number.isFinite(record.durations.current) ? "current" :
      Number.isFinite(record.durations.maintained) ? "maintained" :
      (Number.isFinite(record.durations.build) || Number.isFinite(record.durations.broad) ?
        "legacy" : null));

  const notice = document.getElementById("performance-notice");
  const cards = document.getElementById("performance-cards");
  const latestMeta = document.getElementById("performance-latest");
  const chart = document.getElementById("performance-chart");
  const chartCaption = document.getElementById("performance-chart-caption");
  const metricSelect = document.getElementById("performance-metric");
  const runsBody = document.getElementById("performance-runs");
  const scopeSelect = document.getElementById("performance-scope");
  const scopeDescription = document.getElementById("performance-scope-description");
  const buildHeading = document.getElementById("performance-build-heading");
  let preferredScope = null;
  let currentRecords = [];

  const durationSeconds = (start, finish) => {
    const begin = Date.parse(start);
    const end = Date.parse(finish);
    if (!Number.isFinite(begin) || !Number.isFinite(end) || end < begin) return null;
    return (end - begin) / 1000;
  };

  const stepSeconds = (job, name) => {
    const step = (job.steps || []).find(candidate => candidate.name === name);
    return step ? durationSeconds(step.started_at, step.completed_at) : null;
  };

  const formatDuration = seconds => {
    if (!Number.isFinite(seconds)) return "—";
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const remainder = rounded % 60;
    if (hours > 0) return `${hours} h ${minutes} min`;
    if (minutes > 0) return `${minutes} min ${remainder} s`;
    return `${remainder} s`;
  };

  const shortDuration = seconds => {
    if (!Number.isFinite(seconds)) return "—";
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const remainder = rounded % 60;
    if (hours > 0) return `${hours}h ${minutes}m`;
    if (minutes > 0) return `${minutes}m ${remainder}s`;
    return `${rounded}s`;
  };

  const formatDate = iso => new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  }).format(new Date(iso));

  const setNotice = (message, kind = "normal") => {
    notice.textContent = message;
    notice.dataset.kind = kind;
    notice.hidden = message === "";
  };

  const readCache = () => {
    try {
      const parsed = JSON.parse(localStorage.getItem(cacheKey));
      if (!parsed || !Array.isArray(parsed.records) || parsed.records.length === 0 ||
          !Number.isFinite(parsed.savedAt)) return null;
      return parsed;
    } catch (_error) {
      return null;
    }
  };

  const writeCache = records => {
    try {
      // Keep API history in the visitor's browser; the website needs no data branch or service.
      localStorage.setItem(cacheKey, JSON.stringify({ savedAt: Date.now(), records }));
    } catch (_error) {
      // Private browsing or a storage policy can disable localStorage; the page still works.
    }
  };

  const requestJson = async url => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`GitHub API returned HTTP ${response.status}`);
    return response.json();
  };

  const loadRecords = async () => {
    const runData = await requestJson(runsUrl);
    const records = await Promise.all((runData.workflow_runs || []).map(async run => {
      const jobData = await requestJson(
        `${apiRoot}/actions/runs/${run.id}/jobs?filter=latest&per_page=100`
      );
      const job = (jobData.jobs || []).find(candidate => candidate.name === "build_and_test");
      if (!job || job.conclusion !== "success") return null;
      const durations = { total: durationSeconds(job.started_at, job.completed_at) };
      for (const [key, metric] of Object.entries(metrics)) {
        if (metric.step) durations[key] = stepSeconds(job, metric.step);
      }
      return {
        id: run.id,
        sha: run.head_sha,
        title: run.display_title || run.head_commit?.message || "TorchLean CI run",
        url: run.html_url,
        startedAt: job.started_at || run.run_started_at,
        durations,
        scope: (job.steps || []).some(step => step.name === metrics.native.step) ? "native" :
          (job.steps || []).some(step => step.name === metrics.current.step) ? "current" :
          (job.steps || []).some(step => step.name === metrics.maintained.step) ?
          "maintained" : (job.steps || []).some(step =>
            [metrics.build.step, metrics.broad.step].includes(step.name)) ? "legacy" : null,
      };
    }));
    return records
      .filter(record => record !== null)
      .sort((left, right) => Date.parse(left.startedAt) - Date.parse(right.startedAt));
  };

  const trend = (current, previous) => {
    if (!Number.isFinite(current) || !Number.isFinite(previous) || previous === 0) {
      return { text: "Waiting for another run", direction: "flat" };
    }
    const change = ((current - previous) / previous) * 100;
    if (Math.abs(change) < 0.1) return { text: "No measurable change", direction: "flat" };
    return {
      text: `${change < 0 ? "↓" : "↑"} ${Math.abs(change).toFixed(1)}% from previous`,
      direction: change < 0 ? "shorter" : "longer",
    };
  };

  const renderCards = (records, buildMetric) => {
    const latest = records.at(-1);
    const previous = records.at(-2);
    cards.replaceChildren();
    for (const key of [buildMetric, "tests", "lint", "total"]) {
      const card = document.createElement("div");
      card.className = "performance-card";

      const value = document.createElement("div");
      value.className = "performance-value";
      value.textContent = formatDuration(latest.durations[key]);

      const label = document.createElement("div");
      label.className = "performance-label";
      label.textContent = metrics[key].label;

      const comparison = trend(latest.durations[key], previous?.durations[key]);
      const change = document.createElement("div");
      change.className = "performance-trend";
      change.dataset.direction = comparison.direction;
      change.textContent = comparison.text;

      card.append(value, label, change);
      cards.append(card);
    }

    latestMeta.replaceChildren();
    const link = document.createElement("a");
    link.href = latest.url;
    link.textContent = latest.sha.slice(0, 7);
    link.title = latest.title;
    const date = document.createElement("span");
    date.textContent = formatDate(latest.startedAt);
    latestMeta.append("Commit ", link, date);
  };

  const svgNode = (name, attributes = {}, text = null) => {
    const node = document.createElementNS("http://www.w3.org/2000/svg", name);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    if (text !== null) node.textContent = text;
    return node;
  };

  const renderChart = (records, metricKey) => {
    chart.replaceChildren();
    const valid = records.filter(record => Number.isFinite(record.durations[metricKey]));
    if (valid.length === 0) {
      const empty = document.createElement("div");
      empty.className = "performance-loading";
      empty.textContent = "This CI step has no timing data yet.";
      chart.append(empty);
      chartCaption.textContent = "";
      return;
    }

    const width = 820;
    const height = 280;
    const margin = { top: 22, right: 24, bottom: 48, left: 76 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const values = valid.map(record => record.durations[metricKey]);
    const rawMin = Math.min(...values);
    const rawMax = Math.max(...values);
    const spread = Math.max(rawMax - rawMin, rawMax * 0.08, 1);
    const minimum = Math.max(0, rawMin - spread * 0.35);
    const maximum = rawMax + spread * 0.35;
    const range = Math.max(maximum - minimum, 1);
    const x = index => margin.left +
      (valid.length === 1 ? plotWidth / 2 : (index / (valid.length - 1)) * plotWidth);
    const y = value => margin.top + ((maximum - value) / range) * plotHeight;

    // Native SVG keeps the page responsive without shipping a charting dependency.
    const svg = svgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `${metrics[metricKey].label} timing across ${valid.length} successful CI runs`,
    });

    for (let tick = 0; tick <= 4; tick += 1) {
      const value = maximum - (tick / 4) * range;
      const position = margin.top + (tick / 4) * plotHeight;
      svg.append(svgNode("line", {
        class: "performance-chart-grid",
        x1: margin.left,
        x2: width - margin.right,
        y1: position,
        y2: position,
      }));
      svg.append(svgNode("text", {
        class: "performance-chart-axis",
        x: margin.left - 12,
        y: position + 4,
        "text-anchor": "end",
      }, shortDuration(value)));
    }

    const points = valid.map((record, index) => `${x(index)},${y(record.durations[metricKey])}`);
    svg.append(svgNode("polyline", {
      class: "performance-chart-line",
      points: points.join(" "),
    }));

    valid.forEach((record, index) => {
      const point = svgNode("circle", {
        class: "performance-chart-point",
        cx: x(index),
        cy: y(record.durations[metricKey]),
        r: 5,
      });
      point.append(svgNode("title", {},
        `${record.sha.slice(0, 7)}: ${formatDuration(record.durations[metricKey])}`));
      svg.append(point);

      if (valid.length <= 5 || index === 0 || index === valid.length - 1 || index % 2 === 0) {
        svg.append(svgNode("text", {
          class: "performance-chart-axis performance-chart-commit",
          x: x(index),
          y: height - 18,
          "text-anchor": "middle",
        }, record.sha.slice(0, 7)));
      }
    });

    chart.append(svg);
    chartCaption.textContent =
      `${metrics[metricKey].label} wall-clock time. Lower values mean the CI step finished sooner.`;
  };

  const renderTable = (records, buildMetric) => {
    runsBody.replaceChildren();
    for (const record of [...records].reverse()) {
      const row = document.createElement("tr");
      const commitCell = document.createElement("td");
      const link = document.createElement("a");
      link.href = record.url;
      link.textContent = record.sha.slice(0, 7);
      link.title = record.title;
      commitCell.append(link);
      row.append(commitCell);

      const values = [
        formatDate(record.startedAt),
        formatDuration(record.durations[buildMetric]),
        formatDuration(record.durations.lint),
        formatDuration(record.durations.tests),
        formatDuration(record.durations.total),
      ];
      for (const value of values) {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      }
      runsBody.append(row);
    }
  };

  const selectedRecords = () =>
    currentRecords.filter(record => recordScope(record) === scopeSelect.value);

  const renderSelectedScope = () => {
    const scope = scopes[scopeSelect.value];
    const records = selectedRecords();
    for (const option of metricSelect.options) {
      option.disabled = ["native", "current", "maintained", "build", "broad"].includes(option.value) &&
        option.value !== scope.buildMetric &&
        !(scopeSelect.value === "legacy" && option.value === "broad");
    }
    if (metricSelect.selectedOptions[0].disabled) metricSelect.value = scope.buildMetric;
    scopeDescription.textContent = scope.description;
    buildHeading.textContent = metrics[scope.buildMetric].label;
    renderCards(records, scope.buildMetric);
    renderChart(records, metricSelect.value);
    renderTable(records, scope.buildMetric);
  };

  const render = records => {
    if (!records.length) throw new Error("No successful main-branch CI runs were found");
    // Reject malformed cached records before retaining them as the offline fallback.
    for (const record of records) {
      if (!record || !record.durations || !scopes[recordScope(record)] ||
          typeof record.sha !== "string" || typeof record.url !== "string" ||
          !Number.isFinite(Date.parse(record.startedAt))) {
        throw new Error("Invalid CI timing record");
      }
    }
    currentRecords = records;
    const available = new Set(records.map(recordScope));
    for (const option of scopeSelect.options) option.disabled = !available.has(option.value);
    scopeSelect.value = available.has(preferredScope) ? preferredScope :
      (available.has("native") ? "native" :
        available.has("current") ? "current" :
        available.has("maintained") ? "maintained" : "legacy");
    renderSelectedScope();
  };

  scopeSelect.addEventListener("change", () => {
    preferredScope = scopeSelect.value;
    renderSelectedScope();
  });
  metricSelect.addEventListener("change", () => renderChart(selectedRecords(), metricSelect.value));

  let cache = readCache();
  if (cache) {
    try {
      render(cache.records);
      setNotice("");
    } catch (_error) {
      cache = null;
      try {
        localStorage.removeItem(cacheKey);
      } catch (_storageError) {
        // A storage policy must not prevent fetching fresh timing data.
      }
    }
  }

  if (cache && Date.now() - cache.savedAt < cacheLifetimeMs) return;

  loadRecords()
    .then(records => {
      render(records);
      writeCache(records);
      setNotice("");
    })
    .catch(error => {
      if (cache && cache.records.length) {
        setNotice("GitHub could not be reached; showing the last data saved in this browser.", "warning");
      } else {
        setNotice(`Could not load CI timing history: ${error.message}.`, "error");
        cards.innerHTML = '<div class="performance-loading">Timing data is temporarily unavailable.</div>';
        chart.innerHTML = '<div class="performance-loading">Chart data is temporarily unavailable.</div>';
        runsBody.innerHTML = '<tr><td colspan="6">Run history is temporarily unavailable.</td></tr>';
      }
    });
})();
</script>
