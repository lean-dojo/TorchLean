---
title: Performance
---

# Performance

We time three CPU workloads on the release checkout. Below those results are the timings from
our [GitHub Actions site builds](https://github.com/lean-dojo/TorchLean/actions/workflows/blueprint.yml).

<!-- runtime-bench:start (scripts/docs/runtime_bench.py renders this section) -->
<div class="performance-dashboard">
  <section class="performance-panel" aria-labelledby="performance-runtime-title">
    <h2 id="performance-runtime-title">Runtime benchmarks</h2>
    <p class="performance-panel-intro">
      We ran each command 5 times. The table shows the medians. CPU times varied by at most 4.5%.
    </p>
    <div class="performance-table-wrap">
      <table class="performance-table">
        <thead>
          <tr>
            <th scope="col">Benchmark</th>
            <th scope="col">Workload</th>
            <th scope="col">CPU time</th>
            <th scope="col">Wall time</th>
            <th scope="col">Peak memory</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><code>qs_mlp_ieee</code></td>
            <td>the quickstart MLP's training loop and Adam in FloatLib binary32</td>
            <td>5.83 s</td>
            <td>5.85 s</td>
            <td>78 MiB</td>
          </tr>
          <tr>
            <td><code>gpt2</code></td>
            <td>a byte-level GPT on Tiny Shakespeare</td>
            <td>10.59 s</td>
            <td>10.59 s</td>
            <td>90 MiB</td>
          </tr>
          <tr>
            <td><code>crown_query</code></td>
            <td>exact-rational CROWN bounds of a 24-48-48-48-8 ReLU chain</td>
            <td>6.46 s</td>
            <td>6.47 s</td>
            <td>106 MiB</td>
          </tr>
        </tbody>
      </table>
    </div>
    <p class="performance-chart-caption">
      Measured on 2026-11-20 with Lean 4.36.0 on AMD EPYC 7742 64-Core Processor. Source commit
      <code>534b505e1169</code> with local changes.
    </p>
    <details>
      <summary>Commands</summary>
      <ul>
        <li><code>torchlean quickstart_mlp --device cpu --arithmetic ieee --steps 5000 --seed
          0</code></li>
        <li><code>torchlean gpt2 --device cpu --data-file data/real/text/tiny_shakespeare.txt
          --steps 400 --log false --seed 0</code></li>
        <li><code>verify crown-query q-24-48x3-8.json</code></li>
      </ul>
    </details>
  </section>
</div>
<!-- runtime-bench:end -->

<div class="performance-dashboard" id="performance-dashboard">
  <div class="performance-notice" id="performance-notice" role="status" aria-live="polite">
    Loading build times…
  </div>

  <section class="performance-panel" aria-labelledby="performance-latest-title">
    <div class="performance-section-head">
      <div>
        <h2 id="performance-latest-title">Latest build</h2>
        <p class="performance-panel-intro">
          Times for the latest successful site build.
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
        <h2 id="performance-history-title">Recent builds</h2>
        <p class="performance-panel-intro">
          Successful documentation builds on <code>main</code>. Some runs reuse the API reference and guide.
        </p>
      </div>
      <label for="performance-metric">
        Metric
        <select id="performance-metric">
          <option value="api">API documentation</option>
          <option value="guide">Verso guide</option>
          <option value="website" selected>Website assembly</option>
          <option value="links">Website links and anchors</option>
          <option value="total">Complete build job</option>
        </select>
      </label>
    </div>
    <div class="performance-chart" id="performance-chart">
      <div class="performance-loading">Loading chart…</div>
    </div>
    <p class="performance-chart-caption" id="performance-chart-caption"></p>
  </section>

  <section class="performance-panel" aria-labelledby="performance-runs-title">
    <h2 id="performance-runs-title">Builds</h2>
    <div class="performance-table-wrap">
      <table class="performance-table">
        <thead>
          <tr>
            <th scope="col">Commit</th>
            <th scope="col">Date</th>
            <th scope="col">API docs</th>
            <th scope="col">Guide</th>
            <th scope="col">Website</th>
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
  const timingsUrl = {{ '/assets/ci-timings.json' | relative_url | jsonify }};

  const metrics = {
    api: { label: "API documentation", step: "Build project API documentation" },
    guide: { label: "Verso guide", step: "Build Verso guide from `home_page/blueprint`" },
    website: { label: "Website assembly", step: "Build website using Jekyll" },
    links: { label: "Website links and anchors", step: "Check generated links and anchors" },
    total: { label: "Complete build job", step: null },
  };

  const notice = document.getElementById("performance-notice");
  const cards = document.getElementById("performance-cards");
  const latestMeta = document.getElementById("performance-latest");
  const chart = document.getElementById("performance-chart");
  const chartCaption = document.getElementById("performance-chart-caption");
  const metricSelect = document.getElementById("performance-metric");
  const runsBody = document.getElementById("performance-runs");
  let currentRecords = [];

  const durationSeconds = (start, finish) => {
    const begin = Date.parse(start);
    const end = Date.parse(finish);
    if (!Number.isFinite(begin) || !Number.isFinite(end) || end < begin) return null;
    return (end - begin) / 1000;
  };

  const stepSeconds = (job, metric) => {
    const step = (job.steps || []).find(candidate => candidate.name === metric.step);
    return step ? durationSeconds(step.started_at, step.completed_at) : null;
  };

  const formatDuration = seconds => {
    if (!Number.isFinite(seconds)) return "n/a";
    const rounded = Math.max(0, Math.round(seconds));
    const hours = Math.floor(rounded / 3600);
    const minutes = Math.floor((rounded % 3600) / 60);
    const remainder = rounded % 60;
    if (hours > 0) return `${hours} h ${minutes} min`;
    if (minutes > 0) return `${minutes} min ${remainder} s`;
    return `${remainder} s`;
  };

  const shortDuration = seconds => {
    if (!Number.isFinite(seconds)) return "n/a";
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

  const loadRecords = async () => {
    const response = await fetch(timingsUrl);
    if (!response.ok) throw new Error(`Timing snapshot returned HTTP ${response.status}`);
    const snapshot = await response.json();
    if (snapshot.repository !== repository || snapshot.workflow !== "blueprint.yml" ||
        !Array.isArray(snapshot.runs) ||
        !Number.isFinite(Date.parse(snapshot.generatedAt))) {
      throw new Error("Invalid CI timing snapshot");
    }
    const records = snapshot.runs.map(run => {
      const job = run.job;
      if (!job) return null;
      const durations = { total: durationSeconds(job.started_at, job.completed_at) };
      for (const [key, metric] of Object.entries(metrics)) {
        if (metric.step) durations[key] = stepSeconds(job, metric);
      }
      return {
        id: run.id,
        sha: run.head_sha,
        title: run.display_title || run.head_commit?.message || "TorchLean CI run",
        url: run.html_url,
        startedAt: job.started_at || run.run_started_at,
        durations,
      };
    });
    return {
      generatedAt: snapshot.generatedAt,
      records: records.filter(record => record !== null)
        .sort((left, right) => Date.parse(left.startedAt) - Date.parse(right.startedAt)),
    };
  };

  const trend = (current, previous) => {
    if (!Number.isFinite(current) || !Number.isFinite(previous) || previous === 0) {
      return { text: "Waiting for another run", direction: "flat" };
    }
    const change = ((current - previous) / previous) * 100;
    if (Math.abs(change) < 0.1) return { text: "Less than 0.1% change", direction: "flat" };
    return {
      text: `${change < 0 ? "↓" : "↑"} ${Math.abs(change).toFixed(1)}% from previous`,
      direction: change < 0 ? "shorter" : "longer",
    };
  };

  const renderCards = records => {
    const latest = records.at(-1);
    const previous = records.at(-2);
    const sameBuildSteps = ["api", "guide"].every(key =>
      Number.isFinite(latest.durations[key]) === Number.isFinite(previous?.durations[key]));
    cards.replaceChildren();
    for (const key of ["api", "guide", "website", "total"]) {
      const card = document.createElement("div");
      card.className = "performance-card";

      const value = document.createElement("div");
      value.className = "performance-value";
      value.textContent = latest.durations[key] == null ? "Reused" : formatDuration(latest.durations[key]);

      const label = document.createElement("div");
      label.className = "performance-label";
      label.textContent = metrics[key].label;

      const comparison = trend(latest.durations[key], previous?.durations[key]);
      const change = document.createElement("div");
      change.className = "performance-trend";
      change.dataset.direction = comparison.direction;
      change.textContent = comparison.text;

      card.append(value, label);
      if (latest.durations[key] != null && (key !== "total" || sameBuildSteps)) {
        card.append(change);
      }
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

  const renderTable = records => {
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
        ...["api", "guide", "website"].map(key => formatDuration(record.durations[key])),
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

  const render = records => {
    if (!records.length) throw new Error("No successful main-branch CI runs were found");
    // Check the snapshot before using its records in the chart and run links.
    for (const record of records) {
      if (!record || !record.durations ||
          typeof record.sha !== "string" || typeof record.url !== "string" ||
          !Number.isFinite(Date.parse(record.startedAt))) {
        throw new Error("Invalid CI timing record");
      }
    }
    currentRecords = records;
    renderCards(records);
    renderChart(records, metricSelect.value);
    renderTable(records);
  };

  metricSelect.addEventListener("change", () => renderChart(currentRecords, metricSelect.value));

  loadRecords()
    .then(({ records, generatedAt }) => {
      render(records);
      setNotice(`Timing data updated ${formatDate(generatedAt)}.`);
    })
    .catch(() => {
      setNotice("The saved timing data could not be loaded. Use the source-runs link above to view CI on GitHub.", "error");
      cards.innerHTML = '<div class="performance-loading">Timing data is unavailable.</div>';
      chart.innerHTML = '<div class="performance-loading">Chart data is unavailable.</div>';
      runsBody.innerHTML = '<tr><td colspan="6">Run history is unavailable.</td></tr>';
    });
})();
</script>
