#!/usr/bin/env python3
"""Small post-build polish pass for the generated Verso guide.

Verso owns the main HTML generation. TorchLean adds a thin local layer for the
public guide: responsive figures, readable code blocks, external-link behavior,
copy buttons, route arrows, and a lightweight reading-progress indicator.
"""

from __future__ import annotations

import argparse
import html
import json
import posixpath
import re
from pathlib import Path
from html.parser import HTMLParser


TORCHLEAN_CSS = """

/* TorchLean guide polish: reference-style reading shell. */
:root {
  --tl-accent: #0f5f8f;
  --tl-accent-strong: #0a3f61;
  --tl-accent-soft: rgba(15, 95, 143, 0.10);
  --tl-border: rgba(32, 52, 71, 0.16);
  --tl-surface: #ffffff;
  --tl-surface-soft: #f7fafc;
  --tl-code-bg: #f5f8fb;
  --tl-code-border: rgba(20, 70, 110, 0.18);
}

html,
body {
  max-width: 100%;
  overflow-x: hidden;
}

body {
  scroll-padding-top: 3.4rem;
}

body.tl-progress-mounted {
  padding-top: 0;
}

body.tl-progress-mounted .with-toc {
  margin-top: calc(var(--verso-header-height, 0px) + 2.35rem);
}

body.tl-progress-mounted main [id] {
  scroll-margin-top: calc(var(--verso-header-height, 0px) + 3.6rem);
}

.tl-reading-progress {
  position: fixed;
  z-index: 10000;
  top: var(--verso-header-height, 0px);
  left: 0;
  right: 0;
  height: 2.35rem;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  gap: 0.75rem;
  padding: 0.25rem clamp(0.75rem, 2vw, 1.35rem);
  border-bottom: 1px solid var(--tl-border);
  background: rgba(255, 255, 255, 0.94);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 24px rgba(25, 48, 73, 0.08);
  color: #172536;
  font-size: 0.82rem;
}

.tl-reading-progress-track {
  position: relative;
  height: 0.46rem;
  overflow: hidden;
  border-radius: 999px;
  background: rgba(29, 54, 78, 0.10);
}

.tl-reading-progress-bar {
  width: 0;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #0f5f8f, #37a28f);
  transition: width 140ms linear;
}

.tl-reading-progress-label {
  white-space: nowrap;
  color: #26384c;
  font-weight: 650;
  letter-spacing: 0.01em;
}

main .content-wrapper {
  max-width: 980px;
}

main a[href^="http"]::after,
main a.tl-auto-link::after {
  content: "↗";
  display: inline-block;
  margin-left: 0.16em;
  font-size: 0.72em;
  line-height: 1;
  opacity: 0.62;
  transform: translateY(-0.08em);
}

main :is(h1, h2, h3, h4, h5, h6) .tl-heading-anchor {
  margin-left: 0.38em;
  color: rgba(15, 95, 143, 0.58);
  text-decoration: none;
  font-size: 0.74em;
  opacity: 0;
  transition: opacity 120ms ease, color 120ms ease;
}

main :is(h1, h2, h3, h4, h5, h6):hover .tl-heading-anchor,
main :is(h1, h2, h3, h4, h5, h6) .tl-heading-anchor:focus {
  opacity: 1;
}

main :is(h1, h2, h3, h4, h5, h6) .tl-heading-anchor:hover {
  color: var(--tl-accent-strong);
}

.prev-next-buttons {
  gap: 0.75rem;
  margin: 0.8rem 0 1.7rem;
}

.prev-next-buttons .local-button {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.62rem 0.78rem;
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  background: linear-gradient(180deg, #ffffff, #f7fafc);
  box-shadow: 0 8px 22px rgba(26, 48, 70, 0.07);
  color: #11263a;
  text-decoration: none;
}

.prev-next-buttons .local-button:hover {
  border-color: rgba(15, 95, 143, 0.35);
  background: #f3f9fc;
  text-decoration: none;
}

.prev-next-buttons .arrow {
  display: inline-grid;
  place-items: center;
  width: 1.45rem;
  height: 1.45rem;
  border-radius: 999px;
  background: var(--tl-accent-soft);
  color: var(--tl-accent-strong);
  font-weight: 800;
}

.prev-next-buttons .where {
  font-weight: 700;
}

.torchlean-route-list {
  display: grid;
  gap: 0.75rem;
  padding-left: 0;
  list-style-position: inside;
}

.torchlean-route-list > li {
  margin: 0 !important;
  padding: 0.8rem 0.9rem;
  border: 1px solid var(--tl-border);
  border-radius: 16px;
  background: linear-gradient(180deg, #ffffff, var(--tl-surface-soft));
  box-shadow: 0 10px 28px rgba(24, 47, 70, 0.06);
}

.tl-route-arrow {
  display: inline-grid;
  place-items: center;
  margin: 0 0.32rem;
  width: 1.35rem;
  height: 1.35rem;
  border-radius: 999px;
  background: var(--tl-accent-soft);
  color: var(--tl-accent-strong);
  font-weight: 900;
}

main pre,
main code.hl.lean.block,
main pre.syntax-error {
  box-sizing: border-box;
  max-width: 100%;
  overflow-x: auto;
  margin: 1rem 0;
  padding: 1rem 1.05rem;
  border: 1px solid var(--tl-code-border);
  border-radius: 14px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.55), rgba(255,255,255,0.10)),
    var(--tl-code-bg);
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.85), 0 12px 28px rgba(25, 48, 73, 0.06);
  color: #102236;
  font-family: var(--verso-code-font-family), ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.92rem;
  line-height: 1.52;
  tab-size: 2;
}

main details.bp_code_block {
  box-sizing: border-box;
  margin: 1rem 0;
  border: 1px solid var(--tl-code-border);
  border-radius: 14px;
  background: var(--tl-code-bg);
  box-shadow: 0 12px 28px rgba(25, 48, 73, 0.06);
  overflow: hidden;
}

main details.bp_code_block > summary {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  cursor: pointer;
  padding: 0.62rem 0.9rem;
  border-bottom: 1px solid rgba(20, 70, 110, 0.14);
  background: linear-gradient(180deg, #ffffff, #f4f8fb);
  color: #102236;
  font-weight: 750;
}

main details.bp_code_block > summary::-webkit-details-marker {
  display: none;
}

main details.bp_code_block > summary::marker {
  content: "";
}

.tl-code-actions {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-left: auto;
}

.tl-code-action {
  display: inline-flex;
  align-items: center;
  gap: 0.24rem;
  padding: 0.24rem 0.58rem;
  border: 1px solid rgba(15, 95, 143, 0.26);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  color: var(--tl-accent-strong);
  font: inherit;
  font-size: 0.72rem;
  font-weight: 750;
  line-height: 1.1;
  text-decoration: none;
  cursor: pointer;
}

.tl-code-action:hover {
  background: #eef8fb;
  text-decoration: none;
}

.tl-code-action-live {
  border-color: rgba(55, 162, 143, 0.34);
  color: #0b5a52;
}

main details.bp_code_block > code.hl.lean.block {
  display: block;
  margin: 0;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  max-height: 34rem;
  overflow: auto;
}

.tl-code-wrap {
  position: relative;
  margin: 1rem 0;
}

.tl-code-wrap > pre {
  margin: 0;
  padding-top: 2.35rem;
}

.tl-copy-code {
  position: absolute;
  top: 0.48rem;
  right: 0.55rem;
  z-index: 2;
  padding: 0.24rem 0.55rem;
  border: 1px solid rgba(15, 95, 143, 0.28);
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.88);
  color: var(--tl-accent-strong);
  font: inherit;
  font-size: 0.72rem;
  font-weight: 750;
  cursor: pointer;
}

.tl-copy-code:hover {
  background: #eef8fb;
}

main :not(pre) > code {
  border: 1px solid rgba(32, 52, 71, 0.12);
  border-radius: 0.32em;
  background: rgba(15, 95, 143, 0.07);
  padding: 0.04em 0.22em;
}

main table.tabular {
  width: 100%;
  max-width: 100%;
  margin: 1.1rem 0 1.35rem;
  border-collapse: separate;
  border-spacing: 0;
  border: 1px solid var(--tl-border);
  border-radius: 16px;
  overflow: hidden;
  background: #ffffff;
  box-shadow: 0 12px 30px rgba(25, 48, 73, 0.06);
  table-layout: fixed;
}

main table.tabular th,
main table.tabular td {
  padding: 0.72rem 0.82rem;
  border-right: 1px solid rgba(32, 52, 71, 0.10);
  border-bottom: 1px solid rgba(32, 52, 71, 0.10);
  vertical-align: top;
  overflow-wrap: anywhere;
}

main table.tabular th:last-child,
main table.tabular td:last-child {
  border-right: 0;
}

main table.tabular tr:last-child td {
  border-bottom: 0;
}

main table.tabular thead th {
  background: linear-gradient(180deg, #f7fbfd, #eef6fa);
  color: #102236;
  font-weight: 800;
}

main table.tabular p {
  margin: 0.2rem 0;
}

.tl-table-wrap {
  box-sizing: border-box;
  max-width: 100%;
  margin: 1.1rem 0 1.35rem;
  overflow-x: auto;
  border-radius: 16px;
  box-shadow: 0 12px 30px rgba(25, 48, 73, 0.06);
}

main .tl-table-wrap > table.tabular {
  min-width: 42rem;
  margin: 0;
  box-shadow: none;
}

main .content-wrapper,
main section,
.with-toc {
  min-width: 0;
  max-width: 100%;
  overflow-x: hidden;
}

.bp_math.display {
  box-sizing: border-box;
  contain: inline-size;
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.45rem 0;
}

.katex-display {
  box-sizing: border-box;
  contain: inline-size;
  display: block;
  width: 100%;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.35rem 0;
}

mjx-container[display="true"] {
  box-sizing: border-box;
  contain: inline-size;
  display: block !important;
  width: 100% !important;
  max-width: 100%;
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.35rem 0;
}

mjx-container {
  max-width: 100%;
}

.katex .katex-mathml,
.katex .katex-mathml > math,
.katex .katex-mathml semantics {
  max-width: 1px !important;
  overflow: hidden !important;
}

main img {
  box-sizing: border-box;
  display: block;
  max-width: 100%;
  height: auto;
}

main p > img:only-child {
  margin: 1.5rem auto;
  border-radius: 18px;
  border: 1px solid rgba(31, 49, 73, 0.12);
  box-shadow: 0 18px 42px rgba(18, 49, 74, 0.10);
  background: #ffffff;
}

@media screen and (max-width: 700px) {
  header .header-title-wrapper {
    flex: 0 0 auto;
    min-width: auto;
  }

  header .header-title h1 {
    font-size: 1.55rem;
    white-space: nowrap;
  }

  header #search-wrapper {
    flex: 1 1 7rem;
    min-width: 6rem;
    max-width: 10rem;
  }

  main p > img:only-child {
    margin: 1rem auto;
    border-radius: 12px;
  }

  .tl-reading-progress {
    grid-template-columns: 1fr;
    height: auto;
    gap: 0.28rem;
    padding: 0.35rem 0.75rem;
  }

  body.tl-progress-mounted {
    padding-top: 0;
  }

  body.tl-progress-mounted .with-toc {
    margin-top: calc(var(--verso-header-height, 0px) + 3.25rem);
  }

  .tl-reading-progress-label {
    font-size: 0.74rem;
  }

  .prev-next-buttons {
    flex-direction: column;
    align-items: stretch;
  }

  .prev-next-buttons .local-button {
    justify-content: space-between;
  }

  main .tl-table-wrap > table.tabular {
    min-width: 36rem;
  }
}

/* End TorchLean guide polish. */
"""


TORCHLEAN_JS_BODY = r"""
(function () {
  const pages = TORCHLEAN_GUIDE_PAGES;
  const storageKey = "torchlean-guide-read-v1";
  const script = document.currentScript;
  const rootUrl = new URL(".", script ? script.src : window.location.href);

  function normalizedCurrentPage() {
    const here = new URL(window.location.href);
    const rootPath = rootUrl.pathname.endsWith("/") ? rootUrl.pathname : rootUrl.pathname + "/";
    if (!here.pathname.startsWith(rootPath)) return null;
    let rel = decodeURIComponent(here.pathname.slice(rootPath.length));
    if (rel === "") rel = "index.html";
    else if (rel.endsWith("/")) rel += "index.html";
    else if (!rel.endsWith(".html")) rel += "/index.html";
    return rel;
  }

  function loadReadSet() {
    try {
      return new Set(JSON.parse(localStorage.getItem(storageKey) || "[]"));
    } catch (_) {
      return new Set();
    }
  }

  function saveReadSet(set) {
    try {
      localStorage.setItem(storageKey, JSON.stringify(Array.from(set).sort()));
    } catch (_) {
      /* localStorage can be unavailable in privacy modes; the page still works. */
    }
  }

  function scrollRatio() {
    const doc = document.documentElement;
    const max = Math.max(1, doc.scrollHeight - window.innerHeight);
    return Math.max(0, Math.min(1, window.scrollY / max));
  }

  function mountProgress() {
    const current = normalizedCurrentPage();
    if (!current || pages.indexOf(current) === -1) return;

    const root = document.createElement("div");
    root.className = "tl-reading-progress";
    root.setAttribute("role", "status");
    root.setAttribute("aria-live", "polite");
    root.innerHTML =
      '<div class="tl-reading-progress-track" aria-hidden="true">' +
      '<div class="tl-reading-progress-bar"></div></div>' +
      '<div class="tl-reading-progress-label"></div>';
    document.body.prepend(root);
    document.body.classList.add("tl-progress-mounted");

    const bar = root.querySelector(".tl-reading-progress-bar");
    const label = root.querySelector(".tl-reading-progress-label");
    const read = loadReadSet();

    function maybeMarkRead(ratio) {
      const doc = document.documentElement;
      const max = doc.scrollHeight - window.innerHeight;
      if (ratio >= 0.72 || max < 700) {
        if (!read.has(current)) {
          read.add(current);
          saveReadSet(read);
        }
      }
    }

    function update() {
      const ratio = scrollRatio();
      maybeMarkRead(ratio);
      const pct = Math.round(ratio * 100);
      bar.style.width = pct + "%";
      label.textContent = "Guide progress: " + read.size + "/" + pages.length + " sections read • " + pct + "% of this page";
    }

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    window.setTimeout(update, 1200);
  }

  function shouldSkipTextNode(node) {
    let el = node.parentElement;
    while (el) {
      const tag = el.tagName;
      if (tag === "A" || tag === "CODE" || tag === "PRE" || tag === "SCRIPT" ||
          tag === "STYLE" || tag === "TEXTAREA" || tag === "INPUT") {
        return true;
      }
      el = el.parentElement;
    }
    return false;
  }

  function autoLinkBareUrls() {
    const root = document.querySelector("main");
    if (!root) return;
    const urlRe = /\bhttps?:\/\/[^\s<>"']*[^\s<>"'.,;:!?)\]]/g;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (true) {
      const node = walker.nextNode();
      if (!node) break;
      if (!shouldSkipTextNode(node) && urlRe.test(node.nodeValue)) {
        nodes.push(node);
      }
      urlRe.lastIndex = 0;
    }
    for (const node of nodes) {
      const text = node.nodeValue;
      const frag = document.createDocumentFragment();
      let last = 0;
      text.replace(urlRe, (match, offset) => {
        if (offset > last) frag.appendChild(document.createTextNode(text.slice(last, offset)));
        const a = document.createElement("a");
        a.href = match;
        a.textContent = match;
        a.className = "tl-auto-link";
        frag.appendChild(a);
        last = offset + match.length;
        return match;
      });
      if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
      node.parentNode.replaceChild(frag, node);
    }
  }

  function replaceAsciiArrowsInText() {
    const root = document.querySelector("main");
    if (!root) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (true) {
      const node = walker.nextNode();
      if (!node) break;
      if (!shouldSkipTextNode(node) && node.nodeValue.includes("->")) {
        nodes.push(node);
      }
    }
    for (const node of nodes) {
      node.nodeValue = node.nodeValue.replace(/->/g, "→");
    }
  }

  function externalLinksOpenInNewTabs() {
    const here = window.location.origin;
    document.querySelectorAll('a[href^="http://"], a[href^="https://"]').forEach((a) => {
      let url;
      try { url = new URL(a.href); } catch (_) { return; }
      if (url.origin !== here) {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
      }
    });
    document.querySelectorAll('a[href*="/docs/"], a[href^="../docs/"], a[href^="../../docs/"], a[href^="../../../docs/"]').forEach((a) => {
      a.target = "_blank";
      a.rel = "noopener noreferrer";
    });
  }

  function enhanceRouteLists() {
    document.querySelectorAll("main ol").forEach((ol) => {
      const text = ol.textContent || "";
      if (!text.includes("I want to") || !(text.includes("→") || text.includes("->"))) return;
      ol.classList.add("torchlean-route-list");
      const walker = document.createTreeWalker(ol, NodeFilter.SHOW_TEXT);
      const nodes = [];
      while (true) {
        const node = walker.nextNode();
        if (!node) break;
        if (!shouldSkipTextNode(node) && (node.nodeValue.includes("→") || node.nodeValue.includes("->"))) {
          nodes.push(node);
        }
      }
      for (const node of nodes) {
        const pieces = node.nodeValue.split(/(→|->)/g);
        const frag = document.createDocumentFragment();
        for (const piece of pieces) {
          if (piece === "→" || piece === "->") {
            const span = document.createElement("span");
            span.className = "tl-route-arrow";
            span.textContent = "→";
            frag.appendChild(span);
          } else if (piece) {
            frag.appendChild(document.createTextNode(piece));
          }
        }
        node.parentNode.replaceChild(frag, node);
      }
    });
  }

  function addCopyButtons() {
    document.querySelectorAll("main pre").forEach((pre) => {
      if (pre.closest(".tl-code-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "tl-code-wrap";
      pre.parentNode.insertBefore(wrap, pre);
      wrap.appendChild(pre);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tl-copy-code";
      button.textContent = "Copy";
      button.addEventListener("click", async () => {
        const text = pre.innerText;
        try {
          await navigator.clipboard.writeText(text);
          button.textContent = "Copied";
          window.setTimeout(() => { button.textContent = "Copy"; }, 1100);
        } catch (_) {
          button.textContent = "Select";
          window.setTimeout(() => { button.textContent = "Copy"; }, 1100);
        }
      });
      wrap.appendChild(button);
    });
  }

  function wrapTables() {
    document.querySelectorAll("main table.tabular").forEach((table) => {
      if (table.closest(".tl-table-wrap")) return;
      const wrap = document.createElement("div");
      wrap.className = "tl-table-wrap";
      table.parentNode.insertBefore(wrap, table);
      wrap.appendChild(table);
    });
  }

  function codeTextOf(code) {
    if (!code) return "";
    return code.innerText || code.textContent || "";
  }

  function openInLeanLive(code) {
    const url = "https://live.lean-lang.org/#code=" + encodeURIComponent(code);
    window.open(url, "_blank", "noopener,noreferrer");
  }

  function openLeanCodePanels() {
    document.querySelectorAll("main details.bp_code_block").forEach((details) => {
      details.open = true;
    });
  }

  function addLeanCodePanelActions() {
    document.querySelectorAll("main details.bp_code_block").forEach((details) => {
      const summary = details.querySelector(":scope > summary");
      const code = details.querySelector(":scope > code.hl.lean.block");
      if (!summary || !code || summary.querySelector(".tl-code-actions")) return;

      const actions = document.createElement("span");
      actions.className = "tl-code-actions";
      actions.addEventListener("click", (event) => event.stopPropagation());

      const copy = document.createElement("button");
      copy.type = "button";
      copy.className = "tl-code-action";
      copy.textContent = "Copy";
      copy.title = "Copy this Lean snippet for local use in the TorchLean repository.";
      copy.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(codeTextOf(code));
          copy.textContent = "Copied";
          window.setTimeout(() => { copy.textContent = "Copy"; }, 1100);
        } catch (_) {
          copy.textContent = "Select";
          window.setTimeout(() => { copy.textContent = "Copy"; }, 1100);
        }
      });

      const live = document.createElement("button");
      live.type = "button";
      live.className = "tl-code-action tl-code-action-live";
      live.textContent = "Live ↪";
      live.title = "Open in live.lean-lang.org. TorchLean-specific snippets still need the local TorchLean project to typecheck.";
      live.addEventListener("click", () => openInLeanLive(codeTextOf(code)));

      actions.appendChild(copy);
      actions.appendChild(live);
      summary.appendChild(actions);
    });
  }


  function addHeadingAnchors() {
    document.querySelectorAll("main h1[id], main h2[id], main h3[id], main h4[id], main h5[id], main h6[id]").forEach((heading) => {
      if (heading.querySelector(".tl-heading-anchor")) return;
      const a = document.createElement("a");
      a.className = "tl-heading-anchor";
      a.href = "#" + heading.id;
      a.setAttribute("aria-label", "Link to this section");
      a.textContent = "§";
      heading.appendChild(a);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    replaceAsciiArrowsInText();
    autoLinkBareUrls();
    externalLinksOpenInNewTabs();
    enhanceRouteLists();
    wrapTables();
    addCopyButtons();
    openLeanCodePanels();
    addLeanCodePanelActions();
    addHeadingAnchors();
    mountProgress();
  });
})();
"""


def guide_pages(root: Path) -> list[str]:
    """Return generated guide pages in the reading order used by the progress widget.

    Verso also emits search pages and hyphen-prefixed implementation pages; those
    are useful assets, but they are not chapters a reader should advance through.
    """
    pages: list[str] = []
    for path in sorted(root.rglob("index.html")):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("-") or "/-" in rel or rel.startswith("find/"):
            continue
        pages.append(rel)
    if "index.html" in pages:
        pages.remove("index.html")
        pages.insert(0, "index.html")
    return pages


def write_js(root: Path) -> None:
    """Write the shared JavaScript bundle used by all polished guide pages."""
    pages_json = json.dumps(guide_pages(root), indent=2)
    js = "const TORCHLEAN_GUIDE_PAGES = " + pages_json + ";\n" + TORCHLEAN_JS_BODY
    (root / "torchlean-guide-polish.js").write_text(js)


def inject_script(root: Path) -> None:
    """Install the shared guide script into every generated HTML page."""
    marker = "torchlean-guide-polish.js"
    script_re = re.compile(r'\s*<script defer src="[^"]*torchlean-guide-polish\.js"></script>\n?')
    # Verso emits a <base> tag on every generated page. A bare script URL is
    # therefore resolved relative to the guide root, even from nested pages.
    tag = '    <script defer src="torchlean-guide-polish.js"></script>\n'
    for path in root.rglob("*.html"):
        html = path.read_text()
        if marker in html:
            new_html = script_re.sub("\n" + tag, html, count=1)
            if new_html != html:
                path.write_text(new_html)
            continue
        if "</head>" not in html:
            continue
        path.write_text(html.replace("  </head>", tag + "  </head>", 1))


def rewrite_repository_links(root: Path) -> None:
    """Turn repository-relative links into public API or source links.

    The guide source is written inside `blueprint/TorchLeanBlueprint`, so links
    like `../../NN/...` are convenient while editing. In the generated website
    those paths point outside the published guide. This post-build pass rewrites them. Lean modules
    with generated DocGen pages go to the API
    reference; other repository files go to GitHub source.
    """

    repo_root = Path(__file__).resolve().parents[2]
    docs_root = repo_root / "home_page" / "docs"
    github_root = "https://github.com/lean-dojo/TorchLean"
    repo_prefixes = (
        "NN/",
        "NN.lean",
        "csrc/",
        "scripts/",
        "blueprint/",
        "home_page/",
        ".github/",
        "README",
        "AI_USAGE",
        "CONTRIBUTING",
        "TRUST_BOUNDARIES",
        "THIRD_PARTY",
        "lakefile",
        "lake-manifest",
        "lean-toolchain",
    )

    def api_href_for(path: Path, normalized: str) -> str | None:
        """Return a relative DocGen URL for a Lean source path when one exists."""
        if not normalized.endswith(".lean"):
            return None
        doc_rel = normalized[:-5] + ".html"
        if not (docs_root / doc_rel).exists():
            return None
        rel_page = path.relative_to(root)
        source_dir = posixpath.join("blueprint", rel_page.parent.as_posix())
        target = posixpath.join("docs", doc_rel)
        return posixpath.relpath(target, source_dir)

    def rewrite_href(path: Path, match: re.Match[str]) -> str:
        """Rewrite one `href=` attribute from generated guide HTML."""
        quote = match.group(1)
        href = match.group(2)
        if "://" in href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            return match.group(0)

        path_part, sep, frag = href.partition("#")
        normalized = path_part
        while normalized.startswith("../"):
            normalized = normalized[3:]
        if normalized.startswith("./"):
            normalized = normalized[2:]

        if not normalized.startswith(repo_prefixes):
            return match.group(0)

        api_href = api_href_for(path, normalized)
        if api_href is not None:
            return f'href={quote}{api_href}{quote}'

        local_target = repo_root / normalized
        if local_target.is_dir() or normalized.endswith("/"):
            kind = "tree"
            normalized = normalized.rstrip("/")
        else:
            kind = "blob"
        new_href = f"{github_root}/{kind}/main/{normalized}"
        if sep:
            new_href += "#" + frag
        return f'href={quote}{new_href}{quote}'

    href_re = re.compile(r'href=([\"\'])([^\"\']+)\1')
    api_link_re = re.compile(
        r'<a([^>]*href=[\"\'][^\"\']*docs/NN/([^\"\']+)\.html[^\"\']*[\"\'][^>]*)>'
        r'([^<]*?\.lean|NN/[^<]*?)</a>'
    )
    anchor_re = re.compile(r'<a\b([^>]*\bhref=([\"\'])([^\"\']+)\2[^>]*)>')
    inline_module_re = re.compile(r"<code>(NN/[A-Za-z0-9_./-]+\.lean)</code>")
    inline_tree_re = re.compile(r"<code>(NN/[A-Za-z0-9_./-]+)/\*</code>")

    def clean_api_label(match: re.Match[str]) -> str:
        """Replace noisy source-file link labels with stable module API labels."""
        attrs = match.group(1)
        module = "NN/" + match.group(2)
        label = match.group(3).strip()
        if ".lean" not in label and not label.startswith("NN/"):
            return match.group(0)
        module_label = module.replace("/", ".") + " API"
        return f"<a{attrs}>{module_label}</a>"

    def link_inline_module(path: Path, match: re.Match[str]) -> str:
        """Turn inline `NN/...lean` code spans into API links when DocGen has them."""
        normalized = match.group(1)
        api_href = api_href_for(path, normalized)
        if api_href is None:
            return match.group(0)
        module_label = normalized[:-5].replace("/", ".") + " API"
        return f'<a href="{api_href}" target="_blank" rel="noopener noreferrer">{module_label}</a>'

    def link_inline_tree(match: re.Match[str]) -> str:
        """Turn inline `NN/.../*` code spans into GitHub source-tree links."""
        normalized = match.group(1).rstrip("/")
        label = normalized.replace("/", ".") + " source tree"
        href = f"{github_root}/tree/main/{normalized}"
        return f'<a href="{href}" target="_blank" rel="noopener noreferrer">{label}</a>'

    def add_blank_target(match: re.Match[str]) -> str:
        """Open external and API-reference links in a separate tab."""
        attrs = match.group(1)
        href = match.group(3)
        is_external = href.startswith(("http://", "https://"))
        is_api_ref = (
            "/docs/" in href
            or href.startswith(("docs/", "../docs/", "../../docs/", "../../../docs/"))
        )
        if not (is_external or is_api_ref) or re.search(r"\btarget=", attrs):
            return match.group(0)
        rel = "" if re.search(r"\brel=", attrs) else ' rel="noopener noreferrer"'
        return f'<a{attrs} target="_blank"{rel}>'

    for path in root.rglob("*.html"):
        html = path.read_text()
        rewritten = href_re.sub(lambda match: rewrite_href(path, match), html)
        rewritten = api_link_re.sub(clean_api_label, rewritten)
        rewritten = inline_module_re.sub(lambda match: link_inline_module(path, match), rewritten)
        rewritten = inline_tree_re.sub(link_inline_tree, rewritten)
        rewritten = anchor_re.sub(add_blank_target, rewritten)
        if rewritten != html:
            path.write_text(rewritten)


class _GuideHtmlRefs(HTMLParser):
    """Small parser for generated guide ids, links, and base hrefs."""

    def __init__(self) -> None:
        super().__init__()
        self.base: str | None = None
        self.ids: set[str] = set()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {k: v for k, v in attrs if v is not None}
        if tag == "base" and data.get("href"):
            self.base = data["href"]
        if data.get("id"):
            self.ids.add(data["id"])
        if data.get("name"):
            self.ids.add(data["name"])
        if data.get("href"):
            self.hrefs.append(data["href"])


def add_fragment_aliases(root: Path) -> None:
    """Add hidden anchor aliases for generated local links whose fragments lack ids.

    Verso's TOC and local navigation sometimes point at page-level or tag-level
    fragments that are meaningful in the manual data but are not emitted as DOM
    ids on the standalone page. Adding zero-size aliases keeps those internal
    links stable without changing visible content.
    """

    pages = [path for path in root.rglob("*.html") if "/-verso-" not in path.as_posix()]
    parsed: dict[Path, _GuideHtmlRefs] = {}

    def parse(path: Path) -> _GuideHtmlRefs:
        if path not in parsed:
            p = _GuideHtmlRefs()
            p.feed(path.read_text(errors="ignore"))
            parsed[path] = p
        return parsed[path]

    def resolve(path: Path, href: str) -> tuple[Path | None, str | None]:
        if href.startswith(("#", "mailto:", "tel:", "javascript:")):
            if href.startswith("#") and len(href) > 1:
                return path, href[1:]
            return None, None
        if "://" in href:
            return None, None
        page = parse(path)
        doc_url = "/" + path.relative_to(root).as_posix()
        base_url = posixpath.normpath(posixpath.join(posixpath.dirname(doc_url), page.base or ""))
        href_path, sep, frag = href.partition("#")
        if not sep or not frag:
            return None, None
        target_url = posixpath.normpath(posixpath.join(base_url, href_path))
        if target_url.startswith("../"):
            return None, None
        target = root / target_url.lstrip("/")
        if target.is_dir():
            target = target / "index.html"
        elif not target.exists() and not target.suffix and (target / "index.html").exists():
            target = target / "index.html"
        if target.exists() and target.is_relative_to(root):
            return target, frag
        return None, None

    aliases: dict[Path, set[str]] = {}
    for path in pages:
        page = parse(path)
        for href in page.hrefs:
            target, frag = resolve(path, href)
            if target is None or frag is None:
                continue
            target_page = parse(target)
            if frag not in target_page.ids:
                aliases.setdefault(target, set()).add(frag)

    for path, ids in aliases.items():
        if not ids:
            continue
        page = parse(path)
        missing = [frag for frag in sorted(ids) if frag not in page.ids]
        if not missing:
            continue
        html_text = path.read_text()
        alias_html = "".join(
            f'<span id="{html.escape(frag, quote=True)}" class="tl-anchor-alias" aria-hidden="true"></span>'
            for frag in missing
        )
        if "<main" in html_text:
            html_text = re.sub(r"(<main\b[^>]*>)", r"\1" + alias_html, html_text, count=1)
        elif "<body" in html_text:
            html_text = re.sub(r"(<body\b[^>]*>)", r"\1" + alias_html, html_text, count=1)
        else:
            html_text = alias_html + html_text
        path.write_text(html_text)


def main() -> int:
    """CLI entry point for the post-Verso guide polish pass."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--guide",
        type=Path,
        required=True,
        help="Generated Verso html-multi directory, e.g. _out/blueprint/html-multi",
    )
    args = parser.parse_args()

    css_path = args.guide / "book.css"
    if not css_path.exists():
        raise SystemExit(f"missing generated stylesheet: {css_path}")

    css = css_path.read_text()
    marker = "/* TorchLean guide polish"
    idx = css.find(marker)
    if idx != -1:
        css = css[:idx].rstrip()
    css_path.write_text(css.rstrip() + TORCHLEAN_CSS)
    write_js(args.guide)
    rewrite_repository_links(args.guide)
    add_fragment_aliases(args.guide)
    inject_script(args.guide)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
