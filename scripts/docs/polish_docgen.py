#!/usr/bin/env python3
"""Polish DocGen output for the public TorchLean website.

DocGen owns the declaration pages. This script only adds a nicer landing page and a thin visual
layer after `lake build NN:docs` has produced `home_page/docs`.

The script keeps DocGen as the source of truth for search data, declaration pages,
source links, sidebars, and module navigation. The post-processing below is small
and deterministic: it makes those generated artifacts feel like part of the
TorchLean website.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit


# `append_style` is idempotent: it removes everything after this marker before
# re-appending the TorchLean-specific CSS.  That lets `build_site.sh` and local
# preview loops run repeatedly without duplicating the polish block.
STYLE_MARKER = "/* TorchLean public docs polish */"

DOCGEN_DEPENDENCY_MODULES = {
    "Aesop",
    "Batteries",
    "ImportGraph",
    "Init",
    "Lake",
    "Lean",
    "LeanSearchClient",
    "Mathlib",
    "Plausible",
    "ProofWidgets",
    "Qq",
    "Std",
}

UPSTREAM_DOCGEN_BASE = "https://leanprover-community.github.io/mathlib4_docs/"

HREF_RE = re.compile(r'href="([^"]+)"')

DOCGEN_MATHJAX_TEX_OPTIONS = r'''    packages: {"[-]": ["noundefined"]},
    macros: {
      llbracket: "{\\lbrack\\!\\lbrack}",
      rrbracket: "{\\rbrack\\!\\rbrack}",
    },
'''


DOC_THEME_SCRIPT = """
<script>
(function () {
  const key = "torchlean-theme";
  const root = document.documentElement;

  function preferredTheme() {
    const stored = window.localStorage && window.localStorage.getItem(key);
    if (stored === "light" || stored === "dark") {
      return stored;
    }
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark"
      : "light";
  }

  function applyTheme(theme) {
    root.setAttribute("data-theme", theme);
    if (window.localStorage) {
      window.localStorage.setItem("theme", theme);
    }
    const button = document.querySelector("[data-doc-theme-toggle]");
    if (button) {
      const isDark = theme === "dark";
      button.textContent = isDark ? "Light" : "Dark";
      button.setAttribute("aria-label", isDark ? "Use light theme" : "Use dark theme");
      button.setAttribute("title", isDark ? "Use light theme" : "Use dark theme");
    }
    document.querySelectorAll("iframe.navframe").forEach(function (frame) {
      try {
        if (frame.contentDocument) {
          frame.contentDocument.documentElement.setAttribute("data-theme", theme);
        }
      } catch (_err) {
        // Same-origin in local/docs builds; ignore if a browser blocks access.
      }
    });
  }

  applyTheme(preferredTheme());

  window.addEventListener("DOMContentLoaded", function () {
    applyTheme(preferredTheme());
    document.querySelectorAll("iframe.navframe").forEach(function (frame) {
      frame.addEventListener("load", function () {
        applyTheme(root.getAttribute("data-theme") || preferredTheme());
      });
    });
    const button = document.querySelector("[data-doc-theme-toggle]");
    if (!button) {
      return;
    }
    button.addEventListener("click", function () {
      const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      window.localStorage && window.localStorage.setItem(key, next);
      applyTheme(next);
    });
  });
})();
</script>
"""


def prune_dependency_pages(docs: Path) -> None:
    """Keep the published API reference focused on TorchLean modules.

    DocGen generates pages for every imported package. That is useful in a local
    theorem-library browser, but expensive on GitHub Pages: Mathlib and Lean's
    own generated pages dominate the artifact size while sending users away
    from the `NN` API surface they came to inspect. We keep shared DocGen assets
    and the TorchLean `NN` pages, and rely on upstream docs for dependencies.
    """
    for name in DOCGEN_DEPENDENCY_MODULES:
        path = docs / name
        if path.is_dir():
            shutil.rmtree(path)
        page = docs / f"{name}.html"
        if page.exists():
            page.unlink()


def rewrite_dependency_links(docs: Path) -> None:
    """Rewrite links to pruned dependencies to the upstream Lean/mathlib docs.

    The public TorchLean site publishes `NN` declaration pages, not a second
    copy of Lean, Std, Mathlib, and other imported packages.  DocGen still emits
    local links to those declarations.  After `prune_dependency_pages` removes
    the dependency pages, those links would become local 404s unless we rewrite
    them to the canonical upstream DocGen site.
    """

    for path in docs.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        def repl(match: re.Match[str]) -> str:
            url = match.group(1)
            if (
                not url
                or url.startswith("#")
                or urlsplit(url).scheme
            ):
                return match.group(0)

            path_part, sep, suffix = url.partition("#")
            query = ""
            if "?" in path_part:
                path_part, query_sep, query = path_part.partition("?")
                suffix = query_sep + query + (sep + suffix if sep else "")
            elif sep:
                suffix = sep + suffix

            if not path_part:
                return match.group(0)

            target = (path.parent / unquote(path_part)).resolve()
            try:
                rel = target.relative_to(docs)
            except ValueError:
                return match.group(0)

            parts = rel.parts
            if not parts:
                return match.group(0)
            first = parts[0].removesuffix(".html")
            if first not in DOCGEN_DEPENDENCY_MODULES:
                return match.group(0)

            upstream = UPSTREAM_DOCGEN_BASE + rel.as_posix()
            return f'href="{upstream}{suffix}"'

        updated = HREF_RE.sub(repl, text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def rewrite_search_links(docs: Path) -> None:
    """Keep dependency search, instance lists, and import links usable after pruning."""
    index_path = docs / "declarations" / "declaration-data.bmp"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for collection, field in [("declarations", "docLink"), ("modules", "url")]:
        for entry in index[collection].values():
            url = entry[field]
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            relative = parsed.path.removeprefix("./")
            root = relative.split("/", 1)[0].removesuffix(".html")
            if root in DOCGEN_DEPENDENCY_MODULES:
                entry[field] = UPSTREAM_DOCGEN_BASE + url.removeprefix("./")
    index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    # DocGen prefixes every index URL with SITE_ROOT. Absolute upstream URLs need
    # URL resolution instead, both on the landing page and in nested modules.
    base = "new URL(SITE_ROOT, window.location.href)"
    replacements = {
        "search.js": [("SITE_ROOT + result[j].docLink",
                       f"new URL(result[j].docLink, {base}).href")],
        "find/find.js": [("SITE_ROOT + result.docLink",
                          f"new URL(result.docLink, {base}).href"),
                         ("window.location.replace(result.link)",
                          "window.location.replace(result.docLink)")],
        "instances.js": [("${SITE_ROOT}${instanceLink}",
                          f"${{new URL(instanceLink, {base}).href}}")],
        "importedBy.js": [("${SITE_ROOT}${moduleLink}",
                           f"${{new URL(moduleLink, {base}).href}}")],
        "how-about.js": [("a.href = docLink;", f"a.href = new URL(docLink, {base}).href;")],
    }
    for filename, edits in replacements.items():
        path = docs / filename
        source = path.read_text(encoding="utf-8")
        for old, new in edits:
            if old not in source and new not in source:
                raise SystemExit(f"DocGen URL handling changed in {path}; update the polish step")
            source = source.replace(old, new)
        path.write_text(source, encoding="utf-8")


def _nearest_existing_doc_page(docs: Path, target: Path) -> Path | None:
    """Return the nearest published module page for a missing DocGen target.

    The public API site intentionally prunes dependency pages and only publishes
    a curated slice of TorchLean module pages.  DocGen can still emit links to a
    deeper `NN/...` module that was typechecked but not published as an HTML
    page.  Rather than ship a local 404, send readers to the nearest published
    ancestor module page.
    """
    try:
        rel = target.relative_to(docs)
    except ValueError:
        return None

    parts = list(rel.parts)
    if not parts or parts[0] != "NN":
        return None

    if parts[-1].endswith(".html"):
        parts[-1] = parts[-1][:-5]

    while len(parts) > 1:
        candidate = docs.joinpath(*parts).with_suffix(".html")
        if candidate.exists():
            return candidate
        parts.pop()

    candidate = docs / "NN.html"
    return candidate if candidate.exists() else None


def rewrite_missing_nn_links(docs: Path) -> None:
    """Rewrite missing local TorchLean links to the nearest emitted module page.

    This keeps generated docs and hand-written DocGen landing pages honest after
    the public site prunes the full DocGen universe.  The link text still names
    the precise declaration/module; the href lands on the closest page that is
    actually present in the published reference.
    """

    for path in docs.rglob("*.html"):
        text = path.read_text(encoding="utf-8")

        def repl(match: re.Match[str]) -> str:
            url = match.group(1)
            if (
                not url
                or url.startswith("#")
                or urlsplit(url).scheme
            ):
                return match.group(0)

            path_part, sep, suffix = url.partition("#")
            query = ""
            if "?" in path_part:
                path_part, query_sep, query = path_part.partition("?")
                suffix = query_sep + query + (sep + suffix if sep else "")
            elif sep:
                suffix = sep + suffix

            if not path_part:
                return match.group(0)

            target = (path.parent / unquote(path_part)).resolve()
            if target.exists():
                return match.group(0)

            replacement = _nearest_existing_doc_page(docs, target)
            if replacement is None:
                return match.group(0)

            rel = os.path.relpath(replacement, path.parent).replace(os.sep, "/")
            return f'href="{quote(rel, safe="/.#?=&:%")}{suffix}"'

        updated = HREF_RE.sub(repl, text)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def rewrite_floatlib_profile_link(docs: Path) -> None:
    """Repair one misresolved prose link only when its precise module page exists."""
    page = docs / "FloatLib/Floats/Formats/BinaryInterchange/Transcendentals.html"
    target = page.parent / "Info/Profile.html"
    if not page.is_file() or not target.is_file():
        return

    text = page.read_text(encoding="utf-8")
    # FloatLib has used both spellings in its prose. DocGen resolves them from the
    # documentation root, but Profile is a sibling module beneath Info/.
    updated = text
    for link in ("Info/Profile.html", "BinaryInterchange/Info/Profile.html"):
        updated = updated.replace(f'href="../../../.././{link}"', 'href="./Info/Profile.html"')
    if updated != text:
        page.write_text(updated, encoding="utf-8")


def write_index(docs: Path) -> None:
    """Replace DocGen's default index with a TorchLean-specific API index.

    DocGen's stock index is a raw module tree.  That is technically complete,
    but hard to scan on a project website. The replacement keeps DocGen search
    and the module drawer, then exposes the main TorchLean declaration groups.
    """
    (docs / "index.html").write_text(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="./style.css">
  <link rel="icon" href="./favicon.svg">
  <link rel="mask-icon" href="./favicon.svg" color="#000000">
  <link rel="prefetch" href=".//declarations/declaration-data.bmp" as="image">
  <title>TorchLean API Reference</title>
  <script defer src="./mathjax-config.js"></script>
  <script defer src="https://cdnjs.cloudflare.com/polyfill/v3/polyfill.min.js?features=es6"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <script>const SITE_ROOT="./";</script>
  <script type="module" src="./jump-src.js"></script>
  <script type="module" src="./search.js"></script>
  <script type="module" src="./expand-nav.js"></script>
  <script type="module" src="./how-about.js"></script>
  <script type="module" src="./instances.js"></script>
  <script type="module" src="./importedBy.js"></script>
</head>
  <body class="tl-docs-index">
  <input id="nav_toggle" type="checkbox">
  <header>
    <h1><label for="nav_toggle"></label><span>TorchLean API</span></h1>
    <div class="tl-docsite-links" aria-label="TorchLean site links">
      <a href="../">Home</a>
      <a href="../installation/">Installation</a>
      <a href="../blueprint/">Guide</a>
      <a href="../examples/">Examples</a>
      <a href="../graphs/">Graphs</a>
      <a href="../tools/">Tools</a>
    </div>
    <button class="tl-doc-theme-toggle" type="button" data-doc-theme-toggle aria-label="Use dark theme" title="Use dark theme">Dark</button>
    <form id="search_form">
      <input type="text" name="q" autocomplete="off" placeholder="Search declarations">
      <button id="search_button" onclick="javascript: form.action='./search.html';">Search</button>
    </form>
  </header>
  <main>
    <a id="top"></a>
    <section class="tl-api-grid" aria-label="Main API entrypoints">
      <a class="tl-api-card" href="./NN.html">
        <strong>Start with one import</strong>
        <span>NN</span>
        <p>The complete TorchLean library.</p>
      </a>
      <a class="tl-api-card" href="./NN/API.html">
        <strong>Write model code</strong>
        <span>NN.API</span>
        <p>The focused application API for models, data, and training.</p>
      </a>
      <a class="tl-api-card" href="./NN/Tensor.html">
        <strong>Work with tensors</strong>
        <span>NN.Tensor</span>
        <p>The focused tensor, shape, and operation import.</p>
      </a>
      <a class="tl-api-card" href="./NN/IR/Graph.html">
        <strong>Inspect graph IR</strong>
        <span>NN.IR.Graph</span>
        <p>The op-tagged graph representation used by lowering and verification.</p>
      </a>
      <a class="tl-api-card" href="./NN/IR/Semantics.html">
        <strong>Read operator semantics</strong>
        <span>NN.IR.Semantics</span>
        <p>The executable meaning attached to IR operators and graph evaluation.</p>
      </a>
      <a class="tl-api-card" href="./NN/Runtime.html">
        <strong>Run training code</strong>
        <span>NN.Runtime</span>
        <p>Runtime autograd, typed graph execution, and training support.</p>
      </a>
      <a class="tl-api-card" href="./NN/Verification.html">
        <strong>Check certificates</strong>
        <span>NN.Verification</span>
        <p>The registered certificate and verification command surface.</p>
      </a>
      <a class="tl-api-card" href="./NN/Floats.html">
        <strong>Use FloatLib formats</strong>
        <span>NN.Floats</span>
        <p>TorchLean adapters for FloatLib's configured binary formats and rounded-real models.</p>
      </a>
    </section>

    <section class="tl-api-section">
      <h2>Browse By Layer</h2>
      <div class="tl-link-list">
        <a href="./NN/API.html">API</a>
        <a href="./NN/Spec.html">Spec</a>
        <a href="./NN/Runtime.html">Runtime</a>
        <a href="./NN/IR/Graph.html">IR</a>
        <a href="./NN/GraphSpec.html">GraphSpec</a>
        <a href="./NN/Proofs.html">Proofs</a>
        <a href="./NN/Verification.html">Verification</a>
        <a href="./NN/Examples.html">Examples</a>
      </div>
    </section>

    <section class="tl-api-section tl-task-grid" aria-label="Common documentation tasks">
      <article>
        <h2>Common Tasks</h2>
        <p>
          Start from the thing you want to do, then jump into the declaration namespace from there.
        </p>
      </article>
      <a href="./NN/API.html">
        <strong>Write model code</strong>
        <span>NN.API</span>
      </a>
      <a href="./NN/IR/Graph.html">
        <strong>Inspect lowering</strong>
        <span>NN.IR.Graph</span>
      </a>
      <a href="./NN/Runtime.html">
        <strong>Run autograd</strong>
        <span>NN.Runtime</span>
      </a>
      <a href="./NN/Verification.html">
        <strong>Check certificates</strong>
        <span>NN.Verification</span>
      </a>
    </section>

    <section class="tl-api-section tl-kind-section" aria-label="Declaration kind legend">
      <h2>Declaration Legend</h2>
      <p>The colored stripe on each declaration page marks what Lean generated.</p>
      <div class="tl-kind-legend">
        <span class="tl-kind-chip tl-kind-def">def / instance</span>
        <span class="tl-kind-chip tl-kind-theorem">theorem</span>
        <span class="tl-kind-chip tl-kind-structure">structure / class</span>
        <span class="tl-kind-chip tl-kind-axiom">axiom / opaque</span>
      </div>
    </section>
  </main>
  <nav class="nav"><iframe src="./navbar.html" class="navframe" frameborder="0"></iframe></nav>
  {DOC_THEME_SCRIPT}
</body>
</html>
""".replace("{DOC_THEME_SCRIPT}", DOC_THEME_SCRIPT),
        encoding="utf-8",
    )


def append_style(docs: Path) -> None:
    """Append the public-site visual layer to DocGen's generated stylesheet.

    The script appends instead of rewriting the base stylesheet because DocGen's CSS also
    controls behavior-heavy pieces such as the module drawer, declaration blocks,
    search results, and responsive sidebar toggling.  The appended rules should
    therefore be mostly cosmetic and conservative: nicer cards, clearer spacing,
    readable sidebars, and a few explanatory widgets.
    """
    style = docs / "style.css"
    current = style.read_text(encoding="utf-8")
    if STYLE_MARKER in current:
        current = current.split(STYLE_MARKER, 1)[0].rstrip() + "\n\n"
    style.write_text(
        current
        + """

{STYLE_MARKER}
""".replace("{STYLE_MARKER}", STYLE_MARKER)
        + """
/* Shared polish variables -------------------------------------------------
   DocGen already defines semantic colors for declaration kinds.  These
   TorchLean variables cover site chrome: teal accents, soft borders, and
   shadows used by the landing page and sidebar cards. */
header h1 span {
  letter-spacing: 0;
}

:root {
  --tl-teal: #157878;
  --tl-teal-soft: rgba(21, 120, 120, 0.13);
  --tl-border: rgba(17, 44, 64, 0.16);
  --tl-shadow: 0 16px 44px rgba(17, 44, 64, 0.08);
  --content-width: clamp(36rem, 64vw, 78rem);
}

html,
body {
  max-width: 100%;
  overflow-x: hidden;
}

body {
  font-size: 18px;
  background:
    radial-gradient(circle at top left, color-mix(in srgb, var(--code-bg) 35%, transparent), transparent 28rem),
    var(--body-bg);
}

/* Header and search -------------------------------------------------------
   Keep DocGen's fixed header model, but make it feel like the rest of the
   public site.  Search remains the real DocGen search form. */
header {
  border-bottom: 1px solid rgba(127, 127, 127, 0.18);
  box-shadow: 0 8px 28px rgba(17, 44, 64, 0.06);
  backdrop-filter: blur(10px);
}

header h1 {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  min-width: 0;
  flex-shrink: 0;
  white-space: nowrap;
}

header .header_filename {
  color: color-mix(in srgb, var(--text-color) 66%, transparent);
  font-size: 0.92rem;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

#search_form {
  white-space: nowrap;
}

#search_form input,
#search_page_query {
  border: 1px solid rgba(127, 127, 127, 0.28);
  border-radius: 999px;
  padding: 0.42rem 0.72rem;
  background: color-mix(in srgb, var(--body-bg) 92%, var(--code-bg));
  color: var(--text-color);
}

#search_button {
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  padding: 0.38rem 0.72rem;
  background: var(--tl-teal);
  color: white;
  cursor: pointer;
}

#search_button:hover {
  filter: brightness(0.94);
}

/* Cross-site links --------------------------------------------------------
   Generated API pages sit under `/docs/`, while the human guide and examples
   sit elsewhere in the Jekyll site.  These links connect the generated API
   pages back to the rest of the public site. */
.tl-docsite-links {
  display: flex;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.86rem;
}

.tl-docsite-links a {
  color: color-mix(in srgb, var(--text-color) 78%, transparent);
  text-decoration: none;
  padding: 0.24rem 0.48rem;
  border-radius: 999px;
}

.tl-docsite-links a:hover {
  color: var(--tl-teal);
  background: var(--tl-teal-soft);
  text-decoration: none;
}

/* Leave room for the module drawer and search before the header links crowd
   those controls. */
@media (max-width: 1200px) {
  .tl-docsite-links {
    display: none;
  }
}

.tl-doc-theme-toggle {
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  padding: 0.28rem 0.62rem;
  background: color-mix(in srgb, var(--body-bg) 90%, var(--code-bg));
  color: color-mix(in srgb, var(--text-color) 82%, transparent);
  cursor: pointer;
  font-size: 0.84rem;
  font-weight: 700;
  line-height: 1;
}

.tl-doc-theme-toggle:hover {
  color: var(--tl-teal);
  border-color: color-mix(in srgb, var(--tl-teal) 45%, transparent);
  background: var(--tl-teal-soft);
}

/* Landing-page module drawer ---------------------------------------------
   The landing page uses the same hidden checkbox / iframe navbar that DocGen
   emits everywhere else. The custom index hides the drawer by default and
   expose it with a human label, "Modules", instead of the raw hamburger glyph. */
.tl-docs-index {
  --content-width: min(1080px, calc(100vw - 3rem));
  --tl-doc-drawer-width: 20rem;
  --tl-doc-open-content-width: min(960px, calc(100vw - var(--tl-doc-drawer-width) - 5rem));
}

.tl-docs-index:has(#nav_toggle:checked) {
  display: grid;
  grid-template-columns: var(--tl-doc-drawer-width) var(--tl-doc-open-content-width);
  justify-content: start;
  column-gap: 1.5rem;
  padding: 0 1.25rem 0 clamp(1.5rem, 3vw, 3rem);
}

.tl-docs-index header {
  justify-content: flex-start;
  gap: 1rem;
}

.tl-docs-index #search_form {
  margin-left: auto;
}

.tl-docs-index .header_filename {
  display: none;
}

.tl-docs-index .nav {
  display: none;
}

.tl-docs-index #nav_toggle:checked ~ .nav {
  display: block;
  position: sticky;
  top: calc(var(--header-height) + 1rem);
  grid-column: 1;
  grid-row: 2;
  align-self: start;
  width: 100%;
  max-width: none;
  height: calc(100vh - var(--header-height) - 2rem);
  left: auto;
  z-index: 4;
  background: var(--body-bg);
  border: 1px solid rgba(127, 127, 127, 0.24);
  border-radius: 12px;
  margin: calc(var(--header-height) + 1rem) 0 1rem;
  overflow: hidden;
  padding: 0;
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.2);
}

.tl-docs-index #nav_toggle:checked ~ main {
  grid-column: 2;
  grid-row: 2;
  width: 100%;
  max-width: var(--tl-doc-open-content-width);
  margin: calc(var(--header-height) + 1rem) 0 4rem;
}

.tl-docs-index #nav_toggle:checked ~ main .tl-api-grid,
.tl-docs-index #nav_toggle:checked ~ main .tl-task-grid {
  grid-template-columns: 1fr;
}

.tl-docs-index #nav_toggle:checked ~ main .tl-api-card,
.tl-docs-index #nav_toggle:checked ~ main .tl-task-grid > a,
.tl-docs-index #nav_toggle:checked ~ main .tl-task-grid > article {
  min-height: auto;
  min-width: 0;
  padding: 0.8rem 0.95rem;
}

.tl-docs-index #nav_toggle:checked ~ main .tl-api-grid,
.tl-docs-index #nav_toggle:checked ~ main .tl-api-section {
  margin-top: 0.9rem;
  margin-bottom: 1rem;
}

#settings {
  display: none !important;
}

.tl-docs-index label[for="nav_toggle"] {
  display: inline-block;
  margin-right: 0.5rem;
  border: 1px solid var(--hamburger-border-color);
  padding: 0.25rem 0.62rem;
  cursor: pointer;
  background: var(--hamburger-bg-color);
  border-radius: 999px;
}

.tl-docs-index label[for="nav_toggle"]::before {
  content: 'Modules';
}

/* Landing-page layout -----------------------------------------------------
   These rules are only for `/docs/`: entrypoint cards, layer chips, task
   shortcuts, and the declaration-kind legend. The declaration pages use the
   later `body:not(.tl-docs-index)` rules. */
main {
  padding-bottom: 4rem;
}

.tl-docs-index main {
  max-width: 1080px;
  margin-left: auto;
  margin-right: auto;
}

.tl-api-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.9rem;
  margin: 1.35rem 0 1.75rem;
}

.tl-api-card {
  display: block;
  min-height: 9rem;
  padding: 1.05rem 1.1rem;
  border: 1px solid var(--tl-border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
  color: var(--text-color);
  text-decoration: none;
  transition: border-color 140ms ease, transform 140ms ease, box-shadow 140ms ease;
}

.tl-api-card:hover {
  border-color: rgba(21, 120, 120, 0.56);
  transform: translateY(-2px);
  box-shadow: var(--tl-shadow);
  text-decoration: none;
}

.tl-api-card span {
  display: block;
  color: var(--tl-teal);
  font-family: JuliaMono, monospace;
  font-size: 0.78rem;
  font-weight: 800;
  overflow-wrap: anywhere;
}

.tl-api-card strong {
  display: block;
  margin: 0 0 0.32rem;
  font-size: 1.08rem;
  overflow-wrap: anywhere;
}

.tl-api-card p {
  margin: 0;
  color: color-mix(in srgb, var(--text-color) 78%, transparent);
  font-size: 0.94rem;
}

.tl-api-section {
  margin: 1.45rem 0;
}

.tl-api-section h2 {
  color: var(--tl-teal);
  margin-bottom: 0.45rem;
}

.tl-docs-index .tl-api-section:last-of-type {
  margin-bottom: 0;
}

.tl-link-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}

.tl-link-list a {
  display: inline-block;
  padding: 0.42rem 0.62rem;
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  text-decoration: none;
}

.tl-link-list a:hover {
  border-color: rgba(21, 120, 120, 0.56);
  text-decoration: none;
}

.tl-task-grid {
  display: grid;
  grid-template-columns: minmax(240px, 1.35fr) repeat(2, minmax(210px, 1fr));
  gap: 0.8rem;
  align-items: stretch;
}

.tl-task-grid article,
.tl-task-grid > a {
  border: 1px solid var(--tl-border);
  border-radius: 14px;
  padding: 1rem;
  background: color-mix(in srgb, var(--body-bg) 90%, var(--code-bg));
}

.tl-task-grid article {
  grid-row: span 2;
}

.tl-task-grid article h2 {
  margin-top: 0;
}

.tl-task-grid > a {
  display: block;
  text-decoration: none;
}

.tl-task-grid > a:hover {
  border-color: rgba(21, 120, 120, 0.56);
  box-shadow: var(--tl-shadow);
}

.tl-task-grid strong {
  display: block;
  margin-bottom: 0.25rem;
  color: var(--text-color);
}

.tl-task-grid span {
  display: block;
  color: color-mix(in srgb, var(--text-color) 72%, transparent);
  font-family: JuliaMono, monospace;
  font-size: 0.86rem;
  overflow-wrap: anywhere;
}

.tl-kind-section {
  padding: 0.95rem 1rem;
  border: 1px solid var(--tl-border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--body-bg) 91%, var(--code-bg));
}

.tl-kind-section p {
  margin: 0.1rem 0 0.75rem;
  color: color-mix(in srgb, var(--text-color) 72%, transparent);
}

.tl-kind-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.tl-kind-chip {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  padding: 0.38rem 0.6rem;
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  background: color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
  font-family: JuliaMono, monospace;
  font-size: 0.86rem;
}

.tl-kind-chip::before {
  content: "";
  display: inline-block;
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 999px;
  background: var(--tl-kind-color, var(--text-color));
}

.tl-kind-def {
  --tl-kind-color: var(--def-color);
}

.tl-kind-theorem {
  --tl-kind-color: var(--theorem-color);
}

.tl-kind-structure {
  --tl-kind-color: var(--structure-and-inductive-color);
}

.tl-kind-axiom {
  --tl-kind-color: var(--axiom-and-constant-color);
}

/* Generated declaration pages --------------------------------------------
   DocGen outputs one main module doc block followed by declaration blocks.
   The base HTML is excellent for linking and search, but visually flat.  These
   rules wrap module docs and declarations in subtle cards while preserving the
   colored left/top borders that encode declaration kind. */
body:not(.tl-docs-index) main {
  padding-top: 0.4rem;
}

body:not(.tl-docs-index) .mod_doc {
  margin: 0 0 1.2rem;
  padding: 1rem 1.1rem;
  border: 1px solid var(--tl-border);
  border-radius: 16px;
  background: color-mix(in srgb, var(--body-bg) 91%, var(--code-bg));
  box-shadow: 0 10px 32px rgba(17, 44, 64, 0.05);
}

body:not(.tl-docs-index) .mod_doc h1:first-child,
body:not(.tl-docs-index) main > h1:first-child {
  margin-top: 0;
}

body:not(.tl-docs-index) .decl {
  margin: 1.2rem 0;
}

body:not(.tl-docs-index) .decl > div {
  border-radius: 14px;
  padding: 0.9rem 1rem;
  background: color-mix(in srgb, var(--body-bg) 93%, var(--code-bg));
  box-shadow: 0 10px 28px rgba(17, 44, 64, 0.045);
}

/* Right-side page navigation ---------------------------------------------
   The `.internal_nav` column contains imports, imported-by, and declaration
   anchors for the current page. This polish pass adds cards and hover affordances, plus the
   compact declaration-kind legend on pages with declarations. */
.internal_nav {
  padding: 0.8rem;
  border: 1px solid var(--tl-border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--body-bg) 92%, var(--code-bg));
}

.internal_nav p:first-child {
  margin-top: 0;
}

.internal_nav .imports {
  padding: 0.55rem;
  border: 1px solid rgba(127, 127, 127, 0.16);
  border-radius: 10px;
  background: color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
}

.internal_nav .nav_link {
  margin: 0.28rem 0;
  padding: 0.24rem 0.35rem 0.24rem 2.35ex;
  border-radius: 8px;
}

.internal_nav .nav_link:hover {
  background: var(--tl-teal-soft);
}

.tl-kind-legend-side {
  margin: 0 0 0.8rem;
  padding: 0.65rem;
  border: 1px solid rgba(127, 127, 127, 0.16);
  border-radius: 10px;
  background: color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
}

.tl-kind-legend-side strong {
  display: block;
  margin-bottom: 0.35rem;
  color: color-mix(in srgb, var(--text-color) 80%, transparent);
  font-size: 0.84rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.tl-kind-legend-side .tl-kind-legend {
  gap: 0.35rem;
}

.tl-kind-legend-side .tl-kind-chip {
  padding: 0.22rem 0.42rem;
  font-size: 0.74rem;
}

.tl-kind-legend-side .tl-kind-chip::before {
  width: 0.55rem;
  height: 0.55rem;
}

/* Left module tree --------------------------------------------------------
   The left iframe is generated by DocGen from all imported modules, including
   Lean, Mathlib, and dependencies. A hint and highlight for `NN` show
   users know where the TorchLean modules live.

   DocGen's default desktop layout keeps this tree permanently visible in a
   side column. That works for wide theorem-library docs, but TorchLean pages
   often contain long module docs and code blocks, so the tree can overlap the
   readable content. Treat it as an explicit drawer on every viewport instead:
   the header's Modules tab opens it, and the same tab closes it. */
label[for="nav_toggle"] {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  margin-right: 0.65rem;
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  padding: 0.42rem 0.75rem;
  cursor: pointer;
  background: color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
  color: var(--text-color);
  font-size: 0.88rem;
  font-weight: 700;
  line-height: 1;
  box-shadow: 0 6px 18px rgba(17, 44, 64, 0.08);
}

label[for="nav_toggle"]::before {
  content: "Modules";
}

#nav_toggle:checked ~ header label[for="nav_toggle"] {
  background: var(--tl-teal);
  color: white;
}

#nav_toggle:checked ~ header label[for="nav_toggle"]::before {
  content: "Close modules";
}

#nav_toggle:not(:checked) ~ .nav {
  display: none;
}

#nav_toggle:checked ~ .nav {
  display: block;
  left: 1rem;
  margin-left: 0;
  width: min(28rem, calc(100vw - 2rem));
  max-width: min(28rem, calc(100vw - 2rem));
  z-index: 20;
  padding: 0;
  border-radius: 16px;
  background: color-mix(in srgb, var(--body-bg) 95%, var(--code-bg));
  box-shadow: 0 24px 80px rgba(0, 0, 0, 0.24);
}

#nav_toggle:checked ~ main {
  visibility: visible;
}

.nav {
  padding-right: 0.35rem;
}

.navframe {
  border: 0;
  border-radius: 10px;
  background: transparent;
}

.tl-docs-index .navframe {
  display: block;
  width: 100%;
  height: 100%;
}

body > .navframe {
  width: 100%;
  height: 100%;
  overflow: hidden;
}

body > .navframe > .nav {
  box-sizing: border-box;
  left: 0;
  width: 100%;
  max-width: none;
  margin: 0;
  padding: 1rem 1.05rem 1.25rem;
  font-size: 0.94rem;
}

body > .navframe > .nav h3:first-child {
  margin-top: 0;
}

body > .navframe > .nav h3 {
  margin: 1.15rem 0 0.45rem;
  font-size: 1rem;
  line-height: 1.2;
}

body > .navframe > .nav .nav_link,
body > .navframe > .nav summary {
  line-height: 1.5;
  overflow-wrap: anywhere;
}

.tl-nav-hint {
  margin: 0.25rem 0 0.65rem;
  padding: 0.55rem 0.65rem;
  border: 1px solid rgba(21, 120, 120, 0.28);
  border-radius: 10px;
  color: color-mix(in srgb, var(--text-color) 82%, transparent);
  background: var(--tl-teal-soft);
  font-size: 0.92rem;
  line-height: 1.35;
}

.tl-nav-hint strong {
  color: var(--text-color);
}

.tl-nav-hint span {
  display: block;
  color: var(--tl-teal);
  font-size: 1.3rem;
  line-height: 1.1;
}

.nav details[data-path="./NN.html"] > summary {
  position: relative;
  padding: 0.2rem 0.25rem;
  border-radius: 8px;
  background: var(--tl-teal-soft);
}

.nav details[data-path="./NN.html"] > summary::after {
  content: "TorchLean";
  margin-left: 0.35rem;
  padding: 0.08rem 0.35rem;
  border-radius: 999px;
  background: var(--tl-teal);
  color: white;
  font-size: 0.72rem;
  font-weight: 700;
}

/* Declaration text and search results ------------------------------------
   Long Lean names and types need breathing room. Monospace stays in place for Lean
   fragments, but slightly increase line height and add understated search
   result separators. */
.nav .nav_file a {
  display: inline-block;
  padding: 0.08rem 0.24rem;
  border-radius: 7px;
}

.nav .nav_file a:hover {
  background: var(--tl-teal-soft);
  text-decoration: none;
}

.decl_header {
  line-height: 1.65;
}

.decl_type,
.structure_field_info,
.constructor {
  line-height: 1.58;
}

pre {
  border: 1px solid rgba(127, 127, 127, 0.16);
  overflow-x: auto;
}

code {
  font-size: 0.92em;
}

/* DocGen places the declaration sidebar beside its content column. Keep that
   column's width when allowing long Lean declarations to scroll; expanding main
   to the full viewport would put the text underneath the fixed sidebar. */
body:not(.tl-docs-index) main {
  min-width: 0;
  overflow-x: auto;
}

body:not(.tl-docs-index) .mod_doc,
body:not(.tl-docs-index) .def,
body:not(.tl-docs-index) .theorem,
body:not(.tl-docs-index) .opaque,
body:not(.tl-docs-index) .axiom,
body:not(.tl-docs-index) .class,
body:not(.tl-docs-index) .structure,
body:not(.tl-docs-index) .inductive,
body:not(.tl-docs-index) .instance {
  min-width: 0;
  max-width: 100%;
  overflow-x: auto;
}

body:not(.tl-docs-index) .mod_doc a,
body:not(.tl-docs-index) .mod_doc code,
body:not(.tl-docs-index) li code,
body:not(.tl-docs-index) p code,
body:not(.tl-docs-index) :not(pre) > code {
  overflow-wrap: anywhere;
  word-break: break-word;
  white-space: normal !important;
}

#kinds {
  margin: 0.8rem 0 1rem;
  padding: 0.75rem;
  border: 1px solid var(--tl-border);
  border-radius: 12px;
  background: color-mix(in srgb, var(--body-bg) 90%, var(--code-bg));
}

#search_results .result_link,
#search_results .result_doc {
  border-bottom: 1px solid rgba(127, 127, 127, 0.22);
  padding: 0.45rem 0.35rem;
}

/* Mobile cleanup ----------------------------------------------------------
   Small screens cannot support both the site nav and the DocGen drawer.  The
   drawer stays available through the Modules button; secondary site links and
   the right-sidebar declaration legend collapse away. */
@media (max-width: 700px) {
  .tl-docs-index {
    --content-width: calc(100vw - 1.25rem);
  }

  .tl-docs-index:has(#nav_toggle:checked) {
    display: block;
    padding: 0;
  }

  .tl-docs-index header {
    gap: 0.5rem;
  }

  .tl-docs-index header h1 span {
    display: none;
  }

  .tl-docs-index #search_form {
    flex: 1 1 100%;
    margin-left: 0;
  }

  .tl-doc-theme-toggle {
    order: 2;
  }

  .tl-docs-index #nav_toggle:checked ~ .nav {
    position: fixed;
    top: calc(var(--header-height) + 0.75rem);
    left: 0.75rem;
    right: 0.75rem;
    bottom: 0.75rem;
    width: auto;
    max-width: none;
  }

  .tl-docs-index #nav_toggle:checked ~ main {
    max-width: var(--content-width);
    margin-left: auto;
    margin-right: auto;
  }

  .tl-docsite-links {
    display: none;
  }

  .tl-api-grid,
  .tl-task-grid {
    grid-template-columns: 1fr;
  }

  .tl-task-grid article {
    grid-row: auto;
  }

  .tl-kind-legend-side {
    display: none;
  }
}
""",
        encoding="utf-8",
    )


def add_nav_hint(docs: Path) -> None:
    """Add a small hint above DocGen's module tree.

    The generated tree can still contain dependency names in cached navigation
    metadata. The published pages are pruned to TorchLean modules, so this hint
    points visitors at the `NN` subtree and states the intended scope directly.
    """
    nav = docs / "navbar.html"
    if not nav.exists():
        return
    text = nav.read_text(encoding="utf-8")
    marker = '<h3>Library</h3><div class="module_list">'
    if marker not in text:
        return
    hint = (
        '<h3>Library</h3>'
        '<div class="tl-nav-hint">'
        'Open <strong>NN</strong> for TorchLean modules.'
        '<span aria-hidden="true">↓</span>'
        '</div>'
        '<div class="module_list">'
    )
    text = text.replace(marker, hint, 1)
    nav.write_text(text, encoding="utf-8")


def rename_docgen_header(docs: Path) -> None:
    """Patch generated declaration pages in-place.

    This function performs the per-page DocGen rewrites that are easiest to
    express against generated HTML fragments:

    - add cross-site links to each generated header,
    - normalize path casing in generated doc links,
    - convert `tl-docsite-links` nav wrappers into divs,
    - add a compact declaration-kind legend to pages with declarations.

    These are narrow string rewrites against stable DocGen fragments.
    If DocGen changes its HTML shape, this function should fail harmlessly by
    skipping a rewrite rather than corrupting pages.
    """
    legend = (
        '<div class="tl-kind-legend-side" aria-label="Declaration kind legend">'
        "<strong>Declaration colors</strong>"
        '<div class="tl-kind-legend">'
        '<span class="tl-kind-chip tl-kind-def">def</span>'
        '<span class="tl-kind-chip tl-kind-theorem">theorem</span>'
        '<span class="tl-kind-chip tl-kind-structure">structure</span>'
        '<span class="tl-kind-chip tl-kind-axiom">axiom</span>'
        "</div>"
        "</div>"
    )
    for path in docs.rglob("*.html"):
        # The custom landing page already has its own header and full legend.
        if path.name == "index.html" and path.parent == docs:
            continue
        text = path.read_text(encoding="utf-8")
        has_declarations = 'class="decl"' in text

        # Rebrand DocGen's default header without changing its search form or
        # module name display.
        updated = text.replace("<span>Documentation</span>", "<span>TorchLean API</span>")

        # Normalize the generated Tensor page path for case-sensitive hosting.
        updated = updated.replace(
            "NN/Spec/Core/tensor/Core.html",
            "NN/Spec/Core/Tensor/Core.html",
        )
        semantic_equivalence_source = (
            "https://github.com/lean-dojo/TorchLean/blob/main/"
            "NN/Runtime/Autograd/IRExec/Correctness/SemanticEquivalence.lean"
        )
        updated = re.sub(
            r'href="[^"]*Correctness/SemanticEquivalence\.html"',
            f'href="{semantic_equivalence_source}"',
            updated,
        )
        updated = re.sub(
            r'href="[^"]*https://github\.com/lean-dojo/TorchLean/blob/main/'
            r'NN/Runtime/Autograd/(?:TypedGraph/)?IRExec/Correctness/'
            r'SemanticEquivalence\.lean"',
            f'href="{semantic_equivalence_source}"',
            updated,
        )

        # DocGen's global CSS treats every `nav` as a fixed sidebar, so normalize
        # this tiny cross-site link group to a plain div before adding links.
        updated = updated.replace(
            '<nav class="tl-docsite-links" aria-label="TorchLean site links">',
            '<div class="tl-docsite-links" aria-label="TorchLean site links">',
        )
        updated = updated.replace("</nav><h2 class=\"header_filename", "</div><h2 class=\"header_filename")

        # Header links are relative to each generated page.  `docs_root` points
        # back to `/docs/`; `site_root` points back to the Jekyll site root.
        depth = len(path.relative_to(docs).parent.parts)
        docs_root = "../" * depth if depth else "./"
        site_root = docs_root + "../"
        if "tl-docsite-links" not in updated and '<h2 class="header_filename' in updated:
            links = (
                '<div class="tl-docsite-links" aria-label="TorchLean site links">'
                f'<a href="{docs_root}index.html">Docs Home</a>'
                f'<a href="{site_root}installation/">Installation</a>'
                f'<a href="{site_root}blueprint/">Guide</a>'
                f'<a href="{site_root}examples/">Examples</a>'
                f'<a href="{site_root}graphs/">Graphs</a>'
                f'<a href="{site_root}tools/">Tools</a>'
                "</div>"
            )
            updated = updated.replace('<h2 class="header_filename', links + '<h2 class="header_filename', 1)
        elif "tl-docsite-links" in updated and ">Tools</a>" not in updated:
            # Keep repeated polish runs useful after the cross-site navigation
            # grows. Existing generated pages may already contain the wrapper,
            # so append the new destination instead of treating them as done.
            links_start = updated.find(
                '<div class="tl-docsite-links" aria-label="TorchLean site links">'
            )
            links_end = updated.find("</div>", links_start)
            if links_start != -1 and links_end != -1:
                tools_link = f'<a href="{site_root}tools/">Tools</a>'
                updated = updated[:links_end] + tools_link + updated[links_end:]

        if "data-doc-theme-toggle" not in updated and '<h2 class="header_filename' in updated:
            theme_button = (
                '<button class="tl-doc-theme-toggle" type="button" data-doc-theme-toggle '
                'aria-label="Use dark theme" title="Use dark theme">Dark</button>'
            )
            updated = updated.replace('<h2 class="header_filename', theme_button + '<h2 class="header_filename', 1)

        if "data-doc-theme-toggle" in updated and "torchlean-theme" not in updated:
            updated = updated.replace("</body>", DOC_THEME_SCRIPT + "\n</body>", 1)

        # Only declaration-heavy pages need the color legend.  Module-only pages
        # such as `NN` would waste sidebar space with it.
        if has_declarations and "tl-kind-legend-side" not in updated:
            updated = updated.replace('<nav class="internal_nav">', '<nav class="internal_nav">' + legend, 1)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def configure_math_runtime(docs: Path) -> None:
    """Add the small TeX compatibility layer used by TorchLean docstrings."""
    config = docs / "mathjax-config.js"
    if not config.is_file():
        raise SystemExit(f"missing DocGen MathJax configuration: {config}")

    text = config.read_text(encoding="utf-8")
    if DOCGEN_MATHJAX_TEX_OPTIONS in text:
        return

    anchor = '    displayMath: [["$$", "$$"]],\n'
    if anchor not in text:
        raise SystemExit(f"cannot extend the DocGen MathJax configuration: {config}")
    config.write_text(
        text.replace(anchor, anchor + DOCGEN_MATHJAX_TEX_OPTIONS, 1),
        encoding="utf-8",
    )


def validate_math_runtime(docs: Path) -> None:
    """Require MathJax on every generated page that has document content."""
    config = docs / "mathjax-config.js"
    if not config.is_file() or config.stat().st_size == 0:
        raise SystemExit(f"missing DocGen MathJax configuration: {config}")

    config_text = config.read_text(encoding="utf-8")
    if 'inlineMath: [["$", "$"]]' not in config_text or 'displayMath: [["$$", "$$"]]' not in config_text:
        raise SystemExit(f"DocGen MathJax configuration does not enable dollar delimiters: {config}")
    if DOCGEN_MATHJAX_TEX_OPTIONS not in config_text:
        raise SystemExit(f"DocGen MathJax compatibility macros are not configured: {config}")

    missing: list[Path] = []
    content_pages = 0
    for path in docs.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        if "<main" not in text:
            continue
        content_pages += 1
        if (
            "mathjax-config.js" not in text
            or "mathjax@3/es5/tex-mml-chtml.js" not in text
        ):
            missing.append(path)

    if content_pages == 0:
        raise SystemExit(f"no DocGen content pages found under {docs}")
    if missing:
        sample = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f" (and {len(missing) - 5} more)"
        raise SystemExit(f"MathJax is not loaded by every DocGen content page: {sample}{suffix}")


def main() -> None:
    """Entry point used by `scripts/docs/build_site.sh` and local preview loops."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="home_page/docs", help="DocGen output directory")
    args = parser.parse_args()
    docs = Path(args.docs).resolve()
    if not docs.exists():
        raise SystemExit(f"DocGen output directory does not exist: {docs}")
    prune_dependency_pages(docs)
    rewrite_dependency_links(docs)
    rewrite_search_links(docs)
    write_index(docs)
    append_style(docs)
    add_nav_hint(docs)
    rename_docgen_header(docs)
    rewrite_floatlib_profile_link(docs)
    rewrite_missing_nn_links(docs)
    configure_math_runtime(docs)
    validate_math_runtime(docs)


if __name__ == "__main__":
    main()
