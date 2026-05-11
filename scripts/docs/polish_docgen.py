#!/usr/bin/env python3
"""Polish DocGen output for the public TorchLean website.

DocGen owns the declaration pages. This script only adds a nicer landing page and a thin visual
layer after `lake build NN:docs` has produced `home_page/docs`.

The important design rule is: do not fork DocGen. DocGen keeps producing
search data, declaration pages, source links, sidebars, and module navigation,
then apply small, deterministic post-processing steps that make those artifacts
feel like part of the TorchLean website.
"""

from __future__ import annotations

import argparse
from pathlib import Path


# `append_style` is idempotent: it removes everything after this marker before
# re-appending the TorchLean-specific CSS.  That lets `build_site.sh` and local
# preview loops run repeatedly without duplicating the polish block.
STYLE_MARKER = "/* TorchLean public docs polish */"


def write_index(docs: Path) -> None:
    """Replace DocGen's default index with a TorchLean-specific API landing page.

    DocGen's stock index is a raw module tree.  That is technically complete,
    but not friendly for a project website: users need a few stable entrypoints
    before they start searching thousands of generated declarations.  This page
    keeps the DocGen search scripts and module drawer, but adds a curated front
    door for common TorchLean surfaces.
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
  <title>TorchLean API Docs</title>
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
      <a href="../blueprint/">Guide</a>
      <a href="../examples/">Examples</a>
      <a href="../graphs/">Graphs</a>
    </div>
    <form id="search_form">
      <input type="text" name="q" autocomplete="off" placeholder="Search declarations">
      <button id="search_button" onclick="javascript: form.action='./search.html';">Search</button>
    </form>
  </header>
  <main>
    <a id="top"></a>
    <section class="tl-api-hero">
      <p class="tl-kicker">TorchLean API Docs</p>
      <h1>Exact declarations, organized by API surface.</h1>
      <p>
        These pages are generated from Lean. Use the cards below for the main TorchLean surfaces,
        then use search when you need a specific definition, theorem, or module.
      </p>
      <div class="tl-hero-actions">
        <a href="../blueprint/">Read the Guide</a>
        <a href="../examples/">Run an Example</a>
        <a href="./search.html">Search Declarations</a>
      </div>
    </section>

    <section class="tl-api-grid" aria-label="Main API entrypoints">
      <a class="tl-api-card" href="./NN/Library.html">
        <span>Recommended import</span>
        <strong>NN.Library</strong>
        <p>The broad umbrella import for ordinary downstream use.</p>
      </a>
      <a class="tl-api-card" href="./NN/API/Public.html">
        <span>User facade</span>
        <strong>NN.API.Public</strong>
        <p>The PyTorch-shaped public surface for model code and examples.</p>
      </a>
      <a class="tl-api-card" href="./NN/Tensor/API.html">
        <span>Tensors</span>
        <strong>NN.Tensor.API</strong>
        <p>Tensor operations, shapes, and the user-facing tensor layer.</p>
      </a>
      <a class="tl-api-card" href="./NN/IR/Graph.html">
        <span>Graph IR</span>
        <strong>NN.IR.Graph</strong>
        <p>The op-tagged graph representation used by lowering and verification.</p>
      </a>
      <a class="tl-api-card" href="./NN/IR/Semantics.html">
        <span>Semantics</span>
        <strong>NN.IR.Semantics</strong>
        <p>The executable meaning attached to IR operators and graph evaluation.</p>
      </a>
      <a class="tl-api-card" href="./NN/Runtime/Autograd/TorchLean.html">
        <span>Runtime</span>
        <strong>NN.Runtime.Autograd.TorchLean</strong>
        <p>Runtime autograd, compiled execution, and training support.</p>
      </a>
      <a class="tl-api-card" href="./NN/Verification/CLI.html">
        <span>Verification</span>
        <strong>NN.Verification.CLI</strong>
        <p>The registered certificate and verification command surface.</p>
      </a>
      <a class="tl-api-card" href="./NN/Floats/IEEEExec.html">
        <span>Float32</span>
        <strong>NN.Floats.IEEEExec</strong>
        <p>Executable IEEE-754 binary32 semantics used in float audits.</p>
      </a>
    </section>

    <section class="tl-api-section">
      <h2>Browse By Layer</h2>
      <div class="tl-link-list">
        <a href="./NN/API/Public.html">API</a>
        <a href="./NN/Spec/Core.html">Spec</a>
        <a href="./NN/Runtime/Autograd/TorchLean.html">Runtime</a>
        <a href="./NN/IR/Graph.html">IR</a>
        <a href="./NN/GraphSpec/Core.html">GraphSpec</a>
        <a href="./NN/Proofs/Tensor.html">Proofs</a>
        <a href="./NN/Verification/CLI.html">Verification</a>
        <a href="./NN/Examples/Zoo.html">Examples</a>
      </div>
    </section>

    <section class="tl-api-section tl-task-grid" aria-label="Common documentation tasks">
      <article>
        <h2>Common Tasks</h2>
        <p>
          Start from the thing you want to do, then jump into the declaration namespace from there.
        </p>
      </article>
      <a href="./NN/API/Public.html">
        <strong>Write model code</strong>
        <span>Public tensor and layer facade.</span>
      </a>
      <a href="./NN/IR/Graph.html">
        <strong>Inspect lowering</strong>
        <span>The shared graph object that verification and execution read.</span>
      </a>
      <a href="./NN/Runtime/Autograd/TorchLean.html">
        <strong>Run autograd</strong>
        <span>Runtime training and reverse-mode execution entrypoints.</span>
      </a>
      <a href="./NN/Verification/CLI.html">
        <strong>Check certificates</strong>
        <span>Commands and declarations used by verification demos.</span>
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
</body>
</html>
""",
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
}

header .header_filename {
  color: color-mix(in srgb, var(--text-color) 66%, transparent);
  font-size: 0.92rem;
  min-width: 0;
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
   sit elsewhere in the Jekyll site.  These links prevent the docs from feeling
   like a dead-end generated artifact. */
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

/* Landing-page module drawer ---------------------------------------------
   The landing page uses the same hidden checkbox / iframe navbar that DocGen
   emits everywhere else. The custom index hides the drawer by default and
   expose it with a human label, "Modules", instead of the raw hamburger glyph. */
.tl-docs-index {
  --content-width: min(1080px, calc(100vw - 3rem));
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
  width: min(24rem, calc(100vw - 2rem));
  max-width: none;
  left: 1rem;
  z-index: 4;
  background: var(--body-bg);
  border: 1px solid rgba(127, 127, 127, 0.24);
  border-radius: 12px;
  padding: 0.75rem;
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
   These rules are only for `/docs/`: a hero, entrypoint cards, layer chips,
   task shortcuts, and the declaration-kind legend.  The declaration pages use
   the later `body:not(.tl-docs-index)` rules. */
main {
  padding-bottom: 4rem;
}

.tl-docs-index main {
  max-width: 1080px;
  margin-left: auto;
  margin-right: auto;
}

.tl-api-hero {
  margin: 1.25rem 0 1.5rem;
  padding: 1.7rem 1.8rem;
  border: 1px solid rgba(21, 120, 120, 0.18);
  border-radius: 18px;
  background:
    linear-gradient(135deg, color-mix(in srgb, var(--code-bg) 78%, transparent), transparent),
    color-mix(in srgb, var(--body-bg) 88%, var(--code-bg));
  box-shadow: var(--tl-shadow);
}

.tl-api-hero h1 {
  margin: 0.1rem 0 0.55rem;
  font-size: clamp(1.9rem, 3vw, 2.7rem);
  line-height: 1.08;
}

.tl-api-hero p {
  margin: 0;
  max-width: 780px;
}

.tl-hero-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
  margin-top: 1rem;
}

.tl-hero-actions a {
  display: inline-flex;
  align-items: center;
  padding: 0.5rem 0.72rem;
  border: 1px solid var(--tl-border);
  border-radius: 999px;
  text-decoration: none;
  background: color-mix(in srgb, var(--body-bg) 90%, var(--code-bg));
}

.tl-hero-actions a:first-child {
  background: var(--tl-teal);
  border-color: var(--tl-teal);
  color: white;
}

.tl-kicker {
  color: var(--tl-teal);
  font-weight: 800;
  text-transform: uppercase;
  font-size: 0.78rem;
}

.tl-api-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 0.9rem;
  margin: 1rem 0 1.75rem;
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
  font-size: 0.78rem;
  font-weight: 800;
  text-transform: uppercase;
}

.tl-api-card strong {
  display: block;
  margin: 0.28rem 0 0.36rem;
  font-family: JuliaMono, monospace;
  font-size: 1.03rem;
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
  font-size: 0.95rem;
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
  border: 1px solid var(--tl-border);
  border-radius: 14px;
  background: color-mix(in srgb, var(--body-bg) 92%, var(--code-bg));
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

body:not(.tl-docs-index) main,
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

  .tl-docsite-links {
    display: none;
  }

  .tl-api-hero {
    padding: 1rem;
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

    The generated tree includes every imported library, so a first-time visitor
    sees `Aesop`, `Batteries`, `Lean`, `Mathlib`, and `NN` at the same level.
    For TorchLean docs, `NN` is the useful subtree.  Rather than deleting the
    dependency modules, the script keeps the full tree and adds a clear nudge.
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
    - fix known stale path casing in generated doc links,
    - convert old `tl-docsite-links` nav wrappers into divs,
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

        # Older docstrings used a lowercase `tensor` path.  Linux hosting is
        # case-sensitive, so keep this defensive rewrite until all generated
        # docs come from fresh Lean sources.
        updated = updated.replace(
            "NN/Spec/Core/tensor/Core.html",
            "NN/Spec/Core/Tensor/Core.html",
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
        if "tl-docsite-links" not in updated and '<h2 class="header_filename' in updated:
            depth = len(path.relative_to(docs).parent.parts)
            docs_root = "../" * depth if depth else "./"
            site_root = docs_root + "../"
            links = (
                '<div class="tl-docsite-links" aria-label="TorchLean site links">'
                f'<a href="{docs_root}index.html">Docs Home</a>'
                f'<a href="{site_root}blueprint/">Guide</a>'
                f'<a href="{site_root}examples/">Examples</a>'
                f'<a href="{site_root}graphs/">Graphs</a>'
                "</div>"
            )
            updated = updated.replace('<h2 class="header_filename', links + '<h2 class="header_filename', 1)

        # Only declaration-heavy pages need the color legend.  Module-only pages
        # such as `NN.Library` would waste sidebar space with it.
        if has_declarations and "tl-kind-legend-side" not in updated:
            updated = updated.replace('<nav class="internal_nav">', '<nav class="internal_nav">' + legend, 1)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def main() -> None:
    """Entry point used by `scripts/docs/build_site.sh` and local preview loops."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="home_page/docs", help="DocGen output directory")
    args = parser.parse_args()
    docs = Path(args.docs).resolve()
    if not docs.exists():
        raise SystemExit(f"DocGen output directory does not exist: {docs}")
    write_index(docs)
    append_style(docs)
    add_nav_hint(docs)
    rename_docgen_header(docs)


if __name__ == "__main__":
    main()
