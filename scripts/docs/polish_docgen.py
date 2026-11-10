#!/usr/bin/env python3
"""Polish DocGen output for the public TorchLean website.

Run after building the API documentation and copying it to `home_page/docs`, as
`scripts/docs/build_site.sh` does. The script adds the landing page and the styles of
`assets/docgen.css`, removes local dependency pages and rewrites their links to upstream
documentation. Search data and navigation are adjusted to the published URLs, and DocGen supplies
the declaration content and source links. With `--importgraph PAGE`, the script styles only the
generated import viewer.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote, urlsplit


ASSETS = Path(__file__).resolve().parent / "assets"

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

# The light/dark theme toggle, inlined into every page that has the toggle button.
DOC_THEME_SCRIPT = (
    "\n<script>\n" + (ASSETS / "docgen-theme.js").read_text(encoding="utf-8") + "</script>\n"
)


def prune_dependency_pages(docs: Path) -> None:
    """Keep the published API reference focused on TorchLean modules.

    DocGen generates pages for every imported package. That suits a local theorem-library
    browser, but on GitHub Pages the Mathlib and Lean pages dominate the artifact size and
    lead readers away from the TorchLean API. We delete the pages of the packages in
    `DOCGEN_DEPENDENCY_MODULES`, keep the shared DocGen assets and the TorchLean and FloatLib
    pages, and rely on upstream docs for the deleted packages.
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

    The public TorchLean site publishes the TorchLean and FloatLib declaration pages, with no
    second copy of Lean, Std, Mathlib and the other packages in `DOCGEN_DEPENDENCY_MODULES`.
    DocGen emits local links to those declarations.  After `prune_dependency_pages` removes
    the dependency pages, those links would become local 404s unless we rewrite them to the
    canonical upstream DocGen site.
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

    # DocGen prefixes every index URL with SITE_ROOT, which breaks absolute upstream
    # URLs. Resolving each URL against SITE_ROOT with `new URL` handles relative and
    # absolute URLs alike, on the landing page and in nested modules.
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


def rewrite_floatlib_profile_link(docs: Path) -> None:
    """Repair one misresolved prose link when its page and its target module page exist."""
    page = docs / "FloatLib/Floats/Formats/BinaryInterchange/Transcendentals.html"
    target = page.parent / "Info/Profile.html"
    if not page.is_file() or not target.is_file():
        return

    text = page.read_text(encoding="utf-8")
    # The module doc names `Info.Profile`. DocGen resolves that name from the documentation root,
    # but the module is `BinaryInterchange.Info.Profile`, beneath this page's directory.
    updated = text.replace('href="../../../.././Info/Profile.html"', 'href="./Info/Profile.html"')
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
      <a class="tl-api-card" href="../graphs/">
        <strong>Choose the layers</strong>
        <span>Tensor, CUDA, NN, Verify</span>
        <p>The four libraries and the imports between their modules.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/NN/Trainer/Train.html">
        <strong>Train a model</strong>
        <span>TorchLean.NN.Trainer.Train</span>
        <p>Prediction, dataset training, checkpoint restoration, and stream training.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/Tensor/Constructors.html">
        <strong>Work with tensors</strong>
        <span>TorchLean.Tensor.Constructors</span>
        <p>Tensor literals, flat-index generation, and bounded-index constructors.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/NN/IR/Graph.html">
        <strong>Inspect graph IR</strong>
        <span>TorchLean.NN.IR.Graph</span>
        <p>Graphs of registered operators: nodes, ports and the constant payload.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/NN/IR/Eval.html">
        <strong>Read operator semantics</strong>
        <span>TorchLean.NN.IR.Eval</span>
        <p>The evaluation of graphs of registered operators, node by node.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/NN/Tape.html">
        <strong>Run autograd</strong>
        <span>TorchLean.NN.Tape</span>
        <p>The autograd tape and its reverse sweeps, shared by CPU and CUDA.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/Verify/Capture/Error.html">
        <strong>Check certificates</strong>
        <span>TorchLean.Verify.Capture.Error</span>
        <p>Concrete numerical certificates for captured programs.</p>
      </a>
      <a class="tl-api-card" href="./TorchLean/Verify/Floats/FP32.html">
        <strong>Use FloatLib formats</strong>
        <span>TorchLean.Verify.Floats.FP32</span>
        <p>The rounded-real binary32 specialization of FloatLib's NF model.</p>
      </a>
    </section>

    <section class="tl-api-section">
      <h2>Browse By Layer</h2>
      <p>
        The modules are grouped in libraries, in import order: TorchLean.Tensor,
        TorchLean.CUDA, TorchLean.NN, TorchLean.Verify and TorchLeanExamples. A module
        imports only modules of its own library or of the libraries before it. Open
        TorchLean and TorchLeanExamples in the module list to browse them.
      </p>
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

    The script appends to the base stylesheet, without rewriting it, because DocGen's CSS also
    controls behavior-heavy pieces such as the module drawer, declaration blocks,
    search results, and responsive sidebar toggling.  The appended rules should
    therefore be mostly cosmetic and conservative: nicer cards, clearer spacing,
    readable sidebars, and a few explanatory widgets.
    """
    style = docs / "style.css"
    current = style.read_text(encoding="utf-8")
    if STYLE_MARKER in current:
        current = current.split(STYLE_MARKER, 1)[0].rstrip() + "\n\n"
    css = (ASSETS / "docgen.css").read_text(encoding="utf-8")
    style.write_text(f"{current}\n\n{STYLE_MARKER}\n\n{css}", encoding="utf-8")


def add_nav_hint(docs: Path) -> None:
    """Add a small hint above DocGen's module tree.

    Cached navigation metadata can put dependency names in the generated tree.
    The published pages are pruned to TorchLean modules, so the hint points
    visitors at the `TorchLean` subtree and states the intended scope.
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
        'Open <strong>TorchLean</strong> for the TorchLean modules.'
        '<span aria-hidden="true">↓</span>'
        '</div>'
        '<div class="module_list">'
    )
    text = text.replace(marker, hint, 1)
    nav.write_text(text, encoding="utf-8")


def rename_docgen_header(docs: Path) -> None:
    """Brand every generated page except the landing page as the TorchLean API reference.

    The header title becomes `TorchLean API`, the cross-site links and the theme toggle go before
    the module name, and the toggle script goes at the end of the body. Pages with declarations
    also get the declaration-kind legend at the top of their internal navigation. Each rewrite
    matches a fixed DocGen fragment and skips a page that lacks it.
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

        # Header links are relative to each generated page.  `docs_root` points
        # back to `/docs/` and `site_root` to the Jekyll site root.
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

        if "data-doc-theme-toggle" not in updated and '<h2 class="header_filename' in updated:
            theme_button = (
                '<button class="tl-doc-theme-toggle" type="button" data-doc-theme-toggle '
                'aria-label="Use dark theme" title="Use dark theme">Dark</button>'
            )
            updated = updated.replace('<h2 class="header_filename', theme_button + '<h2 class="header_filename', 1)

        if "data-doc-theme-toggle" in updated and "torchlean-theme" not in updated:
            updated = updated.replace("</body>", DOC_THEME_SCRIPT + "\n</body>", 1)

        # Only pages with declarations need the color legend.  Module-only pages
        # would waste sidebar space with it.
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


def polish_importgraph(page: Path) -> None:
    """Point the generated graph viewer at our API docs and fit mobile controls."""
    html = page.read_text(encoding="utf-8")
    html = html.replace('  <link rel="stylesheet" href="style.css" />\n', "")
    html = html.replace(
        'var doc_url = "https://leanprover-community.github.io/mathlib4_docs/";',
        'var doc_url = new URL("../docs/", window.location.href).href;',
    )
    # The graph includes dependencies whose pages we publish at their upstream site.
    html = html.replace(
        'window.open(doc_url + path.join(\'/\') + ".html");',
        f'const upstream = {json.dumps(sorted(DOCGEN_DEPENDENCY_MODULES))};\n'
        f'    const base = upstream.includes(path[0]) ? "{UPSTREAM_DOCGEN_BASE}" : doc_url;\n'
        '    window.open(base + path.join(\'/\') + ".html");',
    )
    if 'id="torchlean-importgraph-controls"' not in html:
        controls = """<style id="torchlean-importgraph-controls">
    @media (max-width: 700px) {
      .summary .pause-checkbox, .summary .reverse-checkbox {
        position: static;
        display: block;
        padding: .25em .5em;
      }
    }
  </style>
"""
        html = html.replace("</head>", controls + "</head>", 1)
    page.write_text(html, encoding="utf-8")


def main() -> None:
    """Entry point used by `scripts/docs/build_site.sh` and local preview loops."""
    parser = argparse.ArgumentParser()
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--docs", default="home_page/docs", help="DocGen output directory")
    outputs.add_argument("--importgraph", type=Path, help="generated importgraph HTML page")
    args = parser.parse_args()
    if args.importgraph is not None:
        polish_importgraph(args.importgraph)
        return
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
    configure_math_runtime(docs)
    validate_math_runtime(docs)


if __name__ == "__main__":
    main()
