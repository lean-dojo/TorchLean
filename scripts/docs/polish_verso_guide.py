#!/usr/bin/env python3
"""Small post-build polish pass for the generated Verso guide.

Verso owns the main HTML generation. TorchLean adds a thin local layer for the
public guide: responsive figures, readable code blocks, external-link behavior,
copy buttons, responsive tables, stable KaTeX layout, route arrows, and a
low-overhead reading-progress indicator. Its stylesheet and script are `assets/guide.css` and
`assets/guide.js`.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit


ASSETS = Path(__file__).resolve().parent / "assets"
# The block appended to Verso's `book.css`, and the body of `torchlean-guide-polish.js`.
TORCHLEAN_CSS = "\n" + (ASSETS / "guide.css").read_text(encoding="utf-8")
TORCHLEAN_JS_BODY = "\n" + (ASSETS / "guide.js").read_text(encoding="utf-8")


def guide_pages(root: Path) -> list[str]:
    """Return generated guide pages in the reading order used by the progress widget.

    The list skips the search pages and hyphen-prefixed implementation pages that
    Verso also emits, because a reader does not advance through them as chapters.
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
    script_re = re.compile(r'<script defer src="[^"]*torchlean-guide-polish\.js(?:\?v=[^"]*)?"></script>')
    # Verso emits a <base> tag on every generated page. A bare script URL is
    # therefore resolved relative to the guide root, even from nested pages.
    version = hashlib.sha256((root / marker).read_bytes()).hexdigest()[:16]
    tag = f'    <script defer src="{marker}?v={version}"></script>\n'
    for path in root.rglob("*.html"):
        html = path.read_text()
        if marker in html:
            new_html = script_re.sub(tag.strip(), html, count=1)
            if new_html != html:
                path.write_text(new_html)
            continue
        if "</head>" not in html:
            continue
        # Extension-specific styles may leave the closing tag on the same line
        # as `</style>`, so injection must not depend on surrounding whitespace.
        path.write_text(html.replace("</head>", tag + "  </head>", 1))


def inject_favicon(root: Path) -> None:
    """Use the main TorchLean mark for every generated guide page."""
    marker = "data-torchlean-favicon"
    icon_re = re.compile(
        r'\s*<link\s+[^>]*data-torchlean-favicon[^>]*>\n?',
        flags=re.IGNORECASE,
    )
    # Verso's <base> points at the guide root, so this reaches the Jekyll
    # asset both locally and under GitHub Pages' /TorchLean project prefix.
    tag = (
        '    <link rel="icon" href="../assets/media/brand/torchlean-logo.png" '
        'type="image/png" data-torchlean-favicon>\n'
    )
    for path in root.rglob("*.html"):
        html = path.read_text()
        if marker in html:
            new_html = icon_re.sub("\n" + tag, html, count=1)
            if new_html != html:
                path.write_text(new_html)
            continue
        if "</head>" not in html:
            continue
        path.write_text(html.replace("</head>", tag + "  </head>", 1))


def repair_generated_table_css(root: Path) -> None:
    """Correct two inert table-style typos in the generated Verso page headers.

    The pinned Verso table stylesheet writes `margin-left auto;` and selects `th > p:first-child`
    where it means `th > p:last-child`. Repairing the generated pages leaves the pinned
    dependency unmodified.
    """
    replacements = {
        "margin-left auto;": "margin-left: auto;",
        "table.tabular td > p:last-child, table.tabular th > p:first-child":
            "table.tabular td > p:last-child, table.tabular th > p:last-child",
    }
    for path in root.rglob("*.html"):
        original = path.read_text()
        repaired = original
        for old, new in replacements.items():
            repaired = repaired.replace(old, new)
        if repaired != original:
            path.write_text(repaired)


def select_formalization_group_view(root: Path) -> None:
    """Render the curated group overview first while keeping every graph view available."""
    path = root / "Dependency-Graph" / "index.html"
    if not path.exists():
        return

    html_text = path.read_text()
    group_option = '<option value="group">Group View</option>'
    selected_group_option = '<option value="group" selected="">Group View</option>'
    if selected_group_option in html_text:
        return
    if group_option not in html_text:
        raise SystemExit(f"missing Group View selector in {path}")

    html_text = re.sub(
        r'(<option value="full")\s+selected(?:="")?(>)',
        r"\1\2",
        html_text,
        count=1,
    )
    path.write_text(html_text.replace(group_option, selected_group_option, 1))


def rewrite_repository_links(root: Path) -> None:
    """Turn repository-relative links into public API or source links.

    The guide source is written inside `home_page/blueprint/TorchLeanBlueprint`, so links
    like `../../TorchLean/...` are convenient while editing. In the generated website
    those paths point outside the published guide, so the post-build pass rewrites them. Lean
    modules with generated DocGen pages go to the API reference, and other repository files go to
    GitHub source.
    """

    repo_root = Path(__file__).resolve().parents[2]
    docs_root = repo_root / "home_page" / "docs"
    github_root = "https://github.com/lean-dojo/TorchLean"
    # The Lean source roots: the libraries under `TorchLean/`, the examples and the tests.
    source_roots = ("TorchLean/", "TorchLeanExamples/", "TorchLeanTest/")
    repo_prefixes = (
        *source_roots,
        "csrc/",
        "scripts/",
        "docs/",
        "home_page/",
        ".github/",
        "README",
        "lakefile",
        "lake-manifest",
        "lean-toolchain",
    )

    def api_href_for(normalized: str) -> str | None:
        """Return a DocGen URL relative to Verso's guide-root base URL."""
        if not normalized.endswith(".lean"):
            return None
        doc_rel = normalized[:-5] + ".html"
        if not (docs_root / doc_rel).exists():
            return None
        # Every generated page's <base> resolves to /blueprint/, regardless of page depth.
        # Computing it from the page directory would escape a project site's /TorchLean prefix.
        return "../docs/" + doc_rel

    def rewrite_href(match: re.Match[str]) -> str:
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

        if normalized.startswith(tuple(f"docs/{top}" for top in source_roots)):
            return f'href={quote}../{normalized}{sep}{frag}{quote}'

        if not normalized.startswith(repo_prefixes):
            return match.group(0)

        api_href = api_href_for(normalized)
        # Source line anchors belong to GitHub, and declaration anchors to DocGen.
        if api_href is not None and not re.fullmatch(r"L\d+(?:-L\d+)?", frag):
            return f'href={quote}{api_href}{sep}{frag}{quote}'

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
        r'<a([^>]*href=[\"\'][^\"\']*docs/(TorchLean(?:Examples|Test)?/[^\"\']+)\.html'
        r'[^\"\']*[\"\'][^>]*)>([^<]*?\.lean|TorchLean(?:Examples|Test)?/[^<]*?)</a>'
    )
    anchor_re = re.compile(r'<a\b([^>]*\bhref=([\"\'])([^\"\']+)\2[^>]*)>')
    inline_module_re = re.compile(
        r"<code>(TorchLean(?:Examples|Test)?/[A-Za-z0-9_./-]+\.lean)</code>"
    )
    inline_tree_re = re.compile(
        r"<code>(TorchLean(?:Examples|Test)?/[A-Za-z0-9_./-]+)/\*</code>"
    )
    linked_or_block_re = re.compile(
        r"(<a\b[^>]*>.*?</a>|<pre\b[^>]*>.*?</pre>)", re.DOTALL
    )

    def clean_api_label(match: re.Match[str]) -> str:
        """Replace noisy source-file link labels with stable module API labels."""
        attrs = match.group(1)
        module = match.group(2)
        label = match.group(3).strip()
        if ".lean" not in label and not label.startswith(source_roots):
            return match.group(0)
        module_label = module.replace("/", ".") + " API"
        return f"<a{attrs}>{module_label}</a>"

    def link_inline_module(match: re.Match[str]) -> str:
        """Turn inline `TorchLean/...lean` code spans into API links when DocGen has them."""
        normalized = match.group(1)
        api_href = api_href_for(normalized)
        if api_href is None:
            return match.group(0)
        module_label = normalized[:-5].replace("/", ".") + " API"
        return f'<a href="{api_href}" target="_blank" rel="noopener noreferrer">{module_label}</a>'

    def link_inline_tree(match: re.Match[str]) -> str:
        """Turn inline `TorchLean/.../*` code spans into GitHub source-tree links."""
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
        rewritten = href_re.sub(rewrite_href, html)
        rewritten = api_link_re.sub(clean_api_label, rewritten)
        # A source link can already contain a path in <code>. Keep that destination
        # without inserting a second anchor inside it, and leave code blocks literal.
        pieces = linked_or_block_re.split(rewritten)
        for index in range(0, len(pieces), 2):
            pieces[index] = inline_module_re.sub(link_inline_module, pieces[index])
            pieces[index] = inline_tree_re.sub(link_inline_tree, pieces[index])
        rewritten = "".join(pieces)
        rewritten = anchor_re.sub(add_blank_target, rewritten)
        if rewritten != html:
            path.write_text(rewritten)


class _GuideHtmlIds(HTMLParser):
    """Collect the fragment targets (`id` and `name` attributes) of a generated guide page."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {k: v for k, v in attrs if v is not None}
        if data.get("id"):
            self.ids.add(data["id"])
        if data.get("name"):
            self.ids.add(data["name"])


def normalize_source_locations(root: Path) -> None:
    """Replace the builder's checkout paths with package-relative Lean locations."""
    source_path = re.compile(
        r'("(?:path|sourcePath)"\s*:\s*")[^"\n]*[/\\]'
        r'((?:TorchLeanBlueprint|FloatLib|TorchLeanExamples|TorchLeanTest|'
        r'TorchLean[/\\](?:Tensor|CUDA|NN|Verify))[/\\][^"\n]*\.lean")'
    )
    paths = [*root.rglob("*.html"), root / "xref.json",
             root / "-verso-data/blueprint-manifest.json"]
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text()
        portable = source_path.sub(r"\1\2", text)
        if portable != text:
            path.write_text(portable)


def add_fragment_aliases(root: Path) -> None:
    """Add only the page-title anchors identified by Verso's cross-reference data.

    Document, chapter, and page titles may lack DOM ids on their standalone pages.
    Their aliases belong at the start of the page. Deeper headings must have real
    anchors, because inventing targets from incoming links would hide broken fragments
    from the site link checker.
    """

    root = root.resolve()
    xref_path = root / "xref.json"
    if not xref_path.exists():
        return
    xref = json.loads(xref_path.read_text())
    sections = xref.get("Verso.Genre.Manual.section", {}).get("contents", {})
    parsed: dict[Path, _GuideHtmlIds] = {}

    def parse(path: Path) -> _GuideHtmlIds:
        if path not in parsed:
            p = _GuideHtmlIds()
            p.feed(path.read_text(errors="ignore"))
            parsed[path] = p
        return parsed[path]

    aliases: dict[Path, set[str]] = {}
    for entries in sections.values():
        for entry in entries:
            # Context contains book, chapter, page, then headings within the page.
            if len(entry["data"]["context"]) > 3:
                continue
            address = urlsplit(entry["address"])
            if address.scheme or address.netloc:
                continue
            target = (root / unquote(address.path).lstrip("/")).resolve()
            if not target.is_relative_to(root):
                continue
            if target.is_dir():
                target = (target / "index.html").resolve()
            if not target.is_relative_to(root) or target.suffix != ".html" or not target.is_file():
                continue
            frag = entry["id"]
            if frag and frag not in parse(target).ids:
                aliases.setdefault(target, set()).add(frag)

    for path, ids in aliases.items():
        html_text = path.read_text()
        alias_html = "".join(
            f'<span id="{html.escape(frag, quote=True)}" class="tl-anchor-alias" aria-hidden="true"></span>'
            for frag in sorted(ids)
        )
        if "<main" in html_text:
            html_text = re.sub(
                r"(<main\b[^>]*>)", lambda m: m[1] + alias_html, html_text, count=1
            )
        elif "<body" in html_text:
            html_text = re.sub(
                r"(<body\b[^>]*>)", lambda m: m[1] + alias_html, html_text, count=1
            )
        else:
            html_text = alias_html + html_text
        path.write_text(html_text)


def validate_math_runtime(root: Path) -> None:
    """Require the local KaTeX runtime on every generated guide page."""
    runtime = root / "-verso-data" / "katex"
    assets = ("katex.js", "math.js", "katex.css")
    missing_assets = [
        runtime / name
        for name in assets
        if not (runtime / name).is_file() or (runtime / name).stat().st_size == 0
    ]
    if missing_assets:
        missing = ", ".join(str(path) for path in missing_assets)
        raise SystemExit(f"missing generated KaTeX assets: {missing}")

    references = tuple(f"-verso-data/katex/{name}" for name in assets)
    generated_pages = 0
    missing_pages: list[Path] = []
    for path in root.rglob("*.html"):
        page = path.read_text()
        # Asset directories may contain standalone HTML demos. They are not
        # Verso pages and do not inherit the guide's runtime.
        if 'href="book.css"' not in page:
            continue
        generated_pages += 1
        if any(reference not in page for reference in references):
            missing_pages.append(path)
    if generated_pages == 0:
        raise SystemExit(f"no generated Verso pages found under {root}")
    if missing_pages:
        sample = ", ".join(str(path) for path in missing_pages[:5])
        suffix = "" if len(missing_pages) <= 5 else f" (and {len(missing_pages) - 5} more)"
        raise SystemExit(f"KaTeX is not loaded by every guide page: {sample}{suffix}")


DEEP_NUMBER = re.compile(r"^\s*\d+(?:\.\d+){2,}\.?(?:\s|$)")


class GuideLayoutPage(HTMLParser):
    """Collect anchors, headings, and the table of contents' number cells."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.labels: list[str] = []
        self.pending: list[tuple[str, list[str]]] = []
        self.feed(text)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.add(identifier)
        # Restrict the scan to section labels so version links such as "1.2.3"
        # in prose or code do not look like unwanted section numbering.
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} or (
            tag == "td" and "num" in (attributes.get("class") or "").split()
        ):
            self.pending.append((tag, []))

    def handle_data(self, data: str) -> None:
        for _, pieces in self.pending:
            pieces.append(data)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.pending) - 1, -1, -1):
            name, pieces = self.pending[index]
            if name == tag:
                self.labels.append(" ".join("".join(pieces).split()))
                self.pending.pop(index)
                break


def check_layout(root: Path) -> tuple[int, int]:
    """Keep chapter/page numbers, with all deeper sections on their parent page."""
    root = root.resolve()
    pages: dict[Path, GuideLayoutPage] = {}
    errors: list[str] = []
    for path in root.rglob("*.html"):
        text = path.read_text()
        if 'href="book.css"' not in text:
            continue
        page = GuideLayoutPage(text)
        pages[path] = page
        for label in page.labels:
            if DEEP_NUMBER.match(label):
                errors.append(f"{path.relative_to(root)}: deep section number in {label!r}")
    if not pages:
        raise ValueError(f"no rendered guide pages under {root}")

    xref = json.loads((root / "xref.json").read_text())
    sections = [
        entry
        for entries in xref["Verso.Genre.Manual.section"]["contents"].values()
        for entry in entries
    ]
    if not sections:
        raise ValueError("the guide has no section references")

    # Context contains the book, chapter, page, then any headings within that page.
    # Compare addresses, since a slug follows its title and changing a title must not split its
    # children.
    page_addresses = {
        tuple(part["title"] for part in entry["data"]["context"]): entry["address"]
        for entry in sections
        if len(entry["data"]["context"]) == 3
    }
    for entry in sections:
        data = entry["data"]
        title = data["title"]
        context = data["context"]
        number = data.get("sectionNum") or ""
        if DEEP_NUMBER.match(number):
            errors.append(f"{title!r}: deep section number in cross-reference {number!r}")
        if len(context) > 3:
            parent = tuple(part["title"] for part in context[:3])
            address = page_addresses.get(parent)
            if address is None or entry["address"] != address:
                errors.append(f"{title!r}: section is split away from its parent page")

        address = unquote(urlsplit(entry["address"]).path).lstrip("/")
        target = (root / address / "index.html").resolve()
        if not target.is_relative_to(root):
            errors.append(f"{title!r}: section address leaves the guide")
            continue
        page = pages.get(target)
        if page is None or entry["id"] not in page.ids:
            errors.append(f"{title!r}: section anchor is missing from {address or '/'}")

    if errors:
        sample = "\n".join(dict.fromkeys(errors[:20]))
        raise ValueError(f"guide reading-layout check failed ({len(errors)} findings):\n{sample}")
    return len(pages), len(sections)


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
    # The appended block starts with this comment, so a repeated run replaces the block.
    start = css.find("/* Shell input and recorded transcripts")
    if start != -1:
        css = css[:start]
    css_path.write_text(css.rstrip() + TORCHLEAN_CSS)
    write_js(args.guide)
    repair_generated_table_css(args.guide)
    select_formalization_group_view(args.guide)
    rewrite_repository_links(args.guide)
    normalize_source_locations(args.guide)
    add_fragment_aliases(args.guide)
    inject_favicon(args.guide)
    inject_script(args.guide)
    validate_math_runtime(args.guide)
    try:
        pages, sections = check_layout(args.guide)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Guide layout checked: {pages} pages, {sections} sections; numbering stops at N.M.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
