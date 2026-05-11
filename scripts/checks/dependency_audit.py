#!/usr/bin/env python3
"""
TorchLean dependency-graph audit.

This is a lightweight adaptation of the methodology in:

  Xinze Li, Nanyun Peng, Simone Severini, Patrick Shafto,
  "The Network Structure of Mathlib", arXiv:2604.24797, 2026.

Their work extracts Mathlib's module/declaration/namespace graphs at large scale.  This script is
not a replacement for premise-level extraction; it is a lightweight repository audit that runs
cheaply on TorchLean to keep the architecture map aligned with the import graph.

The audit uses only the Python standard library.  It parses Lean import headers,
builds a module graph, reports broad import / layer-boundary smells, and can export JSON for
notebooks or later graph analysis.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The parser is intentionally shallow: TorchLean's public graph page needs module imports,
# namespaces, and declaration headers, not a full elaborated Lean environment.
IMPORT_RE = re.compile(r"^\s*(public\s+)?import\s+([A-Za-z0-9_'.]+)\s*$")
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$")
DECL_RE = re.compile(
    r"^\s*(?:private\s+|protected\s+|partial\s+|unsafe\s+|noncomputable\s+|scoped\s+|local\s+)*"
    r"(def|theorem|lemma|structure|class|inductive|abbrev|axiom|opaque|instance)\b"
)

# Keep generated/build output and vendored research artifacts out of the architecture graph.
SKIP_PARTS = {".lake", "_out", ".git"}
EXTERNAL_TREE_NAMES = {
    "Two-Stage_Neural_Controller_Training",
    "PINN_verification",
}

# Broad imports are fine at umbrella entrypoints and examples, but they are noisy in
# implementation files because they hide which layer a module actually depends on.
BROAD_IMPORTS = {
    "NN",
    "NN.Library",
    "NN.CI.All",
    "Mathlib",
    "Mathlib.Tactic",
}

# These are intentionally excluded from the longest-path metric. Otherwise umbrella modules
# dominate the number and make the graph look deeper than the implementation really is.
CRITICAL_PATH_EXCLUDE = {
    "NN",
    "NN.Library",
    "NN.CI.All",
    "NN.Examples.Zoo",
    "NN.Tests.Suite",
    "NN.Verification.CLI",
}


@dataclass(frozen=True)
class ImportEdge:
    """One direct Lean import edge found in a source file."""

    src: str
    dst: str
    public: bool
    path: str
    line: int


@dataclass(frozen=True)
class Finding:
    """One dependency-audit warning or error."""

    level: str
    path: str
    line: int
    message: str


def iter_lean_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Yield Lean source files that belong to the TorchLean architecture graph."""
    for path in sorted(root.rglob("*.lean")):
        rel_parts = path.relative_to(root).parts
        # Build artifacts and vendored external projects would make the public
        # architecture page report dependencies that are not part of TorchLean.
        if any(part in SKIP_PARTS for part in rel_parts):
            continue
        if any(part in EXTERNAL_TREE_NAMES for part in rel_parts):
            continue
        yield path


def module_name(root: pathlib.Path, path: pathlib.Path) -> str:
    """Convert a Lean source path into a dotted Lean module name."""
    return ".".join(path.relative_to(root).with_suffix("").parts)


def mask_comments_and_strings(text: str) -> str:
    """Replace Lean comments/docstrings/strings with spaces while preserving line numbers.

    This is a lightweight scanner rather than a full Lean parser. It still tracks
    nested block comments and ordinary string escapes so dependency regexes do
    not fire on prose examples or URLs in documentation comments.
    """

    out = list(text)
    i = 0
    n = len(text)
    block_depth = 0
    in_line_comment = False
    in_string = False

    while i < n:
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            else:
                # Preserve newlines so line numbers in findings still match the source file.
                out[i] = " "
            i += 1
            continue

        if block_depth > 0:
            if text.startswith("/-", i):
                out[i] = out[i + 1] = " "
                block_depth += 1
                i += 2
                continue
            if text.startswith("-/", i):
                out[i] = out[i + 1] = " "
                block_depth -= 1
                i += 2
                continue
            if ch != "\n":
                out[i] = " "
            i += 1
            continue

        if in_string:
            if ch == "\n":
                # Lean strings should not span raw newlines in this lightweight scanner.
                # Resetting here keeps a malformed string from masking the rest of the file.
                in_string = False
                i += 1
                continue
            out[i] = " "
            if ch == "\\" and i + 1 < n:
                out[i + 1] = " "
                i += 2
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        if text.startswith("--", i):
            out[i] = out[i + 1] = " "
            in_line_comment = True
            i += 2
            continue
        if text.startswith("/-", i):
            out[i] = out[i + 1] = " "
            block_depth = 1
            i += 2
            continue
        if ch == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue

        i += 1

    return "".join(out)


def layer_of(module: str) -> str:
    """Collapse a module name to the coarse layer used in graph summaries."""
    parts = module.split(".")
    if len(parts) < 2 or parts[0] != "NN":
        return parts[0]
    if parts[1] in {
        "API",
        "CI",
        "Entrypoint",
        "Examples",
        "Floats",
        "GraphSpec",
        "IR",
        "MLTheory",
        "Proofs",
        "Runtime",
        "Spec",
        "Tensor",
        "Tests",
        "Verification",
    }:
        return ".".join(parts[:2])
    return parts[1]


def parse_file(root: pathlib.Path, path: pathlib.Path) -> tuple[list[ImportEdge], list[str], list[Finding]]:
    """Read one Lean file and collect its import edges, namespaces, and warnings."""
    src = module_name(root, path)
    rel = path.relative_to(root).as_posix()
    imports: list[ImportEdge] = []
    namespaces: list[str] = []
    findings: list[Finding] = []
    suppress_broad_import_warning = (
        # Tutorial and integration surfaces deliberately import broad facades.
        rel.startswith("NN/Examples/")
        or rel.startswith("NN/Tests/")
        or rel.startswith("blueprint/")
        or rel in {"NN.lean", "NN/CI/ComparatorAll.lean"}
    )

    visible_text = mask_comments_and_strings(path.read_text(encoding="utf-8"))
    for line_no, line in enumerate(visible_text.splitlines(), start=1):
        if m := IMPORT_RE.match(line):
            public = bool(m.group(1))
            dst = m.group(2)
            imports.append(ImportEdge(src=src, dst=dst, public=public, path=rel, line=line_no))
            if dst in BROAD_IMPORTS and not suppress_broad_import_warning:
                findings.append(
                    Finding(
                        level="WARN",
                        path=rel,
                        line=line_no,
                        message=f"broad import `{dst}`; prefer a narrower entrypoint when possible",
                    )
                )
        elif m := NAMESPACE_RE.match(line):
            # Namespaces are not currently rendered on the public page, but keeping
            # them in JSON makes the artifact useful for future navigation work.
            namespaces.append(m.group(1))

    return imports, namespaces, findings


def longest_path_len(nodes: set[str], edges: list[ImportEdge]) -> int | None:
    """Compute the longest direct-import chain, or `None` if the graph has a cycle.

    The “critical path” reported by this script is a rough layering metric over
    direct imports. It is not a theorem-dependency graph and should not be read
    as a proof of semantic dependency between declarations.
    """
    outgoing: dict[str, list[str]] = defaultdict(list)
    indeg: Counter[str] = Counter()
    for n in nodes:
        indeg[n] += 0
    for e in edges:
        if e.dst in nodes:
            outgoing[e.dst].append(e.src)
            indeg[e.src] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    dist = {n: 0 for n in nodes}
    seen = 0
    while q:
        n = q.popleft()
        seen += 1
        for m in outgoing[n]:
            dist[m] = max(dist[m], dist[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if seen != len(nodes):
        return None
    return max(dist.values(), default=0)


def code_stats(root: pathlib.Path, files: list[pathlib.Path]) -> dict:
    """Compute lightweight repository-size statistics for the public graph page.

    The line counts use the same comment/string masking pass as the dependency parser.
    They are meant as stable repository-scale indicators, not as a semantic Lean
    declaration graph.
    """

    total_lines = 0
    blank_lines = 0
    code_lines = 0
    declaration_counts: Counter[str] = Counter()
    layer_files: Counter[str] = Counter()
    layer_lines: Counter[str] = Counter()
    top_level_files: Counter[str] = Counter()

    for path in files:
        rel = path.relative_to(root)
        raw = path.read_text(encoding="utf-8")
        raw_lines = raw.splitlines()
        # Count declaration-like headers only after comments and strings are hidden,
        # so examples in docstrings do not inflate the public statistics.
        visible_lines = mask_comments_and_strings(raw).splitlines()
        layer = layer_of(module_name(root, path))

        total_lines += len(raw_lines)
        layer_files[layer] += 1
        layer_lines[layer] += len(raw_lines)
        top_level_files[rel.parts[0]] += 1

        for line in visible_lines:
            if not line.strip():
                blank_lines += 1
                continue
            code_lines += 1
            # The declaration regex is deliberately conservative. It is for scale
            # indicators on the website, not for proof coverage accounting.
            if m := DECL_RE.match(line):
                declaration_counts[m.group(1)] += 1

    theorem_like = declaration_counts["theorem"] + declaration_counts["lemma"]
    return {
        "lean_files": len(files),
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_or_comment_lines": total_lines - code_lines,
        "declarations": sum(declaration_counts.values()),
        "theorem_like_declarations": theorem_like,
        "declaration_counts": dict(sorted(declaration_counts.items())),
        "top_level_files": [
            {"directory": key, "files": count}
            for key, count in sorted(top_level_files.items(), key=lambda item: (-item[1], item[0]))
        ],
        "layer_sizes": [
            {"layer": layer, "files": layer_files[layer], "lines": layer_lines[layer]}
            for layer in sorted(layer_files, key=lambda key: (-layer_lines[key], key))
        ],
    }


def audit(root: pathlib.Path) -> dict:
    """Build the JSON-serializable dependency audit report."""
    root = root.resolve()
    modules: set[str] = set()
    edges: list[ImportEdge] = []
    namespaces_by_module: dict[str, list[str]] = {}
    findings: list[Finding] = []
    lean_files = list(iter_lean_files(root))

    for path in lean_files:
        mod = module_name(root, path)
        modules.add(mod)
        # Keep parsing file-local so a malformed or suspicious import can point
        # back to the exact source path and line.
        imports, namespaces, file_findings = parse_file(root, path)
        edges.extend(imports)
        namespaces_by_module[mod] = namespaces
        findings.extend(file_findings)

    internal_edges = [e for e in edges if e.dst in modules]
    public_edges = [e for e in edges if e.public]
    layer_edges = Counter((layer_of(e.src), layer_of(e.dst)) for e in internal_edges)
    fan_in = Counter(e.dst for e in internal_edges)
    fan_out = Counter(e.src for e in internal_edges)

    for e in internal_edges:
        src_layer = layer_of(e.src)
        dst_layer = layer_of(e.dst)
        # Hard architectural boundaries: specs should not depend on runtime code,
        # and reusable runtime code should never depend on examples.
        if src_layer == "NN.Spec" and dst_layer.startswith("NN.Runtime"):
            findings.append(
                Finding(
                    level="ERROR",
                    path=e.path,
                    line=e.line,
                    message="Spec layer imports Runtime; specs should stay backend-independent",
                )
            )
        if src_layer == "NN.Runtime" and dst_layer == "NN.Examples":
            findings.append(
                Finding(
                    level="ERROR",
                    path=e.path,
                    line=e.line,
                    message="Runtime imports Examples; examples must depend on runtime, not the reverse",
                )
            )

    top_fan_in = fan_in.most_common(20)
    top_fan_out = fan_out.most_common(20)
    critical_nodes = modules - CRITICAL_PATH_EXCLUDE
    critical_edges = [
        e for e in internal_edges if e.src in critical_nodes and e.dst in critical_nodes
    ]
    critical_path = longest_path_len(critical_nodes, critical_edges)

    # The public graph page reads this JSON directly. Keep fields explicit and
    # stable so the page can evolve without reparsing Lean in the browser.
    return {
        "paper_citation": {
            "title": "The Network Structure of Mathlib",
            "arxiv": "2604.24797",
            "url": "https://arxiv.org/abs/2604.24797",
            "code": "https://github.com/MathNetwork/mathlib-network",
        },
        "summary": {
            "modules": len(modules),
            "import_edges": len(edges),
            "internal_import_edges": len(internal_edges),
            "public_import_edges": len(public_edges),
            "private_import_edges": len(edges) - len(public_edges),
            "layers": sorted({layer_of(m) for m in modules}),
            "critical_path_import_edges": critical_path,
            "critical_path_excluded_umbrellas": sorted(CRITICAL_PATH_EXCLUDE & modules),
            "findings": len(findings),
            "errors": sum(1 for f in findings if f.level == "ERROR"),
            "warnings": sum(1 for f in findings if f.level == "WARN"),
        },
        "code_stats": code_stats(root, lean_files),
        "layer_edges": [
            {"src_layer": src, "dst_layer": dst, "count": count}
            for (src, dst), count in sorted(layer_edges.items(), key=lambda item: (-item[1], item[0]))
        ],
        "top_fan_in": [{"module": mod, "count": count} for mod, count in top_fan_in],
        "top_fan_out": [{"module": mod, "count": count} for mod, count in top_fan_out],
        "findings": [asdict(f) for f in findings],
        "edges": [asdict(e) for e in edges],
        "namespaces": namespaces_by_module,
    }


def render_markdown(report: dict, *, max_findings: int) -> str:
    """Render a compact Markdown summary from a full audit report."""
    s = report["summary"]
    lines: list[str] = []
    lines.append("# TorchLean Dependency Audit")
    lines.append("")
    lines.append(
        "Inspired by Li, Peng, Severini, and Shafto, "
        '"The Network Structure of Mathlib" (arXiv:2604.24797).'
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Modules: `{s['modules']}`")
    lines.append(f"- Import edges: `{s['import_edges']}`")
    lines.append(f"- Internal import edges: `{s['internal_import_edges']}`")
    lines.append(f"- Public imports: `{s['public_import_edges']}`")
    lines.append(f"- Private imports: `{s['private_import_edges']}`")
    lines.append(f"- Critical-path length over internal imports: `{s['critical_path_import_edges']}`")
    lines.append(f"- Findings: `{s['findings']}` (`{s['errors']}` errors, `{s['warnings']}` warnings)")
    if stats := report.get("code_stats"):
        lines.append(f"- Lean files: `{stats['lean_files']}`")
        lines.append(f"- Lean source lines: `{stats['total_lines']}`")
        lines.append(f"- Declaration headers: `{stats['declarations']}`")
        lines.append(f"- Theorem/lemma headers: `{stats['theorem_like_declarations']}`")
    lines.append("")
    lines.append("## Top Fan-In Modules")
    lines.append("")
    for item in report["top_fan_in"][:10]:
        lines.append(f"- `{item['module']}`: `{item['count']}` incoming imports")
    lines.append("")
    lines.append("## Top Fan-Out Modules")
    lines.append("")
    for item in report["top_fan_out"][:10]:
        lines.append(f"- `{item['module']}`: `{item['count']}` imports")
    lines.append("")
    lines.append("## Layer Edges")
    lines.append("")
    for item in report["layer_edges"][:20]:
        lines.append(
            f"- `{item['src_layer']}` -> `{item['dst_layer']}`: `{item['count']}`"
        )
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    findings = report["findings"]
    if not findings:
        lines.append("No findings.")
    else:
        for f in findings[:max_findings]:
            lines.append(f"- `{f['level']}` {f['path']}:{f['line']}: {f['message']}")
        if len(findings) > max_findings:
            lines.append(f"- ... `{len(findings) - max_findings}` more findings omitted")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    """Parse CLI flags, emit requested reports, and enforce optional failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--json", type=pathlib.Path, help="write full JSON report")
    parser.add_argument("--markdown", type=pathlib.Path, help="write Markdown summary")
    parser.add_argument("--max-findings", type=int, default=80, help="findings shown in Markdown/stdout")
    parser.add_argument("--fail-on-error", action="store_true", help="exit nonzero if ERROR findings exist")
    args = parser.parse_args(argv)

    report = audit(args.root)
    md = render_markdown(report, max_findings=args.max_findings)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(md, encoding="utf-8")
    if not args.json and not args.markdown:
        print(md)

    if args.fail_on_error and report["summary"]["errors"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
