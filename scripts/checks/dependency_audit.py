#!/usr/bin/env python3
"""
TorchLean dependency-graph audit.

The audit adapts to one repository the methodology of:

  Xinze Li, Nanyun Peng, Simone Severini, Patrick Shafto,
  "The Network Structure of Mathlib", arXiv:2604.24797, 2026.

Their work extracts Mathlib's module/declaration/namespace graphs at large scale.  The script here
does no premise-level extraction. It is a cheap repository audit that keeps TorchLean's
architecture map aligned with the import graph.

The audit uses only the Python standard library.  It parses Lean import headers,
builds a module graph, enforces the layering of the six libraries (`layering_gate`:
TorchLean.Tensor < TorchLean.CUDA < TorchLean.NN < TorchLean.Verify < TorchLeanExamples <
TorchLeanTest), reports broad imports, and can export JSON for notebooks or later graph
analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# The parser is intentionally shallow: TorchLean's public graph page needs module imports,
# namespaces, and declaration headers, not a full elaborated Lean environment.
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([A-Za-z0-9_'.]+)\s*$")
DECL_RE = re.compile(
    r"^\s*(?:public\s+|private\s+|protected\s+|partial\s+|unsafe\s+|noncomputable\s+|scoped\s+|local\s+)*"
    r"(def|theorem|lemma|structure|class|inductive|abbrev|axiom|opaque|instance)\b"
)

# Keep generated and build output out of the architecture graph.
SKIP_PARTS = {".lake", "_out", ".git"}

# Broad imports are fine in tests, but they are noisy in implementation files because they
# hide what a module actually depends on.
BROAD_IMPORTS = {
    "Mathlib",
    "Mathlib.Tactic",
}

# These are intentionally excluded from the longest-path metric. Otherwise the aggregators
# (the test suite, the verification CLI) dominate the number and make the graph look deeper
# than the implementation really is.
CRITICAL_PATH_EXCLUDE = {
    "TorchLeanExamples.Verification.CLI",
    "TorchLeanTest.Suite",
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
    """Enumerate project sources without traversing ignored build and package caches."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z", "--", "*.lean"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        result = None
    if result is not None and result.returncode == 0:
        paths = (root / name.decode("utf-8", errors="surrogateescape")
                 for name in result.stdout.split(b"\0") if name)
    else:
        # Prune before descent. Filtering an rglob result would walk the caches first.
        def archive_sources() -> Iterable[pathlib.Path]:
            for directory, children, names in os.walk(root):
                children[:] = [
                    name for name in children
                    if name not in SKIP_PARTS
                ]
                for name in names:
                    if name.endswith(".lean"):
                        yield pathlib.Path(directory) / name
        paths = archive_sources()
    for path in sorted(set(paths)):
        if any(part in SKIP_PARTS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def module_name(root: pathlib.Path, path: pathlib.Path) -> str:
    """Convert a Lean source path into a dotted Lean module name."""
    return ".".join(path.relative_to(root).with_suffix("").parts)


# Lexical classes for `mask_comments_and_strings`: a raw string starts with `r`, any number of
# `#` and a quote, and ends with a quote and the same number of `#`. A character literal is one
# character or escape between single quotes. Neither starts inside an identifier (`fr"`, `h'`).
RAW_STRING_OPEN_RE = re.compile(r'r(#*)"')
CHAR_LITERAL_RE = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F]+\}|.)|[^\\'\n])'")


def _continues_identifier(ch: str) -> bool:
    """Whether `ch` can end an identifier, so a following `r` or `'` belongs to it."""
    return ch.isalnum() or ch in "_'!?.»"


def mask_comments_and_strings(text: str) -> str:
    """Mask Lean comments and strings without changing offsets or line numbers.

    Used by the import audit and the repository linter. Nested block comments, escaped string
    characters, raw strings (`r"…"`, `r#"…"#`) and character literals are handled lexically,
    without a full Lean parser.
    """

    out = list(text)
    n = len(text)
    i = 0

    in_line_comment = False
    block_depth = 0
    in_string = False

    while i < n:
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if block_depth > 0:
            if text.startswith("/-", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth += 1
                i += 2
                continue
            if text.startswith("-/", i):
                out[i] = " "
                if i + 1 < n:
                    out[i + 1] = " "
                block_depth -= 1
                i += 2
                continue
            if ch == "\n":
                i += 1
            else:
                out[i] = " "
                i += 1
            continue

        if in_string:
            # Mask string contents while preserving newlines. Lean strings should
            # not contain raw newlines, but the scanner stays defensive so a
            # missing quote does not mask the rest of the file.
            if ch == "\n":
                in_string = False
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                # Escape sequence: mask both chars.
                out[i] = " "
                if text[i + 1] != "\n":
                    out[i + 1] = " "
                i += 2
                continue
            out[i] = " "
            if ch == '"':
                in_string = False
            i += 1
            continue

        # Outside comments/strings: detect comment/string starts.
        if text.startswith("--", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            in_line_comment = True
            i += 2
            continue

        if text.startswith("/-", i):
            out[i] = " "
            if i + 1 < n:
                out[i + 1] = " "
            block_depth = 1
            i += 2
            continue

        if ch in "r'" and (i == 0 or not _continues_identifier(text[i - 1])):
            # A raw string may span lines (the embedded Python programs). A character literal
            # may be a quote, which must not open a string.
            if ch == "r":
                match = RAW_STRING_OPEN_RE.match(text, i)
                end = None
                if match is not None:
                    close = text.find('"' + match.group(1), match.end())
                    end = n if close < 0 else close + 1 + len(match.group(1))
            else:
                match = CHAR_LITERAL_RE.match(text, i)
                end = match.end() if match is not None else None
            if end is not None:
                for k in range(i, end):
                    if text[k] != "\n":
                        out[k] = " "
                i = end
                continue

        if ch == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue

        i += 1

    return "".join(out)


def mask_fenced_code_blocks(text: str) -> str:
    """Hide Markdown/Verso code fences while preserving offsets and line numbers.

    Verso guide bodies are Lean syntax, so an example line such as ``import NN`` is visible to a
    shallow source scanner even though Lean never executes it as an import command. Fences may use
    backticks or tildes and may carry a language name on the opening line.
    """

    out: list[str] = []
    fence_char: str | None = None
    fence_width = 0

    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        stripped = body.lstrip()

        if fence_char is None:
            match = re.match(r"(`{3,}|~{3,})", stripped)
            if match is None:
                out.append(line)
                continue
            marker = match.group(1)
            fence_char = marker[0]
            fence_width = len(marker)
        else:
            closing = re.fullmatch(
                rf"\s*{re.escape(fence_char)}{{{fence_width},}}\s*",
                body,
            )
            if closing is not None:
                fence_char = None
                fence_width = 0

        # Keep every newline in place so reported source locations stay exact.
        out.append("".join(ch if ch in "\r\n" else " " for ch in line))

    return "".join(out)


def layer_of(module: str) -> str:
    """Collapse a module name to the coarse layer used in graph summaries: the library and its
    top-level directory, `TorchLean.<L>[.<sub>]`, `TorchLeanExamples[.<sub>]` or
    `TorchLeanTest[.<sub>]`."""
    parts = module.split(".")
    width = 2 if parts[0] == "TorchLean" else 1
    if parts[0] not in ("TorchLean", "TorchLeanExamples", "TorchLeanTest") or len(parts) < width:
        return parts[0]
    return ".".join(parts[:width + 1] if len(parts) > width + 1 else parts[:width])


def parse_file(root: pathlib.Path, path: pathlib.Path) -> tuple[list[ImportEdge], list[str], list[Finding]]:
    """Read one Lean file and collect its import edges, namespaces, and warnings."""
    src = module_name(root, path)
    rel = path.relative_to(root).as_posix()
    namespaces: list[str] = []
    findings: list[Finding] = []
    suppress_broad_import_warning = (
        # Tests deliberately exercise the complete public surface. Examples import the modules they
        # use, so their dependencies stay visible to readers.
        rel.startswith("TorchLeanTest/")
        or rel.startswith("home_page/blueprint/")
    )

    raw_text = path.read_text(encoding="utf-8")
    visible_text = mask_comments_and_strings(mask_fenced_code_blocks(raw_text))
    # `layering_gate` reports what `header_imports` cannot read in the six libraries.
    imports, _ = header_imports(src, rel, visible_text)
    for edge in imports:
        if edge.dst in BROAD_IMPORTS and not suppress_broad_import_warning:
            message = f"broad import `{edge.dst}`; prefer a narrower entrypoint when possible"
            findings.append(Finding(level="WARN", path=rel, line=edge.line, message=message))
    for line in visible_text.splitlines():
        if m := NAMESPACE_RE.match(line):
            # The public page does not render namespace names. It reads the table's keys
            # to list modules that have no import edges.
            namespaces.append(m.group(1))

    return imports, namespaces, findings


def longest_path_len(nodes: set[str], edges: list[ImportEdge]) -> int | None:
    """Compute the longest direct-import chain, or `None` if the graph has a cycle.

    The “critical path” reported by this script is an approximate layering metric over
    direct imports. Semantic theorem dependencies require declaration-level proof data,
    which this source-import scan intentionally does not compute.
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
    """Compute repository-size statistics for the public graph page.

    The line counts use the same comment/string masking pass as the dependency parser.
    They are meant as stable repository-scale indicators and carry no Lean semantics.
    """

    total_lines = 0
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
        visible_lines = mask_comments_and_strings(mask_fenced_code_blocks(raw)).splitlines()
        layer = layer_of(module_name(root, path))

        total_lines += len(raw_lines)
        layer_files[layer] += 1
        layer_lines[layer] += len(raw_lines)
        top_level_files[rel.parts[0]] += 1

        for line in visible_lines:
            if not line.strip():
                continue
            code_lines += 1
            # The declaration regex is deliberately conservative. It feeds the scale
            # indicators on the website and plays no part in proof coverage accounting.
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


# No module may define `NN.IR.OpKind`, the operation type of the op-tagged IR.
DELETED_IR_TYPE = "NN.IR.OpKind"
SCOPE_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:public|private|noncomputable|meta)\s+)*"
    r"(namespace|section|mutual|end)\b[ \t]*([A-Za-z0-9_'.«»]*)"
)
DECL_NAME_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:public|private|protected|partial|unsafe|noncomputable|scoped|local|nonrec|meta)\s+)*"
    r"(?:class\s+inductive|def|theorem|lemma|structure|class|inductive|abbrev|axiom|opaque"
    r"|instance)\s+"
    r"([A-Za-z_«][A-Za-z0-9_'.!?«»]*)"
)


def deleted_ir_type_definitions(rel: str, text: str) -> list[Finding]:
    """Report each declaration of `DELETED_IR_TYPE` in a Lean file, read with the namespaces and
    sections open at it (`_root_.` names aside). `text` must have its comments and strings
    masked."""
    findings: list[Finding] = []
    scopes: list[str | None] = []  # one entry per scope: a namespace component, or None
    for line_no, line in enumerate(text.splitlines(), start=1):
        if m := SCOPE_RE.match(line):
            kind, name = m.group(1), m.group(2)
            parts = [p for p in name.split(".") if p]
            if kind == "namespace":
                scopes.extend(parts)
            elif kind in ("section", "mutual"):
                scopes.extend([None] * max(1, len(parts)))
            else:
                del scopes[len(scopes) - min(len(scopes), max(1, len(parts))):]
            continue
        if m := DECL_NAME_RE.match(line):
            name = m.group(1)
            if name.startswith("_root_."):
                full = name[len("_root_."):]
            else:
                full = ".".join([s for s in scopes if s] + [name])
            if full.replace("«", "").replace("»", "") == DELETED_IR_TYPE:
                findings.append(
                    Finding(
                        level="ERROR",
                        path=rel,
                        line=line_no,
                        message=(
                            f"defines `{DELETED_IR_TYPE}`, the operation type of the deleted "
                            "op-tagged IR"
                        ),
                    )
                )
    return findings


HEADER_MODIFIERS = {"module", "prelude", "public", "meta"}
MODULE_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(\.[A-Za-z_][A-Za-z0-9_']*)*")
# An `import` command after the header: one that starts a line, after optional modifiers.
LATE_IMPORT_RE = re.compile(r"(?m)^[ \t]*(?:(?:public|meta)[ \t]+)*import(?![A-Za-z0-9_'!?»])")


def header_imports(src: str, rel: str, text: str) -> tuple[list[ImportEdge], list[tuple[int, str]]]:
    """The import edges of a Lean header read token by token, so an import may span lines:
    `[public] [meta] import [all] Name` repeated after an optional `module`, up to the first
    other token. `text` must have its comments and strings masked. Fails closed: the second
    result lists what it cannot read, a name after `import` that is not a plain dotted ASCII
    name (escaped `«»` parts, other characters, a token glued to the name) and any line after
    the header that starts with an `import` command."""
    edges: list[ImportEdge] = []
    problems: list[tuple[int, str]] = []
    public = False
    expecting = False  # after `import` (and an optional `all`), a module name comes next
    end = len(text)
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        line = text.count("\n", 0, m.start()) + 1
        if expecting:
            if tok == "all":
                continue
            if not MODULE_NAME_RE.fullmatch(tok):
                problems.append((line, f"import of `{tok}` is not a plain module name"))
                end = m.end()
                break
            edges.append(ImportEdge(src=src, dst=tok, public=public, path=rel, line=line))
            expecting = False
            public = False
        elif tok == "import":
            expecting = True
        elif tok in HEADER_MODIFIERS:
            public = public or tok == "public"
        else:
            end = m.start()
            break
    for m in LATE_IMPORT_RE.finditer(text, end):
        line = text.count("\n", 0, m.start()) + 1
        problems.append((line, "`import` after the import header"))
    return edges, problems


# The layering gate. The six libraries are named by their module roots: the four directories
# under TorchLean/, then TorchLeanExamples/ and TorchLeanTest/, ranked by the direction of their
# imports in this order. A module's library is the root that its name starts with. Files outside
# these directories (scripts/checks, bindings, the guide) are not ranked. Lake has no setting that
# enforces import direction between libraries. This check enforces it, and `scripts/lake.sh lint`
# and CI run it.
LIBRARY_ROOTS = (
    "TorchLean.Tensor", "TorchLean.CUDA", "TorchLean.NN", "TorchLean.Verify",
    "TorchLeanExamples", "TorchLeanTest",
)
LAYER_RANKS = {library: rank for rank, library in enumerate(LIBRARY_ROOTS)}
# The directories that hold the six libraries.
SOURCE_DIRECTORIES = ("TorchLean", "TorchLeanExamples", "TorchLeanTest")
# The files of the CPU core's `@[extern]` declarations, implemented in csrc/cpu/torchlean_tensor.c.
TENSOR_NATIVE_FILES = (
    "TorchLean/Tensor/Storage.lean",
    "TorchLean/Tensor/Native/Pointwise.lean",
    "TorchLean/Tensor/Native/Transpose.lean",
)
TENSOR_NATIVE_SOURCE = "csrc/cpu/torchlean_tensor.c"
CUDA_STUB_SOURCE = "csrc/libtorch/unavailable.c"
CUDA_SOURCES = ("csrc/libtorch/torchlean.cpp", "csrc/libtorch/conv_pool.cpp",
                "csrc/libtorch/tree.cpp")
LEAN_PACKAGE = "TorchLean"
# The library check, a file that is not a module: `scripts/lake.sh lint` runs it with `lean` once
# the libraries are built. A chain of its imports must reach every module of the six libraries but
# the roots of executables.
LIBRARY_CHECK = "scripts/checks/Library.lean"

# Commands of a module that only imports (gate rule 3), after attributes and the modifiers in
# `COMMAND_MODIFIERS` are stripped. With `export` too, the module only re-exports names.
IMPORT_ONLY_COMMANDS = {
    "module", "import", "section", "end", "open", "set_option", "namespace", "universe",
    "variable",
}
COMMAND_MODIFIERS = {"public", "private", "meta", "noncomputable"}
# Keywords that start a command on an indented line.
COMMAND_KEYWORDS = IMPORT_ONLY_COMMANDS | COMMAND_MODIFIERS | {
    "export", "attribute", "def", "theorem", "lemma", "abbrev", "instance", "structure", "class",
    "inductive", "coinductive", "opaque", "axiom", "example", "macro", "macro_rules", "syntax",
    "elab", "elab_rules", "notation", "infix", "infixl", "infixr", "prefix", "postfix",
    "initialize", "builtin_initialize", "deriving", "mutual", "local", "scoped", "protected",
    "partial", "unsafe", "nonrec", "declare_syntax_cat", "add_decl_doc", "run_cmd", "run_elab",
    "run_meta", "omit", "include", "irreducible_def", "alias", "register_option", "prelude",
}
DECLARATION_KEYWORDS = {"def", "opaque", "abbrev", "theorem", "instance", "axiom"}
DECLARATION_MODIFIERS = COMMAND_MODIFIERS | {
    "protected", "partial", "unsafe", "nonrec", "scoped", "local",
}
NAME_PART_RE = r"(?:«[^»\n]*»|[^\W\d][\w'!?₀-₉ₐ-ₜᵢ-ᵪ]*)"
DECLARATION_NAME_RE = re.compile(NAME_PART_RE + r"(?:\." + NAME_PART_RE + r"|\.\d+)*")
OPENING_BRACKETS = "([{⦃⟨"
CLOSING_BRACKETS = ")]}⦄⟩"
EXTERN_RE = re.compile(r"\bextern\b")
STRING_LITERAL_RE = re.compile(r'\s*"((?:[^"\\\n]|\\.)*)"')


def library_of(module: str) -> str | None:
    """The library of a module of the six libraries, `None` for every other module."""
    for library in LIBRARY_ROOTS:
        if module.startswith(library + "."):
            return library
    return None


def layer_rank(module: str) -> int | None:
    """The rank of a module of the six libraries, `None` for every other module."""
    library = library_of(module)
    return None if library is None else LAYER_RANKS[library]


def walk_library_sources(root: pathlib.Path) -> list[pathlib.Path]:
    """Every .lean file under the directories of the libraries, found the way Lake's globs find
    them: by walking the directories, whatever git tracks or ignores."""
    found: list[pathlib.Path] = []
    for top in SOURCE_DIRECTORIES:
        for directory, children, names in os.walk(root / top):
            children[:] = sorted(name for name in children if name not in SKIP_PARTS)
            found.extend(pathlib.Path(directory) / name for name in sorted(names)
                         if name.endswith(".lean"))
    return found


def git_lean_files(root: pathlib.Path, *flags: str) -> set[str] | None:
    """The .lean paths under the directories of the libraries that `git ls-files` lists with
    `flags`, or `None` when the tree is not a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", *flags, "--",
             *(f"{top}/*.lean" for top in SOURCE_DIRECTORIES)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return {name.decode("utf-8", errors="surrogateescape")
            for name in result.stdout.split(b"\0") if name}


def module_commands(masked: str) -> list[tuple[int, str]]:
    """(line, keyword) of each command of a masked module. A command starts on a line whose
    first character is not blank, or on an indented line that starts with a command keyword, an
    attribute or `#`. Leading attributes and the modifiers of `COMMAND_MODIFIERS` are skipped."""
    commands: list[tuple[int, str]] = []
    lines = masked.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        first = re.match(r"[^\s(\[{]+", stripped)
        token = first.group(0) if first else stripped[:1]
        if line[0].isspace() and token not in COMMAND_KEYWORDS and not (
            stripped.startswith("@[") or stripped.startswith("#")
        ):
            continue
        rest = "\n".join(lines[index:index + 40])
        pos = len(rest) - len(rest.lstrip())
        while True:
            if rest.startswith("@[", pos):
                depth, j = 0, pos + 1
                while j < len(rest):
                    if rest[j] == "[":
                        depth += 1
                    elif rest[j] == "]":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                pos = j + 1
                while pos < len(rest) and rest[pos].isspace():
                    pos += 1
                continue
            modifier = re.match(r"(public|private|meta|noncomputable)\b\s*", rest[pos:])
            if modifier is None:
                break
            pos += modifier.end()
        keyword = re.match(r"[^\s(\[{]+", rest[pos:])
        commands.append((index + 1, keyword.group(0) if keyword else ""))
    return commands


def module_kind(masked: str) -> str:
    """`import-only`, `alias-only`, `attribute-only` or `code`, by the commands of a masked
    module (`empty` without any)."""
    keywords = [keyword for _, keyword in module_commands(masked)]
    if not keywords:
        return "empty"
    others = [keyword for keyword in keywords if keyword not in IMPORT_ONLY_COMMANDS]
    if not others:
        return "import-only"
    if all(keyword == "export" for keyword in others):
        return "alias-only"
    if all(keyword in {"export", "attribute"} for keyword in others):
        return "attribute-only"
    return "code"


def split_name(name: str) -> list[str]:
    """The components of a dotted Lean name, with `«»` escapes removed."""
    parts, current, k = [], "", 0
    while k < len(name):
        ch = name[k]
        if ch == "«":
            close = name.index("»", k)
            current += name[k:close + 1]
            k = close + 1
            continue
        if ch == ".":
            parts.append(current)
            current = ""
        else:
            current += ch
        k += 1
    parts.append(current)
    return [part.strip("«»") for part in parts]


def skip_attributes(masked: str, pos: int) -> tuple[int, list[tuple[int, int]]]:
    """The offset after the `@[…]` attributes and declaration modifiers at `pos` of a masked
    module, and the spans of the attributes."""
    n, spans = len(masked), []
    while True:
        while pos < n and masked[pos] in " \t\n":
            pos += 1
        if masked.startswith("@[", pos):
            depth, j = 0, pos + 1
            while j < n:
                if masked[j] == "[":
                    depth += 1
                elif masked[j] == "]":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            spans.append((pos, j + 1))
            pos = j + 1
            continue
        modifier = re.match(r"(?:" + "|".join(sorted(DECLARATION_MODIFIERS)) + r")\b",
                            masked[pos:pos + 20])
        if modifier is None:
            return pos, spans
        pos += modifier.end()


def binder_count(masked: str, start: int) -> int:
    """The parameters of the compiled function of the declaration whose signature starts at
    `start`: its binders (explicit, implicit, strict implicit and instance), then the top-level
    arrows of its type."""
    n = len(masked)
    count, depth, group, p = 0, 0, 0, start
    while p < n:
        ch = masked[p]
        if depth == 0:
            if masked.startswith(":=", p) or masked.startswith("where", p) or ch == "|":
                return count
            if ch == ":":
                break
            if ch in OPENING_BRACKETS:
                depth, group = 1, p
            elif not ch.isspace():
                name = DECLARATION_NAME_RE.match(masked, p)
                if name:
                    count += 1
                    p = name.end()
                    continue
        else:
            if ch in OPENING_BRACKETS:
                depth += 1
            elif ch in CLOSING_BRACKETS:
                depth -= 1
                if depth == 0:
                    body = masked[group + 1:p]
                    if masked[group] == "[" and ":" not in body:
                        count += 1
                    else:
                        names = body.split(":", 1)[0]
                        count += max(1, len([t for t in names.split() if t != "@&"]))
        p += 1
    q, depth = p + 1, 0
    while q < n:
        if depth == 0 and (masked.startswith(":=", q) or masked.startswith("\n\n", q)
                           or masked.startswith("where", q)):
            break
        ch = masked[q]
        if ch in OPENING_BRACKETS:
            depth += 1
        elif ch in CLOSING_BRACKETS:
            depth -= 1
        elif depth == 0 and (ch == "→" or masked.startswith("->", q)):
            count += 1
        q += 1
    return count


@dataclass(frozen=True)
class ExternDeclaration:
    """One `@[extern "symbol"]` declaration of a library module."""

    path: str
    line: int
    name: str
    symbol: str
    arity: int


def extern_declarations(rel: str, text: str, masked: str) -> list[ExternDeclaration]:
    """The `@[extern "symbol"]` declarations of a module, with full names: one pass over its
    commands (as in `module_commands`) that tracks `namespace`, `section`, `mutual` and `end`."""
    starts = [0] + [k + 1 for k, ch in enumerate(masked) if ch == "\n"]
    lines = masked.split("\n")
    scopes: list[tuple[str, str]] = []
    found: list[ExternDeclaration] = []
    k = 0
    while k < len(lines):
        line = lines[k]
        stripped = line.strip()
        first = re.match(r"[^\s(\[{]+", stripped)
        token = first.group(0) if first else stripped[:1]
        if not stripped or (line[0].isspace() and token not in COMMAND_KEYWORDS
                            and not stripped.startswith("@[")):
            k += 1
            continue
        pos, spans = skip_attributes(masked, starts[k] + len(line) - len(line.lstrip()))
        keyword_line = masked.count("\n", 0, pos)
        word = re.match(r"[^\s(\[{]+", masked[pos:pos + 100])
        keyword = word.group(0) if word else ""
        eol = masked.find("\n", pos)
        eol = len(masked) if eol < 0 else eol
        named = DECLARATION_NAME_RE.search(masked, pos + len(keyword), eol)
        components = split_name(named.group(0)) if named else []
        if keyword == "namespace":
            scopes.extend(("namespace", c) for c in components)
        elif keyword == "section":
            scopes.extend(("section", c) for c in components or [""])
        elif keyword == "mutual":
            scopes.append(("mutual", ""))
        elif keyword == "end":
            del scopes[max(0, len(scopes) - max(1, len(components))):]
        elif keyword in DECLARATION_KEYWORDS and spans:
            p = pos + len(keyword)
            while p < len(masked) and masked[p] in " \t\n":
                p += 1
            name = DECLARATION_NAME_RE.match(masked, p)
            for a, b in spans if name else []:
                for ext in EXTERN_RE.finditer(masked, a, b):
                    literal = STRING_LITERAL_RE.match(text, ext.end(), b)
                    if literal is None:
                        continue
                    parts = split_name(name.group(0))
                    full = parts[1:] if parts[0] == "_root_" else [
                        c for kind, c in scopes if kind == "namespace"] + parts
                    found.append(ExternDeclaration(
                        path=rel, line=text.count("\n", 0, a) + 1, name=".".join(full),
                        symbol=literal.group(1), arity=binder_count(masked, name.end()),
                    ))
        k = max(k + 1, keyword_line + 1)
    return found


def extern_attributes(text: str, masked: str) -> list[tuple[int, str]]:
    """(line, symbol) of every `extern "symbol"` attribute of a module, in any attribute form."""
    found: list[tuple[int, str]] = []
    for ext in EXTERN_RE.finditer(masked):
        literal = STRING_LITERAL_RE.match(text, ext.end(), min(len(text), ext.end() + 400))
        if literal is not None:
            found.append((text.count("\n", 0, ext.start()) + 1, literal.group(1)))
    return found


def _mangle_component(s: str) -> str:
    """`String.mangle` of Lean's compiler: ASCII alphanumerics kept, `_` doubled, every other
    character escaped."""
    out = ""
    for ch in s:
        code = ord(ch)
        if ch.isascii() and ch.isalnum():
            out += ch
        elif ch == "_":
            out += "__"
        elif code < 0x100:
            out += f"_x{code:02x}"
        elif code < 0x10000:
            out += f"_u{code:04x}"
        else:
            out += f"_U{code:08x}"
    return out


def _needs_disambiguation(mangled: str) -> bool:
    """Whether a mangled component could be read as an escape after a separating `_`."""
    k = 0
    while k < len(mangled) and mangled[k] == "_":
        k += 1
    if k == len(mangled):
        return True
    head, rest = mangled[k], mangled[k + 1:]
    width = {"x": 2, "u": 4, "U": 8}.get(head)
    if width:
        return len(rest) >= width and all(c in "0123456789abcdef" for c in rest[:width])
    return head.isdigit()


def lean_symbol(package: str, name: str, boxed: bool = False) -> str:
    """The C symbol Lean v4.35 gives a declaration of `package` (`mkPackageSymbolPrefix` and
    `Name.mangle`), or its boxed wrapper (`mkMangledBoxedName`)."""
    out = ""
    previous: tuple[str, str] | None = None
    for part in name.split("."):
        if part.isdigit():
            out = f"{part}_" if previous is None else f"{out}_{part}_"
            previous = ("num", part)
            continue
        mangled = _mangle_component(part)
        if previous is None:
            out = ("00" + mangled) if _needs_disambiguation(mangled) else mangled
        else:
            separate = (previous[0] == "str" and previous[1].endswith("_")) or (
                _needs_disambiguation(mangled)
            )
            out += ("_00" if separate else "_") + mangled
        previous = ("str", part)
    symbol = "lp_" + _mangle_component(package) + "_" + out
    if boxed:
        symbol += "_00__boxed" if symbol.endswith("__") else "___boxed"
    return symbol


def _strip_c_comments(text: str) -> str:
    """C/C++ source without comments, with string and character literals emptied except the path
    of an `#include "…"` directive. Newlines are kept."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append("\n" * text[i:n if j < 0 else j + 2].count("\n"))
            i = n if j < 0 else j + 2
            continue
        if text[i] in "\"'":
            quote, j = text[i], i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == "\\" else 1
            line = text[text.rfind("\n", 0, i) + 1:i]
            out.append(text[i:j + 1] if re.fullmatch(r"\s*#\s*include\s*", line)
                       else quote + quote)
            i = j + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _c_call_arguments(text: str, p: int) -> tuple[list[str], int]:
    """The arguments of the call whose `(` is at `p`, and the offset after its `)`."""
    depth, args, current, j = 0, [], "", p
    while j < len(text):
        ch = text[j]
        if ch in "([{":
            depth += 1
            if depth > 1:
                current += ch
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append(current.strip())
                return args, j + 1
            current += ch
        elif ch == "," and depth == 1:
            args.append(current.strip())
            current = ""
        else:
            current += ch
        j += 1
    return args, j


C_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
C_EXPORT_RE = re.compile(r"LEAN_EXPORT\s+((?:TORCHLEAN_WEAK\s+)?)[\w\s\*]*?\b([A-Za-z_]\w*)\s*\(")


def _c_preprocess(path: pathlib.Path) -> str:
    """A C/C++ file with its function-like macros expanded (`##`, `__VA_ARGS__`, and the local
    `#include "…"` files), enough to read the `LEAN_EXPORT` functions that macros declare."""
    macros: dict[str, tuple[list[str], str]] = {}

    def expand(text: str, disabled: frozenset[str] = frozenset()) -> str:
        result, i = "", 0
        while True:
            m = C_CALL_RE.search(text, i)
            if m is None:
                return result + text[i:]
            name = m.group(1)
            if name not in macros or name in disabled:
                result += text[i:m.end()]
                i = m.end()
                continue
            params, body = macros[name]
            args, end = _c_call_arguments(text, m.end() - 1)
            env: dict[str, str] = {}
            if params and params[-1] == "...":
                env["__VA_ARGS__"] = ", ".join(args[len(params) - 1:])
                args = args[:len(params) - 1]
            env.update(zip([p for p in params if p != "..."], args))
            sub = re.sub(r"\s*##\s*", "\0", body)
            sub = re.sub(r"\b[A-Za-z_]\w*\b", lambda t: env.get(t.group(0), t.group(0)), sub)
            result += text[i:m.start()] + expand(sub.replace("\0", ""), disabled | {name})
            i = end

    chunks: list[str] = []

    def run(text: str, here: pathlib.Path) -> None:
        pending: list[str] = []
        for line in re.sub(r"\\\n", " ", text).split("\n"):
            directive = re.match(r"\s*#\s*(define|undef|include)\b\s*(.*)", line)
            if directive is None:
                pending.append(line)
                continue
            chunks.append(expand("\n".join(pending)))
            pending = []
            what, rest = directive.groups()
            if what == "define":
                macro = re.match(r"([A-Za-z_]\w*)\(([^)]*)\)\s*(.*)", rest)
                if macro:
                    params = [p.strip() for p in macro.group(2).split(",") if p.strip()]
                    macros[macro.group(1)] = (params, macro.group(3))
            elif what == "undef":
                macros.pop(rest.strip(), None)
            else:
                local = re.match(r'"([^"]+)"', rest)
                included = here.parent / local.group(1) if local else None
                if included is not None and included.is_file():
                    run(_strip_c_comments(included.read_text(encoding="utf-8")), included)
        chunks.append(expand("\n".join(pending)))

    run(_strip_c_comments(path.read_text(encoding="utf-8")), path)
    return "\n".join(chunks)


def c_exports(path: pathlib.Path) -> dict[str, tuple[bool, int]]:
    """symbol -> (weak, parameter count) of the `LEAN_EXPORT` functions of a C/C++ file."""
    text = _c_preprocess(path)
    exports: dict[str, tuple[bool, int]] = {}
    for m in C_EXPORT_RE.finditer(text):
        args, _ = _c_call_arguments(text, m.end() - 1)
        arity = 0 if args in ([""], ["void"]) else len(args)
        weak = bool(m.group(1)) or "TORCHLEAN_WEAK" in m.group(0)
        exports[m.group(2)] = (weak, arity)
    return exports


LAKEFILE_LIB_RE = re.compile(r"(?m)^(?:@\[[^\]]*\]\s*)?lean_lib\s+(«[^»]+»|\S+)")
LAKEFILE_BLOCK_RE = re.compile(
    r"(?m)^(?:@\[[^\]]*\]\s*)?(?:lean_lib|lean_exe|target|extern_lib|require|package|"
    r"input_file|input_dir|def|script)\s"
)


def lakefile_findings(root: pathlib.Path) -> list[Finding]:
    """Gate rule 2 on lakefile.lean: its libraries are exactly the six, each glob-only, named by
    its module root."""
    path = root / "lakefile.lean"
    if not path.is_file():
        return [Finding(level="ERROR", path="lakefile.lean", line=1,
                        message="lakefile.lean is missing")]
    text = path.read_text(encoding="utf-8")
    masked = mask_comments_and_strings(text)
    starts = [m.start() for m in LAKEFILE_BLOCK_RE.finditer(masked)] + [len(masked)]
    findings: list[Finding] = []
    declared: set[str] = set()
    for m in LAKEFILE_LIB_RE.finditer(masked):
        name = m.group(1)
        line = masked.count("\n", 0, m.start()) + 1
        if name.startswith("«"):
            findings.append(Finding(
                level="ERROR", path="lakefile.lean", line=line,
                message=f"lean_lib {name} declares an atomic name, which `lake build "
                        f"{name.strip('«»')}` does not match: write it as a bare identifier",
            ))
            continue
        declared.add(name)
        body = masked[m.end():min(s for s in starts if s > m.start())]
        if name not in LAYER_RANKS:
            findings.append(Finding(
                level="ERROR", path="lakefile.lean", line=line,
                message=f"lean_lib `{name}` is not one of the six libraries "
                        f"({', '.join(LAYER_RANKS)})",
            ))
            continue
        roots = re.search(r"\broots\s*:=\s*#\[\s*\]", body)
        globs = re.search(
            r"\bglobs\s*:=\s*#\[\s*\.submodules\s+`" + re.escape(name) + r"\s*\]", body
        )
        if roots is None or globs is None:
            findings.append(Finding(
                level="ERROR", path="lakefile.lean", line=line,
                message=f"lean_lib `{name}` must declare `roots := #[]` and "
                        f"`globs := #[.submodules `{name}]`",
            ))
    for name in LAYER_RANKS:
        if name not in declared:
            findings.append(Finding(
                level="ERROR", path="lakefile.lean", line=1,
                message=f"lakefile.lean declares no lean_lib `{name}`",
            ))
    return findings


def executable_roots(root: pathlib.Path) -> set[str]:
    """The root modules of the `lean_exe` targets of lakefile.lean. Each defines `main`, so one
    file cannot import two of them."""
    path = root / "lakefile.lean"
    if not path.is_file():
        return set()
    masked = mask_comments_and_strings(path.read_text(encoding="utf-8"))
    starts = [m.start() for m in LAKEFILE_BLOCK_RE.finditer(masked)] + [len(masked)]
    roots: set[str] = set()
    for m in re.finditer(r"(?m)^(?:@\[[^\]]*\]\s*)?lean_exe\s+(«[^»]+»|\S+)", masked):
        body = masked[m.end():min(s for s in starts if s > m.start())]
        named = re.search(r"\broot\s*:=\s*`(«[^»]+»|[A-Za-z0-9_'.]+)", body)
        roots.add((named or m).group(1).strip("«»"))
    return roots


def library_sinks(modules: set[str], edges: list[ImportEdge]) -> list[str]:
    """The modules of the four libraries that no other module of the four imports."""
    def in_library(module: str) -> bool:
        rank = layer_rank(module)
        return rank is not None and rank <= LAYER_RANKS["TorchLean.Verify"]

    library = {m for m in modules if in_library(m)}
    imported = {e.dst for e in edges if e.src in library and e.dst != e.src}
    return sorted(library - imported)


def layering_gate(root: pathlib.Path) -> tuple[list[Finding], list[str]]:
    """The findings of the five gate rules (rank, ownership, import-only and alias-only modules,
    natives, the reach of the library check) and the library sinks."""
    findings: list[Finding] = []
    sources = walk_library_sources(root)
    modules: set[str] = set()
    edges: list[ImportEdge] = []
    cuda: dict[str, ExternDeclaration] = {}
    tensor: dict[str, ExternDeclaration] = {}

    def error(path: str, line: int, message: str) -> None:
        findings.append(Finding(level="ERROR", path=path, line=line, message=message))

    for top in SOURCE_DIRECTORIES:
        if (root / f"{top}.lean").exists():
            error(f"{top}.lean", 1, f"root file `{top}.lean`: the libraries are glob-only")
    for path in sources:
        rel = path.relative_to(root).as_posix()
        mod = module_name(root, path)
        modules.add(mod)
        if mod in LAYER_RANKS:
            error(rel, 1, f"root file `{rel}`: the libraries are glob-only")
            continue
        layer = library_of(mod)
        if layer is None:
            error(rel, 1, f"`{rel}` is outside the directories of the six libraries "
                          f"({', '.join(name.replace('.', '/') + '/' for name in LIBRARY_ROOTS)})")
            continue
        text = path.read_text(encoding="utf-8")
        masked = mask_comments_and_strings(text)
        mod_edges, problems = header_imports(mod, rel, masked)
        edges.extend(mod_edges)
        for line, what in problems:
            error(rel, line, f"cannot read the imports of `{mod}`: {what}")
        # Rule 1: no import of a module of higher rank, except by the tests.
        if layer != "TorchLeanTest":
            for e in mod_edges:
                rank = layer_rank(e.dst)
                if rank is not None and rank > LAYER_RANKS[layer]:
                    error(rel, e.line, f"{layer} imports {library_of(e.dst)} (order "
                                       f"{' < '.join(LAYER_RANKS)}): `{mod}` imports `{e.dst}`")
        # Rule 3: every module declares something.
        kind = module_kind(masked)
        if kind in {"empty", "import-only", "alias-only"}:
            source = "re-exports from" if kind == "alias-only" else "imports"
            error(rel, 1, f"`{mod}` is {kind}: import the modules it {source} directly")
        # Rule 4: where the `@[extern]` declarations live.
        attributes = extern_attributes(text, masked)
        if not attributes:
            continue
        if layer == "TorchLean.CUDA" or rel in TENSOR_NATIVE_FILES:
            declarations = extern_declarations(rel, text, masked)
            target = cuda if layer == "TorchLean.CUDA" else tensor
            target.update((ext.symbol, ext) for ext in declarations)
            for line, symbol in attributes:
                if not any(ext.symbol == symbol for ext in declarations):
                    error(rel, line, f"`extern \"{symbol}\"` is not on a declaration this "
                                     f"audit reads (`@[extern \"…\"] def|opaque|…`)")
        else:
            for line, symbol in attributes:
                error(rel, line, f"`extern \"{symbol}\"` outside TorchLean/CUDA/ and the "
                                 f"Tensor native modules")

    # Rule 2: what Lake's globs compile.
    findings.extend(lakefile_findings(root))
    tracked = git_lean_files(root, "--cached")
    untracked = git_lean_files(root, "--others", "--exclude-standard")
    ignored = git_lean_files(root, "--others", "--ignored", "--exclude-standard")
    if tracked is None or untracked is None or ignored is None:
        findings.append(Finding(level="WARN", path="lakefile.lean", line=1,
                                message="not a git checkout: the checks for ignored and "
                                        "untracked library files did not run"))
    else:
        for path in sources:
            rel = path.relative_to(root).as_posix()
            if rel in ignored:
                error(rel, 1, "git-ignored .lean file under a library directory, which "
                              "Lake's globs compile")
            elif rel in untracked:
                findings.append(Finding(level="WARN", path=rel, line=1,
                                        message="untracked .lean file under a library "
                                                "directory, which Lake's globs compile"))

    # Rule 4: each extern set agrees with its C sources.
    missing = [name for name in (TENSOR_NATIVE_SOURCE, CUDA_STUB_SOURCE, *CUDA_SOURCES)
               if not (root / name).is_file()]
    for name in missing:
        error(name, 1, f"native source `{name}` is missing")
    if not missing:
        stub = c_exports(root / CUDA_STUB_SOURCE)
        libtorch = {symbol: export for name in CUDA_SOURCES
                    for symbol, export in c_exports(root / name).items()}
        cpu = c_exports(root / TENSOR_NATIVE_SOURCE)
        for symbol in sorted(set(cuda) - set(stub)):
            error(cuda[symbol].path, cuda[symbol].line,
                  f"CUDA extern `{symbol}` has no export in {CUDA_STUB_SOURCE}")
        for symbol in sorted(set(stub) - set(cuda)):
            error(CUDA_STUB_SOURCE, 1, f"stub export `{symbol}` has no CUDA extern")
        for symbol in sorted(set(cuda) - set(libtorch)):
            error(cuda[symbol].path, cuda[symbol].line,
                  f"CUDA extern `{symbol}` has no LEAN_EXPORT in {' or '.join(CUDA_SOURCES)}")
        strong = {symbol for symbol, (weak, _) in cpu.items() if not weak}
        for symbol in sorted(set(tensor) - strong):
            error(tensor[symbol].path, tensor[symbol].line,
                  f"Tensor extern `{symbol}` has no LEAN_EXPORT in {TENSOR_NATIVE_SOURCE}")
        for symbol in sorted(strong - set(tensor)):
            error(TENSOR_NATIVE_SOURCE, 1,
                  f"export `{symbol}` has no extern in the Tensor native modules")
        boxed = {lean_symbol(LEAN_PACKAGE, ext.name, boxed=True): ext for ext in tensor.values()}
        for symbol, (weak, arity) in sorted(cpu.items()):
            if not weak or not symbol.startswith(f"lp_{LEAN_PACKAGE}_"):
                continue
            ext = boxed.get(symbol)
            if ext is None:
                error(TENSOR_NATIVE_SOURCE, 1, f"weak wrapper `{symbol}` matches no extern "
                                               f"declaration of the Tensor native modules")
            elif ext.arity != arity:
                error(TENSOR_NATIVE_SOURCE, 1, f"weak wrapper `{symbol}` takes {arity} "
                                               f"parameters; `{ext.name}` has {ext.arity}")

    # Rule 5: a chain of imports from the library check reaches every module of the six libraries
    # but the roots of executables.
    sinks = library_sinks(modules, edges)
    path = root / LIBRARY_CHECK
    if not path.is_file():
        error(LIBRARY_CHECK, 1, f"`{LIBRARY_CHECK}` is missing")
        return findings, sinks
    check_edges, problems = header_imports(
        module_name(root, path), LIBRARY_CHECK,
        mask_comments_and_strings(path.read_text(encoding="utf-8")))
    for line, what in problems:
        error(LIBRARY_CHECK, line, f"cannot read the imports of `{LIBRARY_CHECK}`: {what}")
    imports: dict[str, set[str]] = {}
    for e in edges:
        imports.setdefault(e.src, set()).add(e.dst)
    reached = {e.dst for e in check_edges}
    queue = list(reached)
    while queue:
        for mod in imports.get(queue.pop(), set()) - reached:
            reached.add(mod)
            queue.append(mod)
    for mod in sorted(modules - reached - executable_roots(root)):
        if library_of(mod) is not None:
            error(mod.replace(".", "/") + ".lean", 1,
                  f"no chain of imports from `{LIBRARY_CHECK}` reaches this module, so the checks "
                  "of that file skip it: import it there")
    return findings, sinks


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

    # The layering gate: rank, ownership, import-only and alias-only modules, natives, the
    # reach of the library check.
    gate_findings, sinks = layering_gate(root)
    findings.extend(gate_findings)

    for path in lean_files:
        findings.extend(
            deleted_ir_type_definitions(
                path.relative_to(root).as_posix(),
                mask_comments_and_strings(path.read_text(encoding="utf-8")),
            )
        )

    top_fan_in = fan_in.most_common(20)
    top_fan_out = fan_out.most_common(20)
    critical_nodes = modules - CRITICAL_PATH_EXCLUDE
    critical_edges = [
        e for e in internal_edges if e.src in critical_nodes and e.dst in critical_nodes
    ]
    critical_path = longest_path_len(critical_nodes, critical_edges)

    # The public graph page reads this JSON as is. Keep the field names stable so
    # the page can change without reparsing Lean in the browser.
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
            "library_sinks": len(sinks),
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
        "library_sinks": sinks,
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
        "This report measures the Lean import graph: which source modules import which other "
        "modules. It is an architecture and maintenance artifact. It is not the runtime graph IR "
        "used to represent neural-network computations, and it is not a declaration-level proof "
        "dependency graph."
    )
    lines.append("")
    lines.append(
        "The audit also enforces the layering of the libraries: a module of TorchLean.Tensor, "
        "TorchLean.CUDA, TorchLean.NN, TorchLean.Verify or TorchLeanExamples imports only "
        "modules of its own or a lower library, in that order, and TorchLeanTest may import "
        "anything. It checks that the six libraries are glob-only, that every module declares "
        "something, that the `@[extern]` declarations live in TorchLean.CUDA and the Tensor "
        "native modules and agree with the C sources, and that a chain of imports from "
        f"`{LIBRARY_CHECK}` reaches every module of the six libraries but the roots of "
        "executables."
    )
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
    lines.append(f"- Library sinks: `{s['library_sinks']}`")
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
    parser.add_argument("--sinks", action="store_true",
                        help="print the library sinks, one per line (for `exe graph --to`)")
    args = parser.parse_args(argv)

    if args.sinks:
        print("\n".join(layering_gate(args.root.resolve())[1]))
        return 0

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
