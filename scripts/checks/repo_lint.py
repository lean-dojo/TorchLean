#!/usr/bin/env python3
"""
TorchLean repo lints (project-specific).

This linter stays dependency-free so it can run in CI and locally.

Checks are split into:
  - errors: must be fixed (fail CI)
  - warnings: reported for visibility (do not fail by default)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys
import urllib.parse
from dataclasses import dataclass
from typing import Iterable

# Do not create the bytecode artifacts this command checks for.
sys.dont_write_bytecode = True
from dependency_audit import mask_comments_and_strings as _mask_lean_comments_and_strings


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_SCOPE_SENTINEL = REPO_ROOT / "NN/MLTheory/CROWN/Lyapunov/Certificate.lean"

# External trees that may exist in a developer checkout but are not part of TorchLean's core sources
# and must not affect repo policy/CI. These are user-cloned repos outside TorchLean's source tree.
#
# Source discovery skips these directories so optional checkouts do not affect `lake lint`.
VENDORED_DIR_NAMES = {
    "Two-Stage_Neural_Controller_Training",  # optional external checkout (α,β-CROWN workflows)
    "PINN_verification",  # user-cloned external repo (gitignored)
}

# Prune these before descent: filtering yielded paths still crawls their contents on EFS.
SOURCE_EXCLUDED_DIR_NAMES = VENDORED_DIR_NAMES | {
    ".git",
    ".lake",
    ".cache",
    ".venv",
    ".bundle",
    ".jekyll-cache",
    ".sass-cache",
    ".verso",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "node_modules",
    "_out",
    "_site",
    "vendor",
}

# These output paths have ordinary names that can also occur in authored source directories.
GENERATED_DOC_DIRS = {
    "docs/manual",
    "home_page/docs",
    "home_page/manual",
    "home_page/blueprint/print",
    "home_page/blueprint/web",
}

# Keep the trusted boundary explicit: axioms must be quarantined, named, and documented.
# TorchLean currently has no custom axioms.
ALLOWED_AXIOMS: dict[str, set[str]] = {}

# Visibility and attributes do not change an axiom's contribution to the trusted boundary.
# Run this on masked Lean source so comments and string literals cannot create declarations.
AXIOM_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:public|private|protected|noncomputable|unsafe)\s+)*"
    r"axiom\s+([^\s:({]+)",
    flags=re.MULTILINE,
)

# Documentation may mention producer-side environment variables only when the implementation hook
# exists in source. This prevents guide text from advertising phantom integration flags.
DOCUMENTED_ENV_VAR_IMPLEMENTATIONS = {
    "ABCROWN_ARTIFACT_OUT": "scripts/verification/abcrown/export_leaf_artifact.py",
}

TRUST_BOUNDARY_DECL_REFS = {
    "NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound": (
        "NN/MLTheory/CROWN/Proofs/GraphCrownCertSoundness.lean",
        re.compile(r"\bdef\s+CrownTransferSound\b"),
    ),
    "NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.OpsExact.Sound": (
        "NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean",
        re.compile(r"\bclass\s+Sound\s*:\s*Prop\b"),
    ),
}

TORCHLEAN_SOURCE_LINK_RE = re.compile(
    r"https://github\.com/lean-dojo/TorchLean/blob/main/([^\s\)\]`]+)"
)

DOCGEN_API_LINK_RE = re.compile(
    r"""(?:['"(])(/docs/[A-Za-z0-9_./-]+\.html(?:#[A-Za-z0-9_'.:-]+)?)"""
)

LOCAL_SOURCE_REF_RE = re.compile(
    r"`((?:NN|blueprint|home_page|scripts|csrc)/[^`\s]+"
    r"\.(?:lean|md|py|json|sh|cu|c|h|yml|yaml))`"
)

LEAN_DOC_COMMENT_RE = re.compile(r"/-(?:!|-).*?-/", flags=re.DOTALL)
# A single backslash starts the delimiters that MD4Lean does not recognize.
# The negative lookbehind leaves TeX line breaks such as `\\[1ex]` alone.
DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE = re.compile(r"(?<!\\)\\[\(\[]")
DOCGEN_DISPLAY_MATH_RE = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)
DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE = re.compile(
    r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+",
    flags=re.MULTILINE,
)
VERSO_TEX_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\\[A-Za-z]+|_\{[^}`]+\})[^`\n]*`"
)
FORMALIZATION_MATH_IN_ORDINARY_CODE_RE = re.compile(
    r"(?<!\$)`[^`\n]*(?:\s[\^+*/<>]=?\s|≤|≥|±)[^`\n]*`"
)

PUBLIC_NUMERICAL_EXAMPLE_PREFIXES = (
    "NN/Examples/Quickstart/",
    "NN/Examples/Models/",
    "NN/Examples/Data/",
    "NN/Examples/Functional/",
    "NN/Examples/Factorization/",
    "NN/Examples/Interop/",
)

PUBLIC_NUMERICAL_SPEC_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bSpec\.(?:fill|mseSpec|getSpec|toScalarSpec|qrQSpec|qrRSpec|qrSpec|"
            r"choleskySpec|linearSpec|matVecMulSpec|matMulSpec|"
            r"convOutSpatial|poolOutSpatialPad)\b|"
            r"\bActivation\.(?:reluSpec|sigmoidSpec|tanhSpec)\b|"
            r"\b(?:Spec\.)?Tensor\.(?:addSpec|subSpec|mulSpec|divSpec|scaleSpec)\b"
        ),
        "runnable numerical examples should call the public `Tensor.*` or `nn.*` operation; "
        "reserve direct computational `Spec.*` calls for proof and specification examples.",
    ),
    (
        re.compile(r"\[[^\]\n]+\]!"),
        "runnable numerical examples should use bounded tensor indices or checked container lookup, "
        "not forced indexing.",
    ),
    (
        re.compile(r"\bunreachable!"),
        "runnable numerical examples should report invalid runtime input explicitly, not use "
        "`unreachable!`.",
    ),
]

# Root-driven Lake targets that collectively typecheck maintained modules outside the dedicated
# example and test libraries. Keep this list aligned with `lakefile.lean`.
LEAN_TYPECHECK_ROOTS = {
    "NN",
    "NN.CI.All",
    "NN.CI.SlowProofs",
    "NN.Docs",
    "NN.Verification.Main",
}

LEAN_TYPECHECK_GLOB_PREFIXES = (
    "NN/Examples/",
    "NN/Tests/",
)

# The tensor compiler's internal language uses Lean vectors for compiler indices,
# proof-recursive lists for syntax.
# Public numerical-container policies apply at `NN.Tensor`, not inside this
# implementation namespace.
TENSOR_INTERNAL_PREFIX = "NN/Tensor/Internal/"
TENSOR_VECTOR_BOUNDARY_FILES = {
    "NN/Tensor/Conversion.lean",
    "NN/Tests/Tensor/Storage.lean",
}



TOP_LEVEL_API_DECL_RE = re.compile(
    r"^\s*(def|structure|inductive|class|abbrev|instance|theorem|lemma)\s+",
    flags=re.MULTILINE,
)


# Lean.Parser.Module.Syntax: optional public, optional meta, import, optional all, module.
LEAN_IMPORT_RE = re.compile(
    r"^[ \t]*(?P<directive>(?P<public>public[ \t]+)?(?:meta[ \t]+)?"
    r"import[ \t]+(?:all[ \t]+)?(?P<module>[A-Za-z0-9_.']+))[ \t]*\r?$",
    flags=re.MULTILINE,
)


def _lean_import_header(masked: str) -> str:
    """Keep module-header imports and their offsets, excluding imports inside body examples."""
    lines: list[str] = []
    for line in masked.splitlines(keepends=True):
        stripped = line.strip()
        if (not stripped or stripped in {"module", "prelude"}
                or LEAN_IMPORT_RE.fullmatch(line.rstrip("\r\n"))):
            lines.append(line)
        else:
            break
    return "".join(lines)

# Import-only API umbrellas should compose focused public modules. Re-exporting these implementation
# roots makes runtime internals part of the user API by accident.
BROAD_LOW_LEVEL_IMPORTS = {
    "NN",
    "NN.Proofs",
    "NN.Runtime",
    "NN.Spec",
    "NN.Verification",
    "NN.Runtime.Autograd.Model",
}

BROAD_LOW_LEVEL_IMPORT_PREFIXES = (
    "NN.Runtime.Autograd.Engine.",
    "NN.Runtime.Autograd.Torch.Core.",
    "NN.Spec.Core.Tensor.Internal.",
)

CONTRACT_SOURCE_FILE_RE = re.compile(
    r"\.sourceFile\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_NATIVE_SYMBOL_RE = re.compile(
    r"\.nativeSymbol\s*\{(?P<body>[^{}]*)\}",
    flags=re.DOTALL,
)
CONTRACT_GUARD_SOURCE_PATH_RE = re.compile(
    r"\.runtimeGuard\s+\"[^\"]*\.(?:c|cc|cpp|cu|cuh|h|hpp)\"",
)


@dataclass(frozen=True)
class Finding:
    """One repository-lint warning or error."""

    level: str  # "ERROR" | "WARN"
    path: pathlib.Path
    line: int | None
    col: int | None
    message: str

    def render(self) -> str:
        """Format the finding for terminal and CI output."""
        rel = self.path.relative_to(REPO_ROOT)
        if self.line is None:
            return f"{self.level}: {rel}: {self.message}"
        if self.col is None:
            return f"{self.level}: {rel}:{self.line}: {self.message}"
        return f"{self.level}: {rel}:{self.line}:{self.col}: {self.message}"


def _iter_lean_files() -> Iterable[pathlib.Path]:
    """Yield tracked and non-ignored project Lean sources without crawling build trees."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        "*.lean",
    ]
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("repo lint requires Git to enumerate project sources") from error
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"failed to enumerate project sources with Git: {message}") from error

    for relative_bytes in result.stdout.split(b"\0"):
        if not relative_bytes:
            continue
        relative = pathlib.Path(relative_bytes.decode("utf-8", errors="surrogateescape"))
        if any(directory in relative.parts for directory in VENDORED_DIR_NAMES):
            continue
        if "_out" in relative.parts:
            continue
        path = REPO_ROOT / relative
        if path.is_file():
            yield path

# FloatLib owns the shared numerical library. The remaining TorchLean adapters follow the
# same source-style rules as the rest of this repository.
MAX_LINE_LENGTH = 100


def _check_line_style(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check the 100-column limit, except for single-line Verso directives."""
    for lineno, line in enumerate(text.split("\n"), start=1):
        # Verso block directives (`:::theorem "label" (lean := "...")`) must stay on one line, so
        # the column limit does not apply to them.
        if line.lstrip().startswith(":::"):
            continue
        if len(line) > MAX_LINE_LENGTH:
            findings.append(
                Finding("ERROR", path, lineno, MAX_LINE_LENGTH + 1,
                        f"line exceeds {MAX_LINE_LENGTH} characters; wrap it.")
            )


def _check_lean_target_coverage(findings: list[Finding]) -> None:
    """Require every maintained `NN` module to belong to a typecheck target."""

    nn_files = {
        path.relative_to(REPO_ROOT).with_suffix("").as_posix().replace("/", "."): path
        for path in _iter_lean_files()
        if path == REPO_ROOT / "NN.lean" or path.is_relative_to(REPO_ROOT / "NN")
    }
    imports: dict[str, list[str]] = {}
    for module, path in nn_files.items():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        masked = _mask_lean_comments_and_strings(text)
        imports[module] = [
            match.group("module")
            for match in LEAN_IMPORT_RE.finditer(_lean_import_header(masked))
            if match.group("module") in nn_files
        ]

    covered: set[str] = set()
    pending = list(LEAN_TYPECHECK_ROOTS)
    while pending:
        module = pending.pop()
        if module in covered or module not in nn_files:
            continue
        covered.add(module)
        pending.extend(imports.get(module, []))

    for module, path in sorted(nn_files.items()):
        rel = path.relative_to(REPO_ROOT).as_posix()
        if module in covered or rel.startswith(LEAN_TYPECHECK_GLOB_PREFIXES):
            continue
        findings.append(
            Finding(
                "ERROR",
                path,
                None,
                None,
                "maintained Lean module is not reachable from `NN`, `NNCI`, `NNSlowProofs`, "
                "`TorchLeanDocs`, or an executable root, and is not covered by the `NNExamples` "
                "or `NNTests` globs.",
            )
        )


def _iter_source_files(
    root: pathlib.Path, suffix: str, *, exclude_dirs: Iterable[str] = ()
) -> Iterable[pathlib.Path]:
    """Yield authored files, including untracked sources, without entering generated trees."""

    excluded_names = SOURCE_EXCLUDED_DIR_NAMES | set(exclude_dirs)
    excluded_paths = {REPO_ROOT / relative for relative in GENERATED_DOC_DIRS}
    if root.name in excluded_names or root in excluded_paths:
        return
    for directory, directories, files in os.walk(root, topdown=True, followlinks=False):
        parent = pathlib.Path(directory)
        directories[:] = sorted(
            name for name in directories
            if name not in excluded_names and parent / name not in excluded_paths
        )
        for name in sorted(files):
            if name.endswith(suffix):
                yield parent / name


def _iter_authored_public_docs() -> Iterable[pathlib.Path]:
    """Yield maintained guide and website sources, excluding generated and vendored trees."""

    yield REPO_ROOT / "README.md"
    yield from (REPO_ROOT / "docs").glob("*.md")
    yield from _iter_source_files(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint", ".lean")
    yield from _iter_source_files(
        REPO_ROOT / "home_page", ".md", exclude_dirs={"blueprint", "docs"}
    )


def _iter_doc_paths() -> Iterable[pathlib.Path]:
    """Yield guide, website, blueprint, and source-local documentation for link checks."""

    yield from REPO_ROOT.glob("README.md")
    yield from (REPO_ROOT / "docs").glob("*.md")
    yield from _iter_source_files(REPO_ROOT / "home_page/blueprint", ".lean")
    for relative in ("home_page", "NN", "scripts"):
        yield from _iter_source_files(REPO_ROOT / relative, ".md")


def _iter_generated_script_artifacts() -> Iterable[pathlib.Path]:
    """Generated files that stay outside the checked-in `scripts/` tree."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or p.name == ".DS_Store":
            yield p


def _iter_script_files() -> Iterable[pathlib.Path]:
    """Yield checked-in support scripts and helper files under `scripts/`."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if p.is_file():
            yield p


def _is_executable(path: pathlib.Path) -> bool:
    """Return whether any executable bit is set for `path`."""

    return bool(path.stat().st_mode & 0o111)


def _has_shebang(text: str) -> bool:
    """Return whether `text` starts with a Unix shebang line."""

    return text.startswith("#!")


def _has_python_module_docstring(text: str) -> bool:
    """Return whether a Python script starts with a module docstring after an optional shebang."""

    lines = text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines = lines[1:]
    body = "\n".join(lines).lstrip()
    return body.startswith(('"""', "'''"))


def _line_col(text: str, idx: int) -> tuple[int, int]:
    """Translate a string offset into 1-based line and column coordinates."""
    # 1-based (Lean-style).
    line = text.count("\n", 0, idx) + 1
    last_nl = text.rfind("\n", 0, idx)
    col = idx - last_nl
    return line, col


def _has_nn_header(path: pathlib.Path, text: str) -> bool:
    """Check whether an `NN/` source file carries the standard TorchLean header."""
    # TorchLean policy: NN sources carry a consistent header at the top of the file.
    if not path.is_relative_to(REPO_ROOT / "NN"):
        return True
    head = "\n".join(text.splitlines()[:10])
    return "Copyright (c) 2026 TorchLean" in head


def _has_lean_module_docstring(text: str) -> bool:
    """Return whether a Lean source contains a module docstring (`/-! ... -/`)."""
    return "/-!" in text


def _mask_verso_prose(text: str) -> str:
    """Preserve Lean examples in a `#doc` body without treating its prose as Lean code."""

    doc = re.search(r"^#doc\b[^\n]*=>[ \t]*\n", text, flags=re.MULTILINE)
    if doc is None:
        return text
    out = [text[:doc.end()]]
    fence: str | None = None
    lean_block = False
    for line in text[doc.end():].splitlines(keepends=True):
        marker = re.match(r"^[ \t]*(`{3,}|~{3,})(.*?)[\r\n]*$", line)
        if marker is not None and fence is None:
            fence = marker.group(1)
            language = marker.group(2).strip().split()
            lean_block = not language or language[0] in {"lean", "leanTerm", "leanInit"}
        elif (
            marker is not None
            and fence is not None
            and marker.group(1)[0] == fence[0]
            and len(marker.group(1)) >= len(fence)
            and not marker.group(2).strip()
        ):
            fence = None
            lean_block = False
        elif lean_block:
            out.append(line)
            continue
        # Inline Lean roles also elaborate terms; retain them for the banned-construct checks.
        masked = list(re.sub(r"[^\r\n]", " ", line))
        if fence is None:
            for role in re.finditer(r"\{lean(?:\s[^}\n]*)?\}(`+)(.*?)\1", line):
                start, end = role.span(2)
                masked[start:end] = line[start:end]
        out.append("".join(masked))
    return "".join(out)




def _check_local_source_refs(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check backtick-quoted local source paths in authored docs/comments."""

    for m in LOCAL_SOURCE_REF_RE.finditer(text):
        raw_target = m.group(1).split("#", 1)[0]
        if any(marker in raw_target for marker in ("*", "<", ">", "...")):
            continue
        target = pathlib.Path(urllib.parse.unquote(raw_target))
        if target.is_absolute():
            continue
        if not (REPO_ROOT / target).exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead local source reference: `{raw_target}` does not exist in this checkout.",
                )
            )


def _check_docgen_api_links(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check website links to generated API pages against the corresponding Lean source."""

    for m in DOCGEN_API_LINK_RE.finditer(text):
        url = urllib.parse.unquote(m.group(1))
        module_path = url.removeprefix("/docs/").split("#", 1)[0].removesuffix(".html")
        target = REPO_ROOT / f"{module_path}.lean"
        if not target.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead generated API link: `/docs/{module_path}.html` has no `{module_path}.lean` source.",
                )
            )


def _check_lean_doc_math(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Reject documentation math that DocGen cannot pass intact to MathJax."""

    for comment in LEAN_DOC_COMMENT_RE.finditer(text):
        for match in DOCGEN_UNSUPPORTED_MATH_DELIMITER_RE.finditer(comment.group()):
            offset = comment.start() + match.start()
            line, col = _line_col(text, offset)
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "DocGen does not preserve this math delimiter; use `$...$` or `$$...$$`.",
                )
            )
        for display in DOCGEN_DISPLAY_MATH_RE.finditer(comment.group()):
            for match in DOCGEN_MARKDOWN_LIST_IN_DISPLAY_MATH_RE.finditer(display.group(1)):
                offset = comment.start() + display.start(1) + match.start()
                line, col = _line_col(text, offset)
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "a display-math line starts like a Markdown list item; move the operator "
                        "to the preceding TeX line so DocGen keeps the equation together.",
                    )
                )


def _check_verso_math_roles(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Catch mathematical TeX that would remain an ordinary monospace code span."""

    patterns = [VERSO_TEX_IN_ORDINARY_CODE_RE]
    if path.is_relative_to(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint/FormalizationMap"):
        patterns.append(FORMALIZATION_MATH_IN_ORDINARY_CODE_RE)
    for pattern in patterns:
        for match in pattern.finditer(text):
            line, col = _line_col(text, match.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "mathematical prose is in an ordinary code span; use a Verso `$` math role.",
                )
            )


def _lean_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lean_optional_string_field(body: str, field: str) -> str | None:
    m = re.search(rf"\b{re.escape(field)}\s*:=\s*some\s+\"([^\"]+)\"", body)
    return m.group(1) if m else None


def _lake_declares_target(lake_text: str, target: str) -> bool:
    return re.search(
        rf"^\s*(?:target|lean_exe|lean_lib)\s+{re.escape(target)}\b",
        lake_text,
        flags=re.MULTILINE,
    ) is not None


def _check_backend_contract_refs(
    path: pathlib.Path,
    text: str,
    lake_text: str,
    findings: list[Finding],
) -> None:
    """Check structured backend contract references to local sources and native symbols."""

    for m in CONTRACT_GUARD_SOURCE_PATH_RE.finditer(text):
        line, col = _line_col(text, m.start())
        findings.append(
            Finding(
                "ERROR",
                path,
                line,
                col,
                "native source paths belong in structured `.sourceFile` or `.nativeSymbol` provenance, not a runtime-guard label.",
            )
        )

    for m in CONTRACT_SOURCE_FILE_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        if raw_path is None:
            line, col = _line_col(text, m.start())
            findings.append(Finding("ERROR", path, line, col, "`.sourceFile` provenance is missing `path := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.sourceFile` provenance points to missing source `{raw_path}`.",
                )
            )

    for m in CONTRACT_NATIVE_SYMBOL_RE.finditer(text):
        body = m.group("body")
        raw_path = _lean_string_field(body, "path")
        symbol = _lean_string_field(body, "symbol")
        build_target = _lean_optional_string_field(body, "buildTarget?")
        line, col = _line_col(text, m.start())
        if raw_path is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `path := ...`."))
            continue
        if symbol is None:
            findings.append(Finding("ERROR", path, line, col, "`.nativeSymbol` provenance is missing `symbol := ...`."))
            continue
        source = REPO_ROOT / raw_path
        if not source.exists():
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance points to missing source `{raw_path}`.",
                )
            )
            continue
        try:
            source_text = source.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if symbol not in source_text:
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names `{symbol}`, but it does not occur in `{raw_path}`.",
                )
            )
        if build_target is not None and not _lake_declares_target(lake_text, build_target):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"`.nativeSymbol` provenance names Lake target `{build_target}`, but `lakefile.lean` does not declare it.",
                )
            )


def lint_repo(*, fail_on_warn: bool) -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []
    try:
        lake_text = (REPO_ROOT / "lakefile.lean").read_text(encoding="utf-8")
    except OSError as e:
        lake_text = ""
        findings.append(Finding("ERROR", REPO_ROOT / "lakefile.lean", None, None, f"failed to read file: {e}"))

    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see "
                "NN/MLTheory/CROWN/Lyapunov/Certificate.lean.",
            )
        )

    _check_lean_target_coverage(findings)

    for path in _iter_generated_script_artifacts():
        findings.append(
            Finding(
                "ERROR",
                path,
                None,
                None,
                "generated Python/cache artifact under `scripts/`; remove it from the source tree.",
            )
        )



    for env_var, rel_impl in DOCUMENTED_ENV_VAR_IMPLEMENTATIONS.items():
        docs_mention = False
        for path in _iter_authored_public_docs():
            try:
                if env_var in path.read_text(encoding="utf-8"):
                    docs_mention = True
                    break
            except OSError:
                continue
        if docs_mention:
            impl = REPO_ROOT / rel_impl
            try:
                impl_text = impl.read_text(encoding="utf-8")
            except OSError as e:
                findings.append(Finding("ERROR", impl, None, None, f"documented env var `{env_var}` has no readable implementation: {e}"))
                continue
            if env_var not in impl_text:
                findings.append(
                    Finding(
                        "ERROR",
                        impl,
                        None,
                        None,
                        f"documented env var `{env_var}` is not implemented in its declared producer helper.",
                    )
                )

    trust_file = REPO_ROOT / "docs/TRUST_BOUNDARIES.md"
    try:
        trust_text = trust_file.read_text(encoding="utf-8")
    except OSError as e:
        trust_text = ""
        findings.append(Finding("ERROR", trust_file, None, None, f"failed to read file: {e}"))

    for fq_name, (rel_source, decl_re) in TRUST_BOUNDARY_DECL_REFS.items():
        if fq_name not in trust_text:
            findings.append(
                Finding(
                    "ERROR",
                    trust_file,
                    None,
                    None,
                    f"trust-boundary declaration `{fq_name}` is missing from docs/TRUST_BOUNDARIES.md.",
                )
            )
        source = REPO_ROOT / rel_source
        try:
            source_text = source.read_text(encoding="utf-8")
        except OSError as e:
            findings.append(Finding("ERROR", source, None, None, f"failed to read file: {e}"))
            continue
        if not decl_re.search(source_text):
            findings.append(
                Finding(
                    "ERROR",
                    source,
                    None,
                    None,
                    f"docs/TRUST_BOUNDARIES.md cites `{fq_name}`, but the expected declaration was not found.",
                )
            )

    for path in _iter_doc_paths():
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if path.suffix != ".lean":
            _check_local_source_refs(path, text, findings)
            _check_docgen_api_links(path, text, findings)
        elif path.is_relative_to(REPO_ROOT / "home_page/blueprint/TorchLeanBlueprint"):
            _check_verso_math_roles(path, text, findings)
        for m in TORCHLEAN_SOURCE_LINK_RE.finditer(text):
            raw_target = m.group(1).split("#", 1)[0]
            target = urllib.parse.unquote(raw_target)
            if not (REPO_ROOT / target).exists():
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"dead TorchLean source link: `{raw_target}` does not exist in this checkout.",
                    )
                )

    for path in _iter_script_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        if path.suffix in {".py", ".sh"}:
            has_shebang = _has_shebang(text)
            is_executable = _is_executable(path)
            if has_shebang and not is_executable:
                findings.append(
                    Finding("ERROR", path, 1, 1, "script has a shebang but is not executable.")
                )
            if is_executable and not has_shebang:
                findings.append(
                    Finding("ERROR", path, 1, 1, "executable script should start with a shebang.")
                )

        if path.suffix == ".py" and not _has_python_module_docstring(text):
            findings.append(
                Finding("ERROR", path, 1, 1, "Python scripts/helpers should start with a module docstring.")
            )

    banned_regexes: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bnative_decide\b"), "`native_decide` is banned in TorchLean."),
        (re.compile(r"\bsorry\b"), "`sorry` is banned in TorchLean sources."),
        (re.compile(r"\badmit\b"), "`admit` is banned in TorchLean sources."),
        (re.compile(r"\bsimp\s*\[\s*\*(\s*[,\]])"), "`simp [*]` is banned; prefer `simp [h₁, h₂]` or `simp (config := ...)` with explicit hypotheses."),
        (
            re.compile(r"\bset_option\s+maxHeartbeats\b"),
            "proof-level `maxHeartbeats` overrides are not allowed; split the declaration or isolate expensive normalization behind reusable lemmas.",
        ),
        (
            re.compile(r"^\s*public\s+import\s+Mathlib\.Tactic\b", flags=re.MULTILINE),
            "Do not `public import Mathlib.Tactic.*`; import the specific tactic modules you use (non-public).",
        ),
        (
            re.compile(r"^\s*import\s+Mathlib\.Tactic(?!\.)\b", flags=re.MULTILINE),
            "Do not `import Mathlib.Tactic` (umbrella import). Import the specific `Mathlib.Tactic.*` modules you use.",
        ),
        (re.compile(r"@\[\s*de" r"precated\b"), "`@[de" "precated]` is banned in TorchLean sources."),
    ]

    for path in _iter_lean_files():
        try:
            raw = path.read_bytes()
        except OSError as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue

        # Enforce LF-only; CRLF and stray CR cause confusing diffs and occasional parser weirdness.
        if b"\r" in raw:
            findings.append(Finding("ERROR", path, None, None, "contains CR (`\\r`) characters (use LF)."))
            continue

        text = raw.decode("utf-8", errors="replace")
        masked = _mask_lean_comments_and_strings(_mask_verso_prose(text))
        rel = path.relative_to(REPO_ROOT).as_posix()
        _check_line_style(path, text, findings)
        _check_local_source_refs(path, text, findings)
        _check_lean_doc_math(path, text, findings)
        _check_backend_contract_refs(path, text, lake_text, findings)

        if rel.startswith(("NN/API/", "NN/Examples/")):
            for match in re.finditer(r"_root_\.", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public API and example code must resolve canonical namespaces without "
                        "`_root_.`; fix the namespace or import boundary instead.",
                    )
                )

        if rel.startswith("NN/Examples/Quickstart/"):
            quickstart_internals = [
                (re.compile(r"\bTensorPack\b"), "heterogeneous runtime packs"),
                (re.compile(r"\bRuntime\.Autograd\b"), "runtime autograd internals"),
                (re.compile(r"\bNN\.IR\b"), "raw compiler IR"),
                (re.compile(r"\bTensor\.ofFn\b"), "proof-level tensor construction"),
                (re.compile(r"\bList\.finRange\b"), "bounded-index proof plumbing"),
            ]
            for pattern, description in quickstart_internals:
                for match in pattern.finditer(masked):
                    line, col = _line_col(text, match.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"quickstarts must use reader-facing APIs, not {description}; "
                            "move the internal demonstration to `DeepDives` or add a public wrapper.",
                        )
                    )

        runtime_example_prefixes = (
            "NN/Examples/Quickstart/",
            "NN/Examples/Data/",
            "NN/Examples/Factorization/",
            "NN/Examples/Models/Supervised/",
            "NN/Examples/Models/Vision/",
            "NN/Examples/Models/Sequence/",
            "NN/Examples/Models/Generative/",
            "NN/Examples/Models/Operators/",
        )
        if rel.startswith(runtime_example_prefixes):
            for match in re.finditer(r"\bSpec\.", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "runtime-facing examples must call the public executable API, not `Spec`; "
                        "move proof-only material to a proof example or add a public operation.",
                    )
                )

        if rel.startswith("NN/Examples/"):
            for match in re.finditer(r"\b(?:nn\.)?State\.Internal\b", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "examples must use the public opaque `nn.State` and public execution or "
                        "verification APIs, not unpack its internal tensor representation.",
                    )
                )

        if rel.startswith("NN/Examples/Models/"):
            prefixed_name_re = re.compile(
                r'\bdef\s+exeName\s*:\s*String\s*:=\s*"torchlean(?:\s|")'
            )
            for match in prefixed_name_re.finditer(text):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "model examples store only their CLI subcommand in `exeName`; help text "
                        "adds `lake exe torchlean` at the rendering boundary.",
                    )
                )


        if rel.startswith("NN/Examples/Models/") or rel == "NN/API/RL/Cli.lean":
            for match in re.finditer(r"\.drop\s+10\b", masked):
                line, col = _line_col(text, match.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "do not recover a CLI subcommand by stripping a fixed prefix; store the "
                        "subcommand directly.",
                    )
                )


        import_directives: dict[str, int] = {}
        # Lean module imports form one contiguous block at the start of a file. Restrict the
        # check to that block so `import ...` lines in Verso code examples are not mistaken for
        # dependencies of the documentation module itself.
        import_header = _lean_import_header(masked)
        for match in LEAN_IMPORT_RE.finditer(import_header):
            directive = " ".join(match.group("directive").split())
            module_name = match.group("module")
            line, col = _line_col(text, match.start())
            if rel.startswith("NN/CI/") and match.group("public"):
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "CI import targets compile dependencies but must not re-export them; "
                        "use a private `import`.",
                    )
                )
            previous_line = import_directives.get(directive)
            if previous_line is None:
                import_directives[directive] = line
            else:
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"duplicate import `{module_name}`; it was already imported on line "
                        f"{previous_line}.",
                    )
                )

        if rel.startswith("NN/API") and TOP_LEVEL_API_DECL_RE.search(masked) is None:
            for match in LEAN_IMPORT_RE.finditer(import_header):
                if not match.group("public"):
                    continue
                module_name = match.group("module")
                if (
                    module_name in BROAD_LOW_LEVEL_IMPORTS
                    or module_name.startswith(BROAD_LOW_LEVEL_IMPORT_PREFIXES)
                ):
                    line, col = _line_col(text, match.start("module"))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"import-only API umbrella re-exports low-level module `{module_name}`; "
                            "export a focused API module instead.",
                        )
                    )

        ownership_sensitive_cuda_modules = {
            "NN/Runtime/Autograd/Engine/LibTorch/Buffer.lean",
            "NN/Runtime/Autograd/Engine/LibTorch/Kernels.lean",
            "NN/Runtime/Autograd/Engine/LibTorch/ConvPool.lean",
        }
        if rel in ownership_sensitive_cuda_modules:
            unsafe_extern = re.compile(r"@\[(?![^\]]*\bnever_extract\b)[^\]]*\bextern\b[^\]]*\]")
            for m in unsafe_extern.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "CUDA buffer externs must use `never_extract`; these calls allocate, "
                        "observe, or mutate native resources and may not be commoned or deleted.",
                    )
                )

        if not _has_nn_header(path, text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing TorchLean header in the first ~10 lines (expected `Copyright (c) 2026 TorchLean`).",
                )
            )

        if path.is_relative_to(REPO_ROOT / "NN") and not _has_lean_module_docstring(text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing Lean module docstring (`/-! ... -/`); add purpose, main declarations, and import guidance.",
                )
            )

        # Line-level whitespace hygiene: Lean rejects tabs, and trailing whitespace is noisy in reviews.
        for i, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                col = line.find("\t") + 1
                findings.append(Finding("ERROR", path, i, col, "tab character found (use spaces)."))
            if line.endswith(" "):
                findings.append(Finding("ERROR", path, i, len(line), "trailing whitespace."))

        if rel.startswith("NN/"):
            fixed_vector_re = re.compile(r"\b(?:List\.)?Vector\b|#v\[")
            if (
                not rel.startswith(TENSOR_INTERNAL_PREFIX)
                and rel not in TENSOR_VECTOR_BOUNDARY_FILES
            ):
                for m in fixed_vector_re.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            "fixed-shape numerical data must use `TorchLean.Tensor`; use `Array` for "
                            "dynamic homogeneous storage.",
                        )
                    )


            dynamic_numeric_list_re = re.compile(
                r"\bList\s+(?:Float|Rat|Int|Bool|UInt8|UInt16|UInt32|UInt64)\b|"
                r"\bList\s*\(\s*(?:Probe|Sample\.Supervised|LinParams)\b|"
                r"\bList\s*\(\s*FlatAffine\b|"
                r"\bList\s+PinnLayer\b|"
                r"\bIO\.Ref\s*\(\s*List\s+Nat\s*\)|"
                r"\bList\s+NN\.Backend\.(?:AcceptedKernel|Provider)\b|"
                r"\bList\s*\(\s*NN\.Backend\.KernelHandler\b|"
                r"\bhiddenDims\s*:\s*List\s+Nat\b"
            )
            if not rel.startswith(TENSOR_INTERNAL_PREFIX):
                for m in dynamic_numeric_list_re.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            "dynamic homogeneous numerical collections must use `Array`; reserve "
                            "`List` for type-level or proof-recursive structure.",
                        )
                    )

        for rx, msg in banned_regexes:
            for m in rx.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))

        # Keep the FloatLib adapters below TorchLean's spec, proof, runtime, and verification
        # layers. Their TorchLean imports are restricted to fellow adapters and core definitions.
        if rel == "NN/Floats.lean" or rel.startswith("NN/Floats/"):
            for m in re.finditer(
                r"^\s*(?:public\s+)?import\s+(NN\.[A-Za-z0-9_.]+)\s*$",
                masked,
                flags=re.MULTILINE,
            ):
                imported = m.group(1)
                if not (imported == "NN.Floats" or imported.startswith("NN.Floats.")
                        or imported == "NN.Core" or imported.startswith("NN.Core.")):
                    line, col = _line_col(text, m.start(1))
                    findings.append(
                        Finding(
                            "ERROR",
                            path,
                            line,
                            col,
                            f"floating-point core imports `{imported}`; move this integration to the spec, proof, runtime, or verification layer.",
                        )
                    )

        is_shape_generic_public_api = (
            rel == "NN/API/Tensor.lean"
            or rel.startswith("NN/API/Models/")
            or rel.startswith("NN/API/Neural/")
        )
        if is_shape_generic_public_api:
            for m in re.finditer(r"\bPNat\b", masked):
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public tensor and model APIs use ordinary `Nat` dimensions and validate "
                        "positivity at construction time; do not expose proof-carrying `PNat` "
                        "configuration fields.",
                    )
                )
            for m in re.finditer(r"\bList\s+Nat\b", masked):
                line_start = masked.rfind("\n", 0, m.start()) + 1
                binder = masked[line_start:m.start()]
                if re.search(r"\b(?:hiddenWidths|modelWidths)\s*:\s*$", binder):
                    continue
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        "public tensor and model geometry must use `Spec.Shape` for static "
                        "shape indices or `Tensor Nat [d]` for computed geometry, not `List Nat`; "
                        "ordinary lists are reserved for explicitly named recursive architecture "
                        "plans named `hiddenWidths` or `modelWidths`.",
                    )
                )

        if rel == "NN/API/Trainer/Train.lean":
            m = TOP_LEVEL_API_DECL_RE.search(masked)
            if m:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                            "`NN.API.Trainer.Train` must stay an import-only aggregator; put training implementation in `NN.API.Trainer.Train.*` modules.",
                    )
                )

        if rel == "NN/API/Neural.lean" and re.search(
            r"^\s*public\s+import\s+NN\.API\.Trainer\s*$", masked, flags=re.MULTILINE
        ):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Neural` must not re-export `NN.API.Trainer`; use `TorchLean.Trainer` for ordinary code and import the advanced training module explicitly when needed.",
                )
            )
        is_trainer_api = rel == "NN/API/Trainer.lean" or rel.startswith("NN/API/Trainer/")
        is_training_entrypoint = rel in {"NN/API.lean", "NN/API/Data/Training.lean"}
        if re.search(
            r"^\s*public\s+import\s+NN\.API\.Trainer\s*$", masked, flags=re.MULTILINE
        ) and not (is_trainer_api or is_training_entrypoint):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    None,
                    None,
                    "`NN.API.Trainer` should only be imported by the Trainer API; keep the callback-heavy training layer out of broad application API imports.",
                )
            )

        if rel.endswith(".lean") and any(
            rel.startswith(prefix) for prefix in PUBLIC_NUMERICAL_EXAMPLE_PREFIXES
        ):
            for rx, msg in PUBLIC_NUMERICAL_SPEC_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        # Axioms must be quarantined and named explicitly.
        allowed_axiom_names = ALLOWED_AXIOMS.get(rel, set())
        for m in AXIOM_DECL_RE.finditer(masked):
            axiom_name = m.group(1)
            if axiom_name not in allowed_axiom_names:
                line, col = _line_col(text, m.start())
                findings.append(
                    Finding(
                        "ERROR",
                        path,
                        line,
                        col,
                        f"axiom `{axiom_name}` is not allowlisted; quarantine and document trusted axioms.",
                    )
                )

        # Warning and visibility checks are part of the build contract. Do not hide them locally:
        # fix the declaration or proof that emits the diagnostic.
        suppressed_linter_re = re.compile(
            r"set_option\s+linter\.([A-Za-z0-9_]+)\s+false(?:\s+in)?"
        )
        for m in suppressed_linter_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"suppresses Lean linter `{m.group(1)}`; fix the diagnostic instead.",
                )
            )

        nolint_re = re.compile(
            r"(?:@\[\s*|attribute\s+\[\s*)nolint\b"
        )
        for m in nolint_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "suppresses a Lean linter with `nolint`; fix the diagnostic instead.",
                )
            )

        private_compat_re = re.compile(r"\bbackward\.privateInPublic(?:\.warn)?\b")
        for m in private_compat_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "overrides Lean's strict private-in-public boundary; remove the override.",
                )
            )

        hidden_warning_re = re.compile(
            r"(?:set_option\s+warningAsError\s+false|"
            r"⟨\s*`warningAsError\s*,\s*false\s*⟩)"
        )
        for m in hidden_warning_re.finditer(masked):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    "disables warnings-as-errors; keep compiler warnings visible and fix them.",
                )
            )

    if not fail_on_warn:
        return findings
    # Promote warnings to errors.
    return [
        Finding("ERROR" if f.level == "WARN" else f.level, f.path, f.line, f.col, f.message)
        for f in findings
    ]


def main() -> int:
    """CLI entry point used by local checks and CI."""
    ap = argparse.ArgumentParser(description="TorchLean repo lints (project policies).")
    ap.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Treat warnings as errors (useful for tightening policies over time).",
    )
    args = ap.parse_args()

    findings = lint_repo(fail_on_warn=args.fail_on_warn)
    errors = [f for f in findings if f.level == "ERROR"]
    warns = [f for f in findings if f.level == "WARN"]

    for f in findings:
        print(f.render())

    if errors:
        print(f"\nFAILED: {len(errors)} error(s), {len(warns)} warning(s).")
        return 1

    if warns:
        print(f"\nOK (with warnings): {len(warns)} warning(s).")
        return 0

    print("OK: no issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
