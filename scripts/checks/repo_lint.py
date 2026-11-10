#!/usr/bin/env python3
"""
TorchLean repo lints (project-specific).

The linter stays dependency-free so it can run in CI and locally.

Every finding is an error: the command prints each one and exits with status 1 if there is any.
"""

from __future__ import annotations

import argparse
import ast
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
LINT_SCOPE_SENTINEL = REPO_ROOT / "TorchLean/Verify/CROWN.lean"

# Prune these before descent, since filtering the yielded paths would walk everything below them.
SOURCE_EXCLUDED_DIR_NAMES = {
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
    "home_page/docs",
    "home_page/blueprint/print",
    "home_page/blueprint/web",
}

# Axioms must be quarantined, named and documented, so that the trusted boundary stays visible.
# TorchLean has no custom axioms.
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
# exists in source, so guide text cannot advertise phantom integration flags.
DOCUMENTED_ENV_VAR_IMPLEMENTATIONS = {
    "ABCROWN_ARTIFACT_OUT": "scripts/verification/abcrown/export_leaf_artifact.py",
}

TORCHLEAN_SOURCE_LINK_RE = re.compile(
    r"https://github\.com/lean-dojo/TorchLean/blob/main/([^\s\)\]`]+)"
)

DOCGEN_API_LINK_RE = re.compile(
    r"""(?:['"(])(/docs/[A-Za-z0-9_./-]+\.html(?:#[A-Za-z0-9_'.:-]+)?)"""
)

LOCAL_SOURCE_REF_RE = re.compile(
    r"`((?:TorchLean|TorchLeanExamples|TorchLeanTest|python|blueprint|home_page|scripts|csrc)"
    r"/[^`\s]+"
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

NUMERICAL_EXAMPLE_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
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

# The tensor compiler's internal language uses Lean vectors for compiler indices and
# proof-recursive lists for syntax. The public numerical-container policies apply everywhere else.
TENSOR_COMPILER_PATHS = (
    "TorchLean/Tensor/Einops/",
    "TorchLean/Tensor/Native/",
    "TorchLean/Tensor/Rep/",
    "TorchLean/Tensor/Literal.lean",
    "TorchLean/Tensor/Promotion.lean",
    "TorchLean/Tensor/Rep.lean",
    "TorchLean/Tensor/Shape/Coord.lean",
    "TorchLean/Tensor/Storage.lean",
)
# Public API and example code resolve canonical namespaces without `_root_.`.
PUBLIC_API_PATHS = (
    "TorchLean/NN/Arithmetic.lean",
    "TorchLean/NN/Autograd/",
    "TorchLean/NN/Builder.lean",
    "TorchLean/NN/Builder/",
    "TorchLean/NN/CLI/",
    "TorchLean/NN/Checkpoint.lean",
    "TorchLean/NN/Data/CausalLM.lean",
    "TorchLean/NN/Data/Dataset.lean",
    "TorchLean/NN/Data/Image.lean",
    "TorchLean/NN/Data/Loader.lean",
    "TorchLean/NN/Data/Source.lean",
    "TorchLean/NN/Data/Synthetic.lean",
    "TorchLean/NN/LoRA.lean",
    "TorchLean/NN/Models/CNN.lean",
    "TorchLean/NN/Models/CausalTransformer.lean",
    "TorchLean/NN/Models/CausalTransformer/",
    "TorchLean/NN/Models/Diffusion/NoisePredictor.lean",
    "TorchLean/NN/Models/FNO.lean",
    "TorchLean/NN/Models/Generative.lean",
    "TorchLean/NN/Models/KAN.lean",
    "TorchLean/NN/Models/Mamba.lean",
    "TorchLean/NN/Models/MaskedPatchReconstructor.lean",
    "TorchLean/NN/Models/PPO.lean",
    "TorchLean/NN/Models/Recurrent.lean",
    "TorchLean/NN/Models/ResNet.lean",
    "TorchLean/NN/Models/UNet.lean",
    "TorchLean/NN/Models/ViT.lean",
    "TorchLean/NN/Module.lean",
    "TorchLean/NN/Module/",
    "TorchLean/NN/Optimizer/",
    "TorchLean/NN/RL/CLI.lean",
    "TorchLean/NN/RL/PPO/",
    "TorchLean/NN/Random.lean",
    "TorchLean/NN/Sample.lean",
    "TorchLean/NN/SelfSupervised.lean",
    "TorchLean/NN/Sequential/",
    "TorchLean/NN/Session/",
    "TorchLean/NN/Text/",
    "TorchLean/NN/Trainer.lean",
    "TorchLean/NN/Trainer/",
    "TorchLean/NN/Training/",
    "TorchLean/Verify/Lowering.lean",
    "TorchLean/Verify/Report.lean",
    "TorchLean/Verify/Trainer.lean",
    "TorchLeanExamples/",
)
TENSOR_VECTOR_BOUNDARY_FILES = {
    "TorchLean/Tensor/Conversion.lean",
}


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

CONTRACT_GUARD_SOURCE_PATH_RE = re.compile(
    r"\.runtimeGuard\s+\"[^\"]*\.(?:c|cc|cpp|cu|cuh|h|hpp)\"",
)


# The source roots: the four libraries under `TorchLean/`, the examples and the tests.
SOURCE_ROOTS = ("TorchLean", "TorchLeanExamples", "TorchLeanTest")

# Path-scoped rules name files and directories under these roots. A rule whose path matches no
# file passes vacuously, so `_check_rule_paths` fails on such a path.
RULE_PATH_RE = re.compile(
    r"(?:TorchLean|TorchLeanExamples|TorchLeanTest|python)(?:/[^\s\"'`*?\[\]]*)?"
)

# `{src "path"}` and `{srcDir "path"}` (TorchLeanBlueprint/Roles.lean) link a file or a directory.
VERSO_SOURCE_ROLE_RE = re.compile(r'\{(src|srcDir)[ \t]+"((?:[^"\\\n]|\\.)*)"[ \t]*\}')


@dataclass(frozen=True)
class Finding:
    """One repository-lint error."""

    level: str  # always "ERROR", the prefix `render` prints
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
    yield from _iter_project_files("*.lean")


def _iter_project_files(pattern: str) -> Iterable[pathlib.Path]:
    """Yield tracked and non-ignored project files matching a Git pathspec."""
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        pattern,
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
        if "_out" in relative.parts:
            continue
        path = REPO_ROOT / relative
        if path.is_file():
            yield path

# FloatLib owns the shared numerical library. The TorchLean adapters follow the same
# source-style rules as the rest of this repository.
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


def _in_source_roots(path: pathlib.Path) -> bool:
    """Check whether `path` lies under one of the source roots."""

    return any(path.is_relative_to(REPO_ROOT / top) for top in SOURCE_ROOTS)


def _check_rule_paths(findings: list[Finding]) -> None:
    """Fail on a path in this file's rules (a string literal under `TorchLean/`,
    `TorchLeanExamples/`, `TorchLeanTest/` or `python/`) that matches no project file, so a rule
    cannot pass vacuously after files move."""

    source = pathlib.Path(__file__).resolve()
    tree = ast.parse(source.read_text(encoding="utf-8"))
    files = [path.relative_to(REPO_ROOT).as_posix() for path in _iter_project_files("*")]
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
            continue
        if RULE_PATH_RE.fullmatch(node.value) and not any(
            rel.startswith(node.value) for rel in files
        ):
            findings.append(
                Finding(
                    "ERROR",
                    source,
                    node.lineno,
                    node.col_offset + 1,
                    f"rule path `{node.value}` matches no file; update the rule or delete it.",
                )
            )


def _check_verso_source_roles(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Check that every `{src "…"}` names a file and every `{srcDir "…"}` a directory."""

    for m in VERSO_SOURCE_ROLE_RE.finditer(text):
        role, target = m.group(1), m.group(2)
        kind = "file" if role == "src" else "directory"
        resolved = REPO_ROOT / target
        if not (resolved.is_file() if role == "src" else resolved.is_dir()):
            line, col = _line_col(text, m.start())
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    line,
                    col,
                    f"dead `{{{role}}}` link: `{target}` is not a {kind} in this checkout.",
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
    for relative in (
        "home_page", "TorchLean", "TorchLeanExamples", "TorchLeanTest", "python",
        "scripts",
    ):
        yield from _iter_source_files(REPO_ROOT / relative, ".md")


def _iter_generated_script_artifacts() -> Iterable[pathlib.Path]:
    """Yield generated files under `scripts/` (`__pycache__`, `.pyc`, `.pyo`, `.DS_Store`)."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or p.name == ".DS_Store":
            yield p


def _iter_script_files() -> Iterable[pathlib.Path]:
    """Yield every file on disk under `scripts/`, tracked or not."""

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


TORCHLEAN_HEADER = (
    "Copyright (c) 2026 TorchLean",
    "Released under MIT license as described in the file LICENSE.",
    "Authors: TorchLean Team",
)


def _has_torchlean_header(path: pathlib.Path, text: str) -> bool:
    """Check whether a source file under `TorchLean/`, `TorchLeanExamples/` or
    `TorchLeanTest/` carries the standard TorchLean header."""
    # TorchLean policy: library, example and test sources carry a consistent header at the top
    # of the file.
    if not _in_source_roots(path):
        return True
    head = text.splitlines()[:10]
    return all(line in head for line in TORCHLEAN_HEADER)


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
        # Inline Lean roles also elaborate terms, so keep them for the banned-construct checks.
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
    """Catch mathematical TeX that would render as an ordinary monospace code span."""

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


def _check_backend_contract_refs(
    path: pathlib.Path,
    text: str,
    findings: list[Finding],
) -> None:
    """Reject native source paths used as backend runtime-guard labels."""

    for m in CONTRACT_GUARD_SOURCE_PATH_RE.finditer(text):
        line, col = _line_col(text, m.start())
        findings.append(
            Finding(
                "ERROR",
                path,
                line,
                col,
                "a runtime-guard label names the guard, not a native source path.",
            )
        )


def _check_python_native_decide(path: pathlib.Path, text: str, findings: list[Finding]) -> None:
    """Reject `native_decide` in a Python source, including in the Lean proofs it writes."""

    for m in re.finditer(r"\bnative_decide\b", text):
        line, col = _line_col(text, m.start())
        findings.append(
            Finding(
                "ERROR",
                path,
                line,
                col,
                "`native_decide` is banned in TorchLean, including in generated Lean proofs.",
            )
        )


def lint_repo() -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []
    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see "
                "TorchLean/Verify/CROWN.lean.",
            )
        )

    _check_rule_paths(findings)

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
            _check_verso_source_roles(path, text, findings)
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

        # Enforce LF-only line endings. CRLF and stray CR cause confusing diffs and occasional
        # parser weirdness.
        if b"\r" in raw:
            findings.append(Finding("ERROR", path, None, None, "contains CR (`\\r`) characters (use LF)."))
            continue

        text = raw.decode("utf-8", errors="replace")
        masked = _mask_lean_comments_and_strings(_mask_verso_prose(text))
        rel = path.relative_to(REPO_ROOT).as_posix()
        _check_line_style(path, text, findings)
        _check_local_source_refs(path, text, findings)
        _check_lean_doc_math(path, text, findings)
        _check_backend_contract_refs(path, text, findings)

        if rel.startswith(PUBLIC_API_PATHS):
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

        if rel.startswith("TorchLeanExamples/Quickstart/"):
            quickstart_internals = [
                (re.compile(r"(?<![\w.'])(?:TorchLean\.)?Pack\b"),
                 "heterogeneous runtime packs"),
                (re.compile(r"(?<![\w.'])(?:TorchLean\.)?(?:Tape|Dual|TypedGraph)\b"),
                 "runtime autograd internals"),
                (re.compile(r"(?<![\w.'])(?:TorchLean\.)?IR\b"), "raw compiler IR"),
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
            "TorchLeanExamples/Quickstart/",
            "TorchLeanExamples/Data/",
            "TorchLeanExamples/Models/Supervised/",
            "TorchLeanExamples/Models/Vision/",
            "TorchLeanExamples/Models/Sequence/",
            "TorchLeanExamples/Models/Generative/",
            "TorchLeanExamples/Models/Operators/",
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

        if rel.startswith("TorchLeanExamples/Models/"):
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


        if rel.startswith("TorchLeanExamples/Models/") or rel == "TorchLean/NN/RL/CLI.lean":
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

        ownership_sensitive_cuda_modules = {
            "TorchLean/CUDA/LibTorch/Buffer.lean",
            "TorchLean/CUDA/LibTorch/Kernels.lean",
            "TorchLean/CUDA/LibTorch/ConvPool.lean",
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

        if not _has_torchlean_header(path, text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing TorchLean header in the first 10 lines (expected the lines `Copyright (c) 2026 TorchLean`, "
                    "`Released under MIT license as described in the file LICENSE.` and `Authors: TorchLean Team`).",
                )
            )

        if _in_source_roots(path) and not _has_lean_module_docstring(text):
            findings.append(
                Finding(
                    "ERROR",
                    path,
                    1,
                    1,
                    "missing Lean module docstring (`/-! ... -/`); add purpose, main declarations, and import guidance.",
                )
            )

        # Line-level whitespace hygiene: Lean rejects tabs, and trailing whitespace is diff noise.
        for i, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                col = line.find("\t") + 1
                findings.append(Finding("ERROR", path, i, col, "tab character found (use spaces)."))
            if line.endswith(" "):
                findings.append(Finding("ERROR", path, i, len(line), "trailing whitespace."))

        if _in_source_roots(path):
            fixed_vector_re = re.compile(r"\b(?:List\.)?Vector\b|#v\[")
            if (
                not rel.startswith(TENSOR_COMPILER_PATHS)
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
                r"\bList\s*\(\s*(?:Probe|Sample\.Supervised|SomeLinear)\b|"
                r"\bIO\.Ref\s*\(\s*List\s+Nat\s*\)|"
                r"\bList\s+(?:TorchLean\.)?Backend\.(?:AcceptedKernel|Provider)\b|"
                r"\bhiddenDims\s*:\s*List\s+Nat\b"
            )
            if not rel.startswith(TENSOR_COMPILER_PATHS):
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

        is_shape_generic_public_api = (
            rel.startswith(("TorchLean/NN/Models/", "TorchLean/NN/Builder/"))
            or rel.startswith("TorchLean/NN/Sequential/")
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
                        "public tensor and model geometry must use `Shape` for static "
                        "shape indices or `Tensor Nat [d]` for computed geometry, not `List Nat`; "
                        "ordinary lists are reserved for explicitly named recursive architecture "
                        "plans named `hiddenWidths` or `modelWidths`.",
                    )
                )

        if rel.endswith(".lean") and rel.startswith("TorchLeanExamples/"):
            for rx, msg in NUMERICAL_EXAMPLE_BANNED_PATTERNS:
                for m in rx.finditer(masked):
                    line, col = _line_col(text, m.start())
                    findings.append(Finding("ERROR", path, line, col, msg))

        # Axioms must be quarantined and listed by name in `ALLOWED_AXIOMS`.
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

    # Python generators write Lean proofs that Lean checks later, so the same ban covers them.
    for path in _iter_project_files("*.py"):
        if path == pathlib.Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            findings.append(Finding("ERROR", path, None, None, f"failed to read file: {e}"))
            continue
        _check_python_native_decide(path, text, findings)

    return findings


def main() -> int:
    """Print every finding of `lint_repo`. Return 1 if there is any, else 0."""
    ap = argparse.ArgumentParser(description="TorchLean repo lints (project policies).")
    ap.parse_args()

    findings = lint_repo()
    for f in findings:
        print(f.render())

    if findings:
        print(f"\nFAILED: {len(findings)} error(s).")
        return 1

    print("OK: no issues found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
