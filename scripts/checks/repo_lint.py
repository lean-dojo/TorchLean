#!/usr/bin/env python3
"""
TorchLean repo lints (project-specific).

This linter stays lightweight and dependency-free so it can run in CI and locally.

Checks are split into:
  - errors: must be fixed (fail CI)
  - warnings: reported for visibility (do not fail by default)
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
LINT_SCOPE_SENTINEL = REPO_ROOT / "NN/MLTheory/CROWN/Lyapunov/Oracle.lean"

# External trees that may exist in a developer checkout but are not part of TorchLean's core sources
# and must not affect repo policy/CI. These are user-cloned repos outside TorchLean's source tree.
#
# Important: `Path.rglob` walks the filesystem, not git-tracked files, so this linter explicitly skips
# these directories to ensure `lake lint` does not start depending on optional checkouts.
VENDORED_DIR_NAMES = {
    "Two-Stage_Neural_Controller_Training",  # optional external checkout (α,β-CROWN workflows)
    "PINN_verification",  # user-cloned external repo (gitignored)
}

# Keep the trusted boundary explicit: axioms must be quarantined, named, and documented.
ALLOWED_AXIOMS = {
    "NN/MLTheory/CROWN/Lyapunov/Oracle.lean": {"crown_oracle"},
    "NN/Runtime/Autograd/Engine/Cuda/Trusted.lean": {"instNonemptyBuffer"},
}

# Narrow allowlist for linter suppressions that are noisy in facade files but do
# not weaken proofs. Keep this list small and review each addition.
ALLOWED_LINTER_SUPPRESSION_FILES = {
    "NN/Tensor/API.lean",
}


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
    """Yield project Lean files while skipping vendored and generated trees."""
    for p in REPO_ROOT.rglob("*.lean"):
        # Never lint vendored dependencies.
        if ".lake" in p.parts:
            continue
        if any(d in p.parts for d in VENDORED_DIR_NAMES):
            continue
        # `_out` can contain generated artifacts; keep policy focused on sources.
        if "_out" in p.parts:
            continue
        yield p


def _iter_generated_script_artifacts() -> Iterable[pathlib.Path]:
    """Generated files that should never live under the checked-in `scripts/` tree."""

    scripts_dir = REPO_ROOT / "scripts"
    if not scripts_dir.exists():
        return
    for p in scripts_dir.rglob("*"):
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"} or p.name == ".DS_Store":
            yield p


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


def _mask_lean_comments_and_strings(text: str) -> str:
    """
    Return a same-length string where Lean comments/docstrings and string literals are replaced
    with spaces (newlines preserved).

    This prevents repo-lint regexes like `\\bsorry\\b` from triggering on policy mentions in
    docstrings/comments (and avoids false positives in string literals).

    Notes:
      - Lean block comments `/- ... -/` nest; the scanner tracks nesting depth.
      - The scanner remains lightweight (no full parser), but it avoids treating comment
        markers inside strings as comments.
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

        if ch == '"':
            out[i] = " "
            in_string = True
            i += 1
            continue

        i += 1

    return "".join(out)


def lint_repo(*, fail_on_warn: bool) -> list[Finding]:
    """Run TorchLean's repository hygiene checks and return all findings."""
    findings: list[Finding] = []

    if not LINT_SCOPE_SENTINEL.exists():
        findings.append(
            Finding(
                "ERROR",
                LINT_SCOPE_SENTINEL,
                None,
                None,
                "repo linter is not rooted at TorchLean; expected to see NN/MLTheory/CROWN/Lyapunov/Oracle.lean.",
            )
        )

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

    banned_regexes: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"\bnative_decide\b"), "`native_decide` is banned in TorchLean."),
        (re.compile(r"\bsorry\b"), "`sorry` is banned in TorchLean sources."),
        (re.compile(r"\badmit\b"), "`admit` is banned in TorchLean sources."),
        (re.compile(r"\bby\s+omega\b"), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"^\s*omega\b", flags=re.MULTILINE), "`omega` is banned in TorchLean; prefer `linarith`/`nlinarith`/`grind` or small arithmetic lemmas."),
        (re.compile(r"\bsimp\s*\[\s*\*(\s*[,\]])"), "`simp [*]` is banned; prefer `simp [h₁, h₂]` or `simp (config := ...)` with explicit hypotheses."),
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

    axiom_re = re.compile(r"^\s*axiom\s+([A-Za-z0-9_'.]+)\b", flags=re.MULTILINE)

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
        masked = _mask_lean_comments_and_strings(text)

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

        # Line-level whitespace hygiene: Lean rejects tabs, and trailing whitespace is noisy in reviews.
        for i, line in enumerate(text.splitlines(), start=1):
            if "\t" in line:
                col = line.find("\t") + 1
                findings.append(Finding("ERROR", path, i, col, "tab character found (use spaces)."))
            if line.endswith(" "):
                findings.append(Finding("ERROR", path, i, len(line), "trailing whitespace."))

        for rx, msg in banned_regexes:
            for m in rx.finditer(masked):
                line, col = _line_col(text, m.start())
                findings.append(Finding("ERROR", path, line, col, msg))

        # Axioms must be quarantined and named explicitly.
        rel = path.relative_to(REPO_ROOT).as_posix()
        allowed_axiom_names = ALLOWED_AXIOMS.get(rel, set())
        for m in axiom_re.finditer(masked):
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

        # Warn by default; callers can promote these warnings with `--fail-on-warn`.
        if "set_option linter." in masked and " false" in masked:
            # Only a coarse signal; report once per file.
            if re.search(r"set_option\s+linter\.[A-Za-z0-9_]+\s+false", masked):
                rel_posix = path.relative_to(REPO_ROOT).as_posix()
                # Examples/tests/tools are allowed to be a bit rough; keep the signal focused on
                # library-facing code.
                if (
                    rel_posix in ALLOWED_LINTER_SUPPRESSION_FILES
                    or
                    rel_posix.startswith("NN/Examples/")
                    or rel_posix.startswith("NN/Tests/")
                    or rel_posix.startswith("scripts/")
                    # The compiled correctness proofs scope linter options locally to
                    # keep proof scripts readable; warning here is usually not actionable.
                    or rel_posix.startswith("NN/Runtime/Autograd/Compiled/IRExec/Correctness/")
                ):
                    pass
                else:
                    findings.append(
                        Finding(
                            "WARN",
                            path,
                            None,
                            None,
                            "suppresses a linter (`set_option linter.* false`). Prefer fixing the warning or scoping the option tightly.",
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
