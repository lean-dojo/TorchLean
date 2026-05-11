#!/usr/bin/env python3
"""
Run `leanprover/comparator` against a Lake project.

TorchLean vendors Comparator as a Lake dependency to make Comparator runs reproducible against the
same Lean toolchain as TorchLean.

Security note:
  Only the final `comparator` invocation is sandboxed by `landrun`. Running `lake update` or
  `lake build` inside an arbitrary `--project` executes that project's `lakefile` and dependency
  code. For untrusted projects, skip update/build (this script does so by default unless the
  project is TorchLean itself) or run inside a dedicated container.

Prerequisites (external):
  - `landrun` in PATH: https://github.com/Zouuup/landrun

What this script does:
  1. (optionally) `lake update` (trusted projects only)
  2. (optionally) build `lean4export` and `comparator` *dependency executables* (trusted projects only)
  3. run comparator via `lake env`, with PATH extended so it can find `lean4export` (sandboxed via landrun)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _die(msg: str, *, code: int = 2) -> None:
    """Print a consistent fatal error message and exit with the chosen code."""
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _run(cmd: list[str], *, cwd: pathlib.Path, env: dict[str, str] | None = None) -> None:
    """Run a subprocess in a selected Lake project, raising on failure."""
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> int:
    """Validate trust settings, build required tools if allowed, then run Comparator."""
    ap = argparse.ArgumentParser(description="Run Comparator against a Lake project.")
    ap.add_argument(
        "config",
        type=pathlib.Path,
        help="Path to comparator JSON config (see https://github.com/leanprover/comparator).",
    )
    ap.add_argument(
        "--project",
        type=pathlib.Path,
        default=REPO_ROOT,
        help="Lake project root to run comparator in (default: TorchLean repo root).",
    )
    ap.add_argument(
        "--trusted-project",
        action="store_true",
        help=(
            "Allow running `lake update`/`lake build` inside --project even when it is not TorchLean. "
            "WARNING: this executes arbitrary code from that project."
        ),
    )
    ap.add_argument("--no-update", action="store_true", help="Skip `lake update`.")
    ap.add_argument("--no-build", action="store_true", help="Skip building comparator/lean4export executables.")
    args = ap.parse_args()

    landrun = shutil.which("landrun")
    if landrun is None:
        _die("`landrun` not found in PATH. Install it from https://github.com/Zouuup/landrun.")
    try:
        subprocess.run([landrun, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except Exception as e:
        _die(f"`landrun` was found but could not be executed ({e}). Check GLIBC compatibility or rebuild from source.")

    # Comparator uses `landrun --best-effort`; on kernels without Landlock support this may degrade
    # to a much weaker sandbox (or effectively no sandbox). Warn loudly so users don't rely on it
    # for adversarial `Solution.lean` checks on older kernels.
    rel = platform.release()
    try:
        major_s, minor_s, *_ = rel.split(".")
        major = int(major_s)
        minor = int(minor_s)
        if (major, minor) < (5, 13):
            print(
                f"WARNING: kernel {rel} is older than 5.13; Landlock may be unavailable. "
                "Comparator will run with `--best-effort` and sandboxing may be significantly weaker.",
                file=sys.stderr,
            )
    except Exception:
        # If parsing fails, keep going; this is only a warning.
        pass

    project = args.project.resolve()
    config = args.config.resolve()
    if not config.is_file():
        _die(f"config not found: {config}")
    if not (project / "lakefile.lean").exists() and not (project / "lakefile.toml").exists():
        _die(f"--project does not look like a Lake project (missing lakefile.*): {project}")

    # Safety: building arbitrary Lake projects executes code (lakefile + dependencies).
    # Restrict update/build to TorchLean by default; require explicit opt-in for other projects.
    if project != REPO_ROOT and not args.trusted_project:
        args.no_update = True
        args.no_build = True
        print(
            "NOTE: --project is not TorchLean and --trusted-project was not set; "
            "skipping `lake update` and `lake build` for safety.",
            file=sys.stderr,
        )

    if not args.no_update:
        _run(["lake", "update"], cwd=project)

    # Build dependency executables in the selected Lake project workspace.
    if not args.no_build:
        _run(["lake", "build", "@lean4export/lean4export"], cwd=project)
        _run(["lake", "build", "@Comparator/comparator"], cwd=project)

    comparator_bin = project / ".lake" / "packages" / "Comparator" / ".lake" / "build" / "bin" / "comparator"
    if not comparator_bin.exists():
        _die(f"comparator binary not found (did the build succeed?): {comparator_bin}")

    lean4export_bin_dir = project / ".lake" / "packages" / "lean4export" / ".lake" / "build" / "bin"
    if not (lean4export_bin_dir / "lean4export").exists():
        _die(f"lean4export binary not found (did the build succeed?): {lean4export_bin_dir / 'lean4export'}")

    env = os.environ.copy()
    env["PATH"] = str(lean4export_bin_dir) + os.pathsep + env.get("PATH", "")

    # Comparator's README recommends `lake env path/to/comparator/binary path/to/config.json`.
    _run(["lake", "env", str(comparator_bin), str(config)], cwd=project, env=env)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
