#!/usr/bin/env python3
"""Export TorchLean alpha-beta-CROWN-style leaf artifacts.

This script is the producer-side bridge for TorchLean's
`abcrown_leaf_artifact_v0_1` checker.  It does not run alpha-beta-CROWN.  Instead,
it converts a small raw leaf/domain dump from an external verifier into the
JSON schema consumed by:

  lake exe verify -- abcrown-leaf <artifact.json>

For external integrations, import `write_abcrown_leaf_artifact` and pass the root
box plus terminal leaves.  If `out_path` is omitted, the helper writes to the
`ABCROWN_ARTIFACT_OUT` environment variable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Sequence


FORMAT = "abcrown_leaf_artifact_v0_1"
ENV_OUT = "ABCROWN_ARTIFACT_OUT"


class ArtifactExportError(ValueError):
    """Raised when a raw verifier dump cannot be converted into the TorchLean schema."""


def _as_object(value: Any, ctx: str) -> Mapping[str, Any]:
    """Return `value` as a JSON object or raise a useful exporter error."""

    if not isinstance(value, dict):
        raise ArtifactExportError(f"{ctx}: expected JSON object")
    return value


def _get_any(obj: Mapping[str, Any], names: Sequence[str], ctx: str) -> Any:
    """Return the first present field from `names`."""

    for name in names:
        if name in obj:
            return obj[name]
    joined = ", ".join(names)
    raise ArtifactExportError(f"{ctx}: expected one of fields: {joined}")


def _get_optional(obj: Mapping[str, Any], names: Sequence[str]) -> Any | None:
    """Return the first present optional field from `names`."""

    for name in names:
        if name in obj:
            return obj[name]
    return None


def _float_list(value: Any, ctx: str) -> list[float]:
    """Parse a flat JSON number list."""

    if not isinstance(value, list):
        raise ArtifactExportError(f"{ctx}: expected a list of numbers")
    out: list[float] = []
    for i, item in enumerate(value):
        try:
            x = float(item)
        except (TypeError, ValueError) as exc:
            raise ArtifactExportError(f"{ctx}[{i}]: expected number, got {item!r}") from exc
        if not math.isfinite(x):
            raise ArtifactExportError(f"{ctx}[{i}]: expected finite number, got {x!r}")
        out.append(x)
    return out


def _float_list_or_scalar(value: Any, dim: int, ctx: str) -> list[float]:
    """Parse a threshold vector, accepting a scalar by broadcasting it to `dim`."""

    if isinstance(value, list):
        return _float_list(value, ctx)
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactExportError(f"{ctx}: expected number or list of numbers") from exc
    if not math.isfinite(x):
        raise ArtifactExportError(f"{ctx}: expected finite number, got {x!r}")
    return [x for _ in range(dim)]


def _parse_csv_floats(text: str, ctx: str) -> list[float]:
    """Parse a comma-separated float list from the command line."""

    if not text.strip():
        raise ArtifactExportError(f"{ctx}: expected at least one number")
    return _float_list([part.strip() for part in text.split(",")], ctx)


def _best_witness(lb: Sequence[float], threshold: Sequence[float]) -> tuple[int, float]:
    """Return the coordinate with the largest positive `lb - threshold` margin."""

    if len(lb) != len(threshold):
        raise ArtifactExportError(
            f"leaf: lb and threshold lengths differ ({len(lb)} vs {len(threshold)})"
        )
    if not lb:
        raise ArtifactExportError("leaf: lb/threshold vectors must be nonempty")
    margins = [float(a) - float(b) for a, b in zip(lb, threshold)]
    idx = max(range(len(margins)), key=lambda i: margins[i])
    margin = margins[idx]
    if not margin > 0.0:
        raise ArtifactExportError(
            "leaf: no verified witness found; expected some coordinate with lb > threshold"
        )
    return idx, margin


def _box_root_from_leaves(leaves: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    """Derive a root box from the componentwise min/max over leaves."""

    if not leaves:
        raise ArtifactExportError("cannot infer root from an empty leaf list")
    first_lo = _float_list(_get_any(leaves[0], LEAF_LO_FIELDS, "leaf[0]"), "leaf[0].lo")
    first_hi = _float_list(_get_any(leaves[0], LEAF_HI_FIELDS, "leaf[0]"), "leaf[0].hi")
    root_lo = list(first_lo)
    root_hi = list(first_hi)
    for idx, leaf in enumerate(leaves[1:], start=1):
        lo = _float_list(_get_any(leaf, LEAF_LO_FIELDS, f"leaf[{idx}]"), f"leaf[{idx}].lo")
        hi = _float_list(_get_any(leaf, LEAF_HI_FIELDS, f"leaf[{idx}]"), f"leaf[{idx}].hi")
        if len(lo) != len(root_lo) or len(hi) != len(root_hi):
            raise ArtifactExportError(f"leaf[{idx}]: input dimension does not match first leaf")
        root_lo = [min(a, b) for a, b in zip(root_lo, lo)]
        root_hi = [max(a, b) for a, b in zip(root_hi, hi)]
    return root_lo, root_hi


LEAF_LO_FIELDS = ("lo", "input_lo", "x_L", "x_l", "domain_lo", "lower")
LEAF_HI_FIELDS = ("hi", "input_hi", "x_U", "x_u", "domain_hi", "upper")
LEAF_LB_FIELDS = ("lb", "lower_bound", "lower_bounds", "output_lb", "margin_lb")
LEAF_THRESHOLD_FIELDS = ("threshold", "thresholds", "rhs", "unsafe_threshold")
LEAF_LIST_FIELDS = ("leaves", "domains", "verified_domains", "terminal_domains")


def _has_box_fields(obj: Mapping[str, Any]) -> bool:
    """Return whether `obj` carries both endpoints of an input box under any accepted alias."""

    return (
        any(name in obj for name in LEAF_LO_FIELDS)
        and any(name in obj for name in LEAF_HI_FIELDS)
    )


def normalize_leaf(
    leaf: Mapping[str, Any],
    *,
    index: int,
    input_dim: int,
    default_threshold: Any | None,
) -> dict[str, Any]:
    """Normalize one raw external leaf into TorchLean's artifact schema."""

    ctx = f"leaf[{index}]"
    lo = _float_list(_get_any(leaf, LEAF_LO_FIELDS, ctx), f"{ctx}.lo")
    hi = _float_list(_get_any(leaf, LEAF_HI_FIELDS, ctx), f"{ctx}.hi")
    if len(lo) != input_dim or len(hi) != input_dim:
        raise ArtifactExportError(
            f"{ctx}: input dimension mismatch; expected {input_dim}, got lo={len(lo)} hi={len(hi)}"
        )
    for i, (a, b) in enumerate(zip(lo, hi)):
        if a > b:
            raise ArtifactExportError(f"{ctx}: invalid box at coordinate {i}: lo={a} > hi={b}")

    lb = _float_list(_get_any(leaf, LEAF_LB_FIELDS, ctx), f"{ctx}.lb")
    threshold_raw = _get_optional(leaf, LEAF_THRESHOLD_FIELDS)
    if threshold_raw is None:
        if default_threshold is None:
            raise ArtifactExportError(
                f"{ctx}: missing threshold; provide leaf threshold or top-level default_threshold"
            )
        threshold_raw = default_threshold
    threshold = _float_list_or_scalar(threshold_raw, len(lb), f"{ctx}.threshold")
    witness_idx, witness_margin = _best_witness(lb, threshold)

    return {
        "lo": lo,
        "hi": hi,
        "lb": lb,
        "threshold": threshold,
        "witness_idx": witness_idx,
        "witness_margin": witness_margin,
    }


def build_abcrown_leaf_artifact(
    raw: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    root_lo: Sequence[float] | None = None,
    root_hi: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a TorchLean `abcrown_leaf_artifact_v0_1` object from a raw leaf dump."""

    if isinstance(raw, dict) and raw.get("format") == FORMAT:
        return dict(raw)

    if isinstance(raw, list):
        leaves_raw = [_as_object(item, f"leaf[{i}]") for i, item in enumerate(raw)]
        top: Mapping[str, Any] = {}
    else:
        top = _as_object(raw, "top-level")
        leaves_value = _get_optional(top, LEAF_LIST_FIELDS)
        if leaves_value is not None:
            # A dump that carries a domain list is a branch-and-bound run, even when the top
            # level also carries x_L/x_U: those are the root box, which is exactly what we want
            # to read from there.  Checking the list first matters because alpha-beta-CROWN
            # writes both, and the single-leaf reading would silently drop every domain.
            if not isinstance(leaves_value, list):
                raise ArtifactExportError("top-level leaves/domains field must be a list")
            leaves_raw = [_as_object(item, f"leaf[{i}]") for i, item in enumerate(leaves_value)]
        elif _has_box_fields(top):
            leaves_raw = [top]
            top = {}
        else:
            # Neither a domain list nor a leaf box.  Name the list aliases, since that is the
            # field a producer is most likely to have called something else.
            joined = ", ".join(LEAF_LIST_FIELDS)
            raise ArtifactExportError(f"top-level: expected one of fields: {joined}")

    if root_lo is None or root_hi is None:
        root_obj = _get_optional(top, ("root", "input_box"))
        if root_obj is not None:
            root = _as_object(root_obj, "root")
            root_lo = _float_list(_get_any(root, ("lo", "x_L", "lower"), "root"), "root.lo")
            root_hi = _float_list(_get_any(root, ("hi", "x_U", "upper"), "root"), "root.hi")
        else:
            root_lo_raw = _get_optional(top, ("root_lo", "input_lo", "x_L"))
            root_hi_raw = _get_optional(top, ("root_hi", "input_hi", "x_U"))
            if root_lo_raw is not None and root_hi_raw is not None:
                root_lo = _float_list(root_lo_raw, "root.lo")
                root_hi = _float_list(root_hi_raw, "root.hi")
            else:
                root_lo, root_hi = _box_root_from_leaves(leaves_raw)

    root_lo = [float(x) for x in root_lo]
    root_hi = [float(x) for x in root_hi]
    if len(root_lo) != len(root_hi):
        raise ArtifactExportError(
            f"root dimension mismatch; lo has {len(root_lo)} entries, hi has {len(root_hi)}"
        )
    for i, (a, b) in enumerate(zip(root_lo, root_hi)):
        if not math.isfinite(a) or not math.isfinite(b):
            raise ArtifactExportError(f"root coordinate {i}: expected finite bounds")
        if a > b:
            raise ArtifactExportError(f"root coordinate {i}: lo={a} > hi={b}")

    default_threshold = _get_optional(top, ("default_threshold", "threshold", "thresholds", "rhs"))
    leaves = [
        normalize_leaf(
            leaf,
            index=i,
            input_dim=len(root_lo),
            default_threshold=default_threshold,
        )
        for i, leaf in enumerate(leaves_raw)
    ]

    return {
        "format": FORMAT,
        "input_dim": len(root_lo),
        "root": {"lo": root_lo, "hi": root_hi},
        "leaves": leaves,
    }


def write_abcrown_leaf_artifact(
    *,
    root_lo: Sequence[float] | None,
    root_hi: Sequence[float] | None,
    leaves: Sequence[Mapping[str, Any]],
    out_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Write a TorchLean leaf artifact.

    If `out_path` is omitted, `ABCROWN_ARTIFACT_OUT` supplies the output path.  This is the function to
    call from an instrumented external verifier once it has terminal leaf boxes and lower bounds.
    """

    out = Path(out_path or os.environ.get(ENV_OUT, ""))
    if not str(out):
        raise ArtifactExportError(f"no output path supplied; pass out_path or set {ENV_OUT}")
    artifact = build_abcrown_leaf_artifact(list(leaves), root_lo=root_lo, root_hi=root_hi)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _read_json(path: Path) -> Any:
    """Read JSON or newline-delimited JSON leaves from `path`."""

    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        return json.loads(text)
    except JSONDecodeError:
        leaves = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                leaves.append(json.loads(stripped))
            except JSONDecodeError as exc:
                raise ArtifactExportError(
                    f"{path}:{lineno}: expected JSON object line in JSONL dump"
                ) from exc
        if not leaves:
            raise ArtifactExportError(f"{path}: empty JSON/JSONL input")
        return leaves


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    """Write JSON to `path`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_hook() -> None:
    """Print a short integration snippet for external verifier instrumentation."""

    print(
        """\
from scripts.verification.abcrown.export_leaf_artifact import write_abcrown_leaf_artifact

# Call this after the external verifier has collected terminal verified leaves.
# Each leaf should contain lo/hi input-box bounds, lb lower bounds, and threshold(s).
write_abcrown_leaf_artifact(
    root_lo=root_lo,
    root_hi=root_hi,
    leaves=terminal_leaves,
)
"""
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Raw external leaf/domain JSON dump.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output artifact path. Defaults to ${ENV_OUT} if set.",
    )
    parser.add_argument("--root-lo", help="Override root lower box as comma-separated floats.")
    parser.add_argument("--root-hi", help="Override root upper box as comma-separated floats.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run `lake exe verify -- abcrown-leaf` on the exported artifact.",
    )
    parser.add_argument(
        "--print-hook",
        action="store_true",
        help="Print the importable Python hook for external verifier instrumentation.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.print_hook:
        _print_hook()
        return 0
    if args.input is None:
        raise ArtifactExportError("--input is required unless --print-hook is used")

    out = args.out or os.environ.get(ENV_OUT)
    if out is None:
        raise ArtifactExportError(f"missing --out; pass --out or set {ENV_OUT}")
    root_lo = _parse_csv_floats(args.root_lo, "--root-lo") if args.root_lo else None
    root_hi = _parse_csv_floats(args.root_hi, "--root-hi") if args.root_hi else None

    raw = _read_json(args.input)
    artifact = build_abcrown_leaf_artifact(raw, root_lo=root_lo, root_hi=root_hi)
    out_path = Path(out)
    _write_json(out_path, artifact)
    print(f"Wrote TorchLean alpha-beta-CROWN-style leaf artifact to {out_path}", flush=True)

    if args.check:
        subprocess.run(
            ["lake", "exe", "verify", "--", "abcrown-leaf", str(out_path)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArtifactExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
