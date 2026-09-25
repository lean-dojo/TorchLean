#!/usr/bin/env python3
"""Build the dedicated IRExec driver and measure each graph in a fresh Linux process."""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import re
import statistics
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[2]
METRICS = ("lower_ns", "eval_ns", "drop_ns", "rss_kb", "peak_kb")
SAMPLE = re.compile(
    r"n=(?P<n>\d+) lower_ns=(?P<lower_ns>\d+) eval_ns=(?P<eval_ns>\d+) "
    r"drop_ns=(?P<drop_ns>\d+) hash=(?P<hash>\d+) "
    r"VmHWM:\s+(?P<hwm_kb>\d+) kB VmRSS:\s+(?P<rss_kb>\d+) kB"
)


def build() -> pathlib.Path:
    # Lake resolves the package relative to its config, so keep the temporary config at the root.
    # The original config, dependency paths, build directory, and test registry stay untouched.
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="lakefile.irexec-scaling.", suffix=".lean", dir=ROOT, delete=False
    ) as handle:
        config = pathlib.Path(handle.name)
        handle.write((ROOT / "lakefile.lean").read_text())
        handle.write(
            "\nlean_exe irexec_scaling where\n"
            "  root := `NN.Tests.Runtime.IRExecScaling\n"
        )
    try:
        subprocess.run(
            ["lake", "-f", config.name, "-Kcuda=false", "build", "irexec_scaling"],
            cwd=ROOT,
            check=True,
        )
    finally:
        config.unlink()
    return ROOT / ".lake/build/bin/irexec_scaling"


def sample(binary: pathlib.Path, n: int, lower_only: bool) -> dict[str, int]:
    command = [str(binary), *(["--lower-only"] if lower_only else []), str(n)]
    result = subprocess.run(
        ["/usr/bin/time", "-f", "peak_kb=%M", *command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    match = SAMPLE.search(result.stdout)
    peak = re.search(r"^peak_kb=(\d+)$", result.stderr, flags=re.MULTILINE)
    if match is None or peak is None:
        raise RuntimeError(f"unrecognized driver output:\n{result.stdout}\n{result.stderr}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    values["peak_kb"] = int(peak[1])
    if values["n"] != n:
        raise RuntimeError("driver measured a different graph size")
    return values


def check_growth(medians: list[dict[str, float]]) -> None:
    first, last = medians[0], medians[-1]
    factor = last["n"] / first["n"]
    if factor < 8:
        raise ValueError("--assert-linear needs graph sizes spanning at least a factor of eight")
    # A coarse regression guard, not a complexity proof. Repetitions and a generous multiplier
    # tolerate scheduler/allocator noise while rejecting the old quadratic time and retention.
    failures = []
    for metric in ("lower_ns", "drop_ns", "rss_kb", "peak_kb"):
        growth = last[metric] / max(first[metric], 1)
        if growth > 2.5 * factor:
            failures.append(f"{metric} grew {growth:.2f}x for {factor:.2f}x more nodes")
    if failures:
        raise RuntimeError("scaling regression: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, help="measure an existing driver; skip building")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 2000, 4000, 8000, 16000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--lower-only", action="store_true", help="measure construction and disposal")
    parser.add_argument("--skip-check", action="store_true", help="for older baseline drivers")
    parser.add_argument("--check-only", action="store_true", help="run semantic regressions and exit")
    parser.add_argument("--assert-linear", action="store_true", help="check coarse empirical growth")
    parser.add_argument("--output", type=pathlib.Path, help="write all samples and medians as JSON")
    args = parser.parse_args()
    sizes = sorted(set(args.sizes))
    if not sizes or sizes[0] <= 0 or args.repeats <= 0:
        parser.error("sizes and repetitions must be positive")
    if args.check_only and args.skip_check:
        parser.error("--check-only cannot be combined with --skip-check")
    if args.assert_linear and sizes[-1] < 8 * sizes[0]:
        parser.error("--assert-linear needs graph sizes spanning at least a factor of eight")

    binary = args.binary.resolve() if args.binary else build()
    if not args.skip_check:
        subprocess.run([str(binary), "--check"], cwd=ROOT, check=True)
    if args.check_only:
        return

    samples = []
    hashes = {}
    for repeat in range(1, args.repeats + 1):
        for n in sizes:
            row = sample(binary, n, args.lower_only)
            row["repeat"] = repeat
            if n in hashes and hashes[n] != row["hash"]:
                raise RuntimeError(f"execution hash changed across repetitions for n={n}")
            hashes[n] = row["hash"]
            samples.append(row)
            print(
                f"repeat={repeat} n={n} lower_ms={row['lower_ns'] / 1e6:.3f} "
                f"eval_ms={row['eval_ns'] / 1e6:.3f} drop_ms={row['drop_ns'] / 1e6:.3f} "
                f"rss_kb={row['rss_kb']} peak_kb={row['peak_kb']} hash={row['hash']}",
                flush=True,
            )

    medians = [
        {
            "n": n,
            **{
                metric: statistics.median(row[metric] for row in samples if row["n"] == n)
                for metric in METRICS
            },
        }
        for n in sizes
    ]
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "binary": str(binary),
                    "platform": platform.platform(),
                    "mode": "lower-only" if args.lower_only else "lower-evaluate-drop",
                    "samples": samples,
                    "medians": medians,
                },
                indent=2,
            )
            + "\n"
        )
    if args.assert_linear:
        check_growth(medians)
        print("IRExec empirical scaling guard passed")


if __name__ == "__main__":
    main()
