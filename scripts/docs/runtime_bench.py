#!/usr/bin/env python3
"""Render the runtime benchmark section of the performance page from its saved results.

`--from SUMMARY.json` first saves the results of a `scripts/bench/bench.py` run, and `--check`
exits with status 1 if the page does not match the saved results, writing nothing.
"""

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys
import textwrap
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "home_page/assets/runtime-bench.json"
PAGE = ROOT / "home_page/performance/index.md"
START = "<!-- runtime-bench:start (scripts/docs/runtime_bench.py renders this section) -->"
END = "<!-- runtime-bench:end -->"


def from_summary(summary: dict) -> dict:
    """The saved results of a bench.py summary.json, which must have no failed repetition."""
    meta = summary["meta"]
    jobs = {job["name"]: job for job in meta["jobs"]}
    benchmarks = []
    for name, s in summary["bench"].items():
        if s["ok"] != s["n"]:
            raise ValueError(f"{name} failed in {s['n'] - s['ok']} of {s['n']} repetitions")
        benchmarks.append({
            "name": name,
            "about": jobs[name]["about"],
            "command": jobs[name]["command"],
            "runsPerRepetition": jobs[name]["runs"],
            "repetitions": s["n"],
            "instructions": s.get("instr"),
            "instructionsMin": s.get("instr_min"),
            "instructionsMax": s.get("instr_max"),
            "cpuSeconds": s["cpu_s"],
            "cpuSecondsMin": s["cpu_s_min"],
            "cpuSecondsMax": s["cpu_s_max"],
            "wallSeconds": s["wall_s"],
            "maxRssKiB": s["maxrss_kb"],
            "outputSha256": s["out_sha"],
        })
    git = meta.get("git") or {}
    return {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "measuredAt": meta["end"],
        "tool": "scripts/bench/bench.py",
        "suite": meta["suite"],
        "repetitions": meta["reps"],
        "tree": {
            "commit": git.get("commit"),
            "modified": git.get("modified"),
            "sourceDigest": meta["source_digest"],
            "sourceFiles": meta["source_files"],
            "leanToolchain": meta["lean_toolchain"],
            "executables": meta["binaries_sha256_16"],
            "inputs": meta["inputs_sha256_16"],
        },
        "machine": {
            "cpu": meta["machine"]["cpu"],
            "cpus": meta["machine"]["cpus"],
            "system": meta["machine"]["system"],
            "coresAllowed": len(meta["affinity"]) if meta.get("affinity") else None,
            "counters": meta["counters"],
        },
        "benchmarks": benchmarks,
    }


def spread(rows: list[dict], key: str) -> "float | None":
    """The largest range of `key` between repetitions, in percent of its median."""
    ranges = [100.0 * (r[key + "Max"] - r[key + "Min"]) / r[key]
              for r in rows if r.get(key) and r.get(key + "Min") is not None]
    return max(ranges) if ranges else None


def code(text: str) -> str:
    """`text` in an HTML `<code>` element."""
    return f"<code>{html.escape(text, quote=False)}</code>"


def paragraph(cls: str, text: str) -> list[str]:
    """A `<p>` of class `cls` around `text` (HTML), wrapped at 100 columns."""
    return [f'    <p class="{cls}">'] + textwrap.wrap(
        text, 100, initial_indent="      ", subsequent_indent="      ",
        break_long_words=False, break_on_hyphens=False) + ["    </p>"]


def render(data: dict) -> str:
    """The page section of the saved results, between the two markers."""
    rows = data["benchmarks"]
    instr, cpu = spread(rows, "instructions"), spread(rows, "cpuSeconds")
    intro = (f"We ran each command {data['repetitions']} times. The table shows the medians. "
             f"CPU times varied by at most {cpu:.1f}%.")
    if instr is not None:
        variation = "less than 0.01%" if instr < 0.01 else f"at most {instr:.2f}%"
        intro += f" Instruction counts varied by {variation}."
    show_instructions = any(row["instructions"] is not None for row in rows)
    out = [START, '<div class="performance-dashboard">',
           '  <section class="performance-panel" aria-labelledby="performance-runtime-title">',
           '    <h2 id="performance-runtime-title">Runtime benchmarks</h2>']
    out += paragraph("performance-panel-intro", intro)
    out += ['    <div class="performance-table-wrap">', '      <table class="performance-table">',
            "        <thead>", "          <tr>"]
    headers = ["Benchmark", "Workload"]
    if show_instructions:
        headers.append("Instructions")
    headers.extend(("CPU time", "Wall time", "Peak memory"))
    out += [f'            <th scope="col">{header}</th>' for header in headers]
    out += ["          </tr>", "        </thead>", "        <tbody>"]
    for row in rows:
        runs = f", {row['runsPerRepetition']} runs" if row["runsPerRepetition"] > 1 else ""
        cells = [code(row["name"]), html.escape(row["about"], quote=False) + runs]
        if show_instructions:
            instructions = row["instructions"]
            cells.append("n/a" if instructions is None else f"{instructions / 1e9:.1f} billion")
        cells.extend((f"{row['cpuSeconds']:.2f} s", f"{row['wallSeconds']:.2f} s",
                      f"{row['maxRssKiB'] / 1024:.0f} MiB"))
        out += ["          <tr>", *(f"            <td>{cell}</td>" for cell in cells), "          </tr>"]
    out += ["        </tbody>", "      </table>", "    </div>"]
    tree, machine = data["tree"], data["machine"]
    lean = html.escape(tree["leanToolchain"].rsplit(":", 1)[-1].lstrip("v"))
    cpu_name = html.escape(machine["cpu"])
    measured_at = datetime.fromisoformat(data["measuredAt"].replace("Z", "+00:00"))
    date = measured_at.astimezone(ZoneInfo("America/Los_Angeles")).date()
    measured = f"Measured on {date} with Lean {lean} on {cpu_name}."
    if tree["commit"]:
        measured += f" Source commit {code(tree['commit'][:12])}"
        measured += " with local changes." if tree["modified"] else "."
    out += paragraph("performance-chart-caption", measured)
    out += ['    <details>',
            '      <summary>Commands</summary>',
            '      <ul>']
    for row in rows:
        out += textwrap.wrap(code(row["command"]), 95, initial_indent="        <li>",
                             subsequent_indent="          ", break_long_words=False,
                             break_on_hyphens=False)
        out[-1] += "</li>"
    out += ["      </ul>", "    </details>", "  </section>", "</div>", END]
    text = "\n".join(out) + "\n"
    if "{{" in text or "{%" in text:
        raise ValueError("the section would contain Liquid syntax")
    return text


def replace_section(page: str, section: str) -> str:
    """`page` with the text between the markers, markers included, replaced by `section`."""
    if page.count(START) != 1 or page.count(END) != 1 or page.index(START) > page.index(END):
        raise ValueError(f"the page needs one '{START}' line and one '{END}' line after it")
    head, rest = page.split(START, 1)
    return head + section.rstrip("\n") + rest.split(END, 1)[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="summary", type=Path,
                        help="summary.json of a bench.py run to save first")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--check", action="store_true",
                        help="exit with status 1 if the page does not match the saved results")
    args = parser.parse_args()
    try:
        if args.summary:
            data = from_summary(json.loads(args.summary.read_text(encoding="utf-8")))
            if not args.check:
                args.data.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
                print(f"Runtime benchmarks: saved {len(data['benchmarks'])} results to "
                      f"{args.data}.")
        else:
            data = json.loads(args.data.read_text(encoding="utf-8"))
        page = args.page.read_text(encoding="utf-8")
        new = replace_section(page, render(data))
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Runtime benchmarks: {error}", file=sys.stderr)
        return 1
    if args.check:
        if new != page:
            print(f"Runtime benchmarks: {args.page} does not match {args.data}; run "
                  "python3 scripts/docs/runtime_bench.py.", file=sys.stderr)
            return 1
        return 0
    if new != page:
        args.page.write_text(new, encoding="utf-8")
        print(f"Runtime benchmarks: rendered {args.page}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
