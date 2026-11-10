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
    varied = "Between repetitions "
    if instr is not None:
        varied += ("the instruction counts varied by less than 0.01%" if instr < 0.01
                   else f"the instruction counts varied by at most {instr:.2f}%") + " and "
    varied += f"the CPU time by at most {cpu:.1f}%."
    intro = (f"We ran the {data['suite']} suite of {code(data['tool'])}: {len(rows)} commands, "
             f"{data['repetitions']} times each. The table shows the median of each measurement. "
             + varied)
    out = [START, '<div class="performance-dashboard">',
           '  <section class="performance-panel" aria-labelledby="performance-runtime-title">',
           '    <h2 id="performance-runtime-title">Runtime benchmarks</h2>']
    out += paragraph("performance-panel-intro", intro)
    out += ['    <div class="performance-table-wrap">', '      <table class="performance-table">',
            "        <thead>", "          <tr>"]
    out += [f"            <th>{h}</th>" for h in
            ("Benchmark", "What it runs", "Instructions", "CPU time", "Wall time", "Max RSS")]
    out += ["          </tr>", "        </thead>", "        <tbody>"]
    for r in rows:
        runs = f", {r['runsPerRepetition']} runs" if r["runsPerRepetition"] > 1 else ""
        cells = [code(r["name"]), html.escape(r["about"], quote=False) + runs,
                 "n/a" if r["instructions"] is None
                 else f"{r['instructions'] / 1e9:.1f} billion",
                 f"{r['cpuSeconds']:.2f} s", f"{r['wallSeconds']:.2f} s",
                 f"{r['maxRssKiB'] / 1024:.0f} MiB"]
        out += ["          <tr>", *(f"            <td>{c}</td>" for c in cells), "          </tr>"]
    out += ["        </tbody>", "      </table>", "    </div>"]
    out += paragraph("performance-chart-caption",
                     f"The commands, run in the checkout with the example data of "
                     f"{code('scripts/datasets/download_example_data.py')} in {code('data/real')} "
                     f"and the query file that {code('scripts/bench/gen_query.py')} writes:")
    out += ["    <ul>"]
    for r in rows:
        out += textwrap.wrap(code(r["command"]), 95, initial_indent="      <li>",
                             subsequent_indent="        ", break_long_words=False,
                             break_on_hyphens=False)
        out[-1] += "</li>"
    out += ["    </ul>"]
    tree, machine = data["tree"], data["machine"]
    if tree.get("commit"):
        where = f"commit {code(tree['commit'][:12])}"
        if tree["modified"]:
            where += " with local changes"
    else:
        where = (f"a checkout whose {tree['sourceFiles']} source files have the bench.py digest "
                 f"{code(tree['sourceDigest'])}")
    cores = (f", with {machine['coresAllowed']} of its {machine['cpus']} logical CPUs available"
             if machine.get("coresAllowed") else "")
    counted = ("Instructions are user-space counts from perf_event_open. "
               if machine["counters"] == "perf_event_open" else "")
    out += paragraph("performance-chart-caption",
                     f"Measured on {data['measuredAt'][:10]} at {where}, with "
                     f"{html.escape(tree['leanToolchain'])}, on {html.escape(machine['cpu'])}"
                     f"{cores}. {counted}Max RSS is the largest resident set of the command's "
                     "process tree.")
    out += ["  </section>", "</div>", END]
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
