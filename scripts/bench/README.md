# Benchmarks

`bench.py` measures how fast TorchLean's executables run on the CPU and how long Lean takes to
elaborate three of its largest modules. It runs one checkout, or two checkouts in alternation so
that both see the same machine load, and compares the results.

## Setup

Build the two executables and download the example data once:

```bash
scripts/lake.sh build torchlean verify
python3 scripts/datasets/download_example_data.py --root data/real \
  --cifar10 --auto-mpg --tiny-shakespeare
```

`SHA256SUMS` holds the digests of the data files that the benchmarks read. Before it starts,
`bench.py` compares the files with these digests and asks `scripts/lake.sh build --no-build`
whether the executables and modules are current. If a file is missing or differs, or Lake would
rebuild something, it stops with exit status 2.

## Commands

```bash
scripts/bench/bench.py run .                  # the quick suite
scripts/bench/bench.py run . --suite full     # every runtime command, then the elaboration
scripts/bench/bench.py ab ../base .           # two checkouts in alternation, with verdicts
scripts/bench/bench.py compare RUN1/summary.json RUN2/summary.json
scripts/bench/bench.py show RUN/summary.json
```

| Suite | Commands | Repetitions | One checkout on our machine |
| --- | --- | --- | --- |
| `quick` (default) | `qs_mlp_ieee`, `gpt2`, `crown_query` | 3 | about 1.5 minutes |
| `cpu` | the twelve runtime commands of the `CPU` table in `bench.py` | 5 | about 4 minutes |
| `elab` | `lean` on the three modules of the `ELAB` table | 5 | about 5 minutes |
| `full` | `cpu`, then `elab` | 5 | about 8 minutes |

`--only gpt2,crown_query` picks commands, `--reps` sets the repetitions and `--cores 2,3` pins
the commands to cores. `--data` and `--out` move the data (`data/real`) and the results
(`_out/bench`). The commands run with `OMP_NUM_THREADS=1` and no visible CUDA device.
`crown_query` runs `verify crown-query` on a file that `gen_query.py` writes, the same bytes on
every run.

## Measurements

Each repetition runs one command, ten times in a row for the five short `verify` commands, and
records the user and system CPU time, the wall time and the largest resident set of its process
tree. When `/proc/sys/kernel/perf_event_paranoid` is 2 or lower, `bench.py` also counts the
tree's user-space instructions and cycles with `perf_event_open`. Otherwise it records times
without counts, and `meta.counters` in `summary.json` says which applied. On our shared build
machine the instruction counts of a runtime command varied by at most 0.01% between repetitions,
and those of elaboration by about 0.1%. CPU time varied by up to 9% at moderate load and by up to
34% when other jobs ran on the same physical cores.

`ab` and `compare` call a benchmark SLOWER or FASTER when its instructions change by more than
5% (`--threshold`). CPU time depends on the load, so it decides in two cases: between the two
sides of one `ab` run, whose repetitions alternate, and when a side has no instruction counts. It
then needs a change of more than 5% and at least 3 repetitions on each side whose ranges do not
overlap. The start of `verify` costs more than the own work of `ibp`, `ibp_ieee`, `crown_ops` and
`crown_ops_ieee`, so their verdicts also use their instructions minus those of `verify_startup`.
The output column compares the SHA-256 of each command's output.

## Results

Each run writes `_out/bench/<label>-<suite>-<UTC time>/` (`<label>-<suite>-ab-<UTC time>/` for
`ab`): `summary.json` and `summary.md` with the median, minimum and maximum of every measurement,
`reps.jsonl` with one line per repetition, `out/` with the output of each repetition, and `log.txt`.
`ab` writes `summaryA.json` and `summaryB.json` for the two checkouts and `compare.md` with the
verdicts. `summary.json` also records the checkout's Git commit, a digest of its sources, the
digests of the executables, the machine and the load.

## The performance page

`home_page/assets/runtime-bench.json` holds the numbers of the runtime section of
`home_page/performance/index.md`. Refresh both from a quick run of the release checkout:

```bash
scripts/bench/bench.py run . --label release --reps 5
python3 scripts/docs/runtime_bench.py --from _out/bench/release-quick-<UTC time>/summary.json
```

Without `--from`, the script renders the section from the saved JSON, as the `timings` step of
`scripts/docs/build_site.sh` does, and `--check` exits with status 1 if the page does not match.
