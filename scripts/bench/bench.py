#!/usr/bin/env python3
"""Measure TorchLean's runtime and elaboration on one checkout, or compare two checkouts.

  bench.py run TREE [--suite S]             one checkout
  bench.py ab TREE_A TREE_B [--suite S]     two checkouts, interleaved (A B A B ...), with verdicts
  bench.py compare BASE.json NEW.json       the summary.json files of two runs
  bench.py show SUMMARY.json                the table of one run

The suite is quick (the default: three commands, 3 repetitions), cpu (twelve runtime commands),
elab (three modules under `lean`) or full (cpu, then elab), with 5 repetitions unless quick.
Each repetition runs one command and records the user and system CPU seconds, the wall time and the
largest resident set of its process tree. Where the kernel allows it (Linux with
/proc/sys/kernel/perf_event_paranoid at most 2), it also counts the user-space instructions and
cycles of the tree with perf_event_open. Results go to OUT/<label>-<suite>-<UTC time>/ (with -ab
before the time for ab): reps.jsonl, summary.json, summary.md and out/, the output of every
repetition.
"""

import argparse
import ctypes
import datetime
import hashlib
import json
import os
import platform
import resource
import signal
import statistics
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SHA256SUMS = os.path.join(HERE, "SHA256SUMS")
GEN_QUERY = os.path.join(HERE, "gen_query.py")
QUERY_NAME, QUERY_DIMS = "q-24-48x3-8.json", [24, 48, 48, 48, 8]
DOWNLOAD = ("python3 scripts/datasets/download_example_data.py --root {} "
            "--cifar10 --auto-mpg --tiny-shakespeare")


def bench(name, exe, args, about, runs=1, files=()):
    """A benchmark: `exe` with `args` (`{data}` is the data directory, `{query}` the generated
    CROWN query), run `runs` times per repetition, and the files it writes (hashed)."""
    return {"name": name, "exe": exe, "args": args.split(), "about": about, "runs": runs,
            "files": list(files)}


CPU = [
    bench("qs_mlp_native", "torchlean",
          "quickstart_mlp --device cpu --arithmetic native --steps 10000 --seed 0",
          "the quickstart MLP's eager training loop and Adam on Lean Float"),
    bench("qs_mlp_ieee", "torchlean",
          "quickstart_mlp --device cpu --arithmetic ieee --steps 5000 --seed 0",
          "the quickstart MLP's training loop and Adam in FloatLib binary32"),
    bench("mlp", "torchlean",
          "mlp --device cpu --csv {data}/auto_mpg/auto_mpg.csv --steps 1500 --log false --seed 0",
          "MLP regression on UCI Auto MPG (392 rows)"),
    bench("cnn", "torchlean",
          "cnn --device cpu --x {data}/cifar10/cifar10_train_X.npy "
          "--y {data}/cifar10/cifar10_train_y.npy --n-total 20 --steps 16 --log false --seed 0",
          "a convolutional classifier on 20 CIFAR-10 images"),
    bench("gpt2", "torchlean",
          "gpt2 --device cpu --data-file {data}/text/tiny_shakespeare.txt --steps 400 "
          "--log false --seed 0",
          "a byte-level GPT on Tiny Shakespeare"),
    bench("ppo_gridworld", "torchlean",
          "ppo_gridworld --device cpu --updates 6 --log false --policy policy.json "
          "--path path.json --seed 0",
          "PPO against the checked GridWorld contract", files=["policy.json", "path.json"]),
    bench("crown_query", "verify", "crown-query {query}",
          "exact-rational CROWN bounds of a 24-48-48-48-8 ReLU chain"),
    bench("verify_startup", "verify", "--help", "the start of verify before any command", runs=10),
    bench("ibp", "verify", "torchlean-ibp", "the IBP workflow on an MLP", runs=10),
    bench("ibp_ieee", "verify", "torchlean-ibp --arithmetic ieee",
          "the IBP workflow on an MLP in FloatLib binary32", runs=10),
    bench("crown_ops", "verify", "torchlean-crown-ops",
          "IBP and CROWN through softmax and MSE", runs=10),
    bench("crown_ops_ieee", "verify", "torchlean-crown-ops --arithmetic ieee",
          "IBP and CROWN through softmax and MSE in FloatLib binary32", runs=10),
]
# verify's start costs more than these commands' own work, so they are also compared by their
# instructions minus those of verify_startup (both 10 runs per repetition).
STARTUP_OF = {"ibp": "verify_startup", "ibp_ieee": "verify_startup",
              "crown_ops": "verify_startup", "crown_ops_ieee": "verify_startup"}
QUICK = ["qs_mlp_ieee", "gpt2", "crown_query"]
ELAB = [
    ("elab_op_table", "TorchLean/Tensor/Op/Table/Rule.lean",
     "the rules of the registered operator table"),
    ("elab_online_softmax", "TorchLean/Verify/RuntimeApprox/OnlineSoftmax.lean",
     "the online softmax error proofs"),
    ("elab_softmax_shift", "TorchLean/Verify/RuntimeApprox/Roundoff/SoftmaxShift.lean",
     "the roundoff automation"),
]
LEAN_OPTIONS = ["-Dpp.unicode.fun=true", "-DautoImplicit=false", "-DrelaxedAutoImplicit=false",
                "-DwarningAsError=true"]  # the leanOptions of lakefile.lean


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def source_digest(tree):
    """sha256 over the paths and contents of the sources that build the executables."""
    h = hashlib.sha256()
    top = ["TorchLean", "TorchLeanExamples", "TorchLeanTest", "csrc", "scripts/checks"]
    files = [f for f in ["lakefile.lean", "lake-manifest.json", "lean-toolchain"]
             if os.path.isfile(os.path.join(tree, f))]
    for t in top:
        for root, dirs, names in os.walk(os.path.join(tree, t)):
            dirs[:] = sorted(d for d in dirs if d not in (".lake", "__pycache__"))
            for n in sorted(names):
                files.append(os.path.relpath(os.path.join(root, n), tree))
    for rel in sorted(files):
        h.update(rel.encode() + b"\0")
        with open(os.path.join(tree, rel), "rb") as f:
            h.update(hashlib.sha256(f.read()).digest())
    return h.hexdigest()[:16], len(files)


def git_state(tree):
    """The commit of a Git checkout and whether its tracked files differ from it, else None."""
    def git(*args):
        return subprocess.run(["git", "-C", tree] + list(args), capture_output=True, text=True,
                              timeout=60)
    try:
        top = git("rev-parse", "--show-toplevel")
        if top.returncode != 0 or os.path.realpath(top.stdout.strip()) != os.path.realpath(tree):
            return None
        head = git("rev-parse", "--verify", "--quiet", "HEAD")
        if head.returncode != 0:
            return None
        status = git("status", "--porcelain", "--untracked-files=no")
        return {"commit": head.stdout.strip(), "modified": bool(status.stdout.strip())}
    except (OSError, subprocess.TimeoutExpired):
        return None


def machine():
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((ln.split(":", 1)[1].strip() for ln in f if ln.startswith("model name")),
                       cpu)
    except OSError:
        pass
    return {"cpu": cpu, "cpus": os.cpu_count(), "system": platform.platform(),
            "python": platform.python_version()}


# perf_event_open: user-space instructions and cycles of the child and of every process it starts.
_NR_PERF_EVENT_OPEN = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
# disabled, inherit, exclude_kernel, exclude_hv, enable_on_exec
_FLAGS = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 6) | (1 << 12)
_READ_FORMAT = 1 | 2  # TOTAL_TIME_ENABLED, TOTAL_TIME_RUNNING
_libc = None


def _open_counter(pid, config):
    global _libc
    if _NR_PERF_EVENT_OPEN is None or platform.system() != "Linux":
        return None
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
    attr = struct.pack("IIQQQQQIIQQ", 0, 72, config, 0, 0, _READ_FORMAT, _FLAGS, 0, 0, 0, 0)
    buf = ctypes.create_string_buffer(attr, 72)
    fd = _libc.syscall(_NR_PERF_EVENT_OPEN, buf, pid, -1, -1, ctypes.c_ulong(8))
    return fd if fd >= 0 else None


def _read_counter(fd):
    if fd is None:
        return None, None
    value, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
    os.close(fd)
    if running == 0:
        return None, 0.0
    return (value * enabled / running if running < enabled else value), running / enabled


def counters():
    """How instructions are counted: perf_event_open if this process may open a counter."""
    fd = _open_counter(0, 1)
    if fd is not None:
        os.close(fd)
        return "perf_event_open"
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            return f"none (perf_event_paranoid {f.read().strip()})"
    except OSError:
        return "none"


def measure(argv, cwd, env, outpath, timeout):
    """Run argv once, its output in outpath, and return the metrics of its process tree. The
    child starts as a copy of this process, so its largest resident set is at least ours."""
    r, w = os.pipe()
    out = os.open(outpath, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    pid = os.fork()
    if pid == 0:
        try:
            os.close(w)
            os.setpgid(0, 0)
            os.read(r, 1)  # start once the counters are attached
            os.chdir(cwd)
            os.dup2(out, 1)
            os.dup2(out, 2)
            os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
            os.execve(argv[0], argv, env)
        except OSError as e:
            os.write(2, f"bench.py: cannot run {argv[0]}: {e}\n".encode())
        finally:
            os._exit(127)
    try:
        os.setpgid(pid, pid)  # so that a timeout kills the whole tree
    except OSError:
        pass  # the child has already done it
    os.close(r)
    os.close(out)
    fds = [_open_counter(pid, 1), _open_counter(pid, 0)]  # instructions, cycles
    t0 = time.monotonic()
    os.write(w, b"x")
    os.close(w)
    while True:
        wpid, status, usage = os.wait4(pid, os.WNOHANG)
        if wpid == pid:
            break
        if time.monotonic() > t0 + timeout:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, status, usage = os.wait4(pid, 0)
            break
        time.sleep(0.02)
    wall = time.monotonic() - t0
    instr, running = _read_counter(fds[0])
    cycles, _ = _read_counter(fds[1])
    with open(outpath, "rb") as f:
        out_sha = hashlib.sha256(f.read()).hexdigest()[:16]
    maxrss = usage.ru_maxrss // 1024 if sys.platform == "darwin" else usage.ru_maxrss  # bytes there
    return {"rc": os.waitstatus_to_exitcode(status), "wall_s": round(wall, 3),
            "user_s": round(usage.ru_utime, 3), "sys_s": round(usage.ru_stime, 3),
            "cpu_s": round(usage.ru_utime + usage.ru_stime, 3), "maxrss_kb": maxrss,
            "instr": instr and int(instr), "cycles": cycles and int(cycles),
            "counter_running": running, "out_sha": out_sha}


def lake(tree, args, env, timeout=1200):
    return subprocess.run([os.path.join(tree, "scripts/lake.sh")] + args, cwd=tree, env=env,
                          capture_output=True, text=True, timeout=timeout)


def last_line(p, what):
    lines = p.stdout.strip().splitlines()
    if p.returncode != 0 or not lines:
        sys.exit(f"bench.py: {what} failed (rc {p.returncode}):\n{(p.stdout + p.stderr)[-2000:]}")
    return lines[-1]


def plan(suite, tree, args, env, log, data):
    """The build directory and the jobs of a suite on one checkout."""
    build = last_line(lake(tree, ["--torchlean-build-dir"], env, 120),
                      "scripts/lake.sh --torchlean-build-dir")
    jobs = []
    if suite in ("quick", "cpu", "full"):
        for b in CPU:
            if suite == "quick" and b["name"] not in QUICK:
                continue
            def fill(d):
                return [a.replace("{data}", d).replace("{query}", QUERY_NAME) for a in b["args"]]
            jobs.append({"name": b["name"], "exe": b["exe"], "kind": "run", "about": b["about"],
                         "argv": [os.path.join(build, "bin", b["exe"])] + fill(data),
                         "command": " ".join([b["exe"]] + fill("data/real")),
                         "data": [a.split("{data}/", 1)[1] for a in b["args"] if "{data}/" in a],
                         "query": "{query}" in b["args"], "runs": b["runs"], "files": b["files"]})
    if suite in ("elab", "full"):
        code = ("import json, os; print(json.dumps({k: os.environ.get(k) "
                "for k in ('LEAN_PATH', 'LEAN_SYSROOT')}))")
        lean_env = json.loads(last_line(lake(tree, ["env", sys.executable, "-c", code], env, 300),
                                        "scripts/lake.sh env"))
        lean = os.path.join(lean_env["LEAN_SYSROOT"], "bin", "lean")
        for name, rel, about in (parse_modules(args.modules) if args.modules else ELAB):
            if not os.path.isfile(os.path.join(tree, rel)):
                log(f"MISSING {name}: {rel} is not in {tree}")
                continue
            jobs.append({"name": name, "kind": "elab", "about": about, "module": rel,
                         "argv": [lean] + LEAN_OPTIONS + [rel], "command": f"lean {rel}",
                         "lean_path": lean_env["LEAN_PATH"], "cwd": tree, "data": [],
                         "query": False, "runs": 1, "files": []})
    if args.only:
        keep = set(args.only.split(","))
        unknown = keep - {j["name"] for j in jobs}
        if unknown:
            sys.exit(f"bench.py: --only names no {suite} benchmark: {','.join(sorted(unknown))}")
        jobs = [j for j in jobs if j["name"] in keep]
    return build, jobs


def parse_modules(spec):
    """--modules name=path,name=path, with paths relative to the checkout."""
    items = [item.split("=", 1) for item in spec.split(",")]
    if any(len(item) != 2 or not all(item) for item in items):
        sys.exit(f"bench.py: --modules takes name=path,name=path, not {spec}")
    return [(name, rel, "") for name, rel in items]


def check_built(tree, jobs, env, log):
    """Whether Lake would rebuild nothing that the jobs run or import."""
    targets = sorted({j["exe"] for j in jobs if j["kind"] == "run"})
    targets += [j["module"].removesuffix(".lean").replace("/", ".")
                for j in jobs if j["kind"] == "elab"]
    p = lake(tree, ["build", "--no-build"] + targets, env, 1800)
    log(f"check: scripts/lake.sh build --no-build {' '.join(targets)} rc={p.returncode}")
    if p.returncode != 0:
        log(f"STALE: build them first: scripts/lake.sh build {' '.join(targets)}\n"
            + (p.stdout + p.stderr)[-2000:])
    return p.returncode == 0


def check_data(data, files, strict, log):
    """sha256 of the data files, or None if one is missing or, when strict, differs from
    SHA256SUMS."""
    sums = {}
    with open(SHA256SUMS) as f:
        for ln in f:
            if ln.strip():
                digest, rel = ln.split(None, 1)
                sums[rel.strip()] = digest
    found, differs = {}, []
    for rel in files:
        path = os.path.join(data, rel)
        if not os.path.isfile(path):
            log(f"MISSING data file {path}; write it with: {DOWNLOAD.format(data)}")
            return None
        digest = sha256_file(path, 64)
        found[rel] = digest[:16]
        if digest != sums.get(rel):
            differs.append(rel)
    if differs:
        log(f"DATA {', '.join(differs)} in {data} differ from scripts/bench/SHA256SUMS"
            + (" (--no-data-check runs anyway)" if strict else ""))
        if strict:
            return None
    return found


def job_argv(j):
    if j["runs"] == 1:
        return j["argv"]
    loop = 'i=0; while [ "$i" -lt %d ]; do "$@" || exit $?; i=$((i + 1)); done' % j["runs"]
    return ["/bin/sh", "-c", loop, "bench-loop"] + j["argv"]


def run_rep(j, rep, outdir, cwd, env, args):
    e = dict(env)
    if j["kind"] == "elab":
        e["LEAN_PATH"] = j["lean_path"]
    for f in j["files"]:
        if os.path.exists(os.path.join(cwd, f)):
            os.remove(os.path.join(cwd, f))
    rec = {"bench": j["name"], "rep": rep, "start": utc(), "load_before": list(os.getloadavg())}
    outpath = os.path.join(outdir, "out", f"{j['name']}-{rep}.txt")
    jcwd = j.get("cwd") or (os.path.join(outdir, "inputs") if j["query"] else cwd)
    rec.update(measure(job_argv(j), jcwd, e, outpath, args.timeout))
    rec["load_after"] = list(os.getloadavg())
    rec["files_sha"] = {f: sha256_file(os.path.join(cwd, f))
                        for f in j["files"] if os.path.exists(os.path.join(cwd, f))}
    return rec


def summarize(recs, meta):
    out = {"meta": meta, "bench": {}}
    by = {}
    for r in recs:
        by.setdefault(r["bench"], []).append(r)
    for name, rs in by.items():
        ok = [r for r in rs if r["rc"] == 0]
        s = {"n": len(rs), "ok": len(ok), "rc": sorted({r["rc"] for r in rs}),
             "out_sha": sorted({r["out_sha"] for r in rs}),
             "files_sha": sorted({json.dumps(r["files_sha"], sort_keys=True) for r in rs})}
        for key in ("cpu_s", "user_s", "sys_s", "wall_s", "maxrss_kb", "instr", "cycles"):
            vals = [r[key] for r in ok if r.get(key) is not None]
            if vals:
                s[key] = statistics.median(vals)
                s[key + "_min"], s[key + "_max"] = min(vals), max(vals)
        loads = [r["load_before"][1] for r in rs] + [r["load_after"][1] for r in rs]
        s["load5_median"] = statistics.median(loads)
        s["load5_min"], s["load5_max"] = min(loads), max(loads)
        s["counter_running_min"] = min((r["counter_running"] or 0) for r in rs)
        out["bench"][name] = s
    return out


def fmt(v, digits=2):
    return "-" if v is None else f"{v:.{digits}f}"


def table(summary):
    rows = ["| benchmark | CPU s (user+sys) | wall s | max RSS MB | instructions (1e9) "
            "| own work (1e6) | cycles (1e9) | load (5-min) | runs | output |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, s in summary["bench"].items():
        if "cpu_s" not in s:
            rows.append(f"| {name} | failed rc={s['rc']} | | | | | | | {s['ok']}/{s['n']} | |")
            continue
        own = own_instr(summary, name)
        rows.append(
            f"| {name} | {fmt(s['cpu_s'])} ({fmt(s['cpu_s_min'])}-{fmt(s['cpu_s_max'])}) "
            f"| {fmt(s['wall_s'])} | {fmt(s['maxrss_kb'] / 1024, 0)} "
            f"| {fmt(s.get('instr') and s['instr'] / 1e9, 3)} "
            f"| {fmt(own / 1e6, 2) if own is not None else ''} "
            f"| {fmt(s.get('cycles') and s['cycles'] / 1e9, 2)} "
            f"| {fmt(s['load5_median'], 0)} ({fmt(s['load5_min'], 0)}-{fmt(s['load5_max'], 0)}) "
            f"| {s['ok']}/{s['n']} | {','.join(s['out_sha'])} |")
    return "\n".join(rows)


def pct(new, old):
    return None if not old or new is None else 100.0 * (new - old) / old


def own_instr(summary, name):
    """Instructions of `name` minus those of its startup benchmark, or None."""
    b = summary["bench"]
    st = STARTUP_OF.get(name)
    if st is None or b.get(name, {}).get("instr") is None or b.get(st, {}).get("instr") is None:
        return None
    return b[name]["instr"] - b[st]["instr"]


def compare(base, new, threshold=5.0):
    """A table, the verdict of each benchmark and a one-line summary. A benchmark is SLOWER or
    FASTER when its instructions (or own work) move by more than `threshold` percent, or its CPU
    time does, the ranges of at least 3 repetitions on each side do not overlap, and either both
    summaries come from one ab run or one of them has no instruction counts."""
    rows = ["| benchmark | CPU s base | CPU s new | CPU Δ% | instr Δ% | own instr Δ% | wall Δ% "
            "| RSS Δ% | output | verdict |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    verdicts, lines = {}, []
    # CPU time follows the load of the machine, which only the two sides of one ab run share
    same_load = base["meta"].get("ab_run") is not None and \
        base["meta"].get("ab_run") == new["meta"].get("ab_run")
    for name, n in new["bench"].items():
        b = base["bench"].get(name)
        if not b or "cpu_s" not in b or "cpu_s" not in n:
            rows.append(f"| {name} | {fmt(b and b.get('cpu_s'))} | {fmt(n.get('cpu_s'))} "
                        "| | | | | | | n/a |")
            continue
        dc, di = pct(n["cpu_s"], b["cpu_s"]), pct(n.get("instr"), b.get("instr"))
        do = pct(own_instr(new, name), own_instr(base, name))
        dw, dm = pct(n["wall_s"], b["wall_s"]), pct(n["maxrss_kb"], b["maxrss_kb"])
        # one or two repetitions say nothing about the spread of CPU time under load
        ranged = (same_load or di is None) and n["ok"] >= 3 and b["ok"] >= 3
        apart_slow = ranged and n["cpu_s_min"] > b["cpu_s_max"]
        apart_fast = ranged and n["cpu_s_max"] < b["cpu_s_min"]
        dc0 = dc or 0.0
        if (dc0 > threshold and apart_slow) or (di is not None and di > threshold) or \
                (do is not None and do > threshold):
            v = "SLOWER"
        elif (dc0 < -threshold and apart_fast) or (di is not None and di < -threshold) or \
                (do is not None and do < -threshold):
            v = "FASTER"
        else:
            v = "same"
        same = "same" if n["out_sha"] == b["out_sha"] else "DIFFERS"
        verdicts[name] = v
        rows.append(f"| {name} | {fmt(b['cpu_s'])} | {fmt(n['cpu_s'])} | {fmt(dc, 1)} "
                    f"| {fmt(di, 1)} | {fmt(do, 1) if do is not None else ''} | {fmt(dw, 1)} "
                    f"| {fmt(dm, 1)} | {same} | {v} |")
        own = f" (own work {fmt(do, 1)}%)" if do is not None else ""
        lines.append(f"{name} CPU {fmt(b['cpu_s'])}->{fmt(n['cpu_s'])} s ({fmt(dc, 1)}%), "
                     f"instr {fmt(di, 1)}%{own}, output {same}, {v}")
    load = [x["load5_median"] for s in (base, new) for x in s["bench"].values()
            if "load5_median" in x]
    worst = ("SLOWER" if "SLOWER" in verdicts.values()
             else "FASTER" if "FASTER" in verdicts.values() else "same")
    line = (f"{new['meta'].get('suite', '?')} {base['meta'].get('label', '?')}->"
            f"{new['meta'].get('label', '?')} (median of {new['meta'].get('reps', '?')}, load "
            f"{fmt(min(load), 0) if load else '-'}-{fmt(max(load), 0) if load else '-'}): "
            + "; ".join(lines) + f"; verdict {worst} (threshold {threshold:g}%)")
    return "\n".join(rows), verdicts, line


def parse_cores(spec):
    cores = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cores.update(range(int(lo), int(hi or lo) + 1))
    return cores


def do_run(args, trees, interleave=False):
    suite = args.suite
    label = args.label or os.path.basename(trees[-1].rstrip("/"))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = os.path.join(os.path.abspath(args.out),
                          f"{label}-{suite}{'-ab' if interleave else ''}-{stamp}")
    for d in ("out", "inputs", "cwd"):
        os.makedirs(os.path.join(outdir, d))
    logf = open(os.path.join(outdir, "log.txt"), "a")

    def log(msg):
        line = f"{utc()} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    if args.cores:
        if not hasattr(os, "sched_setaffinity"):
            sys.exit("bench.py: --cores needs Linux")
        os.sched_setaffinity(0, parse_cores(args.cores))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    data = os.path.abspath(args.data)
    plans = [(tree,) + plan(suite, tree, args, env, log, data) for tree in trees]
    inputs = check_data(data, sorted({f for _, _, jobs in plans for j in jobs for f in j["data"]}),
                        not args.no_data_check, log)
    if inputs is None:
        return 2
    if any(j["query"] for _, _, jobs in plans for j in jobs):
        query = os.path.join(outdir, "inputs", QUERY_NAME)
        subprocess.run([sys.executable, "-B", GEN_QUERY, query] + [str(d) for d in QUERY_DIMS],
                       check=True)
        inputs[QUERY_NAME] = sha256_file(query)
    metas = []
    for tree, build, jobs in plans:
        if not args.no_check and not check_built(tree, jobs, env, log):
            return 2
        bins = {exe: sha256_file(os.path.join(build, "bin", exe))
                for exe in sorted({j["exe"] for j in jobs if j["kind"] == "run"})}
        digest, nfiles = source_digest(tree)
        with open(os.path.join(tree, "lean-toolchain")) as f:
            toolchain = f.read().strip()
        meta = {"tree": tree, "build": build, "suite": suite,
                "label": os.path.basename(tree.rstrip("/")) if interleave else label,
                "start": utc(), "bench_py_sha256_16": sha256_file(os.path.abspath(__file__)),
                "reps": args.reps, "binaries_sha256_16": bins, "source_digest": digest,
                "source_files": nfiles, "git": git_state(tree), "lean_toolchain": toolchain,
                "machine": machine(), "counters": counters(),
                "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                else None, "nice": os.nice(0), "data_dir": data, "inputs_sha256_16": inputs,
                "runner_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "ab_run": os.path.basename(outdir) if interleave else None, "jobs": jobs}
        log(f"tree {tree} build {build} source {digest} ({nfiles} files) binaries {bins} "
            f"counters {meta['counters']}")
        metas.append(meta)
    cwd = os.path.join(outdir, "cwd")
    recs = [[] for _ in plans]
    order = []
    if interleave:
        names = [j["name"] for j in plans[0][2] if any(x["name"] == j["name"] for x in plans[1][2])]
        order = [(t, name, rep) for name in names for rep in range(args.reps)
                 for t in range(len(plans))]
    else:
        order = [(0, j["name"], rep) for j in plans[0][2] for rep in range(args.reps)]
    for t, name, rep in order:
        j = next(x for x in plans[t][2] if x["name"] == name)
        tag = "AB"[t] if interleave else ""
        rec = run_rep(j, f"{tag}{rep}", outdir, cwd, env, args)
        rec["tree"] = plans[t][0]
        recs[t].append(rec)
        with open(os.path.join(outdir, "reps.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        log(f"{tag}{name} rep {rep}: rc={rec['rc']} cpu={rec['cpu_s']} wall={rec['wall_s']} "
            f"instr={rec['instr']} rss={rec['maxrss_kb']} out={rec['out_sha']}")
    sums = []
    for t, meta in enumerate(metas):
        meta["end"] = utc()
        s = summarize(recs[t], meta)
        sums.append(s)
        suffix = "AB"[t] if interleave else ""
        with open(os.path.join(outdir, f"summary{suffix}.json"), "w") as f:
            json.dump(s, f, indent=1)
        with open(os.path.join(outdir, f"summary{suffix}.md"), "w") as f:
            f.write(table(s) + "\n")
        log(f"summary {meta['tree']}:\n" + table(s))
    line = None
    if interleave:
        t, _, line = compare(sums[0], sums[1], args.threshold)
        with open(os.path.join(outdir, "compare.md"), "w") as f:
            f.write(t + "\n\n" + line + "\n")
        log("A/B compare (A = base):\n" + t)
    log(f"DONE {outdir}")
    if line:
        print(line, flush=True)
    return 0 if all(r["rc"] == 0 for rs in recs for r in rs) else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("run", "ab"):
        s = sub.add_parser(c, help="one checkout" if c == "run" else "two checkouts, interleaved")
        s.add_argument("trees", nargs="+", metavar="TREE")
        s.add_argument("--suite", choices=["quick", "cpu", "elab", "full"], default="quick")
        s.add_argument("--reps", type=int, help="repetitions per command (quick 3, else 5)")
        s.add_argument("--only", help="comma-separated benchmark names")
        s.add_argument("--label", help="name of the run (default: the last checkout's name)")
        s.add_argument("--data", default="data/real",
                       help="directory of download_example_data.py's output (default data/real)")
        s.add_argument("--out", default="_out/bench", help="results directory (_out/bench)")
        s.add_argument("--cores", help="run on these CPU cores only, for example 2,3 or 4-7")
        s.add_argument("--timeout", type=float, default=1200, help="seconds per repetition")
        s.add_argument("--threshold", type=float, default=5.0, help="percent for a verdict")
        s.add_argument("--modules", help="elab modules name=path,... instead of the set")
        s.add_argument("--no-check", action="store_true",
                       help="skip the check that the executables and modules are built")
        s.add_argument("--no-data-check", action="store_true",
                       help="run on data that differ from SHA256SUMS")
    c = sub.add_parser("compare", help="compare two summary.json files")
    c.add_argument("base")
    c.add_argument("new")
    c.add_argument("--threshold", type=float, default=5.0)
    sh = sub.add_parser("show", help="print the table of a summary.json file")
    sh.add_argument("summary")
    args = ap.parse_args()
    if args.cmd in ("run", "ab"):
        if args.reps is None:
            args.reps = 3 if args.suite == "quick" else 5
        if args.reps < 1:
            ap.error("--reps must be at least 1")
        trees = [os.path.abspath(t) for t in args.trees]
        if len(trees) != (2 if args.cmd == "ab" else 1):
            ap.error(f"{args.cmd} takes {'two checkouts' if args.cmd == 'ab' else 'one checkout'}")
        for tree in trees:
            if not os.path.isfile(os.path.join(tree, "lakefile.lean")):
                ap.error(f"{tree} is not a TorchLean checkout")
        sys.exit(do_run(args, trees, interleave=args.cmd == "ab"))
    if args.cmd == "compare":
        with open(args.base) as fb, open(args.new) as fn:
            t, _, line = compare(json.load(fb), json.load(fn), args.threshold)
        print(t + "\n\n" + line)
        sys.exit(0)
    with open(args.summary) as f:
        print(table(json.load(f)))


if __name__ == "__main__":
    main()
