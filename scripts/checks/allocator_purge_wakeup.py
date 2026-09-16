"""Deterministically check collection against a concurrent arena free.

The diagnostic includes the pinned allocator implementation with observation
hooks. They coordinate two threads without changing allocator state. A separate
case checks collection across eight arenas. The normal object contains no hooks.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


PROBE = r"""
#include "src/static.c"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <unistd.h>

static std::atomic<int> stage{0};
static std::atomic<int> freed_blocks{0};
static mi_arena_t* target = nullptr;
static bool armed = false;
static int selected_point = 2;
static unsigned hook_count = 0;
static mi_msecs_t scheduled = 0, scheduled_global = 0;

static void require(bool okay, const char* message) {
  if (!okay) { fprintf(stderr, "%s\n", message); abort(); }
}

static size_t rss() {
  FILE* file = fopen("/proc/self/statm", "r");
  size_t size = 0, resident = 0;
  require(file && fscanf(file, "%zu %zu", &size, &resident) == 2, "RSS read");
  fclose(file);
  return resident * static_cast<size_t>(sysconf(_SC_PAGESIZE));
}

static void wait_stage(int expected) {
  const mi_msecs_t deadline = _mi_clock_now() + 10000;
  while (stage.load(std::memory_order_acquire) != expected) {
    require(_mi_clock_now() < deadline, "thread coordination timeout");
    usleep(1000);
  }
}

static void wait_after(mi_msecs_t deadline) {
  require(deadline > 0 && deadline <= _mi_clock_now() + 2000, "invalid deadline");
  while (_mi_clock_now() <= deadline + 40) usleep(1000);
}

static void* worker(void*) {
  mi_thread_init();
  mi_arena_id_t id;
  require(mi_reserve_os_memory_ex(512ull*1024*1024, true, false, true, &id) == 0,
          "target arena reserve");
  mi_heap_t* heap = mi_heap_new_in_arena(id);
  require(heap != nullptr, "target heap");
  void* blocks[100];
  for (int i = 0; i < 100; ++i) {
    blocks[i] = mi_heap_malloc(heap, 4ull*1024*1024);
    require(blocks[i] && mi_arena_contains(id, blocks[i]), "target membership");
    memset(blocks[i], 0x5a, 4ull*1024*1024);
  }
  target = _mi_arena_from_id(id);
  stage.store(1, std::memory_order_release);
  wait_stage(2);
  for (void* block : blocks) {
    mi_free(block);
    freed_blocks.fetch_add(1, std::memory_order_relaxed);
  }
  mi_heap_collect(heap, false);
  stage.store(3, std::memory_order_release);
  // Keep heap destruction and thread teardown after all RSS observations.
  wait_stage(4);
  mi_heap_delete(heap);
  mi_thread_done();
  return nullptr;
}

static void torchlean_purge_checkpoint(int point, mi_subproc_t* subproc,
                                      bool all_visited, bool any_purged) {
  if (!armed || point != selected_point) return;
  armed = false;
  ++hook_count;
  if (point == 2)
    require(all_visited && !any_purged, "collector must finish an empty scan");
  require(subproc == target->subproc, "different subprocess");
  require(mi_atomic_loadi64_acquire(&target->purge_expire) == 0,
          "target already scheduled before free");
  stage.store(2, std::memory_order_release);
  wait_stage(3);
  scheduled = mi_atomic_loadi64_acquire(&target->purge_expire);
  scheduled_global = mi_atomic_loadi64_acquire(&subproc->purge_expire);
  require(scheduled > 0 && scheduled_global > 0, "free did not publish timers");
}

static void budget_check() {
  const size_t before = rss();
  mi_heap_t* heaps[8];
  mi_arena_t* arenas[8];
  void* blocks[8][16];
  for (int i = 0; i < 8; ++i) {
    mi_arena_id_t id;
    require(mi_reserve_os_memory_ex(128ull*1024*1024, true, false, true, &id) == 0,
            "budget arena reserve");
    heaps[i] = mi_heap_new_in_arena(id);
    arenas[i] = _mi_arena_from_id(id);
    require(heaps[i] != nullptr, "budget heap");
    for (int j = 0; j < 16; ++j) {
      blocks[i][j] = mi_heap_malloc(heaps[i], 4ull*1024*1024);
      require(blocks[i][j] && mi_arena_contains(id, blocks[i][j]), "budget membership");
      memset(blocks[i][j], 0x6a, 4ull*1024*1024);
    }
  }
  const size_t allocated = rss();
  mi_msecs_t latest = 0;
  for (int i = 0; i < 8; ++i) {
    for (void* block : blocks[i]) mi_free(block);
    mi_heap_collect(heaps[i], false);
    const mi_msecs_t expires = mi_atomic_loadi64_acquire(&arenas[i]->purge_expire);
    require(expires > 0, "budget free was not scheduled");
    if (expires > latest) latest = expires;
  }
  wait_after(latest);
  mi_subproc_t* subproc = arenas[0]->subproc;
  const size_t count = mi_atomic_load_acquire(&subproc->arena_count);
  require(count < 28, "too many arenas to exercise purge limit");
  mi_collect(false);
  size_t pending = 0;
  for (mi_arena_t* arena : arenas)
    if (mi_atomic_loadi64_acquire(&arena->purge_expire) != 0) ++pending;
  const mi_msecs_t timer_after = mi_atomic_loadi64_acquire(&subproc->purge_expire);
  const mi_msecs_t observed_now = _mi_clock_now();
  for (int i = 0; i < 8; ++i) mi_collect(false);
  size_t remaining = 0;
  for (mi_arena_t* arena : arenas)
    if (mi_atomic_loadi64_acquire(&arena->purge_expire) != 0) ++remaining;
  const size_t released = rss();
  printf("{\"before\":%zu,\"allocated\":%zu,\"released\":%zu,"
         "\"arenaCount\":%zu,\"pendingAfterFirst\":%zu,\"remaining\":%zu,"
         "\"timerAfter\":%lld,\"observedNow\":%lld}\n",
         before, allocated, released, count, pending, remaining,
         (long long)timer_after, (long long)observed_now);
  for (mi_heap_t* heap : heaps) mi_heap_delete(heap);
}

int main(int argc, char** argv) {
  mi_process_init();
  require(argc == 2, "expected checkpoint");
  selected_point = atoi(argv[1]);
  if (selected_point == 3) { budget_check(); return 0; }
  require(selected_point >= 0 && selected_point <= 2, "invalid checkpoint");
  const size_t before = rss();
  pthread_t thread;
  require(pthread_create(&thread, nullptr, worker, nullptr) == 0, "worker creation");
  wait_stage(1);
  const size_t allocated = rss();
  mi_subproc_t* subproc = target->subproc;

  // A real free and ordinary purge leave a nonzero subprocess timer.
  mi_arena_id_t seed_id;
  require(mi_reserve_os_memory_ex(512ull*1024*1024, true, false, true, &seed_id) == 0,
          "seed arena reserve");
  mi_heap_t* seed = mi_heap_new_in_arena(seed_id);
  require(seed != nullptr, "seed heap");
  void* blocks[10];
  for (int i = 0; i < 10; ++i) {
    blocks[i] = mi_heap_malloc(seed, 4ull*1024*1024);
    require(blocks[i] && mi_arena_contains(seed_id, blocks[i]), "seed membership");
    memset(blocks[i], 0x3c, 4ull*1024*1024);
  }
  for (void* block : blocks) mi_free(block);
  mi_heap_collect(seed, false);
  mi_arena_t* seed_arena = _mi_arena_from_id(seed_id);
  wait_after(mi_atomic_loadi64_acquire(&seed_arena->purge_expire));
  mi_collect(false);
  require(mi_atomic_loadi64_acquire(&seed_arena->purge_expire) == 0,
          "seed did not purge");
  const mi_msecs_t primed = mi_atomic_loadi64_acquire(&subproc->purge_expire);
  wait_after(primed);

  armed = true;
  mi_collect(false);
  require(hook_count == 1 && !armed && freed_blocks.load() == 100,
          "controlled interleaving was not exercised");
  const mi_msecs_t global_after = mi_atomic_loadi64_acquire(&subproc->purge_expire);
  const mi_msecs_t target_after = mi_atomic_loadi64_acquire(&target->purge_expire);
  const size_t freed = rss();
  wait_after(scheduled);
  mi_collect(false);
  const size_t delayed = rss();
  const mi_msecs_t target_delayed = mi_atomic_loadi64_acquire(&target->purge_expire);
  mi_collect(true);
  const size_t forced = rss();
  printf("{\"before\":%zu,\"allocated\":%zu,\"freed\":%zu,"
         "\"delayed\":%zu,\"forced\":%zu,\"hookCount\":%u,\"freedBlocks\":%d,"
         "\"primed\":%lld,\"scheduled\":%lld,\"scheduledGlobal\":%lld,"
         "\"globalAfter\":%lld,\"targetAfter\":%lld,\"targetDelayed\":%lld}\n",
         before, allocated, freed, delayed, forced, hook_count, freed_blocks.load(),
         (long long)primed, (long long)scheduled, (long long)scheduled_global,
         (long long)global_after, (long long)target_after, (long long)target_delayed);
  stage.store(4, std::memory_order_release);
  require(pthread_join(thread, nullptr) == 0, "worker join");
  mi_heap_delete(seed);
  return 0;
}
"""


def allocator_module():
    path = Path(__file__).resolve().parents[1] / "lean_allocator.py"
    spec = importlib.util.spec_from_file_location("torchlean_allocator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_checks(lean_prefix, output):
    """Compile both implementations and preserve their sources and raw results."""
    allocator = allocator_module()
    if allocator.sha(lean_prefix / "include/lean/mimalloc.h") != allocator.HEADER_SHA256:
        raise RuntimeError("wakeup check requires the pinned Lean allocator header")
    output.mkdir(parents=True, exist_ok=False)
    archive = allocator.source_archive()
    compiler = Path(shutil.which("c++")).resolve()
    clean_env = {
        key: value for key, value in os.environ.items()
        if not key.startswith("MIMALLOC_") and key not in {"LD_PRELOAD", "LD_LIBRARY_PATH"}
    }
    clean_env.update(MIMALLOC_PURGE_DELAY="100", MIMALLOC_ARENA_PURGE_MULT="1")
    results = {"status": "running", "variants": {}, "compilerSha256": allocator.sha(compiler)}
    commands = []

    def command(label, argv, directory, timeout):
        row = {"label": label, "argv": [str(arg) for arg in argv],
               "cwd": str(directory), "startedUnix": time.time()}
        with (output / (label + ".stdout")).open("xb") as stdout, \
                (output / (label + ".stderr")).open("xb") as stderr:
            proc = subprocess.run(row["argv"], cwd=directory, env=clean_env,
                                  stdout=stdout, stderr=stderr, timeout=timeout)
        row.update(exitCode=proc.returncode, finishedUnix=time.time())
        commands.append(row)
        (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if proc.returncode:
            raise RuntimeError(label + " failed")

    try:
        for name in ["upstream", "patched"]:
            directory = output / name
            directory.mkdir()
            source = allocator.extract_source(archive, directory)
            arena = source / "src/arena.c"
            text = arena.read_text()
            if name == "patched":
                text = allocator.patch_arena_source(text)
            implementation_sha = hashlib.sha256(text.encode()).hexdigest()
            declaration = '#include "bitmap.h"'
            endpoint = ("    if (all_visited && !any_purged) {" if name == "upstream"
                        else "    if (!all_visited || any_purged) {")
            reset = ("    // increase global expire: at most one purge per delay cycle"
                     if name == "upstream" else "    // Acquire published arena updates")
            scan = "    const size_t arena_start ="
            if any(text.count(token) != 1 for token in [declaration, endpoint, reset, scan]):
                raise RuntimeError("diagnostic hook does not apply exactly once")
            text = text.replace(declaration, declaration + "\n\n"
                                "static void torchlean_purge_checkpoint("
                                "int, mi_subproc_t*, bool, bool);")
            text = text.replace(reset, "    torchlean_purge_checkpoint("
                                "0, subproc, false, false);\n" + reset)
            text = text.replace(scan, "    torchlean_purge_checkpoint("
                                "1, subproc, false, false);\n" + scan)
            text = text.replace(endpoint, "    torchlean_purge_checkpoint("
                                "2, subproc, all_visited, any_purged);\n" + endpoint)
            arena.write_text(text)
            probe = source / "wakeup.cpp"
            probe.write_text(PROBE)
            binary = directory / "wakeup"
            flags = [flag for flag in allocator.compiler_flags(source) if flag != "-c"]
            command(name + "-build", [compiler, *flags, probe, "-lpthread", "-ldl", "-lm",
                                     "-o", binary], directory, 180)
            cases = {}
            for point, case in enumerate(["before-reset", "before-scan", "after-scan", "budget"]):
                label = name + "-" + case
                command(label, [binary, str(point)], directory, 30)
                row = json.loads((output / (label + ".stdout")).read_text())
                limit = 64 * 1024 * 1024
                if case == "budget":
                    checks = {
                        "exercised": row["allocated"] >= row["before"] + 450 * 1024 * 1024,
                        "limitedFirstPurge": 0 < row["pendingAfterFirst"] < 8,
                        "immediateRetry": 0 < row["timerAfter"] <= row["observedNow"],
                        "ordinaryPurge": row["remaining"] == 0,
                        "releasedPages": row["released"] <= row["before"] + limit,
                    }
                else:
                    checks = {
                        "interleaving": row["hookCount"] == 1 and row["freedBlocks"] == 100,
                        "exercised": row["allocated"] >= row["before"] + 350 * 1024 * 1024,
                        "scheduled": row["scheduled"] > 0 and row["scheduledGlobal"] > 0
                        and row["targetAfter"] == row["scheduled"],
                        "forcedRelease": row["forced"] <= row["before"] + limit,
                    }
                    if name == "upstream" and point == 2:
                        checks.update(
                            lostWakeup=row["globalAfter"] == 0,
                            pendingAfterDeadline=row["targetDelayed"] == row["scheduled"],
                            retainedPages=row["delayed"] >= row["before"] + 350 * 1024 * 1024)
                    else:
                        checks.update(wakeupPreserved=0 < row["globalAfter"] <= row["scheduled"],
                                      ordinaryPurge=row["targetDelayed"] == 0,
                                      releasedPages=row["delayed"] <= row["before"] + limit)
                cases[case] = {"measurements": row, "checks": checks,
                               "passed": all(checks.values())}
            results["variants"][name] = {
                "cases": cases, "passed": all(row["passed"] for row in cases.values()),
                "binarySha256": allocator.sha(binary), "probeSha256": allocator.sha(probe),
                "arenaImplementationSha256": implementation_sha,
                "arenaInstrumentedSha256": allocator.sha(arena),
            }
        results["status"] = ("passed" if all(row["passed"] for row in
                                             results["variants"].values()) else "failed")
    except BaseException as error:
        results.update(status="failed", error=repr(error))
        raise
    finally:
        (output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
