#!/usr/bin/env python3
"""Check large frees, timed purging, and concurrent wakeups in Lean's pinned allocator."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time

PROBE = r"""
#include <lean/mimalloc.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <time.h>
#include <atomic>
#include <thread>
static size_t rss() {
  FILE* f = fopen("/proc/self/statm", "r");
  size_t size = 0, resident = 0;
  if (!f || fscanf(f, "%zu %zu", &size, &resident) != 2) abort();
  fclose(f);
  return resident * static_cast<size_t>(sysconf(_SC_PAGESIZE));
}
static int arena_probe() {
  const size_t before = rss();
  const size_t bytes = 4ull * 1024 * 1024;
  mi_arena_id_t id;
  if (mi_reserve_os_memory_ex(512ull*1024*1024, true, false, true, &id)) return 2;
  mi_heap_t* heap = mi_heap_new_in_arena(id);
  if (!heap) return 3;
  void* blocks[100];
  for (size_t i = 0; i < 100; ++i) {
    blocks[i] = mi_heap_malloc(heap, bytes);
    if (!blocks[i] || !mi_arena_contains(id, blocks[i])) return 4;
    memset(blocks[i], 0x5a, bytes);
  }
  size_t arena_size = 0;
  void* arena = mi_arena_area(id, &arena_size);
  const size_t allocated = rss();
  for (size_t i = 0; i < 100; ++i) mi_free(blocks[i]);
  const size_t freed = rss();
  // The test sets a 100ms arena deadline. Ordinary collection after that
  // deadline must purge the freed pages without a forced collection.
  usleep(350000);
  mi_collect(false);
  const size_t delayed = rss();
  mi_collect(true);
  printf("{\"before\":%zu,\"allocated\":%zu,\"freed\":%zu,"
         "\"delayed\":%zu,\"forced\":%zu,\"arenaSize\":%zu,\"arenaExists\":%d}\n",
         before, allocated, freed, delayed, rss(), arena_size, arena != nullptr);
  mi_heap_delete(heap);
  return 0;
}
static unsigned long long ms() {
  timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000ull + t.tv_nsec / 1000000;
}
static int staggered_probe() {
  const size_t before = rss();
  mi_arena_id_t ids[2];
  mi_heap_t* heaps[2];
  void* blocks[2][50];
  for (int arm=0; arm<2; ++arm) {
    if (mi_reserve_os_memory_ex(512ull*1024*1024, true, false, true, &ids[arm])) return 2;
    heaps[arm] = mi_heap_new_in_arena(ids[arm]);
    if (!heaps[arm]) return 3;
    for (int i=0; i<50; ++i) {
      blocks[arm][i] = mi_heap_malloc(heaps[arm],4ull*1024*1024);
      if (!blocks[arm][i]) return 4;
      memset(blocks[arm][i],0x5a,4ull*1024*1024);
    }
  }
  const size_t allocated = rss();
  const auto started = ms();
  for (int i=0; i<50; ++i) mi_free(blocks[0][i]);
  usleep(250000);
  for (int i=0; i<50; ++i) mi_free(blocks[1][i]);
  const auto second_freed = ms();
  usleep(250000);
  mi_collect(false);
  const size_t first_due = rss();
  const auto first_collected = ms();
  usleep(450000);
  mi_collect(false);
  const size_t both_due = rss();
  const auto second_collected = ms();
  mi_collect(true);
  printf("{\"before\":%zu,\"allocated\":%zu,\"firstDue\":%zu,\"bothDue\":%zu,"
         "\"forced\":%zu,\"secondFreedMs\":%llu,\"firstCollectedMs\":%llu,"
         "\"secondCollectedMs\":%llu}\n",
         before,allocated,first_due,both_due,rss(),second_freed-started,
         first_collected-started,second_collected-started);
  return 0;
}

static int concurrent_probe() {
  const size_t before = rss();
  std::atomic<int> ready{0}, freed{0};
  std::atomic<bool> start{false}, finish{false};
  std::thread workers[4];
  for (int arm=0; arm<4; ++arm) {
    workers[arm] = std::thread([&, arm]() {
      mi_thread_init();
      mi_arena_id_t id;
      if (mi_reserve_os_memory_ex(512ull*1024*1024, true, false, true, &id)) { fprintf(stderr,"reserve %d failed\n",arm); abort(); }
      mi_heap_t* heap = mi_heap_new_in_arena(id);
      if (!heap) { fprintf(stderr,"heap %d failed\n",arm); abort(); }
      unsigned char* blocks[24];
      for (int i=0; i<24; ++i) {
        blocks[i] = (unsigned char*)mi_heap_malloc(heap, 4ull*1024*1024);
        if (!blocks[i]) { fprintf(stderr,"block %d/%d failed\n",arm,i); abort(); }
        memset(blocks[i], 0x5a + arm, 4ull*1024*1024);
      }
      ready.fetch_add(1);
      while (!start.load()) usleep(1000);
      usleep(70000 * arm);
      for (int i=0; i<24; ++i) {
        for (size_t byte=0; byte<4ull*1024*1024; byte+=4096) {
          if (blocks[i][byte] != 0x5a + arm) { fprintf(stderr,"corrupt %d/%d/%zu: %u\n",arm,i,byte,blocks[i][byte]); abort(); }
        }
        mi_free(blocks[i]);
      }
      freed.fetch_add(1);
      // Keep thread teardown from forcing collection before the observation.
      while (!finish.load()) usleep(1000);
      mi_heap_delete(heap);
    });
  }
  while (ready.load()!=4) { mi_collect(false); usleep(1000); }
  const size_t allocated = rss();
  start.store(true);
  const auto started = ms();
  while (ms() - started < 1200) { mi_collect(false); usleep(20000); }
  if (freed.load() != 4) { fprintf(stderr,"freed incomplete %d\n",freed.load()); abort(); }
  const size_t delayed = rss();
  mi_collect(true);
  const size_t forced = rss();
  finish.store(true);
  for (auto& worker: workers) worker.join();
  printf("{\"before\":%zu,\"allocated\":%zu,\"delayed\":%zu,\"forced\":%zu}\n",
         before, allocated, delayed, forced);
  return 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "concurrent") == 0) return concurrent_probe();
  if (argc > 1 && strcmp(argv[1], "staggered") == 0) return staggered_probe();
  if (argc > 1) return arena_probe();
  const size_t bytes = 3072ull * 3072ull * 8 + 24;
  const size_t before = rss();
  size_t first = 0;
  void* previous[2] = {nullptr, nullptr};
  for (int i = 0; i < 20; ++i) {
    void* next[2] = {mi_malloc(bytes), mi_malloc(bytes)};
    if (!next[0] || !next[1]) return 2;
    memset(next[0], i + 1, bytes);
    memset(next[1], i + 1, bytes);
    mi_free(previous[0]);
    mi_free(previous[1]);
    previous[0] = next[0]; previous[1] = next[1];
    mi_collect(true);
    if (i == 0) first = rss();
  }
  const size_t last = rss();
  mi_free(previous[0]); mi_free(previous[1]);
  mi_collect(true);
  printf("{\"before\":%zu,\"first\":%zu,\"last\":%zu,\"released\":%zu}\n",
         before, first, last, rss());
}
"""


WAKEUP_PROBE = r"""
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


def wakeup_checks(lean_prefix, output):
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
            probe.write_text(WAKEUP_PROBE)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-prefix", type=Path, required=True)
    parser.add_argument("--object", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    limit = 64 * 1024 * 1024
    results = {}
    with tempfile.TemporaryDirectory(prefix="torchlean-allocator-check-") as directory:
        directory = Path(directory)
        source = directory / "probe.cpp"
        source.write_text(PROBE)
        shared = directory / "libtorchlean_allocator.so"
        subprocess.run(["c++", "-shared", str(args.object), "-o", str(shared)],
                       check=True, timeout=60)
        for name, obj in (("installed", args.lean_prefix / "lib/lean/libleanrt.a"),
                          ("private", args.object), ("private-shared", shared)):
            binary = directory / name
            subprocess.run([
                "c++", "-O2", "-std=c++17", "-I" + str(args.lean_prefix / "include"),
                str(source), str(obj), "-L" + str(args.lean_prefix / "lib"),
                "-Wl,-rpath," + str(args.lean_prefix / "lib"),
                "-Wl,-rpath," + str(directory),
                "-lc++", "-lc++abi", "-lunwind", "-lm", "-ldl", "-lpthread",
                "-o", str(binary),
            ], check=True, timeout=60)
            env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
            env["MIMALLOC_DISALLOW_ARENA_ALLOC"] = "1"
            run = subprocess.run([str(binary)], env=env, check=True, capture_output=True,
                                 text=True, timeout=60)
            result = json.loads(run.stdout)
            result["bounded"] = result["last"] <= result["first"] + limit
            result["releasedAll"] = result["released"] <= result["before"] + limit
            result["objectSha256"] = hashlib.sha256(obj.read_bytes()).hexdigest()
            arena_env = {k: v for k, v in os.environ.items() if not k.startswith("MIMALLOC_")}
            arena_env.update(MIMALLOC_PURGE_DELAY="100", MIMALLOC_ARENA_PURGE_MULT="1")
            arena_run = subprocess.run([str(binary), "arena"], env=arena_env, check=True,
                                       capture_output=True, text=True, timeout=30)
            arena = json.loads(arena_run.stdout)
            arena["purged"] = arena["delayed"] <= arena["before"] + limit
            arena["forcedReleased"] = arena["forced"] <= arena["before"] + limit
            arena["exercised"] = (arena["arenaExists"] == 1 and arena["arenaSize"] > 0 and
                                  arena["allocated"] >= arena["before"] + 350*1024*1024)
            result["arena"] = arena
            arena_env["MIMALLOC_PURGE_DELAY"] = "400"
            staggered_run = subprocess.run([str(binary), "staggered"], env=arena_env, check=True,
                                           capture_output=True, text=True, timeout=30)
            staggered = json.loads(staggered_run.stdout)
            staggered["purged"] = staggered["bothDue"] <= staggered["before"] + limit
            staggered["forcedReleased"] = staggered["forced"] <= staggered["before"] + limit
            staggered["exercised"] = (
                staggered["allocated"] >= staggered["before"] + 350*1024*1024 and
                staggered["secondFreedMs"] < 400 and
                400 <= staggered["firstCollectedMs"] < staggered["secondFreedMs"] + 400 and
                staggered["secondCollectedMs"] >= staggered["firstCollectedMs"] + 400)
            result["staggered"] = staggered
            arena_env["MIMALLOC_PURGE_DELAY"] = "100"
            concurrent_run = subprocess.run([str(binary), "concurrent"], env=arena_env, check=True,
                                            capture_output=True, text=True, timeout=30)
            concurrent = json.loads(concurrent_run.stdout)
            concurrent["purged"] = concurrent["delayed"] <= concurrent["before"] + limit
            concurrent["forcedReleased"] = concurrent["forced"] <= concurrent["before"] + limit
            concurrent["exercised"] = concurrent["allocated"] >= concurrent["before"] + 350*1024*1024
            result["concurrent"] = concurrent
            results[name] = result
            for label, completed in [
                ("large", run), ("arena", arena_run),
                ("staggered", staggered_run), ("concurrent", concurrent_run),
            ]:
                (directory / f"{name}-{label}.stdout").write_text(completed.stdout)
                (directory / f"{name}-{label}.stderr").write_text(completed.stderr)
        try:
            controlled = wakeup_checks(args.lean_prefix, directory / "controlled")
        except Exception as error:
            controlled = {"status": "failed", "error": repr(error)}
            recorded = directory / "controlled/result.json"
            if recorded.exists():
                controlled["diagnostic"] = json.loads(recorded.read_text())
        artifacts = {
            str(path.relative_to(directory)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size, "mode": path.stat().st_mode & 0o777,
            }
            for path in sorted(directory.rglob("*")) if path.is_file()
        }
        archive = args.output.with_suffix(".artifacts.tar.gz")
        with tarfile.open(archive, "x:gz", compresslevel=1) as bundle:
            for name in artifacts:
                bundle.add(directory / name, arcname=name, recursive=False)
        seen = {}
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle:
                if not member.isfile() or member.name in seen:
                    raise RuntimeError("Unexpected allocator artifact")
                seen[member.name] = {
                    "sha256": hashlib.file_digest(bundle.extractfile(member), "sha256").hexdigest(),
                    "bytes": member.size, "mode": member.mode,
                }
        if seen != artifacts:
            raise RuntimeError("Retained allocator artifacts changed")
        args.output.with_suffix(".artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    passed = (all(result["arena"]["exercised"] and result["arena"]["forcedReleased"] and
                      result["staggered"]["exercised"] and result["staggered"]["forcedReleased"] and
                      result["concurrent"]["exercised"] and result["concurrent"]["forcedReleased"]
                      for result in results.values())
              and all(results[name]["bounded"] and results[name]["releasedAll"] and
                      results[name]["arena"]["purged"] and results[name]["staggered"]["purged"] and
                      results[name]["concurrent"]["purged"]
                      for name in ("installed", "private", "private-shared"))
              and controlled["status"] == "passed")
    results["controlled"] = controlled
    results["artifacts"] = {
        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "files": len(artifacts), "archive": str(archive),
    }
    results["status"] = "passed" if passed else "failed"
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
