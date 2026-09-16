#!/usr/bin/env python3
"""Check large frees and timed arena purging on Lean 4.34's matching allocators."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile

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
        helper = Path(__file__).with_name("allocator_purge_wakeup.py")
        spec = importlib.util.spec_from_file_location("allocator_purge_wakeup", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        try:
            controlled = module.run_checks(args.lean_prefix, directory / "controlled")
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
