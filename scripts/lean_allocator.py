#!/usr/bin/env python3
"""Build Lean 4.34's mimalloc for TorchLean executables and shared libraries.

Lean 4.34 pins mimalloc 3.4.4, which includes the large-free size correction.
Its arena collector can still clear a concurrently published purge wakeup.
The patch below publishes arena deadlines before their wakeup, clears before
scanning, and preserves concurrent deadlines and the existing retry timing.
Its installed static allocator uses local-exec TLS, which cannot be linked
into TorchLean's shared libraries. Build the same source with PIC and
initial-exec TLS so both native link modes can use it.

Build a private object; never modify the installed Lean runtime. Lake links
this object before libleanrt.a. This serves compiled TorchLean executables;
it does not replace the allocator inside the Lean compiler or `#eval`.

Source is downloaded once and verified by SHA-256. For offline builds, set
TORCHLEAN_MIMALLOC_ARCHIVE to the same verified v3.4.4 source archive.
Review this source and header binding when updating the Lean toolchain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

SOURCE_URL = "https://codeload.github.com/microsoft/mimalloc/tar.gz/refs/tags/v3.4.4"
SOURCE_SHA256 = "8ba991a7266983bd5eefc36e140c24734f720fd9b1fd79ddaeff44ea85d16760"
HEADER_SHA256 = "b457b2365eb25d852efe02f0b5808c99f822b2934d6bcf9acf3a06254c8dcdd1"
OS_SHA256 = "e55a179cbbe6035133e0d76f5a09b0958044af9cc91b49174ca660505814454c"
ARENA_SHA256 = "0919ab2a7a69a90ff88b7cfcdb3d4c67a2bfc791b8a6617cd31b37b7b9295093"
ROOT = Path(__file__).resolve().parent.parent
ARENA_PATCHES = (
    ("""// Schedule a purge. This is usually delayed to avoid repeated decommit/commit calls.""",
     """// Publish preceding arena updates even when a wakeup is already pending.
// A successful same-value RMW joins the release sequence observed by the collector.
static void mi_arenas_schedule_purge(mi_subproc_t* subproc, mi_msecs_t expire) {
  if (expire == 0) expire = 1; // zero is reserved for no pending wakeup
  mi_msecs_t expected = mi_atomic_loadi64_relaxed(&subproc->purge_expire);
  mi_msecs_t desired;
  do {
    desired = (expected == 0 || expire < expected ? expire : expected);
  } while (!mi_atomic_casi64_strong_acq_rel(&subproc->purge_expire, &expected, desired));
}

// Schedule a purge. This is usually delayed to avoid repeated decommit/commit calls."""),
    ("""      // maybe set the global arenas expire as well (if it wasn't set already)
      mi_assert_internal(expire0==0);
      mi_atomic_casi64_strong_acq_rel(&arena->subproc->purge_expire, &expire0, expire);""",
     """      // Publish even if another arena already set the global expiration.
      mi_assert_internal(expire0==0);
      mi_arenas_schedule_purge(arena->subproc, expire);"""),
    ("""  const size_t max_arena = mi_arenas_get_count(subproc);
  if (max_arena == 0) return;

  // allow only one thread to purge at a time""",
     """  // allow only one thread to purge at a time"""),
    ("""    // increase global expire: at most one purge per delay cycle
    if (arenas_expire > now) { mi_atomic_storei64_release(&subproc->purge_expire, now + (delay/10)); }
    const size_t arena_start = tseq % max_arena;""",
     """    // Acquire published arena updates before scanning. A later publication
    // leaves a wakeup for the next collector; no end-of-scan clear can erase it.
    const mi_msecs_t previous_expire =
      mi_atomic_exchange_acq_rel(&subproc->purge_expire, (mi_msecs_t)0);
    const mi_msecs_t retry_expire = (previous_expire == 0 ? now :
      (previous_expire > now ? now + (delay/10) : previous_expire));
    const size_t max_arena = mi_atomic_load_acquire(&subproc->arena_count);
    const size_t arena_start = (max_arena == 0 ? 0 : tseq % max_arena);"""),
    ("""    if (all_visited && !any_purged) {
      mi_atomic_storei64_release(&subproc->purge_expire, (mi_msecs_t)0);
    }""",
     """    if (!all_visited || any_purged) {
      // Keep already-due work eligible and preserve any earlier concurrent timer.
      mi_arenas_schedule_purge(subproc, retry_expire);
    }"""),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_archive():
    supplied = os.environ.get("TORCHLEAN_MIMALLOC_ARCHIVE")
    if supplied:
        archive = Path(supplied)
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "torchlean"
        cache.mkdir(parents=True, exist_ok=True)
        archive = cache / ("mimalloc-" + SOURCE_SHA256 + ".tar.gz")
        if not archive.exists():
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                data = response.read(4 * 1024 * 1024)
            if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
                raise RuntimeError("downloaded mimalloc source checksum mismatch")
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as temporary:
                temporary.write(data)
                name = temporary.name
            os.replace(name, archive)
    if sha(archive) != SOURCE_SHA256:
        raise RuntimeError("mimalloc source checksum mismatch: " + str(archive))
    return archive


def extract_source(archive, directory):
    """Extract the pinned, ordinary-file source tree without applying patches."""
    if sha(archive) != SOURCE_SHA256:
        raise RuntimeError("mimalloc source checksum mismatch: " + str(archive))
    with tarfile.open(archive) as tar:
        for member in tar:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or \
                    relative.parts[0] != "mimalloc-3.4.4":
                raise RuntimeError("unexpected mimalloc archive path")
            if member.isdir():
                continue
            if not member.isfile():
                raise RuntimeError("unexpected non-file in mimalloc archive")
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    source = directory / "mimalloc-3.4.4"
    if sha(source / "src/os.c") != OS_SHA256 or \
            sha(source / "src/arena.c") != ARENA_SHA256 or \
            sha(source / "include/mimalloc.h") != HEADER_SHA256:
        raise RuntimeError("unexpected mimalloc source")
    return source


def patch_arena_source(text):
    """Apply the reviewed wakeup protocol to exactly the pinned arena source."""
    if hashlib.sha256(text.encode()).hexdigest() != ARENA_SHA256:
        raise RuntimeError("unexpected unpatched arena source")
    for old, new in ARENA_PATCHES:
        if text.count(old) != 1:
            raise RuntimeError("arena wakeup patch does not apply exactly once")
        text = text.replace(old, new)
    return text


def compiler_flags(source):
    return [
        "-c", "-x", "c++", "-std=c++17", "-fPIC", "-O3", "-DNDEBUG",
        "-DMI_SHARED_LIB", "-DMI_SHARED_LIB_EXPORT", "-DMI_WIN_NOREDIRECT",
        "-DMI_SECURE=0", "-Wno-unused-function", "-Wno-deprecated",
        # NN's native objects can also be linked into a shared library.
        "-ftls-model=initial-exec", "-include", str(ROOT / "csrc/runtime/lean_libc_compat.h"),
        "-I" + str(source / "include"),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-include", required=True, type=Path)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if sha(args.lean_include / "lean/mimalloc.h") != HEADER_SHA256:
        raise RuntimeError("allocator build requires Lean's pinned mimalloc 3.4.4 header; "
                           "review this binding after a toolchain update")
    archive = source_archive()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mimalloc-", dir=args.output.parent) as directory:
        directory = Path(directory)
        source = extract_source(archive, directory)
        os_source = source / "src/os.c"
        arena_source = source / "src/arena.c"
        arena_source.write_text(patch_arena_source(arena_source.read_text()))
        temporary_object = directory / "mimalloc.o"
        flags = compiler_flags(source)
        subprocess.run([args.compiler, *flags, str(source / "src/static.c"),
                        "-o", str(temporary_object)], check=True, timeout=180)
        os.replace(temporary_object, args.output)
        args.output.with_suffix(".json").write_text(json.dumps({
            "sourceUrl": SOURCE_URL, "sourceSha256": SOURCE_SHA256,
            "headerSha256": HEADER_SHA256, "osSha256": sha(os_source),
            "arenaBeforeSha256": ARENA_SHA256, "arenaSha256": sha(arena_source),
            "objectSha256": sha(args.output), "sourcePatched": True,
            "patch": "arena-wakeup-v2",
            "compiler": args.compiler, "compilerSha256": sha(Path(args.compiler).resolve()),
            "flags": flags,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
