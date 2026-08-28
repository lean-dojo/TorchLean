#!/usr/bin/env bash
# Check that the CUDA compilation target reaches nvcc and invalidates stale objects.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda_arch_target.sh [options]

Verify the `cuda_arch` Lake option end to end without a CUDA toolkit or a GPU.

A recording stand-in for nvcc is placed first on PATH; it logs its argument vector and emits an
empty object file. One extern library is then built repeatedly under different targets, and the
log is asserted against what each build should have compiled:

  no target                      no -arch flag; nvcc keeps its built-in default
  -K cuda_arch=sm_86             -arch=sm_86
  -K cuda_arch=sm_86 (repeated)  no recompilation
  -K cuda_arch=sm_89             recompiled, because the target is a traced argument
  TORCHLEAN_CUDA_ARCH=sm_90      -arch=sm_90, the environment fallback
  both, option and environment   the option wins
  -K cuda_arch=-gencode ...      passed through verbatim, for multi-architecture binaries
  -K cuda=true without -R        nothing compiled: Lake reuses the stored configuration
  the same build with -R         compiled, because the configuration is re-elaborated

The last pair is the one worth stating plainly. `-K` is read when the package configuration is
elaborated, not when the build runs, so a `-K cuda=true` that omits `-R` after a stub build is
accepted, ignored, and links the CPU stubs. That build reports success, and the test suite it
produces passes every test without executing a single kernel; only `ldd` on the executable, or
the suite's own "CUDA kernels: skipped (CPU build)" line, tells the difference. The pair is
checked together because either row alone proves nothing: 0 compilations is also what an
up-to-date target reports.

The stand-in's objects are removed afterwards, so a later real CUDA build cannot mistake them
for its own. Any genuine object for that library goes with them, which costs one recompilation
on the next real CUDA build. The stored build configuration is returned to its default, since
every build here enables CUDA.

Options:
  --target LIB          Extern library to build. Default: torchlean_dgemm_cuda.
  -h, --help            Show this help message.

Environment:
  LAKE                  Lake executable to use (default: lake).

Examples:
  scripts/checks/cuda_arch_target.sh
  LAKE=~/.elan/bin/lake scripts/checks/cuda_arch_target.sh
EOF
}

LAKE="${LAKE:-lake}"
lib="torchlean_dgemm_cuda"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) [[ $# -ge 2 ]] || { echo "--target needs a value" >&2; exit 2; }; lib="$2"; shift 2 ;;
    --target=*) lib="${1#--target=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

work="$(mktemp -d)"
build="$repo/.lake/build"

# The stand-in writes into the normal build directory, so its objects have to go whether the
# check passes or fails: Lake would otherwise record them as up to date and a real CUDA build
# would archive an empty object instead of recompiling.
cleanup() {
  rm -f "$build/$lib.o" "$build/$lib.o.trace" "$build/lib$lib.a" "$build/lib$lib.a.trace"
  rm -rf "$work"
  # Lake remembers the last build configuration, and every build below sets `cuda=true`. Put the
  # default back, or the next plain `lake build` in this checkout reaches for a CUDA toolkit that
  # the developer running this check need not have.
  "$LAKE" -R check-build >/dev/null 2>&1 || true
}
trap cleanup EXIT

mkdir -p "$work/bin"
cat > "$work/bin/nvcc" <<'STUB'
#!/usr/bin/env bash
# Recording stand-in for nvcc: log the argument vector, emit an empty object at -o.
printf '%s\n' "$*" >> "${NVCC_LOG:?}"
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -n "$out" ] || { echo "recording nvcc: no -o in argument vector" >&2; exit 1; }
printf 'int torchlean_recording_nvcc_probe;\n' | cc -x c -c -o "$out" -
STUB
chmod +x "$work/bin/nvcc"

export PATH="$work/bin:$PATH"
export NVCC_LOG="$work/nvcc.log"
: > "$NVCC_LOG"

# Start from no object at all, so the first build below is guaranteed to compile.
rm -f "$build/$lib.o" "$build/$lib.o.trace" "$build/lib$lib.a" "$build/lib$lib.a.trace"

failures=0

# Build once and report how many compilations that build triggered.
compilations_for() {
  local before after
  before="$(wc -l < "$NVCC_LOG")"
  "$@" >/dev/null 2>&1 || { echo "   build failed: $*" >&2; return 1; }
  after="$(wc -l < "$NVCC_LOG")"
  echo "$((after - before))"
}

check() {
  local name="$1" expect_compilations="$2" expect_flags="$3"; shift 3
  local n last
  n="$(compilations_for "$@")" || { failures=$((failures + 1)); return; }
  last="$(tail -n 1 "$NVCC_LOG")"
  if [[ "$n" != "$expect_compilations" ]]; then
    echo "FAIL $name: expected $expect_compilations compilation(s), saw $n" >&2
    failures=$((failures + 1))
    return
  fi
  if [[ -n "$expect_flags" && "$last" != *"$expect_flags"* ]]; then
    echo "FAIL $name: expected argument vector to contain '$expect_flags'" >&2
    echo "   saw: $last" >&2
    failures=$((failures + 1))
    return
  fi
  if [[ -z "$expect_flags" && "$n" != "0" && "$last" == *"-arch"* ]]; then
    echo "FAIL $name: expected no -arch flag" >&2
    echo "   saw: $last" >&2
    failures=$((failures + 1))
    return
  fi
  echo "ok   $name"
}

echo "Checking the CUDA compilation target through a recording nvcc ($lib)"

check "no target leaves nvcc on its default" 1 "" \
  "$LAKE" -R -K cuda=true build "$lib"
check "-K cuda_arch=sm_86 compiles for sm_86" 1 "-arch=sm_86" \
  "$LAKE" -R -K cuda=true -K cuda_arch=sm_86 build "$lib"
check "an unchanged target does not recompile" 0 "" \
  "$LAKE" -R -K cuda=true -K cuda_arch=sm_86 build "$lib"
check "a changed target recompiles" 1 "-arch=sm_89" \
  "$LAKE" -R -K cuda=true -K cuda_arch=sm_89 build "$lib"
check "TORCHLEAN_CUDA_ARCH is the fallback" 1 "-arch=sm_90" \
  env TORCHLEAN_CUDA_ARCH=sm_90 "$LAKE" -R -K cuda=true build "$lib"
check "the option outranks the environment" 1 "-arch=sm_75" \
  env TORCHLEAN_CUDA_ARCH=sm_90 "$LAKE" -R -K cuda=true -K cuda_arch=sm_75 build "$lib"
check "an explicit flag list passes through" 1 "-gencode arch=compute_120,code=[sm_120,compute_120]" \
  "$LAKE" -R -K cuda=true \
  -K cuda_arch="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_120,code=[sm_120,compute_120]" \
  build "$lib"

# `-K` reaches the build only through a reconfiguration. Return to the stub flavour first, so the
# pair below starts from a build that holds no CUDA object at all; then run the SAME command twice,
# differing only in `-R`. Asserting the two together is what makes them evidence — a lone "no
# compilations" result is indistinguishable from an up-to-date target.
"$LAKE" -R build "$lib" >/dev/null 2>&1 || true

check "-K cuda=true without -R reuses the stored configuration" 0 "" \
  "$LAKE" -K cuda=true -K cuda_arch=sm_86 build "$lib"
check "the same build with -R re-elaborates it" 1 "-arch=sm_86" \
  "$LAKE" -R -K cuda=true -K cuda_arch=sm_86 build "$lib"

if [[ "$failures" -ne 0 ]]; then
  echo "$failures check(s) failed" >&2
  exit 1
fi
echo "All CUDA compilation target checks passed."
