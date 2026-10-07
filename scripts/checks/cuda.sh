#!/usr/bin/env bash
# GPU parity and sanitizer checks against TorchLean's LibTorch backend.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$repo_root/scripts/lake.sh}"

parity() {
torchlean_sdk="${TORCHLEAN_LIBTORCH_HOME:-}"
torchlean_lean_prefix="${TORCHLEAN_LEAN_PREFIX:-}"
torchlean_backend="${TORCHLEAN_BACKEND_LIBRARY:-}"
torchlean_cmake="${TORCHLEAN_CMAKE:-cmake}"
declare -a cmake_args=()
sweep=200000
skip_build=false
keep=false

usage() {
  cat <<'HELP'
Usage: scripts/checks/cuda.sh parity [options]

Run on a CUDA host against the already-built production LibTorch backend.
Compare add/mul/div/sqrt/fma with the existing Lean binary32 reference stream.
Finite values and signed zeros must match exactly; both NaNs satisfy AgreeUpToNaN.
NaN encoding differences are counted separately. A finite sweep is validation, not a proof.

The C++ harness batches add/mul/div and raw IEEE sqrt through ATen. It also checks the
production C ABI, including each FMA case through scalar AXPY using addcmul(value=1).
Buffer.sqrt is checked separately with its specified nonpositive-to-zero selection.
FMA cancellation/double-rounding, staged Adam and noAutograd regressions run as well.

Required paths (or set the corresponding environment variables):
  --libtorch-home PATH   TORCHLEAN_LIBTORCH_HOME: selected SDK root containing share/cmake/Torch
  --backend-library PATH TORCHLEAN_BACKEND_LIBRARY: production libtorchlean_libtorch.so
  --lean-prefix PATH     TORCHLEAN_LEAN_PREFIX: pinned Lean root (default: scripts/lake.sh env lean --print-prefix)

Options:
  --sweep N              random cases in addition to curated cases (default: 200000)
                         Large sweeps take longer because every FMA visits the actual scalar C ABI.
  --skip-build           reuse the Lean reference executable; still build the C++ harness
  --keep                 retain temporary build/results and print their directory
  --cmake-arg ARG        repeatable CMake discovery option, e.g. -DCUDA_NVRTC_LIB=/path/libnvrtc.so
  -h, --help             print this message

This harness links the selected prebuilt SDK. Its arithmetic and GPU targets are
determined by that SDK's build configuration. Validation sources are ordinary C++.
TORCHLEAN_CMAKE selects CMake. Arguments are passed literally without shell evaluation.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libtorch-home|--backend-library|--lean-prefix|--sweep|--cmake-arg)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        exit 2
      fi
      case "$1" in
        --libtorch-home) torchlean_sdk="$2" ;;
        --backend-library) torchlean_backend="$2" ;;
        --lean-prefix) torchlean_lean_prefix="$2" ;;
        --sweep) sweep="$2" ;;
        --cmake-arg) cmake_args+=("$2") ;;
      esac
      shift 2
      ;;
    --skip-build) skip_build=true; shift ;;
    --keep) keep=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! "$sweep" =~ ^[0-9]+$ ]]; then
  echo "error: --sweep requires a nonnegative integer" >&2
  exit 2
fi
if [[ ! -f "$torchlean_sdk/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "error: set TORCHLEAN_LIBTORCH_HOME or --libtorch-home to the selected SDK" >&2
  exit 2
fi
if [[ ! -f "$torchlean_backend" ]]; then
  echo "error: set TORCHLEAN_BACKEND_LIBRARY or --backend-library to the built production library" >&2
  exit 2
fi
# Reference generation selects the CPU Lake profile, which can move .lake/build.
# Pin the supplied production artifact before any Lake invocation changes that symlink.
torchlean_backend="$(python3 - "$torchlean_backend" <<'PY'
import os
import sys
print(os.path.realpath(sys.argv[1]))
PY
)"
torchlean_sdk="$(cd "$torchlean_sdk" && pwd -P)"
if [[ -z "$torchlean_lean_prefix" ]]; then
  torchlean_lean_prefix="$("$LAKE" env lean --print-prefix)"
fi

tmp_dir="$(mktemp -d)"
if [[ "$keep" == true ]]; then
  echo "temporary directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

if [[ "$skip_build" == false ]]; then
  "$LAKE" build native_float32_parity
fi
cases="$tmp_dir/cases.txt"
"$LAKE" env "$repo_root/.lake/build/bin/native_float32_parity" \
  --emit-cases --sweep "$sweep" >"$cases"
echo "reference cases: $(wc -l <"$cases")"

"$torchlean_cmake" -S "$repo_root/csrc/libtorch/tests/elementwise" -B "$tmp_dir/build" \
  "${cmake_args[@]}" \
  -DTORCHLEAN_LIBTORCH_HOME="$torchlean_sdk" \
  -DTORCHLEAN_LEAN_PREFIX="$torchlean_lean_prefix" \
  -DTORCHLEAN_BACKEND_LIBRARY="$torchlean_backend"
"$torchlean_cmake" --build "$tmp_dir/build" -j2
"$tmp_dir/build/torchlean_elementwise_regression" <"$cases"
}

sanitize() {
usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda.sh sanitize [options]

Build TorchLean's LibTorch CUDA backend and run the curated CUDA/Lean test suite under
NVIDIA Compute Sanitizer. A CUDA-enabled SDK and a visible GPU are required.

Default:
  scripts/lake.sh -R -K cuda=true build nn_tests_suite
  scripts/lake.sh -R -K cuda=true env compute-sanitizer --tool memcheck \
    --target-processes application-only --error-exitcode 99 .lake/build/bin/nn_tests_suite

Options:
  --tool TOOL           Sanitizer tool to run. May be repeated.
                        Common tools: memcheck, racecheck, initcheck, synccheck.
                        Default: memcheck.
  --all-tools          Run memcheck, racecheck, initcheck, and synccheck.
  --libtorch-home PATH LibTorch SDK root; passes -K libtorch_home=PATH.
  --cuda-home PATH     CUDA development toolkit for SDK discovery and sanitizer lookup.
  --target PATH        Executable to run after building.
                        Default: .lake/build/bin/nn_tests_suite.
  --target-processes MODE
                        Processes instrumented by Compute Sanitizer: application-only or all.
                        Default: application-only.
  --skip-build         Do not rebuild before running sanitizer.
  --sanitizer PATH     CUDA sanitizer executable name/path.
                        Default: compute-sanitizer if available, otherwise cuda-memcheck.
  --                  Remaining arguments are passed to the test executable.
  -h, --help           Show this help message.

Environment:
  LAKE                 Lake command to use (default: scripts/lake.sh).
  TORCHLEAN_LIBTORCH_HOME
                       SDK root when --libtorch-home is omitted (otherwise libtorch/).
                       See scripts/README.md for C++ compiler and CMake controls.

Examples:
  scripts/checks/cuda.sh sanitize
  scripts/checks/cuda.sh sanitize --all-tools
  scripts/checks/cuda.sh sanitize --cuda-home /usr/local/cuda --tool memcheck
  scripts/checks/cuda.sh sanitize --libtorch-home /opt/libtorch --all-tools
EOF
}

sanitizer=""
target=".lake/build/bin/nn_tests_suite"
target_processes="application-only"
libtorch_home=""
cuda_home=""
skip_build=false
declare -a tools=()
declare -a exe_args=()

path_argument() {
  local value="${2:-}"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ -z "$value" || "$value" == -* ]]; then
    echo "error: $1 requires a directory path" >&2
    exit 2
  fi
  printf '%s\n' "$value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      if [[ $# -lt 2 ]]; then
        echo "error: --tool requires a sanitizer tool name" >&2
        exit 2
      fi
      tools+=("$2")
      shift 2
      ;;
    --all-tools)
      tools=(memcheck racecheck initcheck synccheck)
      shift
      ;;
    --libtorch-home)
      libtorch_home="$(path_argument "$1" "${2:-}")"
      shift 2
      ;;
    --cuda-home)
      cuda_home="$(path_argument "$1" "${2:-}")"
      shift 2
      ;;
    --target)
      if [[ $# -lt 2 ]]; then
        echo "error: --target requires an executable path" >&2
        exit 2
      fi
      target="$2"
      shift 2
      ;;
    --target-processes)
      if [[ $# -lt 2 ]]; then
        echo "error: --target-processes requires application-only or all" >&2
        exit 2
      fi
      case "$2" in
        application-only|all)
          target_processes="$2"
          ;;
        *)
          echo "error: --target-processes must be application-only or all" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    --sanitizer)
      if [[ $# -lt 2 ]]; then
        echo "error: --sanitizer requires a command/path" >&2
        exit 2
      fi
      sanitizer="$2"
      shift 2
      ;;
    --)
      shift
      exe_args=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$repo_root"
export TORCHLEAN_REQUIRE_CUDA=1

if [[ ${#tools[@]} -eq 0 ]]; then
  tools=(memcheck)
fi

if [[ -n "$cuda_home" ]]; then
  if [[ "$cuda_home" != /* ]]; then
    cuda_home="$repo_root/$cuda_home"
  fi
  # Select the sanitizer from this toolkit. The backend's SDK-derived RPATH owns library lookup.
  export PATH="$cuda_home/bin:$PATH"
fi

if [[ -z "$sanitizer" ]]; then
  for candidate in \
      compute-sanitizer \
      /usr/local/cuda/bin/compute-sanitizer \
      cuda-memcheck \
      /usr/local/cuda/bin/cuda-memcheck; do
    if command -v "$candidate" >/dev/null 2>&1; then
      sanitizer="$candidate"
      break
    fi
  done
fi

if [[ -z "$sanitizer" || ! -x "$(command -v "$sanitizer" 2>/dev/null)" ]]; then
  echo "error: could not find an NVIDIA CUDA sanitizer" >&2
  echo "hint: install the NVIDIA CUDA toolkit or pass --sanitizer /path/to/compute-sanitizer" >&2
  exit 127
fi

lake_flags=(-R -K cuda=true)
if [[ -n "$libtorch_home" ]]; then
  lake_flags+=(-K "libtorch_home=$libtorch_home")
fi
if [[ -n "$cuda_home" ]]; then
  lake_flags+=(-K "cuda_home=$cuda_home")
fi

run() {
  # Print commands exactly as executed so sanitizer failures are easy to rerun.
  printf '\n==>'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

if [[ "$skip_build" == false ]]; then
  run "$LAKE" "${lake_flags[@]}" build nn_tests_suite
fi

for tool in "${tools[@]}"; do
  # `--error-exitcode` converts sanitizer findings into a nonzero process exit,
  # which lets CI fail even when the test binary itself exits successfully.
  # Repeat the profile flags because scripts/lake.sh selects `.lake/build`
  # under the checkout lock on every invocation.
  # Check the executable only after Lake has selected the requested profile. With --skip-build,
  # .lake/build can still point at the CPU cache when this script starts.
  run "$LAKE" "${lake_flags[@]}" env bash -c '
    target="$1"
    shift
    if [[ ! -x "$target" ]]; then
      echo "error: test executable is missing or not executable: $target" >&2
      echo "hint: run without --skip-build first" >&2
      exit 1
    fi
    exec "$@"
  ' torchlean-sanitizer "$target" \
    "$sanitizer" --tool "$tool" --target-processes "$target_processes" \
    --error-exitcode 99 "$target" "${exe_args[@]}"
done

printf '\nTorchLean LibTorch CUDA sanitizer pass completed.\n'
}

case "${1:-}" in
  parity|sanitize)
    mode="$1"
    shift
    "$mode" "$@"
    ;;
  -h|--help|"")
    printf 'Usage: scripts/checks/cuda.sh {parity|sanitize} [options]\n'
    printf 'Use parity --help or sanitize --help for mode-specific options.\n'
    ;;
  *)
    echo "error: expected parity or sanitize, got: $1" >&2
    exit 2
    ;;
esac
