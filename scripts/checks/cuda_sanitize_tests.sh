#!/usr/bin/env bash
# Run the LibTorch CUDA test suite under NVIDIA sanitizers.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$repo_root/scripts/lake.sh}"

usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda_sanitize_tests.sh [options]

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
  scripts/checks/cuda_sanitize_tests.sh
  scripts/checks/cuda_sanitize_tests.sh --all-tools
  scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda --tool memcheck
  scripts/checks/cuda_sanitize_tests.sh --libtorch-home /opt/libtorch --all-tools
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
  # atomically on every invocation. Root-only instrumentation also lets the
  # suite's intentional self-reexec cache probe complete normally.
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
