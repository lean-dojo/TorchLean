#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda_sanitize_tests.sh [options]

Build TorchLean with native CUDA externs and run the curated CUDA/Lean test suite under
NVIDIA Compute Sanitizer.

Default:
  lake build -R -K cuda=true nn_tests_suite
  lake env compute-sanitizer --tool memcheck .lake/build/bin/nn_tests_suite

Options:
  --tool TOOL           Sanitizer tool to run. May be repeated.
                        Common tools: memcheck, racecheck, initcheck, synccheck.
                        Default: memcheck.
  --all-tools          Run memcheck, racecheck, initcheck, and synccheck.
  --cuda-home PATH     CUDA toolkit root; passes -K cuda_home=PATH and prepends PATH/lib64.
  --target PATH        Executable to run after building.
                        Default: .lake/build/bin/nn_tests_suite.
  --skip-build         Do not rebuild before running sanitizer.
  --sanitizer PATH     CUDA sanitizer executable name/path.
                        Default: compute-sanitizer if available, otherwise cuda-memcheck.
  --                  Remaining arguments are passed to the test executable.
  -h, --help           Show this help message.

Environment:
  LAKE                 Lake executable to use (default: lake).

Examples:
  scripts/checks/cuda_sanitize_tests.sh
  scripts/checks/cuda_sanitize_tests.sh --all-tools
  scripts/checks/cuda_sanitize_tests.sh --cuda-home /usr/local/cuda --tool memcheck
EOF
}

LAKE="${LAKE:-lake}"
sanitizer=""
target=".lake/build/bin/nn_tests_suite"
cuda_home=""
skip_build=false
declare -a tools=()
declare -a exe_args=()

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
    --cuda-home)
      if [[ $# -lt 2 ]]; then
        echo "error: --cuda-home requires a path" >&2
        exit 2
      fi
      cuda_home="$2"
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

if [[ ${#tools[@]} -eq 0 ]]; then
  tools=(memcheck)
fi

if [[ -z "$sanitizer" ]]; then
  # `compute-sanitizer` is NVIDIA's current correctness checker. Keep a
  # `cuda-memcheck` fallback so older CUDA installations can still run the same
  # TorchLean gate.
  if command -v compute-sanitizer >/dev/null 2>&1; then
    sanitizer="compute-sanitizer"
  else
    sanitizer="cuda-memcheck"
  fi
fi

if ! command -v "$sanitizer" >/dev/null 2>&1; then
  echo "error: could not find '$sanitizer' on PATH" >&2
  echo "hint: install the NVIDIA CUDA toolkit or pass --sanitizer /path/to/compute-sanitizer" >&2
  exit 127
fi

lake_flags=(-R -K cuda=true)
if [[ -n "$cuda_home" ]]; then
  lake_flags+=(-K "cuda_home=$cuda_home")
  # Native CUDA tests load shared libraries at runtime. Prepending the requested
  # toolkit keeps this script self-contained on machines with multiple CUDA
  # installations.
  export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
  run "$LAKE" build "${lake_flags[@]}" nn_tests_suite
fi

if [[ ! -x "$target" ]]; then
  echo "error: test executable is missing or not executable: $target" >&2
  echo "hint: run without --skip-build first" >&2
  exit 1
fi

for tool in "${tools[@]}"; do
  # `--error-exitcode` converts sanitizer findings into a nonzero process exit,
  # which lets CI fail even when the test binary itself exits successfully.
  run "$LAKE" env "$sanitizer" --tool "$tool" --error-exitcode 99 "$target" "${exe_args[@]}"
done

printf '\nTorchLean CUDA sanitizer pass completed.\n'
