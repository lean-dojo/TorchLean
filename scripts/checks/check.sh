#!/usr/bin/env bash
# Run the standard local build, test, and lint checks.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$repo_root/scripts/lake.sh}"

usage() {
  cat <<'EOF'
Usage: scripts/checks/check.sh [options]

Run TorchLean's local verification gate.

Default (CPU profile):
  scripts/lake.sh -R -K cuda=false build
  scripts/lake.sh -R -K cuda=false test
  scripts/lake.sh -R -K cuda=false lint

Options:
  --ci-all              Also build NN.CI.All, the broad developer/CI import umbrella.
  --cuda                Build and test the LibTorch CUDA backend; require a visible GPU.
  --libtorch-home PATH   LibTorch SDK root (-K libtorch_home=PATH); implies --cuda.
  --cuda-home PATH       CUDA development toolkit for SDK CMake discovery; implies --cuda.
  --no-build            Skip lake build.
  --no-test             Skip lake test.
  --no-lint             Skip lake lint.
  -h, --help            Show this help message.

Environment:
  LAKE                  Lake command to use (default: scripts/lake.sh).
  TORCHLEAN_LIBTORCH_HOME
                        SDK root when --libtorch-home is omitted (otherwise libtorch/).
                        See scripts/README.md for C++ compiler and CMake controls.

Examples:
  scripts/checks/check.sh
  scripts/checks/check.sh --ci-all
  scripts/checks/check.sh --libtorch-home /opt/libtorch
  scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda
  LAKE=~/.elan/bin/lake scripts/checks/check.sh --ci-all
EOF
}

run_build=true
run_test=true
run_lint=true
run_ci_all=false
cuda=false
libtorch_home=""
cuda_home=""

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
    --ci-all)
      run_ci_all=true
      shift
      ;;
    --cuda)
      cuda=true
      shift
      ;;
    --libtorch-home)
      libtorch_home="$(path_argument "$1" "${2:-}")"
      cuda=true
      shift 2
      ;;
    --cuda-home)
      cuda_home="$(path_argument "$1" "${2:-}")"
      cuda=true
      shift 2
      ;;
    --no-build)
      run_build=false
      shift
      ;;
    --no-test)
      run_test=false
      shift
      ;;
    --no-lint)
      run_lint=false
      shift
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
lake_flags=(-R -K cuda=false)

# Repeat the backend and SDK options for every invocation, including custom LAKE commands
# that do not provide scripts/lake.sh's automatic reconfiguration and profile selection.
if [[ "$cuda" == true ]]; then
  # A requested GPU check must also reject an accidentally linked CPU-stub executable.
  export TORCHLEAN_REQUIRE_CUDA=1
  lake_flags=(-R -K cuda=true)
  if [[ -n "$libtorch_home" ]]; then
    lake_flags+=("-K" "libtorch_home=$libtorch_home")
  fi
  if [[ -n "$cuda_home" ]]; then
    lake_flags+=("-K" "cuda_home=$cuda_home")
  fi
fi

run() {
  # Print the exact command in shell-escaped form before running it. That makes
  # local failure reports copy-pasteable without changing the command semantics.
  printf '\n==> %q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

if [[ "$run_build" == true ]]; then
  run "build" "$LAKE" "${lake_flags[@]}" build
fi

if [[ "$run_ci_all" == true ]]; then
  run "ci-all" "$LAKE" "${lake_flags[@]}" build NN.CI.All
fi

if [[ "$run_test" == true ]]; then
  run "test" "$LAKE" "${lake_flags[@]}" test
fi

if [[ "$run_lint" == true ]]; then
  run "lint" "$LAKE" "${lake_flags[@]}" lint
fi

printf '\nTorchLean local check passed.\n'
