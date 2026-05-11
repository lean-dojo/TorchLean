#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/checks/check.sh [options]

Run TorchLean's local verification gate.

Default:
  lake build
  lake test
  lake lint

Options:
  --ci-all              Also build NN.CI.All, the broad developer/CI import umbrella.
  --cuda                Build and test with real CUDA externs (-R -K cuda=true).
  --cuda-home PATH      CUDA toolkit root; implies --cuda.
  --no-build            Skip lake build.
  --no-test             Skip lake test.
  --no-lint             Skip lake lint.
  -h, --help            Show this help message.

Environment:
  LAKE                  Lake executable to use (default: lake).

Examples:
  scripts/checks/check.sh
  scripts/checks/check.sh --ci-all
  scripts/checks/check.sh --cuda --cuda-home /usr/local/cuda
  LAKE=~/.elan/bin/lake scripts/checks/check.sh --ci-all
EOF
}

run_build=true
run_test=true
run_lint=true
run_ci_all=false
cuda=false
cuda_home=""

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
    --cuda-home)
      if [[ $# -lt 2 ]]; then
        echo "error: --cuda-home requires a path" >&2
        exit 2
      fi
      cuda=true
      cuda_home="$2"
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

LAKE="${LAKE:-lake}"
lake_flags=()

# CUDA builds need both Lake's runtime flag (`-R`) and the TorchLean package
# option selecting native CUDA externs. `--cuda-home` is passed separately so
# local toolkits do not need to live in a global default location.
if [[ "$cuda" == true ]]; then
  lake_flags+=("-R" "-K" "cuda=true")
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
  run "build" "$LAKE" build "${lake_flags[@]}"
fi

if [[ "$run_ci_all" == true ]]; then
  run "ci-all" "$LAKE" build "${lake_flags[@]}" NN.CI.All
fi

if [[ "$run_test" == true ]]; then
  run "test" "$LAKE" test "${lake_flags[@]}"
fi

if [[ "$run_lint" == true ]]; then
  run "lint" "$LAKE" lint "${lake_flags[@]}"
fi

printf '\nTorchLean local check passed.\n'
