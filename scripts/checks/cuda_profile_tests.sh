#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda_profile_tests.sh [options]

Build TorchLean with native CUDA externs and profile the curated CUDA/Lean test suite with
NVIDIA Nsight tools.

Default:
  lake build -R -K cuda=true nn_tests_suite
  lake env nsys profile -t cuda,nvtx,osrt -f true -o data/profiles/cuda/nn_tests_suite_nsys \
    .lake/build/bin/nn_tests_suite

Options:
  --systems            Run Nsight Systems only. This is the default.
  --compute            Run Nsight Compute only.
  --both               Run Nsight Systems, then Nsight Compute.
  --cuda-home PATH     CUDA toolkit root; passes -K cuda_home=PATH and prepends PATH/lib64.
  --target PATH        Executable to run after building.
                        Default: .lake/build/bin/nn_tests_suite.
  --out-dir PATH       Directory for generated reports.
                        Default: data/profiles/cuda.
  --skip-build         Do not rebuild before profiling.
  --nsys PATH          Nsight Systems executable name/path.
                        Default: nsys.
  --ncu PATH           Nsight Compute executable name/path.
                        Default: first working command among ncu and nv-nsight-cu-cli.
  --                  Remaining arguments are passed to the test executable.
  -h, --help           Show this help message.

Environment:
  LAKE                 Lake executable to use (default: lake).

Examples:
  scripts/checks/cuda_profile_tests.sh
  scripts/checks/cuda_profile_tests.sh --both
  scripts/checks/cuda_profile_tests.sh --compute --target .lake/build/bin/nn_tests_suite
EOF
}

LAKE="${LAKE:-lake}"
mode="systems"
target=".lake/build/bin/nn_tests_suite"
out_dir="data/profiles/cuda"
cuda_home=""
skip_build=false
nsys_cmd="nsys"
ncu_cmd=""
declare -a exe_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --systems)
      mode="systems"
      shift
      ;;
    --compute)
      mode="compute"
      shift
      ;;
    --both)
      mode="both"
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
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --out-dir requires a path" >&2
        exit 2
      fi
      out_dir="$2"
      shift 2
      ;;
    --skip-build)
      skip_build=true
      shift
      ;;
    --nsys)
      if [[ $# -lt 2 ]]; then
        echo "error: --nsys requires a command/path" >&2
        exit 2
      fi
      nsys_cmd="$2"
      shift 2
      ;;
    --ncu)
      if [[ $# -lt 2 ]]; then
        echo "error: --ncu requires a command/path" >&2
        exit 2
      fi
      ncu_cmd="$2"
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

lake_flags=(-R -K cuda=true)
if [[ -n "$cuda_home" ]]; then
  lake_flags+=(-K "cuda_home=$cuda_home")
  # Keep the selected toolkit's runtime libraries in front when several CUDA
  # installs are present on a workstation.
  export LD_LIBRARY_PATH="$cuda_home/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

run() {
  # Print exact commands so a profile can be rerun or pasted into a bug report.
  printf '\n==>'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
  "$@"
}

require_tool() {
  local tool="$1"
  local label="$2"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: could not find $label executable '$tool' on PATH" >&2
    exit 127
  fi
  # Some installations leave an `ncu` launcher on PATH even when the actual
  # Nsight Compute payload is missing. Version probing catches that case before
  # we start a long profile run.
  if ! "$tool" --version >/dev/null 2>&1; then
    echo "error: '$tool --version' failed; $label appears to be unavailable or incomplete" >&2
    echo "hint: install the matching NVIDIA Nsight package or pass the full path with --nsys/--ncu" >&2
    exit 127
  fi
}

find_nsight_compute() {
  local candidate
  for candidate in ncu nv-nsight-cu-cli /usr/bin/nv-nsight-cu-cli /usr/local/cuda/bin/nv-nsight-cu-cli; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ "$skip_build" == false ]]; then
  run "$LAKE" build "${lake_flags[@]}" nn_tests_suite
fi

if [[ ! -x "$target" ]]; then
  echo "error: profile target is missing or not executable: $target" >&2
  echo "hint: run without --skip-build first" >&2
  exit 1
fi

mkdir -p "$out_dir"

if [[ "$mode" == "systems" || "$mode" == "both" ]]; then
  require_tool "$nsys_cmd" "Nsight Systems"
  run "$LAKE" env "$nsys_cmd" profile \
    -t cuda,nvtx,osrt \
    -f true \
    -o "$out_dir/nn_tests_suite_nsys" \
    "$target" "${exe_args[@]}"
fi

if [[ "$mode" == "compute" || "$mode" == "both" ]]; then
  if [[ -z "$ncu_cmd" ]]; then
    if ! ncu_cmd="$(find_nsight_compute)"; then
      echo "error: could not find a working Nsight Compute CLI" >&2
      echo "hint: install NVIDIA Nsight Compute or pass the full path with --ncu" >&2
      exit 127
    fi
  fi
  require_tool "$ncu_cmd" "Nsight Compute"
  run "$LAKE" env "$ncu_cmd" \
    --section SpeedOfLight \
    --section LaunchStats \
    --target-processes all \
    --force-overwrite \
    --export "$out_dir/nn_tests_suite_ncu" \
    "$target" "${exe_args[@]}"
fi

printf '\nTorchLean CUDA profiling run completed.\n'
