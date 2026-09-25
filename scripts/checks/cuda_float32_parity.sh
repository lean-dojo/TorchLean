#!/usr/bin/env bash
# Build and run only in the cluster. Every GPU operation comes from the selected ATen SDK.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$repo_root/scripts/lake.sh}"
torchlean_sdk="${TORCHLEAN_LIBTORCH_HOME:-}"
torchlean_lean_prefix="${TORCHLEAN_LEAN_PREFIX:-}"
torchlean_backend="${TORCHLEAN_BACKEND_LIBRARY:-}"
sweep=200000
skip_build=false
keep=false

usage() {
  cat <<'HELP'
Usage: scripts/checks/cuda_float32_parity.sh [options]

Run in the cluster against the already-built production LibTorch backend.
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
  --lean-prefix PATH     TORCHLEAN_LEAN_PREFIX: pinned Lean root (default: lake env lean --print-prefix)

Options:
  --sweep N              random cases in addition to curated cases (default: 200000)
                         Large sweeps take longer because every FMA visits the actual scalar C ABI.
  --skip-build           reuse the Lean reference executable; still build the C++ harness
  --keep                 retain temporary build/results and print their directory
  -h, --help             print this message

This harness links the selected prebuilt SDK. Its arithmetic and GPU targets are
determined by that SDK's build configuration. Validation sources are ordinary C++.
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --libtorch-home|--backend-library|--lean-prefix|--sweep)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a value" >&2
        exit 2
      fi
      case "$1" in
        --libtorch-home) torchlean_sdk="$2" ;;
        --backend-library) torchlean_backend="$2" ;;
        --lean-prefix) torchlean_lean_prefix="$2" ;;
        --sweep) sweep="$2" ;;
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

cmake -S "$repo_root/csrc/libtorch/tests/elementwise" -B "$tmp_dir/build" \
  -DTORCHLEAN_LIBTORCH_HOME="$torchlean_sdk" \
  -DTORCHLEAN_LEAN_PREFIX="$torchlean_lean_prefix" \
  -DTORCHLEAN_BACKEND_LIBRARY="$torchlean_backend"
cmake --build "$tmp_dir/build" -j2
"$tmp_dir/build/torchlean_elementwise_regression" <"$cases"
