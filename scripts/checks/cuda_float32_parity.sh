#!/usr/bin/env bash
# Check the five fields of NativePrimitiveAgreement on this machine's GPU.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAKE="${LAKE:-$repo_root/scripts/lake.sh}"
cuda_home="${CUDA_HOME:-/usr/local/cuda}"

usage() {
  cat <<'EOF'
Usage: scripts/checks/cuda_float32_parity.sh [options]

Compare native binary32 arithmetic against TorchLean's IEEE32Exec reference model, bit for bit,
for the five primitives named by NativePrimitiveAgreement: add, mul, div, fma and sqrt.

The reference bits come from Lean:

  lake exe native_float32_parity --emit-cases --sweep N

and the comparison runs on the host (C library) and on the GPU (round-to-nearest intrinsics)
inside scripts/checks/cuda_float32_parity.cu.

Two verdicts are reported per primitive. "contract" is what NativePrimitiveAgreement asks for,
namely equal bits or two quiet NaNs; "strict" is plain bit equality. They differ, and the run prints
the distinct NaN encodings so you can see why: IEEE 754-2019 leaves the payload of an invalid
operation to the implementation. Only the contract verdict sets the exit status.

Options:
  --sweep N        number of random cases to add to the curated ones (default: 200000)
  --fast-math      also run a second pass compiled with --use_fast_math, which is expected to
                   fail; the point is to see what a violated assumption looks like
  --cuda-arch ARCH value for nvcc -arch (default: all-major; native is rejected)
  --arch ARCH      alias for --cuda-arch
  --cuda-home PATH CUDA toolkit root, whose bin/nvcc is used (default: $CUDA_HOME or
                   /usr/local/cuda)
  --skip-build     do not rebuild the native_float32_parity executable
  --keep           keep the temporary directory and print its path
  -h, --help       show this help message

Environment:
  LAKE             lake command to use (default: scripts/lake.sh)
  CUDA_HOME        CUDA toolkit root (default: /usr/local/cuda)
  NVCC             CUDA compiler, overriding the toolkit root
EOF
}

fast_math=false
sweep=200000
arch="all-major"
skip_build=false
keep=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep) sweep="$2"; shift 2 ;;
    --fast-math) fast_math=true; shift ;;
    --cuda-arch|--arch)
      if [[ $# -lt 2 ]]; then
        echo "error: $1 requires a target such as all-major or sm_80" >&2
        exit 2
      fi
      arch="$2"
      shift 2
      ;;
    --cuda-home) cuda_home="$2"; shift 2 ;;
    --skip-build) skip_build=true; shift ;;
    --keep) keep=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option $1" >&2; usage >&2; exit 2 ;;
  esac
done

# Match Lake's trimAscii: space, tab, carriage return and newline only.
arch_whitespace=$' \t\r\n'
arch="${arch#"${arch%%[!$arch_whitespace]*}"}"
arch="${arch%"${arch##*[!$arch_whitespace]}"}"
arch="${arch:-all-major}"
if [[ "$arch" == "native" ]]; then
  echo "error: native is not supported; use --cuda-arch all-major or an explicit target" >&2
  exit 2
fi
if [[ "$arch" != "all-major" && "$arch" != "all" && ! "$arch" =~ ^sm_[0123456789]+[af]?$ ]]; then
  echo "error: invalid CUDA target: $arch; expected all-major, all, or a real target such as sm_80" >&2
  exit 2
fi

# This wrapper calls nvcc directly. Apply Lake's selector guard here too, so
# environment flags cannot replace the explicit architecture or hide it in an option file.
for flags_name in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
  flags="${!flags_name-}"
  case "$flags" in
    *-arch*|*--gpu-architecture*|*-code*|*--gpu-code*|*-gencode*|*--generate-code*|\
    *-ccbin*|*--compiler-bindir*|*-optf*|*--options-file*|*@*)
      echo "error: $flags_name cannot select architectures, host compilers, or option files" >&2
      echo "hint: use --cuda-arch and NVCC_CCBIN for the target and host compiler" >&2
      exit 2
      ;;
  esac
done

NVCC="${NVCC:-$cuda_home/bin/nvcc}"
if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "error: $NVCC not found; pass --cuda-home or set NVCC" >&2
  exit 2
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
# Lean's executable also needs its toolchain libraries, including when the build is skipped.
"$LAKE" env "$repo_root/.lake/build/bin/native_float32_parity" \
  --emit-cases --sweep "$sweep" >"$cases"
echo "reference cases: $(wc -l <"$cases")"

# No -ffast-math and no -fmad on the host side, and round-to-nearest intrinsics on the device side.
# Those flags are the assumption's known failure modes, so the honest pass compiles without them.
"$NVCC" -O2 -std=c++17 -arch="$arch" -fmad=false \
  -o "$tmp_dir/parity" "$repo_root/scripts/checks/cuda_float32_parity.cu"
"$tmp_dir/parity" <"$cases"

if [[ "$fast_math" == true ]]; then
  echo
  echo "== second pass: --use_fast_math =="
  "$NVCC" -O2 -std=c++17 -arch="$arch" --use_fast_math \
    -o "$tmp_dir/parity_fast" "$repo_root/scripts/checks/cuda_float32_parity.cu"
  if "$tmp_dir/parity_fast" <"$cases"; then
    echo "note: fast math agreed on these cases, which is luck rather than a guarantee"
  else
    status=$?
    if [[ "$status" -eq 1 ]]; then
      echo "expected: fast math breaks the contract, so a fast-math build voids the proofs"
    else
      echo "error: fast-math parity check failed to run (exit $status)" >&2
      exit "$status"
    fi
  fi
fi
