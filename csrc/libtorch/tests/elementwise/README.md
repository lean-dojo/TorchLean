# Elementwise C ABI regression

Configure, compile, generate the Lean reference, and run **only in the cluster**.
This C++ executable links the production backend, selected ATen SDK, and Lean runtime.

From the source root, after building the production LibTorch backend:

```bash
# Set these to the same SDK/toolchain/artifact used by the production cluster build.
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
export TORCHLEAN_LEAN_PREFIX=/path/to/lean
export TORCHLEAN_BACKEND_LIBRARY=/absolute/path/to/libtorchlean_libtorch.so

time scripts/checks/cuda_float32_parity.sh --sweep 1024 --keep
```

The runner resolves the backend path before selecting the CPU reference-generation
profile, builds `native_float32_parity` and the CMake project in this directory,
then runs the production C ABI checks. `--keep` prints and retains the build and
reference directory. `--skip-build` reuses the Lean reference executable.
The default sweep is 200000; use `--sweep 1024` for the initial timing run.
Every supplied case is checked, with one scalar AXPY call per FMA case and batched
add/mul/div/sqrt checks. The reference file must contain all five primitives.
Native regressions require the CUDA SDK and a usable device.

After building `nn_tests_suite` with the native LibTorch configuration, run the
isolated Lean memory probes against that same executable in the cluster:

```bash
for probe in accounting attention-context oom-recovery; do
  TORCHLEAN_REQUIRE_CUDA=1 TORCHLEAN_LIBTORCH_MEMORY_PROBE="$probe" \
    scripts/lake.sh -Kcuda=true env .lake/build/bin/nn_tests_suite
done
```

The ordinary `Stress.run` also launches these three fresh subprocesses. They
check real readback, logical ownership versus native allocated/reserved/peak
accounting, live tensors across `emptyCache`, saved attention inputs across
release/backward, output-context finalization, and `resourceExhausted` recovery
under a temporary allocator memory fraction. No assertion pins an SDK cache
block size or an exact number of bytes returned to the driver. CPU parity builds
check that native counters are zero and explicitly skip the GPU memory probes.

## Coverage

- `NN/Runtime/Autograd/Engine/Cuda/Float32Contract.lean` defines exact finite bits
  and `AgreeUpToNaN`, with no finite tolerance. This regression follows that relation
  and separately counts differing NaN encodings.
- Adam's three outputs are checked bitwise with hand-derived fixtures and
  10 × 1024 staged host-oracle cases. The fixtures isolate both moment FMAs, parameter decay, the final
  update FMA, and the separately rounded initial moment products and gradient square.
  A negative second moment confirms that Adam retains raw IEEE sqrt, including NaN.
  Each of the nine double scalar arguments is independently varied across a
  binary32 rounding midpoint. The cancellation residual `2^-46` detects changes
  that a `1e-6` tolerance would accept.
- The Lean reference file validates production add/mul/div/scalar AXPY and upstream
  tensor `addcmul`/raw sqrt. Buffer.sqrt also runs against the same file with its
  specified `x <= 0 → +0` selection applied.
- FMA cases cover cancellation, overflow cancellation, subnormal/underflow results,
  signed zeros, NaNs, and zero times infinity. The scalar coefficient is checked as
  both a CPU zero-dimensional tensor and a CUDA zero-dimensional tensor, as well as
  through the production C ABI. Tensor operands exercise the other CUDA kernel path.
- A split float32 multiply/add must return zero on the cancellation discriminator.
  A float64 multiply/add must return the **wrong** upper float32 neighbor on the
  double-rounding discriminator. These are negative controls, not accepted fallbacks.
- All 30 arithmetic exports run under both ambient GradMode settings. An ATen
  RecordFunction callback checks that operations execute
  with GradMode disabled inside the export, and that results have no `grad_fn` or
  `requires_grad`. The observer is calibrated against an ordinary graph-recording
  ATen multiplication, so no-grad inputs cannot make the test vacuous. The runtime
  rejects gradient-bearing Buffer inputs; a separate gradient-bearing ATen tensor
  checks `invoke` itself. Both reduction modes run through the canonical LibTorch
  settings API, checking the IO result of both reads and writes and unboxing
  the UInt32 readback. Strict determinism must
  disable cuDNN benchmarking; both settings are restored afterward. Caller
  GradMode and input flags must survive.
- Raw control getter checks require an unknown setting ID to return an IO error,
  followed by a successful unchanged determinism read. The allocator memory-fraction
  getter must return a successful IO result containing a finite Float in `[0, 1]`.
- Deterministic sum fixtures distinguish the specified 256-lane tree from a
  left fold and a double accumulator. They cover partial blocks, recursive
  partial sums, and grid-stride accumulation above the 65535-block cap
  (one 64 MiB input). Mean must round the full sum before multiplying by its
  rounded reciprocal; `[7, 0, 0]` distinguishes this from direct division.
  Empty inputs, signed zero, subnormals, and overflow remain explicit.

Finite sweeps provide validation evidence. Run the selected-gradient, tape traversal,
deterministic reduction, GELU, and end-to-end Lean suites as well.

## Exact SDK source evidence

The selected SDK reports
`torch 2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, C++11 ABI enabled.
The public upstream commit `0291f960b6` has `version.txt = 2.12.0a0`.
The executable prints its header version; retain the full SDK/build
manifest with the results because vendor patches and compiler options also matter.

Official sources at that commit:

- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/ufunc/add.h>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/record_function.h>

For CUDA float32, both `addcmul` kernel paths call `pointwise_op_impl<float>`.
With `value == 1`, that helper explicitly calls
`std::fma(tensor1, tensor2, input)`. Production AXPY therefore passes the rounded
coefficient as **tensor2** and sets **value to one**. CUDA supports a CPU scalar
only in tensor2 for this operation. This provides one explicit fused stage without
adding a GPU scalar-allocation kernel.

For other `value` values the helper first evaluates the binary product, then uses
`std::fma(value, product, input)`. Arbitrarily moving factors into `value` changes
stage boundaries. The CUDA `add(alpha)` ufunc instead contains an ordinary
`self + alpha * other` expression; contraction depends on its build. The test
prints its discriminator result as a diagnostic, not a contractual requirement.

Binary64 is insufficient as a universal FMA replacement: with
`x = 1 + 2^-23`, `y = 1.5`, `z = -2^-80`, the exact product is the midpoint
between float32 `0x3fc00001` and `0x3fc00002`. The tiny negative addend selects the
lower neighbor for a true float32 FMA, but is lost by a binary64 add, which then
rounds to the upper even neighbor.

The source evidence applies to this SDK revision. SDK changes require a fresh source
audit and cluster run with the same finite-bit assertions.
