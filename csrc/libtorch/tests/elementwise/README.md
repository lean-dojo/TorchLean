# Elementwise C ABI regression

The C++ executable `torchlean_elementwise_regression`, built from `regression.cpp`, links the
production backend, the selected ATen SDK and the Lean runtime, and checks the backend's
elementwise C ABI against the Lean binary32 reference and against host arithmetic. It needs a
Linux host, a CUDA device and the same LibTorch SDK as the backend build.

From the source root, after building the production LibTorch backend:

```bash
# Set these to the same SDK, toolchain and artifact used by the production backend build.
export TORCHLEAN_LIBTORCH_HOME=/path/to/torch
export TORCHLEAN_LEAN_PREFIX=/path/to/lean
export TORCHLEAN_BACKEND_LIBRARY=/absolute/path/to/libtorchlean_libtorch.so

time scripts/checks/cuda.sh parity --sweep 1024 --keep
```

The runner first resolves the backend library's real path, because generating the reference selects
the CPU Lake profile and can move `.lake/build`. It then builds `native_float32_parity` and the
CMake project in this directory and runs the C ABI checks on the reference cases. `--keep` keeps the
build and reference directory and prints its path, and `--skip-build` reuses the Lean reference
executable (the C++ harness is built either way). The default sweep adds 200000 random cases to the
curated ones. Start with `--sweep 1024` to time a short run. Every case is checked: each FMA case
through one scalar AXPY call, and add, mul, div and sqrt in batches. The reference file must contain
all five primitives.

After building `nn_tests_suite` with the native LibTorch configuration, run the isolated Lean
memory probes against that executable:

```bash
for probe in accounting attention-buffers oom-recovery; do
  TORCHLEAN_REQUIRE_CUDA=1 TORCHLEAN_LIBTORCH_MEMORY_PROBE="$probe" \
    scripts/lake.sh -Kcuda=true env .lake/build/bin/nn_tests_suite
done
```

`Stress.run` launches the same three probes as fresh subprocesses. They check real readback,
logical ownership against the native allocated, reserved and peak counters, live tensors across
`emptyCache`, saved attention inputs across release and backward, buffer finalization, and
recovery from `resourceExhausted` under a temporary allocator memory fraction. No assertion pins
an SDK cache block size or the number of bytes returned to the driver. A build without LibTorch
checks that the native counters are zero and skips the probes.

## Coverage

- Sigmoid and tanh weighted pullbacks are checked at tails where the local derivative would round to
  zero but the weighted result is representable, in binary32 and binary64, with both signs of input
  and gradient. The tail checks are the only ones here with a tolerance: they compare the production
  C ABI with wider host arithmetic to within `8 * epsilon` relative error.
- GELU and SiLU values and weighted pullbacks are checked bit for bit at both infinities, a NaN and
  both zeros: the values are `-0` at `-inf` and `+inf` at `+inf`, and the slopes `-0` and `1` times
  the gradient.
- `TorchLean/Verify/LibTorch/Contract.lean` defines `EqUpToNaN`: equal values, whose
  bits are equal, or two NaNs, with no finite tolerance. The harness follows that relation and
  counts differing NaN encodings separately.
- Adam's three outputs are checked bit for bit on hand-derived fixtures and on 10 × 1024 staged
  host-oracle cases. The fixtures isolate both moment FMAs, the parameter decay, the final update
  FMA, and the separately rounded initial moment products and finite gradient square. Range fixtures
  cover weighting before an overflowing square, taking the corrected RMS without an overflowing
  second moment, and dividing without an overflowing reciprocal. Retained momentum with a tiny
  denominator, cancellation of an overflowing weighted update and a zero learning rate exercise the
  range-repair path: power-of-two rescaling for a nonzero rate, and the product before the division
  for a zero rate. Ordinary entries keep their staged FMA schedule, and exceptional entries are
  checked against independent host scaling. A negative second moment confirms that Adam keeps the
  raw IEEE sqrt, NaN included. Each of the nine host scalar arguments is perturbed in turn: the
  correction denominators are inverted on the host before rounding, and the other coefficients are
  rounded as given. The cancellation residual `2^-46` detects changes that a `1e-6` tolerance would
  accept.
- The Lean reference file validates production add, mul, div and scalar AXPY, and upstream tensor
  `addcmul` and raw sqrt. `Buffer.sqrtClampNegative` runs against the same file with its specified
  `x <= 0 → +0` selection applied.
- FMA cases cover cancellation, overflow cancellation, subnormal and underflow results, signed
  zeros, NaNs, and zero times infinity. The scalar coefficient is checked as a CPU zero-dimensional
  tensor, as a CUDA zero-dimensional tensor and through the production C ABI, and tensor operands
  exercise the other CUDA kernel path.
- A split float32 multiply and add must return zero on the cancellation discriminator, and a float64
  multiply and add must return the **wrong** upper float32 neighbor on the double-rounding
  discriminator. The two negative controls show that the fixtures reject both replacements.
- The test lists 37 buffer exports, and each runs under both ambient `GradMode` settings. An ATen
  `RecordFunction` callback checks that their operations execute with `GradMode` disabled inside
  the export, and that the results have no `grad_fn` or `requires_grad`. The observer is calibrated
  against an ordinary graph-recording ATen multiplication, so no-grad inputs cannot make the test
  vacuous. The runtime rejects gradient-bearing `Buffer` inputs, so a separate gradient-bearing
  ATen tensor checks `invoke` itself. The test switches between both reduction modes through the
  LibTorch settings API, checking the IO result of every read and write and unboxing the `UInt32`
  readback. Strict determinism must disable cuDNN benchmarking, and both settings are restored
  afterward. The caller's `GradMode` and input flags must survive.
- The raw control getter must return an IO error for an unknown setting ID and leave the determinism
  setting unchanged. The allocator memory-fraction getter must return a successful IO result holding
  a finite `Float` in `[0, 1]`.
- Deterministic sum fixtures tell the specified 256-lane tree from a left fold and from a double
  accumulator. They cover partial blocks, recursive partial sums, and grid-stride accumulation above
  the 65535-block cap (one 64 MiB input). Mean rounds the full sum and divides it once by the count:
  on `[7, 0, 0]` that gives `0x40155555`, where multiplying by the rounded reciprocal of 3, a
  negative control, gives `0x40155556`. Empty inputs, signed zero, subnormals and overflow have
  fixtures of their own.
- Running batch means keep identical maximum finite FP32 and FP64 samples without overflow, handle
  opposite-sign extremes, and propagate NaN and infinity. These fixtures test the running-mean
  primitive, which is separate from the sum-then-scale reduction above.
- The sub and div pullbacks to the second operand use the neg kernel, which gives `-0` at `+0` and
  `+0` at `-0`, while the composed `0 - g`, a negative control, gives `+0` at both. Scaling by `-1`,
  the mse target pullback, gives `-0` at `+0` as well.
- The max and min pullbacks select as the CPU rules do: the strict winner receives `g` bit for bit,
  the strict loser `+0` at every `g`, and a tie, opposite zeros or a NaN operand half of `g` each.
- A Lean external object of another class is rejected before its payload is read as a tensor.

A finite sweep tests the cases it draws. Run the selected-gradient, tape traversal, deterministic
reduction, GELU and end-to-end Lean suites as well.

## SDK source audit

The audited SDK reported `torch 2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2, with the C++11
ABI enabled. The public upstream commit `0291f960b6` has `version.txt = 2.12.0a0`. The executable
prints its header version. Keep the full SDK and build manifest with the results, because vendor
patches and compiler options matter too.

The upstream sources at that commit:

- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/cuda/DeviceAddCmulCdiv.cuh>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/native/ufunc/add.h>
- <https://github.com/pytorch/pytorch/blob/0291f960b6/aten/src/ATen/record_function.h>

For CUDA float32, both `addcmul` kernel paths call `pointwise_op_impl<float>`, and with
`value == 1` that helper calls `std::fma(tensor1, tensor2, input)`. Production AXPY therefore
passes the rounded coefficient as **tensor2** and sets **value to one** (CUDA accepts a CPU scalar
only in tensor2 for this operation). AXPY is then one fused stage, with no kernel to allocate the
scalar on the GPU.

For any other `value` the helper first rounds the product, then computes
`std::fma(value, product, input)`, so moving a factor into `value` moves a rounding boundary. The
CUDA `add(alpha)` ufunc evaluates an ordinary `self + alpha * other` expression whose contraction
depends on the build, so we never use it for AXPY and the test does not exercise it.

Binary64 cannot stand in for FMA in general. With `x = 1 + 2^-23`, `y = 1.5` and `z = -2^-80`,
the exact product is the midpoint between float32 `0x3fc00001` and `0x3fc00002`. The tiny negative
addend selects the lower neighbor in a true float32 FMA, but a binary64 add loses it, and the
result rounds to the upper, even neighbor.

The audit above covers only this SDK revision. A different SDK needs its own source audit and
another CUDA run with the same finite-bit assertions.
