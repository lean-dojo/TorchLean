# Elementwise C ABI regression

The C++ executable `torchlean_elementwise_regression`, built from `regression.cpp`, links the
production backend, ATen and the Lean runtime, and checks the backend's elementwise C ABI against
the Lean binary32 reference and against host arithmetic. It needs a Linux host and a CUDA device.
The harness computes its oracles with ATen's C++ API, so it is built against the torch package or
SDK it runs with. The backend can be a library built against an earlier release (see
[Torch versions](../../README.md#torch-versions)), but its allocator calls fail on a runtime other
than torch 2.13, 2.14 and 2.15. A tensor crosses between the two as an `AtenTensorHandle`, which
both read as the same ATen tensor.

From the source root, after building the production LibTorch backend:

```bash
# The torch package or SDK to run with, the backend's Lean toolchain and the built backend.
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
all five primitives. To run the backend with a release other than the one it is built against,
set `TORCHLEAN_LIBTORCH_HOME` to that release and put its `torch/lib` and NVIDIA libraries first in
`LD_LIBRARY_PATH`.

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
  `RecordFunction` callback checks that their operations execute with `GradMode` disabled inside the
  export, and that the results have no `grad_fn` or `requires_grad`. The observer is calibrated
  against an ordinary graph-recording ATen multiplication, so no-grad inputs cannot make the test
  vacuous. The callback sees the backend's calls, which go through torch's dispatcher. A `Buffer`
  never holds a gradient-bearing tensor (the Stable ABI cannot read the flag, so the harness refuses
  to box one), so a separate gradient-bearing ATen tensor checks `invoke` itself. The test switches
  between both reduction modes through the LibTorch settings API, checking the IO result of every
  read and write and unboxing the `UInt32` readback. Strict determinism must disable cuDNN
  benchmarking, and both settings are restored afterward. The caller's `GradMode` and input flags
  must survive.
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

## Fused multiply-add

Production AXPY (`Buffer.axpy`) does not use an ATen operator. A kernel the backend generates
computes `fma(b, c, a)` with `fmaf` or `fma`, after rounding the coefficient `c` to the operands'
format, and NVRTC compiles it with `--fmad=false`. Each entry rounds once with any torch runtime.

The harness also checks ATen's `addcmul` with `value` one on every FMA case, as an oracle
independent of the backend: with the coefficient as a CPU 0-dim tensor and as a CUDA 0-dim tensor
in `tensor2`, and with tensor operands. For CUDA float32 and `value == 1`, both of its kernel paths
call `pointwise_op_impl`, which returns `std::fma(tensor1, tensor2, input)`. For any other `value`
it first rounds the product, then computes `std::fma(value, product, input)`, so moving a factor
into `value` moves a rounding boundary. `ATen/native/cuda/DeviceAddCmulCdiv.cuh` holds that helper,
the same file in the headers of torch 2.13.0 and 2.15.0. These checks test the loaded runtime's
`addcmul`, so a failure in them alone points at torch, not at the backend. The CUDA `add(alpha)`
ufunc evaluates an ordinary `self + alpha * other` expression whose contraction depends on the
build, so the backend never uses it for AXPY and the test does not exercise it.

Binary64 cannot stand in for FMA in general. With `x = 1 + 2^-23`, `y = 1.5` and `z = -2^-80`,
the exact product is the midpoint between float32 `0x3fc00001` and `0x3fc00002`. The tiny negative
addend selects the lower neighbor in a true float32 FMA, but a binary64 add loses it, and the
result rounds to the upper, even neighbor.

The executable prints the version of the LibTorch headers it is built against. Keep it with the
results, with the backend's `TorchLean.LibTorch.version` and build manifest.
