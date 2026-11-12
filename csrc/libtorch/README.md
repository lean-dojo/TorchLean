# TorchLean LibTorch backend

`csrc/libtorch` holds the CUDA backend behind TorchLean's GPU buffer ABI: C++ that calls ATen
operators through the LibTorch Stable ABI of torch 2.13, so one built library runs with torch 2.13
and later releases (see [Torch versions](#torch-versions)). Lean checks shapes and dispatches
through the extern symbols. Native memory safety, torch's behavior and floating-point execution
lie outside Lean's kernel.

## Layout

The backend builds into one shared library, `libtorchlean_libtorch.so`, from these files:

| File | Contents |
| --- | --- |
| `torchlean.cpp` | The runtime and the ATen adapters of the buffer API. |
| `conv_pool.cpp` | Grouped and transposed convolutions and the pools, with their backward calls. |
| `tree.cpp` | Tree arenas in device memory, and the launches of the row kernels of folds, expansions and unfolds, level by level. |
| `kernels.cpp` | Generated CUDA kernels, compiled by NVRTC for the device's architecture and launched through the CUDA driver API, and the device count, memory and synchronization the bridge reports. |
| `stable.h` | The call of an ATen operator by its schema, `aten<"name.overload">(...)`, through torch's dispatcher, and what else the bridge takes from the Stable ABI: scalar operands, addition, padding, host views, grad mode, and the CUDA device guard and stream. |
| `schemas.h` | The generated table of the operator schemas the bridge calls. |
| `unstable.cpp`, `unstable.h` | The one source compiled against torch's C++ API: the process-wide backend settings and the CUDA caching allocator's statistics and controls, which the Stable ABI cannot express. |
| `operations.h` | The shared operation list that generates the exports common to both builds. |
| `torchlean_libtorch.h` | The buffer representation, the Lean object, size and error helpers, and the handle functions for native tests. |
| `binary.h` | The NVRTC header, bundled into the library, for configured binary arithmetic. |
| `CMakeLists.txt` | The CMake project `scripts/libtorch_build.py` configures against the SDK. |

A build without LibTorch links `unavailable.c` instead. It exports every symbol that a Lean `extern`
names, reports `RuntimeStatus.notLinked`, and fails every buffer operation with a message that says
how to rebuild. The Lean tape owns differentiation, and native calls run with torch's grad mode off.
Lean loads the CPU tensor library (Lake target `torchlean_tensor_cpu_shared`) for `#eval`, and the
GPU tests run as compiled executables.

On Linux, interpreted GPU calls can load Lean's bundled exception unwinder alongside the unwinder
of the GCC-built bridge and torch. If a caught C++ exception aborts under `#eval` or `lean --run`,
select the host's `libgcc_s.so.1` with `LD_PRELOAD` for that process only, never globally. The
C++ regression harness links the host unwinder (`gcc_s`) ahead of the Lean runtime.

## Build selection

`scripts/lake.sh build` is the default build: the CPU runtime, with no LibTorch SDK or CUDA
toolkit. `-Kcuda=true` builds and links the full LibTorch CUDA backend.

The [LibTorch build instructions](../../scripts/README.md#libtorch-cuda-build) cover SDK
selection, prerequisites, compiler controls, cache directories, and the build and test commands.
The backend also links the toolkit's NVRTC and CUDA driver libraries. Generated custom operations
compile their CUDA source with them at runtime, so they need no handwritten kernels.

## Torch versions

The bridge builds against torch 2.13 or later and runs with the torch runtime the process loads,
torch 2.13 or a later release, without a rebuild, apart from the allocator calls below.
`torchlean.cpp`, `conv_pool.cpp`, `tree.cpp` and `kernels.cpp` are compiled with
`TORCH_TARGET_VERSION` set to torch 2.13 and include from torch only the headers of the LibTorch
Stable ABI. Each ATen operator call names its schema, `aten<"name.overload">(...)`: `stable.h`
checks the arguments against the schema in the generated table `schemas.h` at compile time, and
torch's dispatcher finds the operator by name when the call runs. A scalar operand is a CPU 0-dim
tensor, which ATen's elementwise operators take as they take a `Scalar` overload's wrapped number.
An operator that ATen registers with a device check rejects such a tensor beside a CUDA tensor, so a
scalar operand has its own C++ type, which `aten<Key>` accepts only for the operators of the list
`no_device_check` in `schemas.h`, generated from torch's declarations, and passing one to any other
operator does not compile. Everything else these files take from torch goes through its C shims.
`unstable.cpp` holds what the Stable ABI cannot express, the process-wide backend settings and the
CUDA caching allocator's statistics and controls. It is compiled against torch's C++ API. The
settings functions bind by name, and the allocator calls go through virtual functions at the slots
of the headers the library is built against. Torch 2.13, 2.14 and 2.15 have the same slots, and the
allocator calls fail with an error unless the loaded runtime is one of these releases. The check
reads only the runtime's release, so it guards the slots only for a library built against one of
these releases. [LibTorch Stable ABI](../../docs/TRUST_BOUNDARIES.md#libtorch-stable-abi) lists each
schema, C shim and unstable symbol the library uses, and the operators its C shims run, with the
minimum version and a reason.

`scripts/checks/libtorch_abi.py` checks those lists, and `scripts/lake.sh lint` runs it on the
sources, which also checks that `call` in `stable.h` is the only use of the dispatcher. After each
build, `scripts/libtorch_build.py` runs it on the library and the SDK it is built against and
records the result in `libtorch/abi_check.json`: the SDK's generated operator headers declare each
listed schema verbatim, the SDK defines every torch symbol the library imports, and the library
names exactly the table's operators, imports exactly the listed C shims and nothing else from torch
but the unstable remainder, apart from functions of the C++ standard library that `libtorch.so` also
exports, and does not link the CUDA runtime. A failed check fails the build. With a pip package's
`torchgen` beside the SDK, it also checks `no_device_check` against the package's declarations.
`libtorch_abi.py --torch DIR` checks the operator headers of another torch package without a build,
`--torch DIR --library SO` also checks that the package defines every torch symbol a built library
imports, and `--write --torch DIR` regenerates `schemas.h` after a source adds an operator.

The library finds torch through its RUNPATH, the SDK's `lib` directory, and torch's libraries find
their own dependencies through their own search paths (see the
[build instructions](../../scripts/README.md#libtorch-cuda-build)). `LD_LIBRARY_PATH` comes before
the RUNPATH, so naming another release's `torch/lib`, and its NVIDIA libraries, there makes the
same library run with that release. The library binds every import when it loads (`-z now`), so a
runtime that lacks a function the library imports fails to load and names the function.
`TorchLean.LibTorch.version` reports the release of the loaded runtime.

We build the library with pip torch 2.13.0+cu130 (CUDA 13.0) and run that one library, without a
rebuild, on an NVIDIA A100-SXM4-80GB with the runtimes of torch 2.13.0+cu130 and 2.15.0+cu132: the
CUDA suite, `tree_tests` and the elementwise C++ harness below pass with each. The harness is built
against each release's SDK and computes its reference with that release's ATen. On the cases of one
`--sweep 1024` run, the 43289 values it compares, the library's and the reference's, have the same
bits with both runtimes, NaN payloads included. `libtorch_abi.py --torch DIR --library SO` also
passes with torch 2.14.0+cu130, whose runtime we have not run, and other releases are untested. A
later release can still change numerical results, because the library calls the loaded runtime's
kernels, so rerun the CUDA suite and the elementwise harness with each runtime. Attention is
composed in Lean from numerical primitives, with no private ATen attention selectors or paired
attention kernels.

Two rounding contracts rest on TorchLean's generated kernels rather than on ATen. `Buffer.axpy`
rounds once because its kernel calls `fma`. The ordered sums behind column sums, scan gradients and
row scatters fold each column of a segment in order from its initial value, in one thread.
`TorchLeanTest/CUDA/Numerics.lean` runs both on binary32 and binary64 cases whose results change
when the product rounds first or the fold runs in another order.

An operator's error, like a failure to box its arguments, carries torch's message without its
backtrace: an `IO` operation returns it as an IO error, and a pure buffer operation ends the
process with Lean's internal panic, which prints it. The methods of torch's `torch::stable::Tensor`
print their own failures to standard error, but they fail only on torch's internal errors.

The optional discovery controls are the architecture list and the toolkit root:

```bash
TORCH_CUDA_ARCH_LIST=8.0 scripts/lake.sh -Kcuda=true -Kcuda_home=/usr/local/cuda build
```

TorchLean sets no architecture override. `TORCH_CUDA_ARCH_LIST` controls the SDK's configure
probes and does not rebuild the SDK's packaged GPU kernels. Pass the same SDK and toolkit settings
to every later `build`, `exe` and `env` command.

The build helper `scripts/libtorch_build.py` runs on every Lake build, cache hits included. It
fingerprints the SDK's CMake and version files, its headers and libraries (by file metadata), the
compiler, the build environment, the backend sources and the dependencies CMake discovered. It
writes the manifest to `libtorch/build.json`, the SDK settings to `libtorch/cmake/sdk.txt` and the
ABI check's record to `libtorch/abi_check.json` under the selected build directory. Lake links
`libtorch/libtorchlean_libtorch.so` by its resolved absolute path, so keep that file for execution.
The build environment includes `LD_LIBRARY_PATH`, so a Lake command that sets it to select another
torch runtime rebuilds the library. To run the built library with another runtime, run the built
executables directly with `LD_LIBRARY_PATH` set.

## Generated custom operations

`TorchLean.CUDA.Kernel.Launch` generates CUDA source from typed kernel expressions and hands it to
this bridge. NVRTC compiles machine code for the current device's compute capability, so the
driver never translates PTX, and it disables multiply-add contraction
and flush-to-zero, and rounds division and square root correctly (`--prec-div`, `--prec-sqrt`).
The compilation inherits no fast-math setting from the caller. These options apply only to
generated kernels, not to LibTorch's own operations.

The process keeps at most sixteen compiled modules, keyed by source and CUDA context, and an
evicted module stays loaded while a call holds it. Each operation launches on LibTorch's current
stream with read-only inputs and a fresh output. Before returning, the bridge reads the device's
bounds record, and an out-of-bounds read becomes an IO error instead of a partial output.
Compilation errors include NVRTC's log. An operation with an empty output is compiled but not
launched.

Custom tensor computations move complete scalar words through ATen as bytes and keep those words
for saved values and gradients on the same tape. Native binary32 and binary64 keep the hardware
arithmetic, and configured binary formats share the device implementation in [`binary.h`](binary.h).
It uses a native IEEE operation where the complete format and the GPU architecture allow it, and
integer-limb arithmetic for custom formats and for half and bfloat16 division. NaN handling follows
the configured payload rules. The bridge supplies the header to NVRTC in memory, so execution needs
no source files. The [Lean emitter](../../TorchLean/CUDA/Kernel/Emit/Binary.lean) selects the format
and emits exact literals, which keeps wide encodings, signed zeros and NaN metadata without
narrowing wide values through native floats. Device arithmetic is tested against FloatLib, not
proved by Lean, and adds no dtypes to the model runner.

The [ordinary-Lean frontend](../../TorchLean/NN/Kernel/Function.lean) recognizes a supported scalar
subset, and the [graph runner](../../TorchLean/NN/Kernel/Graph.lean) combines custom bodies with
supported canonical IR operations. The
[custom tensor example](../../home_page/examples/custom-computations/index.md) shows the public API
and its limits. Arbitrary Lean functions do not compile to CUDA. The NVRTC, driver, memory and
execution boundary lies outside Lean's proofs.

## Execution and memory

LibTorch supplies PyTorch's C++ libraries, and ATen is the tensor operation layer this backend
calls. A CUDA buffer reaches C++ as a Lean external object that owns a `torch::stable::Tensor`, a
handle to a torch tensor. The bridge unwraps the tensor, calls ATen operators through torch's
dispatcher and returns the result as another owned buffer through the Lean C ABI. The backend
needs no Python interpreter, although an installed CUDA-enabled PyTorch package can supply the SDK
for building and running.

Configure the selected device and the process-wide SDK settings before any concurrent runtime
work, as `TorchLean.CUDA.LibTorch.Controls` requires. The live-wrapper check prevents sequential
device changes while any owner is live, but it is not a lock against concurrent configuration and
allocation. Releasing a buffer also needs exclusive use of it. The atomic counters stay correct
with concurrent independent owners, but mutating one owner from several threads is unsafe.

For a linear layer, the CUDA entry of `Op.linear` in `LibTorch.Kernel.ofOp` sends the matrix product
and the bias addition to the buffer API, and `LibTorch.Tape.apply` records the result, its parents
and its backward rule on the runtime tape. During backward, Lean traverses the tape and calls the
matching gradient operations. Each native call runs with torch's grad mode off, so LibTorch
records no autograd graph of its own. Some operations call ATen's backward kernels, such as
`aten::convolution_backward`. Attention's forward and local VJP are composed in
`TorchLean/NN/LibTorch/Attention.lean`. `Buffer.attentionForward` takes
Q, K and V, an optional mask, the dimensions and the scale, and returns
`Except String (Buffer × Buffer)` for the output and the probabilities. `Buffer.attentionBackward`
takes Q, K and V, those probabilities, the output cotangent, the dimensions and the scale. The tape
keeps and releases these saved buffers like any others, with no native attention context or fused
attention selection. The full score and probability matrices take memory quadratic in the sequence
length.

Spectral layers compose FFT, frequency mixing, and inverse FFT in Lean from the backend's
numerical primitives. Model composition and saved-buffer ownership stay with TorchLean.

The grouped convolution `Buffer.conv` and its backward `Buffer.convBackward` fold the axes before
the channel axis into a batch and call ATen with PyTorch's packed grouped kernel. They support
dilation and separate zero padding before and after each axis: ATen pads a nonempty input with
symmetric padding itself, and any other padding is materialized first. Both check the geometry
and the buffer lengths before calling ATen. On invalid metadata the forward returns an IO error
and the backward aborts, as every pure primitive does. Neither records gradients. The CUDA tape
and the graph runner `IR.Graph.runBuffers` call them through `Op.conv`'s entry in
`LibTorch.Kernel.ofOp`.

Resident buffers hold binary32 for `Float32` and binary64 for `Float` through the forward pass,
local VJPs, gradient accumulation and optimizer updates. Allocation and transfer take a dtype,
binary32 when a call names none, and operation outputs inherit it. The resident graph runner
(`IR.Graph.runBuffers`) and its custom programs take binary32 buffers only. Checkpoints keep
binary64 parameters and moments at full width. Model sessions in other scalar formats use the CPU
tape, without narrowing. Configured-binary custom arithmetic can record on the GPU tape, but native
model operators and optimizer checkpoints stay binary32 or binary64.

Native batch training averages the unscaled sample losses and gradients with running means. For
finite operands the update computes `mean + w * (sample - mean)` when the signs agree and
`(1 - w) * mean + w * sample` when they differ, so neither form overflows. It clamps the result
between the two operands to contain rounding overshoot. It does not sum preweighted
samples, which can overflow near the dtype's range limit. A NaN or an infinite operand propagates
through the weighted form.

Memory policy of the eager CUDA tape:

- Trainable parameters stay on the device as cached mirrors across eager CUDA steps.
- Optimizer steps can update those mirrors on the device.
- Forward scratch buffers kept only for the backward pass are listed on their tape nodes and
  released after the step.
- Gradient accumulation releases the dense-gradient buffers it overwrites.
- After each CUDA tape is retired, in training and in evaluation, `Buffer.collectGarbage` returns
  unused host allocator pages to the system, and LibTorch keeps its device cache for reuse.

Releasing buffers at these points means they seldom wait for Lean's external object finalizers.
Execution is eager, with no CUDA Graph capture or replay.

## Sanitizers and profilers

The [GPU regression instructions](../../scripts/README.md#gpu-regression-tools) cover the Compute
Sanitizer tools, SDK selection and the suite's subprocess instrumentation. The sanitizer runs
exercise the native boundary and the SDK on the paths the tests take, and a clean run is evidence
for those paths only.

For performance work, pair the correctness suite with NVIDIA Nsight Systems for end-to-end traces
and Nsight Compute for per-kernel profiles. They are profilers with no pass or fail result, so the
test suite does not run them. Run them on the executable under study:

```bash
scripts/lake.sh -R -K cuda=true build nn_tests_suite
scripts/lake.sh -R -K cuda=true env nsys profile -t cuda,nvtx,osrt \
  -o /tmp/torchlean-cuda .lake/build/bin/nn_tests_suite
scripts/lake.sh -R -K cuda=true env ncu --section SpeedOfLight \
  --section LaunchStats .lake/build/bin/nn_tests_suite
```

Nsight Compute can be slow on the full suite. Use a focused executable for kernel-level work.

## CUDA tests

The CUDA regression suite is in `TorchLeanTest/CUDA`. Most of its tests compare the Lean CPU
eager tape with the CUDA eager tape on small examples. Apart from the CPU run of
`KernelGradient.lean`, they run only in a `-Kcuda=true` build. The default build skips them, so a
CPU build says nothing about GPU execution. The tree kernels have their own tests, which
`scripts/lake.sh -Kcuda=true exe tree_tests` runs on the host and the device.

Coverage by test module:

| Test module | Main coverage |
| --- | --- |
| `TorchLeanTest/CUDA/Numerics.lean` | Fused `Buffer.axpy`, ordered sums, and cross-entropy retaining a representable combined tail when individual exponentials underflow, in binary32 and binary64. |
| `TorchLeanTest/CUDA/Softmax.lean` | `softmax` and `log_softmax`, forward and backward. |
| `TorchLeanTest/CUDA/Elementwise.lean` | Scalar elementwise ops, activations, safe logs, products, `sum`, `dropout` against its binary32 specification, and the Huber terms and slopes (`huberTerms`, `huberSlope`) against the CPU rules. |
| `TorchLeanTest/CUDA/Attention.lean` | Attention values and gradients against the CPU reference, including batches. |
| `TorchLeanTest/CUDA/SelectiveScan.lean` | Diagonal selective-scan buffer primitives used by the Mamba/SSM runtime path. |
| `TorchLeanTest/CUDA/Graph.lean` | Graph evaluation through the CUDA entries: slices, products, pooling, layer and batch normalization, grouped dilated convolution with its tape pullbacks, and `detach`. |
| `TorchLeanTest/CUDA/FFT.lean` | Packed real FFT, inverse FFT, spectral convolution, and finite-difference gradient checks. |
| `TorchLeanTest/CUDA/Stress.lean` | RNG determinism, explicit release, duplicate-parent gradient accumulation, large buffers, reductions, the division by a count (`divCount`), the mean squared error and its slope (`mseMean`, `mseSlope`), and rectangular matmul. |
| `TorchLeanTest/CUDA/Trainer.lean` | Training state, frozen parameters, optimizer history, finite batch means at FP32/FP64 range limits, failure without state mutation, and DQN Huber loss on the CUDA eager tape. |
| `TorchLeanTest/CUDA/Indexing.lean` | `gather`, `indexAt` and `scatter .sum` against the CPU tape in `Float32`, values and gradients bit for bit, with deterministic reductions. |
| `TorchLeanTest/CUDA/ReduceTree.lean` | The deterministic flat sum `Buffer.reduceSum` against its tree specification `LibTorch.reduceSumTree`, bit for bit (a NaN result as a NaN), on fixtures: empty and one-element inputs, cancellation, block boundaries, signed zeros, overflow, subnormals and NaN. |
| `TorchLeanTest/CUDA/ParameterAliases.lean` | Parameter snapshots in recordings, tied AdamW parameters, rejection of a retied layout, and the AdamW checkpoint round trip. |
| `TorchLeanTest/CUDA/MoE.lean` | `nn.moe` gating and mixing on ties, signed zeros and NaNs, values and both gradients against the CPU tape. |
| `TorchLeanTest/CUDA/Scheduler.lean` | Learning-rate schedules through `Optimizer.setLr` for SGD and Adam. |
| `TorchLeanTest/CUDA/SessionRefIdentity.lean` | Backward entry points refuse foreign, unowned and stale output references on a CUDA session. |
| `TorchLeanTest/CUDA/KernelGradient.lean` | The tangent and gradient kernels of four custom operations in binary32 and binary64, against their analytic derivatives on the CPU and against the CPU bits on the GPU. |
| `TorchLeanTest/CUDA/Suite.lean` | The unified entrypoint the repository suite imports. |

Elementwise arithmetic, activations and whole-buffer reductions share the operation list in
`operations.h`, which generates both the LibTorch exports and their counterparts in
`unavailable.c`. The shared call adapter converts the arguments, checks buffer lengths and dtypes,
rounds scalar arguments to the buffer dtype, turns off native autograd and boxes the result. Each
operator's schema is checked at compile time, with no operator-ID switch, and torch's dispatcher
finds the operator by name when the call runs. Operations with shape metadata or saved backward
state keep adapters of their own.

To add a symbol that Lean calls outside that list, add its failing export to `unavailable.c`, add a
row to the table above, and test it against the Lean CPU tape at least once. If the symbol takes
part in autograd, test the forward value and the VJP or gradient buffers. If it uses atomics, decide
whether deterministic mode needs a test of its own.

The separate [elementwise C++ harness](tests/elementwise/README.md) links the production backend
and reads Lean binary32 cases. Its wrapper is
`scripts/checks/cuda.sh parity --libtorch-home PATH --backend-library PATH --lean-prefix PATH`.
It checks finite results and signed zeros bit for bit, reports differing NaN encodings, and
includes staged Adam and no-autograd regressions. The harness computes its oracle with ATen's C++
API, so it is rebuilt against each torch release it runs with, and links the backend library as
built. A different torch runtime needs its own harness and suite results.

Convolution values and their kernel, bias and input pullbacks are compared exactly with
hand-computed sums and with the CPU rule of `Op.conv`, and `gather`, `indexAt` and `scatter .sum`
bit for bit with the CPU tape. Pooling values and batch normalization with running statistics are
compared with hand-computed values that are exact in binary32, and the suite has no test of pooling
gradients. Layer normalization is compared only with the CUDA entry on the matrix of its axes.
Batch normalization with batch statistics, which the CUDA entry composes from row reductions, is
checked only through the running statistics that training steps leave behind in
`TorchLeanTest/CUDA/Trainer.lean`, within `1e-5`. Transposed convolution has no row in the suite.

## Numerics and SDK changes

- A torch upgrade can change numerical behavior, with or without a rebuild, because the library
  calls the loaded runtime's kernels. ABI compatibility says nothing about agreement with
  TorchLean's floating-point contracts, so keep the build manifest and the loaded runtime's
  release with the test results. For example, the CUDA `relu` and `clamp_min(x, 0)` of torch 2.15,
  which the bridge does not call, return `-0` at `-0`, where those of torch 2.13 return `+0`.
- The deterministic controls request the SDK's supported deterministic algorithms. Repeatable
  results on the tested SDK and device do not imply bitwise agreement across releases, devices or
  algorithms.
- The sigmoid and SiLU values and pullbacks and the tanh and GELU pullbacks call the device `exp`,
  and the tanh and GELU values the device `tanh`. The CUDA Math API documents `expf` within 2 ulps
  and `exp` within 1 ulp, so these words can differ from those of the Lean rules, which call the
  host functions. The bitwise sigmoid and tanh pullback rows of `checkPointwiseVJPBits`
  (`TorchLeanTest/CUDA/Elementwise.lean`) sit at `x = ±0`, where the device `exp` returns
  `1` exactly, and for sigmoid in binary32 at `x = -200`, where the weighted slope rounds to zero.
- `max` and `min` are the IEEE 754-2019 maximum and minimum (`ieee_maximum` and `ieee_minimum` in
  `torchlean.cpp`): `aten::maximum` and `aten::minimum` propagate a NaN, and at a pair of zeros a
  sum picks the larger or the smaller one. `clamp` is `minimum (maximum x lo) hi` with them, so the
  CUDA entries of `Op.max`, `Op.min` and `Op.clamp` compute the CPU tape's values. The CUDA tests
  compare the `max` and `min` pullbacks with the CPU rules bit for bit at ties, infinities, signed
  zeros and NaN operands, and the forward values on the finite nonzero inputs of a composite test.
- Attention uses matrix products composed in Lean and the softmax VJP over probabilities the tape
  owns. The attention tests in the compiled Lean CUDA suite compare values and gradients with the
  CPU reference and exercise saved-buffer ownership.
- Run the GPU suite after any change to native exports, ownership or numerics.
