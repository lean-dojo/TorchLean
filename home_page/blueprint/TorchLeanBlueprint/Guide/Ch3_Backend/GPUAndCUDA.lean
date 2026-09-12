import VersoManual
-- Everything the live blocks in this chapter evaluate: the tensor type with its literal notation,
-- the backend report and the profile table, the native-CUDA capsule registry, the reference CPU
-- capsules the CUDA ones get compared against, and the two softmax specifications the masking
-- section contrasts.
import NN.Tensor
import NN.Backend.Report
import NN.Backend.NativeCUDA
import NN.Backend.Reference
import NN.Spec.Layers.Activation
import NN.Spec.Layers.Attention
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- `Tensor` and its list literals live under `TorchLean`; capsules, profiles, and reports live under
-- `NN.Backend`. Opening all three keeps the code blocks inside Verso's column width, and the
-- printed output still names every capsule and profile in full.
open TorchLean
open NN
open NN.Backend

-- The plan report prints wider than this file's 100-column limit, so its `leanOutput` block asks
-- for `whitespace := lax` and is wrapped in the source. The rendered page shows each line exactly
-- as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "GPU and CUDA" =>
%%%
tag := "gpu-and-cuda"
file := "From-A-Tensor-Operation-To-A-GPU-Kernel"
%%%

When the running MLP evaluates

$$`y=W x+b,`

the mathematical specification sees matrix-vector multiplication and addition. The CUDA runtime
sees something quite different: contiguous buffers, dimensions, an execution stream, a matrix
kernel, a broadcast, and several error checks. TorchLean keeps both views and records the contract
at the boundary between them.

We will follow the linear layer through its buffers, selected kernel, and backward rule, then
examine how reduction order affects its numerical contract. CUDA is the maintained accelerator
today. The capsule and device types also allow a future Metal, ROCm, TPU, or custom-chip provider
to state the same kinds of obligations; those device names do not imply an available runtime.

# Native CUDA Build Configuration

An ordinary CPU build compiles stub archives for CUDA symbols. The stubs let the package link on a
machine without the NVIDIA toolchain, but they reject CUDA session creation. To compile the native
implementation:

```terminal
# Compile the native provider and its CUDA library
# dependencies.
scripts/lake.sh build -R -K cuda=true
```

Run this command from the repository root. `-R` rebuilds targets affected by the Lake configuration,
and `-K cuda=true` selects the CUDA source and link configuration. The build compiles TorchLean's
CUDA code and links the CUDA runtime, cuBLAS, and cuFFT where those libraries are used.

Compilation and session creation answer different questions. A native build can be produced on a
machine that has the toolchain but no visible GPU. Its symbols are present, yet a CUDA session
still cannot allocate a usable device buffer. Conversely, a successful CPU build says nothing
about the native implementation: its stubs deliberately preserve the linking interface so CPU
users can import the same Lean modules.

Run two optimizer steps and print the selected kernel contracts:

```terminal
# Exercise two updates and print the contracts selected by
# this session.
scripts/lake.sh -R -K cuda=true exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

The model reports the same 25-example dataset as the CPU run. The backend report has one line per
selected capsule, followed by one line per contract obligation when the detailed report is
requested:

```
matmul: native_cuda.matmul provider=native-cuda trust=checked vjp=backend-vjp
    reduction=implementation-defined
    shape: shape safety for matmul; guarded at runtime by ...
    layout: ...; ...
    value: matmul forward refines its TorchLean semantics; covered by test suite ...
    vjp: ...; ...
add: native_cuda.add provider=native-cuda trust=checked vjp=backend-vjp reduction=n/a
relu: native_cuda.relu provider=native-cuda trust=checked vjp=backend-vjp reduction=n/a
mse_loss: native_cuda.mse_loss provider=native-cuda trust=checked vjp=backend-vjp
    reduction=implementation-defined
```

The model description contains two *linear layers*. Each becomes a sequence of reshapes,
permutations, matrix multiplications, broadcasts, and additions, so the report names operations
below the level of a whole layer. Each capsule describes an operation that crossed a backend
boundary. The only numerical field is the reduction policy; a native accumulation is marked
implementation-defined so that a fixed-left range certificate cannot be applied to it.

Read this report as an inventory of selected operation contracts. Two linear layers can reuse the
same matmul capsule, and one operation can launch more than one native kernel. Counting the printed
rows therefore gives neither the number of layers nor the number of launches. The two optimizer
steps exercise the chosen route, but the report contains no timing measurement from which to infer
its speed.

# Linear Layer Execution

For an unbatched input `x : [2]` and weight `W : [8,2]`, the eager CUDA path proceeds roughly as:

```
typed Tensor α [2]
    ↓ upload / existing CUDA handle
opaque contiguous float32 buffer
    ↓ reshape and matrix-layout preparation
native_cuda.matmul
    ↓ broadcast bias [8] to output shape
native_cuda.add
    ↓
typed runtime Tensor α [8]
```

In this path, the weight has eight rows of two elements. Multiplication turns the two input
coordinates into eight outputs, and the final addition uses one bias for each output. The Lean
wrapper knows these logical shapes and element counts. The device buffer is opaque; Lean does not
inspect its contents by reducing a theorem. Before each FFI call, wrappers check the conditions
they can observe:

- the session really targets CUDA;
- the selected capsule implements the requested operation;
- flat lengths match the logical shapes;
- every shape axis, runtime index, and element count fits the `UInt32` CUDA ABI;
- ranks, axes, strides, and operation-specific dimensions are supported.

Shape-erased values, output seeds, and supplied gradients pass through `AnyBuffer.validate` before
shape-derived indexing. Convolution and pooling also check spatial products independently, so a
zero channel dimension cannot hide an overflowing spatial shape. Their forward wrappers validate
the native output length immediately after the FFI returns, before storing the result on the tape.

Those checks prevent many ABI and memory errors. They do not prove that a CUDA thread computes the
correct arithmetic expression.

The backward shapes make the same division of responsibility concrete. If the output cotangent is
`δ : [8]`, the input cotangent has shape `[2]` and is mathematically `Wᵀδ`. The weight cotangent has
shape `[8,2]`, with entry `δ[i] * x[j]`, and the bias cotangent is `δ` itself. These formulas
explain
which operands backward must retain. A buffer with the right length is necessary for all three
computations, but length alone cannot show that the right operands or transpose flags were used.

For a concrete row, take weights `[1, 2]`, input `[0.25, -0.75]`, and bias `0.1`. That neuron's
preactivation is `-1.15`. With cotangent one, its contribution to the input gradient is `[1, 2]`
and its weight gradient is `[0.25, -0.75]`. If the next operation is ReLU, however, the negative
preactivation makes the incoming cotangent zero. The tape must compose those local rules in the
right order to obtain the model's gradient.

# Kernel Capsules

A `KernelCapsule` is the audit record for one operation-provider pair:

```
-- The record keeps dispatch identity separate from the
-- evidence for each obligation.
structure KernelCapsule where
  name             : String
  op               : BackendOp
  provider         : Provider
  device           : Device
  trustLevel       : TrustLevel
  supportsForward  : Bool
  vjpMode          : VJPMode
  shapeContract    : ContractDescriptor
  layoutContract   : ContractDescriptor
  valueContract    : ContractDescriptor
  vjpContract      : ContractDescriptor
  numericalPolicy  : NumericalPolicy
```

The source gives two fields defaults omitted from this listing: `supportsForward` defaults to
`true` and `vjpMode` to `.none`. A capsule that leaves these fields at their defaults therefore
claims forward execution and no backward support. Registering it still requires evidence for the
forward claim.

The four descriptors separate shape, storage layout, forward values, and backward values. The
registered native matmul capsule shows how each obligation is recorded:

```lean (name := gpuCapsuleRecord)
-- Inspect the four obligations and the reduction policy
-- attached to native matmul.
#eval NativeCUDA.matmul
```

```leanOutput gpuCapsuleRecord (whitespace := lax)
{ name := "native_cuda.matmul",
  op := NN.Backend.BackendOp.matmul,
  provider := NN.Backend.Provider.nativeCuda,
  device := NN.Backend.Device.cuda,
  trustLevel := NN.Backend.TrustLevel.checked,
  supportsForward := true,
  vjpMode := NN.Backend.VJPMode.backendVJP,
  shapeContract := ContractDescriptor(NN.Backend.ContractClaim.shapeSafety
      (NN.Backend.BackendOp.matmul), runtimeGuard(CUDA FFI size/rank checks at the Lean/native
      boundary)),
  layoutContract := ContractDescriptor(NN.Backend.ContractClaim.layoutCompatibility
      (NN.Backend.BackendOp.matmul) (NN.Backend.TensorLayout.flatRowMajor), runtimeGuard(flat
      row-major Cuda.Buffer layout checks)),
  valueContract := ContractDescriptor(NN.Backend.ContractClaim.valueRefinement
      (NN.Backend.BackendOp.matmul), testSuite(NN.Tests.Runtime.Cuda.Suite)),
  vjpContract := ContractDescriptor(NN.Backend.ContractClaim.vjpRefinement
      (NN.Backend.BackendOp.matmul) (NN.Backend.VJPMode.backendVJP),
      testSuite(NN.Tests.Runtime.Cuda.Suite)),
  numericalPolicy := { reduction := NN.Backend.ReductionPolicy.implementationDefined } }
```

Every claim in that record is paired with its evidence. Shape and layout say `runtimeGuard` and
name the check that runs at the boundary. Value and VJP say `testSuite` and name
`NN.Tests.Runtime.Cuda.Suite`. These entries give two different kinds of assurance: a guard checks
the current call, while a test suite compares results on its selected cases. There is no theorem
entry claiming that the CUDA C source refines the specification. The `implementationDefined`
reduction policy further limits which arithmetic certificates can apply, as we will see below.

## Shape contract

For matrix multiplication, the inner dimensions must agree and determine the result shape.
Lean-side runtime guards and shape-indexed source objects often support this field. In the linear
layer above, multiplying `[8,2]` by `[2]` must produce `[8]`.

## Layout contract

For contiguous row-major storage, `[m,n]` means that the last axis varies fastest: the first
`n` elements form the first row. The wrapper and native kernel must use the same interpretation.
A correct matrix formula paired with a transposed native layout still computes the wrong model.

For the `[8,2]` weight, row `i` occupies flat positions `2*i` and `2*i+1`. A column-major reading
would instead group values from different output neurons. Both interpretations use sixteen
scalars, which is why the shape and layout descriptors are separate. Reshaping a view can preserve
storage, whereas changing this interpretation may require different strides or a physical copy.

## Value contract

This descriptor names the mathematical function the forward result should refine. Evidence is a
runtime guard, a test suite, an explicit trusted boundary, or `notApplicable`. There is no theorem
variant for a Lean proof covering the native code. The evidence variant is visible in reports, so
the reader can distinguish a checked input condition from a tested numerical result.

## VJP contract

The VJP contract records backward ownership and the derivative specification to implement.
`backend-vjp` means the runtime calls a backend derivative kernel while retaining TorchLean's
tape structure.
`torchLeanTape` means TorchLean owns the graph and reverse traversal even if a capsule uses a named
backend kernel for its local VJP. A capsule marked `backend-vjp` makes that local numerical boundary
explicit in the audit.

Capsules record contracts. Runtime code supplies a typed `KernelHandler` for the result type of the
operation. Binding produces an `ExecutableKernel` only when the handler and capsule have equal
operation, provider, and device fields:

```
-- Binding compares these declared identities before calling
-- the stored IO action.
structure KernelHandler (β : Type) where
  name     : String
  op       : BackendOp
  provider : Provider
  device   : Device
  execute  : KernelCapsule → IO β

structure ExecutableKernel (β : Type) where
  capsule             : KernelCapsule
  handler             : KernelHandler β
  operation_matches   : handler.op = capsule.op
  provider_matches    : handler.provider = capsule.provider
  device_matches      : handler.device = capsule.device
```

The identity check rejects a handler whose declared provider or device disagrees with the capsule.
It does not inspect the closure body: a falsely labelled handler would still require source review
and execution checks. Numerical correctness depends on the guard, test, or trusted-boundary
evidence.

The proof fields in `ExecutableKernel` identify the call being made. They prevent accidentally
attaching a CPU handler to a CUDA contract, even when both implement matmul and return the same
Lean type. The function stored in `execute` remains an `IO` computation. Its type does not encode
the matrix formula, and its identity equalities do not derive that formula from the closure.

To isolate the identity check, pair the registered native matmul capsule with a CPU reference
handler for the same operation:

```lean (name := gpuHandlerIdentity)
-- Keep the operation fixed and change provider/device to
-- expose the binding check.
#eval do
  let cap := NativeCUDA.matmul
  let cpu : KernelHandler Unit :=
    { name := "reference_cpu.matmul"
      op := .matmul
      provider := .torchLean
      device := .cpu
      execute := fun _ => pure () }
  let capDev := cap.device.cliName
  let cpuDev := cpu.device.cliName
  IO.println s!"capsule : {cap.name} @ {capDev}"
  IO.println s!"handler : {cpu.name} @ {cpuDev}"
  IO.println s!"bindable: {cpu.matchesCapsule cap}"
```

```leanOutput gpuHandlerIdentity (whitespace := lax)
capsule : native_cuda.matmul @ cuda
handler : reference_cpu.matmul @ cpu
bindable: false
```

The operations agree. The provider and the device do not, so no `ExecutableKernel` can be built and
this mismatched handler is rejected before execution.

The four contract fields get the same treatment from `KernelCapsule.contractsAligned`. A
value-refinement descriptor names a different obligation from a shape descriptor. Moving it into
the shape field therefore makes the capsule inadmissible, even though the descriptor itself has
not changed:

```lean (name := gpuMisplacedEvidence)
-- Put valid value evidence in the wrong field and observe
-- alignment and admission.
#eval do
  let good := NativeCUDA.matmul
  let moved :=
    { good with shapeContract := good.valueContract }
  let policy := BackendProfile.checkedCuda.policy
  IO.println s!"as registered : {good.contractsAligned}"
  IO.println s!"evidence moved: {moved.contractsAligned}"
  let ok := KernelCapsule.admissible policy moved
  IO.println s!"planner admits: {ok}"
```

```leanOutput gpuMisplacedEvidence (whitespace := lax)
as registered : true
evidence moved: false
planner admits: false
```

The third line connects the descriptor check to selection. `KernelCapsule.admissible` calls
`contractsAligned`, so the planner rejects the misplaced descriptor before selecting the capsule.

# Planning, Selection, And Execution

Selection proceeds from descriptions to an executable call:

```
registry
  all known capsule descriptions
       ↓ profile + availability + contract check
accepted plan
  capsules allowed under this policy
       ↓ bind matching typed handler
executable kernel
  operation/provider/device identities agree
       ↓ provider-aware runtime dispatch
executed operation
  the native symbol actually called
```

At the first transition, `BackendProfile.acceptGraph` exposes an `AcceptedGraphKernelPlan` only
after planning, grouping, and the contract check accept every obligation. Binding then checks the
handler identities shown above. Neither step probes the hardware: a profile's `Availability`
declares which devices and providers planning may consider, while CUDA session creation calls
`Cuda.Buffer.requireNativeRuntime` to distinguish native CUDA, a native build with no visible GPU,
and the host-memory parity stubs.

A profile is a small piece of data, so the first two stages are computable on a machine with no GPU
in it. The three maintained profiles describe themselves:

```lean (name := gpuProfiles)
-- Compare forward assurance with ownership of the overall
-- backward traversal.
#eval do
  IO.println BackendProfile.checkedCpu.summary
  IO.println BackendProfile.checkedCuda.summary
  IO.println BackendProfile.libTorchForwardCuda.summary
```

```leanOutput gpuProfiles (whitespace := lax)
profile=checked_cpu device=cpu assurance=checked vjp=torchlean-tape
profile=checked_cuda device=cuda assurance=checked vjp=torchlean-tape
profile=libtorch_forward_cuda device=cuda assurance=external vjp=torchlean-tape
```

In the third line, `assurance=external` records that LibTorch supplies forward values, while
`vjp=torchlean-tape` records that TorchLean still owns backward traversal. These are independent
choices: using a library for forward values need not transfer the tape to that library.

This also explains why `vjp=torchlean-tape` in the profile summary and `vjp=backend-vjp` in the
matmul row can appear together. The profile describes who assembles and traverses the whole
backward computation. The capsule describes how one node computes its contribution. TorchLean can
walk the tape while a native kernel evaluates that node's matrix derivative.

Planning produces the same kind of data, so the CUDA plan for one operation can be printed here, on
a CPU build, without executing a GPU kernel:

```lean (name := gpuPlanReport)
-- Plan one CUDA operation from metadata; printing the plan
-- does not launch it.
#eval do
  let p := BackendProfile.checkedCuda
  match p.planReport #[.matmul] with
  | .ok report => IO.println report
  | .error e => IO.println s!"planning failed: {e}"
```

```leanOutput gpuPlanReport (whitespace := lax)
profile=checked_cuda device=cuda assurance=checked vjp=torchlean-tape
trusted external boundary: none
  matmul: native_cuda.matmul provider=native-cuda trust=checked vjp=backend-vjp
    reduction=implementation-defined
    shape: shape safety for matmul; guarded at runtime by CUDA FFI size/rank checks at the
    Lean/native boundary
    layout: flat-row-major layout compatibility for matmul; guarded at runtime by flat row-major
    Cuda.Buffer layout checks
    value: matmul forward refines its TorchLean semantics; covered by test suite
    NN.Tests.Runtime.Cuda.Suite
    vjp: matmul backend-vjp VJP refines its TorchLean semantics; covered by test suite
    NN.Tests.Runtime.Cuda.Suite
```

The report names two runtime guards and two test suites. Its `checked` label must be read together
with those fields; it does not mean that matmul's native implementation has been proved correct.
The line `trusted external boundary: none` records that this plan added no external library to the
trust assumptions. The value contract has no theorem variant to select.

Availability and dispatch are checked at different points:

1. a plan rejects providers marked unavailable in its supplied availability metadata;
2. provider-aware wrappers reject a selected provider they have not wired up.

Most fixed CUDA wrappers currently require `provider = nativeCuda`. If a profile selects an
unwired provider, execution fails with an error. There is no hidden CPU fallback for an unsupported
CUDA operation, because moving a tensor between devices behind the user's back would change both
performance and the execution claim.

# Runtime Availability Errors

Build without native CUDA and request it:

```terminal
# Show that a CPU stub build rejects a requested CUDA
# session.
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cuda --steps 1
```

On a stub build, session initialization rejects the request. The CLI's `Device.cuda` value
describes the requested target, while
`Cuda.Buffer.requireNativeRuntime` probes whether this build can execute it.

Likewise, names such as `metal`, `rocm`, `tpu`, and `trainium` parse as devices so profiles and
future integrations can describe them. The maintained profile lookup currently returns no runtime
for those devices. Selecting one produces a clear “no maintained runtime profile” error rather than
running on CPU.

The maintained profile lookup exposes that distinction directly:

```lean (name := gpuDeviceTable)
-- Separate a recognized device name from an available
-- maintained profile.
#eval do
  for d in [Device.cpu, .cuda, .metal, .rocm, .tpu] do
    let label :=
      match BackendProfile.maintainedForDevice? d with
      | some p => p.name
      | none => "no maintained runtime profile"
    IO.println s!"{d.cliName}: {label}"
```

```leanOutput gpuDeviceTable (whitespace := lax)
cpu: checked_cpu
cuda: checked_cuda
metal: no maintained runtime profile
rocm: no maintained runtime profile
tpu: no maintained runtime profile
```

The return type of `maintainedForDevice?` is an `Option`: parsing a device name can succeed even
when profile lookup returns `none`. Callers must handle that case, and maintained runtime
entrypoints report the unsupported selection.

# Native CUDA Versus cuBLAS

“CUDA” names the NVIDIA programming platform, not one kernel implementation. TorchLean's CUDA path
can use:

- custom `.cu` kernels for elementwise, reduction, indexing, convolution, attention, and other
  operations;
- cuBLAS for tuned dense linear algebra;
- cuFFT for Fourier transforms;
- CUDA runtime calls for allocation, copies, streams, and launch management.

cuBLAS is a vendor library inside the CUDA ecosystem. Calling it can be much faster than a simple
handwritten matrix kernel because it chooses algorithms specialized for dimensions, datatype, and
GPU generation. It is also a larger external trust boundary. The capsule should identify the
provider and numerical policy precisely enough that “native CUDA” does not hide whether the
arithmetic came from custom code or a vendor library.

# Maintained CUDA Operations

The eager CUDA tape currently covers elementwise arithmetic and activations, reductions and
broadcasting, shape transforms, gather/scatter, dense and batched matrix multiplication,
normalization and softmax, rank-polymorphic convolution and transposed convolution,
max/average/smooth-max pooling, composed attention, a direct attention reference kernel, and fused
spectral convolution. Backward rules are recorded as tape nodes rather than delegated to an
invisible global autograd engine. Differentiable real FFT, inverse real FFT, and selective scan
also have native routes, with generic differentiable reference implementations for interpreters
that do not provide the native hooks.

Layer normalization and tanh-approximate GELU are examples of the distinction between semantics
and scheduling. Each is one operation on the TorchLean tape, with a local forward rule and VJP.
The CUDA implementation evaluates that rule with one forward kernel and one backward kernel rather
than constructing the formula from a chain of temporary tensors. The fusion changes launch and
allocation behavior; it does not replace the TorchLean operation or hand backward ownership to an
external autograd engine.

Matrix derivatives use the same principle without introducing a fused model operation. Products
such as $`A^\mathsf{T}B` and $`AB^\mathsf{T}` are passed to cuBLAS with logical transpose flags.
The old path first copied an operand into a transposed buffer and then multiplied it. The new path
reads the original row-major storage directly. This applies to ordinary and batched matrix
multiplication, linear layers, projection weights, and any model built from them; attention is only
one caller. TorchLean still records the original matrix operation and its local VJP. Only the
schedule used to evaluate that VJP has changed.

This list does not mean every TorchLean operation has a CUDA implementation. Provider-aware
wrappers reject unsupported capsules and shapes; they do not copy a tensor to CPU and continue
silently. The native source map on `NN.Runtime.Autograd.Engine.Cuda.Trusted` identifies the
Lean declarations and corresponding files under `csrc/cuda`.

The registry gives the capsule count and identifies operations with no registered VJP:

```lean (name := gpuCudaRegistry)
-- Count registry entries and name the ones that have no
-- differentiable tensor inputs.
#eval do
  let cs := NativeCUDA.capsules
  IO.println s!"registered: {cs.size}"
  let fwd := cs.filter fun (c : KernelCapsule) =>
    c.vjpMode matches VJPMode.none
  IO.println s!"forward only: {fwd.size}"
  for c in fwd do IO.println s!"  {c.name}"
```

```leanOutput gpuCudaRegistry (whitespace := lax)
registered: 46
forward only: 2
  native_cuda.rand_uniform
  native_cuda.bernoulli_mask
```

The count excludes attention, which has a dedicated semantic split and appears through its own
capsules rather than as one entry in `NativeCUDA.capsules`.

The two forward-only entries create random uniform buffers and Bernoulli masks from a seed.
They have no differentiable tensor inputs. Selective scan has a backward recurrence for its
coefficients, inputs, and initial state. When coefficients are shared across tokens, their
gradients accumulate across time. The eager CUDA path records that recurrence on the tape;
generic interpreters compose the differentiable scalar and tensor operations.

For a diagonal scan, a time step has the form `h[t] = A[t] * h[t-1] + B[t] * x[t]`, with products
taken coordinatewise. Backward must account for an input's immediate contribution and its effect
on later states. If `A[t]` or `B[t]` came from an input projection, returning their cotangents lets
the surrounding tape continue through that projection. Treating them as constants would preserve
the forward recurrence while dropping part of the model's derivative.

The Fourier route has a different bookkeeping issue. A real input of length `n` stores only
`n / 2 + 1` complex bins, represented by a final real/imaginary axis of length two. The inverse
accepts `n` explicitly because lengths four and five both store three bins. Its normalization and
conjugate completion are part of the operation, so its adjoint must include their scaling. The
imaginary DC coordinate, and the imaginary Nyquist coordinate for even lengths, do not affect the
inverse result and have zero derivative. Generic execution uses dense specification matrices;
the native route uses cuFFT. Matching the operation does not fix their floating-point order.

# Batched Attention

A transformer block receives a tensor of shape `(batch, tokens, modelDim)`. The mathematical
operation applies the same attention layer to every batch entry, with shared projection matrices.
TorchLean keeps that description in the specification and typed graph. Each sample is expressed
through the existing attention node, so its forward map, JVP, and VJP are the same definitions used
for unbatched attention.

The eager CUDA runtime schedules the work differently. It flattens `(batch, tokens)` for the four
shared projections and folds `(batch, head)` into the batch axis of the matrix multiplications.
Hard-masked softmax and the local backward rule still belong to TorchLean. The backward pass sums
the four projection-matrix gradients across the full batch.

This removes the old host loop over samples without introducing a second mathematical operation.
The verifier lowers batched attention to the per-sample graph, while CUDA executes one
batch-aware tape node. Regression tests compare the batched forward value, input gradient, and
shared weight gradients with repeated single-sample attention. The comparison is runtime evidence;
the cuBLAS calls and float32 behavior remain covered by the capsule's stated boundary.

Shared weights are the reason the comparison must inspect parameter gradients as well as output
values. Each batch entry contributes to the same projection matrix. A backward implementation
that kept only the last sample's contribution could still produce correct forward values and
input gradients. Summing those contributions is the layer's job; any division by batch size comes
from the surrounding loss reduction and must not be added a second time inside attention.

# Forward Values And Backward Ownership

There are three useful configurations for an operation:

:::table +header
*
  * Forward
  * Backward
  * TorchLean owns
*
  * TorchLean native
  * TorchLean native VJP
  * graph, tape, value/VJP rules, native boundary
*
  * external fast kernel
  * TorchLean VJP
  * graph, tape, backward semantics; external forward boundary
*
  * external autograd
  * external autograd
  * only the surrounding contract and imported gradients
:::

The middle row is the preferred scaling direction when practical. An external provider computes a
fast forward value, but the wrapper still records a TorchLean tape node and applies TorchLean's
selected VJP. This keeps parameter ownership and optimizer flow local.

The local derivative must describe the forward operation that actually ran. Attention scale, mask
convention, and any stochastic choices must agree across the boundary. Reusing a familiar VJP
with a differently configured external forward would differentiate the wrong function even if
all shapes matched. Retained forward values and configuration therefore form part of the
interface between that provider and the TorchLean tape.

The forward capsule still needs guard, test, or trusted-boundary evidence connecting the external
value to the spec. Keeping the VJP local does not strengthen that forward evidence.

The checked CUDA profile does not use this hybrid LibTorch row by default. Its attention path is
the composed TorchLean operation described above; LibTorch forward is available only through the
explicit `libTorchForwardCuda` profile.

# LibTorch Attention

LibTorch is PyTorch's C++ distribution. ATen is the lower tensor/operator library used inside it.
TorchLean's optional adapter calls LibTorch/ATen scaled-dot-product attention; it does not embed a
Python interpreter.

Build the optional provider with:

```terminal
# Build the optional forward provider and run its attention
# comparison.
scripts/lake.sh -R -K cuda=true -K libtorch=true build
scripts/lake.sh -K cuda=true -K libtorch=true exe libtorch_sdpa_test
```

The maintained `libTorchForwardCuda` profile delegates scaled-dot-product-attention *forward* to
LibTorch while keeping a TorchLean tape VJP. A raw LibTorch backward test exists to compare
gradients, but the maintained profile does not hand default backward ownership to LibTorch
autograd.

Programmatic selection is explicit:

```
-- Choose the external forward profile explicitly and retain
-- its audit report.
let run : Trainer.RunConfig :=
  ({} : Trainer.RunConfig)
    |>.withBackendProfile NN.Backend.BackendProfile.libTorchForwardCuda
    |>.withBackendReport true
```

This cannot be obtained by merely spelling `--device cuda`; ordinary CUDA selects
`BackendProfile.checkedCuda`.

# Hard Attention Masks

TorchLean's boolean attention meaning is:

```
true  = this key participates
false = this key has exactly zero softmax numerator
```

A fully blocked row returns zero. Native fused attention and the LibTorch adapter must preserve this
convention. The adapter constructs a boolean CUDA mask rather than replacing `false` by $`-1000`.

Replacing blocked logits by a finite sentinel such as $`-1000` can fail when surviving logits
are smaller than the replacement. If an allowed logit is $`-5000` and a blocked one is set to
$`-1000`, the blocked entry becomes the largest score in the row and takes almost all the weight:

```lean (name := gpuSentinel)
-- Compare normalization over a finite sentinel with removal
-- of a blocked key.
#eval do
  let scores : Tensor Float [2] := [-5000.0, -1000.0]
  let mask : Tensor Bool [2] := [true, false]
  let sentinel := Activation.softmaxSpec 0 scores
  let hard := Spec.hardMaskedSoftmaxSpec scores mask
  IO.println s!"finite sentinel : {sentinel}"
  IO.println s!"hard mask       : {hard}"
```

```leanOutput gpuSentinel (whitespace := lax)
finite sentinel : [0.000000, 1.000000]
hard mask       : [1.000000, 0.000000]
```

Position $`0` is the only key allowed to participate. A hard mask removes the other entry from
normalization, leaving weight one on the allowed key. The sentinel version includes both scores
in softmax, where their difference makes the blocked key dominate; the allowed weight prints as
zero. Large score differences can expose this failure; training duration alone does not imply
them. Negative infinity expresses a hard support restriction, with a separate convention needed
for fully blocked rows. A finite additive bias can approximate that restriction when the remaining
score differences make the blocked weights negligible.
PyTorch's attention interface supports both boolean masks and floating-point additive masks
({Informal.citet pytorch2019}[]).

The specification also defines the result when no key participates:

```lean (name := gpuBlockedRows)
-- Exercise an ordinary masked row and the explicitly
-- defined all-blocked case.
#eval do
  let row : Tensor Float [3] := [1.0, 2.0, 3.0]
  let scores : Tensor Float [2, 3] := [row, row]
  let mask : Tensor Bool [2, 3] :=
    [[true, true, false], [false, false, false]]
  IO.println s!"{Spec.hardMaskedSoftmaxSpec scores mask}"
```

```leanOutput gpuBlockedRows (whitespace := lax)
[[0.268941, 0.731059, 0.000000], [0.000000, 0.000000, 0.000000]]
```

The first row renormalizes over the two surviving keys, and the blocked entry is exactly $`0`.
The second row has no surviving keys and returns zeros rather than dividing by zero. Returning
zero is the chosen convention; rejecting fully blocked rows is another possible API.
`Spec.hardMaskedSoftmaxSpec` records the zero-row convention in its docstring. The native CUDA
providers and the LibTorch adapter are tested against that same definition.

In the first row, removing the third key leaves scores one and two. Their exponentials have ratio
`1 : exp(1)`, giving the printed weights approximately `0.269` and `0.731`. In the second row there
is no such ratio to normalize. Returning zero specifies the whole row's value and gives backward
a defined case to handle. The boolean mask itself is discrete input data; the attention VJP
tracks the numerical inputs while holding that choice of participating keys fixed.

The attention regression suite includes masked forward values, $`\mathrm dQ`, $`\mathrm dK`,
$`\mathrm dV`, and fully blocked
rows. Those are tests of concrete cases. The pure FlashAttention theorem separately identifies its
denotation with ordinary attention.
The current named tiled definition delegates to the full attention formula and does not prove a
separately implemented tiled online-softmax algorithm correct.

# Reduction Order In The Numerical Contract

Every capsule carries a `NumericalPolicy`, and the reduction field is the part the certificates
actually consume. Compare the CPU reference capsule with the CUDA one for the same operation:

```lean (name := gpuReduction)
-- Read the reduction-order field for the same operation
-- under two providers.
#eval do
  let line (c : KernelCapsule) : String :=
    let p := c.numericalPolicy.reduction.label
    s!"{c.name}: reduction={p}"
  IO.println (line Reference.matmul)
  IO.println (line NativeCUDA.matmul)
```

```leanOutput gpuReduction (whitespace := lax)
reference.matmul: reduction=fixed-left
native_cuda.matmul: reduction=implementation-defined
```

`fixed-left` says the reference sums in one pinned order, so its result is reproducible and a proof
may quantify over that fold. `implementation-defined` leaves the CUDA reduction tree to the
implementation, including its block and warp geometry. A certificate for a fixed-left fold cannot
assume that this kernel performs the same sequence of rounded additions.

Reduction order is one of several choices that affect native floating-point values:

- Hardware operations usually round to nearest-even, subject to instruction and library choices.
- Subnormal inputs may be preserved or flushed depending on hardware mode and kernel.
- Multiplication and addition may be contracted into FMA.
- Parallel reductions can choose a tree that differs from the pure interpreter's order.

The accumulator's precision matters too. The native LayerNorm kernel stores inputs and outputs
in binary32, but computes the row mean and centered variance in binary64. It also keeps the mean
in binary64 while subtracting it from each input. Rounding the mean back to binary32 first can
lose the differences between nearby values in a row with a large common offset. The saved
normalized coordinates and inverse standard deviation are binary32 buffers used by backward.
These arithmetic choices mean that a native result need not match a scalar Float32 fold bit for
bit; its accuracy and its relationship to the real-valued specification are separate questions.

The reason a reduction tree matters at all is that floating-point addition is not associative:

$$`(a+b)+c \ne a+(b+c).`

With three decimal literals, changing the parentheses changes the stored result even though the
default printer shows the same decimal text:

```lean (name := gpuAssoc)
-- Compare stored values and bits after the decimal printer
-- has hidden the difference.
#eval do
  let left := (0.1 + 0.2) + 0.3
  let right := 0.1 + (0.2 + 0.3)
  IO.println s!"left  = {left}"
  IO.println s!"right = {right}"
  IO.println s!"equal = {left == right}"
  IO.println s!"apart = {left.toBits - right.toBits} ulp"
```

```leanOutput gpuAssoc (whitespace := lax)
left  = 0.600000
right = 0.600000
equal = false
apart = 1 ulp
```

The equality check compares the stored values and returns `false`; the bit-pattern difference
shows they are one ULP apart. Six printed digits hide that difference
({Informal.citep goldberg1991}[]). Formal floating-point models such as Flocq retain the rounding
steps that real arithmetic omits ({Informal.citep flocq2011}[], {Informal.citep boldo2015}[]).
TorchLean's `IEEE32Exec` serves that role here; {ref "fp32-soundness"}[Float32 Soundness] connects
the executable model to its proofs.

This example uses Lean `Float`, so it demonstrates binary64 rounding rather than directly running
the CUDA binary32 kernel. The same need to specify evaluation order applies to both formats. Here
the positive finite results have adjacent bit patterns, making the subtraction a useful ULP count.
It is not a general distance formula for arbitrary signed values, infinities, or NaNs.

A mathematically equivalent reduction tree can therefore produce different final bits. Numerical
certificates must use an error model or an exact policy strong enough for the claim; shape safety
alone is insufficient.

# Deterministic Reduction Mode

For reproducibility experiments, the CUDA buffer runtime can replace supported atomic reductions
with fixed-order implementations:

```
-- Set the process runtime flag before the reductions whose
-- repeatability is measured.
def enableDeterministic : IO Unit :=
  Runtime.Autograd.Cuda.Buffer.setDeterministicReductions true
```

The environment variable `TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS=1` selects the same mode before
startup. Coverage includes scalar/axis/broadcast reductions, gather/scatter accumulation, and the
pooling backward kernels. It does not make seeded RNG unnecessary, force cuBLAS to a universal
bitwise contract, or prove equality across GPU models. `getDeterministicReductions` reports the
active setting so a training log can record it.

## Scatter Reduction Reproducibility

The following experiment holds the input values and destination indices fixed. It scatter-adds
65,536 uniform float32 values into a single output slot, so every `atomicAdd` updates the same
accumulator. Repeating that identical call a hundred times in each mode isolates the effect of
reduction scheduling:

```
-- Keep values and indices fixed across calls; collect the
-- result bits, not rounded text.
open Runtime.Autograd.Cuda

def survey (label : String) (det : Bool) (runs : Nat) : IO Unit := do
  Buffer.setDeterministicReductions det
  let k : UInt32 := 65536
  let x ← Buffer.zerosIO 1
  let values ← Buffer.randUniformIO k 20260909
  let idx : Array Nat := Array.replicate k.toNat 0
  let mut seen : Std.HashSet UInt64 := {}
  let mut lo := 0.0
  let mut hi := 0.0
  for i in [0:runs] do
    let out ← IO.lazyPure fun _ =>
      Buffer.toFloatArray (Buffer.scatterAdd x values 1 idx k)
    let v := out.get! 0
    if i == 0 then lo := v; hi := v
    lo := min lo v
    hi := max hi v
    seen := seen.insert v.toBits
  let det ← Buffer.getDeterministicReductions
  IO.println s!"{label} deterministic={det} distinct={seen.size}/{runs} spread={hi - lo}"

def main : IO Unit := do
  survey "atomic       :" false 100
  survey "fixed order  :" true 100
```

Built with `-K cuda=true` and run on an A100-SXM4-80GB:

```
atomic       : deterministic=false distinct=52/100 spread=0.191406
fixed order  : deterministic=true distinct=1/100 spread=0.000000
```

The atomic mode produced 52 distinct bit patterns. Every call adds the same values, but scheduling
changes their order at the accumulator. Because each addition rounds, different orders leave
different accumulated errors; the reported spread is the largest observed result minus the
smallest. No result is necessarily the correctly rounded exact sum. With fixed order enabled,
all hundred calls produced one bit pattern. This repeatability helps reproduce a run, while
comparisons between different reduction policies still need an appropriate numerical tolerance.

The `52/100` count is the number of distinct results observed, not a failure rate or a measure of
how often each result occurred. The program stores a set of bit patterns and discards their
frequencies. Its spread also says nothing about which end is closer to the exact sum. To measure
accuracy separately, the experiment would need a reference for the sum of these particular
sampled values, with its own stated precision.

A comparable PyTorch experiment uses normally distributed values and a larger input. It also
compares repeated GPU results with CPU float32 and float64 sums:

```terminal +output
$ python3 - <<'PY'
# Reuse the sampled source for every call, then change only
# reduction determinism.
import torch
torch.manual_seed(0)
src = torch.randn(1 << 20, device="cuda")
idx = torch.zeros(1 << 20, dtype=torch.long, device="cuda")

def once():
    y = torch.zeros(1, device="cuda")
    y.index_add_(0, idx, src)          # one atomicAdd per element
    return y.item()

runs = {once() for _ in range(50)}
print("distinct results, 50 identical runs:", len(runs))
print("cpu, fixed order                   :", src.cpu().sum().item())
print("float64 reference                  :", src.cpu().double().sum().item())
torch.use_deterministic_algorithms(True)
print("deterministic algorithms, 50 runs  :", len({once() for _ in range(50)}))
PY
distinct results, 50 identical runs: 46
cpu, fixed order                   : -1047.3753662109375
float64 reference                  : -1047.3751972975947
deterministic algorithms, 50 runs  : 1
```

The CPU float32 sum differs from the float64 reference even though its order is fixed.
Repeatability specifies whether successive runs agree. Accuracy specifies how far a result is
from the mathematical sum; these measurements answer different questions.

The two transcripts use different distributions, element counts, and repetition counts. Their
numbers illustrate the same scheduling issue, but they do not compare TorchLean and PyTorch speed
or error on a matched workload. The float64 sum in the second transcript is a higher-precision
comparison for those same stored inputs; it is still a floating-point sum. Neither transcript
measures the time or memory cost of enabling deterministic reductions.

These are transcripts rather than live `#eval`s for a structural reason. This book elaborates on a
CPU build, where the CUDA externs resolve to the parity stubs in `torchlean_cuda_tensor_stub.c`, and
the interpreter cannot call the native symbols at all. Evidence about device nondeterminism has to
come from a device, which is why the corresponding contract lives in
`NN/Tests/Runtime/Cuda/DeterministicReductions.lean` under `-K cuda=true`: it pins the mode on and
then asserts *exact* equality between two runs of `scatterAdd` and of the average-pooling backward
kernel, with `==` rather than a tolerance, because bit stability is the entire claim.

## Reading Mutable Native State

The experiment also exposed a problem in how Lean observed the native setting.
`getDeterministicReductions` used to be a pure `Bool`:

```
-- Historical pure wrapper: later reads can observe the
-- module-initialization snapshot.
def getDeterministicReductions : Bool :=
  getDeterministicReductionsRaw 0 != 0
```

Although the `extern` beneath it is marked `@[never_extract]`, this wrapper answered `false` after
fixed-order reduction had been enabled. A nullary definition denotes a value: the compiler
evaluates it once when the module initializes and gives later readers that startup snapshot.
The getter therefore reported the state of
`TORCHLEAN_CUDA_DETERMINISTIC_REDUCTIONS` before any Lean code ran. Adding `@[never_extract]` to the
wrapper does not help, because the attribute governs extraction of applications rather than the
initialization of a constant. The fix is to make the read an effect:

```
-- The current getter performs a fresh effectful observation
-- of native mutable state.
def getDeterministicReductions : IO Bool :=
  IO.lazyPure fun _ => getDeterministicReductionsRaw 0 != 0
```

This is the convention the rest of the file already followed for the allocator counters, where
`allocatorStatsWithToken` takes an ignored token for the same purpose, and the reason
`setDeterministicReductions` is an `IO` action wrapping a `*_checked` native call. Reads and writes
of mutable native state need an effectful observation boundary. An argument alone
does not generally establish ordering or freshness; the low-level token and compiler attributes
must also prevent unwanted sharing. A pure nullary reader can become a startup snapshot.

This affects interpretation of the log itself. A correct fixed-order kernel paired with a stale
getter could print `deterministic=false`, sending a numerical investigation toward the wrong
execution mode. The observed flag must be read after the setter in the same effectful sequence.
The example thus checks two interfaces: the reduction's numerical behavior and the host program's
ability to observe the configuration that selected it.

# Device Reuse Cache Limits

Dropped CUDA buffers are not always returned immediately to the driver. TorchLean keeps exact-size
blocks in a process-wide reuse cache so a later allocation can avoid another `cudaMalloc`. This is
useful in a steady training loop, but a workload that visits many distinct tensor sizes can retain
more device memory than it will reuse.

Set a byte limit before starting the process:

```terminal
# Install the cache allowance before the native allocator
# first initializes.
TORCHLEAN_CUDA_CACHE_CAP_BYTES=$((512 * 1024 * 1024)) \
  scripts/lake.sh -K cuda=true exe torchlean gpt2 --device cuda --steps 100
```

The limit is read once by the native allocator. A returned block that would exceed it waits for its
recorded CUDA event and is then freed instead of cached. `0`, or an unset variable, keeps the cache
unbounded. The value must be a decimal byte count; malformed and overflowing values produce a
warning and are ignored.

This limit applies only to released blocks in the reuse cache. It does not cap live model
parameters, activations, gradients, or workspaces.

The command requests a 512 MiB cache allowance: `512 * 1024 * 1024` bytes. If live tensors already
occupy most of the GPU, this setting cannot make the next large activation fit. Its benefit is to
limit memory retained after owners release buffers, especially when successive workloads use
different sizes. Exact-size reuse also means a large cached block need not satisfy a smaller
request. Interpreting an allocation failure therefore requires live and cached totals together.

Allocator telemetry distinguishes live tensors from reusable memory:

```
-- Read allocator counters after the workload at the point
-- whose ownership matters.
open Runtime.Autograd.Cuda

def printCudaMemory : IO Unit := do
  let stats ← Buffer.allocatorStats
  IO.println stats.format
```

`liveBytes` counts payloads owned by live TorchLean buffers. `cacheBytes` counts released device
blocks waiting for reuse and is therefore not included in `liveBytes`. `cacheCapBytes` reports the
limit installed by the native parser, with `0` meaning unbounded. The formatted report includes all
three values together with wrapper counts and `cudaMemGetInfo` totals.

These counters answer ownership questions at different levels. Releasing the last TorchLean
wrapper can reduce `liveBytes` while increasing `cacheBytes`, leaving driver-visible usage nearly
unchanged. A workspace may retain a buffer intentionally even after a local computation ends.
Reading only free device memory would miss both distinctions; reading only the wrapper count
would miss how large each retained allocation is.

The CUDA stress test starts fresh subprocesses because the limit cannot change after the allocator
has initialized. It checks a finite cap, an explicit unbounded control, malformed input, and integer
overflow against the real allocator. The CPU parity stub reports zero cached bytes and zero cap
because it frees host buffers directly.

# CUDA Test Coverage

The maintained CUDA checks are:

```terminal
# Run device checks against the native build rather than
# host parity storage.
scripts/lake.sh build -R -K cuda=true
scripts/checks/check.sh --cuda
```

They exercise allocation, uploads and downloads, shapes, operation values, gradients, error paths,
and selected numerical behavior. Native sanitizer runs add memory diagnostics. The LibTorch SDPA
test compares native and external forward/backward results within declared tolerances.

Here is the shape of a real run on the A100 build used earlier in this chapter, elided in the middle
because the coverage list is long:

```terminal +output
=== Runtime CUDA kernel coverage suite ===
=== CUDA kernel coverage: softmax ===
=== CUDA kernel coverage: elementwise ===
...
== CUDA deterministic reductions ==
== deterministic scatter_add: exact repeatability ==
== deterministic avg_pool backward: exact repeatability ==
== CUDA deterministic reductions: OK ==
...
== large buffer elementwise/reduction stress ==
== cuBLAS matmul parity stress ==
=== CUDA trainer coverage ===
=== CUDA kernel coverage suite completed ===
== TorchLean: all curated tests passed ==
```

Passing these checks establishes that those cases worked on the tested build, driver, and GPU. It
does not universally prove:

- the C/CUDA source refines every spec operation;
- every GPU architecture behaves identically;
- every reduction is deterministic;
- the compiler and driver preserve source semantics.

Those components remain named in `docs/TRUST_BOUNDARIES.md` and in capsule evidence.
