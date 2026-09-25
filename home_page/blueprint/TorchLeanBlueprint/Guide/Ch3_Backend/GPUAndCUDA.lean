import VersoManual
-- Everything the live blocks in this chapter evaluate: the tensor type with its literal notation,
-- the backend report and the profile table, the LibTorch capsule registry, the reference CPU
-- capsules the CUDA ones get compared against, and the two softmax specifications the masking
-- section contrasts.
import NN.Tensor
import NN.Backend.Report
import NN.Backend.LibTorch
import NN.Backend.Reference
import NN.Runtime.Autograd.Engine.Cuda.Buffer
import NN.Runtime.Autograd.Engine.Cuda.LibTorch
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

the specification describes matrix-vector multiplication and addition. Evaluating it on CUDA
requires contiguous buffers, dimensions, an execution stream, a matrix kernel, and a broadcast
for the bias. The wrapper must put the right values in those buffers, select a kernel, and retain
the operands its backward rule will need.

TorchLean records the shape, layout, forward, and backward obligations in a kernel capsule.
Its CUDA execution goes through ATen, the tensor library distributed with LibTorch. TorchLean
still owns the differentiation tape: it records each operation, retains its operands, selects its
local vector-Jacobian product (VJP), and accumulates the resulting gradients. Calling an ATen
backward operator does not create a second autograd graph.

CUDA is the maintained accelerator. The capsule and device types also allow future Metal, ROCm,
TPU, or custom-chip providers to state their contracts; the runtime lookup below shows which
device names currently have implementations.

# Building With LibTorch

An ordinary CPU build compiles portable stubs for the CUDA symbols and needs no LibTorch SDK.
Those stubs let CPU users import the same Lean modules, but they reject CUDA session creation.
A CUDA build needs a CUDA-enabled LibTorch SDK:

```terminal
scripts/lake.sh -R -K cuda=true \
  -K libtorch_home=/path/to/libtorch build
```

Run this command from the repository root. Replace the path with the SDK directory containing
`include`, `lib`, and `share/cmake/Torch/TorchConfig.cmake`. The builder also accepts
`TORCHLEAN_LIBTORCH_HOME`; without either setting it looks for `libtorch/` in the repository.

The build compiles TorchLean's C++ adapter and uses the selected SDK's CMake package for compiler
flags, C++ ABI, and library dependencies. ATen supplies the GPU kernels. Keep the same SDK
selection on later `build`, `exe`, and `env` commands. The build helper records SDK and compiler
inputs so that a changed native configuration cannot silently reuse an incompatible adapter.

Compilation and session creation answer different questions. A linked adapter still needs a
visible, supported GPU to open a CUDA session. A CPU build exercises the portable stubs, so its
success provides no execution evidence about the CUDA adapter.

To run two optimizer steps and print the selected kernel contracts:

```terminal
scripts/lake.sh -K cuda=true \
  -K libtorch_home=/path/to/libtorch exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

The report identifies capsules such as `libtorch.matmul`, `libtorch.add`, and `libtorch.relu`,
together with the evidence for their contracts. The model description contains two *linear
layers*. Each becomes a sequence of reshapes, permutations, matrix multiplications, broadcasts,
and additions, so the report names operations
below the level of a whole layer. Each capsule describes an operation that crossed a backend
boundary. The only numerical field is the reduction policy; a native accumulation is marked
implementation-defined so that a fixed-left range certificate cannot be applied to it.

Two linear layers can reuse the same matmul capsule, and one operation can launch more than one
ATen kernel. The report's row count therefore measures neither layers nor launches. Measuring speed
requires timing the workload with its SDK, shapes, device, and runtime settings recorded.

# Linear Layer Execution

For an unbatched input `x : [2]` and weight `W : [8,2]`, the eager CUDA path proceeds roughly as:

```
typed Tensor α [2]
    ↓ upload / existing CUDA handle
opaque contiguous float32 buffer
    ↓ reshape and matrix-layout preparation
libtorch.matmul
    ↓ broadcast bias [8] to output shape
libtorch.add
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

Those checks prevent many ABI and memory errors. They do not prove that ATen and its compiled
kernels compute the correct arithmetic expression.

The backward shapes make the same division of responsibility concrete. If the output cotangent is
`δ : [8]`, the input cotangent has shape `[2]` and is mathematically `Wᵀδ`. The weight cotangent has
shape `[8,2]`, with entry `δ[i] * x[j]`, and the bias cotangent is `δ` itself. These formulas
explain which operands backward must retain. A buffer with the right length is necessary for all
three computations, but length alone cannot show that the right operands or transpose flags
were used.

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
registered LibTorch matmul capsule shows how each obligation is recorded:

```lean (name := gpuCapsuleRecord)
-- Inspect the evidence attached to each obligation.
#eval do
  let c := LibTorch.matmul
  IO.println c.name
  IO.println s!"shape: {c.shapeContract.evidence.label}"
  IO.println s!"layout: {c.layoutContract.evidence.label}"
  IO.println s!"value: {c.valueContract.evidence.label}"
  IO.println s!"vjp: {c.vjpContract.evidence.label}"
```

```leanOutput gpuCapsuleRecord (whitespace := lax)
libtorch.matmul
shape: guarded at runtime by LibTorch bridge size/rank checks at the Lean/native boundary
layout: guarded at runtime by LibTorch bridge dtype, device, contiguity, and element-count checks
value: covered by test suite NN.Tests.Runtime.Cuda.Suite
vjp: covered by test suite NN.Tests.Runtime.Cuda.Suite
```

Every claim in that record is paired with its evidence. Shape and layout name the runtime check
that runs at the boundary. Value and VJP name `NN.Tests.Runtime.Cuda.Suite`. These entries give two
different kinds of assurance: a guard checks
the current call, while a test suite compares results on its selected cases. Printing a suite's
name does not execute it or certify that a particular build passed. There is no theorem entry
claiming that ATen refines the specification. The `implementationDefined` reduction policy further
limits which arithmetic certificates can apply, as we will see below.

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

To isolate the identity check, pair the registered LibTorch matmul capsule with a CPU reference
handler for the same operation:

```lean (name := gpuHandlerIdentity)
-- Keep the operation fixed and change provider/device to
-- expose the binding check.
#eval do
  let cap := LibTorch.matmul
  let cpu : KernelHandler Unit :=
    { name := "reference_cpu.matmul"
      op := .matmul
      provider := .reference
      device := .cpu
      execute := fun _ => pure () }
  let capDev := cap.device.cliName
  let cpuDev := cpu.device.cliName
  IO.println s!"capsule : {cap.name} @ {capDev}"
  IO.println s!"handler : {cpu.name} @ {cpuDev}"
  IO.println s!"bindable: {cpu.matchesCapsule cap}"
```

```leanOutput gpuHandlerIdentity (whitespace := lax)
capsule : libtorch.matmul @ cuda
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
  let good := LibTorch.matmul
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

`KernelCapsule.admissible` calls `contractsAligned`, which explains the third line: moving value
evidence into the shape field prevents selection.

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
and a build without LibTorch.

We can inspect profiles and plans on a machine with no GPU. The two maintained profiles are:

```lean (name := gpuProfiles)
-- Compare forward assurance with ownership of the overall
-- backward traversal.
#eval do
  IO.println BackendProfile.checkedCpu.summary
  IO.println BackendProfile.checkedCuda.summary
```

```leanOutput gpuProfiles (whitespace := lax)
profile=checked_cpu device=cpu assurance=checked vjp=torchlean-tape
profile=checked_cuda device=cuda assurance=checked vjp=torchlean-tape
```

Both profiles retain TorchLean's tape. The CUDA profile selects LibTorch primitives whose
contracts name runtime guards and regression suites. Calling a library does not by itself assign
its capsules the `external` assurance policy; that policy concerns the evidence recorded for
each obligation.

This also explains why `vjp=torchlean-tape` in the profile summary and `vjp=backend-vjp` in the
matmul row can appear together. The profile describes who assembles and traverses the whole
backward computation. The capsule describes how one node computes its contribution. TorchLean can
walk the tape while a native kernel evaluates that node's matrix derivative.

The CUDA matmul plan can also be printed on this CPU build:

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
trusted-external capsules: none
  matmul: libtorch.matmul provider=libtorch trust=checked vjp=backend-vjp
    reduction=implementation-defined
    shape: shape safety for matmul; guarded at runtime by LibTorch bridge size/rank checks at the
    Lean/native boundary
    layout: libtorch-cuda-view layout compatibility for matmul; guarded at runtime by LibTorch
    bridge dtype, device, contiguity, and element-count checks
    value: matmul forward refines its TorchLean semantics; covered by test suite
    NN.Tests.Runtime.Cuda.Suite
    vjp: matmul backend-vjp VJP refines its TorchLean semantics; covered by test suite
    NN.Tests.Runtime.Cuda.Suite
```

The report names two runtime guards and two test suites. Its `checked` label must be read together
with those fields; it does not mean that matmul's native implementation has been proved correct.
The line `trusted-external capsules: none` means that no selected capsule has the `trustedExternal`
trust level. That list does not enumerate foreign code: the `checked` LibTorch capsule still calls
ATen. ATen, its dependencies, and compiled execution remain outside the Lean proof. The value
contract has no theorem variant to select.

Availability and dispatch are checked at different points:

1. a plan rejects providers marked unavailable in its supplied availability metadata;
2. provider-aware wrappers reject a selected provider they have not wired up.

CUDA primitive wrappers require the LibTorch provider; the attention wrapper also supports the
explicit TorchLean composition, whose constituent tensor operations use ATen. If a profile selects
an unwired provider, execution fails with an error. There is no hidden CPU fallback for an
unsupported CUDA operation, because moving a tensor between devices behind the user's back would
change both performance and the execution claim.

# Runtime Availability Errors

Build without native CUDA and request it:

```terminal
# Show that a build without LibTorch rejects a requested
# CUDA session.
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cuda --steps 1
```

Without LibTorch, session initialization rejects the request. The CLI's `Device.cuda` value
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

# ATen And The CUDA Libraries

The CUDA provider is LibTorch throughout. The adapter calls ATen operations, and ATen dispatches
their implementations using the tensor shapes, dtype, device, and runtime settings. Vendor
libraries such as cuBLAS, cuDNN, and cuFFT sit below that interface.

This makes the boundary easier to locate: the Lean wrapper states the operation, the C++ adapter
expresses it through ATen, and the selected SDK supplies its kernels. A single ATen operator may
perform several launches, and a TorchLean operation may require a composition of ATen operators.
The capsule identifies the provider and contract; it does not report an exact kernel schedule or
predict its speed.

# Maintained CUDA Operations

The eager CUDA tape currently covers elementwise arithmetic and activations, reductions and
broadcasting, shape transforms, gather/scatter, dense and batched matrix multiplication,
normalization and softmax, rank-polymorphic convolution and transposed convolution,
max/average/smooth-max pooling, attention, and spectral convolution. TorchLean records backward
rules as tape nodes. Differentiable real FFT, inverse real FFT, and selective scan also have ATen
routes, with generic differentiable reference implementations for interpreters that do not
provide these runtime hooks.

Layer normalization and tanh-approximate GELU are examples of the distinction between semantics
and scheduling. Each is one operation on the TorchLean tape, with a local forward rule and VJP.
The adapter can use an upstream operation or compose its formula from tensor operations without
changing who walks the tape. Matrix derivatives follow the same rule: TorchLean selects products
such as $`A^\mathsf{T}B` and $`AB^\mathsf{T}`, and ATen evaluates them. Calling those products does
not ask LibTorch autograd to reconstruct the model's graph.

Convolution and pooling retain spatial ranks one through eight. Ordinary one-, two-, and
three-dimensional cases use upstream convolution and pooling operators and their explicit backward
operators where their domains agree. Higher ranks and exceptional shapes use ATen compositions.
The rank remains part of the public operation even when no single upstream operator accepts it.
Those compositions can launch operations for each kernel offset, so supporting a rank does not
promise the performance of a specialized convolution kernel.

The selected gradient matters just as much as the forward formula. Max pooling selects the first
valid maximum in window order; a first valid NaN remains selected, while later NaNs do not replace
the current candidate. Backward routes the cotangent to that selected input. The adapter corrects
upstream indices where these conventions differ. Average pooling includes literal zero padding in
the full kernel-size denominator. Smooth-max pooling keeps zero padding in its exponential sum
and accepts finite nonzero Float32 values of $`\beta`, including negative values. These details
belong to the operation's contract, even when an upstream operator uses a different convention.

This list does not mean every TorchLean operation has a CUDA implementation. Provider-aware
wrappers reject unsupported capsules and shapes; they do not copy a tensor to CPU and continue
silently. The native source map on `NN.Runtime.Autograd.Engine.Cuda.Trusted` identifies the
Lean declarations and the corresponding implementation boundary. The CUDA adapter sources live
under `csrc/libtorch`.

The registry gives the capsule count and identifies operations with no registered VJP:

```lean (name := gpuCudaRegistry)
-- Count registry entries and name the ones that have no
-- differentiable tensor inputs.
#eval do
  let cs := LibTorch.capsules
  IO.println s!"registered: {cs.size}"
  let fwd := cs.filter fun (c : KernelCapsule) =>
    c.vjpMode matches VJPMode.none
  IO.println s!"forward only: {fwd.size}"
  for c in fwd do IO.println s!"  {c.name}"
```

```leanOutput gpuCudaRegistry (whitespace := lax)
registered: 46
forward only: 2
  libtorch.rand_uniform
  libtorch.bernoulli_mask
```

The count excludes attention, which has a dedicated semantic split and appears through its own
capsules rather than as one entry in `LibTorch.capsules`.

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
the CUDA route uses ATen's Fourier operators. Matching the operation does not fix their
floating-point order.

# Batched Attention

A transformer block receives a tensor of shape `(batch, tokens, modelDim)`. The mathematical
operation applies the same attention layer to every batch entry, with shared projection matrices.
TorchLean keeps that description in the specification and typed graph. Each sample is expressed
through the existing attention node, so its forward map, JVP, and VJP are the same definitions used
for unbatched attention.

The eager CUDA runtime schedules the work differently. It flattens `(batch, tokens)` for the four
shared projections and folds `(batch, head)` into the batch axis of the matrix multiplications.
The selected attention capsule supplies the local forward and backward computation. TorchLean
retains the hard-mask semantics and sums the four projection-matrix gradients across the full
batch.

The verifier lowers batched attention to the per-sample graph, while CUDA executes one
batch-aware tape node. Regression tests compare the batched forward value, input gradient, and
shared weight gradients with repeated single-sample attention. The comparison is runtime evidence;
the ATen calls and float32 behavior remain covered by the capsule's stated boundary.

Shared weights are the reason the comparison must inspect parameter gradients as well as output
values. Each batch entry contributes to the same projection matrix. A backward implementation
that kept only the last sample's contribution could still produce correct forward values and
input gradients. Summing those contributions is the layer's job; any division by batch size comes
from the surrounding loss reduction and must not be added a second time inside attention.

# Forward Values And Backward Ownership

Attention has two maintained CUDA routes:

:::table +header
*
  * Capsule
  * Local forward and backward
  * Tape owner
*
  * `libtorch.direct_attention`
  * paired ATen attention route
  * TorchLean
*
  * `torchlean.composed_attention`
  * matrix products, hard-masked softmax, and their VJPs
  * TorchLean
:::

`BackendProfile.checkedCuda` prefers `Attention.libTorchDirectAttention`. Its adapter retains the
chosen forward implementation and the state needed by that implementation's backward rule. The
TorchLean tape later supplies the output cotangent and receives $`\mathrm dQ`, $`\mathrm dK`, and
$`\mathrm dV`. All of this runs with LibTorch gradient recording disabled.

The local derivative must describe the forward operation that actually ran. Attention scale, mask
convention, and any stochastic choices must agree across the boundary. Reusing a familiar VJP
with a differently configured external forward would differentiate the wrong function even if
all shapes matched. Retained forward values and configuration therefore form part of the
interface between that provider and the TorchLean tape.

The direct route asks the SDK to select an eligible attention implementation with a supported
backward pair. Where the math route is needed, backward uses the saved forward probabilities with
ATen softmax backward and matrix products. Keeping those probabilities also avoids recomputing a
potentially different forward softmax. The bridge does not promise that every shape uses Flash
Attention or that every selected route avoids quadratic attention storage.

The composed capsule remains useful for inspecting and comparing the stages. Select it by
setting the CUDA profile's provider preference to `.prefer .torchLean`, as shown in
{ref "backend-selection"}[Backend Selection]. Its matrix products and softmax still execute
through ATen. Both routes need value and VJP evidence; owning the tape does not prove the native
arithmetic correct.

# Hard Attention Masks

TorchLean's boolean attention meaning is:

```
true  = this key participates
false = this key has exactly zero softmax numerator
```

A fully blocked row returns zero. Both attention routes must preserve this convention. The direct
adapter accepts the Boolean support mask, constructs the representation needed by its selected
ATen operator, and handles fully blocked rows explicitly.

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
`Spec.hardMaskedSoftmaxSpec` records the zero-row convention in its docstring. The attention
regression cases compare the runtime routes against that same definition.

For scores one and two, the exponential ratio is `1 : exp(1)`, giving the first row's weights
approximately `0.269` and `0.731`. The attention VJP holds the Boolean mask fixed while
differentiating the numerical inputs, including the defined all-blocked case.

The attention regression suite includes masked forward values, $`\mathrm dQ`, $`\mathrm dK`,
$`\mathrm dV`, and fully blocked rows. Those are tests of concrete cases. The pure FlashAttention
theorem separately identifies its denotation with ordinary attention. The current named tiled
definition delegates to the full attention formula and does not prove a separately implemented
tiled online-softmax algorithm correct.

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
  IO.println (line LibTorch.matmul)
```

```leanOutput gpuReduction (whitespace := lax)
reference.matmul: reduction=fixed-left
libtorch.matmul: reduction=implementation-defined
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

The input and accumulator precision matter too. The LibTorch controls below let a run permit TF32
for matrix products or cuDNN convolutions, or request IEEE input precision. Those settings constrain
eligible implementations; they do not specify every intermediate rounding or the reduction tree.
A native result's repeatability, accuracy, and relationship to the real-valued specification are
separate questions.

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
({Informal.citep goldberg1991}[]). FloatLib makes the format and rounding operations explicit;
{ref "fp32-soundness"}[Float32 Soundness] explains how those values connect to rounded-real
error analysis.

This example uses Lean `Float`, so it demonstrates binary64 rounding rather than directly running
the CUDA binary32 kernel. The same need to specify evaluation order applies to both formats. Here
the positive finite results have adjacent bit patterns, making the subtraction a useful ULP count.
It is not a general distance formula for arbitrary signed values, infinities, or NaNs.

To cover different reduction trees, a numerical certificate needs an error model that permits
their different rounding steps, or a policy that fixes the steps it assumes.

# Runtime Precision And Determinism

The controls in
{src "NN/Runtime/Autograd/Engine/Cuda/LibTorch.lean"}[`Cuda.LibTorch`] configure the linked
SDK. Set them before opening concurrent work, and record their readbacks with a numerical
experiment. This definition configures a run when called; elaborating it does not touch a GPU:

```lean (name := gpuConfigure)
open Runtime.Autograd.Cuda.LibTorch in
def gpuConfigure : IO Unit := do
  setMatmulPrecision .ieee
  setConvPrecision .ieee
  setDeterministic true
  IO.println (← version)
  IO.println s!"deterministic: {← getDeterministic}"
```

The adapter initializes matrix multiplication and cuDNN convolution with IEEE precision requested.
`setMatmulPrecision .tf32` and `setConvPrecision .tf32` permit TF32 for the corresponding
operations. These are runtime permissions, not changes to the tensor's stored Float32 dtype.
Neither `.ieee` nor `.tf32` establishes agreement with a FloatLib evaluation or fixes every
intermediate rounding step. Arbitrary FloatLib formats remain available to the numerical
specifications and interpreters; these controls do not make CUDA tensors arbitrary-precision.

`setDeterministic true` requests strict deterministic algorithms and disables cuDNN benchmarking.
An unsupported deterministic operation must report an error rather than silently proceed with a
warning. The setting does not impose the reference left fold, guarantee equal bits across SDK
versions or GPU models, or make seeded randomness unnecessary. `setCuDNNBenchmark true` is
rejected while strict determinism is enabled. `setCuDNNEnabled` separately controls whether the
native context may use cuDNN.

The distinction between repeatability and accuracy survives a deterministic implementation.
Repeating a sum in one order can reproduce the same bits every time while differing from a sum in
higher precision. To investigate that distinction, keep the input tensor and index map fixed,
repeat the operation, and collect result bits rather than rounded decimal strings. Compare those
results with a reference over the same stored inputs and state its precision. A count of distinct
outputs measures repeatability; an error against the reference measures something else. Neither
measurement on its own measures speed.

## Attention Selection And Saved State

`setSDPEnabled` accepts `.flash`, `.efficient`, `.math`, or `.cuDNN` and a Boolean permission.
These settings apply to `libtorch.direct_attention`. They do not turn the explicitly composed
TorchLean attention capsule into a fused implementation.

A permitted implementation still has to support the actual shape, dtype, device, mask, and
forward/backward pair. The adapter may need the math route for a request that the enabled fused
pairs cannot handle. Disabling math can therefore make such a request fail. The planner's capsule
name identifies the direct route; it does not certify that Flash Attention ran inside it.

Configure selection before recording the forward call. Its result retains the state needed for
backward, which may include probabilities or other upstream intermediates. Backward uses that
saved choice. Changing the deterministic policy between the paired calls is rejected, since the
backward calculation must honor the policy under which forward was recorded.

## Reading Mutable Native State

A runtime setting must be read at the point where the program needs it. The public controls use
`IO`: execute the setter, then the getter in the same sequence, as `gpuConfigure` does above.
Boolean setters also check the readback, so a rejected request cannot be reported as a successful
configuration.

Allocator telemetry follows the same rule. `Buffer.allocatorStats` observes the native counters
when the action runs. Device selection is effectful too: `deviceCount` reports visible devices,
`getDevice` reads the selected index, and `setDevice` selects a device for subsequent bridge work.
The setter requires all existing buffer wrappers to be finalized, including empty and explicitly
released wrappers; it does not migrate their tensors. Select the device before allocating model
state. Without LibTorch, `version` reports `unavailable` and these calls fail.

# Reading CUDA Memory Usage

The LibTorch allocator owns device storage and its reuse policy. TorchLean's counters answer a
related but different question: which logical payloads and Lean buffer wrappers remain owned?
Read both levels at the point in the workload whose lifetime matters:

```lean (name := gpuMemory)
def gpuPrintMemory : IO Unit := do
  let stats ← Runtime.Autograd.Cuda.Buffer.allocatorStats
  IO.println stats.format
```

`liveBytes` and `peakBytes` count logical payloads owned by TorchLean handles. A view can share
storage with another handle, so these counts can count the same storage twice. Conversely, an ATen
temporary or saved attention context can own storage without a separate TorchLean handle.
`allocCount` and `freeCount` count payload lifetimes, not calls to the CUDA allocator. The wrapper
counters also include empty and explicitly released objects until Lean finalizes those wrappers.
These ownership counters are process-wide.

`allocatedBytes` and `reservedBytes` come from the LibTorch allocator for the selected CUDA device.
Allocations include library workspaces, such as a cuBLAS workspace, that can outlive every
TorchLean handle. Zero ownership counters therefore need not imply zero allocated bytes.
The corresponding peaks are `peakAllocatedBytes` and `peakReservedBytes`. Reservation includes
storage retained for reuse, but subtracting allocated bytes does not give a promise of immediately
reclaimable memory. `deviceFreeBytes` and `deviceTotalBytes` report the driver's view of that
same device. Without LibTorch these device counters are zero. The fields are read separately, so
concurrent activity can make the report differ from an atomic snapshot.

For example, retiring a temporary tensor can reduce allocated bytes while leaving reserved bytes
unchanged. That is compatible with an allocator keeping storage for another operation. A forward
attention result can also retain the context needed by backward; dropping an unrelated local
handle does not retire that context. Reading only free device memory would conflate allocator
reuse with values that the tape still needs.

An application can set the selected device's allocation limit and later release unused cache:

```lean (name := gpuMemoryControls)
def gpuLimitMemory : IO Unit :=
  Runtime.Autograd.Cuda.LibTorch.setMemoryFraction 0.8

def gpuReleaseUnusedMemory : IO Unit :=
  Runtime.Autograd.Cuda.LibTorch.emptyCache
```

The fraction must be finite and lie in `(0, 1]`. It configures the upstream allocator's limit;
it is neither a cache-only allowance nor a reservation against other processes. Read it back with
`getMemoryFraction`, allowing for native byte granularity. It does not release live tensors.

`emptyCache` releases unused allocator blocks. Live parameters, optimizer state, saved forward
state, and library workspaces remain allocated, so both allocated and reserved bytes may remain
after the call. Calling it after every operation can throw away useful reuse.
For timing a completed phase, `LibTorch.synchronize` waits for the
selected device's work; record whether that synchronization is included in a measurement.

# CUDA Test Coverage

Build against the SDK selected for the run, then execute the maintained device checks:

```terminal
export TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
scripts/lake.sh -R -K cuda=true build
scripts/checks/check.sh --cuda
```

The CUDA suite covers allocation, uploads and downloads, shapes, operation values, gradients,
error paths, and selected numerical behavior. Convolution and pooling fixtures include unusual
ranks, padding, empty shapes, max-pool selections, and backward values. Attention fixtures inspect
both forward values and the input cotangents, including hard masks and fully blocked rows.

A capsule names its test source; it does not store a passing result for the current build. To
assess a run, keep its complete test output with the revision, SDK, driver, GPU, and relevant
settings. The chapter's executable planner examples and scalar calculations establish only what
those examples evaluate.

Passing device checks establishes that their cases worked on the recorded configuration. A claim
that ATen, its dependencies, the compiler, and the GPU refine every TorchLean specification would
require a further argument. Those components remain part of the implementation boundary described
in `docs/TRUST_BOUNDARIES.md` and in the capsule records.
