import VersoManual
-- Everything the live blocks in this chapter evaluate: the tensor type with its literal notation,
-- the backend report and the profile table, the LibTorch contracts and CUDA implementation table,
-- the reference CPU contracts the CUDA ones get compared against, and the two softmax
-- specifications the masking section contrasts.
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.LinearAlgebra
import TorchLean.NN.Backend.Report
import TorchLean.NN.Backend.LibTorch
import TorchLean.NN.Backend.Reference
import TorchLean.CUDA.LibTorch.Buffer
import TorchLean.CUDA.LibTorch.Controls
import TorchLean.Tensor.Layers.Activation
import TorchLean.Tensor.Layers.Attention
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- `Tensor` and its list literals live under `TorchLean`, and capsules, profiles, and reports live
-- under `TorchLean.Backend`. Opening both keeps the code blocks inside Verso's column width, and
-- the printed output still names every capsule and profile in full.
open TorchLean
open Backend

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
requires contiguous buffers, dimensions, an execution stream, a matrix kernel, and a broadcast for
the bias, and the wrapper must put the right values in those buffers, select a kernel, and retain
the operands its backward rule will need.

We record the shape, layout, forward, and backward obligations in a kernel capsule. LibTorch is the
standard CUDA backend for model operations. Those operations go through ATen (the tensor library
distributed with LibTorch). For custom scalar functions, {ref "custom-computations"}[Custom Tensor
Computations] generates CUDA through a separate checked frontend and executes it with NVRTC.
TorchLean still owns the differentiation tape: it records each operation, retains its operands,
selects its local vector-Jacobian product (VJP), and accumulates the resulting gradients. Calling
an ATen backward operator doesn't create a second autograd graph.

CUDA is the maintained accelerator, and the runtime lookup below shows the profile each device
selects.

# Building With LibTorch

An ordinary CPU build links one unavailable-backend shim and needs no LibTorch SDK. Such a build
lets CPU users import the same Lean modules, but it can't open native CUDA sessions.

Use `device := .cuda` in Lean and `--device gpu` on the command line. Both select CUDA through
LibTorch (the supported GPU target). For a supported native scalar carrier, an unavailable runtime
is an error, while ordinary model sessions use the CPU tape for carriers without native tensor
support (with a diagnostic and without losing precision). Custom computations also support
configured binary formats, including binary128 values and gradients recorded on the same GPU tape,
and the {ref "custom-computations"}[custom computations chapter] explains which operations we can
record. Types outside the custom GPU interface use CPU with a diagnostic. Neither path hides
compilation or execution errors behind a fallback. Use `cuda` to name the device explicitly. The
default plain build still runs on CPU. A CUDA build needs a CUDA-enabled LibTorch SDK:

```terminal
scripts/lake.sh -Kcuda=true \
  -Klibtorch_home=/path/to/libtorch build
```

Run this command from the repository root, replacing the path with the SDK directory that contains
`include`, `lib`, and `share/cmake/Torch/TorchConfig.cmake`. You can also set
`TORCHLEAN_LIBTORCH_HOME`, and without either setting the builder looks for `libtorch/` in the
repository.

A CUDA build compiles TorchLean's C++ adapter and uses the selected SDK's CMake package for
compiler flags, C++ ABI, and library dependencies. ATen supplies the model-operation kernels, and
custom computations also need the CUDA toolkit's NVRTC and driver libraries. Keep the same SDK
selection on later `build`, `exe`, and `env` commands. Our build helper records SDK and compiler
inputs so that a changed native configuration can't silently reuse an incompatible adapter.

Compilation and session creation answer different questions. A linked adapter still needs a
visible, supported GPU to open a CUDA session, and a CPU build doesn't execute the CUDA adapter, so
its success provides no execution evidence about that backend.

To run two optimizer steps and print the selected kernel contracts:

```terminal
scripts/lake.sh -Kcuda=true \
  -Klibtorch_home=/path/to/libtorch exe torchlean quickstart_mlp \
  --device cuda --steps 2 --seed 2026 --show-backend
```

The report identifies capsules such as `libtorch.matmul`, `libtorch.add`, and `libtorch.relu`,
together with the evidence for their contracts. There are two *linear layers* in the model
description, and each becomes a sequence of reshapes, permutations, matrix multiplications,
broadcasts, and additions, so the report names operations below the level of a whole layer. Each
capsule describes the implementation selected for an operation: the report isn't a trace of native
calls or kernel launches. The reported numerical field is the reduction policy. A native
accumulation is marked implementation-defined so that a fixed-left range certificate can't be
applied to it.

Two linear layers select capsules of the same matmul contract (reported once under its name), and
one operation can launch more than one ATen kernel, so the report's row count measures neither
layers nor launches. To measure speed, time the workload with its SDK, shapes, device, and runtime
settings recorded.

# Linear Layer Execution

For an unbatched input `x : [2]` and weight `W : [8,2]`, the eager CUDA path proceeds roughly as:

```
typed Tensor α [2]
    ↓ upload / existing CUDA handle
opaque contiguous buffer retaining its binary32 or binary64 dtype
    ↓ reshape and matrix-layout preparation
libtorch.matmul
    ↓ broadcast bias [8] to output shape
libtorch.add
    ↓
typed runtime Tensor α [8]
```

In this path, the weight has eight rows of two elements. Multiplication turns the two input
coordinates into eight outputs, and the final addition uses one bias for each output. On the Lean
side, the wrapper knows these logical shapes and element counts, but the device buffer is opaque:
Lean doesn't inspect its contents by reducing a theorem. Before each FFI call, wrappers check the
conditions they can observe:

- the session really targets CUDA
- the selected capsule implements the requested operation
- flat lengths match the logical shapes
- every shape axis, runtime index, and element count fits the `UInt32` CUDA ABI
- ranks, axes, strides, and operation-specific dimensions are supported

Shape-erased values, output seeds, and supplied gradients pass through `SomeBuffer.validate` before
shape-derived indexing. Convolution and pooling also check spatial products independently, so a
zero channel dimension can't hide an overflowing spatial shape, and their forward wrappers validate
the native output length immediately after the FFI returns, before storing the result on the tape.

Those checks prevent many ABI and memory errors. They do not prove that ATen and its compiled
kernels compute the correct arithmetic expression.

On the backward side, the shapes make the same division of responsibility concrete. If the output
cotangent is `δ : [8]`, the input cotangent has shape `[2]` and is mathematically `Wᵀδ`, the weight
cotangent has shape `[8,2]` (with entry `δ[i] * x[j]`), and the bias cotangent is `δ` itself. These
formulas explain which operands backward must retain. A buffer with the right length is necessary
for all three computations, but length alone can't show that the right operands or transpose flags
were used.

For a concrete row, take weights `[1, 2]`, input `[0.25, -0.75]`, and bias `0.1`. That neuron's
preactivation is `-1.15`. With cotangent one, its contribution to the input gradient is `[1, 2]`
and its weight gradient is `[0.25, -0.75]`. If the next operation is ReLU, though, the negative
preactivation makes the incoming cotangent zero. To obtain the model's gradient, the tape must
compose those local rules in the right order.

# Kernel Capsules

A `KernelCapsule` is the audit record for one operator and one provider:

```
-- The record keeps dispatch identity separate from the
-- evidence for each obligation.
structure KernelCapsule where
  name             : String
  op               : Op.Wire
  provider         : Provider
  device           : Device
  trustLevel       : TrustLevel
  supportsForward  : Bool
  vjpMode          : VJPMode
  shapeContract    : Contract.Descriptor
  layoutContract   : Contract.Descriptor
  valueContract    : Contract.Descriptor
  vjpContract      : Contract.Descriptor
  numericalPolicy  : NumericalPolicy
```

In the source, two fields have defaults that this listing leaves out: `supportsForward` defaults to
`true` and `vjpMode` to `.none`. A capsule that leaves them at their defaults therefore claims
forward execution and no backward support, and offering it still requires evidence for the forward
claim.

The `op` field is the wire record of a registered operator, attributes included. The CUDA
capsules are built from contracts. A `KernelContract` holds every capsule field except `op`,
`provider`, and `device`, with each contract field reduced to its explanation and evidence, and a
`LibTorch.Kernel.ofOp` entry carries one in its `contract?` field. `KernelContract.capsule` builds
the capsule of a contract for an operator, and `LibTorch.implementation.capsule? op` returns the
capsule of the contract of the entry for `op`.

Shape, storage layout, forward values, and backward values each get their own descriptor. For a
`[2,3]` by `[3,4]` matrix product, the LibTorch capsule shows how each obligation is recorded:

```lean (name := gpuCapsuleRecord)
-- Inspect the evidence attached to each obligation.
def gpuMatmul :
    Op [.value [2, 3], .value [3, 4]] [.value [2, 4]] :=
  Op.matmul (m := 2) (n := 3) (p := 4) false false
    (fun h => nomatch h) (fun h => nomatch h)
    (Shape.CanBroadcastTo.refl [])
    (Shape.CanBroadcastTo.refl [])

#eval do
  let cuda := LibTorch.implementation
  let some c := cuda.capsule? gpuMatmul
    | IO.println "no CUDA capsule"
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
value: covered by test suite TorchLeanTest.CUDA.Graph
vjp: covered by test suite TorchLeanTest.CUDA.Graph
```

Every claim in that record is paired with its evidence. Shape and layout name the runtime check
that runs at the boundary. Value and VJP name `TorchLeanTest.CUDA.Graph`. Those entries
give two different kinds of assurance: a guard checks the current call, while a test suite compares
results on its selected cases. Printing a suite's name doesn't execute it or certify that a
particular build passed. No theorem entry claims that ATen refines the specification. The
`implementationDefined` reduction policy further limits which arithmetic certificates can apply.

## Shape contract

For matrix multiplication, the inner dimensions must agree and determine the result shape.
Lean-side runtime guards and shape-indexed source objects often support this field. In the linear
layer above, multiplying `[8,2]` by `[2]` must produce `[8]`.

## Layout contract

For contiguous row-major storage, `[m,n]` means that the last axis varies fastest: the first `n`
elements form the first row. Both the wrapper and the native kernel must use the same
interpretation. A correct matrix formula paired with a transposed native layout still computes the
wrong model.

For the `[8,2]` weight, row `i` occupies flat positions `2*i` and `2*i+1`. A column-major reading
would instead group values from different output neurons. Both interpretations use sixteen
scalars, which is why the shape and layout descriptors are separate. Reshaping a view can preserve
storage, whereas changing this interpretation may require different strides or a physical copy.

## Value contract

A value contract names the mathematical function the forward result should refine. Its evidence is a
shared canonical definition, a runtime guard, a test suite, an explicit trusted boundary, or
`notApplicable`, and there's no theorem variant for a Lean proof covering the native code. Reports
show the evidence variant, so a reader can tell a checked input condition from a tested numerical
result.

## VJP contract

A VJP contract records backward ownership and the derivative specification to implement. With
`backend-vjp`, the runtime calls a backend derivative kernel while retaining TorchLean's tape
structure, and a capsule marked `torchLeanTape` composes its local VJP from TorchLean runtime
operations. A profile requesting `torchLeanTape` admits both kinds of capsule: TorchLean owns the
graph and reverse traversal in either case.

`LibTorch.Kernel.ofOp op` is the CUDA implementation of the {ref "registered-operators"}[registered
operator] `op` (attributes included), or `none` where LibTorch has none, and `LibTorch.Tape.apply`
records an operator on the CUDA tape `Tape SomeBuffer` through its entry. An entry holds what the
operator needs on the device: the LibTorch kernel calls, the checks of shapes against the `UInt32`
ABI, the node with its parents and gradient flag, the workspace the node keeps for its backward
(released with the tape), and the release of every temporary buffer. An operator without an entry
is an error on the CUDA tape.

Capsules record contracts. Each CUDA entry of `LibTorch.Kernel.ofOp` but `detach`'s carries a
contract from `Backend.LibTorch`, and the entry's own capsule is that contract's capsule for the
entry's operator. `Session.apply` runs an entry only when `KernelCapsule.implements` holds, that is,
when the entry's capsule and the selected one name the same operator, provider, and device. With a
reference CPU capsule, the operator's rule runs on the CPU tape. `detach` selects no capsule: on
CUDA it records its entry directly, and that entry carries no contract, so the CUDA table offers no
capsule for it:

```
-- The labels compared before an entry runs.
def implements (entry selected : KernelCapsule) : Bool :=
  entry.op == selected.op && entry.provider == selected.provider &&
    entry.device == selected.device
```

The check compares labels and doesn't inspect the entry's code, so an entry carrying a false
capsule would still require source review and execution checks. Numerical correctness depends on
the guard, test, or trusted-boundary evidence.

To isolate the identity check, compare the LibTorch and reference capsules of the same matrix
product:

```lean (name := gpuHandlerIdentity)
-- Keep the operation fixed and change provider/device to
-- expose the identity check.
#eval do
  let cuda := LibTorch.implementation
  let some selected := cuda.capsule? gpuMatmul
    | IO.println "no CUDA capsule"
  let entry :=
    (Reference.contract gpuMatmul).capsule .reference .cpu
      gpuMatmul
  let shown (c : KernelCapsule) : String :=
    s!"{c.name} @ {c.device.cliName}"
  IO.println s!"selected: {shown selected}"
  IO.println s!"entry   : {shown entry}"
  IO.println s!"runs    : {entry.implements selected}"
```

```leanOutput gpuHandlerIdentity (whitespace := lax)
selected: libtorch.matmul @ cuda
entry   : reference.matmul @ cpu
runs    : false
```

Both capsules name the same operation. Their providers and devices don't match, so the reference
implementation doesn't run under the CUDA capsule.

`KernelCapsule.contractsAligned` gives the four contract fields the same treatment. A
value-refinement descriptor names a different obligation from a shape descriptor, so moving it into
the shape field makes the capsule inadmissible, even though the descriptor itself hasn't changed:

```lean (name := gpuMisplacedEvidence)
-- Put valid value evidence in the wrong field and observe
-- alignment and admission.
#eval do
  let cuda := LibTorch.implementation
  let some good := cuda.capsule? gpuMatmul
    | IO.println "no CUDA capsule"
  let moved :=
    { good with shapeContract := good.valueContract }
  let policy := Profile.checkedCUDA.policy
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
implementation tables
  the capsule each table offers per operator
       ↓ profile + availability + contract check
accepted plan
  capsules allowed under this policy
       ↓ implementation with the same identity
implementation entry
  operator/provider/device identities agree
       ↓ provider-aware runtime dispatch
executed operation
  the native symbol actually called
```

At the first transition, `TorchLean.IR.Graph.accept` exposes an `AcceptedGraphKernelPlan` only
after planning, grouping, and the contract check accept every obligation. Execution then checks the
implementation identities shown above. Neither step probes the hardware: a profile's `Availability`
declares which devices and providers planning may consider, while CUDA session creation calls
`LibTorch.Buffer.requireNativeRuntime` to distinguish native CUDA, a native build with no
visible GPU,
and a build without LibTorch.

We can inspect profiles and plans on a machine with no GPU. Here are the default profiles:

```lean (name := gpuProfiles)
-- Compare forward assurance with ownership of the whole
-- backward traversal.
#eval do
  IO.println Profile.checkedCPU.summary
  IO.println Profile.libTorchCUDA.summary
```

```leanOutput gpuProfiles (whitespace := lax)
profile=checked_cpu device=cpu assurance=checked vjp=torchlean-tape
profile=libtorch_cuda device=cuda assurance=external vjp=torchlean-tape
```

Both profiles retain TorchLean's tape. For operations without retained CUDA comparisons, the
default GPU profile admits explicitly recorded LibTorch assumptions, and tested operations name
their comparison source. `Profile.checkedCUDA` remains available when we want to reject
untested operations and keep that external boundary out.

The profile describes who assembles and traverses the whole backward computation, and each capsule
also records how one node computes its contribution. For matmul, Lean composes the two matrix
products in the local derivative, so both entries say `vjp=torchlean-tape`. An operation such as
convolution uses `vjp=backend-vjp` because its local rule calls a native backward primitive.

We can also print the CUDA matmul plan on this CPU build:

```lean (name := gpuPlanReport)
-- Plan one CUDA operation from metadata. Printing the plan
-- does not launch it.
#eval do
  let p := Profile.checkedCUDA
  match p.planReport #[⟨_, _, gpuMatmul⟩] with
  | .ok report => IO.println report
  | .error e => IO.println s!"planning failed: {e}"
```

```leanOutput gpuPlanReport (whitespace := lax)
profile=checked_cuda device=cuda assurance=checked vjp=torchlean-tape
trusted-external capsules: none
  matmul: libtorch.matmul provider=libtorch trust=checked vjp=torchlean-tape
    reduction=implementation-defined
    shape: shape safety for matmul; guarded at runtime by LibTorch bridge size/rank checks at the
    Lean/native boundary
    layout: libtorch-cuda-view layout compatibility for matmul; guarded at runtime by LibTorch
    bridge dtype, device, contiguity, and element-count checks
    value: matmul forward refines its TorchLean semantics; covered by test suite
    TorchLeanTest.CUDA.Graph
    vjp: matmul torchlean-tape VJP refines its TorchLean semantics; covered by test suite
    TorchLeanTest.CUDA.Graph
```

The report names two runtime guards and cites one test suite for both the value and the VJP
contract. Read its `checked` label together with those fields: it doesn't mean that matmul's native
implementation has been proved correct. The line `trusted-external capsules: none` means that no
selected capsule has the `trustedExternal` trust level, but that list doesn't enumerate foreign
code: the `checked` LibTorch capsule still calls ATen. ATen, its dependencies, and compiled
execution remain outside the Lean proof, and the value contract has no theorem variant to select.

Availability and dispatch are checked at different points:

1. a plan rejects providers marked unavailable in its supplied availability metadata
2. provider-aware wrappers reject a selected provider they haven't wired up

CUDA wrappers (attention included) require the LibTorch provider. If a profile selects an unwired
provider, execution fails with an error: there's no hidden CPU fallback for an unsupported CUDA
operation, because moving a tensor between devices behind the user's back would change both
performance and the execution claim.

# Runtime Availability Errors

Build without native CUDA and request it:

```terminal
# Show that a build without LibTorch rejects a requested
# CUDA session.
scripts/lake.sh build
scripts/lake.sh exe torchlean quickstart_mlp --device cuda --steps 1
```

Without LibTorch, session initialization rejects the request: the CLI's `Device.cuda` value
describes the requested target, while `LibTorch.Buffer.requireNativeRuntime` probes whether this
build can execute it.

The profile lookup is pure, and in this CPU-only guide build it still names the CUDA profile:

```lean (name := gpuDeviceTable)
-- Separate a recognized device name from an available
-- maintained profile.
#eval do
  for d in [Device.cpu, .cuda] do
    let label :=
      match Profile.maintainedForDevice? d with
      | some p => p.name
      | none => "no maintained runtime profile"
    IO.println s!"{d.cliName}: {label}"
```

```leanOutput gpuDeviceTable (whitespace := lax)
cpu: checked_cpu
cuda: libtorch_cuda
```

`maintainedForDevice?` returns an `Option`, and `withDevice` turns a `none` into an error rather
than a CPU run.

# ATen And The CUDA Libraries

For the maintained model operations, the CUDA provider is LibTorch. TorchLean's adapter calls ATen,
and ATen dispatches their implementations using the tensor shapes, dtype, device, and runtime
settings (vendor libraries such as cuBLAS, cuDNN, and cuFFT sit below that interface).

That layering makes the boundary easier to locate: the Lean wrapper states the operation, the C++
adapter expresses it through ATen, and the selected SDK supplies its kernels. A single ATen
operator may perform several launches, and a TorchLean operation may require a composition of ATen
operators. A capsule identifies the provider and contract, but it doesn't report an exact kernel
schedule or predict its speed.

# Maintained CUDA Operations

The eager CUDA tape covers elementwise arithmetic and activations, reductions and broadcasting,
shape transforms, gather/scatter, dense and batched matrix multiplication, normalization and
softmax, one-, two-, and three-dimensional convolution and transposed convolution,
max/average/smooth-max pooling, attention, and spectral convolution. We record backward rules as
tape nodes. Differentiable real FFT, inverse real FFT, and selective scan also have ATen routes,
with generic differentiable reference implementations for interpreters that don't provide these
runtime hooks.

Layer normalization and tanh-approximate GELU are examples of the distinction between semantics and
scheduling. Each is one operation on the TorchLean tape (with a local forward rule and VJP), and
the adapter can use an upstream operation or compose its formula from tensor operations without
changing who walks the tape. Matrix derivatives follow the same rule: TorchLean selects products
such as $`A^\mathsf{T}B` and $`AB^\mathsf{T}`, and ATen evaluates them. Calling those products
doesn't ask LibTorch autograd to reconstruct the model's graph.

Convolution, transposed convolution, max pooling, and average pooling follow LibTorch's one-, two-,
and three-dimensional operator families. The adapter handles TorchLean's typed shapes and tape ABI,
then delegates the numerical work and explicit backward operation to ATen, and it doesn't emulate
higher-dimensional kernels or repair ATen's selection rules. Smooth-max pooling is the one
deliberate composition here, because PyTorch has no corresponding primitive. It uses the same
one-to-three spatial-rank boundary and accepts finite nonzero values of $`\beta` in the input dtype
(including negative values).

That list doesn't mean every TorchLean operation has a CUDA implementation. Provider-aware wrappers
reject unsupported capsules and shapes, and they don't copy a tensor to CPU and continue silently.
The native source map on `TorchLean.CUDA.LibTorch.Trusted` identifies the Lean declarations and the
corresponding implementation boundary, and the CUDA adapter sources live under `csrc/libtorch`.

For `Op.max`, `Op.min` and `Op.clamp`, the CUDA entries compute the CPU tape's values. Both tapes
use the IEEE 754-2019 maximum and minimum: ATen's `maximum` and `minimum` propagate a NaN, and at a
pair of zeros the adapter takes the larger or the smaller one by a sum, so `+0` is above `-0`. Thus
`max(NaN, 1)` is NaN on both tapes, `max(+0, -0)` is `+0`, and `clamp` (which is
`minimum (maximum x lo) hi`) gives `+0` for `-0` between `+0` and `1`. Gradients agree as well:
`max` and `min` give each operand half of the cotangent at a tie or an unordered pair, and `clamp`
has the coefficient 1 strictly inside its bounds and 0 elsewhere. `Op.fmax`, `Op.fmin` and
`Op.remainder` have no CUDA entry.

We query the CUDA table one operator at a time, because it's a function of the operator. Three of
these capsules register no VJP, and the table has no capsule for the last operator:

```lean (name := gpuCudaRegistry)
-- Ask the CUDA table for the capsule of each operator and
-- print its VJP mode.
#eval do
  let cuda := LibTorch.implementation
  let ops : List (Σ ins outs, Op ins outs) :=
    [⟨_, _, gpuMatmul⟩, ⟨_, _, Op.relu (s := [4])⟩,
     ⟨_, _, Op.topKMask (s := [4]) ⟨0, by decide⟩ 2⟩,
     ⟨_, _, Op.randUniform (s := [4]) 7⟩,
     ⟨_, _, Op.bernoulliMask (s := [4]) 7⟩,
     ⟨_, _, Op.maskedSoftmax (s := [4]) ⟨0, by decide⟩⟩]
  for e in ops do
    let tag := (Op.encode e).tag
    match cuda.capsule? e.2.2 with
    | some c =>
      IO.println s!"{tag}: {c.name} vjp={c.vjpMode.label}"
    | none => IO.println s!"{tag}: no capsule"
```

```leanOutput gpuCudaRegistry (whitespace := lax)
matmul: libtorch.matmul vjp=torchlean-tape
relu: libtorch.relu vjp=backend-vjp
topKMask: libtorch.top_k_mask vjp=none
randUniform: libtorch.rand_uniform vjp=none
bernoulliMask: libtorch.bernoulli_mask vjp=none
maskedSoftmax: no capsule
```

Masked softmax has no CUDA entry, so the table offers no capsule for it and the maintained CUDA
profiles cannot plan it.

Two of the forward-only entries create random uniform buffers and Bernoulli masks from a seed and
have no differentiable tensor inputs. The third is the top-k mask that routes tokens in a mixture
of experts: it's piecewise constant, so its derivative is zero wherever it exists, and the gradient
reaches the router through the kept probabilities instead.

Selective scan has a backward recurrence for its coefficients, inputs, and initial state, and when
coefficients are shared across tokens, their gradients accumulate across time. On the eager CUDA
path that recurrence is recorded on the tape, while generic interpreters compose the differentiable
scalar and tensor operations.

For a diagonal scan, a time step has the form `h[t] = A[t] * h[t-1] + B[t] * x[t]` (products taken
coordinatewise). Backward must account for an input's immediate contribution and its effect on
later states. If `A[t]` or `B[t]` came from an input projection, returning their cotangents lets
the surrounding tape continue through that projection, whereas treating them as constants would
preserve the forward recurrence while dropping part of the model's derivative.

For the Fourier route, the bookkeeping issue is different. A real input of length `n` stores only
`n / 2 + 1` complex bins (represented by a final real/imaginary axis of length two), so the inverse
takes `n` as an argument: lengths four and five both store three bins. Its normalization and
conjugate completion are part of the operation, so its adjoint must include their scaling. Over the
reals, the imaginary DC coordinate (and the imaginary Nyquist coordinate for even lengths) has zero
contribution and derivative. Generic execution uses dense specification matrices, whose
zero-coefficient products still propagate NaN and infinity, while the CUDA route uses ATen's
Fourier operators. Matching the real-valued operation doesn't fix floating-point order or
nonfinite-value behavior.

Spectral layers compose these transforms with frequency selection and learned channel mixing in
Lean, then call the inverse transform. LibTorch supplies the FFT and tensor primitives, and
TorchLean retains model composition, the local reverse calculation, and the buffers its tape needs.

In the public API, `nn.functional.rfft` and `nn.functional.irfft` transform the last axis and
preserve any leading batch dimensions. A real tensor with shape `[2, 3, 5]`, for example, produces
packed coefficients with shape `[2, 3, 3, 2]`: there are six independent length-five transforms,
each storing three complex bins. Pass `batch := [2, 3]` and the positive-length witness (the
inverse also takes `n := 5`). Both operations accept `path := .denseReference` when you want the
dense calculation. An empty batch is allowed, but the transform length must be positive.

# Batched Attention

A transformer block receives a tensor of shape `(batch, tokens, modelDim)`. Mathematically, the
same attention layer applies to every batch entry, with shared projection matrices. TorchLean keeps
that description in the specification and typed graph. Each sample is expressed through the
existing attention node, so its forward map, JVP, and VJP are the same definitions used for
unbatched attention.

The eager CUDA runtime schedules the work differently. It flattens `(batch, tokens)` for the four
shared projections and folds `(batch, head)` into the batch axis of the matrix multiplications.
LibTorch supplies the local native tensor computations, while TorchLean retains the hard-mask
semantics and sums the four projection-matrix gradients across the full batch.

The verifier lowers batched attention to the per-sample graph, while CUDA executes one batch-aware
tape node, and regression tests compare the batched forward value, input gradient, and shared
weight gradients with repeated single-sample attention. That comparison is runtime evidence. ATen
calls and native floating-point behavior remain covered by the capsule's stated boundary.

Shared weights are the reason the comparison must inspect parameter gradients as well as output
values. Each batch entry contributes to the same projection matrix. A backward implementation that
kept only the last sample's contribution could still produce correct forward values and input
gradients. Summing those contributions is the layer's job. Any division by batch size comes from
the surrounding loss reduction and must not be added a second time inside attention.

# Forward Values And Backward Ownership

GPU multi-head attention has the contract `LibTorch.multiheadAttention`, and its tape node is
composed in {src "TorchLean/NN/LibTorch/Kernel.lean"}[`Kernel.lean`] from
the buffer operations of
{src "TorchLean/NN/LibTorch/Attention.lean"}[`Attention.lean`]. For
Q, K, and V shaped `(batch, n, d)`, `Buffer.attentionForward` takes an optional support mask, the
dimensions, and a scale, then forms the scores, computes the probabilities P, and multiplies P by
V. Its `Except` result contains both the output and P, and the tape retains Q/K/V and P for
backward.

Given output cotangent G, `Buffer.attentionBackward` composes the local rule in Lean:

$$`
\mathrm dP = GV^\mathsf{T},\qquad
\mathrm dS = P \odot \left(\mathrm dP -
  \operatorname{rowsum}(P \odot \mathrm dP)\right),
`

$$`
\mathrm dQ = \mathrm{scale}\,\mathrm dS K,\qquad
\mathrm dK = \mathrm{scale}\,\mathrm dS^\mathsf{T} Q,\qquad
\mathrm dV = P^\mathsf{T}G.
`

In those formulas the row sum is broadcast across each row. Masked positions contribute zero, and
fully blocked rows have zero cotangents. Because the forward probabilities are kept, the rule uses
the same softmax values as forward. LibTorch executes the matrix products and numerical primitives
with gradient recording disabled, while the tape owns and releases the saved buffers.

Composed this way, attention materializes the full score and probability matrices. With B batch
entries, H heads, and n tokens, P alone contains $`BHn^2` values ($`4BHn^2` bytes for `Float32`, or
$`8BHn^2` bytes for `Float`). Scores, backward temporaries, projections, and a mask add further
storage, and doubling sequence length quadruples the probability storage. There's no fused
attention selection or tiled-memory guarantee. Lean composition still needs value and VJP evidence
for its native primitives.

# Hard Attention Masks

TorchLean's boolean attention meaning is:

```
true  = this key participates
false = this key has exactly zero softmax numerator
```

A fully blocked row has zero weights. The CUDA path broadcasts the Boolean support mask over the
folded batch-head axis and applies hard-masked softmax to the score rows. Like the two-pass CPU
reference, it multiplies those zero weights by the values and the cotangents, so the row and its
gradients are zero at finite entries and NaN at an infinite or NaN entry.

Replacing blocked logits by a finite sentinel such as $`-1000` can fail when surviving logits
are smaller than the replacement. If an allowed logit is $`-5000` and a blocked one is set to
$`-1000`, the blocked entry becomes the largest score in the row and takes almost all the weight:

```lean (name := gpuSentinel)
-- Compare normalization over a finite sentinel with removal
-- of a blocked key.
#eval do
  let scores : Tensor Float [2] := [-5000.0, -1000.0]
  let mask : Tensor Bool [2] := [true, false]
  let sentinel := Tensor.softmax scores ⟨0, Nat.one_pos⟩
  let hard := Softmax.hardMasked scores mask
  IO.println s!"finite sentinel : {sentinel}"
  IO.println s!"hard mask       : {hard}"
```

```leanOutput gpuSentinel (whitespace := lax)
finite sentinel : [0.000000, 1.000000]
hard mask       : [1.000000, 0.000000]
```

Position $`0` is the only key allowed to participate. A hard mask removes the other entry from
normalization, leaving weight one on the allowed key, while the sentinel version includes both
scores in softmax, where their difference makes the blocked key dominate (the allowed weight prints
as zero). Large score differences can expose this failure, but training duration alone doesn't
imply them. Negative infinity expresses a hard support restriction (with a separate convention
needed for fully blocked rows), and a finite additive bias can approximate that restriction when
the remaining score differences make the blocked weights negligible. PyTorch's attention interface
supports both boolean masks and floating-point additive masks {Informal.citep pytorch2019}[].

For a row where no key participates, the specification also defines the result:

```lean (name := gpuBlockedRows)
-- Exercise an ordinary masked row and the explicitly
-- defined all-blocked case.
#eval do
  let row : Tensor Float [3] := [1.0, 2.0, 3.0]
  let scores : Tensor Float [2, 3] := [row, row]
  let mask : Tensor Bool [2, 3] :=
    [[true, true, false], [false, false, false]]
  IO.println s!"{Softmax.hardMasked scores mask}"
```

```leanOutput gpuBlockedRows (whitespace := lax)
[[0.268941, 0.731059, 0.000000], [0.000000, 0.000000, 0.000000]]
```

The first row renormalizes over the two surviving keys, and the blocked entry is exactly $`0`. With
no surviving keys, the second row returns zeros without dividing by zero. Returning zero is the
convention we chose, though rejecting fully blocked rows is another possible API.
`Softmax.hardMasked` records the zero-row convention in its docstring, and the attention
regression cases compare the runtime routes against that same definition.

For scores one and two, the exponential ratio is `1 : exp(1)`, giving the first row's weights
approximately `0.269` and `0.731`. Attention's VJP holds the Boolean mask fixed while
differentiating the numerical inputs, including the defined all-blocked case.

Masked forward values, $`\mathrm dQ`, $`\mathrm dK`, $`\mathrm dV`, and fully blocked rows are all
in the attention regression suite. Those are tests of concrete cases. FlashAttention's online
schedule is `Tensor.attention` with `schedule := .online blocks`: online softmax over blocks of
keys of any size, for score matrices with any leading batch axes. Over the reals it computes the
two-pass value for every block size (`Tensor.attention_online_eq`), and
`TorchLean.Verify.RuntimeApprox.OnlineSoftmax` bounds the rounding error of its one-query iterator
in every IEEE format. CUDA attention composes LibTorch primitives and doesn't run this schedule.

# Reduction Order In The Numerical Contract

Every capsule carries the `NumericalPolicy` of its contract, and the reduction field is the part
the certificates actually consume. Compare the CPU reference contract for matrix multiplication
with the CUDA one:

```lean (name := gpuReduction)
-- Read the reduction-order field for the same operation
-- under two providers.
#eval do
  let line (c : KernelContract) : String :=
    let p := c.numericalPolicy.reduction.label
    s!"{c.name}: reduction={p}"
  IO.println (line (Reference.contract gpuMatmul))
  let entry := LibTorch.Kernel.ofOp gpuMatmul
  let some cuda := entry.bind (·.contract?)
    | IO.println "no CUDA contract"
  IO.println (line cuda)
```

```leanOutput gpuReduction (whitespace := lax)
reference.matmul: reduction=fixed-left
libtorch.matmul: reduction=implementation-defined
```

`fixed-left` says the reference sums in one pinned order, so its result is reproducible and a proof
may quantify over that fold. `implementation-defined` leaves the CUDA reduction tree to the
implementation (including its block and warp geometry). A certificate for a fixed-left fold cannot
assume that this kernel performs the same sequence of rounded additions.

Reduction order is one of several choices that affect native floating-point values:

- Hardware operations usually round to nearest-even, subject to instruction and library choices.
- Subnormal inputs may be preserved or flushed depending on hardware mode and kernel.
- Multiplication and addition may be contracted into FMA.
- Parallel reductions can choose a tree that differs from the pure interpreter's order.

Input and accumulator precision matter too. With the LibTorch controls below, a run can permit TF32
for matrix products or cuDNN convolutions, or request IEEE input precision. Those settings
constrain eligible implementations, but they don't specify every intermediate rounding or the
reduction tree. A native result's repeatability, accuracy, and relationship to the real-valued
specification are separate questions.

A reduction tree matters at all because floating-point addition isn't associative:

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

The equality check compares the stored values and returns `false`, and subtracting the bit patterns
shows they're one ULP apart. Six printed digits hide that difference
{Informal.citep goldberg1991}[]. FloatLib spells out the format and rounding operations, and
{ref "fp32-soundness"}[Float32 Soundness] explains how those values connect to rounded-real error
analysis.

This example uses Lean `Float`, so it demonstrates binary64 rounding without running the CUDA
binary32 kernel, but both formats need the evaluation order specified. Here the positive finite
results have adjacent bit patterns, making the subtraction a useful ULP count, though the
subtraction isn't a general distance formula for arbitrary signed values, infinities, or NaNs.

To cover different reduction trees, a numerical certificate needs an error model that permits
their different rounding steps, or a policy that fixes the steps it assumes.

# Runtime Precision And Determinism

The controls in {src "TorchLean/CUDA/LibTorch/Controls.lean"}[`LibTorch`] configure the linked SDK.
Set them before opening concurrent work, and record their readbacks with a numerical experiment.
This definition configures a run when it's called, but elaborating it doesn't touch a GPU:

```lean (name := gpuConfigure)
open TorchLean.LibTorch in
def gpuConfigure : IO Unit := do
  setMatmulPrecision .ieee
  setConvPrecision .ieee
  setDeterministic true
  IO.println (← version)
  IO.println s!"deterministic: {← getDeterministic}"
```

The adapter initializes matrix multiplication and cuDNN convolution with IEEE precision requested,
while `setMatmulPrecision .tf32` and `setConvPrecision .tf32` permit TF32 for the corresponding
binary32 operations. These are runtime permissions, not changes to the tensor's stored dtype:
binary64 buffers remain binary64. Neither `.ieee` nor `.tf32` establishes agreement with a FloatLib
evaluation or fixes every intermediate rounding step. Arbitrary FloatLib formats remain available
to the numerical specifications and interpreters, but these controls don't make CUDA tensors
arbitrary-precision.

`setDeterministic true` requests strict deterministic algorithms and disables cuDNN benchmarking.
An unsupported deterministic operation must then report an error, not silently proceed with a
warning. Strict determinism doesn't impose the reference left fold, guarantee equal bits across SDK
versions or GPU models, or make seeded randomness unnecessary. `setCuDNNBenchmark true` is rejected
while strict determinism is enabled, and `setCuDNNEnabled` separately controls whether the native
context may use cuDNN.

Repeatability and accuracy remain distinct even for a deterministic implementation. Repeating a sum
in one order can reproduce the same bits every time while differing from a sum in higher precision.
To investigate that distinction, keep the input tensor and index map fixed, repeat the operation
and collect result bits (not rounded decimal strings), then compare those results with a reference
over the same stored inputs and state its precision. A count of distinct outputs measures
repeatability, while an error against the reference measures something else. Neither measurement on
its own measures speed.

## Attention Controls And Saved State

`setSDPEnabled` accepts `.flash`, `.efficient`, `.math`, or `.cuDNN` and a Boolean permission.
These SDK permissions don't select an implementation for TorchLean's composed attention: despite
its name, `libtorch.multihead_attention` runs Lean matrix-product and softmax composition, and its
capsule reports `torchLeanTape` for the local VJP.

The tape saves ordinary Q/K/V and probability buffers, and backward receives those buffers and the
output cotangent explicitly, with no native attention context holding a selected pair of forward
and backward kernels. Configure precision and determinism before the workload so that its numerical
primitive calls run under the intended settings.

## Reading Mutable Native State

A runtime setting must be read at the point where the program needs it. The public controls use
`IO`: execute the setter, then the getter in the same sequence, as `gpuConfigure` does above.
Boolean setters also check the readback, so a rejected request cannot be reported as a successful
configuration.

Memory reporting follows the same rule: `Buffer.Memory.read` observes the native counters when the
action runs. Device selection is effectful too: `deviceCount` reports visible devices, `getDevice`
reads the selected index, and `setDevice` selects a device for subsequent bridge work. The setter
requires all existing buffer wrappers to be finalized (including empty and explicitly released
wrappers), and it doesn't migrate their tensors. Select the device before allocating model state.
Without LibTorch, `version` reports `unavailable`, `deviceCount` returns zero, and `getDevice`
returns the stub index zero, but that index doesn't establish availability: the setters fail.

# Reading CUDA Memory Usage

The LibTorch allocator owns device storage and its reuse policy. TorchLean's counters report
something related but different: which logical payloads and Lean buffer wrappers remain owned. Read
both levels at the point in the workload whose lifetime matters:

```lean (name := gpuMemory)
def gpuPrintMemory : IO Unit := do
  let memory ← LibTorch.Buffer.Memory.read
  IO.println memory.format
```

`liveBytes` and `peakBytes` count logical payloads owned by TorchLean handles, and since a view can
share storage with another handle, these counts can count the same storage twice. Conversely, an
ATen temporary or library workspace can own storage without a separate TorchLean handle.
`allocCount` and `freeCount` count payload lifetimes, not calls to the CUDA allocator. The wrapper
counters also include empty and explicitly released objects until Lean finalizes those wrappers.
These ownership counters are process-wide.

`allocatedBytes` and `reservedBytes` come from the LibTorch allocator for the selected CUDA device.
Allocations include library workspaces (a cuBLAS workspace, for instance) that can outlive every
TorchLean handle, so zero ownership counters need not imply zero allocated bytes.
`peakAllocatedBytes` and `peakReservedBytes` are the corresponding peaks. Reservation includes
storage retained for reuse, but subtracting allocated bytes doesn't give a promise of immediately
reclaimable memory. `deviceFreeBytes` and `deviceTotalBytes` report the driver's view of that same
device, and without LibTorch these device counters are zero. Each field is read separately, so
concurrent activity can make the report differ from an atomic snapshot.

For example, retiring a temporary tensor can reduce allocated bytes while leaving reserved bytes
unchanged, which is compatible with an allocator keeping storage for another operation. An
attention tape node also retains Q/K/V and probabilities for backward, so dropping an unrelated
local handle doesn't retire those saved buffers. Reading only free device memory would conflate
allocator reuse with values that the tape still needs.

An application can set the selected device's allocation limit and later release unused cache:

```lean (name := gpuMemoryControls)
def gpuLimitMemory : IO Unit :=
  LibTorch.setMemoryFraction 0.8

def gpuReleaseUnusedMemory : IO Unit :=
  LibTorch.emptyCache
```

The fraction, which must be finite and lie in `(0, 1]`, configures the upstream allocator's limit
and is neither a cache-only allowance nor a reservation against other processes. Read it back with
`getMemoryFraction` (allowing for native byte granularity). Setting the fraction doesn't release
live tensors.

`emptyCache` releases unused allocator blocks. Live parameters, optimizer state, saved forward
state, and library workspaces remain allocated, so both allocated and reserved bytes may remain
after the call. Calling it after every operation can throw away useful reuse. For timing a
completed phase, `LibTorch.synchronize` waits for the selected device's work, so
record whether that synchronization is included in a measurement.

# CUDA Test Coverage

Build against the SDK selected for the run, then execute the maintained device checks:

```terminal
export TORCHLEAN_LIBTORCH_HOME=/path/to/libtorch
scripts/lake.sh -Kcuda=true build
TORCHLEAN_REQUIRE_CUDA=1 scripts/lake.sh -Kcuda=true test
scripts/lake.sh -Kcuda=true lint
```

The retained CUDA suite checks buffer ownership, transfers, selected pointwise operations, softmax,
attention, selective scan, FFTs, parameter sharing, and a small training run. Attention checks
inspect forward values and input cotangents (including hard masks and fully blocked rows). Each
check covers its specific inputs and configuration.

Pooling and batch normalization with batch statistics use LibTorch implementations that the
maintained suite doesn't compare with the Lean reference. Convolution (grouped, dilated and padded,
value and the three pullbacks, against the CPU rule of `Op.conv`), batch normalization with running
statistics, and gather/scatter have retained checks.

A capsule names its test source but doesn't store a passing result for the current build. To assess
a run, keep its complete test output with the revision, SDK, driver, GPU, and relevant settings.
The chapter's executable planner examples and scalar calculations establish only what those
examples evaluate.

Passing device checks establishes that their cases worked on the recorded configuration. A claim
that ATen, its dependencies, the compiler, and the GPU refine every TorchLean specification would
require a further argument. Those components remain part of the implementation boundary described
in `docs/TRUST_BOUNDARIES.md` and in the capsule records.
