# TorchLean Trust Boundaries

TorchLean uses Lean to state and check mathematical claims about neural-network artifacts. Part of
the system sits inside Lean's proof kernel. The rest is executable tools, native runtimes and
external producers, whose outputs Lean may check.

## Levels of Assurance

| Layer | Example | Assurance |
| --- | --- | --- |
| Lean theorem | graph semantics, IR shape soundness, lowered-tape backward pass | checked by Lean |
| Executable checker | certificate parser, backend contract check | checked by code/tests |
| Prop-valued contract | agreement between two independent semantics | hypothesis from the caller |
| FFI/native runtime | LibTorch ATen, CUDA libraries, host tensor kernels | external implementation path |
| External producer | Python, Julia, alpha-beta-CROWN | produces artifacts Lean may check |

Two theorem families anchor the first row. `Graph.run_ports` in `TorchLean/NN/IR/Eval.lean`
proves that entry `j` of a successful run of a graph of registered operators (`Graph.run`) is at the
ports of node `j`, so every value has the element sort and shape its node declares. It needs no
shape-check premise, because each operator is typed by its ports and `Node.eval` reads a graph input
or constant at its node's port. `Node.eval_ports` is the per-node fact behind it. These graphs are
the canonical IR (`IR.Graph`, run by `Graph.run` and `Graph.eval`). Results about them are
forward statements, and the autograd theorems are about the proof-layer `Graph`.

`GraphData.toTape` in `TorchLean/NN/TypedGraph.lean` lowers a proof-carrying `Graph` to the
executable tape, with forward and backward correspondence theorems in
`TorchLean/NN/Tape/Lowering.lean` and `TorchLean/NN/Tape/Lowering/`. The backward chain ends in
`Graph.backwardDenseAll_eq_fderiv` in `TorchLean/NN/Tape/Lowering/BackwardDenseGraph.lean`:
`Tape.backwardDenseAll` on the lowered tape succeeds and returns `backpropAllCtx` of the one-hot
seed. Over `ℝ`, given a correct derivative of every node at the values the forward pass reaches
(`RealGraph.FDerivCorrectAt`), the input block of that context is the adjoint of the Fréchet
derivative of the forward map. No runtime path sweeps a lowered tape. The CPU eager tape that
`Tape.apply` records is linked per node (`TorchLean/NN/Tape/Apply/Correctness.lean`) and, for
layer normalization of a matrix, by `LayerNorm.eager_backward_eq_fderiv` in
`TorchLean/NN/Norm/LayerNorm/Eager.lean`. The checked typed-graph pullback is covered end to end
by `vjpChecked_eq_fderiv` in `TorchLean/NN/TypedGraph/WithData/Checked.lean`. Both families are
about the exact tape and spec models at the given carrier. Neither says anything about `Float`
rounding, compiled code or CUDA.

The local rules carry certificates over `ℝ`, which `@[op_cert]` registers and `#op_certs` lists. The
operators with a fixed number of inputs have 166 value inputs, and the gradient of each has a
`derivative` certificate. At the points its hypotheses name, the stored backward is the adjoint of
the Fréchet derivative of the forward in all the value inputs together. `cat` has
`cat_backward_eq_adjoint`, and the pullback that a custom operation supplies claims no derivative.
At kinks, `convention` certificates state the stored backward at every input.
`scripts/checks/Library.lean` fails when such a gradient has no `derivative` certificate. The
attribute checks that a statement mentions the operator's rule. It does not check that the statement
says what its kind says. `op_deriv` in `TorchLean/NN/Tactic/OpDeriv.lean` proves statements of the
same form for programs outside the table. Its proofs go through Lean's kernel, so the tactic can
fail on a true goal but cannot prove a false one. No theorem bounds the rounding of a stored
backward in `Float` or `Float32`.

For a correctness claim, cite the applicable theorem and hypotheses, checker predicate, or native
validation evidence. A successful command, an accepted artifact, and a proof are different results.

The backend planner and capsule vocabulary are documented in the
[Installation guide](https://lean-dojo.github.io/TorchLean/installation/#from-a-model-to-a-kernel).
A kernel capsule (`TorchLean.Backend.KernelCapsule`) names one registered operator (its
`Op.encode` record, attributes included), a provider and a device. It states a trust level
(`checked` or `trustedExternal`), a VJP mode and a reduction-order policy, and it carries four
contract descriptors: shape, layout, value and VJP. Each descriptor pairs a structured claim with
its evidence: a shared canonical definition, a runtime guard, a regression suite, a named trusted
boundary or not applicable. The planner selects a capsule only when its device, provider
preference, VJP mode and trust level fit the profile and each descriptor states the claim its field
advertises. The contract check (`checkContracts` in `TorchLean/NN/Backend/ContractCheck.lean`,
returning a `Contract.Check`) then confirms that no selected descriptor rests on a trusted boundary
unless the profile's assurance policy allows trusted-boundary evidence. Implementation tables may
extend a backend profile, but every capsule they offer passes through the same selection and
contract check.

The eager runtime (`Session.apply`) runs an operator's CUDA entry in `LibTorch.Kernel.ofOp` under a
selected capsule only when the entry's own capsule names the same operator, provider and device
(`KernelCapsule.implements`). Under a reference CPU capsule it runs the operator's rule `Op.rule?`
on the CPU tape. A backend report therefore cannot name one provider while its dispatch branch runs
another. The check compares those labels only. The entry's arithmetic, FFI code, compiler output,
driver and hardware rest on the evidence and trust level the capsule states. CUDA primitives are
registered under `Provider.libTorch`, with names such as `libtorch.matmul`. GPU multi-head attention
has one implementation, `libtorch.multihead_attention`, which composes the forward pass and the
local VJP in Lean over LibTorch numerical primitives. Scaled dot-product attention is recorded as
the key transpose and the scaled score product, followed by the rest of attention (row softmax, an
optional multiplier and the product with the values) under `libtorch.attention`.

The default `libTorchCUDA` profile prefers LibTorch and keeps TorchLean's global tape. It admits
the recorded LibTorch trust boundaries of operations without CUDA parity checks. The strict
`checkedCUDA` profile rejects those untested capsules and requires runtime guards and named
regression evidence. A `checked` classification does not put LibTorch inside Lean's proof
kernel. Likewise, `KernelPlan.hasTrustedExternal = false` means no capsule has the
`trustedExternal` classification. It does not mean that no foreign code runs. A descriptor that
names a regression suite records which validation is required. It is neither a proof nor the stored
result of a particular test run.

## Lean Axioms and Hidden Implementations

TorchLean declares no custom Lean axioms. The CUDA handle type `Buffer` is the carrier of
`TorchLean.LibTorch.BufferPointed`, an opaque `NonemptyType` in
`TorchLean/CUDA/LibTorch/Trusted.lean`, so its `Nonempty` instance comes from that value
with no asserted proposition. The instance lets compiled extern functions return buffers. It
allocates nothing and proves nothing about native memory.

Opaque definitions and executable replacements are not axioms, but an audit has to find them too.
The command below finds axioms, opaque definitions and both kinds of executable replacement
(`@[implemented_by]` and `@[extern]`):

```bash
rg -n "\\bopaque\\b|\\baxiom\\b|@\\[implemented_by|\\bextern \"" -g'*.lean' \
  TorchLean TorchLeanExamples TorchLeanTest
```

For any theorem used in a claim, `#print axioms theoremName` reports its transitive logical
dependencies. `scripts/lake.sh lint` runs the same check on every theorem of the modules under
`TorchLean/`, `TorchLeanExamples/` and `TorchLeanTest/`, private and generated theorems included,
and fails unless each depends on at most `propext`, `Classical.choice` and `Quot.sound`. The file
that runs it, `scripts/checks/Library.lean`, leaves out the root modules of the executables.

A `roundoff theorem` or `roundoff mixed theorem` declaration states the bound that the analysis
infers from the code, so its statement appears nowhere in the source, and `#check` prints it. Lean
checks the proof, but nothing compares the bound with an earlier one: a change to the code or to
FloatLib's rules can loosen it without failing the build.

`#roundoff_coverage` records the outcome of each route's analysis: proved, conditional, unsupported
or failed. The coverage check of `scripts/lake.sh lint` counts the outcomes for the configured
scalar but doesn't compare them with earlier ones, so a change can turn a proved route into a failed
one without failing the build or the lint.

## Prop-Valued Contracts

Some contracts are declared as `class ... : Prop`, `structure ... : Prop` or `def ... : Prop`. They
are not axioms, and by themselves they add no kernel assumption: a theorem that uses one holds only
when the caller supplies the fields. We count them in the trust model anyway, because public theorem
names and docs should make those assumptions visible.

The main examples:

- The CROWN passes on graphs of registered operators (`TorchLean.CROWN.Op`) are proved at any
  scalar whose endpoint operations are lawful (`LawfulBoundOps`, `LawfulNonlinearBoundOps`):
  `Op.Directed.encloses_of_run` for interval propagation and `CROWN.Op.outputBoxCROWN?_of_run`
  for the backward CROWN output box. The forward α-CROWN pass is proved over `ℝ` only, by
  `Op.Alpha.encloses_of_run`. A backend scalar without those instances needs its own proof of
  them. The executable output-box APIs seed propagation with the requested input region. For
  rounded scalar backends, `Op.outputBoxAlphaCROWN?`, which the verification API's `alphaCROWN`
  algorithm runs, falls back to directed CROWN and never reassociates rounded affine coefficients
  through the exact-real phased α-CROWN rules. An exact-real transfer theorem alone does not
  justify such reassociation in binary32. The node-certificate checker
  `TorchLean.Certificate.CROWNNode` replays the forward α-CROWN pass in binary32, rounded to
  nearest, so a certificate it accepts agrees with that replay and need not enclose the graph. For
  linear/ReLU chains, `TorchLean.Certificate.FiniteArtifact.accepts_graph_sound` proves the
  requested output inequalities of an artifact that `FiniteArtifact.accepts` accepts.
- `TorchLean.Examples.Lyapunov.Cert.ValidFor` is a substantive application contract: parsing
  endpoint values does not prove that they enclose the named Lyapunov function and orbital
  derivative.

## Executable Replacements

`@[implemented_by]` gives a definition one logical body, which proofs and reduction see, and a
different compiled body for performance. Lean theorems are about the logical body, and applying them
to compiled results needs an argument that the two bodies agree.

Parameter alias grouping in `TorchLean/NN/Session.lean` keys its buckets with a private `opaque`
lookup key whose compiled body (`@[implemented_by]`) reads the shape and the full-width addresses
of the three mutable `IO.Ref` cells. The logic has no model of addresses, so no theorem depends on
the key. Every candidate in a bucket must also pass `Cells.same`: equal shape and `IO.Ref.ptrEq`
on the value, CUDA-mirror and host-validity cells. Tensor contents and CUDA allocation addresses
do not decide aliases.

The refinement relies on Lean's address primitive returning stable addresses for live reference
cells. The index holds its representatives until the grouping call ends, which prevents address
reuse while a key is present, and no keys or references are cached between calls. The retie
regression in `TorchLeanTest/CUDA/ParameterAliases.lean` exercises this boundary on the CUDA
backend, but no theorem proves that aliased storages get equal keys.

The `Storage` instances of the packed `Float` and `UInt8` buffers in `TorchLean/Tensor/Storage.lean`
are built from core's `FloatArray` and `ByteArray` operations, and Lean proves the `Storage` laws
about them from core's definitions (for `FloatArray` through a private
`import all Init.Data.FloatArray.Basic`). The folds are proved equal to the array fold by
`FloatArray.foldlM_eq_foldlM_data` and `ByteArray.foldlM_eq_foldlM_data`, and the byte slice is one
`ByteArray.copySlice` call on the clamped interval, proved for all natural bounds by
`data_byteBufferAppendSlice`. What Lean does not check here is core's runtime: the C primitives
behind `FloatArray` and `ByteArray` and core's `implemented_by` loop for `foldlM`. Packed `Float`
slices and transposes run the C kernels described under Native Host Tensors, and the regressions in
`TorchLeanTest/Tensor/Storage.lean` exercise packed reads, updates, and slices.

A `@[csimp]` theorem `@f = @fFast` also makes compiled code run `fFast`, but Lean checks that
equality, so theorems about `f` need no further agreement argument. The core tensor operations in
`TorchLean/Tensor/Rep/Pointwise.lean` take this form: the native loops and slice copies of the
constant tensor, stacking, `map`, `zipWith`, `zipWith3` and `unstack` are proved equal to their
logical definitions. The buffer operations they call on packed storage are those of the `Storage`
instances for `Float` and `UInt8` in `TorchLean/Tensor/Storage.lean`.
`TorchLean/Tensor/Factorization.lean` takes the same form: compiled code runs array functions for
Cholesky columns, the Cholesky solve and the ridge solve, and the `@[csimp]` theorems
`choleskyColsFn_eq_choleskyCols`, `cholSolveFn_eq_cholSolve` and `solveRidgeFn_eq_solveRidge` prove
each equal to its logical definition on every scalar type with a `Context` instance, `Float` and
`Float32` included.

## Native Host Tensors

The public `TorchLean.Tensor` type has one row-major buffer with a certified length. `Float` uses
Lean's unboxed `FloatArray`, `UInt8` uses `ByteArray`, and other scalar types use `Array α`, whose
entries may be boxed. A flat buffer therefore does not guarantee contiguous, unboxed scalar values.
Proofs observe these representations through the ordinary `Storage.toArray` laws.

`csrc/cpu/torchlean_tensor.c` implements selected host operations:

- packed Float contiguous slices, rank-two transpose, and entrywise addition, subtraction,
  multiplication, and division in one kernel that selects the operation once per call
- the same four operations between packed `UInt8` entries, converted exactly to `Float`, and
  packed `Float` entries, in `byteFloatBufferBinary` and `floatByteBufferBinary` for the two
  operand orders
- rank-two transpose for ordinary `Array α` storage

Each operation has a proof-visible Lean model, the slices and transposes in
`TorchLean/Tensor/Storage.lean` and the entrywise kernels in
`TorchLean/Tensor/Native/Pointwise.lean`.
Coordinate and buffer theorems establish the model's public tensor semantics. Compiled execution
uses `@[extern]`, so those theorems do not prove the C implementation,
Lean compiler, or machine instructions. The C code checks sizes and allocation overflow at the
FFI boundary. Regression tests compare Float exceptional cases at the bit level and stress
ordinary-array ownership with repeated rational and complex transposes.

The ordinary-array transpose consumes its input according to Lean's generated calling convention. It
transposes a uniquely owned square array in place, transfers elements from a uniquely owned
rectangular array without changing their reference counts, and retains elements when the array is
shared. Correct reference counting in those branches is part of the native runtime boundary. The C
compiler auto-vectorizes the packed Float pointwise loops. Each scalar operation stays independent,
and vectorization gives no license to reassociate reductions.

## Native Host Allocation

On Linux, `scripts/lean_allocator.py` builds a private mimalloc 3.4.5 object for TorchLean's
native executables and shared libraries. It checks the source archive and Lean SDK header by
SHA-256, then compiles position-independent code with initial-exec thread-local storage. This
supports both native link modes without replacing the installed Lean runtime or the allocator
used by the Lean compiler and `#eval`.

The local patch addresses a lost global purge wakeup in mimalloc 3.4.5: a collector finishing its
scan can erase a deadline that another thread publishes during the scan. With the patch, publishing
an arena deadline completes an acquire-release atomic read-modify-write even when a global deadline
is already pending. The collector consumes that global timer with an acquire-release exchange before
taking its arena snapshot. Later publications survive scan completion, and retry publication
preserves the earliest pending deadline, so expired work left by the purge budget stays eligible.
Zero is reserved for the absence of a wakeup.

Per-arena timer, purge-bitmap, and free-range operations retain their upstream ordering. Controlled
regressions exercise collection/free interleavings and check page release through ordinary
collection. Those observations and the global memory-order argument do not constitute a Lean proof
or a complete concurrency proof of mimalloc, its compiler output, or operating-system behavior.

## LibTorch CUDA Runtime

Established CUDA model primitives cross the LibTorch ATen bridge. TorchLean keeps graph recording,
tape traversal, gradient accumulation policy and the selection of local VJPs. Native forward and
backward calls run with gradient recording disabled, so they create no second LibTorch autograd
graph, local or global. `VJPMode.backendVJP` means that a bridge routine evaluates the selected
local reverse rule. It does not hand differentiation to LibTorch. Attention advertises
`VJPMode.torchLeanTape` because Lean composes its local reverse rule as well as traversing the tape.
Spectral model composition likewise stays in Lean: FFT, frequency mixing, and inverse FFT call
the library's numerical primitives.

- `csrc/libtorch/` contains the native tensor operations. The Lean boundary is under
  `TorchLean/CUDA/LibTorch/`, and its CUDA names identify the execution device and FFI
  ABI. LibTorch, its CUDA dependencies, the compiler and the driver are outside Lean's proof kernel.
- Shape-erased tape values must match their native tensor element counts. Dimensions, indices and
  output sizes must fit the `UInt32` FFI ABI. Native guards must also check dtype, device and
  layout. The identity check `KernelCapsule.implements` compares operator, provider and device, and
  it proves neither those native guards nor the arithmetic correct.
- Native eager CUDA buffers hold contiguous row-major LibTorch binary32 or binary64 tensors. Custom
  configured-binary arithmetic uses byte tensors carrying complete scalar encodings, including saved
  values and gradients on the same tape. Native model operators do not accept those encoded formats.
  Native handles and views follow LibTorch ownership: Lean finalizers release their tensor owners,
  and LibTorch decides when the device memory is freed. Borrowed Lean arguments use `@&`, and native
  calls must respect that calling convention. Both native and encoded interfaces must preserve their
  dtype and buffer-layout requirements.
- Native buffer operations are marked `@[never_extract]` where resource behavior prevents commoning
  or deletion. Effectful constructors and release calls sequence allocation and destruction. A
  `Buffer` is a copyable reference and not a linear capability, so releasing one can invalidate raw
  aliases. LibTorch tensor ownership does not prove TorchLean's handle lifetimes.
- Sparse backward and optimizer paths retain their ownership obligations for seeds, saved values,
  gradient buffers, and parameter mirrors. Lifetime and numerical regressions must exercise the
  linked LibTorch implementation under the settings used by the application.
- The bridge must preserve TorchLean's selected local VJP conventions. These include a zero clamp
  derivative at the endpoints, a totalized square root (`Op.sqrt .clampNegative`) whose VJP
  multiplies the cotangent by zero on nonpositive inputs, equal splitting at eager elementwise
  min/max ties, and the smooth logarithm surrogate `log(softplus(x) + epsilon)`. Delegating
  arithmetic does not authorize replacing those rules with the derivative chosen by a similarly
  named library operation. Piecewise differentiability theorems retain their domain and no-tie
  hypotheses.
- The CUDA entries of `Op.max`, `Op.min` and `Op.clamp` build IEEE 754-2019 maximum and minimum
  from ATen's `maximum` and `minimum`, which propagate a NaN, and settle a pair of zeros by a sum.
  Their CPU rules compute the same values. The CUDA tests compare the `max` and `min` pullbacks
  with the CPU rules bit for bit at ties, infinities, signed zeros and NaN operands. They compare
  the forward values on the finite nonzero inputs of a composite test, so the values at a NaN or a
  signed zero rest on the adapter's construction. `Op.fmax`, `Op.fmin` and `Op.remainder` have no
  CUDA entry.
- Max pooling skips padded cells and sends each output cotangent to the first maximum in row-major
  order at ties, `0` counting above `-0` as in the IEEE 754-2019 `maximum`. NaNs propagate, with
  the last encountered NaN selected by both forward and backward (`poolWins` and `poolIndices` in
  `TorchLean/Tensor/Window.lean`). Average pooling adds padded cells as zero and divides by the
  window cells inside the padded input when `countIncludePad` is set, and by those inside the
  input otherwise (`poolCount`). Implementation-defined accumulation order does not permit
  changing these selection or padding conventions.
- Seeded random operations must preserve the documented seeded-runtime contract or reject an
  unsupported request. Using a library RNG is not itself evidence of agreement with TorchLean's
  seed-to-value mapping.
- ATen chooses the implementations of matrix products, convolutions, reductions, FFTs, scans and
  pooling. Their capsules advertise `implementationDefined` numerical ordering. Deterministic
  execution settings are runtime requests. They establish neither the reference left fold, nor a
  particular multiply-add contraction schedule, nor bit agreement with FloatLib. Unsupported
  requests must be reported, so that none passes silently as fulfilled.
- Runtime numerical evidence must identify the LibTorch/CUDA build, device, dtype, and relevant
  precision and determinism settings. Capsule reduction metadata alone does not record those
  choices. Bit-level agreement requires evidence for the executed operations and their rounding
  sequence.
- Runtime setting and allocator memory-fraction getters return checked `IO` results across the
  native boundary, as do configuration requests. Unknown setting IDs return errors. The bridge maps
  caught native out-of-memory and allocation exceptions to `IO.Error.resourceExhausted`, and other
  caught standard exceptions become IO errors. Successful readback records runtime state and says
  nothing about numerical agreement. Builds without LibTorch return zero for known settings and
  memory fraction, reject unknown setting IDs and reject configuration requests.
- Attention's mathematical definition is in `TorchLean/Tensor/Layers/Attention.lean`.
  `Buffer.attentionForward` composes matrix products and softmax in Lean and returns output plus
  probabilities. `Buffer.attentionBackward` uses Q/K/V, saved probabilities and the output cotangent
  to compose the softmax VJP and matrix products for `dQ`, `dK` and `dV`. The tape owns the saved
  buffers. There is no native attention context or fused-provider selection. Full score and
  probability matrices use quadratic memory in sequence length. Boolean masks keep zero
  contributions at blocked coordinates and zero fully blocked rows. Finite additive attention biases
  are a separate semantic operation. Lean composition depends on the numerical primitive boundary,
  and the specification theorem does not prove those native calls correct.
- Batched attention folds batch and head axes for execution and sums shared parameter cotangents
  over the batch. Its specification and typed-graph proofs describe the mathematical operation.
  Applying those results to the LibTorch execution requires the backend agreement boundary and
  numerical evidence.
- Stream ordering, asynchronous errors, synchronization and allocator behavior belong to the native
  bridge and LibTorch runtime. They need runtime validation, and neither capsule selection nor the
  Lean tape theorem establishes them.

## Custom Tensor Computations

`TorchLean.NN.Kernel.Function` applies scalar functions to tensors on CPU or compiles supported
native and configured binary calculations to CUDA. The field `Program.correct` and the theorems
`Program.lower_correct` and `Program.named_correct` prove source correspondence through the
expression and statement representations. `CUDA.denotes_emit` connects emitted source semantics
under supplied arithmetic and reader contracts. NVRTC, the CUDA driver, hardware arithmetic,
concurrent error recording and foreign memory are outside these theorems. The native bridge checks
reads, keeps input owners alive until completion, and bounds its compiled-module cache. Those checks
are native implementation obligations.

Configured binary arithmetic uses complete encoded words in CUDA buffers, including saved values
and gradients on the autograd tape. The generated arithmetic is compared with FloatLib, and Lean's
kernel does not verify it. The format descriptor, byte transfer, slicing, concatenation and zero
allocation are additional native implementation obligations. Encoded buffers cannot be used as
native optimizer checkpoints or passed to unsupported ATen operations.

`Program.of` accepts only TorchLean's named scalar encodings and checks that the source arithmetic
agrees with that encoding's operations. Overridden operations and variable scalar dictionaries are
rejected for GPU compilation. The public `Scalar` constructor is an extension trust boundary:
manually constructing a `Program` with a forged scalar dictionary does not verify that CUDA
implements those operations, even if its Lean source-equivalence certificate is valid.

Automatic recording has Lean's kernel check that a recipe rebuilt from the source term's functions
equals the original function. The table from source functions to recorded operators and the operand
wiring are trusted. Scalar branches record only the selected path and synchronize to Lean for the
decision. Fixed-parameter recurrences record their ordered steps. These source equalities do not
establish differentiability at branch boundaries or turn floating-point VJPs into exact real
derivatives. The derivative laws apply only under their domain assumptions.

CPU evaluates the original scalar function. CPU-only carriers keep their type through a reported
fallback, and unsupported source for a GPU-capable carrier is an error. Canonical IR stores a custom
node's checked body separately from its shape signature. A source proof supplies neither a
real-enclosure rule nor a derivative: IBP/CROWN, forward-only lowering and PyTorch export reject
unsupported custom paths. The [custom-computation guide](../TorchLean/CUDA/Kernel/README.md) lists
the supported frontend and the precision limits.

`Program.tangent` and `Program.gradient` derive the derivative programs of a custom operation from
its expression. Over `ℝ`, `Expr.hasFDerivAt_evaluate_tangent` and `Expr.gradient_eq_adjoint`
prove that they compute the Fréchet derivative and its adjoint where the evaluation is regular: no
division it reaches has a zero divisor, and no comparison has equal operands unless both are
constants. `Expr.map_gradient_eq_adjoint` carries this to programs over a scalar type whose
reading as reals keeps zero and one, as the IEEE binary formats and Lean's `Float32` and `Float`
do. No theorem bounds the rounding of a run, and in a binary format one infinite or NaN partial
derivative makes every entry of the gradient NaN.

The tree kernels of [`TorchLean/CUDA/Tree`](../TorchLean/CUDA/Tree/README.md) prove that the level
schedule of a fold computes its Lean function (`Fold.eval_encode`, `Step.schedule_eq`) and that
the CUDA text of a row kernel computes its expression for the native formats (`denotes_emitRow`,
`Step.denotes_source`). The device run is trusted: NVRTC, the launches in
`csrc/libtorch/tree.cpp`, the bytes, and the agreement of the device arithmetic with
`Scalar.arithmetic`. The CUDA text of the configured formats is tested only. The certificates of
hylomorphisms and unfolds give partial correctness, and a run past its capacity computes on the
host. A kernel table of a unary function (`TorchLean/CUDA/Kernel/Table.lean`) gives the function
at every input by `Table.eval_eq_of_check`, and its device run is trusted in the same way.

## Executable Floating Point

- FloatLib supplies the executable floating-point formats, software arithmetic, rounding theory and
  interval semantics. `lakefile.lean` tracks its `main` branch, and `lake-manifest.json` locks the
  checked-out revision. Consult that manifest for the exact pin. Lean and mathlib use 4.35.0.
  TorchLean's runtime and certificate interfaces use `ExecFloat.Binary 8 23`. The typed
  tensor/model API supports FloatLib's configured binary family, including custom precision with
  valid widths, bias and storage plans. It does not supply tensor `Context` instances for FloatLib's
  posit, fixed-point or decimal families. Higher-precision training uses typed state, graphs and
  `nn.sgdStep` on CPU, or a supervised session opened with `trainer.openTyped (α := Scalar)`. Typed
  sessions keep `Scalar` through inputs, state, predictions, losses, reports and model-state
  checkpoints, and `finish` takes an independent snapshot. They reject non-CPU devices and custom
  backend profiles, and their results do not provide a verifier. Seeded initialization and
  optimizer/scheduler coefficients start from `Float`, so supply typed initial state when those
  extra digits matter. The ordinary `trainer.open` interface keeps `Float` input/result boundaries
  and uses `arithmetic := .ieee` to select binary32 internally. Native GPU dtype support does not
  provide execution of arbitrary FloatLib formats.
- Lean defines ordinary `Float32` addition, subtraction, multiplication, division, negation,
  absolute value, square root, bit conversion, comparison and classification through the canonical
  `Float32.Model` visible to the kernel. Compiled programs run native `@[extern]` implementations,
  which are a deployment boundary, and Lean's transcendental `Float32` functions are opaque and
  outside the core model bridge. FloatLib's public `FloatLib.Floats.Formats.IEEE754` import supplies
  native conversions and operation-specific correspondence theorems. Configured division
  has its own all-input software-model refinement, which does not prove agreement with arbitrary
  native `Float32` division. The binary64 interfaces use `ofFloat` and `toFloat`, and the binary32
  theorems are these:
  - `ExecFloat.Binary.toFloat32_ofFloat32` proves the exact native round trip for every value.
  - `ExecFloat.Binary.ofFloat32_add_of_isFinite` and `ofFloat32_sub_of_isFinite` require both
    operands to be finite, including signed zeros and subnormals. Their results may overflow.
  - `ExecFloat.Binary.toFloat32_sqrt` covers every configured input through native export, which
    canonicalizes NaNs. It does not preserve arbitrary NaN payloads.
- `TorchLean/Verify/` checks graph-wide binary32 interval
  traces against a graph of registered operators (`IR.Graph`). It rebuilds every range and
  trusts no claimed endpoint, rejects non-finite replay values and re-runs backend planning before
  accepting the embedded execution audit. A `NumericalCertificate.Checked` stores the exact graph
  checked, and `executeIEEE32` can replay only that stored graph. The replay is FloatLib reference
  execution and does not rerun the selected LibTorch kernels. The certificate's provider/capsule
  identities must match the recomputed audit. The audit is planning data and does not show which
  implementation produced an imported tensor.

`TorchLean/Tensor/ConfiguredBinary.lean` supplies a `Context` for FloatLib's configured
binary values. The exponent and fraction widths are part of the scalar type and can exceed
binary64's widths. Tensor storage keeps these values in `Array α`, and integer and rational
constants are rounded directly in the selected format. A value constructed through a native `Float`
first keeps that earlier rounding, whatever the destination precision.

The context makes the shared tensor and model specifications executable with software arithmetic.
It does not supply a CUDA storage representation, a trainer/checkpoint encoding or an ordered field
law for floating-point operations. Typed sessions obtain their checkpoint encoding separately from
`Checkpoint.Encoding`. For an IEEE descriptor its `exp`, `log`, `log1p`, `expm1` and `expMul` are
FloatLib's correctly rounded functions, and `ConfiguredBinary.exp_of_isIEEE` and its siblings prove
that the dictionary selects them. Its `tan`, `exp2`, `log2`, `log10`, `asinh`, `acosh` and `atanh`
round the real function once on every descriptor, with C99's special values
(`TorchLean.Tensor.ConfiguredBinary.Elementary`), and
`TorchLean.Verify.RuntimeApprox.Elementary` proves the rounding for an IEEE descriptor. Its `sin`,
`cos`, `sinh`, `cosh` and `tanh`, and its `exp`, `log`, `log1p` and `expm1` for other descriptors,
are FloatLib's deterministic approximations or compensated formulas, which have no general error or
correct-rounding guarantee. A proof about those functions must supply the bounds it uses.

The context's default safeguard is the rounded rational `1/1000000`. If that rounds to zero in a
coarse format, the context uses its smallest positive subnormal instead. The fallback keeps the
default nonzero, but it can substantially change a guarded formula. The safeguard is neither machine
epsilon nor an accuracy bound, and tolerances belong to the model's numerical specification.

Normalization has a separate default, the exact rational `1/100000` rounded once in the selected
scalar. It stays nonzero in binary16 but can round to zero in a coarser format. The normalization
default has no minimum-subnormal fallback, so callers must supply a positive, representable
normalization epsilon when the default is too small. Validation of a rational model configuration
checks positivity before scalar conversion and does not establish positivity after rounding.

Kernel capsules record one numerical choice, the reduction order (`NumericalPolicy.reduction`):
`fixedLeft`, `fixedLeftOrExact`, `implementationDefined` or `notApplicable`. It is contract data
for the audit and carries no proof. The certificate guard `fixedLeftGuard` in
`TorchLean/Verify/NumericalCertificate/Contracts.lean` reads it as an `Except`
precondition and rejects any node whose transfer folds a reduction from the left (`fixedLeftNode`)
when its capsule does not advertise `fixedLeft`. The scaled matrix contraction transfer requires
`fixedLeftOrExact` instead. Rounding mode, subnormal handling, and multiply-add contraction are not
recorded. A certificate that depends on them must state the assumption itself. Portable reference
accumulations advertise the fixed left fold, except that `scaledMatmul`, `softmax`, `maskedSoftmax`
and `distill` advertise `fixedLeftOrExact` and the reference attention operators advertise
`implementationDefined`. LibTorch matrix products, convolutions, normalizations, pooling operations,
FFT/FNO paths, scans and attention advertise implementation-dependent reductions, so the fixed-left
graph certificate refuses to reuse its transfer for those accelerated paths. A theorem about such a
path needs a backend-specific schedule or an applicable order-independent enclosure from
`TorchLean/Verify/Floats/IEEE754/Reductions.lean`. That enclosure has its own operation and
finiteness hypotheses, and its generic forms also take local-error bounds as hypotheses. The
binary32 corollaries use FloatLib's tree-local unit-roundoff result. The dot-product corollary
bounds the sum of rounded products, so multiplication errors are a separate obligation.
None of these theorems covers an arbitrary ATen algorithm.

For a checked replay, interval validity proves that each endpoint is finite and ordered. The replay
also checks every computed entry for finiteness, and `executeIEEE32` returns a `CheckedExecution`
only when every FloatLib binary32 node value lies inside its checked range. That is the whole of
what the checker proves. It does not prove that the exact-real denotation of the graph lies in those
ranges. That statement is the structure `NumericalCertificate.RealEnclosure` in
`TorchLean/Verify/NumericalCertificate.lean`. Its fields require a real payload and inputs, the
complete real node trace, a proof that the trace is the graph's real denotation, and a pointwise
proof that each real value lies in its checked interval. The caller must supply this structure, and
nothing in the repository constructs one. Given both, the theorem `CheckedExecution.errorTrace`
yields the pointwise interval-width error bound for every intermediate. It is a theorem about the
FloatLib binary32 replay. Transporting it to Lean runtime `Float32`, CUDA, LibTorch, cuBLAS or cuDNN
requires the agreement recorded by that backend's capsule.

The `RevGraph` theorems bound the rounded forward and reverse passes of a `RevGraph` whose nodes
carry local error certificates, and such a graph erases to executable autograd `GraphData`. No
model or `TorchLean.IR.Graph` is lowered to a `RevGraph`, so the caller builds the graph. One
optimizer contract carries those gradient bounds through SGD, momentum SGD and AdamW. For AdamW,
the assumptions of each step include a lower bound on the complete denominator and an error margin
for it. These are Lean theorems about the `NF` rounded-real scalar model.

The ODE corridor theorems in `TorchLeanExamples/Verification/ODE/Enclosure.lean` require
continuity, derivative identities, and corridor inequalities for a real interpretation. Discrete
samples of a floating-point trajectory reach them only through a real-valued interpolant. Sample
values and boxes alone do not establish the clamped-ODE derivative identity of that interpolant.
A directly decoded trajectory of a configured binary format, such as binary32, has finite range.
`decoded_eq_of_continuousOn` in `TorchLeanExamples/Verification/ODE/EnclosureBackends.lean` proves
that when its decoded real values are continuous on `Icc a b`, they are equal at any two points of
`Icc a b`.

Use the float layers as follows:

| Claim | Layer to cite |
| --- | --- |
| executable configured binary behavior inside Lean | FloatLib's `ExecFloat.Binary` and `BinaryInterchange` refinement theorems |
| TorchLean binary32 certificate representation | FloatLib's `ExecFloat.Binary 8 23` |
| finite rounded-real float32 error bound | `TorchLean/Verify/Floats/FP32.lean` |
| precision-parametric rounding theorem | `FloatLib.Floats.Formats.Flocq` |
| endpoint interval enclosure | FloatLib's `BinaryInterchange.IntervalSemantics` |
| meaning of Lean `Float32` core arithmetic | Lean's `Float32.Model` definitions |
| native logical import/export and arithmetic correspondence | FloatLib's `IEEE754.Native` proof modules, with each operation's stated hypotheses |
| compiled `Float`/`Float32`, CUDA, or LibTorch behavior | provider bridge or boundary statement |

## External Numeric Oracles

- PyTorch import/export scripts and training helpers are external producers of weights, examples, or
  JSON artifacts. TorchLean can parse and replay those artifacts, but PyTorch training itself is not
  part of Lean's trusted kernel. The operator records of the `torchlean.ir` artifacts of that
  exchange have a round-trip theorem in Lean (`Op.decode_encode` in
  `TorchLean/Tensor/Op/Codec.lean`), and `TorchLean.IR.load_ok` proves that a loaded artifact's
  program passes `Program.check` and its payload `Payload.check`. The Python side that emits them
  remains a producer.
- The optional Julia wrapper `TorchLeanExamples/Verification/Splines/Julia.lean` follows the same
  pattern. It resolves `TORCHLEAN_JULIA` when set, otherwise falls back to `julia` on `PATH`, and
  does not require Julia at compile time. It supports "untrusted producer, Lean checker" workflows
  such as the piecewise-polynomial spline certificate workflow (the producer script and bundled
  fixtures under `TorchLeanExamples/Verification/Splines/`).
- A Julia-produced spline artifact or a Python-produced PINN artifact is trusted only after a Lean
  checker validates the small certificate data it needs: for example cell domains, polynomial
  coefficients, interval bounds, and claimed residual inequalities. A successful exit doesn't make
  the producer's fitting process, optimizer, GPU use, or floating-point arithmetic trusted.
