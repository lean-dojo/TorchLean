# TorchLean Trust Boundaries

TorchLean uses Lean to state and check mathematical claims about neural-network artifacts. Some
parts of the system are inside Lean's proof kernel; others are executable tools, native runtimes, or
external producers whose outputs may be checked by Lean.

This document records the assumptions that matter for correctness claims: Lean axioms, Prop-valued
contracts, CUDA and FFI code, external numeric oracles, PyTorch import/export scripts, Julia/Python
producers, and artifact-checking conventions.

## Levels of Assurance

| Layer | Example | Assurance |
| --- | --- | --- |
| Lean theorem | graph semantics, IR shape soundness, executed backward sweep | checked by Lean |
| Executable checker | certificate parser, backend contract check | checked by code/tests |
| Prop-valued contract | agreement between two independent semantics | hypothesis from the caller |
| FFI/native runtime | LibTorch ATen, CUDA libraries, host tensor kernels | external implementation path |
| External producer | Python, Julia, Arb, alpha-beta-CROWN | produces artifacts Lean may check |

Two theorem families anchor the first row. `NN/IR/ShapeSoundness.lean` proves that the IR shape
checker and the IR evaluator agree: `checkShapes_sound` and `denoteAll_shape` state that every value
produced by `Graph.denoteAll` has the declared shape of its node, and `evalNodeRaw_shape_of_infer`
is the per-operator fact behind them. For autograd, the executed eager sweep is covered by
`backwardDenseAll_lowerGraphToTape_adjoint_fderiv` in
`NN/Proofs/Autograd/Runtime/Link/BackwardDenseGraph.lean`: running `Tape.backwardDenseAll` on a
tape lowered from a proof-carrying `Graph` succeeds, returns `backpropAllCtx` of the one-hot seed,
and over `ℝ` its input block is the adjoint of the Fréchet derivative of the forward map. Both are
statements about the exact tape and spec models at the given carrier; neither says anything about
`Float` rounding, compiled code, or CUDA.

When writing a correctness claim, name the layer explicitly:

- theorem claim: cite the Lean theorem and its hypotheses;
- executable-checker claim: cite the checker command, artifact schema, and accepted predicate;
- runtime claim: cite the backend, tests, sanitizer/parity evidence, and remaining native boundary;
- producer claim: cite the external tool or script and the artifact that Lean later checks.

This avoids collapsing "the command ran", "the checker accepted this artifact", and "Lean proved a
mathematical statement" into one sentence.

The backend planner and capsule vocabulary are documented in the
[Installation guide](https://lean-dojo.github.io/TorchLean/installation/#from-a-model-to-a-kernel).
A kernel capsule (`NN.Backend.KernelCapsule`) names one operation, provider, and device, states a
trust level (`checked` or `trustedExternal`), a VJP mode, and a reduction-order policy, and carries
four contract descriptors (shape, layout, value, VJP). Each descriptor pairs a structured claim with
its evidence: a runtime guard, a regression suite, a named trusted boundary, or not applicable. The
planner selects a capsule only when its device, provider preference, VJP mode, and trust level fit
the profile and each descriptor states the claim its field advertises. The contract check
(`checkContracts` in `NN/Backend/ContractCheck.lean`, returning a `ContractCheck`) then confirms
that no selected descriptor rests on a trusted boundary unless the profile's assurance policy allows
trusted-boundary evidence. Capsule modules may extend a backend profile, but every contributed capsule
passes through the same selection and contract check.

The eager runtime binds a selected capsule to a typed handler only when operation, provider, and
device agree. This prevents a backend report from naming one provider while its dispatch branch
runs another. The binding proves only that identity agreement; it does not prove the handler's
arithmetic, FFI code, compiler output, driver, or hardware. Those obligations retain the evidence
and trust level stated by the capsule. CUDA primitives are registered under `Provider.libTorch`,
with names such as `libtorch.matmul`. Attention also exposes `torchlean.composed_attention`, where
Lean composes LibTorch primitives, and `libtorch.direct_attention`, where the native bridge evaluates
forward and the selected local VJP. Both keep the global tape in TorchLean.

The maintained `checkedCuda` profile prefers LibTorch, including direct attention, and retains
TorchLean's global tape. An explicit `.prefer .torchLean` provider preference selects composed
attention for comparison. The profile requires runtime guards and named regression evidence. Its
`checked` classification does not mean that LibTorch is inside Lean's proof kernel. Likewise,
`KernelPlanAudit.hasTrustedExternal = false` means no capsule has the `trustedExternal`
classification; it does not mean that no foreign code runs. A descriptor naming a regression suite
records the required validation, not a proof or a stored result from a particular test run.

## Lean Axioms and Hidden Implementations

TorchLean currently declares no custom Lean axioms. The CUDA handle type is instead carried by
`Runtime.Autograd.Cuda.BufferImpl`, an opaque `NonemptyType` in
`NN/Runtime/Autograd/Engine/Cuda/Trusted.lean`. Its `Nonempty` instance is obtained from that data
carrier rather than asserted as a proposition. This lets compiled extern functions return buffers,
but it neither allocates a buffer nor proves anything about native memory.

Opaque definitions and executable replacements are still important during an audit even though
they are not axioms. The following command finds all three mechanisms:

```bash
rg -n "\\bopaque\\b|\\baxiom\\b|@\\[implemented_by" NN -g'*.lean'
```

For any theorem used in a claim, `#print axioms theoremName` reports its transitive logical
dependencies.

## Prop-Valued Contracts

Some declarations are `class ... : Prop`, `structure ... : Prop`, or `def ... : Prop` rather than
axioms. These are not kernel assumptions by themselves: a theorem using one is conditional on the
caller supplying the fields. We still treat them as part of the trust model, because public theorem
names and docs should make those assumptions visible.

Important examples include:

- `NN.MLTheory.CROWN.Graph.CrownCertSoundness.CrownTransferSound` is the generic graph-CROWN
  transfer interface, the kernel of the certificate checker theorem. TorchLean proves it for the
  concrete α-CROWN and α/β-CROWN transfer rules (`alphaCrown_transfer_sound`,
  `alphaBetaCrown_transfer_sound`). Those
  transfer theorems take the IBP boxes as a hypothesis (`IBPEnclosesVals`).
  `NN/MLTheory/CROWN/Proofs/GraphAlphaCrownTransferSoundness/EndToEnd.lean` discharges that
  hypothesis from the IBP soundness theorem `cert_encloses_semantics`
  (`ibp_encloses_vals_of_cert_local_ok`) and states the composed corollaries
  `alphaCrown_cert_encloses_semantics`, `alphaBetaCrown_cert_encloses_semantics'`, and
  `alphaCrown_cert_encloses_evalGraphRec`, which have no `IBPEnclosesVals` hypothesis.
  `NN/MLTheory/CROWN/Proofs/GraphRunibpEndToEnd.lean` connects the proof-side IBP pass to the
  engine's executable `runIBP` (`runIBP_eq_runIBP?`, `runIBP_encloses_evalGraphRec`). All of these
  are statements over `ℝ`. A new backend-dependent transfer rule must provide its own
  `CrownTransferSound` proof.
- `NN.MLTheory.Proofs.UniversalApproximation.FloatIntervalApprox.OpsExact.Sound` packages local
  finite-IEEE32 interval obligations. Addition, multiplication, and ReLU have proved theorems and
  an unconditional instance in `NN/MLTheory/Proofs/Approximation/FloatInterval/Semantics.lean`;
  callers do not supply this instance.
- `NN.MLTheory.CROWN.Lyapunov.LyapunovCert.ValidFor` remains a substantive application contract:
  parsing endpoint values does not prove that they enclose the named Lyapunov function and orbital
  derivative.
- `roundedTargetExactIntervalImage_of_correctRounding` is conditional on correct rounding, real
  activation conditions, threshold-network construction, and exact interval-semantics
  construction. These premises are mathematical obligations, not facts inferred from execution.

## Executable Replacements

`@[implemented_by]` lets a definition have one logical body for reduction and a different compiled
body for performance. Lean theorems describe the logical body; applying them to compiled results
requires an agreement argument between the two implementations.

`NN/Spec/Core/Tensor/Factorizations.lean` uses this mechanism for Cholesky columns, triangular
solves, and ridge solves. Their strict array implementations are tested executable code, not
consequences of the reconstruction theorems about the logical definitions.

Parameter alias grouping in `NN/Runtime/Autograd/Torch/Core/Session/State.lean` uses a private
executable replacement for its lookup key. The logical key groups by shape; the compiled key also
uses full-width addresses of the three mutable `IO.Ref` cells. Every bucket candidate still passes
`ParameterStorage.same`: equal shape and `IO.Ref.ptrEq` on the value, CUDA-mirror, and host-validity
cells. Tensor contents and CUDA allocation addresses do not decide aliases.

This refinement relies on Lean's address primitive returning stable addresses for live reference
cells. The index retains its representatives until that grouping call ends, preventing address
reuse while a key is present; no keys or references are cached between calls. Collision and retie
regressions exercise this boundary, but do not constitute a formal proof of compiled agreement.

## Native Host Tensors

The public `TorchLean.Tensor` type has one certified contiguous row-major buffer. `Float` uses
Lean's unboxed `FloatArray`, `UInt8` uses `ByteArray`, and other scalar types use `Array α`.
Proofs observe all three through the ordinary `Storage.toArray` laws.

`csrc/cpu/torchlean_tensor.c` implements selected host operations:

- packed Float contiguous slices, rank-two transpose, and entrywise addition, subtraction,
  multiplication, and division;
- fused `UInt8 + Float` and `Float + UInt8` promotion and addition;
- rank-two transpose for ordinary `Array α` storage.

Each operation has a proof-visible Lean model under `NN/Tensor/Internal/Elab/Native/`.
Coordinate and buffer theorems establish the model's public tensor semantics. Compiled execution
uses `@[implemented_by]` and `@[extern]`, so those theorems do not prove the C implementation,
Lean compiler, or machine instructions. The C code checks sizes and allocation overflow at the
FFI boundary. Regression tests compare Float exceptional cases at the bit level and stress
ordinary-array ownership with repeated rational and complex transposes.

The ordinary-array transpose consumes its input according to Lean's generated calling convention.
It transposes a uniquely owned square array in place, transfers elements from a uniquely owned
rectangular array without changing their reference counts, and retains elements when the array is
shared. Correct reference counting in those branches remains part of the native runtime boundary.
The packed Float pointwise loops are auto-vectorized by the C compiler; they preserve independent
scalar operations but do not authorize reassociation of reductions.

## Native Host Allocation

On Linux, `scripts/lean_allocator.py` builds a private mimalloc 3.4.4 object for TorchLean's
native executables and shared libraries. It checks the source archive and Lean SDK header by
SHA-256, then compiles position-independent code with initial-exec thread-local storage. This
supports both native link modes without replacing the installed Lean runtime or the allocator
used by the Lean compiler and `#eval`.

The local patch addresses a lost global purge wakeup: a collector finishing its scan could erase
a deadline another thread had just published. Publishing a newly set arena deadline now completes
an acquire-release atomic read-modify-write even when a global deadline is already pending.
The collector consumes that global timer with an acquire-release exchange before taking its arena
snapshot. Later publications survive scan completion, and retry publication preserves the earliest
pending deadline so expired work left by the purge budget stays eligible. Zero remains reserved
for the absence of a wakeup.

Per-arena timer, purge-bitmap, and free-range operations retain their upstream ordering. Controlled
regressions exercise collection/free interleavings and check page release through ordinary
collection. Those observations and the global memory-order argument do not constitute a Lean proof
or a complete concurrency proof of mimalloc, its compiler output, or operating-system behavior.

## Opaque Non-FFI Declarations

- `NN.MLTheory.CROWN.betaAt` is an opaque executable wrapper around `Array.get!`. Its caller,
  `phaseRelaxVec?`, checks the phase-array length before indexing. The wrapper alone is total and
  would return `get!`'s fallback for an out-of-bounds index; its safety claim therefore belongs to
  the caller's control flow.

## LibTorch CUDA Runtime

All CUDA primitive numerical work crosses the LibTorch ATen bridge. TorchLean retains graph
recording, tape traversal, gradient accumulation policy, and the selection of local VJPs. Native
forward and backward calls execute with gradient recording disabled; they do not create a second
local or global LibTorch autograd graph. `VJPMode.backendVJP` means a bridge routine evaluates the
selected local reverse rule, not that LibTorch owns differentiation.

- `csrc/libtorch/` contains the native tensor operations. The Lean boundary remains under
  `NN/Runtime/Autograd/Engine/Cuda/`; its CUDA names identify the execution device and FFI ABI.
  LibTorch, its CUDA dependencies, the compiler, and the driver remain outside Lean's proof kernel.
- Shape-erased tape values must match their native tensor element counts. Dimensions, indices, and
  output sizes must fit the `UInt32` FFI ABI. Native guards must also check dtype, device, and
  layout. A typed capsule/handler binding proves agreement of operation, provider, and device;
  it does not prove those native guards or arithmetic correct.
- Eager CUDA buffers hold contiguous row-major LibTorch float32 tensors. Native handles and views
  follow LibTorch ownership; Lean finalizers release their tensor owners rather than specifying
  direct `cudaFree` behavior. Borrowed Lean arguments still use `@&`, and native calls must respect
  that calling convention. Dtype-specific interfaces remain separate from FloatLib's configured
  software formats.
- Native buffer operations remain marked `@[never_extract]` where resource behavior prevents
  commoning or deletion. Effectful constructors and explicit release sequence allocation and
  destruction. A `Buffer` is still a copyable reference, not a linear capability: explicit release
  can invalidate raw aliases. LibTorch tensor ownership does not prove TorchLean's handle lifetimes.
- Sparse backward and optimizer paths retain their ownership obligations for seeds, saved values,
  gradient buffers, and parameter mirrors. Lifetime and numerical regressions must exercise the
  linked LibTorch implementation under the settings used by the application.
- The bridge must preserve TorchLean's selected local VJP conventions. These include a zero clamp
  derivative at the endpoints, a totalized square root with zero VJP on nonpositive inputs,
  equal splitting at eager elementwise min/max ties, and the smooth logarithm surrogate
  `log(softplus(x) + epsilon)`. Delegating arithmetic does not authorize replacing those rules
  with the derivative chosen by a similarly named library operation. Piecewise differentiability
  theorems retain their domain and no-tie hypotheses.
- Max pooling skips padded cells and sends each output cotangent to the first row-major winner,
  including at ties (`NN/Spec/Layers/Pooling/Spatial.lean`). Average pooling counts padded cells as
  zero in the whole-window denominator. Implementation-defined accumulation order does not permit
  changing these selection or padding conventions.
- Seeded random operations must preserve the documented seeded-runtime contract or reject an
  unsupported request. Using a library RNG is not itself evidence of agreement with TorchLean's
  seed-to-value mapping.
- ATen chooses the implementations of matrix products, convolutions, reductions, FFT/FNO, scans,
  and pooling. Their capsules advertise `implementationDefined` numerical ordering. Deterministic
  execution settings are runtime requests; they do not establish the reference left fold, a
  particular multiply-add contraction schedule, or FloatLib bit agreement. Unsupported requests
  must be reported rather than silently treated as fulfilled.
- Runtime numerical evidence must identify the LibTorch/CUDA build, device, dtype, and relevant
  precision and determinism settings. Capsule reduction metadata alone does not record those
  choices. Bit-level agreement requires evidence for the executed operations and their rounding
  sequence.
- Runtime setting and allocator memory-fraction getters return checked `IO` results across the
  native boundary, as do configuration requests. Unknown setting IDs return errors. The bridge
  maps caught native out-of-memory and allocation exceptions to `IO.Error.resourceExhausted`;
  other caught standard exceptions become IO errors. Successful readback records runtime state,
  not proof of numerical agreement. Builds without LibTorch return zero for known settings and
  memory fraction, reject unknown setting IDs, and reject configuration requests.
- Attention retains its proof-facing denotation in `NN/Spec/Layers/FlashAttention.lean`. The
  composed route evaluates matrix products and hard-masked softmax through LibTorch primitives;
  the direct bridge returns forward values and local `dQ`, `dK`, and `dV`. Neither capsule promises
  a particular FlashAttention algorithm. Boolean masks retain zero contributions at blocked
  coordinates and zero fully blocked rows. Finite additive attention biases remain a separate
  semantic operation.
- Batched attention still folds batch and head axes for execution and sums shared parameter
  cotangents over the batch. Its specification and typed-graph proofs describe the mathematical
  operation. Applying those results to the LibTorch execution requires the backend agreement
  boundary and numerical evidence.
- Stream ordering, asynchronous errors, synchronization, and allocator behavior belong to the
  native bridge and LibTorch runtime. They require runtime validation; neither capsule selection
  nor the Lean tape theorem establishes them.

## Executable Floating Point

- FloatLib supplies the executable floating-point formats, software arithmetic, rounding theory,
  and interval semantics. `lakefile.lean` tracks its `main` branch; `lake-manifest.json` locks the
  checked-out revision, currently `5ef396e35a856a19e24befde631d0bb1ec0a237b`, with Lean and mathlib 4.34.0.
  TorchLean's runtime and certificate interfaces use `ExecFloat.Binary 8 23` directly.
  The typed tensor/model API supports FloatLib's configured binary family, including custom
  precision with valid widths, bias, and storage plans. This does not supply tensor `Context`
  instances for FloatLib's posit, fixed-point, or decimal families.
  Higher-precision training uses typed state, graphs, and `nn.sgdStep` on CPU. The supervised
  trainer's input/report/checkpoint boundary remains `Float`; `.arithmetic := .ieee` selects
  binary32. Native GPU dtype support does not provide execution of arbitrary FloatLib formats.
- Lean defines ordinary `Float32` addition, subtraction, multiplication, division, negation,
  absolute value, square root, bit conversion, comparison, and classification through the
  canonical `Float32.Model` visible to the kernel. FloatLib's public
  `FloatLib.Floats.Formats.IEEE754` import supplies native conversions and operation-specific
  correspondence theorems directly:
  - `ExecFloat.Binary.toFloat32_ofFloat32` proves the exact native round trip for every value.
  - `ExecFloat.Binary.ofFloat32_add_of_isFinite` and `ofFloat32_sub_of_isFinite` require
    both operands to be finite, including signed zeros and subnormals. Their results may overflow.
  - `ExecFloat.Binary.toFloat32_sqrt` covers every configured input through native export,
    which canonicalizes NaNs. It does not preserve arbitrary NaN payloads.
  The corresponding binary64 interfaces use `ofFloat` and `toFloat`. Configured division has its
  own all-input software-model
  refinement, which does not prove agreement with arbitrary native `Float32` division.
  The `@[extern]` implementations used by compiled programs are still native code and remain a
  deployment boundary. Lean's transcendental `Float32` functions are opaque and are not covered by
  the core model bridge.
- `NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean` checks graph-wide binary32 interval
  traces against the canonical `NN.IR.Graph`. It rebuilds ranges rather than trusting claimed
  endpoints, rejects non-finite replay values, and re-runs backend planning before accepting the
  embedded execution audit. A `RegistryCheckedCertificate` stores the exact graph checked, and
  `executeIEEE32` can replay only that stored graph. This is FloatLib reference execution, not a
  replay of the selected LibTorch kernels. Changing provider/capsule identities changes the audit;
  certificates containing old identities must be regenerated and checked. The audit is planning
  data, not proof of which implementation produced an imported tensor.
- Transcendental functions such as `exp`, `log`, and `tanh` are deterministic approximations unless
  a file states a stronger theorem for a specific operation.

`NN/Spec/Core/FloatInstances.lean` supplies a `Context` for FloatLib's configured binary values.
The exponent and fraction widths are part of the scalar type; they can exceed binary64's widths.
Tensor storage keeps these values in `Array α`, and integer and rational constants are rounded
directly in the selected format. Constructing a value through a native `Float` first retains that
earlier rounding, regardless of the destination precision.

This context makes the shared tensor and model specifications executable with software arithmetic.
It does not supply a CUDA storage representation, a trainer/checkpoint encoding, or an ordered
field law for floating-point operations. Its elementary functions come from FloatLib's explicit
binary transcendental module, whose deterministic approximations have no general error or
correct-rounding guarantee. A proof about those functions must supply the bounds it uses.

The context's default safeguard is the rounded rational `1/1000000`. If that rounds to zero in a
coarse format, the context uses its smallest positive subnormal instead. This keeps the default
nonzero, but it can substantially change a guarded formula. The safeguard is neither machine
epsilon nor an accuracy bound; tolerances remain part of the model's numerical specification.

Normalization has a separate default, the exact rational `1/100000` rounded once in the selected
scalar. It remains nonzero in binary16 but can round to zero in a coarser format. There is no
minimum-subnormal fallback for this constant: callers must supply a positive, representable
normalization epsilon when the default is too small. Validation of a rational model configuration
checks positivity before scalar conversion; it does not establish positivity after rounding.

Kernel capsules record one numerical choice: the reduction order (`NumericalPolicy.reduction`,
one of `fixedLeft`, `implementationDefined`, or `notApplicable`). It is audited contract data, not
proof evidence, and it has exactly one consumer: `requireFixedLeftReduction` in
`NN/Proofs/RuntimeApprox/Graph/NumericalCertificate/Contracts.lean` reads it as an `Except`
precondition and rejects any node whose capsule does not advertise `fixedLeft`. Rounding mode,
subnormal handling, and multiply-add contraction are not recorded; a certificate that depends on
them must state the assumption itself. Portable reference accumulations advertise the fixed left
fold. LibTorch matrix products, convolutions, normalizations, pooling operations,
FFT/FNO paths, scans, and attention advertise implementation-dependent reductions. Consequently,
the fixed-left graph certificate refuses to reuse its transfer for those accelerated paths. A
theorem about such a path needs a backend-specific schedule or an applicable order-independent
enclosure from `NN/Proofs/RuntimeApprox/Reductions/IEEE32.lean`. The latter still requires its
operation, finiteness, and local-error hypotheses; it does not cover an arbitrary ATen algorithm.

For a checked replay, interval validity proves that each endpoint is finite and ordered. The replay
also checks every computed entry for finiteness, and `executeIEEE32` returns a
`RangeCheckedExecution` only when every FloatLib binary32 node value lies inside its checked range. That
is the whole of what the checker proves. It does not prove that the exact-real denotation of the
graph lies in those ranges. That statement is the structure `ProvedRealEnclosure` in
`NN/Proofs/RuntimeApprox/Graph/NumericalCertificate/Certificate.lean`, whose fields require a real
payload and input, the complete real node trace, a proof that the trace is the graph's real
denotation, and a pointwise proof that each real value lies in its checked interval. The caller must
supply this structure; nothing in the repository constructs one today. Given both, the theorem
`RangeCheckedExecution.error_trace` yields the pointwise interval-width error bound for every
intermediate. This is a theorem about the FloatLib binary32 replay. Transporting it to Lean runtime
`Float32`, CUDA, LibTorch, cuBLAS, or cuDNN still requires the agreement recorded by that backend's
capsule.

The proof-bearing `RevGraph` path has rounded forward and VJP theorems and erases to executable
autograd `GraphData`. One optimizer contract carries those gradient bounds through SGD,
momentum-SGD, and AdamW; AdamW supplies additional positivity and denominator-margin evidence at
each step. These are Lean theorems about the `NF` rounded-real scalar model.

Two different lowerings must not be confused here. The lowering of the proof-layer `Graph` to the
executable tape (`lowerGraphToTape` in `NN/Proofs/Autograd/Runtime/Link/`) has both forward and
backward correspondence theorems, ending in `backwardDenseAll_lowerGraphToTape_adjoint_fderiv`
over `ℝ`. The lowering of the canonical `NN.IR.Graph` to the forward-only `ForwardGraph`
(`lowerToForwardGraph` in `NN/Runtime/Autograd/IRExec/`) proves forward semantic preservation only:
`denoteAll_eq_of_lowerToForwardGraph` needs just the `NoRawLog` side condition and covers concat,
rank four and higher matmul, and batched linear, but it is not an autograd theorem and must not be
cited as a backward-certificate theorem.

Use the float layers as follows:

| Claim | Layer to cite |
| --- | --- |
| executable configured binary behavior inside Lean | FloatLib's `ExecFloat.Binary` and `BinaryInterchange` refinement theorems |
| TorchLean binary32 certificate representation | FloatLib's `ExecFloat.Binary 8 23` |
| finite rounded-real float32 error bound | `NN/Floats/FP32` |
| precision-parametric rounding theorem | `FloatLib.Floats.Formats.Flocq` |
| endpoint interval enclosure | FloatLib's `BinaryInterchange.IntervalSemantics` and `NN/Floats/Interval` adapters |
| external high-precision enclosure evidence | `NN/Floats/Arb` plus the oracle boundary |
| meaning of Lean `Float32` core arithmetic | Lean's `Float32.Model` definitions |
| native logical import/export and arithmetic correspondence | FloatLib's `IEEE754.Native` proof modules, with each operation's stated hypotheses |
| compiled `Float`/`Float32`, CUDA, or LibTorch behavior | provider bridge or boundary statement |

## External Numeric Oracles

- CROWN/Lyapunov certificate generation is an external evidence producer. Generated Lean modules
  prove their numeric sign margins, while the final stability theorem requires an explicit
  `LyapunovCert.ValidFor` proof connecting those numbers to the named Lean functions. A checked
  graph workflow can establish that predicate; a Python-only workflow must state it as a local
  assumption rather than inheriting a repository-wide axiom.
- The Arb / `python-flint` integration under `NN/Floats/Arb/` is an external subprocess backend. It
  can produce high-quality interval evidence, but an Arb response is still an oracle result unless
  the relevant certificate is independently checked in Lean.
- PyTorch import/export scripts and training helpers are external producers of weights, examples,
  or JSON artifacts. TorchLean can parse and replay those artifacts, but PyTorch training itself is
  not part of Lean's trusted kernel. The operator tag wire format used by that exchange has a
  round-trip check in Lean (`parse_op_tag`, `parse_op_kind` in `NN/Runtime/PyTorch/Wire.lean`); the
  Python side that emits it remains a producer.
- The optional Julia wrapper `NN/Runtime/External/Julia.lean` follows the same pattern. It resolves
  `TORCHLEAN_JULIA` when set, otherwise falls back to `julia` on `PATH`, and does not require Julia
  at compile time. It supports "untrusted producer, Lean checker" workflows such as the
  piecewise-polynomial spline certificate workflow (producer scripts under
  `scripts/verification/splines/`, bundled fixtures under `NN/Examples/Verification/Splines/`).
- A Julia-produced spline or PINN artifact is trusted only after a Lean checker validates the small
  certificate data it needs: for example cell domains, polynomial coefficients, interval bounds, and
  claimed residual inequalities. Lean does not trust Julia's fitting process, optimizer, GPU use, or
  floating-point arithmetic merely because the subprocess returned successfully.
