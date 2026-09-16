import VersoManual
import NN.Tensor
import NN.Proofs.RuntimeApprox.Core.SpecApprox
import NN.Proofs.RuntimeApprox.Core.Tolerance
import NN.Proofs.RuntimeApprox.Graph.NumericalCertificate
import NN.Proofs.RuntimeApprox.NF.Attention
import NN.Proofs.RuntimeApprox.NF.Convolution
import NN.Proofs.RuntimeApprox.NF.EndToEnd
import NN.Proofs.RuntimeApprox.NF.Normalization
import NN.Proofs.RuntimeApprox.NF.Optimizers
import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid
import NN.Proofs.RuntimeApprox.NF.ReductionOps
import NN.Proofs.RuntimeApprox.NF.SoftmaxAxis
import NN.Proofs.RuntimeApprox.Optimizer
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- Everything quoted in this chapter lives under a root `Proofs.RuntimeApprox` namespace, with the
-- rounded-real operator lemmas one level further down in `NFBackend`. Verso keeps displayed code
-- narrow enough to read beside the prose, so the namespaces are opened here to let each `#check`
-- fit on one line. The scalar namespaces identify the imported FloatLib theory.
open Proofs.RuntimeApprox
open Proofs.RuntimeApprox.Attention
open Proofs.RuntimeApprox.NFBackend
open Proofs.RuntimeApprox.NFBackend.Optimizer
open Proofs.RuntimeApprox.NumericalCertificate
open Proofs.RuntimeApprox.Optimizer
open FloatLib.Numerics
open FloatLib.Floats.Formats.Flocq

-- A few of the signatures below print wider than this file's 100-column limit, so their
-- `leanOutput` blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page
-- still shows each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

#doc (Manual) "Runtime Approximation" =>
%%%
tag := "runtime-approximation"
%%%

Suppose a graph has an ideal output `y_spec` and a rounded output `y_run`. To compare them, we
first interpret the rounded scalar as a specification value using `toSpec`. An approximation
statement then bounds the difference by an explicit budget `ε`:

$$`\operatorname{toSpec}(y_{\mathrm{run}})\approx_\varepsilon y_{\mathrm{spec}}`

For tensors of the same shape, a single budget can bound every coordinate. Writing this with the
maximum norm gives

$$`\left\|\operatorname{toSpec}(Y_{\mathrm{run}})-Y_{\mathrm{spec}}\right\|_\infty
\le \varepsilon.`

# Real And Floating-Point Semantics

A real-valued proof does not account for the arithmetic used to evaluate its formulas. Rounding,
cancellation, overflow, reduction order, fused kernels, and domain guards can affect an execution;
{Informal.citet goldberg1991}[] explains these effects in floating-point arithmetic. A softmax
implementation, for example, combines max-subtraction, exponent approximations, a finite sum, and
division. Its error depends on how those operations are carried out and how their errors interact.

To transfer a real-arithmetic guarantee to execution, we need to:

1. prove the claim over the reals or over the spec;
2. prove an approximation relation between runtime tensors and spec tensors;
3. inflate margins, enclosures, or certificates by the approximation budget;
4. leave native hardware assumptions explicit when they are outside Lean.

This matches standard numerical analysis practice; see Higham's *Accuracy and Stability of
Numerical Algorithms* (https://epubs.siam.org/doi/book/10.1137/1.9780898718027) for the classical
background. It also matches the caution needed in neural network verification, where a tiny
rounding gap can matter if the verified margin is tiny.

There are two sources of discrepancy to follow through this process. A stored weight may already
differ from the intended real weight before evaluation starts. Even when every input agrees
exactly, arithmetic can introduce new error at each node. A useful local theorem accepts the
first kind of error as an input and adds the second kind to its conclusion. This is why a graph
bound needs both initial approximation evidence and rules for the operations; proving that the
initial tensors are close would leave the rounding inside the graph unaccounted for.

# The Core Relation: Spec Tensors And Tolerances

The small vocabulary is defined by
{src "NN/Proofs/RuntimeApprox/Core.lean"}[runtime approximation core],
{src "NN/Proofs/RuntimeApprox/Core/SpecApprox.lean"}[spec approximation], and
{src "NN/Proofs/RuntimeApprox/Core/Tolerance.lean"}[tolerances].

The spec approximation layer defines the basic tensor relation. Given a conversion
`toSpec : α → SpecScalar`, a runtime tensor approximates a spec tensor when every component is close
after conversion to the spec scalar. The key names are:

- `tensorToSpec`: pointwise conversion from runtime tensor values to spec scalars.
- `linfNorm`: max style tensor norm used for error statements.
- `approxWith`: absolute tensor approximation using an explicit error tensor.
- `approxWithTol`: approximation using a tolerance object.
- `approxTensorWithTol`: packaged tensor tolerance relation.
- `Witness`: a small record for carrying a runtime value and its error evidence.

The tolerance API defines `ApproxTol`, with absolute, relative, and slack components.
`ApproxTol.absOnly`, `approxBound`, and `approxR` let later proofs move between simple absolute
error statements and more scale aware claims.

For a single scalar, the scale-aware shape is:

$$`|r-s|
\le
\varepsilon_{\mathrm{slack}}\bigl(\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{rel}}\max(|r|,|s|)\bigr).`

The choice is practical. An absolute tolerance of `1e-6` may be meaningful near zero and
irrelevant near `1e9`. A relative tolerance captures "small compared with the scale of the value."
TorchLean uses both styles, but it makes the choice explicit instead of burying it in test
thresholds.

Since `approxBound` is a real function, the difference between the two readings can simply be
computed. Read `1e-6` as an absolute budget and it stays `1e-6` no matter how large the values are:

```lean
-- An absolute tolerance is independent of the operands’
-- magnitude.
example :
    approxBound (ApproxTol.ofReal 1e-6 0 1) 1e9 1e9
      = 1e-6 := by
  norm_num [approxBound, ApproxTol.ofReal,
    Real.toNNReal]
```

Read the same `1e-6` as a relative budget at the same scale and it is a thousand:

```lean
-- A relative tolerance scales with the values being
-- compared.
example :
    approxBound (ApproxTol.ofReal 0 1e-6 1) 1e9 1e9
      = 1e3 := by
  norm_num [approxBound, ApproxTol.ofReal,
    Real.toNNReal]
```

The relative budget grows with the operands: at this scale it permits an absolute difference of
one thousand. The absolute budget remains one millionth. Both use the literal `1e-6`, so a claim
must identify which tolerance field it supplies.

Near a zero reference value, a purely relative comparison deserves particular care. With zero
absolute tolerance and slack one, comparing a nonzero value to zero would require its magnitude
to be at most `rel` times that same magnitude. For a relative tolerance below one, this cannot
hold. The absolute component supplies room for small discrepancies near zero, while the relative
component scales the allowance for large values. These are different jobs, so an application
should choose them from the claim it needs rather than treat the two fields as interchangeable.

## Tensor Approximation Contracts

The relation defines what it means for two tensors to be close. A theorem using that relation
can quantify over a whole family of inputs, or can concern just one pair of tensors. The two
signatures below expose the scalar interpretation and the proposition to be proved:

```lean (name := coreVocab)
-- Separate pointwise interpretation from the proposition
-- asserting approximation.
#check @tensorToSpec
#check @approxTensorWithTol
```
```leanOutput coreVocab (whitespace := lax)
@tensorToSpec : {α : Type} →
  [inst : TorchLean.Storage α] → {s : Spec.Shape} → (α → Spec.SpecScalar) → TorchLean.Tensor α
    s → Spec.SpecTensor s
```
```leanOutput coreVocab (whitespace := lax)
@approxTensorWithTol : {α : Type} →
  [inst : TorchLean.Storage α] →
    {s : Spec.Shape} → (α → Spec.SpecScalar) → Spec.SpecTensor s → TorchLean.Tensor α s →
      ApproxTol → Prop
```

`tensorToSpec` applies the supplied function `α → Spec.SpecScalar` coordinatewise. It defines
an interpretation of the tensor; it does not itself prove that a runtime operation agrees with a
specification operation. Those operation-level facts are the hypotheses that later graph theorems
compose.

The shape of `approxTensorWithTol` is the contract: a scalar interpretation `toSpec`, a spec tensor,
a runtime tensor, a tolerance, and a `Prop`. Nothing about hardware appears in it, which is exactly
why the interpretation has to be named.

The tolerance determines the following real-valued budget:

```lean (name := coreBound)
-- The budget is a real-valued expression, not evidence that
-- a comparison passes.
#check @approxBound
```

```leanOutput coreBound
approxBound : ApproxTol → ℝ → ℝ → ℝ
```

The proposition says every coordinate of `tensorToSpec toSpec runtime` is close to the matching
coordinate of `spec` under `eps`. That makes the trusted boundary easy to locate. If `toSpec`
interprets an executable rounded-real model, the theorem is about that rounded-real model. If the
actual deployment path is CUDA, cuBLAS, PyTorch, or a fused native kernel, a separate agreement
statement is needed before the theorem says anything about that path.

Thus the scalar interpretation and the actual execution path must match. Numerical tests can
compare selected executions, while a deployment theorem needs an agreement result with the scope
required by its final claim.

The shared shape parameter in the displayed signatures ensures that the comparison has matching
coordinates. It does not decide which coordinates are supposed to correspond semantically.
For example, comparing two parameter tensors after a permutation could satisfy the shape
constraint while comparing the wrong weights. The caller must use the same parameter ordering
and scalar interpretation as the computation being analyzed. Once those choices are fixed, the
coordinatewise relation gives later operators a precise premise to consume.

## Comparison With PyTorch Tolerances

A related comparison appears in PyTorch's `torch.allclose` {Informal.citep pytorch2019}[]. For
finite entries, its condition with `rtol` and `atol` is the elementwise inequality

$$`|a-b|\le \mathrm{atol}+\mathrm{rtol}\,|b|`

In `approxBound`, `abs` plays the role of
`atol` and `rel` that of `rtol`. TorchLean scales by
$`\max(|a|,|b|)` rather than by the second argument alone, and it carries a `slack` factor for
proofs that need to inflate a budget without rewriting it.

Scaling by the maximum makes the relation symmetric, which matters because a tolerance theorem gets
used in both directions. At slack one, it also makes TorchLean's budget the more generous of the two
at identical absolute and relative tolerances:

```lean
-- Nonnegative relative tolerance preserves the inequality
-- when the scale is enlarged.
example (atol rtol x y : ℝ) (h : 0 ≤ rtol) :
    atol + rtol * |y|
      ≤ atol + rtol * max |x| |y| := by
  gcongr
  exact le_max_right _ _
```

The formulas differ in symmetry and slack. Quantification is a separate issue: `allclose` checks
the two supplied arrays, while a theorem about `approxTensorWithTol` may range over all inputs,
weights, or shapes satisfying its hypotheses. Merely writing the relation as a `Prop` does not make
a claim universal; its quantifiers determine the scope.

# Forward Graph Approximation

The forward graph theorem is in
{src "NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean"}[NN.Proofs.RuntimeApprox.Graph.ForwardApprox
API]. It mirrors the structure of the autograd tape proof, but the invariant is approximation rather
than derivative soundness.

The main objects are:

- `EList`: list of scalar error budgets indexed by shape.
- `approxCtx`: approximation relation for a whole context.
- `FwdNode`: a node with spec forward, runtime forward, bound, and local soundness.
- `FwdGraph`: a snoc list graph of forward nodes that carry approximation evidence.
- `FwdGraph.eval_approx`: theorem for the whole forward graph.

The local theorem on a `FwdNode` says: if each runtime input approximates the corresponding spec
input, then the runtime output approximates the spec output within this node's bound.

`FwdGraph.eval_approx` composes those local statements. When a graph appends a node, the proof uses
the node's local `sound` theorem, appends the new bound to the error context, and continues. It uses
the same architecture as the autograd soundness theorem: local correctness first, then a
global induction over the graph.

A tiny example is multiplication. The spec value is the real product, while the runtime value is the
rounded product computed by the chosen scalar model.

If $`x_{\mathrm{run}}` is within $`\varepsilon_x` of $`x_{\mathrm{real}}`, and $`y_{\mathrm{run}}`
is within $`\varepsilon_y` of $`y_{\mathrm{real}}`, the local multiplication lemma supplies a bound
for $`z_{\mathrm{run}}` versus $`z_{\mathrm{real}}`. The graph theorem then lets that
new bound feed the next node.

Write the interpreted runtime inputs as $`x+dx` and $`y+dy`,

$$`
(x+dx)(y+dy)-xy=x\,dy+y\,dx+dx\,dy.
`

The first two terms describe how each input error is scaled by the other exact operand. The
third accounts for the interaction of the two errors. The identity can be checked directly:

```lean
-- Keep the product of the two incoming errors; it is part
-- of finite error propagation.
example (x y dx dy : ℝ) :
    (x + dx) * (y + dy) - x * y
      = x * dy + y * dx + dx * dy := by
  ring
```

Hence input uncertainty alone contributes at most

$$`|x|\,\varepsilon_y+|y|\,\varepsilon_x
  +\varepsilon_x\varepsilon_y.`

The triangle inequality bounds the sum by the sum of the three magnitudes. Applying
$`|ab|=|a||b|` and the two input-error bounds gives the estimate:

```lean
example (x y dx dy εx εy : ℝ)
    (hx : |dx| ≤ εx) (hy : |dy| ≤ εy) :
    |(x + dx) * (y + dy) - x * y|
      ≤ |x| * εy + |y| * εx + εx * εy := by
  -- The last `gcongr` needs `εx` on the correct side
  -- of zero, and the hypothesis on `dx` already says so.
  have hεx : 0 ≤ εx := le_trans (abs_nonneg dx) hx
  have h : (x + dx) * (y + dy) - x * y
      = x * dy + y * dx + dx * dy := by ring
  rw [h]
  calc |x * dy + y * dx + dx * dy|
      ≤ |x * dy + y * dx| + |dx * dy| :=
        abs_add_le _ _
    _ ≤ |x * dy| + |y * dx| + |dx * dy| := by
        gcongr
        exact abs_add_le _ _
    _ = |x| * |dy| + |y| * |dx| + |dx| * |dy| := by
        rw [abs_mul, abs_mul, abs_mul]
    _ ≤ |x| * εy + |y| * εx + εx * εy := by
        gcongr
```

If $`x=2`, $`y=3`, $`\varepsilon_x=0.01`, and $`\varepsilon_y=0.02`, that bound is a number:

```lean
-- Evaluate the perturbation budget before adding the
-- multiplication’s rounding term.
example :
    |(2:ℝ)| * 0.02 + |(3:ℝ)| * 0.01 + 0.01 * 0.02
      = 0.0702 := by
  norm_num
```

The local rule then adds the rounding error of the multiplication itself. The number is not meant
as a universal tolerance; it shows why the graph carries operand scale and incoming error rather
than attaching one unexplained epsilon to every multiplication.

The product term in the multiplication estimate makes it a finite-perturbation bound rather than
only a first-order sensitivity calculation. Dropping that term would be justified for a
derivative limit, but would understate this particular finite error budget. The graph carries
these bounds for all available values, not just the newest output. If a later node reuses an
earlier activation, its local theorem retrieves the bound attached to that activation. The
context invariant therefore follows the same data dependencies as the computation, including
values that bypass several intervening nodes.

# Backward Graph Approximation

Backward approximation is developed in the
{src "NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean"}[backward approximation API].
This file is the runtime approximation analogue of reverse mode AD.

The key objects are:

- `RevNode`: a forward node plus spec/runtime VJP functions and a VJP error transformer.
- `RevGraph`: reverse graph with local approximation evidence.
- `RevGraph.eval_approx`: forward approximation inherited from the forward graph.
- `RevGraph.backprop_approx`: theorem for reverse accumulation across the whole graph.

The theorem `RevGraph.backprop_approx` states that the runtime reverse pass approximates the spec
reverse pass, provided the input context, forward tape, and seed cotangents are appropriately
related. Addition during gradient accumulation is not treated as automatic; the theorem takes an
explicit `addBound` and `addSound` describing how accumulation affects error.

Backward passes are often dominated by sums: cotangents from fanout, reductions, convolution
gradients, and parameter gradient accumulation. Floating point addition is not associative, and
accumulation order matters. By making addition soundness an explicit parameter, the theorem states
the arithmetic model being used.

For a shared parameter, two individually accurate incoming cotangents still need an accurate
addition before their sum becomes its gradient. Their existing errors add, and the rounded
addition introduces another contribution. Forward-tape error also enters the local VJP: a
multiplication backward rule uses a saved operand, whose rounded value may already differ from
the real one. The seed has its own budget because changing the seed changes the linear
functional being differentiated. These three sources explain why the backward theorem asks for
more than approximation of the final forward output.

# Autograd Algebra Correspondence

The bridge file
{src "NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean"}[
NN.Proofs.RuntimeApprox.Graph.LinkAutogradAlgebra API] connects the runtime approximation graph
shape back to the autograd algebra graph.

This bridge is structural. The runtime approximation layer uses the same
shape-indexed tensor-context idea that the typed API exposes as `TensorPack`, so the conversion is
not a semantic reinterpretation. The file defines `toNodeData` and `toGraphData`, then proves:

- `evalRuntime_of_toGraphData`;
- `backpropRuntime_of_toGraphData`.

In words, the runtime approximation graph can be viewed as an autograd algebra graph by forgetting
the approximation evidence and keeping the same forward/VJP structure. That means the two layers
compose cleanly:

autograd algebra: the reverse pass is the correct ideal VJP; runtime approximation: the executable
reverse pass stays close to that ideal VJP.

To compose these results, the spec reverse pass must be the one identified with the derivative
by the autograd proof. Approximation of an arbitrary supplied VJP would establish numerical
agreement without establishing derivative correctness.

The erasure equalities are useful when connecting those premises. They let a proof about a
`RevGraph` refer to the forward and backward functions executed by its erased `GraphData`,
without reproving the traversal. They preserve the supplied scalar functions as well as the
graph structure. An application can therefore combine a local derivative certificate and a local
rounding certificate for the same rule. If the two certificates concern different guards,
parameter orders, or selected slopes, the erasure equality cannot make those rules coincide.

# Numerical Traces For The Canonical IR

The proof-relevant `FwdGraph` and `RevGraph` explain how local approximation theorems compose. Model
export and kernel selection, however, use the canonical op-tagged `NN.IR.Graph`. TorchLean connects
that graph directly to executable binary32 through
{src "NN/Proofs/RuntimeApprox/Graph/NumericalCertificate.lean"}[the graph numerical certificate
checker]. It does not introduce a second deployment graph or a second interval type.

A certificate contains source enclosures, one derived range per IR node, the name of the range
registry, and the existing backend execution audit. A successful `check` stores the graph inside
`RegistryCheckedCertificate`; later replay cannot silently substitute a different graph or rule set.
Checking performs three independent executable validations:

1. validate that source and derived endpoints are finite and ordered;
2. reconstruct every supported range transfer from the graph;
3. re-run kernel selection and compare the selected capsules and numerical policies.

`GraphRangeRegistry` dispatches by primitive operation, not model family. The built-in transfers
cover source and shape-only nodes, pooling, arithmetic, inverse, ReLU, absolute value, directed
square root with a checked nonnegative domain, fixed-left reductions, matrix multiplication, MSE,
LayerNorm, softmax, sigmoid, tanh, sine, and cosine. Exponential is currently unsupported.
An unsupported operation fails at its node id; it is not replaced by an uninformative whole
interval.

```lean (name := certEntries)
-- Inspect generation, executable coverage, replay, and
-- semantic evidence separately.
#check @generateChecked
#check @GraphRangeRegistry
#check @numericalCoverage
#check @executeIEEE32
#check @ProvedRealEnclosure
#check @RangeCheckedExecution.error_trace
#check @tensor_error_le_width_of_check
#check @execution_error_trace_of_check
```

```leanOutput certEntries (whitespace := lax)
generateChecked : NN.Backend.BackendProfile → NN.IR.Graph →
  Array SourceRange → Except String RegistryCheckedCertificate
```

```leanOutput certEntries
ProvedRealEnclosure : RegistryCheckedCertificate → Type
```

The generator's `Except` result preserves a diagnostic on failure, such as a node id whose
operation has no registered rule. That message helps locate unsupported operations; it provides no
semantic evidence about the supported ones.

`ProvedRealEnclosure` is indexed by `RegistryCheckedCertificate`, so its evidence refers to the
same artifact used by replay. This prevents accidental substitution of evidence for another
certificate. The fields still have to prove the required real-semantic facts.

The other checked names describe the stages between these two types. `numericalCoverage`
reports whether each operation has a registry entry; it can locate a missing transfer before
range propagation begins. A registered transfer can still reject the actual domain it receives.
`executeIEEE32` then needs concrete payloads and inputs to produce a checked execution.
`tensor_error_le_width_of_check` supplies the coordinatewise interval-width argument, and
`execution_error_trace_of_check` applies that argument to related arrays of node values.
`RangeCheckedExecution.error_trace` packages it for the certificate's real enclosure and replay.
Thus operation coverage, successful evaluation, and a proved error trace answer successive,
different questions.

`GraphRangeContract.derive` is an executable range transformer, not a soundness theorem.
`generateChecked` and `check` establish that the stored trace is exactly the trace reconstructed by
the selected registry and kernel plan; they do not establish that every reconstructed interval
encloses the graph's real denotation.

`executeIEEE32` evaluates the same `NN.IR.Graph` with FloatLib's configured binary32 through
the FloatLib binary32 context and checks every intermediate value against the stored ranges,
rejecting NaN and infinity. This is a
reference replay, not an agreement theorem for a high-throughput backend.
`ProvedRealEnclosure` carries the separate semantic evidence: its fields require both the real
denotation equality and a pointwise real enclosure proof. When that evidence is supplied,
`RangeCheckedExecution.error_trace` combines the real enclosure with the successful IEEE replay to
prove a pointwise error trace whose bound is the interval width. For each coordinate, both the
real value and the interpreted IEEE value lie between the same two endpoints; their distance is
therefore at most the upper endpoint minus the lower endpoint. This argument requires enclosure of
both values. Checking the rounded replay alone leaves the real half of the argument unproved.

Reduction order is read from the selected capsule. The portable reference capsules advertise the
fixed left fold used by the canonical tensor semantics. Native CUDA and LibTorch accumulations are
marked implementation-defined, so a fixed-left certificate cannot accidentally certify a cuBLAS,
cuDNN, fused-attention, or parallel-reduction schedule. Those paths require the order-independent
reduction bounds described in the floating-point chapter or a stronger backend-specific contract.

The local interval lemmas for arithmetic and selected nonlinear operations follow the inclusion
principle of IEEE 1788-2015. The current registry does not yet compose those lemmas into a theorem
that every accepted graph trace encloses exact graph semantics. The separation between local
rounding facts and a composed global error follows the standard treatment in Higham,
*Accuracy and Stability of Numerical Algorithms*, 2nd edition.

The canonical `NN.IR.Graph` lowering currently proves forward semantic preservation only. Its IR
nodes do not yet carry proved VJPs, so this certificate should not be described as a
canonical-IR backward certificate. Backward numerical theorems use the proof-bearing `RevGraph`
path below, which erases to executable autograd `GraphData` without discarding its VJP rules. An
autograd-capable lowering from canonical IR would need an additional correspondence theorem.

An interval-width bound is deliberately coarse when the two executions share information that
the interval forgets. If both values lie in `[2,3]`, their distance is at most one, even when they
are actually identical. Narrowing the interval improves this estimate only if both enclosure
premises remain valid. This gives the range producer a concrete objective: keep intervals tight
enough for the downstream margin while preserving the real inclusion proof and successful
rounded replay. Merely shrinking a stored endpoint would invalidate one of those checks.

# MLP Numerical Certificate Example

The executable example
{src "NN/Examples/DeepDives/Floats/GraphNumericalCertificate.lean"}[GraphNumericalCertificate.lean]
ends with a two-layer MLP rather than a single isolated operator. Run it from the repository root:

```terminal
# Run generation and replay for the primitive-composed MLP
# certificate.
lake exe torchlean numerical_certificate
```

The model is a matrix pipeline with shapes that remain visible in the IR:

```
-- Each matrix and bias stage contributes its own node
-- range.
input [1,2]
  -> matmul [2,3]
  -> add bias [1,3]
  -> ReLU
  -> matmul [3,1]
  -> add bias [1,1]
```

The report captured before the FloatLib migration shows generation, replay, and one deliberately
corrupted artifact. It is a record of that run, not a validation of the migrated executable:

```terminal +output
TorchLean numerical runtime certificate
  ok  base certificate
  ok  base IEEE replay
  ok  tampered range rejected
  ok  two-layer MLP certificate
  ok  two-layer MLP IEEE replay
All numerical certificate checks passed.
```

The tampered-range check replaces the addition interval with `[0,0]`. The checker regenerates
the range trace and rejects the mismatch. This establishes that this corrupted artifact is refused;
it does not establish soundness of every registered transfer.

`mlpCertificate` checks that all ten graph nodes have a registered numerical rule. It derives every
range, selects the CPU capsules, and stores the graph, registry identity, source assumptions,
ranges, and backend audit in one artifact. `mlpReplay` then supplies concrete weights, biases, and
input values, executes the stored graph with FloatLib binary32, and checks every intermediate
tensor.
The same file demonstrates rejection of a tampered range. These Boolean and `Except` checks are
useful regression evidence; the example does not construct a `ProvedRealEnclosure`, so it is not
by itself a proof that the MLP's exact real execution is enclosed.

There is no MLP-specific branch in this process. The checker sees input, constant, matrix
multiplication, addition, and ReLU nodes. Other architectures can use the same walk when their
primitive operations are covered. New primitives extend the executable registry with a
`GraphRangeContract`; a semantic certificate additionally needs a theorem connecting that
contract's derived interval to the operation's real semantics.

The
[complete numerical-runtime
walkthrough](https://lean-dojo.github.io/TorchLean/examples/numerical-runtime/) shows the model
definitions, the five replay stages, the backend-capsule audit, and the handoff to backward and
optimizer bounds. It also states the current lowering boundary explicitly: canonical IR has checked
forward replay, while backward and optimizer composition currently begins from a proof-bearing
`RevGraph`.

Read the five successful lines as checks of two concrete artifacts and one rejection case.
The base and MLP generation lines concern reconstruction of their ranges and execution plans;
their replay lines concern the supplied values at every intermediate node. The tampering line
confirms that changing a stored range is detected. These observations exercise different parts
of the checker, which is why the example keeps them separate. None of the lines supplies the
real denotation and enclosure fields required by the proof-bearing record above.

# NF Operations: Rounded Real Arithmetic

The largest collection of local rules is
{src "NN/Proofs/RuntimeApprox/NF/Ops.lean"}[NN.Proofs.RuntimeApprox.NF.Ops API]. These rules use
FloatLib's `Floats.Formats.Flocq.NF`, a noncomputable rounded-real model. Its radix, exponent
function, and valid rounder determine the representable values and arithmetic. This is the
rounded-real style used by {Informal.citet flocq2011}[] in Rocq. Expressing an operation as
rounding a real result makes its error available directly to real analysis. A bit-level model
can support such reasoning, but first needs decoding and a theorem relating the decoded
operation to rounding.

The local tensor, graph, and optimizer rules compose those imported scalar facts. Selecting
`ExecFloat.Binary e f` for a program does not instantiate an NF error theorem automatically:
the proof must connect the chosen bit format to the same rounding model and account for finite
range and exceptional values. Likewise, an NF exponential denotes rounding the real exponential;
it is not a theorem about the accuracy of a configured software approximation to that function.

The file includes scalar and tensor approximation lemmas for common operations:

- arithmetic: `approx_add_nf`, `approx_sub_nf`, `approx_mul_nf`, `approx_div_nf_of_pos_lb`;
- unary functions: `approx_exp_nf`, `approx_tanh_nf`, `approx_abs_nf`, `approx_neg_nf`;
- guarded operations: `safeLog`, `safeDiv`, `safe_log`;
- tensor rules: `approxTensor_add_spec`, `approxTensor_mul_spec`, `approxTensor_exp_spec`,
  `approxTensor_relu_spec`;
- graph nodes: `addNode`, `mulNode`, `expNode`, `reluNode`, `safeDivNode`, `softmaxNode`, `sumNode`.

Several of these lemmas make the numerical analysis tradeoff visible. Division requires a positive
lower bound on the exact denominator that survives rounding (`approx_div_nf_of_pos_lb`) or a
guarded form (`safeDiv`). The division budget has a name, `divPosErrorBound`: the numerator error
scaled by the effective margin, the denominator error scaled by the squared margin, and half an ULP
of the quotient. The sigmoid, logistic, and mean bounds are built on that one definition.
`reciprocal_sigmoid_bound_scalar_le_one` bounds the budget for the rounded sequence
$`1/(1+\exp(-x))` by one under explicit small-error hypotheses.
`mean_row_bound_of_exact` gives the closed form of the row-mean budget when the row count is exactly
representable.

The public sigmoid uses $`1/(1+\exp(-x))` on positive inputs and $`\exp(x)/(1+\exp(x))`
otherwise. Both formulas describe the same real function, but their rounded arithmetic differs:
the second sequence also rounds the numerator exponential. `approx_sigmoid_nf` uses the budget
for the branch selected by the rounded input. Its proof allows the real input and rounded input
to lie on different sides of zero, because each branch approximates the same real sigmoid.

Square root and log need domain protection. Exponential uses a mean value bound.
Multiplication propagates both input errors and a product term. These are the exact places where a
proof over real numbers would be too optimistic if copied directly onto a float implementation.

```lean (name := nfOps)
-- The denominator margin links the bound expression to a
-- valid division theorem.
#check @divPosErrorBound
#check @approx_div_nf_of_pos_lb
#check @reciprocal_sigmoid_bound_scalar_le_one
#check @approx_sigmoid_nf
#check @mean_row_bound_of_exact
```

```leanOutput nfOps (whitespace := lax)
@divPosErrorBound : {β : Radix} →
  {fexp : ℤ → ℤ} → [ValidExp fexp] → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
```

The displayed `divPosErrorBound` signature makes the budget's dependencies explicit.

The five real arguments are the exact denominator lower bound `η`, the input-error budgets
`epsx` and `epsy`, and the interpreted runtime operands `xhat` and `yhat`. The associated theorem
requires `epsy < η`: denominator error leaves an effective separation `η - epsy` from zero.
The budget scales the numerator error by the reciprocal of that separation and the denominator
error by its squared reciprocal, then adds rounding of the quotient.

As the surviving separation shrinks, the bound grows. A caller can compare this explicit expression
with a desired margin, but it is a noncomputable real-valued bound, not an executable estimator.
Its ULP terms also need not be smooth.

For example, an exact denominator lower bound of two and denominator error at most one quarter
leave a rounded denominator at least seven quarters. That surviving margin is what permits the
reciprocal estimate. Letting the error approach two would remove this argument even if one
particular rounded denominator happened to remain positive. The theorem uses a uniform bound
that must justify every value admitted by its premises, rather than inspecting only a favorable
sample. The squared reciprocal in the denominator-error term makes this loss of separation
especially costly.

The remaining names in `nfOps` specialize this reasoning in different ways. The reciprocal
sigmoid budget theorem bounds the old evaluation sequence under its stated small-error
conditions. The stable sigmoid theorem instead follows the branch of the public implementation,
including the exponential used as a numerator in the negative branch. The exact-count mean
lemma removes uncertainty about representing the divisor; it still accounts for the rounded
sum and final division. A closed form for one of these budgets is useful only with the
hypotheses belonging to that particular sequence.

A safe division example has three pieces: the mathematical value is a guarded division, the runtime
value is computed by `safeDivR eps xR yR`, and the theorem states that the runtime value
approximates the guarded spec value under the declared tolerance.

The theorem covers the guarded function, including its declared behavior at $`y=0`. To use it
for ordinary division, a caller must additionally show that neither execution activates the guard.

## Error Bounds For Reductions And Softmax

Reductions and normalization layers create coupled
error terms because many inputs flow into one output. TorchLean has explicit reduction approximation
lemmas in
{src "NN/Proofs/RuntimeApprox/NF/ReductionOps.lean"}[NN.Proofs.RuntimeApprox.NF.ReductionOps]:

```lean (name := reductions)
-- Row and column reductions follow different index
-- sequences and expose different budgets.
#check @approxTensor_reduce_sum_rows
#check @approxTensor_reduce_mean_rows
#check @approxTensor_reduce_sum_columns
```
```leanOutput reductions (whitespace := lax)
@approxTensor_reduce_sum_rows : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd] {m n : ℕ}
  {xS : Spec.SpecTensor [m, n]} {xR : TorchLean.Tensor (NF β fexp rnd) [m, n]}
    {eps : ℝ},
  approxTensor toSpec xS xR eps →
    ∀ (hRed : Spec.Shape.NonemptyAxis 1 [m, n]),
      approxTensor toSpec (TorchLean.Tensor.reduceSum 1 xS hRed)
        (TorchLean.Tensor.reduceSum 1 xR hRed)
        (linfNorm (sumRowBoundVec eps xR))
```
```leanOutput reductions (whitespace := lax)
@approxTensor_reduce_mean_rows : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd] {m n : ℕ}
  {xS : Spec.SpecTensor [m, n]} {xR : TorchLean.Tensor (NF β fexp rnd) [m, n]}
    {eps : ℝ},
  approxTensor toSpec xS xR eps →
    ∀ (hRed : Spec.Shape.NonemptyAxis 1 [m, n]),
      approxTensor toSpec (TorchLean.Tensor.reduceMean 1 xS hRed)
        (TorchLean.Tensor.reduceMean 1 xR hRed)
        (linfNorm (meanRowBoundVec eps xR))
```
```leanOutput reductions (whitespace := lax)
@approxTensor_reduce_sum_columns : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd] {m n : ℕ}
  {xS : Spec.SpecTensor [m, n]} {xR : TorchLean.Tensor (NF β fexp rnd) [m, n]}
    {eps : ℝ},
  approxTensor toSpec xS xR eps →
    ∀ (hRed : Spec.Shape.NonemptyAxis 0 [m, n]),
      approxTensor toSpec (TorchLean.Tensor.reduceSum 0 xS hRed)
        (TorchLean.Tensor.reduceSum 0 xR hRed)
        (linfNorm (sumColumnBoundVec eps xR))
```

The row-sum theorem accounts for the number of accumulated
terms and for the same row/column indexing used by the executable reducer. That is the sort of
detail that matters for LayerNorm, attention logits, pooled features, and minibatch losses.

For row sums, `sumRowBoundVec eps xR` contains one accumulated error budget per row.
`linfNorm (sumRowBoundVec eps xR)` takes the maximum of those entries, producing the scalar budget
in the theorem's conclusion. Its dependence on `xR` records the magnitudes encountered during the
fold. Row means use `meanRowBoundVec` to include rescaling and the division's rounding; column sums
use `sumColumnBoundVec` with the other axis.

These expressions are derived bounds. A caller may replace one with a simpler upper bound after
proving that it dominates the derived expression. A row bound does not automatically cover a column
reduction, because it follows different sequences of operands.

Taking the maximum across coordinate budgets changes the precision of the estimate, not the
quantity being compared. It yields one allowance that works for every output entry; it does
not sum the errors across entries. That is appropriate for a later coordinatewise margin
argument. If a later operation sums those entries, its own reduction rule must propagate their
uncertainty and add the rounding introduced by that new sum.

Reduction order also affects the computed result. The following rows both have exact sum `2`, but
left and right association need not agree:

```lean (name := raOrder)
section
open TorchLean
-- The tame row is exact in binary floating point: every
-- partial sum is representable.
def raTameRow : Tensor Float [4] := [0.5, 0.5, 0.5, 0.5]

-- The wild row is not. `1e16 + 1` rounds back to `1e16`,
-- so one of the two `1.0` terms is absorbed and lost.
def raWildRow : Tensor Float [4] :=
  [1.0e16, 1.0, -1.0e16, 1.0]

def raLeftSum {n : Nat} (xs : Tensor Float [n]) : Float :=
  xs.sum

def raRightSum {n : Nat} (xs : Tensor Float [n]) : Float :=
  let reversed := Tensor.ofFn fun i : Fin n =>
    xs.getScalar i.rev
  Tensor.foldl (fun acc x => x + acc) 0.0 reversed

#eval (raLeftSum raTameRow, raRightSum raTameRow)
#eval (raLeftSum raWildRow, raRightSum raWildRow)
end
```
```leanOutput raOrder
(2.000000, 2.000000)
```
```leanOutput raOrder
(1.000000, 0.000000)
```

The small-magnitude row is exact in both orders. In the other row, left association loses one
unit contribution when it is added to a much larger value; right association loses both. The
outputs are `1` and `0`. This absorption depends on the operands' scale, which is why the local
bound follows the runtime values as well as the number of terms. The proof follows the same fold
order as the reducer. {Informal.citet goldberg1991}[] explains the underlying rounding effect.

`Shape.NonemptyAxis 1 [m, n]` records that the selected axis has positive length. The reduction
APIs require this evidence, and the mean proof uses the resulting `0 < n` fact when bounding
division by the row count.

Softmax needs even more care. Scalar logistic-style bounds are not a proof of axis softmax, because
axis softmax couples every coordinate through the denominator. TorchLean's
{src "NN/Proofs/RuntimeApprox/NF/SoftmaxAxis.lean"}[axis softmax approximation API]
proves the conditional NF rounded-real forward theorem `approxTensor_softmaxVecSpec`: it accounts
for max subtraction, exponential approximation, a sequential denominator sum, and division, under an
explicit denominator-error margin. `approxTensor_softmaxRowsSpec` lifts the result rowwise.

Hard masking uses exact Boolean mask semantics, including an exact-zero theorem for an all-blocked
row. `HardMaskedRowsEvidence` records the selected maxima, their approximation proofs, positive
real denominator lower bounds, and rounded denominator margins required by
`approxTensor_hardMaskedSoftmaxRowsSpec_of_max`. Backward bounds are provided by
`approxTensor_softmaxBackwardFromWeightsVecSpec` and `approxTensor_softmaxBackwardVecSpec`. The
analytic facts `sum_softmaxVec`, `sum_softmaxJvp`, and `abs_softmaxJvp_le_two_mul` establish
normalization, zero-sum JVP coordinates, and the dimension-independent bound
$`\lvert\operatorname{vjp}_i\rvert\le 2G`.

These are NF rounded-real theorems, not automatic claims about FloatLib binary32, a fused attention
kernel, or native binary32. The numerical-certificate registry's softmax rule only derives the
coarse range $`[0,1]`; that range is not a forward-error theorem.

Max subtraction gives the real softmax denominator a useful scale. In a nonempty row, a maximum
entry has shifted score zero and hence exponential one; the other shifted exponentials are
positive and at most one. This explains why the denominator can be bounded away from zero
before rounded errors are introduced. A hard-masked row needs an allowed entry to make that
argument. An all-blocked row instead follows the separately declared zero convention. The
evidence object records which case applies, so the division proof never obtains positivity from
an entry the mask has excluded.

## Normalization And Attention

Normalization and attention compose several domain-sensitive operations, so TorchLean records an
intermediate error trace instead of assigning one unexplained tolerance to the layer. The
{src "NN/Proofs/RuntimeApprox/NF/Normalization.lean"}[normalization approximation API]
handles arbitrary tensor rank once the selected mean and variance reductions have been certified.
Its centering, variance stabilization, square root, division, and affine stages expose the lower
bounds needed to keep the denominator away from zero. In particular,
`approxTensor_normalizeCore` assumes approximation evidence for the input, mean, variance, scale,
bias, and epsilon, plus a positive exact stabilized-variance lower bound and strict rounded-error
margins. It does not derive the mean and variance reduction bounds itself.

For BatchNorm, the ideal reverse rule is connected to the mathematical forward map in the
{src "NN/Proofs/Autograd/Tape/Ops/Norm/BatchNorm.lean"}[BatchNorm autograd proof].
The theorem `batchNormJvp_batchNormBackward_adjoint` covers an arbitrary list of spatial axes and
all three cotangents: input, scale, and bias. The rounded normalization bounds can therefore be read
against a proved ideal reverse rule rather than an independently written gradient formula.

The
{src "NN/Proofs/RuntimeApprox/NF/Attention.lean"}[attention approximation API] builds scaled
dot-product attention {Informal.citep transformer2017}[] from matrix multiplication, scaling, stable
axis softmax, and a second matrix multiplication. A hard attention mask is semantic: blocked entries
have zero softmax numerator. It is not represented by adding a large finite negative constant, which
would change the function for
sufficiently large logits. Backend capsules must therefore advertise a matching mask convention
before their output can inherit this theorem. The masked theorem
`approxTensor_scaledDotProductAttention_masked` consumes `HardMaskedRowsEvidence`; the canonical
inverse-square-root scale theorem also requires a positive feature dimension and a square-root
margin. These are conditional NF approximation theorems, not proofs for arbitrary fused attention
implementations. A kernel in the style of {Informal.citet flashattention2022}[] recomputes the
softmax in tiles and accumulates in a different order, so it needs its own agreement statement
before it can inherit anything proved here.

```lean (name := normAttn)
-- The trace records intermediate errors; the theorems
-- require the domain margins to hold.
#check @normalizeCoreErrorTrace
#check @approxTensor_normalizeCore
#check @approxTensor_scaledDotProductAttention_masked
```

The first signature in `normAttn` constructs a trace; the second proves what its final bound
means when the supplied reduction and domain evidence hold. There are two successive margins
to preserve. Error in variance plus epsilon must stay below the exact stabilized-variance lower
bound, keeping the rounded square-root input positive. The resulting square-root error must
then stay below the square root of that lower bound, keeping the division denominator positive.
The theorem also takes broadcasting evidence for the reduced statistics, scale, and bias, so
these quantities are compared at the coordinates where the normalized tensor uses them.

The masked attention theorem consumes the row evidence at the softmax stage between its two
matrix products. The first product's error becomes score error; the second combines uncertain
weights with uncertain values. A blocked entry contributes an exact zero softmax weight under
the hard-mask convention, whereas an allowed entry participates in the denominator and its
error bound. Keeping that distinction through the composition matters even if both masks happen
to produce nearly identical outputs on an ordinary test sentence.

# Rank-Polymorphic Convolution

Convolution is parameterized by the number of spatial axes, with one kernel, stride, and padding
entry for each axis. The same definition therefore describes sequence, image, volume, and
higher-rank convolutions. Its typed lowering and executable IR semantics are connected by the
{src "NN/Runtime/Autograd/IRExec/Correctness/Ops/Convolution.lean"}[convolution
semantic-preservation proof].

The ideal autograd proof is in
{src "NN/Proofs/Autograd/Tape/Ops/Conv/FDeriv.lean"}[Convolution FDeriv].
It proves the Fréchet derivative of the rank-general forward map and shows that the implemented
kernel, bias, and input reverse rules are adjoint to that derivative.

The
{src "NN/Proofs/RuntimeApprox/NF/Convolution.lean"}[rounded convolution proof] then follows the
implementation's actual nested loops. `approx_convSpec_coordinate` bounds one forward coordinate.
The kernel, bias, and input-gradient theorems bound the three fields `convBackwardSpec` returns.
Padding branches, multiplication error, and every rounded accumulation are included; the proofs do
not reorder the sums or assume floating-point associativity.

```lean (name := convLemmas)
-- Match each of the three returned convolution gradients
-- with its coordinate error theorem.
#check @approx_convSpec_coordinate
#check @approx_convKernelDerivSpec_coordinate
#check @approx_convBiasDerivSpec_coordinate
#check @approx_convInputDerivSpec_coordinate
```

The four convolution signatures expose exactly which operands affect each result. The forward
coordinate theorem takes approximation of the kernel, bias, and input, then selects an output
channel and spatial multi-index. A kernel-gradient coordinate instead needs the input and
output-cotangent bounds: that gradient sums products of those two quantities. The bias gradient
needs only the output-cotangent bound because it sums those cotangents. The input gradient
needs the kernel and output-cotangent bounds. These different premise lists follow the
backward formulas; they are not four copies of one undifferentiated layer tolerance.

Each conclusion concerns the chosen coordinate and the fold that computes it. A tensor-wide
budget can be obtained by bounding these coordinate budgets uniformly, but the coordinate
theorems retain the dependence on channel, location, and padding. Near a padded boundary,
some forward products use the specified zero value. The proof handles those branches in the
same index traversal, rather than replacing the convolution by a larger unqualified matrix
product whose accumulation order might differ.

# Rounded Optimizer Steps

The backward theorem produces approximate gradients; a training claim must still account for the
optimizer arithmetic. The generic
{src "NN/Proofs/RuntimeApprox/Optimizer.lean"}[optimizer numerical contract]
records an exact state, a rounded state, their relation, a one-step bound transformer, and the proof
that the relation survives one update. `NumericalStepContract.run_approx` proves the corresponding
finite-run result once for every optimizer satisfying that interface.

The concrete
{src "NN/Proofs/RuntimeApprox/NF/Optimizers.lean"}[NF optimizer proofs] use the optimizer equations
directly:

- SGD propagates learning-rate, gradient, and parameter error;
- momentum SGD additionally propagates the momentum-buffer error;
- AdamW records errors for both moments, bias correction, square root, adaptive division,
  decoupled weight decay, and the final subtraction.

AdamW needs more than a nominal epsilon. Its theorem requires a positive lower bound on the exact
bias-corrected second moment and explicit margins showing that rounding does not cross either the
square-root or division boundary. This is the numerical counterpart of the recurrence in Kingma and
Ba's Adam paper (https://arxiv.org/abs/1412.6980) and of the decoupled decay
{Informal.citet adamw2019}[] introduced. Alongside those update stages, the theorem records
approximation budgets for the state and
derived scalars, and domain margins for the square root and division.

```lean (name := optimizers)
-- Stateful optimizers must preserve the state relation as
-- well as parameter closeness.
#check @NumericalStepContract.run_approx
#check @approxTensor_sgd_update
#check @approxTensor_momentumSGD_update
#check @approxTensor_adamW_update
```
```leanOutput optimizers (whitespace := lax)
@NumericalStepContract.run_approx : ∀ {R : Type} {toSpec : R → ℝ} (contract :
  NumericalStepContract R toSpec)
  {shape : Spec.Shape} {exact : Optim.Step ℝ shape (contract.ExactState shape)}
  {runtime : Optim.Step R shape (contract.RuntimeState shape)} {error : StepError
    contract.StateError shape}
  {steps : Array (contract.StepInput shape)},
  contract.StepStreamApprox exact runtime error steps →
    contract.stateApprox exact.optimizerState runtime.optimizerState error.optimizerStateError →
      approxTensor toSpec exact.parameters runtime.parameters error.parameterError →
        contract.RunApprox exact runtime error steps
```
```leanOutput optimizers (whitespace := lax)
@approxTensor_sgd_update : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd]
  {s : Spec.Shape} {stateS : Optim.SGD.State ℝ s} {stateR : Optim.SGD.State
    (NF β fexp rnd) s}
  {learningRateError : ℝ} {exactParameters : TorchLean.Tensor ℝ s}
  {runtimeParameters : TorchLean.Tensor (NF β fexp rnd) s} {parameterError : ℝ}
  {exactGradients : TorchLean.Tensor ℝ s} {runtimeGradients : TorchLean.Tensor
    (NF β fexp rnd) s}
  {gradientError : ℝ},
  sgdStateApprox stateS stateR learningRateError →
    approxTensor toSpec exactParameters runtimeParameters parameterError →
      approxTensor toSpec exactGradients runtimeGradients gradientError →
        approxTensor toSpec (Optim.SGD.update stateS exactParameters exactGradients).parameters
          (Optim.SGD.update stateR runtimeParameters runtimeGradients).parameters
          (sgdStepError learningRateError parameterError gradientError stateR runtimeParameters
              runtimeGradients).parameterError
```
```leanOutput optimizers (whitespace := lax)
@approxTensor_momentumSGD_update : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd]
  {s : Spec.Shape} {stateS : Optim.MomentumSGD.State ℝ s}
  {stateR : Optim.MomentumSGD.State (NF β fexp rnd) s} {stateError :
    MomentumSGDStateError s}
  {paramsS : TorchLean.Tensor ℝ s} {paramsR : TorchLean.Tensor (NF β fexp
    rnd) s} {paramsError : ℝ}
  {gradsS : TorchLean.Tensor ℝ s} {gradsR : TorchLean.Tensor (NF β fexp rnd)
    s} {gradsError : ℝ},
  momentumSGDStateApprox stateS stateR stateError →
    approxTensor toSpec paramsS paramsR paramsError →
      approxTensor toSpec gradsS gradsR gradsError →
        have nextError := momentumSGDStepError stateError paramsError gradsError stateR
          paramsR gradsR;
        momentumSGDStateApprox (Optim.MomentumSGD.update stateS paramsS gradsS).optimizerState
            (Optim.MomentumSGD.update stateR paramsR gradsR).optimizerState
              nextError.optimizerStateError ∧
          approxTensor toSpec (Optim.MomentumSGD.update stateS paramsS gradsS).parameters
            (Optim.MomentumSGD.update stateR paramsR gradsR).parameters nextError.parameterError
```
```leanOutput optimizers (whitespace := lax)
@approxTensor_adamW_update : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd]
  {s : Spec.Shape} {stateS : Optim.AdamW.State ℝ s} {stateR : Optim.AdamW.State
    (NF β fexp rnd) s}
  {stateError : AdamWStateError s} {derivedErrors : AdamWDerivedErrors} {paramsS :
    TorchLean.Tensor ℝ s}
  {paramsR : TorchLean.Tensor (NF β fexp rnd) s} {paramsError : ℝ} {gradsS :
    TorchLean.Tensor ℝ s}
  {gradsR : TorchLean.Tensor (NF β fexp rnd) s} {gradsError η : ℝ},
  adamWStateApprox stateS stateR stateError →
    approxTensor toSpec paramsS paramsR paramsError →
      approxTensor toSpec gradsS gradsR gradsError →
        |toSpec (1 - stateR.beta1) - (1 - stateS.beta1)| ≤ derivedErrors.oneMinusBeta1 →
          |toSpec (1 - stateR.beta2) - (1 - stateS.beta2)| ≤ derivedErrors.oneMinusBeta2 →
            |toSpec (1 / (1 - Optim.scalarPowNat stateR.beta1 (stateR.stepCount + 1))) -
                    1 / (1 - Optim.scalarPowNat stateS.beta1 (stateS.stepCount + 1))| ≤
                derivedErrors.firstMomentBiasInverse →
              |toSpec (1 / (1 - Optim.scalarPowNat stateR.beta2 (stateR.stepCount + 1))) -
                      1 / (1 - Optim.scalarPowNat stateS.beta2 (stateS.stepCount + 1))| ≤
                  derivedErrors.secondMomentBiasInverse →
                |toSpec (stateR.learningRate * stateR.weightDecay) -
                  stateS.learningRate * stateS.weightDecay| ≤
                    derivedErrors.decayScale →
                  0 < η →
                    0 ≤ stateS.epsilon →
                      (have nextStepCount := stateS.stepCount + 1;
                        have moment2 :=
                          (stateS.secondMoment.scaleSpec stateS.beta2).addSpec
                            (gradsS.squareSpec.scaleSpec (1 - stateS.beta2));
                        TorchLean.Tensor.Forall (fun z => η ≤ z)
                          (moment2.scaleSpec (1 / (1 -
                            Optim.scalarPowNat stateS.beta2
                            nextStepCount)))) →
                        (adamWStepErrorTrace stateError derivedErrors
                          paramsError gradsError η stateR paramsR
                                gradsR).correctedSecondMoment <
                            η →
                          (adamWStepErrorTrace stateError
                            derivedErrors paramsError gradsError η
                            stateR paramsR
                                  gradsR).denominator <
                              √η →
                            have trace :=
                              adamWStepErrorTrace stateError
                                derivedErrors paramsError
                                gradsError η stateR paramsR
                                gradsR;
                            adamWStateApprox (Optim.AdamW.update
                              stateS paramsS gradsS).optimizerState
                                (Optim.AdamW.update stateR paramsR gradsR).optimizerState
                                { learningRate :=
                                  stateError.learningRate, beta1
                                  := stateError.beta1,
                                  beta2 := stateError.beta2, epsilon := stateError.epsilon,
                                  weightDecay :=
                                    stateError.weightDecay,
                                    firstMoment :=
                                    trace.firstMoment,
                                  secondMoment := trace.secondMoment } ∧
                              approxTensor toSpec
                                (Optim.AdamW.update stateS paramsS
                                gradsS).parameters
                                (Optim.AdamW.update stateR
                                  paramsR gradsR).parameters
                                  trace.parameterError
```

The optimizer's retained state determines which relations must survive an update.

SGD keeps nothing between steps, so its conclusion is a single `approxTensor` on the updated
parameters, and its only state hypothesis is `sgdStateApprox`, which relates two learning rates.
Momentum SGD keeps one buffer, and its conclusion becomes a conjunction: a bound for the next
momentum buffer *and* a bound for the parameters. The next update needs the bound on the buffer
produced by the current one. Returning both relations makes the conclusion usable as the next
step's hypothesis.

AdamW keeps two buffers, a step counter, and a bias correction, and its statement is by far the
longest in the chapter. Five of its hypotheses are error budgets for quantities Adam derives rather
than stores: `1 - beta1`, `1 - beta2`, and the two bias-correction reciprocals
`1 / (1 - beta^(t+1))` , together with `learningRate * weightDecay` . Those are computed in floating
point on every step, so they carry their own
rounding, including the arithmetic used for the powers. Then come the domain
conditions: `0 < eta` , `0 ≤ stateS.epsilon` , a `Tensor.Forall` saying the
exact bias-corrected second moment is everywhere at least `eta`, and two strict inequalities
requiring the accumulated error in the corrected second moment and in the denominator to stay below
`eta` and `√eta` . These conditions are stronger than merely choosing positive epsilon: they require
the corrected
second moment itself to have a strictly positive lower bound, excluding a zero second moment even
when epsilon would make the numerical division finite. An application with zero second moments
therefore needs another estimate; positive epsilon alone does not discharge this theorem's
hypotheses. These are domain conditions for the NF proof model, which does not model NaNs.

`NumericalStepContract.run_approx` composes updates for any supplied contract. Its
`StepStreamApprox` hypothesis relates the sequence of step inputs, and its conclusion `RunApprox`
relates the resulting states over the finite run. To apply it to an optimizer, one must instantiate
the contract and supply its state and step evidence; a local update inequality by itself does not
supply those data.

For several updates, the new state relation is part of the input to the next step. Resetting
the momentum or moment error to zero after an update would discard accumulated uncertainty.
Likewise, `StepStreamApprox` is not supplied merely by selecting the same minibatch sequence:
the gradients must be related at the exact and rounded states actually reached. The finite-run
theorem organizes this induction once those step premises are available. It does not infer a
bound on an evolving gradient stream from a gradient comparison performed only at initialization.

# Scale Aware Tolerances

The scale layer is split across
[scale bounds](https://github.com/lean-dojo/TorchLean/blob/main/NN/Proofs/RuntimeApprox/Scale.lean),
{src "NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean"}[scale approximation],
{src "NN/Proofs/RuntimeApprox/Scale/ForwardScale.lean"}[forward scale propagation], and
{src "NN/Proofs/RuntimeApprox/Scale/BackwardScale.lean"}[backward scale propagation].

The scale approximation API defines `BList`, a list of nonnegative scale bounds indexed by shape,
plus helpers such as `scaleTensor`, `scaleCtx`, and `tolFromEpsScale`. An absolute error budget is
computed from a machine-like epsilon times a local scale bound.

A graph can then carry both "how close" and "at what scale" information. The lemmas
`approxTensorWithTol_from_scale` and `approxCtx_get_tolFromEpsScale` connect scale estimates back to
the tolerance API used by graph theorems.

This remains a separate layer because not every proof needs scale aware reasoning. Small examples
and operator proofs written by hand are often clearer with absolute tolerances. Larger deployment
claims usually need scale, because one global absolute epsilon is rarely meaningful across all
activations and gradients.

A scale bound can simplify a value-dependent error expression before it enters the next node.
For instance, an upper bound on operand magnitudes replaces those magnitudes in a product
estimate by quantities already known throughout a region. The resulting estimate may be looser,
but it can be reused for every input in that region. The scale relation is the evidence that
permits this replacement. Recording a convenient number in `BList` is insufficient unless the
actual tensor values satisfy the accompanying bound.

# FP32 And Verification Margins

The
{src "NN/Proofs/RuntimeApprox/FP32/Layers.lean"}[FP32 layer approximation API],
{src "NN/Proofs/RuntimeApprox/FP32/MLP.lean"}[FP32 MLP approximation API], and
{src "NN/Proofs/RuntimeApprox/FP32/CROWN.lean"}[FP32 CROWN bridge API] connect the approximation
style to layerwise and verifier reasoning.

The CROWN connection is where runtime approximation meets certification. A verifier over the reals
may prove a margin, but a float runtime can differ from ideal real arithmetic. To transfer the
claim, the real margin must dominate the runtime approximation budget: the certified lower margin
after subtracting the runtime error still has to be positive.

When that inequality is proved, the runtime prediction is still certified.

The Float32 soundness layer uses the same separation. `FP32` is a proof model, FloatLib binary32
is an
executable bit oriented model, and native hardware remains a named assumption unless a bridge
theorem covers the path being used.

# Classifier Margin Example

Suppose we have a two layer classifier whose hidden layer is $`\operatorname{ReLU}(W_1x+b_1)` and
whose output is $`W_2h+b_2`.

The ideal proof might establish that, for all inputs in a box, the margin over the reals for class
$`0` over class $`1` is at least $`0.05`.

A float deployment theorem adds the runtime approximation statement: for every input in the same
box, $`y_{\mathrm{run}}` approximates $`y_{\mathrm{real}}` within $`0.01` per relevant logit.

Then the margin proof must be adjusted:

$$`f_0^{\mathrm{run}}(x)-f_1^{\mathrm{run}}(x)
\ge 0.05-0.01-0.01
=0.03>0.`

The unfavorable case moves class zero down by its error budget and class one up by its budget.
Subtracting both leaves a positive margin for every input covered by both premises. A pointwise
error estimate at one sample would not justify this claim over the whole box.

Equal logit budgets are convenient but unnecessary. If the winner's error is bounded by one
quantity and the competitor's by another, their sum is the amount subtracted from the real
pairwise margin. For several classes, the calculation must succeed against each competitor.
This can retain useful information when some output coordinates have much tighter numerical
bounds than others. In every case, the input region and parameter interpretation must agree
between the real margin theorem and the approximation theorem used to transfer it.

# End To End Rounded Training

The rounded-real end-to-end file connects the proof-bearing reverse graph to executable autograd
`GraphData`, extracts typed parameter gradients, and composes those gradients with the optimizer
contracts:

```lean (name := endToEnd)
-- Follow the bounds from a graph context to one indexed
-- parameter update.
#check @eval_approx_graphData
#check @backprop_approx_graphData
#check @backprop_gradient_approx_graphData
#check @backprop_optimizer_update_approx_graphData
#check @trainingStepTrace
```

```leanOutput endToEnd (whitespace := lax)
@backprop_optimizer_update_approx_graphData : ∀ {β : Radix} {fexp : ℤ → ℤ}
  [inst : ValidExp fexp] {rnd : ℝ → ℤ}
    [ValidRndToNearest rnd]
  {Γ ss : List Spec.Shape} (g : RevGraph toSpec Γ ss) (i : Fin Γ.length)
  (contract : NumericalStepContract (NF β fexp rnd) toSpec)
  (xS : TorchLean.TensorPack Spec.SpecScalar Γ) (xR : TorchLean.TensorPack
    (NF β fexp rnd) Γ)
  (epsIn : EList Γ) (seedS : TorchLean.TensorPack Spec.SpecScalar (Γ ++ ss))
  (seedR : TorchLean.TensorPack (NF β fexp rnd) (Γ ++ ss)) (epsSeed : EList
    (Γ ++ ss))
  (paramsS : TorchLean.Tensor ℝ (Γ.get i)) (paramsR : TorchLean.Tensor (NF β
    fexp rnd) (Γ.get i))
  (paramsError : ℝ) (stateS : contract.ExactState (Γ.get i)) (stateR : contract.RuntimeState
    (Γ.get i))
  (stateError : contract.StateError (Γ.get i)) (assumptions : contract.StepAssumptions (Γ.get i)),
  approxCtx toSpec xS xR epsIn →
    approxCtx toSpec seedS seedR epsSeed →
      approxTensor toSpec paramsS paramsR paramsError →
        contract.stateApprox stateS stateR stateError →
          (have exactGradients := g.backpropSpec xS seedS;
            have runtimeGradients := (LinkAutogradAlgebra.RevGraph.toGraphData
              g).backpropCtx xR () seedR;
            have gradientError := (g.backpropBounds epsIn xR epsSeed seedR fun {Δ}
              => ctxAddBound).get i;
            contract.assumptionsHold stateS stateR stateError paramsS paramsR
              paramsError (exactGradients.get i)
              (runtimeGradients.get i) gradientError assumptions) →
            have exactGradients := g.backpropSpec xS seedS;
            have runtimeGradients := (LinkAutogradAlgebra.RevGraph.toGraphData
              g).backpropCtx xR () seedR;
            have gradientError := (g.backpropBounds epsIn xR epsSeed seedR fun {Δ}
              => ctxAddBound).get i;
            have nextError :=
              contract.nextError stateError paramsError gradientError stateR
                paramsR (runtimeGradients.get i)
                assumptions;
            contract.stateApprox (contract.updateExact stateS paramsS
              (exactGradients.get i)).optimizerState
                (contract.updateRuntime stateR paramsR (runtimeGradients.get i)).optimizerState
                nextError.optimizerStateError ∧
              approxTensor toSpec (contract.updateExact stateS paramsS
                (exactGradients.get i)).parameters
                (contract.updateRuntime stateR paramsR (runtimeGradients.get
                  i)).parameters nextError.parameterError
```

The statement takes the reverse graph `g`, an input-context index `i`, related exact and runtime
contexts, related seeds, parameter tensors, and optimizer states. It derives the two gradients and
their error from that graph and those contexts. The caller supplies the remaining evidence through
`contract.assumptionsHold`, including any domain margins required by the optimizer.

There is a further choice to make when applying this as a training theorem. `paramsS` and `paramsR`
are separate arguments: the signature does not assert that they equal entry `i` of `xS` and `xR`.
The theorem justifies updating the supplied tensors using the gradient extracted at index `i`.
For a training step that updates the parameters at which the loss was differentiated, instantiate
the parameter arguments with those context entries or provide the corresponding equalities.

These are graph-level bridge theorems:

- `eval_approx_graphData` says evaluating the executable forward graph is close to evaluating the
  spec forward graph when the local node approximation obligations have been supplied.
- `backprop_approx_graphData` says the same style of statement for reverse accumulation, with the
  accumulation error model still explicit.
- `backprop_optimizer_update_approx_graphData` applies any numerical optimizer contract to one typed
  parameter gradient produced by that executable reverse pass. SGD uses trivial step evidence;
  AdamW supplies its bias-correction and positivity margins through `AdamWStepAssumptions`. A model
  with several
  parameter tensors applies the same theorem at each parameter index.
- `trainingStepTrace` computes a proof-free report of forward, backward, gradient, parameter, and
  optimizer-state bounds. Its interpretation comes from the surrounding approximation theorems,
  not from the report record itself.

That is the runtime approximation analogue of the autograd proof architecture. Local operator
lemmas are the leaves; graph theorems compose them; deployment claims then combine the graph theorem
with any scalar/backend assumptions.

The index `i` in the printed theorem chooses a tensor from the input context, so its gradient
has shape `Γ.get i`. `backprop_gradient_approx_graphData` extracts precisely that component
from the whole-context reverse result. This is why one graph theorem can support parameters
of different shapes without flattening them into a common array. Applying the update theorem
at several indices still requires a consistent step: the contexts and seeds must describe the
same loss evaluation, and each optimizer state must belong to the parameter tensor it updates.

## Error Propagation Through An Optimizer Step

Take a parameter tensor at index $`i`. The real reverse pass produces $`g_{\mathrm{spec}}`;
executable rounded reverse mode produces $`g_{\mathrm{run}}`; and
`backprop_gradient_approx_graphData` proves

$$`\|\operatorname{toSpec}(g_{\mathrm{run}})-g_{\mathrm{spec}}\|_\infty
\le \varepsilon_g.`

Suppose the current parameters and learning rate have errors $`\varepsilon_p` and
$`\varepsilon_{\mathrm{lr}}`. For SGD,
the two executions perform the same equation in their respective scalar systems,

$$`p' = p-\eta g.`

`sgdStepError` first bounds the rounded product $`\eta g`, including both input errors and the new
multiplication rounding, then bounds the final subtraction. The graph-level theorem returns

$$`\|\operatorname{toSpec}(p'_{\mathrm{run}})-p'_{\mathrm{spec}}\|_\infty
\le \varepsilon_{p'}`

with $`\varepsilon_{p'}` equal to that computed bound, not a user-chosen test tolerance. Momentum
SGD uses the same theorem and additionally returns a bound for the updated momentum buffer.

For AdamW, the route is longer: update both moments, apply bias correction, take the second-moment
square root, form the adaptive learning rate, apply decoupled decay, and subtract the Adam update.
`AdamWStepErrorTrace` keeps the error after each stage, one field per stage, from
`squaredGradient` through `parameterError`. The evidence that makes those numbers meaningful is
`AdamWStepAssumptions`: a strictly positive `minimumSecondMoment` lying below every entry of the
exact bias-corrected second moment, plus the `AdamWDerivedErrors` budgets for the rounded scalars
`1 - beta1`, `1 - beta2`, the two reciprocal bias corrections, and the product `lr * weightDecay`.
The validity predicate then demands that the trace's own `correctedSecondMoment` error stay below
that margin and its `denominator` error below the margin's square root, which is what keeps the
division by the adaptive denominator away from zero. The optimizer theorem remains the same; only
the local contract's validity evidence is richer than SGD's.

These proofs describe numerical recurrences for explicit states. They do not automatically cover
the native trainer's parameter lookup, mutable moment-buffer allocation, per-parameter clocks, or
checkpoint restore. Those runtime mechanisms must be shown to instantiate the same state and step
relation before `NumericalStepContract.run_approx` can be cited for a concrete training process.

# Runtime Approximation APIs

The definitions are organized in the same order as the proof: define closeness, prove local operator
bounds, compose them over forward and backward graphs, and finally connect the result to autograd.

- The
  {src "NN/Proofs/RuntimeApprox/Core/Tolerance.lean"}[tolerance API] and
  {src "NN/Proofs/RuntimeApprox/Core/SpecApprox.lean"}[spec approximation API] define the
  approximation relation.
- The
  {src "NN/Proofs/RuntimeApprox/Graph/ForwardApprox.lean"}[forward graph approximation API] contains
  `FwdGraph.eval_approx`; the
  {src "NN/Proofs/RuntimeApprox/Graph/BackwardApprox.lean"}[backward graph approximation API]
  contains `RevGraph.backprop_approx`.
- The
  {src "NN/Proofs/RuntimeApprox/NF/Ops.lean"}[rounded-real operator API] supplies local obligations,
  including domain-sensitive operations such as division and safe log.
- The
  {src "NN/Proofs/RuntimeApprox/NF/Convolution.lean"}[rounded convolution proof] gives ordered
  forward and backward bounds at arbitrary spatial rank.
- The
  {src "NN/Proofs/RuntimeApprox/Scale/ScaleApprox.lean"}[scale approximation API] supports
  scale-aware error bounds.
- The
  {src "NN/Proofs/RuntimeApprox/Graph/LinkAutogradAlgebra.lean"}[autograd algebra link API] connects
  approximation to the autograd proof layer.
- The
  {src "NN/Proofs/RuntimeApprox/Optimizer.lean"}[optimizer contract API] and
  {src "NN/Proofs/RuntimeApprox/NF/Optimizers.lean"}[NF optimizer instances] continue the backward
  error budget through parameter updates.

# Runtime Agreement

For supported graph and operator fragments, runtime approximation proves that a runtime or rounded
computation stays within a stated tolerance of a spec computation. CUDA kernels, vendor library
paths, compiler rewrites, and PyTorch-exported graphs need their own agreement statements when a
claim is about those paths.

For a deployment claim, identify the graph, scalar interpretation, execution path, and input
region in each premise. The numerical bound must apply to that same computation throughout the
region used by the real theorem.

When the approximation theorem is present, the bridge is proved. When it is not present, the claim
should say which runtime path or external producer supplies the remaining evidence.
