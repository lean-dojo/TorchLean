import VersoManual
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Reduce.Fold
import TorchLean.Verify.RuntimeApprox
import TorchLean.Verify.RuntimeApprox.Tolerance
import TorchLean.Verify.NumericalCertificate
import TorchLean.Verify.RuntimeApprox.NF.Attention
import TorchLean.Verify.RuntimeApprox.NF.Convolution
import TorchLean.Verify.RuntimeApprox.NF.EndToEnd
import TorchLean.Verify.RuntimeApprox.NF.Normalization
import TorchLean.Verify.RuntimeApprox.NF.Optimizers
import TorchLean.Verify.RuntimeApprox.NF.Elementwise.SafeDivSigmoid
import TorchLean.Verify.RuntimeApprox.NF.ReductionOps
import TorchLean.Verify.RuntimeApprox.NF.Softmax
import TorchLean.Verify.RuntimeApprox.OptimizerContract
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean

-- The TorchLean declarations quoted in this chapter live under the `TorchLean` namespace, with
-- the rounded-real operator lemmas in `TorchLean.NF`. Verso keeps displayed code narrow enough to
-- read beside the prose, so the namespaces are opened here to let each `#check` fit on one
-- line. The scalar namespaces identify the imported FloatLib theory.
open TorchLean ApproxTol Tensor
open ApproxTensor
open NF
open NF.AdamW NF.MomentumSGD NF.SGD
open NumericalCertificate NumericalCertificate.ErrorTrace TensorErrorLe
open Robustness (linfNorm)
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
first interpret the rounded scalar as a specification value with `toSpec`, and an approximation
statement then bounds the difference by a budget `ε`:

$$`\operatorname{toSpec}(y_{\mathrm{run}})\approx_\varepsilon y_{\mathrm{spec}}`

For tensors of the same shape, a single budget can bound every coordinate. Writing this with the
maximum norm gives

$$`\left\|\operatorname{toSpec}(Y_{\mathrm{run}})-Y_{\mathrm{spec}}\right\|_\infty
\le \varepsilon.`

# Real And Floating-Point Semantics

A real-valued proof doesn't account for the arithmetic that evaluates its formulas. Rounding,
cancellation, overflow, reduction order, fused kernels, and domain guards can affect an execution,
and {Informal.citet goldberg1991}[] explains these effects in floating-point arithmetic. Take a
softmax implementation: it combines max-subtraction, exponent approximations, a finite sum, and
division, so its error depends on how those operations are carried out and how their errors
interact.

To transfer a real-arithmetic guarantee to execution, we need to:

1. prove the claim over the reals or over the spec
2. prove an approximation relation between runtime tensors and spec tensors
3. inflate margins, enclosures, or certificates by the approximation budget
4. leave native hardware assumptions explicit when they are outside Lean.

These steps match standard numerical analysis practice, and Higham's *Accuracy and Stability of
Numerical Algorithms* (https://epubs.siam.org/doi/book/10.1137/1.9780898718027) gives the classical
background. They also match the caution neural network verification needs, where a tiny rounding
gap can matter if the verified margin is tiny.

You have to follow two sources of discrepancy through this process. A stored weight may already
differ from the intended real weight before evaluation starts, and even when every input agrees
exactly, arithmetic can introduce error of its own at each node. A useful local theorem accepts the
first kind of error as an input and adds the second kind to its conclusion. That's why a graph
bound needs both initial approximation evidence and rules for the operations: proving that the
initial tensors are close would leave the rounding inside the graph unaccounted for.

# The Core Relation: Spec Tensors And Tolerances

Three files define the small vocabulary: {srcDir "TorchLean/Verify/RuntimeApprox"}[runtime
approximation core], {src "TorchLean/Verify/RuntimeApprox.lean"}[spec
approximation], and {src "TorchLean/Verify/RuntimeApprox/Tolerance.lean"}[tolerances].

We define the basic tensor relation in the spec approximation layer. Given a conversion
`toSpec : α → ℝ`, a runtime tensor approximates a spec tensor when every component is close after
conversion to `ℝ`. In that layer, the key names are:

- `Tensor.toReal`: pointwise conversion from runtime tensor values to spec scalars.
- `linfNorm`: the max-abs tensor norm of `Robustness`, used for error
  statements.
- `ApproxWith`: absolute tensor approximation under a chosen norm with a scalar error budget.
- `ApproxWithTol`: approximation using a tolerance object.
- `ApproxTensorWithTol`: packaged tensor tolerance relation.

In the tolerance API, `ApproxTol` has absolute, relative, and slack components, and
`ApproxTol.absOnly`, `bound`, and `ApproxR` let later proofs move between simple absolute
error statements and more scale aware claims.

For a single scalar, the scale-aware shape is:

$$`|r-s|
\le
\varepsilon_{\mathrm{slack}}\bigl(\varepsilon_{\mathrm{abs}}
+\varepsilon_{\mathrm{rel}}\max(|r|,|s|)\bigr).`

The choice is practical. An absolute tolerance of `1e-6` may be meaningful near zero and irrelevant
near `1e9`, while a relative tolerance captures "small compared with the scale of the value."
TorchLean uses both styles, but it makes the choice visible instead of burying it in test
thresholds.

Since `bound` is a real function, we can compute the difference between the two readings.
Read `1e-6` as an absolute budget and it stays `1e-6` no matter how large the values are:

```lean
-- An absolute tolerance is independent of the operands’
-- magnitude.
example :
    bound (ApproxTol.ofReal 1e-6 0 1) 1e9 1e9
      = 1e-6 := by
  norm_num [bound, ApproxTol.ofReal,
    Real.toNNReal]
```

Read the same `1e-6` as a relative budget at the same scale and it's a thousand:

```lean
-- A relative tolerance scales with the values being
-- compared.
example :
    bound (ApproxTol.ofReal 0 1e-6 1) 1e9 1e9
      = 1e3 := by
  norm_num [bound, ApproxTol.ofReal,
    Real.toNNReal]
```

Near a zero reference value, a purely relative comparison needs particular care. With zero absolute
tolerance and slack one, comparing a nonzero value to zero would require its magnitude to be at
most `rel` times that same magnitude, which can't hold for a relative tolerance below one. Room for
small discrepancies near zero comes from the absolute component, while the relative component
scales the allowance for large values. Those are different jobs, so an application should choose
them from the claim it needs and not treat the two fields as interchangeable.

## Tensor Approximation Contracts

The relation defines what it means for two tensors to be close. A theorem using that relation can
quantify over a whole family of inputs, or it can concern just one pair of tensors. Here are the
two signatures that expose the scalar interpretation and the proposition to be proved:

```lean (name := coreVocab)
-- Separate pointwise interpretation from the proposition
-- asserting approximation.
#check @Tensor.toReal
#check @ApproxTensorWithTol
```
```leanOutput coreVocab (whitespace := lax)
@Tensor.toReal : {α : Type} →
  [inst : Storage α] → {s : Shape} → (α → ℝ) → Tensor α
    s → Tensor ℝ s
```
```leanOutput coreVocab (whitespace := lax)
@ApproxTensorWithTol : {α : Type} →
  [inst : Storage α] →
    {s : Shape} → (α → ℝ) → Tensor ℝ s → Tensor α s →
      ApproxTol → Prop
```

`Tensor.toReal` applies the supplied function `α → ℝ` coordinatewise, which defines an
interpretation of the tensor. It doesn't itself prove that a runtime operation agrees with a
specification operation: those operation-level facts are the hypotheses that later graph theorems
compose.

The shape of `ApproxTensorWithTol` is the contract: a scalar interpretation `toSpec`, a spec
tensor, a runtime tensor, a tolerance, and a `Prop`. Nothing about hardware appears in it. That's
why the interpretation has to be named.

A tolerance determines a real-valued budget:

```lean (name := coreBound)
-- The budget is a real-valued expression, not evidence that
-- a comparison passes.
#check @bound
```

```leanOutput coreBound
bound : ApproxTol → ℝ → ℝ → ℝ
```

The proposition bounds the L∞ distance by `bound tol ‖spec‖∞ ‖runtime‖∞`, and every
coordinate shares that allowance: the relative component uses the whole tensor's maximum norm, not
each entry's magnitude as an elementwise tolerance comparison would. That makes the trusted
boundary easy to locate. If `toSpec` interprets an executable rounded-real model, the theorem is
about that rounded-real model, and if the actual deployment path is CUDA, cuBLAS, PyTorch, or a
fused native kernel, a separate agreement statement is needed before the theorem says anything
about that path.

So the scalar interpretation and the actual execution path must match. Numerical tests can compare
selected executions, but a deployment theorem needs an agreement result with the scope its final
claim requires.

In the displayed signatures, the shared shape parameter ensures that the comparison has matching
coordinates, but it doesn't decide which coordinates are supposed to correspond semantically
(comparing two parameter tensors after a permutation, for example, could satisfy the shape
constraint while comparing the wrong weights). Callers must use the same parameter ordering and
scalar interpretation as the computation being analyzed. Once those choices are fixed, the
coordinatewise relation gives later operators a precise premise to consume.

## Comparison With PyTorch Tolerances

A related comparison appears in PyTorch's `torch.allclose` {Informal.citep pytorch2019}[]. For
finite entries, its condition with `rtol` and `atol` is the elementwise inequality

$$`|a-b|\le \mathrm{atol}+\mathrm{rtol}\,|b|`

In `bound`, `abs` plays the role of `atol` and `rel` that of `rtol`. But TorchLean scales by
$`\max(|a|,|b|)`, not by the second argument alone, and it carries a `slack` factor (for proofs
that need to inflate a budget without rewriting it).

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

Symmetry and slack are where the two formulas differ. Quantification is a separate issue:
`allclose` checks the two supplied arrays, while a theorem about `ApproxTensorWithTol` may range
over all inputs, weights, or shapes satisfying its hypotheses. Writing the relation as a `Prop`
doesn't by itself make a claim universal, because the claim's quantifiers determine the scope.

# Forward Graph Approximation

You'll find the forward graph theorem in
{src "TorchLean/Verify/RuntimeApprox/Graph/Forward.lean"}[
TorchLean.Verify.RuntimeApprox.Graph.Forward API]. It mirrors the structure of the autograd
tape proof, but its invariant is approximation instead of derivative soundness.

Here are the main objects:

- `EList`: list of scalar error budgets indexed by shape.
- `ApproxCtx`: approximation relation for a whole context.
- `FwdNode`: a node with spec forward, runtime forward, bound, and local soundness.
- `FwdGraph`: a snoc list graph of forward nodes that carry approximation evidence.
- `FwdGraph.eval_approx`: theorem for the whole forward graph.

On a `FwdNode`, the local theorem says that if each runtime input approximates the corresponding
spec input, then the runtime output approximates the spec output within this node's bound.

`FwdGraph.eval_approx` composes those local statements. When a graph appends a node, the proof uses
the node's local `sound` theorem, appends that node's bound to the error context, and continues. It
follows the same architecture as the autograd soundness theorem (local correctness first, then a
global induction over the graph).

Multiplication is a tiny example. Its spec value is the real product, while its runtime value is
the rounded product computed by the chosen scalar model.

If $`x_{\mathrm{run}}` is within $`\varepsilon_x` of $`x_{\mathrm{real}}`, and $`y_{\mathrm{run}}`
is within $`\varepsilon_y` of $`y_{\mathrm{real}}`, the local multiplication lemma supplies a bound
for $`z_{\mathrm{run}}` versus $`z_{\mathrm{real}}`. The graph theorem then lets that bound feed
the next node.

Write the interpreted runtime inputs as $`x+dx` and $`y+dy`,

$$`
(x+dx)(y+dy)-xy=x\,dy+y\,dx+dx\,dy.
`

In the expansion, the first two terms describe how each input error is scaled by the other exact
operand, and the third accounts for the interaction of the two errors. The identity can be checked
directly:

```lean
-- Keep the product of the two incoming errors: it is part
-- of finite error propagation.
example (x y dx dy : ℝ) :
    (x + dx) * (y + dy) - x * y
      = x * dy + y * dx + dx * dy := by
  ring
```

Hence input uncertainty alone contributes at most

$$`|x|\,\varepsilon_y+|y|\,\varepsilon_x
  +\varepsilon_x\varepsilon_y.`

By the triangle inequality, the sum is at most the sum of the three magnitudes, and applying
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

On top of this number, the local rule adds the rounding error of the multiplication itself. The
number isn't a universal tolerance, though: it shows why the graph carries operand scale and
incoming error instead of attaching one unexplained epsilon to every multiplication.

The graph carries bounds for all available values (not just the newest output), so if a later node
reuses an earlier activation, its local theorem retrieves that activation's bound. The graph's
context invariant follows the same data dependencies as the computation.

# Backward Graph Approximation

Backward approximation is developed in the
{src "TorchLean/Verify/RuntimeApprox/Graph/Backward.lean"}[backward approximation API].
This file is the runtime approximation analogue of reverse mode AD.

Its key objects are:

- `RevNode`: a forward node plus spec/runtime VJP functions and a VJP error transformer.
- `RevGraph`: reverse graph with local approximation evidence.
- `RevGraph.eval_approx`: forward approximation inherited from the forward graph.
- `RevGraph.backprop_approx`: theorem for reverse accumulation across the whole graph.

`RevGraph.backprop_approx` states that the runtime reverse pass approximates the spec reverse pass,
provided the input context, forward tape, and seed cotangents are appropriately related. Addition
during gradient accumulation isn't treated as automatic: the theorem takes `addBound` and
`addSound` as arguments describing how accumulation affects error.

`FwdGraph.eval_approx`, `RevGraph.eval_approx` and `RevGraph.backprop_approx` take the graph as an
argument. No TorchLean model or `TorchLean.IR.Graph` is lowered to a `FwdGraph` or a `RevGraph`,
so the caller builds the graph and its node certificates.

Backward passes are often dominated by sums (cotangents from fanout, reductions, convolution
gradients, and parameter gradient accumulation). Floating point addition isn't associative, and
accumulation order matters. By making addition soundness a parameter, the theorem states the
arithmetic model being used.

Take a shared parameter. Two individually accurate incoming cotangents still need an accurate
addition before their sum becomes its gradient: their existing errors add, and the rounded addition
introduces another contribution. Forward-tape error also enters the local VJP, since a
multiplication backward rule uses a saved operand (whose rounded value may already differ from the
real one). And the seed has its own budget, because changing the seed changes the linear functional
being differentiated. These three sources explain why the backward theorem asks for more than
approximation of the final forward output.

# Autograd Algebra Correspondence

A bridge file, {src "TorchLean/Verify/RuntimeApprox/Graph/Autograd.lean"}[
TorchLean.Verify.RuntimeApprox.Graph.Autograd API], connects the runtime approximation
graph shape back to the autograd algebra graph.

This bridge is structural: the runtime approximation layer uses the same shape-indexed
tensor-context idea that the typed API exposes as `TensorPack`, so the conversion isn't a semantic
reinterpretation. The file defines `toNodeData` and `toGraphData`, then proves:

- `RevGraph.eval_toGraphData`
- `RevGraph.backpropCtx_toGraphData`

This conversion supports forward evaluation and reverse accumulation only. Its required JVP field
repeats the forward map (it computes no derivative), so we mustn't use the converted graph for
forward-mode differentiation. Nor do the two preservation theorems establish native kernel
correctness or prove that the supplied reverse maps are derivatives.

Put in words, the runtime approximation graph can be viewed as an autograd algebra graph once we
forget the approximation evidence and keep the same forward/VJP structure. So the two layers
compose cleanly:

in autograd algebra the reverse pass is the correct ideal VJP, and in runtime approximation the
executable reverse pass stays close to that ideal VJP.

For these results to compose, the spec reverse pass must be the one that the autograd proof
identifies with the derivative. Approximating an arbitrary supplied VJP would establish numerical
agreement without establishing derivative correctness.

Connecting those premises is where the erasure equalities help. They let a proof about a `RevGraph`
refer to the forward and backward functions executed by its erased `GraphData` without reproving
the traversal, and they preserve the supplied scalar functions as well as the graph structure. So
an application can combine a local derivative certificate and a local rounding certificate for the
same rule. But if the two certificates concern different guards, parameter orders, or selected
slopes, the erasure equality can't make those rules coincide.

# Numerical Traces For The Canonical IR

The proof-relevant `FwdGraph` and `RevGraph` explain how local approximation theorems compose.
Kernel selection, however, plans graphs of registered operators (`IR.Graph`). TorchLean
connects such a graph to executable binary32 through
{srcDir "TorchLean/Verify/NumericalCertificate"}[the graph numerical
certificate checker], without introducing a second deployment graph or a second interval type.

A certificate contains source enclosures, one derived range per IR node, and the backend selection
audit. A successful `check` stores the graph and regenerated ranges in
`Checked`, and `executeIEEE32` reads that stored graph (it accepts no other
graph argument). But the public record carries no proof that it originated from `check`. Checking
performs three executable validations:

1. validate that source and derived endpoints are finite and ordered
2. reconstruct every supported range transfer from the graph
3. re-run kernel selection and compare the selected capsules and numerical policies.

`rangeTransfer?` reads each node's transfer off its constructor of `TorchLean.Op`, never off the
model family. The transfers cover source and shape-only nodes, max, min and average pooling,
arithmetic, inverse, ReLU, SiLU (the hull of its input and zero), absolute value, directed square
root with a checked nonnegative domain, fixed-left reductions, matrix multiplication, the scaled
matrix contraction, MSE, LayerNorm, softmax, sigmoid, tanh, sine, cosine, and interval-hull rules
for max, min, and concatenation. Sum pooling and exponential have no transfer. An unsupported
operation fails at its node id instead of receiving an uninformative whole interval.

```lean (name := certEntries)
-- Inspect generation, executable coverage, replay, and
-- semantic evidence separately.
#check @generateChecked
#check @rangeTransfer?
#check @coverage
#check @executeIEEE32
#check @RealEnclosure
#check @CheckedExecution.errorTrace
#check @TensorErrorLe.of_check
#check @ErrorTrace.of_check
```

```leanOutput certEntries (whitespace := lax)
generateChecked : Backend.Profile → IR.Graph → Array SourceRange → Except String Checked
```

```leanOutput certEntries
RealEnclosure : Checked → Type
```

On failure, the generator's `Except` result preserves a diagnostic (such as a node id whose
operation has no range transfer). That message helps locate unsupported operations, but it provides
no semantic evidence about the supported ones.

`RealEnclosure` is indexed by `Checked`, so its evidence refers to the
same artifact used by replay, which prevents accidental substitution of evidence for another
certificate. The fields still have to prove the required real-semantic facts.

Between these two types sit the stages that the other checked names describe. `coverage`
reports whether each operation has a range transfer, so it can locate a missing transfer before
range propagation begins (a transfer can still reject the actual domain it receives).
`executeIEEE32` then needs concrete payloads and inputs to produce a checked execution.
`TensorErrorLe.of_check` supplies the coordinatewise interval-width argument,
`ErrorTrace.of_check` applies that argument to related arrays of node values, and
`CheckedExecution.errorTrace` packages it for the certificate's real enclosure and replay. So
operation coverage, successful evaluation, and a proved error trace answer successive, different
questions.

A `RangeTransfer` is an executable range transformer, not a soundness theorem.
`generateChecked` and `check` establish that the stored trace is exactly the trace reconstructed by
the operators' range transfers and the kernel plan. They do not establish that every reconstructed
interval encloses the graph's real denotation.

`executeIEEE32` evaluates the same `IR.Graph` with FloatLib's configured binary32 through the
FloatLib binary32 context and checks every intermediate value against the stored ranges (rejecting
NaN and infinity). That's a reference replay, not an agreement theorem for a high-throughput
backend. `RealEnclosure` carries the separate semantic evidence: its fields require both the
real denotation equality and a pointwise real enclosure proof. When that evidence is supplied,
`CheckedExecution.errorTrace` combines the real enclosure with the successful IEEE replay to
prove a pointwise error trace whose bound is the interval width. At each coordinate, the real value
and the interpreted IEEE value both lie between the same two endpoints, so their distance is at
most the upper endpoint minus the lower endpoint. The argument needs enclosure of both values, and
checking the rounded replay alone leaves its real half unproved.

Reduction order is read from the selected capsule. Ordinary reference matrix products and sums
advertise the fixed left fold used by the canonical tensor semantics. Grouped scaled products
instead advertise `fixedLeftOrExact` (matching their staged-or-exact recovery), and their
certificate rule requires that policy. LibTorch CUDA accumulations are marked
implementation-defined, so a fixed-left certificate can't accidentally certify a cuBLAS, cuDNN,
fused-attention, or parallel-reduction schedule. Those paths need the order-independent reduction
bounds described in the floating-point chapter or a stronger backend-specific contract.

The local interval lemmas for arithmetic and selected nonlinear operations follow the inclusion
principle of IEEE 1788-2015. Those of the ReLU, absolute-value, square-root, addition, subtraction,
multiplication and softmax transfers are stated on the operator's rule and registered with
`op_cert` as `enclosure` certificates. No theorem composes them into the statement that every
accepted graph trace encloses exact graph semantics, so `RealEnclosure` carries that
enclosure as evidence. The separation between local rounding facts and a composed global error
follows the standard treatment in Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd
edition.

Because the certificate replays forward values only, it shouldn't be described as a backward
certificate of the graph. Backward numerical theorems use the proof-bearing `RevGraph` path, which
erases to executable autograd `GraphData` without discarding its VJP rules. A backward certificate
of an operator graph would need an additional correspondence theorem.

An interval-width bound is deliberately coarse when the two executions share information that the
interval forgets. If both values lie in `[2,3]`, their distance is at most one (even when they're
actually identical). Narrowing the interval improves this estimate only if both enclosure premises
remain valid, which gives the range producer a concrete objective: keep intervals tight enough for
the downstream margin while preserving the real inclusion proof and successful rounded replay.
Merely shrinking a stored endpoint would invalidate one of those checks.

# MLP Numerical Certificate Example

Our executable example {src "TorchLeanExamples/DeepDives/Floats/GraphNumericalCertificate.lean"}[
GraphNumericalCertificate.lean] ends with a two-layer MLP, not a single isolated operator.

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

The file defines two certificates with their replays: a scalar example and this MLP. It also
defines a deliberately corrupted scalar certificate, in which the addition interval is replaced
with `[0,0]`. The checker regenerates the range trace and compares it with the stored one.

`MLP.certificate` checks that all ten graph nodes have a range transfer, derives every range,
selects the CPU capsules, and stores the graph, source assumptions, ranges, and backend audit in
one artifact. `MLP.replay` then supplies concrete weights, biases, and input values, executes the
stored graph with FloatLib binary32, and checks every intermediate tensor. These `Except` values
exercise range reconstruction, concrete replay, and tamper rejection separately. But the example
doesn't construct a `RealEnclosure`, so it isn't by itself a proof that the MLP's exact real
execution is enclosed.

There is no MLP-specific branch in this process. The checker sees input, constant, matrix
multiplication, addition, and ReLU nodes. Other architectures can use the same walk when their
primitive operations are covered. A primitive is covered by its arm of `rangeTransfer?`. A
semantic certificate also needs a theorem connecting that transfer's derived interval to the
operation's real semantics.

The [complete numerical-runtime
walkthrough](https://lean-dojo.github.io/TorchLean/examples/numerical-runtime/) shows the model
definitions, the backend-capsule audit, and the handoff to backward and optimizer bounds. It also
spells out the boundary: an operator graph has checked forward replay, while backward and optimizer
composition begins from a proof-bearing `RevGraph`.

# Generated Roundoff Theorems

A third route is the `roundoff` tactic: it reads an operator's Lean code and generates an error
theorem from it (an absolute bound or a mixed relative and absolute one) for every IEEE format. The
[numerical-runtime walkthrough](https://lean-dojo.github.io/TorchLean/examples/numerical-runtime/)
describes its theorems.

Coverage is registered on the operator table ({ref "registered-operators"}[Registered Operators]).
`#roundoff_coverage` runs `roundoff` on every route of the declarations it's given and records the
outcome of each. A constructor of `TorchLean.Op` has the routes of its arm of `Op.rule?`: the
forward value at each output, the tangent of each value output where the arm has a forward-mode
rule and no other route computes it, the gradient of each value input from the dense pullback,
and the prepared pullback where it is other code. We register the remaining operator
specifications, the CPU tape builders, the
gradient accumulation and the general tensor operators alongside, by family. Each route is analyzed
for a configured scalar over an arbitrary IEEE format, once for an absolute bound and once for a
mixed one, and once each for native `Float32` and for the rounded reals `FP32`.

A route is proved when its theorem has no premise and its reference and budget decode only inputs,
and conditional when it has premises or caveats. A caveat marks a budget or reference that decodes
computed values, such as a reference that branches on a comparison of computed scalars, or a proof
that uses an enclosure in place of a rounding theorem. In absolute mode most
conditional routes are conditional on computed values: such a theorem holds, but its premises,
budget or reference mention values the computation produces, so it isn't an a priori bound. A route
is unsupported when the analysis met an operation without a numerical rule. For native `Float32`
and for `FP32` all but a few routes stop there, because no rule relates their operations to rounded
real arithmetic. Inverse trigonometric functions have unsupported forward routes as well: the
analysis has no rule for the `quietNaN` their code applies, although the configured formats compute
these functions correctly rounded. We don't register the whole-graph lowerings (which have no fixed
routes), the LibTorch tape builders (whose device buffers have no real semantics here) or the IR
evaluator.

A composite route can exceed the heartbeat budget of one declaration. Such a route is then analyzed
once more with the generated theorems of the operations it calls as rules, each stated against real
references for its inputs, so that the composite bound is a chain of its parts' bounds. That
route's outcome names the theorems it was composed from (the two-pass forward of `Op.attention` is
conditional in this way).

The analysis reads a tensor entry through TorchLean's `Storage` class, and it accepts such a read
only with laws that give each method of the class a defining equality through the array observation
`Storage.toArray` (among them `Storage.get_eq_toArray_getElem`, `Storage.uget_eq_toArray_uget` and
`Storage.copyAt_eq_push_uget`). Those laws don't show that a generated reference agrees with a
separately written specification. What remains assumed is stated by FloatLib, in the trust
assumptions of `FloatLib/Floats/Formats/BinaryInterchange/Automation/Roundoff.lean`. Among them:
the types a storage class supplies stay abstract to index computations, the types that index data
mentions aren't chosen by a rounded value, and the screen for storage types is syntactic and
doesn't unfold definitions. The analysis also folds over states that are tuples, which is how
`TorchLean/Verify/RuntimeApprox/Roundoff/OnlineSoftmax.lean` generates a bound for FlashAttention's
blocked pass ({ref "attention-roundoff"}[Rounding Error In Attention]). `scripts/lake.sh lint` fails
unless every theorem of the libraries, the generated ones included, depends on at most the axioms
`propext`, `Classical.choice` and `Quot.sound`.

# NF Operations: Rounded Real Arithmetic

The largest collection of local rules sits in
{srcDir "TorchLean/Verify/RuntimeApprox/NF"}[`TorchLean/Verify/RuntimeApprox/NF`]. These
rules use FloatLib's `Floats.Formats.Flocq.NF`, a noncomputable rounded-real model whose radix,
exponent function, and valid rounder determine the representable values and arithmetic. This is the
rounded-real style used by {Informal.citet flocq2011}[] in Rocq. Expressing an operation as
rounding a real result makes its error available directly to real analysis. A bit-level model can
support such reasoning too, but it first needs decoding and a theorem relating the decoded
operation to rounding.

We compose those imported scalar facts in the local tensor, graph, and optimizer rules. Selecting
`ExecFloat.Binary e f` for a program doesn't instantiate an NF error theorem automatically: the
proof must connect the chosen bit format to the same rounding model and account for finite range
and exceptional values. Likewise, an NF exponential denotes rounding the real exponential. It isn't
a theorem about the accuracy of a configured software approximation to that function.

For common operations, the file has scalar and tensor approximation lemmas:

- arithmetic: `approx_add`, `approx_sub`, `approx_mul`, `approx_div_of_pos_lb`
- unary functions: `approx_exp`, `approx_tanh`, `approx_abs`, `approx_neg`
- guarded operations: `safeLog`, `safeDiv`, `approx_safeLogSoftplus`
- tensor rules: `ApproxTensor.add`, `ApproxTensor.mul`, `ApproxTensor.exp`,
  `ApproxTensor.relu`
- graph nodes: `mulNode`, `expNode`, `reluNode`, `safeDivNode`, `sigmoidNode`, `sumNode`.

`sigmoidNode` evaluates the branch-stable sigmoid at every entry. Axis-softmax bounds live in
`NF/Softmax.lean`. The sum rule follows `Tensor.sum`'s sequential addition order, so we cannot
apply its budget to a different parallel reduction without a bridge.

Several of these lemmas make the numerical analysis tradeoff visible. Division requires a positive
lower bound on the exact denominator that survives rounding (`approx_div_of_pos_lb`) or a guarded
form (`safeDiv`). Division's budget has a name, `divPosErrorBound`: the numerator error divided by
the surviving margin, the denominator error times the numerator magnitude plus its error divided by
the squared margin, and half an ULP of the quotient. We build the sigmoid bounds on that one
definition, and `reciprocalSigmoidBoundScalar_le_one` bounds the budget for the rounded sequence
$`1/(1+\exp(-x))` by one under small-error hypotheses. A mean divides by a count, and the count is
a natural number, not a rounded denominator: `Context.divNat` rounds the exact quotient once. So
`approx_meanRow` bounds a row mean by half an ULP of the rounded row sum over the count plus the
row-sum budget over the count, at every positive count, with no denominator margin.

TorchLean's public sigmoid uses $`1/(1+\exp(-x))` on positive inputs and $`\exp(x)/(1+\exp(x))`
otherwise. Both formulas describe the same real function, but their rounded arithmetic differs (the
second sequence also rounds the numerator exponential). `approx_sigmoid` uses the budget for the
branch selected by the rounded input, and its proof allows the real input and rounded input to lie
on different sides of zero, because each branch approximates the same real sigmoid.

Square root and log need domain protection. Exponential uses a mean value bound, and multiplication
propagates both input errors and a product term. Those are the places where a proof over real
numbers would be too optimistic if copied onto a float implementation.

```lean (name := nfOps)
-- The denominator margin links the bound expression to a
-- valid division theorem.
#check @divPosErrorBound
#check @approx_div_of_pos_lb
#check @reciprocalSigmoidBoundScalar_le_one
#check @approx_sigmoid
#check @approx_meanRow
```

```leanOutput nfOps (whitespace := lax)
@divPosErrorBound : {β : Radix} →
  {fexp : ℤ → ℤ} → [ValidExp fexp] → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ
```

The displayed `divPosErrorBound` signature spells out what the budget depends on.

Its five real arguments are the exact denominator lower bound `η`, the input-error budgets `epsx`
and `epsy`, and the interpreted runtime operands `xhat` and `yhat`. The associated theorem requires
`epsy < η`: denominator error leaves an effective separation `η - epsy` from zero. The budget
scales the numerator error by the reciprocal of that separation and the denominator error by its
squared reciprocal times `abs xhat + epsx`, then adds rounding of the quotient.

As the surviving separation shrinks, the bound grows. A caller can compare the expression with a
desired margin, but it's a noncomputable real-valued bound, not an executable estimator, and its
ULP terms need not be smooth.

Say the exact denominator lower bound is two and the denominator error is at most one quarter: the
rounded denominator is then at least seven quarters, and that surviving margin is what permits the
reciprocal estimate. Letting the error approach two would remove this argument, even if one
particular rounded denominator happened to remain positive. The theorem's uniform bound must
justify every value admitted by its premises, not only a favorable sample. The squared reciprocal
in the denominator-error term makes this loss of separation especially costly.

A safe division example has three pieces: the mathematical value is a guarded division, the runtime
value is computed by `safeDivR eps xR yR`, and the theorem states that the runtime value
approximates the guarded spec value under the declared tolerance.

That theorem covers the guarded function, including its declared behavior at $`y=0`, so to use it
for ordinary division, a caller must also show that neither execution activates the guard.

## Error Bounds For Reductions And Softmax

Reductions and normalization layers create coupled error terms, because many inputs flow into one
output. TorchLean's reduction approximation lemmas are in
{src "TorchLean/Verify/RuntimeApprox/NF/ReductionOps.lean"}[
TorchLean.Verify.RuntimeApprox.NF.ReductionOps]:

```lean (name := reductions)
-- Row and column reductions follow different index
-- sequences and expose different budgets.
#check @reduce_sum_rows
#check @reduce_mean_rows
#check @reduce_sum_columns
```
```leanOutput reductions (whitespace := lax)
@reduce_sum_rows : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd] {m n : ℕ} {xS : Tensor ℝ [m, n]}
  {xR : Tensor (NF β fexp rnd) [m, n]} {eps : ℝ},
  ApproxTensor NF.toReal xS xR eps →
    ApproxTensor NF.toReal
      (Tensor.castShape («reduce» ReduceOp.sum xS [⟨1, Nat.one_lt_two⟩] false ⋯) ⋯)
      (Tensor.castShape («reduce» ReduceOp.sum xR [⟨1, Nat.one_lt_two⟩] false ⋯) ⋯)
      (linfNorm (sumRowBoundVec eps xR))
```
```leanOutput reductions (whitespace := lax)
@reduce_mean_rows : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd] {m n : ℕ} {xS : Tensor ℝ [m, n]}
  {xR : Tensor (NF β fexp rnd) [m, n]} {eps : ℝ},
  ApproxTensor NF.toReal xS xR eps →
    Shape.NonemptyAxis 1 [m, n] →
      ApproxTensor NF.toReal
        (Tensor.map (fun s => Context.divNat s n)
          (Tensor.castShape («reduce» ReduceOp.sum xS [⟨1, Nat.one_lt_two⟩] false
              ⋯) ⋯))
        (Tensor.map (fun s => Context.divNat s n)
          (Tensor.castShape («reduce» ReduceOp.sum xR [⟨1, Nat.one_lt_two⟩] false
              ⋯) ⋯))
        (linfNorm (meanRowBoundVec eps xR))
```
```leanOutput reductions (whitespace := lax)
@reduce_sum_columns : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd] {m n : ℕ} {xS : Tensor ℝ [m, n]}
  {xR : Tensor (NF β fexp rnd) [m, n]} {eps : ℝ},
  ApproxTensor NF.toReal xS xR eps →
    ApproxTensor NF.toReal
      (Tensor.castShape («reduce» ReduceOp.sum xS [⟨0, Nat.two_pos⟩] false ⋯) ⋯)
      (Tensor.castShape («reduce» ReduceOp.sum xR [⟨0, Nat.two_pos⟩] false ⋯) ⋯)
      (linfNorm (sumColumnBoundVec eps xR))
```

The row-sum theorem accounts for the number of accumulated terms and for the same row/column
indexing the executable reducer uses. That's the sort of detail that matters for LayerNorm,
attention logits, pooled features, and minibatch losses.

For row sums, `sumRowBoundVec eps xR` contains one accumulated error budget per row.
`linfNorm (sumRowBoundVec eps xR)` takes the maximum of those entries, producing the scalar
budget in the theorem's conclusion. Its dependence on `xR` records the magnitudes encountered during
the fold. Row means use `meanRowBoundVec` to include rescaling and the division's rounding, and
column sums use `sumColumnBoundVec` with the other axis.

These expressions are derived bounds, and a caller may replace one with a simpler upper bound after
proving that it dominates the derived expression. A row bound doesn't automatically cover a column
reduction, because the two follow different sequences of operands.

Taking the maximum across coordinate budgets changes the precision of the estimate, not the
quantity being compared: it yields one allowance that works for every output entry, and it doesn't
sum the errors across entries. That suits a later coordinatewise margin argument. If a later
operation does sum those entries, its own reduction rule must propagate their uncertainty and add
the rounding that sum introduces.

Reduction order also affects the computed result. Both rows below have exact sum `2`, but left and
right association need not agree:

```lean (name := raOrder)
section
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

Both orders are exact on the small-magnitude row. In the other row, left association loses one unit
contribution when it's added to a much larger value, and right association loses both, so the
outputs are `1` and `0`. That absorption depends on the operands' scale, which is why the local
bound follows the runtime values as well as the number of terms. Our proof follows the same fold
order as the reducer. {Informal.citet goldberg1991}[] explains the underlying rounding effect.

`Shape.NonemptyAxis 1 [m, n]` records that the selected axis has positive length. We require this
evidence in the reduction APIs, and the mean proof uses the resulting `0 < n` fact when bounding
division by the row count.

Softmax needs even more care. Elementwise sigmoid bounds aren't a proof of axis softmax, because
axis softmax couples every coordinate through the denominator. TorchLean's
{src "TorchLean/Verify/RuntimeApprox/NF/Softmax.lean"}[axis softmax approximation API] proves
the conditional NF rounded-real forward theorem `ApproxTensor.softmaxVec`, which accounts for
max subtraction, exponential approximation, a sequential denominator sum, and division, under a
stated denominator-error margin. `ApproxTensor.softmax_rows` lifts the result rowwise.

Hard masking uses exact Boolean mask semantics, including an exact-zero theorem for an all-blocked
row. `HardMaskedRowsEvidence` records the selected maxima, their approximation proofs, positive
real denominator lower bounds, and rounded denominator margins required by
`ApproxTensor.hardMaskedSoftmax_of_max`. Backward bounds come from
`ApproxTensor.softmaxBackwardFromWeights` and `ApproxTensor.softmaxBackward`. Three analytic facts
(`Vec.sum_softmax`, `Vec.sum_softmaxJVP`, and `Vec.abs_softmaxJVP_le_two_mul`) establish
normalization, zero-sum JVP coordinates, and the dimension-independent bound
$`\lvert\operatorname{jvp}_i\rvert\le 2G`. Since the softmax Jacobian is symmetric, the same
formula also serves as the VJP.

These are NF rounded-real theorems, not automatic claims about FloatLib binary32, a fused attention
kernel, or native binary32. And the numerical certificate's softmax transfer only derives the coarse
range $`[0,1]`, which isn't a forward-error theorem.

Max subtraction gives the real softmax denominator a useful scale. In a nonempty row, a maximum
entry has shifted score zero and hence exponential one, while the other shifted exponentials are
positive and at most one, which is why the denominator can be bounded away from zero before rounded
errors are introduced. A hard-masked row needs an allowed entry to make that argument (an
all-blocked row instead follows the separately declared zero convention). The evidence object
records which case applies, so the division proof never obtains positivity from an entry the mask
has excluded.

## Normalization And Attention

Normalization and attention compose several domain-sensitive operations, so we record an
intermediate error trace instead of assigning one unexplained tolerance to the layer. The
{src "TorchLean/Verify/RuntimeApprox/NF/Normalization.lean"}[normalization approximation API]
handles arbitrary tensor rank once the selected mean and variance reductions have been certified,
and its stages (centering, variance stabilization, square root, division, and affine) expose the
lower bounds needed to keep the denominator away from zero. In particular,
`ApproxTensor.normalize` assumes approximation evidence for the input, mean, variance, scale,
bias, and epsilon, plus a positive exact stabilized-variance lower bound and strict rounded-error
margins. It doesn't derive the mean and variance reduction bounds itself.

For BatchNorm, the {src "TorchLean/NN/Norm/BatchNorm.lean"}[BatchNorm
autograd proof] connects the ideal reverse rule to the mathematical forward map, and its theorem
`BatchNorm.jvp_backward_adjoint` covers an arbitrary spatial `Shape` and all three
cotangents (input, scale, and bias). So the rounded normalization bounds can be read against a
proved ideal reverse rule, not against an independently written gradient formula.

Our {src "TorchLean/Verify/RuntimeApprox/NF/Attention.lean"}[attention approximation API] builds
scaled dot-product attention {Informal.citep transformer2017}[] from matrix multiplication,
scaling, stable axis softmax, and a second matrix multiplication. A hard attention mask is
semantic: blocked entries have zero softmax numerator. We don't represent it by adding a large
finite negative constant, which would change the function for sufficiently large logits. So backend
capsules must advertise a matching mask convention before their output can inherit this theorem.
`scaledDotProduct_masked`, the masked theorem, consumes
`HardMaskedRowsEvidence`, and the canonical inverse-square-root scale theorem also requires a
positive feature dimension and a square-root margin. These are conditional NF approximation
theorems, not proofs for arbitrary fused attention implementations. A kernel in the style of
{Informal.citet flashattention2022}[] computes the softmax in one pass over tiles of keys and
accumulates in a different order, so these theorems don't describe it. TorchLean states that
schedule as `Softmax.Schedule.online` and bounds its rounding in
{ref "attention-roundoff"}[Rounding Error In Attention], but a native kernel still needs an
agreement statement with that schedule.

```lean (name := normAttn)
-- The trace records intermediate errors, and the theorems
-- require the domain margins to hold.
#check @normalizeErrorTrace
#check @ApproxTensor.normalize
#check @scaledDotProduct_masked
```

In `normAttn`, the first signature constructs a trace and the second proves what its final bound
means when the supplied reduction and domain evidence hold. The theorem takes two error margins.
The first asks the error in variance plus epsilon to stay below the exact stabilized-variance lower
bound. The proof doesn't use it, since the square root `Tensor.sqrt _ .clampNegative` takes the
root of `max x 0`. The second, which the proof uses, asks the square-root error to stay below the
square root of that lower bound (keeping the division denominator positive). The theorem also takes
broadcasting evidence for the reduced statistics, scale, and bias, so these quantities are compared
at the coordinates where the normalized tensor uses them.

The masked attention theorem consumes the row evidence at the softmax stage between its two matrix
products: the first product's error becomes score error, and the second combines uncertain weights
with uncertain values. Under the hard-mask convention a blocked entry contributes an exact zero
softmax weight, whereas an allowed entry participates in the denominator and its error bound.
Keeping that distinction through the composition matters even if both masks happen to produce nearly
identical outputs on an ordinary test sentence.

# Rank-Polymorphic Convolution

Convolution is parameterized by the number of spatial axes, with one kernel, stride, and padding
entry for each axis, so one definition describes sequence, image, volume, and higher-rank
convolutions. The verifier's lowering of a convolution node is connected to the IR semantics by the
convolution step of the recorded typed graph
({src "TorchLean/NN/IR/OpGraph.lean"}[`TorchLean.IR.OpGraph`]), whose erasure evaluates as the typed
graph does.

On the ideal side, {src "TorchLean/NN/Conv/FDeriv.lean"}[Convolution
FDeriv] holds the autograd proof: it proves the Fréchet derivative of the rank-general forward map
and shows that the implemented kernel, bias, and input reverse rules are adjoint to that
derivative.

Then the {src "TorchLean/Verify/RuntimeApprox/NF/Convolution.lean"}[rounded convolution proof]
follows the implementation's actual nested loops. `approx_conv_forward` bounds one forward
coordinate, and the kernel, bias, and input-gradient theorems bound the three fields
`Conv.backward` returns. Padding branches, multiplication error, and every rounded accumulation
are included. None of the proofs reorders the sums or assumes floating-point associativity.

```lean (name := convLemmas)
-- Match each of the three returned convolution gradients
-- with its coordinate error theorem.
#check @approx_conv_forward
#check @approx_conv_kernelDeriv
#check @approx_conv_biasDeriv
#check @approx_conv_inputDeriv
```

Those four convolution signatures show which operands affect each result. The forward coordinate
theorem takes approximation of the kernel, bias, and input, then selects an output channel and
spatial multi-index. A kernel-gradient coordinate instead needs the input and output-cotangent
bounds, because that gradient sums products of those two quantities. The bias gradient needs only
the output-cotangent bound (it sums those cotangents), and the input gradient needs the kernel and
output-cotangent bounds. These different premise lists follow the backward formulas, and they
aren't four copies of one undifferentiated layer tolerance.

Each conclusion concerns the chosen coordinate and the fold that computes it. You can get a
tensor-wide budget by bounding these coordinate budgets uniformly, but the coordinate theorems keep
the dependence on channel, location, and padding. Near a padded boundary, some forward products use
the specified zero value, and the proof handles those branches in the same index traversal. It
doesn't replace the convolution by a larger unqualified matrix product whose accumulation order
might differ.

# Rounded Optimizer Steps

The backward theorem produces approximate gradients, but a training claim must still account for
the optimizer arithmetic. The generic
{src "TorchLean/Verify/RuntimeApprox/OptimizerContract.lean"}[optimizer numerical contract] records
an exact state, a rounded state, their relation, a one-step bound transformer, and the proof that
the relation survives one update. `OptimizerContract.runApprox_of_stepStreamApprox` then proves the
corresponding finite-run result once, for every optimizer satisfying that interface.

In the concrete {src "TorchLean/Verify/RuntimeApprox/NF/Optimizers.lean"}[NF optimizer proofs], we
work from the optimizer equations themselves:

- SGD propagates learning-rate, gradient, and parameter error.
- Momentum SGD also propagates the momentum-buffer error.
- AdamW records errors for both moments, bias correction, square root, adaptive division, decoupled
  weight decay, and the final subtraction.

For AdamW, we bound the complete denominator, including epsilon, instead of requiring a positive
second moment. The square-root estimate remains valid at zero, while division needs the
denominator's rounding error to stay below its lower bound. This is the numerical counterpart of
the recurrence in Kingma and Ba's Adam paper (https://arxiv.org/abs/1412.6980) and of the decoupled
decay {Informal.citet adamw2019}[] introduced. Alongside those update stages, the theorem records
approximation budgets for the state and derived scalars, and a margin for division.

```lean (name := optimizers)
-- Stateful optimizers must preserve the state relation as
-- well as parameter closeness.
#check @OptimizerContract.runApprox_of_stepStreamApprox
#check @SGD.approxTensor_update
#check @MomentumSGD.approxTensor_update
#check @AdamW.approxTensor_update
```
```leanOutput optimizers (whitespace := lax)
@OptimizerContract.runApprox_of_stepStreamApprox : ∀ {R : Type} {toSpec : R → ℝ} (contract :
  OptimizerContract R toSpec)
  {shape : Shape} {exact : OptimizerStep ℝ shape (contract.ExactState shape)}
  {runtime : OptimizerStep R shape (contract.RuntimeState shape)}
  {error : OptimizerContract.StepError contract.StateError shape} {steps : Array (contract.StepInput
    shape)},
  contract.StepStreamApprox exact runtime error steps →
    contract.stateApprox exact.optimizerState runtime.optimizerState error.optimizerStateError →
      ApproxTensor toSpec exact.parameters runtime.parameters error.parameterError →
        contract.RunApprox exact runtime error steps
```
```leanOutput optimizers (whitespace := lax)
@SGD.approxTensor_update : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd]
  {s : Shape} {stateS : SGD.State ℝ s} {stateR : SGD.State (NF β fexp rnd) s} {lrError : ℝ}
  {exactParameters : Tensor ℝ s} {runtimeParameters : Tensor (NF β fexp rnd) s} {parameterError : ℝ}
  {exactGradients : Tensor ℝ s} {runtimeGradients : Tensor (NF β fexp rnd) s} {gradientError : ℝ},
  SGD.StateApprox stateS stateR lrError →
    ApproxTensor NF.toReal exactParameters runtimeParameters parameterError →
      ApproxTensor NF.toReal exactGradients runtimeGradients gradientError →
        ApproxTensor NF.toReal (SGD.update stateS exactParameters exactGradients).parameters
          (SGD.update stateR runtimeParameters runtimeGradients).parameters
          (SGD.stepError lrError parameterError gradientError stateR runtimeParameters
            runtimeGradients).parameterError
```
```leanOutput optimizers (whitespace := lax)
@MomentumSGD.approxTensor_update : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd] {s : Shape} {stateS : MomentumSGD.State ℝ s} {stateR : MomentumSGD.State
    (NF β fexp rnd) s}
  {stateError : MomentumSGD.StateError s} {paramsS : Tensor ℝ s} {paramsR : Tensor (NF β fexp rnd)
    s} {paramsError : ℝ}
  {gradsS : Tensor ℝ s} {gradsR : Tensor (NF β fexp rnd) s} {gradsError : ℝ},
  MomentumSGD.StateApprox stateS stateR stateError →
    ApproxTensor NF.toReal paramsS paramsR paramsError →
      ApproxTensor NF.toReal gradsS gradsR gradsError →
        have nextError := MomentumSGD.stepError stateError paramsError gradsError stateR paramsR
          gradsR;
        MomentumSGD.StateApprox (MomentumSGD.update stateS paramsS gradsS).optimizerState
            (MomentumSGD.update stateR paramsR gradsR).optimizerState nextError.optimizerStateError
              ∧
          ApproxTensor NF.toReal (MomentumSGD.update stateS paramsS gradsS).parameters
            (MomentumSGD.update stateR paramsR gradsR).parameters nextError.parameterError
```
```leanOutput optimizers (whitespace := lax)
@AdamW.approxTensor_update : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd]
  {s : Shape} {stateS : AdamW.State ℝ s} {stateR : AdamW.State (NF β fexp rnd) s} {stateError :
    AdamW.StateError s}
  {derivedErrors : DerivedErrors} {paramsS : Tensor ℝ s} {paramsR : Tensor (NF β fexp rnd) s}
    {paramsError : ℝ}
  {gradsS : Tensor ℝ s} {gradsR : Tensor (NF β fexp rnd) s} {gradsError η : ℝ},
  AdamW.StateApprox stateS stateR stateError →
    ApproxTensor NF.toReal paramsS paramsR paramsError →
      ApproxTensor NF.toReal gradsS gradsR gradsError →
        |(1 - stateR.beta1).toReal - (1 - stateS.beta1)| ≤ derivedErrors.oneMinusBeta1 →
          |(1 - stateR.beta2).toReal - (1 - stateS.beta2)| ≤ derivedErrors.oneMinusBeta2 →
            |(1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta1)).toReal -
                    1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta1)| ≤
                derivedErrors.firstMomentBiasInverse →
              |(1 / (1 - npowRec (stateR.stepCount + 1) stateR.beta2)).toReal -
                      1 / (1 - npowRec (stateS.stepCount + 1) stateS.beta2)| ≤
                  derivedErrors.secondMomentBiasInverse →
                |(stateR.lr * stateR.weightDecay).toReal - stateS.lr * stateS.weightDecay| ≤
                  derivedErrors.decayScale →
                  (have nextStepCount := stateS.stepCount + 1;
                    have moment2 := stateS.secondMoment.scale stateS.beta2 + gradsS.square.scale (1
                      - stateS.beta2);
                    have corrected := moment2.scale (1 / (1 - npowRec nextStepCount stateS.beta2));
                    Forall (fun z => η ≤ z) (corrected.sqrt SqrtPolicy.clampNegative + Tensor.full s
                      stateS.epsilon)) →
                    (traceStep stateError derivedErrors paramsError gradsError η stateR paramsR
                      gradsR).denominator <
                        η →
                      have trace := traceStep stateError derivedErrors paramsError gradsError η
                        stateR paramsR gradsR;
                      AdamW.StateApprox (AdamW.update stateS paramsS gradsS).optimizerState
                          (AdamW.update stateR paramsR gradsR).optimizerState
                          { lr := stateError.lr, beta1 := stateError.beta1, beta2 :=
                            stateError.beta2,
                            epsilon := stateError.epsilon, weightDecay := stateError.weightDecay,
                            firstMoment := trace.firstMoment, secondMoment := trace.secondMoment } ∧
                        ApproxTensor NF.toReal (AdamW.update stateS paramsS gradsS).parameters
                          (AdamW.update stateR paramsR gradsR).parameters trace.parameterError
```

What state an optimizer retains determines which relations must survive an update.

SGD has no momentum buffer or step counter, so its conclusion is a single `ApproxTensor` on the
updated parameters, and its only state hypothesis is `SGD.StateApprox` (which relates two learning
rates). Momentum SGD keeps one buffer, and its conclusion becomes a conjunction: a bound for the
next momentum buffer *and* a bound for the parameters. Because the next update needs the bound on
the buffer produced by the current one, returning both relations makes the conclusion usable as
that update's hypothesis.

AdamW keeps two buffers, a step counter, and a bias correction, and its statement is by far the
longest in the chapter. Five of its hypotheses are error budgets for quantities Adam derives but
doesn't store: `1 - beta1`, `1 - beta2`, and the two bias-correction reciprocals
`1 / (1 - beta^(t+1))`, together with `lr * weightDecay`. Those are computed in floating point on
every step, so they carry their own rounding (including the arithmetic used for the powers). The
`Tensor.Forall` hypothesis bounds every entry of the exact denominator,
`sqrt(max correctedSecondMoment 0) + epsilon`, from below by `eta`, and the accumulated denominator
error must stay strictly below `eta`. Because the error bound is nonnegative, this strict margin
also establishes that `eta` is positive.

At a zero-gradient first step, the second moment can be zero, and epsilon supplies the denominator.
Since the square-root estimate uses `|sqrt(a) - sqrt(b)| ≤ sqrt(|a - b|)` for nonnegative
arguments, it needs no positive lower bound on the second moment. But positive epsilon doesn't
settle the whole numerical claim: the application must still show that the denominator error is
smaller than its chosen margin. These are conditions for the NF proof model, which doesn't model
NaNs.

`OptimizerContract.runApprox_of_stepStreamApprox` composes updates for any supplied contract. Its
`StepStreamApprox` hypothesis relates the sequence of step inputs, and its conclusion `RunApprox`
relates the resulting states over the finite run. To apply it to an optimizer, you must instantiate
the contract and supply its state and step evidence, which a local update inequality doesn't supply
by itself.

Over several updates, the state relation an update produces is part of the input to the following
one, and resetting the momentum or moment error to zero after an update would discard accumulated
uncertainty. Likewise, selecting the same minibatch sequence doesn't by itself supply
`StepStreamApprox`: the gradients must be related at the exact and rounded states actually reached.
The finite-run theorem organizes this induction once those step premises are available. It doesn't
infer a bound on an evolving gradient stream from a gradient comparison performed only at
initialization.

# Weakening Tolerances

Weakening an absolute bound to an absolute-plus-relative tolerance needs no magnitude hypothesis:
`ApproxTol.ofReal` builds the tolerance, `ApproxWith.approxWithTol_absOnly` turns an absolute
bound into the absolute-only tolerance, and `ApproxWithTol.mono` loosens it. But the result retains
the full absolute error budget, and dividing that budget by an upper magnitude bound doesn't
establish a relative-only error estimate, especially for values near zero.

# FP32 And Verification Margins

The {src "TorchLean/Verify/RuntimeApprox/NF/Layers.lean"}[linear-layer bounds],
{src "TorchLean/Verify/RuntimeApprox/NF/MLP.lean"}[MLP bounds] and
{src "TorchLean/Verify/RuntimeApprox/NF/CROWN.lean"}[CROWN/IBP box bounds] of the `NF` model connect
the approximation style to layerwise and verifier reasoning. `FP32` is their instance at binary32.

Runtime approximation meets certification in the CROWN connection. A verifier over the reals may
prove a margin, but a float runtime can differ from ideal real arithmetic. To transfer the claim,
the real margin must dominate the runtime approximation budget: the certified lower margin, after
subtracting the runtime error, still has to be positive.

When that inequality is proved, the runtime prediction stays certified.

The Float32 soundness layer keeps the same separation: `FP32` is a proof model, FloatLib binary32
is an executable bit oriented model, and native hardware remains a named assumption unless a bridge
theorem covers the path being used.

# Classifier Margin Example

Suppose we have a two layer classifier whose hidden layer is $`\operatorname{ReLU}(W_1x+b_1)` and
whose output is $`W_2h+b_2`.

An ideal proof might establish that, for all inputs in a box, the margin over the reals for class
$`0` over class $`1` is at least $`0.05`.

A float deployment theorem adds the runtime approximation statement: for every input in the same
box, $`y_{\mathrm{run}}` approximates $`y_{\mathrm{real}}` within $`0.01` per relevant logit.

Then the margin proof must be adjusted:

$$`f_0^{\mathrm{run}}(x)-f_1^{\mathrm{run}}(x)
\ge 0.05-0.01-0.01
=0.03>0.`

In the unfavorable case, class zero moves down by its error budget and class one moves up by its
budget, and subtracting both leaves a positive margin for every input covered by both premises. A
pointwise error estimate at one sample wouldn't justify this claim over the whole box.

Equal logit budgets are convenient but unnecessary. If the winner's error is bounded by one
quantity and the competitor's by another, their sum is the amount subtracted from the real pairwise
margin, and with several classes the calculation must succeed against each competitor. Unequal
budgets can retain useful information when some output coordinates have much tighter numerical
bounds than others. In every case, the input region and parameter interpretation must agree between
the real margin theorem and the approximation theorem that transfers it.

# End To End Rounded Training

In the rounded-real end-to-end file, we connect the proof-bearing reverse graph to executable
autograd `GraphData`, extract typed parameter gradients, and compose those gradients with the
optimizer contracts:

```lean (name := endToEnd)
-- Follow the bounds from a graph context to one indexed
-- parameter update.
#check @eval_approx_graphData
#check @backprop_approx_graphData
#check @gradient_approx_graphData
#check @update_approx_graphData
#check @traceTrainingStep
```

```leanOutput endToEnd (whitespace := lax)
@update_approx_graphData : ∀ {β : Radix} {fexp : ℤ → ℤ} [inst : ValidExp fexp] {rnd : ℝ → ℤ}
  [ValidRndToNearest rnd]
  {Γ ss : List Shape} (g : RevGraph NF.toReal Γ ss) (i : Fin Γ.length)
  (contract : OptimizerContract (NF β fexp rnd) NF.toReal) (xS : TensorPack ℝ Γ) (xR : TensorPack
    (NF β fexp rnd) Γ)
  (epsIn : EList Γ) (seedS : TensorPack ℝ (Γ ++ ss)) (seedR : TensorPack (NF β fexp rnd) (Γ ++ ss))
  (epsSeed : EList (Γ ++ ss)) (paramsS : Tensor ℝ (Γ.get i)) (paramsR : Tensor (NF β fexp rnd)
    (Γ.get i))
  (paramsError : ℝ) (stateS : contract.ExactState (Γ.get i)) (stateR : contract.RuntimeState (Γ.get
    i))
  (stateError : contract.StateError (Γ.get i)) (assumptions : contract.StepAssumptions (Γ.get i)),
  ApproxCtx NF.toReal xS xR epsIn →
    ApproxCtx NF.toReal seedS seedR epsSeed →
      ApproxTensor NF.toReal paramsS paramsR paramsError →
        contract.stateApprox stateS stateR stateError →
          (have exactGradients := g.backpropReal xS seedS;
            have runtimeGradients := g.toGraphData.backpropCtx xR () seedR;
            have gradientError := Pack.get (g.backpropBounds epsIn xR epsSeed seedR fun {Δ} =>
              ctxAddBound) i;
            contract.assumptionsHold stateS stateR stateError paramsS paramsR paramsError (Pack.get
              exactGradients i)
              (Pack.get runtimeGradients i) gradientError assumptions) →
            have exactGradients := g.backpropReal xS seedS;
            have runtimeGradients := g.toGraphData.backpropCtx xR () seedR;
            have gradientError := Pack.get (g.backpropBounds epsIn xR epsSeed seedR fun {Δ} =>
              ctxAddBound) i;
            have nextError :=
              contract.nextError stateError paramsError gradientError stateR paramsR (Pack.get
                runtimeGradients i)
                assumptions;
            contract.stateApprox (contract.updateExact stateS paramsS (Pack.get exactGradients
              i)).optimizerState
                (contract.updateRuntime stateR paramsR (Pack.get runtimeGradients i)).optimizerState
                nextError.optimizerStateError ∧
              ApproxTensor NF.toReal (contract.updateExact stateS paramsS (Pack.get exactGradients
                i)).parameters
                (contract.updateRuntime stateR paramsR (Pack.get runtimeGradients i)).parameters
                nextError.parameterError
```

The statement takes the reverse graph `g`, an input-context index `i`, related exact and runtime
contexts, related seeds, parameter tensors, and optimizer states, and it derives the two gradients
and their error from that graph and those contexts. Callers supply the remaining evidence through
`contract.assumptionsHold` (including any domain margins the optimizer requires).

Applying this as a training theorem takes one more choice. `paramsS` and `paramsR` are separate
arguments: the signature doesn't assert that they equal entry `i` of `xS` and `xR`. The theorem
justifies updating the supplied tensors using the gradient extracted at index `i`. For a training
step that updates the parameters at which the loss was differentiated, instantiate the parameter
arguments with those context entries or provide the corresponding equalities.

These are graph-level bridge theorems:

- `eval_approx_graphData` says evaluating the executable forward graph is close to evaluating the
  spec forward graph when the local node approximation obligations have been supplied.
- `backprop_approx_graphData` says the same style of statement for reverse accumulation, with the
  accumulation error model still spelled out.
- `update_approx_graphData` applies any numerical optimizer contract to one
  typed parameter gradient produced by that executable reverse pass. SGD uses trivial step
  evidence, while AdamW supplies its bias-correction and denominator margins through
  `StepAssumptions`. A model with several parameter tensors applies the same theorem at each
  parameter index.
- `traceTrainingStep` computes a proof-free report of forward, backward, gradient, parameter, and
  optimizer-state bounds. Its interpretation comes from the surrounding approximation theorems, not
  from the report record itself.

In the printed theorem, the index `i` chooses a tensor from the input context, so its gradient has
shape `Γ.get i`, and `gradient_approx_graphData` extracts precisely that component from
the whole-context reverse result. That's why one graph theorem can support parameters of different
shapes without flattening them into a common array. Applying the update theorem at several indices
still requires a consistent step, though: the contexts and seeds must describe the same loss
evaluation, and each optimizer state must belong to the parameter tensor it updates.

## Error Propagation Through An Optimizer Step

Take a parameter tensor at index $`i`, with $`g_{\mathrm{spec}}` from the real reverse pass and
$`g_{\mathrm{run}}` from executable rounded reverse mode. Then `gradient_approx_graphData`
proves

$$`\|\operatorname{toSpec}(g_{\mathrm{run}})-g_{\mathrm{spec}}\|_\infty
\le \varepsilon_g.`

Suppose the current parameters and learning rate have errors $`\varepsilon_p` and
$`\varepsilon_{\mathrm{lr}}`. For SGD,
the two executions perform the same equation in their respective scalar systems,

$$`p' = p-\eta g.`

`SGD.stepError` first bounds the rounded product $`\eta g`, including both input errors and the
multiplication's own rounding, then bounds the final subtraction. From the graph-level theorem we
get

$$`\|\operatorname{toSpec}(p'_{\mathrm{run}})-p'_{\mathrm{spec}}\|_\infty
\le \varepsilon_{p'}`

with $`\varepsilon_{p'}` equal to that computed bound, not a user-chosen test tolerance. Momentum
SGD uses the same theorem and also returns a bound for the updated momentum buffer.

For AdamW the route is longer: update both moments, apply bias correction, take the second-moment
square root, form the adaptive learning rate, apply decoupled decay, and subtract the update.
`AdamW.ErrorTrace` keeps the error after each stage (one field per stage, from `squaredGradient`
through `parameterError`). What makes those numbers meaningful is the evidence in
`StepAssumptions`: `minimumDenominator` lies below every entry of
`sqrt(max correctedSecondMoment 0) + epsilon`, alongside `DerivedErrors` for the rounded
scalars `1 - beta1`, `1 - beta2`, the two reciprocal bias corrections, and the product
`lr * weightDecay`. That margin must stay above the denominator error. A square-root bound valid at
zero accounts for the second-moment error, so a zero-gradient first step isn't excluded (epsilon
supplies its positive denominator). Nonzero moments can supply a larger margin. These are proof
obligations, not executable tests.

These proofs describe numerical recurrences for explicit states, and they don't automatically cover
the native trainer's parameter lookup, mutable moment-buffer allocation, per-parameter clocks, or
checkpoint restore. Before `OptimizerContract.runApprox_of_stepStreamApprox` can be cited for a
concrete training process, those runtime mechanisms must be shown to instantiate the same state and
step relation.

# Runtime Agreement

For supported graph and operator fragments, runtime approximation proves that a runtime or rounded
computation stays within a stated tolerance of a spec computation. LibTorch execution,
compiler rewrites, and PyTorch-exported graphs need their own agreement statements when a
claim is about those paths.

For a deployment claim, identify the graph, scalar interpretation, execution path, and input region
in each premise: the numerical bound must apply to that same computation throughout the region used
by the real theorem.

When the approximation theorem is present, the bridge is proved. When it isn't, the claim should
say which runtime path or external producer supplies the remaining evidence.
