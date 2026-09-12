import VersoManual
import NN.API
import NN.Examples.BugZoo.ShapeAndBroadcast
import NN.Examples.BugZoo.StableLoss
import NN.Examples.BugZoo.IgnoredLabelLoss
import NN.Examples.BugZoo.AutogradDomain
import NN.Examples.BugZoo.CompilerBoundary
import NN.Examples.BugZoo.FloatBoundary
import NN.Examples.BugZoo.NormalizationState
import NN.Examples.BugZoo.LayerNormDegenerateAxis
import NN.Examples.BugZoo.ConstantNormalizationSlice
import NN.Examples.BugZoo.BatchInvariance
import NN.Examples.BugZoo.KVCache
import NN.Examples.BugZoo.RoPEPosition
import NN.Examples.BugZoo.TokenizerBoundary
import NN.Examples.BugZoo.Geometry3DProjection
import NN.Proofs.Models.Attention.CausalMask
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open NN.Examples.BugZoo
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "BugZoo Catalog" =>
%%%
tag := "bugzoo-catalog"
%%%

A causal-attention optimization can preserve the output shape while allowing a token to attend to
the future. A shape check cannot detect that change; the relevant condition is that every
strict-future attention weight is zero. BugZoo collects failures in frameworks, compilers,
deployment tools, and serving systems, and identifies the object or proposition needed to express
each intended behavior.

The {srcDir "NN/Examples/BugZoo"}[BugZoo directory] focuses on the TorchLean fragment itself: once a
computation enters a typed TorchLean spec, shape changes, masks, token bounds, finite precision
choices, stateful normalization parameters, and backend semantics become named objects that can be
checked. The failure families are not invented for the occasion. Compiler fuzzing found silent
wrong-code bugs across TVM, TensorRT, ONNXRuntime, and PyTorch
{Informal.citep nnsmith2023}[], coverage-guided fuzzing found rare numerical failures in trained
models {Informal.citep tensorfuzz2019}[], and attention masking has its own long tail of polarity
and layout mistakes going back to the original architecture {Informal.citep transformer2017}[].

Each entry isolates a contract such as “strict-future attention weight is zero.” The motivating
incident explains where the condition can fail, and the Lean declaration states the inputs and
hypotheses under which it holds.

The examples below pair TorchLean reference computations with captured PyTorch runs
{Informal.citep pytorch2019}[]. The `leanOutput` blocks record elaborator output; the PyTorch
transcripts were captured with torch 2.13.0 on the machine that produced the manual. Their version
and device matter when comparing numerical residuals or framework behavior.

The small inputs serve different purposes. Repeated attention scores make the mask the only
reason rows differ; a one-feature normalization axis makes its centered activation exactly zero;
a camera with rational coordinates makes projection arithmetic inspectable by hand. Choosing such
inputs removes competing explanations for a surprising result. The companion general statements
then show which conclusion survives beyond those particular numbers. Read the value, its scalar
type, and the theorem's hypotheses together: a tensor shape can express an interface, a checked
example can expose a discrepancy, and a quantified equality can identify the behavior another
implementation must preserve.

# Catalog Build

All entries are imported by {src "NN/Examples/BugZoo/All.lean"}[NN/Examples/BugZoo/All.lean]. Build
them together with:

```terminal
# Elaborate the maintained catalog through its shared import
# module.
lake build NN.Examples.BugZoo.All
```

A successful build means every definition and theorem in the catalog elaborated;
it does not mean that every external framework implementation satisfies those contracts.

`All.lean` is also the completeness boundary for the maintained catalog: an example file not
imported there is not covered by this build command. The contracts describe TorchLean reference
objects; external framework conformance requires a separate importer, refinement theorem, or
explicit assumption.

This chapter imports the fourteen non-attention entries individually. The attention entry also
states $`\exp(-\infty)=0` in `EReal`; the chapter uses the real-valued hard-mask theorem directly,
through
{lean}`NN.Proofs.Models.Attention.hardMaskedSoftmaxSpec_causal_future_zero`, which needs no extended
reals.

For a more interactive pass, create `BugZooAudit.lean`:

```
-- Compare the exact mask contract with explicit
-- shape-changing operations.
import NN.Examples.BugZoo.AttentionMask
import NN.Examples.BugZoo.ShapeAndBroadcast

#check NN.Examples.BugZoo.AttentionMask.exactMaskedLogit_blocked_exp_zero
#check NN.Examples.BugZoo.ShapeAndBroadcast.addSingletonBatch
#check NN.Examples.BugZoo.ShapeAndBroadcast.broadcastRowToMatrix_firstRow
```

Open the file in the Lean Infoview. The shape declarations expose the singleton batch insertion and
the proof-carrying broadcast as different operations. These are the contracts. The motivating
PyTorch snippets in the source comments explain the bug family, but are not imported as evidence.

# Example Anatomy

A good BugZoo example has four parts:

- the pattern in the framework that goes wrong;
- the TorchLean object that names the intended behavior;
- the theorem, structure, or definition that marks the checked boundary;
- the external conformance obligation or unsupported scope that remains outside the checked claim.

Returning a tensor with the expected shape leaves its values, dependency structure, and state
transitions unconstrained. The catalog gives those additional conditions explicit statements.

The common contract shape is:

$$`\text{bug pattern}
\;\leadsto\;
\text{TorchLean object}
\;\leadsto\;
\text{checked claim}`

Some representative contract shapes:

:::table +header
*
  * Example
  * Contract shape
*
  * Attention mask
  * $`j>i\Rightarrow A_{ij}=0`
*
  * Batch invariance
  * $`\operatorname{unstack}(\operatorname{mapBatch}(f,X),i)=f(X_i)`
*
  * Tokenizer boundary
  * token ids inhabit `Fin vocabularySize`
*
  * KV cache
  * appended key and value appear at the final slot
*
  * Float boundary
  * core arithmetic agrees with Lean's logical model after NaN canonicalization
*
  * Compiler boundary
  * target output equals source output
*
  * Stable loss
  * logits path uses log-softmax semantics
*
  * Ignored labels
  * inactive labels contribute zero
*
  * LayerNorm degenerate axis
  * zero-variance normalization follows an explicit epsilon policy
*
  * 3D projection
  * an accepted certificate implies the projection lies inside the claimed box
:::

# The Examples

## Shape And Broadcast

{src "NN/Examples/BugZoo/ShapeAndBroadcast.lean"}[NN.Examples.BugZoo.ShapeAndBroadcast source]
uses a missing batch dimension and a reduction followed by broadcasting. In NumPy-style tensor
libraries, a reduced vector can silently expand across a matrix. TorchLean's ordinary elementwise
operations require equal shapes; an explicit broadcast carries `Shape.CanBroadcastTo` evidence.

Summing the rows of a two by three matrix removes the leading dimension:

```lean (name := bzMatDef)
/-- A small matrix for the reduce then broadcast walk. -/
def bzMat : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

#eval ShapeAndBroadcast.reduceRows bzMat
```
```leanOutput bzMatDef (whitespace := lax)
[5.000000, 7.000000, 9.000000]
```

The result has shape `[3]`, which the catalog names `RowShape`. In PyTorch, adding it back to the
matrix is legal and produces a two by three tensor:

```terminal +output
== 11. reduce then add, silently broadcast ==
rows: [5.0, 7.0, 9.0]
m + rows shape: (2, 3)
m + rows: [[6.0, 9.0, 12.0], [9.0, 12.0, 15.0]]
```

This expression is legal whether the broadcast was intended or introduced by a mistaken reduction.
TorchLean's `addSpec` requires the two operand shapes to agree, so it rejects this version:

```lean (name := bzBadAdd) +error
/--
Elementwise addition requires an explicit broadcast of the
reduced row.
-/
def bzBadAdd : Tensor Float [2, 3] :=
  Tensor.addSpec bzMat (ShapeAndBroadcast.reduceRows bzMat)
```
```leanOutput bzBadAdd (whitespace := lax)
Application type mismatch: The argument
  ShapeAndBroadcast.reduceRows bzMat
has type
  Tensor Float ShapeAndBroadcast.RowShape
but is expected to have type
  Tensor Float [2, 3]
in the application
  bzMat.addSpec (ShapeAndBroadcast.reduceRows bzMat)
```

If the broadcast was intended, it has to be written down:

```lean (name := bzBroadcast)
-- Expand the reduced row explicitly across both rows of the
-- original matrix.
#eval ShapeAndBroadcast.broadcastRowToMatrix
  (ShapeAndBroadcast.reduceRows bzMat)
```
```leanOutput bzBroadcast (whitespace := lax)
[[5.000000, 7.000000, 9.000000], [5.000000, 7.000000, 9.000000]]
```

The values agree with PyTorch's implicit expansion. The difference is that the expansion is now a
term with a name, so a reviewer reading a diff sees it appear or disappear.

The catalog keeps a second, inference-driven spelling and proves the two agree, so a proof written
against one form can be reused for the other:

```lean (name := bzInferred)
-- Check that inferred broadcast evidence selects the same
-- tensor operation.
#check @ShapeAndBroadcast.inferredRowBroadcastToMatrix_eq
```
```leanOutput bzInferred (whitespace := lax)
@ShapeAndBroadcast.inferredRowBroadcastToMatrix_eq :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Inhabited α]
    (x : Tensor α ShapeAndBroadcast.RowShape),
  ShapeAndBroadcast.inferredRowBroadcastToMatrix x =
    ShapeAndBroadcast.broadcastRowToMatrix x
```

For an image, `addSingletonBatch` inserts the leading batch dimension. Its result type records
that specific shape change:

```lean (name := bzBatchSig)
-- Read the inserted leading batch dimension in the result
-- type.
#check @ShapeAndBroadcast.addSingletonBatch
```
```leanOutput bzBatchSig (whitespace := lax)
@ShapeAndBroadcast.addSingletonBatch : {α : Type} →
  [inst : Storage α] →
    Tensor α ShapeAndBroadcast.ImageShape →
      Tensor α ShapeAndBroadcast.SingletonBatchImageShape
```

In PyTorch, `m.unsqueeze(0) + m` also succeeds: the rank-two operand broadcasts against the
rank-three operand. Success therefore does not distinguish an intentional singleton batch from an
accidental extra dimension.

The empirical motivation is tensor shape fault work such as
[SFData](https://doi.org/10.1145/3533767.3534383), plus
[numerical bug studies](https://doi.org/10.1145/3551349.3559561)
that found bad reductions and accidental broadcasting in real deep learning programs.

The three reduced entries are column totals: the first is `1 + 4`, the second `2 + 5`, and the
third `3 + 6`. Broadcasting repeats that entire vector for each original row. This is a useful
example because both the intended and accidental computations can return six numbers. The output
shape alone therefore cannot explain whether the reduction axis was chosen correctly. The type
error locates the missing operation, while the broadcast theorem says that two explicit spellings
of that operation agree. Neither chooses the modeling intention on the author's behalf.

## Stable Loss

{src "NN/Examples/BugZoo/StableLoss.lean"}[NN.Examples.BugZoo.StableLoss source] is about losses
that look mathematically harmless but fail numerically. The classic sketch is `softmax` followed by
`log`: if a probability rounds to zero, the log path can produce infinities and downstream NaNs.
TorchLean keeps two APIs separate. Logits should use `crossEntropyLogitsSpec`, which unfolds through
the log-softmax; probability inputs use the clipped probability form `crossEntropySpec`.

The logits below make two of the three probabilities underflow in binary64. Direct log-softmax
retains finite log-probabilities:

```lean (name := bzLogSoftmax)
/-- Logits far enough apart that the softmax underflows. -/
def bzLogits : Tensor Float [3] :=
  [-800.0, -1600.0, 0.0]

#eval Activation.logSoftmaxSpec 0 bzLogits
```
```leanOutput bzLogSoftmax (whitespace := lax)
[-800.000000, -1600.000000, 0.000000]
```

Computing the probabilities first loses that information before the logarithm runs:

```lean (name := bzLogOfSoftmax)
-- Taking log after underflow cannot recover the discarded
-- log-probabilities.
#eval Tensor.logSpec (Activation.softmaxSpec 0 bzLogits)
```
```leanOutput bzLogOfSoftmax (whitespace := lax)
[-inf, -inf, 0.000000]
```

The captured PyTorch run shows the same finite and infinite results:

```terminal +output
== 3. softmax then log vs log_softmax ==
log(softmax): [[-inf, -inf, 0.0]]
log_softmax : [[-800.0, -1600.0, 0.0]]
```

For finite real logits, the two formulas define the same function. In this computation, however,
the probability path has already rounded two entries to zero. Applying `log` gives $`-\infty`;
multiplying such an entry by a zero target can then produce NaN. The logits-loss contract fixes the
log-softmax expression used by the reference implementation:

```lean (name := bzCEUnfold)
-- Expose the log-softmax expression and the reduction used
-- by logits cross-entropy.
#check @StableLoss.crossEntropyLogits_uses_logSoftmax
```
```leanOutput bzCEUnfold (whitespace := lax)
@StableLoss.crossEntropyLogits_uses_logSoftmax :
  ∀ {s : Shape} (axis : ℕ)
    [inst : Shape.AxisInBounds axis s] {α : Type}
    [inst_1 : Storage α] [inst_2 : Context α]
    (logits target : Tensor α s),
  Spec.crossEntropyLogitsSpec axis logits target =
    have logp := Activation.logSoftmaxSpec axis logits;
    have total := (target.mulSpec logp).sumSpec;
    Spec.meanOverAxisSlices axis (-total)
```

This unfolding lemma exposes the equation a replacement implementation must preserve.
`crossEntropyProbabilities_clips_before_log` records the corresponding choice for probability
inputs. Neither equation by itself proves conformance of a fused runtime kernel.

This example is motivated by [TensorFuzz](https://proceedings.mlr.press/v97/odena19a.html), which
targeted rare numerical failures, and by
[empirical studies of numerical bugs](https://doi.org/10.1145/3551349.3559561)
involving `log`, `sqrt`, division, `exp`, and reductions.

The underflow example isolates an intermediate representation problem. A log-probability near
`-1600` is a usable finite number even when exponentiating it produces a probability too small for
the format. Once that probability has become zero, a later logarithm has no way to reconstruct the
original value. Keeping the logits until log-softmax avoids that particular loss of information.
The theorem's final reduction also matters: it states how target-weighted log-probabilities become
a scalar, so changing a sum into a different mean would change the loss contract even if every
individual log-probability remained correct.

## Ignored Labels

{src "NN/Examples/BugZoo/IgnoredLabelLoss.lean"}[NN.Examples.BugZoo.IgnoredLabelLoss source]
examines a mean loss when no labels remain active.
[PyTorch issue #75181](https://github.com/pytorch/pytorch/issues/75181) reported an `ignore_index`
case where every label was ignored and the result was `nan`. The captured run below has that
behavior:

```terminal +output
== 1. all labels ignored ==
cross_entropy: nan
```

TorchLean defines each label's contribution using an explicit active flag:

```lean (name := bzIgnored)
-- An inactive label contributes zero even when its supplied
-- loss is nonzero.
#eval IgnoredLabelLoss.labelContribution false 2.5
```
```leanOutput bzIgnored (whitespace := lax)
0.000000
```

```lean (name := bzActive)
-- Activating the same label restores its supplied
-- contribution.
#eval IgnoredLabelLoss.labelContribution true 2.5
```
```leanOutput bzActive (whitespace := lax)
2.500000
```

The policy is the definition:

$$`\operatorname{labelContribution}(active,loss)
=
\begin{cases}
loss, & active\\
0, & \neg active
\end{cases}`

and the checked form is the one-line theorem that the false branch really is zero at any element
type with a zero:

```lean (name := bzIgnoredThm)
-- The ignored-label equation is independent of the chosen
-- scalar loss value.
#check @IgnoredLabelLoss.ignored_label_contributes_zero
```
```leanOutput bzIgnoredThm (whitespace := lax)
@IgnoredLabelLoss.ignored_label_contributes_zero :
  ∀ {α : Type} [inst : Zero α] (loss : α),
  IgnoredLabelLoss.labelContribution false loss = 0
```

The reduction must also specify what happens when the active count is zero. Dividing the zero
loss total by that count is undefined. The catalog adds epsilon to the denominator and evaluates
this policy on an empty active set:

```lean (name := bzMaskedMean)
-- Evaluate the chosen denominator policy when both the
-- total and count are zero.
#eval IgnoredLabelLoss.safeMaskedMean 0.0 0.0
```
```leanOutput bzMaskedMean (whitespace := lax)
0.000000
```

Returning zero for a fully ignored batch is a modeling choice. A training loop could instead skip
the step, preserve a running average, or report an error. Adding epsilon also changes nonempty
means: with one active label, the real-valued result is scaled by $`1/(1+\varepsilon)`.
`Context.defaultEpsilon` at `Float` is $`10^{-6}`.

The two contribution outputs distinguish exclusion from a small numerical weight. An ignored
label follows a branch that returns zero; an active label retains its loss. The later reduction
must then count the same set of active labels, otherwise the numerator and denominator describe
different batches. In this compact helper `activeCount` has the scalar type, so its type alone
does not enforce a nonnegative integer count. The empty-set computation uses the intended count
zero. A caller handling arbitrary external counts must establish their meaning before relying on
the denominator policy.

## Autograd Domain

{src "NN/Examples/BugZoo/AutogradDomain.lean"}[NN.Examples.BugZoo.AutogradDomain source] follows
[PyTorch's own autograd note about division by
zero](https://docs.pytorch.org/docs/main/notes/autograd.html#division-by-zero-in-autograd). If a
graph computes $`x/0` and masks the bad value afterward, the forward result may look clean while
the backward graph still contains the undefined operation.

TorchLean's example names the difference between dividing first and masking later, and dividing
safely and then masking. Evaluate both at a masked-out element whose denominator is zero:

```lean (name := bzSafeDiv)
-- Shift the zero denominator before division, then apply
-- the zero mask.
#eval AutogradDomain.maskAfterSafeDiv
  (Tensor.full [1] 0.0) (Tensor.full [1] 1.0)
  (Tensor.full [1] 0.0)
```
```leanOutput bzSafeDiv (whitespace := lax)
[0.000000]
```

```lean (name := bzUnsafeDiv)
-- This expression divides by zero before multiplying by the
-- mask.
#eval AutogradDomain.unsafeDivThenMask
  (Tensor.full [1] 0.0) (Tensor.full [1] 1.0)
  (Tensor.full [1] 0.0)
```
```leanOutput bzUnsafeDiv (whitespace := lax)
[NaN]
```

The unsafe form already produces `NaN` in the forward direction here, because the mask multiplies
zero by an infinity. In PyTorch the forward value can look fine and the gradient is what carries
the damage:

```terminal +output
== 5. divide then mask in autograd ==
grad after divide-then-mask: [nan]
grad after safe-divide     : [0.0]
```

The unfolding lemma pins down where the epsilon belongs in the reference expression. A fused
kernel needs separate conformance evidence that it preserves this formula:

```lean (name := bzSafeDivThm)
-- Inspect the exact denominator shift that distinguishes
-- the guarded expression.
open AutogradDomain in
#check @maskAfterSafeDiv_uses_epsilon_denominator
```
```leanOutput bzSafeDivThm (whitespace := lax)
@maskAfterSafeDiv_uses_epsilon_denominator :
  ∀ {s : Shape} {α : Type} [inst : Storage α]
    [inst_1 : Context α]
    (mask numerator denominator : Tensor α s),
  maskAfterSafeDiv mask numerator denominator =
    mask.mulSpec
      (Tensor.map2Spec
        (fun a b => a / (b + Context.defaultEpsilon)) numerator
        denominator)
```

Adding epsilon is a denominator shift, not a clamp away from zero: the shifted denominator is
still zero when the original denominator is $`-\varepsilon`. This example checks the displayed
zero-denominator case and unfolds the formula; it does not prove finite values or gradients for
arbitrary inputs.

Keeping the unsafe expression beside the guarded one exposes the relevant graph change: the
zero denominator is used before masking in the first expression and shifted before division in
the second.

Masking is particularly subtle in differentiation because the graph records how a value was
computed. A zero at the final output does not remove an earlier division node or its local
derivative. The reference examples deliberately keep the unsafe operation visible, rather than
checking only whether the selected forward entries look finite. For a gradient comparison, the
relevant observations are the numerator and denominator cotangents as well as the loss. The
captured PyTorch probe supplies one such numerical observation; the local unfolding theorem
specifies the guarded forward formula that a derivative argument would have to follow.

## Attention Mask

{src "NN/Examples/BugZoo/AttentionMask.lean"}[NN.Examples.BugZoo.AttentionMask source] is the
catalog entry for causal mask semantics. Attention masks fail by polarity, layout, fake negative
infinity, fully masked rows, and interactions with API flags. PyTorch has had relevant reports for
`MultiheadAttention`, including
[`is_causal=True` being ignored when
`need_weights=True`](https://github.com/pytorch/pytorch/issues/99282) and
[fully masked heads producing NaNs](https://github.com/pytorch/pytorch/issues/160064) when weights
are requested.

TorchLean's causal mask for three tokens is:

```lean (name := bzCausalMask)
-- Rows are queries; true entries mark keys visible to that
-- query.
#eval Spec.causalMask 3
```
```leanOutput bzCausalMask (whitespace := lax)
[[true, false, false], [true, true, false], [true, true, true]]
```

A `true` entry makes a key visible to a query. The first query can see only its own key; each later
query can also see the preceding keys. Identical score rows isolate the mask's effect on the
weights:

```lean (name := bzMaskedSoftmax)
/-- Three query rows with identical scores. -/
def bzScores : Tensor Float [3, 3] :=
  [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]

#eval Spec.hardMaskedSoftmaxSpec bzScores
  (Spec.causalMask 3)
```
```leanOutput bzMaskedSoftmax (whitespace := lax)
[[1.000000, 0.000000, 0.000000],
 [0.268941, 0.731059, 0.000000],
 [0.090031, 0.244728, 0.665241]]
```

PyTorch, given the same scores and the same mask through `masked_fill` and `softmax`:

```terminal +output
== 4. causal mask polarity ==
causal weights:
tensor([[1.0000, 0.0000, 0.0000],
        [0.2689, 0.7311, 0.0000],
        [0.0900, 0.2447, 0.6652]])
```

The visible weights agree at the printed precision. Flipping the mask polarity fully blocks the
last row in this PyTorch computation:

```
flipped polarity weights:
tensor([[0.0000, 0.2689, 0.7311],
        [0.0000, 0.0000, 1.0000],
        [   nan,    nan,    nan]])
```

The last row contains only negative infinities after `masked_fill`. Direct exponentiation gives a
zero numerator and normalizer; max-subtraction also encounters the undefined difference
$`-\infty-(-\infty)`. This `softmax` call produces NaNs. The captured
`scaled_dot_product_attention` probe instead returns zeros for a fully masked row, so the two API
paths have different behavior in this build.

The checked claim quantifies over every strict-future position, at every sequence length:

```lean (name := bzCausalThm)
-- The theorem quantifies over all sequence lengths and
-- strict-future coordinates.
open NN.Proofs.Models.Attention in
#check @hardMaskedSoftmaxSpec_causal_future_zero
```
```leanOutput bzCausalThm (whitespace := lax)
@hardMaskedSoftmaxSpec_causal_future_zero :
  ∀ {n : ℕ} (scores : Tensor ℝ [n, n]) (i j : Fin n),
  ↑i < ↑j →
  Spec.get2
      (Spec.hardMaskedSoftmaxSpec scores
        (Spec.causalMask n)) i j =
    0
```

This theorem rules out direct attention to a strict-future value. Causality of a complete model
also requires its other operations to preserve that restriction. The catalog connects the
hard-mask formula to a negative-infinity logit by first stating the exact exponential identity in
`EReal`, since `ℝ` contains no negative infinity:

```
-- Exact negative infinity gives a zero numerator in the
-- extended-real presentation.
theorem exactMaskedLogit_blocked_exp_zero (score : ℝ) :
    EReal.exp (exactMaskedLogit score false) = 0 := by
  simp [exactMaskedLogit]
```

TorchLean's ordinary attention spec uses a hard-masked numerator: a blocked entry contributes
zero before normalization. The theorem therefore needs no extended reals. If every entry in a row
is blocked, the spec defines the entire weight row to be zero.

Each causal row provides a different check. Row zero has one allowed key, so its normalized
weight must be one. Row one compares two allowed scores while forcing the third entry to zero.
The last row allows all three scores and recovers ordinary softmax. These cases separate mask
polarity from score scaling and normalization. A fully blocked row adds the denominator boundary
case. Keeping that case explicit is necessary when comparing APIs that share a Boolean mask type
but make different choices for empty attention support.

## Compiler Boundary

{src "NN/Examples/BugZoo/CompilerBoundary.lean"}[NN.Examples.BugZoo.CompilerBoundary source] is the
wrong-code example. It is about optimized graphs that run, return tensors, and are nevertheless not
the same computation as the source graph. [NNSmith](https://arxiv.org/abs/2207.13066) found
compiler bugs across TVM, TensorRT, ONNXRuntime, and PyTorch,
[FreeFuzz](https://arxiv.org/abs/2201.06589) found framework and API bugs by mining real snippets,
and a recent [PyTorch compiler correctness study](https://arxiv.org/abs/2604.08720) focuses directly
on silent `torch.compile` wrong outputs.

The local preservation theorem relates TorchLean's IR graph to the executable forward graph
produced by lowering:

```lean (name := bzLowering)
-- Read both lowering success and the raw-log exclusion
-- before using the equality.
open CompilerBoundary in
#check @successfulLowering_preservesDenotation
```
```leanOutput bzLowering (whitespace := lax)
@successfulLowering_preservesDenotation :
  ∀ {α : Type} [inst : Storage α] [inst_1 : Context α]
    (graph : NN.IR.Graph) (payload : NN.IR.Payload α)
    (executable : Runtime.Autograd.IRExec.ForwardGraph α),
  Runtime.Autograd.IRExec.NoRawLog graph →
  Runtime.Autograd.IRExec.lowerToForwardGraph graph
        payload =
      Except.ok executable →
  ∀ (input : Tensor α executable.inShape),
  graph.denoteAll payload
        { shape := executable.inShape, tensor := input } =
      Except.ok (executable.denoteAll input)
```

There are two hypotheses: the graph contains no raw logarithm node, and lowering succeeds. Under
both conditions, the source and lowered denotations agree for every input of the declared shape.
The compared functions are Lean denotations; generated machine code and external kernels require
separate conformance evidence.

The reusable statement is:

$$`\operatorname{NoRawLog}(g)\;\land\;\operatorname{lower}(g)=\mathrm{ok}(e)
\quad\Longrightarrow\quad
\forall x,\; e(x)=g(x)`

`NoRawLog` excludes graphs with a raw logarithm, whose domain treatment differs across the two
paths. Successful lowering alone is insufficient to apply this theorem.

The lowering theorem compares all intermediate node values, not only the final tensor. This
makes it useful when a later proof depends on a hidden activation or when debugging the first
point at which two evaluations diverge. The payload is part of the comparison: using a different
constant or parameter tensor would describe a different source computation. The universal input
quantifier is also stronger than checking a calibration batch. Once the two hypotheses are met,
the equality applies to every input of the lowered graph's declared shape under the same scalar
context and payload.

## Float Boundary

{src "NN/Examples/BugZoo/FloatBoundary.lean"}[NN.Examples.BugZoo.FloatBoundary source]
separates real arithmetic from the binary32 model. Finite precision, exceptional values, fused
operations, subnormal behavior, and changed reduction order can invalidate a property proved only
over reals {Informal.citep jiarinard2020}[]. Goldberg describes the underlying rounding effects
{Informal.citep goldberg1991}[]; Flocq {Informal.citep flocq2011}[] and the Coq treatment of
numerical
programs {Informal.citep boldo2015}[] develop mechanized ways to reason about them.

`IEEE32Exec` is the executable bit-level binary32 model. Lean gives ordinary `Float32` core
arithmetic a logical definition through `Float32.Model`. We can put the two side by side on the
oldest example in the subject, adding a tenth to a fifth:

```lean (name := bzBits)
open Floats.IEEE754 Floats.IEEE754.Float32Bridge in
/-- Lean's own binary32 add next to the executable model. -/
def bzBits (a b : Float32) : UInt32 × UInt32 :=
  ((a + b).toBits,
    (IEEE32Exec.add (toIEEE32Exec a) (toIEEE32Exec b)).bits)

#eval bzBits 0.1 0.2
```
```leanOutput bzBits (whitespace := lax)
(1050253722, 1050253722)
```

Identical bit patterns, and the same pattern PyTorch reports for its own float32 addition:

```terminal +output
== 7. float32 0.1 + 0.2 bit pattern ==
sum: 0.30000001192092896 bits: 1050253722
```

The bit pattern identifies the rounded result even when a decimal printer hides low-order bits.
This one addition is a runtime comparison. The following theorem gives the general relation
between Lean's logical `Float32` addition and `IEEE32Exec`:

```lean (name := bzAddThm)
-- The general addition bridge canonicalizes NaNs on the
-- executable-model side.
#check @Floats.IEEE754.Float32Bridge.toIEEE32Exec_add
```
```leanOutput bzAddThm (whitespace := lax)
Floats.IEEE754.Float32Bridge.toIEEE32Exec_add :
  ∀ (a b : Float32),
  Floats.IEEE754.Float32Bridge.toIEEE32Exec (a.add b) =
    Floats.IEEE754.Float32Bridge.canonicalize
      ((Floats.IEEE754.Float32Bridge.toIEEE32Exec a).add
        (Floats.IEEE754.Float32Bridge.toIEEE32Exec b))
```

NaN canonicalization discards payload and sign distinctions retained by `IEEE32Exec` but absent
from Lean's logical model. TorchLean proves analogous agreement for finiteness, infinity, NaN,
comparison, and the canonical results of subtraction, multiplication, division, square root,
negation, and absolute value. A compiled CPU instruction or CUDA kernel is a separate execution
boundary; the theorem does not establish conformance of a particular vectorized kernel.

The pair of integers in the addition output is useful because the comparison is about stored
bits, not a tolerance chosen after printing. The theorem then makes its exceptional-value policy
explicit through `canonicalize`. For finite results this still identifies the model's arithmetic
result; for NaNs it deliberately identifies patterns whose payloads differ. A property about NaN
payload preservation would therefore require a more precise relation. Likewise, a theorem about
one addition does not fix the order in which a network's long reduction performs many additions;
that order belongs to the surrounding computation.

## Normalization State

{src "NN/Examples/BugZoo/NormalizationState.lean"}[NN.Examples.BugZoo.NormalizationState source]
covers BatchNorm style bugs where the formula or the state is wrong but the layer still emits a
tensor {Informal.citep batchnorm2015}[].
[CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf) reported a
BatchNorm epsilon placement issue across backends, while
[LEMON](https://lingming.cs.illinois.edu/courses/cs598ast-f20/paper-dnn-lib-testing.pdf) found
moving-statistics bugs and BatchNorm layers that produce NaNs.

Putting epsilon inside or outside the square root changes the normalization formula. The
following binary64 computations use a fixed mean of zero and variance of zero, with incoming value
2. Such fixed statistics need not equal the mean and variance of the incoming batch:

```lean (name := bzEpsIn)
/-- Epsilon inside the root, the way the spec has it. -/
def bzEpsInside (x mean variance gamma beta eps : Float) :
    Float :=
  ((x - mean) / (variance + eps).sqrt) * gamma + beta

#eval bzEpsInside 2 0 0 1 0 1e-5
```
```leanOutput bzEpsIn (whitespace := lax)
632.455532
```

```lean (name := bzEpsOut)
/-- Epsilon outside the square root, the classic slip. -/
def bzEpsOutside (x mean variance gamma beta eps : Float) :
    Float :=
  ((x - mean) / (variance.sqrt + eps)) * gamma + beta

#eval bzEpsOutside 2 0 0 1 0 1e-5
```
```leanOutput bzEpsOut (whitespace := lax)
200000.000000
```

The second result is about 316 times the first. The captured PyTorch `batch_norm` result follows
the inside-root convention, while the hand-written outside-root expression gives 200000:

```terminal +output
== 6. batchnorm epsilon placement, zero variance ==
eps inside sqrt: [632.45556640625, ...]
eps outside sqrt: [200000.0, ...]
```

The Lean result uses binary64 and the PyTorch result uses float32. Their displayed values,
632.455532 and 632.45556640625, are close but unequal; the transcript does not establish bitwise
agreement across scalar types.

The spec-side contract states the placement as an equation over the reals, where there is no
rounding to argue about:

```lean (name := bzEpsThm)
-- The scalar equation fixes epsilon inside the guarded
-- square root.
open NormalizationState in
#check @normalizeCore_scalar_uses_variance_plus_epsilon
```
```leanOutput bzEpsThm (whitespace := lax)
normalizeCore_scalar_uses_variance_plus_epsilon :
  ∀ (x mean variance gamma beta epsilon : ℝ),
  Spec.normalizeCore [] [] [] [] [] epsilon
        (Tensor.full [] x) (Tensor.full [] mean)
        (Tensor.full [] variance) (Tensor.full [] gamma)
        (Tensor.full [] beta) ⋯ ⋯ ⋯ ⋯ =
    Tensor.full []
      ((x - mean) /
            MathFunctions.sqrt (max (variance + epsilon) 0) *
          gamma +
        beta)
```

The formula clamps `variance + epsilon` to zero before taking its square root. This guard is part
of the reference expression; it does not say that an unguarded native square root returns zero on
negative inputs.

BatchNorm inference also depends on its running statistics. Passing them explicitly lets the
reference theorem fix that state and show that inference is affine in the input:

```lean (name := bzAffine)
-- The scale and bias are chosen before the input is
-- quantified.
open NormalizationState in
#check @batchNormEvalWithStats_affine
```
```leanOutput bzAffine (whitespace := lax)
@batchNormEvalWithStats_affine :
  ∀ {channels : ℕ} {sSpatial : Shape}
    (stats : RunningStats channels)
    (gamma beta : Tensor ℝ [channels])
    (epsilon : optParam ℝ normalizationEpsilon),
  ∃ scale bias,
    ∀ (x : Tensor ℝ (sSpatial.prependDim channels)),
      batchNormEvalWithStats x stats gamma beta epsilon =
        (x.mulSpec scale).addSpec bias
```

The existential witnesses precede the input quantifier: the same scale and bias work for every
input. That order is what makes this an affine-map statement, rather than an equality whose
witnesses could change with each input. Once the running statistics are fixed, downstream interval
bounds and Lipschitz arguments can use that common affine representation.

Fixed inference statistics explain why the large normalization value here is possible. The
incoming value is two while the supplied running mean and variance are zero, so the numerator is
nonzero and the small denominator amplifies it. In a one-feature LayerNorm computation, by
contrast, the mean is computed from that same feature and the centered numerator vanishes. The
two examples should therefore have different outputs even with identical epsilon values. The
affine BatchNorm theorem concerns fixed statistics; changing those statistics during training
changes the affine map and requires a statement about the state update as well.

## LayerNorm Degenerate Axis

{src "NN/Examples/BugZoo/LayerNormDegenerateAxis.lean"}[NN.Examples.BugZoo.LayerNormDegenerateAxis
source] records a related but different normalization bug family. LayerNorm can receive an axis with
one element. The tensor output still has the expected shape, but the variance term is zero and the
epsilon convention decides whether the result is finite and meaningful.

With one feature the answer is forced: the centred value is $`x-x=0`, so the output is the bias and
the input and scale gradients vanish in the finite, positive-epsilon case. The bias gradient
does not vanish in general; it is the upstream cotangent. The catalog evaluates that at a
deliberately awkward input, $`x=10^6`, with
weight 2, bias 3, and epsilon $`10^{-5}`:

```lean (name := bzLnFwd)
-- A one-feature normalized value should leave only the
-- affine bias.
#eval LayerNormDegenerateAxis.reproLayerNormForward
```
```leanOutput bzLnFwd (whitespace := lax)
3.000000
```

```lean (name := bzLnDw)
-- The scale gradient multiplies the zero centered
-- activation.
#eval LayerNormDegenerateAxis.reproLayerNormDWeight
```
```leanOutput bzLnDw (whitespace := lax)
0.000000
```

```lean (name := bzLnDx)
-- The input gradient cancels when the normalization axis
-- has one feature.
#eval LayerNormDegenerateAxis.reproLayerNormDX
```
```leanOutput bzLnDx (whitespace := lax)
0.000000
```

The forward value is the bias, as it must be, and both gradients are zero. The matching contract
is a theorem over the reals rather than a sampled check:

```lean (name := bzLnThm)
-- Inspect the real-valued scale-gradient contract behind
-- the numerical probe.
open LayerNormDegenerateAxis in
#check @one_feature_layernorm_scale_grad_contract
```
```leanOutput bzLnThm (whitespace := lax)
one_feature_layernorm_scale_grad_contract :
  ∀ (x dy epsilon : ℝ),
  dy * ((x - x) / MathFunctions.sqrt (max (0 + epsilon) 0)) =
    0
```

Now run the same configuration through PyTorch at float32 and at float64, sweeping the input
magnitude:

```terminal
# Sweep input magnitudes and scalar types while recording
# normalization residuals.
python3 scripts/verification/normalization_contract_probe.py --device cpu
```

The probe requires PyTorch and prints its version along with the forward and backward residuals.
It also covers constant slices for LayerNorm, GroupNorm, InstanceNorm, and training-mode BatchNorm.
The transcript below is the historical run described at the start of this chapter; a different
PyTorch build or device can give different residuals.

```
torch.float32    x=         1  y=3          dw=0              dx=0.0
torch.float32    x=      1000  y=3          dw=0.00195312     dx=0.0
torch.float32    x=     1e+06  y=3          dw=7.20312        dx=0.0
torch.float32    x=     1e+07  y=3          dw=-55.9688       dx=0.0
torch.float64    x=         1  y=3          dw=0              dx=0.0
torch.float64    x=      1000  y=3          dw=-3.63798e-12   dx=0.0
torch.float64    x=     1e+06  y=3          dw=9.40054e-09    dx=0.0
torch.float64    x=     1e+07  y=3          dw=2.13215e-07    dx=0.0
```

The forward value is 3 everywhere, exactly as the contract says. The weight gradient is not zero. At
float32 and $`x=10^6` it is 7.2, and at $`10^7` it is negative 56, on a quantity the mathematics
pins to exactly zero. These residuals expose numerical error in the native backward path, even
though the displayed forward output agrees with the reference. The much smaller float64 residuals
are consistent with cancellation amplified by the reciprocal standard deviation; identifying the
exact instruction sequence requires inspecting the selected kernel, which this probe does not do.

The magnitude sweep matters here: a small input can produce a small residual even when the same
backward path has a large error at $`10^6`. The theorem identifies the exact target value, while
the probe measures deviations for the tested builds, devices, and magnitudes.

The three Lean outputs separate the forward value, scale gradient, and input gradient so that
agreement in one cannot hide disagreement in another. Bias three survives normalization, whereas
the scale multiplies a zero normalized activation. The magnitude sweep keeps these mathematical
targets unchanged while varying a quantity that can affect floating-point cancellation. This
makes the residual meaningful: it measures departure from the same zero target at every listed
magnitude. It does not estimate training accuracy or show how an optimizer would amplify or damp
that error over many steps.

## Constant Normalization Slice

{src "NN/Examples/BugZoo/ConstantNormalizationSlice.lean"}[
NN.Examples.BugZoo.ConstantNormalizationSlice source] keeps the constant-slice case visible. A
constant row or channel slice is not exotic; it appears in padding-heavy batches, masked tokens,
uniform images, and clipped signals. If the implementation assumes positive variance, a constant
slice can produce a division by zero, a `NaN`, or a backend-specific branch.

The constant-slice contract records the variance and the epsilon-protected expression. Its
forward and scale-gradient statements cover the same degeneracy as the one-feature case:

```lean (name := bzConstSlice)
-- This theorem assumes the constant-slice mean and variance
-- have their stated values.
open ConstantNormalizationSlice in
#check @constant_slice_scale_grad_zero
```
```leanOutput bzConstSlice (whitespace := lax)
constant_slice_scale_grad_zero :
  ∀ (dy x epsilon : ℝ),
  dy * ((x - x) / MathFunctions.sqrt (max (0 + epsilon) 0)) =
    0
```

The displayed scale-gradient theorem has the same proposition as the one-feature LayerNorm
scale-gradient theorem. The companion `constant_slice_normalizeCore_outputs_bias` states that the
forward output is the bias. The separate names let readers find the result through either a
constant slice or a one-element axis.

The constant-slice theorem starts after the statistics have been supplied. Its scalar input,
mean, and variance are respectively `x`, `x`, and zero; it does not itself prove that a reduction
kernel computes those statistics exactly. That distinction matters for longer constant slices,
where summation and division introduce their own rounding. It also explains why several
normalization layers share this scalar contract despite using different axes. To apply it to a
complete layer, first identify the slice and connect that layer's statistics to the values in the
pointwise statement.

## Batch Invariance

{src "NN/Examples/BugZoo/BatchInvariance.lean"}[NN.Examples.BugZoo.BatchInvariance source] is about
serving systems that change outputs depending on which other requests share a batch. This can come
from dynamic batching, kernel selection, reduction order, and scheduling details even when user
randomness is off. The catalog points at recent LLM serving work:
[Thinking Machines on inference
nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and an
[LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

The reference semantics is `Tensor.mapLeading [batch]`: apply the same function to each row
independently. Take the
second row out of a batched squaring:

```lean (name := bzBatchMap)
-- Square each batch row independently, then select the
-- second output row.
#eval (Tensor.mapLeading [2]
  (fun r => Tensor.mulSpec r r) bzMat).unstack 1
```
```leanOutput bzBatchMap (whitespace := lax)
[16.000000, 25.000000, 36.000000]
```

and now compute that row on its own:

```lean (name := bzBatchOne)
-- Compare with squaring that same row outside the batch
-- map.
#eval Tensor.mulSpec (bzMat.unstack 1) (bzMat.unstack 1)
```
```leanOutput bzBatchOne (whitespace := lax)
[16.000000, 25.000000, 36.000000]
```

The general equality follows from the definition of `mapLeading`:

```lean (name := bzBatchThm)
-- The map/unstack equality states row independence for an
-- arbitrary function.
open TorchLean.Tensor in
#check @unstack_mapLeading
```
```leanOutput bzBatchThm (whitespace := lax)
@unstack_mapLeading :
  ∀ {α : Type} [inst : Storage α] {batch : ℕ}
    {inShape outShape : Shape} (f : Tensor α inShape → Tensor α outShape)
    (xs : Tensor α (inShape.prependDim batch)) (i : Fin batch),
  (mapLeading [batch] f xs).unstack i = f (xs.unstack i)
```

In the captured PyTorch probe, applying a 256 by 256 linear layer to one row alone and to that row
inside a batch of 64 gives different bit patterns:

```terminal +output
== 9. batch invariance of a linear layer ==
bit identical: False
max abs diff: 5.960464477539062e-07
== 10. same, batch 1 vs batch 2 of a matmul ==
bit identical: False max abs diff: 3.4332275390625e-05
```

The probe runs in inference mode without dropout. At width 512 the reported maximum discrepancy
is about $`3\cdot 10^{-5}`. Changes in kernel selection or reduction order can cause such
differences, but this comparison does not identify the instruction sequence responsible. A
perturbation can change an argmax when the leading scores are sufficiently close. Applying the
reference batch-invariance theorem to this runtime would require a separate conformance result or
an explicitly bounded numerical discrepancy.

Batch independence is a property of the chosen reference computation. The function passed to
`mapLeading` receives one row, so it cannot inspect neighboring requests through that input. An
operation that intentionally computes statistics across the batch has a different interface and
should not satisfy this rowwise equation. This distinction helps diagnose the captured linear
layer comparison: the intended operation is rowwise, while the runtime result varies with batch
composition. The measured maximum difference describes that implementation and input; it is not
a bound for every batch size or every matrix.

## KV Cache

{src "NN/Examples/BugZoo/KVCache.lean"}[NN.Examples.BugZoo.KVCache source] models cache accounting
in autoregressive inference. LLM engines fail through shifted caches, wrong cache slots, config and
shape mismatches, resource scheduling, and interactions with positions or tokenizers. The broader
source trail is again the [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

Append a key and a value to a one-token cache and look at the keys:

```lean (name := bzKV)
open KVCache in
/-- A one token cache holding a zero key and value. -/
def bzCache : Cache Float 1 2 :=
  { keys := Tensor.full [1, 2] 0.0
    values := Tensor.full [1, 2] 0.0 }

#eval (KVCache.appendKV bzCache [1.0, 2.0] [3.0, 4.0]).keys
```
```leanOutput bzKV (whitespace := lax)
[[0.000000, 0.000000], [1.000000, 2.000000]]
```

The output places the new key in the last slot and retains the old row first. The result type
records the new length, two; it does not determine the row contents. The final-slot theorem
establishes where the appended value appears:

```lean (name := bzKVThm)
-- Check which cache slot contains the key after appending
-- one token.
open KVCache in
#check @appendKV_last_key
```
```leanOutput bzKVThm (whitespace := lax)
@appendKV_last_key :
  ∀ {α : Type} [inst : Storage α] {seqLen headDim : ℕ}
    (cache : Cache α seqLen headDim)
    (newKey newValue : Tensor α [headDim]),
  (appendKV cache newKey newValue).keys[seqLen] = newKey
```

This theorem covers the newly written slot. Preservation of earlier slots is visible in the
example and follows the append definition, but the catalog does not state that preservation
theorem. Proving it would require reasoning about indices in the left part of the concatenation;
the final-slot theorem uses the right-part indexing lemma.

The cache has two axes with separate roles: sequence length counts stored token positions and
head dimension counts coordinates of each key or value. Appending must change the former while
preserving the latter. Checking only the final key would still miss a value written to the wrong
slot, so the source includes the corresponding final-value theorem too. Full cached decoding
needs more than these append facts: the query must read the matching key/value prefix using the
same mask and positions as full-sequence attention. These local contracts identify pieces of
that larger comparison without claiming it has been proved here.

## RoPE Position

{src "NN/Examples/BugZoo/RoPEPosition.lean"}[NN.Examples.BugZoo.RoPEPosition source] pairs naturally
with the KV cache example. Rotary position embeddings make position accounting part of the model's
meaning. A decode position off by one can be hard to notice because the shapes still line up and the
model still produces tokens.

A schedule is a function from slots to positions, so appending is a total operation with a printable
result:

```lean (name := bzRope)
open RoPEPosition in
/-- Positions already issued for two decoded tokens. -/
def bzSchedule : PositionSchedule 2 :=
  { pos := fun i => i.val }

#eval List.ofFn
  (RoPEPosition.appendNextPosition bzSchedule).pos
```
```leanOutput bzRope (whitespace := lax)
[0, 1, 2]
```

```lean (name := bzRopeThm)
-- The appended position follows the zero-based
-- sequence-length convention.
open RoPEPosition in
#check @appendNextPosition_last
```
```leanOutput bzRopeThm (whitespace := lax)
@appendNextPosition_last :
  ∀ {seqLen : ℕ} (sched : PositionSchedule seqLen),
  (appendNextPosition sched).pos ⟨seqLen, ⋯⟩ = seqLen
```

In this zero-based example, the appended position is the old sequence length. The theorem fixes
that rule even for an arbitrary input schedule: it does not compute “previous position plus one.”
A cache using an offset or nonconsecutive positions would need a schedule rule that represents
those positions explicitly.

A position schedule separates a token's storage slot from the position used by its embedding.
They coincide in the displayed `[0, 1, 2]` example, which makes the append convention easy to read.
They need not coincide after a window has been shifted or a prefix has been assigned an offset.
The theorem's arbitrary `sched` is useful for spotting that limit: the new position is always
`seqLen`, regardless of the last stored position. No rotation is evaluated by this example; it
checks the bookkeeping that supplies positions to a rotation implementation.

## Tokenizer Boundary

{src "NN/Examples/BugZoo/TokenizerBoundary.lean"}[NN.Examples.BugZoo.TokenizerBoundary source] marks
the boundary before tensors even reach the model. Tokenizer and config mismatches can disagree about
vocabulary size, padding, end-of-sequence, or special token ids while the network code itself looks
ordinary. The [LLM inference engine bug study](https://arxiv.org/abs/2506.09713) lists this as a
real serving class.

A contract is a vocabulary size plus the special ids, each carrying its own bound:

```lean (name := bzTok)
open TokenizerBoundary in
/-- A tiny tokenizer configuration with eight ids. -/
def bzTokenizer : TokenizerContract :=
  { vocabularySize := 8
    paddingTokenId := ⟨0, by decide⟩
    endOfSequenceTokenId := ⟨7, by decide⟩ }

#eval (bzTokenizer.paddingTokenId.val,
  bzTokenizer.endOfSequenceTokenId.val)
```
```leanOutput bzTok (whitespace := lax)
(0, 7)
```

An end-of-sequence id of `⟨8, by decide⟩` would require a proof of $`8<8` and therefore fails to
elaborate. The following theorem extracts the bound already carried by the id:

```lean (name := bzTokThm)
-- Recover the vocabulary bound carried by every token in
-- the sequence.
open TokenizerBoundary in
#check @tokenId_isValid
```
```leanOutput bzTokThm (whitespace := lax)
@tokenId_isValid :
  ∀ {vocabularySize sequenceLength : ℕ}
    (sequence : TokenSequence vocabularySize sequenceLength)
    (position : Fin sequenceLength),
  ↑(sequence.tokenAt position) < vocabularySize
```

Every token id constructed at this type is in range. An importer or tokenizer bridge must establish
that bound when converting external data, and `paddingTokenId_isValid` gives the corresponding
bound for padding. These bounds do not establish that two tokenizers assign the same text or
special-token meaning to an id.

The printed special ids are valid indices into an eight-entry vocabulary, but the type does
not require them to be distinct. Nor does it describe the embedding vector associated with either
id. Those are separate configuration agreements between tokenizer and model. This is why the
bounds theorem is still useful despite extracting information already carried by `Fin`: an
external integer must pass that boundary before it can be used to index the typed sequence.
Afterward, ordinary tensor code can rely on the bound while a separate importer contract handles
token meanings and special-token conventions.

## Geometry 3D Projection

{src "NN/Examples/BugZoo/Geometry3DProjection.lean"}[NN.Examples.BugZoo.Geometry3DProjection source]
shows why BugZoo is not limited to language-model incidents. Vision and robotics pipelines project
3D points to image coordinates with a camera matrix. The formula is familiar:

$$`(u,v)=\left(\frac{x}{z},\frac{y}{z}\right)`

up to intrinsics and coordinate conventions. The bug family is familiar too: depth is zero or has
the wrong sign, coordinate frames are swapped, or the denominator convention is implicit. A tensor
can still be emitted, but the geometry claim no longer matches the camera model.

This entry uses a decidable certificate check. The example camera has focal length 100 and
principal point at the image centre:

```lean (name := bzCam)
open NN.Verification.Geometry3D.Box3D in
/-- A pinhole camera, focal length 100, centre at 50. -/
def bzCamera : CameraP ℚ :=
  [[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]]

open NN.Verification.Geometry3D.Box3D in
/-- Eight cube corners, four to six units away. -/
def bzCorners : BoxCorners ℚ :=
  [[-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4],
   [-1, -1, 6], [1, -1, 6], [1, 1, 6], [-1, 1, 6]]

#check bzCorners
```
```leanOutput bzCam (whitespace := lax)
bzCorners : NN.Verification.Geometry3D.Box3D.BoxCorners ℚ
```

At the near face, depth is 4, so the corners project to $`100\cdot(\pm 1)/4+50`, giving 25 and
75. The far face has depth 6 and projects to $`50\pm 100/6`, inside that range. The enclosing image
box is therefore $`[25,25,75,75]`. Using rationals keeps these calculations exact:

```lean (name := bzCert)
open NN.Verification.Geometry3D.Box3D in
/-- A detector box that does enclose the projection. -/
def bzCert : BoxCameraCert ℚ :=
  { width := 100, height := 100, tol := 0
    camera := bzCamera, corners := bzCorners
    bbox := [25, 25, 75, 75] }

#eval NN.Verification.Geometry3D.Box3D.checkCert bzCert
```
```leanOutput bzCert (whitespace := lax)
true
```

Shrinking the claimed box by five pixels on each side excludes the projected near-face corners,
so the same checker rejects it:

```lean (name := bzCertTight)
open NN.Verification.Geometry3D.Box3D in
/-- The same scene, with a box five pixels too tight. -/
def bzCertTight : BoxCameraCert ℚ :=
  { bzCert with bbox := [30, 30, 70, 70] }

#eval NN.Verification.Geometry3D.Box3D.checkCert bzCertTight
```
```leanOutput bzCertTight (whitespace := lax)
false
```

The rejection distinguishes this checker from a constant-true function on the two tested inputs.
It exercises the enclosure condition; it does not independently test every conjunct. `checkCert`
also checks positive image size, box ordering, that the box lies in the image, positive depths,
and that the projection lies in the image.

The soundness theorem connects acceptance to the geometric predicate:

```lean (name := bzGeoThm)
-- Connect Boolean certificate acceptance to all fields of
-- the geometric predicate.
open Geometry3DProjection
  NN.Verification.Geometry3D.Box3D in
#check @accepted_camera_box_certificate_is_verified
```
```leanOutput bzGeoThm (whitespace := lax)
@accepted_camera_box_certificate_is_verified :
  ∀ {α : Type} [inst : OfNat α 0] [inst_1 : OfNat α 1]
    [inst_2 : Add α] [inst_3 : Sub α] [inst_4 : Mul α]
    [inst_5 : Div α] [inst_6 : LE α] [inst_7 : LT α]
    [inst_8 : DecidableRel fun x1 x2 => x1 ≤ x2]
    [inst_9 : DecidableRel fun x1 x2 => x1 < x2]
    {cert : BoxCameraCert α},
  checkCert cert = true → Verified3DBox cert
```

The typeclasses provide arithmetic operations and decidable comparisons. At `ℚ`, those
operations compute the projection exactly, so acceptance establishes the rational geometric
claim. At `Float`, the same theorem establishes a predicate about rounded float computations;
it does not bound their discrepancy from exact real projection. The companion
`homogeneous_projection_uncertainty_stays_inside_bbox` instead assumes nonnegative enclosing
intervals for the homogeneous numerators and a depth interval bounded away from zero. Applying
that result to a float executor requires showing that its intervals enclose both the geometric
uncertainty and
rounding error.

# Contract Review

For the causal-attention optimization in the opening, the strict-future-zero theorem supplies a
precise reference condition.

The runtime comparison should include visible keys, strict-future keys, and fully blocked rows.
The first causal row has exactly one visible key. A polarity flip can preserve the output shape
while changing all three cases: the current key becomes blocked, future keys become visible, and
the last row becomes fully blocked. Comparing the resulting weights reveals which condition fails.

A layout change also needs to preserve which axis represents queries and which represents keys.
A fused attention implementation {Informal.citep flashattention2022}[] needs to preserve the chosen
mask and normalization semantics. If a separate loss optimization combines softmax and logarithm,
the stable-loss equation specifies its reference expression. For compiler lowering, the theorem
above applies only under `NoRawLog` and successful lowering; for float32 execution, a numerical
conformance argument must also account for rounding and the selected kernels.

After adding a catalog entry, import it in `All.lean` so the maintained catalog build includes it:

```terminal
# Check that every maintained catalog entry remains
# reachable from All.
lake env lean NN/Examples/BugZoo/All.lean
```

# Verification Scope

The theorems describe TorchLean reference objects. The captured framework runs measure separate
implementations: a weight gradient of 7.2 where the real-valued expression gives zero, a batch
discrepancy of about $`3\cdot 10^{-5}`, and different fully masked-row behavior in two attention
API paths. Establishing a deployment guarantee requires connecting the chosen reference object to
the particular runtime.

Remaining obligations and scope distinctions:

- The KV cache append preserves earlier slots by construction, but only the final-slot half is a
  theorem. The missing proof must establish preservation of indices in the left part of the
  concatenation.
- The attention entry also proves $`\exp(-\infty)=0` in `EReal`, using the pinned Mathlib
  extended-real exponential. This chapter uses the real-valued hard-mask theorem directly; the
  maintained `BugZoo.All` build includes the extended-real entry.
- The one-feature LayerNorm and constant-slice scale-gradient theorems state the same proposition.
  Both names need to remain consistent if that shared contract changes.
- The masked-mean epsilon is a policy choice. With one active label it multiplies the real-valued
  mean by $`1/(1+\varepsilon)`, a relative shrinkage of $`\varepsilon/(1+\varepsilon)`. An
  alternative fully ignored-batch policy needs its own definition.
- The Float32 bridge proves agreement between Lean's model and the executable binary32 model after
  NaN canonicalization. Compiled CPU instructions, vectorized reductions, and CUDA kernels are
  separate boundaries with no theorem here.
- The 3D projection check is exact at `ℚ`. At `Float` the same theorem still establishes
  `Verified3DBox` for the float-valued
  predicate, but that is not containment for exact real projection; the interval statement is the
  intended replacement, and wiring it
  to a float executor is future work.
