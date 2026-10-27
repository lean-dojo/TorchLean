import VersoManual
import FloatLib.Floats.Formats.IEEE754
import NN.API
import NN.Examples.BugZoo.ShapeAndBroadcast
import NN.Examples.BugZoo.StableLoss
import NN.Examples.BugZoo.IgnoredLabelLoss
import NN.Examples.BugZoo.AutogradDomain
import NN.Verification.Builtin.Proved.Correctness.Eval.Main
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
import NN.Proofs.Autograd.FDeriv.SoftmaxSpec
import NN.Proofs.Analysis.Normalization
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
checked. These failure families have been studied empirically. Compiler fuzzing found silent
wrong-code bugs across TVM, TensorRT, ONNXRuntime, and PyTorch
{Informal.citep nnsmith2023}[], coverage-guided fuzzing found rare numerical failures in trained
models {Informal.citep tensorfuzz2019}[], and attention masking has its own long tail of polarity
and layout mistakes going back to the original architecture {Informal.citep transformer2017}[].

Each entry isolates a contract such as “strict-future attention weight is zero.” The motivating
incident explains where the condition can fail, and the Lean declaration states the inputs and
hypotheses under which it holds.

The `leanOutput` blocks below are checked when the guide builds. Each computation sits beside
its general contract, so we can inspect both a concrete value and the conditions behind a theorem.

I choose small inputs that let us isolate a cause. Repeated attention scores make the mask the only
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
scripts/lake.sh build NN.Examples.BugZoo.All
```

A successful build means every definition and theorem in the catalog elaborated;
it does not mean that every external framework implementation satisfies those contracts.

`All.lean` is also the completeness boundary for the maintained catalog: an example file not
imported there is not covered by this build command. The contracts describe TorchLean reference
objects; external framework conformance requires a separate importer, refinement theorem, or
explicit assumption.

This chapter imports each non-attention entry individually. The attention entry also
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
#check TorchLean.Tensor.repeatLeading
#check NN.Examples.BugZoo.ShapeAndBroadcast.broadcast_first
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

The condition depends on the operation:

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
  * native import/export, finite-input add/sub, and total square-root export agree with FloatLib
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

/-- Sum the leading axis to get three column totals. -/
def bzRow : Tensor Float [3] :=
  Tensor.reduceSum 0 bzMat Spec.Shape.NonemptyAxis.zero

#eval bzRow
```
```leanOutput bzMatDef (whitespace := lax)
[5.000000, 7.000000, 9.000000]
```

The result has shape `[3]`. Adding this row back to the
matrix by broadcasting would produce `[[6, 9, 12], [9, 12, 15]]`. The shape cannot tell us whether
that broadcast was intended. TorchLean's `addSpec` requires the two operand shapes to agree, so it
rejects
an implicit broadcast:

```lean (name := bzBadAdd) +error -keep
/--
Elementwise addition requires an explicit broadcast of the
reduced row.
-/
def bzBadAdd : Tensor Float [2, 3] :=
  Tensor.addSpec bzMat bzRow
```
```leanOutput bzBadAdd (whitespace := lax)
Application type mismatch: The argument
  bzRow
has type
  Tensor Float [3]
but is expected to have type
  Tensor Float [2, 3]
in the application
  bzMat.addSpec bzRow
```

If the broadcast was intended, it has to be written down:

```lean (name := bzBroadcast)
-- Expand the reduced row explicitly across both rows of the
-- original matrix.
#eval Tensor.broadcastTo
  ShapeAndBroadcast.row_broadcast bzRow
```
```leanOutput bzBroadcast (whitespace := lax)
[[5.000000, 7.000000, 9.000000], [5.000000, 7.000000, 9.000000]]
```

The values agree with PyTorch's implicit expansion. The difference is that we write the expansion
explicitly, so a reviewer reading a diff sees it appear or disappear.

We can also let Lean infer the broadcast evidence. Both spellings use the same tensor operation,
and the catalog proves they agree:

```lean (name := bzInferred)
-- Check that inferred broadcast evidence selects the same
-- tensor operation.
example :
    Tensor.broadcastTo (s₂ := [2, 3])
      Spec.Shape.BroadcastTo.proof bzRow =
    Tensor.broadcastTo
      ShapeAndBroadcast.row_broadcast bzRow :=
  ShapeAndBroadcast.broadcast_infer_eq bzRow
```

For an image, `Tensor.repeatLeading 1` inserts the singleton batch dimension. We use the existing
tensor operation rather than an image-specific wrapper; the result type records the shape change:

```lean (name := bzBatchSig)
-- Read the inserted leading batch dimension in the result
-- type.
example {α : Type} [Storage α]
    (image : Tensor α [100, 100, 3]) :
    Tensor α [1, 100, 100, 3] :=
  Tensor.repeatLeading 1 image
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

Max shifting prevents a large exponential, but it does not prevent every rounding failure.
For binary32 logits `[9, -9, -9]`, the two small exponentials survive while adding them to `1`
loses them. A direct `log` of that rounded normalizer then returns zero. Our log-softmax sums
the non-pivot exponentials separately and uses `log1p`; its backward rule also keeps the small
complement instead of subtracting a rounded probability from one.

Distillation has a different cancellation problem: subtracting independently rounded student
and teacher probabilities can erase a small gradient. `TorchLean.Loss.distill` accepts their
logits, holds the teacher fixed, and computes the student gradient from logit differences.
The shared specification supports configurable scalar precision; the native CUDA tape uses
binary32 or binary64. The real derivative theorem is
`Proofs.Autograd.hasFDerivAt_distillSpec_vec` for nonempty class vectors. It does not assert a
rounding-error bound for every format or verify ATen's execution.

These numerical issues and the logit-difference formulation are discussed in
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642).

Long rows can also exhaust the denominator's precision. Adding `1` repeatedly in binary16
stops changing the sum at `2048`, so a left-fold softmax over `4096` equal logits would return
twice the correct probability. A format with a small exponent range can overflow the sum even
earlier. The masked and unmasked specifications share a normalizer that detects range loss or
a swallowed nonzero term, then reconstructs the sum and quotients from the finite input
coefficients before rounding the outputs. It cannot recover an exponential that was already
rounded to zero. These summation limitations are discussed by
[Blanchard, Higham and Higham](https://arxiv.org/abs/1909.03469); they are separate from backward
cancellation.

Activation derivatives need the same care. We keep their inputs available: sigmoid uses a decaying
exponential ratio, and tanh squares a scaled reciprocal cosh instead of subtracting its rounded
output's square from one. GELU's tanh-approximation gate is evaluated as an equivalent sigmoid;
ELU uses `expm1` near zero. SiLU splits its negative exponential into two factors and multiplies
by the input before the second factor: the product can remain representable even when the
sigmoid alone underflows. GRUs compute complementary update gates as `sigmoid (-x)` instead of
`1 - sigmoid x`, so rounding the update gate to one need not erase the candidate contribution.
These changes preserve the real functions and their proved derivatives
while retaining small floating-point results that direct subtraction can lose.

The incoming gradient matters too. For binary32, the derivative of `tanh 60` rounds to zero on
its own, but multiplying the exact derivative by an incoming gradient of `1e30` gives about
$`3.067 \times 10^{-22}`, which is representable. Forming the rounded slope first loses that
answer. The sigmoid and tanh pullbacks keep the incoming gradient inside their range repair;
the CPU tape, forward-mode rules and recurrent gates share those pullbacks. Standalone slope
functions use the same repair without inserting a multiplication by one. Native GPU calls
use scaled exponentials for the same purpose. Configured precision moves these boundaries but
does not remove them. Near zero, differentiating an exponential quotient can also subtract nearly
equal terms, even when none of them underflows. Nested floating-point derivatives use reciprocal
cosh identities instead: their odd coefficients retain the small input through `sinh`. They check
the input, hyperbolic factor and slope for range loss. A factor rounded to one also requires a
complete enclosure; a product of small directions can vanish even at a zero input. When an
intermediate loses range, scalar and CPU
pullbacks can enclose the complete expression from the original finite inputs. A cheap exponential
bound first checks whether every coefficient must underflow. Otherwise, they refine the working
precision and approximation degree, accepting only matching finite rounded coefficient bounds.
Each candidate carries a Lean proof of real containment for every coefficient. Final codec
rounding and correspondence with each caller's derivative remain separate obligations.
An inconclusive search retains the staged calculation. This additional path does not verify
the GPU implementation or assert correct rounding for every transcendental call.

{src "NN/Examples/BugZoo/StableLoss.lean"}[NN.Examples.BugZoo.StableLoss source] is about losses
that look mathematically harmless but fail numerically. One example is `softmax` followed by
`log`: if a probability rounds to zero, the log path can produce infinities and downstream NaNs.
TorchLean keeps two APIs separate. Logits use `crossEntropyLogitsSpec`, which retains the original
logits alongside log-softmax; probability inputs use the clipped probability form
`crossEntropySpec`.

For probability-based binary cross-entropy, even a nonzero input can lose its loss before the
logarithm. With target zero, the term is $`-\log(1-p)`: a sufficiently small positive $`p` leaves
`1 - p` rounded to one. `binaryCrossEntropySpec` uses `log1p (-p)` inside the clipping interval
to retain that term. At the upper clamp it keeps epsilon as the complementary probability.
The clipping thresholds and selected gradients are unchanged; epsilon must still be positive
and suitable for the chosen format.

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

For finite real logits, the two formulas define the same function. In this computation, however,
the probability path has already rounded two entries to zero. Applying `log` gives $`-\infty`;
multiplying such an entry by a zero target can then produce NaN.

Even direct log-softmax has a finite-range limit. The difference between two finite logits can
overflow, although a zero or small target would make its weighted loss contribution representable.
For those entries, we keep the original logits and weight each operand before subtracting them.
Ordinary finite log-probabilities keep the existing multiplication.

```lean (name := bzExtremeCE)
/-- Finite logits with an overflowing class difference. -/
def bzExtremeLogits : Tensor Float32 [3] :=
  [3e38, 3e38, -3e38]

#eval (Spec.crossEntropyLogitsSpec 0 bzExtremeLogits
  (Tensor.from (#[1, 0, 0] : Array Float32))).toFloat
```
```leanOutput bzExtremeCE (whitespace := lax)
0.693147
```

This loss is about $`\log 2`, rather than NaN. A zero-target mask could produce the right loss
while discarding a valid derivative with respect to that target weight. We retain both pullbacks:

```lean
open Proofs.Autograd in
#check crossEntropyLogitsSpec_eq_dot
open Proofs.Autograd in
#check hasFDerivAt_crossEntropyLogitsSpec_vec
open Proofs.Autograd in
#check hasFDerivAt_crossEntropyTargets_vec
```

These theorems state the real formula and its logits and target-weight derivatives for class
vectors, without assuming the weights sum to one. They do not assert bitwise equivalence with the
old floating-point expression or prove ATen's execution. Overflowing reductions and nonfinite
inputs still need separate treatment. Related zero-weight failures have been
[reported in Optax](https://github.com/google-deepmind/optax/issues/896).

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

The backward calculation matters in attention too. TorchLean centers the incoming gradient before
its weighted softmax reduction, so a dominant probability rounded to one need not erase its logit
gradient. Both attention derivatives use that calculation, including Boolean masks. Fully blocked
rows and rows with one allowed key have zero key derivative over the reals. If large finite
incoming gradients overflow during centering, supported CPU scalars replay the complete weighted
calculation from finite coefficients. This still uses the supplied weights: it cannot reconstruct
a probability already rounded to zero. The rounded-real error bound includes the centering
subtraction; native GPU execution remains a separate boundary.

### Squaring before scaling

Cosine loss compares directions. Multiplying a nonzero vector by a positive scalar should not
change its angle, but squaring large coordinates can overflow before we have a chance to normalize
them. TorchLean scales before squaring, normalizes each tensor, and then takes their dot product.
The epsilon floor is scaled too; it still clips the norm rather than adding to its square.

```lean (name := bzCosine)
def bzLargeVector : Tensor Float32 [2] := [3e20, 4e20]

#eval (Spec.cosineSimilaritySpec bzLargeVector
  (Tensor.from (#[1, 0] : Array Float32))).toFloat
```
```leanOutput bzCosine (whitespace := lax)
0.400000
```

Computing the raw squares first would give a loss of `1` for this input, rather than `0.4`.
The pullback also uses normalized coordinates and separate divisions: a product of large norms
can overflow and erase a representable gradient even when the forward loss looks reasonable.
The real identities cover any tensor shape and preserve the clipped branch, including at its
epsilon floor:

```lean
open Proofs.Normalization in
#check cosineSimilarity_eq
open Proofs.Normalization in
#check cosineSimilarityDeriv_eq
```

Normalizing before the dot product is established practice; PyTorch explains the same issue in its
[source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Distance.cpp).
Scaling the norm calculation additionally keeps large coordinates from being squared directly.
These real identities are not a bound on all floating-point errors or a proof of external kernels.

Huber's quadratic branch has a smaller ordering trap. An FP32 residual of $`2^{64}` has a
representable half-square, $`2^{127}`, although its square overflows. We compute `(x / 2) * x`,
not `(x * x) / 2`. Its linear branch also keeps the prediction and target available: if their
difference overflows, a small positive `delta` can still make the weighted loss representable.
We weight each operand before subtracting, then retain the quadratic offset. The Huber branch
condition, mean reduction, and selected gradient are unchanged. Genuine result overflow and
overflowing reductions still need separate treatment.

### Correcting an optimizer moment

Adam averages squared gradients and then corrects the initial bias in that average. These are
two opportunities to overflow before a representable update. Let's take a single FP32 gradient
of $`2^{68}` and set both decay coefficients to $`1-2^{-10}`:

```lean (name := bzAdamRange)
#eval do
  let parameters : Tensor Float32 [1] := [0]
  let g : Float32 := Rat.cast (2^68 : Rat)
  let gradients := Tensor.full [1] g
  let beta : Float32 := Rat.cast (1023/1024 : Rat)
  let state := Optim.Adam.init 1 beta beta 1 parameters
  let next := Optim.Adam.update state parameters gradients
  return (next.parameters.getScalar ⟨0, by decide⟩).toFloat
```
```leanOutput bzAdamRange (whitespace := lax)
-1.000000
```

The gradient's square, $`2^{136}`, does not fit. Its weighted contribution, $`2^{126}`, does,
so we apply the weight before a square that would overflow. Bias correction would restore the
unrepresentable $`2^{136}`, but the update only needs its square root, $`2^{68}`. In that case,
we divide the square root of the stored moment by the square root of its correction denominator.
If a correction reciprocal itself overflows, we divide the moment directly instead.

There is a different trap at small learning rates. In binary16, a gradient of $`256`, the same
decay coefficients, epsilon $`1`, and learning rate $`2^{-19}` should produce a nonzero update.
Dividing the learning rate by the corrected RMS first rounds that factor to zero. Adam, RMSProp
and AdaGrad now normalize the moment or gradient before applying the learning rate. If the
supported scalar's checked operations detect range loss, the CPU update replays that weighted
quotient from its finite input coefficients and rounds it in the selected format. This also
handles the opposite case, where the unscaled quotient overflows before a small rate is applied.
It cannot restore information lost before those inputs were formed.

These are evaluation-order changes to the [Adam equations](https://arxiv.org/abs/1412.6980),
not a different optimizer. RMSProp and Adadelta share the weighted-square calculation. The
rounded-real AdamW theorem follows the actual normalized-moment calculation, with a revised
parameter-error budget for division followed by learning-rate scaling. That model has no upper
exponent limit, and its scalar context does not enable finite-coefficient replay. Its theorem
does not establish the external GPU adapter's execution. Stored moments, parameter updates
and their sums can still overflow; this example is not a guarantee that every finite gradient
produces a finite optimizer state.

## Ignored Labels

{src "NN/Examples/BugZoo/IgnoredLabelLoss.lean"}[NN.Examples.BugZoo.IgnoredLabelLoss source]
examines a mean loss when no labels remain active.
[PyTorch issue #75181](https://github.com/pytorch/pytorch/issues/75181) reported an `ignore_index`
case where every label was ignored and the result was `nan`. A mean over no active labels needs
an explicit convention: dividing a zero loss sum by a zero active count does not supply one.

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

The active flag selects a branch:

$$`\operatorname{labelContribution}(active,loss)
=
\begin{cases}
loss, & active\\
0, & \neg active
\end{cases}`

The corresponding theorem states that the false branch is zero at any element
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
zero by an infinity. A derivative of the division also depends on that zero denominator. Masking
the final value does not remove the division from the computation being differentiated.

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
relevant observations are the numerator and denominator cotangents as well as the loss. The local
unfolding theorem specifies the guarded forward formula that a derivative argument would have
to follow.

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

The displayed weights are zero at every strict-future position. Reversing the mask would fully
block the last row. A direct softmax over only negative infinities encounters a zero normalizer;
max-subtraction also encounters the undefined difference $`-\infty-(-\infty)`.
TorchLean's hard-mask contract gives blocked positions an explicit zero weight, including fully
blocked rows. The Lean composition over CUDA primitives preserves that choice in both forward
evaluation and its local VJP.

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

Finite arithmetic adds another difficulty. A dot product can overflow before division by
$`\sqrt{d}`, even when the scaled score fits. Pre-scaling the queries avoids some overflows, but
can instead round a tiny query to zero before it meets a large key. The reference attention
calculation keeps its ordinary order and, when checked operations lose range or absorb a nonzero
term, reconstructs the complete scaled contraction from its original finite coefficients.
The model-operation interface `Ops.matmul` accepts a fixed `scale`, so biased and dropout attention
can keep that scale inside the score contraction too. Its IR node and backend audit distinguish
this calculation from an ordinary fixed-left matrix product followed by multiplication.

The native GPU contraction uses a conservative range flag. A flagged call can download the
original finite operands and run the same coefficient recovery on CPU before returning a GPU
buffer. The buffer interface keeps its binary64 scale even for binary32 operands: narrowing
$`10^{-46}` first would turn it into zero before large inputs could amplify it.
Reading the flag synchronizes the device; exact-zero and orthogonal products can also
take this recovery path. Original nonfinite inputs retain the native result without this transfer.
This is not a proof of ATen or of correctly rounded softmax: later native exponential, division
and aggregation stages still round. The grouped IR operation has a separate numerical range rule.
Its scale is an ordinary scalar parent, so source-range and payload checks bind it just like a
weight. The rule encloses the exact-real contraction and includes the finite staged interval when
available. Its reference policy records the choice between fixed-left arithmetic and exact
coefficient replay; it does not accept an implementation-defined native reduction. Concrete
binary32 execution must still pass the node-by-node interval replay.

Even constructing the denominator needs care: FP16 cannot represent $`65536`, but its square
root is $`256`. Casting first would give infinity and erase the score scale. `Context.sqrtNat`
keeps the ordinary calculation when the count cast does not overflow, and otherwise asks
FloatLib to round the exact rational root directly in the chosen format. Dual and complex
contexts lift this same constant construction from their component scalar type.

The same boundary matters in backward mode. The product $`dO\,V^\top` can overflow before the
softmax pullback cancels it. A fully blocked row should have zero gradient; a singleton softmax
should have zero score tangent. The fused reference pullback retains the original contractions
rather than trying to recover them from an already infinite intermediate tensor. This repair uses
the existing scalar coefficient interface, including configured binary precisions and nested duals.
Biased and dropout attention use `Ops.softmaxMatmul` to keep that boundary intact on the existing
tape. Its multiplier can carry gradients too; a sampled dropout mask follows the existing
stop-gradient convention. Both the JVP and VJP retain the original contraction terms, so recovery
can also handle cancellation between large tangent terms or a tiny intermediate amplified later.
The real-valued proofs in
{src "NN/Proofs/Autograd/FDeriv/HardMaskedAttention.lean"}[`HardMaskedAttention`] differentiate
this forward function and show that its JVP and backward rule are adjoint for a fixed optional mask,
including fully blocked rows.
The native GPU backward checks the contraction and centered pullback. When recovery is needed,
it can bring the original finite coefficients to CPU and return the recovered gradient to GPU.
This includes the native attention adapter; it does not prove the ATen kernels or a numerical
error bound for every native execution. Fully blocked rows discard their incoming cotangent;
the reference rule checks all scalar metadata before treating a weight row as zero.

## Compiler Boundary

This entry is about optimized graphs that run, return tensors, and are nevertheless not the same
computation as the source program. [NNSmith](https://arxiv.org/abs/2207.13066) found compiler bugs
across TVM, TensorRT, ONNXRuntime, and PyTorch,
[FreeFuzz](https://arxiv.org/abs/2201.06589) found framework and API bugs by mining real snippets,
and a recent [PyTorch compiler correctness study](https://arxiv.org/abs/2604.08720) focuses directly
on silent `torch.compile` wrong outputs.

The translation a verifier depends on is the lowering of a TorchLean forward program to the IR
graph it bounds. Its preservation theorem is
{src "NN/Verification/Builtin/Proved/Correctness/Eval/Main.lean"}[`runForwardIR_eq_evalForward`]:

```lean (name := bzLowering)
-- Lowering a typed forward program and running its IR
-- gives the source program's result.
open NN.Verification.Builtin.Proved.Correctness in
#check @runForwardIR_eq_evalForward
```
```leanOutput bzLowering (whitespace := lax)
@runForwardIR_eq_evalForward : ∀ {α : Type} [inst : Storage α]
  [inst_1 : Context α] {paramShapes : List Shape}
  {inShape outShape : Shape}
  (p : NN.Verification.Builtin.Proved.ForwardProgram α
    paramShapes inShape outShape)
  (params : TensorPack α paramShapes) (x : Tensor α inShape),
  NN.Verification.Builtin.runForwardIR
      (NN.Verification.Builtin.Proved.lowerForwardProgramToIR
        p params) x =
    NN.Verification.Builtin.Proved.evalForward p params x
```

There is no success premise and no excluded operator. The equality is between two `Except`
results, so it covers failures too: a raw logarithm of a nonpositive input fails the same way in
the source and in the lowered graph. The compared functions are Lean denotations; generated machine
code and external kernels require separate conformance evidence.

The reusable statement is:

$$`\forall p\;\theta\;x,\quad
\operatorname{runIR}(\operatorname{lower}(p,\theta),x)=\operatorname{eval}(p,\theta,x)`

The parameters are part of the comparison: lowering stores them in the graph's payload, and using a
different constant or parameter tensor would describe a different source computation. The
universal input quantifier is also stronger than checking a calibration batch.

## Float Boundary

{src "NN/Examples/BugZoo/FloatBoundary.lean"}[NN.Examples.BugZoo.FloatBoundary source]
separates real arithmetic from the binary32 model. Finite precision, exceptional values, fused
operations, subnormal behavior, and changed reduction order can invalidate a property proved only
over reals {Informal.citep jiarinard2020}[]. Goldberg describes the underlying rounding effects
{Informal.citep goldberg1991}[]; Flocq {Informal.citep flocq2011}[] and the Coq treatment of
numerical
programs {Informal.citep boldo2015}[] develop mechanized ways to reason about them.

Lean gives ordinary `Float32` core arithmetic a logical definition through `Float32.Model`.
FloatLib's `ExecFloat.Binary 8 23` supplies configured binary32 arithmetic and native-model
connections. Adding a tenth to a fifth lets us compare the two results at the bit level:

```lean (name := bzBits)
open FloatLib.Floats in
/-- Lean's own binary32 add next to the executable model. -/
def bzBits (a b : Float32) : UInt32 × UInt32 :=
  ((a + b).toBits,
    ExecFloat.Binary.toBits32
      ((ExecFloat.Binary.ofFloat32 a :
          FloatLib.Floats.Binary32) +
        ExecFloat.Binary.ofFloat32 b))

#eval bzBits 0.1 0.2
```
```leanOutput bzBits (whitespace := lax)
(1050253722, 1050253722)
```

The identical bit patterns identify the rounded result even when a decimal printer hides
low-order bits. This one addition is a checked example. The following proofs state the general
relation for
addition and subtraction, including the finite-input hypotheses:

```lean (name := bzAddThm)
open FloatLib.Floats in
example (a b : Float32)
    (ha : a.isFinite = true) (hb : b.isFinite = true) :
    (ExecFloat.Binary.ofFloat32 (a + b) :
      FloatLib.Floats.Binary32) =
        ExecFloat.Binary.ofFloat32 a +
          ExecFloat.Binary.ofFloat32 b ∧
      (ExecFloat.Binary.ofFloat32 (a - b) :
        FloatLib.Floats.Binary32) =
        ExecFloat.Binary.ofFloat32 a -
          ExecFloat.Binary.ofFloat32 b :=
  ⟨ExecFloat.Binary.ofFloat32_add_of_isFinite a b ha hb,
    ExecFloat.Binary.ofFloat32_sub_of_isFinite a b ha hb⟩
```

The inputs may be signed zeros or subnormals, and the result may overflow to infinity. NaN and
infinite inputs are outside these two statements. FloatLib's square-root export theorem has a
different domain: it covers every configured input, with NaNs canonicalized when exported to Lean.
The catalog also shows configured division refining its software model; that theorem does not
assert native `Float32` division agreement. A compiled CPU instruction or CUDA kernel is a separate
execution boundary; none of these theorems establishes conformance of a particular vectorized
kernel.

The pair of integers in the addition output is useful because the comparison is about stored
bits, not a tolerance chosen after printing. The theorem identifies those same configured values
for arbitrary finite operands. Its input conditions are part of that conclusion, not an optional
runtime check. Native export has a separate NaN policy: payload and sign distinctions can be lost
when Lean's canonical NaN is constructed. A property about payload preservation would require a
more precise relation. Likewise, a theorem about one addition does not fix the order in which a
network's long reduction performs many additions; that order belongs to the surrounding computation.

The conversion into a scalar type matters too. A uniform draw of $`2^{30}/2^{32}` is just $`1/4`,
but casting its numerator and denominator separately to FP16 gives infinity divided by infinity.
That NaN can turn a valid dropout keep decision into a dropped entry. `Spec.Random.sampleUnit`
forms the exact rational fraction first and uses the scalar context's cast. It keeps the same
integer generator, seed and counter; the chosen precision rounds the resulting fraction, which
can still round up to $`1` at the top of the grid.

Parameter initialization has a related range boundary. Two finite uniform bounds can have an
infinite difference: the interval from minus the largest binary64 value to that value is one
example. The CPU and CUDA initializers keep their ordinary affine formulas when the width is
finite in the destination type; CUDA retains its fused multiply-add. If the width overflows,
they instead form the convex combination of the two finite endpoints, avoiding that intermediate
infinity.

Xavier and He bounds can lose range in the fan count rather than the root. Casting an enormous
count to infinity makes $`\sqrt{6/\mathrm{fan}}` zero, even when the bound is representable.
Their shared initializer preserves the ordinary formula for nonzero bounds; otherwise it asks
FloatLib to round the square root of the exact rational ratio. This also handles bounds in the
binary64 subnormal range without first rounding the ratio to zero.

At a minimum or maximum tie, TorchLean selects an equal split between the inputs. Its JVP
therefore averages their supplied tangents. Halving two smallest subnormal values separately
would give zero, although their average is still representable. Dual numbers and typed graphs
use the same checked average: if halving loses range, they combine the finite coefficients
before rounding the result. The half constant is cast from an exact rational, so the rule also
works in configured formats that cannot represent `2`. Reverse mode still rounds distinct input
nodes' cotangents separately; a half that is itself unrepresentable cannot be stored in that
scalar type. When both arguments refer to the same tape node, the checked CPU path and native
GPU path combine its contributions before graph accumulation. This retains a smallest-subnormal
gradient for $`\max(x,x)` or $`\min(x,x)`. It also makes the accumulation convention explicit:
the combined contribution is added once, rather than adding two separately rounded halves.
Unsupported coefficient carriers accumulate the two contributions separately.

Square-root gradients need the upstream value too. Forming $`1/(2\sqrt{x})` first can lose range
before multiplication by that value, and some configured formats cannot represent `2` at all.
Dual numbers, typed graphs and the CPU tape share a weighted quotient that replays finite
coefficients when a checked intermediate loses range. It uses the root already rounded by the
forward arithmetic; it does not claim correctly rounded evaluation of the unrounded analytic
derivative. The existing zero-factor convention outside the positive region is unchanged.
The CUDA bridge uses direct division by twice the root, avoiding a separately rounded
reciprocal. It retains its literal-zero branch outside the positive region; unlike multiplying
by a zero factor, this branch also returns zero for a nonfinite upstream value.

The angle in a complex logarithm has a related cancellation. Moving radially should leave the
angle unchanged. Scaling tiny coordinates can nevertheless make both tangent terms infinite,
turning their zero difference into NaN. The dual angle rule checks the scaled calculation and,
after range loss or cancellation, reconstructs the whole finite quotient from the original
coordinates and direction. It leaves the primal angle and the origin's selected zero tangent
unchanged. A principal angle still has a branch cut; this rule does not make it differentiable
across that cut. The {src "NN/Proofs/Autograd/Dual/Domain.lean"}[real-valued derivative proof]
uses Mathlib's derivative of the complex logarithm away from the cut.

Powers have a domain issue as well. Applying the logarithmic derivative formula to $`x^2` at
zero or a negative input evaluates a logarithm or quotient that the polynomial does not need.
The dual rule uses $`yx^{y-1}x'` when the exponent tangent is identically zero. Its zero tests
inspect every nested coefficient; an active mixed derivative must not disappear just because
its primal is zero. The same test guards the captured-exponent VJP. The
{src "NN/Proofs/Autograd/Dual/Domain.lean"}[constant-power derivative theorem] checks the actual
dual rule over the reals. It is not a floating-point accuracy theorem: a staged power or slope
can still lose range before multiplication by the incoming derivative. For example, with
$`x=2^{100}` and incoming derivative $`2^{-100}`, the cubic's weighted derivative is the finite
binary32 number $`3\cdot 2^{100}`, although the local factor $`3x^2` overflows. The dual rule and
captured-exponent VJP check these intermediates. For a constant exponent, they can refine the
whole weighted expression from its original inputs using FloatLib's logarithm and exponential
enclosures. With a varying exponent and a positive base, recovery instead encloses the complete
expression $`\exp(y\log x)(y'\log x+yx'/x)`. Keeping both terms together matters: an overflowing
quotient can be multiplied by an underflowed power even when the final derivative fits.
Recovery requires every lower and upper derivative coefficient to round alike, including mixed
coefficients of nested duals. Refinement has a bounded work budget based on the input encodings;
it declines oversized exponential arguments before evaluating them. Nonfinite inputs, unsupported
domains and inconclusive bounds retain the ordinary result. The real-valued derivative theorem
checks the varying-exponent rule on positive bases; neither it nor the recovery routine verifies
native power evaluation.

For a mean, we must choose the accumulation precision as well. A left-to-right sum of 65,536 FP16
ones loses increments, and the FP16 conversion of the element count overflows. The same-precision
mean returns zero although the exact mean is one. TorchLean keeps this scalar policy explicit:
{src "NN/Tensor/Reductions.lean"}[`Tensor.foldl`] lets us convert each entry into a wider
accumulator without allocating a second tensor. Converting the tensor before calling `mean` is
another option. Both give one for this example with a binary32 accumulator; neither promises exact
arithmetic or prevents overflow for every input. The ordinary `mean` keeps its reference definition
and does not silently promote the scalar type.

## Normalization State

Normalization also needs care before the square root. In binary32, squaring `[3e20, 4e20]`
overflows even though its normalized entries are ordinary numbers. RMSNorm and regularized L2
normalization scale the input before squaring and divide epsilon by the same scale twice.
Scaling only the input would change the model's epsilon. The scale is detached because it
cancels from the real function; the existing tape records the remaining operations as usual.
`Spec.normalizeL2Regularized_eq` checks that real identity, including the mean-square variant,
for positive epsilon. It is not a floating-point error bound.

Safe norm scaling is established numerical analysis, not a new failure mechanism; see
[Algorithm 978](https://doi.org/10.1145/3061665) and
[PyTorch's notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).
The chosen format still limits the reduction and gradients, and nonfinite inputs can propagate
NaN differently from the unscaled formula.

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

The second result is about 316 times the first. Both computations use Lean's binary64 `Float`,
so the difference comes from the placement of epsilon. Comparing this reference with a native
binary32 implementation also requires accounting for the change in scalar precision.

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
#check @batchNorm_affine
```
```leanOutput bzAffine (whitespace := lax)
@batchNorm_affine :
  ∀ {channels : ℕ} {sSpatial : Shape}
    (stats : RunningStats channels)
    (gamma beta : Tensor ℝ [channels])
    (epsilon : optParam ℝ normalizationEpsilon),
  ∃ scale bias,
    ∀ (x : Tensor ℝ (sSpatial.prependDim channels)),
      batchNorm x stats gamma beta epsilon =
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

Over real arithmetic, one feature has centred value $`x-x=0`, so the output is the bias and
the input and scale gradients vanish when epsilon is positive. The bias gradient
does not vanish in general; it is the upstream cotangent. The catalog evaluates that at a
deliberately awkward input, $`x=10^6`, with
weight 2, bias 3, and epsilon $`10^{-5}`:

```lean (name := bzLnFwd)
-- A one-feature normalized value should leave only the
-- affine bias.
#eval LayerNormDegenerateAxis.forward
```
```leanOutput bzLnFwd (whitespace := lax)
3.000000
```

```lean (name := bzLnDw)
-- The scale gradient multiplies the zero centered
-- activation.
#eval LayerNormDegenerateAxis.backward.scaleGradient[0]
```
```leanOutput bzLnDw (whitespace := lax)
0.000000
```

```lean (name := bzLnDx)
-- The input gradient cancels when the normalization axis
-- has one feature.
#eval
  let dx := LayerNormDegenerateAxis.backward.inputGradient
  dx[((0 : Fin 1), (0 : Fin 1))]
```
```leanOutput bzLnDx (whitespace := lax)
0.000000
```

These evaluations return the bias and two zero gradients. The scale-gradient contract below
quantifies over real inputs:

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

The three Lean outputs separate the forward value, scale gradient, and input gradient so that
agreement in one cannot hide disagreement in another. Bias three survives normalization, whereas
the scale multiplies a zero normalized activation. These outputs do not measure native-framework
residuals or training accuracy.

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
independently. Take the second row out of a batched squaring:

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

Batch independence is a property of the chosen reference computation. The function passed to
`mapLeading` receives one row, so it cannot inspect neighboring requests through that input. An
operation that intentionally computes statistics across the batch has a different interface and
should not satisfy this rowwise equation.

A native implementation introduces another question: does evaluating a row alone produce the same
bits as evaluating it inside a larger batch? Different matrix shapes can select different reduction
orders. To test the native path, compare the same row and weights at both batch sizes with dropout
disabled. Applying the reference theorem to that execution requires a conformance result or an
explicit numerical error bound; a small observed discrepancy supplies neither by itself.

## KV Cache

{src "NN/Examples/BugZoo/KVCache.lean"}[NN.Examples.BugZoo.KVCache source] models cache accounting
in autoregressive inference. LLM engines fail through shifted caches, wrong cache slots, config and
shape mismatches, resource scheduling, and interactions with positions or tokenizers. The broader
source trail is again the [LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

Append a key and a value to a one-token cache and look at the keys:

```lean (name := bzKV)
open KVCache in
/-- A one token cache holding a zero key and value. -/
def bzCache : Cache Float 1 [2] :=
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
  ∀ {α : Type} [inst : Storage α] {seqLen : ℕ} {tokenShape : Shape}
    (cache : Cache α seqLen tokenShape)
    (newKey newValue : Tensor α tokenShape),
  (appendKV cache newKey newValue).keys[seqLen] = newKey
```

This theorem covers the newly written slot. Preservation of earlier slots is visible in the
example and follows the append definition, but the catalog does not state that preservation
theorem. Proving it would require reasoning about indices in the left part of the concatenation;
the final-slot theorem uses the right-part indexing lemma.

This example has two axes with separate roles: sequence length counts stored token positions and
head dimension counts coordinates of each key or value. The general cache accepts any token shape;
appending increases the sequence length while preserving that shape.
Checking only the final key would still miss a value written to the wrong
slot, so the source includes the corresponding final-value theorem too. Full cached decoding
needs more than these append facts: the query must read the matching key/value prefix using the
same mask and positions as full-sequence attention. These local contracts identify pieces of
that larger comparison without claiming it has been proved here.

## RoPE Position

{src "NN/Examples/BugZoo/RoPEPosition.lean"}[NN.Examples.BugZoo.RoPEPosition source] pairs naturally
with the KV cache example. Rotary position embeddings make position accounting part of the model's
meaning. A decode position off by one can be hard to notice because the shapes still line up and the
model still produces tokens.

We'll keep the positions in a tensor. Appending grows its length by one, and the default position
is the old sequence length:

```lean (name := bzRope)
/-- Positions already issued for two decoded tokens. -/
def bzSchedule : Tensor Nat [2] := [0, 1]

#eval RoPEPosition.append bzSchedule
```
```leanOutput bzRope (whitespace := lax)
[0, 1, 2]
```

```lean (name := bzRopeThm)
-- The appended position follows the zero-based
-- sequence-length convention.
open RoPEPosition in
#check @append_last
```
```leanOutput bzRopeThm (whitespace := lax)
@append_last :
  ∀ {seqLen : ℕ} (schedule : Tensor ℕ [seqLen])
    (position : optParam ℕ seqLen),
  (append schedule position)[seqLen] = position
```

In this zero-based example, the appended position is the old sequence length, not “previous
position plus one.” If our cache uses an offset or nonconsecutive positions, we pass the next
position explicitly:

```lean (name := bzRopeOffset)
#eval RoPEPosition.append bzSchedule (position := 12)
```
```leanOutput bzRopeOffset (whitespace := lax)
[0, 1, 12]
```

A position schedule separates a token's storage slot from the position used by its embedding.
They coincide in the displayed `[0, 1, 2]` example, which makes the append convention easy to read.
They need not coincide after a window has been shifted or a prefix has been assigned an offset.
The theorem covers whichever position we supply. The default is `seqLen`, regardless of the last
stored position. No rotation is evaluated by this example; it checks the bookkeeping that supplies
positions to a rotation implementation.

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
def bzTokenizer : Metadata :=
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
#check @token_lt_vocabulary
```
```leanOutput bzTokThm (whitespace := lax)
@token_lt_vocabulary :
  ∀ {vocabularySize sequenceLength : ℕ}
    (sequence : Tensor (Fin vocabularySize) [sequenceLength])
    (position : Fin sequenceLength),
  ↑sequence[position] < vocabularySize
```

Every token id constructed at this type is in range. An importer or tokenizer bridge must establish
that bound when converting external data, and `padding_lt_vocabulary` gives the corresponding
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
checks a camera projection. Vision and robotics pipelines use a camera matrix to project 3D points
to image coordinates:

$$`(u,v)=\left(\frac{x}{z},\frac{y}{z}\right)`

up to intrinsics and coordinate conventions. This can fail when depth is zero or has
the wrong sign, coordinate frames are swapped, or the denominator convention is implicit. A tensor
can still be emitted, but the geometry claim no longer matches the camera model.

This entry uses a decidable certificate check. The example camera has focal length 100 and
principal point at the image centre:

```lean (name := bzCam)
open NN.Verification.Geometry3D.Box3D in
/-- A pinhole camera, focal length 100, centre at 50. -/
def bzCamera : CameraP ℚ :=
  [[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]]

/-- Eight cube corners, four to six units away. -/
def bzCorners : Tensor ℚ [8, 3] :=
  [[-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4],
   [-1, -1, 6], [1, -1, 6], [1, 1, 6], [-1, 1, 6]]

#check bzCorners
```
```leanOutput bzCam (whitespace := lax)
bzCorners : Tensor ℚ [8, 3]
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

Changing only the claimed box tests the enclosure condition while keeping the camera and corners
fixed. The pair does not independently test every conjunct. `checkCert`
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
uncertainty and rounding error.

The certificate carries its own `pointCount`; eight is the default used by this cuboid example.
Its point tensor has shape `[pointCount, 3]`, and the depth and containment conditions quantify
over those supplied points. A different count describes the same projection problem for another
finite point set. With zero points, the pointwise conditions are vacuous, while the image and box
checks still apply. Acceptance does not establish that the points are the corners of a cuboid or
that a detector found the right object.

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
above covers the verifier's typed fragment; for float32 execution, a numerical conformance argument
must also account for rounding and the selected kernels.

After adding a catalog entry, import it in `All.lean` so the maintained catalog build includes it:

```terminal
# Check that every maintained catalog entry remains
# reachable from All.
scripts/lake.sh env lean NN/Examples/BugZoo/All.lean
```

# Verification Scope

The theorems describe TorchLean reference objects. A numerical comparison measures a particular
implementation under its selected dtype, device, and execution policy. Establishing a deployment
guarantee requires connecting that runtime to the reference object and the theorem's hypotheses.

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
- FloatLib's native add/sub proofs require finite operands. Its square-root export theorem
  covers all configured inputs through NaN canonicalization. Compiled CPU instructions, vectorized
  reductions, and CUDA kernels are separate boundaries with no theorem here.
- The 3D projection check is exact at `ℚ`. At `Float` the same theorem still establishes
  `Verified3DBox` for the float-valued predicate. Exact real containment needs the interval
  statement described above and an argument that the float executor's intervals enclose the
  projection.
