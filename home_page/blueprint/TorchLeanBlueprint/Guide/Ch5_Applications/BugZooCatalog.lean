import VersoManual
import FloatLib.Floats.Formats.IEEE754
import TorchLean.NN.LoRA
import TorchLean.NN.Autograd.Function
import TorchLean.NN.Autograd.Frontend
import TorchLean.NN.Autograd.Differential
import TorchLean.NN.Autograd.Complex
import TorchLean.NN.CLI.Training.Command
import TorchLean.NN.Data.Synthetic
import TorchLean.NN.Data.CausalLM
import TorchLean.NN.Models.KAN
import TorchLean.NN.Models.CNN
import TorchLean.NN.Models.ResNet
import TorchLean.NN.Models.UNet
import TorchLean.NN.Models.Recurrent
import TorchLean.NN.Models.CausalTransformer.LanguageModel
import TorchLean.NN.Models.Mamba
import TorchLean.NN.Models.MaskedPatchReconstructor
import TorchLean.NN.Models.Diffusion.NoisePredictor
import TorchLean.NN.Models.FNO
import TorchLean.NN.Models.PPO
import TorchLean.NN.Builder.MoE
import TorchLean.NN.Sequential.Training
import TorchLean.NN.RL.CLI
import TorchLean.NN.RL.PPO.ActorCritic
import TorchLean.NN.SelfSupervised
import TorchLean.NN.Text.Vocabulary
import TorchLean.NN.Text.GPT2BPE
import TorchLean.NN.Text.Generation
import TorchLean.NN.Trainer.Constructor
import TorchLean.NN.Trainer.Train
import TorchLean.NN.Data.Image
import TorchLeanExamples.BugZoo.Broadcasting
import TorchLeanExamples.BugZoo.StableLoss
import TorchLeanExamples.BugZoo.IgnoredLabel
import TorchLeanExamples.BugZoo.AutogradDomain
import TorchLean.Verify.CROWN.Arithmetic
import TorchLeanExamples.BugZoo.BatchNorm
import TorchLeanExamples.BugZoo.LayerNormAxis
import TorchLeanExamples.BugZoo.ConstantSlice
import TorchLeanExamples.BugZoo.KVCache
import TorchLeanExamples.BugZoo.RoPEPosition
import TorchLeanExamples.BugZoo.Tokenizer
import TorchLeanExamples.Verification.Geometry3D.Box3D
import TorchLean.NN.Attention.CausalMask
import TorchLean.NN.Softmax.Correctness
import TorchLean.NN.Loss.Cosine
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean
open Examples.BugZoo
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)
open FloatLib.Numerics (MathFunctions)

#doc (Manual) "BugZoo Catalog" =>
%%%
tag := "bugzoo-catalog"
%%%

A causal-attention optimization can preserve the output shape while allowing a token to attend to
the future. A shape check can't detect that change: the condition that matters is that every
strict-future attention weight is zero. BugZoo collects failures in frameworks, compilers,
deployment tools, and serving systems, and identifies the object or proposition needed to express
each intended behavior.

Our {srcDir "TorchLeanExamples/BugZoo"}[BugZoo directory] focuses on the TorchLean fragment itself:
once a computation enters a typed TorchLean spec, shape changes, masks, token bounds, finite
precision choices, stateful normalization parameters, and backend semantics become named objects
that can be checked. These failure families have been studied empirically: compiler fuzzing found
silent wrong-code bugs across TVM, TensorRT, ONNXRuntime, and PyTorch
{Informal.citep nnsmith2023}[], coverage-guided fuzzing found rare numerical failures in trained
models {Informal.citep tensorfuzz2019}[], and attention masking has its own long tail of polarity
and layout mistakes going back to the original architecture {Informal.citep transformer2017}[].

Each entry isolates a contract such as “strict-future attention weight is zero.” Its motivating
incident explains where the condition can fail, and its Lean declaration states the inputs and
hypotheses under which it holds.

When the guide builds, the `leanOutput` blocks below are checked. Each computation sits beside its
general contract, so we can inspect both a concrete value and the conditions behind a theorem.

I choose small inputs that let us isolate a cause: repeated attention scores make the mask the only
reason rows differ, a one-feature normalization axis makes its centered activation exactly zero,
and a camera with rational coordinates lets you inspect the projection arithmetic by hand. Inputs
like these remove competing explanations for a surprising result, and the companion general
statements then show which conclusion survives beyond those particular numbers. Read the value, its
scalar type, and the theorem's hypotheses together: a tensor shape can express an interface, a
checked example can expose a discrepancy, and a quantified equality can identify the behavior
another implementation must preserve.

# Catalog Build

{src "TorchLeanExamples/BugZoo/README.md"}[TorchLeanExamples/BugZoo/README.md] lists the entries.
The `TorchLeanExamples` library builds every module under `TorchLeanExamples/`, the catalog entries
included. Build them together with:

```terminal
# Elaborate the maintained catalog with the other examples.
scripts/lake.sh build TorchLeanExamples
```

A successful build means every definition and theorem in the catalog elaborated. It doesn't mean
that every external framework implementation satisfies those contracts.

The `TorchLeanExamples` library takes every module under `TorchLeanExamples/`, so this build covers
every `.lean` file under `TorchLeanExamples/BugZoo/`. The contracts describe TorchLean reference
objects. External framework conformance requires a separate importer, refinement theorem, or
explicit assumption.

This chapter imports each non-attention entry individually. The attention entry also states
$`\exp(-\infty)=0` in `EReal`, but the chapter uses the real-valued hard-mask theorem directly,
through {lean}`Softmax.hardMasked_causalMask_eq_zero`, which needs no
extended reals.

For a more interactive pass, create `BugZooAudit.lean`:

```
-- Compare the exact mask contract with explicit
-- shape-changing operations.
import TorchLeanExamples.BugZoo.AttentionMask
import TorchLeanExamples.BugZoo.Broadcasting

#check TorchLean.Examples.BugZoo.AttentionMask.exp_logit_false
#check TorchLean.Tensor.repeatLeading
#check TorchLean.Examples.BugZoo.Broadcasting.broadcast_first
```

Open the file in the Lean Infoview. Its shape declarations expose the singleton batch insertion and
the proof-carrying broadcast as different operations. These are the contracts. The motivating
PyTorch snippets in the source comments explain the bug family, but aren't imported as evidence.

# Example Anatomy

A good BugZoo example has four parts:

- the pattern in the framework that goes wrong
- the TorchLean object that names the intended behavior
- the theorem, structure, or definition that marks the checked boundary
- the external conformance obligation or unsupported scope that remains outside the checked claim.

Returning a tensor with the expected shape leaves its values, dependency structure, and state
transitions unconstrained. We state those additional conditions in the catalog.

Contracts commonly have this shape:

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
  * lowered graph output equals recorded graph output
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

{src "TorchLeanExamples/BugZoo/Broadcasting.lean"}[TorchLeanExamples.BugZoo.Broadcasting
source] uses a missing batch dimension and a reduction followed by broadcasting. In NumPy-style
tensor libraries, a reduced vector can silently expand across a matrix, but TorchLean's ordinary
elementwise operations require equal shapes, and an explicit broadcast carries its placement with
`Shape.BroadcastDims` evidence (or `Shape.CanBroadcastTo` evidence, for PyTorch's right-aligned
`broadcast_to`).

Summing the rows of a two by three matrix removes the leading dimension:

```lean (name := bzMatDef)
/-- A small matrix for the reduce then broadcast walk. -/
def bzMat : Tensor Float [2, 3] :=
  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

/-- Sum the leading axis to get three column totals. -/
def bzRow : Tensor Float [3] :=
  Tensor.castShape
    (Tensor.reduce .sum bzMat [⟨0, by decide⟩] false)
    (Shape.reduce_singleton _ _)

#eval bzRow
```
```leanOutput bzMatDef (whitespace := lax)
[5.000000, 7.000000, 9.000000]
```

That sum has shape `[3]`. Adding this row back to the matrix by broadcasting would produce
`[[6, 9, 12], [9, 12, 15]]`, and the shape can't tell us whether that broadcast was intended.
TorchLean's `+` requires the two operand shapes to agree, so it rejects an implicit broadcast:

```lean (name := bzBadAdd) +error -keep
/--
Elementwise addition requires an explicit broadcast of the
reduced row.
-/
def bzBadAdd : Tensor Float [2, 3] :=
  bzMat + bzRow
```
```leanOutput bzBadAdd (whitespace := lax)
failed to synthesize instance of type class
  HAdd (Tensor Float [2, 3]) (Tensor Float [3]) ?m.11

Hint: Type class instance resolution failures can be inspected with the
`set_option trace.Meta.synthInstance true` command.
```

If the broadcast was intended, it has to be written down:

```lean (name := bzBroadcast)
-- Expand the reduced row explicitly across both rows of the
-- original matrix.
#eval Tensor.broadcast bzRow [2, 3]
  Broadcasting.row_broadcast.dims
  Broadcasting.row_broadcast.broadcastDims
```
```leanOutput bzBroadcast (whitespace := lax)
[[5.000000, 7.000000, 9.000000], [5.000000, 7.000000, 9.000000]]
```

The values agree with PyTorch's implicit expansion. The difference is that we write the expansion
out, so a reviewer reading a diff sees it appear or disappear.

We can also let Lean infer the broadcast evidence. Both spellings use the same tensor operation,
and the catalog proves they agree:

```lean (name := bzInferred)
-- Check that inferred broadcast evidence selects the same
-- tensor operation.
example :
    Tensor.broadcast bzRow [2, 3]
        Broadcasting.inferred_broadcast.dims
        Broadcasting.inferred_broadcast.broadcastDims =
      Tensor.broadcast bzRow [2, 3]
        Broadcasting.row_broadcast.dims
        Broadcasting.row_broadcast.broadcastDims :=
  Broadcasting.broadcast_infer_eq bzRow
```

For an image, `Tensor.repeatLeading 1` inserts the singleton batch dimension. We reuse that tensor
operation instead of an image-specific wrapper, and the result type records the shape change:

```lean (name := bzBatchSig)
-- Read the inserted leading batch dimension in the result
-- type.
example {α : Type} [Storage α]
    (image : Tensor α [100, 100, 3]) :
    Tensor α [1, 100, 100, 3] :=
  Tensor.repeatLeading 1 image
```

In PyTorch, `m.unsqueeze(0) + m` also succeeds, because the rank-two operand broadcasts against the
rank-three operand. So success doesn't distinguish an intentional singleton batch from an
accidental extra dimension.

Empirical motivation comes from tensor shape fault work such as
[SFData](https://doi.org/10.1145/3533767.3534383), plus [numerical bug
studies](https://doi.org/10.1145/3551349.3559561) that found bad reductions and accidental
broadcasting in real deep learning programs.

All three reduced entries are column totals: the first is `1 + 4`, the second `2 + 5`, and the
third `3 + 6`. Broadcasting repeats that entire vector for each original row. This example is
useful because the intended and the accidental computation can both return six numbers, so the
output shape alone can't explain whether the reduction axis was chosen correctly. Here the type
error locates the missing operation, while the broadcast theorem says that two explicit spellings
of that operation agree. Neither chooses the modeling intention on the author's behalf.

## Stable Loss

Max shifting prevents a large exponential, but it doesn't prevent every rounding failure. For
binary32 logits `[9, -9, -9]`, the two small exponentials survive while adding them to `1` loses
them. A direct `log` of that rounded normalizer then returns zero. Our log-softmax sums the
non-pivot exponentials separately and uses `log1p`. Its backward rule also keeps the small
complement instead of subtracting a rounded probability from one.

Distillation has a different cancellation problem: subtracting independently rounded student and
teacher probabilities can erase a small gradient. `TorchLean.Loss.distill` accepts their logits,
holds the teacher fixed, and computes the student gradient from logit differences. Its shared
specification supports configurable scalar precision, while the native CUDA tape uses binary32 or
binary64. For nonempty class vectors, `Tensor.hasFDerivAt_distillLoss` is the real
derivative theorem. It doesn't assert a rounding-error bound for every format or verify ATen's
execution.

These numerical issues and the logit-difference formulation are discussed in
[A Fix for Softmax and Cross-Entropy](https://arxiv.org/abs/2610.31642).

Long rows can also exhaust the denominator's precision: adding `1` repeatedly in binary16 stops
changing the sum at `2048`, so a left-fold softmax over `4096` equal logits would return twice the
correct probability. A format with a small exponent range can overflow the sum even earlier.
TorchLean's masked and unmasked specifications share a normalizer that detects range loss or a
swallowed nonzero term, then reconstructs the sum and quotients from the finite input coefficients
before rounding the outputs, but it can't recover an exponential that was already rounded to zero.
These summation limitations, discussed by [Blanchard, Higham and
Higham](https://arxiv.org/abs/1909.03469), are separate from backward cancellation.

Activation derivatives need the same care. We keep their inputs available: sigmoid uses a decaying
exponential ratio, and tanh squares a scaled reciprocal cosh instead of subtracting its rounded
output's square from one. GELU's tanh-approximation derivative evaluates its gate as an equivalent
sigmoid, and ELU uses `expm1` near zero. SiLU splits its negative exponential into two factors and
multiplies by the input before the second factor: the product can remain representable even when
the sigmoid alone underflows. GRUs compute complementary update gates as `sigmoid (-x)` instead of
`1 - sigmoid x`, so rounding the update gate to one need not erase the candidate contribution.
These rewrites preserve the real functions and their proved derivatives while retaining small
floating-point results that direct subtraction can lose.

The incoming gradient matters too: for binary32, the derivative of `tanh 60` rounds to zero on its
own, but multiplying the exact derivative by an incoming gradient of `1e30` gives about
$`3.067 \times 10^{-22}`, which is representable. Forming the rounded slope first loses that
answer. Our sigmoid and tanh pullbacks keep the incoming gradient inside their range repair, and
the CPU tape, forward-mode rules and recurrent gates share those pullbacks. Standalone slope
functions use the same repair without inserting a multiplication by one, and native GPU calls use
scaled exponentials for the same purpose. Configured precision moves these boundaries but doesn't
remove them. Near zero, differentiating an exponential quotient can also subtract nearly equal
terms, even when none of them underflows. Nested floating-point derivatives use reciprocal cosh
identities instead (their odd coefficients retain the small input through `sinh`), and they check
the input, hyperbolic factor and slope for range loss. A factor rounded to one also requires a
complete enclosure, and a product of small directions can vanish even at a zero input. When an
intermediate loses range, scalar and CPU pullbacks can enclose the complete expression from the
original finite inputs. A cheap exponential bound first checks whether every coefficient must
underflow. Otherwise, they refine the working precision and approximation degree, accepting only
matching finite rounded coefficient bounds. Each candidate carries a Lean proof of real containment
for every coefficient. Final codec rounding and correspondence with each caller's derivative remain
separate obligations. An inconclusive search retains the staged calculation. This additional path
doesn't verify the GPU implementation or assert correct rounding for every transcendental call.

{src "TorchLeanExamples/BugZoo/StableLoss.lean"}[TorchLeanExamples.BugZoo.StableLoss source] is
about losses that look mathematically harmless but fail numerically. One example is `softmax`
followed by `log`: if a probability rounds to zero, the log path can produce infinities and
downstream NaNs. TorchLean's cross-entropy names its input kind: `crossEntropy .logits` retains the
original logits alongside log-softmax, and `crossEntropy (.probabilities epsilon)` clips
probability inputs.

For probability-based binary cross-entropy, even a nonzero input can lose its loss before the
logarithm. With target zero, the term is $`-\log(1-p)`: a sufficiently small positive $`p` leaves
`1 - p` rounded to one. `BinaryCrossEntropy.term` uses `log1p (-p)` inside the clipping interval to
retain that term. The upper clamp has the same problem in the other term. A certain prediction is
clamped to `1 - epsilon`, whose loss $`-\log(1-\epsilon)` is about epsilon, and rounding
`1 - epsilon` first loses it: in bfloat16 at the default epsilon, and in binary32 at
`epsilon = 1e-8`, the loss comes out zero. Both `BinaryCrossEntropy.term` and
`crossEntropy (.probabilities epsilon)` take that logarithm as `log1p (-epsilon)`, and the binary
loss keeps epsilon as the complementary probability. The clipping thresholds, the real loss and the
selected gradients are unchanged, but epsilon must still be positive and suitable for the chosen
format. `TorchLean.Examples.Clamp` checks the standard and the stable losses
in bfloat16, binary32 and binary64 with the kernel.

The logits below make two of the three probabilities underflow in binary64. Direct log-softmax
retains finite log-probabilities:

```lean (name := bzLogSoftmax)
/-- Logits far enough apart that the softmax underflows. -/
def bzLogits : Tensor Float [3] :=
  [-800.0, -1600.0, 0.0]

#eval Tensor.logSoftmax bzLogits ⟨0, Nat.one_pos⟩
```
```leanOutput bzLogSoftmax (whitespace := lax)
[-800.000000, -1600.000000, 0.000000]
```

Computing the probabilities first loses that information before the logarithm runs:

```lean (name := bzLogOfSoftmax)
-- Taking log after underflow cannot recover the discarded
-- log-probabilities.
#eval Tensor.log (Tensor.softmax bzLogits ⟨0, Nat.one_pos⟩)
```
```leanOutput bzLogOfSoftmax (whitespace := lax)
[-inf, -inf, 0.000000]
```

For finite real logits, the two formulas define the same function. In this computation, however,
the probability path has already rounded two entries to zero. Applying `log` gives $`-\infty`, and
multiplying such an entry by a zero target can then produce NaN.

Even direct log-softmax has a finite-range limit: the difference between two finite logits can
overflow, although a zero or small target would make its weighted loss contribution representable.
For those entries, we keep the original logits and weight each operand before subtracting them.
Ordinary finite log-probabilities keep the usual multiplication.

```lean (name := bzExtremeCE)
/-- Finite logits with an overflowing class difference. -/
def bzExtremeLogits : Tensor Float32 [3] :=
  [3e38, 3e38, -3e38]

#eval (Tensor.crossEntropy .logits bzExtremeLogits
  (Tensor.from (#[1, 0, 0] : Array Float32))
  ⟨0, by decide⟩).toFloat
```
```leanOutput bzExtremeCE (whitespace := lax)
0.693147
```

For these logits, the loss is about $`\log 2`, not NaN. A zero-target mask could produce the right
loss while discarding a valid derivative with respect to that target weight. We retain both
pullbacks:

```lean
open CrossEntropy in
#check logits_eq_dot
open CrossEntropy in
#check hasFDerivAt_logits
open CrossEntropy in
#check hasFDerivAt_target
```

These theorems state the real formula and its logits and target-weight derivatives for class
vectors, without assuming the weights sum to one. They don't assert bitwise equivalence with the
usual floating-point expression or prove ATen's execution. Overflowing reductions and nonfinite
inputs still need separate treatment. Related zero-weight failures have been [reported in
Optax](https://github.com/google-deepmind/optax/issues/896).

This example is motivated by [TensorFuzz](https://proceedings.mlr.press/v97/odena19a.html), which
targeted rare numerical failures, and by
[empirical studies of numerical bugs](https://doi.org/10.1145/3551349.3559561)
involving `log`, `sqrt`, division, `exp`, and reductions.

That underflow example isolates an intermediate representation problem: a log-probability near
`-1600` is a usable finite number even when exponentiating it produces a probability too small for
the format. Once that probability has become zero, a later logarithm has no way to reconstruct the
original value. Keeping the logits until log-softmax avoids that particular loss of information.
The theorem's final reduction also matters: it states how target-weighted log-probabilities become
a scalar, so changing a sum into a different mean would change the loss contract even if every
individual log-probability remained correct.

The backward calculation matters in attention too: TorchLean centers the incoming gradient before
its weighted softmax reduction, so a dominant probability rounded to one need not erase its logit
gradient. Both attention derivatives use that calculation, including Boolean masks. Fully blocked
rows and rows with one allowed key have zero key derivative over the reals. If large finite
incoming gradients overflow during centering, supported CPU scalars replay the complete weighted
calculation from finite coefficients. That replay still uses the supplied weights: it can't
reconstruct a probability already rounded to zero. Our rounded-real error bound includes the
centering subtraction, but native GPU execution remains a separate boundary.

### Squaring before scaling

Cosine loss compares directions. Multiplying a nonzero vector by a positive scalar shouldn't change
its angle, but squaring large coordinates can overflow before we have a chance to normalize them.
TorchLean scales before squaring and normalizes each tensor. We scale the epsilon floor too, and it
still clips the norm instead of adding to its square. When both norms are positive and above their
floors, for any epsilon including zero, the loss is then half the squared distance between the two
unit vectors. That equals one minus their dot product, but for nearly aligned vectors subtracting
the dot product from one cancels, and the loss can round to zero.
`TorchLean.Examples.Cancellation` and `CosineGuard` check both formulas with
the kernel (the second at `epsilon = 0`).

```lean (name := bzCosine)
def bzLargeVector : Tensor Float32 [2] := [3e20, 4e20]

#eval (Tensor.cosineLoss bzLargeVector
  (Tensor.from (#[1, 0] : Array Float32))
  [⟨0, by decide⟩] false Context.defaultEpsilon ()).toFloat
```
```leanOutput bzCosine (whitespace := lax)
0.400000
```

Computing the raw squares first would give a loss of `1` for this input, not `0.4`. We use
normalized coordinates and separate divisions in the pullback too: a product of large norms can
overflow and erase a representable gradient even when the forward loss looks reasonable. The real
identities cover any tensor shape and preserve the clipped branch, including at its epsilon floor:

```lean
open Tensor in
#check cosineLoss_eq
open CosineLoss in
#check fiberLossDeriv_eq
```

Normalizing before the dot product is established practice, and PyTorch explains the same issue in
its [source](https://github.com/pytorch/pytorch/blob/v2.11.0/aten/src/ATen/native/Distance.cpp).
Scaling the norm calculation also keeps large coordinates from being squared directly. These real
identities aren't a bound on all floating-point errors or a proof of external kernels.

Huber's quadratic branch has a smaller ordering trap. An FP32 residual of $`2^{64}` has a
representable half-square, $`2^{127}`, although its square overflows. We compute `(x / 2) * x`, not
`(x * x) / 2`. Huber's linear branch also keeps the prediction and target available: if their
difference overflows, a small positive `delta` can still make the weighted loss representable. We
weight each operand before subtracting, then retain the quadratic offset. The Huber branch
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
  let state := Adam.init 1 beta beta 1 parameters
  let next := Adam.update state parameters gradients
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

There's a different trap at small learning rates. In binary16, a gradient of $`256`, the same decay
coefficients, epsilon $`1`, and learning rate $`2^{-19}` should produce a nonzero update. Dividing
the learning rate by the corrected RMS first rounds that factor to zero. Adam, RMSProp and AdaGrad
normalize the moment or gradient before applying the learning rate. If the supported scalar's
checked operations detect range loss, the CPU update replays that weighted quotient from its finite
input coefficients and rounds it in the selected format. That replay also handles the opposite
case, where the unscaled quotient overflows before a small rate is applied. It can't restore
information lost before those inputs were formed.

These are evaluation-order changes to the [Adam equations](https://arxiv.org/abs/1412.6980), not a
different optimizer. RMSProp and Adadelta share the weighted-square calculation. Our rounded-real
AdamW theorem follows the actual normalized-moment calculation, with a parameter-error budget for
division followed by learning-rate scaling. That model has no upper exponent limit, and its scalar
context doesn't enable finite-coefficient replay. Its theorem doesn't establish the external GPU
adapter's execution. Stored moments, parameter updates and their sums can still overflow. This
example doesn't guarantee that every finite gradient produces a finite optimizer state.

### The same failures in other libraries

`TorchLeanExamples.DeepDives.Floats.LibraryBugs` writes formulas from PyTorch, JAX, Hugging Face
Transformers, TRL, Stable-Baselines3 and diffusers with TorchLean's scalar operations, in nine
sections, and checks a failing input of each with the kernel. PyTorch's soft-margin loss
`log1p (exp (-y x))` isn't finite at the margin `-12` in binary16 and its eager gradient is NaN,
while softplus of the margin is finite. PyTorch's KL divergence of two normal distributions adds
the squared mean difference to the variance ratio before subtracting one, so in binary32 a mean
shift of `2⁻¹³` gives zero, and its one-dimensional multivariate formula is negative at a scale
ratio of `1 + 2⁻¹³`. GELU's tanh approximation `x (1 + tanh z) / 2` (the default of `jax.nn.gelu`)
returns zero at `x = -6` in binary32 and at `x = -4` in bfloat16, where the same function written
`x * sigmoid (2 z)` keeps a negative value. In binary32 PyTorch's kernel (and `Activation.geluTanh`
with it) also returns zero at `x = -6`. The clamp of `torch.special.logit` rounds `1 - eps` to one
in binary32 at `eps = 1e-8`, and the logit of one isn't finite. The sections for these four
formulas prove that the stable formula has the library's real value (for the tanh-squash term
below, its value without the epsilon), and every section but the moving average below state mixed
roundoff bounds, in every IEEE format, for the computation in which they differ.

`TorchLeanExamples.DeepDives.Floats.ProbBugs.SurvivalComplement` checks three more formulas.
PyTorch's CPU `logsumexp` and last-dimension `log_softmax` keep the sum of `exp (x - max)` over a
binary16 row in binary16: for 151936 zero logits (the width of the Qwen2 vocabulary), the sum rounds
to infinity and neither result is finite, while a binary32 sum gives `± 1527 / 2⁷`, about `11.93`.
PyTorch Geometric's variance aggregation `mean (x * x) - mean x * mean x` is `-2` in binary32 on the
neighborhood `[4095, 4097, 4097]`, whose variance is `8 / 9`. Subtracting the mean first gives the
binary32 value nearest `8 / 9`. TensorFlow Probability's Weibull survival function `1 - cdf` is
zero at `t = (x / scale) ^ k = 32` in binary32 and its logarithm isn't finite, while `exp (-t)` and
`-t` keep both.

`TorchLeanExamples.DeepDives.Floats.LossBugs.FocalPower` checks three formulas in gradients and
decay factors. torchvision's `sigmoid_focal_loss` with `γ = 1 / 2` differentiates `(1 - p_t) ^ γ`
through `1 / 2 * (1 / sqrt (1 - p_t))`: at the logit `-20` with target `0` in binary32, `1 - p_t`
rounds to zero and the gradient is NaN, while the same derivative written as
`sqrt p * (p + c / 2 * (1 - p))` decodes to `10357061 / 2⁶⁶`, about `1.4 * 10⁻¹³`. xLSTM's parallel
mLSTM backend takes each forget factor as the exponential of a difference of prefix sums of log
forget gates: in bfloat16, after the prefix sum `-64`, the factor for the log gate `-199 / 2¹²` is
`1` instead of about `0.9526`, and the exponential of the gate itself gives `61 / 64`. PyTorch's
backward of `vector_norm` of order three multiplies `x * |x|` by `g / (n * n)`. In binary16 at four
entries `2⁻¹³`, the product underflows to zero, the quotient overflows and the gradient is NaN,
while `g * (x / n) * |x / n|` gives `407 / 1024`.

The sections of `LibraryBugs` on reinforcement-learning code for language models and control show
the same failures. The k3 KL estimator `exp d - d - 1` of TRL's GRPO loss decodes to zero at the
binary32 log-ratio `2⁻¹²`, where `expm1 d - d` keeps the value `2⁻²⁵`. Stable-Baselines3 and
CleanRL take the log-probability correction of a tanh-squashed action as
`log (1 - tanh² u + 1e-6)`, which is `log 1e-6` (about `-13.8`) once `tanh u` rounds to one, but at
`u = 10` the term is about `-18.6`, and the form `2 (log 2 - u - softplus (-2 u))` of
`torch.distributions.TanhTransform` gets it.

The last three sections of `LibraryBugs` show reductions and state updates failing by range and by
resolution. PyTorch's label-smoothed cross-entropy sums the negated log-probabilities of the whole
batch in the logits' format: in binary16 that sum overflows for a thousand classes and a batch of
thirty-two although the loss is about ten. We check three rows of the log-probabilities `0` and
`-30000` with the kernel: each row sum is representable, the negated batch sum `90000` isn't, and
the term at `eps = 1 / 8` isn't finite, while summing terms already divided by the counts gives the
exact term `1875`. The exponential moving average of `torch.optim.swa_utils.get_ema_multi_avg_fn`,
kept in bfloat16 with decay `0.999`, rounds every update back to the average itself once the step
is below half a unit in the last place, so the average stays at its first value. Keeping the
average in binary32 moves it. diffusers' `add_noise` rounds `alpha_bar` to binary16 when the
latents are binary16 (as its own training script makes them), so at the second timestep of the
default linear schedule `sqrt (1 - alpha_bar)` is zero instead of about `0.0148`.
`sqrt (-expm1 (Σ log1p (-beta)))` computes the same noise level without forming `alpha_bar`.

## Ignored Labels

{src "TorchLeanExamples/BugZoo/IgnoredLabel.lean"}[TorchLeanExamples.BugZoo.IgnoredLabel
source] examines a mean loss when no labels remain active. [PyTorch
issue #75181](https://github.com/pytorch/pytorch/issues/75181) reported an `ignore_index` case
where every label was ignored and the result was `nan`. A mean over no active labels needs a
convention: dividing a zero loss sum by a zero active count doesn't supply one.

TorchLean's cross-entropy losses take no ignore mask, so the example defines each label's
contribution through an active flag:

```lean (name := bzIgnored)
-- An inactive label contributes zero even when its supplied
-- loss is nonzero.
#eval IgnoredLabel.contribution false 2.5
```
```leanOutput bzIgnored (whitespace := lax)
0.000000
```

```lean (name := bzActive)
-- Activating the same label restores its supplied
-- contribution.
#eval IgnoredLabel.contribution true 2.5
```
```leanOutput bzActive (whitespace := lax)
2.500000
```

The active flag selects a branch:

$$`\operatorname{contribution}(active,loss)
=
\begin{cases}
loss, & active\\
0, & \neg active
\end{cases}`

We prove that the false branch is zero at any element type with a zero:

```lean (name := bzIgnoredThm)
-- The ignored-label equation is independent of the chosen
-- scalar loss value.
#check @IgnoredLabel.contribution_false
```
```leanOutput bzIgnoredThm (whitespace := lax)
@IgnoredLabel.contribution_false :
  ∀ {α : Type} [inst : Zero α] (loss : α),
  IgnoredLabel.contribution false loss = 0
```

The reduction must also specify what happens when the active count is zero. Dividing the zero loss
total by that count is undefined. We add epsilon to the denominator and evaluate this policy on an
empty active set:

```lean (name := bzMaskedMean)
-- Evaluate the chosen denominator policy when both the
-- total and count are zero.
#eval IgnoredLabel.safeMaskedMean 0.0 0.0
```
```leanOutput bzMaskedMean (whitespace := lax)
0.000000
```

Returning zero for a fully ignored batch is a modeling choice. A training loop could instead skip
the step, preserve a running average, or report an error. Adding epsilon also changes nonempty
means: with one active label, the real-valued result is scaled by $`1/(1+\varepsilon)`.
`Context.defaultEpsilon` at `Float` is $`10^{-6}`.

The two contribution outputs distinguish exclusion from a small numerical weight. An ignored label
follows a branch that returns zero, while an active label retains its loss. The later reduction
must then count the same set of active labels, otherwise the numerator and denominator describe
different batches. In this compact helper `activeCount` has the scalar type, so its type alone
doesn't enforce a nonnegative integer count. The empty-set computation uses the intended count
zero. A caller handling arbitrary external counts must establish their meaning before relying on
the denominator policy.

## Autograd Domain

{src "TorchLeanExamples/BugZoo/AutogradDomain.lean"}[TorchLeanExamples.BugZoo.AutogradDomain
source] follows [PyTorch's own autograd note about division by
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
#eval AutogradDomain.maskAfterDiv
  (Tensor.full [1] 0.0) (Tensor.full [1] 1.0)
  (Tensor.full [1] 0.0)
```
```leanOutput bzUnsafeDiv (whitespace := lax)
[NaN]
```

The unsafe form already produces `NaN` in the forward direction here, because the mask multiplies
zero by an infinity. A derivative of the division also depends on that zero denominator. Masking
the final value doesn't remove the division from the computation being differentiated.

Our unfolding lemma pins down where the epsilon belongs in the reference expression. A fused kernel
needs separate conformance evidence that it preserves this formula:

```lean (name := bzSafeDivThm)
-- Inspect the exact denominator shift that distinguishes
-- the guarded expression.
open AutogradDomain in
#check @maskAfterSafeDiv_eq
```
```leanOutput bzSafeDivThm (whitespace := lax)
@maskAfterSafeDiv_eq :
  ∀ {s : Shape} {α : Type} [inst : Storage α]
    [inst_1 : Context α]
    (mask numerator denominator : Tensor α s),
  maskAfterSafeDiv mask numerator denominator =
    mask * Tensor.zipWith
        (fun a b => a / (b + Context.defaultEpsilon)) numerator
        denominator
```

Adding epsilon is a denominator shift, not a clamp away from zero: the shifted denominator is still
zero when the original denominator is $`-\varepsilon`. This example checks the displayed
zero-denominator case and unfolds the formula, but it doesn't prove finite values or gradients for
arbitrary inputs.

Keeping the unsafe expression beside the guarded one exposes the relevant graph change: the
zero denominator is used before masking in the first expression and shifted before division in
the second.

Masking is particularly subtle in differentiation because the graph records how a value was
computed. A zero at the final output doesn't remove an earlier division node or its local
derivative. Our reference examples deliberately keep the unsafe operation visible instead of
checking only whether the selected forward entries look finite. For a gradient comparison, the
relevant observations are the numerator and denominator cotangents as well as the loss. A
derivative argument would have to follow the guarded forward formula that the local unfolding
theorem specifies.

## Attention Mask

{src "TorchLeanExamples/BugZoo/AttentionMask.lean"}[TorchLeanExamples.BugZoo.AttentionMask source]
is the catalog entry for causal mask semantics. Attention masks fail by polarity, layout, fake
negative infinity, fully masked rows, and interactions with API flags. PyTorch has had relevant
reports for `MultiheadAttention`, including [`is_causal=True` being ignored when
`need_weights=True`](https://github.com/pytorch/pytorch/issues/99282) and [fully masked heads
producing NaNs](https://github.com/pytorch/pytorch/issues/160064) when weights are requested.

TorchLean's causal mask for three tokens is:

```lean (name := bzCausalMask)
-- Rows are queries, and true entries mark keys visible to
-- that query.
#eval Attention.causalMask 3
```
```leanOutput bzCausalMask (whitespace := lax)
[[true, false, false], [true, true, false], [true, true, true]]
```

A `true` entry makes a key visible to a query. The first query can see only its own key, while each
later query can also see the preceding keys. Identical score rows isolate the mask's effect on the
weights:

```lean (name := bzMaskedSoftmax)
/-- Three query rows with identical scores. -/
def bzScores : Tensor Float [3, 3] :=
  [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]

#eval Softmax.hardMasked bzScores
  (Attention.causalMask 3)
```
```leanOutput bzMaskedSoftmax (whitespace := lax)
[[1.000000, 0.000000, 0.000000],
 [0.268941, 0.731059, 0.000000],
 [0.090031, 0.244728, 0.665241]]
```

The displayed weights are zero at every strict-future position. Reversing the mask would fully
block the last row. A direct softmax over only negative infinities encounters a zero normalizer,
and max-subtraction also encounters the undefined difference $`-\infty-(-\infty)`. TorchLean's
hard-mask contract assigns blocked positions a zero weight, including fully blocked rows. Our Lean
composition over CUDA primitives preserves that choice in both forward evaluation and its local
VJP.

The checked claim quantifies over every strict-future position, at every sequence length:

```lean (name := bzCausalThm)
-- The theorem quantifies over all sequence lengths and
-- strict-future coordinates.
open Softmax in
#check @hardMasked_causalMask_eq_zero
```
```leanOutput bzCausalThm (whitespace := lax)
@hardMasked_causalMask_eq_zero :
  ∀ {n : ℕ} (scores : Tensor ℝ [n, n]) (i j : Fin n),
  ↑i < ↑j →
  (hardMasked scores (Attention.causalMask n)).get2 i j = 0
```

This theorem rules out direct attention to a strict-future value. Causality of a complete model
also requires its other operations to preserve that restriction. We connect the hard-mask formula
to a negative-infinity logit by first stating the exact exponential identity in `EReal`, since `ℝ`
contains no negative infinity:

```
-- Exact negative infinity gives a zero numerator in the
-- extended-real presentation.
theorem exp_logit_false (score : ℝ) :
    EReal.exp (logit score false) = 0 := by
  simp [logit]
```

TorchLean's ordinary attention spec uses a hard-masked numerator: a blocked entry contributes zero
before normalization. So the theorem needs no extended reals. If every entry in a row is blocked,
the spec defines the entire weight row to be zero.

Each causal row provides a different check. Row zero has one allowed key, so its normalized weight
must be one. Row one compares two allowed scores while forcing the third entry to zero. The last
row allows all three scores and recovers ordinary softmax. These cases separate mask polarity from
score scaling and normalization. A fully blocked row adds the denominator boundary case. Spelling
that case out is necessary when comparing APIs that share a Boolean mask type but make different
choices for empty attention support.

Finite arithmetic adds another difficulty. A dot product can overflow before division by
$`\sqrt{d}`, even when the scaled score fits. Pre-scaling the queries avoids some overflows, but
can instead round a tiny query to zero before it meets a large key. Our reference attention
calculation keeps its ordinary order and, when checked operations lose range or absorb a nonzero
term, reconstructs the complete scaled contraction from its original finite coefficients. The
registered `Op.scaledMatmul` (which a model calls through `Ops.apply`) takes a fixed scale, so
biased and dropout attention can keep that scale inside the score contraction too. Its IR node and
backend audit distinguish this calculation from an ordinary fixed-left matrix product followed by
multiplication.

On the GPU, the native contraction uses a conservative range flag, and a flagged call can download
the original finite operands and run the same coefficient recovery on CPU before returning a GPU
buffer. Our buffer interface keeps its binary64 scale even for binary32 operands: narrowing
$`10^{-46}` first would turn it into zero before large inputs could amplify it. Reading the flag
synchronizes the device, and exact-zero and orthogonal products can also take this recovery path.
Original nonfinite inputs retain the native result without this transfer. This recovery isn't a
proof of ATen or of correctly rounded softmax: later native exponential, division and aggregation
stages still round. The grouped IR operation has a separate numerical range rule. Its scale is an
ordinary scalar parent, so source-range and payload checks bind it just like a weight. That rule
encloses the exact-real contraction and includes the finite staged interval when available. Its
reference policy records the choice between fixed-left arithmetic and exact coefficient replay, and
it doesn't accept an implementation-defined native reduction. Concrete binary32 execution must
still pass the node-by-node interval replay.

Even constructing the denominator needs care: FP16 cannot represent $`65536`, but its square
root is $`256`. Casting first would give infinity and erase the score scale. `Context.sqrtNat`
keeps the ordinary calculation when the count cast does not overflow, and otherwise asks
FloatLib to round the exact rational root directly in the chosen format. Dual and complex
contexts lift this same constant construction from their component scalar type.

Backward mode has the same boundary: the product $`dO\,V^\top` can overflow before the softmax
pullback cancels it. A fully blocked row should have zero gradient, and a singleton softmax should
have zero score tangent. Our fused reference pullback retains the original contractions instead of
trying to recover them from an already infinite intermediate tensor. That repair reuses the scalar
coefficient interface, including configured binary precisions and nested duals. Biased and dropout
attention apply `Op.attentionFromScores` with a multiplier to keep that boundary intact on the
ordinary tape. Its multiplier can carry gradients too, while a sampled dropout mask follows the
usual stop-gradient convention. Both the JVP and VJP retain the original contraction terms, so
recovery can also handle cancellation between large tangent terms or a tiny intermediate amplified
later. The real-valued proofs in
{src "TorchLean/NN/Attention/HardMask/FDeriv.lean"}[`HardMaskedAttention`]
differentiate this forward function and show that its JVP and backward rule are adjoint for a fixed
optional mask, including fully blocked rows.
{src "TorchLean/NN/Attention/Adjoint.lean"}[`Attention`] proves the same pairing for
`Tensor.attentionJVP` and `Tensor.attentionBackward` in the query, key, value and multiplier
tangents together, over any leading batch axes. On the GPU, the native backward checks the
contraction and centered pullback, and when recovery is needed it can bring the original finite
coefficients to CPU and return the recovered gradient to GPU. This includes the native attention
adapter, but it doesn't prove the ATen kernels or a numerical error bound for every native
execution. Fully blocked rows discard their incoming cotangent, and the reference rule checks all
scalar metadata before treating a weight row as zero.

## Compiler Boundary

This entry is about optimized graphs that run, return tensors, and are nevertheless not the same
computation as the source program. [NNSmith](https://arxiv.org/abs/2207.13066) found compiler bugs
across TVM, TensorRT, ONNXRuntime, and PyTorch, [FreeFuzz](https://arxiv.org/abs/2201.06589) found
framework and API bugs by mining real snippets, and a recent [PyTorch compiler correctness
study](https://arxiv.org/abs/2604.08720) focuses on silent `torch.compile` wrong outputs.

The translation a verifier depends on is the lowering of a TorchLean forward program to the IR
graph it bounds. `IR.lowerForward` records the program as a typed graph and builds its operator
graph. On every input, the lowered graph returns the recorded graph's value at its output
({src "TorchLean/NN/IR/Lowering.lean"}[`IR.lowerForward_eq_ok`]). That comparison includes the
parameters: lowering stores them in the graph's payload, and a different constant or parameter
tensor would describe a different computation. No theorem relates the recorded graph to the source
program, so a wrong recording is outside the theorem. The compared functions are Lean denotations.
Generated machine code and external kernels require separate conformance evidence.

## Float Boundary

This entry separates real arithmetic from the binary32 model. Finite precision, exceptional values,
fused operations, subnormal behavior, and changed reduction order can invalidate a property proved
only over reals {Informal.citep jiarinard2020}[]. Goldberg describes the underlying rounding
effects {Informal.citep goldberg1991}[], while Flocq {Informal.citep flocq2011}[] and the Coq
treatment of numerical programs {Informal.citep boldo2015}[] develop mechanized ways to reason
about them.

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

The identical bit patterns identify the rounded result even when a decimal printer hides low-order
bits. This one addition is a checked example, while the following proofs state the general relation
for addition and subtraction, including the finite-input hypotheses:

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
Configured division refines its software model by `ExecFloat.Binary.toModel_div`, which does not
assert native `Float32` division agreement. A compiled CPU instruction or CUDA kernel is a separate
execution boundary, and none of these theorems establishes conformance of a particular vectorized
kernel.

The pair of integers in the addition output is useful because the comparison is about stored bits,
not a tolerance chosen after printing. For arbitrary finite operands, the theorem identifies those
same configured values. Its input conditions are part of that conclusion, not an optional runtime
check. Native export has a separate NaN policy: payload and sign distinctions can be lost when
Lean's canonical NaN is constructed. A property about payload preservation would require a more
precise relation. Likewise, a theorem about one addition doesn't fix the order in which a network's
long reduction performs many additions: that order belongs to the surrounding computation.

Conversion into a scalar type matters too. A uniform draw of $`2^{30}/2^{32}` is just $`1/4`, but
casting its numerator and denominator separately to FP16 gives infinity divided by infinity. That
NaN can turn a valid dropout keep decision into a dropped entry. `Random.sampleUnit` forms the
exact rational fraction first and uses the scalar context's cast. It keeps the same integer
generator, seed and counter, while the chosen precision rounds the resulting fraction, which can
still round up to $`1` at the top of the grid.

Parameter initialization has a related range boundary. Two finite uniform bounds can have an
infinite difference: the interval from minus the largest binary64 value to that value is one
example. `TorchLean.Init.sampleAt` keeps the ordinary affine formula when the width is finite in
binary64, but if the width overflows, it instead forms the convex combination of the two finite
endpoints, avoiding that intermediate infinity. CUDA parameters hold these binary64 samples,
rounded once to the device dtype.

Xavier and He bounds can lose range in the fan count, not in the root. Casting an enormous count to
infinity makes $`\sqrt{6/\mathrm{fan}}` zero, even when the bound is representable. Their shared
initializer preserves the ordinary formula for nonzero bounds, and otherwise it asks FloatLib to
round the square root of the exact rational ratio, a fallback that also handles bounds in the
binary64 subnormal range without first rounding the ratio to zero.

At a minimum or maximum tie, TorchLean selects an equal split between the inputs, so its JVP
averages their supplied tangents. Halving two smallest subnormal values separately would give zero,
although their average is still representable. Dual numbers and typed graphs use the same checked
average: if halving loses range, they combine the finite coefficients before rounding the result.
We cast the half constant from an exact rational, so the rule also works in configured formats that
can't represent `2`. Reverse mode still rounds distinct input nodes' cotangents separately, and a
half that is itself unrepresentable can't be stored in that scalar type. When both arguments refer
to the same tape node, the checked CPU path and native GPU path combine its contributions before
graph accumulation. This retains a smallest-subnormal gradient for $`\max(x,x)` or $`\min(x,x)`. It
also spells out the accumulation convention: we add the combined contribution once instead of
adding two separately rounded halves. Unsupported coefficient carriers accumulate the two
contributions separately.

Square-root gradients need the upstream value too. Forming $`1/(2\sqrt{x})` first can lose range
before multiplication by that value, and some configured formats can't represent `2` at all. Dual
numbers, typed graphs and the CPU tape share a weighted quotient that replays finite coefficients
when a checked intermediate loses range. That quotient uses the root already rounded by the forward
arithmetic, and it doesn't guarantee correctly rounded evaluation of the unrounded analytic
derivative. The clamped rule and the CUDA bridge divide by twice the root in one rounding. Outside
the positive region, both multiply a zero factor by the upstream value, which gives `-0` for a
negative value and a NaN for an infinite one.

The angle in a complex logarithm has a related cancellation. Moving radially should leave the angle
unchanged. Scaling tiny coordinates can nevertheless make both tangent terms infinite, turning their
zero difference into NaN. TorchLean's dual angle rule checks the scaled calculation and,
after range loss or cancellation, reconstructs the whole finite quotient from the original
coordinates and direction, leaving the primal angle and the origin's selected zero tangent
unchanged. A principal angle still has a branch cut, and this rule doesn't make it differentiable
across that cut. For the {src "TorchLean/NN/Dual/Domain.lean"}[real-valued
derivative proof], we use Mathlib's derivative of the complex logarithm away from the cut.

Powers have a domain issue as well: applying the logarithmic derivative formula to $`x^2` at zero
or a negative input evaluates a logarithm or quotient that the polynomial doesn't need. When the
exponent tangent is identically zero, the dual rule uses $`yx^{y-1}x'`. Its zero tests inspect
every nested coefficient: an active mixed derivative must not disappear just because its primal is
zero. That test also guards the captured-exponent VJP. The
{src "TorchLean/NN/Dual/Domain.lean"}[constant-power derivative theorem] checks the
actual dual rule over the reals. It isn't a theorem about floating-point accuracy: a staged power
or slope can still lose range before multiplication by the incoming derivative. For example, with
$`x=2^{100}` and incoming derivative $`2^{-100}`, the cubic's weighted derivative is the finite
binary32 number $`3\cdot 2^{100}`, although the local factor $`3x^2` overflows. Both the dual rule
and the captured-exponent VJP check these intermediates. For a constant exponent, they can refine
the whole weighted expression from its original inputs using FloatLib's logarithm and exponential
enclosures, but with a varying exponent and a positive base, recovery instead encloses the complete
expression $`\exp(y\log x)(y'\log x+yx'/x)`. Keeping both terms together matters: an overflowing
quotient can be multiplied by an underflowed power even when the final derivative fits. Recovery
requires every lower and upper derivative coefficient to round alike, including mixed coefficients
of nested duals. Refinement has a bounded work budget based on the input encodings, and it declines
oversized exponential arguments before evaluating them. Nonfinite inputs, unsupported domains and
inconclusive bounds retain the ordinary result. On positive bases, the real-valued derivative
theorem checks the varying-exponent rule, but neither it nor the recovery routine verifies native
power evaluation.

For a mean, we must choose the accumulation precision as well. A left-to-right sum of 65,536 FP16
ones loses increments, and the FP16 conversion of the element count overflows. The same-precision
mean returns zero although the exact mean is one. TorchLean keeps this scalar policy explicit:
{src "TorchLean/Tensor/Rep.lean"}[`Tensor.foldl`] lets us convert each entry into a wider
accumulator without allocating a second tensor. Converting the tensor before calling `mean` is
another option. Both give one for this example with a binary32 accumulator, but neither promises
exact arithmetic or prevents overflow for every input. `mean` itself keeps its reference definition
and doesn't silently promote the scalar type.

## Normalization State

Normalization also needs care before the square root. In binary32, squaring `[3e20, 4e20]`
overflows even though its normalized entries are ordinary numbers. RMSNorm and regularized L2
normalization scale the input before squaring and divide epsilon by the same scale twice. Scaling
only the input would change the model's epsilon. We detach the scale because it cancels from the
real function, and the tape records the remaining operations as usual.
`Tensor.normalizeL2_eq` checks that real identity, including the mean-square variant, for
positive epsilon, but it isn't a floating-point error bound.

Safe norm scaling is established numerical analysis, not a new failure mechanism. See [Algorithm
978](https://doi.org/10.1145/3061665) and [PyTorch's
notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html). The chosen format still
limits the reduction and gradients, and nonfinite inputs can propagate NaN differently from the
unscaled formula.

{src "TorchLeanExamples/BugZoo/BatchNorm.lean"}[TorchLeanExamples.BugZoo.BatchNorm
source] covers BatchNorm style bugs where the formula or the state is wrong but the layer still
emits a tensor {Informal.citep batchnorm2015}[].
[CRADLE](https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf) reported a
BatchNorm epsilon placement issue across backends, while
[LEMON](https://lingming.cs.illinois.edu/courses/cs598ast-f20/paper-dnn-lib-testing.pdf) found
moving-statistics bugs and BatchNorm layers that produce NaNs.

Putting epsilon inside or outside the square root changes the normalization formula. The following
binary64 computations use a fixed mean of zero and variance of zero, with incoming value 2. Such
fixed statistics need not equal the mean and variance of the incoming batch:

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

The spec-side contract states the placement as an equation over the reals (where there's no
rounding to argue about):

```lean (name := bzEpsThm)
-- On constant inputs batch normalization is the scalar
-- formula with epsilon inside the guarded square root.
open Examples.BugZoo.BatchNorm in
#check @normalize_scalar
```
```leanOutput bzEpsThm (whitespace := lax)
@normalize_scalar :
  ∀ {s : Shape} (dim : Fin s.rank) (x mean variance gamma beta epsilon : ℝ),
  (Tensor.full s x).batchNorm dim (Tensor.full [s.extent dim] gamma)
      (Tensor.full [s.extent dim] beta)
      (BatchNorm.Stats.running (Tensor.full [s.extent dim] mean)
        (Tensor.full [s.extent dim] variance))
      epsilon =
    Tensor.full s
      ((x - mean) / MathFunctions.sqrt (max (max variance 0 + epsilon) 0) * gamma + beta)
```

The formula clamps the running variance at zero and clamps its sum with epsilon at zero before
taking the square root. These guards are part of the reference expression. They do not say that an
unguarded native square root returns zero on negative inputs.

BatchNorm inference also depends on its running statistics, and passing them as arguments lets the
reference theorem fix that state and show that inference is affine in the input:

```lean (name := bzAffine)
-- The scale and bias are chosen before the input is
-- quantified.
open RunningStats in
#check @batchNorm_affine
```
```leanOutput bzAffine (whitespace := lax)
@batchNorm_affine :
  ∀ {channels : ℕ} {sSpatial : Shape}
    (stats : RunningStats channels)
    (gamma beta : Tensor ℝ [channels])
    (epsilon : optParam ℝ normalizationEpsilon),
  ∃ scale bias,
    ∀ (x : Tensor ℝ (Shape.dim channels sSpatial)),
      x.batchNorm ⟨0, ⋯⟩ gamma beta
          (BatchNorm.Stats.running stats.mean stats.variance) epsilon =
        x * scale + bias
```

The existential witnesses precede the input quantifier: the same scale and bias work for every
input. That order is what makes this an affine-map statement instead of an equality whose witnesses
could change with each input. Once the running statistics are fixed, downstream interval bounds and
Lipschitz arguments can use that common affine representation.

Fixed inference statistics explain why the large normalization value here is possible: with an
incoming value of two and a supplied running mean and variance of zero, the numerator is nonzero
and the small denominator amplifies it. In a one-feature LayerNorm computation, by contrast, the
mean is computed from that same feature and the centered numerator vanishes. So the two examples
should have different outputs even with identical epsilon values. The affine BatchNorm theorem
concerns fixed statistics, and changing those statistics during training changes the affine map and
requires a statement about the state update as well.

## LayerNorm Degenerate Axis

{src "TorchLeanExamples/BugZoo/LayerNormAxis.lean"}[
TorchLeanExamples.BugZoo.LayerNormAxis source] records a related but different
normalization bug family. LayerNorm can receive an axis with one element. Its tensor output still
has the expected shape, but the variance term is zero and the epsilon convention decides whether
the result is finite and meaningful.

Over real arithmetic, one feature has centred value $`x-x=0`, so the output is the bias and the
input and scale gradients vanish when epsilon is positive, but the bias gradient doesn't vanish in
general: it's the upstream cotangent. The catalog evaluates that at a deliberately awkward input
($`x=10^6`), with weight 2, bias 3, and epsilon $`10^{-5}`:

```lean (name := bzLnFwd)
-- A one-feature normalized value should leave only the
-- affine bias.
#eval LayerNormAxis.forward
```
```leanOutput bzLnFwd (whitespace := lax)
3.000000
```

```lean (name := bzLnDw)
-- The scale gradient multiplies the zero centered
-- activation.
#eval LayerNormAxis.backward.scaleGradient[0]
```
```leanOutput bzLnDw (whitespace := lax)
0.000000
```

```lean (name := bzLnDx)
-- The input gradient cancels when the normalization axis
-- has one feature.
#eval
  let dx := LayerNormAxis.backward.inputGradient
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
open LayerNormAxis in
#check @scaleGrad_eq_zero
```
```leanOutput bzLnThm (whitespace := lax)
@scaleGrad_eq_zero :
  ∀ {seqLen : ℕ} (hseq : 0 < seqLen)
    (x : Tensor ℝ [seqLen, 1]) (gamma : Tensor ℝ [1])
    (dy : Tensor ℝ [seqLen, 1]) (epsilon : ℝ),
  (LayerNorm.backward hseq Nat.one_pos x gamma dy
        epsilon).scaleGradient =
    0
```

We keep the forward value, scale gradient, and input gradient in three separate Lean outputs so
that agreement in one can't hide disagreement in another. Bias three survives normalization,
whereas the scale multiplies a zero normalized activation. These outputs don't measure
native-framework residuals or training accuracy.

## Constant Normalization Slice

{src "TorchLeanExamples/BugZoo/ConstantSlice.lean"}[
TorchLeanExamples.BugZoo.ConstantSlice source] keeps the constant-slice case visible.
A constant row or channel slice isn't exotic: it appears in padding-heavy batches, masked tokens,
uniform images, and clipped signals. If the implementation assumes positive variance, a constant
slice can produce a division by zero, a `NaN`, or a backend-specific branch.

The constant-slice contract passes the mean `x` and the variance zero as running statistics. Its
forward and scale-gradient statements cover the same degeneracy as the one-feature case:

```lean (name := bzConstSlice)
-- This theorem assumes the constant-slice mean and variance
-- have their stated values.
open ConstantSlice in
#check @scaleGrad_eq_zero
```
```leanOutput bzConstSlice (whitespace := lax)
@scaleGrad_eq_zero :
  ∀ {s : Shape} (dim : Fin s.rank) (x epsilon : ℝ)
    (gamma : Tensor ℝ [s.extent dim]) (dy : Tensor ℝ s),
  (BatchNorm.runningBackward (Tensor.full s x) dim gamma
        (Tensor.full [s.extent dim] x) (Tensor.full [s.extent dim] 0)
        epsilon dy).scaleGradient =
    0
```

The one-feature LayerNorm theorem above is a statement about `LayerNorm.backward`, and
`ConstantSlice.scaleGrad_eq_zero` is a statement about `BatchNorm.runningBackward`. The companion
`normalize_const` states that batch normalization of a constant input,
with that constant as running mean and zero as running variance, outputs the bias.

The constant-slice theorem starts after the statistics have been supplied. Every entry of its
input and running mean is `x` and every entry of its running variance is zero, but it doesn't
itself prove that a reduction kernel computes those statistics exactly. That distinction matters
for longer constant slices, where summation and division introduce their own rounding. It also
explains why several normalization layers share this contract despite using different axes. To
apply it to a complete layer, first identify the slice and connect that layer's statistics to the
values in the statement.

## Batch Invariance

This entry is about serving systems that change outputs depending on which other requests share a
batch. Such changes can come from dynamic batching, kernel selection, reduction order, and
scheduling details even when user randomness is off. We point to recent LLM serving work: [Thinking
Machines on inference
nondeterminism](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) and an
[LLM inference engine bug study](https://arxiv.org/abs/2506.09713).

Our reference semantics is `Tensor.mapLeading [batch]`: apply the same function to each row
independently. Take the second row out of a batched squaring:

```lean (name := bzBatchMap)
-- Square each batch row independently, then select the
-- second output row.
#eval (Tensor.mapLeading [2]
  (fun r => r * r) bzMat).unstack 1
```
```leanOutput bzBatchMap (whitespace := lax)
[16.000000, 25.000000, 36.000000]
```

and then compute that row on its own:

```lean (name := bzBatchOne)
-- Compare with squaring that same row outside the batch
-- map.
#eval (bzMat.unstack 1 * bzMat.unstack 1)
```
```leanOutput bzBatchOne (whitespace := lax)
[16.000000, 25.000000, 36.000000]
```

From the definition of `mapLeading` we get the general equality:

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
    (xs : Tensor α (Shape.dim batch inShape)) (i : Fin batch),
  (mapLeading [batch] f xs).unstack i = f (xs.unstack i)
```

Batch independence is a property of the chosen reference computation. A function passed to
`mapLeading` receives one row, so it can't inspect neighboring requests through that input, while
an operation that intentionally computes statistics across the batch has a different interface and
shouldn't satisfy this rowwise equation.

A native implementation raises a further question: whether evaluating a row alone produces the same
bits as evaluating it inside a larger batch. Different matrix shapes can select different reduction
orders. To test the native path, compare the same row and weights at both batch sizes with dropout
disabled. Applying the reference theorem to that execution requires a conformance result or a
numerical error bound. A small observed discrepancy supplies neither by itself.

## KV Cache

{src "TorchLeanExamples/BugZoo/KVCache.lean"}[TorchLeanExamples.BugZoo.KVCache source] models cache
accounting in autoregressive inference. LLM engines fail through shifted caches, wrong cache slots,
config and shape mismatches, resource scheduling, and interactions with positions or tokenizers.
Again, the broader source trail is the [LLM inference engine bug
study](https://arxiv.org/abs/2506.09713).

Append a key and a value to a one-token cache and look at the keys:

```lean (name := bzKV)
/-- A one token cache holding a zero key and value. -/
def bzCache : KVCache Float 1 [2] :=
  { keys := Tensor.full [1, 2] 0.0
    values := Tensor.full [1, 2] 0.0 }

#eval (KVCache.append bzCache [1.0, 2.0] [3.0, 4.0]).keys
```
```leanOutput bzKV (whitespace := lax)
[[0.000000, 0.000000], [1.000000, 2.000000]]
```

The output places the new key in the last slot and retains the old row first. The result type
records the new length (two), but it doesn't determine the row contents. The final-slot theorem
establishes where the appended value appears:

```lean (name := bzKVThm)
-- Check which cache slot contains the key after appending
-- one token.
open KVCache in
#check @append_last_key
```
```leanOutput bzKVThm (whitespace := lax)
@append_last_key :
  ∀ {α : Type} [inst : Storage α] {seqLen : ℕ} {tokenShape : Shape}
    (cache : KVCache α seqLen tokenShape)
    (newKey newValue : Tensor α tokenShape),
  (cache.append newKey newValue).keys[seqLen] = newKey
```

This theorem covers the newly written slot. `appendToken_past` covers the earlier slots: below
`seqLen`, the appended sequence reads as the cached one, and `append` appends both keys and values
with `appendToken`. The final-slot theorem uses the right-part indexing lemma of the concatenation,
`Tensor.unstack_concat_right`, and `appendToken_past` uses the left-part one,
`Tensor.unstack_concat_left`.

This example has two axes with separate roles: sequence length counts stored token positions and
head dimension counts coordinates of each key or value. The general cache accepts any token shape,
and appending increases the sequence length while preserving that shape. Checking only the final
key would still miss a value written to the wrong slot, so the source includes the corresponding
final-value theorem too. Full cached decoding needs more than these append facts: the query must
read the matching key/value prefix using the same mask and positions as full-sequence attention.
These local contracts identify pieces of that larger comparison, but the comparison itself isn't
proved here.

## RoPE Position

{src "TorchLeanExamples/BugZoo/RoPEPosition.lean"}[TorchLeanExamples.BugZoo.RoPEPosition source]
pairs naturally with the KV cache example. Rotary position embeddings make position accounting part
of the model's meaning. A decode position off by one can be hard to notice because the shapes still
line up and the model still produces tokens.

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
position as an argument:

```lean (name := bzRopeOffset)
#eval RoPEPosition.append bzSchedule (position := 12)
```
```leanOutput bzRopeOffset (whitespace := lax)
[0, 1, 12]
```

A position schedule separates a token's storage slot from the position used by its embedding. They
coincide in the displayed `[0, 1, 2]` example, which makes the append convention easy to read, but
they need not coincide after a window has been shifted or a prefix has been assigned an offset. The
theorem covers whichever position we supply. The default is `seqLen`, regardless of the last stored
position. No rotation is evaluated by this example: it checks the bookkeeping that supplies
positions to a rotation implementation.

## Tokenizer Boundary

{src "TorchLeanExamples/BugZoo/Tokenizer.lean"}[TorchLeanExamples.BugZoo.Tokenizer
source] marks the boundary before tensors even reach the model. Tokenizer and config mismatches can
disagree about vocabulary size, padding, end-of-sequence, or special token ids while the network
code itself looks ordinary. The [LLM inference engine bug study](https://arxiv.org/abs/2506.09713)
lists this as a real serving class.

A contract is a vocabulary size plus the special ids, each carrying its own bound:

```lean (name := bzTok)
/-- A tiny tokenizer configuration with eight ids. -/
def bzTokenizer : Tokenizer :=
  { vocabularySize := 8
    paddingTokenId := ⟨0, by decide⟩
    endOfSequenceTokenId := ⟨7, by decide⟩ }

#eval (bzTokenizer.paddingTokenId.val,
  bzTokenizer.endOfSequenceTokenId.val)
```
```leanOutput bzTok (whitespace := lax)
(0, 7)
```

An end-of-sequence id of `⟨8, by decide⟩` would require a proof of $`8<8`, so it fails to
elaborate. We extract the bound already carried by the id with the following theorem:

```lean (name := bzTokThm)
-- Recover the vocabulary bound carried by every token in
-- the sequence.
open Tokenizer in
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

Both printed special ids are valid indices into an eight-entry vocabulary, but the type doesn't
require them to be distinct. Nor does it describe the embedding vector associated with either id.
Those are separate configuration agreements between tokenizer and model. This is why the bounds
theorem is still useful despite extracting information already carried by `Fin`: an external
integer must pass that boundary before we can use it to index the typed sequence. Afterward,
ordinary tensor code can rely on the bound while a separate importer contract handles token
meanings and special-token conventions.

## Geometry 3D Projection

{src "TorchLeanExamples/Verification/Geometry3D/Box3D.lean"}[TorchLean.Examples.Geometry3D
source] checks a camera projection. Vision and robotics pipelines use a camera matrix to project 3D
points to image coordinates:

$$`(u,v)=\left(\frac{x}{z},\frac{y}{z}\right)`

up to intrinsics and coordinate conventions. Projection can fail when depth is zero or has the
wrong sign, coordinate frames are swapped, or the denominator convention is implicit. A tensor can
still be emitted, but the geometry claim doesn't match the camera model.

This entry uses a decidable certificate check. The example camera has focal length 100 and
principal point at the image centre:

```lean (name := bzCam)
open Examples.Geometry3D in
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

At the near face, depth is 4, so the corners project to $`100\cdot(\pm 1)/4+50`, giving 25 and 75.
The far face has depth 6 and projects to $`50\pm 100/6`, inside that range. So the enclosing image
box is $`[25,25,75,75]`. Using rationals keeps these calculations exact:

```lean (name := bzCert)
open Examples.Geometry3D in
/-- A detector box that does enclose the projection. -/
def bzCert : BoxCameraCert ℚ :=
  { width := 100, height := 100, tol := 0
    camera := bzCamera, corners := bzCorners
    bbox := [25, 25, 75, 75] }

#eval Examples.Geometry3D.checkCert bzCert
```
```leanOutput bzCert (whitespace := lax)
true
```

Shrinking the claimed box by five pixels on each side excludes the projected near-face corners,
so the same checker rejects it:

```lean (name := bzCertTight)
open Examples.Geometry3D in
/-- The same scene, with a box five pixels too tight. -/
def bzCertTight : BoxCameraCert ℚ :=
  { bzCert with bbox := [30, 30, 70, 70] }

#eval Examples.Geometry3D.checkCert bzCertTight
```
```leanOutput bzCertTight (whitespace := lax)
false
```

Changing only the claimed box tests the enclosure condition while keeping the camera and corners
fixed. The pair doesn't independently test every conjunct: `checkCert` also checks positive image
size, box ordering, that the box lies in the image, positive depths, and that the projection lies
in the image.

The soundness theorem connects acceptance to the geometric predicate:

```lean (name := bzGeoThm)
-- Connect Boolean certificate acceptance to all fields of
-- the geometric predicate.
open Examples.Geometry3D in
#check @checkCert_sound
```
```leanOutput bzGeoThm (whitespace := lax)
@checkCert_sound :
  ∀ {α : Type} [inst : Storage α] [inst_1 : OfNat α 0]
    [inst_2 : OfNat α 1] [inst_3 : Add α] [inst_4 : Sub α]
    [inst_5 : Mul α] [inst_6 : Div α] [inst_7 : LE α]
    [inst_8 : LT α]
    [inst_9 : DecidableRel fun x1 x2 => x1 ≤ x2]
    [inst_10 : DecidableRel fun x1 x2 => x1 < x2]
    {cert : BoxCameraCert α},
  checkCert cert = true → Verified cert
```

The typeclasses provide the storage of tensor entries, arithmetic operations and decidable
comparisons. At `ℚ`, those operations compute the projection exactly, so acceptance establishes the
rational geometric claim. At `Float`, the same theorem establishes a predicate about rounded float
computations, but it doesn't bound their discrepancy from exact real projection. Its companion
`ofHomogeneous_inside_bbox` instead assumes nonnegative
enclosing intervals for the homogeneous numerators and a depth interval bounded away from zero.
Applying that result to a float executor requires showing that its intervals enclose both the
geometric uncertainty and rounding error.

The certificate carries its own `pointCount` (eight is the default used by this cuboid example).
Its point tensor has shape `[pointCount, 3]`, and the depth and containment conditions quantify
over those supplied points. A different count describes the same projection problem for another
finite point set. With zero points, the pointwise conditions are vacuous, while the image and box
checks still apply. Acceptance doesn't establish that the points are the corners of a cuboid or
that a detector found the right object.

# Contract Review

For the causal-attention optimization in the opening, the strict-future-zero theorem supplies a
precise reference condition.

The runtime comparison should include visible keys, strict-future keys, and fully blocked rows.
The first causal row has exactly one visible key. A polarity flip can preserve the output shape
while changing all three cases: the current key becomes blocked, future keys become visible, and
the last row becomes fully blocked. Comparing the resulting weights reveals which condition fails.

A layout change also needs to preserve which axis represents queries and which represents keys. A
fused attention implementation {Informal.citep flashattention2022}[] needs to preserve the chosen
mask and normalization semantics. If a separate loss optimization combines softmax and logarithm,
the stable-loss equation specifies its reference expression. For compiler lowering, the theorem
above covers the step from the recorded typed graph to the IR, and for float32 execution, a
numerical conformance argument must also account for rounding and the selected kernels.

Placing a module under `TorchLeanExamples/BugZoo/` is enough to add it to the maintained catalog
build, because the examples library's glob covers every `.lean` file there. After adding a catalog
entry, build that library:

```terminal
# Elaborate every catalog entry with the other examples.
scripts/lake.sh build TorchLeanExamples
```

# Verification Scope

Our theorems describe TorchLean reference objects. A numerical comparison measures a particular
implementation under its selected dtype, device, and execution policy. Establishing a deployment
guarantee requires connecting that runtime to the reference object and the theorem's hypotheses.

Remaining obligations and scope distinctions:

- The KV cache theorems cover one append: the final slot (`appendToken_last`) and every earlier slot
  (`appendToken_past`). Agreement of cached decoding with full-sequence attention is not proved.
- The attention entry also proves $`\exp(-\infty)=0` in `EReal`, using the pinned Mathlib
  extended-real exponential. This chapter uses the real-valued hard-mask theorem directly, while the
  maintained `TorchLeanExamples` build includes the extended-real entry.
- The one-feature LayerNorm scale-gradient theorem concerns the `scaleGradient` of
  `LayerNorm.backward`, and the constant-slice theorem concerns a scalar expression. They are
  separate propositions.
- The masked-mean epsilon is a policy choice. With one active label it multiplies the real-valued
  mean by $`1/(1+\varepsilon)`, a relative shrinkage of $`\varepsilon/(1+\varepsilon)`. An
  alternative fully ignored-batch policy needs its own definition.
- FloatLib's native add/sub proofs require finite operands. Its square-root export theorem
  covers all configured inputs through NaN canonicalization. Compiled CPU instructions, vectorized
  reductions, and CUDA kernels are separate boundaries with no theorem here.
- The 3D projection check is exact at `ℚ`. At `Float` the same theorem still establishes
  `Verified` for the float-valued predicate. Exact real containment needs the interval
  statement described above and an argument that the float executor's intervals enclose the
  projection.
