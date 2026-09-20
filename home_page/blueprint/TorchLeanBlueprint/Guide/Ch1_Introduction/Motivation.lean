import VersoManual
import NN.API
import NN.Spec.Layers.Attention
import Mathlib.Tactic.Linarith
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean

#doc (Manual) "Motivation" =>
%%%
tag := "motivation"
file := "Why-Execution-Alone-Is-Not-Enough"
%%%

Suppose a classifier returns class `3` on an image. We have learned what happened at one point. A
robustness claim asks a larger question: does class `3` remain ahead of every competitor throughout
a whole neighborhood of that image? That question is the one adversarial examples made unavoidable
{Informal.citep szegedy2014}[], because the neighbors that flip the answer are usually invisible to
the eye and may be absent from a test set.

The difference is visible in the quantifiers. Let $`f_\theta(x)` be the vector of class scores
for parameters $`\theta`, and let $`c_\theta(x)` select a class with maximal score, using a fixed
rule for ties. For the image $`x_0`, a prediction is one computation:

$$`c_\theta(x_0)=y`.

A local robustness statement concerns every point in a region. In the score-based expression
below, $`j` names a competing class and the indexed outputs are class scores. The
$`\ell_\infty` distance bounds the change in each input coordinate by $`\varepsilon`:

$$`\forall x,\quad \lVert x-x_0\rVert_\infty\leq\varepsilon
  \Longrightarrow
  f_\theta(x)_y-f_\theta(x)_j>0
  \quad\text{for every }j\ne y`.

Over real-valued inputs, the second line ranges over an uncountable set. Sampling can find a
counterexample, but checking sampled predictions alone does not establish the universal claim.
A verifier needs a description of the region and a way to bound the network everywhere inside it.

# Attention Masking

An attention mask should give a blocked key zero weight. Subtracting a large constant from its
score often produces that result in tests, yet the subtraction can leave almost all the attention
on the blocked key.

For query $`i`, let $`s_{ij}` be its score for key $`j`, and let $`A_i` be the keys that are
allowed to receive attention. When there is at least one allowed key, a hard mask means

$$`
\operatorname{attention}_{ij}
=
\begin{cases}
\dfrac{\exp(s_{ij})}
      {\sum_{k\in A_i}\exp(s_{ik})}, & j\in A_i,\\[1.2ex]
0, & j\notin A_i.
\end{cases}
`

Blocked entries never enter the denominator and receive exactly zero weight. A common numerical
shortcut instead subtracts a large positive constant $`C` from a blocked logit before softmax. For a
blocked position $`j\notin A_i`, its weight is

$$`
\widetilde{\operatorname{attention}}_{ij}
=
\frac{\exp(s_{ij}-C)}
     {\sum_{k\in A_i}\exp(s_{ik})
       +\sum_{k\notin A_i}\exp(s_{ik}-C)}.
`

For ordinary logits and a large $`C`, this value may underflow to zero in a particular
floating-point run. Mathematically, however, it is positive for every finite $`C`. Worse, the
shortcut is not safe for arbitrary logits. If a blocked score is $`C+100`, then subtracting $`C`
leaves the very large score $`100`; the supposedly blocked key can dominate the softmax.

## Hard Masks And Additive Masks

Consider three scores with the third key blocked and a mask shift of $`C=10^9`. The blocked
score is larger than the shift, so subtraction leaves it above the two allowed scores:

```lean (name := moMaskDefs)
-- Make the blocked score exceed the finite penalty so the
-- two masking rules disagree.
def moScores : Tensor Float [3] :=
  Tensor.ofFn fun i => [1.0, 2.0, 1000000100.0][i.val]!

def moMask : Tensor Bool [3] :=
  Tensor.ofFn fun i => [true, true, false][i.val]!

def moBig : Float := 1000000000.0

def moShortcut (s : Tensor Float [3]) : Tensor Float [3] :=
  Activation.softmaxVecSpec (Tensor.ofFn fun i =>
    if moMask.getScalar i then s.getScalar i
    else s.getScalar i - moBig)
```

{lean}`Spec.hardMaskedSoftmaxVecSpec moScores moMask` is TorchLean's specification applied to those
scores: the hard-mask equation, written directly. {lean}`moShortcut moScores` subtracts the finite
penalty before softmax:

```lean (name := moMaskEval)
-- Compare exclusion from normalization with merely
-- subtracting a large finite number.
#eval Spec.hardMaskedSoftmaxVecSpec moScores moMask
#eval moShortcut moScores
```

```leanOutput moMaskEval
[0.268941, 0.731059, 0.000000]
```

```leanOutput moMaskEval
[0.000000, 0.000000, 1.000000]
```

The first two scores differ by one, so their unnormalized allowed weights have ratio
$`1:\exp(1)`. Dividing by their sum produces the first two entries of the hard-mask result.
The third score never participates in that sum. In the shortcut, subtracting a billion still leaves
a score of one hundred, much larger than either allowed score. Normalization then gives almost all
the weight to the very position meant to be excluded. The problem is therefore already present in
the forward equation, before gradients, training, or backend selection enter the picture.

Lower the blocked score to `0.5`, and the displayed results agree:

```lean (name := moTameEval)
-- Lower only the blocked score: this easier input can hide
-- the faulty masking rule.
def moTame : Tensor Float [3] :=
  Tensor.ofFn fun i => [1.0, 2.0, 0.5][i.val]!

#eval Spec.hardMaskedSoftmaxVecSpec moTame moMask
#eval moShortcut moTame
```

```leanOutput moTameEval
[0.268941, 0.731059, 0.000000]
```

```leanOutput moTameEval
[0.268941, 0.731059, 0.000000]
```

:::table +header
*
  * scores
  * hard mask
  * additive shortcut
*
  * `[1.0, 2.0, 0.5]`
  * `[0.268941, 0.731059, 0.0]`
  * `[0.268941, 0.731059, 0.0]`
*
  * `[1.0, 2.0, 1e9+100]`
  * `[0.268941, 0.731059, 0.0]`
  * `[0.0, 0.0, 1.0]`
:::

The shortcut agrees on the first row because $`\exp(0.5-10^9)` underflows to zero in binary64.
Over the reals that exponential remains positive. The second row shows why agreement on moderate
scores cannot justify substituting the shortcut for the hard-mask definition.

## Attention Masking In PyTorch

The same comparison in PyTorch 2.13.0 uses `masked_fill` with `-inf` for the hard mask and
subtraction for the shortcut:

```
>>> # A forbidden score can exceed a finite penalty;
>>> # replacing it by -inf excludes it.
>>> scores = torch.tensor([1.0, 2.0, 1_000_000_100.0])
>>> mask   = torch.tensor([True, True, False])
>>> torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=0)
tensor([0.2689, 0.7311, 0.0000])
>>> torch.softmax(scores - (~mask) * 1_000_000_000.0, dim=0)
tensor([0., 0., 1.])
```

Both implementations accept the same tensor shapes. The
{ref "tensors-shapes"}[shape interfaces] cannot distinguish them; the
{ref "spec-layer"}[specification layer] can, because it defines the hard-mask function with zero
numerators for blocked entries. A runtime provider needs a contract relating its computation to
that function.

## Limits Of Bounded Random Testing

To understand which tests can expose the difference, consider where the blocked exponential stops
underflowing. Binary64
exponentials near $`-745` show the relevant scale. Stable softmax first subtracts the row maximum,
so the threshold for a score also depends on the other scores. Rounding can hide differences even
when a blocked exponential is nonzero:

```lean (name := moUnderflow)
-- Locate two adjacent integer exponents where Float changes
-- from tiny nonzero to zero.
#eval (Float.exp (-745.0) == 0.0,
  Float.exp (-746.0) == 0.0)

-- Undo the additive mask shift for this illustrative
-- exponent threshold (before softmax's max shift).
#eval moBig - 745.0

-- What a generator of ordinary logits actually reaches.
#eval ((List.range 21).map fun k =>
  Float.exp ((-10.0 + k.toFloat) - moBig)).foldl max 0.0
```

```leanOutput moUnderflow
(false, true)
```

```leanOutput moUnderflow
999999255.000000
```

```leanOutput moUnderflow
0.000000
```

The Boolean pair shows that the exponential at minus 745 is still nonzero, whereas the one at
minus 746 has rounded to zero. The next number translates an exponent threshold back through the
finite mask penalty. It is not a universal cutoff for a softmax row: the stability shift also
depends on the row maximum. The final check samples a grid of moderate scores.

For the moderate allowed scores used here, a blocked score must approach $`10^9` before its
exponential survives the masking and stability shifts. Scores in $`[-10,10]` underflow with room to
spare in this example. A random
test restricted to that range never proposes a point where the large-score failure occurs.
Increasing its sample count does not change its support, even if the tests execute both branches
of the mask.

On the large-score example, the blocked key takes the entire displayed distribution, weight
$`1.0`. A specification quantified over all logits includes this case regardless of the
distribution used to generate tests.

# Input Normalization

The function being checked also depends on preprocessing. Suppose the model consumes normalized
vectors, where $`\mu_i` and $`\sigma_i` are the mean and positive scale for coordinate $`i`:

$$`N(x)_i=\frac{x_i-\mu_i}{\sigma_i}`.

If the raw input lies in a box

$$`x_i\in[\ell_i,u_i]`,

then, when $`\sigma_i>0`, the normalized box is

$$`N(x)_i\in
  \left[
    \frac{\ell_i-\mu_i}{\sigma_i},
    \frac{u_i-\mu_i}{\sigma_i}
  \right]`.

For example, take the MNIST normalization constants below and the raw pixel range $`[0,1]`:

```lean (name := moNormBox)
-- Move the raw endpoints and perturbation radius into the
-- coordinates the model sees.
def moMu : Float := 0.1307
def moSigma : Float := 0.3081

def moNormBox (lo hi : Float) : Float × Float :=
  ((lo - moMu) / moSigma, (hi - moMu) / moSigma)

#eval moNormBox 0.0 1.0
#eval 0.1 / moSigma
```

```leanOutput moNormBox
(-0.424213, 2.821487)
```

```leanOutput moNormBox
0.324570
```

The positive standard deviation preserves endpoint order. For a perturbation, the mean cancels
between the original and perturbed inputs, leaving the radius divided by `moSigma`. These Float
calculations identify the change of coordinates; a certified numerical enclosure would also need
to account for endpoint rounding.

A verifier that starts after normalization needs the first pair as its pixel bounds. Using
$`[0,1]` there would describe a different region: the transformed interval is more than three
times as wide and has a different center. The
second number is the matching correction for a perturbation radius. An $`\varepsilon` of $`0.1` in
raw pixels is a radius of about $`0.3246` after dividing by $`\sigma`, so a verifier fed the raw
radius would check a region roughly a third of the intended size. A successful check would then
cover too little of the intended neighborhood. Both regions have the same tensor shape, so the
coordinate convention must be recorded separately.

# Interval Bound Propagation

To see how a region becomes an output bound, run the interval propagation workflow from the
repository root:

```terminal
# Lower the fixed small MLP and propagate the input box
# through its graph.
lake exe verify -- torchlean-ibp
```

It prints:

```terminal +output
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: native binary32
lowered IR nodes: 18
output box lo: [1.904000]
output box hi: [2.256001]
```

The source is
{src "NN/Verification/Builtin/IBPWorkflow.lean"}[`NN/Verification/Builtin/IBPWorkflow.lean`].
It constructs a two-input, three-hidden-unit ReLU MLP with an explicit parameter payload. It lowers
that forward program to `NN.IR.Graph`, places an $`\ell_\infty` box of radius $`0.1` around
$`(0.5,0.8)`, and runs interval bound propagation.

The two vectors are the computed lower and upper endpoints of an output box. Before accounting
for floating-point rounding, the ideal endpoints are `1.904` and `2.256`, whose midpoint is

$$`\frac{1.904+2.256}{2}=2.08`

and radius

$$`\frac{2.256-1.904}{2}=0.176`.

## Derivation Of The Output Bounds

We can derive these endpoints from the parameters. The first
layer has weights $`W_1` with rows $`(0.1,0.2)`, $`(0.3,0.4)`, $`(0.5,0.6)` and bias
$`b_1=(0.1,0.2,0.3)`, and the second layer has $`W_2=(0.7,0.8,0.9)` with bias $`b_2=0.4`. The
input box is $`x_1\in[0.4,0.6]`, $`x_2\in[0.7,0.9]`.

Every weight is positive, so each pre-activation is largest at the box corner $`(0.6,0.9)` and
smallest at $`(0.4,0.7)`. At the low corner the three pre-activations are

$$`(0.1)(0.4)+(0.2)(0.7)+0.1=0.28,\quad 0.6,\quad 0.92`,

all positive, so on this box every ReLU is the identity and the network is affine. The output at
the low corner is

$$`(0.7)(0.28)+(0.8)(0.6)+(0.9)(0.92)+0.4=1.904`,

and the same computation at $`(0.6,0.9)` gives $`2.256`. Because the map is affine and monotone
here, those corner values *are* the exact range, and interval propagation loses nothing.

## Real-Valued Enclosure Proof

The corner calculation suggests a bound on the whole box. To prove that bound, define the same
network over the reals, with `max 0` for ReLU:

```lean (name := moNetDef)
-- Fix the real-valued network and its parameters before
-- stating an enclosure for all inputs.
noncomputable def moNet (x1 x2 : ℝ) : ℝ :=
  0.7 * max 0 (0.1 * x1 + 0.2 * x2 + 0.1)
    + 0.8 * max 0 (0.3 * x1 + 0.4 * x2 + 0.2)
    + 0.9 * max 0 (0.5 * x1 + 0.6 * x2 + 0.3)
    + 0.4
```

The four hypotheses below give the lower and upper bounds on the two inputs. The proof first uses
them to show that each pre-activation is nonnegative, replaces the ReLUs by their inputs, and then
proves the two affine inequalities.

```lean (name := moProof)
-- Prove both output inequalities for any point satisfying
-- the four input inequalities.
theorem moEnclosure (x1 x2 : ℝ)
    (h1 : 0.4 ≤ x1) (h2 : x1 ≤ 0.6)
    (h3 : 0.7 ≤ x2) (h4 : x2 ≤ 0.9) :
    1.904 ≤ moNet x1 x2 ∧ moNet x1 x2 ≤ 2.256 := by
  -- On this box every pre-activation is positive, so each
  -- `max 0 ·` is the identity and the network is affine.
  have e1 : max 0 (0.1 * x1 + 0.2 * x2 + 0.1)
      = 0.1 * x1 + 0.2 * x2 + 0.1 :=
    max_eq_right (by linarith)
  have e2 : max 0 (0.3 * x1 + 0.4 * x2 + 0.2)
      = 0.3 * x1 + 0.4 * x2 + 0.2 :=
    max_eq_right (by linarith)
  have e3 : max 0 (0.5 * x1 + 0.6 * x2 + 0.3)
      = 0.5 * x1 + 0.6 * x2 + 0.3 :=
    max_eq_right (by linarith)
  rw [moNet, e1, e2, e3]
  constructor <;> nlinarith
```

The variables `x1` and `x2` remain arbitrary throughout the proof. The names `h1` through `h4`
refer to evidence that those variables lie inside the chosen rectangle; they do not assign values
to them. The result joined by `∧` contains two proofs, one for the lower endpoint and one for the
upper endpoint. Each intermediate equality removes a ReLU only after establishing its input is
nonnegative over the entire rectangle. Once those three equalities are substituted, the remaining
inequalities concern an affine expression, which the arithmetic tactic can discharge.

An enclosure may be loose. Evaluating the two corners proves that these endpoints are attained:

```lean (name := moCorners)
-- Show that neither real endpoint can be tightened: each is
-- reached at a box corner.
theorem moLowerCorner : moNet 0.4 0.7 = 1.904 := by
  norm_num [moNet]

theorem moUpperCorner : moNet 0.6 0.9 = 2.256 := by
  norm_num [moNet]
```

Endpoint attainment answers a question that enclosure alone leaves open. A sound procedure could
return a much wider interval and still contain every output. Here the lower-corner equality rules
out raising the lower bound, and the upper-corner equality rules out lowering the upper bound.
These are exact real equalities with the displayed decimal constants interpreted as rationals.
They do not assert that either native endpoint has the same bit representation as those rationals;
the later comparison of encodings concerns that separate numerical question.

Taken together, {lean}`moEnclosure`, {lean}`moLowerCorner`, and {lean}`moUpperCorner` say that
$`[1.904,2.256]` is the exact range of this network over this box. Exactness depends on the
structure of this example. Dependency loss can make intervals overapproximate, especially across
multiple layers; a crossing ReLU or mixed weight signs alone do not force a loose result.
Affine relaxations can retain relationships that separate intervals discard
{Informal.citep crown2018}[]{Informal.citep autolirpa2020}[].

This theorem concerns $`\mathbb{R}`, while the command above ran in binary32. Connecting the
two requires the arithmetic argument developed in
{ref "fp32-soundness"}[the soundness chapter]. First, consider a source of looseness that arises
even with exact arithmetic: dependencies between intermediate values.

# Dependency Loss And Affine Relaxations

In the preceding network, every weight was positive and every ReLU was active across the whole box.
All intermediate maxima occurred at the same corner, so combining their bounds lost no
information. Consider instead one input feeding two hidden ReLU units with weights $`+1` and
$`-1`, whose outputs are added together.

$$`g(x)=\operatorname{ReLU}(x)+\operatorname{ReLU}(-x)=|x|.`

Over the box $`x\in[-1,1]` the true range of $`g` is $`[0,1]`. Interval propagation bounds each
hidden unit separately, then adds their intervals without retaining the relationship between them:

```lean (name := moBoxOps)
-- Interval arithmetic for the two operations we need,
-- which is all IBP does at a ReLU and at an addition.
def moReluBox (lo hi : Float) : Float × Float :=
  (max 0.0 lo, max 0.0 hi)

def moAddBox (a b : Float × Float) : Float × Float :=
  let (aLo, aHi) := a
  let (bLo, bHi) := b
  (aLo + bLo, aHi + bHi)

def moAbsF (x : Float) : Float :=
  max 0.0 x + max 0.0 (-x)

-- `x ∈ [-1, 1]` gives `-x ∈ [-1, 1]`, so both hidden
-- units get the same input box.
#eval moAddBox (moReluBox (-1.0) 1.0)
  (moReluBox (-1.0) 1.0)
```

```leanOutput moBoxOps
(0.000000, 2.000000)
```

The lost information is the relation between the two branches. The interval for `relu(x)` reaches
one at `x = 1`, while the interval for `relu(-x)` reaches one at `x = -1`. Adding interval endpoints
allows those two maxima to occur together. There is no such input. Keeping a common symbolic input
in a linear bound can prevent that combination, which is why a more expensive bound representation
may improve the result even when both procedures use perfectly correct arithmetic.

The propagated box is $`[0,2]`. The function never leaves $`[0,1]`; the theorem below proves the
upper bound. A finite grid
provides an illustrative numerical check:

```lean (name := moGrid)
-- Sample 201 points as a numerical check; the following
-- proof handles the whole interval.
#eval ((List.range 201).map fun k =>
  moAbsF (-1.0 + 0.01 * k.toFloat)).foldl max 0.0
```

```leanOutput moGrid
1.000000
```

The upper bound is off by a factor of two on a network with two hidden units. Nothing went wrong in
the arithmetic: $`\operatorname{ReLU}(x)\in[0,1]` and $`\operatorname{ReLU}(-x)\in[0,1]` are both
correct, and $`[0,1]+[0,1]=[0,2]` is the correct interval sum. What is lost is that the two summands
cannot both be $`1`. Interval arithmetic drops the dependency between them the moment it replaces
each one by a box, and that loss is systematic rather than a rounding artifact.

A CROWN relaxation retains some of this dependency by bounding each ReLU with a line in the
input. On this interval, the line joining the endpoints of the ReLU graph gives the upper bound

$$`\operatorname{ReLU}(x)\leq\frac{x+1}{2}
\qquad\text{for }x\in[-1,1].`

The two bounds remain functions of the same variable. Their slopes cancel when they are added:
$`\frac{x+1}{2}+\frac{-x+1}{2}=1`. The first theorem below proves the line bound; the second applies
it at both inputs and adds the resulting inequalities:

```lean (name := moRelax)
-- Bound each ReLU by a line in the same input x so their
-- shared dependence is retained.
noncomputable def moAbsR (x : ℝ) : ℝ :=
  max 0 x + max 0 (-x)

theorem moReluRelax (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 1) :
    max 0 x ≤ (x + 1) / 2 := by
  rcases le_total 0 x with h | h
  · rw [max_eq_right h]; linarith
  · rw [max_eq_left h]; linarith

theorem moAbsUpper (x : ℝ) (h1 : -1 ≤ x) (h2 : x ≤ 1) :
    moAbsR x ≤ 1 := by
  -- The same relaxation at `x` and at `-x`. Because both
  -- lines are linear in `x`, adding them cancels `x`.
  have hx := moReluRelax x h1 h2
  have hn := moReluRelax (-x) (by linarith) (by linarith)
  rw [moAbsR]
  linarith

theorem moAbsAttained : moAbsR 1 = 1 := by
  norm_num [moAbsR]
```

The hypotheses of `moReluRelax` are exactly the range on which its sloping line is an upper bound.
For a negative input, ReLU is zero and the line is nonnegative; for a positive input, the inequality
reduces to `x ≤ 1`. Applying the same statement to `-x` gives a second line whose slope has the
opposite sign. Their sum is one, so the dependence on `x` cancels before any interval endpoints are
chosen. `moAbsAttained` then checks an input reaching that upper bound. The argument explains the
improvement without relying on the grid experiment.

{lean}`moAbsUpper` recovers $`1`, and {lean}`moAbsAttained` says $`1` is reached, so the linear
relaxation is exact on this example while the interval upper bound was off by a factor of two.
Retaining input dependence is the reason to carry affine bounds through a graph, as developed in
{ref "certificates"}[the certificates chapter] and the CROWN literature
{Informal.citep crown2018}[]{Informal.citep autolirpa2020}[].

The linear relaxation is not always exact. Here its two slopes cancel and its upper bound is
attained at an endpoint; piecewise linearity alone does not guarantee that result. Dependency loss
can also grow across layers. Training with an IBP objective can encourage more useful interval
bounds {Informal.citep gowal2018}[].

Looseness alone does not invalidate a sound bound. An IBP result of $`[0,2]` cannot establish
$`g\leq1`, but it also supplies no counterexample. This test is inconclusive; a tighter relaxation
can establish the property, as the affine calculation above does.

# Native And Reference IEEE Bounds

The workflow takes an `--arithmetic` flag. Running the same graph, the same parameters, and the same
input box through the reference IEEE binary32 semantics instead of the host's native `Float32`
changes the answer:

```terminal +output
$ lake exe verify -- torchlean-ibp --arithmetic=ieee
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: IEEE-754 binary32 reference
lowered IR nodes: 18
output box lo: [1.904000]
output box hi: [2.256000]
```

The upper endpoint printed `2.256001` in the native run and prints `2.256000` here. The bit
patterns expose the difference more precisely than the six-decimal output. Both runs return
binary32 numbers; here are approximate offsets from the ideal decimal endpoints, in units of
$`10^{-9}`:

```lean (name := moUlp)
-- Decode recorded binary32 endpoints and magnify their
-- signed offsets for inspection.
def moOff (bits : UInt32) (exact : Float) : Float :=
  ((Float32.ofBits bits).toFloat - exact) * 1000000000.0

#eval (moOff 1072936516 1.904, moOff 1072936517 1.904)
#eval (moOff 1074815568 2.256, moOff 1074815567 2.256)
```

```leanOutput moUlp
(-194.549560, -75.340271)
```

```leanOutput moUlp
(518.798828, 280.380249)
```

The first component of each pair is the native run, the second is the IEEE reference run.
Both lower endpoints lie below the comparison value and both upper endpoints above it.
Multiplication by a billion makes the offsets readable but adds no precision: `moOff` computes
in host binary64, including its decimal reference value. These are numerical diagnostics, not
exact real identities or a proof of outward rounding for the native implementation.

:::table +header
*
  * endpoint
  * native `Float32`
  * IEEE binary32 reference
  * exact
*
  * lower, bits
  * `1072936516`
  * `1072936517`
  * not representable
*
  * lower, offset
  * `-1.945e-7`
  * `-7.534e-8`
  * `1.904`
*
  * upper, bits
  * `1074815568`
  * `1074815567`
  * not representable
*
  * upper, offset
  * `+5.188e-7`
  * `+2.804e-7`
  * `2.256`
:::

The listed lower endpoints lie below the ideal lower bound and the upper endpoints above the
ideal upper bound, so both listed boxes contain the real-valued range proved earlier. The two
runs differ by exactly one unit in the last place at each endpoint, the ULP near $`2.256` being
$`2.384\times10^{-7}`. The native box is wider in this example. This comparison does not establish
a soundness theorem for either backend over arbitrary graphs or inputs.

The relative widths in this example do not establish an ordering for other inputs. A verifier that
computes bounds in one arithmetic and reports them as if they held in another has an unproved step
in the middle, and that step has been turned into working attacks on published verifiers
{Informal.citep jiarinard2020}[]. Neither decimal endpoint is even a binary32 number, so there is no
reading of `output box hi: [2.256000]` under which the printed digits are the computed value.

# Certificate Soundness

A useful verification claim has a chain of named objects:

$$`
\begin{aligned}
\text{model source}
&\longrightarrow \text{initialized architecture and parameters}\\
&\longrightarrow \text{semantic graph}\\
&\longrightarrow \text{input region}\\
&\longrightarrow \text{bound or certificate}\\
&\longrightarrow \text{proved property}.
\end{aligned}
`

Each arrow carries one piece of the argument.

- Did initialization produce the parameter payload that was later analyzed?
- Did lowering preserve the model's forward computation?
- Does the region describe raw inputs or already transformed inputs?
- Did the checker interpret each graph operation with the intended scalar arithmetic?
- Does acceptance imply the property written in the theorem?

## Stale Parameter Payloads

The first connection can fail when parameters change after verification. Take the network whose
real-valued range we proved and change one weight, as an optimizer step might: the second
layer's $`W_2=(0.7,0.8,0.9)` becomes $`(0.7,0.8,0.95)`. The graph structure and parameter
shapes remain the same.

```lean (name := moStale)
-- Change one readout weight while preserving every tensor
-- shape and input coordinate.
def moW2 : Tensor Float [3] := [0.7, 0.8, 0.9]
def moW2' : Tensor Float [3] := [0.7, 0.8, 0.95]

def moHidden (x1 x2 : Float) : Tensor Float [3] :=
  [ max 0.0 (0.1 * x1 + 0.2 * x2 + 0.1)
   , max 0.0 (0.3 * x1 + 0.4 * x2 + 0.2)
   , max 0.0 (0.5 * x1 + 0.6 * x2 + 0.3) ]

def moOutput (w : Tensor Float [3])
    (x1 x2 : Float) : Float :=
  let h := moHidden x1 x2
  (Tensor.mul w h).sum + 0.4

#eval do
  let a := moOutput moW2 0.6 0.9
  let b := moOutput moW2' 0.6 0.9
  IO.println s!"certified upper   = 2.256"
  IO.println s!"analyzed payload  = {a}"
  IO.println s!"deployed payload  = {b}"
  let ok : Bool := b <= 2.256
  IO.println s!"still in the box  = {ok}"
```

```leanOutput moStale
certified upper   = 2.256
analyzed payload  = 2.256000
deployed payload  = 2.313000
still in the box  = false
```

At the upper corner, the third hidden activation is 1.14. Increasing its readout coefficient by
0.05 raises the prediction from 2.256 to 2.313, overshooting the old bound by $`0.057`, about a
third of the box's radius. The graph structure and shapes have not changed.
{lean}`moEnclosure` remains true for the weights it names; applying it to the new payload fails
because it describes a different function. A verification pipeline must track the parameter values
it analyzed through to deployment, even when the architecture is unchanged.

This is why {name}`Spec.hardMaskedSoftmaxVecSpec` and its relatives take their parameters as
arguments, and why `denote` in {ref "graphs-and-ir"}[the IR chapter] takes a payload rather than
reading weights from somewhere ambient. A soundness theorem whose statement mentions $`\theta`
forces the caller to say which $`\theta`, and the two-stage workflows in
{ref "twostage"}[that chapter] carry the same payload through search and checking for
exactly this reason.

Let $`g` be the graph, $`\theta` its parameter payload, $`B` the input region, and $`c` a
certificate supplied to the checker. A checker soundness theorem usually has the form

$$`\operatorname{check}(g,\theta,B,c)=\mathrm{true}
  \Longrightarrow
  \operatorname{Property}(\operatorname{denote}(g,\theta),B)`.

The certificate may come from an expensive external search;
the checker validates the resulting artifact. Branch-and-bound
verifiers make the split concrete, exploring a tree of neuron splits and emitting a bound per leaf
{Informal.citep bunel2020}[]; {ref "certificates"}[the certificates chapter] reads such artifacts
and checks the leaves in Lean without reproducing the search.

# Executable Verification Workflows

The command

```terminal
# Discover the workflow names before choosing the checker
# and its artifact.
lake exe verify -- list
```

shows the executable verification workflows in the current checkout, one line each:

```
  lirpa-mlp [<path>]   -- IBP cert: feed-forward MLP
  lirpa-attention [<path>]   -- IBP cert: attention softmax block
  pinn-cert [<path>]   -- PINN certificate recomputation check
  abcrown-leaf [<path>]   -- α,β-CROWN leaf artifact structural check
  torchlean-ibp   -- TorchLean → IR → IBP workflow (MLP)
  torchlean-crown-ops   -- TorchLean → IR → IBP+CROWN workflow
  ode   -- ODE enclosure verification (sub/super NN bounds)
  vnncomp-mnistfc   -- VNN-COMP-style suite: MNIST-FC (vnncomp2022)
  twostage-torchlean-cegis-van   -- all-in-Lean two-stage refinement
```

The full output also prints each tool's default artifact path.

# Floating-Point Arithmetic

For real numbers $`a`, $`b`, and $`c`, the expression

$$`(a+b)+c`

can be regrouped without changing its value. Binary floating-point addition rounds after each
operation, so regrouping may change the result:

```lean (name := moAssoc)
-- Change only the parentheses, then compare the complete
-- binary64 encodings.
#eval (0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3)
#eval ((0.1 + 0.2) + 0.3).toBits
#eval (0.1 + (0.2 + 0.3)).toBits
```

```leanOutput moAssoc
false
```

```leanOutput moAssoc
4603579539098121012
```

```leanOutput moAssoc
4603579539098121011
```

The two integer encodings differ by one, even though decimal formatting hides the distinction.
Each parenthesization chooses a different intermediate sum to round. Real associativity therefore
cannot justify replacing one execution order by the other without an additional numerical argument.
Both sides print as `0.600000`; a comparison against a threshold can expose the difference. Fused
multiply-add, reduction order, subnormal handling, NaNs, and overflow introduce further
distinctions; {ref "floating-point-literature"}[the literature chapter] traces how Flocq and its
relatives handle the same questions in Rocq {Informal.citep flocq2011}[].

The arithmetic named in the statement matters:

- exact real-valued specifications;
- configurable rounded-real arithmetic with checked positive format precision;
- `FP32`, a rounded-real model with binary32 precision and gradual-underflow parameters but no
  upper exponent bound or IEEE special values;
- executable bit-level IEEE binary32;
- native `Float32`, host `Float`, CUDA, and external runtime providers.

A real-valued enclosure theorem cannot be silently relabeled as a theorem about every GPU execution.
A bridge theorem or an explicit backend contract must carry the result across that boundary. The
{ref "floats"}[floating-point chapter] develops these relationships in detail.

A change in formula can also reduce overflow risk while preserving the real-valued function.
For smooth-max pooling, let $`\beta` be a nonzero scale and choose a shift $`p` from the extreme
input value as follows:

$$`
p+\frac{1}{\beta}
  \log\!\left(\sum_i e^{\beta(x_i-p)}\right),
\qquad
p=
\begin{cases}
\max_i x_i,&\beta>0,\\
\min_i x_i,&\beta<0.
\end{cases}
`

The shift keeps every exponential argument nonpositive. With $`\beta=1000` and inputs
$`\{0,1\}`, evaluating the
unshifted sum of exponentials overflows:

```lean (name := moShiftBad)
-- The larger exponent overflows even though the smooth
-- maximum itself is near one.
def moBeta : Float := 1000.0

#eval Float.exp (moBeta * 0.0) + Float.exp (moBeta * 1.0)
```

```leanOutput moShiftBad
inf
```

Subtracting the maximum first gives an algebraically equivalent expression with finite
intermediates in this example:

```lean (name := moShiftGood)
-- Shift by the largest input so the exponent arguments are
-- zero or negative.
def moP : Float := 1.0

#eval moP + Float.log
    (Float.exp (moBeta * (0.0 - moP))
      + Float.exp (moBeta * (1.0 - moP))) / moBeta
```

```leanOutput moShiftGood
1.000000
```

For these inputs the shifted exponentials are `exp(-1000)` and `exp(0)`. The latter supplies one
to the sum, while the former is too small to affect this Float result. In real arithmetic the
smooth maximum is slightly above one, by `log(1 + exp(-1000)) / 1000`; the printed one is the
rounded evaluation. This is a useful example of separating two goals: avoid an infinite
intermediate, and describe the error of the finite answer that remains. The algebraic rewrite
addresses the first; a numerical error statement is still needed for the second.

TorchLean uses the same shifted weights in the forward and derivative paths, and executable entry
points reject zero or non-finite $`\beta`. The real identity explains the transformation; the
specification over $`\mathbb{R}` fixes the intended function. Validation and backend tests check
its executable form in the selected format.

# Proof Dependencies And Executable Examples

Lean lets the program, its mathematical interpretation, the checker, and the theorem refer to the
same definitions. A schema change can expose a parser's missing cases, and changing an operation
can invalidate proofs that depend on its old equation. Applying a soundness theorem requires
every hypothesis in its statement, including those that connect the artifact to the intended model.

The named Lean output blocks provide another check: the guide build reruns the broken mask and
checks the two bit patterns of `0.6`. Shell transcripts such as the IBP workflow and Python examples
still need separate execution, and prose claims need review. Executable documentation catches
drift in the examples it actually runs.

# References

Adversarial examples and the quantifier problem come from {Informal.citet szegedy2014}[]. The bound
propagation used above, and its use during training, is developed in
{Informal.citet gowal2018}[]; the linear-relaxation family that tightens it is
{Informal.citet crown2018}[], {Informal.citet autolirpa2020}[], and
{Informal.citet betacrown2021}[], with the branch-and-bound search in
{Informal.citet bunel2020}[]. The floating-point gap between a verifier's arithmetic and the
network's is the subject of {Informal.citet jiarinard2020}[], and the Rocq formalization we compare
against throughout the floating-point chapters is {Informal.citet flocq2011}[].
