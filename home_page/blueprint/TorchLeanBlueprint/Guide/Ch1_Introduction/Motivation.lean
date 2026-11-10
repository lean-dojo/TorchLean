import VersoManual
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
import TorchLean.Tensor.Layers.Attention
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.NN.Loss
import TorchLean.NN.Session.Random
import TorchLean.Verify.Trainer

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open TorchLean

#doc (Manual) "Motivation" =>
%%%
tag := "motivation"
file := "Why-Execution-Alone-Is-Not-Enough"
%%%

Suppose a classifier returns class `3` on an image. We've learned what happened at one point. A
robustness claim asks a larger question: whether class `3` remains ahead of every competitor
throughout a whole neighborhood of that image. Adversarial examples made that question unavoidable
{Informal.citep szegedy2014}[], because nearby inputs that flip the answer can be hard to
distinguish by eye and may be absent from a test set.

You can see the difference in the quantifiers. Let $`f_\theta(x)` be the vector of class scores for
parameters $`\theta`, and let $`c_\theta(x)` select a class with maximal score, using a fixed rule
for ties. For the image $`x_0`, a prediction is one computation:

$$`c_\theta(x_0)=y`.

A local robustness statement concerns every point in a region. In the score-based expression below,
$`j` names a competing class and the indexed outputs are class scores, while the $`\ell_\infty`
distance bounds the change in each input coordinate by $`\varepsilon`:

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
floating-point run, but mathematically it's positive for every finite $`C`. Worse, the shortcut
isn't safe for arbitrary logits. If a blocked score is $`C+100`, then subtracting $`C` leaves the
very large score $`100`, and the supposedly blocked key can dominate the softmax.

## Hard Masks And Additive Masks

Consider three scores with the third key blocked and a mask shift of $`C=10^9`. Since the blocked
score is larger than the shift, subtraction leaves it above the two allowed scores:

```lean (name := moMaskDefs)
-- Make the blocked score exceed the finite penalty so the
-- two masking rules disagree.
def moScores : Tensor Float [3] :=
  Tensor.ofFn fun i => [1.0, 2.0, 1000000100.0][i.val]!

def moMask : Tensor Bool [3] :=
  Tensor.ofFn fun i => [true, true, false][i.val]!

def moBig : Float := 1000000000.0

def moShortcut (s : Tensor Float [3]) : Tensor Float [3] :=
  Tensor.softmax (Tensor.ofFn fun i =>
    if moMask.getScalar i then s.getScalar i
    else s.getScalar i - moBig) ⟨0, Nat.one_pos⟩
```

{lean}`Softmax.hardMaskedVec moScores moMask` is TorchLean's specification applied to those
scores: the hard-mask equation as written. {lean}`moShortcut moScores` subtracts the finite penalty
before softmax:

```lean (name := moMaskEval)
-- Compare exclusion from normalization with merely
-- subtracting a large finite number.
#eval Softmax.hardMaskedVec moScores moMask
#eval moShortcut moScores
```

```leanOutput moMaskEval
[0.268941, 0.731059, 0.000000]
```

```leanOutput moMaskEval
[0.000000, 0.000000, 1.000000]
```

Since the first two scores differ by one, their unnormalized allowed weights have ratio
$`1:\exp(1)`, and dividing by their sum produces the first two entries of the hard-mask result. The
third score never participates in that sum. In the shortcut, subtracting a billion still leaves a
score of one hundred (much larger than either allowed score), and normalization then gives almost
all the weight to the very position meant to be excluded. So the problem is already present in the
forward equation, before gradients, training, or backend selection enter the picture.

Lower the blocked score to `0.5`, and the displayed results agree:

```lean (name := moTameEval)
-- Lower only the blocked score: this easier input can hide
-- the faulty masking rule.
def moTame : Tensor Float [3] :=
  Tensor.ofFn fun i => [1.0, 2.0, 0.5][i.val]!

#eval Softmax.hardMaskedVec moTame moMask
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

On the first row the shortcut agrees because $`\exp(0.5-10^9)` underflows to zero in binary64 (over
the reals that exponential remains positive). The second row shows why agreement on moderate scores
can't justify substituting the shortcut for the hard-mask definition.

## Attention Masking In PyTorch

In PyTorch 2.13.0, the same comparison uses `masked_fill` with `-inf` for the hard mask and
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

Both implementations accept the same tensor shapes, so the {ref "tensors-shapes"}[shape interfaces]
can't distinguish them. The {ref "spec-layer"}[specification layer] can, because it defines the
hard-mask function with zero numerators for blocked entries. A runtime provider needs a contract
relating its computation to that function.

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

In the Boolean pair, the exponential at minus 745 is still nonzero, whereas the one at minus 746
has rounded to zero. The next number translates an exponent threshold back through the finite mask
penalty. It isn't a universal cutoff for a softmax row (the stability shift also depends on the row
maximum). And the final check samples a grid of moderate scores.

For the moderate allowed scores used here, a blocked score must approach $`10^9` before its
exponential survives the masking and stability shifts. Scores in $`[-10,10]` underflow with room to
spare in this example, so a random test restricted to that range never proposes a point where the
large-score failure occurs. Increasing its sample count doesn't change its support, even if the
tests execute both branches of the mask.

On the large-score example, the blocked key takes the entire displayed distribution, weight $`1.0`.
A specification quantified over all logits includes this case regardless of the distribution that
generates the tests.

# Input Normalization

Which function is being checked also depends on preprocessing. Suppose the model consumes
normalized vectors, where $`\mu_i` and $`\sigma_i` are the mean and positive scale for coordinate
$`i`:

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

Because the standard deviation is positive, endpoint order is preserved. For a perturbation, the
mean cancels between the original and perturbed inputs, leaving the radius divided by `moSigma`.
These Float calculations identify the change of coordinates, but a certified numerical enclosure
would also need to account for endpoint rounding.

A verifier that starts after normalization needs the first pair as its pixel bounds. Using $`[0,1]`
there would describe a different region: the transformed interval is more than three times as wide
and has a different center. For a perturbation radius, the second number is the matching
correction. An $`\varepsilon` of $`0.1` in raw pixels is a radius of about $`0.3246` after dividing
by $`\sigma`, so a verifier fed the raw radius would check a region roughly a third of the intended
size. A successful check would then cover too little of the intended neighborhood. Both regions
have the same tensor shape, so the coordinate convention must be recorded separately.

# Interval Bound Propagation

To see how a region becomes an output bound, run the interval propagation workflow from the
repository root:

```terminal
# Lower the fixed small MLP and propagate the input box
# through its graph.
scripts/lake.sh exe verify -- torchlean-ibp
```

A recorded run printed:

```terminal +output
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: native binary32
lowered IR nodes: 18
output box lo: [1.903999]
output box hi: [2.256001]
```

That workflow's source is {src "TorchLeanExamples/Verification/IBPWorkflow.lean"}[
`TorchLeanExamples/Verification/IBPWorkflow.lean`]. It constructs a two-input, three-hidden-unit
ReLU MLP with its parameter payload written out. Then it lowers that forward program to a graph of
registered operators (`IR.Graph`) with its payload, places an $`\ell_\infty` box of radius
$`0.1` around $`(0.5,0.8)`, and runs interval bound propagation on that graph
(`IR.Imported.runIBP`).

Those two vectors are the computed lower and upper endpoints of an output box. Before accounting
for floating-point rounding, the ideal endpoints are `1.904` and `2.256`, whose midpoint is

$$`\frac{1.904+2.256}{2}=2.08`

and radius

$$`\frac{2.256-1.904}{2}=0.176`.

## Derivation Of The Output Bounds

We can derive these endpoints from the parameters. The first layer has weights $`W_1` with rows
$`(0.1,0.2)`, $`(0.3,0.4)`, $`(0.5,0.6)` and bias $`b_1=(0.1,0.2,0.3)`, and the second layer has
$`W_2=(0.7,0.8,0.9)` with bias $`b_2=0.4`. Our input box is $`x_1\in[0.4,0.6]`, $`x_2\in[0.7,0.9]`.

Every weight is positive, so each pre-activation is largest at the box corner $`(0.6,0.9)` and
smallest at $`(0.4,0.7)`. At the low corner the three pre-activations are

$$`(0.1)(0.4)+(0.2)(0.7)+0.1=0.28,\quad 0.6,\quad 0.92`,

all positive, so on this box every ReLU is the identity and the network is affine. At the low
corner its output is

$$`(0.7)(0.28)+(0.8)(0.6)+(0.9)(0.92)+0.4=1.904`,

and the same computation at $`(0.6,0.9)` gives $`2.256`. Because the map is affine and monotone
here, those corner values *are* the exact range, and interval propagation loses nothing.

## Real-Valued Enclosure Proof

That corner calculation suggests a bound on the whole box. To prove that bound, we define the same
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

In the theorem below, four hypotheses give the lower and upper bounds on the two inputs. Its proof
first uses them to show that each pre-activation is nonnegative, replaces the ReLUs by their
inputs, and then proves the two affine inequalities.

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

Throughout the proof, `x1` and `x2` remain arbitrary. The names `h1` through `h4` refer to evidence
that those variables lie inside the chosen rectangle, but they don't assign values to them. Joined
by `∧`, the result contains two proofs, one for the lower endpoint and one for the upper endpoint.
Each intermediate equality removes a ReLU only after establishing that its input is nonnegative
over the entire rectangle. Once those three equalities are substituted, the remaining inequalities
concern an affine expression, which the arithmetic tactic can discharge.

An enclosure may be loose. Evaluating the two corners proves that these endpoints are attained:

```lean (name := moCorners)
-- Show that neither real endpoint can be tightened: each is
-- reached at a box corner.
theorem moLowerCorner : moNet 0.4 0.7 = 1.904 := by
  norm_num [moNet]

theorem moUpperCorner : moNet 0.6 0.9 = 2.256 := by
  norm_num [moNet]
```

Endpoint attainment answers a question that enclosure alone leaves open, because a sound procedure
could return a much wider interval and still contain every output. Here the lower-corner equality
rules out raising the lower bound, and the upper-corner equality rules out lowering the upper
bound. These are exact real equalities, with the displayed decimal constants interpreted as
rationals. They don't assert that either native endpoint has the same bit representation as those
rationals. That separate numerical question is the subject of the later comparison of encodings.

Taken together, {lean}`moEnclosure`, {lean}`moLowerCorner`, and {lean}`moUpperCorner` say that
$`[1.904,2.256]` is the exact range of this network over this box. That exactness depends on the
structure of this example. Dependency loss can make intervals overapproximate, especially across
multiple layers, but a crossing ReLU or mixed weight signs alone don't force a loose result. Affine
relaxations can retain relationships that separate intervals discard
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
-- A toy interval calculation illustrating dependency loss.
-- The checker also follows its selected rounding policy.
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

Here the lost information is the relation between the two branches. For `relu(x)` the interval
reaches one at `x = 1`, while the interval for `relu(-x)` reaches one at `x = -1`. Adding interval
endpoints allows those two maxima to occur together. There's no such input. Keeping a common
symbolic input in a linear bound can prevent that combination, which is why a more expensive bound
representation may improve the result even when both procedures use perfectly correct arithmetic.

Propagation gives the box $`[0,2]`. The function never leaves $`[0,1]` (the theorem below proves
the upper bound). A finite grid provides an illustrative numerical check:

```lean (name := moGrid)
-- Sample 201 points as a numerical check. The following
-- proof handles the whole interval.
#eval ((List.range 201).map fun k =>
  moAbsF (-1.0 + 0.01 * k.toFloat)).foldl max 0.0
```

```leanOutput moGrid
1.000000
```

On a network with two hidden units, the upper bound is off by a factor of two. Nothing went wrong
in the arithmetic: $`\operatorname{ReLU}(x)\in[0,1]` and $`\operatorname{ReLU}(-x)\in[0,1]` are
both correct, and $`[0,1]+[0,1]=[0,2]` is the correct interval sum. What's lost is that the two
summands can't both be $`1`. Interval arithmetic drops the dependency between them the moment it
replaces each one by a box, and that loss is systematic, not a rounding artifact.

A CROWN relaxation retains some of this dependency by bounding each ReLU with a line in the
input. On this interval, the line joining the endpoints of the ReLU graph gives the upper bound

$$`\operatorname{ReLU}(x)\leq\frac{x+1}{2}
\qquad\text{for }x\in[-1,1].`

Both bounds remain functions of the same variable, and their slopes cancel when they're added:
$`\frac{x+1}{2}+\frac{-x+1}{2}=1`. The first theorem below proves the line bound, and the second
applies it at both inputs and adds the resulting inequalities:

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
For a negative input, ReLU is zero and the line is nonnegative, and for a positive input the
inequality reduces to `x ≤ 1`. Applying the same statement to `-x` gives a second line whose slope
has the opposite sign. Their sum is one, so the dependence on `x` cancels before any interval
endpoints are chosen. `moAbsAttained` then checks an input reaching that upper bound. That argument
explains the improvement without relying on the grid experiment.

{lean}`moAbsUpper` recovers $`1`, and {lean}`moAbsAttained` says $`1` is reached, so the linear
relaxation is exact on this example while the interval upper bound was off by a factor of two.
Retaining input dependence is the reason to carry affine bounds through a graph, as developed in
{ref "certificates"}[the certificates chapter] and the CROWN literature
{Informal.citep crown2018}[]{Informal.citep autolirpa2020}[].

But the linear relaxation isn't always exact. Here its two slopes cancel and its upper bound is
attained at an endpoint, a result that piecewise linearity alone doesn't guarantee. Dependency loss
can also grow across layers. Training with an IBP objective can encourage more useful interval
bounds {Informal.citep gowal2018}[].

Looseness alone doesn't invalidate a sound bound. An IBP result of $`[0,2]` can't establish
$`g\leq1`, but it also supplies no counterexample. So this test is inconclusive. A tighter
relaxation can establish the property, as the affine calculation above does.

# Native And Reference IEEE Bounds

That workflow takes an `--arithmetic` flag. Running the same graph, the same parameters, and the
same input box through the reference IEEE binary32 semantics instead of the host's native `Float32`
changes the answer:

```terminal +output
$ scripts/lake.sh exe verify -- torchlean-ibp --arithmetic=ieee
=== TorchLean → IR → IBP (small MLP) workflow ===
[TorchLean] arithmetic: IEEE-754 binary32 reference
lowered IR nodes: 18
output box lo: [1.9039997]
output box hi: [2.2560005]
```

Where the native run printed `1.903999` and `2.256001`, this run prints `1.9039997` and
`2.2560005`. Their bit patterns expose the difference more precisely than the printed decimals.
Both runs return binary32 numbers. Here are approximate offsets from the ideal decimal endpoints,
in units of $`10^{-9}`:

```lean (name := moUlp)
-- Decode recorded binary32 endpoints and magnify their
-- signed offsets for inspection.
def moOff (bits : UInt32) (exact : Float) : Float :=
  ((Float32.ofBits bits).toFloat - exact) * 1000000000.0

#eval (moOff 1072936511 1.904, moOff 1072936515 1.904)
#eval (moOff 1074815571 2.256, moOff 1074815568 2.256)
```

```leanOutput moUlp
(-790.596008, -313.758850)
```

```leanOutput moUlp
(1234.054566, 518.798828)
```

In each pair the first component is the native run and the second is the IEEE reference run. Both
lower endpoints lie below the comparison value and both upper endpoints above it. Multiplying by a
billion makes the offsets readable but adds no precision: `moOff` computes in host binary64,
including its decimal reference value. These are numerical diagnostics, not exact real identities
or a proof of outward rounding for the native implementation.

:::table +header
*
  * endpoint
  * native `Float32`
  * IEEE binary32 reference
  * exact
*
  * lower, bits
  * `1072936511`
  * `1072936515`
  * not representable
*
  * lower, offset
  * `-7.906e-7`
  * `-3.138e-7`
  * `1.904`
*
  * upper, bits
  * `1074815571`
  * `1074815568`
  * not representable
*
  * upper, offset
  * `+1.234e-6`
  * `+5.188e-7`
  * `2.256`
:::

Since the listed lower endpoints lie below the ideal lower bound and the upper endpoints above the
ideal upper bound, both listed boxes contain the real-valued range proved earlier. The two runs
differ by four units in the last place at the lower endpoint and by three at the upper endpoint
(one ULP is $`1.192\times10^{-7}` near $`1.904` and $`2.384\times10^{-7}` near $`2.256`). In this
example the native box is wider. But this comparison doesn't establish a soundness theorem for
either backend over arbitrary graphs or inputs.

And the relative widths in this example don't establish an ordering for other inputs. A verifier
that computes bounds in one arithmetic and reports them as if they held in another has an unproved
step in the middle, and that step has been turned into working attacks on published verifiers
{Informal.citep jiarinard2020}[]. Neither decimal endpoint is even a binary32 number, nor is any
printed endpoint, so there's no reading of `output box hi: [2.2560005]` under which the printed
digits are the computed value.

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

Each arrow carries one piece of the argument, and each piece has to settle one question:

- whether initialization produced the parameter payload that was later analyzed
- whether lowering preserved the model's forward computation
- whether the region describes raw inputs or already transformed inputs
- whether the checker interpreted each graph operation with the intended scalar arithmetic
- whether acceptance implies the property written in the theorem.

## Stale Parameter Payloads

The first connection can fail when parameters change after verification. Take the network whose
real-valued range we proved and change one weight, as an optimizer step might: the second layer's
$`W_2=(0.7,0.8,0.9)` becomes $`(0.7,0.8,0.95)`. Its graph structure and parameter shapes stay the
same.

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

At the upper corner, the third hidden activation is 1.14, so increasing its readout coefficient by
0.05 raises the prediction from 2.256 to 2.313. That overshoots the proved bound by $`0.057` (about
a third of the box's radius). {lean}`moEnclosure` remains true for the weights it names, but
applying it to the changed payload fails, because that payload describes a different function. A
verification pipeline must track the parameter values it analyzed through to deployment, even when
the architecture is unchanged.

That's why {name}`Softmax.hardMaskedVec` and its relatives take their parameters as
arguments, and why `Graph.eval` in {ref "graphs-and-ir"}[the IR chapter] takes a payload (it
doesn't read weights from anywhere ambient). A soundness theorem whose statement mentions $`\theta`
forces the caller to say which $`\theta`, and the two-stage workflows in {ref "twostage"}[that
chapter] carry the same payload through search and checking for that reason.

Let $`g` be the graph, $`\theta` its parameter payload, $`B` the input region, and $`c` a
certificate supplied to the checker. A checker soundness theorem usually has the form

$$`\operatorname{check}(g,\theta,B,c)=\mathrm{true}
  \Longrightarrow
  \operatorname{Property}(\operatorname{denote}(g,\theta),B)`.

The certificate may come from an expensive external search, while the checker validates the
resulting artifact. Branch-and-bound verifiers make the split concrete, exploring a tree of neuron
splits and emitting a bound per leaf {Informal.citep bunel2020}[]. In {ref "certificates"}[the
certificates chapter] we inspect the imported leaf boxes and their claimed bounds. The external
leaf-artifact checker checks box coverage and threshold witnesses without repeating the search, but
it doesn't recompute the claimed bounds. That chapter treats this consistency check separately from
theorem-backed bound checking.

# Executable Verification Workflows

The command

```terminal
# Discover the workflow names before choosing the checker
# and its artifact.
scripts/lake.sh exe verify -- list
```

lists the executable verification workflows in the current checkout. Selected entries include:

```
  ibp-cert [<path>]   -- IBP certificate check for a 3 → 4 → 2 ReLU MLP
  pinn-cert [<path>]   -- PINN certificate recomputation check
  torchlean-ibp   -- TorchLean → IR → IBP workflow (MLP)
  torchlean-crown-ops   -- TorchLean → IR → IBP+CROWN workflow (softmax/mse_loss ops)
  ode   -- ODE enclosure verification (sub/super NN bounds)
  vnncomp-mnistfc   -- VNN-COMP-style suite: MNIST-FC (vnncomp2022) via exported JSON
  twostage-torchlean-cegis-van   -- all-in-Lean two-stage/CEGIS Lyapunov refinement and bound check
```

Each tool's default artifact path appears in the full output as well.

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

The two integer encodings differ by one, even though decimal formatting hides the distinction. Each
parenthesization chooses a different intermediate sum to round, so real associativity can't justify
replacing one execution order by the other without an additional numerical argument. Both sides
print as `0.600000`, but a comparison against a threshold can expose the difference. Fused
multiply-add, reduction order, subnormal handling, NaNs, and overflow introduce further
distinctions, and {ref "floating-point-literature"}[the literature chapter] traces how Flocq and
its relatives handle the same questions in Rocq {Informal.citep flocq2011}[].

Which arithmetic the statement names matters:

- exact real-valued specifications
- configurable rounded-real arithmetic with checked positive format precision
- `FP32`, a rounded-real model with binary32 precision and gradual-underflow parameters but no
  upper exponent bound or IEEE special values
- executable bit-level IEEE binary32
- native `Float32`, host `Float`, CUDA, and external runtime providers.

A real-valued enclosure theorem can't be silently relabeled as a theorem about every GPU execution,
so a bridge theorem or a stated backend contract must carry the result across that boundary. We
develop these relationships in detail in the {ref "floats"}[floating-point chapter].

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

That shift keeps every exponential argument nonpositive. With $`\beta=1000` and inputs $`\{0,1\}`,
evaluating the unshifted sum of exponentials overflows:

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

For these inputs the shifted exponentials are `exp(-1000)` and `exp(0)`. The latter supplies one to
the sum, while the former is too small to affect this Float result. In real arithmetic the smooth
maximum is slightly above one, by `log(1 + exp(-1000)) / 1000` (the printed one is the rounded
evaluation). This is a useful example of separating two goals: avoid an infinite intermediate, and
describe the error of the finite answer that remains. The algebraic rewrite addresses the first,
but the second still needs a numerical error statement.

TorchLean uses the same shifted weights in the forward and derivative paths, except that the
forward shifts by zero when the extreme input is infinite, where $`x_i - p` would be NaN (the
derivative weights are then NaN), and executable entry points reject zero or non-finite $`\beta`.
The real identity explains the transformation, while the specification over $`\mathbb{R}` fixes the
intended function. Validation and backend tests check its executable form in the selected format.

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

Adversarial examples and the quantifier problem come from {Informal.citet szegedy2014}[]. Bound
propagation as used above, and its use during training, is developed in
{Informal.citet gowal2018}[]. The linear-relaxation family that tightens it is
{Informal.citet crown2018}[], {Informal.citet autolirpa2020}[], and
{Informal.citet betacrown2021}[], with the branch-and-bound search in {Informal.citet bunel2020}[].
The floating-point gap between a verifier's arithmetic and the network's is the subject of
{Informal.citet jiarinard2020}[], and the Rocq formalization we compare against throughout the
floating-point chapters is {Informal.citet flocq2011}[].
