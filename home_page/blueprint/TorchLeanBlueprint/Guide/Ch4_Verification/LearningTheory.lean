import VersoManual
import FloatLib
import NN.MLTheory.LearningTheory.DifferentialPrivacy
import NN.MLTheory.LearningTheory.Robustness
import NN.MLTheory.LearningTheory.Stability
import NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.IEEE32Exec
import NN.MLTheory.Proofs.Verification.Robustness.LipschitzCertified
import NN.MLTheory.Proofs.Verification.Robustness.MlpRobustness
import NN.Tensor
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Floats (ExecFloat)

-- Learning theory is the most scattered layer in the repository: privacy lives under
-- `NN.MLTheory.LearningTheory`, the robustness vocabulary under `NN.MLTheory.Robustness`, the
-- dynamics diagnostics under `NN.MLTheory.Stability.Runtime`, and the proofs that connect them
-- under `NN.MLTheory.Proofs.Verification.Robustness`. Opening all of them here is what lets the
-- displayed signatures below read the way a caller would type them, inside Verso's narrow column.
open TorchLean
open Spec
open NN.MLTheory.LearningTheory
open NN.MLTheory.LearningTheory.Stability
open NN.MLTheory.LearningTheory.Stability.RidgeRegression1D
open NN.MLTheory.Robustness.Spec
open NN.MLTheory.Robustness.Runtime
open NN.MLTheory.Robustness.Runtime.Empirical
open NN.MLTheory.Stability.Runtime
open NN.MLTheory.Proofs.Verification.Robustness
open MeasureTheory

-- Several signatures below are wider than this file's 100-column limit, so their `leanOutput`
-- blocks ask for `whitespace := lax` and are wrapped in the source. The rendered page still shows
-- each message exactly as Lean printed it.
open Lean.Elab.Tactic.GuardMsgs.WhitespaceMode (lax)

#doc (Manual) "Learning Theory" =>
%%%
tag := "learning-theory"
%%%

A training script can compute a loss curve, run an attack, add noise to an update, or fit a ridge
regression model. To turn those computations into guarantees, we need to specify what is
quantified: all neighboring datasets for privacy, all allowed perturbations for robustness, or
all replacement examples for algorithmic stability.

TorchLean defines these predicates alongside executable diagnostics. To use a diagnostic as
evidence for a theorem, we need to show how its result addresses the theorem's quantifiers.
A failed attack, for example, leaves open what happens at perturbations the attack never tried.
Named Lean blocks below are checked during the page build; Python and CLI transcripts record
separate experiments.

# Learning Theory Claims

The learning theory material is organized around five concrete objects:

- *Randomized mechanism*: a map from inputs to probability measures over outputs; Lean states
  $`(\varepsilon,\delta)` privacy, pure privacy, monotonicity in $`\delta` and $`\varepsilon`, and
  post processing.
- *Robustness predicate*: a classifier is stable on a perturbation ball; Lean names tensor norms,
  local balls, Lipschitz predicates, and certified robustness, and proves the implications between
  them.
- *Algorithmic stability*: replacing one training example changes the learned loss only slightly;
  Lean gives typed datasets, `replaceAt`, `removeAt`, learning maps, and loss change bounds.
- *Dynamical stability*: trajectories stay bounded or converge under stated hypotheses; Lean names
  Lyapunov, ISS, BIBO, incremental, practical, and finite time predicates.
- *Ridge regression case study*: a one-dimensional strongly regularized ERM theorem, with a real
  stability theorem plus a FloatLib binary32 execution bridge.

These objects do not share one proof method. Differential privacy is an inequality between
measures. Robustness quantifies over a neighborhood of an input. Algorithmic stability compares
two executions of a learning algorithm on neighboring datasets. Dynamical stability concerns an
entire trajectory. The ridge result is a concrete algebraic proof whose constants can be carried
into a numerical analysis. Their common feature is that the quantified object and its boundary are
stated before any runtime evidence is interpreted.

# Differential Privacy

The core privacy definitions live in
{src "NN/MLTheory/LearningTheory/DifferentialPrivacy/Core.lean"}[
NN.MLTheory.LearningTheory.DifferentialPrivacy.Core API]. Here is the central definition, printed
by Lean rather than paraphrased:

```lean (name := dpDef)
-- Expose the input relation, output measure, and two
-- privacy budgets.
#check @DifferentialPrivacy
```

```leanOutput dpDef (whitespace := lax)
@DifferentialPrivacy : {α β : Type} →
  (α → α → Prop) →
    [inst : MeasurableSpace β] →
      Mechanism α β → ℝ → ENNReal → Prop
```

Read the arguments left to right. `α` is the input type, typically a dataset. `β` is the output
type, typically a model or a released statistic. The relation `α → α → Prop` is the *adjacency*
relation: it says which pairs of inputs count as neighbors, and it is a parameter because "differ
in one record" and "differ in one person's whole history" are different privacy claims about the
same mechanism. `Mechanism α β` is an abbreviation for `α → ProbabilityMeasure β`, so a mechanism is
a randomized map. The last two arguments are the budget $`\varepsilon` and the slack $`\delta`.

The square brackets around `[MeasurableSpace β]` request a typeclass instance: the output type
must come with a collection of measurable events. The braces around `{α β : Type}` make those
arguments implicit in ordinary calls. Prefixing the name with `@` exposes them so we can see the
whole interface. The final `Prop` means that this declaration defines a proposition about a
mechanism. Merely constructing a value of type `Mechanism α β` does not supply a proof of that
proposition.

Adjacency also determines the direction of comparison. The definition does not require the
relation to be symmetric. If an application uses symmetric adjacency, it can apply the same
inequality with the datasets exchanged; otherwise that reverse comparison needs its own adjacency
premise. This matters when reading ratios in the example below.

The body is the standard event inequality: for every adjacent pair and every measurable event `S`,

$$`\forall D\sim D',\;\forall S,\qquad
\Pr[M(D)\in S]\le e^\varepsilon \Pr[M(D')\in S]+\delta.`

Probabilities and the slack use `ENNReal`, the extended nonnegative reals. Nonnegativity is built
into the type, and addition and multiplication are monotone. This lets the closure proofs work
directly with the event inequality. Subtraction in this type is truncated at zero, so the
additive form also avoids importing ordinary real-subtraction identities.

The output is a `ProbabilityMeasure`, not a density or a sampler. That covers discrete mechanisms
such as randomized response, continuous mechanisms such as Laplace or Gaussian noise, and whole
randomized training procedures, with one definition. The definition says what any privacy proof
must establish; it deliberately says nothing about how the randomness is produced.

## Constant Mechanisms

A mechanism that ignores its input has the same output distribution on every neighboring pair.
It therefore satisfies pure privacy with zero budget:

```lean
-- An input-independent distribution gives identical
-- probabilities for every event.
example {α β : Type} [MeasurableSpace β]
    (Adj : α → α → Prop) (μ : ProbabilityMeasure β) :
    PureDP (α := α) (β := β) Adj (fun _ => μ) 0 := by
  intro a a' _ S _
  simp
```

The proof is `simp` because after `intro` the goal is
$`\mu(S)\le e^0\cdot\mu(S)+0`, and $`e^0=1`. Both sides concern the same event under the same
measure. `PureDP` is the $`\delta=0` abbreviation.

Note which quantifier order made this easy. Privacy is universally quantified over adjacent pairs
*and* over measurable events, so `intro a a' _ S _` discharges all five binders and leaves an
arithmetic goal. The proof works for any probability measure on the output space.

## Monotonicity Of The Budget

Two closure facts get their own theorems. The first, `differentialPrivacy_mono_delta`, says a
mechanism private for $`\delta_1` is private for any looser $`\delta_2\ge\delta_1`. The second
allows the same relaxation of $`\varepsilon`:

```lean (name := dpMono)
-- Keep the mechanism fixed while increasing its allowed
-- epsilon budget.
#check @differentialPrivacy_mono_eps
```

```leanOutput dpMono (whitespace := lax)
@differentialPrivacy_mono_eps : ∀ {α β : Type}
  {Adj : α → α → Prop} [inst : MeasurableSpace β]
  {M : Mechanism α β} {ε₁ ε₂ : ℝ} {δ : ENNReal},
  ε₁ ≤ ε₂ →
    DifferentialPrivacy Adj M ε₁ δ →
      DifferentialPrivacy Adj M ε₂ δ
```

Together the two lemmas say the predicate is monotone in the whole budget
$`(\varepsilon,\delta)`. A caller may therefore use a looser budget than the one already proved.
For the epsilon lemma, monotonicity of $`\exp` increases the multiplier on the right-hand side;
multiplying a nonnegative event probability preserves that inequality. This is the comparison
handled by `gcongr`.

For example, a proof at $`(0.5,0)` can be reused at $`(1,0)` by supplying the real inequality
$`0.5\le1`. The theorem keeps the mechanism and adjacency fixed. It does not describe the budget
of two successive releases: comparing two numbers on the right-hand side of one event inequality
is different from analyzing a joint output distribution. In Lean, the two arrows in the signature
make the use explicit: first give the budget comparison, then the existing privacy proof, and the
result is the relaxed privacy proof.

## Postprocessing

```lean (name := dpPost)
-- The measurable output map is the only new premise for
-- postprocessing.
#check @differentialPrivacy_postprocess
```

```leanOutput dpPost (whitespace := lax)
@differentialPrivacy_postprocess : ∀ {α β γ : Type}
  {Adj : α → α → Prop} [inst : MeasurableSpace β]
  [inst_1 : MeasurableSpace γ] {M : Mechanism α β}
  {ε : ℝ} {δ : ENNReal} {f : β → γ}
  (hf : Measurable f),
  DifferentialPrivacy Adj M ε δ →
    DifferentialPrivacy Adj (postprocess M f hf) ε δ
```

Here `f` is a fixed measurable function of the released output. It may, for example, convert a
private model into an exported representation or a report. Applying it preserves the budget,
provided it uses only the released output and fixed public information. An evaluator that reads
the private dataset again is outside this theorem.

The proof is short for a mathematical reason. For a measurable output event `T`, the post-processed
mechanism lands in `T` exactly when the original mechanism lands in the measurable preimage
$`f^{-1}(T)`. Applying the DP inequality to that preimage gives

$$`\begin{aligned}
\Pr[f(M(D))\in T]
&=\Pr[M(D)\in f^{-1}(T)]\\
&\le e^\varepsilon\Pr[M(D')\in f^{-1}(T)]+\delta\\
&=e^\varepsilon\Pr[f(M(D'))\in T]+\delta.
\end{aligned}`

A useful model example is a private vector of weights followed by a fixed prediction rule at a
public input. Here `β` is the weight space, `γ` is the prediction space, and `f` performs inference
at that input. An event such as “the prediction is class 2” becomes the set of weights for which
inference returns class 2. Privacy of the weights already bounds the probability of that set.
This explains why the proof uses preimages: the two spaces can have quite different
representations, but the probability comparison is still about the original release.

The measurability hypothesis `hf` licenses the preimage step. Together with the original privacy
premise, it is sufficient; the proof does not inspect the representation of the model or report.
The reference point
is the event inequality of {Informal.citet dwork2006}[].

## Sampling And Privacy Guarantees

Sampling alone cannot establish the universally quantified privacy predicate. The same limitation
affects robustness and asymptotic stability: finite experiments test cases, not all quantified
inputs.

Take the Laplace mechanism on a counting query. Two adjacent datasets differ in one record, so the
count differs by at most the sensitivity $`\Delta=1`. Releasing the count plus
$`\mathrm{Laplace}(0,\Delta/\varepsilon)` noise is $`(\varepsilon,0)`-DP. With $`\varepsilon=0.5`
the noise scale is $`2`. Take the true counts $`10` and $`11`, and the tail event
$`S=\{\text{output}\ge 12\}`, which is the direction that favors the larger count and is therefore
where the inequality is tightest. Two million samples per dataset in PyTorch
{Informal.citep pytorch2019}[]:

```
# Compare one Laplace tail event in both directions of
# dataset adjacency.
import torch, math
torch.manual_seed(0)
lap = torch.distributions.Laplace(torch.tensor(0.), torch.tensor(2.))
zD  = 10. + lap.sample((2_000_000,))
zDp = 11. + lap.sample((2_000_000,))
pD  = (zD  >= 12.).float().mean().item()
pDp = (zDp >= 12.).float().mean().item()
print(f"P[M(D)>=12]  emp {pD:.5f}  exact {0.5 * math.exp(-1):.5f}")
print(f"P[M(D')>=12] emp {pDp:.5f}  exact {0.5 * math.exp(-0.5):.5f}")
print(f"ratio D/D'   {pD / pDp:.5f}")
print(f"ratio D'/D   {pDp / pD:.5f}")
print(f"e^eps        {math.exp(0.5):.5f}")
```

```
P[M(D)>=12]  emp 0.18385  exact 0.18394
P[M(D')>=12] emp 0.30330  exact 0.30327
ratio D/D'   0.60618
ratio D'/D   1.64968
e^eps        1.64872
```

The exact tail probabilities are $`\tfrac12 e^{-1}` and $`\tfrac12 e^{-1/2}`, and their ratio is
exactly $`e^{1/2}`, so this event attains the privacy bound in the direction from the larger count
to the smaller one.

Now look at the last three lines again. The empirical ratio is $`1.64968` and the bound is
$`e^{0.5}=1.64872`. The estimate violates the inequality it is supposed to illustrate, by about six
parts in ten thousand, because each probability was estimated from a finite sample. More samples
can improve estimates for this fixed event, but cannot establish the inequality for every
measurable event and every neighboring pair. Rare events are especially difficult to estimate:
few or no sampled outputs may land in them. The analytical tail calculation explains this
experiment; a privacy proof must cover the full quantification.

## Privacy Verification Gaps

No privacy budget should be inferred from the presence of the DP namespace. The current source
supplies the semantic target and the closure laws. It does not contain a runtime privacy accountant,
a calibration theorem for the Laplace or Gaussian mechanism, a composition theorem, or an
end-to-end theorem for a training command. Concretely, the gap between this file and a DP-SGD claim
is: Poisson or uniform subsampling with its amplification argument, per-example gradient clipping
with a sensitivity bound, noise calibration to that sensitivity, and composition accounting across
steps. Each of those is a separate implementation and proof obligation. What the file does give you
is the target they must hit, plus the measurable, data-independent post-processing theorem.

# Robustness Specifications And Runtime Checks

Robustness is split by trust boundary into two files:

- {src "NN/MLTheory/LearningTheory/Robustness/Spec.lean"}[Robustness.Spec] defines the mathematical
  predicates, scalar-polymorphically;
- {src "NN/MLTheory/LearningTheory/Robustness/Runtime.lean"}[Robustness.Runtime] defines executable
  `Float` diagnostics.

The spec side works over TorchLean tensors without committing to one runtime scalar. It defines
`tensorLinfNorm` and `tensorL2Norm`, `tensorDistance`, closed tensor balls, global and local
Lipschitz continuity, adversarial robustness at a point, certified robustness for classifiers,
uniform robustness over a finite dataset, contraction mappings, and local sensitivity ratios. A
typical local robustness predicate has the form

$$`\operatorname{Robust}(f,x,y,\varepsilon)
\Longleftrightarrow
\forall x',\;\|x'-x\|_\infty\le \varepsilon
\Rightarrow
\operatorname*{argmax} f(x')=y,`

This states a perturbation model of the kind studied by {Informal.citet szegedy2014}[]: every
input in the allowed ball must retain the specified label.

The central robustness predicates take input and output norms as separate arguments. This records
the metric used by each step of an argument. A Lipschitz bound proved in $`\ell^2` can feed a
margin argument using $`\ell^\infty` after a norm-comparison theorem connects them. The
certificate below keeps `normIn` general while fixing the output norm needed to control logits.

## Tensor Norms And Perturbation Balls

Everything on the runtime side is ordinary executable code. Here is a two by two linear map, its
output, and the two norms:

```lean
-- Fix a linear map and two directions with different
-- Euclidean gains.
def lipW : Tensor Float [2, 2] :=
  [[3, 1],
   [0, 2]]

def lipLayer : LinearSpec Float 2 2 :=
  { weights := lipW, bias := [0, 0] }

def lipMap (x : Tensor Float [2]) : Tensor Float [2] :=
  linearSpec lipLayer x

def lipX : Tensor Float [2] := [1, 1]
def lipY : Tensor Float [2] := [0, 1]

-- The top right singular direction of `lipW`, to
-- seven digits. It is where the L2 gain is largest.
def lipTop : Tensor Float [2] :=
  [0.8816746, 0.4718579]

def lipZero : Tensor Float [2] := [0, 0]
```

```lean (name := robNorms)
-- Evaluate the map, then compare output norms and the
-- proposed input ball.
#eval lipMap lipX
#eval tensorLinfNormFloat (lipMap lipX)
#eval tensorL2NormFloat (lipMap lipX)
#eval tensorL2DistanceFloat lipX lipY
#eval inLinfBallFloat lipX 0.5 lipY
```

```leanOutput robNorms
[4.000000, 2.000000]
```

```leanOutput robNorms
4.000000
```

```leanOutput robNorms
4.472136
```

```leanOutput robNorms
1.000000
```

```leanOutput robNorms
false
```

The first output is $`W(1,1)^\top=(3+1,0+2)^\top=(4,2)^\top`, whose $`\ell^\infty` norm is the
larger absolute coordinate $`4` and whose $`\ell^2` norm is $`\sqrt{20}\approx4.4721360`. The inputs
$`(1,1)` and
$`(0,1)` differ only in the first coordinate, so their distance is $`1` in every $`\ell^p`. And
$`(0,1)` is not in the $`\ell^\infty` ball of radius $`0.5` around $`(1,1)`, because $`1>0.5`, so
the ball test returns `false`.

The ball test checks whether this pair satisfies the proposed perturbation constraint. Its norm
and radius must match the ones used by a robustness theorem.

The corresponding PyTorch calculation prints the same values:

```
# Compute the same linear output and norms with PyTorch
# tensors.
W = torch.tensor([[3., 1.], [0., 2.]])
x = torch.tensor([1., 1.])
print("f(x)      ", (W @ x).tolist())
print(f"linf f(x)  {torch.linalg.vector_norm(W @ x, float('inf')).item():.6f}")
print(f"l2   f(x)  {torch.linalg.vector_norm(W @ x).item():.6f}")
```

```
f(x)       [4.0, 2.0]
linf f(x)  4.000000
l2   f(x)  4.472136
```

The choice of ball has a direct geometric effect. Around $`(1,1)`, the infinity-norm ball of
radius $`0.5` allows each coordinate to move independently by at most $`0.5`. Its corner
$`(1.5,1.5)` is at Euclidean distance $`\sqrt{0.5}` from the center, so it is outside the
Euclidean ball with the same numeric radius. A radius therefore needs its norm to be meaningful.
For a model whose input is normalized before inference, the tensor in the theorem is the input
at that stage; a radius in normalized coordinates requires a conversion before it describes a
perturbation of raw pixels or physical measurements.

## Empirical Lipschitz Ratios

The runtime layer also provides empirical helpers: Lipschitz ratios over a finite set of input
pairs, and deterministic perturbation sampling. Changing the sampled direction changes the
observed gain, even for this two-dimensional linear map:

```lean (name := robRatio)
-- Each singleton array measures one direction, rather than
-- all displacements.
#eval maxL2LipschitzRatio lipMap #[(lipX, lipY)]
#eval maxL2LipschitzRatio lipMap #[(lipTop, lipZero)]
```

```leanOutput robRatio
Except.ok 3.000000
```

```leanOutput robRatio
Except.ok 3.256617
```

The first pair differs in the first coordinate, so it measures the gain on $`(1,0)^\top`, which is
$`\|(3,0)^\top\|_2=3`. The second pair approximates the top singular direction. The spectral norm,
with its decimal expansion rounded, is $`\sigma_{\max}(W)=\sqrt{7+\sqrt{13}}\approx3.2566165`.
In exact arithmetic, ratios at distinct points lower-bound the least Lipschitz constant.
Here the direction and arithmetic are rounded, so the diagnostic is an approximation, not a proved
lower bound. The second direction closely approximates the maximizing singular vector.

```
# Compare the sharp spectral constant with the Frobenius
# upper bound.
print(f"spectral norm {torch.linalg.matrix_norm(W, 2).item():.6f}")
print(f"frobenius     {torch.linalg.matrix_norm(W, 'fro').item():.6f}")
```

```
spectral norm 3.256617
frobenius     3.741657
```

For the first sampled direction, the exact ratio, spectral norm, and Frobenius norm have
the following order:

$$`\underbrace{3}_{\text{sampled pair}}
\;\le\;
\underbrace{\sqrt{7+\sqrt{13}}}_{\sigma_{\max}(W),\ \text{least Lipschitz constant}}
\;\le\;
\underbrace{\sqrt{14}}_{\|W\|_F,\ \text{Frobenius bound}}.`

The middle number is what a robustness argument would like to use. The right number is what
`linear_layer_lipschitz_bound` in
{src "NN/MLTheory/Proofs/Verification/Robustness/MlpRobustness.lean"}[MlpRobustness]
proves, by bounding the operator norm with the Frobenius norm. The squared Frobenius norm is the
sum of squared entries, so a Cauchy-Schwarz argument gives the bound without computing singular
values. The resulting bound is loose by about $`15\%` on this matrix and can be loose by a
factor of $`\sqrt{\min(m,n)}` in general. If a downstream certificate ever needs the sharper
constant, that is a new theorem, not a new diagnostic.

The wrapper `Except.ok` says that the ratio computation returned a result through its success
branch. The number inside is still the ratio for the supplied pairs. To use the Frobenius upper
bound in this example, one instead reasons about an arbitrary displacement $`v`: each output
coordinate is a row dot product, Cauchy–Schwarz bounds its square, and summing the row bounds gives
$`\|Wv\|_2^2\le\|W\|_F^2\|v\|_2^2`. The bias disappears when subtracting two outputs.
This proof covers directions absent from the sample array, which is precisely the information a
robustness certificate needs.

There is a related helper for one-directional sensitivity:

```lean (name := robSens)
-- Perturb only the first input coordinate when measuring
-- sensitivity.
#eval sensitivity (α := Float) lipMap tensorL2NormFloat
  tensorL2NormFloat lipX ([0.001, 0] : Tensor Float [2])
```

```leanOutput robSens
3.000000
```

This computes the finite-difference ratio
$`\|f(x+p)-f(x)\|_2/\|p\|_2`. Over the reals, for a linear map and a fixed nonzero direction, the
ratio is independent of step size. For a nonlinear network it may also vary with the point and
step size. In either case, floating subtraction of nearby outputs can lose precision.

## Lipschitz Bounds And Margin Certificates

The implications needed to turn a Lipschitz bound into a robustness certificate are proved in
{src "NN/MLTheory/Proofs/Verification/Robustness/LipschitzCertified.lean"}[LipschitzCertified]. The
first step turns a Lipschitz bound into an output-perturbation bound:

```lean (name := lipThm)
-- Read the output radius as the input radius multiplied by
-- the Lipschitz bound.
#check @is_adversarially_robust_of_lipschitz
```

```leanOutput lipThm (whitespace := lax)
@is_adversarially_robust_of_lipschitz : ∀ {s₁ s₂ : Shape}
  {f : Tensor ℝ s₁ → Tensor ℝ s₂}
  {norm₁ norm₂ : {s : Shape} → Tensor ℝ s → ℝ} {L : ℝ},
  0 ≤ L →
    isLipschitzContinuous f (fun {s} => norm₁)
        (fun {s} => norm₂) L →
      ∀ (x₀ : Tensor ℝ s₁) (ε : ℝ),
        IsAdversariallyRobust f (fun {s} => norm₁)
          (fun {s} => norm₂) x₀ ε (L * ε)
```

Read the conclusion: the output radius is $`L\varepsilon`, with no hypothesis on $`\varepsilon`
beyond what `isLipschitzContinuous` already gives. For an input satisfying the ball premise,
multiply its distance bound by the nonnegative Lipschitz constant. This bounds output drift;
preserving a label also requires a margin between logits.

The second step is about labels, and it needs a quantity the drift bound does not mention: how far
ahead the winning logit is.

```lean (name := marginDef)
-- A margin fixes one class and compares its logit with
-- every competitor.
#check @HasLogitMargin
```

```leanOutput marginDef
@HasLogitMargin : {n : ℕ} → Tensor ℝ [n] → Fin n → ℝ → Prop
```

`HasLogitMargin y c m` says class `c` beats every competitor by at least `m`. Combining the two
gives the classical margin-over-Lipschitz certificate:

```lean (name := certThm)
-- The final strict inequality keeps every competing logit
-- below the winner.
#check @is_certified_robust_of_lipschitz_of_logitMargin
```

```leanOutput certThm (whitespace := lax)
@is_certified_robust_of_lipschitz_of_logitMargin :
  ∀ {s₁ : Shape} {n : ℕ} {f : Tensor ℝ s₁ → Tensor ℝ [n]}
  {normIn : {s : Shape} → Tensor ℝ s → ℝ} {L : ℝ},
  0 ≤ L →
    isLipschitzContinuous f (fun {s} => normIn)
        (fun {s} => tensorLinfNorm) L →
      ∀ {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n},
        0 ≤ ε →
          0 < m →
            HasLogitMargin (f x₀) c m →
              2 * (L * ε) < m →
                IsCertifiedRobust
                  (fun x => argmaxClassifier (f x))
                  (fun {s} => normIn) x₀ ε
```

The side condition is $`2L\varepsilon<m`. The winning logit can
fall by $`L\varepsilon` while a competitor rises by $`L\varepsilon`, so the gap can close at twice
the drift rate. For $`L>0`, this becomes $`\varepsilon<m/(2L)`, which is why a loose
Lipschitz constant costs radius linearly, and why the $`15\%` gap between $`\|W\|_F` and
$`\sigma_{\max}(W)` above reduces the certified radius.

Note also that the output norm in this theorem is pinned to $`\ell^\infty` while the input norm
`normIn` stays free. That asymmetry is forced by the proof: the margin argument compares individual
logits, and the norm that controls individual coordinates is the sup norm. The companion lemma
`is_lipschitz_continuous_linf_of_l2` in the same file is what lets an $`\ell^2` bound be fed into
this slot.

The strict inequality in the margin condition removes ties. If the gap could shrink exactly to
zero, the classifier's tie-breaking convention would become relevant. With a positive remaining
gap, every competitor is strictly below `c`, so the same class is selected regardless of that
convention. The conclusion `IsCertifiedRobust` compares the prediction throughout the ball with
the prediction at its center. It does not compare either prediction with an external ground-truth
label. To state robust accuracy, a caller also supplies the fact that the center's predicted class
is the intended label.

## Logit-Margin Certificate Example

For a concrete certificate, start at logits $`(2,0)` with margin $`2`
and the identity map as the classifier's tail, so $`L=1`:

```lean
-- Discharge the margin and radius conditions for the
-- two-logit identity map.
def marginLogits : Tensor ℝ [2] := [2, 0]

example :
    HasLogitMargin marginLogits ⟨0, by decide⟩ 2 := by
  intro k hk
  fin_cases k
  · exact absurd rfl hk
  · norm_num [marginLogits]

example : IsCertifiedRobust (α := ℝ)
    (classifier := fun y : Tensor ℝ [2] =>
      argmaxClassifier (n := 2) y)
    (norm := tensorLinfNorm (α := ℝ))
    marginLogits 0.4 := by
  have hLip :
      isLipschitzContinuous (α := ℝ)
        (fun y : Tensor ℝ [2] => y)
        (tensorLinfNorm (α := ℝ))
        (tensorLinfNorm (α := ℝ)) 1 := by
    intro x y
    simp
  have hMargin :
      HasLogitMargin marginLogits ⟨0, by decide⟩ 2 := by
    intro k hk
    fin_cases k
    · exact absurd rfl hk
    · norm_num [marginLogits]
  exact
    is_certified_robust_of_lipschitz_of_logitMargin
      (L := 1) zero_le_one hLip (by norm_num)
      (by norm_num) hMargin (by norm_num)
```

The margin proof is two cases because `HasLogitMargin` quantifies over competitors: `fin_cases k`
splits on the class index, the winning class is excluded by the hypothesis `k ≠ c`, and the
competitor obligation is $`0\le 2-2`, which `norm_num` closes after unfolding the literal. That
unfolding works because `getScalar_ofList` is a simp lemma, so a rank-one tensor literal reduces to
list lookup without any manual `Rep` reasoning.

The three `by norm_num` arguments at the end are the numeric side conditions:
$`0\le 0.4`, $`0<2`, and $`2\cdot(1\cdot 0.4)=0.8<2`. The last one is the certificate. It is also
where the radius stops: at $`\varepsilon=1.2` the same call fails, because
$`2\cdot(1\cdot 1.2)=2.4` is not below the margin $`2`. A failed sufficient condition alone does
not establish the existence of an adversarial example at radius $`1.2`. For this identity map,
we can supply a separate witness: $`(0.8,1.2)` is within radius
$`1.2` of $`(2,0)` and class 1 wins. That witness supplies the information that failure of a
sufficient condition alone would not.

Within radius `0.4` in the infinity norm, each logit can move by at most `0.4`. The worst change
for the predicted class lowers its logit and raises its competitor, reducing the gap from `2`
to at least `1.2`. This calculation explains the factor of two in the general condition. For
an earlier layer of a model, the Lipschitz bound first converts an input perturbation radius
into this bound on logit movement.

## MLP Lipschitz Bounds

The same machinery composes to a network, in
{src "NN/MLTheory/Proofs/Verification/Robustness/MlpRobustness.lean"}[MlpRobustness]:

```lean (name := mlpThm)
-- The conclusion supplies a positive constant together with
-- its Lipschitz proof.
#check @NN.MLTheory.Proofs.mlp_is_lipschitz_continuous_l2
```

```leanOutput mlpThm (whitespace := lax)
@NN.MLTheory.Proofs.mlp_is_lipschitz_continuous_l2 :
  ∀ {inDim hidDim outDim : ℕ}
  (l1 : LinearSpec ℝ inDim hidDim)
  (l2 : LinearSpec ℝ hidDim outDim),
  l1.weights ≠ Tensor.full [hidDim, inDim] 0 →
    l2.weights ≠ Tensor.full [outDim, hidDim] 0 →
      ∃ L > 0,
        isLipschitzContinuous
            (fun x => Examples.mlpForward l1 l2 x)
            (fun {s} => Proofs.tensorL2Norm)
          (fun {s} => Proofs.tensorL2Norm) L
```

The conclusion is existential, so using it in a certificate means obtaining a witness `L`
and its Lipschitz proof. The witness is the product of the two Frobenius norms, and
`mlp_lipschitz_frobenius` in the same file states the inequality with that product written out
and without the nonzero-weight hypotheses. Those hypotheses are there only because the existential
form promises a strictly positive constant. The margin lemma
itself only requires $`L\ge0`;
its separate $`m>0` condition does not force $`L>0`. The ReLU in the middle is handled by
`relu_activation_lipschitz`, which is the sharp
statement for ReLU: it is exactly 1-Lipschitz in $`\ell^2`, so the activation contributes no
looseness at all. Slack can come from bounding each linear layer and from multiplying layerwise
constants whose
maximizing directions may not align.

The norm slot here is `Proofs.tensorL2Norm`, the Mathlib-backed real norm from
{src "NN/Proofs/Analysis/Lipschitz/Norm.lean"}[Proofs.Analysis.Lipschitz.Norm], not the
`Context`-dictionary `tensorL2Norm` from the robustness spec. They agree at $`\mathbb R`, and the
predicate takes the norm as a parameter precisely so that a proof may use whichever formulation its
lemmas are stated for. Applying a result across the two formulations requires that norm identity.

To use the existential MLP result in a proof, one can unpack its conclusion into a real number,
a proof that the number is positive, and a proof of the Lipschitz inequality. Those are three
different pieces of information. The inequality can then be applied to any two inputs of shape
`[inDim]`; the hidden dimension constrains the two layer types but disappears from the final
input/output predicate. If a deployment needs a numerical radius, the proof must additionally
relate the chosen witness to a concrete bound that can be evaluated or certified, which is what
the explicit `mlp_lipschitz_frobenius` form provides. Unpacking an existential alone does not
print a decimal value for it.

## Runtime Margin Reports

The bundled logit-bound report makes the spec-versus-evidence difference visible:

```terminal
# Read the counts obtained from the bundled logit-bound
# report.
lake exe verify -- margin-report
```

```terminal +output
[margin report] examples=360
[margin report] nominal_ok=349
[margin report] positive_margin=318
```

`nominal_ok` counts correctly predicted recorded examples. `positive_margin` counts entries whose
stored bounds imply the required margin. This command checks the report's arithmetic; it does not
prove that the stored bounds enclose the model. That stronger claim depends on the verifier that
produced the bounds and on a theorem connecting its output to model semantics
{Informal.citep wongkolter2018}[]. Likewise, the ordering
$`3.000000\le 3.256617\le 3.741657` above shows why an observed ratio alone cannot supply the
upper bound required by a Lipschitz certificate.

# Algorithmic Stability

The {src "NN/MLTheory/LearningTheory/Stability/Core.lean"}[algorithmic stability API] makes one
central representation choice: a dataset of size `n` is a `TorchLean.Tensor Z [n]`, so the sample
size is part of the type rather than a list length that has to be remembered separately. The
definitions on top of it are coordinate access, `replaceAt` and `removeAt`, deterministic learning
maps `Dataset n Z → H`, real-valued losses, empirical error, i.i.d. sampling of datasets, and
uniform replace-one stability (`UniformStableReplace`).

For a dataset containing three examples, replacement preserves the size and removal reduces it:

```lean
-- Replacement keeps three entries; removal changes the
-- dataset type to size two.
def dataset3 : Dataset 3 Nat :=
  Dataset.ofFn (fun i => i.val + 10)

def dataset3Replaced : Dataset 3 Nat :=
  replaceAt dataset3 ⟨1, by decide⟩ 99

def dataset3Dropped : Dataset 2 Nat :=
  removeAt (n := 2) dataset3 ⟨1, by decide⟩
```

```lean (name := dsRun)
-- Display the original data beside its replacement and
-- removal variants.
#eval dataset3
#eval dataset3Replaced
#eval dataset3Dropped
```

```leanOutput dsRun
[10, 11, 12]
```

```leanOutput dsRun
[10, 99, 12]
```

```leanOutput dsRun
[10, 12]
```

Replace-one keeps the size and changes coordinate `1`. Remove-one drops that coordinate and lands in
a smaller type, `Dataset 2`, which is why the leave-one-out definitions in the file are stated at
`n + 1` and produce results at `n`.

The replacement index has type `Fin 3`, so it must carry a proof that it is less than three.
An out-of-range index fails during elaboration:

```lean +error (name := dsBad)
-- Index three is invalid because a three-element dataset
-- ends at index two.
def datasetBad : Dataset 3 Nat :=
  replaceAt dataset3 ⟨3, by decide⟩ 99
```

```leanOutput dsBad
Tactic `decide` proved that the proposition
  3 < 3
is false
```

The attempted coordinate is `3`, which would require the false inequality `3 < 3`. Lean rejects
the dataset operation before it can be used by a learner.

Coordinate lookup after removal is described by reindexing:

```lean
-- After removing index one, the new index one refers to the
-- old index two.
example (S : Dataset 3 Nat) :
    Dataset.get (removeAt (n := 2) S ⟨1, by decide⟩)
        ⟨1, by decide⟩
      = Dataset.get S ⟨2, by decide⟩ := by
  simp [Fin.succAbove]
```

Dropping coordinate `1` makes the old coordinate `2` the new coordinate `1`, and `Fin.succAbove` is
the reindexing that says so.

Empirical error is a plain average, and it computes symbolically:

```lean
-- The empirical loss includes the sample-size
-- normalization: (0 + 1 + 2) / 3.
example :
    empiricalError (n := 3) (H := ℝ) (Z := ℝ)
        (fun _ z => z) 0
        (Dataset.ofFn (fun i => (i.val : ℝ)))
      = 1 := by
  norm_num [empiricalError, Fin.sum_univ_three]
```

The loss here ignores the hypothesis and returns the example itself, the dataset is $`(0,1,2)`, and
the mean is $`1`. This checks the normalization: `empiricalError` carries
the explicit $`1/n`, so downstream constants do not silently absorb a factor of the sample size.

The stability predicate itself:

```lean (name := stabDef)
-- The proposition compares retrained losses at every common
-- test example.
#check @UniformStableReplace
```

```leanOutput stabDef (whitespace := lax)
@UniformStableReplace : {Z H : Type} →
  {n : ℕ} →
    [DecidableEq (Fin n)] →
      LearningMap n Z H → Stability.Loss H Z → ℝ → Prop
```

which unfolds to the standard uniform stability inequality

$$`\forall S,S^{(i)},z,\qquad
|\ell(A(S),z)-\ell(A(S^{(i)}),z)|\le \beta,`

quantified over all datasets, all replaced indices, and all test points. The classical reference is
{Informal.citet bousquet2002}[], and the TorchLean definitions follow its proof habit: first make
the dataset perturbation explicit, then state how much the learned loss can move.

The test point `z` and replacement example `z'` play different roles. The learner is retrained on
`replaceAt S i z'`, but both resulting hypotheses are evaluated on the same `z`. Allowing the
test point to change at the same time would mix sensitivity of the learner with variation of the
loss across examples. Uniformity means the bound holds for every such `z`, including one that
never appeared in training. No sampling distribution is needed for this pointwise definition.
The `DecidableEq (Fin n)` instance supports selecting the replaced coordinate; it is an
implementation requirement for replacement, not a statistical assumption.

# Ridge Regression Stability

The most concrete learning theory development is
{src "NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/Real.lean"}[
NN.MLTheory.LearningTheory.Stability.RidgeRegression1D.Real API]. It proves a replace-one uniform
stability bound for one-dimensional ridge regression with squared loss under bounded inputs. The
argument is the classical strongly convex ERM one, in a setting small enough to read end to end:

1. Each example is a bounded pair $`(x,y)` with $`\lvert x\rvert\le X` and $`\lvert y\rvert\le Y`,
   carried as a subtype so the bounds travel with the data.
2. The closed form fit is
   $`\hat w(S) = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda N}`.
3. Replacing one example changes the numerator by at most $`2XY` and the denominator by at most
   $`2X^2`.
4. Those two bounds combine into a bound on $`|\hat w(S)-\hat w(S')|`, using
   $`\sum_i x_i^2+\lambda N\ge\lambda N` to control the reciprocal.
5. A difference of squares argument converts the weight change bound into a loss change bound.

```lean (name := ridgeThm)
-- The bounded-example type carries the data restrictions
-- used in this constant.
#check @Ridge1D.ridgeFit1D_sqLoss_uniformStableReplace
```

```leanOutput ridgeThm (whitespace := lax)
@Ridge1D.ridgeFit1D_sqLoss_uniformStableReplace :
  ∀ {n : ℕ} {X Y lam : ℝ},
  0 < lam →
    UniformStableReplace (fun S => ridgeFit1D lam S)
      (fun w z => sqLoss w z)
      (4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 /
        (lam ^ 3 * Ridge1D.N))
```

Only `0 < lam` appears as an explicit mathematical premise in the printed theorem, but the data
restrictions have not disappeared. They are encoded by `BoundedExample X Y`, the type of each
training, replacement, and test example. A value of that subtype carries both coordinates and
proofs of their bounds. The dataset has size `n + 1`, so its nonemptiness is also part of the
type. These choices explain why the denominator can be proved positive: the sum of squares is
nonnegative and $`\lambda(n+1)>0`. With an arbitrary unbounded test point, the stated uniform
squared-loss constant would no longer be justified.

The denominator also specifies how regularization is normalized. The objective
$`N^{-1}\sum_i(wx_i-y_i)^2+\lambda w^2` has derivative zero at the displayed quotient: multiplying
the stationarity equation by `N` produces the term $`\lambda N`. If the data-fit objective were
an unnormalized sum with the same penalty coefficient, its denominator would instead contain
$`\lambda`. Thus the theorem's sample-size dependence belongs to this particular estimator;
one should match the training objective before substituting its regularization parameter.

Writing $`N=n+1`, the proved constant is

$$`\beta
=\frac{4X^2Y^2(\lambda+X^2)^2}{\lambda^3N}.`

The $`1/N` comes from the ridge denominator: adding $`\lambda N` to a sum of squares means one
example can move the fit only by $`O(1/N)`. The $`\lambda^{-3}` records the cost of controlling both
the fitted weight and the change in its reciprocal denominator. This is an explicit valid bound, not
a claim of optimality, and the difference matters. Instantiate it:

```lean
-- Substitute sixteen examples and unit bounds into the real
-- stability theorem.
example :
    UniformStableReplace (Z := BoundedExample 1 1) (H := ℝ)
      (A := fun S =>
        ridgeFit1D (n := 15) (X := 1) (Y := 1) 1 S)
      (ℓ := fun w z => sqLoss (X := 1) (Y := 1) w z)
      (β := 1) := by
  have h :=
    Ridge1D.ridgeFit1D_sqLoss_uniformStableReplace
      (n := 15) (X := 1) (Y := 1) (lam := 1) one_pos
  norm_num [Ridge1D.N] at h
  exact h
```

At $`X=Y=\lambda=1` and $`N=16`, the formula gives $`16/16=1`, and `norm_num` checks the
instantiation. This is below the generic loss-range bound of $`4` discussed next.

At $`N=2`, the same formula would give $`8`, a weak bound in that regime. The same file proves
$`|\hat w x-y|\le Y(\lambda+X^2)/\lambda`, which is $`2` here, so the squared loss never exceeds
$`4` and any two losses differ by at most $`4`. Since $`\beta=16/N` at these constants,
the theorem improves that loss-range bound once $`N>4`. The bound is valid at each sample size,
but its inverse dependence on sample size becomes useful in that regime. The displayed
$`N=16` instantiation demonstrates the improvement. A smaller general constant would require a
sharper proof.

The last step of the argument is worth spelling out. For two fits $`w,w'` evaluated at the same
$`(x,y)`, the loss difference factors as

$$`(wx-y)^2-(w'x-y)^2=(w-w')x\bigl((wx-y)+(w'x-y)\bigr).`

The fit-change estimate controls the first factor, the example subtype controls $`|x|`, and the
residual bound controls the last factor. This is how a statement about parameter movement becomes
a statement about prediction loss. In a larger model, a bound on parameter distance would need
an analogous argument for the network and loss being used. Small parameter movement by itself is
not yet the uniform stability predicate displayed above.

## Ridge Regression In Binary32

For ridge regression the ideal theorem lives over $`\mathbb R`, while the executable development is
in {src "NN/MLTheory/LearningTheory/Stability/RidgeRegression1D/IEEE32Exec.lean"}[
the binary32 ridge-regression implementation], which evaluates the algorithm with FloatLib. The
bundled example dataset is $`x=(1,3)`, $`y=(2,4)`, with $`\lambda=1` and $`N=2`:

```lean (name := ridgeExec)
-- Display both the decimal rendering and the binary32
-- representation of the fit.
#eval
  (ExecFloat.Binary.toFloat32
    IEEE32Exec.ExampleDataset.wHat).toFloat
#eval ExecFloat.Binary.toBits32
  IEEE32Exec.ExampleDataset.wHat
```

```leanOutput ridgeExec
1.166667
```

```leanOutput ridgeExec
1066751317
```

By hand: $`\sum x_iy_i=1\cdot2+3\cdot4=14`, $`\sum x_i^2=1+9=10`, the denominator is
$`10+1\cdot2=12`, and $`14/12=1.1\overline{6}`. PyTorch, in float32:

```
# Check the float32 quotient and inspect its underlying
# integer bit pattern.
xs = torch.tensor([1., 3.]); ys = torch.tensor([2., 4.])
w = (xs @ ys) / (xs @ xs + 1.0 * 2.0)
print(w.item(), w.view(torch.int32).item())
```

```
1.1666666269302368 1066751317
```

The bit patterns are identical: `1066751317` is `0x3F955555` on both sides. Every
intermediate here, $`14`, $`10`, $`12`, is exactly representable in binary32, so the only rounding
in the whole computation is the final division, and both implementations round it to nearest with
ties to even as IEEE 754 requires {Informal.citep goldberg1991}[]. A computation with inexact
intermediates would agree only if the two implementations also agreed on association and on
intermediate precision, which is exactly the kind of thing our executable model is meant to let us
check rather than assume {Informal.citep flocq2011}[].

What the executable side proves is narrower than the agreement suggests:

```lean (name := ridgeBridge)
-- This bridge uses a finite evaluation of the
-- expression-tree ridge estimator.
open IEEE32Exec.RidgeIEEEBridge in
#check @ridgeFit1D_execExpr_toReal_eq_fp32Spec_of_finiteEval
```

```leanOutput ridgeBridge (whitespace := lax)
@ridgeFit1D_execExpr_toReal_eq_fp32Spec_of_finiteEval : ∀ {n : ℕ}
  (lam :
    ExecFloat.Binary 8 23 FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.Encoding.ieee
      (FloatLib.Floats.Formats.BinaryInterchange.FloatFormat.Encoding.ieee.defaultBias 8)
      IEEE32Exec.ExampleIEEE32._proof_1 IEEE32Exec.ExampleIEEE32._proof_2
        IEEE32Exec.ExampleIEEE32._proof_3
      IEEE32Exec.ExampleIEEE32._proof_4)
  (S : Dataset (n + 1) IEEE32Exec.ExampleIEEE32) {d : FloatLib.Numerics.Dyadic},
  Floats.IEEE754.IEEE32Exec.FiniteEval (fun x => 0) (ridgeExpr lam S) d →
    (ridgeFit1DExecExpr lam S).toModel.toReal = ridgeFit1DFp32Spec lam S
```

This theorem concerns `ridgeFit1DExecExpr`, whose expression tree has its own association. The
example above executes the separately defined ordered-fold ridge estimator. No theorem in this
module equates those two binary32 evaluation paths for arbitrary data.

This is a finite-evaluation bridge: *if* the executable expression evaluates without overflowing to
infinity or producing a NaN, then its real interpretation agrees with the proof-level FP32
expression. It is not a stability theorem, and it is not an error bound either. What is still
missing to connect the two halves of this section is a bound on
$`|\hat w_{\mathrm{fp32}}(S)-\hat w_{\mathbb R}(S)|` in terms of the unit roundoff, plus a
loss-sensitivity bound that transfers fit error to the compared losses. Both are ordinary numerical
analysis; neither is in the file.

The `FiniteEval` argument in the bridge is evidence about the particular expression `ridgeExpr
lam S`. The variable `d` records its dyadic result, and the equality then compares two
interpretations of that same computation. It does not assert that the equality's right-hand side
is the exact real ridge quotient. A later stability transfer would have to control the numerical
error for both neighboring datasets, since both fits occur in the loss difference. Keeping track
of both executions is essential: bounding the error on the original training set alone leaves
the replacement execution uncontrolled.

# Dynamical Stability

The stability entrypoint also imports
{src "NN/MLTheory/LearningTheory/Stability/Dynamics.lean"}[
NN.MLTheory.LearningTheory.Stability.Dynamics API], which covers recurrences $`x_{t+1}=f(x_t)` and
input-driven systems. The spec file
{src "NN/MLTheory/LearningTheory/Stability/Dynamics/Spec.lean"}[Dynamics.Spec] names Lyapunov
stability, asymptotic stability, exponential stability, input-to-state stability, BIBO stability,
incremental stability, practical stability, finite-time stability, and training and generalization
stability. The runtime file
{src "NN/MLTheory/LearningTheory/Stability/Dynamics/Runtime.lean"}[Dynamics.Runtime] provides
`Float` diagnostics.

Read these predicate names with their actual hypotheses. `IsInputToStateStable` currently asks for
an inequality with supplied comparison functions but does not impose the usual class-K/class-KL
conditions, and its input bound uses indices strictly before the current time. The real
`stabilityMargin` uses a supremum; without nonemptiness and boundedness assumptions it need not
represent an attained or usable radius.

This vocabulary is here because neural-network learning theory is not limited to static supervised
learning: recurrent models, samplers, controllers, RL policies interacting with state, and learned
dynamical systems all need language for trajectories.

Take the simplest contraction, $`x\mapsto x/2`:

```lean
-- Halving gives an exactly representable trajectory for
-- these initial states.
def decayMap (x : Tensor Float [1]) : Tensor Float [1] :=
  Tensor.scaleSpec x 0.5

def decayStart : Tensor Float [1] := [1]
def decayFixed : Tensor Float [1] := [0]
def decayHalf : Tensor Float [1] := [0.5]
```

```lean (name := dynRun)
-- Keep the time horizon and the two diagnostic thresholds
-- explicit.
#eval (generateTrajectory decayMap decayStart 4).map
  tensorL2NormFloat
#eval testLyapunovStability decayMap decayFixed
  #[decayStart] 8 1.0
#eval testAsymptoticStability decayMap decayFixed
  #[decayStart] 8 0.01
```

```leanOutput dynRun
#[1.000000, 0.500000, 0.250000, 0.125000, 0.062500]
```

```leanOutput dynRun
Except.ok true
```

```leanOutput dynRun
Except.ok true
```

The displayed halvings are exact in binary arithmetic, and both tests
pass: the states never leave the unit ball, and after eight steps the state is
$`2^{-8}=0.00390625<0.01`. Now run the aggregate diagnostic:

```lean (name := dynAll)
-- Two distinct initial points also supply a
-- nonzero-distance contractivity pair.
#eval analyzeStability decayMap decayFixed
  #[decayStart, decayHalf] 8
```

```leanOutput dynAll (whitespace := lax)
Except.ok { isLyapunovStable := false,
  isAsymptoticallyStable := true,
  isContractive := true,
  isBiboStable := true,
  stabilityMargin := some 0.200000,
  convergenceRate := some 0.693147 }
```

The convergence rate is $`0.693147\approx\ln 2`, which is exactly $`-\ln(d_1/d_0)` for a contraction
factor of $`1/2`, so the estimator recovers the rate of the system it was given. And
`isLyapunovStable` is `false`, for a system that is Lyapunov stable, asymptotically stable, and
globally exponentially stable.

The `false` comes from the test's fixed tolerance. `analyzeStability` uses
fixed thresholds: tolerance $`0.1` for the Lyapunov test, $`0.01` for the asymptotic test,
$`0.9` for
contractivity, $`1.0` for BIBO. Our trajectory starts at distance $`1` from the equilibrium, so it
sits outside a $`0.1` ball at $`t=0` and the test reports a violation. What the field actually
answers is "did every state of every trajectory stay within $`0.1` of the equilibrium", which is a
different question from Lyapunov stability, whose $`\varepsilon` and $`\delta` are quantified, not
fixed at $`0.1`. The result therefore does not refute Lyapunov stability. Conversely, a finite
test could return `true` for a system that leaves the ball at step nine. Use the individual
diagnostic functions when you need explicit thresholds, and report the thresholds, initial
points, and time horizon beside their results. The aggregate record retains
these fixed defaults; its field names are not theorem claims.

There are also two levels of optionality in this printed record. `Except.ok` means the aggregate
computation returned successfully. A field such as `convergenceRate := some ...` means that its
particular estimator produced a value; `none` would mean no value was available from that
estimator. The rate uses the first two distances to equilibrium, so eight simulated steps do not
make it an eight-step fitted rate. For this recurrence the ratio is always one half, which is why
the first-step estimate describes the whole exact trajectory. A nonlinear recurrence need not
keep that ratio, even if its first step looks equally favorable.

The aggregate diagnostic builds its contractivity pairs by rotating the test-point array.
A single test point yields only the pair $`(x_0,x_0)`, whose input distance is zero.
The contractivity test skips zero-distance pairs and requires at least one pair with positive
input distance; otherwise it returns an `Except.error`. The two distinct points passed above
supply such a pair.

# Runtime And Spec Splits

The learning theory tree repeats one pattern: a spec predicate or theorem is kept separate from the
executable runtime diagnostic, and a float32 or artifact bridge connects them only once the
hypotheses have been stated.

For robustness the split is `Robustness.Spec` versus `Robustness.Runtime`, and the proofs that join
them live in a third place, `NN.MLTheory.Proofs.Verification.Robustness`. For dynamical stability it
is `Stability.Dynamics.Spec` versus `Stability.Dynamics.Runtime`, with the joining theorems still
missing. For ridge regression, `RidgeRegression1D.Real` proves ideal stability, while
the ridge-regression bridge connects one executable binary32 expression to an FP32 expression.
That finite-evaluation bridge does not yet transfer the real stability bound.

A theorem over the reals does not automatically apply to a floating-point computation.
For ridge regression, the real stability bound and the finite-evaluation bridge leave a specific
obligation: bound the numerical error in each of the two fits whose losses we compare.

# Learning Theory Assumptions And Evidence

A learning-theory claim has four visible fields:

```
object       : mechanism, classifier, algorithm, dataset, trajectory, or estimator
property     : privacy, robustness, stability, convergence, or bounded residual
evidence     : theorem, checker, runtime diagnostic, or imported artifact
boundary     : real semantics, finite binary32 refinement, or external producer assumption
```

These four fields separate a theorem from nearby runtime evidence. In particular:

- If a script says an optimizer is differentially private because it used a DP library, the formal
  claim must define a mechanism and prove the DP event inequality or import a checked theorem.
- If no attack found an adversarial example, that is runtime evidence unless it implies
  `IsCertifiedRobust`. Exact sampled ratios lower-bound the least Lipschitz constant;
  a floating-point diagnostic also requires accounting for its own numerical error.
- If a model seems stable when retrained, the formal statement is a replace-one stability predicate
  over `Dataset n Z`, and its constant has to be read at the sample size you actually have.
- If a dynamical system stayed bounded in simulation, the formal statement is a BIBO, ISS,
  Lyapunov, or related stability predicate, and the diagnostic's thresholds are part of what it
  reported.
- If the theorem is over the reals but the code uses float32, the missing link is a
  FloatLib binary32/FP32 bridge with explicit finite-path hypotheses.

These distinctions help decide what evidence to collect for a model. A robustness application
needs a bound on output change and a margin at the particular input. A learning-stability
application instead needs a comparison of two trained hypotheses, with a common test loss.
A recurrent controller needs an assertion about its state evolution and admissible inputs.
Although all three may use a norm estimate internally, their outer quantifiers differ. Writing
the desired conclusion first makes it possible to check whether a proposed theorem supplies the
right object, and whether a runtime report has measured anything that the theorem actually asks
for.

# Open Verification Problems

The gaps in this layer are specific enough to name:

- *Privacy*: no accountant, no calibration theorem for the Laplace or Gaussian mechanism, no
  composition theorem, and therefore no path from these definitions to a DP-SGD claim.
- *Robustness constants*: the proved Lipschitz bound is the Frobenius one, and the MLP result
  returns an existential rather than a formula, so using the explicit Frobenius layer bound directly
  can produce a concrete mathematical radius.
  The existential MLP theorem alone does not expose its witness formula.
- *Stability*: the only proved bound is for the closed-form one-dimensional ridge estimator. An
  iterative solver, a minibatch trainer, or an early-stopped run needs its own argument, and the
  generalization consequence of stability is not formalized here at all.
- *Dynamics*: every predicate in `Dynamics.Spec` is currently a definition without a theorem, and
  the trajectory tests are finite-horizon diagnostics. Individual functions accept thresholds;
  the aggregate `analyzeStability` uses fixed defaults.
- *Numerics*: the ridge bridge gives agreement under a finite-evaluation hypothesis, not an error
  bound, so the float32 fit is not yet connected to the real stability constant.
