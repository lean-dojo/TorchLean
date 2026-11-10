import VersoManual
import FloatLib
import TorchLean.NN.LearningTheory.DifferentialPrivacy
import TorchLean.NN.Robustness.Empirical
import TorchLeanExamples.DeepDives.Stability.RidgeRegression1D
import TorchLean.Verify.Robustness.LipschitzCertified
import TorchLean.Verify.Robustness.MLP
import TorchLean.Tensor.Einops.Elab.Einsum
import TorchLean.Tensor.Einops.Elab.Pack
import TorchLean.Tensor.Literal
import TorchLean.Tensor.Einops.Elab.Transform
import TorchLean.Tensor.Tactic.Einops.Report
import TorchLean.Tensor.LinearAlgebra
import TorchLeanBlueprint.Bib
import TorchLeanBlueprint.Roles
import TorchLean.Tensor.Constructors

open Verso.Genre Manual
open Verso.Genre.Manual.InlineLean
open FloatLib.Floats (ExecFloat)

-- Privacy lives under `TorchLean.LearningTheory`, and the robustness definitions and the theorems
-- that connect them under `TorchLean.Robustness`. Opening them here lets the displayed signatures
-- below read the way a caller would type them, inside Verso's narrow column.
open TorchLean
open Linear
open LearningTheory DifferentialPrivacy
open Dataset
open Examples Ridge
open Robustness
open IsCertifiedRobust IsAdversariallyRobust
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
evidence for a theorem, we have to show how its result addresses the theorem's quantifiers. A
failed attack, for example, leaves open what happens at perturbations the attack never tried. Named
Lean blocks below are checked during the page build, while Python and CLI transcripts record
separate experiments.

# Learning Theory Claims

We organize the learning theory material around five concrete objects:

- *Randomized mechanism*: a map from inputs to probability measures over outputs. Lean states
  $`(\varepsilon,\delta)` privacy, pure privacy, monotonicity in $`\delta` and $`\varepsilon`, and
  post processing.
- *Robustness predicate*: a classifier is stable on a perturbation ball. Lean names tensor norms,
  local balls, Lipschitz predicates, and certified robustness, and proves the implications between
  them.
- *Algorithmic stability*: replacing one training example changes the learned loss only slightly.
  Lean gives typed datasets, `replaceAt`, `removeAt`, learning maps, and loss change bounds.
- *Dynamical stability*: trajectories stay near an equilibrium or converge to it. Lean names
  contraction, Lyapunov and local asymptotic stability.
- *Ridge regression case study*: a one-dimensional regularized ERM theorem, with a real stability
  theorem plus a FloatLib binary32 execution bridge.

These objects don't share one proof method. Differential privacy is an inequality between measures.
Robustness quantifies over a neighborhood of an input. Algorithmic stability compares two executions
of a learning algorithm on neighboring datasets. Dynamical stability concerns an entire trajectory.
The ridge result is a concrete algebraic proof whose constants can be carried into a numerical
analysis. What they have in common is that the quantified object and its boundary are
stated before any runtime evidence is interpreted.

# Differential Privacy

TorchLean's core privacy definitions live in
{src "TorchLean/NN/LearningTheory/DifferentialPrivacy.lean"}[
TorchLean.NN.LearningTheory.DifferentialPrivacy API]. Here's the central definition as
Lean prints it, not a paraphrase:

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

Read the arguments left to right. `α` is the input type (typically a dataset), and `β` is the
output type (typically a model or a released statistic). Next comes the *adjacency* relation, of
type `α → α → Prop`: it says which pairs of inputs count as neighbors, and it's a parameter because
"differ in one record" and "differ in one person's whole history" are different privacy claims
about the same mechanism. `Mechanism α β` abbreviates `α → ProbabilityMeasure β`, so a mechanism is
a randomized map. Last come the budget $`\varepsilon` and the slack $`\delta`.

Square brackets around `[MeasurableSpace β]` request a typeclass instance: the output type must
come with a collection of measurable events. Braces around `{α β : Type}` make those arguments
implicit in ordinary calls, and prefixing the name with `@` exposes them so we can see the whole
interface. The final `Prop` means the declaration defines a proposition about a mechanism. Merely
constructing a value of type `Mechanism α β` doesn't supply a proof of that proposition.

Adjacency also determines the direction of comparison, and the definition doesn't require the
relation to be symmetric. If an application uses symmetric adjacency, it can apply the same
inequality with the datasets exchanged. Otherwise the reverse comparison needs its own adjacency
premise. Direction matters when you read the ratios in the example below.

Budget restrictions are left to the caller as well. For the usual privacy guarantee, we use
$`\varepsilon\ge0` and $`0\le\delta<1`. `ENNReal` ensures that delta is nonnegative, but permits
values at least one and infinity, which the predicate itself doesn't exclude.

The predicate's body is the standard event inequality: for every adjacent pair and every measurable
event `S`,

$$`\forall D\sim D',\;\forall S,\qquad
\Pr[M(D)\in S]\le e^\varepsilon \Pr[M(D')\in S]+\delta.`

Probabilities and the slack use `ENNReal` (the extended nonnegative reals). Nonnegativity is built
into the type, and addition and multiplication are monotone. So the closure proofs can work with
the event inequality itself. Subtraction in this type is truncated at zero, so the additive form
also avoids importing ordinary real-subtraction identities.

A mechanism's output is a `ProbabilityMeasure`, not a density or a sampler. So one definition
covers discrete mechanisms such as randomized response, continuous mechanisms such as Laplace or
Gaussian noise, and whole randomized training procedures. The definition says what any privacy
proof must establish, and it deliberately says nothing about how the randomness is produced.

## Constant Mechanisms

A mechanism that ignores its input has the same output distribution on every neighboring pair, so
it satisfies pure privacy with zero budget:

```lean
-- An input-independent distribution gives identical
-- probabilities for every event.
example {α β : Type} [MeasurableSpace β]
    (Adj : α → α → Prop) (μ : ProbabilityMeasure β) :
    PureDP (α := α) (β := β) Adj (fun _ => μ) 0 := by
  intro a a' _ S _
  simp
```

For this input-independent mechanism the proof is `simp`, because after `intro` the goal is
$`\mu(S)\le e^0\cdot\mu(S)+0`, and $`e^0=1`. Both sides concern the same event under the same
measure. `PureDP` is the $`\delta=0` abbreviation.

## Monotonicity Of The Budget

Two closure facts get their own theorems. First, `mono_delta` says a mechanism
private for $`\delta_1` is private for any looser $`\delta_2\ge\delta_1`. A second theorem allows
the same relaxation of $`\varepsilon`:

```lean (name := dpMono)
-- Keep the mechanism fixed while increasing its allowed
-- epsilon budget.
#check @mono_eps
```

```leanOutput dpMono (whitespace := lax)
@mono_eps : ∀ {α β : Type}
  {Adj : α → α → Prop} [inst : MeasurableSpace β]
  {M : Mechanism α β} {ε₁ ε₂ : ℝ} {δ : ENNReal},
  ε₁ ≤ ε₂ →
    DifferentialPrivacy Adj M ε₁ δ →
      DifferentialPrivacy Adj M ε₂ δ
```

Together the two lemmas say the predicate is monotone in the whole budget $`(\varepsilon,\delta)`,
so a caller may use a looser budget than the one already proved. For the epsilon lemma,
monotonicity of $`\exp` increases the multiplier on the right-hand side, and multiplying a
nonnegative event probability preserves that inequality. That's the comparison `gcongr` handles.

For example, a proof at $`(0.5,0)` can be reused at $`(1,0)` by supplying the real inequality
$`0.5\le1`. The theorem keeps the mechanism and adjacency fixed, and it doesn't describe the budget
of two successive releases: comparing two numbers on the right-hand side of one event inequality is
different from analyzing a joint output distribution. In Lean, the two arrows in the signature
spell out how to use it: first give the budget comparison, then the existing privacy proof, and the
result is the relaxed privacy proof.

## Postprocessing

```lean (name := dpPost)
-- The measurable output map is the only new premise for
-- postprocessing.
#check @postprocess
```

```leanOutput dpPost (whitespace := lax)
@postprocess : ∀ {α β γ : Type}
  {Adj : α → α → Prop} [inst : MeasurableSpace β]
  [inst_1 : MeasurableSpace γ] {M : Mechanism α β}
  {ε : ℝ} {δ : ENNReal} {f : β → γ},
  Measurable f → DifferentialPrivacy Adj M ε δ →
    DifferentialPrivacy Adj (M.postprocess f) ε δ
```

Here `f` is a fixed measurable function of the released output. It may, for example, convert a
private model into an exported representation or a report. Applying it preserves the budget,
provided it uses only the released output and fixed public information. An evaluator that reads
the private dataset again is outside this theorem.

Post-processing has a short proof for a mathematical reason. For a measurable output event `T`, the
post-processed mechanism lands in `T` exactly when the original mechanism lands in the measurable
preimage $`f^{-1}(T)`. Applying the DP inequality to that preimage gives

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
That's why the proof uses preimages: the two spaces can have quite different representations, but
the probability comparison is still about the original release.

The measurability hypothesis `hf` licenses the preimage step. Together with the original privacy
premise it's sufficient, and the proof doesn't inspect the representation of the model or report.
As a reference point we use the event inequality of {Informal.citet dwork2006}[].

## Sampling And Privacy Guarantees

Sampling alone can't establish the universally quantified privacy predicate. Robustness and
asymptotic stability have the same limitation: finite experiments test cases, not all quantified
inputs.

Take the Laplace mechanism on a counting query. Two adjacent datasets differ in one record, so the
count differs by at most the sensitivity $`\Delta=1`. Releasing the count plus
$`\mathrm{Laplace}(0,\Delta/\varepsilon)` noise is $`(\varepsilon,0)`-DP. With $`\varepsilon=0.5`
the noise scale is $`2`. Take the true counts $`10` and $`11`, and the tail event
$`S=\{\text{output}\ge 12\}` (the direction that favors the larger count, and therefore where the
inequality is tightest). Two million samples per dataset in PyTorch {Informal.citep pytorch2019}[]:

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

Analytically, the tail probabilities are $`\tfrac12 e^{-1}` and $`\tfrac12 e^{-1/2}`, and their
ratio is exactly $`e^{1/2}`, so this event attains the privacy bound in the direction from the
larger count to the smaller one.

Look at the last three lines again. The empirical ratio is $`1.64968` and the bound is
$`e^{0.5}=1.64872`. So the estimate violates the inequality it's supposed to illustrate (by about
six parts in ten thousand), because each probability was estimated from a finite sample. More
samples can improve estimates for this fixed event, but they can't establish the inequality for
every measurable event and every neighboring pair. Rare events are especially hard to estimate: few
or no sampled outputs may land in them. The analytical tail calculation explains this experiment. A
privacy proof must cover the full quantification.

## Privacy Verification Gaps

Don't infer a privacy budget from the presence of the DP namespace. The source supplies the
semantic target and the closure laws, but it doesn't contain a runtime privacy accountant, a
calibration theorem for the Laplace or Gaussian mechanism, a composition theorem, or an end-to-end
theorem for a training command. Concretely, the gap between this file and a DP-SGD claim is:
Poisson or uniform subsampling with its amplification argument, per-example gradient clipping with
a sensitivity bound, noise calibration to that sensitivity, and composition accounting across
steps. Each of those is a separate implementation and proof obligation. What the file does give you
is the target they must hit, plus the measurable, data-independent post-processing theorem.

# Robustness Specifications And Runtime Checks

Robustness is split by trust boundary into two files:

- {src "TorchLean/NN/Robustness.lean"}[Robustness] defines the
  mathematical predicates, scalar-polymorphically.
- {src "TorchLean/NN/Robustness/Empirical.lean"}[Robustness.Empirical] defines
  executable `Float` diagnostics.

On the spec side we work over TorchLean tensors without committing to one runtime scalar. There we
define `linfNorm`, `distance`, closed tensor balls, global Lipschitz
continuity, adversarial robustness at a point, certified robustness for classifiers, contraction
mappings, and local sensitivity ratios. A typical local robustness predicate has the form

$$`\operatorname{Robust}(f,x,y,\varepsilon)
\Longleftrightarrow
\forall x',\;\|x'-x\|_\infty\le \varepsilon
\Rightarrow
\operatorname*{argmax} f(x')=y,`

That predicate states a perturbation model of the kind studied by {Informal.citet szegedy2014}[]:
every input in the allowed ball must retain the specified label.

Our central robustness predicates take input and output norms as separate arguments, which records
the metric used by each step of an argument. A Lipschitz bound proved in $`\ell^2` can feed a
margin argument using $`\ell^\infty` after a norm-comparison theorem connects them. The certificate
below keeps `normIn` general while fixing the output norm needed to control logits.

## Tensor Norms And Perturbation Balls

Everything on the runtime side is ordinary executable code. Here is a two by two linear map, its
output, and the two norms:

```lean
-- Fix a linear map and two directions with different
-- Euclidean gains.
def lipW : Tensor Float [2, 2] :=
  [[3, 1],
   [0, 2]]

def lipLayer : Linear Float 2 2 :=
  { weights := lipW, bias := [0, 0] }

def lipMap (x : Tensor Float [2]) : Tensor Float [2] :=
  forward lipLayer x

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
#eval Tensor.maxAbs (lipMap lipX)
#eval l2NormFloat (lipMap lipX)
#eval l2NormFloat (lipX - lipY)
#eval decide (Tensor.maxAbsDiff lipX lipY ≤ 0.5)
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

Multiplying out, the first output is $`W(1,1)^\top=(3+1,0+2)^\top=(4,2)^\top`, whose $`\ell^\infty`
norm is the larger absolute coordinate $`4` and whose $`\ell^2` norm is
$`\sqrt{20}\approx4.4721360`. The inputs $`(1,1)` and $`(0,1)` differ only in the first coordinate,
so their distance is $`1` in every $`\ell^p`. And $`(0,1)` isn't in the $`\ell^\infty` ball of
radius $`0.5` around $`(1,1)` (because $`1>0.5`), so the ball test returns `false`.

Here the ball test checks whether this pair satisfies the proposed perturbation constraint. Its
norm and radius must match the ones a robustness theorem uses.

In PyTorch, the corresponding calculation prints the same values:

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

Choosing the ball has a direct geometric effect. Around $`(1,1)`, the infinity-norm ball of radius
$`0.5` allows each coordinate to move independently by at most $`0.5`. Its corner $`(1.5,1.5)` is
at Euclidean distance $`\sqrt{0.5}` from the center, so it's outside the Euclidean ball with the
same numeric radius. A radius therefore needs its norm to be meaningful. For a model whose input is
normalized before inference, the tensor in the theorem is the input at that stage, so a radius in
normalized coordinates requires a conversion before it describes a perturbation of raw pixels or
physical measurements.

## Empirical Lipschitz Ratios

On the runtime side we also provide empirical helpers: Lipschitz ratios over a finite set of input
pairs, and deterministic perturbation sampling. Changing the sampled direction changes the observed
gain, even for this two-dimensional linear map:

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

Of the two sampled pairs, the first differs in the first coordinate, so it measures the gain on
$`(1,0)^\top`, which is $`\|(3,0)^\top\|_2=3`. The second pair closely approximates the top
singular direction. With its decimal expansion rounded, the spectral norm is
$`\sigma_{\max}(W)=\sqrt{7+\sqrt{13}}\approx3.2566165`. In exact arithmetic, ratios at distinct
points lower-bound the least Lipschitz constant. Here the direction and arithmetic are rounded, so
the diagnostic is an approximation, not a proved lower bound.

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

A robustness argument would like to use the middle number, but `Linear.dist_forward_le` in
{src "TorchLean/Verify/Robustness/MLP.lean"}[Verify.Robustness.MLP] proves the one on the right,
by bounding the operator norm with the Frobenius norm. Since the squared Frobenius norm is the sum
of squared entries, a Cauchy-Schwarz argument gives the bound without computing singular values.
That bound is loose by about $`15\%` on this matrix, and in general it can be loose by a factor of
$`\sqrt{\min(m,n)}`. If a downstream certificate ever needs the sharper constant, getting it means
proving another theorem, not writing another diagnostic.

The wrapper `Except.ok` says that the ratio computation returned a result through its success
branch, but the number inside is still the ratio for the supplied pairs. To use the Frobenius upper
bound in this example, you reason instead about an arbitrary displacement $`v`: each output
coordinate is a row dot product, Cauchy–Schwarz bounds its square, and summing the row bounds gives
$`\|Wv\|_2^2\le\|W\|_F^2\|v\|_2^2`. The bias disappears when you subtract two outputs. That proof
covers directions absent from the sample array, which is the information a robustness certificate
needs.

A related helper measures one-directional sensitivity:

```lean (name := robSens)
-- Perturb only the first input coordinate when measuring
-- sensitivity.
#eval sensitivity (α := Float) lipMap l2NormFloat
  l2NormFloat lipX ([0.001, 0] : Tensor Float [2])
```

```leanOutput robSens
3.000000
```

That helper computes the finite-difference ratio $`\|f(x+p)-f(x)\|_2/\|p\|_2`. Over the reals, for
a linear map and a fixed nonzero direction, the ratio is independent of step size. For a nonlinear
network it may also vary with the point and step size. Either way, floating subtraction of nearby
outputs can lose precision.

## Lipschitz Bounds And Margin Certificates

We prove the implications needed to turn a Lipschitz bound into a robustness certificate in
{src "TorchLean/Verify/Robustness/LipschitzCertified.lean"}[LipschitzCertified]. First, a Lipschitz
bound becomes an output-perturbation bound:

```lean (name := lipThm)
-- Read the output radius as the input radius multiplied by
-- the Lipschitz bound.
#check @IsAdversariallyRobust.of_lipschitz
```

```leanOutput lipThm (whitespace := lax)
@IsAdversariallyRobust.of_lipschitz : ∀ {s₁ s₂ : Shape}
  {f : Tensor ℝ s₁ → Tensor ℝ s₂}
  {norm₁ norm₂ : {s : Shape} → Tensor ℝ s → ℝ} {L : ℝ},
  0 ≤ L →
    IsLipschitzContinuous f (fun {s} => norm₁)
        (fun {s} => norm₂) L →
      ∀ (x₀ : Tensor ℝ s₁) (ε : ℝ),
        IsAdversariallyRobust f (fun {s} => norm₁)
          (fun {s} => norm₂) x₀ ε (L * ε)
```

Read the conclusion: the output radius is $`L\varepsilon`, with no hypothesis on $`\varepsilon`
beyond what `IsLipschitzContinuous` already gives. For an input satisfying the ball premise,
multiply its distance bound by the nonnegative Lipschitz constant. That bounds output drift.
Preserving a label also requires a margin between logits.

Next comes a step about labels, and it needs a quantity the drift bound doesn't mention: how far
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
#check @IsCertifiedRobust.of_lipschitz
```

```leanOutput certThm (whitespace := lax)
@IsCertifiedRobust.of_lipschitz :
  ∀ {s₁ : Shape} {n : ℕ} {f : Tensor ℝ s₁ → Tensor ℝ [n]}
  {normIn : {s : Shape} → Tensor ℝ s → ℝ} {L : ℝ},
  0 ≤ L →
    IsLipschitzContinuous f (fun {s} => normIn)
        (fun {s} => linfNorm) L →
      ∀ {x₀ : Tensor ℝ s₁} {ε m : ℝ} {c : Fin n},
        0 ≤ ε →
          HasLogitMargin (f x₀) c m →
            2 * (L * ε) < m →
              IsCertifiedRobust
                (fun x => argmaxClassifier (f x))
                (fun {s} => normIn) x₀ ε
```

In the certificate, the side condition is $`2L\varepsilon<m`. Since the winning logit can fall by
$`L\varepsilon` while a competitor rises by $`L\varepsilon`, the gap can close at twice the drift
rate. For $`L>0`, the condition becomes $`\varepsilon<m/(2L)`, which is why a loose Lipschitz
constant costs radius linearly, and why the $`15\%` gap between $`\|W\|_F` and $`\sigma_{\max}(W)`
above reduces the certified radius.

In this theorem the output norm is pinned to $`\ell^\infty`, while the input norm `normIn` stays
free. The proof forces that asymmetry: the margin argument compares individual logits, and the norm
that controls individual coordinates is the sup norm. To feed an $`\ell^2` bound into this slot,
use the companion lemma `IsLipschitzContinuous.linf_of_l2` in the same file.

Ties are ruled out by the strict inequality in the margin condition. If the gap could shrink to
zero, the classifier's tie-breaking convention would become relevant. With a positive remaining
gap, every competitor is strictly below `c`, so the same class is selected regardless of that
convention. In the conclusion, `IsCertifiedRobust` compares the prediction throughout the ball with
the prediction at its center, and it doesn't compare either prediction with an external
ground-truth label. To state robust accuracy, a caller also supplies the fact that the center's
predicted class is the intended label.

## Logit-Margin Certificate Example

For a concrete certificate, start at logits $`(2,0)` with margin $`2`
and the identity map as the classifier's tail, so $`L=1`:

```lean
-- Discharge the margin and radius conditions for the
-- two-logit identity map.
def marginLogits : Tensor ℝ [2] := [2, 0]

example : IsCertifiedRobust (α := ℝ)
    (classifier := fun y : Tensor ℝ [2] =>
      argmaxClassifier (n := 2) y)
    (norm := linfNorm (α := ℝ))
    marginLogits 0.4 := by
  have hLip :
      IsLipschitzContinuous (α := ℝ)
        (fun y : Tensor ℝ [2] => y)
        (linfNorm (α := ℝ))
        (linfNorm (α := ℝ)) 1 := by
    intro x y
    simp
  have hMargin :
      HasLogitMargin marginLogits ⟨0, by decide⟩ 2 := by
    intro k hk
    fin_cases k
    · exact absurd rfl hk
    · norm_num [marginLogits]
  exact
    IsCertifiedRobust.of_lipschitz
      (L := 1) zero_le_one hLip (by norm_num)
      hMargin (by norm_num)
```

Here the margin proof has two cases, because `HasLogitMargin` quantifies over competitors:
`fin_cases k` splits on the class index, the winning class is excluded by the hypothesis `k ≠ c`,
and the competitor obligation is $`0\le 2-2`, which `norm_num` closes after unfolding the literal.
That unfolding works because `getScalar_ofList` is a simp lemma, so a rank-one tensor literal
reduces to list lookup without any manual `Rep` reasoning.

At the end, the two `by norm_num` arguments are the numeric side conditions $`0\le 0.4` and
$`2\cdot(1\cdot 0.4)=0.8<2`. That last one is the certificate. It's also where the radius
stops: at $`\varepsilon=1.2` the same call fails, because $`2\cdot(1\cdot 1.2)=2.4` isn't below the
margin $`2`. A failed sufficient condition alone doesn't establish the existence of an adversarial
example at radius $`1.2`. For this identity map, we can supply a separate witness: $`(0.8,1.2)` is
within radius $`1.2` of $`(2,0)` and class 1 wins. That witness supplies the information a failed
sufficient condition alone wouldn't.

Within radius `0.4` in the infinity norm, each logit can move by at most `0.4`. In the worst case
for the predicted class, its logit falls and its competitor's rises, reducing the gap from `2` to
at least `1.2`. That calculation explains the factor of two in the general condition. For an
earlier layer of a model, the Lipschitz bound first converts an input perturbation radius into this
bound on logit movement.

## MLP Lipschitz Bounds

In {src "TorchLean/Verify/Robustness/MLP.lean"}[Verify.Robustness.MLP], the same machinery
composes to a network:

```lean (name := mlpThm)
-- The constant is the product of the two Frobenius norms.
#check @MLP.forward_lipschitz
```

```leanOutput mlpThm (whitespace := lax)
@MLP.forward_lipschitz :
  ∀ {inDim hidDim outDim : ℕ}
  (l1 : Linear ℝ inDim hidDim)
  (l2 : Linear ℝ hidDim outDim),
  IsLipschitzContinuous
    (fun x => MLP.forward l1 l2 x)
    (fun {s} x => ‖x‖)
    (fun {s} x => ‖x‖)
    (‖l2.weights‖ * ‖l1.weights‖)
```

Here the constant is the product of the two Frobenius norms, the bound of `MLP.forward_dist_le` in
the same file, so a certificate can evaluate $`L` and check the margin condition
$`2(L\varepsilon)<m` of `IsCertifiedRobust.of_l2_lipschitz`, the margin lemma for the Euclidean
output norm. That lemma only requires $`L\ge0`. For the ReLU in the middle, the proof uses
`Tensor.l2Dist_relu_le`, the sharp statement for ReLU: ReLU is exactly 1-Lipschitz in
$`\ell^2`, so the activation contributes no looseness at all. Slack can come from bounding each
linear layer and from multiplying layerwise constants whose maximizing directions may not align.

Both norm slots hold `fun {s} x => ‖x‖`, Mathlib's Euclidean norm on real tensors from
{src "TorchLean/Tensor/Euclidean.lean"}[Euclidean], so every Mathlib lemma about norms and
distances applies to the bound.

From the ordering $`3.000000\le 3.256617\le 3.741657` above, you can see why an observed ratio
alone can't supply the upper bound required by a Lipschitz certificate.

# Algorithmic Stability

In the {src "TorchLean/NN/LearningTheory/Stability.lean"}[algorithmic stability API]
we make one central representation choice: a dataset of size `n` is a `TorchLean.Tensor Z [n]`, so
the sample size is part of the type, not a list length you have to remember separately. Built on
that representation are coordinate access, `replaceAt` and `removeAt`, deterministic learning maps
`Dataset n Z → H`, real-valued losses, empirical error, i.i.d. sampling of datasets, and uniform
replace-one stability (`UniformStableReplace`).

For a dataset containing three examples, replacement preserves the size and removal reduces it:

```lean
-- Replacement keeps three entries, while removal changes
-- the dataset type to size two.
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

Replace-one keeps the size and changes coordinate `1`. Remove-one drops that coordinate and lands
in a smaller type (`Dataset 2`), which is why `removeAt` is stated at `n + 1` and produces a
dataset at `n`.

Here the replacement index has type `Fin 3`, so it must carry a proof that it's less than three. An
out-of-range index fails during elaboration:

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

In that failing call, the attempted coordinate is `3`, which would require the false inequality
`3 < 3`. So Lean rejects the dataset operation before a learner can use it.

Coordinate lookup after removal is described by reindexing:

```lean
-- After removing index one, the new index one refers to the
-- old index two.
example (S : Dataset 3 Nat) :
    Dataset.toFn (removeAt (n := 2) S ⟨1, by decide⟩)
        ⟨1, by decide⟩
      = Dataset.toFn S ⟨2, by decide⟩ := by
  exact toFn_removeAt S ⟨1, by decide⟩ ⟨1, by decide⟩
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
  simp only [empiricalError, Fin.sum_univ_three,
    Dataset.toFn_ofFn_apply]
  norm_num
```

In that computation the loss ignores the hypothesis and returns the example itself, the dataset is
$`(0,1,2)`, and the mean is $`1`. That checks the normalization: `empiricalError` carries the
$`1/n` in its definition, so downstream constants don't silently absorb a factor of the sample
size.

Here's the stability predicate itself:

```lean (name := stabDef)
-- The proposition compares retrained losses at every common
-- test example.
#check @UniformStableReplace
```

```leanOutput stabDef (whitespace := lax)
@UniformStableReplace : {Z H : Type} → {n : ℕ} →
  LearningMap n Z H → Loss H Z → ℝ → Prop
```

which unfolds to the standard uniform stability inequality

$$`\forall S,S^{(i)},z,\qquad
|\ell(A(S),z)-\ell(A(S^{(i)}),z)|\le \beta,`

quantified over all datasets, all replaced indices, and all test points. The classical reference is
{Informal.citet bousquet2002}[], and our definitions follow its proof habit: first spell out the
dataset perturbation, then state how much the learned loss can move.

Here the test point `z` and the replacement example `z'` play different roles. The learner is
retrained on `replaceAt S i z'`, but both resulting hypotheses are evaluated on the same `z`.
Letting the test point change at the same time would mix sensitivity of the learner with variation
of the loss across examples. Uniformity means the bound holds for every such `z`, including one
that never appeared in training. No sampling distribution is needed for this pointwise definition.
Finite indices have decidable equality, so replacement can select the coordinate without an
additional hypothesis in `UniformStableReplace`.

# Ridge Regression Stability

Our most concrete learning theory development is
{src "TorchLeanExamples/DeepDives/Stability/RidgeRegression1D.lean"}[
TorchLeanExamples.DeepDives.Stability.RidgeRegression1D API]. It proves a replace-one uniform
stability bound for one-dimensional ridge regression with squared loss under bounded inputs. Its
argument is the classical strongly convex ERM one, in a setting small enough to read end to end:

1. Each example is a bounded pair $`(x,y)` with $`\lvert x\rvert\le X` and $`\lvert y\rvert\le Y`,
   carried as a subtype so the bounds travel with the data.
2. In closed form, the fit is $`\hat w(S) = \frac{\sum_i x_i y_i}{\sum_i x_i^2 + \lambda N}`.
3. Replacing one example changes the numerator by at most $`2XY` and the denominator by at most
   $`2X^2`.
4. Those two bounds combine into a bound on $`|\hat w(S)-\hat w(S')|`, using
   $`\sum_i x_i^2+\lambda N\ge\lambda N` to control the reciprocal.
5. A difference of squares argument converts the weight change bound into a loss change bound.

```lean (name := ridgeThm)
-- The bounded-example type carries the data restrictions
-- used in this constant.
#check @Ridge.uniformStableReplace_estimator
```

```leanOutput ridgeThm (whitespace := lax)
@uniformStableReplace_estimator :
  ∀ {n : ℕ} {X Y lam : ℝ},
  0 < lam →
    UniformStableReplace (fun S => estimator lam S)
      (fun w z => sqLoss w z)
      (4 * X ^ 2 * Y ^ 2 * (lam + X ^ 2) ^ 2 /
        (lam ^ 3 * sampleSize))
```

In the printed theorem, only `0 < lam` is stated as a mathematical premise, but the data
restrictions haven't disappeared. They're encoded by `BoundedExample X Y` (the type of each
training, replacement, and test example), and a value of that subtype carries both coordinates and
proofs of their bounds. The dataset has size `n + 1`, so its nonemptiness is also part of the type.
These choices explain why the denominator can be proved positive: the sum of squares is nonnegative
and $`\lambda(n+1)>0`. With an arbitrary unbounded test point, the stated uniform squared-loss
constant wouldn't be justified.

The denominator also specifies how regularization is normalized. Here the objective
$`N^{-1}\sum_i(wx_i-y_i)^2+\lambda w^2` has derivative zero at the displayed quotient: multiplying
the stationarity equation by `N` produces the term $`\lambda N`. If the data-fit objective were an
unnormalized sum with the same penalty coefficient, its denominator would instead contain
$`\lambda`. So the theorem's sample-size dependence belongs to this particular estimator, and you
should match the training objective before substituting its regularization parameter.

Writing $`N=n+1`, the proved constant is

$$`\beta
=\frac{4X^2Y^2(\lambda+X^2)^2}{\lambda^3N}.`

In that constant, the $`1/N` comes from the ridge denominator: adding $`\lambda N` to a sum of
squares means one example can move the fit only by $`O(1/N)`. And the $`\lambda^{-3}` records the
cost of controlling both the fitted weight and the change in its reciprocal denominator. This
constant is an explicit, valid bound, with no claim of optimality, and the difference matters.
Instantiate it:

```lean
-- Substitute sixteen examples and unit bounds into the real
-- stability theorem.
example :
    UniformStableReplace (Z := BoundedExample 1 1) (H := ℝ)
      (A := fun S =>
        estimator (n := 15) (X := 1) (Y := 1) 1 S)
      (ℓ := fun w z => sqLoss (X := 1) (Y := 1) w z)
      (β := 1) := by
  have h :=
    Ridge.uniformStableReplace_estimator
      (n := 15) (X := 1) (Y := 1) (lam := 1) one_pos
  norm_num [Ridge.sampleSize] at h
  exact h
```

At $`X=Y=\lambda=1` and $`N=16`, the formula gives $`16/16=1`, and `norm_num` checks the
instantiation. That value is below the generic loss-range bound of $`4` discussed next.

At $`N=2`, the same formula would give $`8`, a weak bound in that regime. A private lemma in the
same file proves $`|\hat w x-y|\le Y(\lambda+X^2)/\lambda` (which is $`2` here), so the squared
loss never exceeds $`4` and any two losses differ by at most $`4`. Since $`\beta=16/N` at these
constants, the theorem improves that loss-range bound once $`N>4`. The bound is valid at each
sample size, but its inverse dependence on sample size becomes useful in that regime, and the
displayed $`N=16` instantiation demonstrates the improvement. A smaller general constant would
require a sharper proof.

It's worth spelling out the last step of the argument. For two fits $`w,w'` evaluated at the same
$`(x,y)`, the loss difference factors as

$$`(wx-y)^2-(w'x-y)^2=(w-w')x\bigl((wx-y)+(w'x-y)\bigr).`

In that factorization, the fit-change estimate controls the first factor, the example subtype
controls $`|x|`, and the residual bound controls the last factor. That's how a statement about
parameter movement becomes a statement about prediction loss. In a larger model, a bound on
parameter distance would need an analogous argument for the network and loss being used. By itself,
small parameter movement isn't the uniform stability predicate displayed above.

## Ridge Regression In Binary32

For ridge regression the ideal theorem lives over $`\mathbb R`, while the executable development is
in {src "TorchLeanExamples/DeepDives/Stability/RidgeRegression1D.lean"}[the binary32
ridge-regression implementation], which evaluates the algorithm with FloatLib. Its bundled example
dataset is $`x=(1,3)`, $`y=(2,4)`, with $`\lambda=1` and $`N=2`:

```lean (name := ridgeExec)
-- Display both the decimal rendering and the binary32
-- representation of the fit.
#eval
  (ExecFloat.Binary.toFloat32
    Sample.wHat).toFloat
#eval ExecFloat.Binary.toBits32
  Sample.wHat
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

Both sides give the same bit pattern: `1066751317`, which is `0x3F955555`. Every intermediate here
($`14`, $`10`, $`12`) is exactly representable in binary32, so the only rounding in the whole
computation is the final division, and both implementations round it to nearest with ties to even
{Informal.citep goldberg1991}[]. A computation with inexact intermediates would agree only if the
two implementations also agreed on association and on intermediate precision, and our executable
model is meant to let us check that kind of agreement instead of assuming it
{Informal.citep flocq2011}[].

What the executable side proves is narrower than the agreement suggests:

```lean (name := ridgeBridge)
-- The expression and direct fit have the same ordered
-- binary32 operations. We require finite evaluation.
#check @toReal_fit
```

```leanOutput ridgeBridge (whitespace := lax)
@toReal_fit : ∀ {n : ℕ}
  {format : FloatLib.Floats.Formats.BinaryInterchange.FloatFormat}
  {plan : FloatLib.Floats.Formats.BinaryInterchange.Configured.StoragePlan format} {code : Type}
  [inst : ExecFloat.ModelCodec plan (FloatLib.Floats.Formats.BinaryInterchange.Model format) code]
  (lam : ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan))
  (S :
    Dataset (n + 1)
      (ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan) ×
        ExecFloat (FloatLib.Floats.Formats.BinaryInterchange.Configured.Family format code plan)))
  {d : FloatLib.Numerics.Dyadic},
  IEEE754.FiniteEval (fun x => 0) (fitExpr lam S) d →
    autoParam (format.isIEEE = true) toReal_fit._auto_1 →
      (fit lam S).toModel.toReal = fitReal lam S
```

This theorem covers the fit we ran above. We prove that evaluating `fitExpr` gives exactly
`fit`, with the same left-to-right additions, multiplications, and final division for any
dataset. We need no associativity law for floating-point addition.

The theorem is a finite-evaluation bridge: *if* the executable expression evaluates without
overflowing to infinity or producing a NaN, then its real interpretation agrees with the
proof-level FP32 expression. It isn't a stability theorem, and it isn't an error bound either.
Connecting the two halves of this section would take a bound on
$`|\hat w_{\mathrm{fp32}}(S)-\hat w_{\mathbb R}(S)|` in terms of the unit roundoff, plus a
loss-sensitivity bound that transfers fit error to the compared losses. Both are ordinary numerical
analysis, and neither is in the file.

The `FiniteEval` argument in the bridge is evidence about the particular expression `fitExpr
lam S`. The variable `d` records its dyadic result, and the equality then compares two
interpretations of that same computation. It does not assert that the equality's right-hand side
is the exact real ridge quotient. A later stability transfer would have to control the numerical
error for both neighboring datasets, since both fits occur in the loss difference. Keeping track
of both executions is essential: bounding the error on the original training set alone leaves
the replacement execution uncontrolled.

# Runtime And Spec Splits

One pattern repeats across the learning theory tree: a spec predicate or theorem is kept separate
from the executable runtime diagnostic, and a float32 or artifact bridge connects them only once
the hypotheses have been stated.

For robustness the split is between the modules `TorchLean.NN.Robustness` and
`TorchLean.NN.Robustness.Empirical`, and the proofs that join them live in a third place, the
`Verify.Robustness` modules. For ridge regression, `uniformStableReplace_estimator` proves ideal
stability, while the ridge-regression bridge connects one executable binary32 expression to an FP32
expression. That finite-evaluation bridge does not transfer the real stability bound.

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
- If no attack found an adversarial example, that's runtime evidence unless it implies
  `IsCertifiedRobust`. Exact sampled ratios lower-bound the least Lipschitz constant, while a
  floating-point diagnostic also requires accounting for its own numerical error.
- If a model seems stable when retrained, the formal statement is a replace-one stability predicate
  over `Dataset n Z`, and its constant has to be read at the sample size you actually have.
- If a dynamical system stayed bounded in simulation, that is runtime evidence on the inputs tried.
  The formal predicates are Lyapunov and local asymptotic stability of an equilibrium.
- If the theorem is over the reals but the code uses float32, the missing link is a FloatLib
  binary32/FP32 bridge with its finite-path hypotheses spelled out.

These distinctions help you decide what evidence to collect for a model. A robustness application
needs a bound on output change and a margin at the particular input. A learning-stability
application instead needs a comparison of two trained hypotheses, with a common test loss. And a
recurrent controller needs an assertion about its state evolution and admissible inputs. All three
may use a norm estimate internally, but their outer quantifiers differ. If you write the desired
conclusion first, you can check whether a proposed theorem supplies the right object, and whether a
runtime report has measured anything the theorem actually asks for.

# Open Verification Problems

Each gap in this layer is specific enough to name:

- *Privacy*: no accountant, no calibration theorem for the Laplace or Gaussian mechanism, no
  composition theorem, and therefore no path from these definitions to a DP-SGD claim.
- *Robustness constants*: `MLP.forward_dist_le` exposes the product of the two layerwise
  Frobenius norms. This supplies an explicit upper bound for a radius calculation, but a sharper
  spectral bound would need another theorem.
- *Stability*: the only proved bound is for the closed-form one-dimensional ridge estimator. An
  iterative solver, a minibatch trainer, or an early-stopped run needs its own argument, and the
  generalization consequence of stability is not formalized here at all.
- *Numerics*: the ridge bridge gives agreement under a finite-evaluation hypothesis, not an error
  bound, so no theorem connects the float32 fit to the real stability constant.
