/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.Transition
public import TorchLean.NN.RL.PolicyGradient.Autograd
public import TorchLean.Tensor.Random
public import TorchLean.Tensor.Layers.Softmax

/-!
# Policy-Gradient Objectives

Typed objectives for categorical policies over a finite action space: REINFORCE, advantage
actor-critic, KL-penalized (trust-region style) policy gradients, entropy regularization, the
soft actor-critic policy term and PPO's clipped surrogate, plus a seeded categorical sampler. The
helpers take logits and are pure functions, so eager runtime code and proof-oriented spec code
can share them.

Selected log-probabilities come in two forms. `logProbability` clamps the softmax probability
into `[epsilon, 1 - epsilon]` before taking the logarithm, and REINFORCE and actor-critic use it.
`logProb` is the unclamped log-softmax that PPO caches and recomputes.

We wrote each objective by hand from its paper. `reinforceLoss` is the REINFORCE step (13.8) of
Sutton and Barto read as a loss, and an advantage passed for the return gives their baseline form
(13.11). Their boxed algorithm also scales the step by $\gamma^t$, which we leave to the caller's
`returnOrAdvantage`. `actorCriticLoss` follows Section 4 of Mnih et al. (2016): the advantage
term, the squared error $(R - V)^2$ of their Algorithm S3, and an entropy bonus whose weight their
Atari runs set to 0.01 and our default sets to 0. The guard in `logProbability` moves small
probabilities. At the default `epsilon = 1e-6` a probability of `1e-10` reads as
$\log 10^{-6} \approx -13.8$, where `logProb` gives $-23.0$.

From PPO come `ratio` (eq. (6)), `PPO.objective` (eq. (7)), `klLoss` (eq. (8), negated) and
`PPO.loss` (eq. (9), negated). `kl` puts the policy that sampled the batch first, as their eqs. (4)
and (8) do. The defaults of `PPO.loss` take $c_1 = 1$ from their Table 5 and leave out the entropy
bonus, as their MuJoCo runs in Section 6.1 do. `sacActorLoss` is eq. (12) of Christodoulou, which
computes the expectation over actions in eq. (7) of Haarnoja et al. as a sum. `kl` and
`klFromLogits` compute the divergence with `klTilted`, the form of Section 4.1 of Amos (2026),
which does not subtract the nearly equal logarithms of the textbook sum, centered at the largest
log-ratio instead of the mean, and with the old policy given by its logarithms. Their values over
ℝ are in `TorchLean.NN.RL.PolicyGradient.Real`. In floating point nothing is proved:
`klFromLogits`'s docstring states the logit range, the inputs that `TorchLeanTest.RL.PolicyKL`
checks and two limits of the rounding.
`RL.PolicyGradient.toReal_ratio_self` proves that `ratio x x` decodes to 1 for a finite `x` at
IEEE formats. `TorchLeanTest.RL.PolicyKL` checks `kl` and `klFromLogits` against mpmath, and no
TorchLeanTest test runs the other objectives.

Primary references:

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992), Machine Learning 8(3-4):229-256: https://doi.org/10.1023/A:1022672621406
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., MIT Press, 2018),
  Sections 13.3 and 13.4, eqs. (13.8) and (13.11):
  http://incompleteideas.net/book/the-book-2nd.html
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016), ICML 2016, Section 4
  and Algorithm S3: https://arxiv.org/abs/1602.01783
- Schulman et al., "Trust Region Policy Optimization" (2015), ICML 2015:
  https://arxiv.org/abs/1502.05477
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, eqs. (4)
  and (6) to (9), Section 6.1 and Table 5: https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016: https://arxiv.org/abs/1506.02438
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2018), arXiv:1812.05905,
  eq. (7): https://arxiv.org/abs/1812.05905
- Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019), arXiv:1910.07207,
  eq. (12): https://arxiv.org/abs/1910.07207
- Amos, "On numerical issues of divergences between nearly equal distributions" (2026),
  arXiv:2611.18511, Section 4.1, "Fixing the forward KL": https://arxiv.org/abs/2611.18511
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean
namespace RL
namespace PolicyGradient

open TorchLean.Tensor

variable {α : Type} [Storage α] [Context α]

/-- Softmax policy induced by a vector of logits. -/
def probabilities {nActions : Nat} (logits : Tensor α [nActions]) :
    Tensor α [nActions] :=
  Tensor.softmax (α := α) logits ⟨0, Nat.one_pos⟩

/-- Guarded selected softmax probability `Context.clamp p epsilon (1 - epsilon)`: a NaN `p`
stays NaN, and `1 - epsilon` wins when `epsilon > 1 - epsilon`.

`epsilon` is used as supplied, without validation of the guard bounds. -/
def probability {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (epsilon : α := Context.defaultEpsilon) : α :=
  let probs := probabilities (α := α) logits
  let p := Tensor.getScalar probs action
  Context.clamp p epsilon ((1 : α) - epsilon)

/-- Logarithm of the guarded selected probability from `probability`. -/
def logProbability {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (epsilon : α := Context.defaultEpsilon) : α :=
  MathFunctions.log (probability (α := α) logits action epsilon)

/-- Selected log-probability `log_softmax(logits)[action]`, unclamped: `Autograd.logProbability`
run by the pure interpreter on a batch of one. `PPO.collect` caches this value and
`PPO.objectiveFromLogits` recomputes it. `PPO.Autograd.objective` records the same program, and
its docstring states when its interpreter reproduces the value bit for bit. -/
def logProb {nActions : Nat} [NeZero nActions] (logits : Tensor α [nActions])
    (action : Fin nActions) : α :=
  let batched : Tensor α [1, nActions] := logits.reshape [1, nActions] (by simp)
  let actions : Tensor (Fin nActions) [1] := Tensor.ofFn fun _ => action
  let selected : Tensor α [1] :=
    Autograd.logProbability (m := Id) (α := α) (batch := 1) batched actions
  Tensor.getScalar selected ⟨0, Nat.one_pos⟩

/-- Guarded entropy bonus `-Σ q(a) log q(a)`, where `p = softmax logits` and
`q = clamp p epsilon (1 - epsilon)`.

The clamped probabilities serve as both the weights and the logarithm inputs and are not
renormalized, so for positive `epsilon` the result can differ from the Shannon entropy of `p`. -/
def entropy {nActions : Nat} (logits : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  let probs := probabilities (α := α) logits
  let clamped := Tensor.clamp probs (Tensor.full _ epsilon) (Tensor.full _ ((1 : α) -
      epsilon)) .interior
  let entropy := Tensor.sum (clamped * Tensor.log clamped)
  Neg.neg entropy

/-- REINFORCE loss `-G_t * log π(a_t | s_t)` for one sampled action, with the guarded
`logProbability` and `returnOrAdvantage` as `G_t` (or an advantage, for a baseline). -/
def reinforceLoss {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (returnOrAdvantage : α) (epsilon : α := Context.defaultEpsilon) : α :=
  Neg.neg (returnOrAdvantage * logProbability (α := α) logits action epsilon)

/-- Value-regression loss `valueCoef * (valuePrediction - valueTarget)²` of actor-critic and PPO
critics. -/
def criticLoss (valuePrediction valueTarget : α) (valueCoef : α := 1) : α :=
  valueCoef * squaredError (α := α) valuePrediction valueTarget

/-- Advantage actor-critic loss: the REINFORCE term with the advantage, plus `criticLoss`, minus
`entropyCoef` times the guarded `entropy`. -/
def actorCriticLoss {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (advantage valuePrediction valueTarget : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Context.defaultEpsilon) : α :=
  reinforceLoss (α := α) logits action advantage epsilon
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropy (α := α) logits epsilon

/-- Importance ratio `π_new(a|s) / π_old(a|s)`, computed as `exp (newLogProb - oldLogProb)`. -/
def ratio (newLogProb oldLogProb : α) : α :=
  MathFunctions.exp (newLogProb - oldLogProb)

/-- `exp l * (exp s - 1 - s)`, a term of `klTilted` whose weight is given by its logarithm `l`,
where `x` is `l + s` and `quarter` is `s / 4` as `klTilted` evaluates them. From `s = -2` to
`s = 1` it is `exp l * Context.expm1x s`, which FloatLib rounds once and never makes negative.
Above `1` it is `exp x - exp l - exp l * s`, where `exp s` alone can overflow, and a weight `exp l`
that underflows to zero still contributes `exp x`. Below `-2` it is
`exp x - exp l - 4 * (exp l * quarter)`, so a log-ratio `s` that overflowed to `-∞` still
contributes its finite share. A term below half the least subnormal rounds to `0` in every order,
as at binary32 `l = -110` and `s = 1/2`, where the term is about `2.51e-49`. -/
def weightedExpm1x (l s x quarter : α) : α :=
  let p := MathFunctions.exp l
  if s > 1 then MathFunctions.exp x - p - p * s
  else if -s > 2 then
    let pq := p * quarter
    MathFunctions.exp x - p - ((pq + pq) + (pq + pq))
  else p * Context.expm1x s

/-- `log (Σ_a exp (l(a) + d(a))) - t` as two parts `(u, r)` whose sum it is, for log-weights `l`
whose exponentials sum to one, exponents `d` at most `0`, some of them possibly `-∞`, `gap` the sum
`Σ_a exp (l(a)) * expm1 (d(a))`, and `e = (d - t) + l` as `klTilted` rounds it, with `t = 0` when
`gap` is above `-1/2`. There `u` is `0` and `r` is `log1p gap`, which keeps the small values of a
near-equality accurate. Otherwise `u` is the first strict maximum of `e` and `r` is
`log (Σ_a exp (e(a) - u))`, the logarithm of a sum of at least one even where the weights at the
largest exponents underflow to zero. The parts stay apart from `t` and each other because `r`, at
most `log nActions`, can be below half a unit in the last place of `t`: binary32 `t = -2^27` and
`r = log 2` round to the sum `-2^27`. -/
def logWeightedMass {nActions : Nat} (gap : α) (e : Tensor α [nActions]) (h : 0 < nActions) :
    α × α :=
  if gap > -((1 : α) / 2) then (0, Context.log1p gap)
  else
    let top := e.getScalar (Softmax.pivot e h)
    (top, MathFunctions.log (Tensor.sum (Tensor.exp (e - Tensor.full _ top))))

/-- A vector less its entry at `Softmax.pivot`, its first strict maximum, so that every entry is
at most `0`. -/
def centered {nActions : Nat} (t : Tensor α [nActions]) (h : 0 < nActions) :
    Tensor α [nActions] :=
  t - Tensor.full _ (t.getScalar (Softmax.pivot t h))

/-- A quarter of `t`, `centered`: `t / 4` less its largest entry. Its entries lie between minus half
the largest finite value and `0`, so the difference of two of them is finite, and so is that
difference `centered` again. -/
def quarterCentered {nActions : Nat} (t : Tensor α [nActions]) (h : 0 < nActions) :
    Tensor α [nActions] :=
  centered (t / Tensor.full _ (4 : α)) h

/--
The relative entropy `KL(p || q) = Σ_a p(a) * (log p(a) - log q(a))` of the tilted distribution
`q ∝ p * exp δ` from the weights `p = exp l`, for log-weights `l` and the shift `δ = a - b`, in the
form of Amos (2026), Section 4.1: `Σ_a p(a) * expm1x (s(a))` with
`s = δ - log (Σ_a p(a) * exp (δ(a)))`, where `expm1x s = exp s - 1 - s`.

The textbook sum subtracts nearly equal logarithms when `q` is close to `p`, and its rounding
error can then exceed the divergence and make it negative. Here each term is `weightedExpm1x`,
which is `p(a) * Context.expm1x (s(a))` from `s(a) = -2` to `1`. Amos centers `δ` by its mean
`Σ_a p(a) * δ(a)`, where `expm1` overflows once the two policies are far apart (binary32 logits
`[0, 0]` and `[-100, 100]`) or a weight is far below the rounding unit (`kl` in binary16 with its
default `epsilon`). This form instead takes `d = 4 * centered (quarterCentered a - quarterCentered
b)`, which is at most `0`, and `gap = Σ_a p(a) * expm1 (d(a))`. When `gap` is above `-1/2`, `t`
is `0`; otherwise `t` is the first strict maximum of `d + l`. Then `e = (d - t) + l`, rounded once
in that order, the parts `(u, r)` of `logWeightedMass gap e`, and `s = ((d - t) - u) - r`. A
quarter of `d` is finite for any finite `a` and `b`, while `d` itself may overflow to `-∞`, and
then `weightedExpm1x` takes the term's linear part from the quarter. The weights enter by their
logarithms, so a weight that underflows to zero keeps its share `exp ((e - u) - r)` of the mass.
That exponent `l + s` comes from the same rounded `e` whose largest entry `logWeightedMass` takes as
`u`, so in the log-sum-exp branch each entry of `e` is at most `u` and `r` is the logarithm of a
sum with the term `exp 0 = 1`, and the exponent is at most `0`; in the `log1p` branch, where `t`
and `u` are `0`, it is at most about `log 2`. That is an argument about the rounding, not a
theorem. `d - t` is exact where `d` is near `t`, so `e` keeps `l` at the precision of the term's
own exponent, not at that of `d`. Two other orders failed: `(d - t) + l - r` with `u` left out
could be positive by about half a unit in the last place of an entry of `d` (binary32
`klFromLogits [0, -384, -2^33] [0, 2^33, 2^32]` gave `+∞`, where the divergence rounds to
`2^33`), and `(d + l - t) - r` lost `l` at the precision of `d`, so `exp (l + s)` could fall below
the linear part `exp l * (1 + s)` (`[0.5625, 0, -2^26]` against `[0.5625, 2.25, -2^25]` gave about
`-0.214`). The value of the form does not depend on the centering. Over ℝ, for log-weights
whose exponentials sum to one, the value is `log (Σ_a p(a) * exp (δ(a))) - Σ_a p(a) * δ(a)`
(`RL.PolicyGradient.klTilted_eq_log_sum_exp`). An empty vector gives `0`.

In floating point no theorem bounds the result. `d` is rounded at the magnitude of its entries
and of the centered quarters of `a` and `b`, so a part of the shift below about half a unit in
the last place of those values is lost; `klFromLogits` gives two such inputs.
-/
def klTilted {nActions : Nat} (logWeights a b : Tensor α [nActions]) : α :=
  if h : 0 < nActions then
    let quarter := centered (quarterCentered a h - quarterCentered b h) h
    let d := (quarter + quarter) + (quarter + quarter)
    let gap := Tensor.sum (Tensor.exp logWeights * Tensor.expm1 d)
    let w := d + logWeights
    let t := if gap > -((1 : α) / 2) then 0 else w.getScalar (Softmax.pivot w h)
    let e := (d - Tensor.full _ t) + logWeights
    let mass := logWeightedMass gap e h
    Tensor.sum (Tensor.ofFn fun i =>
      weightedExpm1x (logWeights.getScalar i) (d.getScalar i - t - mass.1 - mass.2)
        (e.getScalar i - mass.1 - mass.2)
        (quarter.getScalar i - t / (4 : α) - mass.1 / (4 : α) - mass.2 / (4 : α)))
  else 0

/-- The probabilities that `kl` compares: each entry `p` replaced by
`Context.clamp p epsilon (1 - epsilon)`, that is `min (max p epsilon) (1 - epsilon)`, then the
vector divided by its sum. A NaN `p` stays NaN, and `1 - epsilon` wins when `epsilon > 1 - epsilon`:
at `epsilon = 0.75` every entry is `0.25` before the division. The tangent of an entry is cleared
unless `epsilon < p < 1 - epsilon` (`Tensor.clamp` at `.interior`). -/
def guardedProbabilities {nActions : Nat} (probs : Tensor α [nActions]) (epsilon : α) :
    Tensor α [nActions] :=
  let clamped := Tensor.clamp probs (Tensor.full _ epsilon) (Tensor.full _ ((1 : α) - epsilon))
    .interior
  clamped / Tensor.full _ (Tensor.sum clamped)

/--
Categorical KL divergence `KL(old || new) = Σ_a old(a) * (log old(a) - log new(a))` of the
probability vectors after `guardedProbabilities` limits each entry by `epsilon` and `1 - epsilon`
and divides the vector by its sum. It is `klTilted` of the logarithms of the guarded old vector
and the shift `log new - log old` (the second vector `0`).

Over ℝ, for `0 < epsilon < 1`, the result is the sum above for the guarded vectors
(`RL.PolicyGradient.kl_eq_sum`), which is `KL(old || new)` when clamping leaves normalized inputs
unchanged. An empty action vector gives `0`.
-/
def kl {nActions : Nat}
    (oldProbs newProbs : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  let oldLog := Tensor.log (guardedProbabilities oldProbs epsilon)
  let newLog := Tensor.log (guardedProbabilities newProbs epsilon)
  klTilted oldLog (newLog - oldLog) (Tensor.full _ 0)

/--
Categorical KL divergence `KL(softmax old || softmax new)` from logits: `klTilted` of the old
policy's log-softmax and the shift `new - old`, which `klTilted` forms from a quarter of each logit
vector, first `centered` at its own largest entry. `klTilted` does not change when a constant is
added to the shift, so the difference stands in for the log-ratios without either softmax's
normalizer, and nothing is clamped. The old weights enter as the log-softmax, so an old
probability that underflows keeps its share of the mass.

In floating point nothing is proved. `TorchLeanTest.RL.PolicyKL` compares these binary32 inputs,
and only these, with 100-digit values: equal policies at a common offset (`[2^127, 2^127]`
against `[-2^127, -2^127]` gives `0`), a small change (`[1, 1]` against `[0, 2^(-26)]`), old
probabilities that underflow (`[0, -110]`, `[0, -2^27]` and `[0, -2^127]` against `[0, 0]` give
`log 2`), policies as far apart as the format allows (`[0, 0]` against `[-2^127, 2^127]` gives
`2^127`, the divergence `log (cosh (2^127))` rounded), and the two cases of `klTilted`'s exponent
order (`[0, -384, -2^33]` against `[0, 2^33, 2^32]` gives `2^33`, the divergence rounded, and
`[0.5625, 0, -2^26]` against `[0.5625, 2.25, -2^25]` gives about `0.4737`, the second limit
below, where the divergence is about `0.5896`). Rounding the shift in the input format limits the
accuracy, and the tests check these two limits too:
- a change of a logit below about half a unit in the last place of its distance from its
  vector's largest entry reads as `0`: `[0, 1]` against `[2^(-26), 1]` gives `0`, where the
  divergence is about `2.2e-17`;
- an entry of the shift far below its largest entry keeps only the precision of that distance,
  so the result can be off by about half a unit in the last place of the distance: against
  `[0, 5]`, where the divergence is about `5.0067`, `[0, -2^24]` gives `log (1 + e^4)`, about
  `4.0181`, since `-2^24 - 5` rounds to `-2^24 - 4`, and `[0, -2^100]` gives `log 2`.

The old logits' log-softmax overflows to `-∞` when their entries are more than the largest finite
value apart, and then the result can be NaN (`[-2^127, 2^127]` against `[0, 0]`). While they are
at most that far apart, the only infinite values a step forms from finite logits are entries of
`d` in `klTilted`, which `weightedExpm1x` replaces by their quarters, exponents below the most
negative finite value, whose exponential is `0` either way, and a term or a sum beyond the
largest finite value. Each term's exponent is at most about `log 2`, as `klTilted` argues; that
is an argument about the ranges of the steps, not a theorem.

Over ℝ the result is `Σ_a p(a) * (log p(a) - log q(a))` for the softmaxes `p` and `q` of the two
logit vectors (`RL.PolicyGradient.klFromLogits_eq_sum`). An empty action vector gives `0`.
-/
def klFromLogits {nActions : Nat} (oldLogits newLogits : Tensor α [nActions]) : α :=
  klTilted (Tensor.logSoftmax oldLogits ⟨0, Nat.one_pos⟩) newLogits oldLogits

/--
KL-penalized policy-gradient loss `-(ratio * A) + β * KL(old || new)` with `β = penaltyCoef`.

It is the differentiable scalar objective commonly used as a surrogate or a diagnostic for
trust-region updates. TRPO's constrained optimizer step is not implemented.
-/
def klLoss (ratio advantage kl penaltyCoef : α) : α :=
  Neg.neg (ratio * advantage) + penaltyCoef * kl

/--
Finite-action SAC actor objective, minimized over actor logits:
`∑ a, π(a|s) * (temperature * log π(a|s) - Q(s,a))`.

The weights `π(a|s)` come from max-shifted exponentials, and `π(a|s) * log π(a|s)` is computed
from the logit `z` as `p * ((z - max) - log Σ)`. No zero probability reaches a logarithm, and
the gradient flows through the weights and the normalizer. When `z < 0 < max`, where `z - max`
could overflow, `z` and `max` are each multiplied by `p` before the subtraction. An action whose
probability `p` is zero (`Context.isZero p`) contributes the entropy term `0`, the limit of
`p log p` as `p` tends to zero.

For an actor-only gradient, hold `qValues` and `temperature` constant with respect to the actor
parameters. With two critics, pass their pointwise minimum as `qValues`. The sum runs over the
actions of one state, with no average across states.
-/
def sacActorLoss {nActions : Nat} [NeZero nActions]
    (logits qValues : Tensor α [nActions]) (temperature : α) : α := by
  cases nActions with
  | zero => exact False.elim (NeZero.ne 0 rfl)
  | succ n =>
      exact
        let maximum := (Softmax.maxVec logits).item
        let weights := Softmax.maxShiftedExpVec logits
        let normalizer := Tensor.sum weights
        let logNormalizer := MathFunctions.log normalizer
        Tensor.sum <|
          Tensor.dim fun action : Fin (Nat.succ n) =>
            let z := Tensor.getScalar logits action
            let p := Tensor.getScalar weights action / normalizer
            -- A zero weight meets an infinite log-probability, and `p log p` tends to zero.
            let entropyTerm :=
              if Context.isZero p then 0
              else if (0 : α) > z ∧ maximum > (0 : α) then
                (p * z - p * maximum) - p * logNormalizer
              else
                p * ((z - maximum) - logNormalizer)
            Tensor.scalar <|
              temperature * entropyTerm - p * Tensor.getScalar qValues action

/--
PPO clipped surrogate `min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)` from a precomputed importance
ratio, with `ε = clipEps`.

Callers that already hold the ratio, for instance from cached log-probabilities, use it to skip
recomputing the ratio from logits. The clip and the minimum are `Context.clamp` and
`Context.minimum` (IEEE 754-2019 9.6), so `-0` orders below `+0`. A NaN argument gives NaN, as
does a NaN `ratio * advantage` from an infinite ratio and a zero advantage.
-/
def _root_.TorchLean.RL.PPO.objective (ratio advantage clipEps : α) : α :=
  let clippedRatio := Context.clamp ratio ((1 : α) - clipEps) ((1 : α) + clipEps)
  let unclipped := ratio * advantage
  let clipped := clippedRatio * advantage
  Context.minimum unclipped clipped

/-- PPO clipped surrogate for one sampled action, to be maximized:
`min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)` with `r_t = exp (newLogProb - oldLogProb)`.

The new log-probability comes from `logProb`, the value `PPO.collect` caches.
-/
def _root_.TorchLean.RL.PPO.objectiveFromLogits {nActions : Nat} [NeZero nActions]
    (newLogits : Tensor α [nActions])
    (action : Fin nActions) (oldLogProb advantage clipEps : α) : α :=
  let newLogProb := logProb (α := α) newLogits action
  let ratio := ratio (α := α) newLogProb oldLogProb
  PPO.objective (α := α) ratio advantage clipEps

/-- PPO loss to minimize, `-L_clip + c_v * value_loss - c_e * entropy newLogits epsilon`, where
`epsilon` guards only the entropy term and the defaults are `c_v = 1`, `c_e = 0`.
`PPO.Autograd.loss` differs: it uses `Autograd.entropy` and the defaults `1/2` and `1/100`. -/
def _root_.TorchLean.RL.PPO.loss {nActions : Nat} [NeZero nActions]
    (newLogits : Tensor α [nActions])
    (action : Fin nActions) (oldLogProb advantage valuePrediction valueTarget clipEps : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Context.defaultEpsilon) : α :=
  Neg.neg (PPO.objectiveFromLogits (α := α) newLogits action oldLogProb advantage clipEps)
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropy (α := α) newLogits epsilon

/--
Sample a categorical action from probability weights, or from logits (`logits := true`), and
return it with the next counter `counter + 1`.

The draw `u` is the scalar `Random.uniform` keyed by `(seed, counter)`. It lies in `[0, 1]`,
because rounding can return the endpoint `1`. The result is the first action whose cumulative
weight exceeds `u`, and the last action when none does. The weights are used as given, without
validation or normalization.
-/
def sample {nActions : Nat} [NeZero nActions]
    (seed counter : Nat) (values : Tensor α [nActions]) (logits : Bool := false) :
    Nat × Fin nActions :=
  let probs := if logits then probabilities (α := α) values else values
  let key := Random.keyOf seed counter
  let u : α :=
    Tensor.item (Random.uniform (α := α) key (s := Shape.scalar))
  let default : Fin nActions :=
    ⟨nActions - 1, Nat.pred_lt (NeZero.ne nActions)⟩
  Id.run do
    let mut cum : α := 0
    for h : k in [:nActions] do
      let action : Fin nActions := ⟨k, h.2.1⟩
      let pk : α := Tensor.item (Tensor.unstack probs action)
      cum := cum + pk
      if Context.gtBool cum u then
        return (counter + 1, action)
    return (counter + 1, default)

end PolicyGradient
end RL
end TorchLean
