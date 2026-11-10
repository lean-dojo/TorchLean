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
computes the expectation over actions in eq. (7) of Haarnoja et al. as a sum.
`RL.PolicyGradient.toReal_ratio_self` proves that `ratio x x` decodes to 1 for a finite `x` at
IEEE formats. No TorchLeanTest test runs these objectives.

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

/--
Categorical KL divergence `Σ_a old(a) * (log old(a) - log new(a))` after each probability vector
is clamped into `[epsilon, 1-epsilon]` and divided by its sum.

For finite inputs and `0 < epsilon < 1/2` the guarded vectors are probability distributions, and
the result is `KL(old || new)` up to floating-point rounding when clamping leaves normalized
inputs unchanged. An empty action vector gives `0`.
-/
def kl {nActions : Nat}
    (oldProbs newProbs : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  match nActions with
  | 0 => 0
  | Nat.succ _ =>
      let oldClamped := Tensor.clamp oldProbs (Tensor.full _ epsilon) (Tensor.full _ ((1 : α) -
          epsilon)) .interior
      let newClamped := Tensor.clamp newProbs (Tensor.full _ epsilon) (Tensor.full _ ((1 : α) -
          epsilon)) .interior
      let oldNormalized := oldClamped / Tensor.full _ (Tensor.sum oldClamped)
      let newNormalized := newClamped / Tensor.full _ (Tensor.sum newClamped)
      Tensor.sum (oldNormalized *
        (Tensor.log oldNormalized - Tensor.log newNormalized))

/--
Categorical KL divergence from logits, using the clamped, normalized policies of
`kl`.
-/
def klFromLogits {nActions : Nat}
    (oldLogits newLogits : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  kl (α := α)
    (oldProbs := probabilities (α := α) oldLogits)
    (newProbs := probabilities (α := α) newLogits)
    (epsilon := epsilon)

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
