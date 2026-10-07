/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Spec.Core.Random

/-!
# Policy-Gradient Objectives

This module exposes typed helpers for the main categorical-policy objectives that modern
policy-gradient code tends to rely on:

- REINFORCE,
- advantage actor-critic,
- trust-region / KL-penalized policy-gradient helpers,
- entropy regularization,
- soft actor-critic policy terms,
- PPO's clipped surrogate.

The helpers operate on logits for a finite action space and stay purely functional so they can be
used from either eager runtime code or proof-oriented spec code.

Primary references:

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992): https://doi.org/10.1023/A:1022672621406
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016):
  https://arxiv.org/abs/1602.01783
- Schulman et al., "Trust Region Policy Optimization" (2015):
  https://arxiv.org/abs/1502.05477
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017):
  https://arxiv.org/abs/1707.06347
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015): https://arxiv.org/abs/1506.02438
-/

@[expose] public section

namespace Runtime
namespace RL
namespace PolicyGradient

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- Softmax policy induced by a vector of logits. -/
def probabilities {nActions : Nat} (logits : Tensor α [nActions]) :
    Tensor α [nActions] :=
  Activation.softmaxVecSpec (α := α) (n := nActions) logits

/-- Guarded selected softmax probability `min (1 - epsilon) (max epsilon p)`.

`epsilon` is used as supplied; the guard bounds are not validated. -/
def probability {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (epsilon : α := Context.defaultEpsilon) : α :=
  let probs := probabilities (α := α) logits
  let p := Tensor.getScalar probs action
  Min.min ((1 : α) - epsilon) (Max.max epsilon p)

/-- Logarithm of the guarded selected probability from `probability`. -/
def logProbability {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (epsilon : α := Context.defaultEpsilon) : α :=
  MathFunctions.log (probability (α := α) logits action epsilon)

/-- Unclamped log-probability `log_softmax(logits)[action]`.

`Autograd.logProbability` additionally clamps log-probabilities to `[-10^30, 10^30]`
before one-hot multiplication. This helper agrees with that selected value only when the clamp
is inactive and the reductions agree. Rollouts used by that objective must apply the same clamp
to cached old log-probabilities; an unclamped tail value can otherwise produce a nonunit ratio
even when the policy parameters are unchanged. -/
def logSoftmax {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) : α :=
  Tensor.getScalar (Activation.logSoftmaxVecSpec (α := α) (n := nActions) logits) action

/-- Guarded entropy bonus `-Σ q(a) log q(a)`, where `p = softmax logits` and
`q = clamp p epsilon (1 - epsilon)`.

Clamped probabilities are used both as weights and as logarithm inputs, without
renormalization. For positive `epsilon`, this can differ from the Shannon entropy of `p`. -/
def entropy {nActions : Nat} (logits : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  let probs := probabilities (α := α) logits
  let clamped := clampSpec probs epsilon ((1 : α) - epsilon)
  let entropy := sumSpec (mulSpec clamped (logSpec clamped))
  Neg.neg entropy

/-- REINFORCE loss for one sampled action:
`-G_t * log π(a_t | s_t)`. -/
def reinforceLoss {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (returnOrAdvantage : α) (epsilon : α := Context.defaultEpsilon) : α :=
  Neg.neg (returnOrAdvantage * logProbability (α := α) logits action epsilon)

/-- Value-regression loss used by actor-critic and PPO critics. -/
def criticLoss (valuePrediction valueTarget : α) (valueCoef : α := 1) : α :=
  valueCoef * Core.squaredError (α := α) valuePrediction valueTarget

/-- Combined advantage actor-critic loss:
policy term + value regression - entropy bonus. -/
def actorCriticLoss {nActions : Nat} (logits : Tensor α [nActions])
    (action : Fin nActions) (advantage valuePrediction valueTarget : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Context.defaultEpsilon) : α :=
  reinforceLoss (α := α) logits action advantage epsilon
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropy (α := α) logits epsilon

/-- Importance ratio `π_new(a|s) / π_old(a|s)` computed from log-probabilities. -/
def ratio (newLogProb oldLogProb : α) : α :=
  MathFunctions.exp (newLogProb - oldLogProb)

/--
Categorical KL divergence after clamping and normalizing both probability vectors.

For finite inputs and `0 < epsilon < 1/2`, clamp each vector into `[epsilon, 1-epsilon]`,
then divide by its sum before computing `Σ_a old(a) * (log old(a) - log new(a))`.
Normalization keeps the guarded inputs probability distributions. The result agrees with
`KL(old || new)` when clamping leaves normalized inputs unchanged, up to floating-point rounding.
Empty action vectors return zero.
-/
def kl {nActions : Nat}
    (oldProbs newProbs : Tensor α [nActions])
    (epsilon : α := Context.defaultEpsilon) : α :=
  match nActions with
  | 0 => 0
  | Nat.succ _ =>
      let oldClamped := clampSpec oldProbs epsilon ((1 : α) - epsilon)
      let newClamped := clampSpec newProbs epsilon ((1 : α) - epsilon)
      let oldNormalized := divSpec oldClamped (replicate (Tensor.scalar (sumSpec oldClamped)))
      let newNormalized := divSpec newClamped (replicate (Tensor.scalar (sumSpec newClamped)))
      sumSpec (mulSpec oldNormalized
        (subSpec (logSpec oldNormalized) (logSpec newNormalized)))

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
KL-penalized policy-gradient loss:
`-(ratio * A) + β * KL(old || new)`.

This is not the full constrained TRPO optimizer; it is the differentiable scalar objective commonly
used as a practical surrogate or diagnostic when implementing trust-region updates.
-/
def klLoss (ratio advantage kl penaltyCoef : α) : α :=
  Neg.neg (ratio * advantage) + penaltyCoef * kl

/--
Finite-action SAC actor objective, minimized over actor logits:
`∑ a, π(a|s) * (temperature * log π(a|s) - Q(s,a))`.

Max-shifted exponentials supply normalized policy weights. When a negative logit and positive
maximum could overflow their difference, weight each before subtracting. Otherwise keep the
usual shifted expression. Temperature multiplies the weighted entropy term. These rearrangements
retain differentiation through the weights and normalization without taking the log of a zero
probability.

For an actor-only gradient, callers hold `qValues` and `temperature` constant with respect to actor
parameters. If two critics are used, pass their pointwise minimum as `qValues`. This function sums
over actions for one state without averaging across states.
-/
def sacActorLoss {nActions : Nat} [NeZero nActions]
    (logits qValues : Tensor α [nActions]) (temperature : α) : α := by
  cases nActions with
  | zero => exact False.elim (NeZero.ne 0 rfl)
  | succ n =>
      exact
        let maximum := (Activation.maxVecSpec logits).item
        let weights := Activation.maxShiftedExpVecSpec logits
        let normalizer := Tensor.sumSpec weights
        let logNormalizer := MathFunctions.log normalizer
        Tensor.sumSpec <|
          Tensor.dim fun action : Fin (Nat.succ n) =>
            let z := Tensor.getScalar logits action
            let p := Tensor.getScalar weights action / normalizer
            let entropyTerm :=
              if (0 : α) > z ∧ maximum > (0 : α) then
                (p * z - p * maximum) - p * logNormalizer
              else
                p * ((z - maximum) - logNormalizer)
            Tensor.scalar <|
              temperature * entropyTerm - p * Tensor.getScalar qValues action

namespace PPO

/--
PPO clipped surrogate objective from a precomputed importance ratio:

`min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`.

This helper is useful when you already have the ratio (e.g. from cached log-probabilities) and want
to avoid recomputing it from logits.
-/
def objective (ratio advantage clipEps : α) : α :=
  let clippedRatio := Min.min ((1 : α) + clipEps) (Max.max ((1 : α) - clipEps) ratio)
  let unclipped := ratio * advantage
  let clipped := clippedRatio * advantage
  Min.min unclipped clipped

/-- PPO clipped surrogate objective for one sampled action.

This is the objective to maximize:
`min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)`.
-/
def objectiveFromLogits {nActions : Nat} (newLogits : Tensor α [nActions])
    (action : Fin nActions) (oldLogProb advantage clipEps : α)
    (epsilon : α := Context.defaultEpsilon) : α :=
  let newLogProb := logProbability (α := α) newLogits action epsilon
  let ratio := ratio (α := α) newLogProb oldLogProb
  objective (α := α) ratio advantage clipEps

/-- PPO loss to minimize:
`-L_clip + c_v * value_loss - c_e * entropy`. -/
def loss {nActions : Nat} (newLogits : Tensor α [nActions])
    (action : Fin nActions) (oldLogProb advantage valuePrediction valueTarget clipEps : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Context.defaultEpsilon) : α :=
  Neg.neg (objectiveFromLogits (α := α) newLogits action oldLogProb advantage clipEps epsilon)
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropy (α := α) newLogits epsilon

end PPO

/--
Sample a categorical action from probability weights, or from logits (`logits := true`).

`seed` and `counter` form an explicit RNG stream identifier. The function returns the incremented
counter together with the sampled action index.

Probability weights are used without validation or normalization. The sampler scans their
cumulative sum in action order and falls back to the final action if no earlier threshold is
crossed.
-/
def sample {nActions : Nat} [NeZero nActions]
    (seed counter : Nat) (values : Tensor α [nActions]) (logits : Bool := false) :
    Nat × Fin nActions :=
  let probs := if logits then probabilities (α := α) values else values
  let key := Spec.Random.keyOf seed counter
  let u : α :=
    Tensor.item (Spec.Random.uniform (α := α) key (s := Shape.scalar))
  let default : Fin nActions :=
    ⟨nActions - 1, Nat.pred_lt (NeZero.ne nActions)⟩
  Id.run do
    let mut cum : α := 0
    for h : k in [:nActions] do
      let action : Fin nActions := ⟨k, h.2.1⟩
      let pk : α := Tensor.item (get probs action)
      cum := cum + pk
      if Context.gtBool cum u then
        return (counter + 1, action)
    return (counter + 1, default)

end PolicyGradient
end RL
end Runtime
