/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.RL.Core
public import NN.Spec.Layers.Activation
public import NN.Runtime.Autograd.TorchLean.Random

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

open Spec
open Tensor

variable {α : Type} [Context α]

/-- Softmax policy induced by a vector of logits. -/
def actionPolicy {nActions : Nat} (logits : Tensor α (.dim nActions .scalar)) :
    Tensor α (.dim nActions .scalar) :=
  Activation.softmaxVecSpec (α := α) (n := nActions) logits

/-- Probability of a selected action under a categorical policy. -/
def actionProbability {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (epsilon : α := Numbers.epsilon) : α :=
  let probs := actionPolicy (α := α) logits
  let p := Tensor.vecGet probs action
  Min.min ((1 : α) - epsilon) (Max.max epsilon p)

/-- Log-probability of a selected action. -/
def actionLogProbability {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (epsilon : α := Numbers.epsilon) : α :=
  MathFunctions.log (actionProbability (α := α) logits action epsilon)

/-- Entropy bonus for a categorical policy:
`-Σ p(a) log p(a)`. -/
def entropyBonus {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (epsilon : α := Numbers.epsilon) : α :=
  let probs := actionPolicy (α := α) logits
  let clamped := clampSpec probs epsilon ((1 : α) - epsilon)
  let entropy := sumSpec (mulSpec clamped (logSpec clamped))
  Neg.neg entropy

/-- REINFORCE loss for one sampled action:
`-G_t * log π(a_t | s_t)`. -/
def reinforceLoss {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (returnOrAdvantage : α) (epsilon : α := Numbers.epsilon) : α :=
  Neg.neg (returnOrAdvantage * actionLogProbability (α := α) logits action epsilon)

/-- Advantage actor-critic policy loss:
`-A_t * log π(a_t | s_t)`. -/
def actorLoss {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (advantage : α) (epsilon : α := Numbers.epsilon) : α :=
  reinforceLoss (α := α) logits action advantage epsilon

/-- Value-regression loss used by actor-critic and PPO critics. -/
def criticLoss (valuePrediction valueTarget : α) (valueCoef : α := 1) : α :=
  valueCoef * Core.squaredError (α := α) valuePrediction valueTarget

/-- Combined advantage actor-critic loss:
policy term + value regression - entropy bonus. -/
def actorCriticLoss {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (advantage valuePrediction valueTarget : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Numbers.epsilon) : α :=
  actorLoss (α := α) logits action advantage epsilon
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropyBonus (α := α) logits epsilon

/-- Advantage actor-critic loss with an explicit entropy bonus coefficient.

This is the A2C/A3C-shaped single-sample objective:
`-A_t log π(a_t|s_t) + c_v value_loss - c_e H(π(.|s_t))`.
-/
def a2cLoss {nActions : Nat} (logits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (advantage valuePrediction valueTarget : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Numbers.epsilon) : α :=
  actorCriticLoss (α := α) logits action advantage valuePrediction valueTarget valueCoef entropyCoef
    epsilon

/-- Importance ratio `π_new(a|s) / π_old(a|s)` computed from log-probabilities. -/
def importanceRatio (newLogProb oldLogProb : α) : α :=
  MathFunctions.exp (newLogProb - oldLogProb)

/--
Categorical KL divergence `KL(old || new)` from two probability vectors:
`Σ_a old(a) * (log old(a) - log new(a))`.

Both distributions are clamped into `[epsilon, 1-epsilon]` before taking logs. This is the scalar
penalty used by TRPO-style diagnostics and KL-penalized policy-gradient objectives.
-/
def categoricalKL {nActions : Nat}
    (oldProbs newProbs : Tensor α (.dim nActions .scalar))
    (epsilon : α := Numbers.epsilon) : α :=
  let oldClamped := clampSpec oldProbs epsilon ((1 : α) - epsilon)
  let newClamped := clampSpec newProbs epsilon ((1 : α) - epsilon)
  sumSpec (mulSpec oldClamped (subSpec (logSpec oldClamped) (logSpec newClamped)))

/-- KL divergence `KL(π_old(.|s) || π_new(.|s))` from old/new logits. -/
def categoricalKLFromLogits {nActions : Nat}
    (oldLogits newLogits : Tensor α (.dim nActions .scalar))
    (epsilon : α := Numbers.epsilon) : α :=
  categoricalKL (α := α)
    (oldProbs := actionPolicy (α := α) oldLogits)
    (newProbs := actionPolicy (α := α) newLogits)
    (epsilon := epsilon)

/--
TRPO-style surrogate objective from a precomputed importance ratio:
`ratio * A`.

TRPO maximizes this surrogate subject to a KL trust-region constraint. We expose the scalar
surrogate separately from the constraint so callers can choose line search / penalty / diagnostics.
-/
def trpoSurrogateFromRatio (ratio advantage : α) : α :=
  ratio * advantage

/--
KL-penalized policy-gradient loss:
`-(ratio * A) + β * KL(old || new)`.

This is not the full constrained TRPO optimizer; it is the differentiable scalar objective commonly
used as a practical surrogate or diagnostic when implementing trust-region updates.
-/
def klPenalizedPolicyLoss (ratio advantage kl penaltyCoef : α) : α :=
  Neg.neg (trpoSurrogateFromRatio (α := α) ratio advantage) + penaltyCoef * kl

/--
Soft actor-critic categorical actor objective:
`temperature * log π(a|s) - Q(s,a)`.

For continuous SAC the action is reparameterized; for finite actions this scalar is the sampled-action
form used inside a categorical policy update.
-/
def sacCategoricalActorLoss {nActions : Nat}
    (logits qValues : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (temperature : α)
    (epsilon : α := Numbers.epsilon) : α :=
  temperature * actionLogProbability (α := α) logits action epsilon
    - Tensor.vecGet qValues action

/--
PPO clipped surrogate objective from a precomputed importance ratio:

`min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`.

This helper is useful when you already have the ratio (e.g. from cached log-probabilities) and want
to avoid recomputing it from logits.
-/
def ppoClippedObjectiveFromRatio (ratio advantage clipEps : α) : α :=
  let clippedRatio := Min.min ((1 : α) + clipEps) (Max.max ((1 : α) - clipEps) ratio)
  let unclipped := ratio * advantage
  let clipped := clippedRatio * advantage
  Min.min unclipped clipped

/-- PPO clipped surrogate objective for one sampled action.

This is the objective to maximize:
`min(r_t A_t, clip(r_t, 1-ε, 1+ε) A_t)`.
-/
def ppoClippedObjective {nActions : Nat} (newLogits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (oldLogProb advantage clipEps : α)
    (epsilon : α := Numbers.epsilon) : α :=
  let newLogProb := actionLogProbability (α := α) newLogits action epsilon
  let ratio := importanceRatio (α := α) newLogProb oldLogProb
  ppoClippedObjectiveFromRatio (α := α) ratio advantage clipEps

/-- PPO loss to minimize:
`-L_clip + c_v * value_loss - c_e * entropy`. -/
def ppoLoss {nActions : Nat} (newLogits : Tensor α (.dim nActions .scalar))
    (action : Fin nActions) (oldLogProb advantage valuePrediction valueTarget clipEps : α)
    (valueCoef : α := 1) (entropyCoef : α := 0)
    (epsilon : α := Numbers.epsilon) : α :=
  Neg.neg (ppoClippedObjective (α := α) newLogits action oldLogProb advantage clipEps epsilon)
    + criticLoss (α := α) valuePrediction valueTarget valueCoef
    - entropyCoef * entropyBonus (α := α) newLogits epsilon

/--
Sample from a categorical distribution represented as a probability vector.

`seed` and `counter` form an explicit RNG stream identifier. The function returns the incremented
counter together with the sampled action index.

Implementation note: this uses the standard cumulative-sum / inverse-CDF sampler.
-/
def sampleCategorical {nActions : Nat} [Fact (0 < nActions)]
    (seed counter : Nat) (probs : Tensor α (.dim nActions .scalar)) :
    Nat × Fin nActions :=
  let key := _root_.Runtime.Autograd.TorchLean.Random.keyOf seed counter
  let u : α :=
    Tensor.toScalar (_root_.Runtime.Autograd.TorchLean.Random.uniform (α := α) key (s := Shape.scalar))
  let default : Fin nActions :=
    ⟨nActions - 1, Nat.pred_lt (Nat.ne_of_gt (Fact.out : 0 < nActions))⟩
  Id.run do
    let idxs : Array (Fin nActions) := Array.ofFn (fun i => i)
    let mut cum : α := 0
    let mut chosen : Option (Fin nActions) := none
    for k in idxs do
      if chosen.isNone then
        let pk : α := Tensor.toScalar (get probs k)
        cum := cum + pk
        if Context.gtBool cum u then
          chosen := some k
    return (counter + 1, chosen.getD default)

/-- Sample an action from logits by applying softmax then `sampleCategorical`. -/
def sampleActionFromLogits {nActions : Nat} [Fact (0 < nActions)]
    (seed counter : Nat) (logits : Tensor α (.dim nActions .scalar)) :
    Nat × Fin nActions :=
  sampleCategorical (α := α) (nActions := nActions) (seed := seed) (counter := counter)
    (probs := actionPolicy (α := α) logits)

end PolicyGradient
end RL
end Runtime
