/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Sequential

/-!
# Autograd Policy-Gradient Objectives

Differentiable policy-gradient and actor-critic helpers written against TorchLean's backend-generic
`Ops` interface, so they run under:

- the eager runtime backend (imperative autograd, GPU-capable), and
- typed graph execution (recorded graph and proof tooling).

The module sits in the RL runtime because these are RL objectives that happen to be differentiable
through TorchLean. It is the autograd companion of
`TorchLean.NN.RL.PolicyGradient`, whose pure helpers work with concrete spec
tensors (`Tensor α s`). The helpers here work with backend refs (`Ref (m := m) (α := α) s`), so
autograd can differentiate the objectives.

## Action Encoding

We assume **categorical** (finite-action) policies parameterized by logits. A sampled action is a
`Fin nActions` index carried as non-differentiable data, so an out-of-range action is
unrepresentable. The selected log-probability is read from the log-softmax by
`Loss.nllUnreduced`'s gather, and the pure helpers of
`TorchLean.NN.RL.PolicyGradient` run the same program through the pure
interpreter.

We matched `torch.distributions.Categorical` (PyTorch 2.8 docs) by hand. `logProbability` reads
the log-softmax at the action, as its `log_prob` does. Torch's `entropy` clamps the same log factor
from below at the dtype's most negative finite value, about $-3.4 \cdot 10^{38}$ in float32. Ours
clamps into $[-10^{30}, 10^{30}]$ in every carrier, so in binary16, where $10^{30}$ overflows, the
clamp does nothing. It also averages over the batch, where torch returns one entropy per row.
`PPO.Autograd.objective` is eq. (7) of Schulman et al. (2017) per
sample, and `loss` is their eq. (9), negated and averaged over the batch. Its defaults
`clipEps = 1/5` and `entropyCoef = 1/100` are the $\epsilon = 0.2$ of their Section 3 and the
$c_2 = 0.01$ of their Table 5. The $c_1$ of that table is 1, twice our `valueCoef`. We found no
test that runs these objectives against PyTorch.

## Primary References

- Williams, "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement
  Learning" (1992), Machine Learning 8(3-4):229-256: https://doi.org/10.1023/A:1022672621406
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), Chapters 12 and 13:
  http://incompleteideas.net/book/the-book-2nd.html
- Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
  (2015), ICLR 2016: https://arxiv.org/abs/1506.02438
- Schulman et al., "Proximal Policy Optimization Algorithms" (2017), arXiv:1707.06347, Section 3,
  eqs. (7) and (9) and Table 5: https://arxiv.org/abs/1707.06347
- `torch.distributions.Categorical` (PyTorch 2.8 docs), `log_prob` and `entropy`:
  https://docs.pytorch.org/docs/2.8/distributions.html
-/

@[expose] public section

namespace TorchLean
namespace RL


variable {α : Type} [Storage α] [Context α]

/-! ## Log-probabilities and entropy (batched) -/

/--
Per-sample selected log-probability `log_softmax(logits)[i, actions[i]]` of a batched categorical
policy: the log-softmax along the action axis, read at each action by `Loss.nllUnreduced`'s gather
and negated, which is exact.

Input shapes: `logits : (N × A)` and the actions `(N)` as data. Output shape: `(N)`.

Only the selected entry is read, so an unselected `-inf` logit reaches neither the value nor the
gradient, and no clamp is applied. `PolicyGradient.logProb` runs this program through the pure
interpreter. The eager CPU tape and the typed graph evaluate each operator by its `Op.rule?` rule,
as the pure interpreter does, while the eager CUDA tape runs LibTorch kernels, whose values need
not agree with those bit for bit.
-/
def PolicyGradient.Autograd.logProbability
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (logits :
      Ops.Ref (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actions : Ops.DataRef (m := m) (α := α) (Fin nActions) [batch]) :
    m (Ops.Ref (m := m) (α := α) (.dim batch .scalar)) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.HasNonemptyAxis 1 s :=
    Shape.hasNonemptyAxis_of_lt_rank (by simp [s])
  let logp ← Ops.apply (m := m) (α := α)
      (.logSoftmax (s := s) ⟨1, Shape.AxisInBounds.proof⟩) (logits, ())
  let negated ← Loss.nllUnreduced (m := m) (α := α) (leading := [batch])
    (trailing := []) (classes := nActions) 1 rfl logp actions
  Ops.apply (m := m) (α := α) (.scale (s := [batch]))
      (negated, .data (Ops.dataConst (m := m) (α := α)
        (Tensor.scalar (-1))), ())

/--
Mean categorical entropy with a clamped logarithmic factor.

For `logp = logSoftmax(logits)` and `bound = (10^30 : α)`, this computes
`mean_i[ -Σ_a exp(logp_i(a)) * clamp(logp_i(a), -bound, bound) ]`.
Input shape is `(N × A)` and the output is scalar.

The expression agrees with ordinary entropy when the clamp is inactive. For Float and binary32,
finite tails below the bound already exponentiate to zero, and the finite bound prevents
`0 * -inf = NaN` in both forward and backward. The guard needs a finite representable bound, so it
is ineffective for binary16, where `10^30` overflows. Exact or wider-range carriers can have
nonzero probabilities below the bound, and there the clipped expression can differ from their
entropy. Rows whose log-softmax is itself NaN are not repaired by this guard.
-/
def PolicyGradient.Autograd.entropy
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (logits :
      Ops.Ref (m := m) (α := α) (.dim batch (.dim nActions .scalar))) :
    m (Ops.Ref (m := m) (α := α) Shape.scalar) := do
  let s : Shape := .dim batch (.dim nActions .scalar)
  let _ : Shape.WellFormed s := by infer_instance
  let _ : Shape.HasNonemptyAxis 1 s :=
    Shape.hasNonemptyAxis_of_lt_rank (by simp [s])
  let logp ← Ops.apply (m := m) (α := α)
      (.logSoftmax (s := s) ⟨1, Shape.AxisInBounds.proof⟩) (logits, ())
  let probs ← Ops.apply (m := m) (α := α) (.exp (s := s)) (logp, ())
  let bound : α := ((10 ^ 30 : Nat) : α)
  let finiteLogp ← Ops.apply (m := m) (α := α) (.clamp (s := s))
      (logp, .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-bound))), .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar bound)), ())
  let plogp ← Ops.apply (m := m) (α := α) (.mul (s := s))
      (probs, finiteLogp, ())
  let sumActions ←
      (Ops.Ref.cast (Shape.reduce_singleton s ⟨1, Shape.AxisInBounds.proof⟩) <$>
        Ops.apply (m := m) (α := α)
          (.reduce (s := s) .sum [⟨1, Shape.AxisInBounds.proof⟩] false
            (fun h => by rcases h with h | h <;> cases h)) (plogp, ()))
  let entropyVec ← Ops.apply (m := m) (α := α)
      (.scale (s := .dim batch .scalar))
      (sumActions, .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-1))), ())
  nn.functional.mean (m := m) (α := α) (s := .dim batch .scalar) entropyVec

/-! ## PPO (batched) -/

namespace PPO.Autograd

open PolicyGradient.Autograd (entropy logProbability)

/--
PPO clipped surrogate objective (the thing to maximize), computed per sample:

`L_clip_i = min(r_i * A_i, clip(r_i, 1-ε, 1+ε) * A_i)`

where `r_i = exp(logπ_new(a_i|s_i) - logπ_old(a_i|s_i))`, with the new log-probability from
`logProbability`. When the interpreter running this objective reproduces bit for bit the logits
`PPO.collect` used and the `PolicyGradient.logProb` value it cached, the ratio at unchanged
parameters is that of a value with itself, which for a finite value decodes to exactly one at IEEE
configured formats (`RL.PolicyGradient.toReal_ratio_self`). The eager CPU tape and the typed graph
evaluate the `Op.rule?` rules the pure interpreter evaluates. The eager CUDA tape runs LibTorch
kernels, so under `--device cuda` the ratio at unchanged parameters need not be exactly one.
-/
def objective
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (newLogits :
      Ops.Ref (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actions : Ops.DataRef (m := m) (α := α) (Fin nActions) [batch])
    (oldLogProb : Ops.Ref (m := m) (α := α) (.dim batch .scalar))
    (advantage : Ops.Ref (m := m) (α := α) (.dim batch .scalar))
    (clipEps : α := (1 : α) / ((5 : Nat) : α)) :
    m (Ops.Ref (m := m) (α := α) (.dim batch .scalar)) := do
  let sVec : Shape := .dim batch .scalar
  let newLogProb ←
    logProbability (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actions
  let diff ← Ops.apply (m := m) (α := α) (.sub (s := sVec))
      (newLogProb, oldLogProb, ())
  let ratio ← Ops.apply (m := m) (α := α) (.exp (s := sVec)) (diff, ())
  let clippedRatio ←
    Ops.apply (m := m) (α := α) (.clamp (s := sVec))
        (ratio, .data
          (Ops.dataConst (m := m) (α := α)
            (Tensor.scalar ((1 : α) - clipEps))), .data
          (Ops.dataConst (m := m) (α := α)
            (Tensor.scalar ((1 : α) + clipEps))), ())
  let unclipped ← Ops.apply (m := m) (α := α) (.mul (s := sVec))
      (ratio, advantage, ())
  let clipped ← Ops.apply (m := m) (α := α) (.mul (s := sVec))
      (clippedRatio, advantage, ())
  Ops.apply (m := m) (α := α) (.min (s := sVec)) (unclipped, clipped, ())

/--
PPO scalar loss to *minimize* (mean over batch):

`loss = -mean(L_clip) + c_v * MSE(v, v_target) - c_e * mean(entropy)`

It is the standard discrete-action PPO loss of many reference implementations. The defaults are
`clipEps = 1/5`, `valueCoef = 1/2` for `c_v` and `entropyCoef = 1/100` for `c_e`.
-/
def loss
    {m : Type → Type} [Monad m] [Ops (m := m) (α := α)]
    {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (newLogits :
      Ops.Ref (m := m) (α := α) (.dim batch (.dim nActions .scalar)))
    (actions : Ops.DataRef (m := m) (α := α) (Fin nActions) [batch])
    (oldLogProb : Ops.Ref (m := m) (α := α) (.dim batch .scalar))
    (advantage : Ops.Ref (m := m) (α := α) (.dim batch .scalar))
    (valuePred valueTarget :
      Ops.Ref (m := m) (α := α) (.dim batch (.dim 1 .scalar)))
    (clipEps : α := (1 : α) / ((5 : Nat) : α))
    (valueCoef : α := (1 : α) / ((2 : Nat) : α))
    (entropyCoef : α := (1 : α) / ((100 : Nat) : α)) :
    m (Ops.Ref (m := m) (α := α) Shape.scalar) := do
  let obj ←
    objective (m := m) (α := α) (batch := batch) (nActions := nActions)
      newLogits actions oldLogProb advantage (clipEps := clipEps)
  let objMean ← nn.functional.mean (m := m) (α := α) (s := .dim batch .scalar) obj
  let policyLoss ← Ops.apply (m := m) (α := α) (.scale (s := Shape.scalar))
      (objMean, .data (Ops.dataConst (m := m) (α := α) (Tensor.scalar (-1))),
        ())
  let valueLoss ←
    Loss.mse (m := m) (α := α) (s := .dim batch (.dim 1 .scalar)) valuePred valueTarget
  let valueLossScaled ← Ops.apply (m := m) (α := α)
      (.scale (s := Shape.scalar))
      (valueLoss, .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar valueCoef)), ())
  let entropy ← entropy (m := m) (α := α) (batch := batch) (nActions := nActions) newLogits
  let entropyScaled ← Ops.apply (m := m) (α := α)
      (.scale (s := Shape.scalar))
      (entropy, .data
        (Ops.dataConst (m := m) (α := α) (Tensor.scalar entropyCoef)), ())
  let policyValueLoss ← Ops.apply (m := m) (α := α)
      (.add (s := Shape.scalar)) (policyLoss, valueLossScaled, ())
  Ops.apply (m := m) (α := α) (.sub (s := Shape.scalar))
      (policyValueLoss, entropyScaled, ())

/-! ## PPO module wrapper (two-model actor/critic) -/

/--
Bundle an actor and critic into an `ObjectiveDef` whose inputs are a PPO minibatch:

- `states : (N × stateDim)`
- `oldLogProb : (N)`
- `advantages : (N)`
- `valueTarget : (N × 1)`

and whose data input is the sampled actions `(N)` as `Fin A`. The model state is
`actor.state ++ critic.state`, and one optimizer step updates both models.
-/
def create
    {stateShape : Shape} {batch nActions : Nat} [NeZero batch] [NeZero nActions]
    (actor : nn.Sequential stateShape (.dim batch (.dim nActions .scalar)))
    (critic : nn.Sequential stateShape (.dim batch (.dim 1 .scalar))) :
    ObjectiveDef (Fin nActions)
      (nn.Sequential.stateShapes actor ++
        nn.Sequential.stateShapes critic)
      [stateShape, (.dim batch .scalar), (.dim batch .scalar), (.dim batch (.dim 1 .scalar))]
      [[batch]] :=
  { initState :=
      Pack.append
        (ss₁ := nn.Sequential.stateShapes actor)
        (ss₂ := nn.Sequential.stateShapes critic)
        (nn.Sequential.initState actor)
        (nn.Sequential.initState critic)
    requiresGrad := nn.Sequential.requiresGrad actor ++
      nn.Sequential.requiresGrad critic
    validate := do
      nn.Sequential.validate actor
      nn.Sequential.validate critic
    loss := fun {α} => by
      intro _ _; exact
        (fun {m} _ _ =>
          CurriedRef.curry
            (Ref := fun sh => Ops.Ref (m := m) (α := α) sh)
            (ss := (nn.Sequential.stateShapes actor ++
              nn.Sequential.stateShapes critic) ++
              [stateShape, (.dim batch .scalar), (.dim batch .scalar),
                (.dim batch (.dim 1 .scalar))])
            (β := Ops.DataRef (m := m) (α := α) (Fin nActions) [batch] →
              m (Ops.Ref (m := m) (α := α) Shape.scalar))
            (fun args actions => do
              let (ps, xs) :=
                Pack.split
                  (F := fun sh => Ops.Ref (m := m) (α := α) sh)
                  (ss₁ := (nn.Sequential.stateShapes actor ++
                    nn.Sequential.stateShapes critic))
                  (ss₂ := [stateShape, (.dim batch .scalar), (.dim batch .scalar),
                    (.dim batch (.dim 1 .scalar))])
                  args
              let (psActor, psCritic) :=
                Pack.split
                  (F := fun sh => Ops.Ref (m := m) (α := α) sh)
                  (ss₁ := nn.Sequential.stateShapes actor)
                  (ss₂ := nn.Sequential.stateShapes critic)
                  ps
              let .cons states (.cons oldLogProb
                  (.cons advantages (.cons valueTarget .nil))) := xs
              let logits ←
                nn.Sequential.forwardState
                  (model := actor) (α := α) (m := m) .train psActor states
              let values ←
                nn.Sequential.forwardState
                  (model := critic) (α := α) (m := m) .train psCritic states
              loss (m := m) (α := α) (batch := batch) (nActions := nActions)
                logits actions oldLogProb advantages values valueTarget))
  }

end PPO.Autograd
end RL
end TorchLean
