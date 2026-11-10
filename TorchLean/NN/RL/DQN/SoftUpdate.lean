/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL.DQN
public import TorchLean.Tensor.Context.Real

/-!
# DQN Runtime Algebra Proofs

Algebraic facts about the DQN runtime helpers, chiefly the soft target-network update
`updateTarget tau online target = tau * online + (1 - tau) * target`. The helper asks only for
`One`, `Add`, `Sub` and `Mul`, with no laws, and floating-point arithmetic is not a ring, so the
identities are stated over `ℝ`.

We proved them by hand, with `simp` and `ring` after unfolding `updateTarget`. Against a fixed
online value, `updateTarget_sub_target` leaves `1 - tau` of the gap after each update, so at
`tau = 0.001` half of it is gone after 693 updates (0.999^693 ≈ 0.4999).

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015), Nature
  518(7540):529-533, Methods, target networks: https://doi.org/10.1038/nature14236
- Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992), SIAM
  Journal on Control and Optimization 30(4):838-855, averaging: https://doi.org/10.1137/0330046
-/

@[expose] public section

namespace TorchLean
namespace RL
namespace DQN

/-- With `τ = 0`, a soft target update leaves the target unchanged. -/
@[simp] theorem updateTarget_zero (online target : ℝ) :
    RL.DQN.updateTarget (α := ℝ) 0 online target = target := by
  simp [RL.DQN.updateTarget]

/-- With `τ = 1`, a soft target update copies the online value. -/
@[simp] theorem updateTarget_one (online target : ℝ) :
    RL.DQN.updateTarget (α := ℝ) 1 online target = online := by
  simp [RL.DQN.updateTarget]

/-- When online and target agree, a soft update returns that common value for every `tau`. -/
@[simp] theorem updateTarget_self (tau x : ℝ) :
    RL.DQN.updateTarget (α := ℝ) tau x x = x := by
  unfold RL.DQN.updateTarget
  ring

/--
A soft update moves the target by `tau` times the online-target gap:
`updateTarget tau online target - target = tau * (online - target)`.

The identity explains target-network lag: with a small `tau`, each update closes only that
fraction of the gap.
-/
theorem updateTarget_sub_target (tau online target : ℝ) :
    RL.DQN.updateTarget (α := ℝ) tau online target - target =
      tau * (online - target) := by
  unfold RL.DQN.updateTarget
  ring

end DQN
end RL
end TorchLean
