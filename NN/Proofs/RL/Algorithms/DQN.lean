/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Data.Real.Basic
public import NN.Runtime.RL.Algorithms.DQN

/-!
# DQN Runtime Algebra Proofs

This module proves small but high-value algebraic facts about the DQN runtime helpers.

The main target is target-network soft updates. TorchLean's scalar runtime interface is intentionally
law-light, so algebraic identities are stated over `ℝ`, where ring laws are available. This avoids
claiming that arbitrary executable scalar backends satisfy mathematical ring axioms definitionally.

References:
- Mnih et al., "Human-level control through deep reinforcement learning" (2015), target networks.
- Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging" (1992), averaging.
-/

@[expose] public section

namespace Proofs
namespace RL
namespace DQN

/-- With `τ = 0`, a soft target update leaves the target unchanged. -/
@[simp] theorem softUpdateScalar_zero_tau_real (online target : ℝ) :
    Runtime.RL.DQN.softUpdateScalar (α := ℝ) 0 online target = target := by
  simp [Runtime.RL.DQN.softUpdateScalar]

/-- With `τ = 1`, a soft target update copies the online value. -/
@[simp] theorem softUpdateScalar_one_tau_real (online target : ℝ) :
    Runtime.RL.DQN.softUpdateScalar (α := ℝ) 1 online target = online := by
  simp [Runtime.RL.DQN.softUpdateScalar]

/-- If online and target parameters agree, soft update keeps that fixed point. -/
@[simp] theorem softUpdateScalar_fixed_point_real (tau x : ℝ) :
    Runtime.RL.DQN.softUpdateScalar (α := ℝ) tau x x = x := by
  unfold Runtime.RL.DQN.softUpdateScalar
  ring

/--
Soft update is exactly a convex-combination difference identity:

`softUpdate τ online target - target = τ * (online - target)`.

This is the key algebraic fact behind target-network lag: each update moves the target by a
`τ`-scaled online-target gap.
-/
theorem softUpdateScalar_sub_target_real (tau online target : ℝ) :
    Runtime.RL.DQN.softUpdateScalar (α := ℝ) tau online target - target =
      tau * (online - target) := by
  unfold Runtime.RL.DQN.softUpdateScalar
  ring

end DQN
end RL
end Proofs
