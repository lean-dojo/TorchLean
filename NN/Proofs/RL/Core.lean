/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.RL.Core
public import Mathlib.Algebra.Order.Field.Basic

/-!
# RL Core Proofs

The shared tensor horizon enforces trajectory alignment in the type. The pointwise law below states
what each reconstructed return means, rather than restating a container-size invariant.

The one-step facts about `discountedBackup` over `ℝ` live here because every Bellman development
(deterministic, finite stochastic, Markov kernel) builds its state-action value from it. Each of
those files only has to bound its bootstrap term; the discount and the terminal mask are handled
once.
-/

@[expose] public section

namespace Proofs
namespace RL
namespace Core

/-- Every lambda-return is its advantage plus its baseline value at the same timestep. -/
theorem returnsFromAdvantages_getScalar {α : Type} [TorchLean.Storage α] [Add α] {n : Nat}
    (advantages values : TorchLean.Tensor α [n]) (index : Fin n) :
    (Spec.RL.returnsFromAdvantages advantages values).getScalar index =
      advantages[index] + values[index] := by
  simp [Spec.RL.returnsFromAdvantages]

open Spec.RL

/-- `continueMask` is always nonnegative. -/
theorem continueMask_nonneg (done : Bool) : 0 ≤ (continueMask (α := ℝ) done : ℝ) := by
  cases done <;> norm_num [continueMask]

/-- A backup is monotone in its bootstrap value when the discount is nonnegative. -/
theorem discountedBackup_mono {reward gamma b₁ b₂ : ℝ} (done : Bool) (hγ : 0 ≤ gamma)
    (hb : b₁ ≤ b₂) :
    discountedBackup reward gamma b₁ done ≤ discountedBackup reward gamma b₂ done := by
  unfold discountedBackup
  gcongr
  exact mul_nonneg hγ (continueMask_nonneg done)

/-- A backup is `gamma`-Lipschitz in its bootstrap value. -/
theorem discountedBackup_abs_sub_le {reward gamma b₁ b₂ bound : ℝ} (done : Bool)
    (hγ : 0 ≤ gamma) (hbound : 0 ≤ bound) (hb : |b₁ - b₂| ≤ bound) :
    |discountedBackup reward gamma b₁ done - discountedBackup reward gamma b₂ done| ≤
      gamma * bound := by
  cases done
  · simpa [discountedBackup, continueMask, ← mul_sub, abs_mul, abs_of_nonneg hγ] using
      mul_le_mul_of_nonneg_left hb hγ
  · simpa [discountedBackup, continueMask] using mul_nonneg hγ hbound

/-- A distance that is at most `gamma` times itself, with `gamma < 1`, is zero. -/
theorem eq_zero_of_le_mul_self {gamma d : ℝ} (hγ : gamma < 1) (hd : 0 ≤ d)
    (h : d ≤ gamma * d) : d = 0 := by
  nlinarith

end Core
end RL
end Proofs
