/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.RL
public import Mathlib.Algebra.Order.Field.Basic

/-!
# RL Core Proofs

Advantages, values and returns share the tensor shape `[n]`, so their lengths agree by type and the
returns theorem can be pointwise: it says what each reconstructed return means.

The one-step facts about `discountedBackup` over `ℝ` live here because every Bellman development
(deterministic, finite stochastic, Markov kernel) builds its state-action value from it. Each of
those files then bounds only its bootstrap term, and the discount and the terminal case are
handled once, here.
-/

@[expose] public section

namespace TorchLean
namespace RL

/-- Every lambda-return is its advantage plus its baseline value at the same timestep. -/
theorem getScalar_returnsFromAdvantages {α : Type} [Storage α] [Add α] {n : Nat}
    (advantages values : Tensor α [n]) (index : Fin n) :
    (RL.returnsFromAdvantages advantages values).getScalar index =
      advantages[index] + values[index] := by
  simp [RL.returnsFromAdvantages]

/-- A backup is monotone in its bootstrap value when the discount is nonnegative. -/
theorem discountedBackup_mono {reward gamma b₁ b₂ : ℝ} (done : Bool) (hγ : 0 ≤ gamma)
    (hb : b₁ ≤ b₂) :
    discountedBackup reward gamma b₁ done ≤ discountedBackup reward gamma b₂ done := by
  cases done
  · simpa [discountedBackup] using mul_le_mul_of_nonneg_left hb hγ
  · simp [discountedBackup]

/-- For `0 ≤ gamma`, a backup is `gamma`-Lipschitz in its bootstrap value. -/
theorem discountedBackup_abs_sub_le {reward gamma b₁ b₂ bound : ℝ} (done : Bool)
    (hγ : 0 ≤ gamma) (hb : |b₁ - b₂| ≤ bound) :
    |discountedBackup reward gamma b₁ done - discountedBackup reward gamma b₂ done| ≤
      gamma * bound := by
  cases done
  · simpa [discountedBackup, ← mul_sub, abs_mul, abs_of_nonneg hγ] using
      mul_le_mul_of_nonneg_left hb hγ
  · simpa [discountedBackup] using mul_nonneg hγ ((abs_nonneg _).trans hb)

/-- For `gamma < 1`, a nonnegative `d` with `d ≤ gamma * d` is zero. -/
theorem eq_zero_of_le_mul {gamma d : ℝ} (hγ : gamma < 1) (hd : 0 ≤ d)
    (h : d ≤ gamma * d) : d = 0 := by
  nlinarith

end RL
end TorchLean
