/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.Verify.RuntimeApprox.OnlineSoftmax.Binary32

/-!
# `softmax_bound` at sizes the library does not state

`softmax_bound` proves the closed form of online softmax over 1024 keys in 16 blocks of 64, with
values of magnitude at most one in `binary32`, which no library theorem states. It leaves a bound in
a variable open as the inequality between the evaluated closed form and that variable, and proves
neither a bound below the evaluated closed form nor the range premises below the normalizer's range.
-/

@[expose] public section

namespace TorchLeanTest.SoftmaxBound

open TorchLean.OnlineSoftmax

/-- The premises and the bound `b` of the closed form of online softmax over 1024 keys in blocks of
64, with values of magnitude at most one, unit roundoff `u`, underflow allowance `η` and largest
finite value `mx`. -/
def Closed (u η mx b : ℝ) : Prop :=
  let cl := u * ((64 : ℕ) + 3 * (16 : ℕ) - 1)
  let co := u * ((64 : ℕ) + 3 * (16 : ℕ))
  let al := (Real.exp cl - 1) + 2 * u * Real.exp cl * Real.log (1024 : ℕ)
  let ao := (Real.exp co - 1) + 2 * u * Real.exp co * Real.log (1024 : ℕ)
  let ρl := (16 : ℕ) * residualGrowth u η ^ 16 * blockResidual u η 1024 64 1 cl η
  let ρo := (16 : ℕ) * residualGrowth u η ^ 16 *
    blockResidual u η 1024 64 1 co (productResidual u η 0 0 1)
  al + ρl < 1 ∧ ((1024 : ℕ) : ℝ) * 1 * Real.exp cl + ρl ≤ mx ∧
    ((1024 : ℕ) : ℝ) * 1 * Real.exp co + ρo ≤ mx ∧ (1 + 1 * ao + ρo) / (1 - al - ρl) ≤ mx ∧
    (1 + u) * ((1 * (ao + al + ρl) + ρo) / (1 - al - ρl)) + u * 1 + η ≤ b

/-- In `binary32` the closed form at 1024 keys in blocks of 64 is at most 1.6e-5. -/
example {u η mx : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) : Closed u η mx (16 / 1000000) := by
  softmax_bound hmx (log_two_pow_le 10) le_rfl

/-- A bound in a variable `b` stays open, and `linarith` closes it from `16 / 1000000 ≤ b`. -/
example {u η mx b : ℝ} (hu : u = 2 ^ (-24 : ℤ)) (hη : η = 2 ^ (-150 : ℤ))
    (hmx : (2 : ℝ) ^ 100 ≤ mx) (hb : 16 / 1000000 ≤ b) : Closed u η mx b := by
  softmax_bound hmx (log_two_pow_le 10) le_rfl
  linarith

/-- The evaluated closed form is about 1.50042e-5, so `softmax_bound` does not prove the bound
1.5e-5. The normalizer ranges up to about 1024.007, so it does not prove the range premises for a
largest finite value of `2 ^ 10`. -/
example : True := by
  fail_if_success
    have : ∀ {u η mx : ℝ}, u = 2 ^ (-24 : ℤ) → η = 2 ^ (-150 : ℤ) → (2 : ℝ) ^ 100 ≤ mx →
        Closed u η mx (15 / 1000000) := by
      intro u η mx hu hη hmx
      softmax_bound hmx (log_two_pow_le 10) le_rfl
  fail_if_success
    have : ∀ {u η mx : ℝ}, u = 2 ^ (-24 : ℤ) → η = 2 ^ (-150 : ℤ) → (2 : ℝ) ^ 10 ≤ mx →
        Closed u η mx (16 / 1000000) := by
      intro u η mx hu hη hmx
      softmax_bound hmx (log_two_pow_le 10) le_rfl
  trivial

end TorchLeanTest.SoftmaxBound
