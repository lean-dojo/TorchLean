/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Context.Real
public import TorchLean.Tensor.Layers.RealFFT

/-!
# Real Fourier matrices over the reals

`TorchLean.RealFFT.forwardMatrix` and `TorchLean.RealFFT.inverseMatrix` take the angle of time `t`
and frequency `f` as `2π r / n` with `r = t f mod n`, reduced exactly before the one cast into the
scalar. Over the reals the two angles differ by whole turns, so both matrices are the ones of the
angle `2π t f / n`.
-/

@[expose] public section

namespace TorchLean.RealFFT

/-- The reduced angle and `2π t f / n` have the same cosine and sine. -/
theorem cos_sin_reduced_angle (n time frequency : Nat) :
    Real.cos (2 * Real.pi * (((time * frequency % n : Nat) / n : Rat) : ℝ)) =
        Real.cos (2 * Real.pi * time * frequency / n) ∧
      Real.sin (2 * Real.pi * (((time * frequency % n : Nat) / n : Rat) : ℝ)) =
        Real.sin (2 * Real.pi * time * frequency / n) := by
  have hturns : 2 * Real.pi * time * frequency / n =
      2 * Real.pi * (((time * frequency % n : Nat) / n : Rat) : ℝ) +
        ((time * frequency / n : Nat) : ℝ) * (2 * Real.pi) := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · have hn' : (n : ℝ) ≠ 0 := by positivity
      have hsplit : (time : ℝ) * frequency =
          n * ((time * frequency / n : Nat) : ℝ) + ((time * frequency % n : Nat) : ℝ) := by
        exact_mod_cast (Nat.div_add_mod (time * frequency) n).symm
      rw [Rat.cast_div, Rat.cast_natCast, Rat.cast_natCast, mul_assoc (2 * Real.pi), hsplit]
      field_simp
      ring
  rw [hturns, Real.cos_add_nat_mul_two_pi, Real.sin_add_nat_mul_two_pi]
  exact ⟨rfl, rfl⟩

/-- Over the reals the forward matrix is the one of the angle `2π t f / n`. -/
theorem forwardMatrix_real (n : Nat) :
    forwardMatrix (α := ℝ) n = Tensor.dim fun time => Tensor.dim fun packed =>
      let frequency := packed.val / 2
      let angle := 2 * Real.pi * (time.val : ℝ) * (frequency : ℝ) / (n : ℝ)
      Tensor.scalar <|
        if packed.val % 2 == 0 then Real.cos angle
        else if isEndpoint n frequency then 0
        else -Real.sin angle := by
  unfold forwardMatrix
  congr 1; funext time; congr 1; funext packed
  obtain ⟨hcos, hsin⟩ := cos_sin_reduced_angle n time.val (packed.val / 2)
  exact congrArg Tensor.scalar (by rw [← hcos, ← hsin]; rfl)

/-- Over the reals the inverse matrix is the one of the angle `2π t f / n`. -/
theorem inverseMatrix_real (n : Nat) :
    inverseMatrix (α := ℝ) n = Tensor.dim fun packed => Tensor.dim fun time =>
      let frequency := packed.val / 2
      let angle := 2 * Real.pi * (time.val : ℝ) * (frequency : ℝ) / (n : ℝ)
      let weight := (multiplicity n frequency : ℝ) / n
      Tensor.scalar <|
        if packed.val % 2 == 0 then weight * Real.cos angle
        else if isEndpoint n frequency then 0
        else -weight * Real.sin angle := by
  unfold inverseMatrix
  congr 1; funext packed; congr 1; funext time
  obtain ⟨hcos, hsin⟩ := cos_sin_reduced_angle n time.val (packed.val / 2)
  exact congrArg Tensor.scalar (by rw [← hcos, ← hsin]; rfl)

end TorchLean.RealFFT
