/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Op.Pointwise
public import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# Derivative of `pow` in the base

We prove that the stored backward of `Op.pow` at the default element, the entrywise power rule
`(if e c = 0 then 0 else e c * x c ^ (e c - 1)) * δ c`, is the adjoint of the Fréchet derivative
of the real forward `x ↦ x ^ e` in the base `x`, for a frozen exponent tensor `e`. Over the reals
the power is Mathlib's `Real.rpow`, and `Numeric.powerBackward` takes its ordinary branch, since
the real dictionary is not a supported quotient carrier
(`TorchLean.Numeric.QuotientArithmetic.supported_real`). The certificate holds on the exact set
where at every entry the base is nonzero, the exponent is zero or the exponent is at least one. We
prove it from `Real.hasDerivAt_rpow_const` and `TorchLean.Op.pointwise_backward_eq_adjoint`.

Off that set a base entry is zero and its exponent `e` lies in `(0, 1)`, where `y ↦ y ^ e` has an
infinite right derivative, or below zero, where `y ^ e` is unbounded near zero. These points are
singularities and not kinks, so no `convention` certificate records the stored value there, which
over the reals is zero since `0 ^ (e - 1) = 0` for `e ≠ 1`, and in a binary format an infinity,
or a NaN where the cotangent is zero.
At a negative base and a non-integer exponent the real model is Mathlib's
`x ^ e = exp (log x * e) * cos (e * π)`, where the float operator returns a NaN.
-/

@[expose] public section

namespace TorchLean

namespace Numeric

/-- Over the reals, the power rule `powerBackward Context.isZero x e g` is
`(if e = 0 then 0 else e * x ^ (e - 1)) * g`. -/
theorem powerBackward_real (x e g : ℝ) :
    powerBackward Context.isZero x e g = (if e = 0 then 0 else e * x ^ (e - 1)) * g := by
  have hz : Context.isZero e = (e == 0) := rfl
  simp only [powerBackward, QuotientArithmetic.supported_real, Bool.not_false, ↓reduceIte, hz,
    beq_iff_eq]

end Numeric

namespace Op

variable {s : Shape}

/-- Where every entry has a nonzero base, a zero exponent or an exponent of at least one, the real
forward of `Op.pow` is differentiable in the base and its stored backward is the adjoint of the
derivative. This is the exact set: for `s = [1]` and the base `0`, `y ^ e` has an infinite right
derivative at `0` for `0 < e < 1` and is unbounded near `0` for `e < 0`. -/
@[op_cert TorchLean.Op.pow derivative 0]
theorem pow_backward_eq_adjoint (x e δ : Tensor ℝ s) (h : ∀ c, x c ≠ 0 ∨ e c = 0 ∨ 1 ≤ e c) :
    DifferentiableAt ℝ (fun y => (((Op.pow (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, e, ())).1) x ∧
      (((Op.pow (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, e, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.pow (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, e, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => y ^ e c)
      (if e c = 0 then 0 else e c * x c ^ (e c - 1)) (x c) := fun c => by
    rcases eq_or_ne (e c) 0 with he | he
    · rw [ite_eq_left he, he]
      simp only [Real.rpow_zero]
      exact hasDerivAt_const (x c) 1
    · rw [ite_eq_right he]
      exact Real.hasDerivAt_rpow_const ((h c).imp_right fun h' => h'.resolve_left he)
  have hp := pointwise_backward_eq_adjoint _ (fun c y => y ^ e c)
    (fun c => if e c = 0 then 0 else e c * x c ^ (e c - 1)) x δ _
    (fun y c => Tensor.zipWith_apply _ y e c) hd
    (fun c => by
      show (Tensor.Rep.ofFlatFn fun index => Numeric.powerBackward Context.isZero
        (Tensor.Rep.getFlat x index) (Tensor.Rep.getFlat e index)
        (Tensor.Rep.getFlat δ index)) c = _
      rw [Tensor.Rep.get_ofFlatFn]
      exact Numeric.powerBackward_real (x c) (e c) (δ c))
  exact ⟨hp.1.differentiableAt, hp.2⟩

/-- Where every base entry is positive, the real forward of `Op.powValue` has the sum of the slope
maps `y x ^ (y - 1)` and `x ^ y log x` as derivative on the L2 product of base and exponent, and
its pair of pullbacks is the adjoint of the derivative. At a negative base the real model is
Mathlib's `exp (log x * y) * cos (y * π)`, whose slope in the exponent is not `x ^ y log x`. -/
@[op_cert TorchLean.Op.powValue derivative]
theorem powValue_backward_eq_adjoint (x y δ : Tensor ℝ s) (hx : ∀ c, 0 < x c) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.powValue (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        ((slopeCLM fun c => y c * x c ^ (y c - 1)).comp
            (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
          (slopeCLM fun c => x c ^ y c * Real.log (x c)).comp
            (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
        (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2
          ((((Op.powValue (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
            (((Op.powValue (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.powValue (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
          (WithLp.toLp 2 (x, y))).adjoint δ :=
  pointwise₂_backward_eq_adjoint _ (fun _ q => q.1 ^ q.2) (fun c => y c * x c ^ (y c - 1))
    (fun c => x c ^ y c * Real.log (x c)) x y δ _ _
    (fun p c => Tensor.zipWith_apply _ p.fst p.snd c)
    (fun c => (Real.hasStrictFDerivAt_rpow_of_pos (x c, y c) (hx c)).hasFDerivAt)
    (fun c => by
      show (Tensor.powBackward x y δ).1 c = _
      simp only [Tensor.powBackward, Tensor.zipWith3_apply, Numeric.powerBackward_real]
      split_ifs with h
      · simp only [h, zero_mul]
      · rfl)
    (fun c => by
      show (Tensor.powBackward x y δ).2 c = _
      have hz : Context.isZero (x c) = false := by
        show (x c == 0) = false
        exact beq_false_of_ne (hx c).ne'
      simp only [Tensor.powBackward, Tensor.zipWith3_apply, hz, Bool.false_and]
      exact mul_comm _ _)

end Op

end TorchLean
