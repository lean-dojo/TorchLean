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

end Op

end TorchLean
