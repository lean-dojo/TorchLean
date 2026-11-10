/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Elementwise

/-!
# NF Elementwise Bounds: Binary Arithmetic

Tensor bounds for elementwise addition, subtraction, multiplication and squaring in `NF`. The
first three lift a scalar bound through `zipWith`, and squaring
reuses the multiplication bound.
-/

@[expose] public section

namespace TorchLean
namespace ApproxTensor

open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TorchLean.NF (mulBoundTensor addBoundTensor subBoundTensor)

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/--
`ApproxTensor` bound for elementwise addition (`+`) over arbitrary tensor shapes.

The output epsilon is `linfNorm (addBoundTensor epsx epsy xR yR)`: each entry pays both
input errors and half an ULP for its rounding.
-/
theorem add {s : Shape} :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxTensor (α := R) (toSpec := NF.toReal) yS yR epsy →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (xS + yS) (xR + yR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    zipWith
      (α := R) (toSpec := NF.toReal)
      (s := s)
      (fS := fun a b => a + b) (fR := fun a b => a + b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + ulp β fexp (a + b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (NF.approx_add (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) hx hy))
  simpa [add_def, addBoundTensor] using h

/--
`ApproxTensor` bound for elementwise subtraction (`-`) over arbitrary tensor shapes.

The scalar bound `NF.approx_sub` lifts through `zipWith`, and the
output epsilon is `linfNorm (subBoundTensor epsx epsy xR yR)`.
-/
theorem sub {s : Shape} :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxTensor (α := R) (toSpec := NF.toReal) yS yR epsy →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (xS - yS) (xR - yR)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    zipWith (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := fun a b => a - b) (fR := fun a b => a - b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + ulp β fexp (a - b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (NF.approx_sub (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y := y) (xR := xR) (yR :=
            yR) hx hy))
  simpa [sub_def, subBoundTensor] using h

/--
`ApproxTensor` bound for elementwise multiplication (`*`) over arbitrary tensor shapes.

The scalar bound `NF.approx_mul` lifts componentwise, and the output epsilon is the
`linfNorm` of `mulBoundTensor`.
-/
theorem mul {s : Shape} :
    ∀ {xS yS : Tensor ℝ s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxTensor (α := R) (toSpec := NF.toReal) yS yR epsy →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (xS * yS) (xR * yR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    zipWith
      (α := R) (toSpec := NF.toReal)
      (s := s)
      (fS := fun a b => a * b) (fR := fun a b => a * b)
      (bnd := fun a b epsx epsy =>
        (abs a + epsx) * epsy + (abs b + epsy) * epsx +
          ulp β fexp (a * b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (NF.approx_mul (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) hx hy))
  simpa [mul_def, mulBoundTensor] using h

/-- Squaring is elementwise multiplication of a tensor with itself, so its budget is the
multiplication budget `mulBoundTensor eps eps xR xR`. -/
theorem square {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.square xS) (Tensor.square xR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) eps eps xR xR)) := by
  intro xS xR eps hx
  have h := mul
    (β := β) (fexp := fexp) (rnd := rnd) hx hx
  simpa only [square_eq_mul] using h

end

end

end ApproxTensor
end TorchLean
