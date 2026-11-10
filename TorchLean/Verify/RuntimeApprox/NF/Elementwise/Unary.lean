/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Elementwise
public import TorchLean.Tensor.Numeric.Real
public import TorchLean.Tensor.Elementwise

/-!
# NF Elementwise Bounds: Unary Operations

Tensor bounds for the unary elementwise `NF` operations: scaling, negation, absolute value,
exponential, square root, `tanh` and ReLU. Each theorem lifts a scalar bound entrywise and takes the
`linfNorm` of a per-entry budget tensor as its output epsilon. The clamped square root has two
bounds, a Hölder bound `√eps` that holds at zero entries and a bound `eps / √η`, linear in `eps`,
for exact inputs at least `η > 0`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Robustness

noncomputable section

section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open TorchLean.NF

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/--
`ApproxTensor` bound for `Tensor.scale` by a runtime constant over arbitrary tensor shapes.

The scalar bound `approx_scale` lifts entrywise. The spec side scales by the coefficient's own
value `NF.toReal c`, so the budget `scaleBoundTensor eps c xR` carries no coefficient error.
-/
theorem ApproxTensor.scale {s : Shape} (c : R) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.scale (α := ℝ) (s := s) xS
            (NF.toReal c))
          (Tensor.scale (α := R) (s := s) xR c)
          (linfNorm
            (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps c xR)) := by
  intro xS xR eps hx
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := fun x => x * NF.toReal c)
      (fR := fun xR => xR * c)
      (bnd := fun a eps =>
        _root_.abs (NF.toReal c) * eps +
          ulp β fexp (a * NF.toReal c) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_scale (β := β) (fexp := fexp) (rnd := rnd) (c := c)
          (x := x) (xR := xR) (eps := eps) hx))
  simpa [Tensor.scale, scaleBoundTensor] using h

/-- Tensor scaling with an approximate runtime coefficient.

If `xR` approximates `xS` within `eps` and the runtime coefficient `cR` approximates `cS` within
`epsC`, the budget `scaleApproxBoundTensor` covers both perturbations and the final rounded
multiplication. With an exact coefficient, `cS = NF.toReal cR` and `epsC = 0`, the budget reduces to
that of `scale`.
-/
theorem ApproxTensor.scale_of_approx {s : Shape} (cS : ℝ) (cR : R) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps epsC : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
      _root_.abs (NF.toReal cR - cS) ≤ epsC →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.scale (α := ℝ) (s := s) xS cS)
          (Tensor.scale (α := R) (s := s) xR cR)
          (linfNorm (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) eps epsC cR xR)) := by
  intro xS xR eps epsC hx hc
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s) (fS := fun x => x * cS) (fR := fun xR => xR * cR)
      (bnd := fun a eps =>
        (_root_.abs a + eps) * epsC +
          (_root_.abs (NF.toReal cR) + epsC) * eps +
          ulp β fexp
            (a * NF.toReal cR) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_mul (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := cS) (xR := xR) (yR := cR)
            (epsx := eps) (epsy := epsC) hx hc))
  simpa [Tensor.scale, scaleApproxBoundTensor] using h

/-- `ApproxTensor` bound for elementwise negation (`Tensor.neg`) over arbitrary tensor shapes. -/
theorem ApproxTensor.neg {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.neg xS) (Tensor.neg xR)
          (linfNorm
            (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := Neg.neg) (fR := Neg.neg)
      (bnd := fun a eps =>
        eps + ulp β fexp (-a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_neg (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [Tensor.neg, negBoundTensor] using h

/-- `ApproxTensor` bound for elementwise absolute value (`Tensor.abs`) over arbitrary tensor
shapes. -/
theorem ApproxTensor.abs {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.abs xS) (Tensor.abs xR)
          (linfNorm
            (absBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := Numerics.MathFunctions.abs) (fR := Numerics.MathFunctions.abs)
      (bnd := fun a eps =>
        eps + ulp β fexp (_root_.abs a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        -- `Numerics.MathFunctions.abs` is definitional `abs` on `ℝ`.
        simpa [Numerics.MathFunctions.abs] using
          (approx_abs (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [Tensor.abs, absBoundTensor] using h

/--
`ApproxTensor` bound for elementwise exponentiation (`Tensor.exp`) over arbitrary tensor shapes.

The scalar mean value bound `approx_exp` lifts entrywise.
-/
theorem ApproxTensor.exp {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.exp xS) (Tensor.exp xR)
          (linfNorm
            (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := Numerics.MathFunctions.exp) (fR := Numerics.MathFunctions.exp)
      (bnd := fun a eps => expErrorBound (β := β) (fexp := fexp) a eps)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa [MathFunctions.exp_real] using
          (approx_exp (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [Tensor.exp, expBoundTensor] using h

/-- Shape-generic bound for the clamped square root `Tensor.sqrt · .clampNegative`, which needs no
positive lower bound and so covers zero entries, such as AdamW's bias-corrected second moments. -/
theorem ApproxTensor.sqrt {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal)
      xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.sqrt xS .clampNegative) (Tensor.sqrt xR .clampNegative)
      (linfNorm (sqrtBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)) := by
  have bound := map (α := R)
    (toSpec := NF.toReal)
    (fS := fun x => Numerics.MathFunctions.sqrt (max x 0))
    (fR := fun xR => Numerics.MathFunctions.sqrt (max xR 0))
    (bnd := fun a eps => Real.sqrt eps + ulp β fexp (Real.sqrt (max a 0)) / 2) hx (by
      intro x xR hx
      simpa [MathFunctions.sqrt_real] using
        approx_sqrt_clamp (β := β) (fexp := fexp) (rnd := rnd) hx)
  simpa [Tensor.sqrt, sqrtBoundTensor] using bound

/-- Shape-generic square-root approximation on a certified positive tensor domain.

The lower bound `η ≤ x` holds at every entry of the exact tensor, as a `Tensor.Forall` hypothesis.
The runtime operation is the clamped `sqrt (max · 0)`, so no margin between `eps` and `η` is needed:
the scalar bound `approx_sqrt_clamp_of_lb` holds even when a rounded input falls below zero.
-/
theorem ApproxTensor.sqrt_of_pos_lb {s : Shape} (η : ℝ) (hη : 0 < η) :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
      Tensor.Forall (fun x : ℝ => η ≤ x) xS →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.sqrt xS .clampNegative) (Tensor.sqrt xR .clampNegative)
          (linfNorm (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) η eps xR)) := by
  intro xS xR eps hx hdom
  have h :=
    map_of_forall
      (α := R) (toSpec := NF.toReal)
      (s := s) (predicate := fun x : ℝ => η ≤ x)
      (fS := fun x => Numerics.MathFunctions.sqrt (max x 0))
      (fR := fun xR => Numerics.MathFunctions.sqrt (max xR 0))
      (bnd := fun a eps =>
        eps / Real.sqrt η +
          ulp β fexp (Real.sqrt (max a 0)) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx hdom (by
        intro x xR hdomain hx
        simpa [MathFunctions.sqrt_real] using
          (approx_sqrt_clamp_of_lb
            (β := β) (fexp := fexp) (rnd := rnd)
            hη (le_trans hdomain (le_max_left x 0)) hx))
  simpa [Tensor.sqrt, sqrtPosBoundTensor] using h

/--
`ApproxTensor` bound for elementwise hyperbolic tangent (`tanh`) over arbitrary tensor shapes.

It lifts the scalar bound `approx_tanh`, so if `xR` approximates `xS` within `eps`, then `tanh xR`
approximates `tanh xS` within `linfNorm (tanhBoundTensor eps xR)`.
-/
theorem ApproxTensor.tanh {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (shape := s) Numerics.MathFunctions.tanh xS)
          (Tensor.map (shape := s) Numerics.MathFunctions.tanh xR)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    map (α := R)
      (toSpec := NF.toReal)
      (s := s)
      (fS := Numerics.MathFunctions.tanh) (fR := Numerics.MathFunctions.tanh)
      (bnd := fun a eps => min eps 2 + ulp β fexp (Real.tanh a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa [MathFunctions.tanh_real] using
          (approx_tanh (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [tanhBoundTensor] using h

-- Real ReLU is non-expansive. The NF wrapper rounds afterward, so its generic bound
-- includes that rounding unless representability of the selected value is established.

/-- Rounded ReLU scalar op for `NF`: apply `max · 0` then round. -/
noncomputable def NF.reluR (x : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (max (NF.toReal x) 0)

/--
Per-entry bound tensor for ReLU (`max · 0`).

ReLU is 1-Lipschitz (`|max x 0 - max y 0| ≤ |x - y|`), so each entry `eps + ulp (max a 0) / 2` adds
only the final rounding in `reluR` to the input error.
-/
def NF.reluBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : Tensor ℝ s :=
  Tensor.map
    (fun a => eps + ulp β fexp (max a 0) / 2)
    (Tensor.toReal (α := R) (toSpec := NF.toReal) xR)

/--
`ApproxTensor` bound for elementwise ReLU (`max · 0`) over arbitrary tensor shapes.

Combines the 1-Lipschitz property of `max` with one rounding step for `reluR`.
-/
theorem ApproxTensor.relu {s : Shape} :
    ∀ {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ},
      ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps →
        ApproxTensor (α := R) (toSpec := NF.toReal)
          (Tensor.map (fun x => max x 0) xS)
          (Tensor.map (reluR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm
            (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    map
      (α := R) (toSpec := NF.toReal)
      (s := s)
      (fS := fun x => max x 0)
      (fR := reluR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun a eps => eps + ulp β fexp (max a 0) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        let xhat := NF.toReal xR
        have hround :
            _root_.abs
                (NF.toReal
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max xhat 0) ≤
              ulp β fexp (max xhat 0) / 2 := by
          simpa [reluR, xhat, NF.toReal,
            NF.ofReal, NF.roundR] using
            (Flocq.error_bound_ulp
              (β := β) (fexp := fexp) rnd (max xhat 0))
        have hmax : _root_.abs (max xhat 0 - max x 0) ≤ _root_.abs (xhat - x) :=
          abs_max_sub_max_le_abs xhat x 0
        have hxhat : _root_.abs (xhat - x) ≤ eps := by
          simpa [xhat] using hx
        calc
          _root_.abs
                (NF.toReal
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max x 0)
              ≤
            _root_.abs
                (NF.toReal
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max xhat 0) +
              _root_.abs (max xhat 0 - max x 0) := by
                exact _root_.abs_sub_le _ _ _
          _ ≤ ulp β fexp (max xhat 0) / 2 + _root_.abs (xhat - x) :=
            add_le_add hround hmax
          _ ≤ ulp β fexp (max xhat 0) / 2 + eps := by
            exact add_le_add_right hxhat _
          _ = eps + ulp β fexp (max xhat 0) / 2 := by ring)
  simpa [reluBoundTensor] using h

end

end

end TorchLean
