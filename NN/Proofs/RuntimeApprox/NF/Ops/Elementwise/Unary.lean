/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Core
public import NN.Proofs.Utils.MathFunctions

/-!
# NF Elementwise Bounds: Unary Operations
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/--
`approxTensor` bound for scaling by a runtime constant (`scaleSpec`) over arbitrary tensor shapes.

This is the tensor-level wrapper around the scalar scaling lemma `approx_scale_nf`.
-/
theorem approxTensor_scale_spec {s : Shape} (c : R) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (scaleSpec (α := SpecScalar) (s := s) xS (toSpec (β := β) (fexp := fexp) (rnd := rnd) c))
          (scaleSpec (α := R) (s := s) xR c)
          (linfNorm (scaleBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps c xR)) :=
            by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := fun x => x * toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
      (fR := fun xR => xR * c)
      (bnd := fun a eps =>
        abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
          ulp β fexp (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using (approx_scale_nf (β := β) (fexp := fexp) (rnd := rnd) (c := c)
          (x := x) (xR := xR) (eps := eps) hx))
  simpa [scaleSpec, scaleBoundTensor] using h

/-- Tensor scaling with an approximate runtime coefficient.

If `xR` approximates `xS` by `eps` and the runtime coefficient `cR` approximates `cS` by `epsC`,
then this theorem accounts for both perturbations and the final rounded multiplication. It is the
coefficient-aware counterpart of `approxTensor_scale_spec`.
-/
theorem approxTensor_scale_spec_of_approx {s : Shape} (cS : ℝ) (cR : R) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps epsC : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) cR - cS) ≤ epsC →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (scaleSpec (α := SpecScalar) (s := s) xS cS)
          (scaleSpec (α := R) (s := s) xR cR)
          (linfNorm (scaleApproxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) eps epsC cR xR)) := by
  intro xS xR eps epsC hx hc
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s) (fS := fun x => x * cS) (fR := fun xR => xR * cR)
      (bnd := fun a eps =>
        (abs a + eps) * epsC +
          (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) cR) + epsC) * eps +
          ulp β fexp
            (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) cR) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := cS) (xR := xR) (yR := cR)
            (epsx := eps) (epsy := epsC) hx hc))
  simpa [scaleSpec, scaleApproxBoundTensor] using h

/-- `approxTensor` bound for elementwise negation (`negSpec`) over arbitrary tensor shapes. -/
theorem approxTensor_neg_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (negSpec xS) (negSpec xR)
          (linfNorm (negBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Neg.neg) (fR := Neg.neg)
      (bnd := fun a eps =>
        eps + ulp β fexp (-a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa using
          (approx_neg_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [negSpec, negBoundTensor] using h

/-- `approxTensor` bound for elementwise absolute value (`absSpec`) over arbitrary tensor
shapes. -/
theorem approxTensor_abs_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (absSpec xS) (absSpec xR)
          (linfNorm (absBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Numerics.MathFunctions.abs) (fR := Numerics.MathFunctions.abs)
      (bnd := fun a eps =>
        eps + ulp β fexp (abs a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        -- `Numerics.MathFunctions.abs` is definitional `abs` on `ℝ`.
        simpa [Numerics.MathFunctions.abs] using
          (approx_abs_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [absSpec, absBoundTensor] using h

/--
`approxTensor` bound for elementwise exponentiation (`expSpec`) over arbitrary tensor shapes.

This lifts the scalar mean-value-theorem bound `approx_exp_nf`.
-/
theorem approxTensor_exp_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (expSpec xS) (expSpec xR)
          (linfNorm (expBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Numerics.MathFunctions.exp) (fR := Numerics.MathFunctions.exp)
      (bnd := fun a eps => expErrorBound (β := β) (fexp := fexp) a eps)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa [Proofs.mathfunc_exp_eq_rexp] using
          (approx_exp_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [expSpec, expBoundTensor] using h

/-- Shape-generic square-root approximation on a certified positive tensor domain.

The pointwise lower bound is carried by `Tensor.Forall`; the global condition `eps < η` guarantees
that every rounded input remains positive. The output budget is assembled entrywise and reduced by
the same infinity norm used throughout `approxTensor`.
-/
theorem approxTensor_sqrt_spec_of_pos_lb {s : Shape} (η : ℝ) (hη : 0 < η) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
      Tensor.Forall (fun x : ℝ => η ≤ x) xS →
      eps < η →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (sqrtSpec xS) (sqrtSpec xR)
          (linfNorm (sqrtPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (s := s) η eps xR)) := by
  intro xS xR eps hx hdom _hbudget
  have h :=
    approxTensor_map_spec_of_scalar_bound_of_forall
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s) (predicate := fun x : ℝ => η ≤ x)
      (fS := fun x => Numerics.MathFunctions.sqrt (max x 0))
      (fR := fun xR => Numerics.MathFunctions.sqrt (max xR 0))
      (bnd := fun a eps =>
        eps / Real.sqrt η +
          ulp β fexp (Real.sqrt (max a 0)) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx hdom (by
        intro x xR hdomain hx
        simpa [Proofs.mathfunc_sqrt_eq_rsqrt] using
          (approx_sqrt_clamp_nf_of_lb
            (β := β) (fexp := fexp) (rnd := rnd)
            hη (le_trans hdomain (le_max_left x 0)) hx))
  simpa [sqrtSpec, sqrtPosBoundTensor] using h

/--
`approxTensor` bound for elementwise hyperbolic tangent (`tanh`) over arbitrary tensor shapes.

Currently uses the coarse unconditional scalar bound `approx_tanh_nf` (boundedness of `tanh`).
-/
theorem approxTensor_tanh_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) Numerics.MathFunctions.tanh xS)
          (mapSpec (s := s) Numerics.MathFunctions.tanh xR)
          (linfNorm (tanhBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd :=
      rnd))
      (s := s)
      (fS := Numerics.MathFunctions.tanh) (fR := Numerics.MathFunctions.tanh)
      (bnd := fun a _eps => (2 : ℝ) + ulp β fexp (Real.tanh a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        simpa [Proofs.mathfunc_tanh_eq_rtanh] using
          (approx_tanh_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (xR := xR) (eps := eps) hx))
  simpa [tanhBoundTensor] using h

-- Real ReLU is non-expansive. The NF wrapper rounds afterward, so its generic bound
-- includes that rounding unless representability of the selected value is established.

private theorem abs_max0_sub_max0_le (x y : ℝ) : abs (max x 0 - max y 0) ≤ abs (x - y) := by
  simpa using (abs_max_sub_max_le_abs x y (0 : ℝ))

/-- Rounded ReLU scalar op for `NF`: apply `max · 0` then round. -/
noncomputable def reluR (x : R) : R :=
  NF.ofReal (β := β) (fexp := fexp) (rnd := rnd)
    (max (toSpec (β := β) (fexp := fexp) (rnd := rnd) x) 0)

/--
Per-entry bound tensor for ReLU (`max · 0`).

ReLU is 1-Lipschitz (`|max x 0 - max y 0| ≤ |x - y|`), so the only new error is the final rounding
step in `reluR`.
-/
def reluBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + ulp β fexp (max a 0) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxTensor` bound for elementwise ReLU (`max · 0`) over arbitrary tensor shapes.

Combines the 1-Lipschitz property of `max` with one rounding step for `reluR`.
-/
theorem approxTensor_relu_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (fun x => max x 0) xS)
          (mapSpec (reluR (β := β) (fexp := fexp) (rnd := rnd)) xR)
          (linfNorm (reluBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)) := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := fun x => max x 0)
      (fR := reluR (β := β) (fexp := fexp) (rnd := rnd))
      (bnd := fun a eps => eps + ulp β fexp (max a 0) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        let xhat := toSpec (β := β) (fexp := fexp) (rnd := rnd) xR
        have hround :
            abs
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max xhat 0) ≤
              ulp β fexp (max xhat 0) / 2 := by
          simpa [reluR, xhat, toSpec, NF.toReal,
            NF.ofReal, NF.roundR,
            Proofs.RuntimeRoundingApprox.roundR] using
            (Proofs.RuntimeRoundingApprox.roundR_abs_error
              (β := β) (fexp := fexp) (rnd := rnd) (max xhat 0))
        have hmax : abs (max xhat 0 - max x 0) ≤ abs (xhat - x) :=
          abs_max0_sub_max0_le xhat x
        have hxhat : abs (xhat - x) ≤ eps := by
          simpa [xhat] using hx
        calc
          abs
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max x 0)
              ≤
            abs
                (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                    (reluR (β := β) (fexp := fexp) (rnd := rnd) xR) -
                  max xhat 0) +
              abs (max xhat 0 - max x 0) := by
                exact abs_sub_le _ _ _
          _ ≤ ulp β fexp (max xhat 0) / 2 + abs (xhat - x) :=
            add_le_add hround hmax
          _ ≤ ulp β fexp (max xhat 0) / 2 + eps := by
            exact add_le_add_right hxhat _
          _ = eps + ulp β fexp (max xhat 0) / 2 := by ring)
  simpa [reluBoundTensor] using h


end NFBackend

end

end RuntimeApprox
end Proofs
