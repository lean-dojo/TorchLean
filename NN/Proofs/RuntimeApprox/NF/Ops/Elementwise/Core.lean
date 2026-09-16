/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Plumbing
public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar

/-!
# NF Elementwise Bounds: Core Arithmetic Budgets
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

-- Elementwise bounds lifted to tensors via `linf_norm`.

/--
Per-entry bound tensor for addition.

`add_bound_tensor epsx epsy xR yR` computes an elementwise error budget for `xR + yR`. Its
  `linfNorm`
is used as the output epsilon in `approxTensor_add_spec`.
-/
def addBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + ulp β fexp (a + b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for subtraction.

Analogous to `addBoundTensor`, but for `xR - yR` (and the corresponding spec subtraction).
-/
def subBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      epsx + epsy + ulp β fexp (a - b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for multiplication.

This is the elementwise lifting of the scalar bound `approx_mul_nf`, tracking first-order error
propagation plus one rounding term.
-/
def mulBoundTensor {s : Shape} (epsx epsy : ℝ) (xR yR : Tensor R s) : SpecTensor s :=
  map2Spec
    (fun a b =>
      (abs a + epsx) * epsy + (abs b + epsy) * epsx + ulp β fexp (a * b) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yR)

/--
Per-entry bound tensor for scaling by a runtime constant.

`scale_bound_tensor eps c xR` bounds the error of `xR * c` assuming the input is approximated within
`eps` and treating `c` as exact (relative to its own `toSpec` value).
-/
def scaleBoundTensor {s : Shape} (eps : ℝ) (c : R) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) * eps +
        ulp β fexp (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c)
          / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry budget for scaling when the scalar coefficient is itself approximate.

`scaleBoundTensor` is the zero-coefficient-error specialization. This general form is required by
constants such as `1 / sqrt(d)` in attention, where constructing the runtime coefficient already
incurs rounding.
-/
def scaleApproxBoundTensor {s : Shape} (eps epsC : ℝ) (c : R)
    (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (abs a + eps) * epsC +
        (abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) c) + epsC) * eps +
        ulp β fexp
          (a * toSpec (β := β) (fexp := fexp) (rnd := rnd) c) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for negation. -/
def negBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + ulp β fexp (-a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry bound tensor for absolute value. -/
def absBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps + ulp β fexp (abs a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for exponentiation (`exp`).

This matches `approx_exp_nf`: a mean-value-theorem bound on the real `exp` plus one rounding term.
-/
def expBoundTensor {s : Shape} (eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec (fun a => expErrorBound (β := β) (fexp := fexp) a eps)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/-- Per-entry square-root budget on a domain with exact lower bound `η`. -/
def sqrtPosBoundTensor {s : Shape} (η eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => eps / Real.sqrt η + ulp β fexp (Real.sqrt (max a 0)) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
Per-entry bound tensor for hyperbolic tangent (`tanh`).

Currently uses the coarse unconditional bound from `approx_tanh_nf` (boundedness of `tanh`).
-/
def tanhBoundTensor {s : Shape} (_eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a => (2 : ℝ) + ulp β fexp (Real.tanh a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

-- Safe log (clamped) bound.

/--
Per-entry bound tensor for `safeLog`.

`safeLog_bound_tensor ε eps xR` is the elementwise bound used by `approxTensor_safeLog_spec`,
combining a `(1/ε)` Lipschitz propagation term with one rounding-ULP term.
-/
def safeLogBoundTensor {s : Shape} (ε eps : ℝ) (xR : Tensor R s) : SpecTensor s :=
  mapSpec
    (fun a =>
      (1 / ε) * eps +
        ulp β fexp (safeLog (ε := ε) a) / 2)
    (tensorToSpec (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)

/--
`approxTensor` bound for clamped log (`safeLog`) lifted to arbitrary tensor shapes.

This is the tensor-level wrapper around `approx_safeLog_nf`, using
  `approxTensor_map_spec_of_scalar_bound`.
-/
theorem approxTensor_safeLog_spec {s : Shape} (ε : ℝ) (hε : 0 < ε) :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mapSpec (s := s) (safeLog (ε := ε)) xS)
          (mapSpec (s := s) (safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε) xR)
          (linfNorm (safeLogBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := s) ε eps xR))
            := by
  intro xS xR eps hx
  have h :=
    approxTensor_map_spec_of_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s) (fS := safeLog (ε := ε))
      (fR := safeLogR (β := β) (fexp := fexp) (rnd := rnd) ε)
      (bnd := fun a inputError =>
        (1 / ε) * inputError + ulp β fexp (safeLog (ε := ε) a) / 2)
      (xS := xS) (xR := xR) (eps := eps) hx (by
        intro x xR hx
        exact approx_safeLog_nf (β := β) (fexp := fexp) (rnd := rnd) hε hx)
  simpa [safeLogBoundTensor] using h
end NFBackend

end

end RuntimeApprox
end Proofs
