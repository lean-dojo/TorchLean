/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Core
public import NN.Spec.Core.FloatInstances.NF

/-!
# NF Elementwise Bounds: Binary Arithmetic
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

omit [ValidRndToNearest rnd] in
/--
`approxTensor` bound for elementwise addition (`addSpec`) over arbitrary tensor shapes.

The output epsilon is computed as `linf_norm (add_bound_tensor epsx epsy xR yR)`, which combines the
input epsilons and one rounding-ULP term per element.
-/
theorem approxTensor_add_spec {s : Shape} [ValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (addSpec xS yS) (addSpec xR yR)
          (linfNorm (addBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := fun a b => a + b) (fR := fun a b => a + b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + ulp β fexp (a + b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR)
      (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) hx hy))
  simpa [addSpec, addBoundTensor] using h

/--
`approxTensor` bound for elementwise subtraction (`subSpec`) over arbitrary tensor shapes.

This is obtained by lifting the scalar subtraction bound `approx_sub_nf` via
`approxTensor_map2_spec_of_scalar_bound`.
-/
theorem approxTensor_sub_spec {s : Shape} :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (subSpec xS yS) (subSpec xR yR)
          (linfNorm (subBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (s := s)
      (fS := fun a b => a - b) (fR := fun a b => a - b)
      (bnd := fun a b epsx epsy =>
        epsx + epsy + ulp β fexp (a - b) / 2)
      (xS := xS) (yS := yS) (xR := xR) (yR := yR) (epsx := epsx) (epsy := epsy)
      hx hy (by
        intro x y xR yR hx hy
        simpa using
          (approx_sub_nf (β := β) (fexp := fexp) (rnd := rnd) (x := x) (y := y) (xR := xR) (yR :=
            yR) hx hy))
  simpa [subSpec, subBoundTensor] using h

omit [ValidRndToNearest rnd] in
/--
`approxTensor` bound for elementwise multiplication (`mulSpec`) over arbitrary tensor shapes.

The scalar core is `approx_mul_nf`, lifted componentwise; the resulting bound is packaged as
`mulBoundTensor` and reduced with `linfNorm`.
-/
theorem approxTensor_mul_spec {s : Shape} [ValidRndToNearest rnd] :
    ∀ {xS yS : SpecTensor s} {xR yR : Tensor R s} {epsx epsy : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsx →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsy →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (mulSpec xS yS) (mulSpec xR yR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) epsx epsy xR yR)) := by
  intro xS yS xR yR epsx epsy hx hy
  have h :=
    approxTensor_map2_spec_of_scalar_bound
      (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
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
          (approx_mul_nf (β := β) (fexp := fexp) (rnd := rnd)
            (x := x) (y := y) (xR := xR) (yR := yR) hx hy))
  simpa [mulSpec, mulBoundTensor] using h

/-- Squaring is the diagonal specialization of elementwise multiplication, so it uses the same
rounded multiplication bound rather than a separate numerical rule. -/
theorem approxTensor_square_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (squareSpec xS) (squareSpec xR)
          (linfNorm (mulBoundTensor (β := β) (fexp := fexp) eps eps xR xR)) := by
  intro xS xR eps hx
  have h := approxTensor_mul_spec
    (β := β) (fexp := fexp) (rnd := rnd) hx hx
  simpa only [squareSpec_eq_mulSpec] using h
end NFBackend

end

end RuntimeApprox
end Proofs
